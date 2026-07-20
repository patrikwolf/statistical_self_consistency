import json
import numpy as np

from experiments_wvs.wvs_consistency_law_total_prob.build_tree_nodes import get_attribute_combinations
from experiments_wvs.wvs_consistency_law_total_prob.questions import get_question_answer_list
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def _compute_aggregated_estimate(
        attribute_nodes: list,
        llm_estimate_dict: dict,
) -> dict:
    # Extract shared attributes
    shared_attributes = _get_shared_attributes(nodes=attribute_nodes)

    # Find prior values and conditional estimated
    unconditional_priors = []
    answer_distributions = []
    for node in attribute_nodes:
        # Find LLM estimates in list
        llm_estimates = next((entry for entry in llm_estimate_dict if entry["combination"] == node), None)

        # Save prior and conditional estimates
        unconditional_priors.append(llm_estimates["llm_prior"])
        answer_distributions.append(llm_estimates["llm_answer_distribution"])

    if shared_attributes == []:
        assert abs(sum(unconditional_priors) - 1) < 1e-8, (f"LLM priors should up to 1 when the nodes have no shared "
                                                           f"attributes since we are aggregating back to the root node. "
                                                           f"Expected sum to be 1, but got {sum(unconditional_priors)}")

    # Compute conditional prior for both children based on unconditional prior
    conditional_priors = (np.array(unconditional_priors) / sum(unconditional_priors)).tolist()

    # Compute aggregated estimate
    aggregated_estimate = sum(prior * np.array(distribution) for prior, distribution
                              in zip(conditional_priors, answer_distributions))

    # Assert that shared attributes are at the same conditioning position
    _assert_shared_attribute_position(nodes=attribute_nodes, shared_attributes=shared_attributes)

    # Collect results
    results = {
        "shared_combination_of_aggregate": shared_attributes,
        "aggregated_estimate": aggregated_estimate.tolist(),
        "aggregated_prior": sum(unconditional_priors)
    }

    return results


def _get_shared_attributes(nodes: list[list[dict]]) -> list[dict]:
    # print("*" * 80)
    # print("Nodes")
    # print(nodes)

    # Initialize shared set with attributes from first node
    shared = set((d["attribute_description"], d["value_description"]) for d in nodes[0])

    # Compute intersection with attributes from all other nodes
    for node in nodes[1:]:
        current = set((d["attribute_description"], d["value_description"]) for d in node)
        shared &= current

    return [{"attribute_description": attr, "value_description": value} for attr, value in shared]


def _assert_shared_attribute_position(nodes: list[list[dict]], shared_attributes: list[dict], verbose: bool = False) -> None:
    # Convert to set of tuples
    shared_attribute_tuples = {
        (d["attribute_description"], d["value_description"])
        for d in shared_attributes
    }

    # Map shared attributes to their index in the first node
    shared_positions = {
        (d["attribute_description"], d["value_description"]): idx
        for idx, d in enumerate(nodes[0])
        if (d["attribute_description"], d["value_description"]) in shared_attribute_tuples
    }

    # Check that every shared attribute appears at the same index in every node
    for node_idx, node in enumerate(nodes):
        for attr, expected_idx in shared_positions.items():
            current = (
                node[expected_idx]["attribute_description"],
                node[expected_idx]["value_description"],
            )

            assert current == attr, (
                f"Shared attribute {attr} appears at index {expected_idx} "
                f"in the first node, but node {node_idx} has {current} there."
            )

    if verbose:
        print("All shared attributes appear at the correct position")


def compute_all_aggregated_estimates_synthetic(
        experiment_folder: str,
        aggregation_dict: dict,
        llm_estimate_dict: dict
) -> dict:

    print("*" * 80)
    print("Computing aggregated estimates")
    print("*" * 80)

    # Compute aggregated estimates (within tree)
    aggregated_estimates_within = []
    for key, val in aggregation_dict["within_tree"].items():
        if len(val) > 2:
            print("     [Within tree] WARNING: Skipping aggregations over more than 2 nodes.")
            continue
        else:
            print(f"     [Within tree] Computing aggregated estimate for {key}")
            print(f"                   Aggregating {len(val)} nodes")

        aggregated_estimate = _compute_aggregated_estimate(attribute_nodes=val,
                                                           llm_estimate_dict=llm_estimate_dict)
        aggregated_estimates_within.append(aggregated_estimate)

    print("                   -----------")

    # Compute aggregated estimates (cross tree)
    aggregated_estimates_cross = []
    for key, val in aggregation_dict["cross_tree"].items():
        if len(val) > 2:
            print("     [Cross tree] WARNING: Skipping aggregations over more than 2 nodes.")
            continue
        else:
            print(f"     [Cross tree] Computing aggregated estimate for {key}")
            print(f"                   Aggregating {len(val)} nodes")

        aggregated_estimate = _compute_aggregated_estimate(attribute_nodes=val,
                                                           llm_estimate_dict=llm_estimate_dict)
        aggregated_estimates_cross.append(aggregated_estimate)

    # Add results for the given question to dict
    aggregation_results = {
        "aggregated_estimates_within_tree": aggregated_estimates_within,
        "aggregated_estimates_cross_tree": aggregated_estimates_cross,
    }

    # Save to file
    results_path = save_as_json(
        data=aggregation_results,
        experiment=experiment_folder,
        filename="aggregation_results.json",
    )
    print(f"\nAggregated LLM estimates saved to {results_path}")

    return aggregation_results


if __name__ == "__main__":
    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name="wvs_tennis_estimates")

    # Get question answer list
    question_answer_list = get_question_answer_list(experiment_folder=experiment_folder)

    # Splits
    binary_splits = {
        "court surface": [
            "clay",
            "non-clay",
        ],
        "first set": [
            "Federer wins the first set",
            "Nadal wins the first set",
        ]
    }

    # Combinations
    combination_dict = get_attribute_combinations(experiment_folder=experiment_folder, splits=binary_splits, max_depth=2)

    # LLM estimates
    with open("./test_llm_estimate_dict.json", "r") as f:
        llm_estimate_dict = json.load(f)

    # Find nodes for aggregation (2 children)
    attribute = "court surface"
    attribute_nodes = []
    level = 1
    for comb_list in combination_dict[level]:
        assert len(comb_list) == 1, f"Expected combination with one attribute for level 1, but got {len(comb_list)}"
        if comb_list[0]["attribute_description"] == attribute:
            attribute_nodes.append(comb_list)

    print(f"Found {len(attribute_nodes)} child nodes: {attribute_nodes}\n")
    print("\n" + "*" * 80 + "\n")

    # Get aggregated results
    question_identifier = "Q1"
    aggregated_results = _compute_aggregated_estimate(
        attribute_nodes=attribute_nodes,
        llm_estimate_dict=llm_estimate_dict
    )
    print(aggregated_results)
    print("\n" + "*" * 80 + "\n")

    # Find nodes for aggregation (level 2 into level 1)
    attribute = "court surface"
    value = "clay"
    attribute_nodes = []
    level = 2
    for comb_list in combination_dict[level]:
        assert len(comb_list) == 2, f"Expected combination with two attributes for level 2, but got {len(comb_list)}"
        if (comb_list[1]["attribute_description"] == attribute) and (comb_list[1]["value_description"] == value):
            attribute_nodes.append(comb_list)

    print(f"Found {len(attribute_nodes)} child nodes: {attribute_nodes}\n")
    print("\n" + "*" * 80 + "\n")

    # Get aggregated results
    question_identifier = "Q1"
    aggregated_results = _compute_aggregated_estimate(
        attribute_nodes=attribute_nodes,
        llm_estimate_dict=llm_estimate_dict
    )
    print(aggregated_results)
    print("\n" + "*" * 80 + "\n")

    # Test function that extracts shared attributes
    nodes = [
        [
            {'attribute_description': 'sex', 'value_description': 'female'},
            {'attribute_description': 'age', 'value_description': '0 - 5 years'}
        ],
        [
            {'attribute_description': 'sex', 'value_description': 'male'},
            {'attribute_description': 'age', 'value_description': '0 - 5 years'},
        ]
    ]

    shared_attributes = _get_shared_attributes(nodes=nodes)
    print(f"Shared attributes: {shared_attributes}")

    _assert_shared_attribute_position(nodes=nodes, shared_attributes=shared_attributes)
