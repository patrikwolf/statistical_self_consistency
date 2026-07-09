import json
import numpy as np

from file_logging.read_and_write_json import save_as_json


def _compute_aggregated_estimate(
        attribute_nodes: list,
        llm_estimate_dict: list[dict],
) -> dict:

    # Extract shared attributes
    shared_attributes = _get_shared_attributes(nodes=attribute_nodes)

    # Find prior values and conditional estimated
    unconditional_priors = []
    answer_distributions = []
    for node in attribute_nodes:
        # Find LLM estimates in list
        llm_estimates = next(
            (entry for entry in llm_estimate_dict
             if entry["combination"] == _serialize_combination(node)), None)

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


def _serialize_combination(combination: list[dict]) -> list[dict]:
    return [d["value_description"].serialize() for d in combination]


def _get_shared_attributes(nodes: list[list[dict]]) -> list[dict]:
    # Initialize shared set with attributes from first node
    shared = set(
        (
            d["attribute_description"],
            json.dumps(d["value_description"].serialize(), sort_keys=False)
        ) for d in nodes[0])

    # Compute intersection with attributes from all other nodes
    for node in nodes[1:]:
        current = set(
            (
                d["attribute_description"],
                json.dumps(d["value_description"].serialize(), sort_keys=False)
            ) for d in node)
        shared &= current

    return [{"attribute_description": attr, "value_description": value} for attr, value in shared]


def _assert_shared_attribute_position(nodes: list[list[dict]], shared_attributes: list[dict], verbose: bool = False) -> None:
    # Convert to set of tuples
    shared_attribute_tuples = {
        (
            d["attribute_description"],
            d["value_description"]
        )
        for d in shared_attributes
    }

    # Map shared attributes to their index in the first node
    shared_positions = {
        (
            d["attribute_description"],
            json.dumps(d["value_description"].serialize(), sort_keys=False)
        ): idx
        for idx, d in enumerate(nodes[0])
        if (d["attribute_description"],
            json.dumps(d["value_description"].serialize(), sort_keys=False)) in shared_attribute_tuples
    }

    # Check that every shared attribute appears at the same index in every node
    for node_idx, node in enumerate(nodes):
        for attr, expected_idx in shared_positions.items():
            current = (
                node[expected_idx]["attribute_description"],
                json.dumps(node[expected_idx]["value_description"].serialize(), sort_keys=False),
            )

            assert current == attr, (
                f"Shared attribute {attr} appears at index {expected_idx} "
                f"in the first node, but node {node_idx} has {current} there."
            )

    if verbose:
        print("All shared attributes appear at the correct position")


def compute_all_aggregated_estimates(
        experiment_folder: str,
        aggregation_dict: dict,
        llm_estimate_dict: list[dict]
) -> dict:

    print("*" * 80)
    print("Computing all aggregated estimates")
    print("*" * 80)

    # Compute aggregated estimates (within tree)
    aggregated_estimates_within = []
    for key, val in aggregation_dict["within_tree"].items():
        print(f"     [Within tree] Computing aggregated estimate for {key}")
        aggregated_estimate = _compute_aggregated_estimate(attribute_nodes=val,
                                                           llm_estimate_dict=llm_estimate_dict)
        aggregated_estimates_within.append(aggregated_estimate)

    # Compute aggregated estimates (cross tree)
    aggregated_estimates_cross = []
    for key, val in aggregation_dict["cross_tree"].items():
        print(f"     [Cross tree] Compute aggregated estimate for {key}")
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
