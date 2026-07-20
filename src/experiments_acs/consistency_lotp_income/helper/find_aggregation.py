import json

from experiments_wvs.wvs_consistency_law_total_prob.build_tree_nodes import get_attribute_combinations
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def _get_l1_l0_aggregations(l1_combinations: list, shared_attribute_type: str) -> list[list[dict]]:
    """
    Search for lists of level 1 nodes that share the first attribute type (not value; e.g., {sex})
    """
    nodes_to_aggregate = []
    for idx, combination in enumerate(l1_combinations):
        assert len(combination) == 1, f"Expected combination with one attribute for level 1, but got {len(combination)}"
        if combination[0]["attribute_description"] == shared_attribute_type:
            nodes_to_aggregate.append(combination)

    return [nodes_to_aggregate]


def _get_l2_l0_aggregations(l2_combinations: list, shared_attribute_types: list[str]) -> list[list[dict]]:
    """
    Search for lists of level 2 nodes for which both attribute types are shared (not value; e.g., {sex, age})
    """
    nodes_to_aggregate = []
    for idx, combination in enumerate(l2_combinations):
        assert len(combination) == 2, f"Expected combination with one attribute for level 2, but got {len(combination)}"
        if (combination[0]["attribute_description"] == shared_attribute_types[0]
                and combination[1]["attribute_description"] == shared_attribute_types[1]):
            nodes_to_aggregate.append(combination)

    return [nodes_to_aggregate]


def _get_l2_l1_aggregations(l2_combinations: list, first_attribute_type: str, first_value_split: list) -> list[list[dict]]:
    """
    Search for lists  of level 2 nodes that share the first attribute type AND attribute value (= will be attributes
    of parent node)
    """
    list_of_aggregations = []
    for value_description in first_value_split:
        nodes_to_aggregate = []
        for idx, combination in enumerate(l2_combinations):
            assert len(combination) == 2, f"Expected combination with one attribute for level 2, but got {len(combination)}"
            if (combination[0]["attribute_description"] == first_attribute_type
                    and combination[0]["value_description"] == value_description):
                nodes_to_aggregate.append(combination)

        # Add to list
        list_of_aggregations.append(nodes_to_aggregate)

    return list_of_aggregations


def get_all_aggregations(
        experiment_folder: str,
        combination_dict: dict,
        binary_splits: dict,
) -> dict:
    # Extract attribute descriptions for original tree and reshaped tree
    attribute_types = list(binary_splits.keys())
    interchanged_attribute_types = attribute_types[::-1]

    # Level 1 to level 0 (2 checks; 1 check per tree)
    within_tree_l1_l0 = _get_l1_l0_aggregations(l1_combinations=combination_dict[1],
                                                shared_attribute_type=attribute_types[0])
    cross_tree_l1_l0 = _get_l1_l0_aggregations(l1_combinations=combination_dict[1],
                                               shared_attribute_type=attribute_types[1])

    # Level 2 to level 0 (2 checks; 1 check per tree)
    within_tree_l2_l0 = _get_l2_l0_aggregations(l2_combinations=combination_dict[2],
                                                shared_attribute_types=attribute_types)
    cross_tree_l2_l0 = _get_l2_l0_aggregations(l2_combinations=combination_dict[2],
                                               shared_attribute_types=interchanged_attribute_types)

    # Level 2 to level 1 (within-tree)
    first_attribute_type = list(binary_splits.keys())[0]
    first_value_split = binary_splits[first_attribute_type]
    within_tree_l2_l1 = _get_l2_l1_aggregations(l2_combinations=combination_dict[2],
                                                first_attribute_type=first_attribute_type,
                                                first_value_split=first_value_split)

    # Level 2 to level 1 (cross-tree)
    first_attribute_type = list(binary_splits.keys())[1]
    first_value_split = binary_splits[first_attribute_type]
    cross_tree_l2_l1 = _get_l2_l1_aggregations(l2_combinations=combination_dict[2],
                                               first_attribute_type=first_attribute_type,
                                               first_value_split=first_value_split)

    # Collect all 8 aggregated estimates for this question
    within_tree_dict = {
        f"aggregation_{idx + 1}": agg for idx, agg in
        enumerate(within_tree_l1_l0 + within_tree_l2_l0 + within_tree_l2_l1)
    }
    cross_tree_dict = {
        f"aggregation_{idx + 1}": agg for idx, agg in enumerate(cross_tree_l1_l0 + cross_tree_l2_l0 + cross_tree_l2_l1)
    }
    aggregation_dict = {
        "within_tree": within_tree_dict,
        "cross_tree": cross_tree_dict
    }

    # Serialize for JSON output
    serialized_agg_dict = {
        "within_tree": serialize_dict(within_tree_dict),
        "cross_tree": serialize_dict(cross_tree_dict)
    }

    # Save to file
    results_path = save_as_json(
        data=serialized_agg_dict,
        experiment=experiment_folder,
        filename="aggregation_dict.json",
    )
    print(f"\nAll possible aggregations saved to {results_path}")

    return aggregation_dict


def serialize_dict(aggregation_dict: dict) -> dict:
    return {
        key: [[
            {
                "attribute_description": d["attribute_description"],
                "value_description": d["value_description"].serialize(),
            } for d in inner_list] for inner_list in value]
        for key, value in aggregation_dict.items()
    }


if __name__ == "__main__":
    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name="wvs_llm_estimates")

    # Splits
    binary_splits = {
        "sex": ["female", "male"],
        "age": ["16 - 44 years old", "45 - 103 years old"]
    }
    attributes = list(binary_splits.keys())
    interchanged_attributes = attributes[::-1]

    # Combinations
    combination_dict = get_attribute_combinations(experiment_folder=experiment_folder, splits=binary_splits, max_depth=2)

    # Level 1 to level 0 (search for lists of level 1 nodes that share the first attribute type (not value; e.g., {sex}))
    within_tree_l1_l0 = _get_l1_l0_aggregations(l1_combinations=combination_dict[1],
                                                shared_attribute_type=attributes[0])
    cross_tree_l1_l0 = _get_l1_l0_aggregations(l1_combinations=combination_dict[1],
                                               shared_attribute_type=interchanged_attributes[0])

    print(f"Within tree (n = {len(within_tree_l1_l0)}):\n{within_tree_l1_l0}")
    print("\n" + "*" * 80 + "\n")
    print(f"Cross tree (n = {len(cross_tree_l1_l0)}):\n{cross_tree_l1_l0}")
    print("\n" + "*" * 80 + "\n")

    # Level 2 to level 0 (2 checks; 1 check per tree)
    within_tree_l2_l0 = _get_l2_l0_aggregations(l2_combinations=combination_dict[2],
                                                shared_attribute_types=attributes)
    cross_tree_l2_l0 = _get_l2_l0_aggregations(l2_combinations=combination_dict[2],
                                               shared_attribute_types=interchanged_attributes)

    print(f"Within tree (n = {len(within_tree_l2_l0)}):\n{within_tree_l2_l0}")
    print("\n" + "*" * 80 + "\n")
    print(f"Cross tree (n = {len(cross_tree_l2_l0)}):\n{cross_tree_l2_l0}")
    print("\n" + "*" * 80 + "\n")

    # Level 2 to level 1 (within-tree)
    first_attribute_type = list(binary_splits.keys())[0]
    first_value_split = binary_splits[first_attribute_type]
    within_tree_l2_l1 = _get_l2_l1_aggregations(l2_combinations=combination_dict[2],
                                                first_attribute_type=first_attribute_type,
                                                first_value_split=first_value_split)

    print(f"Within tree (n = {len(within_tree_l2_l1)}):\n{within_tree_l2_l1}")
    print("\n" + "*" * 80 + "\n")
    print(f"Cross tree (n = {len(within_tree_l2_l1)}):\n{within_tree_l2_l1}")
    print("\n" + "*" * 80 + "\n")

    all_aggregations = get_all_aggregations(
        experiment_folder=experiment_folder,
        combination_dict=combination_dict,
        binary_splits=binary_splits,
    )
    print(json.dumps(all_aggregations, indent=4))
