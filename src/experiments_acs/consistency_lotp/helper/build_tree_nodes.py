from itertools import permutations, product

from data_loader_acs.data_loader import load_original_attribute_dict
from data_loader_acs.value_map import get_value_map
from experiments_acs.filtering.filter import Filter
from experiments_acs.filtering.filter_definitions import extend_generic_filter
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def get_attribute_combinations(experiment_folder: str, splits: dict, max_depth: int) -> dict:
    # Root node
    root = []

    # Initialize dict
    combinations = {
        0: [root],
    }

    attributes = list(splits.keys())
    for depth in range(1, max_depth + 1):
        for attr_order in permutations(attributes, depth):
            value_lists = [splits[attr] for attr in attr_order]
            for values in product(*value_lists):
                combination = [
                    {
                        "attribute_description": attr,
                        "value_description": value
                    }
                    for attr, value in zip(attr_order, values)
                ]

                # Add to dict
                if depth not in combinations:
                    combinations[depth] = [combination]
                else:
                    combinations[depth].append(combination)

    # Serialize
    serialized_splits = {
        key: [d.serialize() for d in value]
        for key, value in splits.items()
    }

    serialized_dict = {
        key: [[
            {
                "attribute_description": d["attribute_description"],
                "value_description": d["value_description"].serialize(),
            } for d in inner_list] for inner_list in value]
        for key, value in combinations.items()
    }

    # Save to file
    save_as_json(
        data=serialized_splits,
        experiment=experiment_folder,
        filename="splits.json",
    )
    results_path = save_as_json(
        data=serialized_dict,
        experiment=experiment_folder,
        filename="combinations.json",
    )
    print(f"\nCombinations list saved to {results_path}")

    return combinations


def get_original_attribute_combinations(experiment_folder: str, splits: dict, max_depth: int) -> dict:
    # Root node
    root = []

    # Initialize dict
    combinations = {
        0: [root],
    }

    attributes = list(splits.keys())
    for depth in range(1, max_depth + 1):
        # Attributes for current depth
        level_attributes = attributes[:depth]
        value_lists = [splits[attr] for attr in level_attributes]
        for values in product(*value_lists):
            combination = [
                {
                    "attribute_description": attr,
                    "value_description": value
                }
                for attr, value in zip(level_attributes, values)
            ]

            # Add to dict
            if depth not in combinations:
                combinations[depth] = [combination]
            else:
                combinations[depth].append(combination)

    # Save to file
    save_as_json(
        data=splits,
        experiment=experiment_folder,
        filename="splits.json",
    )
    results_path = save_as_json(
        data=combinations,
        experiment=experiment_folder,
        filename="combinations.json",
    )
    print(f"\nCombinations list saved to {results_path}")

    return combinations


if __name__ == "__main__":
    # Experiment folder
    date, time, experiment_folder = get_experiment_timestamp(experiment_name="wvs_llm_estimates")

    # Attribute dict
    attribute_dict = load_original_attribute_dict()

    # Splits
    splits = {
        "AGEP": [
            Filter(
                attribute="AGEP",
                values=[float("nan"), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
                value_map=get_value_map(attribute="AGEP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
            Filter(
                attribute="AGEP",
                values=[31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68],
                value_map=get_value_map(attribute="AGEP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            )
        ],
        "COW": [
            Filter(
                attribute="COW",
                values=[8, 9, float("nan")],
                value_map=get_value_map(attribute="COW", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
            Filter(
                attribute="COW",
                values=[1, 2, 3, 4, 5, 6, 7],
                value_map=get_value_map(attribute="COW", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            )
        ]
    }

    # Get all combinations
    combinations_dict = get_attribute_combinations(experiment_folder=experiment_folder, splits=splits, max_depth=2)

    # Print
    num_comb = 0
    for depth, combinations in combinations_dict.items():
        print(f"\nDepth: {depth}")
        for c in combinations:
            print(f"Combination: {c}")
            num_comb += 1

    print(f"\nGenerated {num_comb} combinations.")
    print("*" * 80 + "\n")

    # Get combinations of original tree only
    combinations_dict = get_original_attribute_combinations(experiment_folder=experiment_folder, splits=splits, max_depth=2)

    # Print
    num_comb = 0
    for depth, combinations in combinations_dict.items():
        print(f"\nDepth: {depth}")
        for c in combinations:
            print(f"Combination: {c}")
            num_comb += 1

    print(f"\nGenerated {num_comb} combinations.")
