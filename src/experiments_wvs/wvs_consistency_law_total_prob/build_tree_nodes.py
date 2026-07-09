from itertools import permutations, product

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

    # Splits
    splits = {
        "sex": ["female", "male"],
        "age": ["16 - 44 years old", "45 - 103 years old"]
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
