from itertools import product
from file_logging.read_and_write_json import save_as_json


def get_attribute_combinations(experiment_folder: str, splits: dict, max_depth: int | None = None) -> dict:
    # Root node
    root = []

    # Initialize dict
    combinations = {
        0: [root],
    }

    if max_depth is None:
        max_depth = len(splits)

    attributes = list(splits.keys())
    for depth in range(1, max_depth + 1):
        depth_attr = attributes[:depth]
        value_lists = [splits[attr] for attr in depth_attr]
        for values in product(*value_lists):
            combination = [
                {
                    "attribute_description": attr,
                    "value_description": value
                }
                for attr, value in zip(depth_attr, values)
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
