import math
import itertools

from data_loader_acs.value_map import get_value_map
from experiments_acs.filtering.filter import Filter
from experiments_acs.filtering.filter_definitions import extend_generic_filter


def assert_decomposition_validity(decomposition_attributes: dict, attribute_dict: dict) -> None:
    for level, filter in decomposition_attributes.items():
        attribute = filter["attribute"]
        splits = filter["splits"]
        values_in_splits = {
            value
            for split in splits
            for value in split
            if not math.isnan(value)
        }
        allowed_values = set(int(choice["data_value"]) for choice in attribute_dict[attribute]["choices"])

        assert values_in_splits == allowed_values, (
            f"Invalid decomposition attribute: {attribute} at {level}. "
            f"Values in splits: {values_in_splits} should be equal to allowed values: {allowed_values}."
        )


def process_hyperparameters(
        seeds: list[int],
        decomposition_attributes: dict,
        attribute_dict: dict
) -> dict:
    # Root node with no filters
    root = {
        "node_id": 0,
        "parent_id": None,
        "level": "level_0",
        "filters": []
    }

    # Initialize lists
    decomposition_filters = [root]
    prev_filters = [root]
    node_count = 1

    # Process decomposition attributes level by level
    for level, split_desc in decomposition_attributes.items():
        attribute = split_desc["attribute"]
        splits = split_desc["splits"]
        new_filters = []

        for prev in prev_filters:
            prev_filters_list = prev.get("filters", [])
            for value_split in splits:
                new_filter = Filter(
                    attribute=attribute,
                    getter=extend_generic_filter,
                    values=value_split,
                    description=attribute_dict[attribute]["description"],
                    value_map=get_value_map(attribute=attribute, attribute_dict=attribute_dict),
                )
                new_filters.append({
                    "node_id": node_count,
                    "parent_id": prev.get("node_id", None),
                    "level": level,
                    "filters": prev_filters_list + [new_filter]
                })

                # Increment node count for unique node IDs across the tree
                node_count += 1

        # Add new filters to decomposition filters
        decomposition_filters.extend(new_filters)
        prev_filters = new_filters

    # Create hyperparameters for each combination
    hyperparameters = {}
    for idx, (seed, df) in enumerate(itertools.product(seeds, decomposition_filters)):
        hyperparameters[f'shard_{idx}'] = {
            "seed": seed,
            "decomposition_filters": df,
            "decomposition_attribute_list": [decomposition_attributes[level]["attribute"]
                                             for level in decomposition_attributes.keys()],
        }

    return hyperparameters
