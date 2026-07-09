from data_loader_acs.data_loader import load_original_attribute_dict
from data_loader_acs.value_map import get_value_map
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from experiments_acs.filtering.filter import Filter
from experiments_acs.filtering.filter_definitions import extend_generic_filter


def generate_filters(decomposition_attributes: dict, attribute_dict: dict) -> list:
    # Root
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

    return decomposition_filters


if __name__ == "__main__":
    # Attribute dict
    attribute_dict = load_original_attribute_dict()

    # Decomposition attributes#
    standard_values = load_standard_values()
    decomposition_attributes = standard_values["decomposition_attributes"]

    # Filters
    filter_tree = generate_filters(decomposition_attributes=decomposition_attributes, attribute_dict=attribute_dict)
    print(f"Generated {len(filter_tree)} filters")
