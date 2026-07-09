from data_loader_acs.value_map import get_value_map
from data_loader_acs.data_loader import load_original_attribute_dict
from experiments_acs.filtering.filter import Filter


def extend_generic_filter(filter_desc: Filter, attribute_dict: dict) -> Filter:
    return filter_desc.with_updates(
        description=attribute_dict[filter_desc.attribute]["description"],
        value_map=get_value_map(attribute=filter_desc.attribute, attribute_dict=attribute_dict),
    )


def extend_age_filter(filter_desc: Filter, attribute_dict: dict) -> Filter:
    return filter_desc.with_updates(
        description=attribute_dict[filter_desc.attribute]["description"],
        value_map=lambda i: f"{i} year old",
    )


def construct_base_filters(
        filter_descriptions: list[Filter],
        decomposition_attributes: list[str],
        attribute_dict: dict) -> list[Filter]:
    """Construct base filters for survey data_acs_income.

    Args:
        filter_descriptions: List of filter dictionaries
        decomposition_attributes: List of attributes to exclude from base filters
        attribute_dict: Dictionary containing attribute metadata

    Returns:
        List of filter dictionaries with attribute, description, values, and value_map

    Raises:
        AssertionError: If any decomposition attribute is already in base filters
    """
    extended_filters = []
    for filter_definition in filter_descriptions:
        extended_filters.append(filter_definition.getter(filter_desc=filter_definition, attribute_dict=attribute_dict))

    # Ensure decomposition attributes are not in base filters
    base_filter_attributes = [filter_definition.attribute for filter_definition in extended_filters]
    for attr in decomposition_attributes:
        assert attr not in base_filter_attributes, (
            f"Decomposition attribute '{attr}' cannot be in base filters. "
            f"Base filters: {base_filter_attributes}"
        )

    return extended_filters


if __name__ == "__main__":
    # Load attribute dictionary
    attribute_dict = load_original_attribute_dict()

    # Filter description example
    filter_desc = Filter(attribute="SEX", values=[2])

    # Extend filters
    filters = extend_generic_filter(
        filter_desc=filter_desc,
        attribute_dict=attribute_dict,
    )

    # Print
    print(filters)
