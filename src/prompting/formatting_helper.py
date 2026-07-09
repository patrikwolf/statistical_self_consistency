import json
import numpy as np

from typing import Any
from data_loader_acs.value_map import convert_value_to_human_readable
from experiments_acs.filtering.filter import Filter
from utility.directories import get_prompt_template_dir


def _open_template(template_filename: str) -> str:
    """Load and cache a prompt template from disk."""
    template_dir = get_prompt_template_dir()
    template_path = template_dir / template_filename
    return template_path.read_text(encoding="utf-8")


def _generate_attribute_values(filter_definition: Filter) -> str:
    value_map = filter_definition.value_map
    human_readable_values = [
        f"'{map_filter_value(value=value, value_map=value_map)}'"
        for value in filter_definition.values
    ]
    return ", or ".join(human_readable_values)


def _generate_attribute_list(filters: list[Filter], improved_age_desc: bool) -> str:
    if improved_age_desc:
        print("   ---> Using improved age description")
        return _generate_improved_attribute_list(filters)
    else:
        return _generate_standard_attribute_list(filters)


def _generate_standard_attribute_list(filters: list[Filter]) -> str:
    """Render filter definitions into the bullet list expected by the templates."""
    return "\n".join(
        f"- {filter_definition.description}: {_generate_attribute_values(filter_definition)}"
        for filter_definition in filters
    )


def _generate_improved_attribute_list(filters: list[Filter]) -> str:
    """Render filter definitions into the bullet list expected by the templates."""
    attribute_list = []
    for filter_definition in filters:
        if filter_definition.description != "age":
            attribute_list.append(f"- {filter_definition.description}: {_generate_attribute_values(filter_definition)}")
        else:
            # Improved age description: group ages into ranges
            age_intervals = format_age_intervals(np.array(filter_definition.values))
            attribute_list.append(f"- {filter_definition.description}: {age_intervals}")

    return "\n".join(attribute_list)


def format_age_intervals(values: np.ndarray) -> str:
    if len(values) == 1:
        return f"{values[0]} years old"

    # Separate missing values
    has_unknown = np.isnan(values).any()

    # Keep only valid numeric values and sort them
    ages = np.sort(values[~np.isnan(values)].astype(int))

    if len(ages) == 0:
        return "unknown" if has_unknown else ""

    # Group consecutive ages
    intervals = []
    start = prev = ages[0]

    for age in ages[1:]:
        if age == prev + 1:
            prev = age
        else:
            intervals.append((start, prev))
            start = prev = age

    intervals.append((start, prev))

    # Format intervals
    parts = []
    if has_unknown:
        parts.append("unknown")

    for start, end in intervals:
        if start == end:
            parts.append(f"{start} years old")
        else:
            parts.append(f"{start}–{end} years old")

    return " or ".join(parts)


def _generate_attribute_list_from_combination(combination: list[dict]) -> str:
    """Render filter definitions into the bullet list expected by the templates."""
    return "\n".join(
        f"- {attribute["attribute_description"]}: {attribute["value_description"]}"
        for attribute in combination
    )


def format_numeric_value(value: int | float) -> str:
    return f"{value:.0f}"


def map_filter_value(value: Any, value_map: Any) -> str:
    if hasattr(value_map, "map_value"):
        return convert_value_to_human_readable(value=value, value_map=value_map)
    if callable(value_map):
        mapped_value = value_map(value)
        return str(mapped_value)
    raise TypeError("filter['value_map'] must be callable or expose a map_value method.")


def format_template(template_filename: str, **template_values: str) -> str:
    template = _open_template(template_filename)
    return template.format(**template_values)


def assemble_prompt_with_optional_filters(
        filters: list[Filter],
        filtered_template_filename: str,
        no_filter_template_filename: str,
        improved_age_desc: bool = False,
        **template_values: str,
) -> str:
    if not filters:
        return format_template(no_filter_template_filename, **template_values)

    return format_template(
        filtered_template_filename,
        attributes=_generate_attribute_list(filters=filters, improved_age_desc=improved_age_desc),
        **template_values,
    )


def make_example_json(
        num_groups: int,
        decimals: int = 2,
        seed: int | None = None,
        group_descriptor: str = "Group",
        uniform_distribution: bool = False,
) -> str:
    if uniform_distribution:
        probs = 1 / num_groups * np.ones(num_groups)
    else:
        rng = np.random.default_rng(seed)

        # Draw random positive values and normalize them to sum to one.
        probs = rng.random(num_groups)
        probs = probs / probs.sum()

    # Round while preserving exact sum after rounding.
    rounded = np.round(probs, decimals)
    rounded[-1] = np.round(1.0 - rounded[:-1].sum(), decimals)
    example = {
        f"{group_descriptor} {i + 1}": float(rounded[i])
        for i in range(num_groups)
    }

    return json.dumps(example, indent=2)


def example_filters() -> list[Filter]:
    """Return a small demo payload for manual smoke testing."""
    import numpy as np

    return [
        Filter(
            attribute="AGEP",
            description="age",
            values=[float("nan"), 24, 25, 26, 27, 28, 29, 30, 40, 41, 42, 80],
            value_map=lambda i: f"{i} years old",
        ),
        Filter(
            attribute="SEX",
            description="sex",
            values=[2],
            value_map=lambda i: {1: "Male", 2: "Female"}.get(i, None),
        ),
        Filter(
            attribute="CIT",
            description="citizenship status",
            values=list(np.array([1.0, 2.0])),
            value_map=lambda i: {
                1: "Citizen",
                2: "Non-citizen",
            }.get(i, None),
        ),
    ]


def example_filter_partition() -> list[list[Filter]]:
    """Return a small demo payload for manual smoke testing."""
    return [
        [
            Filter(
                attribute="SEX",
                description="sex",
                values=[2],
                value_map=lambda i: {1: "Male", 2: "Female"}.get(i, None),
            )
        ],
        [
            Filter(
                attribute="SEX",
                description="sex",
                values=[1],
                value_map=lambda i: {1: "Male", 2: "Female"}.get(i, None),
            )
        ],
    ]
