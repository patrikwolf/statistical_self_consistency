import math

from dataclasses import dataclass
from typing import Mapping


@dataclass()
class ValueMapper:
    """Map raw ACS attribute values to their human-readable labels."""

    attribute: str
    choices: Mapping[str, str]

    def __call__(self, value: str) -> str | None:
        """Return the mapped label for a normalized string value."""
        return self.choices.get(value)

    def map_value(self, value: int | float | str) -> str:
        """Convert a raw value to a human-readable label."""
        normalized_value = convert_to_string(value)
        mapped_value = self(normalized_value)
        if mapped_value is None:
            return "Unknown"
        return mapped_value


def convert_to_string(value: int | float | str) -> str | None:
    """Normalize supported raw values to the string keys used in ACS metadata."""
    if isinstance(value, float):
        if math.isnan(value):
            return "Unknown"
        return str(int(value))

    if isinstance(value, int):
        return str(value)

    return value


def get_value_map(attribute: str, attribute_dict: dict) -> ValueMapper:
    """Build a mapper from the attribute metadata dictionary."""
    attribute_metadata = attribute_dict.get(attribute)
    if attribute_metadata is None:
        raise KeyError(f"Attribute {attribute!r} not found in attribute metadata.")

    raw_choices = attribute_metadata.get("choices")
    if raw_choices is None:
        raise ValueError(f"Attribute {attribute} has no choices defined.")

    choices = {
        str(choice["data_value"]): choice["text"]
        for choice in raw_choices
    }
    return ValueMapper(attribute=attribute, choices=choices)


def convert_value_to_human_readable(value: int | float | str, value_map: ValueMapper) -> str:
    """Convert a raw value to a human-readable label using a ValueMapper."""
    return value_map.map_value(value)


if __name__ == "__main__":
    from data_loader_acs.data_loader import load_original_attribute_dict

    # Load attribute dictionary
    attribute_dict = load_original_attribute_dict()

    # Example usage
    value_map = get_value_map(attribute="CIT", attribute_dict=attribute_dict)
    print(f"2 mapped to: {value_map('2')}")

    # Example usage of convert_value_to_human_readable
    print(f"Value 1.0 mapped to: {convert_value_to_human_readable(value=1.0, value_map=value_map)}")
    print(f"Value 2.0 mapped to: {convert_value_to_human_readable(value=2.0, value_map=value_map)}")
    print(f"Value NaN mapped to: {convert_value_to_human_readable(value=float('nan'), value_map=value_map)}")
