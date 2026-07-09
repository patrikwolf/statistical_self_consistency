from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


FilterValueMap = Any
FilterGetter = Callable[["Filter", dict[str, Any]], "Filter"]


@dataclass()
class Filter:
    """Canonical project-wide filter representation."""

    attribute: str
    values: list[Any]
    description: str | None = None
    value_map: FilterValueMap | None = None
    getter: FilterGetter | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "attribute": self.attribute,
            "values": self.values,
        }
        if self.description is not None:
            data["description"] = self.description
        if self.value_map is not None:
            data["value_map"] = self.value_map
        if self.getter is not None:
            data["getter"] = self.getter
        return data

    def serialize(self) -> dict[str, Any]:
        data = {
            "attribute": self.attribute,
            "values": self.values,
        }
        if self.description is not None:
            data["description"] = self.description
        return data

    def with_updates(self, **changes: Any) -> "Filter":
        data = self.to_dict()
        data.update(changes)
        return Filter(
            attribute=data["attribute"],
            values=list(data["values"]),
            description=data.get("description"),
            value_map=data.get("value_map"),
            getter=data.get("getter"),
        )


def ensure_filter(filter_definition: Filter | Mapping[str, Any]) -> Filter:
    """Normalize a dict-like or Filter object into a Filter instance."""
    if isinstance(filter_definition, Filter):
        return filter_definition

    return Filter(
        attribute=str(filter_definition["attribute"]),
        values=list(filter_definition["values"]),
        description=filter_definition.get("description"),
        value_map=filter_definition.get("value_map"),
        getter=filter_definition.get("getter"),
    )
