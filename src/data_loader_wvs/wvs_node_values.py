from dataclasses import dataclass


@dataclass()
class WvsNodeValues:

    description: str
    values: list[str | float]
