import json
from pathlib import Path


def load_standard_values(depth=4):
    # Base path
    base_path = Path(__file__).resolve().parent

    # Decomposition attributes
    with open(base_path / "decomposition_attributes.json", "r", encoding="utf-8") as f:
        decomposition_attributes = json.load(f)

    # Filter tree depth
    decomposition_attributes = {
        key: value
        for key, value in decomposition_attributes.items()
        if int(key.removeprefix("level_")) <= depth
    }

    # Initialize
    standard_values = {
        "decomposition_attributes": decomposition_attributes,
        "model": "openai/gpt-5.4",
        "income_threshold": 40_000
    }

    return standard_values


if __name__ == '__main__':
    standard_values = load_standard_values()
    print(json.dumps(standard_values, indent=2))
