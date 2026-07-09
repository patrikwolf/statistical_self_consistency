import json
from pathlib import Path


def load_standard_values():
    # Base path
    base_path = Path(__file__).resolve().parent

    # Decomposition attributes
    with open(base_path / "decomposition_attributes.json", "r", encoding="utf-8") as f:
        decomposition_attributes = json.load(f)

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
