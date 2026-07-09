import json
from pathlib import Path


def save_decomposition_dict(decomposition_attributes: dict, split_name: str) -> dict:
    with open(f"{split_name}.json", 'w') as f:
        json.dump(decomposition_attributes, f)


def load_decomposition_dict(split_name: str):
    # Base path
    base_path = Path(__file__).resolve().parent

    # Decomposition attributes
    with open(base_path / f"{split_name}.json", "r", encoding="utf-8") as f:
        decomposition_attributes = json.load(f)

    return decomposition_attributes


if __name__ == "__main__":
    decomposition_attributes = load_decomposition_dict("esr_splits")
    print(json.dumps(decomposition_attributes, indent=4))
