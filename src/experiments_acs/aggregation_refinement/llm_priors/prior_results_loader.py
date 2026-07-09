import json

from pathlib import Path


def load_joint_prior_results() -> dict:
    # Determine base directory
    base_dir = Path(__file__).parent
    file_path = base_dir / "combined_results.json"

    # Read file
    with file_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


if __name__ == "__main__":
    combined_results = load_joint_prior_results()
    print(json.dumps(combined_results, indent=4))
