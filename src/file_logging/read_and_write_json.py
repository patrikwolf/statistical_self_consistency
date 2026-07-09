import json

from datetime import datetime
from typing import Any
from utility.directories import get_results_dir

JsonData = dict[str, Any] | list[dict[str, Any]]


def save_as_json(
        data: JsonData,
        experiment: str,
        filename: str | None = None,
        timestamp: bool = False,
        cluster: bool = False,
) -> str:
    """Save JSON data under the experiment results directory and return the file path."""
    if timestamp:
        timestamp_prefix = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        if filename is None:
            filename = f"{timestamp_prefix}.json"
        else:
            normalized_name = filename if filename.endswith(".json") else f"{filename}.json"
            filename = f"{timestamp_prefix}_{normalized_name}"

    if not filename.endswith(".json"):
        filename += ".json"
        print(f"   ---> WARNING: adding .json to the filename: {filename}")

    experiment_dir = get_results_dir(experiment_name=experiment, cluster=cluster, use_timestamp=False, create_dir=True)
    path = experiment_dir / filename

    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

    return str(path)


def read_json(experiment_name: str, filename: str, cluster: bool = False) -> JsonData:
    """Read JSON data from an experiment results directory."""
    experiment_name = get_results_dir(experiment_name=experiment_name, cluster=cluster, use_timestamp=False, create_dir=True)
    path = experiment_name / filename

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file: {path}") from exc


if __name__ == "__main__":
    sample_data = {
        "name": "Alice",
        "age": 30,
        "city": "New York",
    }

    path = save_as_json(sample_data, experiment="test", filename="some_name", timestamp=False)
    print(f"Saved to path: {path}")

    loaded_data = read_json(experiment_name="test", filename="some_name")
    print(f"Loaded data: {loaded_data}")
