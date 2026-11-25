import yaml
import json

from pathlib import Path
from file_logging.read_and_write_json import read_json


def load_experiment_timestamps() -> tuple[list[dict], list[float]]:
    base_dir = Path(__file__).parent
    parent_dir = base_dir.parent

    # Load experiment logs
    with open(parent_dir / "experiments.yaml", "r") as f:
        eval_results = yaml.safe_load(f)

    return eval_results["experiments"], eval_results["epsilon_values"]


def load_eval_results() -> dict:
    # Load experiment timestamps and epsilon values
    experiments, epsilon_values = load_experiment_timestamps()

    # Results
    results = {}
    for exp in experiments:
        # Experiment name
        experiment_name = f"decision_model_evaluation/{exp['timestamp']}"
        indicated_model = exp["model"]
        cluster = exp["cluster"]
        decision_model = exp["decision_model"]

        # Initialize results for this experiment
        for eps in epsilon_values:
            # Read results
            eval_results = read_json(
                experiment_name=experiment_name,
                filename=f"evaluation_{eps}.json",
                cluster=cluster,
            )

            # Extract model
            model = eval_results["model"]

            # Assert that the model matches the indicated model
            assert model == indicated_model, (
                f"Model mismatch: indicated model '{indicated_model}' "
                f"does not match model in results '{model}'"
            )

            # Add results for this epsilon
            if model not in results:
                results[model] = {
                    "decision_model": decision_model,
                    "evaluation_results": {}
                }
            results[model]["evaluation_results"][eps] = eval_results

    return results


if __name__ == "__main__":
    # Load evaluation results
    experiments = load_eval_results()

    print(json.dumps(experiments, indent=2))
