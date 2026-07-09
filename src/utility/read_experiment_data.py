import yaml

from pathlib import Path
from typing import Tuple
from file_logging.read_and_write_json import read_json
from utility.directories import get_log_dir


def get_list_of_result_files(experiment_subdir: str, cluster: bool) -> list:
    # Construct results directory path
    base_dir = Path(__file__).resolve().parent
    parent_dir = base_dir.parent.parent

    if cluster:
        results_dir = parent_dir / "results_cluster" / experiment_subdir
    else:
        results_dir = parent_dir / "results" / experiment_subdir

    # List all files in the results directory
    all_files = sorted(f.name for f in results_dir.iterdir() if f.is_file())

    return all_files


def read_experiment_logs(experiment_name: str, experiment_logs_path: Path) -> dict:
    """
    Reads the experiment logs from a YAML file and returns the logs for a specific experiment.

    Args:
        experiment_logs_path (str): The name of the experiment to retrieve logs for.
    Returns:
        dict: A dictionary containing the logs for the specified experiment.
    """
    # Load experiment logs
    with open(experiment_logs_path, "r") as f:
        logs = yaml.safe_load(f)

    # Find the specific experiment
    experiment_found = next(
        (experiment for experiment in logs["experiments"] if experiment["name"] == experiment_name),
        None
    )

    if experiment_found is None:
        raise ValueError(f"Experiment with name '{experiment_name}' not found in the logs.")

    return experiment_found


def load_sharded_results_dict(experiment_desc: dict, file_suffix: str = "summary_results.json") -> dict:
    """
    Loads sharded results for different experiment settings.

    Args:
        experiment_desc (dict): The description of the experiment containing paths.
        file_suffix (str): The suffix of the files to be loaded.

    Returns:
        dict: A dictionary containing the loaded results for each experiment setting.
    """

    return load_sharded_results(experiment_subdir=experiment_desc["base_path"],
                                cluster=experiment_desc["cluster"],
                                file_suffix=file_suffix)


def load_sharded_results(experiment_subdir: str, cluster: bool, file_suffix: str = "summary_results.json") -> dict:
    # Get list of result files
    all_files = get_list_of_result_files(experiment_subdir=experiment_subdir, cluster=cluster)
    filtered_files = [f for f in all_files if file_suffix in f]

    # Load results for different experiment settings
    results = {}
    for filename in filtered_files:
        # Read file
        summary = read_json(
            experiment_name=experiment_subdir,
            filename=filename,
            cluster=cluster,
        )

        # Extract key
        key = summary["income_threshold"]

        # Load summary
        results[key] = summary

    return results


def load_non_sharded_results(experiment_desc: dict, filename: str) -> dict:
    """
    Loads non-sharded results for different experiment settings.

    Args:
        experiment_desc (dict): The description of the experiment containing paths.
        filename (str): The name of the file to be loaded.

    Returns:
        dict: A dictionary containing the loaded results for each experiment setting.
    """
    # Load results for different experiment settings
    results = {}
    for key, val in experiment_desc["paths"].items():
        # Load summary
        results[key] = read_json(
            experiment_name=val,
            filename=filename,
            cluster=experiment_desc["cluster"],
        )

    return results


def read_experiment_summary(experiment_name: str, experiment_logs_path: Path) -> Tuple[dict, dict]:
    """
    Reads the experiment data for a specific experiment.

    Args:
        experiment_name (str): The name of the experiment to retrieve data for.
    Returns:
        Tuple[dict, dict]: A tuple containing the results dictionary and metadata dictionary.
    """
    # Read experiment logs to get paths and filenames
    experiment_found = read_experiment_logs(experiment_name=experiment_name, experiment_logs_path=experiment_logs_path)

    # Sharded flag
    sharded = experiment_found["sharded"]

    # Load results
    if sharded:
        results = load_sharded_results_dict(experiment_desc=experiment_found, file_suffix="summary_results.json")
    else:
        results = load_non_sharded_results(experiment_desc=experiment_found, filename="summary_results.json")

    # Extract properties
    first_run = list(results.keys())[0]
    metadata = {
        "filters": results[first_run]["base_filters"],
        "decomposition_attributes": results[first_run]["decomposition_attributes"],
        "num_of_samples": results[first_run]["num_of_samples"],
        "applied_person_weights": results[first_run]["weighted"],
    }

    return results, metadata


def read_full_experiment_data(experiment_name: str, experiment_logs_path: Path) -> Tuple[dict, dict]:
    """
    Reads the experiment data for a specific experiment.

    Args:
        experiment_name (str): The name of the experiment to retrieve data for.
    Returns:
        Tuple[dict, dict]: A tuple containing the results dictionary and metadata dictionary.
    """
    # Read experiment logs to get paths and filenames
    experiment_found = read_experiment_logs(experiment_name=experiment_name, experiment_logs_path=experiment_logs_path)

    # Sharded flag
    sharded = experiment_found["sharded"]

    # Base path
    base_path = experiment_found["base_path"]

    # Experiments
    if sharded:
        # Get list of result files
        all_files = get_list_of_result_files(experiment_subdir=experiment_found["base_path"],
                                             cluster=experiment_found["cluster"])

        # Extract shards
        shards = sorted({name.split("__", 1)[0] for name in all_files})

        # Load results for different experiment settings
        results = {}
        for shard in shards:
            # Load summary
            summary = read_json(
                experiment_name=experiment_found["base_path"],
                filename=f"{shard}__summary_results.json",
                cluster=experiment_found["cluster"],
            )

            # Load LLM direct prompting results
            llm_direct_prompting = read_json(
                experiment_name=experiment_found["base_path"],
                filename=f"{shard}__llm_direct_prompting.json",
                cluster=experiment_found["cluster"],
            )

            # Load LLM law of total probability results
            llm_law_of_total_prob = read_json(
                experiment_name=experiment_found["base_path"],
                filename=f"{shard}__llm_law_of_total_prob.json",
                cluster=experiment_found["cluster"],
            )

            # Extract key
            key = summary["income_threshold"]

            # Save results
            results[key] = {}
            results[key]["summary"] = summary
            results[key]["llm_direct_prompting"] = llm_direct_prompting
            results[key]["llm_law_of_total_prob"] = llm_law_of_total_prob
    else:
        experiments = experiment_found["paths"]

        # Load results for different experiment settings
        results = {}
        for key, val in experiments.items():
            results[key] = {}
            experiment_dir = f"{base_path}/{val}"

            # Load summary
            results[key]["summary"] = read_json(
                experiment_name=experiment_dir,
                filename="summary_results.json",
                cluster=experiment_found["cluster"],
            )

            # Load LLM direct prompting results
            results[key]["llm_direct_prompting"] = read_json(
                experiment_name=experiment_dir,
                filename="llm_direct_prompting.json",
                cluster=experiment_found["cluster"],
            )

            # Load LLM law of total probability results
            results[key]["llm_law_of_total_prob"] = read_json(
                experiment_name=experiment_dir,
                filename="llm_law_of_total_prob.json",
                cluster=experiment_found["cluster"],
            )

    # Extract properties
    first_run = list(results.keys())[0]
    metadata = {
        "filters": results[first_run]["summary"]["base_filters"],
        "decomposition_attributes": results[first_run]["summary"]["decomposition_attributes"],
        "num_of_samples": results[first_run]["summary"]["num_of_samples"],
        "applied_person_weights": results[first_run]["summary"]["weighted"],
    }

    return results, metadata


if __name__ == "__main__":
    # Example usage
    experiment_logs_path = get_log_dir(cluster=False) / "test.yaml"
    experiment_name = "experiment_0__test"

    # Load experiment summary
    results_summary, meta_summary = read_experiment_summary(experiment_name=experiment_name,
                                                            experiment_logs_path=experiment_logs_path)
    print(results_summary)
    print(meta_summary)

    # Load full experiment data
    results, metadata = read_full_experiment_data(experiment_name=experiment_name,
                                                  experiment_logs_path=experiment_logs_path)
    # print(results)
    # print(metadata)
