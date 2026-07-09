import numpy as np

from file_logging.read_and_write_json import save_as_json


def compute_final_scores(
    experiment_folder: str,
    sanity_check_results: dict,
    epsilon: float,
    cluster: bool = False,
) -> dict:

    distance_keys = {
        "within_tree": "wasserstein_distances_within_tree",
        "cross_tree": "wasserstein_distances_cross_tree",
    }

    # Initialize results
    results = {
        "epsilon": epsilon
    }

    # Iterate over within- and cross-tree results
    for label, key in distance_keys.items():
        distances = np.asarray(sanity_check_results[key])
        num_satisfied = float(np.sum(distances < epsilon))
        total_num_checks = len(distances)

        results[f"num_{label}_checks_satisfied"] = num_satisfied
        results[f"fraction_of_{label}_checks_satisfied"] = (
            num_satisfied / total_num_checks if total_num_checks > 0 else np.nan
        )

    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"final_scores_{epsilon}.json",
        cluster=cluster,
    )
    print(f"\nFinal score saved to {results_path}")

    return results
