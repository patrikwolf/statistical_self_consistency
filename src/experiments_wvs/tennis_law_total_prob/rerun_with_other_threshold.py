import json

from experiments_wvs.tennis_law_total_prob.final_scoring import compute_final_scores_synthetic
from file_logging.read_and_write_json import read_json


def rerun_epsilon_evaluation(
        timestamp: str,
        epsilon: float,
        cluster: bool,
):
    experiment_folder = f"tennis_self_consistency/{timestamp}"

    # For every question, evaluate aggregated versus direct estimate (Wasserstein)
    sanity_check_results = read_json(experiment_name=experiment_folder,
                                     filename="sanity_check_results.json",
                                     cluster=cluster
                                     )

    # Compute final scores
    results = compute_final_scores_synthetic(
        experiment_folder=experiment_folder,
        sanity_check_results=sanity_check_results,
        epsilon=epsilon,
        num_checks_per_question=4,
        rerun=True,
        cluster=cluster,
    )

    return results


if __name__ == "__main__":
    # Experiment timestamp
    timestamp = "2026-05-07__09-07-52"

    # Threshold for final scoring (in Wasserstein distance)
    epsilon = 0.02

    # Cluster
    cluster = False

    # Recompute results
    results = rerun_epsilon_evaluation(epsilon=epsilon, timestamp=timestamp, cluster=cluster)

    print("\n" + "*" * 80)
    print("Final results:")
    print("*" * 80 + "\n")
    print(json.dumps(results, indent=4))
