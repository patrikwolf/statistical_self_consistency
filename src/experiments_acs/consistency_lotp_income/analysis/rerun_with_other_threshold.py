import json

from experiments_acs.consistency_lotp_income.helper.evaluation import evaluate_aggregated_results
from experiments_acs.consistency_lotp_income.helper.final_scoring import compute_final_scores
from file_logging.read_and_write_json import read_json


def rerun_epsilon_evaluation(
        timestamp: str,
        epsilon: float,
        cluster: bool,
):
    experiment_folder = f"self_consistency_llm_estimates/{timestamp}"

    # Load partial results
    llm_estimates_list = read_json(experiment_name=experiment_folder,
                                   filename="combined_llm_answer_results.json",
                                   cluster=cluster
                                   )

    aggregation_results = read_json(experiment_name=experiment_folder,
                                    filename="aggregation_results.json",
                                    cluster=cluster
                                    )

    # For every question, evaluate aggregated versus direct estimate (Wasserstein)
    sanity_check_results = evaluate_aggregated_results(
        experiment_folder=experiment_folder,
        llm_estimates_list=llm_estimates_list,
        aggregation_results=aggregation_results,
    )

    # Compute final scores
    results = compute_final_scores(
        experiment_folder=experiment_folder,
        sanity_check_results=sanity_check_results,
        epsilon=epsilon,
    )

    return results


if __name__ == "__main__":
    # Experiment timestamp
    timestamp = "2026-06-05__22-09-33"

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
