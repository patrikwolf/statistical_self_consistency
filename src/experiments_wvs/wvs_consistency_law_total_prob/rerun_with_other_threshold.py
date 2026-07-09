import json

from experiments_wvs.wvs_consistency_law_total_prob.evaluation import evaluate_aggregated_results
from experiments_wvs.wvs_consistency_law_total_prob.final_scoring import compute_final_scores
from file_logging.read_and_write_json import read_json


def rerun_epsilon_evaluation(
        timestamp: str,
        epsilon: float,
        cluster: bool,
):
    experiment_folder = f"wvs_llm_estimates/{timestamp}"

    # Load partial results
    question_answer_list = read_json(experiment_name=experiment_folder,
                                     filename="question_answer_list.json",
                                     cluster=cluster
                                     )
    llm_estimate_dict = read_json(experiment_name=experiment_folder,
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
        question_answer_list=question_answer_list,
        llm_estimate_dict=llm_estimate_dict,
        aggregation_results=aggregation_results,
    )

    # Compute final scores
    results = compute_final_scores(
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
    timestamp = "2026-05-06__15-34-17"

    # Threshold for final scoring (in Wasserstein distance)
    epsilon = 0.02

    # Cluster
    cluster = True

    # Recompute results
    results = rerun_epsilon_evaluation(epsilon=epsilon, timestamp=timestamp, cluster=cluster)

    print("\n" + "*" * 80)
    print("Final results:")
    print("*" * 80 + "\n")
    print(json.dumps(results, indent=4))
