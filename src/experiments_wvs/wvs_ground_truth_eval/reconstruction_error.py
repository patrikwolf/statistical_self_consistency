import json
import numpy as np

from experiments_wvs.wvs_ground_truth_eval.evaluation import evaluate_wasserstein_distance
from file_logging.read_and_write_json import save_as_json
from utility.prior_weighted_avg import weighted_avg
from utility.time_helper import get_experiment_timestamp


def _compute_aggregated_answer_distribution(level_nodes: list) -> list[float]:
    # Compute weighted average over distributions
    answer_distribution_list = []
    prior_list = []
    for level_node in level_nodes:
        answer_distribution_list.append(level_node["llm_answer_distribution"])
        prior_list.append(level_node["llm_prior"])

    assert abs(sum(prior_list) - 1) < 1e-8, f"LLM-estimated prior is not normalized, it sums to: {sum(prior_list)}"

    # Prior-weighted average
    aggregated_answer_distribution = weighted_avg(
        values=np.array(answer_distribution_list),
        priors=np.array(prior_list),
    )

    return aggregated_answer_distribution.tolist()


def compute_reconstruction_errors(
        experiment_folder: str,
        llm_estimate_dict: dict,
        ground_truth_distributions: dict
) -> dict:

    reconstruction_errors = {}
    reconstruction_node_aggregation = {}
    for question_identifier, llm_estimates in llm_estimate_dict.items():
        print(f"Computing reconstruction error for question {question_identifier}")
        llm_estimates_for_question = llm_estimates["llm_estimates"]
        ground_truth_distribution = ground_truth_distributions[question_identifier]["answer_distribution"]

        # Extract LLM estimates at root-level
        l0_estimate = next(item for item in llm_estimates_for_question if item["combination"] == [])

        # Compare direct LLM estimate against ground-truth at root-level
        error_0 = evaluate_wasserstein_distance(ground_truth_distribution=ground_truth_distribution,
                                                llm_estimate=l0_estimate["llm_answer_distribution"])

        # Aggregate level 1 estimates
        level = 1
        print(f"     Computing reconstruction error for level {level}")
        level_1_llm_estimates = [item for item in llm_estimates_for_question if item["level"] == level]
        assert len(level_1_llm_estimates) == 2, f"Expected 2 nodes at level {level}, but found {len(level_1_llm_estimates)}"
        aggregated_l1_llm_estimate = _compute_aggregated_answer_distribution(level_nodes=level_1_llm_estimates)
        error_1 = evaluate_wasserstein_distance(ground_truth_distribution=ground_truth_distribution,
                                                llm_estimate=aggregated_l1_llm_estimate)

        # Aggregate level 1 estimates
        level = 2
        print(f"     Computing reconstruction error for level {level}")
        level_2_llm_estimates = [item for item in llm_estimates_for_question if item["level"] == level]
        assert len(level_2_llm_estimates) == 4, f"Expected 4 nodes at level {level}, but found {len(level_2_llm_estimates)}"
        aggregated_l2_llm_estimate = _compute_aggregated_answer_distribution(level_nodes=level_2_llm_estimates)
        error_2 = evaluate_wasserstein_distance(ground_truth_distribution=ground_truth_distribution,
                                                llm_estimate=aggregated_l2_llm_estimate)

        # Add results to dict
        reconstruction_errors[question_identifier] = {
            "error_0": error_0,
            "error_1": error_1,
            "error_2": error_2,
        }

        # Add aggregated nodes to separate dict for easier access when plotting
        reconstruction_node_aggregation[question_identifier] = {
            "aggregated_l1_nodes": level_1_llm_estimates,
            "aggregated_l1_llm_estimate": aggregated_l1_llm_estimate,
            "aggregated_l2_nodes": level_2_llm_estimates,
            "aggregated_l2_llm_estimate": aggregated_l2_llm_estimate,
        }

    # Save results to file
    results_path = save_as_json(
        data=reconstruction_errors,
        experiment=f"{experiment_folder}",
        filename="reconstruction_errors.json",
    )
    print(f"\nReconstruction errors for root saved to {results_path}")

    # Save results to file
    save_as_json(
        data=reconstruction_node_aggregation,
        experiment=f"{experiment_folder}",
        filename="reconstruction_node_aggregation.json",
    )

    return reconstruction_errors


if __name__ == "__main__":
    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name="wvs_llm_reconstruction")

    # Load test LLM estimate dict (FOR TESTING ONLY)
    with open("./test_llm_estimate_dict.json", "r") as f:
        llm_estimate_dict = json.load(f)

    # Load test ground-truth distributions (FOR TESTING ONLY)
    with open("./test_ground_truth_distributions.json", "r") as f:
        ground_truth_distributions = json.load(f)

    # For every question, compute the reconstruction errors at level 0, 1, and 2
    reconstruction_errors = compute_reconstruction_errors(
        experiment_folder=experiment_folder,
        llm_estimate_dict=llm_estimate_dict,
        ground_truth_distributions=ground_truth_distributions,
    )
    print("*" * 80)
    print("Reconstruction errors:")
    print(json.dumps(reconstruction_errors, indent=4))
