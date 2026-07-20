from file_logging.read_and_write_json import save_as_json
from utility.matching_pairs import find_and_evaluate_matching_pairs
from utility.wasserstein_helper import _evaluate_normalized_wasserstein_distance


def _eval_normalized_wasserstein_for_all_aggregations(
        aggregation_estimates: list,
        direct_estimates_for_question: list[dict],
) -> list[float]:

    ws_distances_for_question = []
    for agg_res in aggregation_estimates:
        # Extract combination
        aggregated_combination = agg_res["shared_combination_of_aggregate"]

        # Get direct estimate for the same question and combination
        direct_estimate = next((item for item in direct_estimates_for_question
                                if item["combination"] == aggregated_combination), None)

        # Evaluate Wasserstein distance
        ws_distance = _evaluate_normalized_wasserstein_distance(direct_estimate=direct_estimate["llm_answer_distribution"],
                                                                aggregated_estimate=agg_res["aggregated_estimate"])

        # Add results to list
        ws_distances_for_question.append(ws_distance)

    return ws_distances_for_question


def evaluate_aggregated_results(
        experiment_folder: str,
        question_answer_list: list[dict],
        llm_estimate_dict: dict,
        aggregation_results: dict,
):
    sanity_check_results = {}

    # For every question, evaluate all aggregated estimates against the direct estimates
    for question_answer_dict in question_answer_list:
        question_identifier = str(question_answer_dict["question_identifier"])

        # Get all direct estimates for this question
        direct_estimates_for_question = llm_estimate_dict[question_identifier]["llm_estimates"]

        # Evaluate Wasserstein distance for all aggregated estimates for this question
        ws_distances_for_question_within = _eval_normalized_wasserstein_for_all_aggregations(
            aggregation_estimates=aggregation_results[question_identifier]["aggregated_estimates_within_tree"],
            direct_estimates_for_question=direct_estimates_for_question,
        )
        ws_distances_for_question_cross = _eval_normalized_wasserstein_for_all_aggregations(
            aggregation_estimates=aggregation_results[question_identifier]["aggregated_estimates_cross_tree"],
            direct_estimates_for_question=direct_estimates_for_question,
        )

        # Store results
        sanity_check_results[question_identifier] = {
            "question_identifier": question_identifier,
            "wasserstein_distances_within_tree": ws_distances_for_question_within,
            "wasserstein_distances_cross_tree": ws_distances_for_question_cross,
        }

    # Save to file
    results_path = save_as_json(
        data=sanity_check_results,
        experiment=experiment_folder,
        filename="sanity_check_results.json",
    )
    print(f"\nSanity check evaluation saved to {results_path}")

    return sanity_check_results


def evaluate_order_consistency(
        experiment_folder: str,
        question_answer_list: list[dict],
        llm_estimate_dict: dict,
):
    order_consistency_results = {}

    # For every question, evaluate all aggregated estimates against the direct estimates
    for question_answer_dict in question_answer_list:
        question_identifier = str(question_answer_dict["question_identifier"])

        # Get all direct estimates for this question
        direct_estimates_for_question = llm_estimate_dict[question_identifier]["llm_estimates"]

        # Find all level-2 nodes
        level_two_nodes = [entry for entry in direct_estimates_for_question if len(entry["combination"]) == 2]
        assert len(level_two_nodes) == 8

        # Find matching pairs
        matching_pairs = find_and_evaluate_matching_pairs(level_two_nodes)
        assert len(matching_pairs) == 4

        # Extract wasserstein distances
        wasserstein_distances_for_question = [entry["wasserstein_distance"] for entry in matching_pairs]

        # Store results
        order_consistency_results[question_identifier] = {
            "question_identifier": question_identifier,
            "wasserstein_distances": wasserstein_distances_for_question,
        }

    # Save to file
    results_path = save_as_json(
        data=order_consistency_results,
        experiment=experiment_folder,
        filename="order_consistency_results.json",
    )
    print(f"\nOrder consistency evaluation saved to {results_path}")

    return order_consistency_results
