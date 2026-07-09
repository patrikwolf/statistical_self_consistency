from file_logging.read_and_write_json import save_as_json


def _eval_abs_error_for_all_aggregations(
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

        print(f"Direct estimate: {direct_estimate['llm_answer_distribution']}")
        print(f"Aggregated results: {agg_res['aggregated_estimate']}")

        # Add results to list
        error = abs(direct_estimate["llm_answer_distribution"] - agg_res['aggregated_estimate'])
        ws_distances_for_question.append(error)

    return ws_distances_for_question


def evaluate_aggregated_results_synthetic(
        experiment_folder: str,
        llm_estimate_dict: list[dict],
        aggregation_results: dict,
):
    # Evaluate Wasserstein distance for all aggregated estimates for this question
    ws_distances_for_question_within = _eval_abs_error_for_all_aggregations(
        aggregation_estimates=aggregation_results["aggregated_estimates_within_tree"],
        direct_estimates_for_question=llm_estimate_dict,
    )
    ws_distances_for_question_cross = _eval_abs_error_for_all_aggregations(
        aggregation_estimates=aggregation_results["aggregated_estimates_cross_tree"],
        direct_estimates_for_question=llm_estimate_dict,
    )

    # Store results
    sanity_check_results = {
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
