import json

from collections import defaultdict
from file_logging.read_and_write_json import save_as_json
from utility.wasserstein_helper import _evaluate_normalized_wasserstein_distance


def _eval_wasserstein_for_all_aggregations(
        aggregation_estimates: list,
        direct_estimates_list: list[dict],
) -> list[float]:

    ws_distances_for_question = []
    for agg_res in aggregation_estimates:
        # Extract combination
        aggregated_combination = [f["value_description"] for f in agg_res["shared_combination_of_aggregate"]]

        # Get direct estimate for the same question and combination
        direct_estimate = next((item for item in direct_estimates_list
                                if [json.dumps(f) for f in item["combination"]] == aggregated_combination), None)

        # Evaluate Wasserstein distance
        ws_distance = _evaluate_normalized_wasserstein_distance(direct_estimate=direct_estimate["llm_answer_distribution"],
                                                                aggregated_estimate=agg_res["aggregated_estimate"])

        # Add results to list
        ws_distances_for_question.append(ws_distance)

    return ws_distances_for_question


def evaluate_aggregated_results(
        experiment_folder: str,
        llm_estimates_list: list[dict],
        aggregation_results: dict,
):
    # Evaluate Wasserstein distance for all aggregated estimates for this question
    ws_distances_for_question_within = _eval_wasserstein_for_all_aggregations(
        aggregation_estimates=aggregation_results["aggregated_estimates_within_tree"],
        direct_estimates_list=llm_estimates_list,
    )
    ws_distances_for_question_cross = _eval_wasserstein_for_all_aggregations(
        aggregation_estimates=aggregation_results["aggregated_estimates_cross_tree"],
        direct_estimates_list=llm_estimates_list,
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


def find_and_evaluate_matching_pairs(estimates: list[dict]) -> list[dict]:
    """Find pairs whose combinations match up to reordering."""
    groups = defaultdict(list)

    # Save in dict by canonical combination
    for estimate in estimates:

        # Get canonical combination
        canonical_combination = tuple(
            sorted(
                (item["description"], tuple(item["values"]))
                for item in estimate["combination"]
            )
        )
        groups[canonical_combination].append(estimate)

    matching_pairs = []
    for combination, matches in groups.items():
        if len(matches) != 2:
            raise ValueError(f"Expected exactly 2 estimates for {combination}, but found {len(matches)}.")

        # Compute discrepancy
        discrepancy = _evaluate_normalized_wasserstein_distance(
            direct_estimate=matches[0]["llm_answer_distribution"],
            aggregated_estimate=matches[1]["llm_answer_distribution"]
        )

        # Store results
        matching_pairs.append(
            {
                "canonical_combination": combination,
                "wasserstein_distance": discrepancy,
                "first_estimate": matches[0],
                "second_estimate": matches[1],
            }
        )

    return matching_pairs


def evaluate_order_consistency(
        experiment_folder: str,
        llm_estimates_list: list[dict],
):
    # Find all level-2 nodes
    level_two_nodes = [entry for entry in llm_estimates_list if len(entry["combination"]) == 2]
    assert len(level_two_nodes) == 8

    # Find matching pairs
    matching_pairs = find_and_evaluate_matching_pairs(level_two_nodes)
    assert len(matching_pairs) == 4

    # Extract absolute differences
    wasserstein_distances = [entry["wasserstein_distance"] for entry in matching_pairs]

    # Save to file
    results_path = save_as_json(
        data=wasserstein_distances,
        experiment=experiment_folder,
        filename="order_consistency_results.json",
    )
    print(f"\nOrder consistency evaluation saved to {results_path}")

    return wasserstein_distances


if __name__ == "__main__":
    direct_estimates = [0.71, 0.22, 0.055, 0.015]
    aggregated_estimates = [0.6825, 0.24, 0.0575, 0.02]

    w1 = _evaluate_normalized_wasserstein_distance(direct_estimates, aggregated_estimates)
    print(f"Wasserstein distance: {w1}")
