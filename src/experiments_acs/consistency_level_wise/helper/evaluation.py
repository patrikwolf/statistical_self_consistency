import json

from file_logging.read_and_write_json import save_as_json
from utility.wasserstein_helper import _evaluate_normalized_wasserstein_distance


def evaluate_aggregated_results(
        experiment_folder: str,
        llm_estimates_list: list[dict],
        aggregation_results: dict,
):
    ws_distances = {}
    for level, agg_res in aggregation_results.items():
        # Extract combination
        aggregated_combination = [f["value_description"] for f in agg_res["shared_combination_of_aggregate"]]

        # Get direct estimate for the same question and combination
        direct_estimate = next((item for item in llm_estimates_list
                                if [json.dumps(f) for f in item["combination"]] == aggregated_combination), None)

        # Evaluate Wasserstein distance
        ws_distance = _evaluate_normalized_wasserstein_distance(
            direct_estimate=direct_estimate["llm_answer_distribution"],
            aggregated_estimate=agg_res["aggregated_estimate"])

        # Add results to list
        ws_distances[level] = ws_distance

    # Save to file
    results_path = save_as_json(
        data=ws_distances,
        experiment=experiment_folder,
        filename="sanity_check_results.json",
    )
    print(f"\nSanity check evaluation saved to {results_path}")

    return ws_distances
