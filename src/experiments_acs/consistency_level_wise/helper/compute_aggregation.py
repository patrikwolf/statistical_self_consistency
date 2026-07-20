from experiments_acs.consistency_lotp_income.helper.compute_aggregation import _compute_aggregated_estimate
from file_logging.read_and_write_json import save_as_json


def compute_all_aggregated_estimates(
        experiment_folder: str,
        combination_dict: dict,
        llm_estimate_dict: list[dict],
) -> dict:

    print("*" * 80)
    print("Computing all aggregated estimates")
    print("*" * 80)

    # Compute aggregated estimates (within tree)
    aggregated_estimates = {}
    for level, val in combination_dict.items():
        print(f"  ---> Computing aggregated estimate for level {level}")
        aggregated_estimate = _compute_aggregated_estimate(attribute_nodes=val,
                                                           llm_estimate_dict=llm_estimate_dict)
        aggregated_estimates[level] = aggregated_estimate

    # Save to file
    results_path = save_as_json(
        data=aggregated_estimates,
        experiment=experiment_folder,
        filename="aggregation_results.json",
    )
    print(f"\nAggregated LLM estimates saved to {results_path}")

    return aggregated_estimates
