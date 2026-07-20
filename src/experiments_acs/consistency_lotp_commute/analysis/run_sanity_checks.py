import json
import time

from config.model_config import ModelConfig
from experiments_acs.consistency_lotp_commute.helper.llm_estimates import get_all_llm_estimates
#
from experiments_acs.consistency_lotp_income.helper.compute_aggregation import compute_all_aggregated_estimates
from experiments_acs.consistency_lotp_income.helper.evaluation import evaluate_aggregated_results, \
    evaluate_order_consistency
from experiments_acs.consistency_lotp_income.helper.final_scoring import compute_new_final_scores
from experiments_acs.consistency_lotp_income.helper.find_aggregation import get_all_aggregations
#
from experiments_acs.filtering.filter_definitions import extend_generic_filter
from experiments_acs.consistency_lotp_income.helper.build_tree_nodes import get_attribute_combinations
from data_loader_acs.data_loader import load_original_attribute_dict
from data_loader_acs.value_map import get_value_map
from experiments_acs.filtering.filter import Filter
from utility.directories import get_results_dir
from utility.time_helper import get_experiment_timestamp


def main(
        cfg: ModelConfig,
        binary_splits: dict,
        bin_edges: list,
        epsilon: float,
) -> tuple[dict, str]:
    print(f"\nModel: {cfg.model_name}")
    print(f"Epsilon: {epsilon}\n")

    # Start timer
    start_time = time.time()

    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Save config
    experiment_path = get_results_dir(experiment_name=experiment_folder, cluster=False, use_timestamp=False,
                                      create_dir=True)
    config_path = experiment_path / "config.json"
    cfg.save_json(path=config_path)

    # Combinations of attribute splits
    combination_dict = get_attribute_combinations(experiment_folder=experiment_folder, splits=binary_splits, max_depth=2)

    # Extract attributes
    attributes = list(binary_splits.keys())

    # Sample LLM estimate for all combinations
    llm_estimates_list = get_all_llm_estimates(
        cfg=cfg,
        experiment_folder=experiment_folder,
        attributes=attributes,
        combination_dict=combination_dict,
        bin_edges=bin_edges,
    )

    # Get aggregation dict for the given question
    aggregation_dict = get_all_aggregations(
        experiment_folder=experiment_folder,
        combination_dict=combination_dict,
        binary_splits=binary_splits,
    )

    # Compute the 8 aggregated estimates and store them
    aggregation_results = compute_all_aggregated_estimates(
        experiment_folder=experiment_folder,
        llm_estimate_dict=llm_estimates_list,
        aggregation_dict=aggregation_dict,
    )

    # Evaluate aggregated versus direct estimate (Wasserstein)
    split_consistency_results = evaluate_aggregated_results(
        experiment_folder=experiment_folder,
        llm_estimates_list=llm_estimates_list,
        aggregation_results=aggregation_results,
    )

    # For all questions, get the order consistency pairs and evaluate discrepancy
    order_consistency_results = evaluate_order_consistency(
        experiment_folder=experiment_folder,
        llm_estimates_list=llm_estimates_list,
    )

    # Compute final scores
    results = compute_new_final_scores(
        experiment_folder=experiment_folder,
        split_consistency_results=split_consistency_results,
        order_consistency_results=order_consistency_results,
        epsilon=epsilon,
        num_split_checks=3,
        num_order_checks=4,
    )

    # Timer
    duration_sec = time.time() - start_time
    print(f"\nTIMER: Script finished after {duration_sec:.0f} seconds (= {duration_sec / 60:.1f} minutes).")

    return results, experiment_folder


if __name__ == "__main__":
    # Load attribute dict
    attribute_dict = load_original_attribute_dict()

    # Binary split
    binary_splits = {
        "AGEP": [
            Filter(
                attribute="AGEP",
                description="age",
                values=[float("nan"), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
                value_map=get_value_map(attribute="AGEP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
            Filter(
                attribute="AGEP",
                description="age",
                values=[31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68],
                value_map=get_value_map(attribute="AGEP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            )
        ],
        "COW": [
            Filter(
                attribute="COW",
                description="class of worker",
                values=[8, 9, float("nan")],
                value_map=get_value_map(attribute="COW", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
            Filter(
                attribute="COW",
                description="class of worker",
                values=[1, 2, 3, 4, 5, 6, 7],
                value_map=get_value_map(attribute="COW", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            )
        ]
    }

    # Bins (optimized for approximately uniform distribution)
    bin_edges = [0, 1, 15, 30, 195]

    # Slack
    epsilon = 0.02

    # Model
    # model = "openai/gpt-5.4"
    # model = "anthropic/claude-sonnet-4.6"
    # model = "x-ai/grok-4.3"
    model = "qwen/qwen3.6-plus"

    # Config
    cfg = ModelConfig(
        experiment_name="acs_commute_self_consistency",
        model_name=model,
        reasoning_effort="none",
        sampling_temperature=1.0,
        # todo: increase to 20
        num_of_samples=20,
        max_number_attempts=10,
    )

    # Run analysis
    # --> Run time is approximately 10 minutes (with 20 samples; GPT 5.4)
    results, experiment_folder = main(
        cfg=cfg,
        binary_splits=binary_splits,
        bin_edges=bin_edges,
        epsilon=epsilon,
    )
    timestamp = experiment_folder.split("/")[1]

    print("\n" + "*" * 80)
    print(f"Final results for model {cfg.model_name}:")
    print(f"Sanity checks: {timestamp}")
    print("*" * 80 + "\n")
    print(json.dumps(results, indent=4))
