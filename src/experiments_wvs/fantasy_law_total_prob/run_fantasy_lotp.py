import json

from config.model_config import ModelConfig
from experiments_wvs.wvs_consistency_law_total_prob.build_tree_nodes import get_attribute_combinations
from experiments_wvs.tennis_law_total_prob.compute_aggregation import compute_all_aggregated_estimates_synthetic
from experiments_wvs.tennis_law_total_prob.evaluation import evaluate_aggregated_results_synthetic
from experiments_wvs.tennis_law_total_prob.final_scoring import compute_final_scores_synthetic
from experiments_wvs.fantasy_law_total_prob.llm_estimates import get_all_llm_estimates_synthetic
from experiments_wvs.wvs_consistency_law_total_prob.find_aggregation import get_all_aggregations
from utility.directories import get_results_dir
from utility.time_helper import get_experiment_timestamp


def main(
        cfg: ModelConfig,
        binary_splits: dict,
        epsilon: float,
):
    print(f"Model: {cfg.model_name}")

    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Save config
    experiment_path = get_results_dir(experiment_name=experiment_folder, cluster=False, use_timestamp=False,
                                      create_dir=True)
    config_path = experiment_path / "config.json"
    cfg.save_json(path=config_path)

    # Extract attributes
    attributes = list(binary_splits.keys())

    # Combinations
    combination_dict = get_attribute_combinations(experiment_folder=experiment_folder, splits=binary_splits, max_depth=2)

    # For every question, sample LLM estimate for all combinations
    llm_estimate_dict = get_all_llm_estimates_synthetic(
        cfg=cfg,
        experiment_folder=experiment_folder,
        attributes=attributes,
        combination_dict=combination_dict,
    )

    # Get aggregation dict for the given question
    aggregation_dict = get_all_aggregations(
        experiment_folder=experiment_folder,
        combination_dict=combination_dict,
        binary_splits=binary_splits,
    )

    # For all questions, compute the 8 aggregated estimates and store them
    aggregation_results = compute_all_aggregated_estimates_synthetic(
        experiment_folder=experiment_folder,
        llm_estimate_dict=llm_estimate_dict,
        aggregation_dict=aggregation_dict,
    )

    # For every question, evaluate aggregated versus direct estimate (absolute error)
    sanity_check_results = evaluate_aggregated_results_synthetic(
        experiment_folder=experiment_folder,
        llm_estimate_dict=llm_estimate_dict,
        aggregation_results=aggregation_results,
    )

    # print(json.dumps(sanity_check_results, indent=2))

    # Compute final scores
    results = compute_final_scores_synthetic(
        experiment_folder=experiment_folder,
        sanity_check_results=sanity_check_results,
        epsilon=epsilon,
        num_checks_per_question=4,
    )

    return results


if __name__ == "__main__":
    # Config
    cfg = ModelConfig(
        experiment_name="fantasy_self_consistency",
        country="",
        model_name="openai/gpt-5.4",
        reasoning_effort="none",
        sampling_temperature=1.0,
        # todo: increase
        num_of_samples=1,
        max_number_attempts=10,
    )

    # Splits
    binary_splits = {
        "time of day": [
            "day",
            "night",
        ],
        "weather conditions": [
            "windy",
            "no wind (calm)",
        ]
    }

    # Threshold for final scoring
    epsilon = 0.02

    # Run analysis
    results = main(
        cfg=cfg,
        binary_splits=binary_splits,
        epsilon=epsilon,
    )
    print(json.dumps(results, indent=2))
