import json
import time

from typing import Literal
from config.model_config import ModelConfig
from experiments_wvs.ablation_logprobs_consistency.compute_aggregation import OLD_compute_all_aggregated_estimates
from experiments_wvs.ablation_logprobs_consistency.final_scoring import compute_final_scores
from experiments_wvs.ablation_logprobs_consistency.logprob_estimates import get_all_llm_estimates
from experiments_wvs.config.loader import load_shared_wvs_config
from experiments_wvs.wvs_consistency_law_total_prob.build_tree_nodes import get_attribute_combinations
from experiments_wvs.wvs_consistency_law_total_prob.evaluation import evaluate_aggregated_results
from experiments_wvs.wvs_consistency_law_total_prob.find_aggregation import get_all_aggregations
from experiments_wvs.wvs_consistency_law_total_prob.questions import get_question_answer_list
from utility.directories import get_results_dir
from utility.time_helper import get_experiment_timestamp


def main(
        cfg: ModelConfig,
        binary_splits: dict,
        epsilon: float,
        question_identifiers: list = None,
        truncate: int = None,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"] = "sociodemographic",
):
    print(f"\nModel: {cfg.model_name}")
    print(f"Prompting scheme: {prompting_scheme}")
    print(f"Reasoning effort: {cfg.reasoning_effort}")
    print(f"Country: {cfg.country}")
    print(f"Epsilon: {epsilon}\n")

    # Start timer
    start_time = time.time()

    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Save config
    experiment_path = get_results_dir(experiment_name=experiment_folder, cluster=False, use_timestamp=False, create_dir=True)
    config_path = experiment_path / "config.json"
    cfg.save_json(path=config_path)

    # Get question answer list
    question_answer_list = get_question_answer_list(experiment_folder=experiment_folder,
                                                    question_identifiers=question_identifiers,
                                                    truncate=truncate)

    # Combinations
    combination_dict = get_attribute_combinations(experiment_folder=experiment_folder, splits=binary_splits, max_depth=2)

    # Extract attributes
    attributes = list(binary_splits.keys())

    # For every question, sample LLM estimate for all combinations
    llm_estimate_dict = get_all_llm_estimates(
        cfg=cfg,
        experiment_folder=experiment_folder,
        question_answer_list=question_answer_list,
        attributes=attributes,
        combination_dict=combination_dict,
        prompting_scheme=prompting_scheme,
    )

    # Get aggregation dict for the given question
    aggregation_dict = get_all_aggregations(
        experiment_folder=experiment_folder,
        combination_dict=combination_dict,
        binary_splits=binary_splits,
    )

    # For all questions, compute the 8 aggregated estimates and store them
    aggregation_results = OLD_compute_all_aggregated_estimates(
        experiment_folder=experiment_folder,
        question_answer_list=question_answer_list,
        llm_estimate_dict=llm_estimate_dict,
        aggregation_dict=aggregation_dict,
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
    )

    # Timer
    duration_sec = time.time() - start_time
    print(f"\nTIMER: Script finished after {duration_sec:.0f} seconds (= {duration_sec / 60:.1f} minutes).")

    return results, experiment_folder


if __name__ == "__main__":
    # Load shared config
    shared_config = load_shared_wvs_config()

    # Config
    cfg = ModelConfig(
        experiment_name="wvs_sc_ablation_logprobs",
        country=shared_config["country"],
        model_name=shared_config["model"],
        reasoning_effort=shared_config["reasoning_effort"],
        sampling_temperature=1.0,
        # todo: leave at 1 because token logprobs are unchanged for repeated runs
        num_of_samples=1,
        max_number_attempts=10,
    )

    # Splits
    binary_splits = shared_config["binary_splits"]

    # Threshold for final scoring (in Wasserstein distance)
    epsilon = shared_config["epsilon"]

    # Question identifiers
    question_identifiers = ["Q177", "Q252", "Q162", "Q245", "Q186", "Q246", "Q251", "Q241", "Q148", "Q112", "Q188",
                            "Q244", "Q120", "Q106", "Q250", "Q109", "Q185", "Q242", "Q184", "Q247", "Q143"]
    question_identifiers = question_identifiers[:shared_config["truncate_questions"]]

    # Run analysis
    results, experiment_folder = main(
        cfg=cfg,
        binary_splits=binary_splits,
        epsilon=epsilon,
        question_identifiers=question_identifiers,
        prompting_scheme="sociodemographic",
    )
    timestamp = experiment_folder.split("/")[1]

    print("\n" + "*" * 80)
    print(f"Final results for model {cfg.model_name} and country {cfg.country}:")
    print(f"Sanity checks: {timestamp}")
    print("*" * 80 + "\n")
    print(json.dumps(results, indent=4))
