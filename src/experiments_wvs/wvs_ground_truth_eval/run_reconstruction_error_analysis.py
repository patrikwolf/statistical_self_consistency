import json
import time

from config.model_config import ModelConfig
from experiments_wvs.config.loader import load_shared_wvs_config
from experiments_wvs.wvs_ground_truth_eval.ground_truth_distribution import get_ground_truth_distributions
from experiments_wvs.wvs_ground_truth_eval.llm_estimates import get_all_llm_estimates_for_original_tree
from experiments_wvs.wvs_ground_truth_eval.reconstruction_error import compute_reconstruction_errors
from experiments_wvs.wvs_consistency_law_total_prob.build_tree_nodes import get_original_attribute_combinations
from experiments_wvs.wvs_consistency_law_total_prob.questions import get_question_answer_list
from utility.directories import get_results_dir
from utility.time_helper import get_experiment_timestamp


def main(
        cfg: ModelConfig,
        binary_splits: dict,
        country: str,
        question_identifiers: list = None,
        truncate: int = None,
):
    print(f"\nModel: {cfg.model_name}")
    print(f"Country: {country}\n")

    # Start timer
    start_time = time.time()

    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Save config
    experiment_path = get_results_dir(experiment_name=experiment_folder, cluster=False, use_timestamp=False,
                                      create_dir=True)
    config_path = experiment_path / "config.json"
    cfg.save_json(path=config_path)

    # Get question answer list
    question_answer_list = get_question_answer_list(experiment_folder=experiment_folder,
                                                    question_identifiers=question_identifiers,
                                                    truncate=truncate)
    if question_identifiers:
        extracted_question_identifiers = [item["question_identifier"] for item in question_answer_list]
        assert set(extracted_question_identifiers) == set(question_identifiers), ("Extracted question identifiers do "
                                                                                  "not match the provided ones.")

    # Combinations (only the combination for one of the trees --> not the reshaped one)
    combination_dict = get_original_attribute_combinations(experiment_folder=experiment_folder,
                                                           splits=binary_splits,
                                                           max_depth=2)

    # For every question, sample LLM estimate for all combinations
    llm_estimate_dict = get_all_llm_estimates_for_original_tree(
        cfg=cfg,
        experiment_folder=experiment_folder,
        question_answer_list=question_answer_list,
        combination_dict=combination_dict,
    )

    # Get ground-truth answer distribution at root-level for all questions
    ground_truth_distributions = get_ground_truth_distributions(
        experiment_folder=experiment_folder,
        country=country,
        question_identifiers=question_identifiers,
    )

    # For every question, compute the reconstruction errors at level 0, 1, and 2
    reconstruction_errors = compute_reconstruction_errors(
        experiment_folder=experiment_folder,
        llm_estimate_dict=llm_estimate_dict,
        ground_truth_distributions=ground_truth_distributions,
    )

    # Timer
    duration_sec = time.time() - start_time
    print(f"\nTIMER: Script finished after {duration_sec:.0f} seconds (= {duration_sec / 60:.1f} minutes).")

    return reconstruction_errors, experiment_folder


if __name__ == "__main__":
    # Load shared config
    shared_config = load_shared_wvs_config()

    # Config
    cfg = ModelConfig(
        experiment_name="wvs_llm_reconstruction",
        country=shared_config["country"],
        model_name=shared_config["model"],
        reasoning_effort=shared_config["reasoning_effort"],
        sampling_temperature=1.0,
        num_of_samples=shared_config["num_of_samples"],
        max_number_attempts=10,
    )

    # Splits
    binary_splits = shared_config["binary_splits"]

    # Question identifiers
    question_identifiers = ["Q177", "Q252", "Q162", "Q245", "Q186", "Q246", "Q251", "Q241", "Q148", "Q112", "Q188",
                            "Q244", "Q120", "Q106", "Q250", "Q109", "Q185", "Q242", "Q184", "Q247", "Q143"]
    question_identifiers = question_identifiers[:shared_config["truncate_questions"]]

    # Run analysis
    """
    Sample complexity: num_questions * num_nodes * num_of_repetitions (* max_number_attempts)
      – num_questions = 200
      – num_nodes = 7 for original tree
      – num_of_repetitions = 20
    """
    results, experiment_folder = main(
        cfg=cfg,
        binary_splits=binary_splits,
        country=cfg.country,
        question_identifiers=question_identifiers,
    )
    timestamp = experiment_folder.split("/")[1]

    print("\n" + "*" * 80)
    print(f"Final results for model {cfg.model_name} and country {cfg.country}:")
    print(f"Reconstruction error: {timestamp}")
    print("*" * 80 + "\n")
    print(json.dumps(results, indent=4))
