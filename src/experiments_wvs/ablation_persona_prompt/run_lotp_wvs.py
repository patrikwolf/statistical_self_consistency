import json

from config.model_config import ModelConfig
from experiments_wvs.config.loader import load_shared_wvs_config
from experiments_wvs.wvs_consistency_law_total_prob.run_lotp_sc import main


if __name__ == "__main__":
    # Load shared config
    shared_config = load_shared_wvs_config()

    # Config
    cfg = ModelConfig(
        experiment_name="wvs_llm_estimates",
        country=shared_config["country"],
        model_name=shared_config["model"],
        reasoning_effort=shared_config["reasoning_effort"],
        sampling_temperature=1.0,
        num_of_samples=shared_config["num_of_samples"],
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
    """
    Sample complexity: num_questions * num_nodes * num_of_repetitions (* max_number_attempts)
      – num_questions = 200
      – num_nodes = 13 (7 per tree, but root is shared)
      – num_of_repetitions = 20
    """
    results, experiment_folder = main(
        cfg=cfg,
        binary_splits=binary_splits,
        epsilon=epsilon,
        question_identifiers=question_identifiers,
        # todo: change
        prompting_scheme="persona",
    )
    timestamp = experiment_folder.split("/")[1]

    print("\n" + "*" * 80)
    print(f"Final results for model {cfg.model_name} and country {cfg.country}:")
    print(f"Sanity checks: {timestamp}")
    print("*" * 80 + "\n")
    print(json.dumps(results, indent=4))
