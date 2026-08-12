import os

from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.split_analysis.standard_value_loader import load_standard_values
from experiments_acs.tree_income_threshold.analysis.binary_threshold_split import main
from utility.argument_parser import parse_arguments
from utility.hyperparameters import assert_decomposition_validity, process_hyperparameters


if __name__ == "__main__":
    print(f"Running script with name: {os.path.basename(__file__)}")

    # Model
    # todo: change model
    # model_name = "openai/gpt-5.4"
    # model_name = "openai/gpt-4o-mini"
    # model_name = "google/gemini-3.1-pro-preview"
    # model_name = "google/gemini-3-flash-preview"
    # model_name = "anthropic/claude-opus-4.7"
    # model_name = "x-ai/grok-4.20"
    model_name = "qwen/qwen3.6-plus"
    # model_name = "moonshotai/kimi-k2.6"

    # Threshold
    # todo: change threshold
    income_threshold = 40_000

    # Hyperparameters
    seeds = [42]
    standard_values = load_standard_values(depth=6)
    decomposition_attributes = standard_values["decomposition_attributes"]

    # Configuration
    cfg = MinimalExperimentConfig(
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        experiment_name="aggregation_refinement",
        model_name=model_name,
        reasoning_effort="none",
        prompting_scheme="sociodemographic",
        prompt_reasoning=False,
        llm_aggregation_priors=False,
        num_of_samples=50,
        max_number_attempts=10,
        income_threshold=income_threshold,
        weighted=True,
        income_greater_than_threshold=True,
        log_csv="aggregation_refinement_results.csv",
    )

    if cfg.llm_aggregation_priors:
        print("*" * 80)
        print("WARNING: Using LLM-estimated priors!")
        print("*" * 80)

    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=cfg.survey_year, set_nan_to_zero=cfg.set_nan_to_zero)
    print(f"Successfully loaded ACS data from year {cfg.survey_year}...")

    # Assert validity of decomposition attributes
    assert_decomposition_validity(decomposition_attributes=decomposition_attributes, attribute_dict=attribute_dict)

    # Process hyperparameters
    hyperparameters = process_hyperparameters(
        seeds=seeds,
        decomposition_attributes=decomposition_attributes,
        attribute_dict=attribute_dict,
    )

    # Parse arguments
    shard_id, shard, datetime = parse_arguments()
    print(f"Shard ID: {shard_id} ({shard_id + 1} out of {len(hyperparameters)})")

    # Run main experiment
    main(survey_df=survey_df,
         attribute_dict=attribute_dict,
         cfg=cfg,
         shard=shard,
         timestamp=datetime,
         hyperparameters=hyperparameters[shard],
         template_filename="ACS_income_v3.txt",
         no_filter_template_filename="ACS_income_no_filters_v3.txt",
         extract_confidence=False,
         )
