import os

from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from experiments_acs.tree_income_threshold.analysis.binary_threshold_split import main
from utility.argument_parser import parse_arguments
from utility.hyperparameters import assert_decomposition_validity, process_hyperparameters


if __name__ == "__main__":
    print(f"Running script with name: {os.path.basename(__file__)}")

    # Hyperparameters
    seeds = [42]
    standard_values = load_standard_values()
    decomposition_attributes = standard_values["decomposition_attributes"]

    # Configuration
    cfg = MinimalExperimentConfig(
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        experiment_name="aggregation_refinement",
        model_name="openai/gpt-5.4",
        reasoning_effort="none",
        prompting_scheme="sociodemographic",
        prompt_reasoning=False,
        llm_aggregation_priors=False,
        # todo: increase to 100
        num_of_samples=100,
        max_number_attempts=10,
        # todo: change threshold
        income_threshold=10_000,
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
