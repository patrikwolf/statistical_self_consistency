import os

from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from experiments_acs.micro_macro_multi.config.loader import load_shared_micro_macro_config
from experiments_acs.tree_income_threshold.analysis.binary_threshold_split import main
from utility.argument_parser import parse_arguments
from utility.hyperparameters import assert_decomposition_validity, process_hyperparameters


if __name__ == "__main__":
    print(f"Running script with name: {os.path.basename(__file__)}")

    # todo: change
    income_threshold = 100

    # Hyperparameters
    seeds = [42]
    standard_values = load_standard_values()
    decomposition_attributes = standard_values["decomposition_attributes"]

    # Configuration
    shared_config = load_shared_micro_macro_config()
    cfg = MinimalExperimentConfig(
        survey_name=shared_config["survey_name"],
        survey_year=shared_config["survey_year"],
        set_nan_to_zero=shared_config["set_nan_to_zero"],
        model_name=shared_config["model_name"],
        reasoning_effort=shared_config["reasoning_effort"],
        prompting_scheme=shared_config["prompting_scheme"],
        prompt_reasoning=shared_config["prompt_reasoning"],
        improved_age_desc=shared_config["improved_age_desc"],
        llm_aggregation_priors=shared_config["llm_aggregation_priors"],
        num_of_samples=shared_config["num_of_samples"],
        max_number_attempts=shared_config["max_number_attempts"],
        income_threshold=income_threshold,
        weighted=shared_config["weighted"],
        income_greater_than_threshold=shared_config["income_greater_than_threshold"],
        #
        experiment_name="micro_macro_aggregation_baseline",
        log_csv="micro_macro_aggregation_baseline_results.csv",
    )

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
