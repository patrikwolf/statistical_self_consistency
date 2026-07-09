import os

from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.aggregation_refinement.split_selection.split_helper import load_decomposition_dict
from experiments_acs.tree_income_threshold.analysis.binary_threshold_split import main
from utility.argument_parser import parse_arguments
from utility.hyperparameters import assert_decomposition_validity, process_hyperparameters


if __name__ == "__main__":
    print(f"Running script with name: {os.path.basename(__file__)}")

    # Hyperparameters
    seeds = [42]
    split_name = "COW_MAR_ESR_AGEP"
    decomposition_attributes = load_decomposition_dict(split_name=split_name)

    # Model
    model = "anthropic/claude-sonnet-4.6"
    reasoning_effort = "none"

    print("*" * 80)
    print([decomposition_attributes[level]["attribute"] for level in decomposition_attributes.keys()])
    print("*" * 80)

    # Configuration
    cfg = MinimalExperimentConfig(
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        experiment_name="aggregation_refinement",
        model_name=model,
        reasoning_effort=reasoning_effort,
        prompting_scheme="sociodemographic",
        prompt_reasoning=False,
        llm_aggregation_priors=False,
        # todo: increase to 100
        num_of_samples=1,
        max_number_attempts=10,
        income_threshold=40_000,
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
