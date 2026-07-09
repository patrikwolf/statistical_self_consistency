import os
import numpy as np

from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from experiments_acs.distribution.experiment_ground_truth_distribution import compute_and_save_survey_distribution
from utility.hyperparameters import assert_decomposition_validity, process_hyperparameters
from utility.argument_parser import parse_arguments


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
        experiment_name="ground_truth_few_bins",
        weighted=True,
        log_csv="ground_truth_few_bins_results.csv",
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

    # Bins (equal probability mass)
    # bins_edges = np.array([-11_500, 2_500, 16_900, 33_000, 52_000, 85_000, 300_000])

    # Bins (equal width)
    # bins_edges = np.linspace(start=-11_500, stop=300_000, num=1 + 6)

    # Bins (equal width, no truncation)
    # bins_edges = np.array([-11_500, 2_800, 17_500, 34_200, 54_000, 90_000, 1_849_000])

    # 4 bins (after setting NaN to zero)
    bins_edges = np.array([-11500, 1, 25000, 60000, 1849000])

    # 2 bins (after setting NaN to zero)
    # bins_edges = np.array([-11500, 25000, 1849000])

    # Run main experiment
    compute_and_save_survey_distribution(survey_df=survey_df,
                                         cfg=cfg,
                                         shard=shard,
                                         timestamp=datetime,
                                         hyperparameters=hyperparameters[shard],
                                         attribute_dict=attribute_dict,
                                         target_col="PINCP",
                                         bins_edges=bins_edges,
                                         density=False,
                                         )
