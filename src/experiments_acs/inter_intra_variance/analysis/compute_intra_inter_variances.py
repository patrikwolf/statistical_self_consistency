import os
import time
import numpy as np
import pandas as pd

from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from utility.hyperparameters import assert_decomposition_validity, process_hyperparameters
from utility.argument_parser import parse_arguments
from experiments_acs.filtering.filter_survey_data import filter_survey_data
from utility.time_helper import get_experiment_timestamp


def main(
        cfg: MinimalExperimentConfig,
        shard: str,
        hyperparameters: dict,
        survey_df: pd.DataFrame,
        attribute_dict: dict,
        timestamp: str | None,
        target_col: str = "PINCP",
        weight_col: str = "PWGTP",
):
    print(f"Hyperparameters: {hyperparameters}")

    # Start timer
    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name,
                                                                  timestamp=timestamp)

    # Initialize results
    results = {
        "filename": os.path.basename(__file__),
        "timestamp": f"{date}__{timestamp}",
        "shard": shard,
        "survey_name": cfg.survey_name,
        "survey_year": cfg.survey_year,
        "set_nan_to_zero": cfg.set_nan_to_zero,
        "node_id": hyperparameters["decomposition_filters"]["node_id"],
        "parent_id": hyperparameters["decomposition_filters"]["parent_id"],
        "level": hyperparameters["decomposition_filters"]["level"],
        "filters": hyperparameters["decomposition_filters"]["filters"],
        "experiment_folder": experiment_folder,
        "weighted": cfg.weighted,
        "target_column": target_col,
        "target_description": attribute_dict[target_col]["description"],
    }

    # Filter data
    filtered_data, sizes = filter_survey_data(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=hyperparameters["decomposition_filters"]["filters"],
        target_attribute=target_col)

    weighted_prior = sizes["weighted_prior"]

    # Extract income and survey weights
    target_array = np.array(filtered_data[target_col])
    weight_array = np.array(filtered_data[weight_col])
    total_weight = np.sum(weight_array)

    # Apply weights to income
    target_array = np.repeat(target_array, weight_array.astype(int))

    print(f"Income array of length {len(target_array)}: {target_array}")
    print(f"Mean income: {target_array.mean()} USD")
    print(f"Variance in income: {target_array.var()} USD")

    # Collect results
    results["weighted_prior"] = weighted_prior
    results["total_weight"] = float(total_weight)
    results["target_mean"] = target_array.mean()
    results["target_var"] = target_array.var()

    # Pre-processing for logging
    results["filters"] = [f.serialize() for f in results["filters"]]

    # Log to JSON & CSV
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"{shard}__{cfg.summary_results_file}",
        timestamp=False)

    # Remove more attributes
    results.pop("bin_edges", None)
    results.pop("counts", None)
    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "shard", "node_id", "target_mean", "target_var"]
    try:
        print_tabulated(filename=cfg.log_csv, cols=cols, head=10)
    except Exception as e:
        print("  ---> Cannot print logs...")
        print(f"  ---> Exception: {e}")

    end_time = time.time()

    # Print
    print("\n" + "*" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds = {(end_time - start_time) / 60:.2f} minutes")
    print("*" * 80)

    return


if __name__ == "__main__":
    # Hyperparameters
    seeds = [42]
    standard_values = load_standard_values()
    decomposition_attributes = standard_values["decomposition_attributes"]

    # Configuration
    cfg = MinimalExperimentConfig(
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        experiment_name="inter_intra_variance",
        weighted=True,
        log_csv="inter_intra_variance_results.csv",
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

    # Run analysis
    main(
        cfg=cfg,
        shard=shard,
        hyperparameters=hyperparameters[shard],
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        timestamp=datetime,
    )
