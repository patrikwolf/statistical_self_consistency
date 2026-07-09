import os
import time
import numpy as np
import pandas as pd

from typing import Any
from config.minimal_experiment_config import MinimalExperimentConfig
from experiments_acs.distribution.empirical_survey_distribution import get_empirical_survey_distribution
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def compute_and_save_survey_distribution(
        survey_df: pd.DataFrame,
        cfg: MinimalExperimentConfig,
        shard: str,
        timestamp: str | None,
        hyperparameters: dict,
        attribute_dict: dict,
        target_col: str,
        bins_edges: np.ndarray,
        density: bool = True,
) -> None:
    print(f"Hyperparameters: {hyperparameters}")

    # Start timer
    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name,
                                                                  timestamp=timestamp)

    # Initialize results
    results: dict[str, Any] = {
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

    # Compute ground truth probability distribution
    target_array, total_weight, weighted_prior, counts, bin_edges, probability_mass = get_empirical_survey_distribution(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=hyperparameters["decomposition_filters"]["filters"],
        weighted=True,
        target_col=target_col,
        bins=bins_edges,
        density=density,
    )

    # Compute mean and variance
    target_mean = np.mean(target_array)
    target_std = np.std(target_array)

    # Store ground truth distribution
    results["total_weight"] = float(total_weight)
    results["weighted_prior"] = float(weighted_prior)
    results["counts"] = counts.tolist()
    results["bin_edges"] = bin_edges.tolist()
    results["captured_probability_mass"] = probability_mass
    results["target_mean"] = target_mean
    results["target_std"] = target_std

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
    cols = ["date", "time", "shard", "node_id", "target_mean", "target_std"]
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
