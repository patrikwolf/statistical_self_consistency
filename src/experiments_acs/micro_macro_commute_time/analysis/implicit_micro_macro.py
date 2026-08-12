import os
import time
import numpy as np
import pandas as pd

from typing import Any
from experiments_acs.micro_macro_commute_time.config.loader import load_shared_micro_macro_config
from language_models.batch_prompting_direct import llm_direct_prompting
from prompting.assemble_acs_prompts import assemble_thresholded_commute_time_prompt, assemble_commute_time_micro_macro_prompt
from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from experiments_acs.split_analysis.standard_value_loader import load_standard_values
from utility.argument_parser import parse_arguments
from utility.hyperparameters import assert_decomposition_validity
from utility.time_helper import get_experiment_timestamp


def main(survey_df: pd.DataFrame,
         cfg: MinimalExperimentConfig,
         shard_id: int,
         commute_time: int,
         set_commute_nan_to_zero: bool,
         mtm_template_filename: str,
         template_direct_root: str,
         timestamp: str | None = None
         ) -> None:
    filename = os.path.basename(__file__)
    print(f"Running {filename}")
    print(f"Model: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}")
    print(f"Commute time threshold: {commute_time} minutes")
    print(f"set_commute_nan_to_zero: {set_commute_nan_to_zero}\n")

    assert cfg.prompting_scheme == "sociodemographic", "Currently, only 'sociodemographic' prompting scheme is implemented."

    # Start timer
    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name, timestamp=timestamp)

    # File prefix
    if shard_id == -1:
        prefix = ""
    else:
        prefix = f"shard_{shard_id}__"

    # Initialize results
    results: dict[str, Any] = {
        "filename": filename,
        "timestamp": f"{date}__{timestamp}",
        "shard": prefix.rstrip("_"),
        "model": cfg.model_name,
        "reasoning_effort": cfg.reasoning_effort,
        "prompting_scheme": cfg.prompting_scheme,
        "prompt_reasoning": cfg.prompt_reasoning,
        "llm_aggregation_priors": cfg.llm_aggregation_priors,
        "survey_name": cfg.survey_name,
        "survey_year": cfg.survey_year,
        "set_nan_to_zero": cfg.set_nan_to_zero,
        "num_of_samples": cfg.num_of_samples,
        "max_number_attempts": cfg.max_number_attempts,
        "income_threshold": cfg.income_threshold,
        "set_commute_nan_to_zero": set_commute_nan_to_zero,
        "commute_time": commute_time,
        "experiment_folder": experiment_folder,
        "weighted": cfg.weighted,
        "prompt_template": mtm_template_filename,
        "income_greater_than_threshold": cfg.income_greater_than_threshold,
    }

    # Compute ground truth probability of commute time > threshold
    ground_truth_probability = compute_ground_truth_prob(
        survey_df=survey_df,
        target_threshold=commute_time,
        set_commute_nan_to_zero=set_commute_nan_to_zero,
        target_col="JWMNP",
        weight_col="PWGTP",
    )

    # Store ground truth
    results["ground_truth_probability"] = ground_truth_probability

    # Assemble prompt for direct root level estimate (no micro to macro reasoning)
    prompt = assemble_thresholded_commute_time_prompt(
        filters=[],
        commute_time=commute_time,
        prompting_scheme=cfg.prompting_scheme,
        filtered_template_filename="",
        no_filter_template_filename=template_direct_root,
    )

    # Direct prompting
    print("Direct root level estimate (no micro to macro)")
    log_direct_root = llm_direct_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Save LLM output in results file
    save_as_json(
        data=log_direct_root,
        experiment=experiment_folder,
        filename=f"{prefix}direct_root_llm.json",
        timestamp=False)

    # Store results in dictionary
    llm_prediction_list = [log_direct_root[f"run_{idx}"]["prediction"] for idx in range(cfg.num_of_samples)]
    results["direct"] = {
        "llm_prompt": prompt,
        "llm_prediction_list": llm_prediction_list,
        "llm_prediction_avg": float(np.mean(llm_prediction_list)),
    }

    # Assemble prompt for micro to macro reasoning
    prompt = assemble_commute_time_micro_macro_prompt(
        commute_time=commute_time,
        prompting_scheme=cfg.prompting_scheme,
        template_filename=mtm_template_filename,
    )

    # LLM prediction of commute time > threshold
    print("Micro to macro reasoning (with implicit partitioning)")
    log_direct_reasoning = llm_direct_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Save LLM output in results file
    save_as_json(
        data=log_direct_reasoning,
        experiment=experiment_folder,
        filename=f"{prefix}marco_micro_reasoning.json",
        timestamp=False)

    # LLM predictions
    llm_prediction_list = [log_direct_reasoning[f"run_{idx}"]["prediction"] for idx in range(cfg.num_of_samples)]

    # Add to main results dict
    results["micro_macro"] = {
        "llm_prompt": prompt,
        "llm_prediction_list": llm_prediction_list,
        "llm_prediction_avg": float(np.mean(llm_prediction_list)),
    }

    # Log to JSON & CSV
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"{prefix}{cfg.summary_results_file}",
        timestamp=False)
    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "model", "num_of_samples", "income_threshold", "commute_time"]
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


def compute_ground_truth_prob(
        survey_df: pd.DataFrame,
        target_threshold: int,
        set_commute_nan_to_zero: bool,
        target_col: str = "JWMNP",
        weight_col: str = "PWGTP",
) -> float:
    # Filtering
    df_copy = survey_df.copy()
    if set_commute_nan_to_zero:
        # Set commute time for workers with NaN commute time to zero -> not a worker or worker who worked at home
        filtered_survey_df = df_copy
        filtered_survey_df[target_col] = filtered_survey_df[target_col].fillna(0)
        print(f"  ----> Set all NaN values in {target_col} column to zero")
    else:
        # Drop NaN entries
        filtered_survey_df = df_copy.dropna(subset=[target_col])
        nan_share = 1 - filtered_survey_df.shape[0] / survey_df.shape[0]
        print(f"  ----> Filtered out all NaN values in {target_col} column ({100 * nan_share:.0f}%)")

    # Compute total weighted count
    total_weight = filtered_survey_df[weight_col].sum()

    # Zero weight
    if total_weight == 0:
        print("    ---> WARNING: Total weight is zero after filtering.")
        return 0

    # Compute probability of target > threshold
    target_after_threshold = filtered_survey_df[filtered_survey_df[target_col] > target_threshold]
    thresholded_weight = target_after_threshold[weight_col].sum()

    return float(thresholded_weight / total_weight)


if __name__ == "__main__":
    # Hyperparameters
    seeds = [42]
    standard_values = load_standard_values()
    decomposition_attributes = standard_values["decomposition_attributes"]
    commute_time_thresholds = [10, 20, 30, 40, 50, 60, 80, 100, 120]

    # todo: toggle
    set_commute_nan_to_zero = True

    # Parse arguments
    shard_id, shard, datetime = parse_arguments()
    print(f"Shard ID: {shard_id} ({shard_id + 1} out of {len(commute_time_thresholds)})")

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
        llm_aggregation_priors=shared_config["llm_aggregation_priors"],
        num_of_samples=shared_config["num_of_samples"],
        max_number_attempts=shared_config["max_number_attempts"],
        income_threshold=shared_config["income_threshold"],
        weighted=shared_config["weighted"],
        income_greater_than_threshold=shared_config["income_greater_than_threshold"],
        #
        experiment_name="micro_macro_commute_time",
        log_csv="micro_macro_commute_time_results.csv",
    )

    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=cfg.survey_year, set_nan_to_zero=cfg.set_nan_to_zero)
    print(f"Successfully loaded ACS data from year {cfg.survey_year}...")

    # Assert validity of decomposition attributes
    assert_decomposition_validity(decomposition_attributes=decomposition_attributes, attribute_dict=attribute_dict)

    if set_commute_nan_to_zero:
        # Set commute time for workers with NaN commute time to zero -> not a worker or worker who worked at home
        mtm_template_filename = "ACS_commute_time_micro_macro_v1.txt"
        template_direct_root = "ACS_commute_time_no_filters_v1.txt"
    else:
        # Only consider workers who commute to a workplace (with non-NaN commute time)
        mtm_template_filename = "ACS_commute_time_micro_macro_working_population.txt"
        template_direct_root = "ACS_commute_time_no_filters_working_population.txt"

    # Run main experiment
    main(survey_df=survey_df,
         cfg=cfg,
         shard_id=shard_id,
         timestamp=datetime,
         set_commute_nan_to_zero=set_commute_nan_to_zero,
         commute_time=commute_time_thresholds[shard_id],
         mtm_template_filename=mtm_template_filename,
         template_direct_root=template_direct_root,
         )
