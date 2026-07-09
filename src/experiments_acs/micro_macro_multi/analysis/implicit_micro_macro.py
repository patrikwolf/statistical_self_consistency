import os
import time
import numpy as np
import pandas as pd

from typing import Any
from experiments_acs.distribution.thresholded import compute_ground_truth_prob
from experiments_acs.micro_macro_multi.config.loader import load_shared_micro_macro_config
from language_models.batch_prompting_direct import llm_direct_prompting
from prompting.assemble_acs_prompts import assemble_thresholded_implicit_micro_macro_prompt
from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from utility.argument_parser import parse_arguments
from utility.hyperparameters import assert_decomposition_validity
from utility.time_helper import get_experiment_timestamp


def main(survey_df: pd.DataFrame,
         attribute_dict: dict,
         cfg: MinimalExperimentConfig,
         shard_id: int,
         mtm_template_filename: str,
         template_direct_root: str,
         timestamp: str | None = None
         ) -> None:
    filename = os.path.basename(__file__)
    print(f"Running {filename}")
    print(f"Model: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}")
    print(f"Income above threshold: {cfg.income_threshold}\n")

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
        "experiment_folder": experiment_folder,
        "weighted": cfg.weighted,
        "prompt_template": mtm_template_filename,
        "income_greater_than_threshold": cfg.income_greater_than_threshold,
    }

    # Compute ground truth probability of income > threshold
    ground_truth_results, _ = compute_ground_truth_prob(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=[],
        target_threshold=cfg.income_threshold,
        target_greater_than_threshold=cfg.income_greater_than_threshold,
        weighted=cfg.weighted,
        target_col="PINCP",
        weight_col="PWGTP",
    )

    # Store ground truth
    results["ground_truth_probability"] = ground_truth_results["high_income_probability"]
    results["ground_truth_prior"] = ground_truth_results["prior"]

    """
    # Assemble prompt for direct root level estimate (no micro to macro reasoning)
    prompt = assemble_income_prompt(
        filters=[],
        income_threshold=cfg.income_threshold,
        filtered_template_filename="",
        no_filter_template_filename=template_direct_root,
        prompting_scheme=cfg.prompting_scheme,
        income_greater_than_threshold=cfg.income_greater_than_threshold,
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
    results["llm_direct_root"] = llm_prediction_list
    results["llm_prediction_avg"] = np.mean(llm_prediction_list)
    results["llm_prediction_std"] = np.std(llm_prediction_list)
    """

    # Assemble prompt for micro to macro reasoning
    prompt = assemble_thresholded_implicit_micro_macro_prompt(
        income_threshold=cfg.income_threshold,
        filtered_template_filename=mtm_template_filename,
        prompting_scheme=cfg.prompting_scheme,
    )

    # LLM prediction of income > threshold
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
        "llm_prediction_std": float(np.std(llm_prediction_list))
    }

    # Log to JSON & CSV
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"{prefix}{cfg.summary_results_file}",
        timestamp=False)
    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "model", "num_of_samples", "income_threshold"]
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
    income_thresholds = [100, 20_000, 40_000, 60_000, 80_000]
    # income_thresholds = [100, 1_000, 10_000, 20_000, 40_000, 50_000, 60_000, 80_000]

    # Parse arguments
    shard_id, shard, datetime = parse_arguments()
    print(f"Shard ID: {shard_id} ({shard_id + 1} out of {len(income_thresholds)})")

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
        income_threshold=income_thresholds[shard_id],
        weighted=shared_config["weighted"],
        income_greater_than_threshold=shared_config["income_greater_than_threshold"],
        #
        experiment_name="micro_macro_implicit",
        log_csv="micro_macro_implicit_results.csv",
    )

    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=cfg.survey_year, set_nan_to_zero=cfg.set_nan_to_zero)
    print(f"Successfully loaded ACS data from year {cfg.survey_year}...")

    # Assert validity of decomposition attributes
    assert_decomposition_validity(decomposition_attributes=decomposition_attributes, attribute_dict=attribute_dict)

    # Run main experiment
    main(survey_df=survey_df,
         attribute_dict=attribute_dict,
         cfg=cfg,
         shard_id=shard_id,
         timestamp=datetime,
         mtm_template_filename="ACS_income_thresholded_implicit_micro_macro_v2.txt",
         template_direct_root=""      # "ACS_income_no_filters_v3.txt",
         )
