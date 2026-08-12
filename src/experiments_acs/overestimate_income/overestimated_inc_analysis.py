import os
import time
import numpy as np
import pandas as pd

from typing import Any
from experiments_acs.distribution.thresholded import compute_ground_truth_prob
from language_models.batch_prompting_direct import llm_direct_prompting
from prompting.assemble_acs_prompts import assemble_income_prompt
from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def main(survey_df: pd.DataFrame,
         attribute_dict: dict,
         cfg: MinimalExperimentConfig,
         shard: str,
         timestamp: str | None,
         hyperparameters: dict,
         template_filename: str,
         no_filter_template_filename: str,
         extract_confidence: bool,
         confidence: int | float = None,
         prompt_len_ablation: str = "none",
         fixed_num_of_attributes: int | None = None,
         ) -> dict:
    print(f"\nHyperparameters: {hyperparameters}")
    print(f"Model: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}")
    print(f"LLM-estimated priors: {cfg.llm_aggregation_priors}")
    print(f"Income above threshold: {cfg.income_threshold}")
    print(f"Number of samples: {cfg.num_of_samples}")
    print(f"Improved age description: {cfg.improved_age_desc}\n")

    assert cfg.prompting_scheme == "sociodemographic", "Currently, only 'sociodemographic' prompting scheme is implemented."

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
        "decomposition_attribute_list": hyperparameters["decomposition_attribute_list"],
        "node_id": hyperparameters["decomposition_filters"]["node_id"],
        "parent_id": hyperparameters["decomposition_filters"]["parent_id"],
        "level": hyperparameters["decomposition_filters"]["level"],
        "filters": hyperparameters["decomposition_filters"]["filters"],
        "experiment_folder": experiment_folder,
        "weighted": cfg.weighted,
        "prompt_template": template_filename,
        "prompt_template_no_filter": no_filter_template_filename,
        "improved_age_desc": cfg.improved_age_desc,
        "extract_confidence": extract_confidence,
        "confidence": f"{confidence:.0f}%" if confidence else None,
        "income_greater_than_threshold": cfg.income_greater_than_threshold,
    }

    # Filter dataset based on hyperparameters
    filters = hyperparameters["decomposition_filters"]["filters"]

    # Compute ground truth probability of income > threshold
    print("Computing ground truth results")
    ground_truth_results, _ = compute_ground_truth_prob(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=filters,
        target_threshold=cfg.income_threshold,
        target_greater_than_threshold=cfg.income_greater_than_threshold,
        weighted=cfg.weighted,
        target_col="PINCP",
        weight_col="PWGTP",
    )

    # Store ground truth
    results["ground_truth_probability"] = ground_truth_results["high_income_probability"]
    results["ground_truth_prior"] = ground_truth_results["prior"]

    # Assemble prompt
    prompt = assemble_income_prompt(
        filters=filters,
        income_threshold=cfg.income_threshold,
        filtered_template_filename=template_filename,
        no_filter_template_filename=no_filter_template_filename,
        extract_confidence=extract_confidence,
        confidence=confidence,
        prompting_scheme=cfg.prompting_scheme,
        income_greater_than_threshold=cfg.income_greater_than_threshold,
        improved_age_desc=cfg.improved_age_desc,
        prompt_len_ablation=prompt_len_ablation,
        fixed_num_of_attributes=fixed_num_of_attributes,
    )

    # LLM prediction of income > threshold
    print("\nComputing LLM-estimated conditionals")
    log_direct = llm_direct_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
        extract_confidence=extract_confidence,
    )

    # Store LLM direct predictions
    results["llm_prompt"] = prompt
    results["llm_prediction_list"] = [log_direct[f"run_{idx}"]["prediction"] for idx in range(cfg.num_of_samples)]
    results["llm_prediction_avg"] = float(np.mean(results["llm_prediction_list"]))
    results["llm_prediction_std"] = float(np.std(results["llm_prediction_list"]))
    if extract_confidence:
        results["llm_confidence_width_list"] = [log_direct[f"run_{idx}"]["upper_bound"]
                                                - log_direct[f"run_{idx}"]["lower_bound"]
                                                for idx in range(cfg.num_of_samples)]
        results["ground_truth_within_llm_confidence_interval"] = sum([
            log_direct[f"run_{idx}"]["lower_bound"] <= results["ground_truth"]
            <= log_direct[f"run_{idx}"]["upper_bound"]
            for idx in range(cfg.num_of_samples)])
        results["ground_truth_below_confidence_interval"] = sum([results["ground_truth"]
                                                                 < log_direct[f"run_{idx}"]["lower_bound"]
                                                                 for idx in range(cfg.num_of_samples)])
        results["ground_truth_above_confidence_interval"] = sum([results["ground_truth"]
                                                                 > log_direct[f"run_{idx}"]["upper_bound"]
                                                                 for idx in range(cfg.num_of_samples)])
        results["llm_confidence_width_avg"] = float(np.mean(results["llm_confidence_width_list"]))

    # Save LLM output in results file
    save_as_json(
        data=log_direct,
        experiment=experiment_folder,
        filename=f"{shard}__{cfg.direct_results_file}",
        timestamp=False)

    # Pre-processing for logging
    results["filters"] = [f.serialize() for f in results["filters"]]

    # Log to JSON & CSV
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"{shard}__{cfg.summary_results_file}",
        timestamp=False)
    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "shard", "model", "num_of_samples", "income_threshold",
            "llm_prediction_avg", "llm_prediction_std", "confidence", "node_id"]
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

    return results


if __name__ == "__main__":
    print(f"Running script with name: {os.path.basename(__file__)}")

    # Model
    model_name = "openai/gpt-5.4"

    # Threshold
    income_threshold = 40_000

    # Configuration
    cfg = MinimalExperimentConfig(
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        experiment_name="overestimated_inc_analysis",
        model_name=model_name,
        reasoning_effort="none",
        prompting_scheme="sociodemographic",
        prompt_reasoning=False,
        llm_aggregation_priors=False,
        num_of_samples=20,
        max_number_attempts=10,
        income_threshold=income_threshold,
        weighted=True,
        income_greater_than_threshold=True,
        log_csv="overestimated_inc_analysis_results.csv",
    )

    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=cfg.survey_year, set_nan_to_zero=cfg.set_nan_to_zero)
    print(f"Successfully loaded ACS data from year {cfg.survey_year}...")

    # Hyperparameters
    datetime = None
    hyperparameters = {
        "seed": 42,
        "decomposition_filters":
            {
                "node_id": 0,
                "parent_id": None,
                "level": "level_0",
                "filters": []
            },
        "decomposition_attribute_list": []
    }

    # Template
    template = "ACS_income_no_filters_v3.txt"
    # template = "ACS_income_above_16_years_v1.txt"

    # Run main experiment
    results = main(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        cfg=cfg,
        shard="none",
        timestamp=datetime,
        hyperparameters=hyperparameters,
        template_filename="",
        no_filter_template_filename=template,
        extract_confidence=False,
    )

    print(f"\nUsing template: {template}")
    print(f"Average prediction: {np.average(results['llm_prediction_list'])}")
