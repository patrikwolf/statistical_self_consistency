import time
from typing import Literal

import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.experiment_config import ExperimentConfig
from data_loader_acs.compute_prior import compute_prior
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.filtering.filter import Filter
from experiments_acs.filtering.filter_definitions import extend_generic_filter, construct_base_filters
from experiments_acs.filtering.filter_survey_data import filter_survey_data
from language_models.prior_estimation import get_individual_llm_prior_estimate
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def compute_unnormalized_llm_priors(
        model_name: str,
        model: AutoModelForCausalLM | None,
        tokenizer: AutoTokenizer | None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        num_of_samples: int,
        max_number_attempts: int,
        grouped: pd.DataFrame,
        attribute_dict: dict,
        base_filters: list[Filter],
        decomposition_attributes: list[str],
) -> dict:

    llm_results = {}
    for idx, row in tqdm(grouped.iterrows(), total=len(grouped)):
        print(f"Decomposition group {idx + 1} / {len(grouped)}")

        label = "__".join([f"{attr}-{row[attr]}" for attr in decomposition_attributes])
        value_combination = [row[attr] for attr in decomposition_attributes]

        # Filters for this combination
        decomposition_filters = [
            extend_generic_filter(
                filter_desc=Filter(attribute=attr, values=[value]),
                attribute_dict=attribute_dict
            ) for attr, value in zip(decomposition_attributes, value_combination)
        ]

        unnormalized_llm_priors = get_individual_llm_prior_estimate(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            reasoning_effort=reasoning_effort,
            num_of_samples=num_of_samples,
            max_number_attempts=max_number_attempts,
            base_filters=base_filters,
            decomposition_filters=decomposition_filters,
        )

        # Add to results
        for run, value in unnormalized_llm_priors.items():
            if run not in llm_results:
                llm_results[run] = {}

            llm_results[run][label] = {
                "gt_prior": row["prior"],
                "unnormalized_llm_results": value,
            }

    return llm_results


def main(cfg: ExperimentConfig,
         income_col: str = "PINCP",
         weight_col: str = "PWGTP"
         ) -> dict:

    print(f"\nIncome threshold: {cfg.income_threshold} USD")
    print(f"Model: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}")
    print(f"Income above threshold: {cfg.income_threshold}\n")

    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=cfg.survey_year, set_nan_to_zero=cfg.set_nan_to_zero)
    print(f"Successfully loaded ACS data from year {cfg.survey_year}...")

    # Extend base filters
    base_filters = construct_base_filters(filter_descriptions=cfg.base_filters,
                                          decomposition_attributes=cfg.decomposition_attributes,
                                          attribute_dict=attribute_dict)

    # Filter data
    base_filtered_data, _ = filter_survey_data(survey_df=survey_df,
                                               attribute_dict=attribute_dict,
                                               filters=base_filters,
                                               target_attribute=income_col)

    # Compute ground-truth priors
    grouped = compute_prior(
        survey_df=base_filtered_data,
        decomposition_attributes=cfg.decomposition_attributes,
        weighted=cfg.weighted,
        weight_col=weight_col,
    )

    # Load model and tokenizer if necessary
    if cfg.model_name.startswith("Qwen"):
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype="auto",
            device_map="auto"
        )
        model.eval()
    else:
        model = None
        tokenizer = None

    # Compute unnormalized LLM priors
    llm_results = compute_unnormalized_llm_priors(
        model_name=cfg.model_name,
        model=model,
        tokenizer=tokenizer,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
        grouped=grouped,
        attribute_dict=attribute_dict,
        base_filters=base_filters,
        decomposition_attributes=cfg.decomposition_attributes,
    )

    # Normalize LLM results
    normalized_results = {}
    for run, run_results in llm_results.items():
        ground_truth_priors_across_values = [item["gt_prior"] for item in run_results.values()]
        unnormalized_llm_priors_across_values = [item["unnormalized_llm_results"]["prediction"]
                                                 for item in run_results.values()]

        # Compute sum
        gt_sum = sum(ground_truth_priors_across_values)
        unnormalized_sum = sum(unnormalized_llm_priors_across_values)

        assert abs(gt_sum - 1) < 1e-8, f"Ground-truth priors in run '{run}' do not sum to 1!"

        for label, value_results in run_results.items():
            print(f"{label}: {value_results}")
            print("****")

            if label not in normalized_results:
                normalized_results[label] = {
                    "gt_prior_list": [],
                    "normalized_llm_prior_list": [],
                    "unnormalized_llm_prior_list": [],
                }

            unnormalized_llm_prior = value_results["unnormalized_llm_results"]["prediction"]

            # Add results to dictionary
            normalized_results[label]["gt_prior_list"].append(value_results["gt_prior"])
            normalized_results[label]["normalized_llm_prior_list"].append(unnormalized_llm_prior / unnormalized_sum)
            normalized_results[label]["unnormalized_llm_prior_list"].append(unnormalized_llm_prior)

    # Averaging and assertions
    for value_comb, value_results in normalized_results.items():
        assert len(set(value_results["gt_prior_list"])) == 1, f"Ground-truth prior must be unique for values {value_comb}."
        normalized_results[value_comb]["gt_prior"] = value_results["gt_prior_list"][0]
        normalized_results[value_comb].pop("gt_prior_list")

        # Averaging
        normalized_results[value_comb]["avg_normalized_llm_prior"] = np.mean(value_results["normalized_llm_prior_list"])

    # More assertions
    for run in range(cfg.num_of_samples):
        run_wise_normalized_priors = [value_results["normalized_llm_prior_list"][run]
                                      for value_results in normalized_results.values()]
        assert (sum(run_wise_normalized_priors) - 1) < 1e-8, f"Normalized LLM priors in run '{run}' do not sum to 1!"

    # Combine stats and normalized results
    results = {
        "description": {
            "model_name": cfg.model_name,
            "reasoning_effort": cfg.reasoning_effort,
            "prompting_scheme": cfg.prompting_scheme,
            "survey_name": cfg.survey_name,
            "survey_year": cfg.survey_year,
            "num_of_samples": cfg.num_of_samples,
            "max_number_attempts": cfg.max_number_attempts,
            "base_filters": [f.serialize() for f in cfg.base_filters],
            "decomposition_attributes": cfg.decomposition_attributes,
        },
        "normalized_results": normalized_results,
    }

    # Save results
    prior_results_file = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=cfg.summary_results_file,
        timestamp=False)

    # Print
    print("\n" + "*" * 80)
    print(f"Results saved to: {prior_results_file}")
    print(f"Total time: {time.time() - start_time}")
    print("*" * 80)

    return results


if __name__ == "__main__":
    # Configuration
    cfg = ExperimentConfig(
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        weighted=True,
        base_filters=[
            Filter(attribute="SEX", values=[1], getter=extend_generic_filter)
        ],
        decomposition_attributes=["MAR"],
        model_name="openai/gpt-5.4",
        reasoning_effort="none",
        prompting_scheme="sociodemographic",
        num_of_samples=20,
        max_number_attempts=10,
        experiment_name="compare_llm_and_gt_priors"
    )

    # Main
    main(cfg=cfg)
