import pandas as pd
import numpy as np

from typing import Any
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from config.experiment_config import ExperimentConfig
from data_loader_acs.compute_prior import compute_prior
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.filtering.filter import Filter
from experiments_acs.filtering.filter_definitions import construct_base_filters, extend_generic_filter
from experiments_acs.filtering.filter_survey_data import filter_survey_data
from file_logging.read_and_write_csv import log_to_csv
from file_logging.read_and_write_json import save_as_json
from prompting.assemble_acs_prompts import assemble_sociodemographic_income_prompt, assemble_generic_prompt, \
    assemble_income_prompt_with_ground_truth
from language_models.prompt_local_llm import prompt_llm
from utility.completion_postprocessing import extract_number


def load_and_filter_data(cfg: ExperimentConfig) -> tuple[list[Filter], dict, pd.DataFrame]:
    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=2024, set_nan_to_zero=cfg.set_nan_to_zero)

    # Filters
    base_filters = construct_base_filters(filter_descriptions=cfg.base_filters,
                                          decomposition_attributes=cfg.decomposition_attributes,
                                          attribute_dict=attribute_dict)

    # Filter data
    base_filtered_survey_df, _ = filter_survey_data(survey_df=survey_df, attribute_dict=attribute_dict,
                                                    filters=base_filters, target_attribute="PINCP")

    # Compute prior probabilities for all combinations
    grouped = compute_prior(
        survey_df=base_filtered_survey_df,
        decomposition_attributes=cfg.decomposition_attributes,
        weighted=cfg.weighted,
        weight_col="PWGTP",
    )

    # Check for 'nan' rows in prior probabilities
    if grouped[cfg.decomposition_attributes].isna().any().any():
        mask = grouped[cfg.decomposition_attributes].isna().any(axis=1)
        nan_mass = sum(grouped.loc[mask, "prior"].values)
        print(f"  ---> WARNING: Found 'nan' row in prior probabilities with probability mass: "
              f"{100 * nan_mass:.2f}%")
        if nan_mass > 0.05:
            raise ValueError(f"More than 5% probability mass on 'NaN' row: {100 * nan_mass:.2f}%")
    assert np.abs(np.sum(grouped["prior"]) - 1.0) < 1e-5, "Prior probabilities do not sum to 1."

    return base_filters, attribute_dict, grouped


def assemble_prompt(
        filters: list[Filter],
        prompt_properties: dict,
        improved_age_desc: bool
) -> str:
    # Case distinction based on question type
    if prompt_properties["question"] == "income_above_threshold":
        return assemble_sociodemographic_income_prompt(
            filters=filters,
            income_threshold=prompt_properties["threshold"],
        )
    elif prompt_properties["question"] == "income_above_threshold_with_ground_truth":
        return assemble_income_prompt_with_ground_truth(
            filters=filters,
            income_threshold=prompt_properties["threshold"],
            true_income=prompt_properties["true_income"],
        )
    elif prompt_properties["question"] == "car_vs_public_transport":
        return assemble_generic_prompt(
            filters=filters,
            template_filename="car_vs_public_transport_v1.txt",
            improved_age_desc=improved_age_desc,
        )
    elif prompt_properties["question"] == "basic_programming_ability":
        return assemble_generic_prompt(
            filters=filters,
            template_filename="basic_programming_v1.txt",
            improved_age_desc=improved_age_desc,
        )
    elif prompt_properties["question"] == "democrat":
        return assemble_generic_prompt(
            filters=filters,
            template_filename="democrat_v1.txt",
            improved_age_desc=improved_age_desc,
        )
    else:
        raise ValueError(f"Unknown prompt question type: {prompt_properties['question']}")


def llm_direct_prompting(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        max_number_attempts: int,
) -> tuple[float, dict]:
    # Prompting LLM until we get a valid response
    for attempt in range(max_number_attempts):
        response_dict = prompt_llm(model=model, tokenizer=tokenizer, prompt=prompt, logprobs=False)
        probability, _ = extract_number(response_dict["response"])

        if probability is not None and 0.0 <= probability <= 1.0:
            return probability, response_dict
        else:
            print(f"    ---> WARNING: Could not extract probability from response."
                  f" Attempt {attempt + 1}/{max_number_attempts}...")

    # All attempts failed
    raise ValueError(f"Failed to extract a valid probability after {max_number_attempts} attempts.")


def llm_decomposed_prompting(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribute_dict: dict,
        grouped_survey_df: pd.DataFrame,
        base_filters: list[Filter],
        decomposition_attributes: list[str],
        max_number_attempts: int,
        prompt_properties: dict,
) -> tuple[float, list[dict]]:
    # Initialization
    total_probability = 0.0
    combinations = []

    # Iterate over all combinations
    for _, row in tqdm(grouped_survey_df.iterrows(), total=len(grouped_survey_df)):
        value_combination = tuple(row[attr] for attr in decomposition_attributes)
        prior = row["prior"]

        # Filters for this combination
        filters_for_combination = base_filters + [
            extend_generic_filter(filter_desc=Filter(attribute=attr, values=[value]), attribute_dict=attribute_dict)
            for attr, value in zip(decomposition_attributes, value_combination)
        ]

        # Assemble prompt
        prompt = assemble_prompt(filters=filters_for_combination, prompt_properties=prompt_properties)

        # Compute LLM probability for each combination
        llm_prediction, llm_output = llm_direct_prompting(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_number_attempts=max_number_attempts,
        )

        # Update total probability
        total_probability += prior * llm_prediction

        # Save in combinations
        combinations.append({
            "value_combination": value_combination,
            "prior_probability": prior,
            "llm_prediction": llm_prediction,
            "llm_output": llm_output,
        })

    return total_probability, combinations


def main(cfg: ExperimentConfig, prompt_properties: dict, improved_age_desc: bool) -> None:
    # Load LLM model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype="auto",
        device_map="auto"
    )
    model.eval()

    # Load data
    base_filters, attribute_dict, grouped = load_and_filter_data(cfg)

    # Assemble prompt
    prompt_direct = assemble_prompt(
        filters=base_filters,
        prompt_properties=prompt_properties,
        improved_age_desc=improved_age_desc,
    )

    # Run experiment for specified number of samples
    results: dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "survey_name": cfg.survey_name,
        "survey_year": cfg.survey_year,
        "set_nan_to_zero": cfg.set_nan_to_zero,
        "model_name": cfg.model_name,
        "num_of_samples": cfg.num_of_samples,
        "max_number_attempts": cfg.max_number_attempts,
        "base_filters": [
            {k: (v.__name__ if callable(v) else v) for k, v in filter_definition.to_dict().items()}
            for filter_definition in cfg.base_filters
        ],
        "decomposition_attributes": cfg.decomposition_attributes,
        "weighted": cfg.weighted,
        "prompt": prompt_properties,
        "results": [],
    }
    for idx in range(cfg.num_of_samples):
        print(f"\n=== Sample {idx + 1}/{cfg.num_of_samples} ===")

        # Reference prediction
        prob_direct, _ = llm_direct_prompting(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_direct,
            max_number_attempts=cfg.max_number_attempts,
        )

        # Decomposed predictions
        prob_decomp, _ = llm_decomposed_prompting(
            model=model,
            tokenizer=tokenizer,
            attribute_dict=attribute_dict,
            grouped_survey_df=grouped,
            base_filters=base_filters,
            decomposition_attributes=cfg.decomposition_attributes,
            max_number_attempts=cfg.max_number_attempts,
            prompt_properties=prompt_properties,
        )

        # Add to results list
        results["results"].append({
            "sample_index": idx,
            "direct_prediction": prob_direct,
            "decomposed_prediction": prob_decomp,
        })

        # Log results
        print(f"Direct prediction: {prob_direct}")
        print(f"Decomposed prediction: {prob_decomp}")

    # Summarize results
    results["summary"] = {
        "direct_mean": np.mean([res["direct_prediction"] for res in results["results"]]),
        "direct_std": np.std([res["direct_prediction"] for res in results["results"]]),
        "decomposed_mean": np.mean([res["decomposed_prediction"] for res in results["results"]]),
        "decomposed_std": np.std([res["decomposed_prediction"] for res in results["results"]]),
    }

    # Save results
    date = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H-%M-%S")
    results_file = save_as_json(
        data=results,
        experiment=cfg.experiment_name,
        filename=f"results__{date}_{timestamp}",
        timestamp=False)
    print(f"\nResults saved to: {results_file}")

    # Prepare for logging
    results["direct_mean"] = results["summary"]["direct_mean"]
    results["direct_std"] = results["summary"]["direct_std"]
    results["decomposed_mean"] = results["summary"]["decomposed_mean"]
    results["decomposed_std"] = results["summary"]["decomposed_std"]
    results.pop("results")
    results.pop("summary")

    # Log to CSV
    log_file = log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)
    print("Results logged to CSV:", log_file)


if __name__ == "__main__":
    # Experiment
    experiment_name = "sanity_check_1_self_consistency"

    # Configuration
    cfg = ExperimentConfig(
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        experiment_name=experiment_name,
        # todo: change model name
        model_name="Qwen/Qwen3-0.6B",
        # todo: increase to 20
        num_of_samples=2,
        max_number_attempts=10,
        base_filters=[Filter(attribute="SEX", getter=extend_generic_filter, values=[2])],
        decomposition_attributes=["ESR"],
        log_csv=f"{experiment_name}.csv",
        weighted=True,
    )

    # Prompt properties
    prompt_properties = {
        "question": "democrat",
    }

    # Alternative prompt properties
    """
    prompt_properties = {
        "question": "income_above_threshold",
        "threshold": cfg.income_threshold,
    }

    prompt_properties = {
        "question": "car_vs_public_transport",
    }

    prompt_properties = {
        "question": "basic_programming_ability",
    }

    prompt_properties = {
        "question": "democrat",
    }
    """

    # Toggle improved age description
    improved_age_desc = False

    # Run main experiment
    main(cfg=cfg, prompt_properties=prompt_properties, improved_age_desc=improved_age_desc)
