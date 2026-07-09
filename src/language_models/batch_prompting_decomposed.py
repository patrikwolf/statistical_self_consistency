import pandas as pd
import numpy as np

from typing import Literal
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from experiments_acs.filtering.filter import Filter
from language_models.batch_prompting_direct import llm_direct_prompting
from language_models.prior_estimation import get_individual_llm_prior_estimate
from data_loader_acs.compute_prior import compute_prior
from experiments_acs.filtering.filter_definitions import extend_generic_filter
from prompting.assemble_acs_prompts import assemble_income_prompt


def llm_probability_decomposition(
        model_name: str,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        base_filters: list[Filter],
        base_filtered_survey_df: pd.DataFrame,
        decomposition_attributes: list[str],
        attribute_dict: dict,
        weighted: bool,
        income_threshold: int,
        llm_aggregation_priors: bool,
        num_of_samples: int,
        template_filename,
        no_filter_template_filename,
        income_greater_than_threshold: bool,
        weight_col: str = "PWGTP",
        max_number_attempts: int = 5,
        extract_confidence: bool = False,
):
    # Load model and tokenizer if necessary
    if model_name.startswith("Qwen"):
        assert model is not None, f"{model_name} requires model to be provided"
        assert tokenizer is not None, f"{model_name} requires tokenizer to be provided"

    # Compute prior probabilities for all combinations using groupby
    grouped = compute_prior(
        survey_df=base_filtered_survey_df,
        decomposition_attributes=decomposition_attributes,
        weighted=weighted,
        weight_col=weight_col,
    )

    assert np.abs(np.sum(grouped["prior"]) - 1.0) < 1e-5, "Prior probabilities do not sum to 1."

    # Iterate over all combinations
    results = {}
    for idx, row in tqdm(grouped.iterrows(), total=len(grouped)):
        print(f"Decomposition group {idx + 1} / {len(grouped)}")
        value_combination = [row[attr] for attr in decomposition_attributes]
        prior = row["prior"]

        # Filters for this combination
        decomposition_filters = [
            extend_generic_filter(
                filter_desc=Filter(attribute=attr, values=[value]),
                attribute_dict=attribute_dict
            ) for attr, value in zip(decomposition_attributes, value_combination)
        ]
        filters_for_combination = base_filters + decomposition_filters

        if llm_aggregation_priors:
            print("Computing LLM prior estimate for this subpopulation...")
            log_unnormalized_llm_priors = get_individual_llm_prior_estimate(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                reasoning_effort=reasoning_effort,
                num_of_samples=num_of_samples,
                max_number_attempts=max_number_attempts,
                base_filters=base_filters,
                decomposition_filters=decomposition_filters,
            )
        else:
            log_unnormalized_llm_priors = None

        # Assemble prompt
        prompt = assemble_income_prompt(
            filters=filters_for_combination,
            income_threshold=income_threshold,
            filtered_template_filename=template_filename,
            no_filter_template_filename=no_filter_template_filename,
            prompting_scheme=prompting_scheme,
            income_greater_than_threshold=income_greater_than_threshold,
        )

        # Direct prompting on subgroup
        print("Computing LLM income estimate for subpopulation...")
        log_direct = llm_direct_prompting(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            reasoning_effort=reasoning_effort,
            num_of_samples=num_of_samples,
            max_number_attempts=max_number_attempts,
            extract_confidence=extract_confidence,
        )

        # Save results in dict
        for run, value in log_direct.items():
            if run not in results:
                results[run] = {
                    "combinations": []
                }

            # Extract LLM-based prior estimate
            if log_unnormalized_llm_priors:
                llm_prior_output = log_unnormalized_llm_priors[run]["output"]
                llm_prior = log_unnormalized_llm_priors[run]["prediction"]
            else:
                llm_prior_output = None
                llm_prior = None

            # Add entry to list
            results[run]["combinations"].append({
                "value_combination": value_combination,
                "survey_prior_probability": prior,
                "unnormalized_llm_prior_probability": llm_prior,
                "unnormalized_llm_prior_output": llm_prior_output,
                "prediction": value["prediction"],
                "llm_output": value["output"]
            })

    # Normalize LLM priors over base-filtered subpopulation to get conditional prior of "subgroup given base filters"
    if llm_aggregation_priors:
        for run, value in results.items():
            unnormalized_priors = [comb["unnormalized_llm_prior_probability"] for comb in value["combinations"]]
            results[run]["aggregated_unnormalized_llm_prior"] = sum(unnormalized_priors)

    # Compute aggregated prediction
    for run, value in results.items():
        aggregated_prediction = 0
        aggregated_prior = 0
        combinations = value["combinations"]
        for comb in combinations:
            if llm_aggregation_priors:
                prior = comb["unnormalized_llm_prior_probability"] / value["aggregated_unnormalized_llm_prior"]
            else:
                prior = comb["survey_prior_probability"]
            aggregated_prediction += prior * comb["prediction"]
            aggregated_prior += prior
        results[run]["aggregated_prediction"] = aggregated_prediction
        results[run]["aggregated_normalized_prior"] = aggregated_prior

    return results
