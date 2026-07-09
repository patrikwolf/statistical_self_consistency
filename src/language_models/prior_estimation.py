from typing import Literal
from transformers import PreTrainedModel, PreTrainedTokenizer
from experiments_acs.filtering.filter import Filter
from language_models.batch_prompting_direct import llm_direct_prompting, llm_distribution_prompting
from prompting.assemble_acs_prompts import assemble_individual_prior_estimation_prompt, assemble_joint_prior_estimation_prompt


def get_individual_llm_prior_estimate(
        model_name: str,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        num_of_samples: int,
        max_number_attempts: int,
        base_filters: list[Filter],
        decomposition_filters: list[Filter],
        template_filename: str = "ACS_prior_estimation_v1.txt",
) -> dict:

    # Assemble prompt
    prompt = assemble_individual_prior_estimation_prompt(
        base_filters=base_filters,
        decomposition_filters=decomposition_filters,
        template_filename=template_filename,
    )

    # Direct prompting on subgroup
    log_direct = llm_direct_prompting(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        num_of_samples=num_of_samples,
        max_number_attempts=max_number_attempts,
        extract_confidence=False,
    )

    return log_direct


def get_joint_llm_prior_estimates(
        model_name: str,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        num_of_samples: int,
        max_number_attempts: int,
        base_filters: list[Filter],
        filter_list: list[list[Filter]],
        improved_age_desc: bool,
        template_filename: str = "ACS_joint_prior_estimation_v1.txt",
) -> tuple[dict, list]:
    # Assemble prompt
    prompt, group_wise_filters = assemble_joint_prior_estimation_prompt(
        base_filters=base_filters,
        filter_list=filter_list,
        template_filename=template_filename,
        improved_age_desc=improved_age_desc,
    )

    # Direct prompting on subgroup
    log_direct = llm_distribution_prompting(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        num_of_samples=num_of_samples,
        max_number_attempts=max_number_attempts,
    )

    return log_direct, group_wise_filters
