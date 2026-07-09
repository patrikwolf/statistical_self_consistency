from typing import Literal
from transformers import PreTrainedModel, PreTrainedTokenizer
from language_models.prompt_unified import prompt_llm_and_extract, prompt_llm_and_extract_json


def llm_direct_prompting(
        model_name: str,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        prompt: str,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        num_of_samples: int,
        max_number_attempts: int = 5,
        extract_confidence: bool = False,
) -> dict:
    # Load model and tokenizer if necessary
    if model_name.startswith("Qwen"):
        assert model is not None, f"{model_name} requires model to be provided"
        assert tokenizer is not None, f"{model_name} requires tokenizer to be provided"

    # LLM prediction of income > threshold
    log_direct = {}
    for idx in range(num_of_samples):
        print(f"Sample {idx + 1}/{num_of_samples}")
        # LLM direct prompting
        llm_prediction, llm_lower_bound, llm_upper_bound, llm_output = prompt_llm_and_extract(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            reasoning_effort=reasoning_effort,
            prompt=prompt,
            max_number_attempts=max_number_attempts,
            extract_confidence=extract_confidence,
        )

        # Store log
        log_direct[f"run_{idx}"] = {
            "prediction": llm_prediction,
            "lower_bound": llm_lower_bound,
            "upper_bound": llm_upper_bound,
            "output": llm_output
        }

    return log_direct


def llm_distribution_prompting(
        model_name: str,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        prompt: str,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        num_of_samples: int,
        max_number_attempts: int = 5,
        expected_json_keys: list[str] | None = None,
) -> dict:
    # Load model and tokenizer if necessary
    if model_name.startswith("Qwen"):
        assert model is not None, f"{model_name} requires model to be provided"
        assert tokenizer is not None, f"{model_name} requires tokenizer to be provided"

    # LLM prediction of income > threshold
    log_direct = {}
    for idx in range(num_of_samples):
        print(f"Sample {idx + 1}/{num_of_samples}")
        # LLM direct prompting
        llm_distribution, llm_output = prompt_llm_and_extract_json(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            reasoning_effort=reasoning_effort,
            prompt=prompt,
            max_number_attempts=max_number_attempts,
            expected_json_keys=expected_json_keys,
        )

        # Store log
        log_direct[f"run_{idx}"] = {
            "llm_distribution": llm_distribution,
            "output": llm_output
        }

    return log_direct
