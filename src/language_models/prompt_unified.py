from typing import Literal

from transformers import PreTrainedModel, PreTrainedTokenizer
from language_models.prompt_local_llm import prompt_llm
from language_models.prompt_open_router import prompt_open_router_api
from utility.completion_postprocessing import extract_estimates, extract_textual_response, extract_json_response


def prompt_llm_and_extract(
        model_name: str,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        prompt: str,
        max_number_attempts: int = 5,
        extract_confidence: bool = False,
) -> tuple[float, float, float, dict]:
    """Get LLM prediction of sociodemographic probability.

    Args:
        model_name: Name of the model to use for prediction
        reasoning_effort: Reasoning effort to use
        max_number_attempts: Maximum number of attempts to extract valid probability
        prompt: Optional pre-assembled prompt to use instead of generating a new one
        extract_confidence: Whether to extract confidence from response

    Returns:
        Tuple of (probability, lower_bound, upper_bound, result_dict) where result_dict contains full LLM output

    Raises:
        AssertionError: If no valid probability could be extracted after max attempts
    """
    # Initialization
    probability = None
    lower_bound = None
    upper_bound = None
    result = {}

    for k in range(max_number_attempts):
        # Generate response
        if model_name.startswith("Qwen"):
            assert reasoning_effort == "none", (f"Reasoning effort {reasoning_effort} not compatible with local "
                                                f"Qwen models! Set the reasoning effort to 'none'.")
            result = prompt_llm(model=model, tokenizer=tokenizer, prompt=prompt, logprobs=False)
        else:
            result = prompt_open_router_api(model=model_name, prompt=prompt, reasoning_effort=reasoning_effort, logprobs=False)

        # Parse response
        probability, lower_bound, upper_bound, _ = extract_estimates(text=result["response"],
                                                                     extract_confidence=extract_confidence)

        if reasoning_effort != "none" and result["reasoning"] is None:
            # Missing reasoning output
            print(f"    ---> WARNING: Reasoning trace was missing in LLM output, despite asking for {reasoning_effort} effort."
                  f" Attempt {k + 1}/{max_number_attempts}...")
            continue
        if (probability is not None) and (0.0 <= probability <= 1.0):
            if not extract_confidence:
                # Valid estimates
                break
            else:
                # Additional checks for confidence interval
                if ((lower_bound is not None)
                        and (upper_bound is not None)
                        and (0.0 <= lower_bound <= probability)
                        and (probability <= upper_bound <= 1.0)):
                    # Valid estimates
                    break
                else:
                    print(
                        f"    ---> WARNING: Could not extract valid confidence interval from response."
                        f" Attempt {k + 1}/{max_number_attempts}...")
        else:
            print(
                f"    ---> WARNING: Could not extract probability estimate or valid confidence interval from response."
                f" Attempt {k + 1}/{max_number_attempts}...")
            print(f"    ---> Response: {result['response']}")

    # Ensure we have a probability estimate
    assert probability is not None, (f"Could not extract probability estimate (or valid confidence interval) "
                                     f"from response within {max_number_attempts} attempts.")
    if reasoning_effort != "none":
        assert result["reasoning"] is not None, (f"Reasoning trace was missing in LLM output, "
                                                 f"despite asking for {reasoning_effort} effort.")

    return probability, lower_bound, upper_bound, result


def prompt_llm_and_extract_text(
        model_name: str,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        prompt: str,
        max_number_attempts: int,
) -> tuple[str, dict]:
    # Initialization
    text_response = None
    result = {}

    for k in range(max_number_attempts):
        # Generate response
        if model_name.startswith("Qwen"):
            assert reasoning_effort == "none", (f"Reasoning effort {reasoning_effort} not compatible with local "
                                                f"Qwen models! Set the reasoning effort to 'none'.")
            result = prompt_llm(model=model, tokenizer=tokenizer, prompt=prompt, logprobs=False)
        else:
            result = prompt_open_router_api(model=model_name, prompt=prompt, reasoning_effort=reasoning_effort, logprobs=False)

        # Parse response
        text_response, index = extract_textual_response(text=result["response"])

        if text_response is not None:
            # Valid estimate
            break
        else:
            print(f"    ---> WARNING: Could not extract estimate from response. Attempt {k + 1}/{max_number_attempts}...")
            print(f"    ---> Response: {result['response']}")

    # Ensure we have a valid estimate
    assert text_response is not None, f"Could not extract estimate from response within {max_number_attempts} attempts."

    return text_response, result


def prompt_llm_and_extract_json(
        model_name: str,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        prompt: str,
        max_number_attempts: int,
        expected_json_keys: list[str] | None = None,
) -> tuple[dict, dict]:
    # Initialization
    json_response = None
    result = {}

    for k in range(max_number_attempts):
        # Generate response
        if model_name.startswith("Qwen"):
            assert reasoning_effort == "none", (f"Reasoning effort {reasoning_effort} not compatible with local "
                                                f"Qwen models! Set the reasoning effort to 'none'.")
            result = prompt_llm(model=model, tokenizer=tokenizer, prompt=prompt, logprobs=False)
        else:
            result = prompt_open_router_api(model=model_name, prompt=prompt, reasoning_effort=reasoning_effort, logprobs=False)

        if result["response"] is None:
            print(f"    ---> WARNING: Result was 'None'. Attempt {k + 1}/{max_number_attempts}...")
            continue

        # Parse response
        json_response, index = extract_json_response(text=result["response"])

        if json_response is not None:
            # Sum of probabilities should be 1
            if abs(sum([val for val in json_response.values()]) - 1) > 1e-6:
                print(f"    ---> WARNING: Extracted JSON does not sum to 1. Attempt {k + 1}/{max_number_attempts}...")
                continue

            if expected_json_keys is not None:
                # Check if keys match
                if set(json_response.keys()) == set(expected_json_keys):
                    # Valid estimate
                    # print("    ---> COOL: All expected keys are present in the LLM-estimated distribution")
                    break
                else:
                    print(
                        f"    ---> WARNING: Extracted JSON does not contain expected keys {expected_json_keys}. "
                        f"Attempt {k + 1}/{max_number_attempts}...")
                    print(f"    ---> Extracted JSON: {json_response}")
            else:
                # Valid estimate
                break
        else:
            print(f"    ---> WARNING: Could not extract estimate from response. Attempt {k + 1}/{max_number_attempts}...")
            print(f"    ---> Response: {result['response']}")

    # Ensure we have a valid estimate
    assert json_response is not None, f"Could not extract JSON from response within {max_number_attempts} attempts."

    return json_response, result


if __name__ == "__main__":
    # Select model
    model = "openai/gpt-5.4"

    # Define prompt
    prompt = ("What is the probability that someone in France earns for than 40'000 EUR per year? Please answer "
              "format the answer exactly as: [[p]] where p is your probability estimate (for example: [[0.5]]).")

    # Reasoning effort
    reasoning_effort = "none"

    print("Send prompt to model via API call...")
    probability, lower_bound, upper_bound, result = prompt_llm_and_extract(
        model_name=model,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
    )

    print("*** Probability ***")
    print(probability)
    print("*** Lower bound ***")
    print(lower_bound)
    print("*** Upper bound ***")
    print(upper_bound)
    print("*** Response ***")
    print(result["response"])
    print("\n*** Reasoning ***")
    print(result["reasoning"])
    print("\n*** Logprobs ***")
    print(result["logprobs"])
