import json
import numpy as np

from openai import OpenAI
from decision_models.utility.read_questions import read_filtered_questions
from utility.completion_postprocessing import extract_number
from utility.secret_config_helper import load_secret_config


def multi_prompting(
        model: str,
        question: str,
        output_format: str,
        num_runs: int,
) -> tuple[list, float]:
    estimates = []
    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}...")
        prob_estimate, output = prompt_language_model(
            model=model,
            question=question,
            output_format=output_format
        )
        estimates.append(prob_estimate)

    # Compute the average of the estimates
    average = float(np.mean(estimates))

    return estimates, average


def prompt_language_model(
        model: str,
        question: str,
        output_format: str,
) -> tuple[float, dict]:
    # Load API key
    secret_config = load_secret_config()
    api_key = secret_config["open_router_key_celestine"]

    # Client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Message list
    messages = [
        {
            "role": "user",
            "content": question + output_format
        }
    ]

    # Create chat completion
    print(f"   ---> Prompting model '{model}'...")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    if completion.choices is None:
        print(f"   ---> WARNING: No choices returned by model. Response: {completion}")
    else:
        # Response
        response = completion.choices[0].message.content

    # Collect results
    result = {
        "model": model,
        "prompt": question,
        "response": response,
    }

    # Extract estimate
    prob_estimate, _ = extract_number(text=response)

    if prob_estimate is None:
        print(f"   ---> WARNING: No valid probability estimate extracted from model response. Response: {response}")
        raise ValueError("No valid probability estimate extracted.")

    return prob_estimate, result


if __name__ == "__main__":
    # Model
    model = "openai/gpt-6-luna"

    # Read filtered questions
    questions = read_filtered_questions()
    q_item = questions[0]  # Limit to first 3 questions for testing

    # Extract questions
    parent = q_item["root"]["question"]
    left_child = q_item["children"][0]["question"]
    right_child = q_item["children"][1]["question"]
    split = q_item["split"]["question"]

    # todo: select question
    question = split

    # Output format
    output_format = """

Output requirements (strict):
- Return a single real number between 0 and 1 (inclusive).
- Format the answer exactly as: [[p]] where p is your probability estimate (for example: [[0.5]]).
"""

    # Example usage
    prob_estimate, output = prompt_language_model(
        model=model,
        question=question,
        output_format=output_format,
    )

    print(f"Estimate: {prob_estimate:.3f}")
    print("Output:")
    print(json.dumps(output, indent=4))

    # Example usage
    estimates, average = multi_prompting(
        model=model,
        question=question,
        output_format=output_format,
        num_runs=2)

    print(f"Estimates: {estimates}")
    print(f"Average: {average}")
