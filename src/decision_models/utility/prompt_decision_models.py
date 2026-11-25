import json
import requests
import laya
import torch
import numpy as np

from functools import lru_cache
from utility.secret_config_helper import load_secret_config


def multi_prompting(
        model: str,
        context: str,
        question: str,
        options: dict[str, str],
        num_runs: int,
        verbose: bool = False
) -> tuple[dict, dict]:
    estimates = []
    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}...")
        distribution, output = prompt_decision_model(
            model=model,
            context=context,
            question=question,
            options=options,
            verbose=verbose
        )
        estimates.append(distribution)

    # Compute the average of the estimates
    average = {
        key: float(np.mean([estimate[key] for estimate in estimates]))
        for key in estimates[0]
    }

    return estimates, average


def prompt_decision_model(
        model: str,
        context: str,
        question: str,
        options: dict[str, str],
        verbose: bool = False
) -> tuple[dict, dict]:
    if model == "convaiinnovations/laya":
        return prompt_laya(
            model=model,
            context=context,
            question=question,
            options=options,
            verbose=verbose,
        )
    else:
        return prompt_openrouter(
            model=model,
            context=context,
            question=question,
            options=options,
            verbose=verbose,
        )


def prompt_openrouter(
        model: str,
        context: str,
        question: str,
        options: dict[str, str],
        verbose: bool = False,
) -> tuple[dict, dict]:
    # Load API key
    secret_config = load_secret_config()
    api_key = secret_config["open_router_key_celestine"]

    # Create payload
    payload = {
        "model": model,
        "state": context,
        "questions": {
            "outcome": {
                "type": "choice",
                "instructions": question,
                "criteria": options,
            }
        },
    }

    if verbose:
        print(json.dumps(payload, indent=2))

    # Send request to the decision model API
    response = requests.post(
        "https://openrouter.ai/api/alpha/decisions",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=120,
    )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise requests.exceptions.HTTPError(
            f"{exc}\nResponse body: {response.text}",
            response=response,
            request=response.request,
        ) from exc

    # Extract JSON response
    result = response.json()

    distribution = result["answers"]["outcome"]["probabilities"]

    return distribution, result


@lru_cache(maxsize=1)
def load_laya(model: str):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return laya.load(model, device=device)


def prompt_laya(
        model: str,
        context: str,
        question: str,
        options: dict[str, str],
        verbose: bool = False,
) -> tuple[dict, dict]:
    agent = load_laya(model)

    if verbose:
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Options: {options}")

    # Prompt the model
    output = agent.predict(
        state=context,
        questions={
            "decision": {
                "type": "choice",
                "instructions": question,
                "criteria": options,
            }
        },
    )

    distribution = {
        key: float(value)
        for key, value in
        output["answers"]["decision"]["probabilities"].items()
    }

    return distribution, output


if __name__ == "__main__":
    # Model
    model = "inception/mercury-decide"
    # model = "convaiinnovations/laya"

    # Example usage
    distribution, output = prompt_decision_model(
        model=model,
        context="The Australian government awards no such funded development contract by December 31, 2028",
        question="What is the probability that Australia has a fully operational sovereign satellite navigation system "
                 "under its primary control, capable of independent navigation services for its territory and maritime "
                 "regions, by December 31, 2030, given that the Australian government awards no such funded development "
                 "contract by December 31, 2028? Estimate the probability under the specified context.",
        options={
            "yes": "Australia operates its own independent satellite navigation system by December 31, 2030.",
            "no": "Australia does not operate such a system by December 31, 2030.",
        },
        verbose=True,
    )

    print(distribution)
    print(output)
