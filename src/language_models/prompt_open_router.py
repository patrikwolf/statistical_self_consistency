from typing import Literal

from openai import OpenAI
from utility.secret_config_helper import load_secret_config


def prompt_open_router_api(
        model: str,
        prompt: str,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        logprobs: bool = False,
        num_top_logprobs: int = 5,
        token=None,
        temperature: float = 1.0,
) -> dict:
    # Initialize OpenAI client for OpenRouter API
    if token is not None:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=token
        )
    else:
        secret_config = load_secret_config()
        api_key = secret_config["open_router_key_celestine"]
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    # Message list
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Create chat completion
    if logprobs:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            logprobs=True,
            top_logprobs=num_top_logprobs,
            temperature=temperature,
            extra_body={
                "reasoning": {
                    "effort": reasoning_effort
                }
            },
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            extra_body={
                "reasoning": {
                    "effort": reasoning_effort
                }
            },
        )

    # Check if answer is valid
    response = None
    logprobs_list = None
    reasoning = None
    if completion.choices is None:
        print(f"   ---> WARNING: No choices returned by model. Response: {completion}")
    else:
        # Response
        response = completion.choices[0].message.content

        # Check if logprobs are present
        if completion.choices[0].logprobs is not None:
            logprobs_list = completion.choices[0].logprobs.content

        # Reasoning
        if completion.choices[0].message.reasoning is not None:
            reasoning = completion.choices[0].message.reasoning

    # Collect results
    result = {
        "model": model,
        "prompt": prompt,
        "response": response,
        "reasoning": reasoning,
        "logprobs": logprobs_list,
    }

    return result


if __name__ == "__main__":
    # Select model
    """
    #model = "deepseek/deepseek-r1:free"
    #model = "qwen/qwen3-235b-a22b:free"
    #model = "meta-llama/llama-3.3-70b-instruct:free"
    """
    model = "openai/gpt-5.4"

    # Define prompt
    prompt = "How tall is the Eiffel tower?"

    # Reasoning effort
    reasoning_effort = "none"

    print("Send prompt to model via API call...")
    result = prompt_open_router_api(
        model=model,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        logprobs=False,
        temperature=1.0,
    )

    print("*** Response ***")
    print(result["response"])
    print("\n*** Reasoning ***")
    print(result["reasoning"])
    print("\n*** Logprobs ***")
    print(result["logprobs"])
