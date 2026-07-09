import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from prompting.assemble_acs_prompts import assemble_sociodemographic_income_prompt
from config.constants import NUM_TOP_LOGPROBS, MAX_NEW_TOKENS
from utility.completion_postprocessing import extract_number


def prompt_llm(model: PreTrainedModel,
               tokenizer: PreTrainedTokenizer,
               prompt: str,
               logprobs: bool = False,
               temperature: float = 1.0) -> dict:
    """Prompt a language model and generate a response.

    Args:
        model: Pre-trained language model
        tokenizer: Pre-trained tokenizer
        prompt: Input prompt string
        logprobs: Whether to compute log probabilities for generated tokens
        temperature: The sampling temperature

    Returns:
        Dictionary containing:
            - prompt: Original prompt
            - response: Generated response text
            - logprobs: List of token log probabilities (if logprobs=True)
    """
    # Set model to evaluation mode
    model.eval()

    # Messages
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    )

    # Tokenize prompt
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Parameters
    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "output_scores": logprobs,
        "return_dict_in_generate": True,
        "do_sample": True,
        "temperature": temperature,
    }

    # Generate response
    with torch.no_grad():
        generated_outputs = model.generate(**model_inputs, **generate_kwargs)

    # Slicing off the prompt
    output_ids = generated_outputs.sequences[0][len(model_inputs.input_ids[0]):].tolist()

    # Decode response
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    # Compute logprobs for generated tokens (excluding the prompt)
    if logprobs:
        logprobs_list = compute_logprobs(
            tokenizer=tokenizer,
            model=model,
            output_ids=output_ids,
            generated_outputs=generated_outputs,
        )
    else:
        logprobs_list = None

    # Collect results
    result = {
        "prompt": prompt,
        "response": response,
        "logprobs": logprobs_list,
    }

    return result


def compute_logprobs(tokenizer, model, output_ids, generated_outputs) -> list:
    """Compute log probabilities for generated tokens.

    Args:
        tokenizer: Pre-trained tokenizer
        model: Pre-trained language model
        output_ids: List of generated token IDs
        generated_outputs: Output from model.generate()

    Returns:
        List of dictionaries, each containing:
            - token: Token string
            - logprob: Log probability of the token
            - top_logprobs: List of top-k token log probabilities
    """
    # Convert output IDs to tokens
    generated_tokens = tokenizer.convert_ids_to_tokens(output_ids)

    # Compute transition scores
    transition_scores = model.compute_transition_scores(
        sequences=generated_outputs.sequences,
        scores=generated_outputs.scores,
        normalize_logits=True
    )[0]

    # Compute top-k logprobs for each generated step
    top_logprobs = []
    for step_logits in generated_outputs.scores:
        # step_logits shape: [batch, vocab_size]
        log_probs = F.log_softmax(step_logits[0], dim=-1)
        topk_logprobs, topk_ids = torch.topk(log_probs, k=NUM_TOP_LOGPROBS)
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids.tolist())
        top_logprobs.append([
            {"token": token, "logprob": logprob}
            for token, logprob in zip(topk_tokens, topk_logprobs.tolist())
        ])

    # Combine all per-token info
    token_info = []
    for tok, logp, topk in zip(generated_tokens, transition_scores.tolist(), top_logprobs):
        token_info.append({
            "token": tok,
            "logprob": logp,
            "top_logprobs": topk
        })

    return token_info


if __name__ == "__main__":
    income_threshold = 40000
    filters = [
        {
            "attribute": "AGEP",
            "description": "age of the person",
            "values": [59],
            "value_map": lambda i: f"{i} years old",
        },
        {
            "attribute": "SCHL",
            "description": "highest educational attainment",
            "values": [20],
            "value_map": lambda i: {"20": "Associate's degree", "2": "Female"}.get(i, None),
        }
    ]
    prompt = assemble_sociodemographic_income_prompt(
        filters=filters,
        income_threshold=income_threshold,
        filtered_template_filename="ACS_income_v1.txt",
    )

    # Load model and tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )

    # Generate response
    logprobs = True
    temperature = 1.0
    result = prompt_llm(model=model, tokenizer=tokenizer, prompt=prompt, logprobs=logprobs, temperature=temperature)
    choice, _ = extract_number(result["response"])

    print(f"Response: {result['response']}\n")
    print(f"Choice: {choice}")
    if logprobs:
        print("Logprobs: ")
        print(result["logprobs"][0:2])
