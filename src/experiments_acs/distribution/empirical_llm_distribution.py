import os
import time
import numpy as np
import pandas as pd

from typing import Any, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from config.minimal_experiment_config import MinimalExperimentConfig
from experiments_acs.filtering.filter_survey_data import filter_survey_data
from language_models.prompt_local_llm import prompt_llm
from language_models.prompt_unified import prompt_open_router_api
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from utility.completion_postprocessing import extract_number
from utility.time_helper import get_experiment_timestamp


def get_empirical_llm_distribution(
        survey_df: pd.DataFrame,
        attribute_dict: dict[str, Any],
        cfg: MinimalExperimentConfig,
        shard: str,
        timestamp: str | None,
        hyperparameters: dict,
        target_col: str,
        prompt: str,
        bins: np.ndarray,
        weight_col: str = "PWGTP",
) -> None:
    print(f"\nHyperparameters: {hyperparameters}")
    print(f"Model: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}\n")

    # Start timer
    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name,
                                                                  timestamp=timestamp)

    # Initialize results
    results: dict[str, Any] = {
        "filename": os.path.basename(__file__),
        "timestamp": f"{date}__{timestamp}",
        "shard": shard,
        "model": cfg.model_name,
        "reasoning_effort": cfg.reasoning_effort,
        "prompting_scheme": cfg.prompting_scheme,
        "prompt_reasoning": cfg.prompt_reasoning,
        "temperature": cfg.sampling_temperature,
        "survey_name": cfg.survey_name,
        "survey_year": cfg.survey_year,
        "set_nan_to_zero": cfg.set_nan_to_zero,
        "num_of_samples": cfg.num_of_samples,
        "max_number_attempts": cfg.max_number_attempts,
        "node_id": hyperparameters["decomposition_filters"]["node_id"],
        "parent_id": hyperparameters["decomposition_filters"]["parent_id"],
        "level": hyperparameters["decomposition_filters"]["level"],
        "filters": hyperparameters["decomposition_filters"]["filters"],
        "experiment_folder": experiment_folder,
        "weighted": cfg.weighted,
        "target_column": target_col,
        "target_description": attribute_dict[target_col]["description"],
        "prompt": prompt,
    }

    # Filter data
    filtered_data, sizes = filter_survey_data(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=hyperparameters["decomposition_filters"]["filters"],
        target_attribute=target_col,
    )

    # Compute weight
    weight_array = np.array(filtered_data[weight_col])
    total_weight = np.sum(weight_array)
    weighted_prior = sizes["weighted_prior"]

    # Load model and tokenizer
    if cfg.model_name.startswith("Qwen"):
        print(f"   ---> Locally evaluating model '{cfg.model_name}'...")
        if cfg.reasoning_effort != "none":
            raise ValueError(f"Reasoning effort {cfg.reasoning_effort} not compatible with local Qwen models! Set the "
                             f"reasoning effort to 'none'.")

        # Load model and tokenizer
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
        print(f"   ---> Evaluating model '{cfg.model_name}' on OpenRouter API...")

    # Get LLM predictions
    llm_results = {
        "predicted_target": [],
    }
    for idx in range(cfg.num_of_samples):
        print(f" ----> Evaluation {idx + 1} of {cfg.num_of_samples} samples...")
        # Prompt LLM
        target, result = prompt_income_and_extract(
            model_name=cfg.model_name,
            model=model,
            tokenizer=tokenizer,
            reasoning_effort=cfg.reasoning_effort,
            max_number_attempts=cfg.max_number_attempts,
            prompt=prompt,
            temperature=cfg.sampling_temperature,
        )

        # Store result
        llm_results["predicted_target"].append(target)

    # Post-processing
    target_array = np.array(llm_results["predicted_target"])
    counts, bin_edges = np.histogram(target_array, bins=bins, density=True)
    mask = (target_array < bin_edges[0]) | (target_array > bin_edges[-1])
    probability_mass = 1 - len(target_array[mask]) / len(target_array)

    # Compute mean and variance
    target_mean = np.mean(target_array)
    target_std = np.std(target_array)

    # Store data about distribution
    results["total_weight"] = float(total_weight)
    results["weighted_prior"] = float(weighted_prior)
    results["predicted_target"] = target_array.tolist()
    results["counts"] = counts.tolist()
    results["bin_edges"] = bin_edges.tolist()
    results["probability_mass"] = probability_mass
    results["target_mean"] = target_mean
    results["target_std"] = target_std

    # Pre-processing for logging
    results["filters"] = [f.serialize() for f in results["filters"]]

    # Log to JSON & CSV
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"{shard}__{cfg.summary_results_file}",
        timestamp=False)

    # Remove more attributes
    results.pop("bin_edges", None)
    results.pop("counts", None)
    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "shard", "node_id", "target_mean", "target_std", "model", "reasoning_effort",
            "num_of_samples", "temperature"]
    try:
        print_tabulated(filename=cfg.log_csv, cols=cols, head=10)
    except Exception as e:
        print("  ---> Cannot print logs...")
        print(f"  ---> Exception: {e}")

    end_time = time.time()

    # Print
    print("\n" + "*" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds = {(end_time - start_time) / 60:.2f} minutes")
    print("*" * 80)

    return


def prompt_income_and_extract(
        model_name: str,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        max_number_attempts: int,
        prompt: str,
        temperature: float,
) -> tuple[float, dict]:

    # Initialization
    income_prediction = None
    result = {}

    for k in range(max_number_attempts):
        # Generate LLM response
        if model_name.startswith("Qwen"):
            assert reasoning_effort == "none", (f"Reasoning effort {reasoning_effort} not compatible with local "
                                                f"Qwen models! Set the reasoning effort to 'none'.")
            result = prompt_llm(model=model, tokenizer=tokenizer, prompt=prompt, logprobs=False, temperature=temperature)
        else:
            result = prompt_open_router_api(model=model_name, prompt=prompt, reasoning_effort=reasoning_effort,
                                            logprobs=False, temperature=temperature)

        # Parse response
        income_prediction, _ = extract_number(text=result["response"])

        if income_prediction is not None:
            break
        else:
            print(
                f"    ---> WARNING: Could not extract probability estimate or valid confidence interval from response."
                f" Attempt {k + 1}/{max_number_attempts}...")
            print(f"    ---> Response: {result['response']}")

    # Ensure we have a probability estimate
    assert income_prediction is not None, (f"Could not extract estimate from response "
                                           f"within {max_number_attempts} attempts.")

    return income_prediction, result
