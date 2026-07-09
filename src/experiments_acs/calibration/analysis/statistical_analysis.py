import itertools
import os
import time
import math
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.distribution.thresholded import compute_ground_truth_prob
from language_models.batch_prompting_decomposed import llm_probability_decomposition
from language_models.batch_prompting_direct import llm_direct_prompting
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from config.experiment_config import ExperimentConfig
from experiments_acs.filtering.filter_definitions import construct_base_filters
from prompting.assemble_acs_prompts import assemble_income_prompt
from utility.argument_parser import parse_arguments
from utility.time_helper import get_experiment_timestamp


def process_hyperparameters(
        seeds: list[int],
        income_thresholds: list[int],
) -> dict:
    """Process hyperparameters for the experiment.

    Returns:
        A dictionary containing hyperparameters for each shard.
    """
    hyperparameters = {}

    for idx, (seed, it) in enumerate(itertools.product(seeds, income_thresholds)):
        hyperparameters[f'shard_{idx}'] = {
            "seed": seed,
            "income_threshold": it,
        }

    return hyperparameters


def main(cfg: ExperimentConfig,
         shard_id: int,
         template_filename: str,
         no_filter_template_filename: str,
         timestamp: str | None = None) -> None:
    """Main function to run sociodemographic prompting experiment.

    Args:
        cfg: Experiment configuration.
        shard_id: Shard ID.
        timestamp: Timestamp string for loading existing experiment. Format: YYYY-MM-DD_HH-MM-SS
    """
    assert cfg.prompt_reasoning is False, ("Prompt-based reasoning is not implemented yet. "
                                           "Please set prompt_reasoning to False.")

    print(f"\nIncome threshold: {cfg.income_threshold} USD")
    print(f"Model: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}")
    print(f"Prompting scheme: {cfg.prompting_scheme}")
    print(f"Income above threshold: {cfg.income_threshold}\n")

    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name, timestamp=timestamp)

    # File prefix
    if shard_id == -1:
        prefix = ""
    else:
        prefix = f"shard_{shard_id}__"

    # Load survey dataset
    assert cfg.survey_name == "ACS", "Only ACS is supported for now."
    survey_df, attribute_dict = load_survey_data(year=cfg.survey_year, set_nan_to_zero=cfg.set_nan_to_zero)
    print("Successfully loaded ACS data...")

    # Filters
    base_filters = construct_base_filters(filter_descriptions=cfg.base_filters,
                                          decomposition_attributes=cfg.decomposition_attributes,
                                          attribute_dict=attribute_dict)

    # Initialize results
    results = {
        "filename": os.path.basename(__file__),
        "timestamp": f"{date}__{timestamp}",
        "shard": prefix.rstrip("_"),
        "model": cfg.model_name,
        "survey_name": cfg.survey_name,
        "survey_year": cfg.survey_year,
        "set_nan_to_zero": cfg.set_nan_to_zero,
        "reasoning_effort": cfg.reasoning_effort,
        "prompting_scheme": cfg.prompting_scheme,
        "prompt_reasoning": cfg.prompt_reasoning,
        "llm_aggregation_priors": cfg.llm_aggregation_priors,
        "num_of_samples": cfg.num_of_samples,
        "max_number_attempts": cfg.max_number_attempts,
        "income_threshold": cfg.income_threshold,
        "base_filters": base_filters,
        "decomposition_attributes": cfg.decomposition_attributes,
        "experiment_folder": experiment_folder,
        "weighted": cfg.weighted,
        "income_greater_than_threshold": cfg.income_greater_than_threshold,
    }

    # Compute ground truth probability of income > threshold
    ground_truth_results, base_filtered_survey_df = compute_ground_truth_prob(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=base_filters,
        target_threshold=cfg.income_threshold,
        weighted=cfg.weighted,
        target_greater_than_threshold=cfg.income_greater_than_threshold,
    )

    # Store ground truth
    results["ground_truth"] = ground_truth_results["high_income_probability"]

    # Load model and tokenizer if necessary
    if cfg.model_name.startswith("Qwen"):
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

    # Assemble prompt
    prompt = assemble_income_prompt(
        filters=base_filters,
        income_threshold=cfg.income_threshold,
        filtered_template_filename=template_filename,
        no_filter_template_filename=no_filter_template_filename,
        prompting_scheme=cfg.prompting_scheme,
        income_greater_than_threshold=cfg.income_greater_than_threshold,
    )

    # Direct prompting
    log_direct = llm_direct_prompting(
        model_name=cfg.model_name,
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Save LLM output log file
    llm_direct_results_file = save_as_json(
        data=log_direct,
        experiment=experiment_folder,
        filename=f"{prefix}{cfg.direct_results_file}",
        timestamp=False)

    # Store LLM direct predictions
    results["llm_prediction_list"] = [log_direct[f"run_{idx}"]["prediction"] for idx in range(cfg.num_of_samples)]
    results["llm_prediction_avg"] = float(np.mean(results["llm_prediction_list"]))
    results["llm_prediction_std"] = float(np.std(results["llm_prediction_list"]))
    results["llm_prediction_results_file"] = llm_direct_results_file

    # LLM with law of total probability
    print("\nLoTP decomposition\n")
    log_decomposed = llm_probability_decomposition(
        model_name=cfg.model_name,
        model=model,
        tokenizer=tokenizer,
        reasoning_effort=cfg.reasoning_effort,
        prompting_scheme=cfg.prompting_scheme,
        base_filters=base_filters,
        base_filtered_survey_df=base_filtered_survey_df,
        decomposition_attributes=cfg.decomposition_attributes,
        attribute_dict=attribute_dict,
        weighted=cfg.weighted,
        income_threshold=cfg.income_threshold,
        llm_aggregation_priors=cfg.llm_aggregation_priors,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
        template_filename=template_filename,
        no_filter_template_filename=no_filter_template_filename,
        income_greater_than_threshold=cfg.income_greater_than_threshold,
    )

    # Save LLM with law of total probability log file
    llm_law_of_total_prob_results_file = save_as_json(
        data=log_decomposed,
        experiment=experiment_folder,
        filename=f"{prefix}{cfg.decomposition_results_file}",
        timestamp=False)

    # Store LLM decomposed predictions
    results["llm_law_of_total_prob_list"] = [log_decomposed[f"run_{idx}"]["aggregated_prediction"]
                                             for idx in range(cfg.num_of_samples)]
    results["llm_law_of_total_prob_sum_of_priors"] = [log_decomposed[f"run_{idx}"]["aggregated_normalized_prior"]
                                                      for idx in range(cfg.num_of_samples)]
    results["llm_law_of_total_prob_avg"] = float(np.mean(results["llm_law_of_total_prob_list"]))
    results["llm_law_of_total_prob_std"] = float(np.std(results["llm_law_of_total_prob_list"]))
    results["llm_law_of_total_prob_results_file"] = llm_law_of_total_prob_results_file

    # Store conditional LLM estimates for extended plots
    results["llm_lotp_conditional_predictions"] = {}
    for run, res in log_decomposed.items():
        for combination in res["combinations"]:
            value_combination = combination["value_combination"]
            value_string = ""
            for value in value_combination:
                if isinstance(value, float):
                    if math.isnan(value):
                        value_string += "nan_"
                    else:
                        value_string += f"{value:.1f}_"
                elif isinstance(value, str):
                    value_string += f"{int(value)}_"
                else:
                    print(f"Unexpected type '{type(value)}' for value {value}")
            value_comb_string = "_".join(cfg.decomposition_attributes) + "__" + value_string[:-1]

            # Drop information
            combination.pop("unnormalized_llm_prior_probability")
            combination.pop("unnormalized_llm_prior_output")
            combination.pop("llm_output")

            if value_comb_string not in results["llm_lotp_conditional_predictions"]:
                results["llm_lotp_conditional_predictions"][value_comb_string] = [combination]
            else:
                results["llm_lotp_conditional_predictions"][value_comb_string].append(combination)

    # Prepare for logging
    results["base_filters"] = [f.serialize() for f in results["base_filters"]]

    # Log to JSON & CSV
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"{prefix}{cfg.summary_results_file}",
        timestamp=False)
    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "shard", "model", "reasoning_effort", "num_of_samples", "income_threshold",
            "income_greater_than_threshold", "llm_aggregation_priors", "ground_truth", "llm_prediction_avg",
            "llm_law_of_total_prob_avg", "llm_prediction_std", "llm_law_of_total_prob_std"]
    print_tabulated(filename=cfg.log_csv, cols=cols, head=10)

    # Print
    print("\n" + "*" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Total time: {time.time() - start_time}")
    print("*" * 80)


if __name__ == "__main__":
    # Hyperparameters
    seeds = [42]
    income_thresholds = [100, 5_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000, 200_000]
    hyperparameters = process_hyperparameters(
        seeds=seeds,
        income_thresholds=income_thresholds,
    )

    # Parse arguments
    shard_id, shard, datetime = parse_arguments()
    print(f"Shard ID: {shard_id} ({shard_id + 1} out of {len(hyperparameters)})")

    # Configuration
    cfg = ExperimentConfig(
        experiment_name="calibration_multiple_samples",
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        income_threshold=hyperparameters[shard]["income_threshold"],
        # todo: change model to Qwen3-8B
        model_name="openai/gpt-5.4",     # "Qwen/Qwen3-0.6B", "openai/gpt-5.4", "qwen/qwen3-8b"
        reasoning_effort="none",
        # todo: implement different prompting schemes
        prompting_scheme="sociodemographic",      # "persona", "sociodemographic"
        # todo: implement prompt-based reasoning
        prompt_reasoning=False,
        # todo: change
        llm_aggregation_priors=False,
        sampling_temperature=1.0,
        # todo: increase number of samples to 20 or 50
        num_of_samples=100,
        max_number_attempts=10,
        base_filters=[],
        decomposition_attributes=["ESR"],
        log_csv="sociodemographic_multi_run.csv",
        weighted=True,
        # todo: change to False for CDF-style results
        income_greater_than_threshold=True,
    )

    # Prompt templates
    if cfg.prompting_scheme == "sociodemographic":
        template_filename = "ACS_income_v3.txt"
        no_filter_template_filename = "ACS_income_no_filters_v3.txt"
    elif cfg.prompting_scheme == "persona":
        template_filename = "ACS_persona_income_v1.txt"
        no_filter_template_filename = ""
    elif cfg.prompting_scheme == "unspecified":
        raise ValueError("Please specify prompting scheme. The option 'unspecified' is invalid.")
    else:
        raise ValueError(f"Unknown prompting scheme: {cfg.prompting_scheme}")

    # Run main experiment
    main(cfg=cfg,
         shard_id=shard_id,
         template_filename=template_filename,
         no_filter_template_filename=no_filter_template_filename,
         timestamp=datetime)
