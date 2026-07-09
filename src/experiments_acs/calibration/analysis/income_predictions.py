import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from experiments_acs.distribution.thresholded import compute_ground_truth_prob
from experiments_acs.filtering.filter import Filter
from language_models.batch_prompting_decomposed import llm_probability_decomposition
from language_models.batch_prompting_direct import llm_direct_prompting
from file_logging.read_and_write_csv import log_to_csv
from file_logging.read_and_write_json import save_as_json
from config.experiment_config import ExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.filtering.filter_definitions import construct_base_filters, extend_generic_filter
from prompting.assemble_acs_prompts import assemble_income_prompt
from utility.time_helper import get_experiment_timestamp


def main(cfg: ExperimentConfig,
         template_filename: str,
         no_filter_template_filename: str,
         ) -> None:
    assert cfg.num_of_samples == 1, "Use script 'statistical_analysis.py' for multi-sample experiments."

    print(f"\nIncome threshold: {cfg.income_threshold} USD")
    print(f"Model: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}\n")

    # Start timer
    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

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
        "model": cfg.model_name,
        "survey_name": cfg.survey_name,
        "survey_year": cfg.survey_year,
        "set_nan_to_zero": cfg.set_nan_to_zero,
        "reasoning_effort": cfg.reasoning_effort,
        "prompting_scheme": cfg.prompting_scheme,
        "prompt_reasoning": cfg.prompt_reasoning,
        "num_of_samples": cfg.num_of_samples,
        "max_number_attempts": cfg.max_number_attempts,
        "income_threshold": cfg.income_threshold,
        "llm_aggregation_priors": cfg.llm_aggregation_priors,
        "base_filters": base_filters,
        "decomposition_attributes": cfg.decomposition_attributes,
        "experiment_folder": experiment_folder,
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

    results["ground_truth"] = ground_truth_results["high_income_probability"]
    print(f"   ---> TIMER: Computed ground truth probability after {time.time() - start_time:.2f} seconds.")

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
        data=log_direct["run_0"],
        experiment=experiment_folder,
        filename=cfg.direct_results_file,
        timestamp=False)

    # Store LLM prediction
    results["llm_prediction"] = log_direct["run_0"]["prediction"]
    results["llm_prediction_results_file"] = llm_direct_results_file
    print(f"   ---> TIMER: Computed direct LLM prediction after {time.time() - start_time:.2f} seconds.")

    # LLM with law of total probability
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
        filename=cfg.decomposition_results_file,
        timestamp=False)

    # Store LLM with law of total probability prediction
    results["llm_law_of_total_probability"] = log_decomposed["run_0"]["aggregated_prediction"]
    results["llm_law_of_total_probability_results_file"] = llm_law_of_total_prob_results_file
    print(f"   ---> TIMER: Computed law of total probability prediction after {time.time() - start_time:.2f} seconds.")

    # Prepare for logging
    results["base_filters"] = [f.serialize() for f in results["base_filters"]]

    # Log to CSV
    save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=cfg.summary_results_file,
        timestamp=False)
    log_file = log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print
    print("\n" + "*" * 80)
    print(f"Model: {cfg.model_name}")
    print(f"Ground Truth Probability of Income > {cfg.income_threshold} USD: {100 * results['ground_truth']:.2f}%")
    print(f"LLM Predicted Probability of Income > {cfg.income_threshold} USD: {100 * results['llm_prediction']:.2f}%")
    print(f"LLM with Law of Total Probability: {100 * results['llm_law_of_total_probability']:.2f}%")
    print("*" * 80)
    print(f"Results saved to:\n"
          f"      {llm_direct_results_file}\n"
          f"      {llm_law_of_total_prob_results_file}")
    print(f"Results logged to CSV:\n"
          f"      {log_file}")
    print("*" * 80)


if __name__ == "__main__":
    # Configuration
    cfg = ExperimentConfig(
        experiment_name="calibration_single_sample",
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        income_threshold=40_000,
        model_name="openai/gpt-5.4",      # "Qwen/Qwen3-0.6B", "openai/gpt-5.4"
        reasoning_effort="none",
        # todo: implement different prompting schemes
        prompting_scheme="persona",
        # todo: implement prompt-based reasoning
        prompt_reasoning=False,
        # todo: change
        llm_aggregation_priors=False,
        num_of_samples=1,
        max_number_attempts=10,
        base_filters=[
            Filter(
                attribute="SEX",
                getter=extend_generic_filter,
                values=[2],
            )
        ],
        decomposition_attributes=["ESR"],
        log_csv="sociodemographic_single_run.csv",
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

    # Run experiment
    main(cfg=cfg,
         template_filename=template_filename,
         no_filter_template_filename=no_filter_template_filename,
         )
