import os
import time
import numpy as np

from typing import Any
from config.goqa_config import GlobalOpinionQAConfig
from data_loader_global_opinion_qa.data_loader import load_goqa_data
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from language_models.batch_prompting_direct import llm_direct_prompting
from prompting.assemble_goqa_prompts import assemble_goqa_probability_prompt
from utility.time_helper import get_experiment_timestamp


def main(
        cfg: GlobalOpinionQAConfig,
        shard: str,
        timestamp: str | None,
        survey_item: dict,
        direct_template: str,
        micro_to_macro_template: str,
):
    filename = os.path.basename(__file__)
    print(f"Running {filename}")
    print(f"Model: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}\n")

    assert cfg.prompting_scheme == "sociodemographic", "Currently, only 'sociodemographic' prompting scheme is implemented."

    # Start timer
    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name,
                                                                  timestamp=timestamp)

    # Initialize results
    results: dict[str, Any] = {
        "filename": filename,
        "timestamp": f"{date}__{timestamp}",
        "shard": shard,
        "model": cfg.model_name,
        "reasoning_effort": cfg.reasoning_effort,
        "prompting_scheme": cfg.prompting_scheme,
        "num_of_samples": cfg.num_of_samples,
        "max_number_attempts": cfg.max_number_attempts,
        "experiment_folder": experiment_folder,
        "question_identifier": survey_item["question_identifier"],
        "question": survey_item["question"],
        "options": survey_item["options"],
        "selections": survey_item["selections"],
        "direct_template": direct_template,
        "micro_to_macro_template": micro_to_macro_template,
        "evaluation_results": [],
    }

    print(results["question"])
    print(results["options"])

    if shard:
        prefix = f"{shard}_"
    else:
        prefix = ""

    # Initialize question-specific LLM dictionaries
    direct_question_dict = {}
    micro_macro_question_dict = {}

    for country in survey_item["selections"]:
        print(f"\nProcessing country: {country}")

        # Direct prompting
        print("Direct prompting:")
        direct_results = direct_prompting(
            cfg=cfg,
            direct_template=direct_template,
            prefix=prefix,
            experiment_folder=experiment_folder,
            question_identifier=survey_item["question_identifier"],
            question=survey_item["question"],
            options=survey_item["options"],
            country=country,
        )
        direct_question_dict[country] = direct_results

        # Micro-to-macro prompting
        print("Micro prompting:")
        micro_macro_results = micro_macro_prompting(
            cfg=cfg,
            micro_to_macro_template=micro_to_macro_template,
            prefix=prefix,
            experiment_folder=experiment_folder,
            question_identifier=survey_item["question_identifier"],
            question=survey_item["question"],
            options=survey_item["options"],
            country=country,
        )
        micro_macro_question_dict[country] = micro_macro_results

        # Collect results
        country_results = {
            "country": country,
            "ground_truth": survey_item["selections"][country],
            "direct_prompting": {
                "llm_prediction_list": direct_results["llm_prediction_list"],
                "avg_llm_prediction": direct_results["avg_llm_prediction"],
            },
            "micro_to_macro": {
                "llm_prediction_list": micro_macro_results["llm_prediction_list"],
                "avg_llm_prediction": micro_macro_results["avg_llm_prediction"],
            },
        }

        # Add to list
        results["evaluation_results"].append(country_results)

    # Log to JSON
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"{prefix}{cfg.summary_results_file}",
        timestamp=False)

    save_as_json(
        data=direct_question_dict,
        experiment=experiment_folder,
        filename=f"{prefix}{cfg.direct_results_file}",
        timestamp=False)

    save_as_json(
        data=micro_macro_question_dict,
        experiment=experiment_folder,
        filename=f"{prefix}{cfg.micro_macro_results_file}",
        timestamp=False)

    # Log to CSV
    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "model", "reasoning_effort", "num_of_samples", "question_identifier"]
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


def direct_prompting(
        cfg: GlobalOpinionQAConfig,
        direct_template: str,
        prefix: str,
        experiment_folder: str,
        question_identifier: str,
        question: str,
        options: list,
        country: str,
) -> dict:
    # Create prompt
    prompt = assemble_goqa_probability_prompt(
        country=country,
        prompting_scheme=cfg.prompting_scheme,
        template_filename=direct_template,
    )

    # LLM distribution prompting
    log_direct = llm_direct_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Extract information
    llm_prediction_list = [item["prediction"] for item in log_direct.values()]

    # Save as JSON
    save_as_json(
        data=log_direct,
        experiment=f"{experiment_folder}/{prefix}llm_results",
        filename=f"{question_identifier}_{country}_direct_prompting.json",
        timestamp=False)

    # Collect results
    results = {
        "question": question,
        "options": options,
        "country": country,
        "prompt": prompt,
        "llm_prediction_list": llm_prediction_list,
        "avg_llm_prediction": np.average(llm_prediction_list),
    }

    return results


def micro_macro_prompting(
        cfg: GlobalOpinionQAConfig,
        micro_to_macro_template: str,
        prefix: str,
        experiment_folder: str,
        question_identifier: str,
        question: str,
        options: list,
        country: str,
) -> dict:
    # Create prompt
    prompt = assemble_goqa_probability_prompt(
        country=country,
        prompting_scheme=cfg.prompting_scheme,
        template_filename=micro_to_macro_template,
    )

    # LLM distribution prompting
    log_micro_macro = llm_direct_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Extract information
    llm_prediction_list = [item["prediction"] for item in log_micro_macro.values()]

    # Save as JSON
    save_as_json(
        data=log_micro_macro,
        experiment=f"{experiment_folder}/{prefix}llm_results",
        filename=f"{question_identifier}_{country}_micro_macro_prompting.json",
        timestamp=False)

    # Collect results
    results = {
        "question": question,
        "options": options,
        "country": country,
        "prompt": prompt,
        "llm_prediction_list": llm_prediction_list,
        "avg_llm_prediction": np.average(llm_prediction_list),
    }

    return results


if __name__ == "__main__":
    # Config
    cfg = GlobalOpinionQAConfig(
        experiment_name="goqa_micro_macro_thresholded",
        model_name="openai/gpt-5.4",
        reasoning_effort="none",
        prompting_scheme="sociodemographic",
        num_of_samples=1,
        max_number_attempts=10,
        log_csv="micro_macro_goqa_thresholded.csv"
    )

    # Load GlobalOpinionQA dataset
    countries = ["Australia", "Italy", "Spain"]
    row_index = 55         # 55, 136, 573, 839, 1108, 1445, 1512, 1531
    print(f"Question: Q{row_index}")
    survey_data = load_goqa_data(row_indices=[row_index], countries=countries)
    assert len(survey_data) == 1

    # Prompt templates
    direct_template = f"GOQA_Q{row_index}_direct.txt"
    micro_to_macro_template = f"GOQA_Q{row_index}_micro_macro.txt"

    # Run main experiment
    main(
        cfg=cfg,
        shard="",
        timestamp=None,
        survey_item=survey_data[0],
        direct_template=direct_template,
        micro_to_macro_template=micro_to_macro_template
    )
