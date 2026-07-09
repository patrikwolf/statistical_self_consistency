import os
import time

from typing import Any
from config.goqa_config import GlobalOpinionQAConfig
from data_loader_global_opinion_qa.data_loader import load_goqa_data, get_question_indices_for_country
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from language_models.batch_prompting_direct import llm_distribution_prompting
from prompting.assemble_goqa_prompts import assemble_goqa_distribution_prompt
from utility.time_helper import get_experiment_timestamp


def main(
        cfg: GlobalOpinionQAConfig,
        survey_data: list[dict],
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
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Initialize results
    results: dict[str, Any] = {
        "filename": filename,
        "timestamp": f"{date}__{timestamp}",
        "model": cfg.model_name,
        "reasoning_effort": cfg.reasoning_effort,
        "prompting_scheme": cfg.prompting_scheme,
        "num_of_samples": cfg.num_of_samples,
        "max_number_attempts": cfg.max_number_attempts,
        "experiment_folder": experiment_folder,
        "direct_template": direct_template,
        "micro_to_macro_template": micro_to_macro_template,
        "evaluation_results": [],
    }

    # Initialize LLM result dictionaries
    direct_result_dict = {}
    micro_macro_result_dict = {}

    # Iterate over questions
    for idx, row in enumerate(survey_data):
        print(f"\nProcessing row {idx + 1} / {len(survey_data)}...")

        # Initialize results
        eval_result = {
            "question": row["question"],
            "question_identifier": row["question_identifier"],
            "options": row["options"],
            "distributions": [],
        }

        # Initialize question-specific LLM dictionaries
        direct_question_dict = {}
        micro_macro_question_dict = {}

        for country in row["selections"]:
            print(f"\nProcessing country: {country}")

            # Direct prompting
            print("Direct prompting:")
            direct_results = direct_prompting(
                cfg=cfg,
                direct_template=direct_template,
                experiment_folder=experiment_folder,
                question_identifier=row["question_identifier"],
                question=row["question"],
                options=row["options"],
                country=country,
            )
            direct_question_dict[country] = direct_results

            # Micro-to-macro prompting
            print("Micro prompting:")
            micro_macro_results = micro_macro_prompting(
                cfg=cfg,
                micro_to_macro_template=micro_to_macro_template,
                experiment_folder=experiment_folder,
                question_identifier=row["question_identifier"],
                question=row["question"],
                options=row["options"],
                country=country,
            )
            micro_macro_question_dict[country] = micro_macro_results

            # Collect results
            country_results = {
                "country": country,
                "ground_truth": row["selections"][country],
                "direct_prompting": {
                    "llm_distribution_list": direct_results["llm_distribution_list"],
                    "avg_llm_distribution": direct_results["avg_llm_distribution"],
                },
                "micro_to_macro": {
                    "llm_distribution_list": micro_macro_results["llm_distribution_list"],
                    "avg_llm_distribution": micro_macro_results["avg_llm_distribution"],
                },
            }

            # Add to list
            eval_result["distributions"].append(country_results)

        # Add to list
        results["evaluation_results"].append(eval_result)
        direct_result_dict[f"Q{idx}"] = direct_question_dict
        micro_macro_result_dict[f"Q{idx}"] = micro_macro_question_dict

    # Log to JSON
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"{cfg.summary_results_file}",
        timestamp=False)

    save_as_json(
        data=direct_result_dict,
        experiment=experiment_folder,
        filename=f"{cfg.direct_results_file}",
        timestamp=False)

    save_as_json(
        data=micro_macro_result_dict,
        experiment=experiment_folder,
        filename=f"{cfg.micro_macro_results_file}",
        timestamp=False)

    # Log to CSV
    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "model", "num_of_samples"]
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
        experiment_folder: str,
        question_identifier: str,
        question: str,
        options: list,
        country: str,
) -> dict:
    # Create prompt
    prompt, answer_option_keys = assemble_goqa_distribution_prompt(
        question=question,
        options=options,
        country=country,
        prompting_scheme=cfg.prompting_scheme,
        template_filename=direct_template
    )

    # LLM distribution prompting
    log_direct = llm_distribution_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Extract information
    llm_distribution_list = [item["llm_distribution"] for item in log_direct.values()]

    # Save as JSON
    save_as_json(
        data=log_direct,
        experiment=f"{experiment_folder}/llm_results",
        filename=f"{question_identifier}_direct_prompting.json",
        timestamp=False)

    # Collect results
    results = {
        "question": question,
        "options": options,
        "country": country,
        "prompt": prompt,
        "llm_distribution_list": llm_distribution_list,
        "avg_llm_distribution": compute_average_llm_distribution(llm_distribution_list),
    }

    return results


def micro_macro_prompting(
        cfg: GlobalOpinionQAConfig,
        micro_to_macro_template: str,
        experiment_folder: str,
        question_identifier: str,
        question: str,
        options: list,
        country: str,
):
    # Create prompt
    prompt, answer_option_keys = assemble_goqa_distribution_prompt(
        question=question,
        options=options,
        country=country,
        prompting_scheme=cfg.prompting_scheme,
        template_filename=micro_to_macro_template
    )

    # LLM distribution prompting
    log_micro_macro = llm_distribution_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Extract information
    llm_distribution_list = [item["llm_distribution"] for item in log_micro_macro.values()]

    # Save as JSON
    save_as_json(
        data=log_micro_macro,
        experiment=f"{experiment_folder}/llm_results",
        filename=f"{question_identifier}_micro_macro_prompting.json",
        timestamp=False)

    # Collect results
    results = {
        "question": question,
        "options": options,
        "country": country,
        "prompt": prompt,
        "llm_distribution_list": llm_distribution_list,
        "avg_llm_distribution": compute_average_llm_distribution(llm_distribution_list),
    }

    return results


def compute_average_llm_distribution(distributions: list[dict]) -> dict:
    avg_distribution = {
        key: sum(d[key] for d in distributions) / len(distributions)
        for key in distributions[0].keys()
    }

    # Assert normalization
    assert abs(sum([val for val in avg_distribution.values()]) - 1) < 1e-6

    return avg_distribution


if __name__ == "__main__":
    # Config
    cfg = GlobalOpinionQAConfig(
        experiment_name="micro_macro_global_opinion_qa",
        # todo: change model
        model_name="openai/gpt-5.4",
        reasoning_effort="none",
        prompting_scheme="sociodemographic",
        num_of_samples=20,
        max_number_attempts=10,
    )

    # Load GlobalOpinionQA dataset
    countries = ["United States", "Germany", "France", "Japan"]
    row_indices = get_question_indices_for_country(countries=countries)[:2]
    row_indices = [55, 136, 345, 413, 573, 839, 1108, 1445, 1512, 1531, 1644][:8]
    survey_data = load_goqa_data(row_indices=row_indices, countries=countries)

    # Prompt templates
    direct_template = "GOQA_direct_prompt_v1.txt"
    micro_to_macro_template = "GOQA_implicit_micro_macro.txt"

    # Run main experiment
    main(
        cfg=cfg,
        survey_data=survey_data,
        direct_template=direct_template,
        micro_to_macro_template=micro_to_macro_template
    )
