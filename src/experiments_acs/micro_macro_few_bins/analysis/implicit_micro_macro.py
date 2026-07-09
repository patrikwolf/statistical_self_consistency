import os
import time
import numpy as np
import pandas as pd

from typing import Any
from experiments_acs.distribution.empirical_survey_distribution import get_empirical_survey_distribution
from experiments_acs.micro_macro_few_bins.config.loader import load_shared_micro_macro_config
from language_models.batch_prompting_direct import llm_distribution_prompting
from prompting.assemble_acs_prompts import assemble_few_bin_income_prompt, assemble_few_bin_implicit_micro_macro_prompt
from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_survey_data
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from utility.hyperparameters import assert_decomposition_validity
from utility.time_helper import get_experiment_timestamp


def main(survey_df: pd.DataFrame,
         attribute_dict: dict,
         cfg: MinimalExperimentConfig,
         bin_edges: list[float],
         mtm_template_filename: str,
         template_direct_root: str,
         timestamp: str | None = None,
         target_col: str = "PINCP",
         ) -> None:
    print(f"\nModel: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}")
    print(f"Bins: {bin_edges}\n")

    assert cfg.prompting_scheme == "sociodemographic", "Currently, only 'sociodemographic' prompting scheme is implemented."

    # Start timer
    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name, timestamp=timestamp)

    # Initialize results
    results: dict[str, Any] = {
        "filename": os.path.basename(__file__),
        "timestamp": f"{date}__{timestamp}",
        "model": cfg.model_name,
        "reasoning_effort": cfg.reasoning_effort,
        "prompting_scheme": cfg.prompting_scheme,
        "prompt_reasoning": cfg.prompt_reasoning,
        "llm_aggregation_priors": cfg.llm_aggregation_priors,
        "survey_name": cfg.survey_name,
        "survey_year": cfg.survey_year,
        "set_nan_to_zero": cfg.set_nan_to_zero,
        "num_of_samples": cfg.num_of_samples,
        "max_number_attempts": cfg.max_number_attempts,
        "income_threshold": cfg.income_threshold,
        "experiment_folder": experiment_folder,
        "weighted": cfg.weighted,
        "prompt_template": mtm_template_filename,
        "income_greater_than_threshold": cfg.income_greater_than_threshold,
    }

    # Compute ground truth probability distribution
    _, _, _, counts, _, _ = get_empirical_survey_distribution(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=[],
        weighted=True,
        target_col=target_col,
        bins=np.array(bin_edges),
        density=False,
    )

    # Store ground truth distribution
    results["ground_truth"] = {
        "counts": counts.tolist(),
        "distribution": (counts / counts.sum()).tolist(),
        "bin_edges": bin_edges,
    }

    # Direct prompting
    # few_bin_direct_prompting(
    #     results=results,
    #     cfg=cfg,
    #     experiment_folder=experiment_folder,
    #     prefix=prefix,
    #     bin_edges=bin_edges,
    #     template_direct_root=template_direct_root,
    # )

    # Micro macro prompting
    few_bin_micro_macro_prompting(
        results=results,
        cfg=cfg,
        experiment_folder=experiment_folder,
        prefix="",
        bin_edges=bin_edges,
        mtm_template_filename=mtm_template_filename,
    )

    # Log to JSON & CSV
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=cfg.summary_results_file,
        timestamp=False)
    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "model", "num_of_samples", "income_threshold"]
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


def few_bin_direct_prompting(
        results: dict,
        cfg: MinimalExperimentConfig,
        experiment_folder: str,
        prefix: str,
        bin_edges: list[float],
        template_direct_root: str,
) -> None:
    # Assemble prompt for direct root level estimate (no micro to macro reasoning)
    prompt_direct, bin_details_direct = assemble_few_bin_income_prompt(
        bin_edges=bin_edges,
        filters=[],
        prompting_scheme=cfg.prompting_scheme,
        filtered_template_filename="",
        no_filter_template_filename=template_direct_root,
        uniform_example_distribution=True,
    )

    # LLM prediction for direct prompting
    print("Direct root level estimate (no micro to macro)")
    log_direct_root = llm_distribution_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt_direct,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Save LLM output in results file
    save_as_json(
        data=log_direct_root,
        experiment=experiment_folder,
        filename=f"{prefix}direct_root_llm.json",
        timestamp=False)

    # Average distribution
    for bin_name, bin_detail in bin_details_direct.items():
        average_probability_mass = np.average([item["llm_distribution"][bin_name] for item in log_direct_root.values()])
        bin_detail["average_probability_mass"] = float(average_probability_mass)

    sorted_bin_list = [f"Bin {idx + 1}" for idx in range(len(bin_details_direct))]
    avg_bin_distribution_list = [bin_details_direct[bin_name]["average_probability_mass"] for bin_name in
                                 sorted_bin_list]
    assert abs(sum(avg_bin_distribution_list) - 1) < 1e-6, (f"Averaged LLM distribution does not sum to 1:"
                                                            f" {sum(avg_bin_distribution_list)}")

    # Store LLM direct predictions
    results["direct"] = {
        "prompt": prompt_direct,
        "llm_distribution_list": [item["llm_distribution"] for item in log_direct_root.values()],
        "avg_bin_distribution": bin_details_direct,
        "avg_bin_distribution_list": avg_bin_distribution_list,
    }


def few_bin_micro_macro_prompting(
        results: dict,
        cfg: MinimalExperimentConfig,
        experiment_folder: str,
        prefix: str,
        bin_edges: list[float],
        mtm_template_filename: str,
) -> None:
    # Assemble prompt for micro to macro reasoning
    prompt_micro_macro, bin_details_micro_macro = assemble_few_bin_implicit_micro_macro_prompt(
        bin_edges=bin_edges,
        filtered_template_filename=mtm_template_filename,
        prompting_scheme=cfg.prompting_scheme,
        uniform_example_distribution=True,
    )

    # LLM prediction of income > threshold
    print("Micro to macro reasoning (with implicit partitioning)")
    log_micro_macro = llm_distribution_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt_micro_macro,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Save LLM output in results file
    save_as_json(
        data=log_micro_macro,
        experiment=experiment_folder,
        filename=f"{prefix}marco_micro_reasoning.json",
        timestamp=False)

    # Average distribution
    for bin_name, bin_detail in bin_details_micro_macro.items():
        average_probability_mass = np.average([item["llm_distribution"][bin_name] for item in log_micro_macro.values()])
        bin_detail["average_probability_mass"] = float(average_probability_mass)

    sorted_bin_list = [f"Bin {idx + 1}" for idx in range(len(bin_details_micro_macro))]
    avg_bin_distribution_list = [bin_details_micro_macro[bin_name]["average_probability_mass"] for bin_name in
                                 sorted_bin_list]
    assert abs(sum(avg_bin_distribution_list) - 1) < 1e-6, (f"Averaged LLM distribution does not sum to 1:"
                                                            f" {sum(avg_bin_distribution_list)}")

    # Store LLM direct predictions
    results["micro_macro"] = {
        "prompt": prompt_micro_macro,
        "llm_distribution_list": [item["llm_distribution"] for item in log_micro_macro.values()],
        "avg_bin_distribution": bin_details_micro_macro,
        "avg_bin_distribution_list": avg_bin_distribution_list,
    }


if __name__ == "__main__":
    print(f"Running script with name: {os.path.basename(__file__)}")

    # Hyperparameters
    seeds = [42]
    standard_values = load_standard_values()
    decomposition_attributes = standard_values["decomposition_attributes"]

    # Configuration
    shared_config = load_shared_micro_macro_config()
    cfg = MinimalExperimentConfig(
        survey_name=shared_config["survey_name"],
        survey_year=shared_config["survey_year"],
        set_nan_to_zero=shared_config["set_nan_to_zero"],
        model_name=shared_config["model_name"],
        reasoning_effort=shared_config["reasoning_effort"],
        prompting_scheme=shared_config["prompting_scheme"],
        prompt_reasoning=shared_config["prompt_reasoning"],
        llm_aggregation_priors=shared_config["llm_aggregation_priors"],
        num_of_samples=shared_config["num_of_samples"],
        max_number_attempts=shared_config["max_number_attempts"],
        weighted=shared_config["weighted"],
        income_greater_than_threshold=shared_config["income_greater_than_threshold"],
        #
        experiment_name="micro_macro_implicit_few_bins",
        log_csv="micro_macro_implicit_few_bins_results.csv",
    )

    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=cfg.survey_year, set_nan_to_zero=cfg.set_nan_to_zero)
    print(f"Successfully loaded ACS data from year {cfg.survey_year}...")

    # Assert validity of decomposition attributes
    assert_decomposition_validity(decomposition_attributes=decomposition_attributes, attribute_dict=attribute_dict)

    # 4 bins (after setting NaN to zero)
    bin_edges = [-11500, 1, 25000, 60000, 1849000]

    # Run main experiment
    main(survey_df=survey_df,
         attribute_dict=attribute_dict,
         cfg=cfg,
         bin_edges=bin_edges,
         mtm_template_filename="ACS_income_few_bin_implicit_micro_macro.txt",
         template_direct_root="ACS_income_few_bins_no_filters_v1.txt",
         )
