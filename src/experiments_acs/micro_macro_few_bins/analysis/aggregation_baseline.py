import os
import time
import numpy as np
import pandas as pd

from data_loader_acs.data_loader import load_survey_data
from experiments_acs.distribution.empirical_survey_distribution import get_empirical_survey_distribution
from experiments_acs.micro_macro_few_bins.config.loader import load_shared_micro_macro_config
from file_logging.read_and_write_csv import log_to_csv, print_tabulated
from file_logging.read_and_write_json import save_as_json
from language_models.batch_prompting_direct import llm_distribution_prompting
from prompting.assemble_acs_prompts import assemble_few_bin_income_prompt
from config.minimal_experiment_config import MinimalExperimentConfig
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from utility.hyperparameters import assert_decomposition_validity, process_hyperparameters
from utility.argument_parser import parse_arguments
from utility.time_helper import get_experiment_timestamp


def main(cfg: MinimalExperimentConfig,
         survey_df: pd.DataFrame,
         shard: str,
         timestamp: str | None,
         hyperparameters: dict,
         bin_edges: list[float],
         template_filename: str,
         no_filter_template_filename: str,
         target_col: str = "PINCP",
         ) -> None:
    print(f"\nHyperparameters: {hyperparameters}")
    print(f"Model: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}")
    print(f"Bins: {bin_edges}\n")

    assert cfg.prompting_scheme == "sociodemographic", "Currently, only 'sociodemographic' prompting scheme is implemented."

    # Start timer
    start_time = time.time()

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name,
                                                                  timestamp=timestamp)

    # Extract filters
    filters = hyperparameters["decomposition_filters"]["filters"]

    # Initialize results
    results: dict = {
        "filename": os.path.basename(__file__),
        "timestamp": f"{date}__{timestamp}",
        "shard": shard,
        "model": cfg.model_name,
        "survey_name": "n/a",
        "survey_year": "n/a",
        "reasoning_effort": cfg.reasoning_effort,
        "prompting_scheme": cfg.prompting_scheme,
        "prompt_reasoning": cfg.prompt_reasoning,
        "num_of_samples": cfg.num_of_samples,
        "max_number_attempts": cfg.max_number_attempts,
        "bin_edges": bin_edges,
        "decomposition_attribute_list": hyperparameters["decomposition_attribute_list"],
        "node_id": hyperparameters["decomposition_filters"]["node_id"],
        "parent_id": hyperparameters["decomposition_filters"]["parent_id"],
        "level": hyperparameters["decomposition_filters"]["level"],
        "filters": [f.serialize() for f in filters],
        "experiment_folder": experiment_folder,
        "prompt_template": template_filename,
        "prompt_template_no_filter": no_filter_template_filename,
    }

    # Compute ground truth probability distribution
    print("Computing ground-truth distribution...")
    _, _, weighted_prior, counts, _, _ = get_empirical_survey_distribution(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=hyperparameters["decomposition_filters"]["filters"],
        weighted=True,
        target_col=target_col,
        bins=np.array(bin_edges),
        density=False,
    )

    # Store ground truth distribution
    results["ground_truth"] = {
        "weighted_prior": weighted_prior,
        "counts": counts.tolist(),
        "distribution": (counts / counts.sum()).tolist(),
        "bin_edges": bin_edges,
    }

    if cfg.llm_aggregation_priors:
        raise NotImplementedError("LLM aggregation priors is not implemented yet.")

    # Assemble prompt
    prompt, bin_details = assemble_few_bin_income_prompt(
        bin_edges=bin_edges,
        filters=filters,
        prompting_scheme=cfg.prompting_scheme,
        filtered_template_filename=template_filename,
        no_filter_template_filename=no_filter_template_filename,
    )

    # LLM prediction of categorical distribution
    print("LLM-estimated distribution...")
    log_direct = llm_distribution_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Save LLM output log file
    save_as_json(
        data=log_direct,
        experiment=experiment_folder,
        filename=f"{shard}__{cfg.direct_results_file}",
        timestamp=False)

    # Average distribution
    for bin_name, bin_detail in bin_details.items():
        average_probability_mass = np.average([item["llm_distribution"][bin_name] for item in log_direct.values()])
        bin_detail["average_probability_mass"] = float(average_probability_mass)

    sorted_bin_list = [f"Bin {idx + 1}" for idx in range(len(bin_details))]
    avg_bin_distribution_list = [bin_details[bin_name]["average_probability_mass"] for bin_name in sorted_bin_list]
    assert abs(sum(avg_bin_distribution_list) - 1) < 1e-6, (f"Averaged LLM distribution does not sum to 1:"
                                                            f" {sum(avg_bin_distribution_list)}")

    # Store LLM direct predictions
    results["llm_estimates"] = {
        "prompt": prompt,
        "llm_distribution_list": [item["llm_distribution"] for item in log_direct.values()],
        "avg_bin_distribution": bin_details,
        "avg_bin_distribution_list": avg_bin_distribution_list,
    }

    # Log to JSON
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"{shard}__{cfg.summary_results_file}",
        timestamp=False)

    log_to_csv(results=results, filename=cfg.log_csv, date=date, timestamp=timestamp)

    # Print tabulated results
    cols = ["date", "time", "shard", "model", "num_of_samples", "node_id", "avg_bin_distribution_list"]
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
        improved_age_desc=shared_config["improved_age_desc"],
        llm_aggregation_priors=shared_config["llm_aggregation_priors"],
        num_of_samples=shared_config["num_of_samples"],
        max_number_attempts=shared_config["max_number_attempts"],
        weighted=shared_config["weighted"],
        income_greater_than_threshold=shared_config["income_greater_than_threshold"],
        #
        experiment_name="micro_macro_few_bins_baseline",
        log_csv="micro_macro_few_bins_baseline_results.csv",
    )

    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=cfg.survey_year, set_nan_to_zero=cfg.set_nan_to_zero)
    print(f"Successfully loaded ACS data from year {cfg.survey_year}...")

    # Assert validity of decomposition attributes
    assert_decomposition_validity(decomposition_attributes=decomposition_attributes, attribute_dict=attribute_dict)

    # Process hyperparameters
    hyperparameters = process_hyperparameters(
        seeds=seeds,
        decomposition_attributes=decomposition_attributes,
        attribute_dict=attribute_dict,
    )

    # Parse arguments
    shard_id, shard, datetime = parse_arguments()
    print(f"Shard ID: {shard_id} ({shard_id + 1} out of {len(hyperparameters)})")

    # 4 bins (after setting NaN to zero)
    bin_edges = [-11500, 1, 25000, 60000, 1849000]

    # Run main experiment
    main(cfg=cfg,
         survey_df=survey_df,
         shard=shard,
         timestamp=datetime,
         hyperparameters=hyperparameters[shard],
         bin_edges=bin_edges,
         template_filename="ACS_income_few_bins_v1.txt",
         no_filter_template_filename="ACS_income_few_bins_no_filters_v1.txt",
         )
