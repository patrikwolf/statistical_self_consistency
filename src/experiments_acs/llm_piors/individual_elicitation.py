import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizer
from config.minimal_experiment_config import MinimalExperimentConfig
from data_loader_acs.data_loader import load_original_attribute_dict
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from experiments_acs.llm_piors.filter_helper import generate_filters
from language_models.prior_estimation import get_individual_llm_prior_estimate
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def individual_prior_elicitation(
        cfg: MinimalExperimentConfig,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        filter_list: list[list],
        level: str,
        experiment_folder: str,
        timestamp: str,
) -> list[dict]:
    print(f"\nModel: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}")

    # Loop over all filter choices
    unnormalized_results = []
    for idx, filters in enumerate(filter_list):
        print(f"\nComputing LLM prior estimate for filter {idx + 1} / {len(filter_list)}")
        print("Filters:")
        for f in filters:
            print(f"     Attribute: {f.attribute}, values: {f.values}")

        # Compute prior estimate
        unnormalized_llm_priors = get_individual_llm_prior_estimate(
            model_name=cfg.model_name,
            model=model,
            tokenizer=tokenizer,
            reasoning_effort=cfg.reasoning_effort,
            num_of_samples=cfg.num_of_samples,
            max_number_attempts=cfg.max_number_attempts,
            base_filters=[],
            decomposition_filters=filters,
        )

        results = {
            "filters": [f.serialize() for f in filters],
            "runs": unnormalized_llm_priors,
        }

        # Add to list
        unnormalized_results.append(results)

    # Save LLM output in results file
    save_as_json(
        data=unnormalized_results,
        experiment=experiment_folder,
        filename=f"{cfg.llm_prior_results_file}",
        timestamp=False)

    # Initialize summary results
    summary_results = {
        "experiment_name": cfg.experiment_name,
        "timestamp": timestamp,
        "level": level,
        "model": cfg.model_name,
        "num_of_samples": cfg.num_of_samples,
        "num_of_filters": len(filter_list),
        "results": [],
    }

    # Average unnormalized priors
    unnormalized_averaged_priors = []
    for filter_specific_unnormalized_res in unnormalized_results:
        runs = filter_specific_unnormalized_res["runs"]
        unnormalized_llm_prior_list = [item["prediction"] for item in runs.values()]
        unnormalized_llm_prior_avg = np.average(unnormalized_llm_prior_list)
        unnormalized_averaged_priors.append(unnormalized_llm_prior_avg)

        # Add to results
        summary_results["results"].append(
            {
                "filters": filter_specific_unnormalized_res["filters"],
                "unnormalized_llm_prior_list": unnormalized_llm_prior_list,
                "unnormalized_averaged_prior": unnormalized_llm_prior_avg,
            }
        )

    sum_unnormalized = sum(unnormalized_averaged_priors)
    summary_results["sum_unnormalized_avg_priors"] = sum_unnormalized
    print(f"Sum of unnormalized averaged priors: {sum_unnormalized}")

    # Normalize averaged priors
    normalized_averaged_priors = []
    for res in summary_results["results"]:
        normalized_averaged_prior = float(res["unnormalized_averaged_prior"] / sum_unnormalized)
        res["normalized_averaged_prior"] = normalized_averaged_prior
        normalized_averaged_priors.append(normalized_averaged_prior)

    assert abs(sum(normalized_averaged_priors) - 1) < 1e-8, "Normalized prior estimates do not sum up to 1!"

    # Save LLM output in results file
    save_as_json(
        data=summary_results,
        experiment=experiment_folder,
        filename=f"{cfg.summary_results_file}",
        timestamp=False)

    return normalized_averaged_priors


if __name__ == "__main__":
    # Configuration
    cfg = MinimalExperimentConfig(
        survey_name="ACS",
        survey_year=2024,
        set_nan_to_zero=True,
        experiment_name="individual_prior_elicitation",
        model_name="openai/gpt-5.4",
        reasoning_effort="none",
        prompting_scheme="sociodemographic",
        prompt_reasoning=False,
        num_of_samples=2,
        max_number_attempts=10,
        weighted=True,
        log_csv="individual_prior_results.csv",
        llm_prior_results_file="unnormalized_priors.json",
    )

    # Experiment folder
    date, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Attribute dict
    attribute_dict = load_original_attribute_dict()

    # Decomposition attributes#
    standard_values = load_standard_values()
    decomposition_attributes = standard_values["decomposition_attributes"]

    # Filters
    filter_tree = generate_filters(decomposition_attributes=decomposition_attributes, attribute_dict=attribute_dict)
    print(f"Generated {len(filter_tree)} filters")

    # Extract filters for given level
    filter_level = "level_2"
    filter_list = []
    for f in filter_tree:
        if f["level"] == filter_level:
            filter_list.append(f["filters"])
    print(f"Extracted {len(filter_list)} filters for {filter_level}")

    # Prior elicitation
    normalized_priors = individual_prior_elicitation(
        cfg=cfg,
        model=None,
        tokenizer=None,
        filter_list=filter_list,
        level=filter_level,
        experiment_folder=experiment_folder,
        timestamp=timestamp,
    )

    print(f"\nNormalized (averaged) LLM prior estimates: {normalized_priors}")
