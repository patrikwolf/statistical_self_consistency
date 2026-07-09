import json
import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizer
from config.prior_computation_config import PriorComputationConfig
from data_loader_acs.data_loader import load_original_attribute_dict
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from experiments_acs.llm_piors.filter_helper import generate_filters
from language_models.prior_estimation import get_joint_llm_prior_estimates
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def joint_prior_elicitation(
        cfg: PriorComputationConfig,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        filter_list: list[list],
        level: str,
        experiment_folder: str,
        timestamp: str,
        template_filename: str,
        improved_age_desc: bool,
        verbose: bool = False,
) -> dict:
    print(f"\nModel: {cfg.model_name}")
    print(f"Reasoning effort: {cfg.reasoning_effort}\n")

    print(f"Jointly computing LLM prior estimates for all {len(filter_list)} filters")
    joint_llm_priors, group_wise_filters = get_joint_llm_prior_estimates(
        model_name=cfg.model_name,
        model=model,
        tokenizer=tokenizer,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
        base_filters=[],
        filter_list=filter_list,
        improved_age_desc=improved_age_desc,
        template_filename=template_filename,
    )

    # Normalize prior distribution for each run
    if verbose:
        print("\nNormalizing LLM prior estimates for each run")
    for run in joint_llm_priors.values():
        unnormalized_priors = run["llm_distribution"]

        # Check if number of priors matches number of groups
        assert len(unnormalized_priors) == len(filter_list), (f"Prior estimate has {len(unnormalized_priors)} keys, "
                                                              f"which does not match the {len(filter_list)} filters.\n"
                                                              + json.dumps(unnormalized_priors, indent=4))

        # Normalize
        sum_unnormalized_priors = sum([val for val in unnormalized_priors.values()])
        if verbose:
            print(f"Sum of unnormalized priors: {sum_unnormalized_priors}")
        run["normalized_priors"] = {key: value / sum_unnormalized_priors for key, value in unnormalized_priors.items()}
        assert abs(sum([val for val in run["normalized_priors"].values()]) - 1) < 1e-8, "Prior estimates do not sum up to 1!"

    # Average normalized priors over across runs
    normalized_avg_priors = {}
    for group in joint_llm_priors["run_0"]["normalized_priors"].keys():
        group_specific_normalized_priors = [joint_llm_priors[run]["normalized_priors"][group]
                                            for run in joint_llm_priors.keys()]
        normalized_avg_priors[group] = float(np.average(group_specific_normalized_priors))

    # Save LLM output in results file
    save_as_json(
        data=joint_llm_priors,
        experiment=experiment_folder,
        filename=f"{cfg.llm_prior_results_file}",
        timestamp=False)

    # Summary of results
    summary_results = {
        "experiment_name": cfg.experiment_name,
        "timestamp": timestamp,
        "level": level,
        "model": cfg.model_name,
        "num_of_samples": cfg.num_of_samples,
        "num_of_filters": len(filter_list),
        "results": [
            {
                "group": item["group_name"],
                "filters": item["filters"],
                "normalized_averaged_prior": normalized_avg_priors[item["group_name"]],
            }
            for item in group_wise_filters
        ]
    }

    # Save LLM output in results file
    save_as_json(
        data=summary_results,
        experiment=experiment_folder,
        filename=f"{cfg.summary_results_file}",
        timestamp=False)

    # return [val for val in normalized_avg_priors.values()]
    return summary_results


if __name__ == "__main__":
    # Configuration
    cfg = PriorComputationConfig(
        experiment_name="test_joint_prior_elicitation",
        model_name="openai/gpt-5.4",
        reasoning_effort="none",
        num_of_samples=2,
        max_number_attempts=10,
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

    # Toggle improved age description
    improved_age_desc = False

    # Prior elicitation
    normalized_priors = joint_prior_elicitation(
        cfg=cfg,
        model=None,
        tokenizer=None,
        filter_list=filter_list,
        level=filter_level,
        experiment_folder=experiment_folder,
        timestamp=f"{date}__{timestamp}",
        template_filename="ACS_joint_prior_estimation_v2.txt",
        improved_age_desc=improved_age_desc,
    )

    print(f"\nNormalized (averaged) LLM prior estimates: {normalized_priors}")
