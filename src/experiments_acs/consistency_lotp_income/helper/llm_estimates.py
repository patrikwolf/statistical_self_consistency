import json
import numpy as np

from config.model_config import ModelConfig
from experiments_acs.filtering.filter import Filter
from language_models.batch_prompting_direct import llm_distribution_prompting
from file_logging.read_and_write_json import save_as_json
from prompting.assemble_acs_prompts import assemble_joint_prior_estimation_prompt, assemble_few_bin_income_prompt


def _filter_by_first_attribute_type(
        combinations: list[list[Filter]],
        first_attribute: str,
) -> list[list[Filter]]:
    valid_combinations = []
    for idx, combination in enumerate(combinations):
        if combination[0]["attribute_description"] == first_attribute:
            valid_combinations.append(combination)
    return valid_combinations


def get_llm_estimated_prior_distribution(
        cfg: ModelConfig,
        experiment_folder: str,
        level: int,
        combinations: list[list[Filter]],
        tree_idx: int,
) -> list[dict]:
    print(f"Computing LLM prior results for tree {tree_idx}, level {level}")

    # Assemble prompt
    prompt, group_wise_attribute_lists = assemble_joint_prior_estimation_prompt(
        base_filters=[],
        filter_list=[[d["value_description"] for d in comb] for comb in combinations],
        template_filename="ACS_joint_prior_estimation_v2.txt",
        improved_age_desc=True,
    )

    # Direct prompting on subgroup
    llm_estimated_prior_distribution = llm_distribution_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Save results to JSON file
    results_path = save_as_json(
        data=llm_estimated_prior_distribution,
        experiment=f"{experiment_folder}/llm_priors",
        filename=f"tree-{tree_idx}__level-{level}__llm_prior_results.json",
    )
    print(f"Saved LLM prior results to {results_path}")

    # Normalize prior distribution for each run
    for run in llm_estimated_prior_distribution.values():
        unnormalized_distribution = run["llm_distribution"]

        # Check if number of priors matches number of groups
        assert len(unnormalized_distribution) == len(combinations), (f"Prior estimate has {len(unnormalized_distribution)} "
                                                                     f"keys, which does not match the {len(combinations)} "
                                                                     f"combination.\n"
                                                                     + json.dumps(unnormalized_distribution, indent=4))

        # Normalize
        sum_unnormalized_priors = sum([val for val in unnormalized_distribution.values()])
        run["normalized_priors"] = {key: value / sum_unnormalized_priors for key, value in
                                    unnormalized_distribution.items()}
        assert abs(sum([val for val in run["normalized_priors"].values()]) - 1) < 1e-8, "Prior estimates do not sum up to 1!"

    # Average normalized priors over across runs
    normalized_avg_priors = {}
    for group in llm_estimated_prior_distribution["run_0"]["normalized_priors"].keys():
        group_specific_normalized_priors = [llm_estimated_prior_distribution[run]["normalized_priors"][group]
                                            for run in llm_estimated_prior_distribution.keys()]
        normalized_avg_priors[group] = float(np.average(group_specific_normalized_priors))

    # Convert to list (with the correct order)
    llm_prior_distribution = []
    for combination in combinations:
        serialized_filters = [d["value_description"].serialize() for d in combination]
        group_name = next(item["group_name"] for item in group_wise_attribute_lists if item["filters"] == serialized_filters)
        llm_prior_distribution.append(
            {
                "combination": combination,
                "llm_prior": normalized_avg_priors[group_name],
            }
        )

    return llm_prior_distribution


def get_llm_estimated_answer_distribution(
        cfg: ModelConfig,
        experiment_folder: str,
        combination_idx: int,
        num_combinations: int,
        combination: list,
        bin_edges: list,
) -> dict:
    print(f"Computing LLM results for combination {combination_idx + 1} out of {num_combinations}")

    # Generate prompt
    prompt, bin_details = assemble_few_bin_income_prompt(
        bin_edges=bin_edges,
        filters=[c["value_description"] for c in combination],
        prompting_scheme="sociodemographic",
        filtered_template_filename="ACS_income_few_bins_v1.txt",
        no_filter_template_filename="ACS_income_few_bins_no_filters_v1.txt",
        improved_age_desc=True,
    )

    # Direct prompting on subgroup
    llm_estimated_answer_distribution = llm_distribution_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
    )

    # Save results to JSON file
    save_as_json(
        data=llm_estimated_answer_distribution,
        experiment=f"{experiment_folder}/llm_answer_distributions",
        filename=f"comb-{combination_idx}__llm_answer_results.json",
    )

    # Normalize prior distribution for each run
    for run in llm_estimated_answer_distribution.values():
        unnormalized_distribution = run["llm_distribution"]

        # Check if number of priors matches number of groups
        assert len(unnormalized_distribution) == len(bin_details), (f"Answer distribution has "
                                                                    f"{len(unnormalized_distribution)} keys, "
                                                                    f"which does not match the "
                                                                    f"{len(bin_details)} answer options.\n"
                                                                    + json.dumps(unnormalized_distribution, indent=4))

        # Normalize
        sum_unnormalized_distribution = sum([val for val in unnormalized_distribution.values()])
        run["normalized_distribution"] = {key: value / sum_unnormalized_distribution for key, value in
                                          unnormalized_distribution.items()}
        assert abs(sum([val for val in run["normalized_distribution"].values()]) - 1) < 1e-8, ("Estimated distribution "
                                                                                               "does not sum up to 1!")

    # Average normalized distributions over across runs
    normalized_avg_distribution = {}
    for answer_option in llm_estimated_answer_distribution["run_0"]["normalized_distribution"].keys():
        group_specific_normalized_estimates = [llm_estimated_answer_distribution[run]["normalized_distribution"][answer_option]
                                               for run in llm_estimated_answer_distribution.keys()]
        normalized_avg_distribution[answer_option] = float(np.average(group_specific_normalized_estimates))

    # Convert to list (with the correct order)
    bins = sorted(
        list(bin_details.keys()),
        key=lambda qid: int(qid.removeprefix("Bin ")),
    )
    llm_answer_distribution = [normalized_avg_distribution[bin] for bin in bins]

    # Structure results in dict
    results = {
        "combination": [comb["value_description"].serialize() for comb in combination],
        "llm_answer_distribution": llm_answer_distribution,
    }

    return results


def get_all_llm_estimates(
        cfg: ModelConfig,
        experiment_folder: str,
        attributes: list[str],
        combination_dict: dict,
        bin_edges: list,
) -> list[dict]:
    # Compute priors for all nodes
    llm_prior_list = []
    print("*" * 80)
    print("Computing priors for all levels...")
    print("*" * 80)
    for depth, combinations in combination_dict.items():
        if depth == 0:
            assert len(combinations) == 1, "We only have one root node, so expected one combination for depth 0."
            llm_prior_list.append(
                {
                    "combination": combinations[0],
                    "llm_prior": 1,
                }
            )
            continue

        # Get level-nodes for each tree
        assert len(attributes) == 2, "This only works for tree with two attributes."
        for tree_idx, first_attribute in enumerate(attributes):
            # Filter by first attribute (split original tree and reshaped tree)
            tree_combinations = _filter_by_first_attribute_type(combinations=combinations,
                                                                first_attribute=first_attribute)

            # Compute prior distribution over all nodes in the given level
            level_wise_priors = get_llm_estimated_prior_distribution(
                cfg=cfg,
                experiment_folder=experiment_folder,
                level=depth,
                combinations=tree_combinations,
                tree_idx=tree_idx,
            )

            # Add results to list
            llm_prior_list.extend(level_wise_priors)

    # For every question and every combination, sample LLM-estimated answer distribution
    print("*" * 80)
    print("Computing answer distribution estimates for all combinations...")
    print("*" * 80)

    # Get LLM estimates for all nodes (= combination)
    llm_estimates_for_combinations = []
    combination_idx = 0
    num_combinations = sum([len(combinations) for combinations in combination_dict.values()])
    for depth, combinations in combination_dict.items():
        for combination in combinations:
            # Get LLM-estimated answer distribution for question, conditional on attributes
            llm_results = get_llm_estimated_answer_distribution(
                cfg=cfg,
                experiment_folder=experiment_folder,
                combination_idx=combination_idx,
                num_combinations=num_combinations,
                combination=combination,
                bin_edges=bin_edges,
            )

            # Add prior from level-wise computation
            prior_results = next((item for item in llm_prior_list if item["combination"] == combination), None)
            llm_results["llm_prior"] = prior_results["llm_prior"]

            # Increase counter
            combination_idx += 1

            # Add to list
            llm_estimates_for_combinations.append(llm_results)

    # Save results to file
    results_path = save_as_json(
        data=llm_estimates_for_combinations,
        experiment=f"{experiment_folder}",
        filename="combined_llm_answer_results.json",
    )
    print(f"\nCombined LLM results saved to {results_path}")

    return llm_estimates_for_combinations
