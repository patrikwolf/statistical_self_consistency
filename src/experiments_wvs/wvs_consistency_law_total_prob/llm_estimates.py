import json
import numpy as np

from config.model_config import ModelConfig
from experiments_wvs.wvs_consistency_law_total_prob.build_tree_nodes import get_attribute_combinations
from experiments_wvs.wvs_consistency_law_total_prob.questions import get_question_answer_list
from language_models.batch_prompting_direct import llm_distribution_prompting
from file_logging.read_and_write_json import save_as_json
from prompting.assemble_sc_prompts import assemble_wvs_answer_distribution_prompt, \
    assemble_wvs_prior_distribution_prompt
from utility.time_helper import get_experiment_timestamp


def _filter_by_first_attribute_type(
        combinations: list[list[dict]],
        first_attribute: str,
) -> list[list[dict]]:
    valid_combinations = []
    for idx, combination in enumerate(combinations):
        if combination[0]["attribute_description"] == first_attribute:
            valid_combinations.append(combination)
    return valid_combinations


def get_llm_estimated_prior_distribution(
        cfg: ModelConfig,
        experiment_folder: str,
        level: int,
        combinations: list[list[dict]],
        tree_idx: int,
) -> list[dict]:
    print(f"Computing LLM prior results for tree {tree_idx}, level {level}")

    # Generate prompt
    prompt, group_wise_attribute_lists = assemble_wvs_prior_distribution_prompt(
        country=cfg.country,
        combinations=combinations,
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
    save_as_json(
        data=llm_estimated_prior_distribution,
        experiment=f"{experiment_folder}/llm_priors",
        filename=f"tree-{tree_idx}__level-{level}__llm_prior_results.json",
    )
    # print(f"\nLLM prior results for tree {tree_idx}, level {level} saved to {results_path}")

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

    # todo: maybe save normalized and average results to JSON file

    # Convert to list (with the correct order)
    llm_prior_distribution = []
    for combination in combinations:
        group_name = next(item["group_name"] for item in group_wise_attribute_lists if item["combination"] == combination)
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
        question_idx: int,
        num_questions: int,
        question: str,
        answer_options: list[str],
        combination_idx: int,
        num_combinations: int,
        combination: list,
) -> dict:
    print(f"Computing LLM results for question {question_idx + 1} out of {num_questions} and "
          f"combination {combination_idx + 1} out of {num_combinations}")

    # Generate prompt
    prompt, answer_option_to_label_dict = assemble_wvs_answer_distribution_prompt(
        question=question,
        answer_options=answer_options,
        attributes=combination,
        country=cfg.country,
    )

    # Extract expected keys
    expected_json_keys = list(answer_option_to_label_dict.values())

    # Direct prompting on subgroup
    llm_estimated_answer_distribution = llm_distribution_prompting(
        model_name=cfg.model_name,
        model=None,
        tokenizer=None,
        prompt=prompt,
        reasoning_effort=cfg.reasoning_effort,
        num_of_samples=cfg.num_of_samples,
        max_number_attempts=cfg.max_number_attempts,
        expected_json_keys=expected_json_keys,
    )

    # Save results to JSON file
    save_as_json(
        data=llm_estimated_answer_distribution,
        experiment=f"{experiment_folder}/llm_answers_for_questions",
        filename=f"{question_idx}__comb-{combination_idx}__llm_answer_results.json",
    )
    # print(f"\nLLM results for question {question_identifier} and combination {combination_idx} saved to {results_path}")

    # Normalize prior distribution for each run
    for run in llm_estimated_answer_distribution.values():
        unnormalized_distribution = run["llm_distribution"]

        # Check if number of priors matches number of groups
        assert len(unnormalized_distribution) == len(answer_options), (f"Answer distribution has "
                                                                       f"{len(unnormalized_distribution)} keys, "
                                                                       f"which does not match the "
                                                                       f"{len(answer_options)} answer options.\n"
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

    # todo: maybe save normalized and average results to JSON file

    # Convert to list (with the correct order)
    llm_answer_distribution = [normalized_avg_distribution[answer_option_to_label_dict[answer_option]]
                               for answer_option in answer_options]

    # Structure results in dict
    results = {
        "combination": combination,
        "llm_answer_distribution": llm_answer_distribution,
    }

    return results


def get_all_llm_estimates(
        cfg: ModelConfig,
        experiment_folder: str,
        question_answer_list: list[dict],
        attributes: list[str],
        combination_dict: dict,
) -> dict:
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
    llm_estimate_dict = {}
    print("*" * 80)
    print("Computing answer distribution estimates for all question and combinations...")
    print("*" * 80)
    for question_idx, question_answer_dict in enumerate(question_answer_list):
        question_identifier = str(question_answer_dict["question_identifier"])
        question = question_answer_dict["question"]
        answer_options = question_answer_dict["answer_options"]

        # Get LLM estimates for all nodes (= combination)
        llm_estimates_for_combination = []
        combination_idx = 0
        num_combinations = sum([len(combinations) for combinations in combination_dict.values()])
        for depth, combinations in combination_dict.items():
            for combination in combinations:
                # Get LLM-estimated answer distribution for question, conditional on attributes
                llm_results = get_llm_estimated_answer_distribution(
                    cfg=cfg,
                    experiment_folder=experiment_folder,
                    question_idx=question_idx,
                    num_questions=len(question_answer_list),
                    question=question,
                    answer_options=answer_options,
                    combination_idx=combination_idx,
                    num_combinations=num_combinations,
                    combination=combination,
                )

                # Add prior from level-wise computation
                prior_results = next((item for item in llm_prior_list if item["combination"] == combination), None)
                llm_results["llm_prior"] = prior_results["llm_prior"]

                # Increase counter
                combination_idx += 1

                # Add to list
                llm_estimates_for_combination.append(llm_results)

        # Add to dict
        llm_estimate_dict[question_identifier] = {
            "question_identifier": question_identifier,
            "question": question,
            "answer_options": answer_options,
            "llm_estimates": llm_estimates_for_combination,
        }

    # Save results to file
    results_path = save_as_json(
        data=llm_estimate_dict,
        experiment=f"{experiment_folder}",
        filename="combined_llm_answer_results.json",
    )
    print(f"\nCombined LLM results saved to {results_path}")

    return llm_estimate_dict


def load_test_llm_estimates() -> dict:
    with open("./test_llm_estimate_dict.json", "r") as f:
        llm_estimate_dict = json.load(f)
    return llm_estimate_dict


if __name__ == "__main__":
    # Config
    cfg = ModelConfig(
        experiment_name="wvs_llm_estimates",
        country="Canada",
        model_name="openai/gpt-5.4",
        reasoning_effort="none",
        sampling_temperature=1.0,
        num_of_samples=2,
        max_number_attempts=10,
    )

    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Get question answer list
    question_answer_list = get_question_answer_list(experiment_folder=experiment_folder, truncate=2)

    # Splits
    binary_splits = {
        "sex": ["female", "male"],
        "age": ["16 - 44 years old", "45 - 103 years old"]
    }

    # Extract attributes
    attributes = list(binary_splits.keys())

    # Combinations
    combination_dict = get_attribute_combinations(experiment_folder=experiment_folder, splits=binary_splits, max_depth=2)

    # Get LLM-estimate prior for all nodes in level 1
    """
    level = 1
    level_wise_priors = _get_llm_estimated_prior_distribution(
        cfg=cfg,
        experiment_folder="wvs_llm_estimates/2026-05-05__16-14-00",
        level=level,
        country=country,
        combinations=combination_dict[level][:2],
    )
    print("(Normalized and averaged) level-wise priors:")
    print(level_wise_priors)
    """

    """
    # Get LLM estimates for question 1 and fixed combination
    question_idx = 0
    llm_estimates = _get_llm_estimated_answer_distribution(
        cfg=cfg,
        experiment_folder="wvs_llm_estimates/2026-05-05__16-14-00",
        question_identifier=question_answer_list[question_idx]["question_identifier"],
        question=question_answer_list[question_idx]["question"],
        answer_options=question_answer_list[question_idx]["answer_options"],
        combination_idx=0,
        combination=combination_dict[2][0],
        country=country
    )
    print("(Normalized and averaged) answer distribution estimates:")
    print(llm_estimates)
    """

    # Remove second layer for shorter runtime (during testing)
    combination_dict.pop(2)

    # Get all LLM estimates (all questions + all combinations)
    all_llm_estimates = get_all_llm_estimates(
        cfg=cfg,
        experiment_folder=experiment_folder,
        question_answer_list=question_answer_list,
        attributes=attributes,
        combination_dict=combination_dict,
    )
    print(json.dumps(all_llm_estimates, indent=4))
