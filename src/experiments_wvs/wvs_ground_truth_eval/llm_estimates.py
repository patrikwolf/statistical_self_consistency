import json

from config.model_config import ModelConfig
from experiments_wvs.wvs_consistency_law_total_prob.build_tree_nodes import get_original_attribute_combinations
from experiments_wvs.wvs_consistency_law_total_prob.llm_estimates import get_llm_estimated_prior_distribution, \
    get_llm_estimated_answer_distribution
from experiments_wvs.wvs_consistency_law_total_prob.questions import get_question_answer_list
from file_logging.read_and_write_json import save_as_json
from utility.time_helper import get_experiment_timestamp


def get_all_llm_estimates_for_original_tree(
        cfg: ModelConfig,
        experiment_folder: str,
        question_answer_list: list[dict],
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

        # Compute prior distribution over all nodes in the given level
        level_wise_priors = get_llm_estimated_prior_distribution(
            cfg=cfg,
            experiment_folder=experiment_folder,
            level=depth,
            combinations=combinations,
            tree_idx=0,
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

                # Add level to dict
                llm_results["level"] = depth

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


if __name__ == "__main__":
    # Config
    cfg = ModelConfig(
        experiment_name="wvs_llm_reconstruction",
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

    # Combinations
    combination_dict = get_original_attribute_combinations(experiment_folder=experiment_folder,
                                                           splits=binary_splits,
                                                           max_depth=2)

    # Get all LLM estimates (all questions + all combinations)
    all_llm_estimates = get_all_llm_estimates_for_original_tree(
        cfg=cfg,
        experiment_folder=experiment_folder,
        question_answer_list=question_answer_list,
        combination_dict=combination_dict,
    )
    print(json.dumps(all_llm_estimates, indent=4))
