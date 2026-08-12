from typing import Literal

from config.model_config import ModelConfig
from experiments_wvs.ablation_logprobs_consistency.logprob_estimates import get_llm_estimated_prior_distribution, \
    get_llm_estimated_answer_distribution
from file_logging.read_and_write_json import save_as_json


def get_all_llm_estimates_for_original_tree(
        cfg: ModelConfig,
        experiment_folder: str,
        question_answer_list: list[dict],
        combination_dict: dict,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
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
                    prompting_scheme=prompting_scheme
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
