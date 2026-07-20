from config.model_config import ModelConfig
from experiments_acs.consistency_lotp_income.helper.llm_estimates import get_llm_estimated_prior_distribution, \
    get_llm_estimated_answer_distribution
from file_logging.read_and_write_json import save_as_json


def get_all_llm_estimates(
        cfg: ModelConfig,
        experiment_folder: str,
        combination_dict: dict,
        bin_edges: list,
) -> list[dict]:
    # Compute priors for all nodes
    llm_prior_list = []
    print("*" * 80)
    print("Computing priors for all levels...")
    print("*" * 80)
    for depth, combinations in combination_dict.items():
        # Assertion
        assert len(combinations) == (2 ** depth), (f"Unexpected number of combinations for depth {depth}. "
                                                   f"Expected {2 ** depth} but got {len(combinations)}.")

        # Special case (root node; no aggregation)
        if depth == 0:
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
