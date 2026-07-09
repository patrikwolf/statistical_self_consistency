import numpy as np

from file_logging.read_and_write_json import save_as_json


def compute_final_scores_synthetic(
        experiment_folder: str,
        sanity_check_results: dict,
        epsilon: float,
        num_checks_per_question: int = 4,
        rerun: bool = False,
        cluster: bool = False,
) -> dict:

    # Within-tree
    within_res = sanity_check_results["wasserstein_distances_within_tree"]
    assert len(within_res) == num_checks_per_question
    num_within_tree_checks_satisfied = float(np.sum(np.array(within_res) < epsilon))

    # Cross-tree
    cross_res = sanity_check_results["wasserstein_distances_cross_tree"]
    assert len(cross_res) == num_checks_per_question
    num_cross_tree_checks_satisfied = float(np.sum(np.array(cross_res) < epsilon))

    results = {
        "epsilon": epsilon,
        "num_within_tree_checks_satisfied": num_within_tree_checks_satisfied,
        "num_cross_tree_checks_satisfied": num_cross_tree_checks_satisfied,
        "fraction_of_within_tree_checks_satisfied": num_within_tree_checks_satisfied / num_checks_per_question,
        "fraction_of_cross_tree_checks_satisfied": num_cross_tree_checks_satisfied / num_checks_per_question,
    }

    if rerun:
        filename = f"final_scores_{epsilon}.json"
    else:
        filename = "final_scores.json"

    # Save to file
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=filename,
        cluster=cluster,
    )
    print(f"\nFinal score saved to {results_path}")

    return results
