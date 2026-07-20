import numpy as np

from file_logging.read_and_write_json import save_as_json


def compute_new_final_scores(
        experiment_folder: str,
        split_consistency_results: dict,
        order_consistency_results: dict,
        epsilon: float,
        num_split_checks: int,
        num_order_checks: int,
        cluster: bool = False,
) -> dict:

    # Split consistency within-tree
    within_res = split_consistency_results["absolute_distances_within_tree"]
    assert len(within_res) == num_split_checks
    num_within_tree_checks_satisfied = float(np.sum(np.array(within_res) < epsilon))

    # Split consistency cross-tree
    cross_res = split_consistency_results["absolute_distances_cross_tree"]
    assert len(cross_res) == num_split_checks
    num_cross_tree_checks_satisfied = float(np.sum(np.array(cross_res) < epsilon))

    # Fraction of combined checks
    num_satisfied_combined = num_within_tree_checks_satisfied + num_cross_tree_checks_satisfied
    num_checks_combined = 2 * num_split_checks
    fraction_combined = num_satisfied_combined / num_checks_combined

    # Order consistency
    order_consistency_scores = score_order_consistency(
        order_consistency_results=order_consistency_results,
        epsilon=epsilon,
        num_order_checks_per_question=num_order_checks,
    )

    # Collect results
    results = {
        "epsilon": epsilon,
        "split_consistency": {
            "num_within_tree_checks_satisfied": num_within_tree_checks_satisfied,
            "num_cross_tree_checks_satisfied": num_cross_tree_checks_satisfied,
            "fraction_of_within_tree_checks_satisfied": num_within_tree_checks_satisfied / num_split_checks,
            "fraction_of_cross_tree_checks_satisfied": num_cross_tree_checks_satisfied / num_split_checks,
            "num_satisfied_combined": num_satisfied_combined,
            "num_checks_combined": num_checks_combined,
            "fraction_combined": fraction_combined,
        },
        "order_consistency": order_consistency_scores,
    }

    # Save results
    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"final_scores_{epsilon}.json",
        cluster=cluster,
    )
    print(f"\nFinal score saved to {results_path}")

    return results


def score_order_consistency(
        order_consistency_results: dict,
        epsilon: float,
        num_order_checks_per_question: int,
) -> dict:
    assert len(order_consistency_results) == num_order_checks_per_question
    num_checks_satisfied = float(np.sum(np.array(order_consistency_results) < epsilon))

    # Collect results
    order_consistency_scores = {
        "num_checks_per_question": num_order_checks_per_question,
        "num_checks_satisfied_per_question": num_checks_satisfied,
        "fraction_satisfied": num_checks_satisfied / num_order_checks_per_question,
    }

    return order_consistency_scores


def compute_final_scores(
    experiment_folder: str,
    sanity_check_results: dict,
    epsilon: float,
    cluster: bool = False,
) -> dict:

    distance_keys = {
        "within_tree": "wasserstein_distances_within_tree",
        "cross_tree": "wasserstein_distances_cross_tree",
    }

    # Initialize results
    results = {
        "epsilon": epsilon
    }

    # Iterate over within- and cross-tree results
    for label, key in distance_keys.items():
        distances = np.asarray(sanity_check_results[key])
        num_satisfied = float(np.sum(distances < epsilon))
        total_num_checks = len(distances)

        results[f"num_{label}_checks_satisfied"] = num_satisfied
        results[f"fraction_of_{label}_checks_satisfied"] = (
            num_satisfied / total_num_checks if total_num_checks > 0 else np.nan
        )

    results_path = save_as_json(
        data=results,
        experiment=experiment_folder,
        filename=f"final_scores_{epsilon}.json",
        cluster=cluster,
    )
    print(f"\nFinal score saved to {results_path}")

    return results
