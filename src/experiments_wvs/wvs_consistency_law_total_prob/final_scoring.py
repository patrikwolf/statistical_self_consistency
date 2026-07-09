import numpy as np

from file_logging.read_and_write_json import save_as_json


def _sort_list_by_q_id(items: list[dict]) -> None:
    items.sort(key=lambda x: int(x["question_identifier"].removeprefix("Q")))


def compute_final_scores(
        experiment_folder: str,
        sanity_check_results: dict,
        epsilon: float,
        num_checks_per_question: int = 4,
        rerun: bool = False,
        cluster: bool = False,
) -> dict:

    keys = {
        "within_tree": "wasserstein_distances_within_tree",
        "cross_tree": "wasserstein_distances_cross_tree"
    }
    num_checks_satisfied_per_question = {
        "within_tree": [],
        "cross_tree": []
    }
    total_num_of_checks_satisfied = {
        "within_tree": 0,
        "cross_tree": 0
    }
    total_num_of_checks = {
        "within_tree": 0,
        "cross_tree": 0
    }

    for q_id, entry in sanity_check_results.items():
        for label, dict_key in keys.items():
            # Extract WD results
            wasserstein_distances = entry[dict_key]

            # Make sure that dimension is correct
            assert len(
                wasserstein_distances) == num_checks_per_question, (f"Expected {num_checks_per_question} checks, "
                                                                    f"got {len(wasserstein_distances)}")

            # Check how many values are below threshold
            num_satisfied = float(np.sum(np.array(wasserstein_distances) < epsilon))
            num_checks_satisfied_per_question[label].append(
                {
                    "question_identifier": q_id,
                    "num_satisfied": num_satisfied,
                }
            )
            total_num_of_checks_satisfied[label] += num_satisfied
            total_num_of_checks[label] += len(wasserstein_distances)

    # Sort
    _sort_list_by_q_id(items=num_checks_satisfied_per_question["within_tree"])
    _sort_list_by_q_id(items=num_checks_satisfied_per_question["cross_tree"])

    # Collect results
    results = {
        "epsilon": epsilon,
        "num_checks_per_question": num_checks_per_question,
        "num_within_tree_checks_satisfied_per_question": num_checks_satisfied_per_question["within_tree"],
        "num_cross_tree_checks_satisfied_per_question": num_checks_satisfied_per_question["cross_tree"],
        "fraction_of_within_tree_checks_satisfied": total_num_of_checks_satisfied["within_tree"] / total_num_of_checks[
            "within_tree"],
        "fraction_of_cross_tree_checks_satisfied": total_num_of_checks_satisfied["cross_tree"] / total_num_of_checks[
            "cross_tree"],
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
