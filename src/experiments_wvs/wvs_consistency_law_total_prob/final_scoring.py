import numpy as np

from file_logging.read_and_write_json import save_as_json


def _sort_list_by_q_id(items: list[dict]) -> None:
    items.sort(key=lambda x: int(x["question_identifier"].removeprefix("Q")))


def compute_new_final_scores(
        experiment_folder: str,
        split_consistency_results: dict,
        order_consistency_results: dict,
        epsilon: float,
        num_split_checks_per_question: int,
        num_order_checks_per_question: int,
        rerun: bool = False,
        cluster: bool = False,
) -> dict:

    keys = {
        "within_tree": "wasserstein_distances_within_tree",
        "cross_tree": "wasserstein_distances_cross_tree",
    }
    num_checks_satisfied_per_question = {
        "within_tree": [],
        "cross_tree": [],
    }
    total_num_of_checks_satisfied = {
        "within_tree": 0,
        "cross_tree": 0
    }
    total_num_of_checks = {
        "within_tree": 0,
        "cross_tree": 0
    }

    # Split consistency
    for q_id, entry in split_consistency_results.items():
        # Iterate over within and cross checks
        for label, dict_key in keys.items():
            # Extract WD results
            wasserstein_distances = entry[dict_key]

            # Make sure that dimension is correct
            assert len(
                wasserstein_distances) == num_split_checks_per_question, (
                f"Expected {num_split_checks_per_question} checks, got {len(wasserstein_distances)}")

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

    # Combined number of satisfied checks
    combined_num_checks_satisfied_per_question = []
    for within_entry in num_checks_satisfied_per_question["within_tree"]:
        # Fetch cross tree results
        cross_tree_entry = next(
            (
                entry for entry in num_checks_satisfied_per_question["cross_tree"]
                if entry["question_identifier"] == within_entry["question_identifier"]
            ),
            None,
        )

        # Save
        combined_num_checks_satisfied_per_question.append({
            "question_identifier": within_entry["question_identifier"],
            "num_satisfied": within_entry["num_satisfied"] + cross_tree_entry["num_satisfied"],
        })

    # Fraction of combined checks averaged over all questions
    num_satisfied_combined = total_num_of_checks_satisfied["within_tree"] + total_num_of_checks_satisfied["cross_tree"]
    num_of_checks_combined = total_num_of_checks["within_tree"] + total_num_of_checks["cross_tree"]
    fraction_combined = num_satisfied_combined / num_of_checks_combined

    assert sum([entry["num_satisfied"] for entry in combined_num_checks_satisfied_per_question]) == num_satisfied_combined

    # Order consistency results
    order_consistency_scores = score_order_consistency(
        order_consistency_results=order_consistency_results,
        epsilon=epsilon,
        num_order_checks_per_question=num_order_checks_per_question,
    )

    # Collect results
    results = {
        "epsilon": epsilon,
        "split_consistency": {
            "num_checks_per_question_and_tree": num_split_checks_per_question,
            "num_checks_combined": 2 * num_split_checks_per_question,
            "num_within_tree_checks_satisfied_per_question": num_checks_satisfied_per_question["within_tree"],
            "num_cross_tree_checks_satisfied_per_question": num_checks_satisfied_per_question["cross_tree"],
            "combined_num_checks_satisfied_per_question": combined_num_checks_satisfied_per_question,
            "fraction_of_within_tree_checks_satisfied": (
                total_num_of_checks_satisfied["within_tree"] / total_num_of_checks["within_tree"]),
            "fraction_of_cross_tree_checks_satisfied": (
                total_num_of_checks_satisfied["cross_tree"] / total_num_of_checks["cross_tree"]),
            "num_satisfied_combined": num_satisfied_combined,
            "num_of_checks_combined": num_of_checks_combined,
            "fraction_combined": fraction_combined,
        },
        "order_consistency": order_consistency_scores,
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


def score_order_consistency(
        order_consistency_results: dict,
        epsilon: float,
        num_order_checks_per_question: int,
) -> dict:
    num_checks_satisfied_per_question = []
    for q_id, entry in order_consistency_results.items():
        # Extract WD results
        wasserstein_distances = entry["wasserstein_distances"]

        # Make sure that dimension is correct
        assert len(wasserstein_distances) == num_order_checks_per_question, (
            f"Expected {num_order_checks_per_question} checks, got {len(wasserstein_distances)}"
        )

        # Check how many values are below threshold
        num_satisfied = float(np.sum(np.array(wasserstein_distances) < epsilon))
        num_checks_satisfied_per_question.append(
            {
                "question_identifier": q_id,
                "num_satisfied": num_satisfied,
            }
        )

    # Average over all questions
    num_satisfied_checks = sum([entry["num_satisfied"] for entry in num_checks_satisfied_per_question])
    total_num_of_checks = len(order_consistency_results) * num_order_checks_per_question
    fraction_satisfied = num_satisfied_checks / total_num_of_checks

    # Collect results
    order_consistency_scores = {
        "num_checks_per_question": num_order_checks_per_question,
        "num_checks_satisfied_per_question": num_checks_satisfied_per_question,
        "total_num_of_checks": total_num_of_checks,
        "fraction_satisfied": fraction_satisfied,
    }

    return order_consistency_scores


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
        "cross_tree": "wasserstein_distances_cross_tree",
    }
    num_checks_satisfied_per_question = {
        "within_tree": [],
        "cross_tree": [],
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
        # Iterate over within and cross checks
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

    # Combined number of satisfied checks
    combined_num_checks_satisfied_per_question = []
    for within_entry in num_checks_satisfied_per_question["within_tree"]:
        # Fetch cross tree results
        cross_tree_entry = next(
            (
                entry for entry in num_checks_satisfied_per_question["cross_tree"]
                if entry["question_identifier"] == within_entry["question_identifier"]
            ),
            None,
        )

        # Save
        combined_num_checks_satisfied_per_question.append({
            "question_identifier": within_entry["question_identifier"],
            "num_satisfied": within_entry["num_satisfied"] + cross_tree_entry["num_satisfied"],
        })

    # Fraction of combined checks averaged over all questions
    num_satisfied_combined = total_num_of_checks_satisfied["within_tree"] + total_num_of_checks_satisfied["cross_tree"]
    num_of_checks_combined = total_num_of_checks["within_tree"] + total_num_of_checks["cross_tree"]
    fraction_combined = num_satisfied_combined / num_of_checks_combined

    assert sum([entry["num_satisfied"] for entry in combined_num_checks_satisfied_per_question]) == num_satisfied_combined

    # Collect results
    results = {
        "epsilon": epsilon,
        "num_checks_per_question_and_tree": num_checks_per_question,
        "num_checks_combined": 2 * num_checks_per_question,
        "num_within_tree_checks_satisfied_per_question": num_checks_satisfied_per_question["within_tree"],
        "num_cross_tree_checks_satisfied_per_question": num_checks_satisfied_per_question["cross_tree"],
        "combined_num_checks_satisfied_per_question": combined_num_checks_satisfied_per_question,
        "fraction_of_within_tree_checks_satisfied": total_num_of_checks_satisfied["within_tree"] / total_num_of_checks[
            "within_tree"],
        "fraction_of_cross_tree_checks_satisfied": total_num_of_checks_satisfied["cross_tree"] / total_num_of_checks[
            "cross_tree"],
        "num_satisfied_combined": num_satisfied_combined,
        "num_of_checks_combined": num_of_checks_combined,
        "fraction_combined": fraction_combined,
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
