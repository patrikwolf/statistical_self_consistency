from file_logging.read_and_write_json import save_as_json, read_json


def evaluate_model_samples(
        experiment_timestamp: str,
        epsilon: float,
        cluster: bool,
):
    # Experiment name
    experiment_name = f"decision_model_evaluation/{experiment_timestamp}"

    # Read results
    sampling_results = read_json(
        experiment_name=experiment_name,
        filename="llm_sampling_results.json",
        cluster=cluster,
    )

    print(f"Evaluating model: {sampling_results['model']}")
    print(f"Number of runs: {sampling_results['num_runs']}")

    for sample_set in sampling_results["samples"]:
        print(f"\nQuestion: {sample_set['root']['question']}")

        # Extract model estimates for the current question set
        estimates = sample_set["model_estimates"]

        # Root estimate
        root_yes = estimates["root"]["average"]

        # Aggregate estimate from left and right children
        left_child = estimates["left_child"]
        right_child = estimates["right_child"]
        prior = estimates["prior"]["average"]
        aggregate_yes = left_child["average"] * prior + right_child["average"] * (1 - prior)

        print(f"Root yes: {root_yes}")
        print(f"Aggregate yes: {aggregate_yes}")

        # Consistency check
        absolute_difference = abs(root_yes - aggregate_yes)
        consistent = absolute_difference <= epsilon

        # Collect results
        eval_result = {
            "root_yes": root_yes,
            "aggregate_yes": aggregate_yes,
            "abs_difference": absolute_difference,
            "consistent": consistent
        }

        # Add to dict
        sample_set["evaluation"] = eval_result

    # Compute overall consistency score
    total_samples = len(sampling_results["samples"])
    consistent_samples = sum(1 for sample_set in sampling_results["samples"] if sample_set["evaluation"]["consistent"])
    consistency_score = consistent_samples / total_samples if total_samples > 0 else 0

    print("\n" + "*" * 80)
    print(f"Overall consistency score: {consistency_score}")

    # Add overall consistency score to results
    sampling_results["overall_consistency_score"] = consistency_score
    sampling_results["epsilon"] = epsilon

    # Save augmented results with evaluation
    filename = save_as_json(
        data=sampling_results,
        experiment=experiment_name,
        filename=f"evaluation_{epsilon}.json",
        cluster=cluster,
    )

    print(f"\nSampling results saved to {filename}")


if __name__ == "__main__":
    # Timestamp
    # experiment_timestamp = "2026-10-21__14-32-27"         # test run on openai/gpt-6-luna
    # experiment_timestamp = "2026-10-21__14-44-07"  # test run on openai/gpt-4o-mini
    # experiment_timestamp = "2026-10-21__15-06-38"  # openai/gpt-4o-mini (on cluster)
    # experiment_timestamp = "2026-10-21__15-23-55"  # inception/mercury-2.5 (on cluster)
    experiment_timestamp = "2026-10-21__15-56-24"  # openai/gpt-6-luna (on cluster)

    # Consistency threshold
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3]

    # Evaluate model samples
    for eps in epsilon_values:
        print(f"\nEvaluating with epsilon = {eps}...")
        evaluate_model_samples(
            experiment_timestamp=experiment_timestamp,
            epsilon=eps,
            cluster=True,
        )

    print("\n" + "*" * 80)
    print("Done evaluating model samples!")
    print("*" * 80)
