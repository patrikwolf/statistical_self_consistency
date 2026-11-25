from file_logging.read_and_write_json import save_as_json, read_json


def evaluate_model_samples(
        experiment_timestamp: str,
        epsilon: float,
):
    # Experiment name
    experiment_name = f"decision_model_evaluation/{experiment_timestamp}"

    # Read results
    sampling_results = read_json(
        experiment_name=experiment_name,
        filename="sampling_results.json",
        cluster=False,
    )

    print(f"Evaluating model: {sampling_results['model']}")
    print(f"Number of runs: {sampling_results['num_runs']}")

    for sample_set in sampling_results["samples"]:
        print(f"\nQuestion: {sample_set['root']['question']}")

        # Extract model estimates for the current question set
        estimates = sample_set["model_estimates"]

        # Root estimate
        root_yes = estimates["root"]["average"]["yes"]

        # Aggregate estimate from left and right children
        left_child = estimates["left_child"]
        right_child = estimates["right_child"]
        priors = estimates["prior"]["average"]
        aggregate_yes = left_child["average"]["yes"] * priors[left_child["split_option"]] + right_child["average"][
            "yes"] * priors[right_child["split_option"]]

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
        cluster=False,
    )

    print(f"\nSampling results saved to {filename}")


if __name__ == "__main__":
    # Timestamp
    # experiment_timestamp = "2026-10-15__13-46-42"       # liquid/d1
    # experiment_timestamp = "2026-10-16__08-42-30"       # liquid/d1-ultra
    # experiment_timestamp = "2026-10-16__10-09-08"       # typesafe/jev-1.13
    # experiment_timestamp = "2026-10-16__09-19-12"       # typesafe/jev-1.14
    # experiment_timestamp = "2026-10-16__10-53-46"       # openai/gpt-6-luna-decisions
    # experiment_timestamp = "2026-10-16__11-04-05"       # microsoft/microsoft-decision-1
    # experiment_timestamp = "2026-10-16__11-14-30"       # nace-ai/drex-v1.5
    # experiment_timestamp = "xxxxx"       # nace-ai/drex-v2.0
    # experiment_timestamp = "2026-10-16__11-26-14"       # inception/mercury-decide
    experiment_timestamp = "2026-10-16__10-52-19"         # convaiinnovations/laya

    # Consistency threshold
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3]

    # Evaluate model samples
    for eps in epsilon_values:
        print(f"\nEvaluating with epsilon = {eps}...")
        evaluate_model_samples(
            experiment_timestamp=experiment_timestamp,
            epsilon=eps,
        )

    print("\n" + "*" * 80)
    print("Done evaluating model samples!")
    print("*" * 80)
