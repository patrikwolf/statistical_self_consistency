from file_logging.read_and_write_json import read_json
from experiments_wvs.ablation_logprobs_consistency.rerun_with_other_threshold import rerun_epsilon_evaluation


def load_and_process_sc_data(
        results: dict,
        model_name: str,
        experiment_name: str,
        timestamp: str,
        cluster: bool,
        new_epsilon: float = None,
):
    # Config for sanity checks
    sc_config = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="config.json",
        cluster=cluster,
    )

    # Check model name
    assert sc_config["model_name"] == model_name, (f"Provided model name {model_name} does not match the model"
                                                   f"name in the config file for the sanity checks.")

    # Country
    if "country" in sc_config:
        country = sc_config["country"]
    else:
        print(" ---> WARNING: Using country 'Canada' as default since it is not specified in the sanity check config.")
        country = "Canada"

    # Binary splits for sanity checks
    sc_splits = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="splits.json",
        cluster=cluster,
    )

    # Re-run evaluation if we want to change the epsilon
    if new_epsilon is None:
        # Final scores for sanity check
        final_scores = read_json(
            experiment_name=f"{experiment_name}/{timestamp}",
            filename="final_scores.json",
            cluster=cluster,
        )
    else:
        print(f" ---> WARNING: rerunning evaluation with new epsilon {new_epsilon}")
        sc_config["epsilon"] = new_epsilon

        # Re-run computation
        rerun_epsilon_evaluation(
            timestamp=timestamp,
            epsilon=new_epsilon,
            cluster=cluster,
            experiment_dir=experiment_name,
        )

        # Final scores for sanity check
        final_scores = read_json(
            experiment_name=f"{experiment_name}/{timestamp}",
            filename=f"final_scores_{new_epsilon}.json",
            cluster=cluster,
        )

    # Prepare dict
    if model_name not in results:
        results[model_name] = {}

    # Initialize results
    results[model_name][country] = {
        "timestamp": timestamp,
        "model_name": model_name,
        "sanity_checks": {
            "config": sc_config,
            "splits": sc_splits,
            "final_scores": final_scores,
        },
    }


def load_and_process_reconstruction_error(
        results: dict,
        model_name: str,
        experiment_name: str,
        timestamp: str,
        cluster: bool,
):
    # Config for reconstruction error
    gt_config = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="config.json",
        cluster=cluster,
    )

    # Check model name
    assert gt_config["model_name"] == model_name, (f"Provided model name {model_name} does not match the model"
                                                   f"name in the config file for the sanity checks.")

    # Country
    if "country" in gt_config:
        country = gt_config["country"]
    else:
        print(" ---> WARNING: Using country 'Canada' as default since it is not specified in the sanity check config.")
        country = "Canada"

    # Binary splits for sanity checks
    gt_splits = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="splits.json",
        cluster=cluster,
    )

    # Reconstruction errors
    reconstruction_errors = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="reconstruction_errors.json",
        cluster=cluster,
    )

    # Initialize results
    results[model_name][country]["ground_truth"] = {
        "config": gt_config,
        "splits": gt_splits,
        "reconstruction_errors": reconstruction_errors,
    }
