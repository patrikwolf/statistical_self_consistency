from file_logging.read_and_write_json import read_json


def load_data(
        model_name: str,
        experiment_name: str,
        timestamp: str,
        cluster: bool,
        epsilon: float,
) -> dict:
    # Config for sanity checks
    config = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="config.json",
        cluster=cluster,
    )

    # Check model name
    assert config["model_name"] == model_name, (f"Provided model name {model_name} does not match the model"
                                                f"name in the config file for the sanity checks.")

    # Binary splits for sanity checks
    splits = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="splits.json",
        cluster=cluster,
    )

    # LLM estimate dict
    llm_estimates_list = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="combined_llm_answer_results.json",
        cluster=cluster,
    )

    # Aggregation dict
    aggregation_dict = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="aggregation_dict.json",
        cluster=cluster,
    )

    # Final scores
    final_scores = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename=f"final_scores_{epsilon}.json",
        cluster=cluster,
    )

    # Collect results
    results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "config": config,
        "llm_estimates_list": llm_estimates_list,
        "aggregation_dict": aggregation_dict,
        "final_scores": final_scores,
        "splits": splits,
    }

    return results
