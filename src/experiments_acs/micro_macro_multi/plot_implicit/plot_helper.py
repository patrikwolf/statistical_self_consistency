import json
import yaml
import numpy as np

from experiments_acs.tree_income_threshold.latex.aggregation_helper import load_joint_llm_prior
from experiments_acs.tree_income_threshold.latex.convert_shards_to_json_tree import convert_shards_to_json_tree
from file_logging.read_and_write_json import read_json
from utility.bootstrap import bootstrap_confidence_interval
from utility.directories import get_results_dir
from utility.prior_weighted_avg import weighted_avg


def read_shards_to_json_list(experiment_name: str, datetime: str, cluster: bool) -> list[dict]:
    # Directory
    results_dir = get_results_dir(
        experiment_name=experiment_name,
        cluster=cluster,
        use_timestamp=True,
        timestamp=datetime,
        create_dir=False)

    # Check if results_dir exists
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Specified results directory is not a valid directory: {results_dir}")

    # Load shards
    shards = sorted(results_dir.glob("shard_*__summary_results.json"))

    # Process each shard
    results = []
    for idx, shard in enumerate(shards):
        data = read_json(
            experiment_name=f"{experiment_name}/{datetime}",
            filename=shard.name,
            cluster=cluster,
        )

        # Add to list
        results.append(data)

    return results


def read_baseline_to_json_list(experiment_name: str, experiment_list: list[dict], load_joint_llm_priors: bool) -> list[dict]:
    results = []
    for experiment_dict in experiment_list:
        # Extract data
        datetime = experiment_dict["timestamp"]
        cluster = experiment_dict["cluster"]

        # Convert shards to JSON tree
        tree, metadata, _ = convert_shards_to_json_tree(
            experiment_name=experiment_name,
            datetime=datetime,
            cluster=cluster,
        )

        # Aggregate results across levels
        aggregated_tree = aggregate_baseline_results(tree=tree, load_joint_llm_priors=load_joint_llm_priors)

        # Collect results
        exp_results = {
            "metadata": metadata,
            "aggregated_tree": aggregated_tree,
        }

        # Add to list
        results.append(exp_results)

    return results


def aggregate_baseline_results(tree: dict, load_joint_llm_priors: bool) -> dict:
    aggregated_tree = {}
    for level_str, data in tree.items():
        level = int(level_str.split("_")[1])

        # Aggregate nodes at the same level by computing weighted averages of ground truth and LLM predictions.
        if len(data) == 1:
            assert level == 0
            node = data[0]

            # Extract information
            level_node = {}
            level_node["id"] = 0
            level_node["level"] = f"level_{level}"
            level_node["improved_age_desc"] = node["improved_age_desc"]
            level_node["income_threshold"] = node["income_threshold"]
            level_node["aggregated_ground_truth_probability"] = node["ground_truth_probability"]
            level_node["aggregated_llm_prediction_avg"] = node["llm_prediction_avg"]
            level_node["aggregated_estimation_error"] = abs(node["ground_truth_probability"]
                                                            - node["llm_prediction_avg"])
            level_node["aggregated_estimation_error_list"] = abs(node["ground_truth_probability"]
                                                                 - np.array(node["llm_prediction_list"])).tolist()
        else:
            level_node = build_level_node(
                level=level,
                data=data,
                load_joint_llm_priors=load_joint_llm_priors,
            )

        # Add level node to tree
        aggregated_tree[level] = level_node

    return aggregated_tree


def build_level_node(
        level: int,
        data: list[dict],
        load_joint_llm_priors: bool,
):
    assert len(data) == 2 ** level, (f"Expected {2 ** level} nodes at level {level}, not got {len(data)} "
                                     f"(timestamp: {data[0]['timestamp']})")

    # Extract information
    num_of_samples = data[0]["num_of_samples"]
    improved_age_desc = data[0]["improved_age_desc"] if "improved_age_desc" in data[0] else "n/a"

    # Initialize level nodes
    level_node = {
        "id": level,
        "level": f"level_{level}",
        "num_of_samples": num_of_samples,
        "improved_age_desc": improved_age_desc,
        "income_threshold": data[0]["income_threshold"],
        "attributes": data[0]["attributes"],
        "label": ", ".join(data[0]["attributes"]),
        "load_joint_llm_priors": load_joint_llm_priors,
        "ground_truth_list": [],
        "loaded_joint_priors": [],
        "list_of_llm_prediction_lists": [],
        "llm_prediction_avg_list": [],
        "unnormalized_llm_prior_avg_list": [],
        "ground_truth_prior_list": [],
        "priors": [],
    }

    # Aggregate values from all nodes at this level
    for node in data:
        # Load jointly elicited prior from files
        if load_joint_llm_priors:
            loaded_joint_prior = load_joint_llm_prior(filters=node["filters"],
                                                      model=node["model"])["normalized_averaged_prior"]
        else:
            loaded_joint_prior = -1

        # Save results
        level_node["ground_truth_list"].append(node["ground_truth_probability"])
        level_node["list_of_llm_prediction_lists"].append(node["llm_prediction_list"])
        level_node["llm_prediction_avg_list"].append(node["llm_prediction_avg"])
        level_node["loaded_joint_priors"].append(loaded_joint_prior)
        level_node["ground_truth_prior_list"].append(node["ground_truth_prior"])

    # Assertion for LLM predictions
    num_level_nodes = 2 ** level
    assert len(level_node["llm_prediction_avg_list"]) == num_level_nodes, (
        f"Dimension mismatch, expected {num_level_nodes} entries in level {level}, but got "
        f"{len(level_node['llm_prediction_avg_list'])}")
    assert len(level_node["list_of_llm_prediction_lists"][0]) == num_of_samples, (
        f"Dimension mismatch, expected {num_of_samples} entries LLM predictions for every level nodes, but go "
        f"{len(level_node['llm_prediction_avg_list'][0])}")

    # Assertion for ground-truth prior --> make sure we have collected all nodes
    assert abs(sum(level_node["ground_truth_prior_list"]) - 1.0) < 1e-8, f"Priors on level {level} do not add up to 1.0"

    # Choose pior
    if load_joint_llm_priors:
        # Use jointly elicited priors
        level_node["priors"] = level_node["loaded_joint_priors"]
    else:
        # Resort to ground-truth prior from survey
        level_node["priors"] = level_node["ground_truth_prior_list"]

    # Assertion for prior
    assert abs(sum(level_node["priors"]) - 1.0) < 1e-8, (f"Priors on level {level} do not add up to 1.0, "
                                                         f"sum is {sum(level_node['priors'])}")

    # Weighted average for the level node
    level_node["aggregated_ground_truth_probability"] = weighted_avg(level_node["ground_truth_list"],
                                                                     level_node["priors"])
    level_node["aggregated_llm_prediction_avg"] = weighted_avg(level_node["llm_prediction_avg_list"],
                                                               level_node["priors"])

    # Estimation error (GT vs. LLM estimate) for every node in level
    list_of_estimation_error_list = [abs(gt - np.array(pred_list)) for gt, pred_list in
                                     zip(level_node["ground_truth_list"], level_node["list_of_llm_prediction_lists"])]
    avg_estimation_error_list = [abs(gt - pred) for gt, pred in
                                 zip(level_node["ground_truth_list"], level_node["llm_prediction_avg_list"])]

    assert len(list_of_estimation_error_list) == num_level_nodes, (
        f"Expected {num_level_nodes} estimation error lists, but got {len(list_of_estimation_error_list)}"
    )
    assert len(list_of_estimation_error_list[0]) == num_of_samples, (
        f"Expected {num_of_samples} estimation errors in every list, but got {len(list_of_estimation_error_list[0])}"
    )

    # Weighted average of estimation error for the level node
    level_node["aggregated_estimation_error_list"] = weighted_avg(list_of_estimation_error_list,
                                                                  level_node["priors"]).tolist()
    level_node["aggregated_estimation_error"] = weighted_avg(avg_estimation_error_list, level_node["priors"])

    return level_node


def process_micro_macro_results(results: list[dict]) -> list[dict]:
    plot_results = []

    for result in results:
        # Ground-truth target
        gt_probability = result["ground_truth_probability"]

        # No root level estimate; only deeper levels
        prediction_list = result["micro_macro"]["llm_prediction_list"]
        error_list = abs(gt_probability - np.array(prediction_list))
        mean, lower, upper = bootstrap_confidence_interval(values=error_list, n_boot=1_000, ci=0.9)

        # Collect results
        micro_macro_results = {
            "income_threshold": result["income_threshold"],
            "prediction_list": prediction_list,
            "error_list": error_list,
            "mean_error": mean,
            "lower_error": lower,
            "upper_error": upper,
        }

        # Add to dict
        plot_results.append(micro_macro_results)

    return plot_results


def process_baseline_results(results: list[dict]) -> list[dict]:
    plot_results = []

    for result in results:
        y_mean = []
        y_lower = []
        y_upper = []

        for level_res in result["aggregated_tree"].values():
            aggregated_error_list = level_res["aggregated_estimation_error_list"]
            mean, lower, upper = bootstrap_confidence_interval(values=aggregated_error_list, n_boot=1_000, ci=0.9)

            # Add to lists
            y_mean.append(mean)
            y_lower.append(lower)
            y_upper.append(upper)

        # Collect results
        aggregation_results = {
            "income_threshold": result["metadata"]["income_threshold"],
            "levels": list(result["aggregated_tree"].keys()),
            "y_mean": y_mean,
            "y_lower": y_lower,
            "y_upper": y_upper,
        }

        # Add to dict
        plot_results.append(aggregation_results)

    return plot_results


def load_all_results(
        mtm_experiment_name: str,
        mtm_timestamp: str,
        mtm_cluster: bool,
        baseline_experiment_log_file: str,
        baseline_experiment_name: str,
        load_joint_llm_priors: bool,
) -> tuple[list, list]:
    # Macro to micro
    mtm_results = read_shards_to_json_list(
        experiment_name=mtm_experiment_name,
        datetime=mtm_timestamp,
        cluster=mtm_cluster
    )

    # Load experiment logs
    with open(baseline_experiment_log_file, "r") as f:
        logs = yaml.safe_load(f)

    baseline_experiment_list = logs["experiments_log"]["experiments"]
    baseline_results = read_baseline_to_json_list(
        experiment_name=baseline_experiment_name,
        experiment_list=baseline_experiment_list,
        load_joint_llm_priors=load_joint_llm_priors,
    )

    # Assertions
    assert mtm_results[0]["model"].split("/")[1] == baseline_results[0]["metadata"]["model"], (
        f"Model mismatch, MtM: {mtm_results[0]['model'].split('/')[1]}, "
        f"baseline: {baseline_results[0]['metadata']['model']}")
    assert mtm_results[0]["reasoning_effort"] == baseline_results[0]["metadata"]["reasoning_effort"], \
        "Mismatch in reasoning effort"

    return mtm_results, baseline_results


def process_and_combine_results(
        mtm_results: list[dict],
        baseline_results: list[dict],
) -> dict:
    micro_macro_results = process_micro_macro_results(results=mtm_results)
    agg_results = process_baseline_results(results=baseline_results)

    combined_results = {}
    for mtm_res in micro_macro_results:
        income_threshold = mtm_res["income_threshold"]

        # Fine baseline results for same threshold
        baseline_res = next(el for el in agg_results if el["income_threshold"] == income_threshold)

        combined_results[income_threshold] = {
            "micro_macro": mtm_res,
            "aggregation_baseline": baseline_res
        }

    return combined_results


if __name__ == "__main__":
    experiment_name = "micro_macro_implicit"
    timestamp = "2026-06-01__08-30-38"
    cluster = True

    # Convert shards to JSON tree
    results = read_shards_to_json_list(
        experiment_name=experiment_name,
        datetime=timestamp,
        cluster=cluster,
    )

    print("Done!")
    # print(json.dumps(results, indent=4))

    # Baseline
    experiment_name = "micro_macro_aggregation_baseline"
    experiment_list = [
        {"income_threshold": 1000, "timestamp": "2026-05-30__00-15-22", "cluster": True},
        {"income_threshold": 10000, "timestamp": "2026-05-25__15-49-12", "cluster": True}
    ]

    baseline_results = read_baseline_to_json_list(
        experiment_name=experiment_name,
        experiment_list=experiment_list,
        load_joint_llm_priors=True
    )

    print("Done!")
    print(json.dumps(baseline_results, indent=4))

    plot_results = process_baseline_results(results=baseline_results)

    print("Done!")
    print(json.dumps(plot_results, indent=4))

    # Load all results
    mtm_experiment_name = "micro_macro_implicit"
    mtm_timestamp = "2026-06-01__08-30-38"
    mtm_cluster = True

    # Baseline
    baseline_experiment_log_file = "exp_baseline_gpt54.yaml"
    baseline_experiment_name = "micro_macro_aggregation_baseline"
    load_joint_llm_priors = True

    load_all_results(
        mtm_experiment_name=mtm_experiment_name,
        mtm_timestamp=mtm_timestamp,
        mtm_cluster=mtm_cluster,
        baseline_experiment_log_file=baseline_experiment_log_file,
        baseline_experiment_name=baseline_experiment_name,
        load_joint_llm_priors=load_joint_llm_priors,
    )
