import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from file_logging.read_and_write_json import read_json, save_as_json
from utility.plot_style import set_plot_style
from utility.prior_weighted_avg import weighted_avg
from pathlib import Path
from utility.directories import get_results_dir
from utility.time_helper import get_experiment_timestamp

set_plot_style(sans_serif=False)


def main(
        experiment_name: str,
        datetime: str,
        cluster: bool,
):
    # Experiment folder
    timestamp = datetime.replace("__", "_")
    date, time, experiment_folder = get_experiment_timestamp(experiment_name=experiment_name, timestamp=timestamp)

    # Construct paths
    base_path = get_results_dir(
        experiment_name=experiment_name,
        cluster=cluster,
        use_timestamp=True,
        timestamp=datetime)
    json_path = base_path / "tree"

    # Convert shards to JSON tree
    convert_shards_to_json_tree(
        experiment_name=experiment_name,
        datetime=datetime,
        cluster=cluster,
    )

    # Build node and children mappings
    _, variance_results, total_variance = build_tree(json_path=json_path)

    # Save results to JSON file
    results_path = save_as_json(
        data=variance_results,
        experiment=experiment_folder,
        filename="final_summary_results.json",
        timestamp=False,
        cluster=cluster,
    )

    print(f"\nResults saved to {results_path}")

    # Plot results
    plot_within(variance_results=variance_results)
    plot_between(variance_results=variance_results)
    plot_sum(variance_results=variance_results, total_variance=total_variance)


def convert_shards_to_json_tree(
        experiment_name: str,
        datetime: str,
        cluster: bool,
):
    # Directory
    results_dir = get_results_dir(
        experiment_name=experiment_name,
        cluster=cluster,
        use_timestamp=True,
        timestamp=datetime,
        create_dir=False)

    # Check if directory exists
    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {results_dir}")

    # Load shards
    shards = sorted(results_dir.glob("shard_*__summary_results.json"))

    # Create tree dir
    tree_dir = results_dir / "tree"
    tree_dir.mkdir(parents=True, exist_ok=True)

    # Process each shard
    tree = {}
    for idx, shard in enumerate(shards):
        data = read_json(
            experiment_name=f"{experiment_name}/{datetime}",
            filename=shard.name,
            cluster=cluster,
        )

        # Save metadata from the first shard
        if idx == 0:
            metadata = {
                "survey_name": data["survey_name"],
                "survey_year": data["survey_year"],
                "target_column": data["target_column"] if "target_column" in data.keys() else "Unknown",
                "target_description": data["target_description"] if "target_description" in data.keys() else "Unknown",
                "cluster": cluster,
                "timestamp": datetime,
            }

            with open(tree_dir / "meta.json", "w") as json_file:
                json.dump(metadata, json_file, indent=4)

        # Create node label from filters
        attributes = []
        value_list = []

        # Create tree node
        node = {
            "id": data["node_id"],
            "parent_id": data["parent_id"],
            "level": data["level"],
            "filters": data["filters"],
            "attributes": attributes,
            "decisive_attribute": attributes[-1] if len(attributes) > 0 else "none",
            "decisive_value": value_list[-1] if len(value_list) > 0 else "none",
            "target_mean": data["target_mean"],
            "target_var": data["target_var"],
            "weight_after_filtering": data["total_weight"],
            "weighted_prior": data["weighted_prior"],
        }

        # Add node to tree
        level = data["level"]
        if level in tree.keys():
            tree[level].append(node)
        else:
            tree[level] = [node]

    # Save tree
    for level, nodes in tree.items():
        path = tree_dir / f"{level}.json"
        with open(path, "w") as json_file:
            json.dump(nodes, json_file, indent=4)

    return


def build_tree(json_path: Path) -> tuple[dict, dict, float]:
    # List all level files with distribution results
    level_files = sorted(json_path.glob("level_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(level_files)} level files with distribution results: {[f.name for f in level_files]}")

    level_tree = defaultdict(list)
    for level_file in level_files:
        nodes = json.loads(level_file.read_text())
        for node in nodes:
            level_tree[node["level"]].append(dict(node))

    total_variance = None
    variance_results = {}
    for level, nodes in level_tree.items():
        if len(nodes) <= 1:
            print(nodes[0])
            total_variance = nodes[0]["target_var"]
            continue

        agg_node = build_level_node_distribution(
            level_str=level,
            level_nodes=nodes,
        )

        variance_results[level] = {
            "within_group_variance": agg_node["within_group_variance"],
            "between_group_variance": agg_node["between_group_variance"],
        }

    return dict(level_tree), variance_results, total_variance


def build_level_node_distribution(
        level_str: str,
        level_nodes: list[dict],
) -> dict:
    # Level
    level = int(level_str.split("_")[1])

    # Initialize level nodes
    agg_node = {
        "level": level_str,
        "ground_truth_prior_list": [],
        "mean_list": [],
        "variance_list": [],
    }

    # Collect values from all nodes at this level
    for level_node in level_nodes:
        agg_node["ground_truth_prior_list"].append(level_node["weighted_prior"])
        agg_node["mean_list"].append(level_node["target_mean"])
        agg_node["variance_list"].append(level_node["target_var"])

    # Assertion for ground-truth prior --> make sure we have collected all nodes
    assert abs(sum(agg_node["ground_truth_prior_list"]) - 1.0) < 1e-8, (
        f"Priors on level {level} do not add up to 1.0, got {sum(agg_node['ground_truth_prior_list'])}")

    # Weighted average for the level node
    agg_node["within_group_variance"] = weighted_avg(values=agg_node["variance_list"],
                                                     priors=agg_node["ground_truth_prior_list"])
    agg_node["between_group_variance"] = between_subgroup_variance(subgroup_means=agg_node["mean_list"],
                                                                   priors=agg_node["ground_truth_prior_list"])

    return agg_node


def between_subgroup_variance(subgroup_means, priors):
    priors = np.asarray(priors, dtype=float)
    subgroup_means = np.asarray(subgroup_means, dtype=float)
    aggregate_mean = weighted_avg(values=subgroup_means, priors=priors)
    return np.sum(priors * (subgroup_means - aggregate_mean) ** 2)


def plot_within(variance_results: dict) -> None:
    labels = [r"$\ell=1$", r"$\ell=2$", r"$\ell=3$", r"$\ell=4$"]
    levels = list(variance_results.keys())

    x = np.arange(len(levels))
    within = np.array([variance_results[level]["within_group_variance"] for level in levels])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x, within, label="GT")

    ax.set_title("Within-Group Variance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(x)])
    ax.set_ylabel("Within-subgroup variance")
    ax.set_xlabel("Tree level")

    fig.tight_layout()

    plt.show()


def plot_between(variance_results: dict) -> None:
    labels = [r"$\ell=1$", r"$\ell=2$", r"$\ell=3$", r"$\ell=4$"]
    levels = list(variance_results.keys())
    x = np.arange(len(levels))
    between = np.array([variance_results[level]["between_group_variance"] for level in levels])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x, between, label="GT")

    ax.set_title("Between-Group Variance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(x)])
    ax.set_ylabel("Between-subgroup variance")
    ax.set_xlabel("Tree level")

    # Useful because the current between-group values are extremely close to zero.
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    plt.show()


def plot_sum(variance_results: dict, total_variance: float) -> None:
    labels = [r"$\ell=1$", r"$\ell=2$", r"$\ell=3$", r"$\ell=4$"]
    levels = list(variance_results.keys())
    x = np.arange(len(levels))
    between = np.array([
        variance_results[level]["within_group_variance"] + variance_results[level]["between_group_variance"]
        for level in levels])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x, between, label="GT")
    ax.axhline(y=total_variance, color="red", linestyle="--", label="Total variance")

    ax.set_title("Sum of Variances")
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(x)])
    ax.set_ylabel("Sum of variances")
    ax.set_xlabel("Tree level")

    # Useful because the current between-group values are extremely close to zero.
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Experiments
    experiment_name = "inter_intra_variance"
    datetime = "2026-06-26__16-00-00"
    cluster = False

    # Run main script
    main(
        experiment_name=experiment_name,
        datetime=datetime,
        cluster=cluster
    )
