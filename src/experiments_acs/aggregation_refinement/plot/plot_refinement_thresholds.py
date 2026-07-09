import yaml
import json
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path
from design_elements.color_palette import get_histogram_colors
from file_logging.read_and_write_json import save_as_json
from experiments_acs.aggregation_refinement.plot.helper import add_to_results
from utility.directories import get_results_dir
from experiments_acs.tree_income_threshold.latex.aggregation_helper import build_aggregated_threshold_tree
from experiments_acs.tree_income_threshold.latex.convert_shards_to_json_tree import convert_shards_to_json_tree
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir
from utility.bootstrap import bootstrap_confidence_interval


def process_logs(logs: dict, overwrite_cluster_to_false: bool, load_joint_llm_priors: bool) -> tuple[dict, str]:
    # Initialization
    results = {}

    logs_items = list(logs.items())
    assert len(logs_items) == 1, f"Expected exactly 1 experiment in logs, found {len(logs_items)}"

    # Process experiments
    experiment_name, description = logs_items[0]
    results["decomposition"] = description["decomposition"]
    results["model"] = description["model"]
    results["refinement_results"] = []
    results["income_thresholds"] = []
    experiment_name = description["experiment_name"]

    for experiment in description["experiments"]:
        income_threshold = experiment["income_threshold"]
        datetime = experiment["timestamp"]
        cluster = experiment["cluster"]

        if overwrite_cluster_to_false:
            cluster = False

        # Add to results
        results["income_thresholds"].append(income_threshold)

        # Convert shards to JSON tree
        convert_shards_to_json_tree(
            experiment_name=experiment_name,
            datetime=datetime,
            cluster=cluster,
        )

        # Construct paths
        base_path = get_results_dir(experiment_name=experiment_name, cluster=cluster, use_timestamp=True, timestamp=datetime)
        json_path = base_path / "tree"

        # Load all level files
        level_files = sorted(json_path.glob("level_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        print(f"Found {len(level_files)} level files: {[f.name for f in level_files]}")

        # Load metadata
        meta_path = json_path / "meta.json"
        metadata = json.loads(meta_path.read_text())

        assert metadata["model"] == results["model"].split("/")[1], "Model mismatch"
        if ("decomposition_attribute_list" in metadata
                and metadata["decomposition_attribute_list"] != "unknown"):
            assert metadata["decomposition_attribute_list"] == results["decomposition"], \
                "Decomposition attribute list mismatch"
        else:
            print("  ---> WARNING: could not find decomposition_attribute_list in metadata")

        # Add additional properties to results
        add_to_results(results=results, key="num_of_samples", value=metadata["num_of_samples"])
        add_to_results(results=results, key="reasoning_effort", value=metadata["reasoning_effort"])
        add_to_results(results=results, key="prompting_scheme", value=metadata["prompting_scheme"])
        add_to_results(results=results, key="llm_estimated_priors", value=metadata["llm_estimated_priors"])
        add_to_results(results=results, key="set_nan_to_zero", value=metadata["set_nan_to_zero"])
        add_to_results(results=results, key="income_greater_than_threshold", value=metadata["income_greater_than_threshold"])

        # Build node and children mappings
        tree = build_aggregated_threshold_tree(level_files=level_files, load_joint_llm_priors=load_joint_llm_priors)

        # Assert that ground-truth does not change from level to level
        ground_truth_probability = tree[0]["ground_truth_probability"]
        if not load_joint_llm_priors:
            for key, value in tree.items():
                assert abs(value["aggregated_ground_truth_probability"] - ground_truth_probability) < 1e-8, \
                    (f"Ground-truth probability at level {key} deviates from root-level ground-truth."
                     f"Expected {ground_truth_probability}, but got {value['aggregated_ground_truth_probability']}")
        else:
            print("Aggregating ground-truth estimate at lower levels with LLM-estimated priors might yields "
                  "different results from root-level ground-truth. Therefore, we use the root-level ground-truth "
                  "to evaluate the estimation error.")

        # Extract results from each level file
        refinement_results = {}
        for level_idx, level_results in tree.items():
            refinement_results[level_idx] = {
                "level": level_results["level"],
                "label": level_results["label"],
                "attributes": level_results["attributes"],
                "aggregated_error": abs(ground_truth_probability - level_results["aggregated_llm_prediction_avg"]),
                # List for boostrap confidence intervals
                "aggregated_error_list": level_results["aggregated_estimation_error_list"],
            }

        # Add results to dict
        results["refinement_results"].append({
            "income_threshold": income_threshold,
            "refinement_results": refinement_results,
        })

    return results, experiment_name


def plot_aggregation_refinement(results: dict, plot_dir: Path, experiment_name: str, load_joint_llm_priors: bool):
    # Title
    title = (r"\textbf{Estimation Error}"
             "\n"
             r"\normalsize "
             f"Model: {results['model']}, reasoning effort: '{results['reasoning_effort']}', "
             f"prompting scheme: '{results['prompting_scheme']}'"
             "\n"
             r"\normalsize "
             f"Loaded and used jointly elicited LLM priors: {load_joint_llm_priors}, "
             f"(individual LLM-estimated priors available: {results['llm_estimated_priors']})"
             "\n"
             r"\normalsize "
             f"Task: estimate the probability that an individual belonging to a subpopulation has an income "
             f"above a predefined threshold"
             "\n"
             r"\normalsize "
             f"Income thresholds (USD): {results['income_thresholds']}, Income above or below threshold: "
             f"{results['income_greater_than_threshold']}"
             "\n"
             r"\normalsize "
             f"Number of repetitions per setting: {results['num_of_samples']}"
             "\n")

    # Plot size
    plt.figure(figsize=(9, 9))

    # Colormap
    entries = results["refinement_results"]
    colors = get_histogram_colors(llm=True, num_colors=len(entries))

    # Store values per level for averaging
    values_by_level = defaultdict(list)

    # Plot all results in the list
    all_x = set()
    for i, entry in enumerate(entries):
        x = []
        y_mean = []
        y_lower = []
        y_upper = []

        # Collect data from dict
        for level, result in sorted(entry["refinement_results"].items(), key=lambda t: int(t[0])):
            level_int = int(level)
            all_x.add(level_int)
            error_list = result["aggregated_error_list"]
            mean, lower, upper = bootstrap_confidence_interval(values=error_list, n_boot=1_000, ci=0.9)

            # Add to list
            x.append(level_int)
            y_mean.append(mean)
            y_lower.append(lower)
            y_upper.append(upper)

            # Save in dict
            values_by_level[level_int].append(mean)

        # Plot
        color = colors[i]
        plt.plot(
            x, y_mean,
            "--",
            marker="o",
            color=color,
            alpha=0.8,
            lw=1.4,
            markersize=5,
            zorder=1,
            label=r"$\tau = " + f"{entry['income_threshold']:,}$ USD".replace(",", r"\,")
        )

        # Bootstrap CI
        plt.fill_between(x, y_lower, y_upper, color=color, alpha=0.2, zorder=0)

    # Compute average trend
    x_avg = sorted(values_by_level.keys())
    y_avg = [np.mean(values_by_level[level]) for level in x_avg]

    # Plot average prominently on top
    plt.plot(
        x_avg, y_avg,
        marker="o",
        color="blue",
        lw=3.5,
        markersize=10,
        label="Average",
        zorder=10,
    )

    # Formatting
    plt.title(title)
    plt.xlabel("Level", labelpad=10)
    plt.ylabel("Error of aggregated estimate", labelpad=15)
    x_ticks = list(range(min(all_x), max(all_x) + 1))
    plt.xticks(x_ticks)
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()

    # Add prior to plot name
    prior_usage = "llm_prior" if load_joint_llm_priors else "gt_prior"

    # Save plot
    plot_subdir = plot_dir / prior_usage
    plot_subdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_subdir / f"threshold_{experiment_name}.png", dpi=300)
    plt.savefig(plot_subdir / f"threshold_{experiment_name}.pdf")

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "aggregation_refinement"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    set_plot_style(sans_serif=False, factor=1.225)

    # Line width
    plt.rcParams.update({
        "axes.linewidth": 2,
        "grid.linewidth": 1.75,
    })

    # Load experiment logs
    with open("experiments_refinement_threshold_gt_priors.yaml", "r") as f:
        logs = yaml.safe_load(f)

    # Toggle to use new jointly elicited priors
    load_joint_llm_priors = True

    # Process results
    results, experiment_name = process_logs(logs=logs,
                                            overwrite_cluster_to_false=False,
                                            load_joint_llm_priors=load_joint_llm_priors)

    # Filename
    filename = f"thresholds_{"llm_prior" if load_joint_llm_priors else "gt_prior"}.json"

    # Save results to JSON file
    results_path = save_as_json(
        data=results,
        experiment="figure_aggregation_refinement",
        filename=filename,
        timestamp=False)
    print(f"\nResults saved to {results_path}")

    # Remove some entries
    masked_refinement_res = []
    for entry in results["refinement_results"]:
        if entry["income_threshold"] == 100_000:
            continue
        else:
            masked_refinement_res.append(entry)
    results["refinement_results"] = masked_refinement_res

    # Plot
    plot_aggregation_refinement(results=results,
                                plot_dir=plot_dir,
                                experiment_name=experiment_name,
                                load_joint_llm_priors=load_joint_llm_priors)
