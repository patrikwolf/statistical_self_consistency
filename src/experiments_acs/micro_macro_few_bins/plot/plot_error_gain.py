import yaml
import numpy as np
import matplotlib.pyplot as plt

from experiments_acs.micro_macro_few_bins.plot.convert_baseline_shards_to_json_tree import \
    convert_baseline_shards_to_json_tree
from utility.plot_style import set_plot_style
from file_logging.read_and_write_json import read_json
from utility.directories import get_plot_dir
from experiments_acs.tree_llm_few_bins.latex_helper.aggregation_helper import build_aggregated_distribution_tree
from utility.wasserstein_helper import compute_wasserstein_distance

# Set style
set_plot_style(sans_serif=False, factor=1.225)

# Line width
plt.rcParams.update({
    "axes.linewidth": 2,
    "grid.linewidth": 1.75,
})


def main_error_gain(
        plot_dir_name: str,
        experiment_log_file: str,
        load_joint_llm_priors: bool,
):
    # Load results
    mtm_results, baseline_results = load_all_results(
        experiment_log_file=experiment_log_file,
        load_joint_llm_priors=load_joint_llm_priors,
    )

    # Process results
    mtm_error_list = compute_mtm_error_list(mtm_result=mtm_results)
    baseline_level_wise_error_lists = compute_baseline_error_list(baseline_result=baseline_results)

    # Compute error gains
    bootstrapped_error_gains = compute_bootstrapped_error_gains(
        mtm_error_list=mtm_error_list,
        baseline_level_wise_error_lists=baseline_level_wise_error_lists,
    )

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Title
    title = (r"\huge\textbf{Micro to Macro Reasoning: Error Gains}"
             "\n"
             r"\normalsize "
             f"Model: {mtm_results['model']}, reasoning effort: '{mtm_results['reasoning_effort']}', "
             f"prompting scheme: '{mtm_results['prompting_scheme']}'"
             "\n"
             r"\normalsize "
             f"Bin edges: {mtm_results['ground_truth']['bin_edges']} USD"
             "\n"
             r"\normalsize "
             f"Loaded and used jointly elicited LLM priors: {load_joint_llm_priors}"
             "\n"
             )

    # Plot size
    plt.figure(figsize=(9, 9))

    # Collect results
    labels = ["Micro to macro"]
    entries = [bootstrapped_error_gains["micro_macro"]]
    for level, entry in bootstrapped_error_gains["aggregation_baseline"].items():
        if int(level) == 0:
            continue
        labels.append(f"Level {level}")
        entries.append(entry)

    x = np.arange(len(labels))
    means = np.array([entry["mean"] for entry in entries])
    lower = np.array([entry["lower"] for entry in entries])
    upper = np.array([entry["upper"] for entry in entries])

    # Asymmetric error bars: distances from mean to lower/upper CI bounds
    yerr = np.vstack([
        means - lower,
        upper - means,
    ])

    # Color
    colors = ["tab:orange"] + ["tab:blue"] * (len(labels) - 1)

    # Bar plot
    plt.bar(
        x,
        means,
        color=colors,
        yerr=yerr,
        capsize=5,
        alpha=0.8,
    )

    # Formatting
    plt.title(title)
    plt.xticks(np.arange(len(labels)), labels, rotation="vertical")
    plt.ylabel("Error gain")
    plt.grid()
    plt.tight_layout()

    # Save plot
    model_name = mtm_results['model'].split("/")[1].replace(".", "-")
    plt.savefig(plot_dir / f"error_gains_few_bins_{model_name}.png", dpi=300)
    plt.savefig(plot_dir / f"error_gains_few_bins_{model_name}.pdf")

    # Show plot
    plt.show()


def compute_mtm_error_list(mtm_result: dict):
    # Ground-truth target
    gt_distribution = mtm_result["ground_truth"]["distribution"]

    # No root level estimate; only deeper levels
    bins = sorted(
        list(mtm_result["micro_macro"]["avg_bin_distribution"].keys()),
        key=lambda qid: int(qid.removeprefix("Bin ")),
    )
    raw_distribution_list = mtm_result["micro_macro"]["llm_distribution_list"]
    error_list = []
    for raw_distribution in raw_distribution_list:
        # Convert distribution
        llm_distribution = [raw_distribution[bin] for bin in bins]

        # Compute Wasserstein distance
        error = compute_wasserstein_distance(
            d1=gt_distribution,
            d2=llm_distribution,
        )

        # Add to list
        error_list.append(error)

    return error_list


def compute_baseline_error_list(baseline_result: dict):
    error_lists = {}
    for level, level_res in baseline_result["aggregated_tree"].items():
        aggregated_error_list = level_res["aggregated_estimation_error_list"]
        error_lists[level] = aggregated_error_list

    return error_lists


def compute_bootstrapped_error_gains(
        mtm_error_list: list,
        baseline_level_wise_error_lists: dict,
):
    # Get error list for direct prompting
    direct_prompting_error_list = baseline_level_wise_error_lists[0]

    # Micro to macro error gain
    mtm_error_gains = bootstrap_error_gains(
        direct_prompting_error_list=direct_prompting_error_list,
        comparison_error_list=mtm_error_list)

    # Baseline
    baseline_error_gains = {}
    for level, level_res in baseline_level_wise_error_lists.items():
        baseline_error_gains[level] = bootstrap_error_gains(
            direct_prompting_error_list=direct_prompting_error_list,
            comparison_error_list=level_res
        )

    return {
        "micro_macro": mtm_error_gains,
        "aggregation_baseline": baseline_error_gains,
    }


def bootstrap_error_gains(
        direct_prompting_error_list: list,
        comparison_error_list: list,
        n_boot=1_000,
        ci=0.90) -> dict:
    # Convert to numpy arrays
    direct_arr = np.asarray(direct_prompting_error_list)
    comparison_arr = np.asarray(comparison_error_list)

    assert len(direct_arr) == len(comparison_arr), f"Direct len: {len(direct_arr)}, comparison len: {len(comparison_arr)}"

    # Initialization
    rng = np.random.default_rng()
    alpha = (1 - ci) / 2
    n = len(direct_prompting_error_list)
    boot_values = []

    # Bootstrap runs
    for _ in range(n_boot):
        # Sampling
        idx = rng.integers(0, n, size=n)
        direct = direct_arr[idx].mean()
        comparison_error = comparison_arr[idx].mean()

        # Compute error gain
        error_gain = 1 - comparison_error / direct
        boot_values.append(error_gain)

    # Compute bootstrap results
    results = {
        "mean": np.mean(boot_values),
        "lower": np.quantile(boot_values, alpha),
        "upper": np.quantile(boot_values, (1 - alpha))
    }

    return results


def load_all_results(
        experiment_log_file: str,
        load_joint_llm_priors: bool,
) -> tuple[dict, dict]:
    # Load experiment logs
    with open(experiment_log_file, "r") as f:
        logs = yaml.safe_load(f)

    baseline_logs = logs["experiments_log"]["baseline"]
    micro_macro_logs = logs["experiments_log"]["micro_macro"]
    print(baseline_logs)

    # Macro to micro
    mtm_results = read_json(
        experiment_name=f"{micro_macro_logs['experiment_name']}/{micro_macro_logs['timestamp']}",
        filename="summary_results.json",
        cluster=micro_macro_logs["cluster"],
    )

    baseline_results = read_baseline_to_json(
        experiment_name=baseline_logs["experiment_name"],
        datetime=baseline_logs["timestamp"],
        cluster=baseline_logs["cluster"],
        load_joint_llm_priors=load_joint_llm_priors,
    )

    # Assertions
    assert mtm_results["model"].split("/")[1] == baseline_results["metadata"]["model"], "Model mismatch"
    assert mtm_results["reasoning_effort"] == baseline_results["metadata"]["reasoning_effort"], \
        "Mismatch in reasoning effort"

    return mtm_results, baseline_results


def read_baseline_to_json(experiment_name: str, datetime: str, cluster: bool, load_joint_llm_priors: bool) -> dict:
    # Convert shards to JSON tree
    tree, metadata, _ = convert_baseline_shards_to_json_tree(
        experiment_name=experiment_name,
        datetime=datetime,
        cluster=cluster,
    )

    # Aggregate results across levels
    aggregated_tree = build_aggregated_distribution_tree(level_tree=tree, load_joint_llm_priors=load_joint_llm_priors)

    # Collect results
    exp_results = {
        "metadata": metadata,
        "aggregated_tree": aggregated_tree,
    }

    return exp_results


if __name__ == "__main__":
    # Experiment
    plot_dir_name = "micro_macro_few_bin_implicit"

    # Experiment logs
    experiment_log_file = "experiment_gpt54.yaml"
    load_joint_llm_priors = True

    # Run main script
    main_error_gain(
        plot_dir_name=plot_dir_name,
        experiment_log_file=experiment_log_file,
        load_joint_llm_priors=load_joint_llm_priors,
    )
