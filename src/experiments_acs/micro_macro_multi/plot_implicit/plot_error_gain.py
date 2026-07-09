import numpy as np
import matplotlib.pyplot as plt

from experiments_acs.micro_macro_multi.plot_implicit.plot_helper import load_all_results
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


# Set style
set_plot_style(sans_serif=False, factor=1.225)

# Line width
plt.rcParams.update({
    "axes.linewidth": 2,
    "grid.linewidth": 1.75,
})


def main_error_gain(
        mtm_experiment_name: str,
        mtm_timestamp: str,
        mtm_cluster: bool,
        baseline_experiment_log_file: str,
        baseline_experiment_name: str,
        load_joint_llm_priors: bool,
        plot_dir_name: str,
        income_thresholds: list[int],
):
    # Load results
    mtm_results, baseline_results = load_all_results(
        mtm_experiment_name=mtm_experiment_name,
        mtm_timestamp=mtm_timestamp,
        mtm_cluster=mtm_cluster,
        baseline_experiment_log_file=baseline_experiment_log_file,
        baseline_experiment_name=baseline_experiment_name,
        load_joint_llm_priors=load_joint_llm_priors,
    )

    # Process and combine results
    combined_results = {}
    for mtm_res in mtm_results:
        # Extract income threshold
        income_threshold = mtm_res["income_threshold"]
        print(f"Preparing results for income Threshold: {income_threshold}")

        # Fine baseline results for same threshold
        baseline_res = next(el for el in baseline_results if el["metadata"]["income_threshold"] == income_threshold)

        # Micro to macro error list
        mtm_error_list = compute_mtm_error_list(mtm_result=mtm_res)
        baseline_level_wise_error_lists = compute_baseline_error_list(baseline_result=baseline_res)

        # Compute error gains
        bootstrapped_error_gains = compute_bootstrapped_error_gains(
            mtm_error_list=mtm_error_list,
            baseline_level_wise_error_lists=baseline_level_wise_error_lists,
        )

        # Save results
        combined_results[income_threshold] = bootstrapped_error_gains

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Title
    title = (r"\huge\textbf{Micro to Macro Reasoning: Error Gains}"
             "\n"
             r"\normalsize "
             f"Model: {mtm_results[0]['model']}, reasoning effort: '{mtm_results[0]['reasoning_effort']}', "
             f"prompting scheme: '{mtm_results[0]['prompting_scheme']}'"
             "\n"
             r"\normalsize "
             f"Dashed line: manual aggregation at different tree levels. Solid line: implicit micro-to-macro prompting."
             "\n"
             r"\normalsize "
             f"Task: estimate the probability that an randomly selected person in the U.S. has an income "
             f"above a predefined threshold"
             "\n"
             r"\normalsize "
             f"Loaded and used jointly elicited LLM priors: {load_joint_llm_priors}"
             "\n"
             )

    # Select thresholds
    selected_thresholds = sorted([int(tau) for tau in combined_results.keys() if int(tau) in income_thresholds])
    num_plots = len(selected_thresholds)

    # Plot size
    fig, axs = plt.subplots(1, num_plots, figsize=(7 * num_plots, 8))
    fig.suptitle(title)

    # Plot results
    plot_idx = 0
    for key, res in combined_results.items():
        if int(key) not in income_thresholds:
            print(f"Skipping threshold {key} USD")
            continue

        # Collect results
        labels = ["Micro to macro"]
        entries = [res["micro_macro"]]
        for level, entry in res["aggregation_baseline"].items():
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
        axs[plot_idx].bar(
            x,
            means,
            color=colors,
            yerr=yerr,
            capsize=5,
            alpha=0.8,
        )

        # Formatting
        axs[plot_idx].axhline(0, color="black", linewidth=0.8)
        axs[plot_idx].set_xticks(x)
        axs[plot_idx].set_xticklabels(labels, rotation=30, ha="right")
        axs[plot_idx].set_ylabel("Error gain")
        axs[plot_idx].set_title(f"Income threshold: {key} USD")

        plot_idx += 1

    # Formatting
    fig.tight_layout()

    # Save plot
    plt.savefig(plot_dir / f"error_gains_{mtm_timestamp}.png", dpi=300)
    plt.savefig(plot_dir / f"error_gains_{mtm_timestamp}.pdf")

    # Show plot
    plt.show()


def compute_mtm_error_list(mtm_result: dict):
    # Ground-truth target
    gt_probability = mtm_result["ground_truth_probability"]

    # No root level estimate; only deeper levels
    prediction_list = mtm_result["micro_macro"]["llm_prediction_list"]
    error_list = abs(gt_probability - np.array(prediction_list))

    return error_list.tolist()


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

    assert len(direct_arr) == len(comparison_arr)

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


if __name__ == "__main__":
    # Experiment
    plot_dir_name = "micro_macro_implicit"

    # Load micro to macro results
    # todo: change (find timestamps in exp_micro_macro.yaml)
    mtm_timestamp = "2026-06-01__09-00-00"
    mtm_experiment_name = "micro_macro_implicit"
    mtm_cluster = True

    # Baseline
    # todo: change
    baseline_experiment_log_file = "exp_baseline_gpt54.yaml"
    baseline_experiment_name = "micro_macro_aggregation_baseline"
    load_joint_llm_priors = True

    # Income thresholds
    income_thresholds = [100, 20_000, 40_000, 60_000, 80_000]
    # income_thresholds = [100, 1_000, 10_000, 20_000, 40_000, 50_000, 60_000, 80_000]

    # Plot results
    main_error_gain(
        mtm_experiment_name=mtm_experiment_name,
        mtm_timestamp=mtm_timestamp,
        mtm_cluster=mtm_cluster,
        baseline_experiment_log_file=baseline_experiment_log_file,
        baseline_experiment_name=baseline_experiment_name,
        load_joint_llm_priors=load_joint_llm_priors,
        plot_dir_name=plot_dir_name,
        income_thresholds=income_thresholds,
    )
