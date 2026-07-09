import matplotlib.pyplot as plt

from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir
from experiments_acs.micro_macro_few_bins.plot.plot_error_gain import load_all_results, compute_mtm_error_list, \
    compute_baseline_error_list
from utility.bootstrap import bootstrap_confidence_interval

# Set style
set_plot_style(sans_serif=False, factor=1.225)

# Line width
plt.rcParams.update({
    "axes.linewidth": 2,
    "grid.linewidth": 1.75,
})


def main_abs_error(
        plot_dir_name: str,
        experiment_log_file: str,
        load_joint_llm_priors: bool,
):
    # Load results
    mtm_results, baseline_results = load_all_results(
        experiment_log_file=experiment_log_file,
        load_joint_llm_priors=load_joint_llm_priors,
    )

    # Process micro to macro results
    mtm_error_list = compute_mtm_error_list(mtm_result=mtm_results)
    mean, lower, upper = bootstrap_confidence_interval(values=mtm_error_list, n_boot=1_000, ci=0.9)

    # Collect results
    micro_macro_results = {
        "mean_error": mean,
        "lower_error": lower,
        "upper_error": upper,
    }

    # Process baseline results
    baseline_level_wise_error_lists = compute_baseline_error_list(baseline_result=baseline_results)
    aggregation_results = {
        "levels": [],
        "y_mean": [],
        "y_lower": [],
        "y_upper": [],
    }
    for level, level_error_list in baseline_level_wise_error_lists.items():
        mean, lower, upper = bootstrap_confidence_interval(values=level_error_list, n_boot=1_000, ci=0.9)

        # Add to lists
        aggregation_results["levels"].append(level)
        aggregation_results["y_mean"].append(mean)
        aggregation_results["y_lower"].append(lower)
        aggregation_results["y_upper"].append(upper)

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Title
    title = (r"\huge\textbf{Micro to Macro Reasoning: Absolute Error}"
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

    ##############################################################################
    # Plot baseline results
    ##############################################################################

    # Extract plot points
    x = aggregation_results["levels"]
    y_mean = aggregation_results["y_mean"]
    y_lower = aggregation_results["y_lower"]
    y_upper = aggregation_results["y_upper"]

    # Plot
    line, = plt.plot(
        x, y_mean,
        "--",
        marker="o",
        alpha=0.8,
        lw=1.4,
        markersize=5,
        zorder=1,
        label="Manual aggregation",
    )

    # Extract color
    color = line.get_color()

    # Bootstrap CI for aggregation results
    plt.fill_between(x, y_lower, y_upper, color=color, alpha=0.2, zorder=0)

    ##############################################################################
    # Plot for implicit reasoning results
    ##############################################################################

    # Plot mean error
    line = plt.axhline(
        y=micro_macro_results["mean_error"],
        linestyle="-",
        linewidth=1.4,
        alpha=0.8,
        zorder=2,
        color="tab:orange",
        label="Micro-to-macro"
    )

    # Extract color
    color = line.get_color()

    # Horizontal bootstrap CI band
    plt.fill_between(
        x,
        micro_macro_results["lower_error"],
        micro_macro_results["upper_error"],
        color=color,
        alpha=0.15,
        zorder=0,
    )

    # Formatting
    plt.title(title)
    plt.xlabel("Level", labelpad=10)
    plt.ylabel("Error", labelpad=15)
    plt.legend(loc="upper center",
               bbox_to_anchor=(0.5, 1.065),
               ncol=5,
               prop={"size": 10},
               columnspacing=0.8,
               handletextpad=0.4,
               )
    plt.grid()
    plt.tight_layout()

    # Save plot
    model_name = mtm_results['model'].split("/")[1].replace(".", "-")
    plt.savefig(plot_dir / f"absolute_error_few_bins_{model_name}.png", dpi=300)
    plt.savefig(plot_dir / f"absolute_error_few_bins_{model_name}.pdf")

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Experiment
    plot_dir_name = "micro_macro_few_bin_implicit"

    # Experiment logs
    experiment_log_file = "experiment_gpt54.yaml"
    load_joint_llm_priors = True

    # Run main script
    main_abs_error(
        plot_dir_name=plot_dir_name,
        experiment_log_file=experiment_log_file,
        load_joint_llm_priors=load_joint_llm_priors,
    )
