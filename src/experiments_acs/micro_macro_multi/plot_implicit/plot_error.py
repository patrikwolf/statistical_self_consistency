import matplotlib.pyplot as plt

from experiments_acs.micro_macro_multi.plot_implicit.plot_helper import load_all_results, process_and_combine_results
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


# Set style
set_plot_style(sans_serif=False, factor=1.225)

# Line width
plt.rcParams.update({
    "axes.linewidth": 2,
    "grid.linewidth": 1.75,
})


def main_abs_error(
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

    # Process and combine
    combined_results = process_and_combine_results(
        mtm_results=mtm_results,
        baseline_results=baseline_results,
    )

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Title
    title = (r"\huge\textbf{Micro to Macro Reasoning: Population-Level Error}"
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
             "\n\n"
             )

    # Plot size
    plt.figure(figsize=(9, 9))
    all_x = set()

    # Plot results
    for key, res in combined_results.items():
        if int(key) not in income_thresholds:
            print(f"Skipping threshold {key} USD")
            continue

        ##############################################################################
        # Plot baseline results
        ##############################################################################

        # Extract plot points
        agg_res = res["aggregation_baseline"]
        x = agg_res["levels"]
        y_mean = agg_res["y_mean"]
        y_lower = agg_res["y_lower"]
        y_upper = agg_res["y_upper"]

        # Plot
        line, = plt.plot(
            x, y_mean,
            "--",
            marker="o",
            alpha=0.8,
            lw=1.4,
            markersize=5,
            zorder=1
        )

        # Extract color
        color = line.get_color()

        # Bootstrap CI for aggregation results
        plt.fill_between(x, y_lower, y_upper, color=color, alpha=0.2, zorder=0)

        all_x.update(x)

        ##############################################################################
        # Plot for implicit reasoning results
        ##############################################################################

        # Extract results
        mtm_res = res["micro_macro"]
        threshold = mtm_res["income_threshold"]

        plt.axhline(
            y=mtm_res["mean_error"],
            color=color,
            linestyle="-",
            linewidth=1.4,
            alpha=0.8,
            label=f"{threshold} USD",
            zorder=2,
        )

        # Horizontal bootstrap CI band
        plt.fill_between(
            x,
            mtm_res["lower_error"],
            mtm_res["upper_error"],
            color=color,
            alpha=0.15,
            zorder=0,
        )

    # Formatting
    plt.title(title)
    plt.xlabel("Level", labelpad=10)
    plt.ylabel("Error", labelpad=15)
    x_ticks = list(range(min(all_x), max(all_x) + 1))
    plt.xticks(x_ticks)
    plt.legend(loc="upper center",
               bbox_to_anchor=(0.5, 1.1),
               ncol=5,
               prop={"size": 10},
               columnspacing=0.8,
               handletextpad=0.4,
               )
    plt.grid()
    plt.tight_layout()

    # Save plot
    plt.savefig(plot_dir / f"{mtm_timestamp}.png", dpi=300)
    plt.savefig(plot_dir / f"{mtm_timestamp}.pdf")

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Experiment
    plot_dir_name = "micro_macro_implicit"

    # Load micro to macro results
    # todo: change
    mtm_experiment_name = "micro_macro_implicit"
    mtm_timestamp = "2026-06-01__08-30-38"
    mtm_cluster = True

    # Baseline
    baseline_experiment_log_file = "exp_baseline_gpt54.yaml"
    baseline_experiment_name = "micro_macro_aggregation_baseline"
    load_joint_llm_priors = True

    # Income thresholds
    income_thresholds = [100, 20_000, 40_000, 60_000, 80_000]

    # Plot results
    main_abs_error(
        mtm_experiment_name=mtm_experiment_name,
        mtm_timestamp=mtm_timestamp,
        mtm_cluster=mtm_cluster,
        baseline_experiment_log_file=baseline_experiment_log_file,
        baseline_experiment_name=baseline_experiment_name,
        load_joint_llm_priors=load_joint_llm_priors,
        plot_dir_name=plot_dir_name,
        income_thresholds=income_thresholds,
    )
