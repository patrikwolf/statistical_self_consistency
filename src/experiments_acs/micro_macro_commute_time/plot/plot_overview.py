import re
import numpy as np
import matplotlib.pyplot as plt

from file_logging.read_and_write_json import read_json
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir, get_results_dir

# Set style
set_plot_style(sans_serif=False, factor=1.225)

# Line width
plt.rcParams.update({
    "axes.linewidth": 2,
    "grid.linewidth": 1.75,
})


def main(
        plot_dir_name: str,
        experiment_name: str,
        timestamp: str,
        cluster: bool,
):
    # Results dir
    results_dir = get_results_dir(
        experiment_name=experiment_name,
        cluster=cluster,
        use_timestamp=True,
        timestamp=timestamp,
        create_dir=False
    )

    # Load shards
    shards = sorted(results_dir.glob("shard_*__summary_results.json"),
                    key=lambda p: int(re.search(r"shard_(\d+)__", p.name).group(1)))

    # Load micro to macro results
    raw_results = []
    for idx, shard in enumerate(shards):
        data = read_json(
            experiment_name=f"{experiment_name}/{timestamp}",
            filename=shard.name,
            cluster=cluster,
        )

        # Add to list
        raw_results.append(data)

    # Process results
    plot_results, metadata = process_results(raw_results=raw_results)

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    title = (r"\huge\textbf{Direct vs. Micro-to-Macro Error for ACS Commute Time}"
             "\n"
             r"\normalsize "
             f"Model: {metadata['model']}, reasoning effort: '{metadata['reasoning_effort']}', "
             f"prompting scheme: '{metadata['prompting_scheme']}', number of samples: {metadata['num_of_samples']}"
             "\n"
             r"\normalsize "
             f"Set NaN in commute time column to zero: {metadata['set_commute_nan_to_zero']}"
             "\n"
             r"\normalsize "
             f"Task: estimate the probability that a randomly selected person's one-way commute time to work "
             f"that is longer than the threshold."
             )

    commute_thresholds = list(plot_results.keys())

    n_cols = len(commute_thresholds)

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(3 + 4 * n_cols, 6),
        sharey="row",
        squeeze=False,
    )

    row_y_max = 0
    for col_idx, threshold in enumerate(commute_thresholds):
        ax = axes[0, col_idx]

        # Extract data
        direct_error = plot_results[threshold]["direct_error"]
        mtm_error = plot_results[threshold]["mtm_error"]
        labels = ["Direct", "MtM"]
        values = [direct_error, mtm_error]
        error_gain = 1 - mtm_error / direct_error
        error_gain_color = "red" if error_gain < 0 else "green"
        x = np.arange(len(labels))

        # Colors
        colors = ["tab:orange", "tab:blue"]

        # Bar plot
        ax.bar(x, values, color=colors)

        # Formatting
        row_y_max = max(row_y_max, max(values))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(rf"\Large Commute threshold: {threshold} mins,"
                     + "\n"
                     + rf"\Large Error gain: ${100 * error_gain:.2f}$ \%", color=error_gain_color)

        if col_idx == 0:
            ax.set_ylabel("Absolute error", fontsize=14)

        # Print values
        for i, value in enumerate(values):
            ax.text(
                i,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=14,
            )

    # Set country-specific y limit
    axes[0, 0].set_ylim(0, row_y_max * 1.15)

    # Formatting
    fig.suptitle(title)
    fig.tight_layout()

    if metadata["set_commute_nan_to_zero"]:
        prefix = "nan_zero"
    else:
        prefix = "working_pop"

    # Save plot
    plt.savefig(plot_dir / f"{prefix}_{timestamp}.png", dpi=300)
    plt.savefig(plot_dir / f"{prefix}_{timestamp}.pdf")

    # Show plot
    plt.show()


def process_results(raw_results: list) -> tuple[dict, dict]:
    plot_results = {}
    metadata = {}
    for idx, res in enumerate(raw_results):
        # Metadata
        if idx == 0:
            metadata = {
                "model": res["model"],
                "reasoning_effort": res["reasoning_effort"],
                "prompting_scheme": res["prompting_scheme"],
                "num_of_samples": res["num_of_samples"],
                "set_commute_nan_to_zero": res["set_commute_nan_to_zero"],
            }

        # Ground-truth probability
        ground_truth = res["ground_truth_probability"]

        # Direct error
        direct_error = abs(ground_truth - res["direct"]["llm_prediction_avg"])

        # Micro to macro error
        mtm_error = abs(ground_truth - res["micro_macro"]["llm_prediction_avg"])

        # Extract results
        plot_results[res["commute_time"]] = {
            "direct_error": direct_error,
            "mtm_error": mtm_error,
        }

    return plot_results, metadata


if __name__ == "__main__":
    # Experiment
    plot_dir_name = "micro_macro_commute_time"

    # Specify experiment results
    experiment_name = "micro_macro_commute_time"
    cluster = True
    timestamp = "2026-06-05__10-13-01"

    # Run main function
    main(
        plot_dir_name=plot_dir_name,
        experiment_name=experiment_name,
        timestamp=timestamp,
        cluster=cluster,
    )
