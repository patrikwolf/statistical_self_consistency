import numpy as np
import matplotlib.pyplot as plt

from file_logging.read_and_write_json import read_json
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


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
    # Load micro to macro results
    data = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="summary_results.json",
        cluster=cluster,
    )

    # Process results
    plot_results = process_results(data=data)

    # Plot results
    plot_results_overview(plot_results=plot_results, data=data, plot_dir_name=plot_dir_name, timestamp=timestamp)


def process_results(data: dict) -> dict:
    plot_results = {}
    for item in data["evaluation_results"]:
        country = item['country']
        print(f"\nCountry: {country}")

        # Ground-truth distribution
        idx_yes = 0
        survey_distribution = item["ground_truth"][idx_yes]

        # Compute error for direct prompting
        direct_avg_llm_prediction = item["direct_prompting"]["avg_llm_prediction"]
        direct_error = abs(survey_distribution - direct_avg_llm_prediction)

        # Compute error for micro-to-macro prompting
        mtm_avg_llm_prediction = item["micro_to_macro"]["avg_llm_prediction"]
        mtm_error = abs(survey_distribution - mtm_avg_llm_prediction)

        # Store results in dict
        if country not in plot_results:
            plot_results[country] = {}

        plot_results[country] = {
            "direct_error": direct_error,
            "mtm_error": mtm_error,
        }

    return plot_results


def plot_results_overview(plot_results: dict, data: dict, plot_dir_name: str, timestamp: str) -> None:
    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    title = (r"\huge\textbf{Direct vs. Micro-to-Macro Error by Country for Thresholded Question}"
             "\n"
             r"\normalsize "
             f"Model: {data['model']}, reasoning effort: '{data['reasoning_effort']}', "
             f"prompting scheme: '{data['prompting_scheme']}'"
             "\n"
             r"\normalsize "
             f"Question identifier: {data['question_identifier']}, question: {data['question']}"
             "\n"
             r"\normalsize "
             f"Answer options: {data['options']}"
             )

    countries = list(plot_results.keys())

    n_cols = len(countries)

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(3 + 4 * n_cols, 6),
        sharey="row",
        squeeze=False,
    )

    row_y_max = 0
    for col_idx, country in enumerate(countries):
        ax = axes[0, col_idx]

        # Extract data
        direct_error = plot_results[country]["direct_error"]
        mtm_error = plot_results[country]["mtm_error"]
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
        ax.set_title(rf"\Large Country: {country}, error gain: ${100 * error_gain:.2f}$ \%", color=error_gain_color)

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

    # Save plot
    plt.savefig(plot_dir / f"{data['question_identifier']}_{timestamp}.png", dpi=300)
    plt.savefig(plot_dir / f"{data['question_identifier']}_{timestamp}.pdf")

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Experiment
    plot_dir_name = "goqa_micro_macro_thresholded"

    # Specify experiment results
    # todo: change
    experiment_name = "goqa_micro_macro_thresholded"
    timestamp = "2026-06-10__11-57-04"
    cluster = True

    # Run main function
    main(
        plot_dir_name=plot_dir_name,
        experiment_name=experiment_name,
        timestamp=timestamp,
        cluster=cluster,
    )
