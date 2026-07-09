import numpy as np
import matplotlib.pyplot as plt

from file_logging.read_and_write_json import read_json
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir
from utility.wasserstein_helper import compute_wasserstein_distance


# Set style
set_plot_style(sans_serif=False, factor=1.225)

# Line width
plt.rcParams.update({
    "axes.linewidth": 2,
    "grid.linewidth": 1.75,
})


def main(
        experiment_name: str,
        timestamp: str,
        cluster: bool,
        plot_dir_name: str,
):
    # Load micro to macro results
    data = read_json(
        experiment_name=f"{experiment_name}/{timestamp}",
        filename="summary_results.json",
        cluster=cluster,
    )

    # Process results
    plot_results = process_results(data=data)

    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    title = (r"\huge\textbf{Direct vs. Micro-to-Macro Error by Country and Question}"
             "\n"
             r"\normalsize "
             f"Model: {data['model']}, reasoning effort: '{data['reasoning_effort']}', "
             f"prompting scheme: '{data['prompting_scheme']}'"
             )

    countries = list(plot_results.keys())
    question_ids = sorted(
        {qid for country_results in plot_results.values() for qid in country_results.keys()},
        key=lambda qid: int(qid.removeprefix("Q")),
    )

    n_rows = len(countries)
    n_cols = len(question_ids)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3 + 4 * n_rows),
        sharey="row",
        squeeze=False,
    )

    for row_idx, country in enumerate(countries):
        country_y_max = 0
        for col_idx, qid in enumerate(question_ids):
            ax = axes[row_idx, col_idx]

            # Hide missing results
            if qid not in plot_results[country]:
                ax.axis("off")
                continue

            # Extract data
            direct_error = plot_results[country][qid]["direct_error"]
            mtm_error = plot_results[country][qid]["mtm_error"]
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
            country_y_max = max(country_y_max, max(values))
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title(rf"\Large Question {qid}, error gain: ${100 * error_gain:.2f}$ \%", color=error_gain_color)

            if col_idx == 0:
                ax.set_ylabel("Wasserstein Distance", fontsize=14)

                # Separate country label
                ax.annotate(
                    rf"\bf {country}",
                    xy=(-0.2, 0.5),
                    xycoords="axes fraction",
                    va="center",
                    ha="center",
                    rotation=90,
                    fontsize=16,
                )

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

        print(f"Country: {country}, max error: {country_y_max:.3f}")
        # Set country-specific y limit
        axes[row_idx, 0].set_ylim(0, country_y_max * 1.15)

    # Formatting
    fig.suptitle(title)
    fig.tight_layout()

    # Save plot
    plt.savefig(plot_dir / f"{timestamp}.png", dpi=300)
    plt.savefig(plot_dir / f"{timestamp}.pdf")

    # Show plot
    plt.show()


def process_results(data: dict) -> dict:
    plot_results = {}
    for item in data["evaluation_results"]:
        print(f"\nQuestion: {item['question']}")
        question_identifier = item["question_identifier"]

        # Extract information
        distributions = item["distributions"]

        # Iterate over countries
        for country_res in distributions:
            country = country_res["country"]
            print(f"  ---> {country}")

            # Ground-truth distribution
            survey_distribution = country_res["ground_truth"]

            # Compute error for direct prompting
            direct_avg_llm_distribution = extract_distribution_list(
                distribution_dict=country_res["direct_prompting"]["avg_llm_distribution"])
            direct_error = compute_wasserstein_distance(
                d1=survey_distribution,
                d2=direct_avg_llm_distribution
            )

            # Compute error for micro-to-macro prompting
            mtm_avg_llm_distribution = extract_distribution_list(country_res["micro_to_macro"]["avg_llm_distribution"])
            mtm_error = compute_wasserstein_distance(
                d1=survey_distribution,
                d2=mtm_avg_llm_distribution
            )

            # Store results in dict
            if country not in plot_results:
                plot_results[country] = {}

            plot_results[country][question_identifier] = {
                "direct_error": direct_error,
                "mtm_error": mtm_error,
            }

    return plot_results


def extract_distribution_list(distribution_dict):
    distribution_list = []
    for k in range(len(distribution_dict)):
        distribution_list.append(distribution_dict[f"Answer option {k + 1}"])
    return distribution_list


if __name__ == "__main__":
    # Experiment
    plot_dir_name = "goqa_micro_macro_implicit"

    # Specify experiment results
    # todo: change
    experiment_name = "micro_macro_global_opinion_qa"
    timestamp = "2026-06-03__14-22-06"
    cluster = True

    # Plot results
    main(
        experiment_name=experiment_name,
        timestamp=timestamp,
        cluster=cluster,
        plot_dir_name=plot_dir_name,
    )
