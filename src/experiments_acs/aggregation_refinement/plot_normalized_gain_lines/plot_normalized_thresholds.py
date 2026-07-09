import yaml
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path
from design_elements.color_palette import get_histogram_colors
from design_elements.set_of_markers import get_markers
from experiments_acs.aggregation_refinement.plot.plot_refinement_thresholds import process_logs
from experiments_acs.aggregation_refinement.plot_normalized_gain_lines.normalized_bootstrap import bootstrap_normalized_errors
from file_logging.read_and_write_json import save_as_json
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


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
    markers, marker_sizes = get_markers(num_markers=len(entries))

    # Store values per level for averaging
    values_by_level = defaultdict(list)

    # Plot all results in the list
    all_x = set()
    for i, entry in enumerate(entries):
        # Compute bootstrap intervals for relative gain
        x, y_mean, y_lower, y_upper = bootstrap_normalized_errors(refinement_results=entry["refinement_results"])
        all_x.update([int(level) for level in x])

        # Save in dict
        for level, mean in zip(x, y_mean):
            values_by_level[level].append(mean)

        # Plot
        color = colors[i]
        plt.plot(
            x, y_mean,
            "--",
            marker=markers[i],
            color=color,
            alpha=0.8,
            lw=1.4,
            markersize=marker_sizes[i],
            markeredgewidth=0.1,
            zorder=3,
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
    plt.ylabel("Normalized alignment error", labelpad=15)
    x_ticks = list(range(min(all_x), max(all_x) + 1))
    plt.xticks(x_ticks)
    plt.legend(loc="upper center",
               bbox_to_anchor=(0.5, 1.135),
               ncol=4,
               prop={"size": 10},
               columnspacing=0.8,
               handletextpad=0.4,
               )
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
    plot_dir = plot_base_dir / "aggregation_relative_gain"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    set_plot_style(sans_serif=False, factor=1.225)

    # Line width
    plt.rcParams.update({
        "axes.linewidth": 2,
        "grid.linewidth": 1.75,
    })

    # Load experiment logs
    with open("./../plot/experiments_refinement_threshold_gt_priors.yaml", "r") as f:
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
        experiment="figure_aggregation_relative_gain",
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
