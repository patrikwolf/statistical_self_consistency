import yaml
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from experiments_acs.aggregation_refinement.plot.plot_refinement_splits import process_logs
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


def plot_aggregation_refinement(results: dict, plot_dir: Path, load_joint_llm_priors: bool):
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
             f"Income threshold (USD): {results['income_threshold']}, Income above or below threshold: "
             f"{results['income_greater_than_threshold']}"
             "\n"
             r"\normalsize "
             f"Number of repetitions per setting: {results['num_of_samples']}"
             "\n")

    # Colormap
    entries = results["refinement_results"]

    # Plot all results in the list
    plot_results = {}
    for i, entry in enumerate(entries):
        # Collect data from dict
        level_0_error = entry["refinement_results"][0]["aggregated_error"]
        level_2_error = entry["refinement_results"][2]["aggregated_error"]

        error_gain = (level_0_error - level_2_error) / level_0_error

        # Add to dict
        plot_results[entry["decomposition"]] = {
            "decomposition": entry["decomposition"],
            "error_gain": error_gain,
        }

    # Plot size
    plt.figure(figsize=(9, 9))

    # Convert results to list
    decompositions = []
    error_gains = []
    for decomposition, item in plot_results.items():
        decompositions.append(decomposition)
        error_gains.append(item["error_gain"])

    # Plot
    x = np.arange(len(decompositions))
    error_gains_percent = [100 * e for e in error_gains]
    plt.bar(x, error_gains_percent)

    # Formatting
    plt.title(title)
    plt.xlabel("Decomposition", labelpad=10)
    plt.ylabel("Error gain of aggregated estimate from level 0 to level 2 " + r"[\%]", labelpad=10)
    plt.xticks(x, decompositions, rotation=90)
    plt.grid()
    plt.tight_layout()

    # Add prior to plot name
    prior_usage = "llm_prior" if load_joint_llm_priors else "gt_prior"

    # Save plot
    plot_subdir = plot_dir / prior_usage
    plot_subdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_subdir / "splits_aggregation_improvement.png", dpi=300)
    plt.savefig(plot_subdir / "splits_aggregation_improvement.pdf")

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "aggregation_improvement"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    set_plot_style(sans_serif=False, factor=1.15)

    # Load experiment logs
    with open("../plot/experiments_refinement_splits.yaml", "r") as f:
        logs = yaml.safe_load(f)

    # Toggle to use new jointly elicited priors
    load_joint_llm_priors = False

    # Process results
    results, experiment_name = process_logs(logs=logs,
                                            overwrite_cluster_to_false=False,
                                            load_joint_llm_priors=load_joint_llm_priors)

    # Plot
    plot_aggregation_refinement(results=results, plot_dir=plot_dir, load_joint_llm_priors=load_joint_llm_priors)
