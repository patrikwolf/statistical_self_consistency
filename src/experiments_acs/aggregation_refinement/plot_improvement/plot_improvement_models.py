import yaml
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from experiments_acs.aggregation_refinement.plot.plot_refinement_models import process_logs
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


def plot_aggregation_refinement(
        results: dict,
        plot_dir: Path,
        load_joint_llm_priors: bool,
        income_threshold: float | int,
):
    model_effort_list = [f"{model} ({effort})" for model, effort in zip(results["models"], results["reasoning_efforts"])]
    chunks = [", ".join(model_effort_list[i:i + 3]) for i in range(0, len(model_effort_list), 3)]
    model_effort_text = ("\n" + r"\normalsize ").join(chunks)

    # Title
    title = (r"\textbf{Estimation Error Improvement from Level 0 to Level 2}"
             "\n"
             r"\normalsize "
             f"Models (reasoning effort): {model_effort_text[:80]} ..."
             "\n"
             r"\normalsize "
             f"Prompting scheme: '{results['prompting_scheme']}'"
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

    # Get results
    entries = results["refinement_results"]

    # Plot all results in the list
    plot_results = {}
    for i, entry in enumerate(entries):
        # Collect data from dict
        level_0_error = entry["refinement_results"][0]["aggregated_error"]
        level_2_error = entry["refinement_results"][2]["aggregated_error"]

        error_gain = (level_0_error - level_2_error) / level_0_error

        # Add to dict
        plot_results[entry["model"]] = {
            "model": entry["model"],
            "error_gain": error_gain,
        }

    # Plot size
    plt.figure(figsize=(9, 9))

    # Convert results to list
    models = []
    models_short = []
    error_gains = []
    for model, item in plot_results.items():
        models.append(model)
        models_short.append(model.split("/")[1])
        error_gains.append(item["error_gain"])

    # Plot
    x = np.arange(len(models))
    error_gains_percent = [100 * e for e in error_gains]
    plt.bar(x, error_gains_percent)

    # Formatting
    plt.title(title)
    plt.xlabel("Model name", labelpad=10)
    plt.ylabel("Error gain of aggregated estimate from level 0 to level 2 " + r"[\%]", labelpad=10)
    plt.xticks(x, models_short, rotation=90)
    plt.grid()
    plt.tight_layout()

    # Add prior to plot name
    prior_usage = "llm_prior" if load_joint_llm_priors else "gt_prior"

    # Save plot
    plot_subdir = plot_dir / prior_usage
    plot_subdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_subdir / f"models_aggregation_improvement_{income_threshold}.png", dpi=300)
    plt.savefig(plot_subdir / f"models_aggregation_improvement_{income_threshold}.pdf")

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "aggregation_improvement"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    set_plot_style(sans_serif=False, factor=1.15)

    # todo: change threshold
    income_threshold = 40_000

    # Log name
    if income_threshold == 10_000:
        log_filename = "experiments_refinement_models_10k.yaml"
    elif income_threshold == 40_000:
        log_filename = "experiments_refinement_models_40k.yaml"
    else:
        raise ValueError(f"Invalid income threshold: {income_threshold}")

    # Load experiment logs
    with open(f"../plot/{log_filename}", "r") as f:
        logs = yaml.safe_load(f)

    # Toggle to use new jointly elicited priors
    load_joint_llm_priors = False

    # Process results
    results, experiment_name = process_logs(logs=logs,
                                            overwrite_cluster_to_false=False,
                                            load_joint_llm_priors=load_joint_llm_priors)

    # Plot
    plot_aggregation_refinement(
        results=results,
        plot_dir=plot_dir,
        load_joint_llm_priors=load_joint_llm_priors,
        income_threshold=income_threshold,
    )
