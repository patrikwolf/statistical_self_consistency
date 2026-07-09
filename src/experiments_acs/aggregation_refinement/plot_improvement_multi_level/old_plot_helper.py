import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


def plot_multi_level_error_improvement(
        plot_results: dict,
        variations: list,
        levels: list,
        plot_dir: Path,
        plot_name: str,
        load_joint_llm_priors: bool,
        model_name_split: bool = False,
):
    # Title
    title = (r"\textbf{Estimation Error Improvement from " + levels[0] + " to " + levels[-1] + "}"
             "\n"
             r"\normalsize "
             f"Loaded and used jointly elicited LLM priors: {load_joint_llm_priors}")

    # Figure
    fig, axs = plt.subplots(1, 3, figsize=(11, 6), sharey=True, sharex=True)
    fig.suptitle(title)

    # Top to bottom
    y_positions = np.arange(len(variations))[::-1]

    for ax, level in zip(axs, levels):
        for y, realization in zip(y_positions, variations):
            error_gain = plot_results[realization][level]

            # Conditional color
            color = "green" if error_gain > 0 else "red"

            # Horizontal line from zero improvement to the observed gain
            ax.hlines(
                y=y,
                xmin=0,
                xmax=error_gain,
                color=color,
                linewidth=4.2,
            )

            # Optional: keep a small endpoint marker
            ax.plot(
                error_gain,
                y,
                marker="o",
                color=color,
                markersize=11,
            )

        ax.axvline(0, color="gray", linewidth=1.0)
        ax.set_title(r"\bfseries " + level, fontsize=17)

        # Keep only the bottom axis
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)

    # Axes formatting
    axs[0].set_yticks(y_positions)
    if model_name_split:
        axs[0].set_yticklabels([v.split("/")[1] for v in variations])
    else:
        axs[0].set_yticklabels([v for v in variations])
    axs[0].set_ylim(y_positions.min() - 0.6, y_positions.max() + 0.6)

    # Figure formatting
    fig.supxlabel("Error gain over level 0", fontsize=17)
    fig.tight_layout()

    # Add prior to plot name
    prior_usage = "llm_prior" if load_joint_llm_priors else "gt_prior"

    # Save plot
    plot_subdir = plot_dir / prior_usage
    plot_subdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_subdir / f"{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(plot_subdir / f"{plot_name}.pdf")

    return plt


if __name__ == "__main__":
    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "aggregation_improvement"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    set_plot_style(sans_serif=True)

    # Dummy data
    plot_results = {
        "openai/gpt-5.4": {
            "Level 1": 0.1,
            "Level 2": 0.3,
            "Level 3": 0.4,
        },
        "google/gemini-3.1-pro-preview": {
            "Level 1": 0.5,
            "Level 2": -0.5,
            "Level 3": 0.1,
        },
        "x-ai/grok-4.1-fast": {
            "Level 1": 0.1,
            "Level 2": -0.3,
            "Level 3": -0.4,
        },
    }
    models = ["openai/gpt-5.4", "google/gemini-3.1-pro-preview", "x-ai/grok-4.1-fast"]
    levels = ["Level 1", "Level 2", "Level 3"]

    # Toggle to use new jointly elicited priors
    load_joint_llm_priors = False

    # Plot
    error_plt = plot_multi_level_error_improvement(
        plot_results=plot_results,
        variations=models,
        levels=levels,
        plot_dir=plot_dir,
        plot_name="test",
        load_joint_llm_priors=load_joint_llm_priors,
        model_name_split=True,
    )
    error_plt.show()
