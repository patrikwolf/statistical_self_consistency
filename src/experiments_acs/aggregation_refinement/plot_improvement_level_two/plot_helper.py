import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


def plot_combined_level_two_error_improvement(
        plot_results: dict,
        variations: list,
        plot_dir: Path,
        plot_name: str,
        label_names: list = None,
        loc: str = None
):
    # Title
    title = r"\textbf{Error Gain @ Level 2}"

    # Figure
    plt.figure(figsize=(8, 6))
    plt.title(title, pad=15)

    # Top to bottom
    y_positions = np.arange(len(variations))[::-1]
    y_pos_per_model = {}

    # Formatting
    line_width = 3
    marker_size = 7.5
    pair_gap = 0.28         # distance between GT and LLM within one level

    # Plot
    level = "Level 2"
    print(f"Level: {level}")
    for y, realization in zip(y_positions, variations):
        # Ground-truth y
        y_gt = y

        # Shifted y for LLM priors
        y_llm = y_gt - pair_gap

        # Add y_positions to dict
        if realization in y_pos_per_model:
            y_pos_per_model[realization]["y_pos"].extend([y_gt, y_llm])
        else:
            y_pos_per_model[realization] = {"y_pos": [y_gt, y_llm]}

        # Extract error gains
        gt_error_gain = plot_results["gt_prior"][realization][level]
        llm_error_gain = plot_results["llm_prior"][realization][level]

        # Horizontal lines for GT results
        plt.hlines(
            y=y_gt,
            xmin=0,
            xmax=gt_error_gain,
            color="green" if gt_error_gain > 0 else "red",
            linewidth=line_width,
            zorder=2,
        )

        # Horizontal lines for GT results
        plt.hlines(
            y=y_llm,
            xmin=0,
            xmax=llm_error_gain,
            color="green" if llm_error_gain > 0 else "red",
            linewidth=line_width,
            linestyle="--",
            zorder=2,
        )

        # Endpoint marker for GT
        plt.plot(
            gt_error_gain,
            y_gt,
            marker="o",
            color="green" if gt_error_gain > 0 else "red",
            markersize=marker_size,
            zorder=2,
        )

        # Endpoint marker for LLM
        plt.plot(
            llm_error_gain,
            y_llm,
            marker="o",
            color="green" if llm_error_gain > 0 else "red",
            markersize=marker_size,
            zorder=2,
        )

    # Find ticks position
    y_ticks = []
    for realization in variations:
        min_pos = min(y_pos_per_model[realization]["y_pos"])
        max_pos = max(y_pos_per_model[realization]["y_pos"])
        y_ticks.append(0.5 * (min_pos + max_pos))

    # Ticks, labels, limits
    plt.xlabel(r"$\mathrm{SpecificityGain}(2)$", labelpad=10)
    plt.tick_params(axis="y", pad=8)
    if label_names:
        plt.yticks(
            ticks=y_ticks,
            labels=label_names,
        )
    else:
        plt.yticks(
            ticks=y_ticks,
            labels=variations,
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], color="black", lw=line_width, linestyle="-", label="GT prior"),
        Line2D([0], [0], color="black", lw=line_width, linestyle="--", label="LLM prior"),
    ]
    if loc:
        plt.legend(
            handles=legend_elements,
            loc=loc,
            frameon=True,
        )
    else:
        plt.legend(
            handles=legend_elements,
            loc=loc,
            frameon=True,
        )

    # Formatting
    plt.axvline(0, color="gray", linewidth=1.0)
    plt.grid(axis="x")
    plt.tight_layout()

    # Save plot
    plt.savefig(plot_dir / f"{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(plot_dir / f"{plot_name}.pdf")

    return plt


if __name__ == "__main__":
    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "aggregation_improvement"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    set_plot_style(sans_serif=False)

    # Dummy data
    plot_results = {
        "gt_prior": {
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
        },
        "llm_prior": {
            "openai/gpt-5.4": {
                "Level 1": 0.2,
                "Level 2": 0.2,
                "Level 3": 0.25,
            },
            "google/gemini-3.1-pro-preview": {
                "Level 1": 0.34,
                "Level 2": -0.1,
                "Level 3": 0.3,
            },
            "x-ai/grok-4.1-fast": {
                "Level 1": 0.2,
                "Level 2": -0.1,
                "Level 3": -0.5,
            },
        }
    }
    models = ["openai/gpt-5.4", "google/gemini-3.1-pro-preview", "x-ai/grok-4.1-fast"]
    model_names = ["GPT-5.4", "Gemini 3.1 Pro", "Grok 4.20"]
    levels = ["Level 1", "Level 2", "Level 3"]

    # Plot
    error_plt = plot_combined_level_two_error_improvement(
        plot_results=plot_results,
        variations=models,
        plot_dir=plot_dir,
        plot_name="test",
        label_names=model_names,
    )
    error_plt.show()
