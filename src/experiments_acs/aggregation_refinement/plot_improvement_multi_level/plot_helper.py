import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D
from utility.plot_style import set_plot_style
from utility.directories import get_plot_dir


def plot_combined_multi_level_error_improvement(
        plot_results: dict,
        variations: list,
        levels: list,
        plot_dir: Path,
        plot_name: str,
        model_name_split: bool = False,
        level_label_x_shift: float = 0.07,
):
    # Title
    title = r"\textbf{Estimation Error Gain (" + levels[0] + " to " + levels[-1] + ")}"

    # Figure
    plt.figure(figsize=(14, 9))
    plt.title(title, pad=10)
    ax = plt.gca()

    # Top to bottom
    y_positions = 3 * np.arange(len(variations))[::-1] + 2
    y_pos_per_model = {}

    # Formatting
    line_width = 3
    marker_size = 6
    pair_gap = 0.28         # distance between GT and LLM within one level
    level_gap = 0.84        # distance between levels
    band_padding = 0.16     # vertical padding around each GT/LLM pair
    band_alpha = 0.7

    # Colors
    band_colors = {
        "level_1": "#E0E0E0",  # light gray
        "level_2": "#C0C0C0",  # medium gray
        "level_3": "#A0A0A0",  # darker gray
    }
    line_colors = {
        "level_1": {
            "pos": "green",
            "neg": "red",
        },
        "level_2": {
            "pos": "green",
            "neg": "red",
        },
        "level_3": {
            "pos": "green",
            "neg": "red",
        }
    }

    # Plot
    for idx, level in enumerate(levels):
        print(f"Level: {level}")
        for y, realization in zip(y_positions, variations):
            # Shift ground-truth y for every label
            y_gt = y - idx * level_gap

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

            # Light background band to visually group GT and LLM for this level
            plt.axhspan(
                ymin=y_llm - band_padding,
                ymax=y_gt + band_padding,
                color=band_colors[f"level_{idx + 1}"],
                alpha=band_alpha,
                zorder=0,
            )

            # Horizontal lines for GT results
            plt.hlines(
                y=y_gt,
                xmin=0,
                xmax=gt_error_gain,
                color=line_colors[f"level_{idx + 1}"]["pos"] if gt_error_gain > 0 else line_colors[f"level_{idx + 1}"]["neg"],
                linewidth=line_width,
                zorder=2,
            )

            # Horizontal lines for GT results
            plt.hlines(
                y=y_llm,
                xmin=0,
                xmax=llm_error_gain,
                color=line_colors[f"level_{idx + 1}"]["pos"] if llm_error_gain > 0 else line_colors[f"level_{idx + 1}"]["neg"],
                linewidth=line_width,
                linestyle="--",
                zorder=2,
            )

            # Endpoint marker for GT
            plt.plot(
                gt_error_gain,
                y_gt,
                marker="o",
                color=line_colors[f"level_{idx + 1}"]["pos"] if gt_error_gain > 0 else line_colors[f"level_{idx + 1}"]["neg"],
                markersize=marker_size,
                zorder=2,
            )

            # Endpoint marker for LLM
            plt.plot(
                llm_error_gain,
                y_llm,
                marker="o",
                color=line_colors[f"level_{idx + 1}"]["pos"] if llm_error_gain > 0 else line_colors[f"level_{idx + 1}"]["neg"],
                markersize=marker_size,
                zorder=2,
            )

            # Add level labels only for the top variation instance
            if realization == variations[0]:
                y_center = 0.5 * (y_gt + y_llm)
                ax.text(
                    level_label_x_shift,
                    y_center,
                    f"Level {idx + 1}",
                    # color=band_colors[f"level_{idx + 1}"],
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    clip_on=False,
                )

    # Find ticks position
    y_ticks = []
    for realization in variations:
        min_pos = min(y_pos_per_model[realization]["y_pos"])
        max_pos = max(y_pos_per_model[realization]["y_pos"])
        y_ticks.append(0.5 * (min_pos + max_pos))

    # Ticks, labels, limits
    plt.xlabel("Error gain over level 0", fontsize=16, labelpad=14)
    plt.tick_params(axis="y", pad=16)
    if model_name_split:
        plt.yticks(
            ticks=y_ticks,
            labels=[v.split("/")[1] for v in variations],
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
    plt.legend(
        handles=legend_elements,
        loc="upper right",
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
    levels = ["Level 1", "Level 2", "Level 3"]

    # Plot
    error_plt = plot_combined_multi_level_error_improvement(
        plot_results=plot_results,
        variations=models,
        levels=levels,
        plot_dir=plot_dir,
        plot_name="test",
        model_name_split=True,
    )
    error_plt.show()
