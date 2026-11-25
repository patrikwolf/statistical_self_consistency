import numpy as np
import matplotlib.pyplot as plt

from decision_models.utility.eval_data_loader import load_eval_results
from decision_models.utility.model_name_mapper import replace_model_name


def load_scores(epsilons: list[float], filter_out_llms: bool) -> dict[str, list[float]]:
    # Load all results
    results = load_eval_results()

    scores = {}
    for model, exp_res in results.items():
        values = []

        # Extract eval results
        eval_res = exp_res["evaluation_results"]

        if filter_out_llms and exp_res["decision_model"] is False:
            continue

        # Filter results for the specified epsilons
        for eps in eval_res.keys():
            if eps in epsilons:
                score = float(eval_res[eps]["overall_consistency_score"])
                values.append(score)

        # Add to scores
        scores[model] = values

    return scores


def rank_models(scores: dict, column: int) -> tuple[list[str], dict[str, int]]:
    # Sort models by score for the specified column
    order = sorted(scores, key=lambda model: scores[model][column], reverse=True)

    # Create a dictionary mapping each model to its rank
    ranks = {model: rank for rank, model in enumerate(order, start=1)}

    return order, ranks


def plot_rankings(scores: dict[str, list[float]], epsilons: list[float]):
    # Extract data
    x = np.arange(len(epsilons))
    rankings = [rank_models(scores, column) for column in range(len(epsilons))]
    positions = [
        {model: position for position, model in enumerate(order, start=1)}
        for order, _ in rankings
    ]
    models = rankings[0][0]

    # Colors
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(models), endpoint=False))

    # Create figure
    figure, ax = plt.subplots(figsize=(12 + 2 * (len(epsilons) - 2), max(6, 0.48 * len(models) + 2)))
    figure.subplots_adjust(left=0.30, right=0.70, bottom=0.18, top=0.82)

    # Add vertical lines for each epsilon
    for column in x:
        ax.axvline(column, color="#e4e4e4", linewidth=1, zorder=0)

    # Plot each model's rankings
    for model, color in zip(models, colors):
        y = np.array([position[model] for position in positions])
        ax.fill_between(x, y - 0.15, y + 0.15, color=color, alpha=0.10, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=7, clip_on=False, zorder=3)
        for column, (_, ranks) in enumerate(rankings):
            score = scores[model][column]
            model_name = replace_model_name(model)

            # Determine the label and annotation position based on the column
            if column == 0:
                label = rf"{ranks[model]}. {model_name} ({100 * score:.1f}\%)"
                offset, align = (-12, 0), "right"
            elif column == len(epsilons) - 1:
                label = rf"{ranks[model]}. {model_name} ({100 * score:.1f}\%)"
                offset, align = (12, 0), "left"
            else:
                label = rf"{ranks[model]} ({100 * score:.1f}\%)"
                offset, align = (10, 0), "left"

            # Annotate the plot with the model's rank and score
            ax.annotate(
                label, (column, y[column]), xytext=offset, textcoords="offset points",
                ha=align, va="center", fontsize=10, color=color, annotation_clip=False,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1), zorder=4,
            )

    # Formatting
    ax.set_xticks(x, [rf"$\varepsilon = {epsilon:g}$" for epsilon in epsilons], fontsize=13)
    ax.tick_params(axis="x", length=0, pad=14)
    ax.set_xlim(0, len(epsilons) - 1)
    ax.set_ylim(len(models) + 0.5, 0.5)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title and labels
    figure.suptitle("Self-Consistency Rankings", fontsize=21, y=0.96)
    figure.text(0.5, 0.89, "Ranked by split consistency score. Higher scores are better.",
                ha="center", fontsize=11, color="#555555")

    return figure


if __name__ == "__main__":
    # Define the list of epsilon values to evaluate
    epsilon_list = [0.05, 0.1]

    # Load scores
    scores = load_scores(epsilons=epsilon_list, filter_out_llms=True)

    print(scores)
