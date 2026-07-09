import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments_acs.filtering.filter import Filter
from experiments_acs.filtering.filter_survey_data import filter_survey_data


def compute_mean_incomes(
        survey_df: pd.DataFrame,
        attribute_dict: dict,
        decomposition_attribute: str,
        possible_values: list,
        target_col: str,
        weight_col: str
) -> list[dict]:

    results = []
    for possible_value in possible_values:
        # Define filters
        filters = [Filter(attribute=decomposition_attribute, values=[possible_value])]

        # Filter data
        filtered_data, sizes = filter_survey_data(
            survey_df=survey_df,
            attribute_dict=attribute_dict,
            filters=filters,
            target_attribute=target_col)

        # Extract income, weights, and prior
        target_array = np.array(filtered_data[target_col])
        weight_array = np.array(filtered_data[weight_col])
        prior = sizes["weighted_prior"]

        # Apply weights to incomes
        target_array = np.repeat(target_array, weight_array.astype(int))
        if target_array.shape[0] > 0:
            mean_income = np.mean(target_array)
        else:
            print(f"     WARNING: Target array is empty (target: {target_array}, prior: {prior})")
            mean_income = 0
        print(f"Mean Income for {decomposition_attribute} = {possible_value}: {mean_income:.0f} USD")

        # Store results
        results.append({
            "attribute": decomposition_attribute,
            "value": possible_value,
            "prior": prior,
            "mean_income": mean_income,
        })

    return results


def contains_value(arr, x):
    if isinstance(x, float) and math.isnan(x):
        return any(isinstance(y, float) and math.isnan(y) for y in arr)
    return x in arr


def equal_with_nan(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def print_statistics(
        decomposition_attribute: str,
        values_first_group: list,
        possible_values: list,
        income_results: list,
) -> list[dict]:
    stats = []

    # Create value split
    other_values = sorted([x for x in possible_values if not contains_value(values_first_group, x)])
    values_split = [values_first_group, other_values]

    # Aggregate prior and mean income
    for split in values_split:
        cumulative_prior = 0
        cumulative_mean_income = 0
        for val in split:
            # Find result for this value
            res = next((r for r in income_results if equal_with_nan(r["value"], val)), None)

            # Extract information
            prior = res["prior"]
            mean_income = res["mean_income"]
            cumulative_prior += prior
            cumulative_mean_income += mean_income * prior

        # Renormalize
        cumulative_mean_income = cumulative_mean_income / cumulative_prior

        # Save results
        stats.append({
            "values": split,
            "prior": cumulative_prior,
            "mean_income": cumulative_mean_income,
        })

    print(f"Decomposition attribute: {decomposition_attribute}")
    print("*" * 80)
    for stat in stats:
        print(f"Split stat: {stat['values']}")
        print(f"Prior stat: {stat['prior']}")
        print(f"Mean income: {stat['mean_income']}")
        print("-" * 40)

    # YAML
    print("\nYAML\n")
    print(f"- attribute: {decomposition_attribute}")
    print("  split:")
    for stat in stats:
        print(f"    - values: {stat['values']}")
        print(f"      prior: {stat['prior']:2f}")
        print(f"      mean_income: {stat['mean_income']:.0f}")

    return stats


def side_by_side_income_distribution(
        decomposition_attribute: str,
        first_stats: dict,
        second_stats: dict,
):
    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.bar(first_stats["bin_centers"], first_stats["counts"], width=first_stats["width"], align="edge",
            alpha=0.6, color="blue")
    ax1.axvline(first_stats["mean_income"], color="blue", linestyle="--", linewidth=2, label="Mean")
    ax2.bar(second_stats["bin_centers"], second_stats["counts"], width=second_stats["width"], align="edge",
            alpha=0.6, color="red")
    ax2.axvline(second_stats["mean_income"], color="red", linestyle="--", linewidth=2, label="Mean")

    # Formatting
    if len(first_stats["values"]) > 7:
        first_values = f"{first_stats['values'][0]}, {first_stats['values'][1]}, ..., {first_stats['values'][-1]}"
    else:
        first_values = first_stats["values"]
    if len(second_stats["values"]) > 7:
        second_values = f"{second_stats['values'][0]}, {second_stats['values'][1]}, ..., {second_stats['values'][-1]}"
    else:
        second_values = second_stats["values"]
    fig.suptitle(f"Income Distribution by Subgroup for ``{decomposition_attribute}''")
    ax1.set_title(f"{decomposition_attribute} = {first_values} (prior = {100 * first_stats['prior']:.0f}" + r"\%)")
    ax1.grid()
    ax1.legend()
    ax2.set_title(f"{decomposition_attribute} = {second_values} (prior = {100 * second_stats['prior']:.0f}" + r"\%)")
    ax2.grid()
    ax2.legend()
    plt.tight_layout()

    return plt
