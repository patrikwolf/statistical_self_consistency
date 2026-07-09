import numpy as np
import pandas as pd

from data_loader_acs.data_loader import load_survey_data
from data_loader_acs.value_map import get_value_map
from experiments_acs.filtering.filter import Filter
from experiments_acs.filtering.filter_definitions import extend_generic_filter
from experiments_acs.filtering.filter_survey_data import filter_survey_data


def get_empirical_survey_distribution(
        survey_df: pd.DataFrame,
        attribute_dict: dict,
        filters: list[Filter],
        weighted: bool,
        target_col: str,
        weight_col: str = "PWGTP",
        bins: np.ndarray | int = 200,
        density: bool = True,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, float]:
    # Filter data
    filtered_data, sizes = filter_survey_data(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=filters,
        target_attribute=target_col)

    weighted_prior = sizes["weighted_prior"]

    # Extract income and survey weights
    target_array = np.array(filtered_data[target_col])
    weight_array = np.array(filtered_data[weight_col])
    total_weight = np.sum(weight_array)

    # Apply weights to incomes
    if weighted:
        target_array = np.repeat(target_array, weight_array.astype(int))
    else:
        raise NotImplementedError("Unweighted income distribution not implemented.")

    # Bin incomes
    counts, bin_edges = np.histogram(target_array, bins=bins, density=density)

    # Compute probability mass outside domain
    mask = (target_array < bin_edges[0]) | (target_array > bin_edges[-1])
    captured_probability_mass = 1 - len(target_array[mask]) / len(target_array)

    return target_array, total_weight, weighted_prior, counts, bin_edges, captured_probability_mass


if __name__ == "__main__":
    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=2024, set_nan_to_zero=True)

    # Filter
    decomposition_filter = Filter(
        attribute="SEX",
        getter=extend_generic_filter,
        values=[2],
        description=attribute_dict["SEX"]["description"],
        value_map=get_value_map(attribute="SEX", attribute_dict=attribute_dict),
    )

    # Bins
    bins = np.linspace(start=-20_000, stop=400_000, num=1 + 200)

    # Compute income distribution
    target_array, total_weight, weighted_prior, counts, bin_edges, probability_mass = get_empirical_survey_distribution(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=[decomposition_filter],
        weighted=True,
        target_col="PINCP",
        bins=bins,
    )

    print(target_array[:10])
