import json
import pandas as pd

from data_loader_acs.value_map import get_value_map
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.filtering.filter import Filter


def filter_survey_data(
        survey_df: pd.DataFrame,
        attribute_dict: dict,
        filters: list[Filter],
        target_attribute: str,
        weight_col="PWGTP",
) -> tuple[pd.DataFrame, dict]:
    """
    Filter survey data based on provided filters.

    Args:
        survey_df: DataFrame containing survey data_acs
        attribute_dict: Dictionary mapping attribute names to their metadata
        filters: List of filter dictionaries, each containing:
            - attribute: Column name to filter on
            - description: Human-readable description
            - values: List of values to include
        target_attribute: Column name for target attribute

    Returns:
        Filtered DataFrame
    """
    # Initialize filtered data
    df_copy = survey_df.copy()
    num_samples_incl_nan = len(df_copy)
    weight_incl_nan = df_copy[weight_col].sum()

    # Drop rows with missing values for income
    clean_survey_df = df_copy.dropna(subset=[target_attribute])
    num_samples_without_nan = len(clean_survey_df)
    weight_without_nan = clean_survey_df[weight_col].sum()

    nan_share = 1 - clean_survey_df.shape[0] / survey_df.shape[0]
    print(f"  ----> Filtered out all NaN values in {target_attribute} column ({100 * nan_share:.0f}%)")

    # Iterate over filters and apply them to the survey data
    for f in filters:
        col = f.attribute
        values = f.values

        # Print filtering information
        value_map = get_value_map(attribute=col, attribute_dict=attribute_dict)
        print(f"Constraining the column {col} ({f.description}) to the following values"
              f" {[value_map(str(v)) for v in f.values]}")

        # Apply filtering
        if col in clean_survey_df.columns:
            clean_survey_df = clean_survey_df[clean_survey_df[col].isin(values)]
        else:
            print(f"Attribute {col} not found in data_acs_income columns.")

    num_samples_filtered = len(clean_survey_df)
    weight_after_filtering = clean_survey_df[weight_col].sum()

    sizes = {
        "unfiltered": {
            "num_samples_incl_nan": num_samples_incl_nan,
            "weight_incl_nan": float(weight_incl_nan),
            "num_samples_without_nan": num_samples_without_nan,
            "weight_without_nan": float(weight_without_nan),
        },
        "filtered": {
            "num_samples": num_samples_filtered,
            "weight_filtered": float(weight_after_filtering),
        },
        "unweighted_prior": float(num_samples_filtered / num_samples_without_nan),
        "weighted_prior": float(weight_after_filtering / weight_without_nan),
    }

    return clean_survey_df, sizes


if __name__ == "__main__":
    # Load survey dataset
    survey_df, attribute_dict = load_survey_data(year=2024, set_nan_to_zero=True)

    # Filters
    filters = [
        Filter(
            attribute="AGEP",
            description="Age of the person",
            values=[24, 25, 26, 27, 28, 29, 30],
        ),
        Filter(
            attribute="SEX",
            description="Sex of the person",
            values=[2],
        ),
    ]

    # Filter data
    filtered_data, sizes = filter_survey_data(survey_df=survey_df,
                                              attribute_dict=attribute_dict,
                                              filters=filters,
                                              target_attribute="PINCP")

    # Check for NaN values in filtered data
    assert filtered_data["PINCP"].isna().sum() == 0, "Filtered data contains NaN values in 'PINCP' column (income)."

    print(f"Number of samples after filtering: "
          f"{len(filtered_data)} (= {100 * len(filtered_data) / len(survey_df):.2f}%) out of {len(survey_df)}")

    print(json.dumps(sizes, indent=2))
