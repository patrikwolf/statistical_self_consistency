import json
import pandas as pd

from data_loader_acs.data_loader import load_survey_data
from experiments_acs.filtering.filter import Filter
from experiments_acs.filtering.filter_survey_data import filter_survey_data


def compute_ground_truth_prob(
        survey_df: pd.DataFrame,
        attribute_dict: dict,
        filters: list[Filter],
        target_threshold: int,
        weighted: bool,
        target_greater_than_threshold: bool,
        target_col: str = "PINCP",
        weight_col: str = "PWGTP",
) -> tuple[dict, pd.DataFrame]:
    # Filter data
    filtered_data, sizes = filter_survey_data(survey_df=survey_df,
                                              attribute_dict=attribute_dict,
                                              filters=filters,
                                              target_attribute=target_col)

    # Compute total weighted count
    total_weight = filtered_data[weight_col].sum()
    if weighted:
        if total_weight == 0:
            print("    ---> WARNING: Total weight is zero after filtering.")
            results = {
                "high_income_probability": 0.0,
                "prior": sizes["weighted_prior"],
                "unfiltered_size": sizes["unfiltered"],
                "filtered_size": sizes["filtered"]["num_samples"],
                "filtered_weight": 0.0,
                "filtered_high_income_weight": 0.0,
            }
            return results, filtered_data

        # Compute probability of income > threshold
        if target_greater_than_threshold:
            income_after_threshold = filtered_data[filtered_data[target_col] > target_threshold]
        else:
            income_after_threshold = filtered_data[filtered_data[target_col] <= target_threshold]
        high_income_weight = income_after_threshold[weight_col].sum()
    else:
        raise ValueError("Invalid estimates without survey weighting!")

    # Collect statistics
    results = {
        "high_income_probability": float(high_income_weight / total_weight),
        "prior": sizes["weighted_prior"],
        "unweighted_prior": sizes["unweighted_prior"],
        "unfiltered": sizes["unfiltered"],
        "filtered_size": sizes["filtered"]["num_samples"],
        "filtered_weight": int(total_weight) if weighted else None,
        "filtered_high_income_weight": float(high_income_weight) if weighted else None,
    }

    return results, filtered_data


if __name__ == "__main__":
    survey_df, attribute_dict = load_survey_data(year=2024, set_nan_to_zero=True)
    income_threshold = 40_000
    filter = Filter(attribute="MAR", values=[1, 2])

    # Compute ground truth probability of income > threshold
    ground_truth_results, base_filtered_survey_df = compute_ground_truth_prob(
        survey_df=survey_df,
        attribute_dict=attribute_dict,
        filters=[filter],
        target_threshold=income_threshold,
        weighted=True,
        target_greater_than_threshold=True,
    )

    print(json.dumps(ground_truth_results, indent=4))
