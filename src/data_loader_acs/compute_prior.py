import pandas as pd

from data_loader_acs.data_loader import load_survey_data


def compute_prior(
        survey_df: pd.DataFrame,
        decomposition_attributes: list[str],
        weighted: bool,
        weight_col: str = "PWGTP",
) -> pd.DataFrame:
    if not weighted:
        raise ValueError("Invalid estimates without survey weighting!")

    return compute_weighted_prior(
        survey_df=survey_df,
        decomposition_attributes=decomposition_attributes,
        weight_col=weight_col,
    )


def compute_weighted_prior(
        survey_df: pd.DataFrame,
        decomposition_attributes: list[str],
        weight_col: str = "PWGTP",
) -> pd.DataFrame:
    """Compute prior probabilities for given decomposition attributes using survey weights.

    Args:
        survey_df (pd.DataFrame): Survey dataset containing decomposition attributes and weights.
        decomposition_attributes (list[str]): List of attributes to decompose the prior over.
        weight_col (str): Column name for survey weights. Default is "PWGTP".

    Returns:
        pd.DataFrame: DataFrame with decomposition attributes and their corresponding prior probabilities.
    """
    # Group and sum weights (also supports NaN values)
    grouped = (
        survey_df
        .groupby(decomposition_attributes, as_index=False, dropna=False)[weight_col]
        .sum()
        .rename(columns={weight_col: "weight_sum"})
    )

    # Compute priors
    grouped["prior"] = grouped["weight_sum"] / grouped["weight_sum"].sum()

    return grouped


def compute_unweighted_prior(
        survey_df: pd.DataFrame,
        decomposition_attributes: list[str],
) -> pd.DataFrame:
    """Compute prior probabilities for given decomposition attributes without using survey weights.

    Args:
        survey_df (pd.DataFrame): Survey dataset containing decomposition attributes.
        decomposition_attributes (list[str]): List of attributes to decompose the prior over.

    Returns:
        pd.DataFrame: DataFrame with decomposition attributes and their corresponding prior probabilities.
    """
    # Group and count occurrences
    grouped = survey_df.groupby(decomposition_attributes, dropna=False).size().reset_index(name="count")
    grouped["prior"] = grouped["count"] / len(survey_df)

    return grouped


if __name__ == "__main__":
    # Load survey dataset
    survey_df, _ = load_survey_data(year=2024, set_nan_to_zero=True)

    # Define decomposition attributes
    decomposition_attributes = ["ESR"]

    # Compute weighted prior probabilities
    prior_df = compute_prior(
        survey_df=survey_df,
        decomposition_attributes=decomposition_attributes,
        weighted=True,
        weight_col="PWGTP",
    )

    print("Weighted prior probabilities:")
    print(prior_df)

    # Compute unweighted prior probabilities
    """
    prior_df_unweighted = compute_prior(
        survey_df=survey_df,
        decomposition_attributes=decomposition_attributes,
        weighted=False,
    )

    print("\n" + "*" * 80 + "\n")
    print("Unweighted prior probabilities:")
    print(prior_df_unweighted)
    """
