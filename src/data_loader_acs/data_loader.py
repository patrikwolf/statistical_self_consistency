import json
import pandas as pd

from pathlib import Path
from typing import Any

from utility.directories import get_acs_data_dir


class ACSMultiTaskDataLoader:
    """Load persisted ACS multi-task data and metadata from disk."""

    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parent

    def load_survey_data(self, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return the latest saved survey dataset and attribute mapping for a year."""
        data_dir = get_acs_data_dir()
        year_path = data_dir / "parquet" / str(year)
        if not year_path.exists():
            raise FileNotFoundError(
                f"Directory {year_path} does not exist. "
                "Run the ACS download/parquet conversion step first."
            )

        timestamp_dirs = sorted(path for path in year_path.iterdir() if path.is_dir())
        if not timestamp_dirs:
            raise FileNotFoundError(f"No timestamped parquet directories found in {year_path}.")

        survey_df = pd.read_parquet(timestamp_dirs[-1] / "full_acs_data.parquet")
        return survey_df, self.load_attribute_dict()

    def load_attribute_dict(self) -> dict[str, Any]:
        """Load the ACS attribute dictionary from the local JSON file."""
        file_path = self.base_dir / "acs_attributes.json"
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)


def load_survey_data(
        year: int,
        set_nan_to_zero: bool,
        income_column: str = "PINCP",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Loading full ACS multi-task survey data."""
    survey_df, attribute_dict = ACSMultiTaskDataLoader().load_survey_data(year=year)

    if set_nan_to_zero:
        print("   ----> Setting NaN values in income column to zero.")
        survey_df[income_column] = survey_df[income_column].fillna(0)
    else:
        print("\n" + "*" * 80 + "\n")
        print("   ----> IMPORTANT: Make sure to deal with NaN values in income column!")
        print("\n" + "*" * 80 + "\n")

    return survey_df, attribute_dict


def load_original_attribute_dict() -> dict[str, Any]:
    """Backward-compatible wrapper for loading the original attribute dictionary."""
    return ACSMultiTaskDataLoader().load_attribute_dict()


if __name__ == "__main__":
    selected_year = 2024

    print("Loading full dataset and attribute dictionary...")
    survey_dataframe, attribute_dictionary = load_survey_data(year=selected_year, set_nan_to_zero=True)
