import pandas as pd

from datetime import datetime
from pathlib import Path
from folktexts import ACSDataset
from folktexts.acs.acs_tasks import ACSTaskMetadata, acs_columns_map
from folktexts.col_to_text import ColumnToText
from utility.directories import get_acs_data_dir

# The Census Bureau renamed some PUMS person-record variables in the 2019
# release, but the folktables task definitions still use the pre-2019 names.
PUMS_2019_COLUMN_RENAMES = {
    "RELP": "RELSHIPP",
    "JWTR": "JWTRNS",
}

# RELSHIPP (2019+): Relationship to Reference Person, with new codes 20-38
# replacing the old RELP codes 0-17.
acs_relationship_2019 = ColumnToText(
    "RELSHIPP",
    short_description="relationship to the reference person in the survey",
    value_map={
        20: "The reference person itself",
        21: "Opposite-sex husband/wife/spouse",
        22: "Opposite-sex unmarried partner",
        23: "Same-sex husband/wife/spouse",
        24: "Same-sex unmarried partner",
        25: "Biological son or daughter",
        26: "Adopted son or daughter",
        27: "Stepson or stepdaughter",
        28: "Brother or sister",
        29: "Father or mother",
        30: "Grandchild",
        31: "Parent-in-law",
        32: "Son-in-law or daughter-in-law",
        33: "Other relative",
        34: "Roommate or housemate",
        35: "Foster child",
        36: "Other non-relative",
        37: "Institutionalized group quarters population",
        38: "Non-institutionalized group quarters population",
    },
)

# JWTRNS (2019+): Means of Transportation to Work, replacing JWTR.
acs_commute_method_2019 = ColumnToText(
    "JWTRNS",
    short_description="means of transportation to work",
    missing_value_fill="N/A (not a worker, or worker who worked from home)",
    value_map={
        1: "Car, truck, or van",
        2: "Bus",
        3: "Subway or elevated rail",
        4: "Long-distance train or commuter rail",
        5: "Light rail, streetcar, or trolley",
        6: "Ferryboat",
        7: "Taxicab",
        8: "Motorcycle",
        9: "Bicycle",
        10: "Walked",
        11: "Worked from home",
        12: "Other method",
    },
)

# The tasks' `cols_to_text` all reference this shared module-level map, so
# registering the renamed columns here makes them available to every task.
acs_columns_map["RELSHIPP"] = acs_relationship_2019
acs_columns_map["JWTRNS"] = acs_commute_method_2019


def _get_task_for_year(task_name: str, year: int) -> ACSTaskMetadata:
    """Return the folktexts task, with features renamed for 2019+ survey data."""
    task = ACSTaskMetadata.get_task(task_name)
    # Tasks are module-level singletons, so always start from the pristine
    # folktables feature list instead of renaming a possibly-renamed list.
    original_features = (
        list(task.folktables_obj.features) if task.folktables_obj else task.features
    )
    if year >= 2019:
        task.features = [PUMS_2019_COLUMN_RENAMES.get(f, f) for f in original_features]
    else:
        task.features = original_features
    return task


class ACSMultiTaskDatasetClient:
    """
    Download ACS dataset for a given folktexts task and survey year.

    Available tasks:

    - ACSIncome (PINCP)
        features: ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        predict whether an individual's income is above $50,000
        "What is the probability that this person's estimated yearly income is above $50,000 ?"
        "Below $50,000" or "Above $50,000"

    - ACSPublicCoverage (PUBCOV)
        features: ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE',
                    'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']
        predict whether an individual is covered by public health insurance
        "Does this person have public health insurance coverage?"
        "Yes, person is covered by public health insurance" or "No, person is not covered by public health insurance"

    - ACSMobility (MIG)
        features: ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIL', 'ANC', 'NATIVITY', 'RELP', 'DEAR',
                    'DEYE', 'DREM', 'RAC1P', 'GCL', 'COW', 'ESR', 'WKHP', 'JWMNP', 'PINCP']
        predict whether an individual had the same residential address one year ago
        "Has this person moved in the last year?"
        "No, person has lived in the same house for the last year" or "Yes, person has moved in the last year"

    - ACSEmployment (ESR)
        features: ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR',
                    'DEYE', 'DREM', 'SEX', 'RAC1P']
        predict whether an individual is employed
        "What is this person's employment status?"
        "Employed civilian" or "Unemployed or in the military"

    - ACSTravelTime (JWMNP)
        features: ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'MIG', 'RELP', 'RAC1P', 'PUMA', 'ST', 'CIT',
                    'OCCP', 'JWTR', 'POWPUMA', 'POVPIP']
        predict whether an individual has a commute to work that is longer than 20 minutes
        "What is this person's commute time?"
        "Longer than 20 minutes" or "Less than 20 minutes"
    """

    def __init__(self) -> None:
        self.data_dir = get_acs_data_dir()

    def load_original_task_data(self, year: int, task: str) -> ACSDataset:
        """Create or load the raw ACS dataset for a specific task."""
        cache_dir = self.data_dir / "original" / task
        cache_dir.mkdir(parents=True, exist_ok=True)
        return ACSDataset.make_from_task(
            task=_get_task_for_year(task_name=task, year=year),
            cache_dir=cache_dir,
            survey_year=str(year),
            horizon="1-Year",
        )


class ACSMultiTaskParquetWriter:
    """Persist ACS datasets as timestamped parquet snapshots."""

    def __init__(self) -> None:
        self.data_dir = get_acs_data_dir()

    def save_data(
        self,
        acs_data: pd.DataFrame,
        year: int,
        file_name: str,
    ) -> tuple[Path, str]:
        """Save a dataframe as parquet in a timestamped year directory."""
        file_name = file_name or "full_acs_data.parquet"
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        if not file_name.endswith(".parquet"):
            file_name = f"{Path(file_name).stem}.parquet"

        output_path = self.data_dir / "parquet" / str(year) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)

        acs_data.to_parquet(output_path / file_name, index=False)
        return output_path, file_name


def load_original_acs_task_data(year: int, task: str) -> ACSDataset:
    """Backward-compatible wrapper for loading raw ACS task data."""
    return ACSMultiTaskDatasetClient().load_original_task_data(year=year, task=task)


def save_data(
    acs_data: pd.DataFrame,
    year: int,
    file_name: str,
) -> tuple[Path, str]:
    """Backward-compatible wrapper for saving parquet snapshots."""
    return ACSMultiTaskParquetWriter().save_data(
        acs_data=acs_data,
        year=year,
        file_name=file_name,
    )


if __name__ == "__main__":
    # Survey year
    selected_year = 2024

    print("Loading full ACS dataset...")
    acs_dataset = load_original_acs_task_data(year=selected_year, task="ACSIncome")

    print("Extracting full dataset...")
    full_acs_data = acs_dataset._full_acs_data

    print("Saving full dataset as parquet file...")
    output_path, parquet_filename = save_data(
        acs_data=full_acs_data,
        year=selected_year,
        file_name="full_acs_data.parquet",
    )
    print(f"Saved {parquet_filename} to {output_path}")
