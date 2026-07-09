import math
import pandas as pd

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from tabulate import tabulate
from utility.directories import get_log_dir

TIMESTAMP_COLUMNS = ("date", "time")


def _csv_path(filename: str, cluster: bool = False) -> Path:
    return get_log_dir(cluster=cluster) / filename


def _timestamp_values(date: str | None = None, timestamp: str | None = None) -> tuple[str, str]:
    if date is not None and timestamp is not None:
        return date, timestamp

    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")


def _add_datetime(
        results: Mapping[str, Any],
        date: str | None = None,
        timestamp: str | None = None,
) -> dict[str, Any]:
    row = dict(results)
    row_date, row_time = _timestamp_values(date=date, timestamp=timestamp)
    row["date"] = row_date
    row["time"] = row_time
    return row


def _apply_filters(df: pd.DataFrame, filters: Mapping[str, Any] | None) -> pd.DataFrame:
    if filters is None:
        return df

    filtered_df = df
    for column, value in filters.items():
        if column not in filtered_df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping filter.")
            continue

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            filtered_df = filtered_df[filtered_df[column].isin(value)]
        else:
            filtered_df = filtered_df[filtered_df[column] == value]

    return filtered_df


def _round_float(value: Any, digits: int = 5) -> Any:
    if not isinstance(value, float):
        return value
    if math.isnan(value):
        return value
    return round(value, digits)


def log_to_csv(
        results: Mapping[str, Any],
        filename: str,
        date: str | None = None,
        timestamp: str | None = None,
) -> Path:
    """Append one result row to a semicolon-separated CSV file in the project log directory."""
    path = _csv_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add date and time
    row = _add_datetime(results=results, date=date, timestamp=timestamp)
    df_new = pd.DataFrame([row])

    # Create new file if it does not exist
    if not path.exists():
        df_new.to_csv(path, index=False, sep=";")
        return path

    try:
        df_existing = pd.read_csv(path, sep=";")

        # Combine columns
        all_columns = list(df_existing.columns.union(df_new.columns))

        # Reorder: date, time, then sorted remaining
        remaining_cols = [col for col in all_columns if col not in ["date", "time"]]
        ordered_columns = ["date", "time"] + remaining_cols

        # Ensure both frames have the same columns
        df_existing = df_existing.reindex(columns=ordered_columns)
        df_new = df_new.reindex(columns=ordered_columns)

        # Append new row
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(path, index=False, sep=";")
    except Exception as exc:
        print(f"Failed to update existing CSV: {exc}")

    return path


def read_csv(filename: str, cluster: bool = False) -> pd.DataFrame:
    """Read a semicolon-separated CSV file and return the newest rows first."""
    path = _csv_path(filename, cluster=cluster)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path, sep=";")
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file at {path}: {exc}") from exc

    # Move "date" and "time" to first positions if they exist
    ordered_timestamp_columns = [column for column in TIMESTAMP_COLUMNS if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in TIMESTAMP_COLUMNS]
    df = df.loc[:, ordered_timestamp_columns + remaining_columns]

    return df.iloc[::-1].reset_index(drop=True)


def print_tabulated(
        filename: str,
        cols: list[str] | None = None,
        head: int | None = None,
        cluster: bool = False,
        filters: Mapping[str, Any] | None = None,
        sort_by: str | list[str] | None = None,
        ascending: bool | list[bool] = False,
        truncate_d: int = 5,
) -> tuple[list[str], pd.DataFrame]:
    """Print a CSV file as a formatted table and return the original columns and displayed frame."""
    df = read_csv(filename, cluster=cluster)
    all_cols = df.columns.tolist()

    # Subsample specific columns if provided
    if cols is not None:
        df = df[cols]

    # Filter columns
    if filters is not None:
        for col, value in filters.items():
            if col in df.columns:
                if isinstance(value, list):
                    df = df[df[col].isin(value)]
                else:
                    df = df[df[col] == value]
            else:
                print(f"Warning: Column '{col}' not found in DataFrame! Skipping filter.")

    # Sort columns
    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Truncate all float columns
    float_columns = df.select_dtypes(include="float").columns
    df[float_columns] = df[float_columns].map(lambda value: _round_float(value, digits=truncate_d))

    # Print the DataFrame
    table_df = df.head(head) if head is not None else df
    print(tabulate(table_df, headers="keys", tablefmt="pretty", showindex=False))

    return all_cols, df


if __name__ == "__main__":
    example_results = {
        "model_name": "example_model",
        "epoch": 1,
        "loss": 0.05,
        "tv_distance": 0.1,
        "jsd_distance": 0.2,
    }

    log_to_csv(example_results, "test.csv", date="2025-08-05", timestamp="08:00:00")

    all_columns, dataframe = print_tabulated(filename="test.csv", head=10)
    print(all_columns)

    print("*")

    selected_columns = ["date", "time", "epoch"]
    print_tabulated(filename="test.csv", cols=selected_columns, head=3)
