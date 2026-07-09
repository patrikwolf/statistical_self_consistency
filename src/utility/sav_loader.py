import pandas as pd
import pyreadstat

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseSavLoader(ABC):
    """Shared loader for SPSS `.sav` files with convenience metadata helpers."""

    def __init__(self) -> None:
        self.sav_path = self.get_sav_path()
        self._df, self._meta = pyreadstat.read_sav(self.sav_path)

    @abstractmethod
    def get_sav_path(self) -> Path:
        """Return the absolute path to the dataset `.sav` file."""

    def get_df(self) -> pd.DataFrame:
        return self._df

    def get_meta(self) -> pyreadstat.metadata_container:
        return self._meta

    def load(self) -> tuple[pd.DataFrame, pyreadstat.metadata_container]:
        return self._df, self._meta

    def get_col_labels(self) -> list[str]:
        return list(self._meta.column_labels)

    def get_col_names(self) -> list[str]:
        return list(self._meta.column_names)

    def convert_col_name_to_long_label(self, col_name: str) -> str:
        if col_name in self._meta.column_names_to_labels:
            return self._meta.column_names_to_labels[col_name]
        raise KeyError(f"Column '{col_name}' not found in metadata.")

    def get_all_long_labels(self) -> list[str]:
        return list(self._meta.column_names_to_labels.values())

    def get_value_labels_mapping(self) -> dict[str, Any]:
        return self._meta.value_labels

    def get_variable_to_label_mapping(self) -> dict[str, Any]:
        return self._meta.variable_to_label

    def get_value_label_mapping_for_col_name(self, col_name: str) -> dict[Any, str]:
        if col_name in self._meta.variable_value_labels:
            return self._meta.variable_value_labels[col_name]

        print(f"   WARNING: Column '{col_name}' not found in metadata. Returning empty value map.")
        return {}
