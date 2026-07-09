from pathlib import Path
from utility.directories import get_wvs_data_dir
from utility.sav_loader import BaseSavLoader


class WvsSavLoader(BaseSavLoader):
    """Load the WVS SPSS dataset and expose metadata helpers."""

    def get_sav_path(self) -> Path:
        return get_wvs_data_dir() / "WVS_Cross-National_Wave_7_spss_v6_0.sav"


if __name__ == "__main__":
    data_loader = WvsSavLoader()
    df, meta = data_loader.load()

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)[:5]}")

    print("\n" + "*" * 80 + "\n")

    print(df["W_WEIGHT"][:20])

    print("\n" + "*" * 80 + "\n")

    idx = 4
    column_names = data_loader.get_col_names()
    col_labels = data_loader.get_col_labels()
    long_labels = data_loader.get_all_long_labels()
    print(f"Column names: {column_names[:5]}")
    print(f"Column labels: {col_labels[:5]}")
    print(f"Column label for column name: '{column_names[idx]}' is:\n"
          f"     {data_loader.convert_col_name_to_long_label(col_name=column_names[idx])}\n")

    print("\n" + "*" * 80 + "\n")

    labels = "labels0"
    value_labels = data_loader.get_value_labels_mapping()
    variable_to_label = data_loader.get_variable_to_label_mapping()
    variable_value_labels = data_loader.get_value_label_mapping_for_col_name(col_name=column_names[idx])
    print(f"Value labels for : '{labels}' are given by: {value_labels[labels]}")
    print(f"Label of column: '{column_names[idx]}' is given by {variable_to_label[column_names[idx]]}")
    print(f"Value labels for col: '{column_names[idx]}' is: {variable_value_labels}")

    print("\n" + "*" * 80 + "\n")

    for col_name in column_names[:20]:
        print(f" ---> Column name: {col_name}")
        print(f" ---> Column label for column name: '{col_name}' is:\n"
              f"      {data_loader.convert_col_name_to_long_label(col_name=col_name)}")
        print(f" ---> Value labels for column name: '{col_name}' is:\n"
              f"      {data_loader.get_value_label_mapping_for_col_name(col_name=col_name)}")
        print("--------------------------------")
