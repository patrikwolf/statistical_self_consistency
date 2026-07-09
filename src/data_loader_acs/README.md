## ACS Data from Folktexts Package

Source: [folktexts GitHub repository](https://github.com/socialfoundations/folktexts?tab=readme-ov-file)

#### Download Survey Data

1. Use the `ACSDataset.make_from_task()` function from the `folktexts` package to download
the raw ACS data.
2. Run the Python script `data_to_parquet.py` to convert the data into a `.parquet` file.
This accelerates the data import.

#### Attribute Mapping

- We have extracted the attribute mapping from `acs_columns.py` in the `folktexts` package and create a JSON
file from it. The JSON file is named `acs_attributes.json`,