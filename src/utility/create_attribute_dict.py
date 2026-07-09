import json

from pathlib import Path
from data_pew_gas.pew_sav_loader import PewSavLoader
from utility.directories import get_pew_data_dir
from utility.sav_loader import BaseSavLoader


def save_attribute_dict(
        data_loader: BaseSavLoader,
        file_path: Path,
) -> Path:
    attribute_dict = {}
    for col_name in data_loader.get_col_names():
        attribute_dict[col_name] = {
            "label": data_loader.convert_col_name_to_long_label(col_name=col_name),
            "value_labels": data_loader.get_value_label_mapping_for_col_name(col_name=col_name),
        }

    # Save attribute dict
    with open(file_path, "w") as json_file:
        json.dump(attribute_dict, json_file, indent=4)

    return file_path


def load_attribute_dict(file_path: Path) -> dict:
    with open(file_path, "r") as json_file:
        attribute_dict = json.load(json_file)

    return attribute_dict


if __name__ == "__main__":
    data_loader = PewSavLoader()
    data_path = get_pew_data_dir()
    file_name = "pew_attribute_dict.json"
    file_path = data_path / file_name

    # Save attribute_dict
    save_attribute_dict(data_loader=data_loader, file_path=file_path)
    print(f"Attribute dict saved to {file_path}")

    # Load attribute_dict
    attribute_dict = load_attribute_dict(file_path=file_path)
    print(json.dumps(attribute_dict["urbanicity"], indent=4))
