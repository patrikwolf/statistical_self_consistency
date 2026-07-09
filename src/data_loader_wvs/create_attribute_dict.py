import json

from data_loader_wvs.wvs_sav_loader import WvsSavLoader
from utility.create_attribute_dict import save_attribute_dict, load_attribute_dict
from utility.directories import get_wvs_data_dir


if __name__ == "__main__":
    data_loader = WvsSavLoader()
    data_path = get_wvs_data_dir()
    file_name = "../../data/wvs_wave_7/wvs_attribute_dict.json"
    file_path = data_path / file_name

    # Save attribute_dict
    save_attribute_dict(data_loader=data_loader, file_path=file_path)
    print(f"Attribute dict saved to {file_path}")

    # Load attribute_dict
    attribute_dict = load_attribute_dict(file_path=file_path)
    print(json.dumps(attribute_dict["Q292G"], indent=4))
    print(f"Found {len(list(attribute_dict.keys()))} attributes")
