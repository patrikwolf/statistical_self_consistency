import json
import yaml

from pathlib import Path


if __name__ == "__main__":
    attributes = ["AGEP", "COW", "WKHP", "OCCP", "ESR", "MAR"]
    decomp = {}

    with open("splits.yaml", "r", encoding="utf-8") as f:
        splits_dict = yaml.safe_load(f)

    # Construct dict
    for idx, attr in enumerate(attributes):
        split_info = next((r for r in splits_dict if r["attribute"] == attr), None)

        # Get split
        splits = []
        for item in split_info["split"]:
            # Preprocessing
            sorted_values = sorted(
                item["values"],
                key=lambda x: (x != "nan", x if x != "nan" else 0),
            )
            sorted_values = [float("nan") if value == "nan" else value for value in sorted_values]

            # Add to list
            splits.append(sorted_values)

        # Add to dict
        decomp[f"level_{idx + 1}"] = {
            "attribute": attr,
            "splits": splits,
        }

    # Save file
    path = Path("decomposition_attributes.json")
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(decomp, json_file, indent=4)

    print("Done")
