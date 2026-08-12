import json

from file_logging.read_and_write_json import read_json
from utility.directories import get_results_dir


def convert_llm_shards_to_json_tree(
        experiment_name: str,
        datetime: str,
        cluster: bool,
) -> tuple[dict, dict, int]:
    # Directory
    results_dir = get_results_dir(
        experiment_name=experiment_name,
        cluster=cluster,
        use_timestamp=True,
        timestamp=datetime,
        create_dir=False)

    # Check if directory exists
    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {results_dir}")

    # Load shards
    shards = sorted(results_dir.glob("shard_*__summary_results.json"))

    # Create tree dir
    tree_dir = results_dir / "tree"
    tree_dir.mkdir(parents=True, exist_ok=True)

    # Process each shard
    tree = {}
    metadata = None
    for idx, shard in enumerate(shards):
        data = read_json(
            experiment_name=f"{experiment_name}/{datetime}",
            filename=shard.name,
            cluster=cluster,
        )

        # Save metadata from the first shard
        if idx == 0:
            metadata = {
                "survey_name": data["survey_name"],
                "survey_year": data["survey_year"],
                "target_column": data["target_column"] if "target_column" in data.keys() else "Unknown",
                "target_description": data["target_description"] if "target_description" in data.keys() else "Unknown",
                "cluster": cluster, "timestamp": datetime, "model": data["model"].split("/")[1],
                "reasoning_effort": data["reasoning_effort"], "prompting_scheme": data["prompting_scheme"],
                "prompt_reasoning": data["prompt_reasoning"], "num_of_samples": data["num_of_samples"]}
            if "temperature" in data.keys():
                metadata["temperature"] = data["temperature"]
            else:
                metadata["temperature"] = "–"

            with open(tree_dir / "meta.json", "w") as json_file:
                json.dump(metadata, json_file, indent=4)

        # Create node label from filters
        attributes = []
        value_list = []
        filter_label = []
        for f in data["filters"]:
            attributes.append(f["attribute"])
            values = f["values"]
            value_list.append(values)
            if len(values) > 5:
                condensed_values = f"{values[0]} – {values[-1]}"
                filter_label.append(f"{f['attribute']} = {condensed_values}")
            else:
                filter_label.append(f"{f['attribute']} = {values}")

        # Create tree node
        node = {
            "id": data["node_id"],
            "parent_id": data["parent_id"],
            "level": data["level"],
            "label": ", ".join(filter_label),
            "short_label": filter_label[-1] if len(filter_label) > 0 else "root",
            "model": data["model"],
            "filters": data["filters"],
            "attributes": attributes,
            "decisive_attribute": attributes[-1] if len(attributes) > 0 else "none",
            "decisive_value": value_list[-1] if len(value_list) > 0 else "none",
            "num_of_samples": data["num_of_samples"],
            "avg_bin_distribution_list": data["avg_bin_distribution_list"],
            "bin_edges": data["bin_edges"],
        }

        # Add node to tree
        level = data["level"]
        if level in tree.keys():
            tree[level].append(node)
        else:
            tree[level] = [node]

    # Save tree
    for level, nodes in tree.items():
        path = tree_dir / f"{level}.json"
        with open(path, "w") as json_file:
            json.dump(nodes, json_file, indent=4)

    # Number of levels
    num_levels = len(tree)

    return tree, metadata, num_levels


if __name__ == "__main__":
    experiment_name = "llm_few_bins"
    datetime = "2026-05-13__15-00-00"
    cluster = False

    # Convert shards to JSON tree
    _, _, num_levels = convert_llm_shards_to_json_tree(
        experiment_name=experiment_name,
        datetime=datetime,
        cluster=cluster,
    )

    print(f"Level: {num_levels}")
    print("Done!")
