import json
import yaml

from pathlib import Path


def load_shared_micro_macro_config() -> dict:
    base_dir = Path(__file__).resolve().parent

    # Load shared config
    with open(base_dir / "shared_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config["shared_config"]


if __name__ == "__main__":
    conf = load_shared_micro_macro_config()
    print(json.dumps(conf, indent=2))
