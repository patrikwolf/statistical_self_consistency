import yaml

from pathlib import Path


def load_secret_config():
    # Determine base directory
    base_dir = Path(__file__).parent.parent.parent

    # Load data from chatbots.yaml
    with open(base_dir / "secrets" / "secret_config.yaml", "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_secret_config()

    gemini_api_key = config["gemini_api_key"]

    print(gemini_api_key)
