import json

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
from config.constants import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_SAMPLES,
)


@dataclass
class ModelConfig:
    # Experiment name
    experiment_name: str

    # Country
    country: str = None

    # Model
    model_name: str = DEFAULT_MODEL_NAME
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    sampling_temperature: float = 1.0

    # Experiment control
    num_of_samples: int = DEFAULT_NUM_SAMPLES
    max_number_attempts: int = DEFAULT_MAX_ATTEMPTS

    def save_json(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)


if __name__ == "__main__":
    # Config
    cfg = ModelConfig(
        experiment_name="wvs_llm_estimates",
        country="Canada",
        model_name="x-ai/grok-4.1-fast",
        reasoning_effort="none",
        sampling_temperature=1.0,
        # todo: increase to 20
        num_of_samples=2,
        max_number_attempts=10,
    )

    file_path = Path('/Users/patrikwolf/Desktop/config.json')
    cfg.save_json(path=file_path)
