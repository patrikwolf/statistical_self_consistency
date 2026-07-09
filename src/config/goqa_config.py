from dataclasses import dataclass
from typing import Literal
from config.constants import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SUMMARY_RESULTS_FILE,
    DEFAULT_DIRECT_RESULTS_FILE,
    DEFAULT_MICRO_MACRO_RESULTS_FILE,
    DEFAULT_LOG_CSV,
)


@dataclass
class GlobalOpinionQAConfig:
    # Experiment name
    experiment_name: str

    # Model
    model_name: str = DEFAULT_MODEL_NAME
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    sampling_temperature: float = 1.0

    # Prompting
    prompting_scheme: Literal["sociodemographic", "persona", "unspecified"] = "unspecified"

    # Experiment control
    num_of_samples: int = DEFAULT_NUM_SAMPLES
    max_number_attempts: int = DEFAULT_MAX_ATTEMPTS

    # Logging / output
    summary_results_file: str = DEFAULT_SUMMARY_RESULTS_FILE
    direct_results_file: str = DEFAULT_DIRECT_RESULTS_FILE
    micro_macro_results_file: str = DEFAULT_MICRO_MACRO_RESULTS_FILE
    log_csv: str = DEFAULT_LOG_CSV
