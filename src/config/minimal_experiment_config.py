from dataclasses import dataclass
from typing import Literal
from config.constants import (
    DEFAULT_INCOME_THRESHOLD,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SUMMARY_RESULTS_FILE,
    DEFAULT_LLM_PRIOR_RESULTS_FILE,
    DEFAULT_DIRECT_RESULTS_FILE,
    DEFAULT_DECOMPOSITION_RESULTS_FILE,
    DEFAULT_LOG_CSV, DEFAULT_APPLY_PERSON_WEIGHTING,
)


@dataclass
class MinimalExperimentConfig:
    """Configuration class for sociodemographic prompting experiments.

    Attributes:
        experiment_name: Name of the experiment
        model_name: HuggingFace model identifier
        num_of_samples: Number of samples to run for statistical analysis
        max_number_attempts: Maximum attempts to extract valid probability from LLM
        income_threshold: Income threshold in USD for probability calculations
        direct_results_file: Filename for direct LLM prediction logs
        decomposition_results_file: Filename for decomposition method logs
        log_csv: Filename for results CSV output
    """
    # Experiment name
    experiment_name: str

    # Survey data
    survey_name: str
    survey_year: int
    set_nan_to_zero: bool

    # Model
    model_name: str = DEFAULT_MODEL_NAME
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    sampling_temperature: float = 1.0

    # Prompting
    improved_age_desc: bool = False
    prompting_scheme: Literal["sociodemographic", "persona", "unspecified"] = "unspecified"
    prompt_reasoning: bool = False
    llm_aggregation_priors: bool = False

    # Experiment control
    num_of_samples: int = DEFAULT_NUM_SAMPLES
    max_number_attempts: int = DEFAULT_MAX_ATTEMPTS

    # Threshold
    income_threshold: int = DEFAULT_INCOME_THRESHOLD
    income_greater_than_threshold: bool = True

    # Apply person weighting
    weighted: bool = DEFAULT_APPLY_PERSON_WEIGHTING

    # Logging / output
    summary_results_file: str = DEFAULT_SUMMARY_RESULTS_FILE
    llm_prior_results_file: str = DEFAULT_LLM_PRIOR_RESULTS_FILE
    direct_results_file: str = DEFAULT_DIRECT_RESULTS_FILE
    decomposition_results_file: str = DEFAULT_DECOMPOSITION_RESULTS_FILE
    log_csv: str = DEFAULT_LOG_CSV
