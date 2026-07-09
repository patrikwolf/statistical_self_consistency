"""Constants used across the package."""

# LLM generation parameters
NUM_TOP_LOGPROBS = 5
MAX_NEW_TOKENS = 2000

# Default income threshold
DEFAULT_INCOME_THRESHOLD = 40_000

# Default model configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"

# Default person weighting application
DEFAULT_APPLY_PERSON_WEIGHTING = True

# Experiment defaults
DEFAULT_NUM_SAMPLES = 50
DEFAULT_MAX_ATTEMPTS = 10

# Default decomposition attributes
DEFAULT_DECOMPOSITION_ATTRIBUTES = ["ESR"]

# Logging filenames
DEFAULT_SUMMARY_RESULTS_FILE = "summary_results.json"
DEFAULT_LLM_PRIOR_RESULTS_FILE = "llm_prior_results.json"
DEFAULT_DIRECT_RESULTS_FILE = "llm_direct_prompting.json"
DEFAULT_MICRO_MACRO_RESULTS_FILE = "llm_micro_macro_prompting.json"
DEFAULT_DECOMPOSITION_RESULTS_FILE = "llm_law_of_total_prob.json"
DEFAULT_LOG_CSV = "sociodemographic_statistics.csv"
