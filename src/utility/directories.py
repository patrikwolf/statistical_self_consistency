from datetime import datetime
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
SRC_DIR = MODULE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


def _ensure_dir(path: Path, create_dir: bool) -> Path:
    if create_dir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _build_results_dir(
    experiment_name: str,
    cluster: bool,
    use_timestamp: bool,
    timestamp: str | None,
) -> Path:
    results_root = PROJECT_ROOT / "results_cluster" if cluster else PROJECT_ROOT / "results"
    output_dir = results_root / experiment_name

    if not use_timestamp:
        return output_dir

    timestamp_str = timestamp or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    return output_dir / timestamp_str


def get_results_dir(
    experiment_name: str,
    cluster: bool,
    use_timestamp: bool,
    timestamp: str | None = None,
    create_dir: bool = True,
) -> Path:
    """
    Return the results directory for an experiment.

    When ``use_timestamp`` is enabled, the returned path includes either the
    provided timestamp or a new timestamp generated from the current time.
    """
    results_dir = _build_results_dir(
        experiment_name=experiment_name,
        cluster=cluster,
        use_timestamp=use_timestamp,
        timestamp=timestamp,
    )
    return _ensure_dir(path=results_dir, create_dir=create_dir)


def get_base_dir() -> Path:
    """Return the base directory"""
    return PROJECT_ROOT


def get_plot_dir(create_dir: bool = True) -> Path:
    """Return the plots directory, creating it when requested."""
    return _ensure_dir(path=PROJECT_ROOT / "plots", create_dir=create_dir)


def get_acs_data_dir(create_dir: bool = True) -> Path:
    """Return the data directory for ACS income data."""
    return _ensure_dir(path=PROJECT_ROOT / "data" / "acs", create_dir=create_dir)


def get_wvs_data_dir() -> Path:
    """Return the data directory for WVS Wave 7 data."""
    return _ensure_dir(path=PROJECT_ROOT / "data" / "wvs_wave_7", create_dir=False)


def get_data_hf_dataset_dir() -> Path:
    """Return the data directory for local copy of HF dataset."""
    return _ensure_dir(path=PROJECT_ROOT / "data" / "hf_dataset", create_dir=False)


def get_log_dir(create_dir: bool = True, cluster: bool = True) -> Path:
    """Return the logs directory, creating it when requested."""
    logs_name = "logs" if not cluster else "logs_cluster"
    return _ensure_dir(path=PROJECT_ROOT / logs_name, create_dir=create_dir)


def get_prompt_template_dir() -> Path:
    """Return the prompt template directory for ACS income data."""
    return _ensure_dir(path=SRC_DIR / "prompt_templates", create_dir=False)


def get_latex_template_dir() -> Path:
    """Return the LaTeX template directory for ACS income data."""
    return _ensure_dir(path=SRC_DIR / "latex_templates", create_dir=False)


if __name__ == "__main__":
    results_dir = get_results_dir(experiment_name="test", cluster=False, use_timestamp=False)
    print(results_dir)

    prompt_template_dir = get_prompt_template_dir()
    print(prompt_template_dir)
