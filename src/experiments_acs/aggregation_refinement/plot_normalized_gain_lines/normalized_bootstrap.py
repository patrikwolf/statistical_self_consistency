import numpy as np


def bootstrap_normalized_errors(refinement_results: dict, n_boot=1_000, ci=0.90) -> tuple[list, list, list, list]:
    rng = np.random.default_rng()
    levels = list(refinement_results.keys())
    root_level = levels[0]
    boot_values = {level: [] for level in levels}

    # Convert error lists once
    errors = {
        level: np.asarray(refinement_results[level]["aggregated_error_list"])
        for level in levels
    }

    # Iterate over bootstrap runs
    n = len(errors[root_level])
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        e0 = errors[root_level][idx].mean()

        # Compute relative error for all levels
        for level in levels:
            el = errors[level][idx].mean()
            relative_error = el / e0
            boot_values[level].append(relative_error)

    alpha = (1 - ci) / 2
    means, lowers, uppers = [], [], []

    for level in levels:
        vals = np.asarray(boot_values[level])
        means.append(vals.mean())
        lowers.append(np.quantile(vals, alpha))
        uppers.append(np.quantile(vals, 1 - alpha))

    return levels, means, lowers, uppers


def bootstrap_normalized_error_gains(refinement_results: dict, n_boot=1_000, ci=0.90) -> tuple[list, list, list, list]:
    rng = np.random.default_rng()
    levels = list(refinement_results.keys())
    root_level = levels[0]
    boot_values = {level: [] for level in levels}

    # Convert error lists once
    errors = {
        level: np.asarray(refinement_results[level]["aggregated_error_list"])
        for level in levels
    }

    # Iterate over bootstrap runs
    n = len(errors[root_level])
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        e0 = errors[root_level][idx].mean()

        # Compute relative error for all levels
        for level in levels:
            el = errors[level][idx].mean()
            # relative_error = (el - e0) / e0
            relative_error = (e0 - el) / e0
            boot_values[level].append(relative_error)

    alpha = (1 - ci) / 2
    means, lowers, uppers = [], [], []

    for level in levels:
        vals = np.asarray(boot_values[level])
        means.append(vals.mean())
        lowers.append(np.quantile(vals, alpha))
        uppers.append(np.quantile(vals, 1 - alpha))

    return levels, means, lowers, uppers
