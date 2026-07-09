import numpy as np


def bootstrap_confidence_interval(values: list, n_boot=1_000, ci=0.90):
    values = np.asarray(values)
    mean = np.mean(values)

    n = len(values)
    boot_means = np.empty(n_boot)

    for b in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return mean, lower, upper
