import numpy as np

from scipy.stats import wasserstein_distance


def compute_wasserstein_distance(d1: list[float], d2: list[float]) -> float:
    # Positions of the ordered categories
    x = np.arange(len(d1))

    # Wasserstein distance
    w1 = wasserstein_distance(x, x, u_weights=d1, v_weights=d2)

    return w1


def _evaluate_normalized_wasserstein_distance(direct_estimate: list[float], aggregated_estimate: list[float]) -> float:
    """Compute normalized Wasserstein-1 distance on ordered categories.

    The raw W1 distance on K ordered categories is in [0, K - 1].
    We divide by K - 1 so that the returned value lies in [0, 1].
    """
    if len(direct_estimate) != len(aggregated_estimate):
        raise ValueError("Distributions must have the same number of categories.")

    # Number and positions of the ordered categories
    num_categories = len(direct_estimate)
    x = np.arange(num_categories)

    # Raw Wasserstein distance
    w1 = wasserstein_distance(x, x, u_weights=direct_estimate, v_weights=aggregated_estimate)

    # Normalize by the diameter of the ordered support
    normalized_w1 = w1 / (num_categories - 1)

    return float(normalized_w1)


if __name__ == "__main__":
    d1 = [
        0.6565656565656566,
        0.32323232323232326,
        0.020202020202020204
    ]
    d2 = [
        0.7535000000000001,
        0.133,
        0.1135
    ]

    wd = compute_wasserstein_distance(d1=d1, d2=d2)
    print(f"WD: {wd:.3f}")

    direct_estimates = [0.71, 0.22, 0.055, 0.015]
    aggregated_estimates = [0.6825, 0.24, 0.0575, 0.02]

    w1 = _evaluate_normalized_wasserstein_distance(direct_estimates, aggregated_estimates)
    print(f"Normalized Wasserstein distance: {w1}")
