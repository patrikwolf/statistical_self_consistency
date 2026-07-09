import numpy as np

from scipy.stats import wasserstein_distance


def evaluate_wasserstein_distance(ground_truth_distribution: list[float],
                                  llm_estimate: list[float]
                                  ) -> float:
    raise ValueError("Need to normalize Wasserstein distance!")
    # Positions of the ordered categories
    x = np.arange(len(ground_truth_distribution))

    # Wasserstein distance
    w1 = wasserstein_distance(x, x, u_weights=ground_truth_distribution, v_weights=llm_estimate)

    return w1


if __name__ == "__main__":
    direct_estimates = [0.71, 0.22, 0.055, 0.015]
    aggregated_estimates = [0.6825, 0.24, 0.0575, 0.02]

    w1 = evaluate_wasserstein_distance(direct_estimates, aggregated_estimates)
    print(f"Wasserstein distance: {w1}")
