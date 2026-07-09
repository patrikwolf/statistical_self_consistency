
def weighted_avg(values, priors):
    """Compute the weighted average of a list of values."""
    assert abs(sum(priors) - 1) < 1e-6, f"Sum of priors is {sum(priors)}, but should be 1"
    return sum(prior * v for prior, v in zip(priors, values))


if __name__ == "__main__":
    # Example usage
    values = [0.2, 0.5, 0.3]
    priors = [0.5, 0.3, 0.2]
    avg = weighted_avg(values, priors)
    print(f"Weighted average: {avg:.4f}")
