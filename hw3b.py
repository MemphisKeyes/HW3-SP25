# region imports
from scipy.stats import t
import math


# endregion

# region function definitions
def compute_t_value(confidence_level, sample_size):
    """
    Computes the t-value for a given confidence level and sample size.
    :param confidence_level: Confidence level (e.g., 0.95 for 95% confidence)
    :param sample_size: Number of samples
    :return: t-value
    """
    dof = sample_size - 1  # Degrees of freedom
    alpha = 1 - confidence_level
    t_value = t.ppf(1 - alpha / 2, dof)  # Two-tailed test
    return t_value


def main():
    """
    Interactive program to compute t-values for confidence intervals.
    """
    confidence_level = float(input("Enter confidence level (e.g., 0.95 for 95%): "))
    sample_size = int(input("Enter sample size: "))

    t_value = compute_t_value(confidence_level, sample_size)
    print(f"Computed t-value: {t_value}")


if __name__ == "__main__":
    main()
