# region imports
from numericalMethods import GPDF, Probability
from scipy.optimize import bisect


# endregion

# region function definitions
def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
    """
    Find root using the Secant method.
    :param f: Function for which the root is sought.
    :param x0, x1: Initial guesses.
    :param tol: Convergence tolerance.
    :param max_iter: Maximum iterations.
    :return: Approximate root of f.
    """
    for _ in range(max_iter):
        f_x0, f_x1 = f(x0), f(x1)
        if abs(f_x1) < tol:
            return x1
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        x0, x1 = x1, x2
    raise ValueError("Secant method did not converge")


def main():
    """
    Interactive program to compute probabilities and critical values using numerical integration and the Secant method.
    """
    mu = float(input("Enter mean (mu): "))
    sigma = float(input("Enter standard deviation (sigma): "))
    mode = input("Are you specifying c (Y) or P (N)? ").strip().lower()

    if mode == 'y':
        c = float(input("Enter c value: "))
        prob = Probability(mu - 5 * sigma, c, mu, sigma)
        print(f"Computed probability: {prob}")
    else:
        P_target = float(input("Enter desired probability: "))

        def objective(c):
            return Probability(mu - 5 * sigma, c, mu, sigma) - P_target

        c_solution = secant_method(objective, mu, mu + sigma)
        print(f"Computed critical value c: {c_solution}")


if __name__ == "__main__":
    main()
