import numpy as np
import matrixOperations as mo
import DoolittleMethod as dm


def is_symmetric(A):
    return np.allclose(A, np.transpose(A))


def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def cholesky_factorization(A):
    n = len(A)
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - sum_k)
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]
    return L


def cholesky_solve(A, b):
    L = cholesky_factorization(A)
    Lt = np.transpose(L)

    # Forward substitution: Solve Ly = b
    y = np.linalg.solve(L, b)

    # Back substitution: Solve L^T x = y
    x = np.linalg.solve(Lt, y)
    return x


def solve_system(Aaug):
    A, b = mo.separateAugmented(Aaug)
    if is_symmetric(A) and is_positive_definite(A):
        print("Using Cholesky Method")
        x = cholesky_solve(A, b)
    else:
        print("Using Doolittle Method")
        x = dm.Doolittle(Aaug)
    return x


def main():
    test_matrices = [
        [[4, 12, -16, 5], [12, 37, -43, 10], [-16, -43, 98, 20]],
        [[3, 2, -1, 1], [2, -4, 3, -2], [-1, 3, 6, 3]]
    ]

    for i, Aaug in enumerate(test_matrices, 1):
        print(f"Solving system {i}:")
        x = solve_system(Aaug)
        print("Solution:", [round(xi, 4) for xi in x])
        print()


if __name__ == "__main__":
    main()
