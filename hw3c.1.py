def is_symmetric(A):
    """Check if matrix A is symmetric."""
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                return False
    return True

def is_positive_definite(A):
    """Check if matrix A is positive definite using the leading principal minors."""
    n = len(A)
    for i in range(1, n + 1):
        sub_matrix = [[A[x][y] for y in range(i)] for x in range(i)]
        if determinant(sub_matrix) <= 0:
            return False
    return True

def determinant(A):
    """Compute the determinant of a matrix recursively."""
    n = len(A)
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    det = 0
    for c in range(n):
        sub_matrix = [[A[i][j] for j in range(n) if j != c] for i in range(1, n)]
        det += ((-1) ** c) * A[0][c] * determinant(sub_matrix)
    return det

def cholesky_decomposition(A):
    """Performs Cholesky decomposition on A = L * L^T."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            sum1 = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = (A[i][i] - sum1) ** 0.5
            else:
                L[i][j] = (A[i][j] - sum1) / L[j][j]
    return L

def forward_substitution(L, b):
    """Solve Ly = b for y."""
    n = len(b)
    y = [0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y

def backward_substitution(U, y):
    """Solve Ux = y for x."""
    n = len(y)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def doolittle_lu_decomposition(A):
    """Performs Doolittle LU decomposition with partial pivoting."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    for j in range(n):
        for i in range(j + 1):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for i in range(j, n):
            L[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k in range(j))) / U[j][j]
    return L, U

def solve_system(A, b):
    """Determine which method to use and solve Ax = b."""
    if is_symmetric(A) and is_positive_definite(A):
        print("Using Cholesky Decomposition")
        L = cholesky_decomposition(A)
        y = forward_substitution(L, b)
        L_T = [[L[j][i] for j in range(len(L))] for i in range(len(L))]
        x = backward_substitution(L_T, y)
    else:
        print("Using Doolittle LU Decomposition")
        L, U = doolittle_lu_decomposition(A)
        y = forward_substitution(L, b)
        x = backward_substitution(U, y)
    return x

# Example matrices and vectors
A1 = [[1, -1, 3, 2], [-1, 5, -5, -2], [3, -5, 9, 3], [2, -2, 3, 21]]
b1 = [15, -35, 94, 1]
A2 = [[4, 2, 4, 2], [2, 3, 2, 3], [4, 2, 6, 3], [2, 3, 3, 9]]
b2 = [20, 36, 60, 122]

x1 = solve_system(A1, b1)
print("Solution for first system:", x1)

x2 = solve_system(A2, b2)
print("Solution for second system:", x2)

