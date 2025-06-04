import numpy as np


def monotone_sweep(A: np.ndarray, B: np.ndarray, C: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Solves tridiagonal system using monotone sweep method

    :param A: Subdiagonal coefficients array (A[0] unused)
    :param B: Main diagonal coefficients array
    :param C: Superdiagonal coefficients array (C[-1] unused)
    :param F: Right-hand side vector
    :return: Solution vector U
    """
    N = len(B)
    alpha = np.zeros(N)
    beta = np.zeros(N)
    U = np.zeros(N)

    alpha[1] = -C[0] / B[0]
    beta[1] = F[0] / B[0]

    for k in range(1, N - 1):
        denominator = B[k] + A[k] * alpha[k]
        alpha[k + 1] = -C[k] / denominator
        beta[k + 1] = (F[k] - A[k] * beta[k]) / denominator

    U[-1] = (F[-1] - A[-1] * beta[-1]) / (A[-1] * alpha[-1] + B[-1])

    for k in range(N - 2, -1, -1):
        U[k] = alpha[k + 1] * U[k + 1] + beta[k + 1]

    return U


def check_solution(A: np.ndarray, B: np.ndarray, C: np.ndarray, F: np.ndarray,
                   U: np.ndarray, max_error: float = 1e-10) -> bool:
    """
    Verifies that U satisfies the tridiagonal system within specified tolerance

    :param A: Subdiagonal coefficients array (A[0] unused)
    :param B: Main diagonal coefficients array
    :param C: Superdiagonal coefficients array (C[-1] unused)
    :param F: Right-hand side vector
    :param U: Proposed solution vector
    :param max_error: Maximum allowed residual norm
    :return: True if solution is valid within tolerance, False otherwise
    """
    N = len(B)
    residual = np.zeros(N)

    residual[0] = B[0] * U[0] + C[0] * U[1] - F[0]

    for k in range(1, N - 1):
        residual[k] = A[k] * U[k - 1] + B[k] * U[k] + C[k] * U[k + 1] - F[k]

    residual[-1] = A[-1] * U[-2] + B[-1] * U[-1] - F[-1]

    max_residual = np.max(np.abs(residual))
    print(f"Максимальная разность: {max_residual}")
    return max_residual < max_error


def test_system(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates random tridiagonal system with diagonal dominance

    :param N: System size
    :return: Tuple of (A, B, C, F) where:
        A - Subdiagonal (A[0] = 0)
        B - Main diagonal (with +2 for dominance)
        C - Superdiagonal (C[-1] = 0)
        F - Right-hand side vector
    """
    A = np.random.rand(N)
    B = np.random.rand(N) + 2
    C = np.random.rand(N)
    F = np.random.rand(N)

    A[0] = 0
    C[-1] = 0

    return A, B, C, F


# test 1
A, B, C, F = test_system(5)

print("\nТест 1")
print("\nКоэффициенты системы:")
print("A (поддиагональ):", A)
print("B (главная диагональ):", B)
print("C (наддиагональ):", C)
print("F (правая часть):", F)
print()

U_our = monotone_sweep(A, B, C, F)
print("Решение методом монотонной прогонки:")
print(U_our)
print()

is_correct = check_solution(A, B, C, F, U_our)
print(f"Решение {'верно' if is_correct else 'неверно'}")

# test 2
A, B, C, F = test_system(7)

print("\nТест 2")
print("\nКоэффициенты системы:")
print("A (поддиагональ):", A)
print("B (главная диагональ):", B)
print("C (наддиагональ):", C)
print("F (правая часть):", F)

U_our = monotone_sweep(A, B, C, F)
print("\nРешение методом монотонной прогонки:")
print(U_our)
print()

is_correct = check_solution(A, B, C, F, U_our)
print(f"Решение {'верно' if is_correct else 'неверно'}")

# test 3
A, B, C, F = test_system(15)

print("\nТест 3")
print("\nКоэффициенты системы:")
print("A (поддиагональ):", A)
print("B (главная диагональ):", B)
print("C (наддиагональ):", C)
print("F (правая часть):", F)
print()

U_our = monotone_sweep(A, B, C, F)
print("\nРешение методом монотонной прогонки:")
print(U_our)
print()

is_correct = check_solution(A, B, C, F, U_our)
print(f"Решение {'верно' if is_correct else 'неверно'}")
