import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify


class SplineInterpolator:
    def __init__(self, x: np.ndarray, y: np.ndarray, method: str = 'slopes', boundary_type: int = 1, boundary_values: dict = None) -> None:
        """
        Initializes the spline interpolator with given data and parameters
        :param x: Array of x-coordinates
        :param y: Array of y-coordinates
        :param method: Interpolation method ('slopes' or 'moments')
        :param boundary_type: Type of boundary conditions (1-4)
        :param boundary_values: Dictionary of boundary values
        """
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.n = len(x) - 1
        self.h = np.diff(x)
        self.method = method
        self.boundary_type = boundary_type
        self.boundary_values = boundary_values or {}
        self.spline = None
        self.m = None
        self.M = None

        if method == 'slopes':
            self._build_spline_via_slopes()
        elif method == 'moments':
            self._build_spline_via_moments()

    def _build_spline_via_slopes(self) -> None:
        """
        Constructs the spline using slopes (first derivatives) as parameters
        """
        A = np.zeros((self.n + 1, self.n + 1))
        b = np.zeros(self.n + 1)

        for i in range(1, self.n):
            mu_i = self.h[i] / (self.h[i - 1] + self.h[i])
            lambda_i = self.h[i - 1] / (self.h[i - 1] + self.h[i])

            A[i, i - 1] = mu_i
            A[i, i] = 2
            A[i, i + 1] = lambda_i

            b[i] = 3 * (lambda_i * (self.y[i + 1] - self.y[i]) / self.h[i] +
                        mu_i * (self.y[i] - self.y[i - 1]) / self.h[i - 1])

        if self.boundary_type == 1:  # S'(a) = f'(a), S'(b) = f'(b)
            A[0, 0] = 1
            b[0] = self.boundary_values.get('d1_a', 0)

            A[self.n, self.n] = 1
            b[self.n] = self.boundary_values.get('d1_b', 0)

        elif self.boundary_type == 2:  # S''(a) = f''(a), S''(b) = f''(b)
            A[0, 0] = 2
            A[0, 1] = 1
            b[0] = (3 * (self.y[1] - self.y[0]) / self.h[0] -
                    self.h[0] * self.boundary_values.get('d2_a', 0) / 2)

            A[self.n, self.n - 1] = 1
            A[self.n, self.n] = 2
            b[self.n] = (self.h[-1] * self.boundary_values.get('d2_b', 0) / 2 +
                         3 * (self.y[-1] - self.y[-2]) / self.h[-1])

        elif self.boundary_type == 4:  # S'''(a+) = S'''(a-), S'''(b-) = S'''(b+)
            gamma_1 = self.h[0] / self.h[1]
            A[0, 0] = 1
            A[0, 1] = 1 - gamma_1 ** 2
            A[0, 2] = -gamma_1 ** 2
            b[0] = 2 * ((self.y[1] - self.y[0]) / self.h[0] -
                        gamma_1 ** 2 * (self.y[2] - self.y[1]) / self.h[1])

            gamma_n = self.h[-1] / self.h[-2]
            A[self.n, self.n - 2] = gamma_n ** 2
            A[self.n, self.n - 1] = -(1 - gamma_n ** 2)
            A[self.n, self.n] = -1
            b[self.n] = 2 * (gamma_n ** 2 * (self.y[-2] - self.y[-3]) / self.h[-2] -
                             (self.y[-1] - self.y[-2]) / self.h[-1])

        self.m = np.linalg.solve(A, b)

        self.spline = []
        for i in range(self.n):
            a_i = (6 / self.h[i]) * ((self.y[i + 1] - self.y[i]) / self.h[i] -
                                     (2 * self.m[i] + self.m[i + 1]) / 3)
            b_i = (12 / self.h[i] ** 2) * ((self.m[i] + self.m[i + 1]) / 2 -
                                           (self.y[i + 1] - self.y[i]) / self.h[i])

            def poly(x_val: float, i: int, a_i: float = a_i, b_i: float = b_i) -> float:
                """
                Local cubic polynomial for the i-th segment (slopes method)
                :param x_val: Input x-value
                :param i: Segment index
                :param a_i: Precomputed coefficient
                :param b_i: Precomputed coefficient
                :return: Interpolated y-value
                """
                return (self.y[i] + self.m[i] * (x_val - self.x[i]) +
                        a_i * (x_val - self.x[i]) ** 2 / 2 +
                        b_i * (x_val - self.x[i]) ** 3 / 6)

            self.spline.append(poly)

    def _build_spline_via_moments(self) -> None:
        """
        Constructs the spline using moments (second derivatives) as parameters
        """
        A = np.zeros((self.n + 1, self.n + 1))
        b = np.zeros(self.n + 1)

        for i in range(1, self.n):
            lambda_i = self.h[i - 1] / (self.h[i - 1] + self.h[i])
            mu_i = self.h[i] / (self.h[i - 1] + self.h[i])

            A[i, i - 1] = lambda_i
            A[i, i] = 2
            A[i, i + 1] = mu_i

            b[i] = (6 / (self.h[i - 1] + self.h[i]) *
                    ((self.y[i + 1] - self.y[i]) / self.h[i] -
                     (self.y[i] - self.y[i - 1]) / self.h[i - 1]))

        if self.boundary_type == 1:  # S'(a) = f'(a), S'(b) = f'(b)
            A[0, 0] = 2
            A[0, 1] = 1
            b[0] = (6 / self.h[0] *
                    ((self.y[1] - self.y[0]) / self.h[0] -
                     self.boundary_values.get('d1_a', 0)))

            A[self.n, self.n - 1] = 1
            A[self.n, self.n] = 2
            b[self.n] = (6 / self.h[-1] *
                         (self.boundary_values.get('d1_b', 0) -
                          (self.y[-1] - self.y[-2]) / self.h[-1]))

        elif self.boundary_type == 2:  # S''(a) = f''(a), S''(b) = f''(b)
            A[0, 0] = 1
            b[0] = self.boundary_values.get('d2_a', 0)

            A[self.n, self.n] = 1
            b[self.n] = self.boundary_values.get('d2_b', 0)

        elif self.boundary_type == 4:  # S'''(a+) = S'''(a-), S'''(b-) = S'''(b+)
            gamma_1 = self.h[0] / self.h[1]
            A[0, 0] = 1
            A[0, 1] = -(1 + gamma_1)
            A[0, 2] = gamma_1
            b[0] = 0

            gamma_n = self.h[-1] / self.h[-2]
            A[self.n, self.n - 2] = gamma_n
            A[self.n, self.n - 1] = -(1 + gamma_n)
            A[self.n, self.n] = 1
            b[self.n] = 0

        self.M = np.linalg.solve(A, b)

        self.spline = []
        for i in range(self.n):
            b_i = (self.M[i + 1] - self.M[i]) / self.h[i]
            c_i = ((self.y[i + 1] - self.y[i]) / self.h[i] -
                   self.h[i] * (2 * self.M[i] + self.M[i + 1]) / 6)

            def poly(x_val: float, i: int, c_i: float = c_i, b_i: float = b_i) -> float:
                """
                Local cubic polynomial for the i-th segment (moments method)
                :param x_val: Input x-value
                :param i: Segment index
                :param c_i: Precomputed coefficient
                :param b_i: Precomputed coefficient
                :return: Interpolated y-value
                """
                return (self.y[i] + c_i * (x_val - self.x[i]) +
                        self.M[i] * (x_val - self.x[i]) ** 2 / 2 +
                        b_i * (x_val - self.x[i]) ** 3 / 6)

            self.spline.append(poly)

    def evaluate(self, x_vals: np.ndarray) -> np.ndarray:
        """
        Evaluates the spline at given x-values
        :param x_vals: Array of x-coordinates to evaluate
        :return: Array of interpolated y-values
        """
        x_vals = np.asarray(x_vals)
        results = np.zeros_like(x_vals)

        for i, x_val in enumerate(x_vals):
            idx = np.searchsorted(self.x, x_val) - 1
            idx = max(0, min(idx, self.n - 1))
            results[i] = self.spline[idx](x_val)

        return results

    def derivative(self, x_val: float, order: int = 1) -> float:
        """
        Calculates the derivative of the spline at a given point
        :param x_val: Point where the derivative is evaluated
        :param order: Order of derivative (1 or 2)
        :return: Derivative value as a float
        """
        idx = np.searchsorted(self.x, x_val) - 1
        idx = max(0, min(idx, self.n - 1))

        if self.method == 'slopes':
            a_i = (6 / self.h[idx]) * ((self.y[idx + 1] - self.y[idx]) / self.h[idx] -
                                       (2 * self.m[idx] + self.m[idx + 1]) / 3)
            b_i = (6 / self.h[idx] ** 2) * ((self.m[idx] + self.m[idx + 1]) / 2 -
                                             (self.y[idx + 1] - self.y[idx]) / self.h[idx])

            if order == 1:
                return self.m[idx] + a_i * (x_val - self.x[idx]) + b_i * (x_val - self.x[idx]) ** 2
            elif order == 2:
                return a_i + b_i * (x_val - self.x[idx]) * 2

        elif self.method == 'moments':
            b_i = (self.M[idx + 1] - self.M[idx]) / self.h[idx]
            c_i = ((self.y[idx + 1] - self.y[idx]) / self.h[idx] -
                   self.h[idx] * (2 * self.M[idx] + self.M[idx + 1]) / 6)

            if order == 1:
                return c_i + self.M[idx] * (x_val - self.x[idx]) + b_i * (x_val - self.x[idx]) ** 2 / 2
            elif order == 2:
                return self.M[idx] + b_i * (x_val - self.x[idx])

    def integrate(self, a: float = None, b: float = None) -> float:
        """
        Computes the definite integral of the spline over [a, b]
        :param a: Lower bound
        :param b: Upper bound
        :return: Integral value as a float
        """
        if a is None:
            a = self.x[0]
        if b is None:
            b = self.x[-1]

        total = 0.0

        start_idx = np.searchsorted(self.x, a) - 1
        start_idx = max(0, min(start_idx, self.n - 1))

        end_idx = np.searchsorted(self.x, b) - 1
        end_idx = max(0, min(end_idx, self.n - 1))

        for i in range(start_idx, end_idx + 1):

            if self.method == 'slopes':
                integral = (self.h[i] * (self.y[i + 1] + self.y[i]) / 2 +
                            self.h[i] ** 2 * (self.m[i] - self.m[i + 1]) / 12)

            elif self.method == 'moments':
                integral = (self.h[i] * (self.y[i] + self.y[i + 1]) / 2 -
                            self.h[i] ** 3 * (self.M[i] + self.M[i + 1]) / 24)

            total += integral

        return total


def plot_spline(spline: SplineInterpolator, x_vals: np.ndarray, y_vals: np.ndarray) -> None:
    """
    Plots the spline interpolation along with original data points
    :param spline: Instance of SplineInterpolator containing the spline to plot
    :param x_vals: Array of x-coordinates of the original data points
    :param y_vals: Array of y-coordinates of the original data points
    """
    plt.figure(figsize=(10, 6))
    x_fine = np.linspace(min(x_vals), max(x_vals), 500)
    y_fine = spline.evaluate(x_fine)

    plt.plot(x_fine, y_fine, label='Spline')
    plt.scatter(x_vals, y_vals, color='red', label='Data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


x = sp.symbols('x')
f = sp.log(x) + x
f_prime = sp.diff(f, x)
f_double_prime = sp.diff(f_prime, x)

f_num = lambdify(x, f, 'numpy')
f_prime_num = lambdify(x, f_prime, 'numpy')
f_double_prime_num = lambdify(x, f_double_prime, 'numpy')

a, b = 0.5, 5.5
n_points = 5
x_vals = np.linspace(a, b, n_points)
y_vals = f_num(x_vals)

print("Выберите метод построения сплайна:")
print("1 - через наклоны")
print("2 - через моменты")
method = input("")

print("\nВыберите тип краевых условий:")
print("1 - S'(a)=f'(a), S'(b)=f'(b)")
print("2 - S''(a)=f''(a), S''(b)=f''(b)")
print("3 - периодические условия")
print("4 - S'''(a+)=S'''(a-), S'''(b-)=S'''(b+)")
boundary_type = int(input(""))

print("\nВыберите задачу:")
print("1 - интерполяция")
print("2 - интегрирование")
print("3 - первая производная")
print("4 - вторая производная")
task = input("")

boundary_values = {
    'd1_a': float(f_prime_num(a)),
    'd1_b': float(f_prime_num(b)),
    'd2_a': float(f_double_prime_num(a)),
    'd2_b': float(f_double_prime_num(b))
}

if method == '1':
    spline = SplineInterpolator(x_vals, y_vals, method='slopes',
                                boundary_type=boundary_type,
                                boundary_values=boundary_values)
else:
    spline = SplineInterpolator(x_vals, y_vals, method='moments',
                                boundary_type=boundary_type,
                                boundary_values=boundary_values)

if task == '2':
    integral = spline.integrate()
    true_integral = float(sp.integrate(f, (x, a, b)))
    print(f"\nВычисленный интеграл: {integral}")
    print(f"Точное значение интеграла: {true_integral}")
elif task == '3':
    point = float(input(f"\nВведите точку на отрезке [{a}, {b}] для вычисления производной: "))

    spline_deriv = spline.derivative(point, order=1)
    true_deriv = f_prime_num(point)

    print(f"\nПроизводная сплайна в точке {point}: {spline_deriv}")
    print(f"Точная производная функции в точке {point}: {true_deriv}")
elif task == '4':
    point = float(input(f"\nВведите точку на отрезке [{a}, {b}] для вычисления второй производной: "))

    spline_deriv = spline.derivative(point, order=2)
    true_deriv = f_double_prime_num(point)

    print(f"\nВторая производная сплайна в точке {point}: {spline_deriv}")
    print(f"Точная вторая производная функции в точке {point}: {true_deriv}")

plot_spline(spline, x_vals, y_vals)
