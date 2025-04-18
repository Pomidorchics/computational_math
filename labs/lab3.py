import math
import numpy as np
import sympy as sp
from math import log, e, pi, sin, cos, exp
from sympy import log, pi, sin, cos, exp
from scipy.optimize import minimize_scalar
from typing import Callable


def lagrange_interpolation(a: float, b: float, f_expr: sp.Expr, n: int, nodes: int) -> sp.Expr:
    """
    Construct the Lagrange interpolation polynomial for a given function over an interval.

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f_expr: the function to be interpolated
    :param n: the degree of interpolating polynomial
    :param nodes: the number of nodes to use for interpolation

    :return: a SymPy expression representing the Lagrange interpolation polynomial
    """
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr, 'numpy')

    x_values = np.linspace(a, b, nodes)
    y_values = [f(xi) for xi in x_values]

    L = 0

    for k in range(n + 1):
        term = y_values[k]

        term *= np.prod([(x - x_values[j]) / (x_values[k] - x_values[j])
                         for j in range(n + 1)
                         if j != k])

        L += term

    L = sp.expand(L)

    return L


def derivative(f_expr: sp.Expr, order: int) -> sp.Expr:
    """
    Calculates the derivative of a given function

    :param f_expr: the function to be differentiated
    :param order: the order of derivative to compute

    :return: a SymPy expression representing the derivative
    """

    x_sym = sp.Symbol('x')

    return sp.diff(f_expr, x_sym, order)


def error_estimate(a: float, b: float, f_expr: sp.Expr, x_val: float, n: int, nodes: int, k: int) -> tuple:
    """
    Estimates error bounds for numerical differentiation for a function f(x)
    using a polynomial of degree n.

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f_expr: the function to be differentiated
    :param x_val: the point at which to evaluate the derivative error
    :param n: the degree of interpolating polynomial
    :param nodes: the number of nodes to use for interpolation

    :return: a tuple containing the maximum and minimum estimated error values
    """
    x_sym = sp.Symbol('x')

    x_values = np.linspace(a, b, nodes)[:n + 1]

    w_expr = sp.prod([x_sym - xi for xi in x_values])

    r_min = 0
    r_max = 0

    for m in range(k + 1):
        order = n + m + 1
        dev_expr = derivative(f_expr, order)
        dev = sp.lambdify(x, dev_expr, 'numpy')

        w_dev_expr = derivative(w_expr, k - m)
        w_dev = sp.lambdify(x, w_dev_expr, 'numpy')
        w_dev_val = w_dev(x_val)

        dev_min = minimize_scalar(dev, bounds=(min(x_values), max(x_values)), method='bounded').fun
        dev_max = - minimize_scalar(lambda x: -dev(x), bounds=(min(x_values), max(x_values)), method='bounded').fun

        coeff = math.factorial(k) / (math.factorial(m) * math.factorial(k - m) * math.factorial(n + m + 1))

        r_max += max(dev_min * w_dev_val * coeff, dev_max * w_dev_val * coeff)
        r_min += min(dev_min * w_dev_val * coeff, dev_max * w_dev_val * coeff)

    return r_max, r_min


def error(Ln_expr: sp.Expr, f_expr: sp.Expr, x_val: float) -> float:
    """
    Calculates the interpolation error at a specific point

    :param Ln_expr: the interpolated value of the function f at point x
    :param f_expr: the function to be interpolated
    :param x_val: the point at which to interpolate the function

    :return: the interpolation error at point x
    """
    x = sp.Symbol('x')
    f = sp.lambdify(x, f_expr, 'numpy')
    Ln = sp.lambdify(x, Ln_expr, 'numpy')

    return f(x_val) - Ln(x_val)


a = 1
b = 1.5
n = 5
m = 4
k = 2
nodes = 11
x = sp.Symbol('x')

f_expr = (x - 1) ** 2 - exp(-x)
x_values = np.linspace(a, b, nodes)

x_val = x_values[m]

Ln_expr = lagrange_interpolation(a, b, f_expr, n, nodes)

diff_Ln_expr = derivative(Ln_expr, k)
diff_f_expr = derivative(f_expr, k)

diff_f = sp.lambdify(x, diff_f_expr, 'numpy')
diff_Ln = sp.lambdify(x, diff_Ln_expr, 'numpy')

r_max, r_min = error_estimate(a, b, f_expr, x_val, n, nodes, k)
r = error(diff_Ln_expr, diff_f_expr, x_val)

print(f"Точка x{m}: {x_val}\n")

print(f"Функция f(x): \n {f_expr}")
print(f"Полином Лагранжа степени {n}: \n {Ln_expr}\n")

print(f"Производная {k}-го порядка функции f(x): \n {diff_f_expr}")
print(f"Производная {k}-го порядка полинома Лагранжа степени {n}: \n {diff_Ln_expr}\n")

print(f"Значение производной {k}-го порядка функции f(x) в точке x{m}: \n {diff_f(x_val):.{15}}")
print(f"Значение производной {k}-го порядка полинома Лагранжа степени {n} в точке x{m}: \n {diff_Ln(x_val):.{15}}\n")

print(f"Минимальное и максимальное значения остаточного члена R(x):\n"
      f"minR(x) = {r_min:.{15}f}\n"
      f"maxR(x) = {r_max:.{15}f}\n")
print(f"Значение R(x{m}) = f(x{m}) - L{n}(x{m}):\n"
      f"R(x{m}) = {r:.{15}f}\n")
if r_min < r < r_max:
    print(f"Неравенство minR(x) < R(x{m}) < maxR(x) выполняется\n")
else:
    print(f"Неравенство minR(x) < R(x{m}) < maxR(x) не выполняется\n")


