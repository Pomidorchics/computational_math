import math
import numpy
import sympy
from math import log, e, pi, sin, cos, exp
from sympy import log, pi, sin, cos, exp
from scipy.optimize import minimize_scalar
import pandas
from typing import Callable


def print_table_of_values(a: float, b: float, f: Callable[[float], float], nodes: int) -> None:
    """
    Generates and prints a table of values for a given function f over a specified interval

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function for which the table of values will be generated
    :param nodes: the number of nodes in the interval

    :return: None
    """
    x_values = numpy.linspace(a, b, nodes)
    y_values = [f(xi) for xi in x_values]

    data = {'x': x_values, 'f(x)': y_values}

    print(f"Таблица значений:\n"
          f"{pandas.DataFrame(data)}\n")

    return


def divided_differences(x: numpy.array, y: numpy.array, n: int) -> numpy.array:
    """
    Computes the divided differences table for a given data points

    :param x: a numpy array of x-coordinates
    :param y: a numpy array of y-coordinates corresponding to the x-coordinates
    :param n: the degree of the polynomial to be interpolated

    :return: a 2D numpy array the representing the divided differences table
    """
    table = numpy.zeros((n + 1, n + 1), dtype=float)

    for i in range(n + 1):
        table[i][0] = y[i]

    for j in range(1, n + 1):
        for i in range(n + 1 - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table


def nearest_nodes(a: float, b: float, x: float, nodes: int) -> numpy.array:
    """
    Finds and returns an array of nodes closest to a given point within an interval

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param x: the point for which to find the nearest nodes
    :param nodes: the number of nodes in the interval

    :return: a numpy array containing the nodes, sorted by their distance to x
    """
    x_values = numpy.linspace(a, b, nodes)

    differences_list = [(abs(x - x_values[i]), i) for i in range(nodes)]
    differences_list.sort()

    return [x_values[i[1]] for i in differences_list]


def lagrange_interpolation(a: float, b: float, f: Callable[[float], float], x: float, n: int, nodes: int) -> float:
    """
    Performs Lagrange interpolation of a function at a given point

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function
    :param n: the degree of interpolating polynomial
    :param nodes: the number of nodes to use for interpolation

    :return: the interpolated value of the function at x
    """
    x_values = nearest_nodes(a, b, x, nodes)
    y_values = [f(xi) for xi in x_values]

    result = 0

    for k in range(n + 1):
        f_xk = y_values[k]

        f_xk *= numpy.prod([(x - x_values[j]) / (x_values[k] - x_values[j])
                            for j in range(n + 1)
                            if j != k])

        result += f_xk

    return result


def newton_interpolation(a: float, b: float, f: Callable[[float], float], x: float, n: int, nodes: int) -> float:
    """
    Performs Newton's interpolation of a function at a given point

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function
    :param n: the degree of interpolating polynomial
    :param nodes: the number of nodes to use for interpolation

    :return: the interpolated value of the function at x
    """

    x_values = nearest_nodes(a, b, x, nodes)
    y_values = [f(xi) for xi in x_values]

    table = divided_differences(x_values, y_values, n)

    result = table[0][0]
    t = 1

    for i in range(n):
        t *= (x - x_values[i])
        result += table[0][i + 1] * t

    return result


def error_estimate(a: float, b: float, f: Callable[[float], float], x: float, n: int, nodes: int) -> tuple:
    """
    Estimates the error bounds of polynomial interpolation for a function f(x)
    using a polynomial of degree n.

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function
    :param n: the degree of interpolating polynomial
    :param nodes: the number of nodes to use for interpolation

    :return: a tuple containing the maximum and minimum estimated error values
    """
    x_sym = sympy.Symbol('x')
    f_sym = sympy.sympify(f(x_sym))
    dev_sym = sympy.diff(f_sym, x_sym, n + 1)
    dev = sympy.lambdify(x_sym, dev_sym, 'numpy')

    x_values = nearest_nodes(a, b, x, nodes)[:n + 1]

    dev_min = minimize_scalar(dev, bounds=(min(x_values), max(x_values)), method='bounded').fun
    dev_max = - minimize_scalar(lambda x: -dev(x), bounds=(min(x_values), max(x_values)), method='bounded').fun

    w = numpy.prod([x - xi for xi in x_values])

    r_max = max((dev_min * w / math.factorial(n + 1)),
                dev_max * w / math.factorial(n + 1))

    r_min = min(dev_min * w / math.factorial(n + 1),
                dev_max * w / math.factorial(n + 1))

    return r_max, r_min


def error(ln: float, f: Callable[[float], float], x: float) -> float:
    """
    Calculates the interpolation error at a specific point

    :param ln: the interpolated value of the function f at point x
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function

    :return: the interpolation error at point x
    """
    return f(x) - ln


a = 1.0
b = 1.5
n1 = 1
n2 = 2

nodes = 11
f = lambda x: (x - 1) ** 2 - exp(-x)
x = 1.07

L1 = lagrange_interpolation(a, b, f, x, n1, nodes)
L2 = lagrange_interpolation(a, b, f, x, n2, nodes)

N1 = newton_interpolation(a, b, f, x, n1, nodes)
N2 = newton_interpolation(a, b, f, x, n2, nodes)

r1_max, r1_min = error_estimate(a, b, f, x, n1, nodes)
r1 = error(L1, f, x)

r2_max, r2_min = error_estimate(a, b, f, x, n2, nodes)
r2 = error(L2, f, x)

print_table_of_values(a, b, f, nodes)
print("Результаты линейной интерполяции:\n")
print(f"Результат линейной интерполяции с помощью формулы Лагранжа в точке x*:\n"
      f"L1(x*) = {L1:.{15}f}\n")
print(f"Результат линейной интерполяции с помощью формулы Ньютона в точке x*:\n"
      f"N1(x*) = {N1:.{15}f}\n")
print(f"Истинное значение функции в точке x*:\n"
      f"f(x*) = {f(x):.{15}f}\n")
print(f"Минимальное и максимальное значения остаточного члена R1(x):\n"
      f"minR1(x) = {r1_min:.{15}f}\n"
      f"maxR1(x) = {r1_max:.{15}f}\n")
print(f"Значение R1(x*) = f(x*) - L1(x*):\n"
      f"R1(x*) = {r1:.{15}f}\n")
if r1_min < r1 < r1_max:
    print("Неравенство minR1(x) < R1(x*) < maxR1(x) выполняется\n")
else:
    print("Неравенство minR1(x) < R1(x*) < maxR1(x) не выполняется\n")
if abs(r1_min) <= 1e-4 and abs(r1_max) <= 1e-4:
    print("Линейная интерполяция допустима с погрешностью, не превосходящей 10^(-4)\n")
else:
    print("Линейная интерполяция не допустима с погрешностью, не превосходящей 10^(-4)\n")

print("Результаты квадратичной интерполяции:\n")
print(f"Результат квадратичной интерполяции с помощью формулы Лагранжа в точке x*:\n"
      f"L2(x*) = {L2:.{15}f}\n")
print(f"Результат квадратичной интерполяции с помощью формулы Ньютона в точке x*:\n"
      f"N2(x*) = {N2:.{15}f}\n")
print(f"Истинное значение функции в точке x*:\n"
      f"f(x*) = {f(x):.{15}f}\n")
print(f"Минимальное и максимальное значения остаточного члена R2(x):\n"
      f"minR2(x) = {r2_min:.{15}f}\n"
      f"maxR2(x) = {r2_max:.{15}f}\n")
print(f"Значение R2(x*) = f(x*) - L2(x*):\n"
      f"R2(x*) = {r2:.{15}f}\n")
if r2_min < r2:
    print("Неравенство minR2(x) < R2(x*) < maxR2(x) выполняется\n")
else:
    print("Неравенство minR2(x) < R2(x*) < maxR2(x) не выполняется\n")
if abs(r2_min) <= 1e-4 and abs(r2_max) <= 1e-5:
    print("Квадратичная интерполяция допустима с погрешностью, не превосходящей 10^(-5)\n")
else:
    print("Квадратичная интерполяция не допустима с погрешностью, не превосходящей 10^(-5)\n")
