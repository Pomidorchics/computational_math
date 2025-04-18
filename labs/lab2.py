import math
import numpy as np
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
    x_values = np.linspace(a, b, nodes)
    y_values = [f(xi) for xi in x_values]

    data = {'x': x_values, 'f(x)': y_values}

    print(f"Таблица значений:\n"
          f"{pandas.DataFrame(data)}\n")

    return


def finite_differences(y: np.array) -> np.array:
    """
    Computes the finite difference table

    :param y: numpy array of y-coordinates

    :return: a 2D numpy array the representing the divided differences table
    """

    n = len(y)
    table = np.zeros((n, n), dtype=float)

    for i in range(n):
        table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    return table


def newton_first(a: float, b: float, f: Callable[[float], float], x: float, nodes: int) -> float:
    """
    Performs interpolation of a function at a given point using Newton's forward difference formula

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function
    :param nodes: the number of nodes to use for interpolation

    :return: the interpolated value of the function at x
    """
    x_values = np.linspace(a, b, nodes)
    y_values = [f(xi) for xi in x_values]

    table = finite_differences(y_values)

    n = nodes - 1
    h = (b - a) / n
    t = (x - x_values[0]) / h

    result = table[0][0]
    for i in range(1, len(x_values)):
        product = np.prod(np.array([t - k for k in range(i)]))

        result += (product / math.factorial(i)) * table[0][i]

    return result


def newton_second(a: float, b: float, f: Callable[[float], float], x: float, nodes: int) -> float:
    """
    Performs interpolation of a function at a given point using Newton's forward difference formula

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function
    :param nodes: the number of nodes to use for interpolation

    :return: the interpolated value of the function at x
    """
    x_values = np.linspace(a, b, nodes)
    y_values = [f(xi) for xi in x_values]

    nearest_ind = len(x_values) - 1

    table = finite_differences(y_values)

    n = nodes - 1
    h = (b - a) / n
    t = (x - x_values[nearest_ind]) / h

    result = table[nearest_ind][0]
    for i in range(1, len(x_values)):
        product = np.prod(np.array([t + k for k in range(i)]))

        result += (product / math.factorial(i)) * table[nearest_ind - i][i]

    return result


def gauss_first(a: float, b: float, f: Callable[[float], float], x: float, nodes: int) -> float:
    """
    Performs interpolation of a function at a given point using Gauss's forward difference formula

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function
    :param nodes: the number of nodes to use for interpolation

    :return: the interpolated value of the function at x
    """
    x_values = np.linspace(a, b, nodes)
    y_values = [f(xi) for xi in x_values]

    h = (b - a) / (nodes - 1)
    central_ind = nodes // 2
    t = (x - x_values[central_ind]) / h

    table = finite_differences(y_values)

    result = table[central_ind][0]

    for k in range(1, nodes):
        product = np.prod([t + (-1) ** (i-1) * (i // 2) for i in range(1, k + 1)])

        diff = table[central_ind - (k // 2)][k]

        result += product * diff / math.factorial(k)

    return result


def gauss_second(a: float, b: float, f: Callable[[float], float], x: float, nodes: int) -> float:
    """
    Performs interpolation of a function at a given point using Gauss's backward difference formula

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function
    :param nodes: the number of nodes to use for interpolation

    :return: the interpolated value of the function at x
    """
    x_values = np.linspace(a, b, nodes)
    y_values = [f(xi) for xi in x_values]

    h = (b - a) / (nodes - 1)
    central_ind = nodes // 2
    t = (x - x_values[central_ind]) / h

    table = finite_differences(y_values)

    result = table[central_ind][0]

    for k in range(1, nodes):
        product = np.prod([t + (-1) ** i * (i // 2) for i in range(1, k + 1)])

        diff = table[central_ind - (k+1) // 2][k]

        result += product * diff / math.factorial(k)

    return result


def stirling(a: float, b: float, f: Callable[[float], float], x: float, nodes: int) -> float:
    """
    Performs central interpolation using Stirling's difference formula

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function
    :param nodes: the number of nodes to use for interpolation

    :return: the interpolated value of the function at x
    """
    x_values = np.linspace(a, b, nodes)
    y_values = [f(xi) for xi in x_values]

    h = (b - a) / (nodes - 1)
    central_ind = nodes // 2
    t = (x - x_values[central_ind]) / h

    table = finite_differences(y_values)

    result = table[central_ind][0]

    for k in range(1, nodes):
        if k % 2 == 0:
            product = np.prod([t ** 2 - i ** 2 for i in range(0, k // 2)])
            diff = table[central_ind - (k // 2)][k]

        else:
            product = np.prod([t + (-1) ** i * (i // 2) for i in range(1, k + 1)])
            diff = (table[central_ind - (k // 2)][k] + table[central_ind - (k+1) // 2][k]) / 2

        result += product * diff / math.factorial(k)

    return result


def error_estimate(a: float, b: float, f: Callable[[float], float], x: float, nodes: int) -> tuple:
    """
    Estimates the error bounds of polynomial interpolation for a function f(x)
    using a polynomial of degree n.

    :param a: the left endpoint of the interval
    :param b: the right endpoint of the interval
    :param f: the function to be interpolated
    :param x: the point at which to interpolate the function
    :param nodes: the number of nodes to use for interpolation

    :return: a tuple containing the maximum and minimum estimated error values
    """
    x_sym = sympy.Symbol('x')
    f_sym = sympy.sympify(f(x_sym))
    dev_sym = sympy.diff(f_sym, x_sym, nodes)
    dev = sympy.lambdify(x_sym, dev_sym, 'numpy')

    x_values = np.linspace(a, b, nodes)

    dev_min = minimize_scalar(dev, bounds=(min(x_values), max(x_values)), method='bounded').fun
    dev_max = - minimize_scalar(lambda x: -dev(x), bounds=(min(x_values), max(x_values)), method='bounded').fun

    w = np.prod([x - xi for xi in x_values])

    r_max = max((dev_min * w / math.factorial(nodes)),
                dev_max * w / math.factorial(nodes))

    r_min = min(dev_min * w / math.factorial(nodes),
                dev_max * w / math.factorial(nodes))

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


def print_error_result(a: float, b: float, f: Callable[[float], float], x: float, nodes: int, ln: float) -> None:
    r_max, r_min = error_estimate(a, b, f, x, nodes)
    r = error(ln, f, x)

    print(f"Минимальное и максимальное значения остаточного члена R(x):\n"
          f"minR(x) = {r_min:.{15}f}\n"
          f"maxR(x) = {r_max:.{15}f}\n")
    print(f"Значение R(x*) = f(x*) - L(x*):\n"
          f"R(x*) = {r:.{15}f}\n")
    if r_min < r < r_max:
        print("Неравенство minR(x) < R(x*) < maxR(x) выполняется\n")
    else:
        print("Неравенство minR(x) < R(x*) < maxR(x) не выполняется\n")


a = 0.4
b = 0.9
x1 = 0.42
x2 = 0.63
x3 = 0.87
f = lambda x: x**2 + log(x)
nodes = 11

x_values = np.linspace(a, b, nodes)
h = (b - a) / (nodes - 1)

print_table_of_values(a, b, f, nodes)

print(f'Интерполирование в точке x1 = {x1}')

print('Истинное значение функции f в точке x1:')
print(f'{f(x1):.{15}f}\n')

if x1 - x_values[0] < h:
    print('Для интерполирования подходит первая формула Ньютона, так как x1 находится в начале таблицы:')
    print(f'{newton_first(a, b, f, x1, nodes):.{15}f}\n')

    print_error_result(a, b, f, x1, nodes, newton_first(a, b, f, x1, nodes))

elif x_values[nodes - 1] - x1 < h:
    print('Для интерполирования подходит вторая формула Ньютона, так как x1 находится в конце таблицы:')
    print(f'{newton_second(a, b, f, x1, nodes):.{15}f}\n')

    print_error_result(a, b, f, x1, nodes, newton_second(a, b, f, x1, nodes))

elif x1 - x_values[nodes // 2] < 0:
    print('Для интерполирования подходит первая формула Гаусса, так как x1 находится справа от центральноно узла:')
    print(f'{gauss_first(a, b, f, x1, nodes):.{15}f}\n')

    print_error_result(a, b, f, x1, nodes, gauss_first(a, b, f, x1, nodes))

    print('Также можно использовать формулу Cтирлинга:')
    print(f'{stirling(a, b, f, x1, nodes):.{15}f}\n')

    print_error_result(a, b, f, x1, nodes, stirling(a, b, f, x1, nodes))

elif x1 - x_values[nodes // 2] > 0:
    print('Для интерполирования подходит вторая формула Гаусса, так как x1 находится слева от центрального узла:')
    print(f'{gauss_second(a, b, f, x1, nodes):.{15}f}\n')

    print_error_result(a, b, f, x1, nodes, gauss_second(a, b, f, x1, nodes))

    print('Также можно использовать формулу Cтирлинга:')
    print(f'{stirling(a, b, f, x1, nodes):.{15}f}\n')

    print_error_result(a, b, f, x1, nodes, stirling(a, b, f, x1, nodes))

print(f'Интерполирование в точке x2 = {x2}')

print('Истинное значение функции f в точке x2:')
print(f'{f(x2):.{15}f}\n')

if x2 - x_values[0] < h:
    print('Для интерполирования подходит первая формула Ньютона, так как x2 находится в начале таблицы:')
    print(f'{newton_first(a, b, f, x2, nodes):.{15}f}\n')

    print_error_result(a, b, f, x2, nodes, newton_first(a, b, f, x2, nodes))

elif x_values[nodes - 1] - x2 < h:
    print('Для интерполирования подходит вторая формула Ньютона, так как x2 находится в конце таблицы:')
    print(f'{newton_second(a, b, f, x2, nodes):.{15}f}\n')

    print_error_result(a, b, f, x2, nodes, newton_second(a, b, f, x2, nodes))

elif x2 - x_values[nodes // 2] < 0:
    print('Для интерполирования подходит первая формула Гаусса, так как x2 находится справа от центральноно узла:')
    print(f'{gauss_first(a, b, f, x2, nodes):.{15}f}\n')

    print_error_result(a, b, f, x2, nodes, gauss_first(a, b, f, x2, nodes))

    print('Также можно использовать формулу Cтирлинга:')
    print(f'{stirling(a, b, f, x2, nodes):.{15}f}\n')

    print_error_result(a, b, f, x2, nodes, stirling(a, b, f, x2, nodes))

elif x2 - x_values[nodes // 2] > 0:
    print('Для интерполирования подходит вторая формула Гаусса, так как x2 находится слева от центрального узла:')
    print(f'{gauss_second(a, b, f, x2, nodes):.{15}f}\n')

    print_error_result(a, b, f, x2, nodes, gauss_second(a, b, f, x2, nodes))

    print('Также можно использовать формулу Cтирлинга:')
    print(f'{stirling(a, b, f, x2, nodes):.{15}f}\n')

    print_error_result(a, b, f, x2, nodes, stirling(a, b, f, x2, nodes))

print(f'Интерполирование в точке x3 = {x3}')

print('Истинное значение функции f в точке x3:')
print(f'{f(x3):.{15}f}\n')

if x3 - x_values[0] < h:
    print('Для интерполирования подходит первая формула Ньютона, так как x3 находится в начале таблицы:')
    print(f'{newton_first(a, b, f, x3, nodes):.{15}f}\n')

    print_error_result(a, b, f, x3, nodes, newton_first(a, b, f, x3, nodes))

elif (x_values[nodes - 1] - x3 < h):
    print('Для интерполирования подходит вторая формула Ньютона, так как x3 находится в конце таблицы:')
    print(f'{newton_second(a, b, f, x3, nodes):.{15}f}\n')

    print_error_result(a, b, f, x3, nodes, newton_second(a, b, f, x3, nodes))

elif (x3 - x_values[nodes // 2] < 0):
    print('Для интерполирования подходит первая формула Гаусса, так как x3 находится справа от центральноно узла:')
    print(f'{gauss_first(a, b, f, x3, nodes):.{15}f}\n')

    print_error_result(a, b, f, x3, nodes, gauss_first(a, b, f, x3, nodes))

    print('Также можно использовать формулу Cтирлинга:')
    print(f'{stirling(a, b, f, x3, nodes):.{15}f}\n')

    print_error_result(a, b, f, x3, nodes, stirling(a, b, f, x3, nodes))

elif x3 - x_values[nodes // 2] > 0:
    print('Для интерполирования подходит вторая формула Гаусса, так как x3 находится слева от центрального узла:')
    print(f'{gauss_second(a, b, f, x3, nodes):.{15}f}\n')

    print_error_result(a, b, f, x3, nodes, gauss_second(a, b, f, x3, nodes))

    print('Также можно использовать формулу Cтирлинга:')
    print(f'{stirling(a, b, f, x3, nodes):.{15}f}\n')

    print_error_result(a, b, f, x3, nodes, stirling(a, b, f, x3, nodes))
