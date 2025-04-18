import numpy as np
import sympy as sp
from math import log, e, pi, sin, cos, exp
from sympy import log, pi, sin, cos, exp
import pandas as pd
from typing import Callable, Tuple


def left_rectangles(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Computes the definite integral of a function using the left rectangle rule

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval
    :param n: Number of subintervals to divide the integration range [a,b]

    :return: Approximate value of the integral
    """
    h = (b - a) / n
    x_values = np.linspace(a, b, n + 1)
    f_values = [f(x) for x in x_values[:-1]]

    return h * sum(f_values)


def right_rectangles(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Computes the definite integral of a function using the right rectangle rule

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval
    :param n: Number of subintervals to divide the integration range [a,b]

    :return: Approximate value of the integral
    """
    h = (b - a) / n
    x_values = np.linspace(a, b, n + 1)
    f_values = [f(x) for x in x_values[1:]]

    return h * sum(f_values)


def central_rectangles(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Computes the definite integral of a function using the central rectangle rule

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval
    :param n: Number of subintervals to divide the integration range [a,b]

    :return: Approximate value of the integral
    """
    h = (b - a) / n
    x_values = np.linspace(a, b, n + 1)
    f_values = [f(x) for x in x_values]

    return 2 * h * sum(f_values[1::2])


def trapezoid(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Computes the definite integral of a function using the central trapezoidal rule

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval
    :param n: Number of subintervals to divide the integration range [a,b]

    :return: Approximate value of the integral
    """
    h = (b - a) / n
    x_values = np.linspace(a, b, n + 1)
    f_values = [f(x) for x in x_values]

    return h * (0.5 * (f_values[0] + f_values[-1]) + sum(f_values[1:-1]))


def lagrange_interpolation(f: Callable[[float], float], a: float, b: float, n: int) -> sp.Expr:
    """
    Construct the Lagrange interpolation polynomial for a given function over an interval.

    :param f: The function to be interpolated
    :param a: Lower bound of the interval
    :param b: Upper bound of the interval
    :param n: Degree of interpolating polynomial

    :return: A SymPy expression representing the Lagrange interpolation polynomial
    """
    x_values = np.linspace(a, b, n + 1)
    f_values = [f(x) for x in x_values]

    x = sp.symbols('x')
    L = 0

    for k in range(n + 1):
        term = f_values[k]
        for j in range(n + 1):
            if j != k:
                term *= (x - x_values[j]) / (x_values[k] - x_values[j])
        L += term

    return sp.simplify(L)


def simpson(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Computes the definite integral of a function using the Simpson's rule

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval
    :param n: Number of subintervals to divide the integration range [a,b]

    :return: Approximate value of the integral
    """
    x_values = np.linspace(a, b, n + 1)
    integral = 0.0

    for i in range(0, n, 2):
        x_segment = x_values[i:i + 3]

        x = sp.symbols('x')
        L = lagrange_interpolation(f, x_segment[0], x_segment[2], 2)

        segment_integral = sp.integrate(L, (x, x_segment[0], x_segment[2]))
        integral += float(segment_integral.evalf())

    return integral


def weddle(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Computes the definite integral of a function using the Weddle's rule

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval
    :param n: Number of subintervals to divide the integration range [a,b]

    :return: Approximate value of the integral
    """
    if n % 6 != 0:
        raise ValueError("Число интервалов должно быть кратно 6")

    nodes = n + 1

    x_values = np.linspace(a, b, nodes)
    f_values = [f(x) for x in x_values]

    h = (b - a) / (nodes - 1)

    pattern = [1] + [5, 1, 6, 1, 5, 2] * (n // 6)
    pattern[-1] = 1

    integral = sum([f_values[i] * pattern[i] for i in range(nodes)])

    return integral * 3 * h / 10


def newton_cotes(f: Callable[[float], float], a: float, b: float, order: int) -> float:
    """
    Computes the definite integral of a function using the Newton-Cotes rule

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval
    :param order: Degree of the Newton-Cotes formula (1-6)

    :return: Approximate value of the integral
    """
    coefficients = {
        1: [1 / 2, 1 / 2],
        2: [1 / 6, 4 / 6, 1 / 6],
        3: [1 / 8, 3 / 8, 3 / 8, 1 / 8],
        4: [7 / 90, 32 / 90, 12 / 90, 32 / 90, 7 / 90],
        5: [19 / 288, 75 / 288, 50 / 288, 50 / 288, 75 / 288, 19 / 288],
        6: [41 / 840, 216 / 840, 27 / 840, 272 / 840, 27 / 840, 216 / 840, 41 / 840]
    }

    x_values = np.linspace(a, b, order + 1)
    f_values = [f(x) for x in x_values]

    c = coefficients[order]

    integral = sum([c[k] * f_values[k] for k in range(order + 1)])

    return (b - a) * integral


def gauss(f: Callable[[float], float], a: float, b: float, order: int) -> float:
    """
    Computes the definite integral of a function using the Gauss rule

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval
    :param order: Degree of the Gauss formula (1-4)

    :return: Approximate value of the integral
    """
    gauss_data = {
        1: {'t': [0.0], 'c': [2.0]},
        2: {'t': [-0.577350, 0.577350], 'c': [1.0, 1.0]},
        3: {'t': [-0.774597, 0.0, 0.774597], 'c': [5 / 9, 8 / 9, 5 / 9]},
        4: {'t': [-0.861136, -0.339981, 0.339981, 0.861136],
            'c': [0.347855, 0.652145, 0.652145, 0.347855]}
    }

    data = gauss_data[order]
    t = data['t']
    c = data['c']

    x_values = [(b + a) / 2 + (b - a) * t_k / 2 for t_k in t]
    f_values = [f(x) for x in x_values]

    integral = sum(c[i] * f_values[i] for i in range(order))

    return (b - a) * integral / 2


def integral(f, a, b) -> float:
    """
    Computes the definite integral of a function

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval

    :return: Value of the integral
    """
    x = sp.symbols('x')
    f_expr = sp.sympify(f(x))

    integral = sp.integrate(f_expr, (x, a, b))

    return float(integral.evalf())


def adaptive_integration(f: Callable[[float], float],
                         a: float,
                         b: float,
                         method: Callable,
                         epsilon: float,
                         max_iter: int = 50,
                         intervals: int = 1,
                         order: int = 0) -> Tuple[float, int]:
    """
    Performs numerical integration of a function until desired precision is achieved

    :param f: The function to integrate
    :param a: Lower bound of the integration interval
    :param b: Upper bound of the integration interval
    :param method: Integration method to use (e.g., trapezoid, simpson, gauss, etc.)
    :param epsilon: Desired precision
    :param max_iter: Maximum number of iterations
    :param intervals: Number of subintervals to divide the integration range [a,b]
    :param order: Degree of the Gauss formula (1-4) or Newton-Cotes formula (1-6)

    :return: Tuple containing the computed integral value and the final number of intervals used
    """

    if method == gauss or method == newton_cotes:
        segments = [(a, b)]
        I_prev = method(f, a, b, order)

        for _ in range(max_iter):
            new_segments = []
            intervals *= 2
            for seg_a, seg_b in segments:
                mid = (seg_a + seg_b) / 2
                new_segments.append((seg_a, mid))
                new_segments.append((mid, seg_b))

            I_curr = sum(method(f, seg_a, seg_b, order) for seg_a, seg_b in new_segments)

            if abs(I_prev - I_curr) < epsilon:

                return I_curr, intervals

            segments = new_segments
            I_prev = I_curr

        return I_curr, intervals

    else:
        I_prev = method(f, a, b, intervals)
        for _ in range(max_iter):
            intervals *= 2
            I_curr = method(f, a, b, intervals)

            if abs(I_prev - I_curr) < epsilon:
                return I_curr, intervals
            I_prev = I_curr

        return I_curr, intervals


a = 1.0
b = 1.5
f = lambda x: (x - 1) ** 2 - exp(-x)

methods_info = [
    {
        "name": "Левые прямоугольники",
        "method": left_rectangles,
        "epsilon": 1e-4,
        "intervals": 2,
        "order": None
    },
    {
        "name": "Правые прямоугольники",
        "method": right_rectangles,
        "epsilon": 1e-4,
        "intervals": 2,
        "order": None
    },
    {
        "name": "Центральные прямоугольники",
        "method": central_rectangles,
        "epsilon": 1e-4,
        "intervals": 2,
        "order": None
    },
    {
        "name": "Формула трапеции",
        "method": trapezoid,
        "epsilon": 1e-4,
        "intervals": 2,
        "order": None
    },
    {
        "name": "Формула Симпсона",
        "method": simpson,
        "epsilon": 1e-8,
        "intervals": 2,
        "order": None
    },
    {
        "name": "Формула Веддля",
        "method": weddle,
        "epsilon": 1e-8,
        "intervals": 6,
        "order": None
    },
    {
        "name": "Формула Ньютона-Котеса порядка 4",
        "method": newton_cotes,
        "epsilon": 1e-12,
        "intervals": 1,
        "order": 4
    },
    {
        "name": "Формула Гаусса порядка 4",
        "method": gauss,
        "epsilon": 1e-12,
        "intervals": 1,
        "order": 4
    }
]

results = []

results.append({
        "Метод": "Истинное значение",
        "Значение интеграла": f"{integral(f, a, b):.15f}",
        "Точность": "-",
        "Число интервалов": "-"
    })

for info in methods_info:
    integral_value, final_n = adaptive_integration(
        f, a, b, info["method"],
        epsilon=info["epsilon"],
        intervals=info["intervals"],
        order=info["order"]
    )

    results.append({
        "Метод": info["name"],
        "Значение интеграла": f"{integral_value:.15f}",
        "Точность": f"{info['epsilon']:.1e}",
        "Число интервалов": final_n
    })

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.DataFrame(results)
print(df)
