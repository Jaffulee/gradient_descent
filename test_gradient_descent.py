# Unit tests. run with `pytest -v`
import numpy as np
import sympy as sp
from gradient_descent import gradient_descent_from_function


def assert_close(actual, expected, tol=1e-6):
    """Helper for approximate comparison of arrays/lists."""
    actual = np.array(actual, dtype=float).reshape(-1)
    expected = np.array(expected, dtype=float).reshape(-1)
    assert np.allclose(actual, expected, atol=tol), f"Expected {expected}, got {actual}"


def test_quartic_minimum():
    def f0(x, y): return (x - 1) ** 4 + (y + 1) ** 4 + 1
    def gradf0(x, y): return np.array([4 * (x - 1) ** 3, 4 * (y + 1) ** 3])

    minima, value = gradient_descent_from_function(
        2, gradf0, f0,
        rate_parameter=0.7,
        previous_rate_parameter=0.1,
        momentum_parameter=0.7,
    )
    assert_close(minima, [1, -1], tol=1e-3)
    assert abs(value - 1) < 1e-3


def test_simple_quadratic():
    x1, x2 = sp.symbols("x1 x2", real=True)
    f1 = sp.Matrix([(x1 + x2 - 1) ** 2 + (x1 - x2 + 1) ** 2])
    gradf1 = f1.jacobian([x1, x2])

    f1 = sp.lambdify([x1, x2], f1, "numpy")
    gradf1 = sp.lambdify([x1, x2], gradf1, "numpy")

    minima, value = gradient_descent_from_function(2, gradf1, f1)
    assert_close(minima, [0, 1], tol=1e-3)
    assert abs(value - 0) < 1e-6


def test_exponential_target():
    x1, x2, x3, x4, x5 = sp.symbols("x1 x2 x3 x4 x5", real=True)
    f2 = sp.Matrix([
        (2 ** x1 - 2 ** 8) ** 2 +
        (2 ** x2 - 2 ** 0) ** 2 +
        (2 ** x3 - 2 ** 0) ** 2 +
        (2 ** x4 - 2 ** 8) ** 2 +
        (2 ** x5 - 2 ** 5) ** 2 + 80085
    ])
    gradf2 = f2.jacobian([x1, x2, x3, x4, x5])

    f2 = sp.lambdify([x1, x2, x3, x4, x5], f2, "numpy")
    gradf2 = sp.lambdify([x1, x2, x3, x4, x5], gradf2, "numpy")

    minima, value = gradient_descent_from_function(
        5, gradf2, f2,
        rate_parameter=0.0005,
        previous_rate_parameter=0.00001,
        momentum_parameter=0.95,
        noise_parameter=1,
        noise_interval=100000,
        noise_sigma=2,
    )
    assert_close(minima, [8, 0, 0, 8, 5], tol=1e-1)
    assert abs(value - 80085) < 1e-1


def test_maximization():
    x1 = sp.symbols("x1", real=True)
    f3 = sp.Matrix([-((x1 - 6) ** 2) * (((2 ** x1) + 1) ** 2) + 6])
    gradf3 = f3.jacobian([x1])

    f3 = sp.lambdify([x1], f3, "numpy")
    gradf3 = sp.lambdify([x1], gradf3, "numpy")

    minima, value = gradient_descent_from_function(
        1, gradf3, f3,
        num_attempts=15,
        xinit_radius=15,
        descent=False,   # maximization
        rate_parameter=0.00005,
        previous_rate_parameter=0.00001,
        momentum_parameter=0.97,
        noise_parameter=0.000001,
        noise_interval=50000,
    )
    assert_close(minima, [6], tol=1e-1)
    assert abs(value - 6) < 1e-1
