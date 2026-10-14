"""MPA reference benchmarks F1, F5, F9, F10, F11 and their search bounds."""

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


def _vector(x: ArrayLike) -> np.ndarray:
    vector = np.asarray(x, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("x must be a nonempty, finite one-dimensional vector")
    return vector


def sphere(x: ArrayLike) -> float:
    """F1: sum(x_i^2); optimum zero at x=0; bounds [-100, 100]."""
    x = _vector(x)
    return float(np.sum(x ** 2))


def rastrigin(x: ArrayLike) -> float:
    """F9: sum(x_i^2 - 10*cos(2*pi*x_i)) + 10*d; bounds [-5.12, 5.12]."""
    x = _vector(x)
    return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * x.size)


def rosenbrock(x: ArrayLike) -> float:
    """F5: sum(100*(x_{i+1}-x_i^2)^2+(x_i-1)^2); bounds [-30, 30]."""
    x = _vector(x)
    if x.size < 2:
        raise ValueError("Rosenbrock requires at least two dimensions")
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def ackley(x: ArrayLike) -> float:
    """F10: -20*exp(-.2*sqrt(mean(x^2)))-exp(mean(cos(2*pi*x)))+20+e.

    Bounds [-32, 32], as in Get_Functions_details.m; optimum zero at x=0.
    """
    x = _vector(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2)))
                 - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e)


def griewank(x: ArrayLike) -> float:
    """F11: sum(x_i^2)/4000-prod(cos(x_i/sqrt(i)))+1; bounds [-600,600]."""
    x = _vector(x)
    return float(np.sum(x ** 2) / 4000
                 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1)))) + 1)


@dataclass(frozen=True)
class Benchmark:
    objective: Callable
    lower_bound: float
    upper_bound: float


BENCHMARKS = {
    "sphere": Benchmark(sphere, -100, 100),
    "rastrigin": Benchmark(rastrigin, -5.12, 5.12),
    "rosenbrock": Benchmark(rosenbrock, -30, 30),
    "ackley": Benchmark(ackley, -32, 32),
    "griewank": Benchmark(griewank, -600, 600),
}
