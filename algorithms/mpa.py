"""Marine Predators Algorithm, translated from Faramarzi et al.'s MPA.m.

Reference: https://github.com/afshinfaramarzi/Marine-Predators-Algorithm
The reference workflow, including its strict phase boundaries, is preserved.
"""

from collections.abc import Callable
from math import gamma, pi, sin
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class MPAResult(NamedTuple):
    """Best position (d,), scalar fitness, and best-so-far curve (iterations,)."""

    best_position: FloatArray
    best_fitness: float
    convergence_curve: FloatArray


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _bounds(value: ArrayLike, dimensions: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(dimensions, float(array))
    elif array.shape != (dimensions,):
        raise ValueError(f"{name} must be a scalar or have shape ({dimensions},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _initialize(population_size: int, dimensions: int, lower: FloatArray,
                upper: FloatArray, vector_bounds: bool, rng: np.random.Generator) -> FloatArray:
    """initialization.m: one matrix for scalar bounds, columns for vector bounds."""
    if not vector_bounds:
        return rng.random((population_size, dimensions)) * (upper - lower) + lower
    prey = np.empty((population_size, dimensions))
    for j in range(dimensions):
        prey[:, j] = rng.random(population_size) * (upper[j] - lower[j]) + lower[j]
    return prey


def _correct_bounds(prey: FloatArray, lower: FloatArray, upper: FloatArray) -> FloatArray:
    """Reference masks: replace above/below-bound coordinates with that endpoint."""
    above = prey > upper
    below = prey < lower
    return prey * ~(above | below) + upper * above + lower * below


def _memory(prey: FloatArray, fitness: FloatArray, old_prey: FloatArray,
            old_fitness: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Retain the previous candidate only when its fitness is strictly better."""
    retain = old_fitness < fitness
    return (np.where(retain[:, None], old_prey, prey),
            np.where(retain, old_fitness, fitness))


def _levy(n: int, m: int, beta: float, rng: np.random.Generator) -> FloatArray:
    """levy.m's Mantegna sampler: u / abs(v)**(1/beta), shape (n, m)."""
    numerator = gamma(1 + beta) * sin(pi * beta / 2)
    denominator = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (numerator / denominator) ** (1 / beta)
    u = rng.normal(0, sigma_u, (n, m))
    v = rng.normal(0, 1, (n, m))
    return u / np.abs(v) ** (1 / beta)


def _movement(prey: FloatArray, elite: FloatArray, rb: FloatArray, rl: FloatArray,
              iteration: int, max_iterations: int, cf: float,
              rng: np.random.Generator) -> None:
    """MPA.m Eqs. 12–15; sample R even in branches that do not use it."""
    n, d = prey.shape
    p = 0.5
    for i in range(n):
        for j in range(d):
            r = rng.random()
            if iteration < max_iterations / 3:
                # High velocity: Brownian prey exploration around the elite.
                step = rb[i, j] * (elite[i, j] - rb[i, j] * prey[i, j])
                prey[i, j] += p * r * step
            elif iteration > max_iterations / 3 and iteration < 2 * max_iterations / 3:
                # Unit velocity: Lévy first half; Brownian second half.
                if i + 1 > n / 2:  # MATLAB's one-based population-half condition.
                    step = rb[i, j] * (rb[i, j] * elite[i, j] - prey[i, j])
                    prey[i, j] = elite[i, j] + p * cf * step
                else:
                    step = rl[i, j] * (elite[i, j] - rl[i, j] * prey[i, j])
                    prey[i, j] += p * r * step
            else:
                # Low velocity: Lévy exploitation; also the exact one-third point.
                step = rl[i, j] * (rl[i, j] * elite[i, j] - prey[i, j])
                prey[i, j] = elite[i, j] + p * cf * step


def _fads(prey: FloatArray, lower: FloatArray, upper: FloatArray, cf: float,
          rng: np.random.Generator) -> FloatArray:
    """MPA.m Eq. 16: FADs mask or two independent row permutations."""
    probability = 0.2
    if rng.random() < probability:
        mask = rng.random(prey.shape) < probability
        return prey + cf * ((lower + rng.random(prey.shape) * (upper - lower)) * mask)
    r = rng.random()
    first = rng.permutation(prey.shape[0])
    second = rng.permutation(prey.shape[0])
    step = (probability * (1 - r) + r) * (prey[first] - prey[second])
    return prey + step


def mpa(population_size: int, max_iterations: int, lower_bound: ArrayLike,
        upper_bound: ArrayLike, dimensions: int,
        objective_function: Callable[[FloatArray], float], seed: int | None = None) -> MPAResult:
    """Minimize a finite scalar objective using vanilla MPA.

    Bounds are scalars or arrays of shape (dimensions,). The objective receives
    a copied, bounded vector of shape (dimensions,). Population layout is
    (population_size, dimensions). A seed reproduces runs within this Python
    implementation; it does not reproduce MATLAB's random sequence.

    Return an MPAResult, also unpackable as (best_position, best_fitness, curve).
    curve[k] is the best fitness after reference iteration k (zero-based),
    following both evaluation/memory passes and the unevaluated FADs update.
    There are exactly 2 * population_size * max_iterations objective calls.
    Transient movement/FADs positions may exceed bounds; correction occurs
    before each evaluation, at precisely the two reference locations.
    """
    for value, name in ((population_size, "population_size"),
                        (max_iterations, "max_iterations"), (dimensions, "dimensions")):
        _positive_integer(value, name)
    if not callable(objective_function):
        raise TypeError("objective_function must be callable")
    lower = _bounds(lower_bound, dimensions, "lower_bound")
    upper = _bounds(upper_bound, dimensions, "upper_bound")
    if np.any(lower > upper):
        raise ValueError("lower_bound must be <= upper_bound in every dimension")
    if not np.all(np.isfinite(upper - lower)):
        raise ValueError("bound widths must be finite")
    rng = np.random.default_rng(seed)
    vector_bounds = np.asarray(lower_bound).ndim != 0 or np.asarray(upper_bound).ndim != 0
    prey = _initialize(population_size, dimensions, lower, upper, vector_bounds, rng)
    best_position = np.zeros(dimensions)
    best_fitness = float("inf")
    curve = np.zeros(max_iterations)

    def evaluate(population: FloatArray) -> FloatArray:
        nonlocal best_position, best_fitness
        fitness = np.empty(population_size)
        for i in range(population_size):
            value = np.asarray(objective_function(population[i].copy()))
            if value.ndim != 0 or np.iscomplexobj(value):
                raise ValueError("objective_function must return a finite real scalar")
            fitness[i] = float(value)
            if not np.isfinite(fitness[i]):
                raise ValueError("objective_function must return a finite real scalar")
            if fitness[i] < best_fitness:
                best_fitness = float(fitness[i])
                best_position = population[i].copy()
        return fitness

    for iteration in range(max_iterations):
        prey = _correct_bounds(prey, lower, upper)
        fitness = evaluate(prey)
        if iteration == 0:
            old_fitness, old_prey = fitness.copy(), prey.copy()
        prey, fitness = _memory(prey, fitness, old_prey, old_fitness)
        old_fitness, old_prey = fitness.copy(), prey.copy()

        # Eq. 10: repeated top predator; CF reduces the movement scale over time.
        elite = np.tile(best_position, (population_size, 1))
        cf = (1 - iteration / max_iterations) ** (2 * iteration / max_iterations)
        rl = 0.05 * _levy(population_size, dimensions, 1.5, rng)
        rb = rng.standard_normal((population_size, dimensions))
        _movement(prey, elite, rb, rl, iteration, max_iterations, cf, rng)

        prey = _correct_bounds(prey, lower, upper)
        fitness = evaluate(prey)
        # The source resets memory again in iteration zero, intentionally kept.
        if iteration == 0:
            old_fitness, old_prey = fitness.copy(), prey.copy()
        prey, fitness = _memory(prey, fitness, old_prey, old_fitness)
        old_fitness, old_prey = fitness.copy(), prey.copy()

        prey = _fads(prey, lower, upper, cf, rng)
        curve[iteration] = best_fitness

    return MPAResult(best_position.copy(), best_fitness, curve)
