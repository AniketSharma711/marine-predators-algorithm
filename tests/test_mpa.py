"""Behavioral checks and deterministic regression checks for MATLAB translation."""

import importlib
from math import gamma, pi, sin

import numpy as np
import pytest

from algorithms import mpa
from algorithms.mpa import _correct_bounds, _fads, _levy, _memory, _movement
from benchmarks.functions import BENCHMARKS, ackley, griewank, rastrigin, rosenbrock, sphere


@pytest.mark.parametrize("name", BENCHMARKS)
def test_benchmarks_run_and_result_contract(name):
    benchmark = BENCHMARKS[name]
    result = mpa(12, 24, benchmark.lower_bound, benchmark.upper_bound,
                 5, benchmark.objective, seed=17)
    assert result.best_position.shape == (5,)
    assert isinstance(result.best_fitness, float) and np.isfinite(result.best_fitness)
    assert result.convergence_curve.shape == (24,)
    assert np.all(np.isfinite(result.convergence_curve))
    assert np.all(np.diff(result.convergence_curve) <= 1e-12)
    assert result.convergence_curve[-1] == result.best_fitness
    assert benchmark.objective(result.best_position) == pytest.approx(result.best_fitness)
    assert np.all(result.best_position >= benchmark.lower_bound)
    assert np.all(result.best_position <= benchmark.upper_bound)


def test_repeatability_and_global_random_state_independence():
    first = mpa(15, 30, -100, 100, 7, sphere, seed=42)
    np.random.seed(987)
    np.random.random(50)
    second = mpa(15, 30, -100, 100, 7, sphere, seed=42)
    np.testing.assert_array_equal(first.best_position, second.best_position)
    np.testing.assert_array_equal(first.convergence_curve, second.convergence_curve)
    assert first.best_fitness == second.best_fitness


def test_sphere_multiple_independent_seeds_approach_optimum():
    fitness = [mpa(30, 100, -100, 100, 10, sphere, seed=s).best_fitness for s in range(5)]
    assert np.median(fitness) < 0.01


@pytest.mark.parametrize("n,iterations", [(1, 1), (5, 9), (6, 10)])
def test_all_evaluations_bounded_and_exact_call_count(n, iterations):
    lower = np.array([-3., 1., -8., 2.])
    upper = np.array([2., 4., -1., 2.])
    calls = []

    def objective(x):
        assert x.shape == (4,)
        assert np.all(x >= lower) and np.all(x <= upper)
        calls.append(x.copy())
        return sphere(x)

    result = mpa(n, iterations, lower, upper, 4, objective, seed=7)
    assert len(calls) == 2 * n * iterations
    assert np.all(result.best_position >= lower) and np.all(result.best_position <= upper)


def test_mask_boundary_correction():
    positions = np.array([[-4., 2., 99.], [0., -10., 7.]])
    actual = _correct_bounds(positions, np.array([-3., 1., 7.]), np.array([3., 2., 7.]))
    np.testing.assert_array_equal(actual, [[-3., 2., 7.], [0., 1., 7.]])


def test_memory_retains_only_strictly_better_previous_candidates():
    old = np.array([[1., 2.], [3., 4.], [5., 6.]])
    new = old + 10
    positions, fitness = _memory(new, np.array([2., 2., 2.]), old, np.array([1., 2., 3.]))
    np.testing.assert_array_equal(positions, [old[0], new[1], new[2]])
    np.testing.assert_array_equal(fitness, [1., 2., 2.])
    positions[:] = -999
    assert old[0, 0] == 1 and new[0, 0] == 11


class FixedRandom:
    def __init__(self, value=0.4):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


@pytest.mark.parametrize("iteration,expected", [
    (0, [1.2, 2., 2.8, 3.6, 4.4]),  # Brownian phase.
    (3, [3.2, 2.8, 2.4, 2., 1.6]),  # Exact one-third: final branch.
    (4, [1., 1.2, 1.8, 1.6, 1.4]),  # Odd n: rows 3,4,5 are second half.
    (6, [3.2, 2.8, 2.4, 2., 1.6]),  # Exact two-thirds: final branch.
    (8, [3.2, 2.8, 2.4, 2., 1.6]),
])
def test_movement_equations_and_strict_phase_boundaries(iteration, expected):
    prey = np.arange(1., 6.).reshape(5, 1)
    rng = FixedRandom()
    _movement(prey, np.full((5, 1), 2.), np.ones((5, 1)),
              np.full((5, 1), 2.), iteration, 9, 0.4, rng)
    # RL=2 and elite=2: final branch is 2+.2*2*(4-x)=3.6-.4*x;
    # Phase 1 is x+.2*(2-x); mixed first half x+.2*2*(2-2*x).
    np.testing.assert_allclose(prey[:, 0], expected)
    assert rng.calls == 5


def test_levy_formula_and_normal_draw_order():
    class NormalDraws:
        def __init__(self):
            self.calls = []

        def normal(self, mean, scale, shape):
            self.calls.append((mean, scale, shape))
            return np.full(shape, 2. if len(self.calls) == 1 else -8.)

    rng = NormalDraws()
    sample = _levy(3, 2, 1.5, rng)
    sigma = (gamma(2.5) * sin(0.75 * pi) / (gamma(1.25) * 1.5 * 2 ** .25)) ** (1 / 1.5)
    assert rng.calls == [(0, sigma, (3, 2)), (0, 1, (3, 2))]
    np.testing.assert_allclose(sample, np.full((3, 2), .5))


def test_fads_mask_branch_preserves_signed_uniform_displacement():
    class MaskDraws:
        def __init__(self):
            self.draws = iter([0.1, np.array([[.1, .8], [.9, .0]]), np.full((2, 2), .25)])

        def random(self, shape=None):
            value = next(self.draws)
            if shape is not None:
                assert value.shape == shape
            return value

    actual = _fads(np.ones((2, 2)), np.array([-4., 2.]), np.array([4., 6.]), .5, MaskDraws())
    np.testing.assert_array_equal(actual, [[0., 1.], [1., 2.5]])


def test_non_fads_two_permutations_and_inclusive_probability_boundary():
    class PermutationDraws:
        def __init__(self):
            self.draws = iter([.2, .5])
            self.permutations = iter([np.array([2, 0, 1]), np.array([1, 2, 0])])

        def random(self):
            return next(self.draws)

        def permutation(self, size):
            assert size == 3
            return next(self.permutations)

    actual = _fads(np.array([[1.], [4.], [9.]]), np.array([-10.]),
                   np.array([10.]), .5, PermutationDraws())
    np.testing.assert_allclose(actual[:, 0], [4., -.8, 10.8])


def test_first_iteration_second_memory_reset_and_final_fads_not_evaluated(monkeypatch):
    module = importlib.import_module("algorithms.mpa")
    monkeypatch.setattr(module, "_initialize", lambda *args: np.array([[1.], [2.]]))

    def worse_movement(prey, *args):
        prey[:] = [[3.], [4.]]

    def final_fads(prey, *args):
        # The first iteration's second memory reset keeps worse movement rows.
        np.testing.assert_array_equal(prey, [[3.], [4.]])
        return np.zeros_like(prey)  # This apparent optimum must not be evaluated.

    monkeypatch.setattr(module, "_movement", worse_movement)
    monkeypatch.setattr(module, "_fads", final_fads)
    result = mpa(2, 1, -10, 10, 1, sphere, seed=0)
    np.testing.assert_array_equal(result.best_position, [1.])
    assert result.best_fitness == 1.


@pytest.mark.parametrize("parameter,value", [
    ("population_size", 0), ("population_size", True), ("population_size", 2.5),
    ("max_iterations", 0), ("max_iterations", -5), ("dimensions", 0),
    ("lower_bound", [1, 2]), ("upper_bound", [[1, 2, 3]]),
    ("lower_bound", float("nan")), ("upper_bound", float("inf")),
    ("lower_bound", 6),
])
def test_invalid_parameters(parameter, value):
    arguments = dict(population_size=5, max_iterations=10, lower_bound=-5,
                     upper_bound=5, dimensions=3, objective_function=sphere)
    arguments[parameter] = value
    with pytest.raises(ValueError, match="bound|dimension|population|max_iterations"):
        mpa(**arguments)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [1., 2.], 1j])
def test_invalid_objective_output(value):
    with pytest.raises(ValueError, match="finite real scalar"):
        mpa(2, 2, -1, 1, 3, lambda x: value, seed=0)


def test_objective_mutation_cannot_corrupt_population():
    def objective(x):
        value = sphere(x)
        x[:] = 1e9
        return value

    actual = mpa(5, 9, -1, 1, 3, objective, seed=4)
    expected = mpa(5, 9, -1, 1, 3, sphere, seed=4)
    np.testing.assert_array_equal(actual.best_position, expected.best_position)
    np.testing.assert_array_equal(actual.convergence_curve, expected.convergence_curve)


@pytest.mark.parametrize("function,position", [
    (sphere, np.zeros(5)), (rastrigin, np.zeros(5)), (rosenbrock, np.ones(5)),
    (ackley, np.zeros(5)), (griewank, np.zeros(5)),
])
def test_known_benchmark_optima(function, position):
    assert function(position) == pytest.approx(0, abs=1e-12)


def test_nonzero_benchmark_values():
    assert sphere([1, -2]) == 5
    assert rastrigin([1, -2]) == pytest.approx(5)
    assert rosenbrock([0, 0]) == 1
    assert ackley([1, 1]) == pytest.approx(20 * (1 - np.exp(-.2)))
    assert griewank([1, 2]) == pytest.approx(5 / 4000 - np.cos(1) * np.cos(np.sqrt(2)) + 1)
