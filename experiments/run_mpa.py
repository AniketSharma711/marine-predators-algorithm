"""Run configurable independent MPA experiments and plot convergence."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms import mpa
from benchmarks.functions import BENCHMARKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="sphere")
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--lower-bound", type=float)
    parser.add_argument("--upper-bound", type=float)
    parser.add_argument("--csv", type=Path, help="Save per-run convergence, e.g. results/sphere.csv")
    parser.add_argument("--plot", type=Path, help="Save the convergence figure")
    parser.add_argument("--no-show", action="store_true", help="Run without an interactive window")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    benchmark = BENCHMARKS[args.benchmark]
    lower = benchmark.lower_bound if args.lower_bound is None else args.lower_bound
    upper = benchmark.upper_bound if args.upper_bound is None else args.upper_bound
    figure, axis = plt.subplots(figsize=(8, 5))
    rows = []
    final_fitness = []
    for run in range(args.runs):
        seed = args.seed + run
        result = mpa(args.population_size, args.iterations, lower, upper,
                     args.dimensions, benchmark.objective, seed=seed)
        final_fitness.append(result.best_fitness)
        print(f"Run {run + 1}, seed {seed}: best fitness = {result.best_fitness:.12g}")
        print(f"Best position: {np.array2string(result.best_position, precision=8)}")
        axis.plot(np.arange(1, args.iterations + 1), result.convergence_curve,
                  label=f"seed {seed}")
        rows.extend((run + 1, seed, i + 1, float(value))
                    for i, value in enumerate(result.convergence_curve))
    if args.runs > 1:
        print(f"Final fitness: mean={np.mean(final_fitness):.12g}, "
              f"std={np.std(final_fitness, ddof=1):.12g}, "
              f"min={min(final_fitness):.12g}, max={max(final_fitness):.12g}")
    axis.set(xlabel="Completed iteration", ylabel="Best-so-far fitness",
             title=f"MPA — {args.benchmark} ({args.dimensions} dimensions)")
    # Symlog shows zero or negative objective values without modifying the curve.
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("run", "seed", "iteration", "best_fitness"))
            writer.writerows(rows)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=160)
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
