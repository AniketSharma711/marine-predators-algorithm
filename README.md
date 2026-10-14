# Marine Predators Algorithm in Python

Vanilla MPA for continuous minimization, translated against the complete
`MPA.m`, `initialization.m`, `levy.m`, and `Get_Functions_details.m` from the
[authors' MATLAB repository](https://github.com/afshinfaramarzi/Marine-Predators-Algorithm).
Downloaded source files and the upstream license are retained in `references/`
for inspection, with retrieval date and SHA-256 hashes in `source_manifest.json`.
No quantum, adaptive, SSA, or Quantum-Swarm features are included.

## Setup and running

Use Python 3.10 or newer. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m experiments.run_mpa
```

NumPy and Matplotlib are runtime dependencies; pytest is for validation.
Run the experiment as a module from the root so the local packages resolve.
The default run uses Sphere, population 30, dimensions 10, 100 iterations,
and seed 42. These are development settings, not the paper's configuration.

```powershell
python -m experiments.run_mpa --benchmark rastrigin --dimensions 10 --population-size 30 --iterations 100 --seed 42 --runs 5 --csv results/rastrigin.csv --plot results/rastrigin.png --no-show
```

Available benchmarks: `sphere`, `rastrigin`, `rosenbrock`, `ackley`, `griewank`.
Override bounds using `--lower-bound` and `--upper-bound`. Each independent run
uses seed `seed + run_index`. The runner prints each best solution and fitness;
multiple runs also print mean, sample standard deviation, minimum, and maximum.
CSV records run, seed, completed iteration (one-based), and best fitness.
Plots use a symmetric logarithmic axis to accommodate zero and negative values.

## Python API

```python
from algorithms import mpa
from benchmarks.functions import sphere

result = mpa(
    population_size=30,
    max_iterations=100,
    lower_bound=-100,
    upper_bound=100,
    dimensions=10,
    objective_function=sphere,
    seed=42,
)
print(result.best_fitness, result.best_position)
best_position, best_fitness, convergence_curve = result
```

`population_size`, `max_iterations`, and `dimensions` must be positive integers.
Both bounds may be finite scalars or one-dimensional arrays of length
`dimensions`; scalar/vector combinations are supported. Lower bounds must not
exceed upper bounds; equal bounds fix a coordinate. Bound widths must be finite.
The callable receives a copied, bounded NumPy vector of shape `(dimensions,)`
and must return a finite real scalar. Its own randomness, if any, must be
controlled by the caller. MPA minimizes this value. `seed=None` uses fresh
randomness; an integer controls the local NumPy Generator.

The result is a named tuple with:

- `best_position`: independent array of shape `(dimensions,)`.
- `best_fitness`: Python float, the lowest evaluated objective value.
- `convergence_curve`: array of shape `(max_iterations,)`. Element `k` records
  the best value at the end of reference iteration `k`, after both evaluation
  passes and the FADs operation. There is no extra initial point.

The internal population has shape `(population_size, dimensions)`. There are
exactly `2 * population_size * max_iterations` objective calls. The last FADs
perturbation is never evaluated, matching the source.

## Translation audit

Let `t` be the zero-based iteration, `T` the requested count, `x` a prey
coordinate, and `e` the corresponding elite coordinate. `P=0.5` and `FADs=0.2`.

1. **Initialization — initialization.m:** scalar bounds use a full uniform
   matrix scaled by `upper-lower` and shifted by `lower`. Vector bounds are
   initialized column by column. Top fitness starts at infinity, the top
   position and convergence curve at zero.
2. **Bounds and elite — both detecting-top-predator blocks of MPA.m:** coordinates
   above/below a bound are replaced with that endpoint using the source masks.
   Candidates are evaluated in population order; only strictly smaller fitness
   updates the elite. Copies preserve MATLAB's value semantics.
3. **Memory — both marine-memory blocks:** replace current rows with old rows
   only where `fit_old < fitness`, then save the resulting positions and fitness.
   Ties keep current rows. Both blocks reset memory when `t=0`, including the
   second block after movement. The elite updates before memory restoration.
4. **Elite/control/random arrays:** repeat the best position across rows;
   `CF=(1-t/T)^(2*t/T)`. Generate `RL=0.05*levy(n,d,1.5)` before `RB=N(0,1)`.
   A fresh `R=U(0,1)` is sampled for every coordinate, including branches that
   do not use it.
5. **Movement — the exact source conditions and equations:**
   - If `t < T/3`: `step=RB*(e-RB*x)`; `x=x+P*R*step` (Brownian exploration).
   - If `t > T/3` and `t < 2*T/3`: MATLAB rows `i>n/2` use
     `step=RB*(RB*e-x)`; `x=e+P*CF*step`. Other rows use
     `step=RL*(e-RL*x)`; `x=x+P*R*step`. This splits exploration and exploitation.
     Python uses `i+1>n/2`; odd populations place the extra row in the second half.
   - Otherwise: `step=RL*(RL*e-x)`; `x=e+P*CF*step` (Lévy exploitation).
     Exactly `t=T/3` also enters this branch, as does `t=2*T/3`.
6. **Lévy — levy.m:** independent normal matrices `u~N(0,sigma_u)` and
   `v~N(0,1)` give `u/abs(v)^(1/beta)`, with
   `sigma_u=[Gamma(1+beta)*sin(pi*beta/2)/(Gamma((1+beta)/2)*beta*2^((beta-1)/2))]^(1/beta)`.
   The optimizer uses `beta=1.5` and the source's `0.05` scaling.
7. **FADs — after the second evaluation and memory block:** if a fresh uniform
   scalar is `<0.2`, draw a uniform matrix to form `U=(random<0.2)`, then a
   second uniform matrix `Q`; add `CF*((lower+Q*(upper-lower))*U)`.
   Otherwise draw scalar `r`, then two independent row permutations `a,b`;
   add `(0.2*(1-r)+r)*(Prey[a]-Prey[b])`. This models environmental perturbations.
8. **Convergence:** record the elite after FADs and continue with the next
   iteration's bounds correction and evaluation.

Movement and FADs can temporarily leave the bounds. Every evaluated candidate
and returned best position is bounded. Correcting immediately after FADs would
change reference timing, so no extra correction or evaluation is inserted.

## Benchmark formulas and bounds

The five formulas and bounds come from `Get_Functions_details.m`:

- Sphere (F1): `sum(x_i^2)`, bounds `[-100,100]`.
- Rosenbrock (F5): `sum(100*(x_(i+1)-x_i^2)^2+(x_i-1)^2)`, bounds `[-30,30]`;
  requires at least two dimensions.
- Rastrigin (F9): `sum(x_i^2-10*cos(2*pi*x_i))+10*d`, bounds `[-5.12,5.12]`.
- Ackley (F10): `-20*exp(-0.2*sqrt(mean(x_i^2)))-exp(mean(cos(2*pi*x_i)))+20+e`,
  bounds `[-32,32]`, matching this source rather than the also-common `32.768`.
- Griewank (F11): `sum(x_i^2)/4000-prod(cos(x_i/sqrt(i)))+1`, `i=1,...,d`,
  bounds `[-600,600]`.

All have global minimum zero. Rosenbrock's optimum is the all-ones vector;
the other four use the zero vector. Dimensionality is configurable rather than
fixed at the helper file's 50 dimensions.

## Validation and differences

Executed validation on 2026-10-14 with Python 3.11, NumPy 2.3.5,
Matplotlib 3.10.8, and pytest 9.1.1: `python -m pytest -q` passed all 43 tests.
Compilation checks also passed. Both complete runner commands below used
population 30, dimensions 10, 100 iterations, and independent seeds 42–46:

```powershell
python -m experiments.run_mpa --benchmark sphere --runs 5 --seed 42 --csv results/sphere.csv --plot results/sphere.png --no-show
python -m experiments.run_mpa --benchmark rastrigin --runs 5 --seed 42 --csv results/rastrigin.csv --plot results/rastrigin.png --no-show
```

Sphere's mean final fitness was `0.00100258489041`, sample standard deviation
`0.000572111644522`, and range `0.000434203116083` to `0.0018258099509`.
Rastrigin's mean was `3.55013661111`, sample standard deviation `2.32975805357`,
and range `1.04036150287` to `7.19970465294`. These modest development runs
demonstrate execution and convergence, not reproduction of paper results.
Each CSV was verified to contain 500 data rows; both plots were visually checked.

Tests check output shapes, finite fitness, monotone convergence, objective
consistency, scalar and vector bounds at every evaluation, fixed dimensions,
seed repeatability, several independent Sphere runs, and all five benchmarks.
Deterministic checks cover movement equations and exact phase boundaries,
the odd population split, Lévy draw order and scaling, both FADs branches,
strict memory comparisons, and the first iteration's second memory reset.
They also check the exact evaluation budget, unevaluated final FADs update,
invalid inputs, and isolation from objective-function input mutation.

The equations and workflow have been checked against the source. MATLAB was
not executed for trajectory comparison. NumPy's Generator and array draw order
produce different random trajectories from MATLAB, even for the same seed;
the intended distributions and random-call stages are preserved. Other Python
interface differences are explicit input validation, rejection of non-finite
objective values, copied objective inputs, named return values, and the
Rosenbrock minimum-dimension check. Memory selection uses `np.where` rather
than mask multiplication, equivalent for finite values. Bounds correction is
batched immediately before each population evaluation; objective inputs are
still visited in the original row order. No paper/source discrepancy is claimed;
this implementation follows the repository as the primary authority.

## Attribution

Original MATLAB programming: Afshin Faramarzi and Seyedali Mirjalili.
Lévy helper credits Hemanth Manjunatha and the Yang/Deb cuckoo-search study.
The upstream copyright, redistribution conditions, and disclaimer are retained
in `references/license.txt` and apply to this source translation.

Faramarzi, A., Heidarinejad, M., Mirjalili, S., and Gandomi, A. H. (2020).
*Marine Predators Algorithm: A Nature-inspired Metaheuristic*, Expert Systems
with Applications. [DOI: 10.1016/j.eswa.2020.113377](https://doi.org/10.1016/j.eswa.2020.113377).
