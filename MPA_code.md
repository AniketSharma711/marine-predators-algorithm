# Implementation Brief: Marine Predators Algorithm (MPA) in Python



## 1\. Objective

Implement a faithful Python version of the **Marine Predators Algorithm
(MPA)**, using the original MATLAB implementation as the primary
reference:

* Reference repository:
https://github.com/afshinfaramarzi/Marine-Predators-Algorithm
* Main reference file: `MPA.m`
* Also inspect and translate relevant helper files, especially
`initialization.m`, `levy.m`, and `Get\\\_Functions\\\_details.m`.
* Original paper: Faramarzi et al., *Marine Predators Algorithm: A
Nature-inspired Metaheuristic*, Expert Systems with Applications
(2020), https://doi.org/10.1016/j.eswa.2020.113377



The goal is to produce a clear, testable, reproducible implementation
suitable for a student research project. **Prioritize fidelity to the
reference MATLAB implementation over adding new features or modifying
the algorithm.**



## 2\. Before writing code

1. Inspect the reference repository and read the full contents of
`MPA.m`, `initialization.m`, and `levy.m`.
2. Record the reference algorithm's equations, phase conditions,
iteration order, boundary handling, memory mechanism, and FADs
effect.
3. Do not infer the missing details of the implementation from
secondary descriptions when the source code is available.
4. If the repository or a helper file cannot be accessed, clearly
report that limitation rather than inventing its behavior.



## 3\. Recommended project structure

Use the following structure, adapting only if the existing project
already has a suitable organization:

``` text
project/
├── algorithms/
│   └── mpa.py
├── benchmarks/
│   └── functions.py
├── experiments/
│   └── run\\\_mpa.py
├── tests/
│   └── test\\\_mpa.py
├── results/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

Keep the core algorithm separate from benchmark definitions, experiment
configuration, plotting, and result export.



## 4\. MPA function interface

Create a reusable function with an interface equivalent to:

``` text
MPA(population\\\_size, max\\\_iterations, lower\\\_bound, upper\\\_bound, dimensions, objective\\\_function, seed=None)
```

Choose idiomatic Python naming, such as `mpa(...)`, if appropriate.

The implementation should return at least:

* `best\\\_position`: best solution vector found.
* `best\\\_fitness`: objective-function value of that solution.
* `convergence\\\_curve`: best-so-far fitness recorded across iterations.



Document the input and output shapes. Support scalar bounds and, where
practical, per-dimension bound arrays. Validate incompatible dimensions
and invalid parameter values with useful error messages.

Use NumPy for numerical operations. Use `numpy.random.Generator` for
controlled randomness if it can be done without changing the algorithm's
distributions or operation semantics unnecessarily. Passing a seed
should make a run reproducible.



## 5\. Required algorithm workflow

Translate the reference implementation in its original order. At a
minimum, verify and implement each of these parts:

1. **Initialization**

   * Initialize the prey population within the specified bounds using
the reference initialization procedure.
   * Initialize the elite/top-predator position and fitness, the
convergence curve, and the variables used by the memory
mechanism.
2. **Fitness evaluation and elite update**

   * Evaluate each candidate with the supplied objective function.
   * Update the best-so-far elite solution whenever a better fitness
value is found.
   * Preserve a copy of the elite position; avoid accidental array
aliasing.
3. **Boundary handling**

   * Apply the reference implementation's boundary correction at the
same points in the iteration.
   * Ensure candidates remain within the specified lower and upper
bounds.
   * Do not silently replace the reference handling with a different
clipping or reflection strategy.
4. **Memory mechanism**

   * Translate the original memory mechanism for retaining better
previous candidates.
   * Preserve its evaluation order and comparison logic.
   * Make sure stored populations and fitness arrays are copied where
necessary.
5. **Movement phases**

   * Implement all three MPA movement phases using the exact
equations and phase conditions from `MPA.m`.
   * Translate the Brownian-motion and Lévy-flight updates
faithfully.
   * Preserve the reference use of `RB`, `RL`, `P`, `R`, and the
control factor `CF`, including the way random arrays are
generated and applied.
   * Do not replace the original equations with simplified versions.
6. **Lévy-flight generation**

   * Translate `levy.m` after inspecting its exact formula and
parameters.
   * Preserve the distribution and scaling used by the reference.
   * Check output dimensions and ensure the generated values are
compatible with the update equations.
7. **Fish Aggregating Devices (FADs) effect**

   * Implement the FADs/non-FADs branch using the reference code's
probability, random matrices, permutation logic, and scaling.
   * Preserve the order in which this operation is applied relative
to the movement phases and fitness/memory updates.
8. **Iteration and convergence**

   * Run the full loop for the requested number of iterations.
   * Record the best-so-far objective value at the appropriate point
in each iteration.
   * Return a convergence curve whose length and indexing are clearly
documented.



## 6\. Important MATLAB-to-Python translation details

* MATLAB uses **1-based indexing**; Python uses **0-based indexing**.
Translate population-half conditions carefully rather than copying
index expressions literally.
* Preserve strict versus inclusive phase conditions exactly as written
in the MATLAB source. For example, conditions using `>` and `<`
should not be changed to `>=` or `<=` without a documented reason.
* MATLAB's matrix operations and implicit expansion may differ from
NumPy broadcasting. Check array shapes explicitly.
* Preserve the intended population-by-dimension layout consistently.
* Translate random-number calls carefully. The same seed will not
necessarily produce the same random sequence in MATLAB and NumPy;
the requirement is faithful distributions and reproducibility within
Python, not identical random trajectories across languages.
* Keep the algorithm's objective-function convention consistent: this
implementation is for **minimization**, unless the reference code
explicitly supports otherwise.
* Handle vector bounds correctly if they are supported by the chosen
interface.
* Avoid unnecessary refactoring that changes when fitness is
evaluated, when the elite is updated, or when random arrays are
sampled.



## 7\. Initial benchmark functions

Create independent benchmark functions, each accepting a one-dimensional
NumPy vector and returning a scalar:

* **Sphere:** `f(x) = sum(x\\\_i^2)`. Global minimum is `0` at the
all-zero vector.
* **Rastrigin**
* **Rosenbrock**
* **Ackley**
* **Griewank**



Use standard, documented formulas and bounds. Start validation with
Sphere because its optimum is known and straightforward to check. Do not
hard-code a particular benchmark inside the MPA implementation; pass the
objective function as an argument.



## 8\. Testing and validation

Add tests or a validation script that checks:

* The returned position has the expected dimension.
* The returned best fitness is a finite scalar for valid benchmark
inputs.
* The convergence curve has the documented length and contains valid
values.
* The best-so-far curve does not worsen over time, allowing for
floating-point tolerance.
* Candidate positions remain within bounds during optimization; if
needed, add optional internal diagnostics or tests around the
boundary-handling function.
* A fixed seed produces repeatable results within the Python
implementation.
* Sphere optimization can approach its known optimum under reasonable
settings.
* The implementation runs on at least one additional benchmark, such
as Rastrigin or Rosenbrock.



Because MPA is stochastic, do not expect every run to reach the exact
global optimum. Use multiple independent runs when reporting
performance. Do not write tests that assume one exact final fitness
value for every seed.

For a quick development check, use modest settings such as a population
of 30, 10 dimensions, and 100 iterations. These are **development
settings**, not a claim about the original paper's experimental
configuration. Keep parameters configurable.



## 9\. Experiment runner

Create a small runnable experiment script that:

1. Selects a benchmark function.
2. Defines dimensions, bounds, population size, iteration count, and
seed.
3. Runs MPA.
4. Prints the best fitness and best position.
5. Plots the convergence curve with Matplotlib.
6. Optionally saves results to a CSV file under `results/`.

Keep plotting and file output outside the core algorithm.



## 10\. Code quality and documentation

* Use clear function names, type hints where useful, and concise
docstrings.
* Comment the equations and major algorithm stages, not every line.
* Explain the purpose of the three phases, elite matrix, Brownian
motion, Lévy flight, memory mechanism, control factor, and FADs
effect in comments or documentation.
* Include installation and run instructions in `README.md`.
* Keep dependencies minimal: NumPy and Matplotlib should be sufficient
for the initial implementation; use pytest only if tests are added.
* Do not add unrelated packages, GPU support, or premature
optimization.
* Do not implement SSA or Quantum-Swarm in this task unless explicitly
requested later.



## 11\. Scope boundaries

For this task:

* Implement **vanilla MPA** only.
* Do not add quantum-inspired candidate generation, adaptive
exploration/exploitation, diversity monitoring, or stagnation
detection. Those belong to the planned Quantum-Swarm work and should
remain separate.
* Do not claim that the implementation is identical to the MATLAB
implementation until the equations and workflow have been checked
against the source.
* Do not silently change algorithm behavior to improve benchmark
results.
* If a discrepancy between the paper and MATLAB code is found,
document it and follow the MATLAB repository as the primary
implementation reference unless instructed otherwise.



## 12\. Expected deliverables

Provide:

1. `algorithms/mpa.py` --- core MPA implementation.
2. `benchmarks/functions.py` --- benchmark functions.
3. `experiments/run\\\_mpa.py` --- example experiment and convergence
plot.
4. `tests/test\\\_mpa.py` --- basic correctness and reproducibility tests.
5. `requirements.txt` --- minimal dependencies.
6. `README.md` --- setup, usage, parameters, return values, and
reference attribution.



At completion, summarize: - Which reference files were inspected. - How
each major MPA component maps to the source implementation. - Which
tests and benchmarks were run. - Any deviations from the MATLAB
reference and why they were necessary.

Do not report tests as passing unless they were actually executed.

