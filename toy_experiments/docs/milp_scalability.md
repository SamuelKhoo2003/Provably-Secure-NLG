# MILP Scalability and Gurobi Configuration

This document explains how the toy certificate experiments control and measure
the computational cost of the Gurobi mixed-integer linear programs (MILPs). It
is intended to support the scalability methodology in the project report.

## What Is Being Solved

For each generated instance, the shared certificate minimizes the number of
poisoned shards:

```text
minimise sum_k a[k]
```

where `a[k]` is a binary variable indicating whether shard `k` is poisoned.
The optimum `B*` is the minimum attack budget that can violate the requested
stability or validity property. A budget `B` is certified when `B < B*`.

The benchmark can solve:

- four stability objectives: `(q=1,r=1)`, `(q=1,r=L)`, `(q=N,r=1)`, and
  `(q=N,r=L)`;
- two validity objectives: one harmful sequence (`q=1`) and all prompt rows
  (`q=N`);
- optional per-row solves used to derive certified-fraction budget curves.

Stability always considers every competitor token other than the clean winner.

## Why the MILPs Grow

The principal dimensions are:

- `K`: number of shard models;
- `N`: number of prompt rows;
- `L`: generated tokens per row;
- `T`: vocabulary size represented in the toy vote tensor.

The stability MILP contains approximately:

- `K` shard-selection variables;
- `N*L` attacked-cell variables;
- `N*L*(T-1)` competitor-selection variables;
- `N` row-selection variables.

Its competitor constraints contain shard-dependent expressions, so the number
of nonzero coefficients grows approximately as `O(K*N*L*T)`.

The validity MILP uses fewer binary variables because it does not need a
separate competitor-selection variable for every token. However, it still
compares each harmful target against every other token at every cell, producing
approximately `N*L*(T-1)` main constraints with `O(K*N*L*T)` coefficient
growth.

This is why increasing any of `K`, `N`, `L`, or `T` can increase runtime, and
why joint increases are substantially harder than changing one dimension in
isolation. MILP runtime is not expected to follow a smooth polynomial curve:
branch-and-bound difficulty also depends on the generated vote structure and
the strength of the linear relaxation.

## Gurobi Threads

Each model sets Gurobi's `Threads` parameter:

```yaml
solver:
  gurobi_threads: 4
```

The accepted values are:

- `0`: Gurobi automatic thread selection;
- a positive integer: maximum number of threads available to that solve;
- a negative value: rejected during validation.

For benchmark runs, the precedence is:

1. `GUROBI_THREADS` environment variable;
2. `solver.gurobi_threads` in YAML;
3. default `0`.

For a direct Python solver call, an explicit `gurobi_threads` argument has
priority over the environment variable.

Examples:

```bash
# Use the YAML setting.
CONFIG=toy_experiments/configs/medium.yaml ./toy_experiments/scripts/data.sh

# Override the YAML setting for this run.
GUROBI_THREADS=4 CONFIG=toy_experiments/configs/medium.yaml \
  ./toy_experiments/scripts/data.sh
```

`Threads` limits parallelism inside one Gurobi solve. The benchmark driver
itself runs instances and objectives sequentially; it does not launch multiple
Gurobi models concurrently. A high thread value therefore does not multiply by
several simultaneous benchmark workers, but it can still oversubscribe a
shared machine or provide little benefit on small MILPs.

The current presets use:

| Config | Threads |
| --- | ---: |
| `smoke.yaml` | automatic (`0`) |
| `small.yaml` | automatic (`0`) |
| `medium.yaml` | `8` |
| `large.yaml` | `32` |
| `sweep_K.yaml`, `sweep_N.yaml`, `sweep_L.yaml` | `4` |
| `sweep_degenerate.yaml` | automatic (`0`) |
| `validity_demo.yaml` | `4` |

For fair runtime comparisons, all points within a sweep use the same thread
limit. Runtime results from different machines or different thread settings
should not be treated as directly comparable.

## Time Limits and Other Stopping Criteria

The current repository does **not** set:

- Gurobi `TimeLimit`;
- `NodeLimit`;
- `SolutionLimit`;
- a non-default target `MIPGap`.

Consequently, normal benchmark runs use Gurobi's defaults and are intended to
continue until optimality or another solver-level termination condition.

The result-handling code understands statuses including `TIME_LIMIT`,
`NODE_LIMIT`, `SUBOPTIMAL`, `INTERRUPTED`, and `NUMERIC`. It also records the
incumbent objective, objective bound, and MIP gap when Gurobi exposes them.
This status support does not mean a time limit is currently configured.

This distinction matters in the report:

- `is_optimal=true` means Gurobi proved the reported `B*` optimal;
- `upper_bound` is the best feasible attack budget found for this minimization
  problem;
- `lower_bound` is Gurobi's best bound on the unknown optimum;
- `mip_gap` measures the remaining separation between incumbent and bound;
- a non-optimal incumbent must not be presented as an exact certificate
  threshold.

The main benchmark CSV can store an incumbent from `TIME_LIMIT` or
`SUBOPTIMAL`, accompanied by `is_optimal=false`. Report analysis should filter
or clearly mark these rows. Per-row budget-curve construction is stricter:
only optimal solves are used, while non-optimal rows become unknown values.

If a report experiment later introduces a time limit, it should be added as an
explicit YAML field, passed to every model, written into CSV metadata, and held
fixed across compared sweep points. That has not been implemented yet.

## Controlling the Number of Solves

Solver cost depends on both model size and the number of models constructed.
For one instance:

| Enabled work | Gurobi solves |
| --- | ---: |
| Four report-facing stability objectives | `4` |
| Two report-facing validity objectives | `2` |
| Stability budget curves | `2*N` |
| Validity budget curves | `N` |

Therefore:

- `objective_family: full` with budget curves uses `6 + 3*N` solves per
  instance;
- `objective_family: full` without budget curves uses `6` solves;
- `objective_family: validity_only` with validity curves uses `2 + N` solves;
- `objective_family: validity_only` without curves uses `2` solves.

These counts explain why increasing `N` can raise total experiment time in two
ways: each MILP becomes larger, and optional per-row curves create more MILPs.

The main controls are:

```yaml
objective_family: full       # or validity_only
make_budget_curves: true     # false skips all budget-curve solves
```

The following optional fields override the family defaults:

```yaml
make_stability_objectives: false
make_validity_objectives: true
make_stability_budget_curves: false
make_validity_budget_curves: true
```

The validity demo uses these controls to avoid stability models that are not
part of that experiment's research question.

## `budget_max` Does Not Add MILP Solves

`budget_max` controls the range of budget values written to the certified
fraction curve:

```yaml
budget_max: 20
```

The evaluated budgets are:

```text
0, 1, ..., min(K, budget_max)
```

The MILP is solved once to obtain a radius, and all curve points are derived
from that radius. Increasing `budget_max` increases CSV rows and plotting work,
but does not create one Gurobi solve per budget.

`budget_plot_num_points` similarly downsamples plotted budget points for the
validity demo. It does not reduce the number or size of MILP solves.

## Dry Runs Before Expensive Experiments

Every benchmark should first be expanded as a dry run:

```bash
CONFIG=toy_experiments/configs/medium.yaml DRY_RUN=1 VERBOSE=1 \
  ./toy_experiments/scripts/data.sh
```

For all scalability sweeps:

```bash
MODE=dry-run ./toy_experiments/scripts/sweep_benchmark.sh
```

The dry run:

- validates the YAML;
- prints the expanded `K`, `N`, `L`, and `T` grids;
- reports which objective families and curves are enabled;
- estimates the number of stability and validity Gurobi solves;
- does not instantiate or optimize Gurobi models.

This catches accidental Cartesian-product expansions before compute is spent.

## Runtime Measurement

Each benchmark row records:

```text
runtime_gurobi_total
```

This is wall-clock time measured around the primary objective solves selected
by the objective flags: up to four stability solves and two validity solves. It
includes Gurobi model construction and optimization for those objectives.

It currently excludes:

- synthetic data generation;
- analytical DPA/TPA baselines;
- per-row budget-curve MILP solves;
- CSV writing;
- plotting.

Therefore the sweep runtime plot measures primary certificate objective cost,
not full end-to-end benchmark wall time when budget curves are enabled.

Plots are generated from existing CSVs and do not rerun Gurobi:

```bash
MODE=plot SWEEP=K ./toy_experiments/scripts/sweep_benchmark.sh
```

## Recorded Solver Evidence

For every primary certificate metric, the CSV stores:

```text
<metric>
<metric>_status
<metric>_is_optimal
<metric>_lower_bound
<metric>_upper_bound
<metric>_mip_gap
```

Examples include `row_col_stab_qN_rL_status` and
`row_col_val_qN_is_optimal`. These fields make it possible to audit whether a
runtime result corresponds to a proven optimum.

The sweep plotting workflow also writes `audit_sweep.md`, which reports:

- varied and fixed parameters;
- included and missing metrics;
- observed solver statuses;
- generated plot files.

## How Larger Runs Are Kept Manageable

The current workflow uses the following safeguards:

1. Change one main dimension at a time in the `K`, `N`, and `L` sweeps.
2. Keep `T=5` fixed in the standard scalability sweeps.
3. Use a fixed four-thread limit across standard sweep points.
4. Dry-run every grid to expose instance and solve counts.
5. Disable irrelevant objective families, as in `validity_only`.
6. Allow budget curves to be disabled because they add `O(N)` separate solves.
7. Separate data generation from plotting so plots can be revised without
   rerunning Gurobi.
8. Record status, optimality, bounds, and gap for auditability.
9. Use smoke, small, medium, and large presets rather than jumping directly to
   the largest grid.

These measures control experimental workload, but they do not change the
worst-case complexity of the MILPs. The project demonstrates empirical
scalability over the configured ranges; it does not claim that arbitrary
production-scale vocabularies or sequence lengths are tractable.

## Recommended Report Wording

The following summary is consistent with the implementation:

> Scalability was evaluated by varying the number of shards, prompt rows, and
> generated token positions independently while holding the remaining
> dimensions and data-generation parameters fixed. Gurobi's thread limit was
> fixed across each sweep, and every primary solve recorded its termination
> status, optimality flag, objective bounds, and MIP gap. Before execution, a
> dry-run stage expanded the parameter grid and estimated the total number of
> MILP solves. Optional objective families and per-row budget curves could be
> disabled to avoid solving models unrelated to a given experiment. No
> explicit solver time limit was used in the reported implementation, so exact
> certificate values were identified by `is_optimal=true`; non-optimal
> incumbents were not treated as proven thresholds.

## Relevant Files

- `toy_experiments/milp.py`: model construction, threads, optimization, and
  status extraction.
- `toy_experiments/experiments.py`: benchmark grids, objective selection,
  solve-count estimation, runtime measurement, and CSV output.
- `toy_experiments/configs/`: dimensions, objective flags, budget ranges, and
  thread settings.
- `toy_experiments/scripts/data.sh`: config-driven benchmark entry point.
- `toy_experiments/scripts/sweep_benchmark.sh`: dry-run, data, and plotting
  workflow for scalability sweeps.
