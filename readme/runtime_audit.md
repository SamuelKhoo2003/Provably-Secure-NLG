# Large Benchmark Runtime Audit

Static audit date: 2026-05-29. This report was produced by reading the
benchmark/runtime code only; no large benchmark or Gurobi solve was run.

## Summary

The large config is slow because each benchmark instance runs the main
certificate MILPs, additional per-row shared MILPs for radius-derived budget
curves, and direct damage-at-budget MILPs for every clipped budget. With the
current config, that expands to 4 benchmark instances and an estimated 508
Gurobi solves. The most expensive part is the direct damage curve loop:
3 fixed-budget MILPs are rebuilt and solved for every budget.

The run can appear frozen around `K=40, N=6, L=10, T=12` because Gurobi output
is disabled and the benchmark prints only after an entire instance completes.
There is no configured time limit on the Gurobi models, so one hard MILP can run
silently for a very long time.

## Parameter Expansion

The benchmark loop expands:

- `K_values`: 2 values, `20, 40`
- `N_values`: 2 values, `6, 10`
- `L_values`: 1 value, `10`
- `T_values`: 1 value, `12`
- `delta_stab_values`: 1 value, `0.25`
- `delta_val_values`: 1 value, `0.25`
- `target_bias_values`: 1 value, `0.3`
- `seed`: one integer seed, not a seed list

Expanded grid size: `2 * 2 * 1 * 1 * 1 * 1 * 1 = 4` instances.

Budget handling is clipped correctly in `benchmark_scale`:

```python
budgets = list(range(0, min(K_actual, budget_max) + 1))
```

So with `budget_max=50`:

- `K=20`: budgets `0..20`, 21 budgets
- `K=40`: budgets `0..40`, 41 budgets

No accidental `0..50` budget loop was found in the main benchmark path.

## Estimated Gurobi Solve Counts

Per instance, `_solve_benchmark_certificates` launches 10 main certificate
solves:

- 6 stability certificates: row, column, and structured `(q,r)` values
  `(1,1)`, `(1,L)`, `(N,1)`, `(N,L)`
- 4 validity certificates: row, column, `q=1`, `q=N`

When budget curves are enabled, `compute_radius_derived_budget_curve_rows`
launches per-row shared MILP radii:

- `N` stability row radii for `r=1`
- `N` stability row radii for `r=L`
- `N` validity row radii
- total: `3N` additional MILP solves per instance

When direct damage curves are enabled, `compute_direct_damage_curve_rows`
launches:

- stability, one token per prompt
- stability, full sequence per prompt
- validity, full harmful sequence per prompt
- total: `3 * budget_count` fixed-budget MILP solves per instance

Estimated solve table:

| K | N | budgets | main cert solves | per-row radius solves | direct damage solves | total |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 6 | 21 | 10 | 18 | 63 | 91 |
| 20 | 10 | 21 | 10 | 30 | 63 | 103 |
| 40 | 6 | 41 | 10 | 18 | 123 | 151 |
| 40 | 10 | 41 | 10 | 30 | 123 | 163 |
| total | | | 40 | 96 | 372 | 508 |

Dominant category: direct damage curves, 372 of 508 solves.

Horizon curves and reference baselines do not call Gurobi, but they recompute
some non-MILP baseline arrays that were already computed elsewhere.

## Model Size Growth

In all-competitor stability mode, each stability model creates:

- poisoned shard binaries: `K`
- attacked cell binaries: `N * L`
- competitor binaries: `N * L * (T - 1)`
- row/column helper binaries depending on objective
- competitor constraints with sums over all `K` shards

For `K=40, N=6, L=10, T=12`:

- shard variables: 40
- cell variables: 60
- competitor variables: 660
- total stability variables before row helpers: about 760
- all-competitor stability constraints before objective helpers:
  `2 * 60 * 11 + 60 = 1,380`
- each competitor inequality has dense sums over up to all 40 shards

For `K=40, N=10, L=10, T=12`:

- shard variables: 40
- cell variables: 100
- competitor variables: 1,100
- total stability variables before row helpers: about 1,240
- all-competitor stability constraints before objective helpers:
  `2 * 100 * 11 + 100 = 2,300`
- dense coefficient growth is roughly proportional to
  `K * N * L * (T - 1)`

Validity models do not create separate competitor binaries, but they still add
one dense constraint per cell and non-target competitor:

- `K=40, N=6, L=10, T=12`: 660 validity competitor constraints
- `K=40, N=10, L=10, T=12`: 1,100 validity competitor constraints

Important cost drivers:

- all-competitor stability is exact but much larger than `runner_up`
- dense influence makes every competitor constraint involve all shards
- models are rebuilt for every objective and every damage budget
- direct damage maximization repeats very similar full-grid models many times
- main structured objectives build full-grid constraints even for objectives
  that only require one row or one token

Per-row radius helpers slice to one row, so they do not build the full `N` rows,
but there are `3N` of them per instance.

## Time Limits and Status Handling

No Gurobi time limit is set in `_make_model` or `_make_budget_model`. The large
config also does not specify a time limit, and there is no CLI/config plumbing
for one in the audited path.

If Gurobi returns `TIME_LIMIT` or `SUBOPTIMAL`, the result objects do record
`is_optimal=False`, bounds, status names, and MIP gaps where available. Direct
damage rows mark non-optimal values as `feasible_attacked_lower_bound`, and plot
code skips non-optimal direct-damage rows for exact curves.

However, because no `TimeLimit` parameter is actually set, a hard direct damage
or certificate solve can run indefinitely unless stopped externally.

## Logging and Progress

Gurobi `OutputFlag` is disabled. The benchmark prints one `bench ...` line only
after main certificates, budget curves, damage curves, and horizon curves for
the instance have all completed.

There is no progress line before each instance, before each objective, or before
each direct damage budget solve. There is also no per-solve runtime/status line.
So a long solve is probably not a Python freeze; it is likely a silent Gurobi
optimization.

## Expensive Defaults

`scripts/data.sh` defaults missing config keys to enabled:

- `MAKE_BUDGET_CURVES=1`
- `MAKE_DAMAGE_CURVES=1`
- `MAKE_HORIZON_CURVES=1`

For the large config, `stability_competitor_mode: all`, dense influence, and
direct damage curves dominate runtime. Atomic phrase aggregation and independent
composition are included in radius-derived budget rows, but they are non-Gurobi
baseline computations and are not the main bottleneck.

`scripts/plot.sh` reads existing CSVs and does not trigger data generation.

## Repeated Work and Simple Opportunities

Repeated or avoidable work found:

- direct damage curves rebuild nearly identical models for every budget
- baseline arrays are recomputed in `compute_reference_baselines`,
  `compute_radius_derived_budget_curve_rows`, and horizon generation
- per-row shared MILP radii are separate from the main full-instance
  certificates, which is mathematically different but still adds `3N` solves
- no dry-run estimator exists in the CLI

Simple non-invasive improvements:

- add a dry-run command that expands the grid and prints estimated solve counts
- add progress logging before each instance/objective/budget solve, with flush
- add configurable `TimeLimit` plumbing and record it in CSV metadata
- disable direct damage curves by default for large runs
- keep exact `all` competitor mode for final selected runs, but use
  `runner_up` only for approximate sweeps
- cache non-Gurobi baseline arrays per instance and pass them to curve helpers

## Safer Large-Fast Config

For a quick audit-scale run, prefer a config or environment override like:

```yaml
preset: large-fast

K_values: [20, 40]
N_values: [6, 10]
L_values: [10]
T_values: [12]

delta_stab_values: [0.25]
delta_val_values: [0.25]
target_bias_values: [0.3]

seed: 0
budget_max: 15

influence_mode: dense
stability_competitor_mode: runner_up

make_budget_curves: 1
make_damage_curves: 0
make_horizon_curves: 1
```

For exact final runs, switch `stability_competitor_mode` back to `all` and run
only selected `(K,N,L,T)` points with a solver time limit and progress logging.
