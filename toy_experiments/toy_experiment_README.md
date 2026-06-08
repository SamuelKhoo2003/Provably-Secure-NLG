# Toy Experiments

This folder contains the synthetic, small-scale certificate experiments for the
project. It was previously called `toy_certificate`; the package and commands
now use `toy_experiments`.

The toy experiments are separate from the large VPA adapter experiments:

- `toy_experiments/` contains synthetic data generation, baselines, MILP
  certificate code, configs, scripts, and toy outputs.
- `large_experiments/` contains the newer large-scale VPA integration work.
- `external/VPA-main/` is the external VPA scaffold used by the large
  experiment track.

Use this folder when you want to run or plot the controlled toy certificate
benchmarks. Use `large_experiments/` for the VPA token-vote export and large
adapter workflow.

## Layout

```text
toy_experiments/
  data.py              Synthetic vote/count data helpers
  baselines.py         Pointwise baseline certificates
  milp.py              MILP certificate formulations
  experiments.py       Main CLI entry point
  csv_io.py            CSV helpers
  configs/             Small, medium, large, and validity-demo configs
  scripts/             Shell wrappers for common workflows
  outputs/             Results and plots for toy runs
```

## Configs

The main benchmark configs are:

```text
toy_experiments/configs/small.yaml
toy_experiments/configs/medium.yaml
toy_experiments/configs/large.yaml
toy_experiments/configs/validity_demo.yaml
toy_experiments/configs/smoke.yaml
```

Each config writes results inside `toy_experiments/outputs/`.

The `solver.gurobi_threads` config key controls Gurobi's internal solver
threads. `0` uses Gurobi automatic mode; positive integers request a fixed
thread limit. `GUROBI_THREADS` can override the YAML value for a run. On shared
machines, avoid blindly setting very high values.

For the full scalability methodology, including model-size growth, solve-count
growth, runtime accounting, stopping statuses, and the current absence of an
explicit Gurobi time limit, see
[`docs/milp_scalability.md`](docs/milp_scalability.md).

Stability is always evaluated against all competing tokens. The old
runner-up-only diagnostic has been removed because report-facing stability is
an untargeted any-token-change property. There is no stability competitor-mode
config or CLI option.

The report-facing comparison is:

- Stability: DPA weakest-token baseline and full shared MILP stability.
- Validity: DPA weakest harmful-token diagnostic, TPA max-token phrase
  baseline, and full shared MILP validity.

For standard `small.yaml`, `medium.yaml`, and `large.yaml` results,
`comparisons.txt` also reports summative DPA validity statistics. Summative DPA
is `independent_val_sequence_qN`: independent shard-aware token attack costs
summed across all token positions and prompts. It is compared with
`row_col_val_qN`, where the shared MILP can reuse poisoned shards across cells.

## Coupled Synthetic Sweeps

Synthetic benchmark sweeps use coupled generation by design; no YAML option is
required. For each fixed `(delta_stab, delta_val, target_bias)` tuple, the
benchmark generates one master instance at the maximum requested `K`, `N`, `L`,
and `T`. Smaller points are derived from that master:

- `K`: take the first `K` shards and recompute vote counts and predictions.
- `N`: take the first `N` prompt rows.
- `L`: take the first `L` token positions.
- `T`: retain candidate ids below `T - 1` and merge removed candidates into
  the last retained id, then recompute candidate-dependent arrays.

This replaces the previous behavior where every Cartesian-product point called
the random generator independently. Under that behavior, changing an array
shape changed the RNG trajectory even with the same seed, so sweep points were
different random worlds rather than nested comparisons.

Distribution sweeps are deliberately more limited. Each distinct
`delta_stab`, `delta_val`, and `target_bias` tuple receives its own coupled
master. These parameters therefore still compare separately generated
distributions; the implementation does not claim latent-random-variable
coupling across them.

Configs may use either a scalar `seed` or a `seed_values` list. The standard
small, medium, and large configs remain single-seed runs. The dedicated
`sweep_K`, `sweep_N`, `sweep_L`, and `sweep_degenerate` configs use:

```yaml
seed_values: [0, 25, 50]
```

Each sweep seed receives its own coupled master instance. Raw seed-specific
rows are written together to `benchmark_results.csv` and
`benchmark_budget_curves.csv`, with `seed` identifying the replicate. Two
additional files provide plotting summaries:

- `benchmark_results_seed_aggregate.csv`
- `benchmark_budget_curves_seed_aggregate.csv`

In these summaries, each numeric measurement column contains the mean across
seeds and has matching `_min`, `_max`, `_minus`, and `_plus` columns.
`_minus = mean - min` and `_plus = max - mean`, making asymmetric error bars
directly available. Sweep plots use the mean line and shade the min-to-max
range. Budget curves preserve each seed/configuration as a separate group
before aggregation.

Coupling improves comparability but does not make every metric monotonic:

- Full-sequence validity is nondecreasing with `L` because the attack must
  force every position in a longer exact prefix.
- Any-token stability is nonincreasing with `L` because the attacker has more
  positions available.
- All-token stability is nondecreasing with `L`.
- For `N`, attacking all prompts becomes harder or unchanged, while attacking
  at least one prompt can become easier.
- `K` is not generally monotonic because ensemble size and margins both change.
- Increasing `T` adds candidate competitors, but validity also depends on the
  target representation and candidate projection.
- Delta and target-bias directions depend on the synthetic distribution.

The validity demo generates a master at `L_master` and a sufficiently large
`K_master`. For a derived point,
`K_actual = max(K_requested, min_required_shards(L))`, where
`min_required_shards = group_size + (L - 1) * (group_size - overlap)`.
Heterogeneous useful-shard groups are restricted to the prefix available at
each position, so slicing does not discard an intended group.

New result and budget-curve CSVs include `coupled_generation`, `K_requested`,
`K_actual`, `K_master`, `N_master`, `L_master`, and `T_master`. Plot readers
remain compatible with historical CSVs that lack these columns.

Report-facing caveat: The synthetic sweeps are generated using a coupled master
instance, so changes in `K`, `N`, `L`, or `T` are evaluated on nested variants
of the same underlying vote structure. This isolates each scaling parameter
more clearly than independently regenerating each point.

## Run A Benchmark

Run the small benchmark:

```bash
CONFIG=toy_experiments/configs/small.yaml ./toy_experiments/scripts/data.sh
```

Run the medium benchmark:

```bash
CONFIG=toy_experiments/configs/medium.yaml ./toy_experiments/scripts/data.sh
```

Run the large benchmark:

```bash
CONFIG=toy_experiments/configs/large.yaml ./toy_experiments/scripts/data.sh
```

For a dry run that checks the config and prints the estimated number of solves
without running Gurobi:

```bash
CONFIG=toy_experiments/configs/small.yaml DRY_RUN=1 ./toy_experiments/scripts/data.sh
```

With `VERBOSE=1`, the dry run also prints the master dimensions and every
derived instance. Validity-demo dry runs show `K_requested`, `K_actual`, and
`min_required_shards`.

## Plot Results

Plot all available small, medium, and large benchmark outputs:

```bash
./toy_experiments/scripts/plot.sh
```

The plot script checks for:

```text
toy_experiments/outputs/small/results/benchmark_results.csv
toy_experiments/outputs/medium/results/benchmark_results.csv
toy_experiments/outputs/large/results/benchmark_results.csv
```

When those files exist, it writes plots to:

```text
toy_experiments/outputs/small/plots/
toy_experiments/outputs/medium/plots/
toy_experiments/outputs/large/plots/
```

All generated plots are report-ready PDF files. Regenerating a default plot
directory replaces the existing generated PDF set.

Each plot directory also gets `audit_plot_outputs.txt` and `comparisons.txt`.
`comparisons.txt` is derived from the existing CSV outputs and reports stability
and validity relative-lift summaries such as MILP vs DPA and MILP vs TPA. Running
`plot.sh` regenerates it without rerunning Gurobi or regenerating experiment CSVs.

To plot one specific CSV:

```bash
CSV_PATH=toy_experiments/outputs/small/results/benchmark_results.csv ./toy_experiments/scripts/plot.sh
```

## Smoke Check

Run a lightweight visualization smoke check:

```bash
./toy_experiments/scripts/smoke.sh
```

This writes smoke outputs under:

```text
toy_experiments/outputs/smoke/
```

## Validity Demo

The validity demo uses its own config:

```text
toy_experiments/configs/validity_demo.yaml
```

Run the full validity demo workflow:

```bash
./toy_experiments/scripts/validity_demo.sh
```

This generates validity-demo benchmark results and plots under:

```text
toy_experiments/outputs/validity_demo/results/
toy_experiments/outputs/validity_demo/plots/
```

Each run prints the active config path and its `K`, `N`, `L`, and `T` values.
Plotting also writes tables derived from the actual result CSV:

```text
toy_experiments/outputs/validity_demo/plots/validity_demo_parameters.md
toy_experiments/outputs/validity_demo/plots/validity_demo_parameters.csv
```

These tables list every observed `(K, N, L, T)` combination and its MILP
statuses, which makes stale results or an overridden `CONFIG` visible.

## Python CLI

The direct Python entry point is:

```bash
python -m toy_experiments.experiments
```

Common examples:

```bash
python -m toy_experiments.experiments visualize --save-dir toy_experiments/outputs/smoke
python -m toy_experiments.experiments benchmark --config toy_experiments/configs/small.yaml --dry-run
python -m toy_experiments.experiments plot-csv --csv toy_experiments/outputs/small/results/benchmark_results.csv --save-dir toy_experiments/outputs/small/plots
```

The shell scripts use `toy_experiments/scripts/_python.sh` to resolve the
project Python environment.
