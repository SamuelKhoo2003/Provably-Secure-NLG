# Toy Experiments

`toy_experiments/` contains the synthetic certificate benchmarks used to study
the DPA, TPA, and shared MILP strategies on controlled vote grids. It is separate
from the large VPA adapter workflow under `large_experiments/`.

Use this package when you want to generate toy vote data, run certificate
benchmarks, plot existing CSV outputs, or run the controlled validity demo.

## Layout

```text
toy_experiments/
  data.py              Synthetic vote/count data generation
  baselines.py         DPA and TPA baseline certificates
  milp.py              Shared-shard MILP certificate formulations
  experiments.py       Python CLI and plotting implementation
  csv_io.py            CSV read/write helpers
  configs/             Benchmark and demo YAML configs
  scripts/             Shell wrappers for common workflows
  outputs/             Generated results and plots
```

## Configs

Standard benchmark configs:

```text
toy_experiments/configs/small.yaml
toy_experiments/configs/medium.yaml
toy_experiments/configs/large.yaml
```

Other supported configs:

```text
toy_experiments/configs/smoke.yaml
toy_experiments/configs/validity_demo.yaml
toy_experiments/configs/sweep_K.yaml
toy_experiments/configs/sweep_N.yaml
toy_experiments/configs/sweep_L.yaml
toy_experiments/configs/sweep_degenerate.yaml
```

Each config writes results under `toy_experiments/outputs/`. The
`solver.gurobi_threads` key controls Gurobi's internal solver threads:

- `0` leaves Gurobi in automatic thread mode.
- A positive integer requests that fixed thread limit.
- `GUROBI_THREADS` overrides the YAML value for one run.

Stability is always treated as an untargeted any-token-change property against
all competing tokens. There is no runner-up-only stability mode.

## Standard Benchmarks

Run a benchmark by passing a config to `data.sh`:

```bash
CONFIG=toy_experiments/configs/small.yaml ./toy_experiments/scripts/data.sh
CONFIG=toy_experiments/configs/medium.yaml ./toy_experiments/scripts/data.sh
CONFIG=toy_experiments/configs/large.yaml ./toy_experiments/scripts/data.sh
```

Dry-run a config without invoking Gurobi:

```bash
CONFIG=toy_experiments/configs/small.yaml DRY_RUN=1 ./toy_experiments/scripts/data.sh
```

Add `VERBOSE=1` to print the full derived instance grid.

Standard runs write:

```text
toy_experiments/outputs/<size>/results/benchmark_results.csv
toy_experiments/outputs/<size>/results/benchmark_budget_curves.csv
```

`benchmark_budget_curves.csv` is the source for the report-facing budget-curve
plots. `benchmark_results.csv` is still written as the benchmark manifest and
for summary statistics such as summative DPA versus shared MILP validity.

## Standard Plots

Plot all available standard benchmark outputs:

```bash
./toy_experiments/scripts/plot.sh
```

Or plot one CSV explicitly:

```bash
CSV_PATH=toy_experiments/outputs/small/results/benchmark_results.csv ./toy_experiments/scripts/plot.sh
```

For `small`, `medium`, and `large`, the default plot set is intentionally small:

```text
stability_budget_curve.pdf
validity_budget_curve.pdf
audit_plot_outputs.txt
comparisons.txt
```

`stability_budget_curve.pdf` plots these series together:

- DPA weakest token
- Shared MILP full sequence
- Shared MILP one token per row

`validity_budget_curve.pdf` plots:

- Shared shard-aware MILP full sequence
- TPA max-token phrase blocker
- Plain DPA max-token phrase blocker

`comparisons.txt` is generated from existing CSVs only. Plotting does not rerun
Gurobi or regenerate benchmark data.

## Coupled Generation

Synthetic benchmark grids use coupled generation. For each fixed
`(delta_stab, delta_val, target_bias, seed)` group, the benchmark generates one
master instance at the maximum requested `K`, `N`, `L`, and `T`. Smaller grid
points are slices of that master:

- `K`: take the first `K` shards and recompute counts and predictions.
- `N`: take the first `N` prompt rows.
- `L`: take the first `L` token positions.
- `T`: keep candidate ids below `T - 1`, merge removed ids into the last
  retained id, and recompute candidate-dependent arrays.

This makes scaling points nested variants of the same synthetic world instead
of independent random draws. Coupling improves comparability, but it does not
make every metric monotonic.

CSV outputs include coupling metadata:

```text
coupled_generation
K_requested
K_actual
K_master
N_master
L_master
T_master
seed
```

## Sweep Benchmarks

Run all dedicated scaling sweeps:

```bash
./toy_experiments/scripts/sweep_benchmark.sh
```

Useful variants:

```bash
MODE=dry-run ./toy_experiments/scripts/sweep_benchmark.sh
MODE=data SWEEP=K ./toy_experiments/scripts/sweep_benchmark.sh
MODE=plot SWEEP=L ./toy_experiments/scripts/sweep_benchmark.sh
```

Supported `SWEEP` values are `K`, `N`, `L`, `degenerate`, and `all`.
Supported `MODE` values are `dry-run`, `data`, `plot`, and `all`.

The sweep configs use five seeds:

```yaml
seed_values: [0, 10, 20, 30, 40]
```

Raw seed-specific rows are written to:

```text
toy_experiments/outputs/sweep_benchmark/<sweep>/results/benchmark_results.csv
toy_experiments/outputs/sweep_benchmark/<sweep>/results/benchmark_budget_curves.csv
```

Aggregated seed summaries are written beside them:

```text
benchmark_results_seed_aggregate.csv
benchmark_budget_curves_seed_aggregate.csv
```

For each numeric metric, aggregate rows contain the seed mean plus `_min`,
`_max`, `_minus`, and `_plus` columns. The sweep plots use the mean line and the
available range columns for uncertainty.

## Validity Demo

The controlled validity demo is configured by:

```text
toy_experiments/configs/validity_demo.yaml
```

Run it with:

```bash
./toy_experiments/scripts/validity_demo.sh
```

Outputs:

```text
toy_experiments/outputs/validity_demo/results/
toy_experiments/outputs/validity_demo/plots/
```

The demo is artificial and validity-only. It is designed to expose cases where
per-token or count-only reasoning differs from a shared-shard MILP over the full
harmful sequence. The plotting step writes:

```text
validity_demo_baseline_vs_milp.pdf
validity_demo_certified_fraction_by_budget.pdf
audit_validity_demo.md
comparisons.txt
validity_demo_parameters.md
validity_demo_parameters.csv
```

The parameter files are generated from the actual result CSV so stale or
overridden runs are visible.

## Smoke Visualization

Run a lightweight visualization smoke check:

```bash
./toy_experiments/scripts/smoke.sh
```

This creates instance-level PDFs and `smoke_summary.txt` under:

```text
toy_experiments/outputs/smoke/
```

You can override the visualization dimensions with environment variables such
as `VIS_K`, `VIS_N`, `VIS_L`, `VIS_T`, `SEED`, and `VIS_OUT_DIR`.

## Python CLI

The direct entry point is:

```bash
python -m toy_experiments.experiments <command>
```

Common commands:

```bash
python -m toy_experiments.experiments visualize --save-dir toy_experiments/outputs/smoke
python -m toy_experiments.experiments benchmark --config toy_experiments/configs/small.yaml --dry-run
python -m toy_experiments.experiments plot-csv --csv toy_experiments/outputs/small/results/benchmark_results.csv --save-dir toy_experiments/outputs/small/plots
python -m toy_experiments.experiments plot-sweep --sweep K --csv toy_experiments/outputs/sweep_benchmark/K/results/benchmark_results.csv --save-dir toy_experiments/outputs/sweep_benchmark/K/plots
python -m toy_experiments.experiments plot-validity-demo --csv toy_experiments/outputs/validity_demo/results/benchmark_results.csv --save-dir toy_experiments/outputs/validity_demo/plots
```

The shell scripts use `toy_experiments/scripts/_python.sh` to choose the Python
executable. Set `PYTHON_BIN` if you need a specific environment.
