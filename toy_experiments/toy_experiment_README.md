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
