# Provably-Secure-NLG

EIE Final Year Project 2026.

This repository contains a runnable toy implementation of row/column poisoning certificates for natural-language generation style token voting. The core experiment builds a prompt-by-token vote matrix, solves shared-allocation MILPs with Gurobi, and compares those certificates with DPA-style baselines.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gurobi also needs a valid local license.

Run the lightweight check pipeline:

```bash
./scripts/run_toy_check.sh
```

This runs compile checks, unit tests, and one visualization instance. It does not run the benchmark.

## Common Commands

Run benchmark data generation only:

```bash
./scripts/run_toy_benchmark_data.sh
```

Refresh plots from an existing benchmark CSV:

```bash
./scripts/plot_toy_benchmark.sh
```

Run visualization plus plot refresh from an existing CSV:

```bash
./scripts/run_toy_benchmark.sh
```

Run tests directly:

```bash
python -m unittest discover
```

## Documentation

- `readme/spec.md`: consolidated design/specification notes for what the toy certificate experiment is meant to build.
- `readme/implementation.md`: consolidated implementation, command, benchmark, plot, and baseline explanation.
- `readme/phd_readme.md`: notes on the external `phd_reference` package structure and where the closest reference-code concepts live.

## Repository Layout

- `toy_certificate/`: toy data generator, MILP solvers, experiment CLI, plotting helpers.
- `scripts/run_toy_check.sh`: compile, test, and visualization check run.
- `scripts/run_toy_benchmark_data.sh`: benchmark data generation, writing `benchmark_results.csv`.
- `scripts/plot_toy_benchmark.sh`: plot refresh from an existing CSV.
- `scripts/run_toy_benchmark.sh`: visualization plus plot refresh from an existing CSV.
- `tests/`: lightweight test bench.
- `historical_csvs/`: saved benchmark CSVs.
- `toy_results/`: generated outputs, ignored by git.
