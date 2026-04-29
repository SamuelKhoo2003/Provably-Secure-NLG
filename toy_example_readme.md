# Toy Row/Column Certificate Experiment

This document describes the runnable toy implementation of the row/column poisoning certificate experiment in `toy_example_spec.md`.

The key modelling choice is that all MILPs use one shared poisoning allocation vector:

```text
a[k] in {0, 1}
```

The same corrupted shard allocation is used across every prompt row and token column.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gurobi requires a valid local license.

## Run

```bash
python -m certificate_toy.experiments sanity
python -m certificate_toy.experiments sweep-delta
python -m certificate_toy.experiments sweep-length
python -m certificate_toy.experiments sweep-prompts
```

Default sanity configuration:

```text
K=7, N=3, L=4, T=5, delta=0.2, seed=0
```

## Files

- `certificate_toy/data.py`: vote generation, counts, predictions, targets, and margins.
- `certificate_toy/milp.py`: Gurobi MILP builders and certificate solvers.
- `certificate_toy/experiments.py`: command-line experiments and table printing.
