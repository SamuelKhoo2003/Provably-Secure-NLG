# Toy Row/Column Certificate Experiment

This document describes the runnable toy implementation of the row/column poisoning certificate experiment in `readme/toy_example_spec.md`.

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
python -m toy_certificate.experiments sanity
python -m toy_certificate.experiments visualize
python -m toy_certificate.experiments benchmark
python -m toy_certificate.experiments plot-csv
python -m toy_certificate.experiments sweep-delta
python -m toy_certificate.experiments sweep-length
python -m toy_certificate.experiments sweep-prompts
```

Default sanity configuration:

```text
K=7, N=3, L=4, T=5, delta_stab=0.2, delta_val=0.2, target_bias=0.2, seed=0
```

`delta_stab` controls disagreement for stability votes under the clean prefix.
`delta_val` controls disagreement for validity votes under the harmful prefix.
`target_bias` controls how much natural support the harmful target receives under the harmful prefix.
The shorthand `--delta 0.2` sets both `delta_stab` and `delta_val` unless the explicit flags are provided.

## Console visualization and plots

Print the generated prompt/token grid, shard vote layers, and save heatmap plots:

```bash
python -m toy_certificate.experiments visualize \
  --K 7 --N 3 --L 4 --T 5 \
  --delta-stab 0.2 --delta-val 0.2 --target-bias 0.2 --seed 0 \
  --save-dir toy_results/default_instance
```

The console grid shows each cell as:

```text
pred->target r<runner_up> m<margin> w<winner_count> h<target_count>
```

For example:

```text
row 00: 4->1 r0 m5 w6 h0 | ...
```

Saved SVG plots include:

- `clean_predictions.svg`
- `harmful_targets.svg`
- `stability_margins.svg`
- `winner_counts.svg`
- `target_counts.svg`
- `disagreement_rate.svg`

You can also attach the grid view to the sanity run:

```bash
python -m toy_certificate.experiments sanity --show-grid --save-dir toy_results/sanity
```

## Benchmark ranges

Run the default scale benchmark:

```bash
python -m toy_certificate.experiments benchmark --save-dir toy_results/benchmark_default
```

This writes a focused set of comparison plots:

- `benchmark_results.csv`
- `focused_stability_by_K.svg`, `focused_stability_by_N.svg`, etc.
- `focused_validity_q1_by_K.svg`, `focused_validity_q1_by_N.svg`, etc.
- `focused_validity_qN_by_K.svg`, `focused_validity_qN_by_N.svg`, etc.

The focused plots compare:

- shared-allocation MILP certificate;
- naive DPA per-cell baseline, where token costs are computed independently and then added;
- PHD-style single-cell majority-margin reference, where applicable.

Run a larger benchmark by passing comma-separated ranges:

```bash
python -m toy_certificate.experiments benchmark \
  --Ks 5,7,9,11,15 \
  --Ns 2,3,5,8,12 \
  --lengths 2,4,8 \
  --Ts 3,5,8,12 \
  --deltas 0.0,0.1,0.2,0.3,0.4 \
  --target-bias 0.2 \
  --seed 0 \
  --save-dir toy_results/benchmark_large
```

The number of Gurobi batches is:

```text
len(Ks) * len(Ns) * len(lengths) * len(Ts) * len(deltas)
```

Each batch solves several certificates, so start modest before scaling up.

You can replot an existing CSV without rerunning Gurobi:

```bash
python -m toy_certificate.experiments plot-csv \
  --csv toy_results/benchmark_large/benchmark_results.csv \
  --save-dir toy_results/benchmark_large
```

If the CSV was produced before the baseline columns were added, replotting can only show the shared MILP series. Rerun `benchmark` or `./run_toy_benchmark.sh` to generate the baseline comparison columns.

## `q1` vs `qN`

`row_col_validity_q1` and `row_col_validity_qN` are both targeted sequence-validity certificates.

- `q1`: minimum poison budget needed to force the full harmful target sequence in at least one prompt row.
- `qN`: minimum poison budget needed to force the full harmful target sequence in all `N` prompt rows.

`qN` is the stronger requirement, so it should usually be greater than or equal to `q1`. In the CSV, `qN` means "the current row count for that benchmark row", not a fixed number.

## Bash runner

Run the default visualization and large benchmark:

```bash
./run_toy_benchmark.sh
```

Override ranges with environment variables:

```bash
KS=5,7,9 NS=2,4 LENGTHS=2,4 TS=3,5 DELTAS=0.0,0.2 TARGET_BIAS=0.3 OUT_DIR=toy_results/custom ./run_toy_benchmark.sh
```

## Files

- `toy_certificate/data.py`: vote generation, counts, predictions, targets, and margins.
- `toy_certificate/milp.py`: Gurobi MILP builders and certificate solvers.
- `toy_certificate/experiments.py`: command-line experiments and table printing.
- `run_toy_benchmark.sh`: bash wrapper for visualization plus benchmark runs.
