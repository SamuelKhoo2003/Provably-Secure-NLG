# Toy Certificate Implementation

This document consolidates how the toy certificate code is implemented, how to run it, and how to interpret benchmark outputs.

## Files

```text
toy_certificate/data.py         toy vote generation and counts
toy_certificate/milp.py         Gurobi MILP builders and solvers
toy_certificate/experiments.py  CLI, benchmark runner, baselines, SVG plots
scripts/run_toy_check.sh        compile/test/visualization check run
scripts/run_toy_benchmark_data.sh benchmark CSV generation
scripts/plot_toy_benchmark.sh   plot refresh from existing CSV
scripts/run_toy_benchmark.sh    visualization plus plot refresh from existing CSV
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gurobi requires a valid local license.

## Main Commands

Check the repo without running a benchmark:

```bash
./scripts/run_toy_check.sh
```

Generate benchmark data only:

```bash
./scripts/run_toy_benchmark_data.sh
```

Refresh plots from an existing CSV:

```bash
./scripts/plot_toy_benchmark.sh
```

Generate the visualization instance and refresh plots from an existing CSV:

```bash
./scripts/run_toy_benchmark.sh
```

Run tests directly:

```bash
python -m unittest discover
```

## CLI

The experiment module exposes:

```bash
python -m toy_certificate.experiments sanity
python -m toy_certificate.experiments visualize
python -m toy_certificate.experiments benchmark
python -m toy_certificate.experiments plot-csv
python -m toy_certificate.experiments sweep-delta
python -m toy_certificate.experiments sweep-length
python -m toy_certificate.experiments sweep-prompts
```

`benchmark` writes CSV data only by default. Add `--make-plots` only if you explicitly want plotting in the same run.

## Data Representation

The toy generator returns `ToyData`, including:

```text
stab_votes[k, i, j]
val_votes[k, i, j]
stab_counts[i, j, t]
val_counts[i, j, t]
clean_pred[i, j]
runner_up[i, j]
target[i, j]
influence[k, i, j]
```

The token-voting tensor maps naturally to language-generation DPA partitions:

```text
toy K  = number of partitions / shard models
toy N  = number of prompts
toy L  = generated token horizon
toy T  = vocabulary size
```

In the reference `phd_reference` code, the closest equivalent is the language-generation stability certifier building tokenized responses per partition and per test sample, then counting votes by token position.

## MILP Certificates

The implemented shared-allocation MILP certificates are:

```text
row_stability
column_stability
row_col_stab_q1_r1
row_col_stab_q1_rL
row_col_stab_qN_r1
row_col_stab_qN_rL
row_validity
column_validity_full_column
row_col_val_q1
row_col_val_qN
```

All report:

```text
B* = minimum poisoned-shard count
```

Larger `B*` means stronger robustness for that objective. `B* = 0` means the attack condition is already feasible before poisoning.

## Baselines

Baseline columns are computed in `compute_reference_baselines(data)`.

Confirmed DPA matrix baseline:

```text
dpa_stab_cell_min
dpa_stab_row_radius_q1
dpa_stab_row_radius_qN
dpa_val_cell_min
dpa_val_row_weak_q1
dpa_val_row_weak_qN
```

This baseline computes token-level certificates independently, then represents each prompt row by its weakest token:

```text
row_radius[i] = min_j B_cell[i,j]
```

Independent composition:

```text
independent_stab_full_row_q1
independent_stab_full_row_qN
independent_val_sequence_q1
independent_val_sequence_qN
```

This sums token costs and does not reuse the same poisoned allocation across cells.

Phrase-DPA:

```text
phrase_dpa_val_q1
phrase_dpa_val_qN
phrase_independent_val_q1
phrase_independent_val_qN
```

This treats a full generated row as one atomic label.

Compatibility and diagnostics:

```text
raw_dpa_stab_min_cell
raw_dpa_val_min_cell
independent_stab_qN_rL
independent_val_q1
independent_val_qN
runtime_gurobi_total
```

## Benchmark Data

Default benchmark data generation:

```bash
./scripts/run_toy_benchmark_data.sh
```

Default output:

```text
toy_results/benchmark_large/benchmark_results.csv
```

Default large sweep:

```text
K in {3, 4, ..., 20}
N in {2, 3, ..., 12}
L in {2, 3, ..., 10}
T in {3, 4, ..., 12}
delta in {0.0, 0.1, 0.2, 0.3, 0.4}
target_bias = 0.2
```

This is a large sweep:

```text
18 * 11 * 9 * 10 * 5 = 89,100 generated instances
```

Each instance solves several MILPs, so this can take a long time.

Override ranges with environment variables:

```bash
KS=3,4,5 NS=2,3 LENGTHS=2,3 TS=3,4 DELTAS=0.0,0.2 ./scripts/run_toy_benchmark_data.sh
```

## Plots

Refresh plots from an existing CSV:

```bash
CSV_PATH=toy_results/benchmark_large/benchmark_results.csv \
OUT_DIR=toy_results/benchmark_large \
./scripts/plot_toy_benchmark.sh
```

Aggregate benchmark SVGs:

```text
validity_scaling_by_L.svg
stability_structured_by_L.svg
validity_bias_sweep.svg
```

The plotter groups rows by the x-axis variable and plots mean `B*` for each metric.

### `validity_scaling_by_L.svg`

X-axis:

```text
L = sequence length
```

Curves:

```text
dpa_val_row_weak_q1
independent_val_sequence_q1
phrase_dpa_val_q1
row_col_val_q1
row_col_val_qN
```

This compares harmful-sequence validity robustness as sequence length grows.

### `stability_structured_by_L.svg`

X-axis:

```text
L = sequence length
```

Curves:

```text
dpa_stab_row_radius_q1
dpa_stab_row_radius_qN
row_col_stab_q1_r1
row_col_stab_q1_rL
row_col_stab_qN_r1
row_col_stab_qN_rL
independent_stab_full_row_qN
```

This compares structured stability objectives as sequence length grows.

### `validity_bias_sweep.svg`

X-axis:

```text
target_bias
```

Curves:

```text
dpa_val_row_weak_q1
dpa_val_row_weak_qN
row_col_val_q1
row_col_val_qN
phrase_dpa_val_q1
```

This shows how validity budgets change when the harmful target already has more natural vote support.

## Visualization Outputs

The visualization command:

```bash
python -m toy_certificate.experiments visualize \
  --K 20 --N 12 --L 10 --T 12 \
  --delta-stab 0.2 --delta-val 0.2 --target-bias 0.2 \
  --influence-mode dense \
  --seed 0 \
  --save-dir toy_results/default_instance
```

Writes:

```text
clean_predictions.svg
harmful_targets.svg
stability_margins.svg
validity_target_counts.svg
structured_stability_heatmap.svg
validity_q_curve.svg
```

Meanings:

```text
clean_predictions.svg          clean majority token per cell
harmful_targets.svg            harmful target token per cell
stability_margins.svg          winner-vs-runner-up stability margin
validity_target_counts.svg     current target-token vote counts
structured_stability_heatmap.svg B*(q rows, r changed tokens)
validity_q_curve.svg           B*(full harmful sequence in q rows)
```

## Reading Results

- Higher `B*` means stronger robustness for the plotted attack objective.
- `qN` objectives are stronger than `q1` objectives and should usually have larger or equal budgets.
- `rL` objectives are stronger than `r1` objectives and should usually have larger or equal budgets.
- Shared MILP budgets can be lower than independent-composition budgets because one poisoned shard allocation can satisfy multiple cell objectives.
- Aggregate plots average over other swept parameters, so inspect the CSV directly when debugging a surprising point.
