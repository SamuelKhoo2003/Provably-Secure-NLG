# Toy Row/Column Certificate Experiment

This document describes the runnable toy implementation of the row/column poisoning certificate experiment in `readme/toy_example_spec.md`.

The key modelling choice is that all MILPs use one shared poisoning allocation vector:

```text
a[k] in {0, 1}
```

The same corrupted shard allocation is used across every prompt row and token column.

For a closer explanation of the prompt/token vote tensor and how it relates to `phd_reference`, see `readme/token_voting_readme.md`.

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
`--influence-mode` can be `dense`, `row-local`, or `column-local`; it controls which cells a poisoned shard can affect.

## Console visualization and plots

Print the generated prompt/token grid, shard vote layers, and save heatmap plots:

```bash
python -m toy_certificate.experiments visualize \
  --K 7 --N 3 --L 4 --T 5 \
  --delta-stab 0.2 --delta-val 0.2 --target-bias 0.2 --seed 0 \
  --influence-mode dense \
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
- `validity_target_counts.svg`
- `structured_stability_heatmap.svg`
- `validity_q_curve.svg`

You can also attach the grid view to the sanity run:

```bash
python -m toy_certificate.experiments sanity --show-grid --save-dir toy_results/sanity
```

## Benchmark ranges

Run the default scale benchmark:

```bash
python -m toy_certificate.experiments benchmark --save-dir toy_results/benchmark_default
```

This writes reusable benchmark data:

- `benchmark_results.csv`

By default this does not render plots, so changing graph style does not require rerunning Gurobi. To benchmark and plot in one command, pass `--make-plots`.

Render or refresh plots from an existing CSV:

```bash
python -m toy_certificate.experiments plot-csv \
  --csv toy_results/benchmark_default/benchmark_results.csv \
  --save-dir toy_results/benchmark_default
```

This writes a focused set of comparison plots:

- `validity_scaling_by_L.svg`
- `stability_structured_by_L.svg`
- `validity_bias_sweep.svg`

The focused plots compare:

- shared-allocation MILP certificate;
- confirmed DPA matrix baseline, where each prompt row is represented by its weakest token certificate;
- independent-composition baseline, where token costs are summed separately;
- phrase-DPA baseline, where a full generated row is treated as one class.

See `readme/naive_dpa_readme.md` for the exact baseline formulas and CSV column meanings.
See `readme/toy_benchmark_plots_readme.md` for a full explanation of every benchmark graph and plotted column.

Run a larger benchmark by passing comma-separated ranges:

```bash
python -m toy_certificate.experiments benchmark \
  --Ks 5,7,9,11,15 \
  --Ns 2,3,5,8,12 \
  --lengths 2,4,8 \
  --Ts 3,5,8,12 \
  --deltas 0.0,0.1,0.2,0.3,0.4 \
  --target-bias 0.2 \
  --influence-mode dense \
  --seed 0 \
  --save-dir toy_results/benchmark_large
```

The number of Gurobi batches is:

```text
len(Ks) * len(Ns) * len(lengths) * len(Ts) * len(deltas)
```

Each batch solves several certificates, so start modest before scaling up.

You can replot any existing benchmark CSV without rerunning Gurobi:

```bash
python -m toy_certificate.experiments plot-csv \
  --csv toy_results/benchmark_large/benchmark_results.csv \
  --save-dir toy_results/benchmark_large
```

If the CSV was produced before the renamed spec columns were added, replotting can only show the columns present in that file. Rerun `benchmark` or `./scripts/run_toy_benchmark.sh` to generate the current comparison columns.

Current benchmark CSV columns include:

- `row_col_stab_q1_r1`, `row_col_stab_q1_rL`, `row_col_stab_qN_r1`, `row_col_stab_qN_rL`
- `row_col_val_q1`, `row_col_val_qN`
- `dpa_stab_cell_min`, `dpa_stab_row_radius_q1`, `dpa_stab_row_radius_qN`
- `dpa_val_cell_min`, `dpa_val_row_weak_q1`, `dpa_val_row_weak_qN`
- `independent_stab_full_row_q1`, `independent_stab_full_row_qN`
- `independent_val_sequence_q1`, `independent_val_sequence_qN`
- `phrase_dpa_val_q1`, `phrase_dpa_val_qN`, `phrase_independent_val_qN`

## `q1` vs `qN`

`row_col_validity_q1` and `row_col_validity_qN` are both targeted sequence-validity certificates.

- `q1`: minimum poison budget needed to force the full harmful target sequence in at least one prompt row.
- `qN`: minimum poison budget needed to force the full harmful target sequence in all `N` prompt rows.

`qN` is the stronger requirement, so it should usually be greater than or equal to `q1`. In the CSV, `qN` means "the current row count for that benchmark row", not a fixed number.

## Bash runner

Run only the large benchmark data generation:

```bash
./scripts/run_toy_benchmark_data.sh
```

Refresh plots from an existing benchmark CSV:

```bash
./scripts/plot_toy_benchmark.sh
```

Run the default visualization, large benchmark, and plot refresh:

```bash
./scripts/run_toy_benchmark.sh
```

Run the full pipeline, including compile check, tests, visualization, benchmark, and plot refresh:

```bash
./scripts/full_run_toy_example.sh
```

Default full-run outputs:

- `toy_results/full_run/instance/*.svg`
- `toy_results/full_run/benchmark/benchmark_results.csv`
- `toy_results/full_run/benchmark/*.svg`

Override ranges with environment variables:

```bash
KS=5,7,9 NS=2,4 LENGTHS=2,4 TS=3,5 DELTAS=0.0,0.2 TARGET_BIAS=0.3 INFLUENCE_MODE=row-local OUT_DIR=toy_results/custom ./scripts/full_run_toy_example.sh
```

## Tests

Run the lightweight test bench:

```bash
python -m unittest discover
```

Tests that require Gurobi skip if a local license is not available.

## Files

- `toy_certificate/data.py`: vote generation, counts, predictions, targets, and margins.
- `toy_certificate/milp.py`: Gurobi MILP builders and certificate solvers.
- `toy_certificate/experiments.py`: command-line experiments and table printing.
- `scripts/run_toy_benchmark.sh`: bash wrapper for visualization plus benchmark runs.
- `scripts/full_run_toy_example.sh`: full compile, test, visualization, benchmark, and replot pipeline.
