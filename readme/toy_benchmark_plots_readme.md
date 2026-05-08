# Toy Benchmark Results and Plots

This README explains the outputs produced by the toy row/column certificate benchmark.

The benchmark has two separate stages:

```bash
./scripts/run_toy_benchmark_data.sh
./scripts/plot_toy_benchmark.sh
```

The first command runs the Gurobi-backed benchmark and writes reusable data to:

```text
toy_results/benchmark_large/benchmark_results.csv
```

The second command reads that CSV and redraws the graphs. This means graph styling can be changed without rerunning the benchmark.

## Core Quantity

All certificate and baseline columns report a poison budget:

```text
B* = minimum number of corrupted shards needed to make the attack objective feasible
```

Larger `B*` means the certificate is stronger under the toy poisoning model. A value of `0` means the attack objective is already feasible without poisoning for that generated instance.

The shared MILP columns use one poisoning allocation:

```text
a[k] in {0, 1}
```

The same selected poisoned shards are reused across every required prompt row and token column. This is the main distinction from independent per-token or per-row baselines.

## Benchmark Parameters

Each CSV row corresponds to one generated toy instance:

```text
K               number of shards / base models
N               number of prompt rows
L               token sequence length / number of columns
T               token vocabulary size
delta_stab      disagreement rate for clean-prefix stability votes
delta_val       disagreement rate for harmful-prefix validity votes
target_bias     natural support for the harmful target under harmful-prefix votes
seed            random seed
influence_mode  poisoning influence mask: dense, row-local, or column-local
```

The default large benchmark sweeps:

```text
K in {3, 4, ..., 20}
N in {2, 3, ..., 12}
L in {2, 3, ..., 10}
T in {3, 4, ..., 12}
delta in {0.0, 0.1, 0.2, 0.3, 0.4}
```

The number of generated instances is:

```text
len(Ks) * len(Ns) * len(lengths) * len(Ts) * len(deltas)
```

Each instance solves several MILPs plus deterministic reference baselines.

## Aggregate Benchmark Plots

The aggregate plots are generated from `benchmark_results.csv` by:

```bash
python -m toy_certificate.experiments plot-csv \
  --csv toy_results/benchmark_large/benchmark_results.csv \
  --save-dir toy_results/benchmark_large
```

The plotting helper groups CSV rows by the x-axis variable and plots the mean `B*` for each metric at each x value. For example, in an `L` plot, all rows with the same sequence length are averaged across the other swept parameters.

### `validity_scaling_by_L.svg`

Question answered:

```text
How does harmful-sequence validity robustness scale as generated sequences get longer?
```

X-axis:

```text
L = token sequence length
```

Y-axis:

```text
Mean poison budget B*
```

Curves:

```text
DPA matrix: weakest row token  dpa_val_row_weak_q1
independent sequence q1        independent_val_sequence_q1
phrase-DPA validity            phrase_dpa_val_q1
shared row-column q1           row_col_val_q1
shared row-column qN           row_col_val_qN
```

Interpretation:

- `row_col_val_q1` is the shared-MILP budget to force the full harmful target sequence in at least one prompt row.
- `row_col_val_qN` is the shared-MILP budget to force the full harmful target sequence in all prompt rows.
- `dpa_val_row_weak_q1` is a token-matrix DPA baseline that reduces each row to its weakest token.
- `independent_val_sequence_q1` sums token-level costs across a row, so it represents independent composition rather than shared poisoning.
- `phrase_dpa_val_q1` treats the entire generated row as one atomic phrase class.

This plot is mainly about whether sequence-length growth makes shared row/column certification diverge from token-wise or phrase-wise baselines.

### `stability_structured_by_L.svg`

Question answered:

```text
How much poisoning is needed to destabilise structured parts of the prompt-token matrix as sequence length grows?
```

X-axis:

```text
L = token sequence length
```

Y-axis:

```text
Mean poison budget B*
```

Curves:

```text
DPA matrix q1                  dpa_stab_row_radius_q1
DPA matrix qN                  dpa_stab_row_radius_qN
q1 r1: weakest cell            row_col_stab_q1_r1
q1 rL: full response           row_col_stab_q1_rL
qN r1: all prompts one token   row_col_stab_qN_r1
qN rL: full matrix             row_col_stab_qN_rL
independent qN rL              independent_stab_full_row_qN
```

Notation:

```text
q1 = at least one prompt row
qN = all N prompt rows
r1 = at least one token cell in the selected row
rL = all L token cells in the selected row
```

Interpretation:

- `row_col_stab_q1_r1` is the weakest-cell shared-MILP stability attack.
- `row_col_stab_q1_rL` requires a full generated response row to be destabilised.
- `row_col_stab_qN_r1` requires every prompt row to have at least one destabilised token.
- `row_col_stab_qN_rL` requires the full prompt-token matrix to be destabilised.
- `independent_stab_full_row_qN` sums independent full-row costs across all rows.

This plot is the best view for comparing structured shared poisoning against independent composition.

### `validity_bias_sweep.svg`

Question answered:

```text
How does harmful-target validity robustness change when the harmful target already has more natural support?
```

X-axis:

```text
target_bias = probability mass pushed toward harmful target votes
```

Y-axis:

```text
Mean poison budget B*
```

Curves:

```text
DPA matrix q1          dpa_val_row_weak_q1
DPA matrix qN          dpa_val_row_weak_qN
shared row-column q1   row_col_val_q1
shared row-column qN   row_col_val_qN
phrase-DPA q1          phrase_dpa_val_q1
```

Interpretation:

Increasing `target_bias` should generally lower validity budgets, because the harmful target is closer to winning before poisoning. This plot checks whether the shared MILP and the DPA-style baselines respond similarly to that bias.

## Per-Instance Visualization Plots

Per-instance plots are generated by:

```bash
python -m toy_certificate.experiments visualize \
  --K 7 --N 3 --L 4 --T 5 \
  --delta-stab 0.2 --delta-val 0.2 --target-bias 0.2 \
  --influence-mode dense \
  --seed 0 \
  --save-dir toy_results/default_instance
```

These plots explain a single generated toy instance rather than benchmark averages.

### `clean_predictions.svg`

Heatmap of:

```text
clean_pred[i,j]
```

Each cell shows the clean ensemble majority token for prompt row `i` and token position `j`.

### `harmful_targets.svg`

Heatmap of:

```text
target[i,j]
```

Each cell shows the harmful target token used by validity attacks. The generator chooses targets different from the clean predictions where possible.

### `stability_margins.svg`

Heatmap of:

```text
stab_counts[i,j,clean_pred[i,j]] - stab_counts[i,j,runner_up[i,j]]
```

This is the clean winner's margin over the runner-up token under clean-prefix votes. Smaller margins usually imply lower stability budgets.

### `validity_target_counts.svg`

Heatmap of:

```text
val_counts[i,j,target[i,j]]
```

This shows how many harmful-prefix votes already support the harmful target token in each cell. Higher target counts usually imply lower validity budgets.

### `structured_stability_heatmap.svg`

Heatmap of shared-MILP structured stability budgets:

```text
entry[q-1,r-1] = B*(at least q rows, each with at least r destabilised token cells)
```

Rows of the heatmap correspond to `q = 1..N`. Columns correspond to `r = 1..L`.

This is the per-instance version of the structured stability idea used in `stability_structured_by_L.svg`.

### `validity_q_curve.svg`

Line plot of:

```text
B*(force the full harmful target sequence in at least q prompt rows)
```

The x-axis is `q = 1..N`. The y-axis is the shared-MILP poison budget.

## Shared MILP Certificate Columns

The benchmark writes these direct MILP results:

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

Stability means the clean prediction can be changed or tied by an adversarial competitor. Validity means the harmful target can be made to win or tie all competitors.

## Reference Baseline Columns

The confirmed DPA matrix baseline computes token-level certificates independently and then reduces each prompt row to its weakest token:

```text
dpa_stab_cell_min
dpa_stab_row_radius_q1
dpa_stab_row_radius_qN
dpa_val_cell_min
dpa_val_row_weak_q1
dpa_val_row_weak_qN
```

Independent composition sums token-level costs:

```text
independent_stab_full_row_q1
independent_stab_full_row_qN
independent_val_sequence_q1
independent_val_sequence_qN
```

Phrase-DPA treats a full generated row as one atomic class:

```text
phrase_dpa_val_q1
phrase_dpa_val_qN
phrase_independent_val_q1
phrase_independent_val_qN
```

Compatibility and diagnostic columns are also emitted:

```text
raw_dpa_stab_min_cell
raw_dpa_val_min_cell
independent_stab_qN_rL
independent_val_q1
independent_val_qN
runtime_gurobi_total
```

For the exact formulas behind the DPA-style baselines, see `readme/naive_dpa_readme.md`.

## Reading the Results

Use these checks when interpreting plots:

- Higher `B*` means stronger robustness for the plotted attack objective.
- `qN` objectives are stronger than `q1` objectives, so they should usually have larger or equal budgets.
- `rL` objectives are stronger than `r1` objectives, so they should usually have larger or equal budgets.
- Shared MILP curves can be lower than independent-composition curves because one poisoned shard allocation can affect many required cells at once.
- Phrase-DPA can behave differently from token-level baselines because full sequences are treated as atomic labels, which changes the vote-count geometry.
- Averages in aggregate plots hide variation across `K`, `N`, `T`, `delta`, and `seed`; inspect the CSV directly when debugging a surprising point.
