# Toy Certificate Implementation

This document consolidates how the toy certificate code is implemented, how to run it, and how to interpret benchmark outputs.

`toy_certificate/` is the self-contained first-party toy implementation.
`phd_reference/` is external read-only reference code; do not edit, reformat, or
delete files inside it as part of toy implementation cleanup.

## Files

```text
toy_certificate/data.py         toy vote generation and counts
toy_certificate/milp.py         Gurobi MILP builders and solvers
toy_certificate/experiments.py  CLI, benchmark runner, baselines, SVG plots
scripts/check.sh                compile/test/visualization check run
scripts/data.sh                 benchmark CSV generation
scripts/plot.sh                 plot refresh from existing CSV
scripts/benchmark.sh            data generation followed by plotting
scripts/visualize.sh            one visualization instance
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
./scripts/check.sh
```

Generate benchmark data only:

```bash
./scripts/data.sh
```

Refresh plots from an existing CSV:

```bash
./scripts/plot.sh
```

Generate benchmark data and then refresh plots:

```bash
./scripts/benchmark.sh
```

Generate one visualization instance:

K in {4, 6, 8}
N in {2, 3}
L in {2, 3, 4}
T in {4, 6}
delta_stab = delta_val = 0.2
target_bias = 0.3
```bash
.venv/bin/python -m unittest discover
```

## CLI

The experiment module exposes:

```bash
.venv/bin/python -m toy_certificate.experiments sanity
.venv/bin/python -m toy_certificate.experiments visualize
.venv/bin/python -m toy_certificate.experiments benchmark
.venv/bin/python -m toy_certificate.experiments plot-csv
.venv/bin/python -m toy_certificate.experiments sweep-delta
.venv/bin/python -m toy_certificate.experiments sweep-length
.venv/bin/python -m toy_certificate.experiments sweep-prompts
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

The optimization uses one shared poisoned-shard allocation:

```text
a[k] in {0, 1}
minimize sum_k a[k]
```

The same selected poisoned shards are reused across all required rows and token
positions. This shared allocation is the core row/column coupling. Larger `B*`
means stronger robustness for that objective. `B* = 0` means the attack
condition is already feasible before poisoning.

### Stability and Validity Conditions

Stability means the attacker changes the generated output away from the clean
output. A cell is destabilised if any competitor token `c != clean winner` can
tie or beat the clean winner after poisoning. The MILP checks all competitors,
not only the original runner-up. Runner-up margins remain useful for DPA-style
baselines, but they are not the only MILP competitor.

Stability supports two competitor modes:

```text
all        exact mode; checks every competitor token
runner_up  cheaper DPA-style top-vs-runner-up simplification
```

The default is `all`. The runner-up mode may overestimate robustness if a
non-runner-up competitor is easier to promote. If both modes solve optimally, the
expected relationship is:

```text
B*_runner_up >= B*_all
```

Use all-competitor mode for correctness and report-critical experiments. Use
runner-up mode for large sweeps only after the comparison diagnostic shows it
agrees or is acceptably close on the chosen toy distribution. For dense
influence settings the two modes may often agree, but this should be verified
rather than assumed.

Validity means the attacker forces a specific harmful target token or harmful
target sequence. A validity cell succeeds if the harmful target token ties or
beats every competitor token, which is stricter than only beating the current
clean winner. Validity code uses `counts` terminology because those counts come
from the validity/harmful-prefix vote tensor.

### Report-Facing Objectives

Define `q` as the number of affected prompt rows and `r` as the number of
affected token positions per selected row. Define the shorthand once:

```text
q = 1  one prompt
q = N  all prompts
r = 1  one token
r = L  full sequence
```

Recommended stability objectives use `solve_structured_stability`:

```text
one prompt, one token        q_rows=1, r_cols=1
one prompt, full sequence    q_rows=1, r_cols=L
all prompts, one token each  q_rows=N, r_cols=1
all prompts, full matrix     q_rows=N, r_cols=L
```

Recommended validity objectives use `solve_row_col_validity`:

```text
one harmful sequence                 q_rows=1
harmful sequences for all prompts    q_rows=N
```

### Solver Exactness

`CertificateResult` includes:

```text
is_optimal
mip_gap
lower_bound
upper_bound
```

If `is_optimal=True`, `B_star` is an exact optimum. If Gurobi returns
`TIME_LIMIT` or `SUBOPTIMAL` with a feasible solution, `B_star` is the best
feasible upper bound found, not an exact certificate. Plotting/reporting should
prefer optimal rows or clearly state how many non-optimal rows were included.

`attacked_cells` is diagnostic. Since `z` variables are not secondarily
minimized, the list may include extra feasible cells. The certified quantity is
`B_star`, not the exact diagnostic cell list.

## Baselines

Baseline columns are computed in `compute_reference_baselines(data)`.

There are two main external baselines.

The first is the token-level DPA margin baseline. DPA remains the natural
baseline for untargeted stability, where the adversary tries to change the clean
output. For validity, the same token-level baseline is a weakest harmful-token
reference and should not be interpreted as a full-sequence certificate.

The second is the Ghitu-style phrase-level TPA baseline. It computes targeted
token validity radii for inducing a specific harmful token. For a harmful
sequence, the toy implementation composes token-level TPA radii using the
maximum over token positions, since the attacker must force every token in the
sequence. The older phrase-level aggregation baseline is retained as a naive
atomic-sequence diagnostic, but it should not be interpreted as the main TPA
validity baseline.

DPA matrix / weakest-token baseline:

```text
dpa_stab_cell_min
dpa_stab_row_radius_q1
dpa_stab_row_radius_qN
dpa_val_cell_min
dpa_val_row_weak_q1
dpa_val_row_weak_qN
```

This baseline computes token-level DPA-style margins independently, then represents each prompt row by its weakest/easiest token:

```text
row_radius[i] = min_j B_cell[i,j]
```

For validity, this means "easiest harmful target token", not a full harmful-sequence certificate.

TPA max-token sequence baseline:

```text
tpa_val_cell_min
tpa_val_sequence_q1
tpa_val_sequence_qN
tpa_val_sequence_mean
```

This is the paper-inspired targeted validity baseline. Each token cell uses
`targeted_partition_radius(counts, target)`, the minimum number of non-target
votes that must be changed into the harmful target so the target ties or beats
every competitor. The sequence metric then computes:

```text
R_tpa_sequence[i] = max_j r_tpa[i,j]
tpa_val_sequence_q1 = min_i R_tpa_sequence[i]
tpa_val_sequence_qN = max_i R_tpa_sequence[i]
```

This is not ordinary DPA top-vs-runner-up stability and not phrase aggregation.
It follows the toy MILP tie convention where target ties count as successful
attacks. If the paper convention being compared against is strict plurality,
interpret this as the tie-wins toy adaptation of TPA.

Independent composition:

```text
independent_stab_full_row_q1
independent_stab_full_row_qN
independent_val_sequence_q1
independent_val_sequence_qN
```

This sums token costs and does not reuse the same poisoned allocation across cells. Treat it as a loose/conservative upper reference rather than the main method.

Atomic phrase aggregation:

```text
phrase_dpa_val_q1
phrase_dpa_val_qN
phrase_independent_val_q1
phrase_independent_val_qN
```

This treats a full generated row as one atomic label/phrase and does not reason token-by-token.
It is useful as a crude full-sequence diagnostic, but exact sequence votes diffuse
over many possible sequences as `L` grows, so it should not be read as the main
TPA/PHD validity baseline. The old `phrase_dpa_*` CSV names are retained for
compatibility.

The shared row-column MILP is the proposed method. Row-only, column-only, and
joint row-column MILPs are variants or ablations of the proposed method, not
external baselines.

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
./scripts/data.sh
```

Default output:

```text
toy_results/small_benchmark/benchmark_results.csv
toy_results/small_benchmark/benchmark_budget_curves.csv
toy_results/small_benchmark/benchmark_damage_curves.csv
toy_results/small_benchmark/benchmark_horizons.csv
```

Default small sweep:

```text
K in {4, 6, 8}
N in {2, 3}
L in {2, 3, 4}
T in {4, 6}
delta_stab = delta_val = 0.2
target_bias = 0.3
```
This keeps the default benchmark compact enough for quick plot and label iteration.

```text
3 * 2 * 3 * 2 * 1 * 1 * 1 = 36 generated instances
```

Each instance solves several MILPs, so this can still take a little while.
Direct damage-at-budget curves add three maximization MILPs per budget, so use a
small `BUDGET_MAX` or disable them for larger sweeps.

Override ranges with environment variables:

```bash
KS=4,6 NS=2 LENGTHS=2,3 TS=4 DELTA_STABS=0.2 DELTA_VALS=0.2 TARGET_BIASES=0.3 ./scripts/data.sh
```

Choose the stability competitor mode with:

```bash
STABILITY_COMPETITOR_MODE=runner_up ./scripts/data.sh
```

The shell wrappers default to `STABILITY_COMPETITOR_MODE=runner_up` for quick
iteration. Set it to `all` for exact all-competitor runs.

Budget-sweep controls:

```bash
BUDGET_MAX=15 ./scripts/data.sh
MAKE_BUDGET_CURVES=0 ./scripts/data.sh
MAKE_DAMAGE_CURVES=0 ./scripts/data.sh
MAKE_HORIZON_CURVES=0 ./scripts/data.sh
```

Metric families:

```text
benchmark_results.csv         radius-style B* metrics
benchmark_budget_curves.csv   radius-derived coverage, certified iff B < B*_row
benchmark_damage_curves.csv   direct shared-MILP damage maximization at fixed B
benchmark_horizons.csv        average certified prefix horizon at fixed B
```

Radius-derived coverage solves rows independently and is cheap enough for
baselines and quick plotting. Direct damage-at-budget MILPs are more faithful to
the shared-allocation threat model because the solver chooses one poisoned-shard
set and maximizes damage under that budget.

## Stability Mode Comparison

Run a small exact-vs-runner-up diagnostic:

```bash
.venv/bin/python -m toy_certificate.experiments compare-stability-modes \
  --Ks 4,6 --Ns 2 --lengths 2,3 --Ts 4 --deltas 0.2 \
  --save-dir toy_results/stability_mode_comparison
```

This writes:

```text
stability_mode_comparison.csv
stability_mode_diff_by_L.svg
stability_mode_runtime_by_L.svg
```

The CSV includes `B_star_all`, `B_star_runner_up`, `diff`,
`all_is_optimal`, `runner_up_is_optimal`, status names, and runtimes. Negative
optimal differences are flagged because runner-up-only should not need less
poisoning than all-competitor mode.

## Plots

Refresh plots from an existing CSV:

```bash
CSV_PATH=toy_results/small_benchmark/benchmark_results.csv \
OUT_DIR=toy_results/small_benchmark \
./scripts/plot.sh
```

Report-facing benchmark SVGs:

```text
validity_one_prompt_by_L.svg
validity_all_prompts_by_L.svg
stability_one_prompt_by_L.svg
stability_all_prompts_by_L.svg
validity_independent_overestimate_by_L.svg
stability_independent_overestimate_by_L.svg
certified_fraction_stability_by_budget.svg
certified_fraction_validity_by_budget.svg
certified_fraction_stability_by_L_at_budget.svg
certified_fraction_validity_by_L_at_budget.svg
stability_horizon_by_budget.svg
validity_horizon_by_budget.svg
```

Compatibility aggregate SVGs are still written:

```text
validity_scaling_by_L.svg
stability_structured_by_L.svg
validity_bias_sweep.svg
```

The plotter groups rows by the x-axis variable and plots mean `B*` for each metric.
If a requested metric column is missing, the plotter skips that curve and prints a warning.

The plots are separated by attack objective. Stability plots measure the budget needed to change outputs away from the clean generation. Validity plots measure the budget needed to force specific harmful target tokens or sequences. DPA weakest harmful token is the easiest single harmful token and is not a full-sequence certificate. TPA max-token sequence is the targeted sequence baseline using the maximum over token positions. Atomic phrase aggregation treats the whole sequence as one label. Independent composition sums token costs and therefore ignores poisoned-shard reuse, making it a loose upper reference. The shared row-column MILP uses one poisoned-shard allocation across all required cells, which is the proposed structured method.

### `validity_one_prompt_by_L.svg`

X-axis:

```text
L = sequence length
```

Curves:

```text
DPA weakest harmful token        dpa_val_row_weak_q1
TPA max-token sequence           tpa_val_sequence_q1
atomic phrase aggregation        phrase_dpa_val_q1
shared MILP full sequence        row_col_val_q1
independent full sequence        independent_val_sequence_q1
```

This answers: how much poisoning is needed to force one harmful generated sequence?
The DPA curve is a weak token-level reference, TPA max-token sequence is the targeted sequence baseline, atomic phrase aggregation treats the whole sequence as one label, the shared MILP is the proposed structured certificate, and independent composition is a loose upper reference.

### `validity_all_prompts_by_L.svg`

X-axis:

```text
L = sequence length
```

Curves:

```text
TPA max-token sequences for all prompts   tpa_val_sequence_qN
shared MILP all harmful sequences         row_col_val_qN
DPA weakest harmful token per prompt      dpa_val_row_weak_qN
independent all harmful sequences         independent_val_sequence_qN
```

This answers: how much poisoning is needed to force harmful sequences for all prompts?
The DPA curve means one harmful token per prompt, not full harmful sequences.

### `validity_independent_overestimate_by_L.svg`

X-axis:

```text
L = sequence length
```

Y-axis:

```text
independent_val_sequence_q1 / row_col_val_q1
```

If any shared-MILP denominator is zero, the plot falls back to:

```text
independent_val_sequence_q1 - row_col_val_q1
```

This diagnostic shows how much independent sequence composition overestimates the validity budget for forcing one harmful target sequence because it ignores shared poisoned-shard reuse.

### `stability_one_prompt_by_L.svg`

X-axis:

```text
L = sequence length
```

Curves:

```text
DPA weakest token                 dpa_stab_row_radius_q1
shared MILP: one token            row_col_stab_q1_r1
shared MILP: full sequence        row_col_stab_q1_rL
```

This answers: how much poisoning is needed to destabilise one prompt?

### `stability_all_prompts_by_L.svg`

X-axis:

```text
L = sequence length
```

Curves:

```text
shared MILP: one token per prompt        row_col_stab_qN_r1
shared MILP: full matrix                 row_col_stab_qN_rL
DPA weakest token per prompt             dpa_stab_row_radius_qN
```

This answers: how much poisoning is needed to destabilise all prompts?
Independent composition is kept out of this main plot because it often dominates the y-axis.

### `stability_independent_overestimate_by_L.svg`

X-axis:

```text
L = sequence length
```

Y-axis:

```text
independent_stab_full_row_qN / row_col_stab_qN_rL
```

If the legacy alias is present instead, the numerator may be `independent_stab_qN_rL`. If any shared-MILP denominator is zero, the plot falls back to:

```text
independent_stab_full_row_qN - row_col_stab_qN_rL
```

This diagnostic shows how much independent composition overestimates the stability budget for the full prompt-token matrix because it ignores shared poisoned-shard reuse.

### `structured_stability_heatmap.svg`

X-axis:

```text
affected tokens per prompt, r
```

Y-axis:

```text
affected prompts, q
```

Colour:

```text
poison budget B*(q,r)
```

This visualization is produced by the instance visualization command and shows the row-column stability structure directly.

### `validity_bias_sweep.svg`

X-axis:

```text
target_bias
```

Curves:

```text
DPA weakest harmful token
DPA weakest harmful token per prompt
TPA max-token sequence
shared MILP: one harmful sequence
shared MILP: harmful sequences for all prompts
atomic phrase aggregation
```

This shows how validity budgets change when the harmful target already has more natural vote support.

## Monotonicity Diagnostics

Every benchmark plot refresh checks expected ordering relationships among shared MILP objectives:

```text
row_col_stab_q1_r1 <= row_col_stab_q1_rL
row_col_stab_q1_r1 <= row_col_stab_qN_r1
row_col_stab_q1_rL <= row_col_stab_qN_rL
row_col_stab_qN_r1 <= row_col_stab_qN_rL
row_col_val_q1 <= row_col_val_qN
```

The plotter prints the number of violations. If any are found, it writes `monotonicity_violations.csv` in the plot output directory.

## Visualization Outputs

The visualization command:

```bash
./scripts/visualize.sh
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
- All-prompts objectives are stronger than one-prompt objectives and should usually have larger or equal budgets.
- Full-sequence or full-matrix objectives are stronger than one-token objectives and should usually have larger or equal budgets.
- If `is_optimal=False`, treat `B_star` as a feasible upper bound rather than an exact certificate.
- `attacked_cells` is diagnostic and may include extra feasible cells.
- Shared MILP budgets can be lower than independent-composition budgets because one poisoned shard allocation can satisfy multiple cell objectives.
- Aggregate plots average over other swept parameters, so inspect the CSV directly when debugging a surprising point.
