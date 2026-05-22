# Toy Certificate Spec

This document consolidates the design/specification notes for the toy row/column poisoning certificate experiment.

## Goal

Build a small Python + Gurobi experiment that models poisoning robustness for generated token votes. The toy setting uses a prompt-by-token matrix:

```text
N = number of prompt rows
L = generated token columns / sequence length
K = shards / DPA partitions / base models
T = toy vocabulary size
```

Each shard casts one token vote for every prompt/token cell:

```text
votes[k, i, j] in {0, ..., T-1}
```

The core quantity is:

```text
B* = minimum number of poisoned shards needed to make an attack objective feasible
```

The benchmark also supports budget-sweep data. Radius-derived coverage fixes a
budget `B` and certifies a row if and only if `B < B*_row`. Direct
damage-at-budget MILPs fix `B` and maximize attacked rows using one shared
poisoned-shard allocation. Horizon metrics fix `B` and measure the longest
certified prefix per prompt row.

Certified percentage is always a strict-radius statement. For per-row radius
`B*_i`, row `i` is certified at poison budget `B` exactly when `B < B*_i`.
If `B == B*_i`, the attack is feasible at that budget, so the row is no longer
certified. Unknown or non-finite radii should be recorded as unknown and treated
as not certified in percentages.

The main modelling requirement is that the MILPs use one shared poisoning allocation:

```text
a[k] in {0, 1}
```

The same selected poisoned shards are reused across all required rows and columns. This is the key distinction from token-wise independent certificates.

## Toy Data

The generator should create separate vote tensors for stability and validity:

```text
stab_votes[k, i, j]
val_votes[k, i, j]
```

Stability votes model the clean prefix. For each cell, a clean base token is sampled and shards mostly follow it, with disagreement controlled by:

```text
delta_stab
```

Validity votes model a harmful-target prefix. The generator also samples:

```text
target[i, j] != clean_pred[i, j]
```

Validity support is controlled by:

```text
delta_val
target_bias
```

The count tensors are:

```text
stab_counts[i, j, t] = number of stability votes for token t
val_counts[i, j, t]  = number of validity votes for token t
```

From stability counts, compute:

```text
clean_pred[i, j] = majority token
runner_up[i, j]  = second-ranked token
```

## Poisoning Model

The adversary corrupts shards:

```text
a[k] = 1 means shard k is poisoned
B = sum_k a[k]
```

If shard `k` is poisoned, the attacker can change that shard's vote arbitrarily for any influenced cell. The optional influence mask is:

```text
influence[k, i, j] in {0, 1}
```

Supported influence modes:

```text
dense         every shard influences every prompt/token cell
row-local     a shard influences selected prompt rows
column-local  a shard influences selected token columns
```

## Cell Conditions

### Stability

A stability attack succeeds at cell `(i,j)` if some competitor token can beat or tie the clean winner.

For clean winner `w = clean_pred[i,j]` and competitor `c != w`:

```text
stab_counts[i,j,c]
+ sum_k a[k] * influence[k,i,j] * 1[stab_votes[k,i,j] != c]
>=
stab_counts[i,j,w]
- sum_k a[k] * influence[k,i,j] * 1[stab_votes[k,i,j] == w]
```

The implementation checks all competitors, not only the original runner-up.
The runner-up margin remains useful for simple DPA-style baselines, but the
shared MILP itself is all-competitor.

The stability solver also exposes a `runner_up` competitor mode for diagnostics.
This cheaper mode checks only the original runner-up token and is a DPA-style
top-vs-second simplification. It may overestimate robustness when a
non-runner-up competitor is easier to promote. If both modes solve optimally:

```text
B*_runner_up >= B*_all
```

Use all-competitor mode for the exact threat model and runner-up mode only after
comparison diagnostics show it is acceptably close for the chosen toy setting.

### Validity

A validity attack succeeds at cell `(i,j)` if harmful target `h = target[i,j]` beats or ties every competitor `c != h`:

```text
val_counts[i,j,h]
+ sum_k a[k] * influence[k,i,j] * 1[val_votes[k,i,j] != h]
>=
val_counts[i,j,c]
- sum_k a[k] * influence[k,i,j] * 1[val_votes[k,i,j] == c]
```

This condition is enforced for all competitors.
Validity is unchanged by stability competitor mode.

## MILP Variables

Common variables:

```text
a[k]      binary poisoned-shard allocation
z[i,j]    binary attacked-cell indicator
y_row[i]  optional attacked-row indicator
y_col[j]  optional attacked-column indicator
```

Objective:

```text
minimize sum_k a[k]
```

The MILP returns the minimum poison budget `B*`.

## Certificate Objectives

The experiment should support:

```text
row_stability
column_stability
structured row/column stability
row_validity
column_validity
row/column validity
```

Structured stability uses:

```text
q = number of prompt rows required
r = number of destabilised token cells per selected row
```

Common special cases:

```text
q = 1, r = 1  one prompt, one token
q = 1, r = L  one prompt, full sequence
q = N, r = 1  all prompts, one token each
q = N, r = L  all prompts, full matrix
```

Validity uses:

```text
q = 1  one harmful target sequence
q = N  harmful target sequences for all prompts
```

Report-facing stability objectives should use `solve_structured_stability`.
Report-facing validity objectives should use `solve_row_col_validity`.

## Reference Baselines

There are two main external baselines.

The first is the token-level DPA margin baseline. This is a cell-wise baseline.
For stability it uses the standard DPA top-vs-runner-up margin at each
prompt-token cell and answers how robust a single token prediction is to
arbitrary change. It reduces each prompt row to its weakest token when a
row-level summary is needed:

```text
row_radius[i] = min_j B_cell[i,j]
```

For validity, it uses a simple targeted margin between the harmful target token
and the strongest non-target competitor. This is the easiest harmful target
token in a prompt row, not a full harmful-sequence certificate.

The second is the Ghitu-style phrase-level TPA baseline. This is the row-wise
certified NLG baseline. For phrase stability, a row radius is the minimum over
token-level stability radii because a response is no longer stable if any one
token changes. For phrase validity, the toy implementation computes a targeted
radius for inducing each specific harmful token and composes token-level TPA
radii using the maximum over token positions:

```text
R_tpa_sequence[i] = max_j r_tpa[i,j]
```

The maximum is used because the attacker must force every target token in the
sequence, so the hardest target token controls the phrase certificate. This is
the main paper-inspired validity baseline. It is not ordinary DPA
top-vs-runner-up stability and not atomic phrase aggregation. The toy
implementation follows the MILP tie convention where target ties count as
successful attacks; if a strict-plurality convention is used elsewhere,
interpret this as the tie-wins toy adaptation.

The independent-composition diagnostic sums token costs:

```text
full_row_cost[i] = sum_j B_cell[i,j]
```

This does not reuse the same poisoned-shard allocation and should be treated as a loose/conservative upper reference.

The atomic phrase aggregation diagnostic treats an entire generated row as one atomic class:

```text
phrase_vote[k,i] = tuple(val_votes[k,i,0:L])
```

It is useful as a crude full-sequence diagnostic, but it is not the main TPA/PHD validity baseline. It often weakens as `L` grows because exact sequence agreement across shards becomes rare.

The proposed shared row-column MILP is evaluated against the two external baselines. Row-only, column-only, and joint row-column MILPs are variants or ablations of the proposed method, not external baselines.

## Solver Exactness and Diagnostics

`CertificateResult` includes:

```text
is_optimal
mip_gap
lower_bound
upper_bound
```

If `is_optimal=True`, `B_star` is an exact optimum. If Gurobi stops with
`TIME_LIMIT` or `SUBOPTIMAL` and a feasible solution exists, `B_star` is the best
feasible upper bound found, not an exact certificate.

`attacked_cells` is diagnostic. Since attacked-cell variables are not
secondarily minimized, the list may include extra feasible cells. The certified
quantity is `B_star`.

## Expected Outputs

The benchmark should write reusable CSV data, so plots can be changed without rerunning Gurobi. The default quick workflow uses the small preset and writes to `toy_results/small_benchmark/benchmark_results.csv`:

```bash
./scripts/data.sh
./scripts/plot.sh
```

Larger sweeps remain available through `PRESET=medium`, `PRESET=large`, or direct CLI arguments.

Reusable CSV data:

```text
benchmark_results.csv
benchmark_budget_curves.csv
benchmark_damage_curves.csv
benchmark_horizons.csv
```

Plotting should be a separate step from benchmark data generation.

`benchmark_results.csv` preserves the existing radius-style `B*` columns.
`benchmark_budget_curves.csv` stores cheap radius-derived certified and attacked
fractions. Radius-derived coverage computes per-row radii first, then counts
rows satisfying `B < B*_row`; it does not directly optimize one shared
allocation across all rows at budget `B`. `benchmark_damage_curves.csv` stores
fixed-budget shared-MILP damage maximization results and solver metadata;
non-optimal rows are feasible damage bounds rather than exact maxima.
`benchmark_horizons.csv` stores average, median, minimum, and maximum certified
prefix horizons.

Report-facing plots should be separated by attack objective:

```text
validity_one_prompt_by_L.svg          force one harmful sequence
validity_all_prompts_by_L.svg         force harmful sequences for all prompts
stability_one_prompt_by_L.svg         destabilise one prompt
stability_all_prompts_by_L.svg        destabilise all prompts
validity_independent_overestimate_by_L.svg
stability_independent_overestimate_by_L.svg
certified_fraction_stability_by_budget.svg
certified_fraction_validity_by_budget.svg
certified_fraction_stability_by_L_at_budget.svg
certified_fraction_validity_by_L_at_budget.svg
stability_horizon_by_budget.svg
validity_horizon_by_budget.svg
structured_stability_heatmap.svg      B*(q,r) over affected prompts/tokens
```

Legends should use human-readable objective names such as `DPA weakest token`, `DPA weakest harmful token`, `TPA max-token sequence`, `Atomic phrase aggregation`, `Shared MILP full sequence`, `Shared MILP all harmful sequences`, `Shared MILP one prompt, one token`, `Shared MILP one prompt, full sequence`, `Shared MILP all prompts, one token each`, `Shared MILP full matrix`, and `Independent full sequence`.
Independent-composition diagnostics should be separated from main plots when
they dominate the y-axis.
