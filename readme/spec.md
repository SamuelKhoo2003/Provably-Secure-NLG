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

The shared MILP should be compared against:

```text
DPA weakest-token stability/validity baseline
TPA targeted validity baseline
independent-composition baseline
atomic phrase aggregation baseline
```

The confirmed DPA matrix baseline computes token-level cell certificates independently and reduces each prompt row to its weakest token:

```text
row_radius[i] = min_j B_cell[i,j]
```

For validity, this is the easiest harmful target token in a prompt row, not a
full harmful-sequence certificate.

The updated NLG certification paper separates DPA-style stability from
TPA-style targeted validity. DPA remains the natural baseline for untargeted
stability, where the adversary tries to change the clean output. For validity,
the relevant paper-inspired baseline is Targeted Partition Aggregation, which
computes a targeted radius for inducing a specific harmful token. For a harmful
sequence, the toy implementation composes token-level TPA radii using the
maximum over token positions:

```text
R_tpa_sequence[i] = max_j r_tpa[i,j]
```

The maximum is used because the attacker must force every target token in the
sequence, so the hardest token controls the targeted sequence baseline. This is
not ordinary DPA top-vs-runner-up stability and not phrase aggregation. The toy
implementation follows the MILP tie convention where target ties count as
successful attacks; if a strict-plurality convention is used elsewhere, interpret
this as the tie-wins toy adaptation.

The independent-composition baseline sums token costs:

```text
full_row_cost[i] = sum_j B_cell[i,j]
```

This does not reuse the same poisoned-shard allocation and should be treated as
a loose/conservative upper reference.

The atomic phrase aggregation baseline treats an entire generated row as one
atomic class:

```text
phrase_vote[k,i] = tuple(val_votes[k,i,0:L])
```

It is useful as a crude full-sequence baseline, but it is not the main TPA/PHD
validity baseline. It often weakens as `L` grows because exact sequence
agreement across shards becomes rare.

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

The benchmark should write reusable CSV data, so plots can be changed without rerunning Gurobi:

```text
benchmark_results.csv
```

Plotting should be a separate step from benchmark data generation.

Report-facing plots should be separated by attack objective:

```text
validity_one_prompt_by_L.svg          force one harmful sequence
validity_all_prompts_by_L.svg         force harmful sequences for all prompts
stability_one_prompt_by_L.svg         destabilise one prompt
stability_all_prompts_by_L.svg        destabilise all prompts
validity_independent_overestimate_by_L.svg
stability_independent_overestimate_by_L.svg
structured_stability_heatmap.svg      B*(q,r) over affected prompts/tokens
```

Legends should use human-readable objective names such as `one prompt`, `all prompts`, `one token`, `full sequence`, `full matrix`, `DPA weakest token`, `DPA weakest harmful token`, `TPA max-token sequence`, `atomic phrase aggregation`, `shared MILP full sequence`, and `independent full sequence`.
Independent-composition diagnostics should be separated from main plots when
they dominate the y-axis.
