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

The reported implementation checks all competitors, not only the runner-up.

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
q1 r1 = at least one token in at least one row
q1 rL = full response for at least one row
qN r1 = at least one token in every row
qN rL = full prompt-token matrix
```

Validity uses:

```text
q1 = full harmful target sequence in at least one row
qN = full harmful target sequence in all N rows
```

## Reference Baselines

The shared MILP should be compared against:

```text
confirmed DPA matrix baseline
independent-composition baseline
phrase-DPA baseline
```

The confirmed DPA matrix baseline computes token-level cell certificates independently and reduces each prompt row to its weakest token:

```text
row_radius[i] = min_j B_cell[i,j]
```

The independent-composition baseline sums token costs:

```text
full_row_cost[i] = sum_j B_cell[i,j]
```

The phrase-DPA baseline treats an entire generated row as one atomic class:

```text
phrase_vote[k,i] = tuple(val_votes[k,i,0:L])
```

## Expected Outputs

The benchmark should write reusable CSV data, so plots can be changed without rerunning Gurobi:

```text
benchmark_results.csv
```

Plotting should be a separate step from benchmark data generation.
