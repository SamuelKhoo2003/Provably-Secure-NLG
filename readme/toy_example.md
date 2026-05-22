# Toy Example Guide

This document explains the toy experiment used by the first-party code in
`toy_certificate/`. The toy setting is deliberately small, but it mirrors the
structure of a certified natural-language generation experiment: many shard
models vote on token outputs, and a certificate asks how many shards an
adversary must poison before a stability or validity attack becomes feasible.

## High-Level Picture

The experiment builds a grid of generated outputs:

```text
prompt rows:          i = 0, ..., N-1
token positions:      j = 0, ..., L-1
vocabulary tokens:    t = 0, ..., T-1
shard models:         k = 0, ..., K-1
```

For every shard `k`, prompt row `i`, and token position `j`, the toy generator
stores one token vote:

```text
stab_votes[k, i, j]   vote under the clean prefix
val_votes[k, i, j]    vote under the harmful-target prefix
```

Votes are counted into:

```text
stab_counts[i, j, t]  number of clean-prefix shards voting for token t
val_counts[i, j, t]   number of harmful-prefix shards voting for token t
```

The clean model output at a cell is the majority token:

```text
clean_pred[i, j] = argmax_t stab_counts[i, j, t]
```

Ties are resolved by NumPy's `argmax`, so the smallest tied token id wins.

## Why There Are Two Vote Tensors

The toy experiment separates two attack stories.

Stability uses `stab_votes`. This represents the normal generation setting:
each shard votes for the clean response token at each prompt and position. A
stability attack tries to change the output away from `clean_pred`.

Validity uses `val_votes`. This represents a harmful-target prefix or targeted
generation setting. The adversary wants the harmful target token sequence to win.
A validity attack tries to force `target[i, j]` at every required token
position.

The two tensors are generated from the same dimensions but with different
sampling rules, because they support different certificates.

## How Clean Stability Votes Are Generated

The generator starts by sampling a base clean token for every prompt-token cell:

```text
base_token[i, j] in {0, ..., T-1}
```

Then each shard initially copies that base token. With probability
`delta_stab`, the shard disagrees and instead votes for a different token:

```text
stab_votes[k, i, j] = base_token[i, j]                 with probability 1 - delta_stab
stab_votes[k, i, j] = some token != base_token[i, j]   with probability delta_stab
```

After all `K` shards vote, the code counts votes and defines:

```text
clean_pred[i, j]  majority token at the clean cell
runner_up[i, j]   second-ranked token by clean vote count
```

`runner_up` is mainly used by DPA-style margins and by the optional
`runner_up` approximation mode. The exact shared MILP can check every competing
token, not just the runner-up.

## How Harmful Validity Votes Are Generated

First, the generator samples a harmful target token for every cell:

```text
target[i, j] != clean_pred[i, j]
```

Then it samples a non-target base token:

```text
val_base[i, j] != target[i, j]
```

For every shard, prompt, and token position, validity votes are built as:

```text
val_votes[k, i, j] = val_base[i, j]                    by default
val_votes[k, i, j] = noisy non-base token              with probability delta_val
val_votes[k, i, j] = target[i, j]                      with probability target_bias
```

The final target assignment takes priority. Larger `target_bias` gives the
harmful target more initial support, so validity certificates usually become
smaller as `target_bias` increases. That is why the sensitivity plot uses target
bias on the x-axis.

## Influence Masks

The poisoning variable is shard-level:

```text
a[k] = 1 if shard k is poisoned
a[k] = 0 otherwise
```

The influence mask says whether poisoning shard `k` can affect cell `(i, j)`:

```text
influence[k, i, j] in {0, 1}
```

Supported modes:

```text
dense        every poisoned shard can affect every cell
row-local    each shard affects only selected prompt rows
column-local each shard affects only selected token positions
```

The main shared-budget story is about reusing the same poisoned shard allocation
across the full prompt-token grid. If shard `k` is poisoned, that same choice is
shared by all rows and columns where `influence[k, i, j] = 1`.

## Stability Certificates

Stability asks how many shards must be poisoned before the clean output can be
changed.

For one cell `(i, j)`, let:

```text
w = clean_pred[i, j]
c = competitor token, c != w
```

A stability attack succeeds at that cell if, after poisoning selected shards,
some competitor `c` ties or beats the clean winner `w`.

The exact MILP checks all competitors. The cheap `runner_up` mode checks only
the original runner-up token. The runner-up mode is faster but can overestimate
robustness.

## Validity Certificates

Validity asks how many shards must be poisoned before a harmful target token or
sequence can be forced.

For one cell `(i, j)`, let:

```text
h = target[i, j]
```

A validity attack succeeds at that cell if `h` ties or beats every non-target
competitor token after poisoning. For a full harmful sequence, every token
position in the row must satisfy that target condition.

Ties count as successful attacks for the adversary. Therefore a certificate at
budget `B` is counted as valid only when:

```text
B < B*
```

If `B == B*`, the attack is already feasible at that budget.

## Main Methods

The cleaned plotting pipeline uses these thesis-facing method names:

```text
Joint row-column MILP
DPA weakest-token baseline
PHD sequence baseline
```

`Joint row-column MILP` is the proposed method. It solves for one shared
poisoned-shard allocation across prompt rows and token-position columns.

`DPA weakest-token baseline` is a simple token-level baseline. For a generated
sequence, it reduces the sequence-level robustness to the easiest token to
attack.

`PHD sequence baseline` is mapped from the current `TPA max-token sequence`
export. For validity, it computes targeted token radii and uses the hardest
token position in the sequence.

Row-only and column-only MILPs are ablations of the proposed method. They appear
only in ablation plots:

```text
Row-only MILP ablation
Column-only MILP ablation
```

Independent composition is excluded from main plots because it combines
token-level certificates without faithfully modelling reuse of poisoned shards
across token positions.

## CSV Outputs

`benchmark_results.csv` is the wide summary CSV. It stores one row per
parameter setting and many certificate columns, for example:

```text
row_col_stab_qN_rL     joint full-grid stability certificate
row_col_val_qN         joint validity certificate for all prompt rows
dpa_stab_row_radius_qN DPA weakest-token stability baseline
dpa_val_row_weak_qN    DPA weakest-token validity baseline
tpa_val_sequence_qN    PHD/TPA sequence validity baseline
```

`benchmark_budget_curves.csv` stores radius-derived certified fractions by
budget. It is used for:

```text
main_stability_budget_curve.png
main_validity_budget_curve.png
```

`benchmark_damage_curves.csv` stores direct fixed-budget damage MILPs. It is
used for:

```text
direct_damage_curve_joint_milp.png
```

The direct damage curve is audit evidence: it fixes a poisoned-shard budget and
maximizes how many prompt rows can be attacked under the actual shared-budget
MILP.

## What `plot.sh` Does

`scripts/plot.sh` does not rerun Gurobi or regenerate benchmark data. It reads
existing CSVs and refreshes plots.

For one explicit CSV:

```bash
CSV_PATH=toy_results/medium_benchmark/benchmark_results.csv OUT_DIR=plots ./scripts/plot.sh
```

If `CSV_PATH` is not set, it searches under `toy_results/*/benchmark_results.csv`
and refreshes plots in each benchmark folder.

The current cleaned plotting path writes:

```text
main_stability_budget_curve.png
main_validity_budget_curve.png
stability_certificate_vs_K.png
validity_certificate_vs_K.png
validity_sensitivity_target_bias.png
direct_damage_curve_joint_milp.png
ablation_stability_row_column_joint.png
ablation_validity_row_column_joint.png
audit_milp_vs_phd_equivalence.csv
audit_method_mapping.txt
```

It no longer generates the old SVG plot set.

## Audit Outputs

`audit_method_mapping.txt` records how raw CSV method names map into the thesis
terminology:

```text
Shared MILP             -> Joint row-column MILP
DPA token margin        -> DPA weakest-token baseline for stability
DPA weakest harmful token -> DPA weakest-token baseline for validity
TPA max-token sequence  -> PHD sequence baseline for validity
```

It also records excluded methods and ablation-only methods.

`audit_milp_vs_phd_equivalence.csv` compares the joint MILP validity curve and
the PHD sequence baseline curve for every parameter setting. It records whether
the two curves are exactly identical across budgets, whether they differ, and
the first budget where a difference appears.

This audit is important because similar-looking curves do not automatically mean
the same algorithm was run twice. In the current implementation, the PHD
sequence baseline and the joint MILP are computed by different code paths.

## Useful Commands

Run one sanity instance:

```bash
.venv/bin/python -m toy_certificate.experiments sanity --K 8 --N 2 --L 6 --T 4
```

Generate benchmark CSVs:

```bash
./scripts/data.sh
```

Refresh cleaned plots:

```bash
./scripts/plot.sh
```

Generate data and plots:

```bash
./scripts/benchmark.sh
```

Run tests:

```bash
.venv/bin/python -m unittest discover
```
