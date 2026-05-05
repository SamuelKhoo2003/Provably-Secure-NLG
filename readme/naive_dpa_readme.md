# Naive DPA Baselines

This note explains the DPA-style reference baselines currently implemented in `toy_certificate/experiments.py`.

The important point is that these are not Gurobi MILPs. They are fast per-cell calculations used as comparison baselines against the shared row-column MILP certificates.

## Where It Lives

The relevant functions are:

- `compute_reference_baselines(data)`
- `_cell_stability_budgets(data)`
- `_cell_validity_budgets(data)`
- `_phd_margin_stability_budgets(data)`
- `_phrase_dpa_validity_row_budgets(data)`
- `_min_budget_from_contributions(deficit, contributions)`

These functions are called during:

```bash
python -m toy_certificate.experiments benchmark
```

The results are written as extra columns in `benchmark_results.csv`.

## Poisoning Model

For each shard `k`, prompt row `i`, and token column `j`, the toy setup stores a vote:

```text
stab_votes[k, i, j]
val_votes[k, i, j]
```

The DPA baseline asks how many shards must be poisoned to flip an independent majority vote.

If shard `k` is poisoned and it influences cell `(i, j)`, the attacker is allowed to:

- remove that shard's current vote from its clean class;
- add that shard's vote to the desired attacker class.

The helper code models this with a per-shard contribution of `0`, `1`, or `2`.

## Per-Cell Stability

For stability, each cell has:

```text
w = clean winner token
c = runner-up token
```

The implementation compares only the clean winner against the runner-up:

```python
deficit = stab_counts[i, j, w] - stab_counts[i, j, c]
```

For each shard, the contribution is:

```python
influence[k, i, j] * (
    int(stab_votes[k, i, j] != c)
    + int(stab_votes[k, i, j] == w)
)
```

Interpretation:

- `+1` if the poisoned shard can be changed into the runner-up token `c`;
- `+1` if the poisoned shard currently votes for the winner `w`, so poisoning removes support from `w`;
- `0` if the shard does not influence the cell.

The contributions are sorted from largest to smallest. `_min_budget_from_contributions(...)` returns the smallest number of poisoned shards whose total contribution closes the winner-vs-runner-up deficit.

The raw stability CSV column is:

```text
raw_dpa_stab_min_cell = min over cells of ceil(margin / 2)
```

That column uses `_phd_margin_stability_budgets(...)`, which is the simple majority-margin version:

```python
((margins + 1) // 2)
```

## Per-Cell Validity

For validity, each cell has a harmful target token:

```text
h = target[i, j]
```

The baseline computes the cost to make `h` beat every competitor token. For each competitor `c`, it computes:

```python
deficit = val_counts[i, j, c] - val_counts[i, j, h]
```

For each shard, the contribution is:

```python
influence[k, i, j] * (
    int(val_votes[k, i, j] != h)
    + int(val_votes[k, i, j] == c)
)
```

Interpretation:

- `+1` if the poisoned shard can be changed into the harmful target `h`;
- `+1` if the poisoned shard currently votes for competitor `c`, so poisoning removes support from `c`;
- `0` if the shard does not influence the cell.

The cell budget is the maximum budget needed across all competitors:

```text
B_cell_validity[i, j] = max_c budget_to_make_h_beat_c
```

The raw validity CSV column is:

```text
raw_dpa_val_min_cell = min over cells of B_cell_validity[i, j]
```

This is a weakest-cell diagnostic. It does not certify that a full harmful sequence can be produced.

## Independent Composition

The code also builds simple sequence-level baselines by adding independent per-cell costs.

For stability:

```text
independent_stab_full_row_q1 = min_i sum_j B_cell_stability[i, j]
independent_stab_qN_rL      = sum_i sum_j B_cell_stability[i, j]
```

For validity:

```text
independent_val_q1 = min_i sum_j B_cell_validity[i, j]
independent_val_qN = sum_i sum_j B_cell_validity[i, j]
```

These are intentionally naive. They assume each token position is attacked independently and do not reuse the same poisoned shard across multiple cells.

Because of that, independent composition can overestimate the attack budget compared with the shared row-column MILP.

## Phrase-DPA Baseline

The current implementation also includes a phrase-level DPA reference for validity.

For each row `i`, the length-`L` vote from shard `k` is collapsed into one phrase class:

```text
phrase[k, i] = tuple(val_votes[k, i, 0:L])
```

The harmful target phrase is:

```text
target_phrase[i] = tuple(target[i, 0:L])
```

The code counts phrase votes and computes how many poisoned shards are needed to make the target phrase beat the most frequent non-target phrase.

CSV columns:

```text
phrase_dpa_val_q1 = min_i phrase_row_budget[i]
phrase_dpa_val_qN = sum_i phrase_row_budget[i]
```

This mimics phrase-level DPA where the whole output sequence is treated as one large class. It avoids token-column modelling, but as `L` grows the number of possible phrases grows quickly and votes can diffuse across many phrase classes.

## CSV Columns

Current benchmark columns from `compute_reference_baselines(...)`:

```text
raw_dpa_stab_min_cell
raw_dpa_val_min_cell
independent_stab_full_row_q1
independent_stab_qN_rL
independent_val_q1
independent_val_qN
phrase_dpa_val_q1
phrase_dpa_val_qN
```

Legacy column names are still accepted when replotting older CSV files:

```text
naive_dpa_stability_full_row -> independent_stab_full_row_q1
naive_dpa_validity_q1        -> independent_val_q1
naive_dpa_validity_qN        -> independent_val_qN
phd_ref_stability_any_cell   -> raw_dpa_stab_min_cell
phd_ref_validity_any_cell    -> raw_dpa_val_min_cell
```

## How To Interpret It

Use the naive DPA baselines as references, not as the main certificate.

- Raw per-cell DPA answers: "How easy is one token cell to flip?"
- Independent composition answers: "What if we add token costs without shard reuse?"
- Phrase-DPA answers: "What if the whole generated phrase is one class?"
- The shared row-column MILP answers: "What is the minimum shared poisoned-shard set that satisfies the structured row/column attack objective?"

The shared MILP can be lower than independent composition because one poisoned shard can help attack multiple cells at the same time. That is not a bug; it means the independent baseline was conservative for the joint objective.
