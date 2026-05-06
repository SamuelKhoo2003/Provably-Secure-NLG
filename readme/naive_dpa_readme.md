# Confirmed DPA Baselines

This note explains the DPA-style baselines currently implemented in `toy_certificate/experiments.py`.

The confirmed DPA matrix baseline is not the shared row-column MILP. It computes token-level certificates independently, reduces each prompt row to its weakest token, and then reports prompt-level `q` curves from those row radii.

## Where It Lives

The relevant functions are:

- `compute_reference_baselines(data)`
- `_cell_stability_budgets(data)`
- `_cell_validity_budgets(data)`
- `_phrase_dpa_validity_row_budgets(data)`
- `_min_budget_satisfying_all(deficits, contribs, ignored_class=None)`
- `_min_budget_from_contributions(deficit, contributions)`

These are called by:

```bash
python -m toy_certificate.experiments benchmark
```

and written to `benchmark_results.csv`.

## DPA Matrix Rule

For every prompt row `i` and token column `j`, compute a per-cell DPA radius:

```text
B_cell[i,j]
```

Then the confirmed prompt-level DPA radius is the weakest token in that prompt:

```text
row_radius[i] = min_j B_cell[i,j]
```

The budget at which at least `q` prompts are individually vulnerable is the `q`-th smallest row radius.

Special cases:

```text
q1 = min_i row_radius[i]
qN = max_i row_radius[i]
```

This is different from independent composition, which sums token costs.

## Stability

For stability, a cell is vulnerable if some competitor token can beat or tie the clean winner.

For cell `(i,j)`:

```text
w = clean_pred[i,j]
c = competitor token
```

For each competitor `c != w`, the deficit is:

```python
deficit = stab_counts[i,j,w] - stab_counts[i,j,c]
```

A poisoned shard contributes:

```python
influence[k,i,j] * (
    int(stab_votes[k,i,j] != c)
    + int(stab_votes[k,i,j] == w)
)
```

The implementation checks all competitors and takes:

```text
B_cell_stab[i,j] = min_c budget_to_make_c_tie_or_beat_w
```

The confirmed stability columns are:

```text
dpa_stab_cell_min       = min_{i,j} B_cell_stab[i,j]
dpa_stab_row_radius_q1  = min_i min_j B_cell_stab[i,j]
dpa_stab_row_radius_qN  = max_i min_j B_cell_stab[i,j]
```

`raw_dpa_stab_min_cell` is still emitted as a simple winner-vs-runner-up margin reference.

## Validity

For validity, a cell is vulnerable if the harmful target token can be made to beat or tie every competitor using one shard set for that cell.

For cell `(i,j)`:

```text
h = target[i,j]
```

For every competitor `c != h`, the target condition is:

```text
target_count_after_poisoning >= competitor_count_after_poisoning
```

A poisoned shard contributes against competitor `c` as:

```python
influence[k,i,j] * (
    int(val_votes[k,i,j] != h)
    + int(val_votes[k,i,j] == c)
)
```

The implementation searches for the smallest shard subset whose contributions satisfy all competitors at once. This replaces the older approximation:

```text
max_c independently computed budget_to_beat_c
```

The confirmed validity columns are:

```text
dpa_val_cell_min      = min_{i,j} B_cell_val[i,j]
dpa_val_row_weak_q1   = min_i min_j B_cell_val[i,j]
dpa_val_row_weak_qN   = max_i min_j B_cell_val[i,j]
```

`raw_dpa_val_min_cell` is kept as an alias for the weakest target-token diagnostic.

## Independent Composition

Independent composition is a separate conservative reference. It assumes token or prompt costs are paid separately and does not reuse poisoned shards across cells.

Stability:

```text
independent_stab_full_row_q1 = min_i sum_j B_cell_stab[i,j]
independent_stab_full_row_qN = sum_i sum_j B_cell_stab[i,j]
```

Validity:

```text
independent_val_sequence_q1 = min_i sum_j B_cell_val[i,j]
independent_val_sequence_qN = sum_i sum_j B_cell_val[i,j]
```

Compatibility aliases are still written:

```text
independent_stab_qN_rL = independent_stab_full_row_qN
independent_val_q1     = independent_val_sequence_q1
independent_val_qN     = independent_val_sequence_qN
```

## Phrase-DPA

Phrase-DPA collapses a full generated row into one atomic class.

For row `i`:

```text
phrase_vote[k,i] = tuple(val_votes[k,i,0:L])
target_phrase[i] = tuple(target[i,0:L])
```

The target phrase must beat or tie all observed phrase competitors using one poisoned-shard set. The implementation now checks all observed phrase competitors, not only the most frequent one.

Confirmed phrase-DPA columns:

```text
phrase_dpa_val_q1 = min_i phrase_row_radius[i]
phrase_dpa_val_qN = max_i phrase_row_radius[i]
```

Independent phrase composition is separate:

```text
phrase_independent_val_q1 = min_i phrase_row_radius[i]
phrase_independent_val_qN = sum_i phrase_row_radius[i]
```

## CSV Columns

Current baseline columns:

```text
raw_dpa_stab_min_cell
dpa_stab_cell_min
dpa_stab_row_radius_q1
dpa_stab_row_radius_qN

dpa_val_cell_min
dpa_val_row_weak_q1
dpa_val_row_weak_qN
raw_dpa_val_min_cell

independent_stab_full_row_q1
independent_stab_full_row_qN
independent_stab_qN_rL

independent_val_sequence_q1
independent_val_sequence_qN
independent_val_q1
independent_val_qN

phrase_dpa_val_q1
phrase_dpa_val_qN
phrase_independent_val_q1
phrase_independent_val_qN
```

## Interpretation

Use this language in reports:

```text
The naive DPA matrix baseline computes token-level certificates independently and aggregates a prompt by its weakest token. This mirrors standard per-sample DPA evaluation, but it does not model one shared adversarial allocation across multiple prompts or token positions.

The independent-composition baseline sums token costs and is a conservative reference, not the confirmed DPA matrix baseline.

The phrase-DPA baseline treats each generated sequence as one atomic label. This avoids explicit token-column modelling but suffers from vote diffusion over the exponentially large phrase space.

The proposed shared row-column MILP explicitly enforces one poisoned-shard set across the prompt-token matrix and can evaluate structured stability and sequence-validity objectives more faithfully.
```
