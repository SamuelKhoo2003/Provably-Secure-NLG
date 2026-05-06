# Confirmed Naive DPA Baseline Implementation

This README describes the confirmed DPA-style baseline to use on the final prompt-token matrix.

The goal of this baseline is **not** to solve the shared row-column MILP.  
The goal is to reproduce the simple DPA-style reference:

```text
1. compute a token/cell-level DPA margin certificate;
2. aggregate each prompt row by taking the weakest token certificate;
3. evaluate how many prompt rows are vulnerable/certified at a given poison budget.
```

This matches the confirmed approach:

```text
For every token, generate the DPA margin.
For each row/prompt, take the smallest token certificate.
Then check how many prompts can be affected at budget b.
```

---

## Objects

The toy matrix has:

```python
K = number of shards / base models
N = number of prompts / rows
L = token horizon / columns
T = number of possible tokens/classes
```

Votes:

```python
stab_votes[k, i, j]
val_votes[k, i, j]
```

where:

```text
k = shard index
i = prompt row
j = token column
```

Counts:

```python
stab_counts[i, j, t]
val_counts[i, j, t]
```

Clean prediction:

```python
clean_pred[i, j]
```

Harmful target:

```python
target[i, j]
```

Optional influence mask:

```python
influence[k, i, j] in {0, 1}
```

---

## Important distinction

There are two different baseline styles.

### 1. Confirmed DPA matrix baseline

This is the baseline this README recommends.

It computes:

```text
cell budget -> row budget -> number of vulnerable rows at budget b
```

For a row/prompt:

```text
row_radius[i] = min_j cell_radius[i,j]
```

Then for budget `b`:

```text
affected_prompts(b) = sum_i 1[row_radius[i] <= b]
certified_prompts(b) = N - affected_prompts(b)
```

If you want the budget where at least `q` prompts become vulnerable:

```text
B*_DPA(q) = q-th smallest value in {row_radius[i]}.
```

This matches the usual individual-DPA certificate curve: each prompt is judged independently by its weakest token.

### 2. Independent-composition baseline

This is a different baseline.

It assumes token or prompt attack costs are paid separately:

```text
full_row_cost[i] = sum_j cell_radius[i,j]
B*_independent(q) = sum of q smallest full_row_cost[i]
```

This is useful as a conservative reference, but it is **not** the confirmed DPA matrix baseline.

The implementation should not confuse these two.

---

## Stability DPA baseline

Stability asks whether a token changes away from the clean winner.

For each cell `(i,j)`:

```python
w = clean_pred[i,j]
```

Let `c` be a competing token.

The stability condition for competitor `c` is:

```text
poisoned_count(c) >= poisoned_count(w)
```

Under the DPA-style worst-case vote model, a poisoned shard can:

```text
+1 by adding its vote to competitor c, if it did not already vote for c;
+1 by removing its vote from winner w, if it currently voted for w.
```

So for a fixed competitor `c`, shard `k` contributes:

```python
influence[k, i, j] * (
    int(stab_votes[k, i, j] != c)
    + int(stab_votes[k, i, j] == w)
)
```

The deficit to close is:

```python
deficit = stab_counts[i, j, w] - stab_counts[i, j, c]
```

The budget for competitor `c` is the smallest number of shards whose sorted contributions close the deficit.

### Exact per-cell stability

For exact per-cell DPA stability, check all competitors:

```text
B_cell_stab[i,j] = min_{c != w} budget_to_make_c_beat_or_tie_w
```

Runner-up-only is only an approximation:

```text
B_cell_stab_runner_up[i,j] = budget_to_make_runner_up_beat_or_tie_w
```

The implementation should prefer the all-competitor version for reported results.

### Row-level DPA stability

For each prompt row:

```text
row_stab_radius[i] = min_j B_cell_stab[i,j]
```

Interpretation:

> A prompt row is unstable if any generated token in that row can change.

This is a row-aggregated token certificate.

### Budget curve

For budget `b`:

```text
dpa_stab_affected_rows(b)
=
sum_i 1[row_stab_radius[i] <= b]
```

The budget to make at least `q` rows individually vulnerable is:

```text
B*_dpa_stab(q)
=
q-th smallest row_stab_radius[i]
```

Special cases:

```text
q = 1:
    weakest prompt-level stability radius

q = N:
    budget at which every prompt is individually uncertified
```

This is **not** the same as a shared-allocation MILP saying one attack of size `b` changes `q` prompts.  
It is the naive DPA baseline that ignores cross-prompt budget coupling.

---

## Validity DPA baseline

Validity asks whether a target harmful token wins.

For each cell `(i,j)`:

```python
h = target[i,j]
```

Target validity requires:

```text
target h beats/ties every competitor c != h
```

For a fixed competitor `c`, the deficit is:

```python
deficit = val_counts[i,j,c] - val_counts[i,j,h]
```

A poisoned shard can contribute:

```python
influence[k, i, j] * (
    int(val_votes[k, i, j] != h)
    + int(val_votes[k, i, j] == c)
)
```

Interpretation:

```text
+1 if the shard can be changed into the harmful target h;
+1 if the shard currently votes for competitor c and can be removed from c.
```

### Important correctness issue

Do **not** compute per-cell targeted validity as:

```text
max_c independently_computed_budget_to_beat_c
```

unless clearly labelled as an approximation.

Why?

Because the target token must beat all competitors using the **same poisoned shard set**.

The independently optimal shard set for beating competitor `c1` may be different from the independently optimal shard set for beating competitor `c2`.

Therefore, for exact per-cell targeted validity, use one of the following.

### Exact per-cell validity option A: small MILP

For each cell `(i,j)`, solve a tiny MILP:

Variables:

```python
a[k] in {0,1}
```

Objective:

```python
minimize sum_k a[k]
```

Constraints for every competitor `c != h`:

```text
val_counts[i,j,h]
+ sum_k a[k] * influence[k,i,j] * 1[val_votes[k,i,j] != h]
>=
val_counts[i,j,c]
- sum_k a[k] * influence[k,i,j] * 1[val_votes[k,i,j] == c]
```

This gives:

```text
B_cell_val[i,j]
```

This is the cleanest exact DPA-style per-cell target-validity baseline.

### Exact per-cell validity option B: budget feasibility search

For each budget `b = 0, ..., K`, check whether there exists a set of `b` shards that makes the target beat all competitors.

This can be brute-forced only for very small `K`.

For larger `K`, use the tiny MILP above.

---

## Row-level DPA validity

There are two possible DPA-style row validity baselines.

### A. Weakest-token target diagnostic

This asks whether any target token cell is easy to force:

```text
row_val_weakest_token[i] = min_j B_cell_val[i,j]
```

This is only diagnostic. It does **not** certify that the full harmful sequence can be produced.

### B. Sequence validity composition baseline

For actual sequence validity, the target phrase requires all tokens:

```text
forall j: target[i,j] wins.
```

An independent sequence-composition baseline is:

```text
row_val_cost_independent[i] = sum_j B_cell_val[i,j]
```

The shared row-column MILP is the main proposed method for a tighter sequence-validity certificate because it uses one global `a[k]` across all target tokens.

---

## Phrase-DPA baseline

The phrase-DPA baseline treats a full length-`L` generation as one atomic class.

For row `i`:

```python
phrase_vote[k, i] = tuple(val_votes[k, i, 0:L])
target_phrase[i] = tuple(target[i, 0:L])
```

Count phrase votes:

```python
phrase_counts[i, phrase]
```

Then compute the DPA target certificate for making:

```text
target_phrase[i]
```

win.

### Correct phrase-DPA target condition

Target phrase must beat/tie **all observed phrase competitors**, not just the most frequent competitor.

For each row `i`, solve the same targeted plurality problem at phrase level:

```text
target phrase count after poisoning
>=
competitor phrase count after poisoning
for all competitor phrases.
```

A poisoned shard can:

```text
+1 by moving its phrase vote into the target phrase;
+1 by removing its phrase vote from a competitor phrase.
```

So phrase-DPA is the same as per-cell targeted validity, but with phrase labels instead of token labels.

### Phrase-DPA q curves

For each row, compute:

```text
phrase_row_radius[i]
```

Then:

```text
phrase_dpa_q1 = min_i phrase_row_radius[i]
phrase_dpa_q  = q-th smallest phrase_row_radius[i]
phrase_dpa_qN = max_i phrase_row_radius[i]
```

These are the individual-DPA-style q curves.

If instead you sum row costs:

```text
sum of q smallest phrase_row_radius[i]
```

that is an independent-composition baseline, not the naive DPA q curve.

---

## Shared row-column MILP comparison

The proposed method is different.

It solves:

```text
minimize sum_k a[k]
```

subject to a structured objective such as:

```text
at least q rows have at least r changed tokens       # stability
```

or:

```text
at least q rows contain the full harmful sequence    # validity
```

using one shared allocation vector:

```python
a[k]
```

across all rows and columns.

This can be higher or lower than the naive baselines depending on the comparison:

```text
Compared with weakest-cell DPA:
    shared row-column validity is usually stronger/larger because it requires a full sequence.

Compared with independent-sum composition:
    shared row-column MILP can be smaller because poisoned shards can be reused across cells.

Compared with phrase-DPA:
    shared row-column validity keeps token-level structure instead of collapsing the phrase into one huge class.
```

---

## Recommended CSV columns

Use names that make the distinction clear.

### Per-cell / DPA matrix baselines

```text
dpa_stab_cell_min
dpa_stab_row_radius_q1
dpa_stab_row_radius_qN

dpa_val_cell_min
dpa_val_row_weak_q1
dpa_val_row_weak_qN
```

### Independent composition baselines

```text
independent_stab_full_row_q1
independent_stab_full_row_qN

independent_val_sequence_q1
independent_val_sequence_qN
```

### Phrase-DPA baselines

```text
phrase_dpa_val_q1
phrase_dpa_val_qN

phrase_independent_val_q1
phrase_independent_val_qN
```

### Shared MILP certificates

```text
row_col_stab_q1_r1
row_col_stab_q1_rL
row_col_stab_qN_r1
row_col_stab_qN_rL

row_col_val_q1
row_col_val_qN
```

---

## What to change from the current implementation

If the current implementation does the following:

```text
raw_dpa_stab_min_cell = min over cells of ceil(margin / 2)
independent_val_qN = sum_i sum_j B_cell_validity[i,j]
phrase_dpa_val_qN = sum_i phrase_row_budget[i]
```

then rename/reorganise them as follows.

### Stability

Keep:

```text
raw_dpa_stab_min_cell
```

but label it as:

```text
simple margin DPA stability reference
```

Add or prefer:

```text
dpa_stab_cell_min
```

using the contribution-based all-competitor calculation.

Add:

```text
dpa_stab_row_radius_q1 = min_i row_stab_radius[i]
dpa_stab_row_radius_qN = max_i row_stab_radius[i]
```

where:

```text
row_stab_radius[i] = min_j B_cell_stab[i,j]
```

### Validity

Change per-cell validity to exact shared-competitor per-cell MILP.

Keep:

```text
raw_dpa_val_min_cell
```

but label it as:

```text
weakest target-token diagnostic
```

Do not present it as sequence validity.

### Phrase-DPA

Change phrase-DPA so target phrase beats all observed phrase competitors.

Change:

```text
phrase_dpa_val_qN = sum_i phrase_row_budget[i]
```

to either:

```text
phrase_dpa_val_qN = max_i phrase_row_budget[i]
```

for individual-DPA-style qN, or:

```text
phrase_independent_val_qN = sum_i phrase_row_budget[i]
```

for independent composition.

---

## Interpretation

Use this language in the report:

```text
The naive DPA matrix baseline computes token-level certificates independently and aggregates a prompt by its weakest token. This mirrors the standard per-sample DPA evaluation style, but it does not model one shared adversarial allocation across multiple prompts or token positions.

The phrase-DPA baseline treats each generated sequence as one atomic label. This avoids explicit token-column modelling but suffers from vote diffusion over the exponentially large phrase space.

The proposed shared row-column MILP explicitly enforces one poisoned-shard set across the prompt-token matrix and can therefore evaluate structured stability and sequence-validity objectives more faithfully.
```
