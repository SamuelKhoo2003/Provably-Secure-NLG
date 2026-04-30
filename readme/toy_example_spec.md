# Toy Row/Column Certificate Experiment Spec

## Goal

Build a small Python + Gurobi experiment that simulates row/column poisoning certificates for natural language generation (NLG).

The experiment should generate a toy grid with:

- `K`: number of shards / base models
- `N`: number of prompts / rows
- `L`: token horizon / columns
- `T`: number of possible tokens/classes
- `delta_stab`: disagreement/noise rate for stability votes
- `delta_val`: disagreement/noise rate for validity votes
- `target_bias`: how much natural support harmful target tokens receive under target-prefix decoding
- optional random seed for reproducibility

Each cell `(i, j)` corresponds to:

```text
prompt i, token position j
```

Each shard/base model `k` casts one vote/token for every cell `(i, j)`.

The key modelling principle is:

> Use one shared poisoning allocation vector `a` across all rows and columns.

This is the main difference from independent token-wise or row-wise certificates. The same poisoned shards must explain the whole attack.

---

# Conceptual distinction: stability vs validity

The toy experiment should explicitly separate the vote tensors used for stability and validity.

## Stability

Stability asks:

> Can the attacker make the clean generation change?

For stability, use votes under the **clean autoregressive prefix**.

At token position `j`, each shard is assumed to vote after seeing:

```text
prompt + clean tokens 1...(j-1)
```

Use:

```python
stab_votes[k, i, j]
```

where:

```text
stab_votes[k, i, j] = token predicted by shard k
under the clean prefix for prompt i, token position j.
```

The clean ensemble prediction is:

```python
clean_pred[i, j] = majority_token(stab_votes[:, i, j])
```

Stability fails at cell `(i, j)` if the attacker can make the aggregate winner become **any token other than** `clean_pred[i, j]`.

So stability is untargeted:

```text
winner_after_attack(i, j) != clean_pred[i, j]
```

---

## Validity

Validity asks:

> Can the attacker force a specific harmful target token/phrase/sequence?

For validity, use votes under the **target/harmful autoregressive prefix**.

At token position `j`, each shard is assumed to vote after seeing:

```text
prompt + harmful target tokens 1...(j-1)
```

Use:

```python
val_votes[k, i, j]
```

where:

```text
val_votes[k, i, j] = token predicted by shard k
under the harmful prefix for prompt i, token position j.
```

The harmful target token is:

```python
target[i, j]
```

Validity succeeds at cell `(i, j)` if the attacker can make:

```text
winner_after_attack(i, j) == target[i, j]
```

For row-level validity, the attacker succeeds if the whole harmful sequence appears in at least one prompt row:

```text
exists i such that for all j:
    winner_after_attack(i, j) == target[i, j]
```

This is the key `OR over rows, AND over tokens` structure.

---

# High-level objects

```python
K = number of shards / base models
N = number of prompts / rows
L = number of token positions / columns
T = vocabulary size / number of possible tokens
```

Vote tensors:

```python
stab_votes[k, i, j] in {0, ..., T-1}
val_votes[k, i, j] in {0, ..., T-1}
```

Counts:

```python
stab_counts[i, j, t] = number of shards voting token t in stab_votes[:, i, j]
val_counts[i, j, t]  = number of shards voting token t in val_votes[:, i, j]
```

Clean prediction:

```python
clean_pred[i, j] = argmax_t stab_counts[i, j, t]
```

Harmful target:

```python
target[i, j] in {0, ..., T-1}
```

For validity, choose:

```python
target[i, j] != clean_pred[i, j]
```

where possible.

---

# Toy data generation

Implement:

```python
def generate_toy_votes(
    K: int,
    N: int,
    L: int,
    T: int,
    delta_stab: float = 0.2,
    delta_val: float = 0.2,
    target_bias: float = 0.2,
    seed: int = 0,
):
    """
    Returns:
        stab_votes:   np.ndarray[K,N,L]
        val_votes:    np.ndarray[K,N,L]
        stab_counts:  np.ndarray[N,L,T]
        val_counts:   np.ndarray[N,L,T]
        clean_pred:   np.ndarray[N,L]
        runner_up:    np.ndarray[N,L]
        target:       np.ndarray[N,L]
    """
```

---

## Generate stability votes

For each cell `(i, j)`:

1. Sample a clean majority token:

```python
base_token[i, j] ~ Uniform({0, ..., T-1})
```

2. For each shard `k`, with probability `1 - delta_stab`, set:

```python
stab_votes[k, i, j] = base_token[i, j]
```

3. With probability `delta_stab`, sample a different token uniformly:

```python
stab_votes[k, i, j] = random token != base_token[i, j]
```

This gives controllable disagreement.

Larger `delta_stab` means weaker clean vote margins and smaller stability certificates.

---

## Generate harmful targets

After computing `clean_pred`, choose:

```python
target[i, j] != clean_pred[i, j]
```

A simple implementation:

```python
target[i, j] = random token from {0, ..., T-1} \ {clean_pred[i, j]}
```

Optional extension:

- use the same target sequence for all prompts;
- use structured targets such as `[DROP, TABLE]`;
- use one harmful token repeated across rows/columns.

---

## Generate validity votes

Validity votes should simulate shard predictions under the harmful prefix.

For each cell `(i, j)`, define:

```python
h = target[i, j]
```

Then generate `val_votes[:, i, j]` so that the harmful target has some controllable support.

Suggested method:

1. Sample a non-target majority token:

```python
val_base[i, j] ~ Uniform({0, ..., T-1} \ {h})
```

2. For each shard `k`:

With probability `target_bias`, set:

```python
val_votes[k, i, j] = h
```

With probability `1 - target_bias`, follow the non-target majority with noise:

```python
with probability 1 - delta_val:
    val_votes[k, i, j] = val_base[i, j]

with probability delta_val:
    val_votes[k, i, j] = random token not equal to val_base[i, j]
```

Interpretation:

- larger `target_bias` means the harmful target is already closer to winning;
- smaller `target_bias` means the target is far from winning;
- larger `delta_val` means more disagreement under the harmful prefix.

This is a toy abstraction of worst-case targeted autoregressive generation:

> for token `j`, validity is evaluated under the assumption that previous harmful target tokens `target[i, 0:j]` have already been generated.

---

# Poisoning model

Use a conservative DPA-style poisoning model.

Binary variable:

```python
a[k] ∈ {0,1}
```

where:

```text
a[k] = 1 means shard k is corrupted/poisoned.
```

Budget:

```python
B = sum_k a[k]
```

Worst-case assumption:

> If shard `k` is poisoned, the attacker can change that shard's vote arbitrarily for any cell that shard influences.

For the first toy experiment, assume every poisoned shard can affect every `(i, j)` vote.

Later extension: add an influence mask:

```python
influence[k, i, j] ∈ {0,1}
```

Then a corrupted shard only affects cell `(i, j)` if:

```python
influence[k, i, j] = 1
```

This makes row/column coupling more interesting because not every poisoned shard helps every cell.

---

# Influence mask extension

Optional but strongly recommended after the first prototype.

Generate:

```python
influence[k, i, j] ∈ {0,1}
```

Possible patterns:

## Dense worst-case

```python
influence[k, i, j] = 1 for all k, i, j
```

This is simplest and most conservative.

## Row-local

Shard `k` influences only some prompts:

```python
influence[k, i, j] = row_mask[k, i]
```

## Column-local

Shard `k` influences only some token positions:

```python
influence[k, i, j] = col_mask[k, j]
```

## Block-structured

Shard `k` influences a rectangular subset of rows and columns.

This is useful for testing whether row-column certificates improve over independent row/column relaxations.

When using `influence`, replace every term:

```python
a[k] * condition
```

with:

```python
a[k] * influence[k, i, j] * condition
```

---

# Vote margins under poisoning

## Stability flip condition

Use `stab_votes` and `stab_counts`.

For a cell `(i, j)`, define:

```python
w = clean_pred[i, j]
```

A stability attack succeeds if some competitor `c != w` can beat or tie the clean winner after poisoning.

Under the conservative worst-case model, a corrupted shard can:

1. remove one vote from the clean winner if that shard voted for `w`;
2. add one vote to an adversarial competitor token `c` if that shard did not already vote for `c`.

For a specific competitor `c != w`:

```python
winner_loss_ij = sum_k a[k] * influence[k,i,j] * 1[stab_votes[k,i,j] == w]
competitor_gain_ijc = sum_k a[k] * influence[k,i,j] * 1[stab_votes[k,i,j] != c]
```

Poisoned count bounds:

```python
poisoned_w_count = stab_counts[i,j,w] - winner_loss_ij
poisoned_c_count = stab_counts[i,j,c] + competitor_gain_ijc
```

Flip condition:

```python
poisoned_c_count >= poisoned_w_count
```

Equivalently:

```python
stab_counts[i,j,c]
+ sum_k a[k] * influence[k,i,j] * 1[stab_votes[k,i,j] != c]
>=
stab_counts[i,j,w]
- sum_k a[k] * influence[k,i,j] * 1[stab_votes[k,i,j] == w]
```

### Approximate stability

For a first prototype, compare only against the clean runner-up:

```python
c = runner_up[i, j]
```

This is faster but not exact.

### Exact stability

For exact stability, introduce competitor-level variables:

```python
z_comp[i, j, c] ∈ {0,1}
```

for every competitor `c != w`.

Then:

```text
z_comp[i,j,c] = 1 means competitor c can beat/tie the clean winner at cell (i,j).
```

Cell instability is:

```text
z[i,j] = OR over c != w of z_comp[i,j,c].
```

---

## Validity target condition

Use `val_votes` and `val_counts`.

For a cell `(i, j)`, define:

```python
h = target[i, j]
```

Validity succeeds at `(i, j)` if the harmful target token `h` can be made to win.

For every competitor `c != h`, require:

```python
poisoned_h_count >= poisoned_c_count
```

Conservative bounds:

```python
target_gain_ij = sum_k a[k] * influence[k,i,j] * 1[val_votes[k,i,j] != h]
competitor_loss_ijc = sum_k a[k] * influence[k,i,j] * 1[val_votes[k,i,j] == c]
```

Then:

```python
poisoned_h_count = val_counts[i,j,h] + target_gain_ij
poisoned_c_count = val_counts[i,j,c] - competitor_loss_ijc
```

Target success requires:

```python
val_counts[i,j,h]
+ sum_k a[k] * influence[k,i,j] * 1[val_votes[k,i,j] != h]
>=
val_counts[i,j,c]
- sum_k a[k] * influence[k,i,j] * 1[val_votes[k,i,j] == c]
for all c != h
```

For speed, compare only against the top few competitors.

For exact validity, compare against all `T-1` competitors.

---

# MILP variables

For all formulations:

```python
a[k] ∈ {0,1}          # shard poisoned or not
```

For cell-level success:

```python
z[i,j] ∈ {0,1}        # cell is successfully attacked
```

For exact stability:

```python
z_comp[i,j,c] ∈ {0,1}
```

For row-level success:

```python
y_row[i] ∈ {0,1}
```

For column-level success:

```python
y_col[j] ∈ {0,1}
```

Objective for certificate radius:

```python
minimize sum_k a[k]
```

This directly computes the minimum poison budget:

```text
B*
```

Alternative literature-style evaluation:

```text
fix B, maximize damage, sweep B, then find first B where attack succeeds.
```

But for the toy version, direct minimization is simpler.

---

# Big-M implication template

Many constraints are of the form:

```text
if z[i,j] = 1, then attack condition at cell (i,j) must hold.
```

Use Big-M:

```python
lhs >= rhs - M * (1 - z[i,j])
```

where:

```python
M = 2 * K + 10
```

is safe because all vote counts are between `0` and `K`.

---

# Certificate formulations

## 1. Cell-level stability

Cell `(i, j)` is unstable if any competitor can beat/tie the clean winner.

### Approximate runner-up version

Use only:

```python
c = runner_up[i, j]
```

For each `(i, j)`:

```python
w = clean_pred[i,j]
c = runner_up[i,j]

lhs = stab_counts[i,j,c] + sum_k a[k] * influence[k,i,j] * int(stab_votes[k,i,j] != c)
rhs = stab_counts[i,j,w] - sum_k a[k] * influence[k,i,j] * int(stab_votes[k,i,j] == w)

lhs >= rhs - M * (1 - z[i,j])
```

Meaning:

```text
z[i,j] = 1 => cell (i,j) can be changed.
```

### Exact all-competitor version

For every `c != w`:

```python
lhs_c = stab_counts[i,j,c] + sum_k a[k] * influence[k,i,j] * int(stab_votes[k,i,j] != c)
rhs_c = stab_counts[i,j,w] - sum_k a[k] * influence[k,i,j] * int(stab_votes[k,i,j] == w)

lhs_c >= rhs_c - M * (1 - z_comp[i,j,c])
```

Link:

```python
z[i,j] <= sum_c z_comp[i,j,c]
z[i,j] >= z_comp[i,j,c] for all c != w
```

Since the objective minimizes `sum a[k]` and the model requires some `z`, these constraints are enough to make `z[i,j]` act like an OR over competitors.

---

## 2. Row stability

Meaning:

> A row/prompt is unstable if at least one token in that row can change.

Linearisation:

```python
y_row[i] <= sum_j z[i,j]
y_row[i] >= z[i,j] for all j
```

Minimum budget to destabilise at least one row:

```python
sum_i y_row[i] >= 1
minimize sum_k a[k]
```

Equivalent basic form:

```python
sum_{i,j} z[i,j] >= 1
```

because one changed token already destabilises one row.

---

## 3. Column stability

Meaning:

> A column/token-position is unstable if at least one prompt changes at that token position.

Linearisation:

```python
y_col[j] <= sum_i z[i,j]
y_col[j] >= z[i,j] for all i
```

Minimum budget to destabilise at least one column:

```python
sum_j y_col[j] >= 1
minimize sum_k a[k]
```

Again, this is equivalent to requiring at least one cell flip.

---

## 4. Row + column stability

Implement multiple definitions because “row + column stability” can mean different things.

### Option A: any cell changes

Standard stability failure condition:

```python
sum_{i,j} z[i,j] >= 1
```

This usually equals row stability and column stability.

### Option B: one full row changes

Every token in some row changes:

```python
sum_j z[i,j] >= L * y_row_full[i]
z[i,j] >= y_row_full[i] for all j
sum_i y_row_full[i] >= 1
```

### Option C: one full column changes

Every prompt changes at some token position:

```python
sum_i z[i,j] >= N * y_col_full[j]
z[i,j] >= y_col_full[j] for all i
sum_j y_col_full[j] >= 1
```

### Option D: full matrix changes

All cells change:

```python
sum_{i,j} z[i,j] >= N * L
```

For the FYP, clearly state which definition is used. The standard stability certificate is Option A.

---

## 5. Cell-level validity

Cell `(i, j)` is validly attacked if the harmful target token wins.

Use `val_votes`, `val_counts`, and `target`.

For each `(i,j)`:

```python
h = target[i,j]
```

For every competitor `c != h`:

```python
lhs = val_counts[i,j,h] + sum_k a[k] * influence[k,i,j] * int(val_votes[k,i,j] != h)
rhs = val_counts[i,j,c] - sum_k a[k] * influence[k,i,j] * int(val_votes[k,i,j] == c)

lhs >= rhs - M * (1 - z[i,j])
```

Meaning:

```text
z[i,j] = 1 => harmful target token wins at cell (i,j).
```

For exact validity, include all `T-1` competitors.

For speed, compare only against top competitors:

```python
competitors = top_m_tokens_by_val_count(i, j, m=3)
```

---

## 6. Row validity

Meaning:

> The attacker succeeds if the whole harmful sequence appears in at least one prompt row.

This is the important:

```text
OR over rows, AND over tokens
```

formulation.

For each row `i`:

```python
sum_j z[i,j] >= L * y_row[i]
z[i,j] >= y_row[i] for all j
```

Then require:

```python
sum_i y_row[i] >= 1
```

Objective:

```python
minimize sum_k a[k]
```

This gives:

```text
minimum poison budget needed to force the full harmful phrase in at least one prompt.
```

---

## 7. Column validity

Implement two versions.

### Column validity A: force one harmful target token position across all prompts

For each column `j`:

```python
sum_i z[i,j] >= N * y_col[j]
z[i,j] >= y_col[j] for all i
```

Require:

```python
sum_j y_col[j] >= 1
```

This asks:

```text
Can the attacker force the harmful target token at the same position across every prompt?
```

### Column validity B: force any harmful target token in any prompt/column

This is just cell-level validity:

```python
sum_{i,j} z[i,j] >= 1
```

Use A for a more meaningful column certificate.

---

## 8. Row + column validity

Implement multiple options.

### Option A: force full harmful sequence in at least one row

This is usually the best targeted NLG validity definition:

```python
sum_i y_row[i] >= 1
```

with:

```python
y_row[i] = 1 iff all L target tokens in row i succeed.
```

### Option B: force full harmful sequence in at least q rows

Add parameter:

```python
q_rows
```

Require:

```python
sum_i y_row[i] >= q_rows
```

This gives a robustness curve:

```text
B*(q) = minimum budget to compromise q prompts.
```

### Option C: force entire N x L target matrix

Require:

```python
sum_{i,j} z[i,j] >= N * L
```

This corresponds to the strongest threshold:

```text
tau = N * L
```

Use this only if the intended attack requires every prompt and every token to be targeted.

---

# Implementation plan

Create either:

```text
certificate_toy/
  data.py
  milp.py
  experiments.py
  README.md
```

or a single script first:

```text
toy_certificates.py
```

---

# Functions to implement

## Data generation

```python
def generate_toy_votes(
    K: int,
    N: int,
    L: int,
    T: int,
    delta_stab: float = 0.2,
    delta_val: float = 0.2,
    target_bias: float = 0.2,
    seed: int = 0,
):
    """
    Return:
        stab_votes[K,N,L]
        val_votes[K,N,L]
        stab_counts[N,L,T]
        val_counts[N,L,T]
        clean_pred[N,L]
        runner_up[N,L]
        target[N,L]
        influence[K,N,L]
    """
```

## Utility functions

```python
def compute_counts(votes, T):
    """Return counts[N,L,T]."""


def majority_predictions(counts):
    """Return pred[N,L]."""


def runner_up_tokens(counts, clean_pred):
    """Return runner_up[N,L]."""


def generate_targets(clean_pred, T, seed=0):
    """Return target[N,L] with target[i,j] != clean_pred[i,j]."""


def generate_influence(K, N, L, mode="dense", seed=0):
    """Return influence[K,N,L]."""
```

## MILP builders

```python
def make_model(K):
    """Create Gurobi model and binary poison variables a[k]."""


def add_stability_cell_constraints(
    m, a, stab_votes, stab_counts, clean_pred, runner_up, influence,
    exact=False, M=None,
):
    """Return z[N,L] for stability attack success."""


def add_validity_cell_constraints(
    m, a, val_votes, val_counts, target, influence, T, M=None,
):
    """Return z[N,L] for target-token validity success."""
```

## Solvers

```python
def solve_row_stability(...):
    ...


def solve_col_stability(...):
    ...


def solve_row_col_stability(..., definition="any_cell"):
    ...


def solve_row_validity(...):
    ...


def solve_col_validity(..., definition="full_column"):
    ...


def solve_row_col_validity(..., q_rows=1, definition="at_least_q_rows"):
    ...
```

Each solver should return:

```python
{
    "B_star": objective_value,
    "a": selected_poisoned_shards,
    "z": attacked_cells,
    "y_row": row_indicators if used,
    "y_col": column_indicators if used,
    "status": gurobi_status,
}
```

---

# Gurobi pseudocode

## Base setup

```python
import gurobipy as gp
from gurobipy import GRB


def make_model(K, verbose=False):
    m = gp.Model()
    if not verbose:
        m.Params.LogToConsole = 0

    a = m.addVars(K, vtype=GRB.BINARY, name="a")
    m.setObjective(gp.quicksum(a[k] for k in range(K)), GRB.MINIMIZE)
    return m, a
```

---

## Add approximate stability cell constraints

```python
def add_stability_cell_constraints(
    m, a, stab_votes, stab_counts, clean_pred, runner_up, influence, M=None
):
    K, N, L = stab_votes.shape
    if M is None:
        M = 2 * K + 10

    z = m.addVars(N, L, vtype=GRB.BINARY, name="z_stab")

    for i in range(N):
        for j in range(L):
            w = int(clean_pred[i, j])
            c = int(runner_up[i, j])

            lhs = stab_counts[i, j, c] + gp.quicksum(
                a[k] * int(influence[k, i, j]) * int(stab_votes[k, i, j] != c)
                for k in range(K)
            )

            rhs = stab_counts[i, j, w] - gp.quicksum(
                a[k] * int(influence[k, i, j]) * int(stab_votes[k, i, j] == w)
                for k in range(K)
            )

            m.addConstr(lhs >= rhs - M * (1 - z[i, j]))

    return z
```

---

## Add exact validity cell constraints

```python
def add_validity_cell_constraints(
    m, a, val_votes, val_counts, target, influence, T, M=None
):
    K, N, L = val_votes.shape
    if M is None:
        M = 2 * K + 10

    z = m.addVars(N, L, vtype=GRB.BINARY, name="z_val")

    for i in range(N):
        for j in range(L):
            h = int(target[i, j])

            target_count = val_counts[i, j, h] + gp.quicksum(
                a[k] * int(influence[k, i, j]) * int(val_votes[k, i, j] != h)
                for k in range(K)
            )

            for c in range(T):
                if c == h:
                    continue

                competitor_count = val_counts[i, j, c] - gp.quicksum(
                    a[k] * int(influence[k, i, j]) * int(val_votes[k, i, j] == c)
                    for k in range(K)
                )

                m.addConstr(
                    target_count >= competitor_count - M * (1 - z[i, j])
                )

    return z
```

---

## Row validity solver

```python
def solve_row_validity(val_votes, val_counts, target, influence, T):
    K, N, L = val_votes.shape

    m, a = make_model(K)
    z = add_validity_cell_constraints(
        m, a, val_votes, val_counts, target, influence, T
    )

    y = m.addVars(N, vtype=GRB.BINARY, name="y_row")

    for i in range(N):
        m.addConstr(gp.quicksum(z[i, j] for j in range(L)) >= L * y[i])
        for j in range(L):
            m.addConstr(z[i, j] >= y[i])

    m.addConstr(gp.quicksum(y[i] for i in range(N)) >= 1)

    m.optimize()
    return extract_solution(m, a, z, y_row=y)
```

---

## Row + column validity solver with q compromised rows

```python
def solve_row_col_validity(
    val_votes, val_counts, target, influence, T, q_rows=1
):
    K, N, L = val_votes.shape

    m, a = make_model(K)
    z = add_validity_cell_constraints(
        m, a, val_votes, val_counts, target, influence, T
    )

    y = m.addVars(N, vtype=GRB.BINARY, name="y_row")

    for i in range(N):
        m.addConstr(gp.quicksum(z[i, j] for j in range(L)) >= L * y[i])
        for j in range(L):
            m.addConstr(z[i, j] >= y[i])

    m.addConstr(gp.quicksum(y[i] for i in range(N)) >= q_rows)

    m.optimize()
    return extract_solution(m, a, z, y_row=y)
```

---

## Row + column stability solver

```python
def solve_row_col_stability(
    stab_votes,
    stab_counts,
    clean_pred,
    runner_up,
    influence,
    definition="any_cell",
):
    K, N, L = stab_votes.shape

    m, a = make_model(K)
    z = add_stability_cell_constraints(
        m, a, stab_votes, stab_counts, clean_pred, runner_up, influence
    )

    if definition == "any_cell":
        m.addConstr(gp.quicksum(z[i, j] for i in range(N) for j in range(L)) >= 1)

    elif definition == "full_row":
        y = m.addVars(N, vtype=GRB.BINARY, name="y_full_row")
        for i in range(N):
            m.addConstr(gp.quicksum(z[i, j] for j in range(L)) >= L * y[i])
            for j in range(L):
                m.addConstr(z[i, j] >= y[i])
        m.addConstr(gp.quicksum(y[i] for i in range(N)) >= 1)

    elif definition == "full_col":
        y = m.addVars(L, vtype=GRB.BINARY, name="y_full_col")
        for j in range(L):
            m.addConstr(gp.quicksum(z[i, j] for i in range(N)) >= N * y[j])
            for i in range(N):
                m.addConstr(z[i, j] >= y[j])
        m.addConstr(gp.quicksum(y[j] for j in range(L)) >= 1)

    elif definition == "full_matrix":
        m.addConstr(gp.quicksum(z[i, j] for i in range(N) for j in range(L)) >= N * L)

    else:
        raise ValueError(f"Unknown definition: {definition}")

    m.optimize()
    return extract_solution(m, a, z)
```

---

# Experiments to run

## 1. Sanity check

Use:

```python
K = 7
N = 3
L = 4
T = 5
delta_stab = 0.2
delta_val = 0.2
target_bias = 0.2
seed = 0
```

Print:

```text
clean predictions matrix
harmful target matrix
stability vote margins per cell
validity target margins per cell
B*_row_stab
B*_col_stab
B*_row_col_stab_any_cell
B*_row_col_stab_full_row
B*_row_val
B*_col_val_full_column
B*_row_col_val_q1
B*_row_col_val_qN
```

Expected:

- validity usually requires larger budget than basic stability;
- row-column validity with `q_rows=N` should require at least as much budget as `q_rows=1`;
- basic row/column stability often equals any-cell stability.

---

## 2. Sweep stability disagreement

```python
for delta_stab in [0.0, 0.1, 0.2, 0.3, 0.4]:
    ...
```

Expected:

```text
higher delta_stab -> weaker clean vote margins -> smaller stability B*
```

---

## 3. Sweep validity target bias

```python
for target_bias in [0.0, 0.1, 0.2, 0.3, 0.4]:
    ...
```

Expected:

```text
higher target_bias -> harmful target closer to winning -> smaller validity B*
```

---

## 4. Sweep sequence length

```python
for L in [1, 2, 4, 8]:
    ...
```

Expected:

- basic stability may not grow with `L`, because any one token changing is enough;
- row validity should generally grow with `L`, because all target tokens in a row must be forced;
- with more columns, q1 row validity may sometimes decrease if more weak rows/tokens appear, so average over seeds.

---

## 5. Sweep number of prompts

```python
for N in [1, 2, 4, 8]:
    ...
```

Expected:

- row validity with `q_rows=1` may decrease or stay similar because the attacker has more rows to choose from;
- row validity with `q_rows=N` should increase because all prompts must be compromised.

---

## 6. Compare dense vs structured influence

Run:

```python
for influence_mode in ["dense", "row_local", "col_local", "block"]:
    ...
```

Expected:

- dense influence makes attacks easier because one poisoned shard can help many cells;
- structured influence makes row-column coupling more meaningful;
- joint validity should differ more from independent token-wise relaxations under structured influence.

---

# Notes on exactness and limitations

1. Stability uses clean-prefix votes.
2. Validity uses target-prefix votes.
3. This is a toy abstraction; no real autoregressive LM is run.
4. The first stability implementation using only the runner-up is an approximation.
5. Exact stability should allow any competitor token to win.
6. Exact validity should compare the target token against all `T-1` competitors.
7. The dense poisoning model assumes a corrupted shard can affect every cell. This is conservative but may hide some row-column structure.
8. A structured influence mask is useful for demonstrating where row-column joint optimisation tightens independent relaxations.
9. The main intended contribution is the shared allocation vector `a` across rows and columns, especially for targeted validity.

---

# Deliverable

Build a runnable Python script that prints a table like:

```text
K=7, N=3, L=4, T=5
delta_stab=0.2, delta_val=0.2, target_bias=0.2

Certificate                         B*
---------------------------------------
row_stability                       1
column_stability                    1
row_col_stability_any_cell           1
row_col_stability_full_row           3
row_validity                        4
column_validity_full_column          5
row_col_validity_q1                  4
row_col_validity_qN                  7
```

The exact values depend on the random seed and generated vote structure.
