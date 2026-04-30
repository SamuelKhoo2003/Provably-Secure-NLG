# Toy Row/Column Certificate Experiment Spec

## Goal

Build a small Python + Gurobi experiment that simulates the row/column certificate setup for poisoning robustness in natural language generation.

We want a toy grid with:

- `K`: number of shards / base models
- `N`: number of prompts / rows
- `L`: token horizon / columns
- `T`: number of possible tokens/classes
- `delta`: disagreement/noise rate used to randomly perturb shard votes
- optional random seed for reproducibility

Each cell `(i, j)` corresponds to prompt `i`, token position `j`.

Each shard `k` casts one vote/token for every cell `(i, j)`.

We will generate clean votes, define harmful target tokens, then formulate Gurobi MILPs to compute minimum poison budget `B*` for:

1. row stability
2. column stability
3. row + column stability
4. row validity
5. column validity
6. row + column validity

The main modelling principle is:

> Use one shared poisoning allocation vector `a` across all rows and columns. This is the key difference from independent token-wise or row-wise certificates.

---

## High-level definitions

### Grid

```python
N = number of prompts / rows
L = number of token positions / columns
K = number of shards / base models
T = vocabulary size / number of possible tokens
```

Each clean shard vote is:

```python
votes[k, i, j] in {0, ..., T-1}
```

where:

- `k` indexes shard/base model
- `i` indexes prompt row
- `j` indexes token column

The clean ensemble prediction is:

```python
clean_pred[i, j] = majority_token(votes[:, i, j])
```

The harmful target token is:

```python
target[i, j] in {0, ..., T-1}
```

For validity experiments, choose `target[i, j] != clean_pred[i, j]` where possible.

---

## Toy data generation

Implement:

```python
def generate_toy_votes(K, N, L, T, delta=0.2, seed=0):
    ...
```

Suggested behaviour:

1. For each cell `(i, j)`, sample a clean majority token:

```python
base_token[i, j] ~ Uniform({0, ..., T-1})
```

2. For each shard `k`, with probability `1 - delta`, set:

```python
votes[k, i, j] = base_token[i, j]
```

3. With probability `delta`, sample a different token uniformly:

```python
votes[k, i, j] = random token != base_token[i, j]
```

This gives controllable disagreement. Larger `delta` means weaker vote margins and smaller certificates.

Also compute:

```python
clean_counts[i, j, t] = number of shards voting token t at cell (i, j)
clean_pred[i, j] = argmax_t clean_counts[i, j, t]
runner_up[i, j] = second largest token by vote count
```

---

## Poisoning model

Use a conservative DPA-style worst-case poisoning model.

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

For the toy experiment, assume every poisoned shard can affect every `(i, j)` vote. This is intentionally worst-case and simple.

Later extension: add an influence mask:

```python
influence[k, i, j] ∈ {0,1}
```

so that a corrupted shard only affects cell `(i, j)` if `influence[k, i, j] = 1`.

---

## Vote margins under poisoning

For a cell `(i, j)`, clean winner:

```python
w = clean_pred[i, j]
```

Target token for validity:

```python
h = target[i, j]
```

### Stability flip condition

Stability fails at `(i, j)` if some non-clean token can beat or tie the clean winner after poisoning.

Under the conservative worst-case model, each corrupted shard can both:

1. remove one vote from the clean winner if that shard voted for `w`, and
2. add one vote to an adversarial competitor token.

For a specific competitor `c != w`, define:

```python
winner_loss_ij = sum_k a[k] * 1[votes[k,i,j] == w]
competitor_gain_ijc = sum_k a[k] * 1[votes[k,i,j] != c]
```

Then poisoned counts can be upper/lower bounded as:

```python
poisoned_w_count = clean_counts[i,j,w] - winner_loss_ij
poisoned_c_count = clean_counts[i,j,c] + competitor_gain_ijc
```

A flip to competitor `c` is possible if:

```python
poisoned_c_count >= poisoned_w_count
```

Equivalently:

```python
clean_counts[i,j,c]
+ sum_k a[k] * 1[votes[k,i,j] != c]
>=
clean_counts[i,j,w]
- sum_k a[k] * 1[votes[k,i,j] == w]
```

For a simpler first version, compare only against the clean runner-up instead of every `c != w`.

---

### Validity target condition

Validity succeeds at `(i, j)` if the harmful target token `h = target[i,j]` can be made to win.

For every competing token `c != h`, require:

```python
poisoned_h_count >= poisoned_c_count
```

Use conservative bounds:

```python
poisoned_h_count = clean_counts[i,j,h] + sum_k a[k] * 1[votes[k,i,j] != h]
poisoned_c_count = clean_counts[i,j,c] - sum_k a[k] * 1[votes[k,i,j] == c]
```

So target success requires:

```python
clean_counts[i,j,h]
+ sum_k a[k] * 1[votes[k,i,j] != h]
>=
clean_counts[i,j,c]
- sum_k a[k] * 1[votes[k,i,j] == c]
for all c != h
```

Again, for a simpler first version, compare only against the clean winner or top few competitors.

---

## MILP variables

For all formulations:

```python
a[k] ∈ {0,1}          # shard poisoned or not
```

For cell-level success:

```python
z[i,j] ∈ {0,1}        # cell is successfully attacked
```

For row-level success:

```python
y_row[i] ∈ {0,1}      # row/prompt is successfully attacked
```

For column-level success:

```python
y_col[j] ∈ {0,1}      # column/token-position is successfully attacked
```

Objective for certificate radius:

```python
minimize sum_k a[k]
```

This directly computes the minimum poison budget `B*`.

---

## Big-M implication template

Many constraints are of form:

```text
if z[i,j] = 1, then attack condition at cell (i,j) must hold.
```

Use Big-M:

```python
lhs >= rhs - M * (1 - z[i,j])
```

where `M` should be large enough, e.g.

```python
M = 2*K + 10
```

Since all vote counts are between `0` and `K`, this is safe.

---

# Certificate formulations

## 1. Cell-level stability

A cell is unstable if any competitor can beat/tie the clean winner.

Simplified first version: only use the runner-up competitor `r[i,j]`.

For each `(i,j)`:

```python
w = clean_pred[i,j]
c = runner_up[i,j]

lhs = clean_counts[i,j,c] + sum_k a[k] * int(votes[k,i,j] != c)
rhs = clean_counts[i,j,w] - sum_k a[k] * int(votes[k,i,j] == w)

lhs >= rhs - M * (1 - z[i,j])
```

Meaning:

```text
z[i,j] = 1 => cell (i,j) can be changed.
```

For the exact version, introduce `z_comp[i,j,c]` for each competitor and then:

```python
z[i,j] <= sum_{c != w} z_comp[i,j,c]
z[i,j] >= z_comp[i,j,c] for some selected competitor via objective/constraints
```

But for a first prototype, runner-up-only is fine.

---

## 2. Row stability

Meaning:

> A row/prompt is unstable if at least one token in that row can change.

Linearisation:

```python
y_row[i] <= sum_j z[i,j]
y_row[i] >= z[i,j] for all j
```

To compute minimum budget to destabilise at least one row:

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

To compute minimum budget to destabilise at least one column:

```python
sum_j y_col[j] >= 1
minimize sum_k a[k]
```

Again, this is equivalent to requiring at least one cell flip.

---

## 4. Row + column stability

There are several possible definitions. Implement all three as options.

### Option A: any cell changes

This is the standard stability failure condition:

```python
sum_{i,j} z[i,j] >= 1
```

This will usually equal row stability and column stability.

### Option B: one full row changes

Meaning:

> every token in some row changes.

Use:

```python
sum_j z[i,j] >= L * y_row_full[i]
z[i,j] >= y_row_full[i] for all j
sum_i y_row_full[i] >= 1
```

### Option C: one full column changes

Meaning:

> every prompt changes at some token position.

Use:

```python
sum_i z[i,j] >= N * y_col_full[j]
z[i,j] >= y_col_full[j] for all i
sum_j y_col_full[j] >= 1
```

### Option D: full matrix changes

Meaning:

> all cells change.

Use:

```python
sum_{i,j} z[i,j] >= N * L
```

For the FYP, clearly state which one you mean. The most standard stability certificate is Option A.

---

## 5. Cell-level validity

A cell is validly attacked if the harmful target token wins.

For each `(i,j)`:

```python
h = target[i,j]
```

For every competitor `c != h`:

```python
lhs = clean_counts[i,j,h] + sum_k a[k] * int(votes[k,i,j] != h)
rhs = clean_counts[i,j,c] - sum_k a[k] * int(votes[k,i,j] == c)

lhs >= rhs - M * (1 - z[i,j])
```

Meaning:

```text
z[i,j] = 1 => harmful target token wins at cell (i,j).
```

For speed, compare only against top competitors:

```python
competitors = top_m_tokens_by_clean_count(i,j, m=3)
```

But exact version uses all `T-1` competitors.

---

## 6. Row validity

Meaning:

> The attacker succeeds if the whole harmful sequence appears in at least one prompt row.

This is the important `OR over rows, AND over tokens` formulation.

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

Meaning options depend on interpretation. Implement two versions.

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
Can the attacker force a harmful target token at the same position across every prompt?
```

### Column validity B: force any harmful target token in any column/prompt

This is just cell-level validity:

```python
sum_{i,j} z[i,j] >= 1
```

Use A for a more meaningful column certificate.

---

## 8. Row + column validity

Again, implement multiple options.

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

This corresponds to the strongest threshold `tau = N*L`.

Use this only if the intended attack requires every prompt and every token to be targeted.

---

# Implementation plan

## File structure

Create:

```text
certificate_toy/
  data.py
  milp.py
  experiments.py
  README.md
```

or a single notebook/script first:

```text
toy_certificates.py
```

---

## Functions to implement

### Data generation

```python
def generate_toy_votes(K: int, N: int, L: int, T: int, delta: float, seed: int):
    """Return votes[K,N,L], clean_counts[N,L,T], clean_pred[N,L], target[N,L]."""
```

### Majority/count utilities

```python
def compute_counts(votes, T):
    """Return clean_counts[N,L,T]."""


def majority_predictions(clean_counts):
    """Return clean_pred[N,L]."""


def runner_up_tokens(clean_counts, clean_pred):
    """Return runner_up[N,L]."""


def generate_targets(clean_pred, T, seed=0):
    """Return target[N,L] with target[i,j] != clean_pred[i,j]."""
```

### MILP builders

```python
def build_base_model(votes, clean_counts, clean_pred, target, mode, definition, params):
    """Create Gurobi model and return model plus variables."""
```

Better: implement separate functions:

```python
def solve_row_stability(votes, clean_counts, clean_pred, runner_up):
    ...


def solve_col_stability(votes, clean_counts, clean_pred, runner_up):
    ...


def solve_row_col_stability(votes, clean_counts, clean_pred, runner_up, definition="any_cell"):
    ...


def solve_row_validity(votes, clean_counts, target):
    ...


def solve_col_validity(votes, clean_counts, target, definition="full_column"):
    ...


def solve_row_col_validity(votes, clean_counts, target, q_rows=1, definition="at_least_q_rows"):
    ...
```

Each should return:

```python
{
    "B_star": objective_value,
    "a": selected_poisoned_shards,
    "z": attacked_cells,
    "status": gurobi_status,
}
```

---

# Gurobi pseudocode

## Base setup

```python
import gurobipy as gp
from gurobipy import GRB


def make_model(K):
    m = gp.Model()
    a = m.addVars(K, vtype=GRB.BINARY, name="a")
    m.setObjective(gp.quicksum(a[k] for k in range(K)), GRB.MINIMIZE)
    return m, a
```

---

## Add stability cell constraints

```python
def add_stability_cell_constraints(m, a, votes, clean_counts, clean_pred, runner_up, M=1000):
    K, N, L = votes.shape
    z = m.addVars(N, L, vtype=GRB.BINARY, name="z_stab")

    for i in range(N):
        for j in range(L):
            w = int(clean_pred[i, j])
            c = int(runner_up[i, j])

            lhs = clean_counts[i, j, c] + gp.quicksum(
                a[k] * int(votes[k, i, j] != c)
                for k in range(K)
            )

            rhs = clean_counts[i, j, w] - gp.quicksum(
                a[k] * int(votes[k, i, j] == w)
                for k in range(K)
            )

            m.addConstr(lhs >= rhs - M * (1 - z[i, j]))

    return z
```

---

## Add validity cell constraints

```python
def add_validity_cell_constraints(m, a, votes, clean_counts, target, T, M=1000):
    K, N, L = votes.shape
    z = m.addVars(N, L, vtype=GRB.BINARY, name="z_val")

    for i in range(N):
        for j in range(L):
            h = int(target[i, j])

            target_count = clean_counts[i, j, h] + gp.quicksum(
                a[k] * int(votes[k, i, j] != h)
                for k in range(K)
            )

            for c in range(T):
                if c == h:
                    continue

                competitor_count = clean_counts[i, j, c] - gp.quicksum(
                    a[k] * int(votes[k, i, j] == c)
                    for k in range(K)
                )

                m.addConstr(target_count >= competitor_count - M * (1 - z[i, j]))

    return z
```

---

## Row validity solver

```python
def solve_row_validity(votes, clean_counts, target, T):
    K, N, L = votes.shape
    M = 2 * K + 10

    m, a = make_model(K)
    z = add_validity_cell_constraints(m, a, votes, clean_counts, target, T, M)

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
def solve_row_col_validity(votes, clean_counts, target, T, q_rows=1):
    K, N, L = votes.shape
    M = 2 * K + 10

    m, a = make_model(K)
    z = add_validity_cell_constraints(m, a, votes, clean_counts, target, T, M)

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
def solve_row_col_stability(votes, clean_counts, clean_pred, runner_up, definition="any_cell"):
    K, N, L = votes.shape
    M = 2 * K + 10

    m, a = make_model(K)
    z = add_stability_cell_constraints(m, a, votes, clean_counts, clean_pred, runner_up, M)

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

Use small values:

```python
K = 7
N = 3
L = 4
T = 5
delta = 0.2
```

Print:

```text
clean predictions matrix
harmful target matrix
vote margins per cell
B*_row_stab
B*_col_stab
B*_row_col_stab_any_cell
B*_row_val
B*_col_val
B*_row_col_val_q1
B*_row_col_val_qN
```

Expected behaviour:

- Higher `delta` should lower certificate radii.
- Validity should usually require larger budget than stability.
- Row+column validity with `q_rows=N` should require at least as much budget as `q_rows=1`.

---

## 2. Sweep disagreement

Run:

```python
for delta in [0.0, 0.1, 0.2, 0.3, 0.4]:
    generate data
    solve certificates
    record B*
```

Plot or print table:

```text
delta | row_stab | row_val | row_col_val_q1 | row_col_val_qN
```

---

## 3. Sweep sequence length

Run:

```python
for L in [1, 2, 4, 8]:
    ...
```

Expected:

- Basic stability may not grow with `L`, because any one token changing is enough.
- Validity should generally grow with `L`, because all target tokens in a row must be forced.

---

## 4. Sweep number of prompts

Run:

```python
for N in [1, 2, 4, 8]:
    ...
```

Expected:

- Validity with `q_rows=1` may decrease or stay similar because the attacker has more rows to choose from.
- Validity with `q_rows=N` should increase because all prompts must be compromised.

---

# Notes on exactness and limitations

1. The first stability implementation using only the runner-up is a lower-complexity approximation.
2. Exact stability should allow any competitor token to win.
3. Exact validity should compare the target token against all `T-1` competitors.
4. The toy poisoning model assumes a corrupted shard can change votes independently at every cell. This is conservative but may be too strong.
5. A more realistic extension is to use an influence mask `influence[k,i,j]` or bucket assignment structure.
6. The main intended contribution is not the toy data generation, but the shared allocation vector `a` across rows and columns.

---

# Deliverable

Build a runnable Python script that prints a table like:

```text
K=7, N=3, L=4, T=5, delta=0.2

Certificate                      B*
------------------------------------
row_stability                    1
column_stability                 1
row_col_stability_any_cell        1
row_col_stability_full_row        3
row_validity                     4
column_validity_full_column       5
row_col_validity_q1               4
row_col_validity_qN               7
```

The exact values will depend on the random seed.
---

# Baseline and plotting plan

This section defines the baselines and plots that should be included so the toy experiment can compare the proposed shared row-column MILP against simpler DPA-style references and phrase-level references.

The main comparison should not be phrased as:

```text
our method always gives a larger B*
```

because different baselines can be optimistic or conservative in different ways.

Instead, phrase the claim as:

> The shared row-column MILP gives a tighter joint certificate because it enforces one globally consistent poisoning allocation across rows and columns.

In general:

- larger `B*` means a stronger robustness certificate;
- smaller `B*` means the attack is easier;
- independent-sum baselines can overestimate attack cost because they do not allow shard reuse;
- weakest-cell / min baselines can underestimate sequence-level attack cost because they only require one token to be attacked.

---

## Baseline A: raw per-cell DPA

This is the simplest DPA-style reference.

For each cell `(i, j)`, compute an independent certificate:

```text
B*_cell[i,j]
```

using only the majority margin at that cell.

### Stability

For stability, the raw per-cell DPA certificate asks:

```text
minimum poisoned shards needed to make any non-clean token beat/tie the clean winner
```

For each cell:

```python
w = clean_pred[i,j]
```

Then compute:

```text
B*_stab_cell[i,j]
```

using either:

- runner-up-only approximation; or
- exact all-competitor version.

The raw weakest-cell stability baseline is:

```text
B*_stab_raw_min = min_{i,j} B*_stab_cell[i,j]
```

This corresponds to ordinary any-token stability.

### Validity

For validity, the raw per-cell targeted DPA certificate asks:

```text
minimum poisoned shards needed to make target[i,j] win at cell (i,j)
```

For each cell:

```python
h = target[i,j]
```

compute:

```text
B*_val_cell[i,j]
```

The raw weakest-cell validity baseline is:

```text
B*_val_raw_min = min_{i,j} B*_val_cell[i,j]
```

This is useful diagnostically but should not be treated as a sequence-level validity certificate, because a harmful sequence requires all target tokens in a row to be forced together.

---

## Baseline B: independent composition baseline

This baseline computes per-cell costs independently and combines them with simple min/sum rules.

It is not a true joint optimisation because it does not enforce a single shared poisoning allocation across cells.

### Stability independent composition

For basic any-change stability:

```text
B*_stab_independent_any = min_{i,j} B*_stab_cell[i,j]
```

For full-row stability, where all `L` tokens in some row must change, use:

```text
B*_stab_independent_full_row
=
min_i sum_j B*_stab_cell[i,j]
```

This is conservative because it assumes the attacker must pay separately for each token and cannot reuse the same poisoned shard across columns.

For at least `q` rows each with at least `r` changed tokens:

```text
For each row i:
    row_cost[i,r] = sum of the r smallest B*_stab_cell[i,j] values over j

Then:
    B*_stab_independent(q,r)
    = sum of the q smallest row_cost[i,r] values over i
```

This gives a simple independent baseline for the structured row-column stability certificate.

### Validity independent composition

For row-sequence validity, where the full harmful target sequence must appear in one prompt row:

```text
B*_val_independent_q1
=
min_i sum_j B*_val_cell[i,j]
```

For at least `q` compromised prompt rows:

```text
For each row i:
    row_val_cost[i] = sum_j B*_val_cell[i,j]

Then:
    B*_val_independent(q)
    = sum of the q smallest row_val_cost[i] values over i
```

This is conservative because the same poisoned shard may help force multiple target tokens. The shared row-column MILP may produce a smaller `B*` than this independent-sum baseline, and that does not mean the joint method is weaker. It means the independent baseline was loose.

---

## Baseline C: PHD-style fixed-budget MILP reference

The reference PhD-style solver works mainly in the fixed-budget direction:

```text
given k_poison, maximise number/fraction of flipped predictions
```

and then reports worst-case accuracy.

The toy implementation can compare against this in two equivalent ways:

### Option 1: sweep budget

For each budget:

```python
for b in range(K + 1):
    solve fixed-budget MILP with sum_k a[k] <= b
    record Adv(b)
```

Then compute:

```text
B* = min { b : Adv(b) >= threshold }
```

This is closest to the reference implementation style.

### Option 2: direct minimisation

Solve:

```text
minimise sum_k a[k]
subject to attack-success constraints
```

This is what the toy spec currently does. It directly computes `B*`.

Both are equivalent if solved exactly.

In the README and report, explain that:

> The PhD-style reference is implemented as a fixed-budget worst-case allocation formulation, while the toy implementation often solves the equivalent minimum-budget form directly.

---

## Baseline D: phrase-DPA / phrase-level reference

This baseline is important because the reference paper discusses phrase-level certification, where the whole phrase is treated as one atomic class.

This is not the same as row-column validity.

### Phrase-DPA idea

For each prompt row `i`, each shard emits a whole phrase:

```python
phrase_vote[k, i] = tuple(val_votes[k, i, 0:L])
```

The target harmful phrase is:

```python
target_phrase[i] = tuple(target[i, 0:L])
```

Then count phrase votes:

```python
phrase_counts[i, phrase] = number of shards voting for phrase in row i
```

A phrase-level targeted certificate asks:

```text
minimum poisoned shards needed to make target_phrase[i] win
```

This mirrors the phrase-level method where the length-`L` output is collapsed into one large label space.

### Why phrase-DPA is a useful baseline

Phrase-DPA avoids explicitly modelling token columns. But it suffers from vote diffusion:

```text
number of possible phrases = T^L
```

As `L` grows, shard votes may spread across many distinct phrases, so phrase margins can become weak.

The row-column MILP keeps token-level structure while still requiring one shared poisoning allocation across the whole target sequence.

### Phrase-DPA implementation

Implement:

```python
def compute_phrase_votes(val_votes):
    """
    Return phrase_votes[K,N], where each entry is a tuple of length L.
    """


def solve_phrase_dpa_validity(val_votes, target):
    """
    For each row i, treat each shard's generated length-L phrase as one class.
    Compute the minimum poison budget needed to make target_phrase[i] win.
    Return q1 and qN variants if needed.
    """
```

For `q1`:

```text
B*_phrase_q1 = min_i B*_phrase_row[i]
```

For `qN`, independent phrase composition could use:

```text
B*_phrase_qN_independent = sum_i B*_phrase_row[i]
```

A stronger phrase-level shared-allocation MILP may also be implemented, where one shared `a[k]` is used to make target phrases win in at least `q` rows.

---

# Recommended plots

The experiment should produce a focused set of plots that tell the story clearly.

## Plot 1: validity scaling with sequence length

Purpose:

> Show how sequence-level validity behaves as the token horizon grows.

Run:

```python
for L in [1, 2, 4, 8, 12]:
    ...
```

Plot:

```text
x-axis: L
y-axis: B*
lines:
    raw per-cell DPA weakest target
    independent-sum row validity
    phrase-DPA validity
    shared row-column validity q1
    shared row-column validity qN
```

Expected interpretation:

- raw per-cell DPA is usually too weak for sequence validity;
- independent-sum may grow quickly because it assumes no shard reuse;
- phrase-DPA may degrade as phrase space grows;
- shared row-column validity captures the joint budget needed to force the sequence.

---

## Plot 2: row-column validity curve B*(q)

Purpose:

> Show how much budget is needed to compromise more prompt rows.

For fixed `N` and `L`, compute:

```text
B*_val(q) for q = 1, 2, ..., N
```

Plot:

```text
x-axis: q rows compromised
y-axis: B*
line: shared row-column validity
optional line: independent composition baseline
optional line: phrase-DPA baseline
```

Expected:

```text
B*(q) should generally increase with q.
```

This is one of the clearest demonstrations of shared-budget row-column certification.

---

## Plot 3: structured stability heatmap

Purpose:

> Show that stability becomes meaningful when the failure condition is structured over rows and columns.

Define:

```text
B*_stab(q,r)
=
minimum budget needed to make at least q rows each have at least r changed tokens.
```

Plot heatmap:

```text
x-axis: r changed tokens required per row
y-axis: q affected rows
colour: B*
```

Special cases:

```text
q=1, r=1       basic weakest-cell stability
q=1, r=L       full response instability
q=N, r=1       every prompt has at least one changed token
q=N, r=L       full matrix instability
```

This plot is more useful than basic row stability alone.

---

## Plot 4: column-aggregated stability by token position

Purpose:

> Identify which token positions are easiest to destabilise across prompts.

For each column `j`, compute:

```text
B*_stab_col[j]
```

Possible definitions:

```text
minimum budget to change at least one prompt at column j
minimum budget to change all prompts at column j
minimum budget to change q prompts at column j
```

Plot:

```text
x-axis: token position j
y-axis: B*
```

This is especially relevant for first-token refusal, jailbreak, or tool-call prefix behaviour.

---

## Plot 5: disagreement / margin sweeps

Purpose:

> Verify that certificates behave sensibly as vote margins shrink.

Run:

```python
for delta_stab in [0.0, 0.1, 0.2, 0.3, 0.4]:
    compute stability certificates

for target_bias in [0.0, 0.1, 0.2, 0.3, 0.4]:
    compute validity certificates
```

Plots:

```text
x-axis: delta_stab
y-axis: B*_stab

x-axis: target_bias
y-axis: B*_val
```

Expected:

```text
higher delta_stab -> lower stability B*
higher target_bias -> lower validity B*
```

---

# Recommended benchmark CSV columns

The benchmark CSV should include enough columns to compare all methods without rerunning Gurobi.

Suggested columns:

```text
K
N
L
T
delta_stab
delta_val
target_bias
seed
influence_mode

row_col_stab_q1_r1
row_col_stab_q1_rL
row_col_stab_qN_r1
row_col_stab_qN_rL

row_col_val_q1
row_col_val_qN

raw_dpa_stab_min_cell
raw_dpa_val_min_cell

independent_stab_full_row_q1
independent_stab_qN_rL
independent_val_q1
independent_val_qN

phrase_dpa_val_q1
phrase_dpa_val_qN
```

Optional runtime columns:

```text
runtime_row_col_stab
runtime_row_col_val
runtime_phrase_dpa
gurobi_status
```

---

# Interpretation guidance for report

Use the following language in the final writeup:

```text
Raw per-cell DPA certifies individual token predictions but does not certify structured sequence-level behaviour.

Independent composition combines token certificates but ignores shared poisoning allocation. It can be either too optimistic or too conservative depending on whether min or sum composition is used.

Phrase-DPA treats a whole generated phrase as one atomic class, avoiding explicit token-column modelling, but suffers from vote diffusion over an exponentially large phrase space.

The proposed row-column MILP keeps token-level structure and enforces a single global poisoning allocation across both prompt rows and token columns.
```

Important caveat:

```text
The goal is not to show that the proposed certificate is numerically larger than every baseline. The goal is to show that it is a tighter and more faithful optimisation of the actual joint attack condition.
```

For validity:

```text
The central comparison is between phrase-DPA and row-column validity as L grows.
```

For stability:

```text
The central comparison is not basic row stability, which collapses to weakest-cell stability, but structured stability B*(q,r).
```
