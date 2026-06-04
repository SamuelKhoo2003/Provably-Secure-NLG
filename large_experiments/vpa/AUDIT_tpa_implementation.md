# TPA Implementation Audit

Date: 2026-06-04

Scope: static repository audit only. I inspected source, scripts, docs, and
external reference trees with lightweight read/search commands. I did not run
experiments, real VPA inference, GPU code, training, installs, or deletion.

## Summary Conclusion

The repo preserves the important distinction that standalone TPA is count-based
and that shared-budget MILP-style validity needs shard identities, but the
implementation is incomplete in two ways.

First, standalone token-level TPA exists only in the toy code as
`toy_experiments.baselines.targeted_partition_radius`. It is count-based and is
called by the toy benchmark pipeline through `targeted_validity_token_budgets`,
`aggregate_tpa_sequence_baselines`, and `compute_reference_baselines`. It is not
implemented for VPA JSONL vote artifacts yet.

Second, TPA+MSC / collective TPA is not implemented as a named paper method.
The existing validity MILP is a first-party row-column shard-aware MILP in
`toy_experiments.milp.solve_row_col_validity`. It uses shared poisoned-shard
variables across cells and can certify full harmful rows, but it is not clearly
the paper's TPA+MSC / collective TPA from Definition 4.5 / Theorem B.2.

The VPA integration JSONL schema is good for future standalone TPA and MILP
work: validity rows keep both `vote_counts` plus `target_token_id`, and
`shard_ids` plus `shard_token_ids`. Mock validity export uses target-prefix
decoding. Real VPA validity export is not implemented; the real backend is
currently restricted to stability smoke export.

## Standalone TPA

### Location

- `toy_experiments/baselines.py:104`:
  `targeted_partition_radius(counts, target, tie_wins=True)`
- `toy_experiments/baselines.py:191`:
  `targeted_validity_token_budgets(data)`
- `toy_experiments/baselines.py:176`:
  `aggregate_tpa_sequence_baselines(token_radii)`
- `toy_experiments/baselines.py:10`:
  `compute_reference_baselines(data)` calls the token and sequence functions.
- `toy_experiments/experiments.py:1615`:
  `_compute_benchmark_baselines(...)` calls the TPA baseline path.

Commands/scripts that call it:

```bash
CONFIG=toy_experiments/configs/<config>.yaml ./toy_experiments/scripts/data.sh
```

`toy_experiments/scripts/data.sh:22-33` runs:

```bash
python -m toy_experiments.experiments benchmark --config "$CONFIG"
```

The validity demo script also calls it indirectly:

```bash
./toy_experiments/scripts/validity_demo.sh
```

which invokes `./toy_experiments/scripts/data.sh` at
`toy_experiments/scripts/validity_demo.sh:28`.

### Algorithm 1 Match

`targeted_partition_radius` is standalone/count-based:

- It takes a dense aggregate count vector `counts`.
- It takes a target token index `target`.
- It computes `target_count = counts[target]`.
- It removes the target class to form `competitor_counts`.
- It does not require shard identities.
- It does not use the DPA top-vs-target margin formula.

However, it is not a literal implementation of Algorithm 1:

- It does not explicitly sort/rank all vote counts.
- It does not compute named `Delta_j`, `Phi_s`, or `s_star`.
- Instead, it brute-force searches budgets and checks whether `budget` vote
  moves can raise the target and reduce all competitors enough:
  `required_removals = max(0, competitor_counts - max_competitor_after).sum()`.

This brute-force count search appears equivalent to the targeted reallocation
condition for aggregate counts, but it should be validated against a literal
Algorithm 1 implementation before being called paper-exact.

Tie handling is the main caveat. The default is `tie_wins=True`, so the target
is treated as successful when `target_count >= max(competitor_counts)`. The
paper-style recurrence in the local `external/VPA-main/Theory.md` uses a strict
condition (`Phi_s > v_{c_{s+1}}`) and the formula has a `+1`, suggesting strict
plurality unless a deterministic tie rule is explicitly part of the theorem.
The VPA schema's majority helper breaks ties by first shard order
(`large_experiments/vpa/integration/schemas.py:16-21`), not by target favor.
Therefore the current default tie convention can be optimistic if target ties
do not deterministically win.

Target absent handling is adequate for toy dense vectors: if the target token
has zero votes, `counts[target]` gives `0`. There is no standalone TPA adapter
yet for sparse JSONL `vote_counts` dictionaries, where the target may be absent
as a key.

## Validity Vote Collection

Toy generation marks `val_votes` as harmful-prefix votes:
`toy_experiments/data.py:20-25` and `toy_experiments/data.py:60-65`.
The toy generator is synthetic, so it does not actually run an autoregressive
LLM target-prefix loop; it constructs votes for the harmful-prefix setting.

The large VPA mock exporter implements the intended prefix flow directly:

- Stability: `large_experiments/vpa/integration/export_votes.py:18-60`
  appends `row.majority_token_id`, so it uses clean sequential majority-prefix
  decoding.
- Validity: `large_experiments/vpa/integration/export_votes.py:84-133`
  appends `target_token_id`, so each position is conditioned on prompt plus
  previous target tokens.

`large_experiments/vpa/integration/generate_mock_votes.py` also writes validity
rows with `target_prefix_token_ids`, but its prefix construction uses
`target_token_id - 100 + offset` rather than the exact prior `target_token_id`
sequence. The main `export_votes` mock path is the clearer target-prefix
implementation.

Real VPA validity collection is not implemented. The CLI builds a real backend
only if `args.mode == "stability"`; otherwise `_build_vpa_backend_from_args`
returns "Stage 6 real VPA smoke export supports stability mode only" at
`large_experiments/vpa/integration/export_votes.py:272-278`.

## Multi-Token Standalone TPA

The toy TPA sequence aggregation uses max over per-token TPA radii:

- `toy_experiments/baselines.py:176-188`:
  `row_sequence_radii = token_radii.max(axis=1)`
- `toy_experiments/experiments.py:1813-1840`:
  budget curves label this as `TPA max-token phrase blocker`.

This matches the intended distinction in the request: for a harmful target
sequence, the sequence-level standalone TPA lower bound is the maximum of the
per-token targeted radii when each token is evaluated under target-prefix
decoding.

Potential doc conflict: `external/VPA-main/Theory.md` states a `min_i`
sentence-level validity certificate. That conflicts with the requested
theoretical distinction and with the first-party toy implementation's
`max(axis=1)`. I treat this as an external theory-note inconsistency, not as an
implementation path.

## DPA Validity Use

The repo has a separate plain DPA-like count-margin diagnostic:

- `toy_experiments/baselines.py:133`:
  `plain_dpa_count_margin_radius`
- `toy_experiments/baselines.py:201`:
  `plain_dpa_validity_token_budgets`
- `toy_experiments/experiments.py:1821-1825`:
  plotted as `Plain DPA max-token phrase blocker`.

This is not used as the TPA implementation. TPA has its own function
`targeted_partition_radius`.

There is also a confusingly named shard-aware validity path:

- `toy_experiments/baselines.py:76`:
  `cell_validity_budgets`
- metrics such as `dpa_val_cell_min`, `dpa_val_row_weak_q1`, and
  `raw_dpa_val_min_cell`.

Despite the metric names, `cell_validity_budgets` is not plain DPA. It uses
shard votes, influence masks, and a subset search over shard contributions.
Those labels should be cleaned up later because they can be mistaken for a DPA
validity baseline.

## TPA+MSC / Collective TPA

I did not find a named implementation of:

- `TPA+MSC`
- `collective TPA`
- `multi-sample validity`
- `MSC validity`
- `collective validity MILP`
- Definition 4.5 / Theorem B.2

The first-party candidate is:

- `toy_experiments/milp.py:71`:
  `solve_row_col_validity`
- `toy_experiments/milp.py:164`:
  `_add_validity_cell_constraints`

It uses:

- binary poisoned-shard variables `a[k]` shared across the model
  (`toy_experiments/milp.py:103-110`);
- shard-level votes `votes[k, i, j]`;
- per-cell target-vs-every-competitor constraints
  (`toy_experiments/milp.py:178-190`);
- full-row and q-row objectives
  (`toy_experiments/milp.py:91-100`, `toy_experiments/milp.py:194-199`).

Classification: this is a row-column shard-aware validity MILP inspired by
collective/shared-budget certification. It is not standalone TPA because it
requires shard identities. It is not clearly TPA+MSC / collective TPA because
it does not present the paper's collective TPA formulation, terminology, or
Definition 4.5 / Theorem B.2 mapping. It is a repo-specific MILP over toy
prompt-row/token-position cells.

External reference code has related targeted-certification logic:

- `external/phd_reference/certifiable_learning_stability/gen_validity_certifier.py:256-275`
  contains an Algorithm-1-like targeted radius computation.
- `external/phd_reference/certifiable_learning_stability/solver.py:29-90`
  contains a Gurobi targeted attack batch solver with a shared poisoning vector.

Those are under `external/` and are not wired into first-party toy or VPA
integration commands.

## VPA JSONL Artifact Support

The VPA schema preserves the right fields:

- Aggregate counts: `vote_counts` in both row types
  (`large_experiments/vpa/integration/schemas.py:42-44`,
  `large_experiments/vpa/integration/schemas.py:85-88`).
- Target token: `target_token_id` in validity rows
  (`large_experiments/vpa/integration/schemas.py:79-88`).
- Shard identities: `shard_ids`.
- Shard-level votes: `shard_token_ids`.

The validator checks that `vote_counts` matches `shard_token_ids` and that
validity rows contain `target_id`, `target_prefix_token_ids`, and
`target_token_id` (`large_experiments/vpa/integration/validate_votes.py:63-85`).

This is sufficient storage for:

- standalone TPA from `vote_counts` + `target_token_id`;
- future shard-aware MILPs from `shard_ids` + `shard_token_ids`.

Missing: a first-party function/CLI that reads VPA JSONL validity rows and
computes standalone TPA token and sequence radii.

## External VPA-main Notes

`external/VPA-main/src/certify_vpa.py:87-94` computes a robustness-only radius
`floor((v_t - 1) / 2)` for the majority prediction. This is not standalone TPA
validity.

`external/VPA-main/src/certify_vpa_token.py:68-106` has an Algorithm-1-like
function `compute_targeted_radius_robust`, but it is framed around safe vs
unsafe tool predictions and `majority_prediction`; it is not the clean
standalone TPA API requested here (`vote_counts`, target token `t`, `vt`).
It is also in the preserved external tree and not wired into the first-party
large experiment integration.

## Recommended Fixes

1. Add a first-party standalone TPA module that accepts sparse `vote_counts`
   and `target_token_id`, treats absent targets as `vt = 0`, and implements
   Algorithm 1 explicitly with `Delta`, `Phi`, and `s_star`.
2. Decide and document the tie policy. If the paper requires strict plurality,
   default to strict target success; if deterministic tie-breaking is used, make
   the target's tie outcome explicit and consistent with artifact majority
   computation.
3. Add a VPA JSONL TPA CLI that groups validity rows by
   `(example_id, target_id)`, computes per-row token radii, and reports sequence
   radii as max over target-prefix positions.
4. Rename or document toy metrics beginning with `dpa_val_*`, because
   `cell_validity_budgets` is shard-aware and not plain DPA.
5. Implement real VPA validity export by supplying actual harmful target token
   sequences and using prompt plus previous target tokens for each next-token
   shard vote.
6. Treat TPA+MSC / collective TPA as a separate implementation stage. Reuse the
   JSONL shard fields, but write a paper-mapped formulation with explicit shared
   poisoned-shard variables and a clear Definition 4.5 / Theorem B.2 reference.

## Next Implementation Stages

1. Standalone TPA exact function and unit-level examples.
2. VPA JSONL standalone TPA evaluator.
3. Real VPA target-prefix validity export.
4. Adapter from VPA JSONL artifacts into toy-style tensors for exploratory
   row-column MILPs.
5. Paper-mapped TPA+MSC / collective TPA MILP, separate from the existing
   row-column MILP.
6. Documentation cleanup: tie policy, max-over-token sequence aggregation, and
   naming for DPA vs shard-aware validity diagnostics.

## Status Table

| Component | Implemented? | File/function | Matches paper? | Needs changes? | Notes |
|---|---:|---|---|---:|---|
| DPA stability baseline | Yes | `toy_experiments/baselines.py:54` `cell_stability_budgets`; `toy_experiments/baselines.py:211` `phd_margin_stability_budgets` | Mostly for toy baselines | Maybe | Stability path is separate from TPA audit. |
| DPA validity diagnostic | Yes | `toy_experiments/baselines.py:133` `plain_dpa_count_margin_radius`; `toy_experiments/baselines.py:201` `plain_dpa_validity_token_budgets` | Diagnostic only | Maybe | Correctly separated from TPA, but labels around `dpa_val_*` are confusing. |
| standalone TPA token-level | Partial | `toy_experiments/baselines.py:104` `targeted_partition_radius` | Partly | Yes | Count-based and shard-free, but not literal Algorithm 1 and tie policy needs review. |
| standalone TPA multi-token sequence-level | Yes in toy | `toy_experiments/baselines.py:176` `aggregate_tpa_sequence_baselines` | Yes relative to requested max-over-token rule | No/Maybe | Uses max over token radii. |
| target-prefix validity vote collection | Mock yes, real no | `large_experiments/vpa/integration/export_votes.py:84` `export_validity_votes`; `toy_experiments/data.py:60` synthetic generator | Mock matches; toy synthetic only | Yes | Real VPA validity export is blocked to stability-only. |
| TPA+MSC / collective TPA | No | None found | No | Yes | No named paper-mapped implementation. |
| my row-column validity MILP | Yes | `toy_experiments/milp.py:71` `solve_row_col_validity` | Separate method | Maybe | Shard-aware shared-budget MILP over toy row/column cells, not standalone TPA. |
| VPA real vote export support for TPA | Schema yes, real collection no | `schemas.py:75` `ValidityVoteRow`; `export_votes.py:272-278` real mode restriction | Partial | Yes | JSONL has needed fields; real validity export and TPA evaluator missing. |
| VPA real vote export support for MILP | Schema yes, real collection no | `schemas.py:85-88` shard fields; `vpa_backend.py:110` shard loop | Partial | Yes | Shard IDs/votes are preserved in schema, but real validity rows are not produced. |

