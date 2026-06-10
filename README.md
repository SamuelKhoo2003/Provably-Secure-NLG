# Provably-Secure-NLG

EIE Final Year Project 2026.

Stability is always evaluated against all competing tokens. The old
runner-up-only diagnostic has been removed because report-facing stability is
an untargeted any-token-change property.

Synthetic benchmark sweeps use coupled master instances. For each fixed
distribution-parameter tuple, one maximum-size vote structure is generated and
smaller `K`, `N`, `L`, and `T` points are derived from it. This makes scaling
comparisons nested and avoids confounding each point with a fresh random draw.
Coupling improves comparability but does not imply monotonicity for every
objective or parameter; see
[`toy_experiments/toy_experiment_README.md`](toy_experiments/toy_experiment_README.md#coupled-synthetic-sweeps).

TPA terminology in this repository is deliberately narrow. The toy baseline is
the count-based **TPA max-token phrase baseline**, and the large baseline is
**aggregate TPA final-tool validity** over whole tool-call labels.
Neither uses an MILP, shard identities, or a shared poisoned-shard allocation.
Collective TPA+MSC is not implemented.

The main large-experiment external baselines use final tool-call vote vectors:
**DPA final-tool stability** and **aggregate TPA final-tool validity**. The
proposed joint MILPs use the shard-aware prompt-token grid. Token-grid DPA
curves are optional diagnostics rather than main full-scale baselines.

## Tests

Run the maintained solver-free and utility tests from the repository root:

```bash
.venv/bin/python -m unittest discover -v
```
