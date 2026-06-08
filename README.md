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
