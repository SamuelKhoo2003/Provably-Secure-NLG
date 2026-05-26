# Provably-Secure-NLG

EIE Final Year Project 2026.

## Horizon Plots

`./scripts/data.sh` writes `benchmark_horizons.csv` alongside the benchmark
results. Horizon curves answer a prefix question at fixed poisoning budget `B`:
how many initial token positions remain certified? This differs from certified
fraction plots, which count how many rows or regions are fully certified.

Stability horizons use token-level DPA stability radii for the clean prefix.
Validity horizons use TPA-style targeted token radii for harmful target prefixes,
aggregating each prefix by its hardest target token. `./scripts/plot.sh` reads
the existing CSVs only and writes:

- `stability_horizon_by_budget.svg`
- `validity_horizon_by_budget.svg`
- `stability_horizon_fraction_by_budget.svg`
- `validity_horizon_fraction_by_budget.svg`

## Main Comparison Plots

`./scripts/plot.sh` also writes report-facing comparisons between external
DPA/TPA baselines and shared-MILP methods:

- `stability_one_sequence_main_comparison.png`
- `stability_full_matrix_main_comparison.png`
- `validity_one_sequence_main_comparison.png`
- `validity_all_prompts_main_comparison.png`

Independent composition and atomic phrase aggregation are diagnostic references,
not main baselines. When present, they are kept in diagnostic plots and described
in `audit_baseline_vs_milp_mapping.txt`.
