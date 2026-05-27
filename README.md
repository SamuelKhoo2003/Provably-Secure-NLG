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

`./scripts/plot.sh` also writes report-facing comparisons between the external
DPA stability baseline, the TPA sequence validity baseline, DPA validity
diagnostics, and shared-MILP methods:

- `stability_one_sequence_main_comparison.png`
- `stability_full_matrix_main_comparison.png`
- `validity_one_sequence_main_comparison.png`
- `validity_all_prompts_main_comparison.png`

Independent composition and atomic phrase aggregation are diagnostic references,
not main baselines. When present, they are kept in diagnostic plots and described
in `audit_baseline_vs_milp_mapping.txt`.

## validity_demo

`validity_demo` is an artificial controlled validity demonstration, not a
natural-language data distribution. It is designed to show when count-based TPA
sequence validity and shard-aware shared-MILP validity differ: each harmful
target token looks individually cheap from aggregate counts, but different token
positions require different shard groups.

Run the data and plots with:

```bash
./scripts/validity_demo.sh
```

The main SVG outputs are `validity_demo_baseline_vs_milp.svg`,
`validity_demo_budget_curve.svg`, and `validity_demo_certificate_vs_K.svg`. TPA
is the validity baseline; the joint row-column shared MILP is the proposed
shard-aware method. The DPA weakest harmful-token validity series is diagnostic
only, not a full-sequence validity baseline. Independent composition and atomic
phrase aggregation are not included in the main validity_demo plots. Outputs are
written under `outputs/validity_demo/`.
