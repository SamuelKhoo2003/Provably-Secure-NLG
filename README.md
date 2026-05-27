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

`./scripts/plot.sh` writes the simplified default report plot set to
`outputs/plots/`:

- `main_stability_budget_curve.svg`
- `main_validity_budget_curve.svg`
- `stability_certificate_vs_K.svg`
- `validity_certificate_vs_K.svg`

Certified fraction at poisoned shard budget `B` is
`100 * mean[B < B_star]`, where `B_star` is the minimum attack budget returned by
the corresponding baseline or MILP certificate. The inequality is strict:
if `B == B_star`, the attack is feasible and the region is not certified at that
budget.

Stability and validity are plotted separately. TPA appears only as the sequence
validity baseline. DPA weakest harmful-token validity is labelled as a
diagnostic, not a full-sequence validity baseline. Independent composition,
atomic phrase aggregation, row-only MILP, and column-only MILP are not included
in the default plot set.

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
