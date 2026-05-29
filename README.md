# Provably-Secure-NLG

EIE Final Year Project 2026.

## Experiment Configs

Benchmark/data runs are configured through YAML files. `scripts/data.sh`
requires `CONFIG` and passes that config to the Python benchmark runner:

```bash
CONFIG=configs/validity_demo.yaml ./scripts/data.sh
CONFIG=configs/validity_demo.yaml DRY_RUN=1 VERBOSE=1 ./scripts/data.sh
```

Config files are strict: required fields such as `generator`, `K_values`,
`N_values`, `L_values`, `T_values`, `seed`, `budget_max`, `output_dir`,
`influence_mode`, `stability_competitor_mode`, `objective_family`, and
`make_budget_curves` must be present with the right type. Direct shell
overrides for grid values are not supported by `scripts/data.sh`. The Python `--preset
smoke|small|medium|large` path remains a legacy convenience for manual use; repo
experiment runs should use `--config`.

## Budget Curves

`CONFIG=<path> ./scripts/data.sh` writes `benchmark_budget_curves.csv` alongside
the benchmark results when budget curves are enabled in YAML. These curves
summarize certified fraction at fixed poisoning budget `B`. This is the only
active sidecar CSV in the report-facing pipeline.

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

Stability and validity are plotted separately. The report-facing stability
comparison is DPA weakest-token stability versus the full shared MILP. The
report-facing validity comparison is DPA weakest harmful-token diagnostic versus
TPA max-token sequence and the full shared MILP. Independent composition and
atomic phrase aggregation remain diagnostic references. Row-only and column-only
MILPs are legacy/debug ablations and are not computed or plotted by the normal
benchmark report path.

## validity_demo

`validity_demo` is an artificial controlled validity demonstration, not a
natural-language data distribution. It is designed to show when count-based TPA
sequence validity and shard-aware shared-MILP validity differ: each harmful
target token looks individually cheap from aggregate counts, but different token
positions require different shard groups.

Run the data and plots with:

```bash
./scripts/validity_demo.sh
DRY_RUN=1 VERBOSE=1 ./scripts/validity_demo.sh
```

`scripts/validity_demo.sh` defaults to `configs/validity_demo.yaml`; that YAML
controls the validity-demo size and objective selection. The config sets
`objective_family: validity_only` and disables stability objectives, stability
budget curves, and other stability work so the demo does not accidentally run
expensive stability MILPs. The validity-demo plots show only DPA weakest harmful
token, TPA max-token sequence, and shared MILP full sequence comparisons.

The main SVG outputs are `validity_demo_baseline_vs_milp.svg`,
`validity_demo_budget_curve.svg`, and `validity_demo_certificate_vs_K.svg`. TPA
is the validity baseline; the shared MILP full-sequence objective is the
proposed shard-aware method. The DPA weakest harmful-token validity series is
diagnostic only, not a full-sequence validity baseline. Independent composition
and atomic phrase aggregation are not included in the main validity_demo plots.
Outputs are written under `outputs/validity_demo/`.
