# Synthetic Sweep Benchmark

This benchmark studies how the toy certificates and Gurobi runtime scale with
the number of shards (`K`), prompts (`N`), and generated tokens (`L`). The
vocabulary size (`T=5`) and all data-generation settings remain fixed. There is
intentionally no `T` sweep.

## Experiment design

Each standard sweep changes one parameter:

| Sweep | Varied values | Fixed values |
| --- | --- | --- |
| `K` | `8, 12, 16, 20, 24` | `N=4`, `L=5`, `T=5` |
| `N` | `1, 2, 4, 6, 8` | `K=20`, `L=5`, `T=5` |
| `L` | `1, 2, 4, 6, 8` | `K=20`, `N=4`, `T=5` |

All use the dense toy generator, seed `0`, `delta_stab=0.2`,
`delta_val=0.2`, target bias `0.3`, and the full objective family. Stability is
always evaluated against all competing tokens; the old runner-up-only
diagnostic has been removed because report-facing stability is an untargeted
any-token-change property. Budget-curve generation remains enabled.

The separate degeneracy study evaluates the Cartesian product
`N in {1,4}` and `L in {1,4}` at `K=20`, giving:

- `N=1, L=1`: single prompt and single token.
- `N=1, L>1`: sequence-only structure.
- `N>1, L=1`: multi-prompt, single-token structure.
- `N>1, L>1`: the full prompt-token matrix.

For `N=1, L=1`, the shared MILP is a single-cell sanity check and should
agree with the corresponding token-level DPA-style certificate, apart from
tie-breaking and competitor conventions. The other three cases isolate
sequence structure, cross-prompt structure, and the complete two-dimensional
grid respectively.

## Outputs

Results are isolated under:

```text
toy_experiments/outputs/sweep_benchmark/
  K/{results,plots}/
  N/{results,plots}/
  L/{results,plots}/
  degenerate/{results,plots}/
```

Each `results` directory contains `benchmark_results.csv` and, because budget
curves are enabled, `benchmark_budget_curves.csv`. Each `plots` directory
contains report-facing SVGs or the degeneracy LaTeX table, plus
`audit_sweep.md`.

## Plot interpretation

- `sweep_K_stability_certificate_vs_K.svg` shows how adding shard models
  changes DPA weakest-token stability and the two shared stability objectives.
- `sweep_K_validity_certificate_vs_K.svg` compares the two external validity
  baselines with the proposed shared validity MILP as the ensemble grows.
- The `N` stability and validity plots show the effect of requiring a shared
  poisoning objective across more prompt rows; `sweep_N_runtime_vs_N.svg`
  reports the associated solver cost.
- The `L` stability and validity plots show how longer generated sequences
  affect weakest-token, one-token-per-row, and full-grid certification;
  `sweep_L_runtime_vs_L.svg` reports solver cost.
- `degenerate_study_table.tex` and its compact source
  `degenerate_study.csv` compare the four cases where one or both grid
  dimensions collapse.

The stability plots distinguish:

- DPA weakest-token stability: `dpa_stab_row_radius_qN`.
- Shared MILP with one affected token per prompt row:
  `row_col_stab_qN_r1`.
- Shared MILP over the full prompt-token grid:
  `row_col_stab_qN_rL`.

The validity plots distinguish plain DPA, TPA multi-sample validity, and the
shared shard-aware MILP using `plain_dpa_val_sequence_qN`,
`tpa_val_sequence_qN`, and `row_col_val_qN`.

Runtime plots use `runtime_gurobi_total`, the total Gurobi objective runtime
recorded for each generated instance. The audit records all parameter values,
solver statuses, generated files, and missing or skipped series.

## Commands

Validate every grid and print solve counts without invoking Gurobi:

```bash
MODE=dry-run ./toy_experiments/scripts/sweep_benchmark.sh
```

Generate all result CSVs, then render all plots:

```bash
MODE=all ./toy_experiments/scripts/sweep_benchmark.sh
```

Render plots from existing CSVs without invoking Gurobi:

```bash
MODE=plot ./toy_experiments/scripts/sweep_benchmark.sh
```

Run only one study by setting `SWEEP`, for example:

```bash
MODE=dry-run SWEEP=K ./toy_experiments/scripts/sweep_benchmark.sh
MODE=data SWEEP=N ./toy_experiments/scripts/sweep_benchmark.sh
MODE=plot SWEEP=N ./toy_experiments/scripts/sweep_benchmark.sh
```

The configurations are `toy_experiments/configs/sweep_K.yaml`,
`sweep_N.yaml`, `sweep_L.yaml`, and `sweep_degenerate.yaml`.
