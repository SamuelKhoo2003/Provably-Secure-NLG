# Provably-Secure-NLG

EIE Final Year Project 2026.

This repository contains a runnable toy implementation of row/column poisoning certificates for natural-language generation style token voting. The core experiment builds a prompt-by-token vote matrix, solves shared-allocation MILPs with Gurobi, and compares those certificates with DPA-style stability baselines and TPA-style targeted validity baselines.

The cleaned first-party implementation lives in `toy_certificate/`. The
`phd_reference/` directory is external read-only reference code and should not be
edited, reformatted, linted, or used as the source of generated outputs.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gurobi also needs a valid local license.

Run the lightweight check pipeline:

```bash
./scripts/check.sh
```

This runs compile checks, unit tests, and one visualization instance. It does not run the benchmark.

## Common Commands

Quick smoke test:

```bash
./scripts/check.sh
```

Run benchmark data generation only:

```bash
./scripts/data.sh
```

By default this writes `toy_results/small_benchmark/benchmark_results.csv` for
the small preset: `K={4,6,8}`, `N={2,3}`, `L={2,3,4}`, `T={4,6}`,
`delta_stab=delta_val={0.2}`, and `target_bias={0.3}`. The shell wrappers use
`STABILITY_COMPETITOR_MODE=runner_up` by default for speed; set it to `all` for
exact all-competitor runs.

It also writes reusable long-format sidecar CSVs by default:

```text
benchmark_budget_curves.csv
benchmark_damage_curves.csv
benchmark_horizons.csv
```

You can override the ranges with environment variables:

```bash
KS=4,6 NS=2 LENGTHS=2,3 TS=4 DELTA_STABS=0.2 DELTA_VALS=0.2 TARGET_BIASES=0.3 ./scripts/data.sh
```

Budget-curve controls:

```bash
BUDGET_MAX=15 MAKE_BUDGET_CURVES=1 MAKE_DAMAGE_CURVES=1 MAKE_HORIZON_CURVES=1 ./scripts/data.sh
```

Or choose a preset explicitly:

```bash
PRESET=large ./scripts/data.sh
```

Refresh plots from an existing benchmark CSV:

```bash
./scripts/plot.sh
```

Run data generation followed by plotting:

```bash
./scripts/benchmark.sh
```

This uses the small preset by default. Larger sweeps remain available through
`PRESET=medium`, `PRESET=large`, or direct CLI arguments.

Generate one visualization instance:

```bash
./scripts/visualize.sh
```

## Experiment Taxonomy

The core quantity is:

```text
B* = minimum number of poisoned shards needed to make an attack objective feasible
```

The benchmark records four metric families. Radius-style metrics keep the
existing `B*` columns in `benchmark_results.csv`. Radius-derived coverage fixes
a budget `B` and counts a prompt as certified exactly when `B < B*_row`. Direct
damage-at-budget MILPs also fix `B`, but solve a new maximization problem that
directly maximizes attacked prompt rows under one shared poisoned-shard
allocation. Horizon metrics fix `B` and measure how many initial token positions
remain certified on average.

The radius-derived curves are cheap summaries of per-row certificates. The
direct damage-at-budget curves are more faithful to the shared-allocation threat
model because they solve a new MILP for each budget and maximise adversarial
damage under that budget.

The MILP uses one shared poisoned-shard allocation `a[k] in {0,1}`. The same
selected shards are reused across all required prompt rows and token positions;
this shared allocation is the row/column coupling studied here.

Stability measures the budget needed to change outputs away from the clean
generation. A cell is destabilised if any competitor token can tie or beat the
clean winner after poisoning. The shared MILP checks all competitors, not just
the original runner-up. Runner-up margins remain useful for simple DPA-style
baselines.

Stability has two competitor modes. `all` is exact and checks every competitor
token, matching the stated threat model. `runner_up` checks only the original
runner-up token; it is a cheaper DPA-style top-vs-second approximation that may
overestimate robustness if another competitor is easier to promote. If both
models solve optimally, expect `B*_runner_up >= B*_all`. Use `all` for
correctness and report-critical runs; use `runner_up` for large sweeps only after
the comparison diagnostic shows it is close enough for the chosen toy
distribution.

Validity measures the budget needed to force harmful target tokens or full
harmful sequences. A validity cell succeeds only if the harmful target token ties
or beats every competitor token, which is stricter than only beating the current
clean winner.

There are two main external baselines.
The first is the token-level DPA margin baseline: each prompt-token cell is
certified independently using the standard clean-winner versus runner-up margin
for stability, and a simple targeted margin against the strongest non-target
competitor for validity. This is a weakest harmful-token reference, not a full-
sequence certificate.
The second is the Ghitu-style phrase-level TPA baseline: token certificates are
composed across each generated response. Phrase stability is controlled by the
weakest token, while phrase validity is controlled by the hardest target token,
implemented as the maximum over token-level TPA validity radii.
Atomic phrase aggregation and independent composition are retained only as
diagnostic references. The proposed shared row-column MILP is evaluated against
these two baselines and should not be conflated with them.

For report-facing objectives, use readable names rather than q/r shorthand:
`q` is affected prompt rows and `r` is affected token positions per selected
row. Stability objectives use `solve_structured_stability`: one prompt/one token
is `q_rows=1, r_cols=1`; one prompt/full sequence is `q_rows=1, r_cols=L`; all
prompts/one token each is `q_rows=N, r_cols=1`; all prompts/full matrix is
`q_rows=N, r_cols=L`. Validity objectives use `solve_row_col_validity`: one
harmful sequence is `q_rows=1`; harmful sequences for all prompts is `q_rows=N`.

The row-only, column-only, and joint row-column MILPs are proposed-method
variants and ablations. They are not external baselines.

## Baselines

The DPA weakest harmful token baseline computes token-level margins
independently and represents each prompt by its easiest harmful token,
`row_radius[i] = min_j B_cell[i,j]`. For validity, this is not a full-sequence
certificate.

The TPA max-token sequence baseline is the paper-inspired targeted validity
baseline. It computes targeted token radii and uses `max_j r[i,j]` for a harmful
sequence because every target token must be forced.

Atomic phrase aggregation, kept in CSV columns named `phrase_dpa_*` for
compatibility, treats the whole generated sequence as one label. It is a crude
full-sequence diagnostic and should not be interpreted as the main TPA validity
baseline.

Independent composition sums token-level costs and does not reuse poisoned
shards across cells, so it is a loose upper reference rather than a main
baseline.

The shared row-column MILP is the proposed method. Row-only, column-only, and
joint row-column solves are report-facing variants or ablations of that method,
not external baselines.

## Solver Exactness

`CertificateResult` includes `is_optimal`, `mip_gap`, `lower_bound`, and
`upper_bound`. If `is_optimal=True`, `B_star` is an exact optimum. If Gurobi
returns `TIME_LIMIT` or `SUBOPTIMAL` with a feasible solution, `B_star` is the
best feasible upper bound found, not an exact certificate. The `attacked_cells`
field is diagnostic; because `z` variables are not secondarily minimized, it may
include extra feasible cells. The certified quantity is `B_star`.

## Plot Taxonomy

The plots are separated by attack objective. Stability plots measure the budget
needed to change outputs away from the clean generation. Validity plots measure
the budget needed to force specific harmful target tokens or sequences. Validity
plots use clear labels for four references: DPA weakest harmful token is the
easiest single harmful token and not a full-sequence certificate; TPA max-token
sequence is the targeted sequence baseline using the maximum over token
positions; atomic phrase aggregation treats the whole sequence as one label;
shared row-column MILP is the proposed method using one poisoned-shard
allocation across all required cells.

Preferred report-facing plots:

```text
validity_one_prompt_by_L.svg          force one harmful sequence
validity_all_prompts_by_L.svg         force harmful sequences for all prompts
stability_one_prompt_by_L.svg         destabilise one prompt
stability_all_prompts_by_L.svg        destabilise all prompts
structured_stability_heatmap.svg      B*(q,r) over affected prompts/tokens
validity_independent_overestimate_by_L.svg
stability_independent_overestimate_by_L.svg
certified_fraction_stability_by_budget.svg
certified_fraction_validity_by_budget.svg
certified_fraction_stability_by_L_at_budget.svg
certified_fraction_validity_by_L_at_budget.svg
stability_horizon_by_budget.svg
validity_horizon_by_budget.svg
```

Independent-composition diagnostics are separated because they can dominate the
y-axis and make the shared MILP curves unreadable.

Run tests directly:

```bash
.venv/bin/python -m unittest discover
```

## Documentation

- `readme/spec.md`: consolidated design/specification notes for what the toy certificate experiment is meant to build.
- `readme/implementation.md`: consolidated implementation, command, benchmark, plot, and baseline explanation.
- `readme/phd_readme.md`: notes on the external `phd_reference` package structure and where the closest reference-code concepts live.

## Developer Notes

- Keep new implementation work inside `toy_certificate/`, `scripts/`, `tests/`,
  and `readme/`.
- Treat `phd_reference/` as read-only external reference code.
- Large Gurobi sweeps can be expensive, especially with exact all-competitor
  stability. Use runner-up stability mode only as an approximation/diagnostic
  after checking it against all-competitor mode on the intended toy distribution.
- Use `is_optimal`, `mip_gap`, `lower_bound`, and `upper_bound` to distinguish
  exact optima from time-limited feasible upper bounds.

## Repository Layout

- `toy_certificate/`: toy data generator, MILP solvers, experiment CLI, plotting helpers.
- `phd_reference/`: external read-only reference package.
- `scripts/check.sh`: compile, test, and small visualization smoke test.
- `scripts/data.sh`: small benchmark data generation, writing `benchmark_results.csv`.
- `scripts/plot.sh`: plot refresh from an existing CSV without rerunning Gurobi.
- `scripts/benchmark.sh`: small benchmark data-generation plus plotting workflow.
- `scripts/visualize.sh`: one tiny visualization instance.
- `tests/`: lightweight test bench.
- `historical_csvs/`: saved benchmark CSVs.
- `toy_results/`: generated outputs, ignored by git.
