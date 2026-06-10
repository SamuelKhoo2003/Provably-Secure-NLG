# Large Experiments

`large_experiments/` contains the offline full-scale certification pipeline for
VPA tool-calling vote-vector outputs. It consumes stored JSONL votes, computes
final-tool DPA/TPA baselines, solves shard-aware joint MILPs with Gurobi, and
plots completed certification runs.

It does not train poisoned models or rerun language-model generation.

## Layout

```text
large_experiments/
  scripts/
    certify_vote_vectors_runner.py   Report-facing certification runner
    export_vote_vector_grid.py       Optional CSV inspection export
    plot_certification_curves.py     PDF curve comparison tool
  tests/                             Solver-free runner and plotter tests
  vote_vector_utils.py               Shared deterministic vote-count helpers
  outputs/                           Generated certification and plot artifacts
```

## Input Interface

The report-facing runner expects one JSON object per prompt with:

```text
vote_vector
token_vote_matrix
vote_counts
majority
```

For the current full-scale setup:

- `vote_vector` contains one final tool-call label per shard.
- `token_vote_matrix` contains one generated token sequence per shard.
- `vote_counts` must equal `Counter(vote_vector)`.
- `majority` must equal the deterministic majority of `vote_vector`.
- The runner expects 500 shards by default.
- Every shard sequence in one row must have the same raw list length.

The two 1B configurations are independent clean ensembles:

| Configuration | Meaning |
| --- | --- |
| `1b_full` | OLMo-2-1B-Instruct with Full LoRA shard adapters |
| `1b_last3_lora` | OLMo-2-1B-Instruct with Last-3 LoRA shard adapters |

They are not clean and poisoned versions of the same model. Poisoning is
hypothetical and is represented only by the certification objectives.

## Horizon And Padding Policy

Some shard sequences contain `None` padding after generation terminates. The
certification runner defines a row's usable sequence length as the shortest
non-`None` prefix across all shard sequences:

```python
non_none_prefix_lengths = [
    next((idx for idx, token in enumerate(sequence) if token is None), len(sequence))
    for sequence in matrix
]
usable_sequence_length = min(non_none_prefix_lengths)
```

For horizon `H`, a row is retained only when:

```text
usable_sequence_length >= H
```

Retained rows are truncated to the first `H` real token IDs and transformed
into:

```text
grid[prompt][position][shard] = token_id
```

with shape:

```text
N_H x H x 500
```

The runner does not replace `None` with fake token IDs and does not pad shorter
generations. It preserves raw-length consistency validation.

For `vote_vectors_1b_full_gpu0.jsonl`, the observed usable-prefix retention was:

```text
H=05: 110/110
H=10: 109/110
H=15: 104/110
H=20: 77/110
H=30: 23/110
H=40: 4/110
H=50: 0/110
H=60: 0/110
```

`H=20` remains a useful report horizon, but it does not retain all prompts.
Comparisons across horizons should report their retained prompt counts.

## Full-Scale Comparison

The main comparison deliberately uses different data interfaces for inherited
baselines and the proposed methods.

### Stability

```text
DPA final-tool stability
Joint row-column stability MILP
```

`dpa_final_tool_stability` uses only final tool-call counts from `vote_vector`.
For each prompt:

```text
radius = max(0, floor((winner_votes - runner_up_votes - 1) / 2))
```

The token-grid weakest-token DPA curve is optional diagnostic output, not the
main full-scale stability baseline.

### Validity

```text
Aggregate TPA final-tool validity
Joint row-column validity MILP
```

`aggregate_tpa_final_tool_validity`:

1. counts final tool-call labels in `vote_vector`;
2. treats every observed non-majority label as a possible target;
3. computes standalone count-based TPA for each target;
4. takes the easiest target attack for the prompt.

This TPA baseline does not solve an MILP, use shard identities, or enforce one
shared poisoned-shard allocation. Collective TPA+MSC is not implemented.

The joint validity MILP uses representative target token sequences from shards
that produced observed non-majority final-tool classes. Shared clean-prefix
positions are excluded. Every active target position is constrained against
all observed non-target competitor tokens.

The token-grid max-target-token DPA curve is optional diagnostic output, not the
main full-scale validity baseline.

## Fixed-Budget MILPs

The joint methods solve:

```text
maximize successful prompts
subject to one shared poisoned-shard allocation with sum(a_k) <= B
```

The objective mode is:

```text
fixed_budget_adversarial_success
```

Gurobi is always used. The runner prints:

```text
[stability] solver=gurobi budget=...
[validity] solver=gurobi budget=...
```

For interrupted or time-limited solves, the reported certified fraction uses
the conservative Gurobi objective bound rather than the incumbent objective.

`--top-competitors` controls stability event generation.
`--max-targets-per-prompt` limits the validity MILP target bank only; the
aggregate final-tool TPA baseline still considers every observed non-majority
final-tool class.

## Running Certification

Use the repository virtual environment explicitly:

```bash
.venv/bin/python large_experiments/scripts/certify_vote_vectors_runner.py \
  --input /data/mwicker/VPA/vote_vectors_1b_full_gpu0.jsonl \
  --name 1b_full_final_tool_baselines_top3 \
  --horizon 20 \
  --budgets 0,1,3,5,7,9,25,50,100,150,200,249 \
  --top-competitors 3 \
  --max-targets-per-prompt 2 \
  --milp-time-limit 600 \
  --mip-gap 0.001 \
  --threads 8 \
  --quiet-gurobi \
  --output-dir large_experiments/outputs/certification
```

Useful controls:

| Flag | Meaning |
| --- | --- |
| `--max-prompts N` | Stop after retaining `N` usable prompts |
| `--skip-stability-milp` | Write baselines and omit stability MILP solves |
| `--skip-validity-milp` | Write baselines and omit validity MILP solves |
| `--include-token-grid-dpa-stability-diagnostic` | Include the optional token-grid stability diagnostic |
| `--include-token-grid-dpa-validity-diagnostic` | Include the optional token-grid validity diagnostic |
| `--milp-time-limit SECONDS` | Set the Gurobi time limit |
| `--mip-gap VALUE` | Set the Gurobi relative MIP gap |
| `--threads N` | Set Gurobi threads; `0` uses automatic mode |

Outputs are written to:

```text
large_experiments/outputs/certification/<name>/H<horizon>/
```

When `--max-prompts N` is used, an additional `NNN` directory is added:

```text
large_experiments/outputs/certification/<name>/H020/N010/
```

## Certification Outputs

Default output files:

```text
dpa_final_tool_stability.csv
aggregate_tpa_final_tool_validity.csv
joint_row_column_stability_milp.csv
joint_row_column_validity_milp.csv
budget_curve_summary.csv
summary.json
```

MILP CSV files are omitted when their corresponding `--skip-...-milp` flag is
used.

Optional diagnostic files:

```text
dpa_token_grid_weakest_token_stability_diagnostic.csv
dpa_max_target_token_validity_diagnostic.csv
```

By default, `budget_curve_summary.csv` contains only:

```text
dpa_final_tool_stability
aggregate_tpa_final_tool_validity
joint_row_column_stability_milp
joint_row_column_validity_milp
```

Diagnostic methods are included only when their corresponding flags are passed.

Important `summary.json` fields include:

```text
solver = gurobi
objective_mode = fixed_budget_adversarial_success
horizon_filter_basis = shortest_non_none_prefix_across_shards
padding_policy = no_padding_none_rows_filtered
full_scale_baseline_interface = final_tool_vote_vector
proposed_method_interface = shard_aware_prompt_token_grid
main_stability_baseline = dpa_final_tool_stability
main_validity_baseline = aggregate_tpa_final_tool_validity
validity_constraint = target_ties_or_beats_every_observed_non_target_token
```

It also records total rows read, rows filtered for horizon, rows truncated to
the horizon, retained prompts, event counts, solver controls, and diagnostic
inclusion flags.

## Plotting Certification Curves

Plot one run:

```bash
.venv/bin/python large_experiments/scripts/plot_certification_curves.py \
  --inputs large_experiments/outputs/certification/1b_full_final_tool_baselines_top3/H020 \
  --labels "1B full H=20" \
  --output-dir large_experiments/outputs/certification/plots/1b_full_H020_top3 \
  --title-prefix "1B full H=20" \
  --filename-prefix "1b_full_H020_top3"
```

Compare horizons:

```bash
.venv/bin/python large_experiments/scripts/plot_certification_curves.py \
  --inputs \
    large_experiments/outputs/certification/1b_full_final_tool_baselines_top3/H015 \
    large_experiments/outputs/certification/1b_full_final_tool_baselines_top3/H020 \
  --labels "1B full H=15" "1B full H=20" \
  --output-dir large_experiments/outputs/certification/plots/1b_full_horizon_compare_top3 \
  --title-prefix "1B full horizon comparison" \
  --filename-prefix "1b_full_h15_h20_top3"
```

Add `--milp-only` to show only the joint stability and validity MILPs.

The plotter writes:

```text
<prefix>_stability_budget_curve.pdf
<prefix>_validity_budget_curve.pdf
<prefix>_all_methods_budget_curve.pdf
<prefix>_plot_ready_budget_curves.csv
<prefix>_method_comparison_summary.csv
```

There is no runtime plot.

Certified-fraction plots:

- use PDF output only;
- place 12-point legends dynamically with `loc="best"`;
- use distinct line styles and markers for overlapping methods;
- round the dynamic lower y-axis limit down to the nearest 5%;
- keep the upper y-axis limit fixed at 105%.

The plotter accepts only current method names. Legacy names are not remapped and
raise a `ValueError`; regenerate old certification runs before using them in the
current full-scale report comparison.

## Optional CSV Grid Export

`export_vote_vector_grid.py` writes inspection-oriented CSV representations:

```bash
.venv/bin/python large_experiments/scripts/export_vote_vector_grid.py \
  --input /path/to/vote_vectors.jsonl \
  --name 1b_full \
  --horizon 20 \
  --output-dir large_experiments/outputs/vote_vector_grids \
  --write-shard-grid
```

Outputs:

```text
clean_grid.csv
target_grid.csv
aggregate_tool_votes.csv
summary.json
shard_votes_long.csv        # only with --write-shard-grid
```

This exporter is for inspection. The certification runner is the authoritative
path for full-scale padded JSONL files because it filters horizons using the
shortest non-`None` prefix across shards.

## Tests

Run the maintained solver-free tests:

```bash
MPLCONFIGDIR=/tmp/provably-secure-mpl \
XDG_CACHE_HOME=/tmp/provably-secure-cache \
.venv/bin/python -m unittest discover -v
```

The tests cover:

- shortest non-`None` prefix filtering;
- final-tool DPA and TPA baselines;
- default and diagnostic summary method selection;
- all-competitor validity event construction;
- conservative solver-bound handling;
- strict plot method-name validation;
- filename prefixes and PDF rendering.

They do not run full certification or Gurobi jobs.

## Operational Notes

- Do not rerun model training or generation when only certification is needed.
- Do not treat `1b_full` as clean and `1b_last3_lora` as poisoned.
- Do not pad shorter generated sequences.
- Do not replace `None` padding with artificial token IDs.
- Do not modify toy validity-demo code from the full-scale pipeline.
- Do not commit generated certification CSVs or plots unless they are
  intentionally tracked report artifacts.
