# Large Experiments

This folder contains the full-scale experiment pipeline for applying the certification framework to vote-vector outputs produced by the VPA tool-calling experiments.

The large-scale setting uses ensembles of shard-specific language-model adapters. Each shard model generates a tool-call prediction for each prompt. The resulting vote-vector files are then processed offline to compute stability and validity certificates, including DPA and standalone count-based TPA baselines and the row-column MILP certificates developed in this project.

## Purpose

The aim of this folder is to bridge the full VPA model outputs and the certification code used in the toy experiments.

The full-scale VPA runs produce JSONL files containing shard-level predictions. These files are not poisoned outputs. They are clean ensemble outputs from different training configurations. Certification then asks how many training examples an adversary would need to poison before the clean output could be changed or before an alternative valid tool call could be forced.

## Important distinction

The two 1B configurations are separate clean configurations, not clean versus poisoned models.

| Configuration   | Meaning                                            |
| --------------- | -------------------------------------------------- |
| `1b_full`       | OLMo-2-1B-Instruct with Full LoRA shard adapters   |
| `1b_last3_lora` | OLMo-2-1B-Instruct with Last-3 LoRA shard adapters |

The poisoned behaviour is hypothetical and is analysed by the certificates and MILPs. No poisoned model is trained by this pipeline.

## Expected input artifacts

The main input artifacts are VPA vote-vector JSONL files.

Example paths on Ada:

```bash
/data/mwicker/VPA/vote_vectors_1b_full_gpu0.jsonl
/data/mwicker/VPA/vote_vectors_1b_last3_lora_gpu1.jsonl
```

Each JSONL row corresponds to one prompt. A row contains fields such as:

| Field                  | Meaning                                                 |
| ---------------------- | ------------------------------------------------------- |
| `vote_vector`          | One final tool-call prediction per shard                |
| `token_vote_matrix`    | Token-level outputs for each shard                      |
| `vote_counts`          | Aggregate counts over `vote_vector`                     |
| `majority`             | Majority tool-call prediction                           |
| `robustness_radius`    | Stored aggregate robustness radius from the VPA run     |
| `ground_truth_correct` | Whether the majority tool call matched the ground truth |
| `majority_is_safe`     | Whether the majority tool call was marked safe          |

The key field for the row-column MILPs is `token_vote_matrix`.

For one prompt, it has shape:

```text
K x L_i
```

where `K = 500` is the number of shards and `L_i` is the generated token length for that prompt.

The prompt-token grid used by the MILPs is built by transposing and stacking these matrices:

```text
grid[prompt][position][shard] = token_id
```

For a fixed horizon `H`, the resulting grid has shape:

```text
N_H x H x K
```

where `N_H` is the number of prompts with at least `H` generated tokens.

## Horizon handling

Generated sequences have different lengths across prompts. For this reason, the pipeline uses fixed token horizons.

For a horizon `H`:

1. Keep only prompts with at least `H` generated tokens.
2. Truncate each retained prompt to the first `H` positions.
3. Convert each prompt from `K x H` to `H x K`.
4. Stack all retained prompts into an `N_H x H x K` grid.

Shorter prompts are not padded, because padding would create artificial token positions that were never generated.

The recommended main horizon is:

```text
H = 20
```

This keeps all 110 prompts in the current 1B full-scale runs while still covering meaningful tool-call structure.

Longer horizons such as `H = 40` and `H = 60` can be used as diagnostic sweeps, but they retain fewer prompts.

## Exporting the prompt-token grid

The script below exports the full-scale vote-vector JSONL files into CSV files that are easier to inspect and use with the certification code.

```bash
python large_experiments/scripts/export_vote_vector_grid.py \
  --input /data/mwicker/VPA/vote_vectors_1b_full_gpu0.jsonl \
  --name 1b_full \
  --horizon 20 \
  --output-dir large_experiments/outputs/vote_vector_grids \
  --write-shard-grid
```

For the Last-3 LoRA configuration:

```bash
python large_experiments/scripts/export_vote_vector_grid.py \
  --input /data/mwicker/VPA/vote_vectors_1b_last3_lora_gpu1.jsonl \
  --name 1b_last3_lora \
  --horizon 20 \
  --output-dir large_experiments/outputs/vote_vector_grids \
  --write-shard-grid
```

The expected output directory is:

```text
large_experiments/outputs/vote_vector_grids/<config>/H<horizon>/
```

For example:

```text
large_experiments/outputs/vote_vector_grids/1b_full/H020/
```

## Exported CSV files

The grid export produces the following files.

| File                       | Purpose                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `clean_grid.csv`           | One row per prompt-token cell with the clean majority token and DPA stability radius |
| `target_grid.csv`          | Representative alternative tool-call target tokens for validity analysis             |
| `aggregate_tool_votes.csv` | Aggregate tool-call vote counts from `vote_vector`                                   |
| `shard_votes_long.csv`     | Full shard-aware prompt-token grid in long CSV form                                  |
| `summary.json`             | Metadata about the export, horizon, retained prompts, and input file                 |

### `clean_grid.csv`

This file contains one row per prompt-token cell.

Important columns:

| Column                 | Meaning                                              |
| ---------------------- | ---------------------------------------------------- |
| `prompt_index`         | Index of the retained prompt                         |
| `original_row_index`   | Original row index in the JSONL file                 |
| `position`             | Token position                                       |
| `clean_token_id`       | Majority token at this prompt-token cell             |
| `clean_votes`          | Number of shards voting for the clean token          |
| `runner_up_token_id`   | Strongest competing token                            |
| `runner_up_votes`      | Number of shards voting for the strongest competitor |
| `dpa_stability_radius` | Token-level DPA stability radius                     |

This file can be used for the optional token-grid weakest-token DPA stability
diagnostic. It is not the main full-scale stability baseline.

### `target_grid.csv`

This file contains representative target tokens for professor-style MCP validity.

For each prompt, target classes are the observed non-majority tool-call predictions in `vote_vector`. For each target class, the script selects one shard that predicted that target class and uses that shard's generated token sequence as a representative target sequence.

Important columns:

| Column                       | Meaning                                               |
| ---------------------------- | ----------------------------------------------------- |
| `target_class`               | Alternative observed tool-call class                  |
| `representative_shard_index` | Shard used to obtain the target token sequence        |
| `target_token_id`            | Target token at this position                         |
| `clean_token_id`             | Clean majority token at this position                 |
| `active_position`            | Whether the target token differs from the clean token |
| `dpa_target_radius`          | Token-level DPA target diagnostic radius              |

Validity MILPs should normally use only rows where:

```text
active_position = 1
```

Shared prefix tokens should not be treated as adversarial target events, because the adversary does not need to change tokens that already match the target sequence.

### `aggregate_tool_votes.csv`

This file contains the class-level vote counts used for professor-style MCP validity.

It is derived from `vote_vector`, not from `token_vote_matrix`.

This file is used to compute aggregate TPA final-tool validity over alternative
tool-call classes.

### `shard_votes_long.csv`

This is the full shard-aware prompt-token grid in long form.

Each row corresponds to:

```text
prompt_index, position, shard_index, token_id
```

This is the CSV representation of:

```text
grid[prompt][position][shard] = token_id
```

This file is used by the row-only, column-only, and joint row-column MILPs.

For `N = 110`, `H = 20`, and `K = 500`, this file contains:

```text
110 x 20 x 500 = 1,100,000 rows
```

## Baselines

The large-scale experiment compares inherited final-tool vote-vector baselines
against proposed shard-aware prompt-token-grid MILPs.

### Stability baselines

The main inherited stability baseline operates on the final tool-call labels in
`vote_vector`.

The main stability baseline is:

```text
DPA final-tool stability
```

For each prompt, count final tool-call votes and compute:

```text
radius = floor((winner_votes - runner_up_votes - 1) / 2)
```

Negative radii are clamped to zero.

Token-grid weakest-token DPA remains available only as an optional diagnostic.

### Validity baselines

Validity asks whether an adversary can force an alternative valid MCP or tool-call output.

The professor-style full-scale validity baseline is:

```text
Aggregate TPA final-tool validity
```

For each prompt:

1. Use `vote_vector` to count tool-call predictions.
2. Treat observed non-majority tool calls as target classes.
3. Compute TPA for each target class.
4. Take the minimum over target classes.

The minimum is used because the adversary succeeds if it can force any alternative target class.

This is targeted partition aggregation over final MCP/tool-call label counts
from `vote_vector`. It is aggregate and count-based: it does not solve an
MILP, use shard identities, or enforce one shared poisoned-shard allocation.
The collective TPA+MSC multi-sample MILP is not implemented here. The
row-column validity MILP is a separate proposed method and must not be labeled
TPA+MSC.

A token-grid DPA target diagnostic is available with
`--include-token-grid-dpa-validity-diagnostic`.

## Fixed-budget joint MILP curves

The report-facing runner consumes the raw JSONL files directly. It solves the
fixed-budget adversarial-success view of the same event constraints used by the
toy row-column certificates:

```text
maximise successful prompts
subject to one shared poisoned-shard allocation with sum(a_k) <= B
```

Outputs label this objective as `fixed_budget_adversarial_success`. Row-only
and column-only MILPs are not run by default. `--max-targets-per-prompt` limits
only the MILP target bank; the validity baselines use every observed
non-majority target class.

The default `budget_curve_summary.csv` contains only:

```text
dpa_final_tool_stability
aggregate_tpa_final_tool_validity
joint_row_column_stability_milp
joint_row_column_validity_milp
```

Use `--include-token-grid-dpa-stability-diagnostic` or
`--include-token-grid-dpa-validity-diagnostic` to add the corresponding
token-grid DPA diagnostic to the summary.

## Recommended workflow

### 1. Inspect the input schema

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/data/mwicker/VPA/vote_vectors_1b_full_gpu0.jsonl")

with path.open() as f:
    row = json.loads(next(f))

for k, v in row.items():
    if isinstance(v, list):
        print(f"{k}: list length {len(v)}")
    elif isinstance(v, dict):
        print(f"{k}: dict with {len(v)} keys")
    else:
        print(f"{k}: {type(v).__name__} = {v}")
PY
```

### 2. Check retained prompts by horizon

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/data/mwicker/VPA/vote_vectors_1b_full_gpu0.jsonl")
horizons = [1, 5, 10, 20, 30, 40, 50, 60]

lengths = []
with path.open() as f:
    for line in f:
        row = json.loads(line)
        lens = {len(x) for x in row["token_vote_matrix"]}
        assert len(lens) == 1
        lengths.append(next(iter(lens)))

print("num rows", len(lengths))
for H in horizons:
    kept = sum(l >= H for l in lengths)
    print(f"H={H:02d}: kept {kept}/{len(lengths)}")
PY
```

### 3. Run a smoke certification

```bash
python large_experiments/scripts/certify_vote_vectors_runner.py \
  --input /data/mwicker/VPA/vote_vectors_1b_full_gpu0.jsonl \
  --name 1b_full \
  --horizon 5 \
  --max-prompts 5 \
  --budgets 0,1,3 \
  --top-competitors 1 \
  --max-targets-per-prompt 1 \
  --milp-time-limit 120 \
  --threads 4 \
  --output-dir large_experiments/outputs/certification
```

### 4. Run the H=20 curve

```bash
python large_experiments/scripts/certify_vote_vectors_runner.py \
  --input /data/mwicker/VPA/vote_vectors_1b_full_gpu0.jsonl \
  --name 1b_full \
  --horizon 20 \
  --budgets 0,1,3,5,7,9,25,50,100,150,200,249 \
  --top-competitors 1 \
  --max-targets-per-prompt 2 \
  --milp-time-limit 600 \
  --mip-gap 0.001 \
  --threads 8 \
  --quiet-gurobi \
  --output-dir large_experiments/outputs/certification
```

Repeat with `vote_vectors_1b_last3_lora_gpu1.jsonl` and a distinct `--name`.
The configurations are independent clean ensembles; neither is interpreted as
a poisoned version of the other.

Outputs are written under
`large_experiments/outputs/certification/<name>/H020/`:

```text
dpa_final_tool_stability.csv
aggregate_tpa_final_tool_validity.csv
joint_row_column_stability_milp.csv
joint_row_column_validity_milp.csv
budget_curve_summary.csv
summary.json
```

Optional diagnostic files are:

```text
dpa_token_grid_weakest_token_stability_diagnostic.csv
dpa_max_target_token_validity_diagnostic.csv
```

## Ada setup

Large experiment outputs and virtual environments should be stored on bitbucket-backed or `/data2` storage rather than in the home directory.

The intended layout is:

```text
Repo:
  ~/Projects/Provably-Secure-NLG

Large outputs:
  large_experiments/outputs/

External VPA artifacts:
  /data/mwicker/VPA/
```

If using the Ada helper scripts, activate the large-experiments environment before running certification scripts.

```bash
source large_experiments/scripts/activate_ada_large_experiments.sh
```

## Notes

Do not rerun model training or generation when running certification.

Do not treat `1b_full` as clean and `1b_last3_lora` as poisoned.

Do not pad shorter generated sequences.

Do not modify the toy validity demo from this pipeline.

The full-scale certification pipeline is offline. It consumes stored vote-vector JSONL files and produces certification results.
