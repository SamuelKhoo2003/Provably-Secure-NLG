# VPA Integration Audit - Stage 0

Audit date: 2026-06-01

Scope: inspected the existing repository, with emphasis on `external/VPA-main` and the first-party `toy_certificate` code. No training or inference was run.

## 1. Relevant Existing Files

### VPA-main data, sharding, and training

- `external/VPA-main/src/data_loader.py`
  - Loads a Toucan dataset from disk, filters by top tool names, shuffles with seed `42`, and writes `train.jsonl` and `test.jsonl`.
  - Hard-codes dataset and output paths.
- `external/VPA-main/src/generate_shards.py`
  - Reads `train.jsonl` and writes `K = 1000` shard files named `shard_0000.jsonl`, `shard_0001.jsonl`, etc.
  - Preserves shard identity in filenames only.
  - Deletes/recreates the shard directory when run.
- `external/VPA-main/src/train_shards.py`
  - Orchestrates training over `data/shards/*.jsonl`.
  - Writes PEFT adapter directories named after shard ids under `output/adapters`.
- `external/VPA-main/src/train_worker.py`
  - Trains full-layer LoRA adapters for one shard.
  - Saves adapter with `model.save_pretrained(output_dir)`.
- `external/VPA-main/src/train_shards_last3_lora.py`
  - Sequentially trains last-3-layer LoRA adapters over all shard files.
  - Writes adapter directories under `output/adapters_last3_lora`.
- `external/VPA-main/src/train_last3_lora.py`
  - Worker for last-3-layer LoRA adapters, targeting layers `[13, 14, 15]`.
- Other training/benchmark variants:
  - `external/VPA-main/src/train_last3.py`
  - `external/VPA-main/src/train_fplc.py`
  - `external/VPA-main/src/train_worker_partial.py`
  - `external/VPA-main/src/benchmark_training.py`
  - `external/VPA-main/src/compare_training_approaches.py`

### VPA-main inference, voting, and certification

- `external/VPA-main/src/eval_ensemble.py`
  - Evaluates zero-shot, single-shard, and final tool-name majority voting over `NUM_ENSEMBLE_SHARDS = 20`.
  - Stores per-example predictions in memory as `ensemble_predictions = [[] for _ in range(n)]`.
  - Does not save per-shard votes or a reusable artifact.
- `external/VPA-main/src/certify_vpa.py`
  - Main large certification script: `NUM_SAMPLES = 100`, `NUM_SHARDS = 500`, `MAX_NEW_TOKENS = 60`.
  - Generates one complete output per shard, extracts a final tool name, counts final tool-name votes, computes a robustness radius, and appends JSONL results.
  - Saves only aggregate `vote_counts`, not per-shard vote identities.
- `external/VPA-main/src/rerun_examples.py`
  - Variant of `certify_vpa.py` for rerunning specific examples and resuming.
  - Same final tool-name voting pattern; saves aggregate `vote_counts`.
- `external/VPA-main/src/eval_token_ensemble.py`
  - Implements token-level majority voting during generation.
  - At each token position it collects `next_token_votes`, takes `Counter(...).most_common(1)[0][0]`, appends the majority token, and discards the full vote list.
  - Prints accuracy only; no artifact output.
- `external/VPA-main/src/certify_vpa_token.py`
  - Token-level generation/certification comparison over `NUM_SAMPLES = 10`, `NUM_SHARDS = 50`.
  - At each token step it collects `next_token_votes`, keeps only the majority token, then writes `vpa_token_level_results.json`.
  - The saved JSON contains result summaries, not the token vote matrix.
- Other evaluation helpers:
  - `external/VPA-main/src/eval_models_prompt.py`
  - `external/VPA-main/src/eval_last3_lora.py`
  - `external/VPA-main/src/quick_eval.py`
  - `external/VPA-main/src/cpu_speed_test.py`
  - These are useful for prompt/model checks but are not artifact-producing vote exporters.
- `external/VPA-main/src/test_filter.py`
  - Provides `extract_filter_features`, `is_solvable`, and `filter_test_set`.
  - Imported by the VPA evaluation/certification scripts to select filtered examples.

### First-party certificate code

- `toy_certificate/data.py`
  - Defines `ToyData` with token vote tensors: `stab_votes`, `val_votes`, counts, predictions, targets, and `influence`.
  - Existing expected vote shape is `(K, N, L)` for votes and `(N, L, T)` for counts.
  - Provides reusable helpers such as `compute_counts`, `majority_predictions`, `runner_up_tokens`, `generate_influence`, and `stability_margins`.
- `toy_certificate/baselines.py`
  - Implements stability and validity baselines over token-level vote/count tensors.
  - Useful once a real VPA vote artifact is adapted into the same shape conventions.
- `toy_certificate/milp.py`
  - Gurobi MILP solvers expect shard-level token vote tensors and optional influence masks.
- `toy_certificate/csv_io.py`
  - Small CSV read/write helper module.
- `toy_certificate/experiments.py`
  - Current orchestration for synthetic experiments; it is built around generated `ToyData`, not real VPA artifacts.

## 2. Files With Hard-Coded Cluster Paths

Files with `/data/mwicker/VPA` paths:

- `external/VPA-main/src/benchmark_training.py`
- `external/VPA-main/src/certify_vpa.py`
- `external/VPA-main/src/certify_vpa_token.py`
- `external/VPA-main/src/cpu_speed_test.py`
- `external/VPA-main/src/data_loader.py`
- `external/VPA-main/src/eval_ensemble.py`
- `external/VPA-main/src/eval_last3_lora.py`
- `external/VPA-main/src/eval_models_prompt.py`
- `external/VPA-main/src/eval_token_ensemble.py`
- `external/VPA-main/src/generate_shards.py`
- `external/VPA-main/src/quick_eval.py`
- `external/VPA-main/src/rerun_examples.py`
- `external/VPA-main/src/test_filter.py`
- `external/VPA-main/src/train_fplc.py`
- `external/VPA-main/src/train_last3.py`
- `external/VPA-main/src/train_last3_lora.py`
- `external/VPA-main/src/train_shards.py`
- `external/VPA-main/src/train_shards_last3_lora.py`
- `external/VPA-main/src/train_worker.py`
- `external/VPA-main/src/train_worker_partial.py`

Documentation with cluster-specific paths:

- `external/VPA-main/AI_SETUP.md`
  - Mentions `/data/mwicker/`, `/data/mwicker/datasets/`, `/data/mwicker/output/`, `/vol/bitbucket/mwicker/antigravity-env`, and `/homes/mwicker`.

Observation: path configuration is not centralized. Most scripts set `HF_HOME` directly and define data/adapter/output constants at module scope.

## 3. Scripts That Currently Produce Final Tool-Name Votes

- `external/VPA-main/src/eval_ensemble.py`
  - Final tool-name votes are produced by generating full completions per shard and applying `extract_predicted_tool(output)`.
  - In-memory shape is effectively `ensemble_predictions[example_idx][shard_idx] = pred_tool`.
  - Majority is computed with `Counter(votes)`.
  - No saved vote artifact.
- `external/VPA-main/src/certify_vpa.py`
  - Uses `tool_name_votes = []`.
  - For each `sname` in `shard_dirs`, it loads the adapter, generates a full completion, extracts `pred_tool`, and appends it.
  - Saves result fields including `majority`, `votes`, and `vote_counts`.
- `external/VPA-main/src/rerun_examples.py`
  - Same final tool-name vote logic as `certify_vpa.py`.
  - Saves result fields including `majority`, `votes`, and `vote_counts`.

Uncertain/minor: `eval_models_prompt.py`, `eval_last3_lora.py`, `quick_eval.py`, and `compare_training_approaches.py` extract or check final tool names for evaluation, but they do not implement ensemble vote export or certification-style vote production.

## 4. Scripts That Currently Do Token-Level Voting

- `external/VPA-main/src/eval_token_ensemble.py`
  - `get_next_token_from_adapter(...)` loads one adapter and returns an argmax next-token id.
  - `token_level_majority_vote(...)` collects `next_token_votes` for all selected shards at each generation step and appends the majority token.
- `external/VPA-main/src/certify_vpa_token.py`
  - Same token-level generation pattern in `token_level_generation(...)`.
  - It then extracts one final tool name from the token-majority sequence.

These scripts do token-level majority decoding, not token vote export.

## 5. Existing True Token Vote Matrix Saving

No existing script appears to save a true token vote matrix.

Concrete observations:

- `eval_token_ensemble.py` has `next_token_votes` inside the token loop, but it is local and discarded after computing `vote_counts`.
- `certify_vpa_token.py` has the same local `next_token_votes` pattern.
- `certify_vpa_token.py` writes `vpa_token_level_results.json`, but only with summary fields: `ground_truth`, `majority`, correctness/safety booleans, `robustness_radius`, and timing.
- `certify_vpa.py` and `rerun_examples.py` save final tool-name `vote_counts`, not token-level votes.

Therefore the repository currently lacks an artifact equivalent to:

- `token_votes[k, example_idx, token_pos]`
- `shard_ids[k]`
- `majority_tokens[example_idx, token_pos]`
- per-token count vectors
- enough metadata to reconstruct token ids, text, prompts, and selected examples

## 6. Data Structures Available During Inference

Current final-vote scripts have these structures available:

- `filtered_data`: list of dictionaries. In `certify_vpa.py`, each item contains:
  - `tools_def`
  - `question`
  - `ground_truth_tool`
  - `available_tools`
- `prompt`: formatted string using `V1_ORIGINAL`.
- `inputs`: tokenizer output moved to CUDA, with at least `input_ids` and attention tensors.
- `shard_dirs`: sorted list of adapter directory names such as `shard_0000`.
- Per shard:
  - `sname`
  - `adapter_path`
  - generated output token ids from `model.generate(...)`
  - decoded completion string `output`
  - extracted final tool name `pred_tool`
- After shard loop:
  - `tool_name_votes`: list of predicted tool names in shard iteration order.
  - `vote_counts`: `Counter(tool_name_votes)`.
  - `majority_tool`, `majority_votes`, correctness/safety flags, radius.

Current token-level scripts have these additional structures available:

- `current_ids`: growing prompt-plus-generated token tensor.
- `next_token_votes`: list of next-token ids, one per selected shard, in shard iteration order.
- `vote_counts`: `Counter(next_token_votes)` for the current generation step.
- `majority_token`: selected token id appended to `current_ids`.
- `output_ids`: final token-majority sequence after all steps.

Not currently retained:

- Per-shard generated token sequences in the token-level scripts.
- Per-shard next-token votes across all positions.
- Logits, probabilities, or rank information.
- Stable mapping from saved result rows back to individual shard votes.

## 7. Shard Identity Preservation

Shard identity is preserved in several internal places:

- Shard data files are named `shard_0000.jsonl`, etc. in `generate_shards.py`.
- Adapter output directories are named with the same shard id in `train_shards.py` and `train_shards_last3_lora.py`.
- Inference loops iterate over `shard_dirs`, where each `sname` is the shard id.
- `eval_ensemble.py` keeps prediction order aligned with `shard_dirs` while the process is running.

Shard identity is not preserved in current saved certification outputs:

- `certify_vpa.py` result JSONL stores only aggregate `vote_counts`.
- `rerun_examples.py` result JSONL stores only aggregate `vote_counts`.
- `certify_vpa_token.py` stores no shard-level token votes and no `shard_ids`.
- Logs print progress counts such as `100/500`, but do not record vote rows by shard id.

Conclusion: the existing adapter and shard directory naming is enough to build a new export layer, but the current saved outputs are insufficient for shard-aware MILPs.

## 8. Existing Utilities That Can Be Reused Safely

Safe to reuse as imported helpers or copied logic in a new additive exporter:

- `external/VPA-main/src/test_filter.py`
  - `extract_filter_features` and `is_solvable` can reproduce the current filtered-example selection.
  - Caution: the script has a `__main__` verification block with hard-coded paths; import only the functions.
- Prompt and parsing conventions from:
  - `external/VPA-main/src/eval_ensemble.py`
  - `external/VPA-main/src/certify_vpa.py`
  - `external/VPA-main/src/certify_vpa_token.py`
  - Useful pieces: `V1_ORIGINAL`, `get_tool_call`, `extract_tool_name`, `extract_tool_from_output`, `extract_available_tools`, and safety predicates.
- Adapter iteration pattern from:
  - `certify_vpa.py` for validated adapter discovery under `adapters_last3_lora`.
  - `eval_token_ensemble.py` / `certify_vpa_token.py` for token-step adapter voting.
- First-party tensor utilities:
  - `toy_certificate.data.compute_counts`
  - `toy_certificate.data.majority_predictions`
  - `toy_certificate.data.runner_up_tokens`
  - `toy_certificate.data.stability_margins`
- First-party downstream methods:
  - `toy_certificate.baselines` functions can operate once real votes are converted to expected `(K, N, L)` and count shapes.
  - `toy_certificate.milp` can operate once real votes, counts, targets, and influence masks are defined.
  - `toy_certificate.csv_io` is safe for tabular benchmark summaries.

Reuse caveat: current VPA helper functions are embedded in scripts with global constants and side effects. For the next stage, prefer importing only pure functions where practical or creating a new standalone exporter with local path/config arguments.

## 9. Parts That Should Not Be Touched

Given the integration goal and the constraint not to edit existing VPA-main files:

- Do not modify `external/VPA-main/src/*` in the next stage unless explicitly approved later.
- Do not change training scripts:
  - `data_loader.py`
  - `generate_shards.py`
  - `train_shards.py`
  - `train_worker.py`
  - `train_shards_last3_lora.py`
  - `train_last3_lora.py`
  - other training variants
- Do not alter existing certification scripts in place:
  - `certify_vpa.py`
  - `certify_vpa_token.py`
  - `rerun_examples.py`
  - `eval_ensemble.py`
  - `eval_token_ensemble.py`
- Do not change historical outputs under `outputs/`.
- Do not retrofit the existing toy MILP code directly into VPA-main outputs before a token vote artifact exists.
- Do not use `generate_shards.py` casually because it deletes and recreates the shard directory.
- Do not run large VPA training/inference scripts during integration validation.

## 10. Proposed Minimal Integration Plan For Stage 1

1. Add a new, additive token vote export script outside `external/VPA-main`, or in a new integration directory, without editing VPA-main files.
2. Make paths/config explicit CLI arguments:
   - test JSONL path
   - adapter base path
   - output artifact path
   - number of examples
   - number of shards
   - max generated tokens
   - model id / cache directory
   - device
3. Reuse the current VPA prompt/filter/parsing semantics so exported examples match existing certification behavior.
4. During token-level generation, save the full vote matrix instead of discarding `next_token_votes`.
   - Required minimum fields:
     - `token_votes`: shape `(K, N, L)` or equivalent serialized ragged representation if sequences stop early.
     - `shard_ids`: ordered list matching axis `K`.
     - `example_ids` or source line indices.
     - prompt metadata: `question`, `tools_def` or hash/path reference, `ground_truth_tool`, available tools.
     - `majority_tokens`, decoded majority output, tokenizer/model identifiers.
     - `max_new_tokens`, EOS behavior, and selection/filter seed.
5. Prefer a compact numeric artifact such as `.npz` for arrays plus `.json` metadata, or a single structured JSONL only for very small smoke exports.
6. Add a tiny smoke mode first, for example 1-2 examples and 2-3 shards, to validate artifact shape without expensive inference.
7. Add a separate adapter layer that loads the saved artifact and constructs the arrays expected by `toy_certificate`:
   - `votes` / `val_votes` with shape `(K, N, L)`
   - counts with shape `(N, L, T)` or a compact remapped token vocabulary
   - `clean_pred`, `target`, and `influence`
8. Only after artifact export is validated, run existing stability/validity baselines and MILPs from the saved artifact. Do not call MILPs from inside the VPA inference loop.

Uncertain point: the exact validity target representation for real VPA token artifacts is not yet defined in the repo. The toy code expects harmful target token arrays, but VPA-main currently filters and evaluates ground-truth tool calls. Stage 1 should define whether `target` means ground-truth tool-call tokens, an unsafe target sequence, or a separate harmful-prefix target before running validity MILPs.
