# VPA Integration

This package is the integration boundary between the external VPA scaffold and
the certificate code.

`external/VPA-main` remains an external scaffold. Its role is to train shard
LoRA adapters and, later, run those adapters for token-level inference. This
package will provide the artifact schema and lightweight utilities needed to
export those token votes without coupling certification code to model runtime
code.

The intended flow is:

1. `external/VPA-main` trains shard adapters.
2. A future exporter runs shard adapters and saves token vote artifacts.
3. Stability and validity baselines read the saved artifacts.
4. Full MILPs read the saved artifacts.

Certification should run from saved vote artifacts rather than directly inside
VPA inference scripts. This keeps expensive model execution separate from
certificate computation, makes results auditable, and allows baselines and MILPs
to be rerun without regenerating model outputs.

Pointwise DPA and TPA baselines only need token count vectors. Full MILPs need
more: they require the identity of the shard behind every token vote. For that
reason the JSONL schema stores both `shard_ids` and `shard_token_ids`; aggregate
`vote_counts` alone are not enough.

JSONL is used first for debuggability. It is easy to inspect, diff, and produce
in small smoke runs. A compact format such as NPZ or Parquet can be added later
after the schema is validated against real exports.

The default package path remains lightweight. Real-inference imports such as
`torch` and `transformers` are isolated inside explicit runtime methods in
`vpa_integration/vpa_backend.py`; discovery, validation, schema, metadata, and
mock export paths remain model-free.

## Mock Artifacts

Stage 2 adds deterministic mock artifacts for validating the schema before any
real adapter inference is wired in:

```bash
python -m vpa_integration.generate_mock_votes --output-dir outputs/mock
python -m vpa_integration.validate_votes outputs/mock/stability_votes.jsonl
python -m vpa_integration.validate_votes outputs/mock/validity_votes.jsonl
```

The validator recomputes `vote_counts` from `shard_token_ids` and verifies the
saved `majority_token_id` using the same first-occurrence shard-order tie rule
as the schema helper. This is intentional: inconsistent aggregate counts must be
caught before certificate code consumes the artifact.

## Exporter Scaffold

Stage 3 adds an exporter scaffold with a backend abstraction. It still uses only
a mock backend:

```bash
python -m vpa_integration.export_votes --backend mock --mode stability --output outputs/export_mock/stability_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
python -m vpa_integration.export_votes --backend mock --mode validity --output outputs/export_mock/validity_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
```

Stability mode models clean sequential voting: for each example, the exporter
asks every shard for the next token, writes a `StabilityVoteRow`, commits the
majority token to the prefix, and continues to the next position.

Validity mode models target-prefix voting: for each example and target, the
exporter asks every shard for the next token under the target prefix, writes a
`ValidityVoteRow`, then extends the prefix with the known target token. It does
not use the clean majority prefix.

Every export writes a sidecar metadata file next to the JSONL artifact, such as
`stability_votes.meta.json` or `validity_votes.meta.json`.

## Stage 4 Safety And VPA Discovery

The real VPA backend is scaffolded but not enabled for inference. Stage 4 adds
path discovery and safety checks without loading models:

```bash
python -m vpa_integration.discover_vpa --adapter-dir /data/<username>/output/adapters_last3_lora --test-path /data/<username>/VPA/data/test.jsonl --num-shards 3 --output-dir /data/<username>/output/vpa_integration_smoke --cluster-username <username>
```

Cluster path rules are generalized through configuration. Usernames and virtual
environment names must be supplied or inferred; no professor-specific username
or environment path is hard-coded. The safety module enforces sequential
execution metadata with `MAX_CONCURRENT_JOBS = 1` and
`CONCURRENCY_MODE = "sequential"`, forbids configured paths under `/homes/`, and
warns when local development paths are used instead of configured `/data` roots.

`--backend vpa` is recognized by the exporter only to fail clearly. Real adapter
inference remains disabled until a later stage.

## Stage 5 Real Smoke Backend

Stage 5 adds a tiny real-inference path behind an explicit opt-in flag:

```bash
python -m vpa_integration.export_votes \
  --backend vpa \
  --enable-real-inference \
  --mode stability \
  --adapter-dir /data/<username>/output/adapters_last3_lora \
  --model-name allenai/OLMo-2-0425-1B-Instruct \
  --output /data/<username>/output/vpa_integration_smoke/stability_votes.jsonl \
  --num-examples 1 \
  --num-positions 1 \
  --num-shards 1 \
  --cluster-username <username>
```

The real path is restricted to one example, one position, and one shard in this
stage. It computes a single next-token argmax from the supplied prefix for one
adapter at a time. It does not use `model.generate`, training, multiprocessing,
thread pools, process pools, distributed execution, or adapter batching.

## Stage 6 Adapter Compatibility

The real backend reports `adapter_strategy = "transformers_adapter_methods"`.
Before loading a shard adapter it checks that the loaded model exposes:

- `load_adapter`
- `set_adapter`
- `delete_adapter`

If any method is unavailable, the backend raises a clear compatibility error
instead of silently assuming the runtime supports the expected adapter API. PEFT
is not imported directly in this stage; if a later runtime needs
`PeftModel.from_pretrained`, that strategy should be added explicitly and kept
inside `vpa_integration/vpa_backend.py`.
