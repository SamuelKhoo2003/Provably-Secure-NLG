"""Export token-level vote artifacts through a backend abstraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from .backends import TokenVoteBackend, VoteRequest, make_backend
from .io import write_jsonl
from .metadata import build_export_metadata, metadata_path_for_output, write_metadata
from .safety import safety_metadata, validate_configured_paths
from .schemas import StabilityVoteRow, ValidityVoteRow
from .vpa_backend import VPAAdapterBackend


def export_stability_votes(
    *,
    output: str | Path,
    backend_name: str,
    num_examples: int,
    num_positions: int,
    num_shards: int,
    seed: int = 17,
    backend: TokenVoteBackend | None = None,
    metadata_extra: dict[str, object] | None = None,
    cluster_username: str | None = None,
    real_inference_enabled: bool = False,
) -> Path:
    """Export stability rows using clean sequential majority-prefix decoding."""

    _validate_dimensions(num_examples, num_positions, num_shards)
    output_path = Path(output)
    backend = backend or make_backend(backend_name, seed=seed)
    shard_ids = _select_shard_ids(backend, num_shards)
    rows: list[StabilityVoteRow] = []

    for example_idx in range(num_examples):
        example_id = f"export_example_{example_idx:04d}"
        prefix_token_ids = [101, 10_000 + example_idx]
        for position in range(num_positions):
            request = VoteRequest(
                mode="stability",
                example_id=example_id,
                position=position,
                prefix_token_ids=list(prefix_token_ids),
            )
            shard_token_ids = backend.predict_next_token_for_shards(request, shard_ids)
            row = StabilityVoteRow.from_shard_votes(
                example_id=example_id,
                position=position,
                prefix_token_ids=list(prefix_token_ids),
                shard_ids=shard_ids,
                shard_token_ids=shard_token_ids,
            )
            rows.append(row)
            # Stability follows the clean autoregressive rollout: the voted
            # majority token becomes part of the next clean prefix.
            prefix_token_ids.append(row.majority_token_id)

    write_jsonl(output_path, rows)
    notes = (
        "Real VPA stability smoke export. Majority tokens are committed to the next clean prefix."
        if real_inference_enabled
        else "Mock stability export; no real model inference. Majority tokens are committed to the next clean prefix."
    )
    _write_sidecar(
        output_path=output_path,
        mode="stability",
        backend=backend.name,
        num_examples=num_examples,
        num_positions=num_positions,
        num_shards=num_shards,
        shard_ids=shard_ids,
        notes=notes,
        cluster_username=cluster_username,
        real_inference_enabled=real_inference_enabled,
        extra=metadata_extra,
    )
    return output_path


def export_validity_votes(
    *,
    output: str | Path,
    backend_name: str,
    num_examples: int,
    num_positions: int,
    num_shards: int,
    seed: int = 17,
    backend: TokenVoteBackend | None = None,
    metadata_extra: dict[str, object] | None = None,
    cluster_username: str | None = None,
    real_inference_enabled: bool = False,
) -> Path:
    """Export validity rows using target-prefix decoding."""

    _validate_dimensions(num_examples, num_positions, num_shards)
    output_path = Path(output)
    backend = backend or make_backend(backend_name, seed=seed)
    shard_ids = _select_shard_ids(backend, num_shards)
    rows: list[ValidityVoteRow] = []

    for example_idx in range(num_examples):
        example_id = f"export_example_{example_idx:04d}"
        target_id = "export_target_00"
        target_prefix_token_ids = [501, 60_000 + example_idx]
        for position in range(num_positions):
            target_token_id = 80_000 + example_idx * 100 + position
            request = VoteRequest(
                mode="validity",
                example_id=example_id,
                target_id=target_id,
                position=position,
                prefix_token_ids=list(target_prefix_token_ids),
                target_token_id=target_token_id,
            )
            shard_token_ids = backend.predict_next_token_for_shards(request, shard_ids)
            row = ValidityVoteRow.from_shard_votes(
                example_id=example_id,
                target_id=target_id,
                position=position,
                target_prefix_token_ids=list(target_prefix_token_ids),
                target_token_id=target_token_id,
                shard_ids=shard_ids,
                shard_token_ids=shard_token_ids,
            )
            rows.append(row)
            # Validity does not follow the clean majority prefix. It rolls out
            # the specified target sequence so every position is evaluated under
            # the adversarial/target prefix.
            target_prefix_token_ids.append(target_token_id)

    write_jsonl(output_path, rows)
    notes = (
        "Real VPA validity smoke export. Prefixes are extended with target tokens, not clean majority tokens."
        if real_inference_enabled
        else "Mock validity export; no real model inference. Prefixes are extended with target tokens, not clean majority tokens."
    )
    _write_sidecar(
        output_path=output_path,
        mode="validity",
        backend=backend.name,
        num_examples=num_examples,
        num_positions=num_positions,
        num_shards=num_shards,
        shard_ids=shard_ids,
        notes=notes,
        cluster_username=cluster_username,
        real_inference_enabled=real_inference_enabled,
        extra=metadata_extra,
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export token-level VPA vote artifacts.")
    parser.add_argument("--backend", choices=["mock", "vpa"], required=True)
    parser.add_argument("--mode", choices=["stability", "validity"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--num-positions", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--enable-real-inference", action="store_true")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--cluster-username", type=str, default=None)
    args = parser.parse_args(argv)

    if args.backend == "vpa":
        setup = _build_vpa_backend_from_args(args)
        if isinstance(setup, str):
            print(setup)
            return 2
        backend, metadata_extra = setup
    else:
        backend = None
        metadata_extra = None

    try:
        if args.mode == "stability":
            path = export_stability_votes(
                output=args.output,
                backend_name=args.backend,
                num_examples=args.num_examples,
                num_positions=args.num_positions,
                num_shards=args.num_shards,
                seed=args.seed,
                backend=backend,
                metadata_extra=metadata_extra,
                cluster_username=args.cluster_username,
                real_inference_enabled=args.enable_real_inference and args.backend == "vpa",
            )
        else:
            path = export_validity_votes(
                output=args.output,
                backend_name=args.backend,
                num_examples=args.num_examples,
                num_positions=args.num_positions,
                num_shards=args.num_shards,
                seed=args.seed,
                backend=backend,
                metadata_extra=metadata_extra,
                cluster_username=args.cluster_username,
                real_inference_enabled=args.enable_real_inference and args.backend == "vpa",
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Export failed: {exc}")
        return 1

    print(f"Wrote {path}")
    print(f"Wrote {metadata_path_for_output(path)}")
    return 0


def _write_sidecar(
    *,
    output_path: Path,
    mode: str,
    backend: str,
    num_examples: int,
    num_positions: int,
    num_shards: int,
    shard_ids: list[str],
    notes: str,
    cluster_username: str | None,
    real_inference_enabled: bool,
    extra: dict[str, object] | None,
) -> None:
    metadata = build_export_metadata(
        mode=mode,
        backend=backend,
        num_examples=num_examples,
        num_positions=num_positions,
        num_shards=num_shards,
        shard_ids=shard_ids,
        output_path=output_path,
        notes=notes,
        safety=safety_metadata(cluster_username=cluster_username, real_inference_enabled=real_inference_enabled),
        extra=extra,
    )
    write_metadata(metadata_path_for_output(output_path), metadata)


def _validate_dimensions(num_examples: int, num_positions: int, num_shards: int) -> None:
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")
    if num_positions < 1:
        raise ValueError("num_positions must be at least 1")
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")


def _shard_ids(num_shards: int) -> list[str]:
    return [f"shard_{idx:04d}" for idx in range(num_shards)]


def _select_shard_ids(backend: TokenVoteBackend, num_shards: int) -> list[str]:
    if hasattr(backend, "discover_shards"):
        discovered = backend.discover_shards()
        if len(discovered) < num_shards:
            raise FileNotFoundError(f"Requested {num_shards} shards but found {len(discovered)} adapter shard directories")
        return discovered[:num_shards]
    return _shard_ids(num_shards)


def _build_vpa_backend_from_args(args: argparse.Namespace) -> tuple[VPAAdapterBackend, dict[str, object]] | str:
    if not args.enable_real_inference:
        return "Real VPA inference requires --enable-real-inference. Use --backend mock or run discovery first."
    if args.num_examples != 1 or args.num_positions != 1 or args.num_shards != 1:
        return "Stage 6 real VPA smoke export is limited to --num-examples 1 --num-positions 1 --num-shards 1."
    if args.mode != "stability":
        return "Stage 6 real VPA smoke export supports stability mode only."
    if args.adapter_dir is None:
        return "--backend vpa requires --adapter-dir."
    if args.model_name is None:
        return "--backend vpa requires --model-name."
    if not args.adapter_dir.exists():
        return f"Adapter directory does not exist: {args.adapter_dir}"

    safety_report = validate_configured_paths(
        adapter_dir=args.adapter_dir,
        output_dir=args.output.parent,
        cluster_username=args.cluster_username,
    )
    if safety_report.has_errors:
        return "Safety validation failed: " + "; ".join(message.message for message in safety_report.errors)

    backend = VPAAdapterBackend(
        adapter_dir=args.adapter_dir,
        model_name=args.model_name,
        output_dir=args.output.parent,
        cluster_username=args.cluster_username,
        device=args.device,
        dtype=args.dtype,
        real_inference_enabled=True,
    )
    metadata_extra = {
        "safety_report": safety_report.as_dict(),
    }
    metadata_extra.update(backend.runtime_metadata())
    return backend, metadata_extra


if __name__ == "__main__":
    raise SystemExit(main())
