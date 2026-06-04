"""Lightweight VPA path and shard discovery without model loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from large_experiments.storage import resolve_large_output_path

from .metadata import write_metadata
from .safety import validate_configured_paths


def discover_vpa(
    *,
    adapter_dir: str | Path,
    test_path: str | Path,
    num_shards: int,
    output_dir: str | Path,
    cluster_username: str | None = None,
) -> dict[str, Any]:
    """Inspect configured VPA paths without importing model dependencies."""

    adapter_path = Path(adapter_dir)
    test_file = Path(test_path)
    output_path = resolve_large_output_path(output_dir)
    shard_ids = _discover_shard_ids(adapter_path)
    safety_report = validate_configured_paths(
        adapter_dir=adapter_path,
        dataset_dir=test_file.parent,
        output_dir=output_path,
        cluster_username=cluster_username,
    )
    return {
        "adapter_dir": str(adapter_path),
        "adapter_dir_exists": adapter_path.exists(),
        "num_shard_adapter_dirs_found": len(shard_ids),
        "first_shard_ids": shard_ids[: min(num_shards, 10)],
        "test_path": str(test_file),
        "test_path_exists": test_file.exists(),
        "num_test_examples": _count_jsonl_lines(test_file) if test_file.exists() else None,
        "chosen_num_shards": num_shards,
        "output_dir": str(output_path),
        "output_dir_exists": output_path.exists(),
        "safety": safety_report.as_dict(),
        "real_model_loading_attempted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover VPA adapter/test paths without model loading.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster-username", type=str, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None, help="Optional JSON file for discovery metadata")
    args = parser.parse_args(argv)
    args.output_dir = resolve_large_output_path(args.output_dir)
    if args.metadata_output is not None:
        args.metadata_output = resolve_large_output_path(args.metadata_output)

    result = discover_vpa(
        adapter_dir=args.adapter_dir,
        test_path=args.test_path,
        num_shards=args.num_shards,
        output_dir=args.output_dir,
        cluster_username=args.cluster_username,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.metadata_output is not None:
        write_metadata(args.metadata_output, result)
        print(f"Wrote {args.metadata_output}")
    return 1 if result["safety"]["messages"] and any(message["level"] == "error" for message in result["safety"]["messages"]) else 0


def _discover_shard_ids(adapter_dir: Path) -> list[str]:
    if not adapter_dir.exists():
        return []
    return sorted(path.name for path in adapter_dir.iterdir() if path.is_dir() and path.name.startswith("shard_"))


def _count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


if __name__ == "__main__":
    raise SystemExit(main())
