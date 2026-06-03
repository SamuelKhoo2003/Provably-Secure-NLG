"""Metadata sidecar helpers for exported vote artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vpa-token-votes/v1"


def metadata_path_for_output(output_path: str | Path) -> Path:
    """Return the sidecar path for a JSONL output path."""

    path = Path(output_path)
    if path.suffix == ".jsonl":
        return path.with_suffix(".meta.json")
    return path.with_name(f"{path.name}.meta.json")


def build_export_metadata(
    *,
    mode: str,
    backend: str,
    num_examples: int,
    num_positions: int,
    num_shards: int,
    shard_ids: list[str],
    output_path: str | Path,
    notes: str,
    safety: dict[str, object] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable metadata dictionary."""

    metadata = {
        "mode": mode,
        "backend": backend,
        "num_examples": num_examples,
        "num_positions": num_positions,
        "num_shards": num_shards,
        "shard_ids": list(shard_ids),
        "selected_shard_ids": list(shard_ids),
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "notes": notes,
    }
    if safety is not None:
        metadata["safety"] = safety
    if extra is not None:
        metadata.update(extra)
    return metadata


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    """Write metadata as formatted JSON."""

    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
