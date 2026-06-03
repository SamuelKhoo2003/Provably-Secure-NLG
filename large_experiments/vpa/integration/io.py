"""Small JSONL helpers for vote artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any] | object]) -> None:
    """Write rows to a JSONL file, replacing any existing contents."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_row_to_dict(row), sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any] | object) -> None:
    """Append one row to a JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_row_to_dict(row), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _row_to_dict(row: dict[str, Any] | object) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if is_dataclass(row):
        return asdict(row)
    if hasattr(row, "to_dict"):
        converted = row.to_dict()
        if isinstance(converted, dict):
            return converted
    raise TypeError("JSONL rows must be dictionaries, dataclasses, or expose to_dict()")
