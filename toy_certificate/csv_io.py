"""CSV read/write helpers for benchmark outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write heterogeneous row dictionaries while preserving first-seen order."""
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: Path) -> list[dict[str, object]]:
    """Read benchmark rows."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                parsed[key] = parse_csv_value(key, value)
            rows.append(parsed)
    return rows


def read_optional_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        print(f"Warning: optional CSV not found, skipping: {path}")
        return []
    return read_rows_csv(path)


def parse_csv_value(key: str, value: str | None) -> object:
    if value is None or value == "":
        return "" if value == "" else value
    if value == "True":
        return True
    if value == "False":
        return False
    if key in {
        "K",
        "N",
        "L",
        "T",
        "seed",
        "budget",
        "num_certified",
        "num_known",
        "num_unknown",
        "num_total",
        "max_attacked_rows",
        "max_attacked_cells",
    }:
        return int(float(value))
    if looks_numeric(value):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    return value


def looks_numeric(value: object) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))
