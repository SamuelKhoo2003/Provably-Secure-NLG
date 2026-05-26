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
    """Read benchmark rows and normalize legacy column aliases."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                parsed[key] = parse_csv_value(key, value)
            copy_legacy_csv_columns(parsed)
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


def copy_legacy_csv_columns(row: dict[str, object]) -> None:
    legacy_map = {
        "row_col_stability_any_cell": "row_col_stab_q1_r1",
        "row_col_stability_full_row": "row_col_stab_q1_rL",
        "row_col_validity_q1": "row_col_val_q1",
        "row_col_validity_qN": "row_col_val_qN",
        "naive_dpa_stability_full_row": "independent_stab_full_row_q1",
        "naive_dpa_validity_q1": "independent_val_q1",
        "naive_dpa_validity_qN": "independent_val_qN",
        "phd_ref_stability_any_cell": "raw_dpa_stab_min_cell",
        "phd_ref_validity_any_cell": "raw_dpa_val_min_cell",
        "independent_stab_qN_rL": "independent_stab_full_row_qN",
        "independent_val_q1": "independent_val_sequence_q1",
        "independent_val_qN": "independent_val_sequence_qN",
    }
    for old_key, new_key in legacy_map.items():
        if old_key in row and new_key not in row:
            row[new_key] = row[old_key]
    if "raw_dpa_stab_min_cell" in row:
        row.setdefault("dpa_stab_cell_min", row["raw_dpa_stab_min_cell"])
        row.setdefault("dpa_stab_row_radius_q1", row["raw_dpa_stab_min_cell"])
    if "raw_dpa_val_min_cell" in row:
        row.setdefault("dpa_val_cell_min", row["raw_dpa_val_min_cell"])
        row.setdefault("dpa_val_row_weak_q1", row["raw_dpa_val_min_cell"])
    if "certified_fraction_full_horizon" in row:
        row.setdefault("full_horizon_certified_fraction", row["certified_fraction_full_horizon"])
    if "full_horizon_certified_fraction" in row:
        row.setdefault("certified_fraction_full_horizon", row["full_horizon_certified_fraction"])


def looks_numeric(value: object) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))
