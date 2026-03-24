from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import yaml

from ilp_selector import select_poison_indices


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_ids(rows: List[Dict]) -> None:
    for i, row in enumerate(rows):
        row.setdefault("id", f"sample-{i}")


def add_dummy_scores(rows: List[Dict], score_field: str, seed: int) -> None:
    rng = random.Random(seed)
    for row in rows:
        if score_field not in row:
            row[score_field] = rng.random()


def poison_rows(rows: List[Dict], selected_ids: set[str]) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        row = dict(r)
        row["is_poison"] = row["id"] in selected_ids
        out.append(row)
    return out


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = Path(cfg["paths"]["raw_jsonl"])
    output_dir = Path(cfg["paths"]["output_dir"])
    rates = cfg["poison_rates"]
    score_field = cfg.get("score_field", "score")
    score_seed = int(cfg.get("random_score_seed", 123))
    max_samples = int(cfg.get("max_samples", 0))

    rows = read_jsonl(raw_path)
    if max_samples > 0:
        rows = rows[:max_samples]

    add_ids(rows)
    add_dummy_scores(rows, score_field, score_seed)

    clean_path = output_dir / "clean.jsonl"
    write_jsonl(clean_path, poison_rows(rows, set()))

    manifest = {
        "n_rows": len(rows),
        "rates": rates,
        "files": {"clean": str(clean_path)},
    }

    scores = [float(r[score_field]) for r in rows]
    ids = [str(r["id"]) for r in rows]

    for rate in rates:
        budget = math.floor(len(rows) * float(rate))

        rng = random.Random(1000 + int(rate * 1_000_000))
        rand_indices = list(range(len(rows)))
        rng.shuffle(rand_indices)
        rand_indices = rand_indices[:budget]
        rand_ids = {ids[i] for i in rand_indices}

        random_path = output_dir / f"poison_random_{rate}.jsonl"
        write_jsonl(random_path, poison_rows(rows, rand_ids))

        ilp = select_poison_indices(scores=scores, budget=budget)
        ilp_ids = {ids[i] for i in ilp.selected_indices}

        ilp_path = output_dir / f"poison_ilp_{rate}.jsonl"
        write_jsonl(ilp_path, poison_rows(rows, ilp_ids))

        manifest["files"][f"random_{rate}"] = str(random_path)
        manifest["files"][f"ilp_{rate}"] = str(ilp_path)
        manifest[f"ilp_objective_{rate}"] = ilp.objective_value

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
