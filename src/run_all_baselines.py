from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def build_arms(manifest: Dict[str, Any], rate: str) -> List[Tuple[str, str]]:
    files = manifest.get("files", {})
    clean = files.get("clean")
    random_split = files.get(f"random_{rate}")
    ilp_split = files.get(f"ilp_{rate}")

    missing = [
        name
        for name, value in [
            ("clean", clean),
            (f"random_{rate}", random_split),
            (f"ilp_{rate}", ilp_split),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Manifest missing required entries: {missing}")

    return [("clean", clean), (f"random_{rate}", random_split), (f"ilp_{rate}", ilp_split)]


def run_train(train_script: Path, config_path: Path) -> None:
    cmd = [sys.executable, str(train_script), "--config", str(config_path)]
    subprocess.run(cmd, check=True)


def read_metrics(output_dir: Path) -> Dict[str, Any]:
    metrics_path = output_dir / "train_metrics.json"
    if not metrics_path.exists():
        return {"metrics_path": str(metrics_path), "metrics_found": False}

    payload = load_json(metrics_path)
    payload["metrics_path"] = str(metrics_path)
    payload["metrics_found"] = True
    return payload


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "arm",
        "split",
        "output_dir",
        "metrics_found",
        "train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "epoch",
        "global_step",
        "metrics_path",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("configs/dpo.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    parser.add_argument("--rate", type=str, default="0.01")
    parser.add_argument("--summary-csv", type=Path, default=Path("results/baseline_comparison.csv"))
    parser.add_argument("--work-dir", type=Path, default=Path("results/.tmp_configs"))
    parser.add_argument("--run-prefix", type=str, default="baseline")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    manifest = load_json(args.manifest)

    train_script = Path(__file__).parent / "train_dpo.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing training script: {train_script}")

    arms = build_arms(manifest, args.rate)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_rows: List[Dict[str, Any]] = []

    for arm_name, split_path in arms:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["data"]["train_jsonl"] = split_path

        base_out = str(base_cfg["training"].get("output_dir", "results/dpo"))
        run_out = Path(base_out) / f"{args.run_prefix}_{stamp}_{arm_name}"
        cfg["training"]["output_dir"] = str(run_out)

        arm_cfg_path = args.work_dir / f"dpo_{arm_name}_{args.rate}.yaml"
        write_yaml(arm_cfg_path, cfg)

        print(json.dumps({"status": "running", "arm": arm_name, "config": str(arm_cfg_path)}, indent=2))
        run_train(train_script=train_script, config_path=arm_cfg_path)

        metrics = read_metrics(run_out)
        row: Dict[str, Any] = {
            "arm": arm_name,
            "split": split_path,
            "output_dir": str(run_out),
        }
        row.update(metrics)
        summary_rows.append(row)

    write_summary_csv(args.summary_csv, summary_rows)
    print(json.dumps({"status": "ok", "summary_csv": str(args.summary_csv)}, indent=2))


if __name__ == "__main__":
    main()
