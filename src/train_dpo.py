from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    model_name = cfg["model"]["name"]
    use_lora = bool(cfg["model"].get("use_lora", True))

    train_jsonl = cfg["data"]["train_jsonl"]
    prompt_col = cfg["data"].get("prompt_col", "prompt")
    chosen_col = cfg["data"].get("chosen_col", "chosen")
    rejected_col = cfg["data"].get("rejected_col", "rejected")

    output_dir = cfg["training"]["output_dir"]

    dataset = load_dataset("json", data_files=train_jsonl, split="train")
    required_cols = {prompt_col, chosen_col, rejected_col}
    missing = required_cols - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    # Normalize to canonical column names expected by DPOTrainer.
    if prompt_col != "prompt":
        dataset = dataset.rename_column(prompt_col, "prompt")
    if chosen_col != "chosen":
        dataset = dataset.rename_column(chosen_col, "chosen")
    if rejected_col != "rejected":
        dataset = dataset.rename_column(rejected_col, "rejected")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name) if not use_lora else None

    if use_lora:
        peft_cfg = LoraConfig(
            r=int(cfg["model"].get("lora_r", 16)),
            lora_alpha=int(cfg["model"].get("lora_alpha", 32)),
            lora_dropout=float(cfg["model"].get("lora_dropout", 0.05)),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    tcfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(tcfg.get("learning_rate", 5e-6)),
        num_train_epochs=float(tcfg.get("num_train_epochs", 1)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 200)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        fp16=bool(tcfg.get("fp16", False)),
        bf16=bool(tcfg.get("bf16", False)),
        report_to=[],
    )

    dpo_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": training_args,
        "train_dataset": dataset,
        "beta": float(cfg["dpo"].get("beta", 0.1)),
        "max_length": int(tcfg.get("max_length", 1024)),
        "max_prompt_length": int(tcfg.get("max_prompt_length", 512)),
    }

    # TRL API changed `tokenizer` -> `processing_class` in newer releases.
    sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in sig.parameters:
        dpo_kwargs["processing_class"] = tokenizer
    else:
        dpo_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**dpo_kwargs)
    train_result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_path = Path(output_dir) / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)

    print(json.dumps({"status": "ok", "output_dir": output_dir}, indent=2))


if __name__ == "__main__":
    main()
