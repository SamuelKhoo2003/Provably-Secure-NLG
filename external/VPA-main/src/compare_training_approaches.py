#!/usr/bin/env python3
"""
Compare Training Approaches for VPA
====================================
Compares three approaches on shard_0000 data:
1. Zero-shot (base model, no fine-tuning)
2. Full-layer LoRA (all 16 layers trained)
3. Last-3-layer LoRA (layers 13-15 only)

Metrics: Accuracy on tool name extraction from shard_0000 test samples.
"""

import json
import torch
import os
import time
import re
import argparse
from tqdm import tqdm

os.environ["HF_HOME"] = "/data/mwicker/VPA/cache/huggingface"

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"


def format_prompt(item):
    """Format input prompt for model inference."""
    tools_def = item['tools']
    question = item['question']
    
    prompt = (
        "System: Only respond with the formatted tool string and parameters for the correct tool use. "
        "You may select from the following tools:\n"
        f"{tools_def}\n\n"
        f"User: {question}\n\n"
        "Assistant:"
    )
    return prompt


def get_ground_truth_tool(item):
    """Extract the ground truth tool name from messages."""
    msgs = item.get('messages')
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    
    for m in msgs:
        if m.get('role') == 'tool_call':
            content = m.get('content', '')
            # Extract tool name from format like "tool_name(params)"
            match = re.match(r'^(\w+)\s*\(', content)
            if match:
                return match.group(1)
    return None


def extract_tool_name(generation):
    """Extract tool name from model generation."""
    match = re.match(r'^(\w+)\s*\(', generation.strip())
    if match:
        return match.group(1)
    return None


def evaluate_model(model, tokenizer, test_data, model_name, device="cuda"):
    """Evaluate a model on test data and return accuracy."""
    correct = 0
    total = 0
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    for item in tqdm(test_data, desc=model_name):
        prompt = format_prompt(item)
        gt_tool = get_ground_truth_tool(item)
        
        if not gt_tool:
            continue
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generation = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred_tool = extract_tool_name(generation)
        
        if pred_tool == gt_tool:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Result: {correct}/{total} = {accuracy:.1%} accuracy")
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(description="Compare training approaches for VPA")
    parser.add_argument("--shard", type=str, default="data/shards/shard_0000.jsonl", help="Shard data file")
    parser.add_argument("--full_adapter", type=str, default="output/adapters/shard_0000", help="Full-layer adapter path")
    parser.add_argument("--partial_adapter", type=str, default="output/adapters_partial/shard_0000", help="Partial-layer adapter path")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("VPA Training Approach Comparison")
    print("=" * 60)
    
    # Load test data
    print(f"\nLoading data from {args.shard}...")
    test_data = []
    with open(args.shard, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Use subset for faster evaluation
    test_data = test_data[:args.num_samples]
    print(f"Using {len(test_data)} samples for evaluation")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    # 1. Zero-shot evaluation
    print("\n[1/3] Loading base model for zero-shot evaluation...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    base_model.to("cuda")
    base_model.eval()
    
    acc, correct, total = evaluate_model(base_model, tokenizer, test_data, "Zero-Shot")
    results["Zero-Shot"] = {"accuracy": acc, "correct": correct, "total": total}
    
    del base_model
    torch.cuda.empty_cache()
    
    # 2. Full-layer LoRA evaluation
    if os.path.exists(args.full_adapter):
        print(f"\n[2/3] Loading full-layer adapter from {args.full_adapter}...")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        full_model = PeftModel.from_pretrained(base_model, args.full_adapter)
        full_model.to("cuda")
        full_model.eval()
        
        acc, correct, total = evaluate_model(full_model, tokenizer, test_data, "Full-Layer LoRA (16 layers)")
        results["Full-Layer LoRA"] = {"accuracy": acc, "correct": correct, "total": total}
        
        del full_model, base_model
        torch.cuda.empty_cache()
    else:
        print(f"\n[2/3] Skipping full-layer adapter (not found at {args.full_adapter})")
        results["Full-Layer LoRA"] = None
    
    # 3. Partial-layer LoRA evaluation
    if os.path.exists(args.partial_adapter):
        print(f"\n[3/3] Loading partial-layer adapter from {args.partial_adapter}...")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        partial_model = PeftModel.from_pretrained(base_model, args.partial_adapter)
        partial_model.to("cuda")
        partial_model.eval()
        
        acc, correct, total = evaluate_model(partial_model, tokenizer, test_data, "Last-3-Layer LoRA (layers 13-15)")
        results["Last-3-Layer LoRA"] = {"accuracy": acc, "correct": correct, "total": total}
        
        del partial_model, base_model
        torch.cuda.empty_cache()
    else:
        print(f"\n[3/3] Skipping partial-layer adapter (not found at {args.partial_adapter})")
        results["Last-3-Layer LoRA"] = None
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Approach':<30} {'Accuracy':<15} {'Correct/Total':<15}")
    print("-" * 60)
    for name, res in results.items():
        if res is not None:
            print(f"{name:<30} {res['accuracy']:.1%}{'':>10} {res['correct']}/{res['total']}")
        else:
            print(f"{name:<30} {'N/A (not trained)':<15}")
    print("=" * 60)


if __name__ == "__main__":
    main()
