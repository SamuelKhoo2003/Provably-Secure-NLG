#!/usr/bin/env python3
"""
Benchmark: Zero-Shot vs Full-Train vs Partial-Train
====================================================
Compares accuracy and timing for three training configurations on shard_000.
"""

import json
import torch
import os
import time
import subprocess
import sys
os.environ["HF_HOME"] = "/data/mwicker/VPA/cache/huggingface"

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
SHARD_FILE = "/data/mwicker/VPA/data/shards/shard_0000.jsonl"

# Output directories
FULL_ADAPTER_DIR = "/data/mwicker/VPA/output/benchmark/full_train"
PARTIAL_ADAPTER_DIR = "/data/mwicker/VPA/output/benchmark/partial_train"

def load_shard_data(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_prompt(item):
    tools_def = item['tools']
    question = item['question']
    prompt = "System: Only respond with the formatted tool string and parameters for the correct tool use. You may select from the following tools:\n"
    prompt += tools_def + "\n\n"
    prompt += "User: " + question + "\n\n"
    prompt += "Assistant:"
    return prompt

def get_target(item):
    msgs = item.get('messages')
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    for m in msgs:
        if m.get('role') == 'tool_call':
            return m.get('content')
    return ""

def evaluate_model(model, tokenizer, data, label="Model"):
    """Evaluate accuracy on tool name extraction."""
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for item in data[:20]:  # Evaluate on 20 samples for speed
            prompt = format_prompt(item)
            target = get_target(item)
            if not target:
                continue
                
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Check if tool name matches (first word/function name)
            target_tool = target.split('(')[0].strip() if '(' in target else target.split()[0]
            gen_tool = generated.split('(')[0].strip() if '(' in generated else generated.split()[0] if generated.split() else ""
            
            if target_tool.lower() == gen_tool.lower():
                correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"[{label}] Accuracy: {correct}/{total} = {acc:.1%}")
    return acc

def main():
    print("=" * 60)
    print("VPA Training Benchmark: Zero-Shot vs Full-Train vs Partial-Train")
    print("=" * 60)
    
    # Load data
    data = load_shard_data(SHARD_FILE)
    print(f"Loaded {len(data)} samples from shard_000")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    # ==================== ZERO-SHOT ====================
    print("\n" + "=" * 40)
    print("1. ZERO-SHOT (Base Model)")
    print("=" * 40)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to("cuda")
    
    results['zero_shot'] = {
        'accuracy': evaluate_model(model, tokenizer, data, "Zero-Shot"),
        'train_time': 0.0
    }
    
    del model
    torch.cuda.empty_cache()
    
    # ==================== FULL TRAINING ====================
    print("\n" + "=" * 40)
    print("2. FULL TRAINING (All 16 layers)")
    print("=" * 40)
    
    os.makedirs(FULL_ADAPTER_DIR, exist_ok=True)
    
    start = time.time()
    result = subprocess.run([
        sys.executable, "/data/mwicker/VPA/src/train_worker.py",
        "--input_file", SHARD_FILE,
        "--output_dir", FULL_ADAPTER_DIR,
        "--epochs", "5"
    ], capture_output=True, text=True)
    full_train_time = time.time() - start
    
    print(f"[Full Train] Training time: {full_train_time:.1f}s")
    if result.returncode != 0:
        print(f"[Full Train] Error: {result.stderr}")
    
    # Evaluate full-trained model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, FULL_ADAPTER_DIR)
    model.to("cuda")
    
    results['full_train'] = {
        'accuracy': evaluate_model(model, tokenizer, data, "Full-Train"),
        'train_time': full_train_time
    }
    
    del model
    torch.cuda.empty_cache()
    
    # ==================== PARTIAL TRAINING ====================
    print("\n" + "=" * 40)
    print("3. PARTIAL TRAINING (Last 3 layers)")
    print("=" * 40)
    
    os.makedirs(PARTIAL_ADAPTER_DIR, exist_ok=True)
    
    start = time.time()
    result = subprocess.run([
        sys.executable, "/data/mwicker/VPA/src/train_worker_partial.py",
        "--input_file", SHARD_FILE,
        "--output_dir", PARTIAL_ADAPTER_DIR,
        "--epochs", "5"
    ], capture_output=True, text=True)
    partial_train_time = time.time() - start
    
    print(f"[Partial Train] Training time: {partial_train_time:.1f}s")
    if result.returncode != 0:
        print(f"[Partial Train] Error: {result.stderr}")
    else:
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    
    # Evaluate partial-trained model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, PARTIAL_ADAPTER_DIR)
    model.to("cuda")
    
    results['partial_train'] = {
        'accuracy': evaluate_model(model, tokenizer, data, "Partial-Train"),
        'train_time': partial_train_time
    }
    
    del model
    torch.cuda.empty_cache()
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Accuracy':<15} {'Train Time':<15} {'Speedup'}")
    print("-" * 60)
    
    for name, res in results.items():
        speedup = results['full_train']['train_time'] / res['train_time'] if res['train_time'] > 0 else float('inf')
        speedup_str = f"{speedup:.1f}x" if speedup != float('inf') else "N/A"
        print(f"{name:<20} {res['accuracy']:.1%}{'':>8} {res['train_time']:.1f}s{'':>8} {speedup_str}")
    
    # Save results
    with open("/data/mwicker/VPA/output/benchmark/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /data/mwicker/VPA/output/benchmark/results.json")

if __name__ == "__main__":
    main()
