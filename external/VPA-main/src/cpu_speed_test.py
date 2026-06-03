"""
Quick CPU speed test for VPA inference.
Tests 5 shards on CPU only to estimate throughput.
"""
import os
os.environ["HF_HOME"] = "/data/mwicker/VPA/cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

import json
import torch
import random
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from test_filter import extract_filter_features, is_solvable

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
TEST_FILE = "/data/mwicker/VPA/data/test.jsonl"
ADAPTER_BASE = "/data/mwicker/VPA/output/adapters"
NUM_TEST_SHARDS = 5
MAX_NEW_TOKENS = 60

V1_ORIGINAL = """You are a tool-calling assistant. Given the user query, respond ONLY with a JSON object in this exact format:
{{"name": "<tool_name>", "arguments": "<json_arguments_string>"}}

Do NOT add any explanation, commentary, or code blocks. Output ONLY the raw JSON.

Available tools:
{tools_def}

User: {question}

JSON:"""


def get_tool_call(item):
    msgs = item.get('messages')
    if isinstance(msgs, str): msgs = json.loads(msgs)
    for m in msgs:
        if m.get('role') == 'tool_call':
            return m.get('content')
    return ""


def extract_tool_name(target_str):
    try:
        return eval(target_str).get('name', '')
    except:
        return ''


def main():
    print("=" * 60)
    print("CPU SPEED TEST FOR VPA INFERENCE")
    print("=" * 60)
    print(f"Testing {NUM_TEST_SHARDS} shards on CPU only")
    print(f"CUDA_VISIBLE_DEVICES = '' (CPU forced)")
    print()
    
    # Load one test example
    print("Loading test data...")
    with open(TEST_FILE, "r") as f:
        lines = f.readlines()
    
    random.seed(42)
    random.shuffle(lines)
    
    # Get first filterable example
    test_item = None
    for line in lines:
        item = json.loads(line)
        target = get_tool_call(item)
        if not target:
            continue
        tool_name = extract_tool_name(target)
        features = extract_filter_features(item['tools'], item['question'], tool_name)
        if is_solvable(features):
            test_item = {
                'tools_def': item['tools'],
                'question': item['question'],
            }
            break
    
    if not test_item:
        print("ERROR: No test example found")
        return
    
    print("Loading model on CPU...")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load on CPU with float32 (bfloat16 may not be efficient on CPU)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")
    
    # Get shard directories
    shard_dirs = sorted([d for d in os.listdir(ADAPTER_BASE) if d.startswith("shard_")])[:NUM_TEST_SHARDS]
    print(f"Testing {len(shard_dirs)} shards: {shard_dirs}")
    print()
    
    # Prepare input
    prompt = V1_ORIGINAL.format(tools_def=test_item['tools_def'], question=test_item['question'])
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Time each shard
    times = []
    print("Running inference...")
    for i, sname in enumerate(shard_dirs):
        adapter_path = os.path.join(ADAPTER_BASE, sname)
        
        shard_start = time.time()
        model.load_adapter(adapter_path, adapter_name=sname)
        model.set_adapter(sname)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        
        model.delete_adapter(sname)
        shard_time = time.time() - shard_start
        times.append(shard_time)
        
        print(f"  Shard {i+1}/{NUM_TEST_SHARDS}: {shard_time:.2f}s")
    
    avg_time = sum(times) / len(times)
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Average time per shard (CPU): {avg_time:.2f}s")
    print(f"Time for 500 shards: {avg_time * 500 / 60:.1f} minutes")
    print(f"Time for 100 examples × 500 shards: {avg_time * 500 * 100 / 3600:.1f} hours")
    print()
    
    # Compare to GPU estimate
    gpu_time_per_shard = 0.7  # ~350s per example / 500 shards
    print(f"GPU time per shard (estimated): {gpu_time_per_shard:.2f}s")
    print(f"CPU/GPU ratio: {avg_time / gpu_time_per_shard:.1f}x slower")
    print("=" * 60)


if __name__ == "__main__":
    main()
