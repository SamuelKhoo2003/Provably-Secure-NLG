#!/usr/bin/env python3
"""Quick accuracy evaluation for Last-3-Layer LoRA adapter."""

import json
import torch
import re
import os
import ast
os.environ['HF_HOME'] = '/data/mwicker/VPA/cache/huggingface'

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

MODEL_ID = 'allenai/OLMo-2-0425-1B-Instruct'
NUM_SAMPLES = 60  # Evaluate all valid samples in shard

# V1_ORIGINAL prompt format
V1_ORIGINAL = """You are a tool-calling assistant. Given the user query, respond ONLY with a JSON object in this exact format:
{{"name": "<tool_name>", "arguments": "<json_arguments_string>"}}

Do NOT add any explanation, commentary, or code blocks. Output ONLY the raw JSON.

Available tools:
{tools_def}

User: {question}

JSON:"""

def get_gt_tool(item):
    """Extract ground truth tool name from messages."""
    msgs = item.get('messages')
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    for m in msgs:
        if m.get('role') == 'tool_call':
            content = m.get('content', {})
            if isinstance(content, dict):
                return content.get('name', '')
            elif isinstance(content, str):
                try:
                    c = ast.literal_eval(content)
                    if isinstance(c, dict):
                        return c.get('name', '')
                except:
                    pass
    return None

def extract_tool(gen):
    """Extract tool name from model generation (JSON format)."""
    try:
        match = re.search(r'\{[^}]+\}', gen)
        if match:
            d = json.loads(match.group())
            return d.get('name', '')
    except:
        pass
    match = re.search(r'"name"\s*:\s*"([^"]+)"', gen)
    if match:
        return match.group(1)
    match = re.search(r'(\w+-\w+-\w+[-\w]*)', gen)
    if match:
        return match.group(1)
    return None

def evaluate(model, tokenizer, data, name):
    print(f"\n=== Evaluating: {name} ===")
    correct = 0
    total = 0
    for item in tqdm(data[:NUM_SAMPLES], desc=name):
        gt_tool = get_gt_tool(item)
        if not gt_tool:
            continue
        
        prompt = V1_ORIGINAL.format(tools_def=item['tools'], question=item['question'])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = extract_tool(gen)
        
        if pred == gt_tool:
            correct += 1
        total += 1
    
    acc = 100*correct/total if total > 0 else 0
    print(f"Result: {correct}/{total} = {acc:.1f}%")
    return correct, total

print("="*60)
print("ACCURACY EVALUATION: Last-3-Layer LoRA vs Baselines")
print("="*60)

# Load data
print(f"\nLoading shard_0000 data (using {NUM_SAMPLES} samples)...")
data = []
with open("data/shards/shard_0000.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

gt_count = sum(1 for d in data[:NUM_SAMPLES] if get_gt_tool(d))
print(f"Samples with valid GT: {gt_count}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

results = {}

# 1. Zero-Shot (Base Model)
print("\n[1/3] Loading zero-shot base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
base_model.to("cuda")
base_model.eval()
zs_c, zs_t = evaluate(base_model, tokenizer, data, "Zero-Shot")
results["Zero-Shot"] = (zs_c, zs_t)
del base_model
torch.cuda.empty_cache()

# 2. Full-Layer LoRA
print("\n[2/3] Loading full-layer LoRA adapter...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
full_model = PeftModel.from_pretrained(base_model, "output/adapters/shard_0000")
full_model.to("cuda")
full_model.eval()
full_c, full_t = evaluate(full_model, tokenizer, data, "Full-Layer LoRA")
results["Full-Layer LoRA"] = (full_c, full_t)
del full_model, base_model
torch.cuda.empty_cache()

# 3. Last-3-Layer LoRA (NEW)
print("\n[3/3] Loading last-3-layer LoRA adapter...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
last3_lora = PeftModel.from_pretrained(base_model, "output/adapters_last3_lora/shard_0000")
last3_lora.to("cuda")
last3_lora.eval()
l3_c, l3_t = evaluate(last3_lora, tokenizer, data, "Last-3-Layer LoRA")
results["Last-3-Layer LoRA"] = (l3_c, l3_t)
del last3_lora, base_model
torch.cuda.empty_cache()

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, (c, t) in results.items():
    acc = 100*c/t if t > 0 else 0
    print(f"{name:20s}: {c:2d}/{t:2d} = {acc:5.1f}%")
print("="*60)
