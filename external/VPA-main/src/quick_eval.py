#!/usr/bin/env python3
"""Quick accuracy comparison - using V1_ORIGINAL prompt format from eval_ensemble.py"""

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
NUM_SAMPLES = 20  # Fast run

# V1_ORIGINAL prompt from eval_ensemble.py
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
    # Try JSON parse
    try:
        match = re.search(r'\{[^}]+\}', gen)
        if match:
            d = json.loads(match.group())
            return d.get('name', '')
    except:
        pass
    # Fallback: regex
    match = re.search(r'"name"\s*:\s*"([^"]+)"', gen)
    if match:
        return match.group(1)
    # Check if tool name appears directly
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

# Load data
print(f"Loading shard_0000 data (using {NUM_SAMPLES} samples)...")
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

# 1. Full-Layer LoRA
print("\nLoading full-layer LoRA adapter...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
full_model = PeftModel.from_pretrained(base_model, "output/adapters/shard_0000")
full_model.to("cuda")
full_model.eval()
full_c, full_t = evaluate(full_model, tokenizer, data, "Full-Layer LoRA")
del full_model, base_model
torch.cuda.empty_cache()

# 2. Last-3-Layer
print("\nLoading last-3-layer model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
base_model.to("cuda")
tail_weights = torch.load("output/adapters_last3/shard_0000/tail_weights.pt")
for i in [13, 14, 15]:
    layer = base_model.model.layers[i]
    layer_prefix = f"layers.{i}."
    layer_state = {k[len(layer_prefix):]: v for k, v in tail_weights.items() if k.startswith(layer_prefix)}
    layer.load_state_dict(layer_state)
norm_state = {k[5:]: v for k, v in tail_weights.items() if k.startswith("norm.")}
base_model.model.norm.load_state_dict(norm_state)
lm_head_state = {k[8:]: v for k, v in tail_weights.items() if k.startswith("lm_head.")}
base_model.lm_head.load_state_dict(lm_head_state)
base_model.eval()
last3_c, last3_t = evaluate(base_model, tokenizer, data, "Last-3-Layer")

print("\n" + "="*50)
print("SUMMARY")  
print("="*50)
print(f"Full-Layer LoRA: {full_c}/{full_t} = {100*full_c/full_t:.1f}%")
print(f"Last-3-Layer:    {last3_c}/{last3_t} = {100*last3_c/last3_t:.1f}%")
