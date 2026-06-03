#!/usr/bin/env python3
"""
Model Evaluation with Multiple Prompts - Fast Version
Compares V1_Strict vs Blend_B prompts across Zero-Shot, LoRA, and FPLC models
"""

import os
os.environ["HF_HOME"] = "/data/mwicker/VPA/cache/huggingface"

import json
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from test_filter import extract_filter_features, is_solvable

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
TEST_FILE = "/data/mwicker/VPA/data/test.jsonl"
ADAPTER_BASE = "/data/mwicker/VPA/output/adapters"
LAST3_BASE = "/data/mwicker/VPA/output/adapters_last3/shard_0000/tail_weights.pt"
NUM_SAMPLES = 50
MAX_NEW_TOKENS = 60

# Prompts to compare
PROMPTS = {
    "V1_Strict": """You are a tool-calling assistant. Given the user query, respond ONLY with a JSON object in this exact format:
{{"name": "<tool_name>", "arguments": "<json_arguments_string>"}}

Do NOT add any explanation, commentary, or code blocks. Output ONLY the raw JSON.

Available tools:
{tools_def}

User: {question}

JSON:""",

    "Blend_B": """You are a tool-calling assistant. Given the user query, respond ONLY with a JSON object.
Use the EXACT tool name from the list below.

Format: {{"name": "<exact_tool_name>", "arguments": "<json_arguments_string>"}}

Do NOT add any explanation. Output ONLY the raw JSON.

Available tools:
{tools_def}

User: {question}

JSON:"""
}

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

def evaluate_model(model, tokenizer, data, prompt_template, label="Model"):
    """Single-input evaluation with specified prompt template."""
    correct = 0
    total = 0
    
    for item in tqdm(data, desc=label, leave=False):
        prompt = prompt_template.format(tools_def=item['tools_def'], question=item['question'])
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        
        output = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        if item['tool_name'] in output:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    return correct, total, acc

def main():
    print("=== MODEL + PROMPT COMPARISON ===")
    print(f"Samples: {NUM_SAMPLES}, Max tokens: {MAX_NEW_TOKENS}\n")
    
    # Load and filter data
    print("Loading data (with decision tree filter)...")
    with open(TEST_FILE, "r") as f:
        lines = f.readlines()
    
    random.seed(42)
    random.shuffle(lines)
    
    test_data = []
    for line in lines:
        item = json.loads(line)
        target = get_tool_call(item)
        if not target:
            continue
            
        tool_name = extract_tool_name(target)
        features = extract_filter_features(item['tools'], item['question'], tool_name)
        
        if is_solvable(features):
            test_data.append({
                'tools_def': item['tools'],
                'question': item['question'],
                'target': target,
                'tool_name': tool_name
            })
            
        if len(test_data) >= NUM_SAMPLES:
            break
            
    print(f"Loaded {len(test_data)} solvable samples.\n")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}  # {(model, prompt): accuracy}
    
    # ============ ZERO-SHOT ============
    print("=" * 50)
    print("1. Zero-Shot (Base)")
    print("=" * 50)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda()
    
    for pname, ptemplate in PROMPTS.items():
        c, t, acc = evaluate_model(model, tokenizer, test_data, ptemplate, f"Zero-Shot + {pname}")
        results[("Zero-Shot", pname)] = acc
        print(f"  {pname}: {c}/{t} = {acc:.1%}")
    
    del model
    torch.cuda.empty_cache()
    
    # ============ FULL-TRAIN LoRA ============
    print("\n" + "=" * 50)
    print("2. Full-Train LoRA")
    print("=" * 50)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda()
    model.load_adapter(os.path.join(ADAPTER_BASE, "shard_0000"), adapter_name="shard_0000")
    model.set_adapter("shard_0000")
    
    for pname, ptemplate in PROMPTS.items():
        c, t, acc = evaluate_model(model, tokenizer, test_data, ptemplate, f"LoRA + {pname}")
        results[("Full-Train LoRA", pname)] = acc
        print(f"  {pname}: {c}/{t} = {acc:.1%}")
    
    del model
    torch.cuda.empty_cache()
    
    # ============ LAST-LAYER FPLC ============
    print("\n" + "=" * 50)
    print("3. Last-Layer Full-Weight (FPLC)")
    print("=" * 50)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cuda()
    
    if os.path.exists(LAST3_BASE):
        state_dict = torch.load(LAST3_BASE)
        
        for i in [13, 14, 15]:
            layer = model.model.layers[i]
            layer_prefix = f"layers.{i}."
            layer_state = {k[len(layer_prefix):]: v for k, v in state_dict.items() if k.startswith(layer_prefix)}
            layer.load_state_dict(layer_state)
            
        norm_state = {k[5:]: v for k, v in state_dict.items() if k.startswith("norm.")}
        model.model.norm.load_state_dict(norm_state)
        
        lm_head_state = {k[8:]: v for k, v in state_dict.items() if k.startswith("lm_head.")}
        model.lm_head.load_state_dict(lm_head_state)
        
        for pname, ptemplate in PROMPTS.items():
            c, t, acc = evaluate_model(model, tokenizer, test_data, ptemplate, f"FPLC + {pname}")
            results[("Last-Layer FPLC", pname)] = acc
            print(f"  {pname}: {c}/{t} = {acc:.1%}")
    else:
        print(f"Error: Last-3 weights not found at {LAST3_BASE}")
    
    # ============ SUMMARY TABLE ============
    print("\n" + "=" * 60)
    print("SUMMARY: Model × Prompt Accuracy")
    print("=" * 60)
    print(f"| {'Model':<20} | {'V1_Strict':<10} | {'Blend_B':<10} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*12}|")
    
    for model_name in ["Zero-Shot", "Full-Train LoRA", "Last-Layer FPLC"]:
        v1 = results.get((model_name, "V1_Strict"), 0)
        bb = results.get((model_name, "Blend_B"), 0)
        print(f"| {model_name:<20} | {v1:.1%}      | {bb:.1%}      |")
    print("=" * 60)

if __name__ == "__main__":
    main()
