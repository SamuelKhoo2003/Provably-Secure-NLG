"""
Ensemble Evaluation Script
Evaluates: Zero-shot, Single Shard, Majority Vote (20 shards)
On 100 filtered examples using decision tree filter.
"""
import os
os.environ["HF_HOME"] = "/data/mwicker/VPA/cache/huggingface"

import json
import torch
import random
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from test_filter import extract_filter_features, is_solvable

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
TEST_FILE = "/data/mwicker/VPA/data/test.jsonl"
ADAPTER_BASE = "/data/mwicker/VPA/output/adapters"
NUM_SAMPLES = 100
NUM_ENSEMBLE_SHARDS = 20

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

def extract_predicted_tool(output_str):
    """Extract tool name from model output."""
    # Try to parse JSON
    try:
        # Find JSON-like pattern
        match = re.search(r'\{[^}]+\}', output_str)
        if match:
            parsed = json.loads(match.group())
            return parsed.get('name', '')
    except:
        pass
    
    # Fallback: look for "name": "..."
    match = re.search(r'"name"\s*:\s*"([^"]+)"', output_str)
    if match:
        return match.group(1)
    return ''

def main():
    print("=== ENSEMBLE EVALUATION ===\n")
    
    # Load and filter test data
    print("Loading and filtering test data...")
    with open(TEST_FILE, "r") as f:
        lines = f.readlines()
    
    random.seed(42)
    random.shuffle(lines)
    
    filtered_data = []
    for line in lines:
        item = json.loads(line)
        target = get_tool_call(item)
        if not target:
            continue
        
        tool_name = extract_tool_name(target)
        features = extract_filter_features(item['tools'], item['question'], tool_name)
        
        if is_solvable(features):
            filtered_data.append({
                'tools_def': item['tools'],
                'question': item['question'],
                'tool_name': tool_name
            })
        
        if len(filtered_data) >= NUM_SAMPLES:
            break
    
    n = len(filtered_data)
    print(f"Filtered samples: {n}\n")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    
    def generate_prediction(item):
        prompt = V1_ORIGINAL.format(tools_def=item['tools_def'], question=item['question'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # === Zero-shot evaluation ===
    print("=== Zero-Shot Model ===")
    base_correct = 0
    for i, item in enumerate(filtered_data):
        output = generate_prediction(item)
        if item['tool_name'] in output:
            base_correct += 1
        if (i+1) % 25 == 0:
            print(f"  Progress: {i+1}/{n}")
    
    print(f"Zero-Shot Accuracy: {base_correct}/{n} ({base_correct/n*100:.1f}%)\n")
    
    # === Single Shard (shard_0000) ===
    print("=== Shard 0 ===")
    model.load_adapter(os.path.join(ADAPTER_BASE, "shard_0000"), adapter_name="shard_0000")
    model.set_adapter("shard_0000")
    
    shard0_correct = 0
    for i, item in enumerate(filtered_data):
        output = generate_prediction(item)
        if item['tool_name'] in output:
            shard0_correct += 1
        if (i+1) % 25 == 0:
            print(f"  Progress: {i+1}/{n}")
    
    model.delete_adapter("shard_0000")
    print(f"Shard 0 Accuracy: {shard0_correct}/{n} ({shard0_correct/n*100:.1f}%)\n")
    
    # === Ensemble (Majority Vote of 20 shards) ===
    print(f"=== Ensemble (Majority Vote, {NUM_ENSEMBLE_SHARDS} shards) ===")
    
    # Get available shards
    shard_dirs = sorted([d for d in os.listdir(ADAPTER_BASE) if d.startswith("shard_")])[:NUM_ENSEMBLE_SHARDS]
    print(f"Using shards: {shard_dirs[0]} to {shard_dirs[-1]}")
    
    ensemble_predictions = [[] for _ in range(n)]  # Store predictions per example
    
    for shard_idx, sname in enumerate(shard_dirs):
        model.load_adapter(os.path.join(ADAPTER_BASE, sname), adapter_name=sname)
        model.set_adapter(sname)
        
        for i, item in enumerate(filtered_data):
            output = generate_prediction(item)
            pred_tool = extract_predicted_tool(output)
            ensemble_predictions[i].append(pred_tool)
        
        model.delete_adapter(sname)
        print(f"  Completed shard {shard_idx+1}/{NUM_ENSEMBLE_SHARDS}")
    
    # Compute majority vote accuracy
    ensemble_correct = 0
    for i, item in enumerate(filtered_data):
        votes = ensemble_predictions[i]
        vote_counts = Counter(votes)
        majority_tool = vote_counts.most_common(1)[0][0] if votes else ''
        
        if majority_tool == item['tool_name']:
            ensemble_correct += 1
    
    print(f"\nEnsemble Accuracy: {ensemble_correct}/{n} ({ensemble_correct/n*100:.1f}%)")
    
    # === Summary ===
    print("\n" + "="*50)
    print("=== SUMMARY ===")
    print("="*50)
    print(f"| Model            | Accuracy    |")
    print(f"|------------------|-------------|")
    print(f"| Zero-Shot        | {base_correct}/{n} ({base_correct/n*100:.1f}%) |")
    print(f"| Shard 0          | {shard0_correct}/{n} ({shard0_correct/n*100:.1f}%) |")
    print(f"| Ensemble (20)    | {ensemble_correct}/{n} ({ensemble_correct/n*100:.1f}%) |")
    print("="*50)

if __name__ == "__main__":
    main()
