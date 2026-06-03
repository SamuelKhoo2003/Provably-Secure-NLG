"""
Token-Level Voting Ensemble (Memory-Efficient Version)
Implements majority voting at each token position.
Loads adapters one at a time to avoid OOM.
"""
import os
os.environ["HF_HOME"] = "/data/mwicker/VPA/cache/huggingface"

import json
import torch
import random
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from test_filter import extract_filter_features, is_solvable

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
TEST_FILE = "/data/mwicker/VPA/data/test.jsonl"
ADAPTER_BASE = "/data/mwicker/VPA/output/adapters"
NUM_SAMPLES = 30  # Further reduced for speed
NUM_ENSEMBLE_SHARDS = 10  # Use 10 shards for memory efficiency
MAX_NEW_TOKENS = 80

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

def get_next_token_from_adapter(model, tokenizer, adapter_path, adapter_name, input_ids):
    """Load adapter, get next token prediction, unload adapter."""
    model.load_adapter(adapter_path, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).item()
    
    model.delete_adapter(adapter_name)
    return next_token

def token_level_majority_vote(model, tokenizer, shard_dirs, input_ids, max_new_tokens=MAX_NEW_TOKENS):
    """Generate tokens using majority voting at each step."""
    current_ids = input_ids.clone()
    eos_token_id = tokenizer.eos_token_id
    
    for step in range(max_new_tokens):
        # Collect next token predictions from all adapters
        next_token_votes = []
        
        for sname in shard_dirs:
            adapter_path = os.path.join(ADAPTER_BASE, sname)
            next_token = get_next_token_from_adapter(model, tokenizer, adapter_path, sname, current_ids)
            next_token_votes.append(next_token)
        
        # Majority vote
        vote_counts = Counter(next_token_votes)
        majority_token = vote_counts.most_common(1)[0][0]
        
        # Append voted token
        current_ids = torch.cat([
            current_ids, 
            torch.tensor([[majority_token]], device=current_ids.device)
        ], dim=1)
        
        # Check for EOS
        if majority_token == eos_token_id:
            break
        
        # Log progress every 20 tokens
        if (step + 1) % 20 == 0:
            print(f"    Token {step+1}/{max_new_tokens}")
    
    return current_ids

def main():
    print("=== TOKEN-LEVEL VOTING ENSEMBLE ===\n")
    
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
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda()
    
    # Get shard directories
    shard_dirs = sorted([d for d in os.listdir(ADAPTER_BASE) if d.startswith("shard_")])[:NUM_ENSEMBLE_SHARDS]
    print(f"Using {len(shard_dirs)} shards for ensemble\n")
    
    # === Zero-shot evaluation ===
    print("=== Zero-Shot Model ===")
    base_correct = 0
    for i, item in enumerate(filtered_data):
        prompt = V1_ORIGINAL.format(tools_def=item['tools_def'], question=item['question'])
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        output = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if item['tool_name'] in output:
            base_correct += 1
        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{n}")
    
    print(f"Zero-Shot Accuracy: {base_correct}/{n} ({base_correct/n*100:.1f}%)\n")
    
    # === Shard 0 only ===
    print("=== Shard 0 ===")
    model.load_adapter(os.path.join(ADAPTER_BASE, shard_dirs[0]), adapter_name=shard_dirs[0])
    model.set_adapter(shard_dirs[0])
    
    shard0_correct = 0
    for i, item in enumerate(filtered_data):
        prompt = V1_ORIGINAL.format(tools_def=item['tools_def'], question=item['question'])
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        output = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if item['tool_name'] in output:
            shard0_correct += 1
        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{n}")
    
    model.delete_adapter(shard_dirs[0])
    print(f"Shard 0 Accuracy: {shard0_correct}/{n} ({shard0_correct/n*100:.1f}%)\n")
    
    # === Token-Level Voting Ensemble ===
    print(f"=== Token-Level Voting Ensemble ({NUM_ENSEMBLE_SHARDS} shards) ===")
    print("  (This will be slower due to per-token voting)")
    
    ensemble_correct = 0
    for i, item in enumerate(filtered_data):
        prompt = V1_ORIGINAL.format(tools_def=item['tools_def'], question=item['question'])
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        print(f"  Example {i+1}/{n}...")
        
        # Token-level majority vote generation
        output_ids = token_level_majority_vote(
            model, tokenizer, shard_dirs, 
            inputs['input_ids'], max_new_tokens=MAX_NEW_TOKENS
        )
        
        output = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        if item['tool_name'] in output:
            ensemble_correct += 1
            print(f"    ✓ Correct")
        else:
            print(f"    ✗ Wrong (Expected: {item['tool_name'][:30]}...)")
    
    print(f"\nEnsemble (Token-Level) Accuracy: {ensemble_correct}/{n} ({ensemble_correct/n*100:.1f}%)")
    
    # === Summary ===
    print("\n" + "="*50)
    print("=== SUMMARY ===")
    print("="*50)
    print(f"| Model                  | Accuracy       |")
    print(f"|------------------------|----------------|")
    print(f"| Zero-Shot              | {base_correct}/{n} ({base_correct/n*100:.1f}%) |")
    print(f"| Shard 0                | {shard0_correct}/{n} ({shard0_correct/n*100:.1f}%) |")
    print(f"| Ensemble (Token-Level) | {ensemble_correct}/{n} ({ensemble_correct/n*100:.1f}%) |")
    print("="*50)

if __name__ == "__main__":
    main()
