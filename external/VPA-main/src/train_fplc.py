#!/usr/bin/env python3
"""
Two-Phase Training for VPA: Frozen-Prefix Latent Caching (FPLC)
================================================================
Phase 1: Precompute hidden states after layer 12 (runs once per shard)
Phase 2: Train LoRA on layers 13-15 using cached embeddings

This avoids the HuggingFace Trainer issue with layers_to_transform.
"""

import json
import torch
import os
import time
import argparse
from tqdm import tqdm

os.environ["HF_HOME"] = "/data/mwicker/VPA/cache/huggingface"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Importing libraries...")
from transformers import AutoTokenizer, AutoModelForCausalLM
log("Imports complete")

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
FREEZE_AFTER_LAYER = 12  # Freeze layers 0-12, cache output after layer 12


def format_prompt(item, tokenizer):
    """Format input prompt and get labels."""
    tools_def = item['tools']
    question = item['question']
    
    msgs = item.get('messages')
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    
    tool_call_content = ""
    for m in msgs:
        if m.get('role') == 'tool_call':
            tool_call_content = m.get('content')
            break
    
    if not tool_call_content:
        return None, None, None
    
    prompt = (
        "System: Only respond with the formatted tool string and parameters for the correct tool use. "
        "You may select from the following tools:\n"
        f"{tools_def}\n\n"
        f"User: {question}\n\n"
        "Assistant:"
    )
    
    # Tokenize prompt and target
    prompt_enc = tokenizer(prompt, truncation=True, max_length=2048, add_special_tokens=False, return_tensors="pt")
    target_enc = tokenizer(tool_call_content + tokenizer.eos_token, truncation=True, max_length=512, add_special_tokens=False, return_tensors="pt")
    
    # Combine
    input_ids = torch.cat([prompt_enc["input_ids"], target_enc["input_ids"]], dim=1)
    attention_mask = torch.ones_like(input_ids)
    
    # Labels: -100 for prompt, actual ids for target
    labels = torch.cat([
        torch.full_like(prompt_enc["input_ids"], -100),
        target_enc["input_ids"]
    ], dim=1)
    
    return input_ids, attention_mask, labels


def precompute_embeddings(input_file, output_dir):
    """Phase 1: Pass data through frozen prefix and cache embeddings + position info."""
    log(f"Phase 1: Precomputing embeddings for {input_file}")
    os.makedirs(output_dir, exist_ok=True)
    
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to("cuda")
    model.eval()
    log(f"Model loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Get rotary embedding function for position embeddings
    rotary_emb = model.model.rotary_emb  # OlMo2 has rotary_emb at model level
    
    # Load data
    log("Loading data...")
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    log(f"Loaded {len(data)} samples")
    
    # Process each sample
    cached_count = 0
    for idx, item in enumerate(tqdm(data, desc="Caching embeddings")):
        input_ids, attention_mask, labels = format_prompt(item, tokenizer)
        
        if input_ids is None:
            continue
        
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        
        with torch.no_grad():
            # Get hidden states after the frozen layers
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get hidden state after layer 12 (index 13 since layer 0 is embedding)
            hidden_state = outputs.hidden_states[FREEZE_AFTER_LAYER + 1]  # Shape: [1, seq_len, hidden_dim]
            
            # Also compute position embeddings for later use
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
            cos, sin = rotary_emb(hidden_state, position_ids)
        
        # Save to disk
        cache_path = os.path.join(output_dir, f"sample_{idx:04d}.pt")
        torch.save({
            "hidden_state": hidden_state.cpu().half(),
            "attention_mask": attention_mask.cpu(),
            "labels": labels,
            "seq_len": seq_len,
            "cos": cos.cpu().half(),
            "sin": sin.cpu().half(),
        }, cache_path)
        cached_count += 1
    
    log(f"Cached {cached_count} samples to {output_dir}")
    return cached_count


def train_tail_layers(embeddings_dir, output_dir, epochs=5):
    """Phase 2: Train only the last 3 layers on cached embeddings."""
    log(f"Phase 2: Training last 3 layers on cached embeddings")
    
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    
    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to("cuda")
    
    # Freeze all layers EXCEPT the last 3
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Unfreeze last 3 transformer layers
    for i in [13, 14, 15]:
        for name, param in model.model.layers[i].named_parameters():
            param.requires_grad = True
    
    # Also unfreeze the final layer norm and lm_head
    for param in model.model.norm.parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Load cached embeddings
    cache_files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.pt')])
    log(f"Loading {len(cache_files)} cached samples...")
    
    # Simple training loop
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    
    total_loss = 0
    steps = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for cache_file in tqdm(cache_files, desc=f"Epoch {epoch+1}/{epochs}"):
            cache_path = os.path.join(embeddings_dir, cache_file)
            cached = torch.load(cache_path)
            
            hidden_state = cached["hidden_state"].to("cuda")
            attention_mask = cached["attention_mask"].to("cuda")
            labels = cached["labels"].to("cuda")
            cos = cached["cos"].to("cuda")
            sin = cached["sin"].to("cuda")
            
            # Forward through last 3 layers + head
            x = hidden_state
            position_embeddings = (cos, sin)  # OlMo2 expects this tuple
            
            for layer_idx in [13, 14, 15]:
                layer = model.model.layers[layer_idx]
                layer_outputs = layer(
                    x,
                    attention_mask=None,  # OlMo2 uses causal mask internally
                    position_embeddings=position_embeddings,
                )
                x = layer_outputs[0]
            
            # Final norm and lm_head
            x = model.model.norm(x)
            logits = model.lm_head(x)
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
        
        avg_loss = epoch_loss / len(cache_files)
        log(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    
    # Save the trained model layers
    os.makedirs(output_dir, exist_ok=True)
    
    # Save state dict for last 3 layers
    state_dict = {}
    for i in [13, 14, 15]:
        for name, param in model.model.layers[i].named_parameters():
            state_dict[f"layers.{i}.{name}"] = param.data.cpu()
    state_dict["norm"] = model.model.norm.state_dict()
    state_dict["lm_head"] = model.lm_head.state_dict()
    
    torch.save(state_dict, os.path.join(output_dir, "tail_weights.pt"))
    log(f"Saved tail weights to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Shard JSONL file")
    parser.add_argument("--cache_dir", type=str, default=None, help="Dir for cached embeddings")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir for trained weights")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--phase", type=str, choices=["1", "2", "both"], default="both")
    args = parser.parse_args()
    
    if args.cache_dir is None:
        args.cache_dir = args.output_dir + "_cache"
    
    start = time.time()
    
    if args.phase in ["1", "both"]:
        precompute_embeddings(args.input_file, args.cache_dir)
    
    if args.phase in ["2", "both"]:
        train_tail_layers(args.cache_dir, args.output_dir, args.epochs)
    
    elapsed = time.time() - start
    log(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
