#!/usr/bin/env python3
"""
Simple Two-Phase Training for VPA
==================================
Phase 1: Precompute full outputs through layer 12 (input_ids -> hidden_states)
Phase 2: Fine-tune just the last 3 layers using full model forward but with partial gradient

This is simpler and avoids OlMo2 layer-specific API issues.
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
from torch.optim import AdamW
log("Imports complete")

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"


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
    
    # Tokenize
    prompt_enc = tokenizer(prompt, truncation=True, max_length=2048, add_special_tokens=False, return_tensors="pt")
    target_enc = tokenizer(tool_call_content + tokenizer.eos_token, truncation=True, max_length=512, add_special_tokens=False, return_tensors="pt")
    
    input_ids = torch.cat([prompt_enc["input_ids"], target_enc["input_ids"]], dim=1)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.cat([
        torch.full_like(prompt_enc["input_ids"], -100),
        target_enc["input_ids"]
    ], dim=1)
    
    return input_ids, attention_mask, labels


def train_last_3_layers(input_file, output_dir, epochs=5):
    """Simple approach: freeze first 13 layers, train last 3 + head directly."""
    log(f"Training last 3 layers for {input_file}")
    os.makedirs(output_dir, exist_ok=True)
    
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)  # Use fp32 for stable gradients
    model.to("cuda")
    log(f"Model loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Freeze first 13 layers (0-12)
    log("Freezing layers 0-12...")
    for i in range(13):
        for param in model.model.layers[i].parameters():
            param.requires_grad = False
    
    # Also freeze embedding layer
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Load data
    log("Loading data...")
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    log(f"Loaded {len(data)} samples")
    
    # Prepare training samples
    samples = []
    for item in data:
        input_ids, attention_mask, labels = format_prompt(item, tokenizer)
        if input_ids is not None:
            samples.append((input_ids, attention_mask, labels))
    log(f"Prepared {len(samples)} training samples")
    
    # Optimizer - only for trainable params
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for input_ids, attention_mask, labels in tqdm(samples, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            labels = labels.to("cuda")
            
            # Forward pass (frozen layers don't accumulate gradients)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(samples)
        log(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    
    # Save only the trained layers
    log("Saving trained layers...")
    state_dict = {}
    for i in [13, 14, 15]:
        for name, param in model.model.layers[i].named_parameters():
            state_dict[f"layers.{i}.{name}"] = param.data.cpu()
    for name, param in model.model.norm.named_parameters():
        state_dict[f"norm.{name}"] = param.data.cpu()
    for name, param in model.lm_head.named_parameters():
        state_dict[f"lm_head.{name}"] = param.data.cpu()
    
    torch.save(state_dict, os.path.join(output_dir, "tail_weights.pt"))
    log(f"Saved to {output_dir}/tail_weights.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    start = time.time()
    train_last_3_layers(args.input_file, args.output_dir, args.epochs)
    elapsed = time.time() - start
    log(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
