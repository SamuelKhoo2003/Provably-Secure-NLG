#!/usr/bin/env python3
"""
LoRA Training for Last 3 Layers (VPA)
=====================================
Applies LoRA adapters ONLY to layers 13-15 of OLMo-2-1B.
Uses the simple manual training loop (like train_last3.py) but with PEFT/LoRA.

Key differences from train_last3.py:
- Uses LoRA instead of full parameter updates
- Produces ~4MB adapter instead of ~1.6GB weights
- Compatible with PeftModel loading pattern
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
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
log("Imports complete")

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"

# LoRA on ONLY the last 3 layers
TRAINABLE_LAYERS = [13, 14, 15]


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


def train_last_3_layers_lora(input_file, output_dir, epochs=5):
    """Train LoRA adapters on only the last 3 layers (13, 14, 15)."""
    log(f"Training LoRA on last 3 layers for {input_file}")
    os.makedirs(output_dir, exist_ok=True)
    
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log("Loading model...")
    # Use fp16 for efficiency (LoRA is more stable than full-weight training)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to("cuda")
    log(f"Model loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Apply LoRA ONLY to last 3 layers
    log(f"Applying LoRA to layers {TRAINABLE_LAYERS}...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=TRAINABLE_LAYERS  # KEY: Only layers 13-15
    )
    model = get_peft_model(model, peft_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    
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
    
    # Optimizer - only for trainable (LoRA) params
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for input_ids, attention_mask, labels in tqdm(samples, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            labels = labels.to("cuda")
            
            # Forward pass
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
    
    # Save LoRA adapter (PEFT format, ~4MB)
    log("Saving LoRA adapter...")
    model.save_pretrained(output_dir)
    
    # Verify size
    adapter_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    log(f"Saved adapter to {output_dir} ({adapter_size/1e6:.1f}MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    start = time.time()
    train_last_3_layers_lora(args.input_file, args.output_dir, args.epochs)
    elapsed = time.time() - start
    log(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
