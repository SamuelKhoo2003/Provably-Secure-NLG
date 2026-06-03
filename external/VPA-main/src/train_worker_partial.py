#!/usr/bin/env python3
"""
Partial-Layer Training Worker for VPA
=====================================
Trains LoRA adapters on ONLY the last 3 layers (13-15) of OLMo-2-1B.
Layers 0-12 are completely frozen - no LoRA applied.

This is a separate script from train_worker.py to preserve the original
for comparison benchmarks.
"""

import sys
import json
import os
import time

def log(msg):
    """Print with timestamp and flush."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting imports...")
os.environ["HF_HOME"] = "/data/mwicker/VPA/cache/huggingface"

log("Importing torch...")
import torch
log(f"Torch imported, CUDA available: {torch.cuda.is_available()}")

log("Importing transformers...")
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
log("Transformers imported")

log("Importing peft...")
from peft import LoraConfig, get_peft_model, TaskType
log("All imports complete")

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"

# Frozen-Prefix Configuration: 13 frozen layers, 3 trainable layers
TRAINABLE_LAYERS = [13, 14, 15]  # Last 3 layers get LoRA (of 16 total)

def format_prompt(item):
    tools_def = item['tools']
    question = item['question']
    
    msgs = item.get('messages')
    if isinstance(msgs, str): msgs = json.loads(msgs)
    
    tool_call_content = ""
    for m in msgs:
        if m.get('role') == 'tool_call':
            tool_call_content = m.get('content')
            break
            
    prompt = f"""System: Only respond with the formatted tool string and parameters for the correct tool use. You may select from the following tools:
{tools_def}

User: {question}

Assistant:"""

    return prompt, tool_call_content

def train(input_file, output_dir, epochs=5):
    start_time = time.time()
    log(f"Worker processing {input_file} -> {output_dir}")
    log(f"Training only layers: {TRAINABLE_LAYERS}")
    
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log("Tokenizer loaded")
        
    log("Loading data...")
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    log(f"Loaded {len(data)} samples")
            
    formatted_data = []
    for item in data:
        prompt, target = format_prompt(item)
        if target:
            full_text = prompt + target + tokenizer.eos_token
            formatted_data.append({"text": full_text, "prompt": prompt, "completion": target})
            
    if not formatted_data:
        log("Warning: No valid training data in shard.")
        return
    log(f"Formatted {len(formatted_data)} training samples")

    dataset = Dataset.from_list(formatted_data)
    
    def tokenize(element):
        prompt_enc = tokenizer(element["prompt"], truncation=True, max_length=2048, add_special_tokens=False)
        target_enc = tokenizer(element["completion"] + tokenizer.eos_token, truncation=True, max_length=512, add_special_tokens=False)
        
        input_ids = prompt_enc["input_ids"] + target_enc["input_ids"]
        labels = [-100] * len(prompt_enc["input_ids"]) + target_enc["input_ids"]
        attention_mask = [1] * len(input_ids)
        
        if len(input_ids) > 2048:
            input_ids = input_ids[:2048]
            labels = labels[:2048]
            attention_mask = attention_mask[:2048]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    log("Tokenizing dataset...")
    tokenized_ds = dataset.map(tokenize, remove_columns=["text", "prompt", "completion"])
    log("Dataset tokenized")
    
    # Init Model
    log("Loading model (this may take ~30s)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    log("Model loaded, moving to CUDA...")
    model.to("cuda")
    log(f"Model on GPU, VRAM used: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # PARTIAL TRAINING: LoRA only on last 3 layers (13, 14, 15)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=TRAINABLE_LAYERS  # KEY CHANGE: Only layers 13-15
    )
    log("Applying PEFT config...")
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=epochs, 
        logging_steps=1,  # More frequent logging for debugging
        save_strategy="no",
        fp16=True,
        report_to="none"
    )
    
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_ds,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )
    
    log("Starting training...")
    trainer.train()
    
    log("Saving adapter...")
    model.save_pretrained(output_dir)
    
    elapsed = time.time() - start_time
    log(f"Saved adapter to {output_dir}")
    log(f"Training time: {elapsed:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train(args.input_file, args.output_dir, args.epochs)
