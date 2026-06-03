
import json
import torch
import os
os.environ["HF_HOME"] = "/data/mwicker/VPA/cache/huggingface"
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct" 

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
    print(f"Worker processing {input_file} -> {output_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    formatted_data = []
    for item in data:
        prompt, target = format_prompt(item)
        if target:
            # Full text
            full_text = prompt + target + tokenizer.eos_token
            formatted_data.append({"text": full_text, "prompt": prompt, "completion": target})
            
    if not formatted_data:
        print("Warning: No valid training data in shard.")
        return

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
        
    tokenized_ds = dataset.map(tokenize, remove_columns=["text", "prompt", "completion"])
    
    # Init Model without device_map="auto" to avoid OOM in multi-process
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to("cuda")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4, # Reverted to 4 (stable)
        gradient_accumulation_steps=8, # Effective BS = 32 
        learning_rate=2e-4,
        num_train_epochs=epochs, 
        logging_steps=100, # reduce logging
        save_strategy="no",
        fp16=True,
        report_to="none"
    )
    
    # Silence warnings
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_ds,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )
    
    trainer.train()
    
    # Clean checkpoints and save adapter only
    model.save_pretrained(output_dir)
    print(f"Saved adapter to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    # Sequential training - no delay needed
    # delay = random.uniform(1, 20)
    # print(f"Sleeping {delay:.1f}s before start...")
    # time.sleep(delay)
    
    train(args.input_file, args.output_dir, args.epochs)
