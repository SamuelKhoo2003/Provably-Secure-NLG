import os
import json
import random
from datasets import load_from_disk
from collections import Counter

DATASET_PATH = "/data/mwicker/VPA/datasets/toucan"
OUTPUT_DIR = "/data/mwicker/VPA/data"
TOP_N = 150

def get_top_n_tools(ds, n):
    target_counts = Counter()
    print("Counting tool usage...")
    for item in ds:
        tt = item.get('target_tools')
        if tt:
            if isinstance(tt, str):
                names = [x.strip() for x in tt.split(',')]
                for n_tool in names: target_counts[n_tool] += 1
            elif isinstance(tt, list):
                for n_tool in tt: target_counts[n_tool] += 1
    
    return [t for t, c in target_counts.most_common(n)]

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print(f"Loading dataset from {DATASET_PATH}...")
    ds = load_from_disk(DATASET_PATH)
    
    top_tools = get_top_n_tools(ds, TOP_N)
    print(f"Top {TOP_N} tools identified: {top_tools[:10]}...")
    top_tools_set = set(top_tools)
    
    filtered_data = []
    print("Filtering dataset...")
    for item in ds:
        # Check if ANY target tool is in top_tools
        # Or should it be ALL? User said "associated with 10 tools". 
        # Usually checking if at least one employed tool is in the set is standard.
        tt = item.get('target_tools')
        match = False
        if tt:
             if isinstance(tt, str):
                names = [x.strip() for x in tt.split(',')]
                if any(n in top_tools_set for n in names): match = True
             elif isinstance(tt, list):
                if any(n in top_tools_set for n in tt): match = True
        
        if match:
            filtered_data.append(item)
            
    print(f"Filtered count: {len(filtered_data)}")
    
    # Shuffle and Split
    random.seed(42)
    random.shuffle(filtered_data)
    
    split_idx = int(len(filtered_data) * 0.9)
    train_data = filtered_data[:split_idx]
    test_data = filtered_data[split_idx:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")
    
    # Save
    with open(os.path.join(OUTPUT_DIR, "train.jsonl"), "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(os.path.join(OUTPUT_DIR, "test.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
            
    print("Saved train.jsonl and test.jsonl to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
