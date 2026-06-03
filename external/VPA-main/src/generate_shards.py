import json
import os
import math
import shutil

INPUT_FILE = "/data/mwicker/VPA/data/train.jsonl"
SHARD_DIR = "/data/mwicker/VPA/data/shards"
K = 1000

def generate_shards():
    if os.path.exists(SHARD_DIR):
        shutil.rmtree(SHARD_DIR)
    os.makedirs(SHARD_DIR)
    
    print(f"Loading {INPUT_FILE}...")
    data = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    
    # Calculate shard size
    # We want K shards.
    # Base size = total // K
    # Remainder distributed to first R shards
    base_size = total_samples // K
    remainder = total_samples % K
    
    print(f"Generating {K} shards (Base size: {base_size})...")
    
    start_idx = 0
    for i in range(K):
        # Determine size for this shard
        size = base_size + (1 if i < remainder else 0)
        end_idx = start_idx + size
        
        shard_data = data[start_idx:end_idx]
        shard_file = os.path.join(SHARD_DIR, f"shard_{i:04d}.jsonl")
        
        with open(shard_file, "w") as f:
            for item in shard_data:
                f.write(json.dumps(item) + "\n")
                
        start_idx = end_idx
        
        if i % 100 == 0:
            print(f"Created shard {i} ({len(shard_data)} samples)")
            
    print(f"Done. Generated 1000 shards in {SHARD_DIR}")

if __name__ == "__main__":
    generate_shards()
