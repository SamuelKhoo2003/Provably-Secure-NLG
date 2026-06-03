#!/usr/bin/env python3
"""
Training Orchestrator for Last-3-Layer LoRA (1000 Shards)
=========================================================
Sequential training of LoRA adapters targeting only layers 13-15.
Estimated time: ~26s per shard = ~7.2 hours for 1000 shards.
"""

import subprocess
import sys
import time
import os
import glob

SHARD_DIR = "/data/mwicker/VPA/data/shards"
OUTPUT_BASE = "/data/mwicker/VPA/output/adapters_last3_lora"
WORKER_SCRIPT = "/data/mwicker/VPA/src/train_last3_lora.py"

# Limit CPU threads per worker
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def train_shard(shard_path, gpu_id="0"):
    shard_id = os.path.basename(shard_path).replace(".jsonl", "")
    output_dir = os.path.join(OUTPUT_BASE, shard_id)
    
    # Check if already completed
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        return "skipped"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    cmd = [
        sys.executable, WORKER_SCRIPT,
        "--input_file", shard_path,
        "--output_dir", output_dir,
        "--epochs", "5"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True)
        return "success"
    except subprocess.CalledProcessError as e:
        print(f"Failed {shard_id}: {e}")
        return "failed"

def main():
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "*.jsonl")))
    print(f"Found {len(shards)} shards.")
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Count existing
    existing = sum(1 for s in shards if os.path.exists(
        os.path.join(OUTPUT_BASE, os.path.basename(s).replace(".jsonl", ""), "adapter_config.json")
    ))
    print(f"Already completed: {existing}/{len(shards)}")
    remaining = len(shards) - existing
    print(f"Remaining: {remaining} shards (~{remaining * 26 / 3600:.1f} hours at 26s/shard)")
    
    start_time = time.time()
    completed = existing
    failed = 0
    
    for i, shard in enumerate(shards):
        shard_id = os.path.basename(shard).replace(".jsonl", "")
        
        result = train_shard(shard)
        
        if result == "success":
            completed += 1
        elif result == "failed":
            failed += 1
        # skipped doesn't change counts
        
        # Progress update every 10 shards
        if (i + 1) % 10 == 0 or i == len(shards) - 1:
            elapsed = time.time() - start_time
            rate = (completed - existing) / elapsed if elapsed > 0 else 0
            eta = (remaining - (completed - existing)) / rate / 3600 if rate > 0 else 0
            print(f"[{time.strftime('%H:%M:%S')}] Progress: {completed}/{len(shards)} | "
                  f"Failed: {failed} | Rate: {rate*60:.1f}/min | ETA: {eta:.1f}h")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total completed: {completed}/{len(shards)}")
    print(f"Failed: {failed}")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()
