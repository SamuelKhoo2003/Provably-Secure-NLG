
import subprocess
import sys
import time
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import queue

SHARD_DIR = "/data/mwicker/VPA/data/shards"
OUTPUT_BASE = "/data/mwicker/VPA/output/adapters"
WORKER_SCRIPT = "/data/mwicker/VPA/src/train_worker.py"
MAX_CONCURRENT_JOBS = 1 # Back to 1 GPU for training (GPU 0)

# Thread-safe GPU Queue
gpu_queue = queue.Queue()
gpu_queue.put("0")
# gpu_queue.put("1") # Reserved for VPA Benchmarking

# Limit CPU threads per worker
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def train_shard(shard_path):
    shard_id = os.path.basename(shard_path).replace(".jsonl", "")
    output_dir = os.path.join(OUTPUT_BASE, shard_id)
    
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        print(f"Skipping {shard_id}, already exists.")
        return

    # Acquire GPU
    gpu_id = gpu_queue.get()
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        cmd = [
            sys.executable, WORKER_SCRIPT,
            "--input_file", shard_path,
            "--output_dir", output_dir,
            "--epochs", "5"
        ]
        
        # Run synchronously within this thread (but concurrent with others)
        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"Finished {shard_id} on GPU {gpu_id}")
        except subprocess.CalledProcessError as e:
            print(f"Failed {shard_id} on GPU {gpu_id}: {e}")
            
    finally:
        # Always return GPU
        gpu_queue.put(gpu_id)

def main():
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "*.jsonl")))
    print(f"Found {len(shards)} shards.")
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    print(f"Starting execution with {MAX_CONCURRENT_JOBS} concurrent jobs on GPUs [0, 1]...")
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = [executor.submit(train_shard, shard) for shard in shards]
        
        # Monitor Loop
        completed = 0
        total = len(shards)
        while completed < total:
            completed = sum(1 for f in futures if f.done())
            print(f"Progress: {completed}/{total} ({completed/total:.1%})", end="\r")
            time.sleep(10)
            
    print("\nAll jobs submitted/completed.")

if __name__ == "__main__":
    main()
