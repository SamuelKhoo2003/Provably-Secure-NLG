# 🚀 Project Protocol & Environment Config
**CRITICAL INSTRUCTION FOR AGENT:** Read this file before running any commands or writing code.

> [!CAUTION]
> # 🛑 GPU SAFETY PROTOCOL (STRICT)
> **NEVER** run concurrent training jobs on a single GPU.
> **NEVER** use "job packing" or "process parallelism" for model training on shared servers.
>
> **Reason:** This causes Kernel/Driver Deadlocks (D-state processes) that crash the node and harm other users.
>
> **Allowed Mode:**
> - **SEQUENTIAL ONLY**: `MAX_CONCURRENT_JOBS = 1`
> - **Single Process**: One script per GPU at a time.
> - If you need speed, optimize batch size, NOT concurrency.

## 1. File System Geography (Strict)
Use the repo-local workspace under `/data2/$USER/Projects/Provably-Secure-NLG` and keep shared runtime state inside the repository:

| Purpose | Path | Persistence | Notes |
| :--- | :--- | :--- | :--- |
| **Code & Repos** | `/data2/$USER/Projects/Provably-Secure-NLG` | **Permanent** | **WORKSPACE ROOT.** All git cloning, script writing, and project files MUST go here. |
| **Virtual Env** | `/data2/$USER/Projects/Provably-Secure-NLG/.venv` | **Permanent** | Shared repo-local environment for toy and large experiments. |
| **Caches** | `/data2/$USER/Projects/Provably-Secure-NLG/caches` | **Permanent** | Repo-local pip, torch, HF, and model caches. |

## 2. Environment Activation
Activate the shared repo-local environment in every shell before running Python commands.

```bash
cd /data2/$USER/Projects/Provably-Secure-NLG
python3 -m virtualenv .venv
source .venv/bin/activate

mkdir -p caches/pip-cache caches/torch-cache caches/hf-cache caches/model-cache caches/cache

export PIP_CACHE_DIR="$PWD/caches/pip-cache"
export TORCH_HOME="$PWD/caches/torch-cache"
export HF_HOME="$PWD/caches/hf-cache"
export TRANSFORMERS_CACHE="$PWD/caches/model-cache"
export XDG_CACHE_HOME="$PWD/caches/cache"
export MODEL_CACHE_DIR="$PWD/caches/model-cache"
```
