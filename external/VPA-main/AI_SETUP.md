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
We are running on a cluster with strict storage quotas. You must respect these paths:

| Purpose | Path | Persistence | Notes |
| :--- | :--- | :--- | :--- |
| **Code & Repos** | `/data/mwicker/` | **Permanent** | **WORKSPACE ROOT.** All git cloning, script writing, and project files MUST go here. |
| **Datasets** | `/data/mwicker/datasets/` | **Permanent** | Store heavy raw data here. |
| **Experiments** | `/data/mwicker/output/` | **Permanent** | Save model checkpoints, logs, and graphs here. |
| **Virtual Env** | `/vol/bitbucket/mwicker/antigravity-env` | **Ephemeral** | The python environment lives here. |
| **Home (`~`)** | `/homes/mwicker` | **FORBIDDEN** | **DO NOT WRITE HERE.** 2GB Limit. Writing here will crash the session. |

---

## 2. Environment Activation
The system Python is broken. You **must** activate the virtual environment for every new terminal session or execution context.

**Activation Command:**
```bash
source /vol/bitbucket/mwicker/antigravity-env/bin/activate
