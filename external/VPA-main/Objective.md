
I. High-Level Objective
Demonstrate that by increasing the number of shards to $K=1000$, we can provide formal guarantees for a poisoning budget $k$ that is an order of magnitude larger than current baselines, specifically for Tool-Use (MCP) scenarios.

II. Experiment Design: "The Thousand-Shard Trial"
1. Data & Model Selection
* Dataset: Use the Toucan dataset (released late 2025/early 2026), which contains ~1.5 million trajectories synthesized from nearly 500 real-world Model Context Protocols (MCPs).
    * SFT Prep: Subsample ~100,000 tool-use instances to ensure each of your 1000 shards contains exactly 100 samples.
* Model: Qwen2.5-1.5B or OLMo-1B. A smaller model is critical to completing 1000 training runs within 4 hours; these models are highly capable for tool-calling after alignment.
2. Distributed Sharding & Training (2x L40)
Training 1000 models sequentially is inefficient. Instead, use a Multi-Adapter Batching strategy:
* Setup: Partition the data into 1000 disjoint shards $\{\mathcal{D}_1, \dots, \mathcal{D}_{1000}\}$.
* Parallel Training: Run two parallel processes (one per L40). Each GPU will handle 500 shards.
* PEFT Strategy: Use LoRA with a low rank ($r=8$). This results in adapter files of ~10MB each.
* Throughput Goal: With only 100 samples per shard, 1 epoch of SFT should take ~10–15 seconds per shard. Total training time for 1000 shards: $\approx 500 \text{ shards} \times 15 \text{s} = 125 \text{ minutes}$ (approx. 2.1 hours).
3. Implementation of VPA
Once the 1000 adapters are trained, implement the Valid Partition Aggregation (VPA) algorithm based on the paper's Theorem 1:
* Aggregation: For a given tool-call prompt, generate the next token from all 1000 models.
* Vote Counting: Count votes $v_{c_1}, v_{c_2}, \dots$ for each token.
* Certification: Apply the recurrence from Theorem 1 to calculate the Targeted Validity Radius $r_t$. Note: With $K=1000$, a tool-call can be certified even if an adversary modifies up to ~250–400 training points (depending on the vote margin), a massive jump from the $k=9$ budget in the paper. 

III. Briefing for Antigravity
Project Alpha-K: Scaling Provable Security to $K=1000$
* The Problem: Current certified defenses only handle tiny poisoning budgets (e.g., $k < 10$), which are unrealistic for large-scale data.
* The Innovation: We are moving to a Massive Sharding paradigm. By using 1000 shards, we force an attacker to poison a significantly larger percentage of the global dataset to flip a single tool-call.
* Technical Fit: * Tooling: We utilize the Model Context Protocol (MCP) to standardize the "harmful actions" we want to prevent (e.g., preventing a model from calling rm -rf even if the training data is poisoned).
    * Efficiency: By using PEFT (LoRA), we keep the storage overhead for 1000 models under 10GB total, making multi-adapter inference feasible on standard L40 nodes.
* Expected Result: A formal certificate proving that no adversary with a budget of $k=100$ (or higher) can force the model to execute a specific unauthorized MCP tool call.
