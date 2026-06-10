# Provably-Secure-NLG

EIE Final Year Project 2026.

This repository studies poisoning certificates for partition-aggregated natural
language generation. It contains:

- controlled synthetic experiments under `toy_experiments/`;
- full-scale certification of stored VPA vote-vector outputs under
  `large_experiments/`;
- vendored or external model-training code under `external/`.

The maintained certification code is offline: it consumes generated votes and
does not retrain or poison language models.

## Repository Layout

```text
toy_experiments/       Synthetic data, baselines, MILPs, plots, and sweeps
large_experiments/     Full-scale vote-vector certification and plotting
external/              External training/evaluation code and diagram utilities
requirements.txt       Python dependencies
```

Detailed instructions:

- [Toy experiment documentation](toy_experiments/toy_experiment_README.md)
- [Large experiment documentation](large_experiments/large_experiments_README.md)

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The maintained shell scripts automatically prefer `.venv/bin/python`, then
fall back to `python3`. Set `PYTHON_BIN` to select another interpreter.

Gurobi and a valid Gurobi licence are required for MILP solves. Plotting and
the maintained unit tests do not run Gurobi.

## Method Taxonomy

Stability is evaluated against all competing tokens. Report-facing stability is
an untargeted any-token-change property; there is no runner-up-only stability
mode.

TPA terminology is deliberately narrow:

- The toy count-based baseline is **TPA max-token phrase baseline**.
- The full-scale count-based baseline is **aggregate TPA final-tool validity**.
- Neither method solves an MILP, uses shard identities, or enforces a shared
  poisoned-shard allocation.
- Collective TPA+MSC is not implemented.

The main full-scale comparison uses two interfaces:

- **DPA final-tool stability** and **aggregate TPA final-tool validity** use
  final tool-call vote counts from `vote_vector`.
- **Joint row-column stability MILP** and **joint row-column validity MILP**
  use the shard-aware prompt-token grid extracted from `token_vote_matrix`.

Token-grid DPA curves are optional diagnostics and are excluded from the
default full-scale summary.

## Toy Workflow

Run a standard benchmark:

```bash
CONFIG=toy_experiments/configs/medium.yaml ./toy_experiments/scripts/data.sh
```

Plot existing standard benchmark CSVs:

```bash
./toy_experiments/scripts/plot.sh
```

Run the controlled validity demo:

```bash
./toy_experiments/scripts/validity_demo.sh
```

Synthetic scaling experiments use coupled master instances. For each fixed
distribution-parameter tuple, one maximum-size vote structure is generated and
smaller `K`, `N`, `L`, and `T` points are derived from it. Coupling improves
comparability but does not imply monotonicity for every objective.

## Full-Scale Workflow

The full-scale runner consumes existing VPA JSONL rows containing
`vote_vector`, `token_vote_matrix`, `vote_counts`, and `majority`.

```bash
.venv/bin/python large_experiments/scripts/certify_vote_vectors_runner.py \
  --input /path/to/vote_vectors.jsonl \
  --name example_run \
  --horizon 20 \
  --budgets 0,1,3,5,7,9 \
  --top-competitors 1 \
  --max-targets-per-prompt 2 \
  --threads 8 \
  --quiet-gurobi \
  --output-dir large_experiments/outputs/certification
```

The runner always uses Gurobi for the joint MILPs. Rows are retained only when
every shard has a non-`None` prefix at least as long as the selected horizon.
Short generations are filtered rather than padded.

Plot one or more completed runs:

```bash
.venv/bin/python large_experiments/scripts/plot_certification_curves.py \
  --inputs large_experiments/outputs/certification/example_run/H020 \
  --labels "Example H=20" \
  --output-dir large_experiments/outputs/certification/plots/example \
  --filename-prefix example
```

The full-scale plotter writes PDF curves and CSV summaries. It validates current
method names strictly and rejects legacy method names rather than remapping
them.

## Tests

Run the maintained solver-free and utility tests from the repository root:

```bash
MPLCONFIGDIR=/tmp/provably-secure-mpl \
XDG_CACHE_HOME=/tmp/provably-secure-cache \
.venv/bin/python -m unittest discover -v
```

Run a syntax and bytecode compilation check:

```bash
.venv/bin/python -m compileall -q toy_experiments large_experiments
```

## Generated Artifacts

Generated CSV, JSON, PDF, and model-output files live under the corresponding
`outputs/` directories. Certification and plotting should operate on existing
vote data; they do not require rerunning model training or generation.

## AI-Assisted Development Disclosure

Codex 5.5 was used to support code generation, refactoring, documentation, and
infrastructure setup. All generated or modified code was critically reviewed
and tested before inclusion.
