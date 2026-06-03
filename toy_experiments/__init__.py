"""First-party toy row/column certificate experiment package.

The package is intentionally self-contained: ``data`` generates synthetic
shard-level token votes, ``milp`` solves shared poisoned-shard allocation
certificates, and ``experiments`` provides CLI workflows, baselines, CSV output,
and plots. External reference code lives under ``phd_reference/`` and is not part
of this package API.
"""

from .data import ToyData, generate_toy_votes

__all__ = ["ToyData", "generate_toy_votes"]
