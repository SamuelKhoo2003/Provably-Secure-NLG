"""Storage path helpers for large experiment artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_large_output_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve a large-experiment output path with an optional env override.

    Absolute paths are respected. When ``FYP_LARGE_OUTPUT_ROOT`` is unset,
    relative paths keep their local behavior, optionally relative to ``base_dir``.
    When it is set, relative paths are placed below that root.
    """

    configured = Path(path)
    if configured.is_absolute():
        return configured

    output_root = os.environ.get("FYP_LARGE_OUTPUT_ROOT")
    if not output_root:
        return Path(base_dir) / configured if base_dir is not None else configured

    parts = configured.parts
    if parts[:1] == ("large_experiments",):
        suffix = parts[1:]
    else:
        suffix = parts
    return Path(output_root, *suffix)
