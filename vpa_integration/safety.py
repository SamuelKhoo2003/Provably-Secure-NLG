"""Generalized cluster safety checks for VPA integration."""

from __future__ import annotations

import getpass
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAX_CONCURRENT_JOBS = 1
CONCURRENCY_MODE = "sequential"
HOME_WRITE_FORBIDDEN = True


@dataclass(frozen=True)
class SafetyMessage:
    """One safety check message."""

    level: str
    field: str
    path: str
    message: str


@dataclass(frozen=True)
class SafetyReport:
    """Result of validating configured paths."""

    cluster_username: str | None
    messages: list[SafetyMessage]

    @property
    def has_errors(self) -> bool:
        return any(message.level == "error" for message in self.messages)

    @property
    def warnings(self) -> list[SafetyMessage]:
        return [message for message in self.messages if message.level == "warning"]

    @property
    def errors(self) -> list[SafetyMessage]:
        return [message for message in self.messages if message.level == "error"]

    def as_dict(self) -> dict[str, object]:
        return {
            "cluster_username": self.cluster_username,
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "concurrency_mode": CONCURRENCY_MODE,
            "home_write_forbidden": HOME_WRITE_FORBIDDEN,
            "messages": [message.__dict__ for message in self.messages],
        }


def infer_cluster_username(
    *,
    cluster_username: str | None = None,
    paths: Iterable[str | Path] = (),
) -> str | None:
    """Infer a cluster username from explicit input, paths, or environment."""

    if cluster_username:
        return cluster_username
    for raw_path in paths:
        parts = Path(raw_path).parts
        for marker in ("/data", "/homes"):
            if len(parts) >= 3 and parts[0] == "/" and parts[1] == marker.strip("/"):
                return parts[2]
    try:
        username = getpass.getuser()
    except Exception:
        return None
    return username or None


def validate_configured_paths(
    *,
    base_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    adapter_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    cluster_username: str | None = None,
) -> SafetyReport:
    """Validate configured paths against generalized cluster safety rules."""

    path_items = {
        "base_dir": base_dir,
        "dataset_dir": dataset_dir,
        "adapter_dir": adapter_dir,
        "output_dir": output_dir,
    }
    username = infer_cluster_username(cluster_username=cluster_username, paths=[path for path in path_items.values() if path is not None])
    messages: list[SafetyMessage] = []

    for field, raw_path in path_items.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        path_str = str(path)
        if is_forbidden_home_path(path, username):
            messages.append(
                SafetyMessage(
                    level="error",
                    field=field,
                    path=path_str,
                    message="cluster home paths under /homes are forbidden for project data, datasets, adapters, and outputs",
                )
            )
            continue
        if is_cluster_path(path):
            preferred = _preferred_prefix_for_field(field, username)
            if preferred is not None and not _is_relative_to(path, preferred):
                messages.append(
                    SafetyMessage(
                        level="warning",
                        field=field,
                        path=path_str,
                        message=f"cluster path is outside preferred root {preferred}",
                    )
                )
        else:
            messages.append(
                SafetyMessage(
                    level="warning",
                    field=field,
                    path=path_str,
                    message="non-cluster/local path detected; acceptable for development but use configured /data roots for shared-server runs",
                )
            )

    return SafetyReport(cluster_username=username, messages=messages)


def safety_metadata(cluster_username: str | None = None, *, real_inference_enabled: bool = False) -> dict[str, object]:
    """Return common safety fields for export metadata."""

    return {
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "concurrency_mode": CONCURRENCY_MODE,
        "home_write_forbidden": HOME_WRITE_FORBIDDEN,
        "cluster_username": cluster_username,
        "real_inference_enabled": real_inference_enabled,
    }


def is_forbidden_home_path(path: Path, cluster_username: str | None = None) -> bool:
    """Return True for /homes paths, with or without a known username."""

    parts = path.parts
    if len(parts) >= 2 and parts[0] == "/" and parts[1] == "homes":
        return True
    return False


def is_cluster_path(path: Path) -> bool:
    parts = path.parts
    return len(parts) >= 2 and parts[0] == "/" and parts[1] in {"data", "vol", "homes"}


def _preferred_prefix_for_field(field: str, cluster_username: str | None) -> Path | None:
    if cluster_username is None:
        return None
    if field == "dataset_dir":
        return Path("/data") / cluster_username / "datasets"
    if field == "output_dir":
        return Path("/data") / cluster_username / "output"
    if field in {"base_dir", "adapter_dir"}:
        return Path("/data") / cluster_username
    return None


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True
