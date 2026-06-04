"""Configuration for VPA token vote export integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from large_experiments.storage import resolve_large_output_path


@dataclass(frozen=True)
class VPAIntegrationConfig:
    """Paths and runtime knobs for a future VPA vote exporter.

    The fields are intentionally plain Python types so this module can be
    imported in environments that do not have model or optimization packages.
    """

    base_dir: Path
    vpa_dir: Path
    dataset_dir: Path
    test_path: Path
    adapter_dir: Path
    output_dir: Path
    model_name: str
    num_shards: int
    num_samples: int
    max_new_tokens: int
    device: str
    dtype: str
    stability_horizon: int
    validity_targets_path: Path | None = None

    @classmethod
    def from_base_dir(
        cls,
        base_dir: str | Path,
        *,
        model_name: str = "allenai/OLMo-2-0425-1B-Instruct",
        num_shards: int = 500,
        num_samples: int = 100,
        max_new_tokens: int = 60,
        device: str = "cuda",
        dtype: str = "float16",
        stability_horizon: int = 60,
        validity_targets_path: str | Path | None = None,
        vpa_dir: str | Path | None = None,
        adapter_subdir: str = "output/adapters_last3_lora",
    ) -> "VPAIntegrationConfig":
        """Build a config using the large-experiment repository layout.

        ``vpa_dir`` can point to any external VPA-main checkout. If omitted, the
        optional checkout location under ``large_experiments/vpa/external`` is
        used.
        """

        root = Path(base_dir)
        resolved_vpa_dir = Path(vpa_dir) if vpa_dir is not None else root / "large_experiments" / "vpa" / "external" / "VPA-main"
        dataset_dir = resolved_vpa_dir / "data"
        return cls(
            base_dir=root,
            vpa_dir=resolved_vpa_dir,
            dataset_dir=dataset_dir,
            test_path=dataset_dir / "test.jsonl",
            adapter_dir=resolved_vpa_dir / adapter_subdir,
            output_dir=resolve_large_output_path("large_experiments/vpa/outputs", base_dir=root),
            model_name=model_name,
            num_shards=num_shards,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            device=device,
            dtype=dtype,
            stability_horizon=stability_horizon,
            validity_targets_path=Path(validity_targets_path) if validity_targets_path is not None else None,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary."""

        return {
            "base_dir": str(self.base_dir),
            "vpa_dir": str(self.vpa_dir),
            "dataset_dir": str(self.dataset_dir),
            "test_path": str(self.test_path),
            "adapter_dir": str(self.adapter_dir),
            "output_dir": str(self.output_dir),
            "model_name": self.model_name,
            "num_shards": self.num_shards,
            "num_samples": self.num_samples,
            "max_new_tokens": self.max_new_tokens,
            "device": self.device,
            "dtype": self.dtype,
            "stability_horizon": self.stability_horizon,
            "validity_targets_path": str(self.validity_targets_path) if self.validity_targets_path is not None else None,
        }
