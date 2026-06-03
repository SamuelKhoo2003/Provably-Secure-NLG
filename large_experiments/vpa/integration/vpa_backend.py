"""Real-inference scaffold for VPA adapter token votes.

This module intentionally performs no model loading at import time. Heavy
dependencies are imported only inside explicit runtime methods so discovery,
validation, and ordinary package imports remain lightweight.
"""

from __future__ import annotations

from pathlib import Path

from .backends import TokenVoteBackend, VoteRequest
from .safety import SafetyReport, validate_configured_paths


ADAPTER_STRATEGY_TRANSFORMERS = "transformers_adapter_methods"
REQUIRED_ADAPTER_METHODS = ("load_adapter", "set_adapter", "delete_adapter")


class AdapterCompatibilityError(RuntimeError):
    """Raised when the runtime model cannot load adapters as expected."""


class VPAAdapterBackend:
    """Future backend for VPA shard adapter inference.

    The backend evaluates shards sequentially. It loads one adapter, computes a
    single next-token argmax for the supplied prefix, deletes that adapter, and
    moves to the next shard. It does not call model.generate and does not train.
    """

    name = "vpa"

    def __init__(
        self,
        *,
        adapter_dir: str | Path,
        model_name: str,
        test_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        cluster_username: str | None = None,
        device: str = "cuda",
        dtype: str = "float16",
        real_inference_enabled: bool = False,
    ) -> None:
        self.adapter_dir = Path(adapter_dir)
        self.model_name = model_name
        self.test_path = Path(test_path) if test_path is not None else None
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.cluster_username = cluster_username
        self.device = device
        self.dtype = dtype
        self.real_inference_enabled = real_inference_enabled
        self.adapter_strategy = ADAPTER_STRATEGY_TRANSFORMERS
        self.model = None
        self.tokenizer = None

    def safety_report(self) -> SafetyReport:
        return validate_configured_paths(
            adapter_dir=self.adapter_dir,
            output_dir=self.output_dir,
            dataset_dir=self.test_path.parent if self.test_path is not None else None,
            cluster_username=self.cluster_username,
        )

    def discover_shards(self) -> list[str]:
        """Return shard adapter directory names in deterministic order."""

        if not self.adapter_dir.exists():
            return []
        return sorted(path.name for path in self.adapter_dir.iterdir() if path.is_dir() and path.name.startswith("shard_"))

    def load_base_model_and_tokenizer(self) -> None:
        """Load the base causal LM and tokenizer.

        Heavy imports are intentionally local to this method.
        """

        if not self.real_inference_enabled:
            raise NotImplementedError("Real VPA model loading requires explicit real_inference_enabled=True")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = _torch_dtype(torch, self.dtype)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype)
        model.to(self.device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

    def load_adapter_for_shard(self, shard_id: str) -> None:
        """Load one shard adapter onto the already-loaded base model."""

        if not self.real_inference_enabled:
            raise NotImplementedError("Real adapter loading requires explicit real_inference_enabled=True")
        if self.model is None:
            raise RuntimeError("Base model must be loaded before loading adapters")
        self._check_adapter_api()
        adapter_path = self.adapter_dir / shard_id
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory not found for shard {shard_id}: {adapter_path}")
        self.model.load_adapter(str(adapter_path), adapter_name=shard_id)
        self.model.set_adapter(shard_id)

    def predict_next_token_for_shards(self, request: VoteRequest, shard_ids: list[str]) -> list[int]:
        """Compute one next-token argmax per shard, sequentially.

        The real implementation stays single-process and evaluates one shard at
        a time. It does not batch shards and does not keep all adapters loaded.
        """

        if not self.real_inference_enabled:
            raise NotImplementedError("Real VPA inference requires explicit real_inference_enabled=True")
        if not shard_ids:
            return []
        if self.model is None or self.tokenizer is None:
            self.load_base_model_and_tokenizer()
        self._check_adapter_api()

        token_ids: list[int] = []
        for shard_id in shard_ids:
            self.load_adapter_for_shard(shard_id)
            token_ids.append(self._predict_next_token(request.prefix_token_ids))
            self._delete_adapter(shard_id)
        return token_ids

    def _predict_next_token(self, prefix_token_ids: list[int]) -> int:
        if self.model is None:
            raise RuntimeError("Base model is not loaded")
        if not prefix_token_ids:
            raise ValueError("prefix_token_ids must be non-empty")
        import torch

        input_ids = torch.tensor([prefix_token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            return int(torch.argmax(logits, dim=-1).item())

    def _delete_adapter(self, shard_id: str) -> None:
        if self.model is None:
            return
        self.model.delete_adapter(shard_id)

    def _check_adapter_api(self) -> None:
        if self.model is None:
            raise RuntimeError("Base model is not loaded")
        missing = [method_name for method_name in REQUIRED_ADAPTER_METHODS if not hasattr(self.model, method_name)]
        if missing:
            missing_text = ", ".join(missing)
            required_text = ", ".join(REQUIRED_ADAPTER_METHODS)
            raise AdapterCompatibilityError(
                "Adapter API compatibility check failed: "
                f"strategy={self.adapter_strategy!r} requires model methods [{required_text}], "
                f"but missing [{missing_text}]. This runtime does not expose the expected adapter API."
            )

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "adapter_strategy": self.adapter_strategy,
            "adapter_dir": str(self.adapter_dir),
            "model_name": self.model_name,
            "device": self.device,
            "dtype": self.dtype,
            "real_inference_enabled": self.real_inference_enabled,
        }


def assert_backend_protocol(_: TokenVoteBackend) -> None:
    """Static helper for type checkers; no runtime behavior."""


def _torch_dtype(torch_module: object, dtype: str) -> object:
    if dtype == "float16":
        return torch_module.float16
    if dtype == "bfloat16":
        return torch_module.bfloat16
    if dtype == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported dtype for real VPA backend: {dtype}")
