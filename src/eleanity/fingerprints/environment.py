from __future__ import annotations

import platform
from importlib import metadata
from typing import Iterable

from eleanity.models.schemas import EnvironmentFingerprint

# Packages that influence runtime parity diagnostics.
TRACKED_PACKAGES: tuple[str, ...] = (
    "transformers",
    "tokenizers",
    "torch",
    "vllm",
    "llama-cpp-python",
    "accelerate",
    "bitsandbytes",
    "httpx",
    "pydantic",
    "eleanity",
)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _collect_packages(names: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    return {name: _package_version(name) for name in names}


def collect_environment_fingerprint() -> EnvironmentFingerprint:
    """Snapshot host/runtime identity used for reproducibility notes."""

    cuda_available: bool | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    torch_version: str | None = None

    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_version = getattr(torch.version, "cuda", None)
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:  # pragma: no cover - depends on driver state
                gpu_name = None
    except ImportError:
        pass

    return EnvironmentFingerprint(
        python_version=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or None,
        packages=_collect_packages(),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        torch_version=torch_version,
    )
