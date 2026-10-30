"""Which inference backends this installation can run, and creating one by id.

Backend ids (also the values of the `backend` setting):

    ort-cpu     ONNX Runtime on the CPU (always available)
    ort-dml     ONNX Runtime on the GPU via DirectML (Windows, any DX12 GPU)
    torch-cpu   PyTorch on the CPU          (full build / dev env only)
    torch-cuda  PyTorch on an NVIDIA GPU    (full build / dev env only)

`auto` is resolved by `bench.pick_fastest`.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from scanlake_core.recognition import Engine
from scanlake_inference.metadata import ModelFiles

AUTO = "auto"
BACKEND_IDS = ("ort-cpu", "ort-dml", "torch-cpu", "torch-cuda")
LABELS = {
    "ort-cpu": "ONNX Runtime (CPU)",
    "ort-dml": "ONNX Runtime (GPU, DirectML)",
    "torch-cpu": "PyTorch (CPU)",
    "torch-cuda": "PyTorch (GPU, CUDA)",
}


@dataclass(frozen=True)
class BackendInfo:
    id: str
    label: str
    available: bool
    reason: str = ""  # why not available


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def available_backends(models: Path | None = None) -> list[BackendInfo]:
    """What can run here; with `models`, PyTorch also needs `.safetensors` weights there."""
    from scanlake_inference.ort_backend import dml_available

    infos = [BackendInfo("ort-cpu", LABELS["ort-cpu"], True)]
    dml = dml_available()
    infos.append(
        BackendInfo("ort-dml", LABELS["ort-dml"], dml, "" if dml else "DirectML is not available")
    )
    if not torch_installed():
        reason = "PyTorch is not installed (full build only)"
        infos += [BackendInfo(i, LABELS[i], False, reason) for i in ("torch-cpu", "torch-cuda")]
        return infos
    files = ModelFiles(models) if models is not None else None
    if files is not None and not all(
        (files.directory / f"{name}.safetensors").exists() for name in (files.digit, files.word)
    ):
        reason = "no PyTorch weights (run: scanlake-train checkpoint convert)"
        infos += [BackendInfo(i, LABELS[i], False, reason) for i in ("torch-cpu", "torch-cuda")]
        return infos
    infos.append(BackendInfo("torch-cpu", LABELS["torch-cpu"], True))
    import torch

    cuda = torch.cuda.is_available()
    infos.append(
        BackendInfo("torch-cuda", LABELS["torch-cuda"], cuda, "" if cuda else "no CUDA GPU found")
    )
    return infos


def create_engine(
    backend_id: str, models: Path | ModelFiles, *, device_id: int = 0, threads: int = 2
) -> Engine:
    files = models if isinstance(models, ModelFiles) else ModelFiles(models)
    if backend_id == "ort-cpu":
        from scanlake_inference.ort_backend import OrtEngine

        return OrtEngine(files, "cpu", threads=threads)
    if backend_id == "ort-dml":
        from scanlake_inference.ort_backend import OrtEngine

        return OrtEngine(files, "dml", device_id=device_id, threads=threads)
    if backend_id in ("torch-cpu", "torch-cuda"):
        from scanlake_inference.torch_backend import TorchEngine

        return TorchEngine(files, "cuda" if backend_id == "torch-cuda" else "cpu")
    raise ValueError(f"unknown backend {backend_id!r}")
