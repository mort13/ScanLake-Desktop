"""Export trained checkpoints to ONNX for the web app and the desktop ORT backend, and verify.

The export keeps what the web app's ModelLoader expects: opset 17, input `input`
[batch, 1, 32, 256], output `log_probs` [T, batch, C] (digits) or `logits` [batch, K] (words),
weights in `<name>.onnx.data` next to `<name>.onnx`, metadata in `<name>.json`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scanlake_models import checkpoint

OPSET = 17


def export_onnx(stem: Path, out_stem: Path) -> Path:
    """`<stem>.safetensors` (+ .json) -> `<out_stem>.onnx` + `.onnx.data` + `.json`."""
    import onnx

    model, meta = checkpoint.load(stem)
    kind = checkpoint.kind_of(meta)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = out_stem.with_suffix(".onnx")
    data_name = onnx_path.name + ".data"
    output = "log_probs" if kind == "digit" else "logits"
    output_axes = {1: "batch"} if kind == "digit" else {0: "batch"}

    tmp = onnx_path.with_suffix(".tmp.onnx")
    torch.onnx.export(
        model.cpu().eval(),
        (torch.zeros(1, 1, 32, 256),),
        str(tmp),
        input_names=["input"],
        output_names=[output],
        dynamic_axes={"input": {0: "batch"}, output: output_axes},
        opset_version=OPSET,
        dynamo=False,
    )
    graph = onnx.load(str(tmp))
    tmp.unlink()
    for leftover in tmp.parent.glob(tmp.name + "*"):
        leftover.unlink()
    (onnx_path.parent / data_name).unlink(missing_ok=True)
    onnx.save_model(
        graph,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,
    )

    if stem.with_suffix(".json").resolve() != out_stem.with_suffix(".json").resolve():
        shutil.copyfile(stem.with_suffix(".json"), out_stem.with_suffix(".json"))
    return onnx_path


@dataclass(frozen=True)
class Verification:
    max_abs_diff: float
    same_argmax: bool
    batch_consistent: bool

    @property
    def ok(self) -> bool:
        return self.max_abs_diff < 1e-3 and self.same_argmax and self.batch_consistent


def verify(stem: Path, onnx_stem: Path, batch: int = 8, seed: int = 0) -> Verification:
    """ONNX Runtime output equals PyTorch's, and batch N equals N runs of batch 1."""
    import onnxruntime as ort

    model, meta = checkpoint.load(stem)
    kind = checkpoint.kind_of(meta)
    session = ort.InferenceSession(
        str(onnx_stem.with_suffix(".onnx")), providers=["CPUExecutionProvider"]
    )
    x = np.random.default_rng(seed).random((batch, 1, 32, 256), dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(x)).numpy()
    (got,) = session.run(None, {session.get_inputs()[0].name: x})
    got = np.asarray(got)
    axis = 2 if kind == "digit" else 1
    singles = [np.asarray(session.run(None, {"input": x[i : i + 1]})[0]) for i in range(batch)]
    stacked = np.concatenate(singles, axis=1 if kind == "digit" else 0)
    return Verification(
        max_abs_diff=float(np.abs(expected - got).max()),
        same_argmax=bool((expected.argmax(axis) == got.argmax(axis)).all()),
        batch_consistent=bool(np.allclose(stacked, got, atol=1e-4)),
    )


def describe(stem: Path) -> dict[str, Any]:
    return json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
