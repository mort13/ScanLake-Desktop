"""ONNX Runtime engine: DirectML (any DX12 GPU) or CPU, same .onnx files as the web app."""

from __future__ import annotations

from typing import Literal

import numpy as np
import onnxruntime as ort

from scanlake_core.decode import Floats
from scanlake_inference.metadata import ModelFiles, load_digit_meta, load_word_meta

Provider = Literal["cpu", "dml"]
DML = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"


def dml_available() -> bool:
    return DML in ort.get_available_providers()


class OrtEngine:
    """Runs the digit CRNN and the word CNN on batches of [N, 1, 32, 256] inputs."""

    def __init__(
        self, files: ModelFiles, provider: Provider = "cpu", device_id: int = 0, threads: int = 2
    ) -> None:
        self.provider = provider
        self._digit_meta = load_digit_meta(files.meta(files.digit))
        self._word_meta = load_word_meta(files.meta(files.word))
        self._digit = self._session(str(files.onnx(files.digit)), provider, device_id, threads)
        self._word = self._session(str(files.onnx(files.word)), provider, device_id, threads)

    @staticmethod
    def _session(
        path: str, provider: Provider, device_id: int, threads: int
    ) -> ort.InferenceSession:
        options = ort.SessionOptions()
        # Leave CPU for the game: these models are small, more threads barely help
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers: list[str | tuple[str, dict[str, object]]]
        if provider == "dml":
            # DirectML requires these (memory patterns and parallel execution are unsupported)
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [(DML, {"device_id": device_id}), CPU]
        else:
            providers = [CPU]
        session = ort.InferenceSession(path, sess_options=options, providers=providers)
        if provider == "dml" and session.get_providers()[0] != DML:
            raise RuntimeError("DirectML provider could not be initialised")
        return session

    @property
    def name(self) -> str:
        return f"ort-{self.provider}"

    @property
    def digit_classes(self) -> str:
        return self._digit_meta.char_classes

    @property
    def blank_index(self) -> int:
        return self._digit_meta.blank_index

    @property
    def word_classes(self) -> tuple[str, ...]:
        return self._word_meta.word_classes

    def digits(self, batch: Floats) -> Floats:
        (out,) = self._digit.run(None, {self._digit.get_inputs()[0].name: _contiguous(batch)})
        return np.asarray(out, dtype=np.float32)

    def words(self, batch: Floats) -> Floats:
        (out,) = self._word.run(None, {self._word.get_inputs()[0].name: _contiguous(batch)})
        return np.asarray(out, dtype=np.float32)


def _contiguous(batch: Floats) -> Floats:
    return np.ascontiguousarray(batch, dtype=np.float32)
