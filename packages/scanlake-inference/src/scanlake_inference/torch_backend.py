"""PyTorch engine: runs `<model>.safetensors` checkpoints on CUDA or the CPU.

Only in the full build and the dev environment. Loads checkpoints straight from training, so a
new architecture can be tried without exporting to ONNX first.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from scanlake_core.decode import Floats
from scanlake_core.names import NameScorer, lexicon
from scanlake_core.preprocess import preprocess_of
from scanlake_inference.metadata import ModelFiles
from scanlake_models import checkpoint


class TorchEngine:
    def __init__(self, files: ModelFiles, device: Literal["cpu", "cuda"] = "cpu") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.device = torch.device(device)
        self._digit, digit_meta = checkpoint.load(files.directory / files.digit, self.device)
        self._word, word_meta = checkpoint.load(files.directory / files.names_model, self.device)
        self._digit_classes = str(digit_meta["charClasses"])
        self._blank = int(digit_meta.get("blankIdx", 0))
        self._names: NameScorer | None = None
        if files.reads_names:  # the letter model (see OrtEngine)
            chars, blank = str(word_meta["charClasses"]), int(word_meta.get("blankIdx", 0))
            self._names = NameScorer(lexicon(), chars, blank)
            self._word_classes = self._names.lexicon.classes
        else:
            self._word_classes = tuple(word_meta["wordClasses"])
        self._digit_preprocess = preprocess_of(digit_meta)
        self._word_preprocess = preprocess_of(word_meta)

    @property
    def name(self) -> str:
        return f"torch-{self.device.type}"

    @property
    def digit_classes(self) -> str:
        return self._digit_classes

    @property
    def blank_index(self) -> int:
        return self._blank

    @property
    def word_classes(self) -> tuple[str, ...]:
        return self._word_classes

    @property
    def digit_preprocess(self) -> str:
        return self._digit_preprocess

    @property
    def word_preprocess(self) -> str:
        return self._word_preprocess

    @torch.inference_mode()
    def digits(self, batch: Floats) -> Floats:
        out = self._digit(torch.from_numpy(np.ascontiguousarray(batch)).to(self.device))
        return out.float().cpu().numpy()

    @torch.inference_mode()
    def words(self, batch: Floats) -> Floats:
        out = self._word(torch.from_numpy(np.ascontiguousarray(batch)).to(self.device))
        out = out.float().cpu().numpy()
        return self._names.scores(out) if self._names is not None else out
