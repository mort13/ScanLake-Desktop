"""What is trained: the digit CRNN (CTC) and the word CNN (classification)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from scanlake_core.decode import ctc_greedy
from scanlake_core.preprocess import PREPROCESS_VERSION
from scanlake_models.digit_crnn import BLANK_IDX, CHAR_CLASSES, TIME_STEPS, DigitCRNN
from scanlake_models.word_cnn import WordCNN
from scanlake_training.data import collate_ctc

Batch = Any

# The format regexes the web app's ModelLoader reads from the digit metadata
DIGIT_FORMATS = {
    "decimalPercent": r"^\d{1,2}\.\d{2}%$",
    "decimal": r"^\d{1,3}\.\d{2}$",
    "percent": r"^\d{1,2}%$",
    "integer": r"^\d{1,6}$",
}


class Task(Protocol):
    name: str

    def collate_fn(self) -> Callable[[list[Any]], Batch] | None:
        """The DataLoader collate function (None = PyTorch's default)."""
        ...

    def model(self) -> nn.Module: ...
    def loss(self, model: nn.Module, batch: Batch, device: torch.device) -> Tensor: ...
    def evaluate(
        self, model: nn.Module, loader: DataLoader[Any], device: torch.device
    ) -> dict[str, float]:
        """Metrics on the validation set; `accuracy` decides which epoch is kept."""
        ...

    def meta(self, metrics: dict[str, float]) -> dict[str, Any]:
        """Checkpoint metadata, in the keys of the ONNX exports' JSON."""
        ...


class CrnnTask:
    name = "crnn"

    def collate_fn(self) -> Callable[[list[Any]], Batch] | None:
        return collate_ctc

    def __init__(self, alphabet: str = CHAR_CLASSES) -> None:
        self.alphabet = alphabet
        self.criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    def model(self) -> nn.Module:
        return DigitCRNN(num_classes=len(self.alphabet) + 1)

    def loss(self, model: nn.Module, batch: Batch, device: torch.device) -> Tensor:
        images, targets, input_lengths, target_lengths = (t.to(device) for t in batch)
        log_probs = model(images).float()
        return self.criterion(log_probs, targets, input_lengths, target_lengths)

    @torch.inference_mode()
    def evaluate(
        self, model: nn.Module, loader: DataLoader[Any], device: torch.device
    ) -> dict[str, float]:
        exact = chars = char_total = total = 0
        for images, targets, _, target_lengths in loader:
            log_probs = model(images.to(device)).float().cpu().numpy()
            decoded = ctc_greedy(log_probs, self.alphabet, BLANK_IDX)
            offset = 0
            for result, length in zip(decoded, target_lengths.tolist(), strict=True):
                truth = "".join(self.alphabet[i - 1] for i in targets[offset : offset + length])
                offset += length
                exact += result.text == truth
                chars += sum(p == g for p, g in zip(result.text, truth, strict=False))
                char_total += max(len(result.text), len(truth))
                total += 1
        return {"accuracy": exact / max(total, 1), "char_accuracy": chars / max(char_total, 1)}

    def meta(self, metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "modelType": "crnn",
            "numClasses": len(self.alphabet) + 1,
            "inputShape": [1, 1, 32, 256],
            "outputShape": [TIME_STEPS, 1, len(self.alphabet) + 1],
            "timeSteps": TIME_STEPS,
            "blankIdx": BLANK_IDX,
            "charClasses": self.alphabet,
            "formats": DIGIT_FORMATS,
            "valAccuracy": metrics.get("accuracy", 0.0),
            "preprocess": PREPROCESS_VERSION,
        }


class WordTask:
    name = "word"

    def collate_fn(self) -> Callable[[list[Any]], Batch] | None:
        return None

    def __init__(self, classes: Sequence[str]) -> None:
        self.classes = list(classes)
        self.criterion = nn.CrossEntropyLoss()

    def model(self) -> nn.Module:
        return WordCNN(num_classes=len(self.classes))

    def loss(self, model: nn.Module, batch: Batch, device: torch.device) -> Tensor:
        images, labels = batch
        return self.criterion(model(images.to(device)).float(), labels.to(device))

    @torch.inference_mode()
    def evaluate(
        self, model: nn.Module, loader: DataLoader[Any], device: torch.device
    ) -> dict[str, float]:
        correct = np.zeros(len(self.classes))
        seen = np.zeros(len(self.classes))
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu().numpy()
            truth = labels.numpy()
            np.add.at(seen, truth, 1)
            np.add.at(correct, truth, predicted == truth)
        metrics = {"accuracy": float(correct.sum() / max(seen.sum(), 1))}
        present = seen > 0
        if present.any():
            per_class = np.where(present, correct / np.maximum(seen, 1), 1.0)
            worst = int(per_class.argmin())
            metrics["worst_class_accuracy"] = float(per_class[worst])
            metrics["worst_class"] = worst  # index into classes
        return metrics

    def meta(self, metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "numClasses": len(self.classes),
            "inputShape": [1, 1, 32, 256],
            "wordClasses": self.classes,
            "valAccuracy": metrics.get("accuracy", 0.0),
            "preprocess": PREPROCESS_VERSION,
        }
