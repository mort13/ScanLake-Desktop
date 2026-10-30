"""Training samples and datasets.

Folder layouts (the same as Rock Capture CNN, so existing data keeps working):

    strips:  <folder>/<label>_<n>.png          e.g. 42.50_000123.png   (label before the last _)
    words:   <folder>/<class>/<anything>.png   e.g. quartz/synth_00012.png

Images go through the inference preprocessing (`scanlake_core.preprocess.resize_pad`), so
training and recognition see identical inputs.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from scanlake_core.preprocess import Gray, resize_pad
from scanlake_models.digit_crnn import CHAR_CLASSES, TIME_STEPS
from scanlake_training.augment import STRIP, WORD, Preset, augment

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def load_gray(path: Path) -> Gray:
    """Grayscale image (imdecode: works with non-ASCII Windows paths)."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return np.asarray(image)


def _images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def strip_samples(folder: Path, alphabet: str = CHAR_CLASSES) -> list[Sample]:
    samples = []
    for path in _images(folder):
        label, sep, _ = path.stem.rpartition("_")
        if sep and label and all(c in alphabet for c in label):
            samples.append(Sample(path, label))
    return samples


def word_classes(folder: Path) -> list[str]:
    return sorted(d.name for d in folder.iterdir() if d.is_dir()) if folder.exists() else []


def word_samples(folder: Path, classes: Sequence[str] | None = None) -> list[Sample]:
    names = list(classes) if classes is not None else word_classes(folder)
    return [Sample(p, name) for name in names for p in _images(folder / name)]


def split(
    samples: Sequence[Sample], val_fraction: float, salt: str = "scanlake"
) -> tuple[list[Sample], list[Sample]]:
    """Deterministic train/validation split by file name hash.

    A file always lands on the same side, whatever else is in the folder, so adding data never
    moves old validation samples into training (the old word trainer validated on its training set).
    """
    train: list[Sample] = []
    val: list[Sample] = []
    for s in samples:
        digest = hashlib.sha1(f"{salt}/{s.label}/{s.path.name}".encode()).digest()
        (val if int.from_bytes(digest[:8], "big") / 2**64 < val_fraction else train).append(s)
    return train, val


class _ImageDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        augment: bool,
        preset: Preset,
        oversample: int = 1,
        seed: int = 0,
    ) -> None:
        self.samples = list(samples) * max(1, oversample)
        self.augment = augment
        self.preset = preset
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def _canvas(self, index: int) -> Tensor:
        canvas = resize_pad(load_gray(self.samples[index].path))
        if self.augment:
            # torch's RNG advances per call (and is seeded per DataLoader worker and epoch), so
            # every epoch sees new augmentations while torch.manual_seed keeps runs reproducible
            draw = int(torch.randint(0, 2**31 - 1, (1,)).item())
            canvas = augment(canvas, random.Random(f"{self.seed}-{index}-{draw}"), self.preset)
        return torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)


class StripDataset(_ImageDataset):
    """Digit strips -> (image (1, 32, 256), target indices 1..12)."""

    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        augment: bool = False,
        oversample: int = 1,
        alphabet: str = CHAR_CLASSES,
        seed: int = 0,
    ) -> None:
        super().__init__(samples, augment=augment, preset=STRIP, oversample=oversample, seed=seed)
        self.char_to_index = {c: i + 1 for i, c in enumerate(alphabet)}

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        label = self.samples[index].label
        target = torch.tensor([self.char_to_index[c] for c in label], dtype=torch.long)
        return self._canvas(index), target


class WordDataset(_ImageDataset):
    """Word images -> (image (1, 32, 256), class index)."""

    def __init__(
        self,
        samples: Sequence[Sample],
        classes: Sequence[str],
        *,
        augment: bool = False,
        oversample: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(samples, augment=augment, preset=WORD, oversample=oversample, seed=seed)
        self.class_to_index = {c: i for i, c in enumerate(classes)}

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        label = self.class_to_index[self.samples[index].label]
        return self._canvas(index), torch.tensor(label, dtype=torch.long)


def collate_ctc(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """images (N,1,H,W), flat targets, input lengths (all T), target lengths — for nn.CTCLoss."""
    images, labels = zip(*batch, strict=True)
    return (
        torch.stack(images),
        torch.cat(labels),
        torch.full((len(images),), TIME_STEPS, dtype=torch.long),
        torch.tensor([len(lb) for lb in labels], dtype=torch.long),
    )
