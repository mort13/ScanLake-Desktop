"""Checkpoints: weights in `<name>.safetensors`, metadata in `<name>.json`.

The JSON uses the keys of the ONNX exports' metadata (shared with the web app), so one
`digit_crnn.json` describes both `digit_crnn.onnx` and `digit_crnn.safetensors`:

    digit: {"modelType": "crnn", "charClasses": "0123456789.%", "blankIdx": 0, "timeSteps": 64,
            "numClasses": 13, ...}
    word:  {"wordClasses": [...], "numClasses": 31, ...}

safetensors cannot run code when loaded, unlike the pickled `.pth` files of Rock Capture CNN;
those are only read with `weights_only=True` and converted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from scanlake_models.digit_crnn import BLANK_IDX, CHAR_CLASSES, TIME_STEPS, DigitCRNN
from scanlake_models.word_cnn import WordCNN

Kind = Literal["digit", "word"]


def build(kind: Kind, meta: dict[str, Any]) -> nn.Module:
    if kind == "digit":
        return DigitCRNN(num_classes=int(meta.get("numClasses", len(CHAR_CLASSES) + 1)))
    return WordCNN(num_classes=int(meta.get("numClasses", len(meta["wordClasses"]))))


def kind_of(meta: dict[str, Any]) -> Kind:
    return "word" if "wordClasses" in meta else "digit"


CLASS_KEYS = ("charClasses", "wordClasses", "numClasses")


def save(
    model: nn.Module, stem: Path, meta: dict[str, Any], *, overwrite_meta: bool = False
) -> None:
    """Writes `<stem>.safetensors`, and `<stem>.json` unless one already exists.

    An existing JSON (e.g. the pinned metadata of the ONNX export next to it) is left untouched,
    but must describe the same classes as `meta`, or the weights would be read wrongly.
    Training runs pass `overwrite_meta` to update their own JSON (accuracy, epoch) each save.
    """
    meta_path = stem.with_suffix(".json")
    if meta_path.exists() and not overwrite_meta:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in CLASS_KEYS:
            if key in meta and key in existing and existing[key] != meta[key]:
                raise ValueError(f"{meta_path} has different {key}; write to a new name instead")
    stem.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(stem.with_suffix(".safetensors")))
    if overwrite_meta or not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load(stem: Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    model = build(kind_of(meta), meta)
    model.load_state_dict(load_file(str(stem.with_suffix(".safetensors")), device=str(device)))
    return model.to(device).eval(), meta


def read_legacy(path: Path) -> tuple[Kind, dict[str, torch.Tensor], dict[str, Any]]:
    """A Rock Capture CNN `.pth` checkpoint: (kind, state_dict, metadata in the JSON keys)."""
    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    state = raw["model_state_dict"]
    if "word_classes" in raw:
        classes = list(raw["word_classes"])
        meta: dict[str, Any] = {"wordClasses": classes, "numClasses": len(classes)}
        kind: Kind = "word"
    else:
        chars = raw.get("char_classes", CHAR_CLASSES)
        meta = {
            "modelType": "crnn",
            "charClasses": chars,
            "numClasses": int(raw.get("num_classes", len(chars) + 1)),
            "blankIdx": BLANK_IDX,
            "timeSteps": TIME_STEPS,
        }
        kind = "digit"
    if "val_accuracy" in raw:
        meta["valAccuracy"] = float(raw["val_accuracy"])
    return kind, state, meta


def convert_legacy(path: Path, stem: Path) -> dict[str, Any]:
    """Converts a legacy `.pth` to `<stem>.safetensors` + `<stem>.json`; returns the metadata."""
    kind, state, meta = read_legacy(path)
    model = build(kind, meta)
    model.load_state_dict(state)
    save(model, stem, meta)
    return meta
