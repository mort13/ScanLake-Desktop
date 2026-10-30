"""One training loop for every task (replaces Rock Capture CNN's three copies).

Adam, optional early stopping, mixed precision on CUDA (`torch.amp`), and the best epoch by
validation accuracy saved as `<out>.safetensors` + `<out>.json`. Progress goes to a `report`
callback as plain dicts; the CLI prints them as JSON lines that the dev-mode launcher reads.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch.utils.data import DataLoader, Dataset

from scanlake_models import checkpoint
from scanlake_training.tasks import Task

Report = Callable[[dict[str, Any]], None]


class SizedDataset(Protocol):
    """What train() takes: an indexable dataset with a length (both datasets in data.py)."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "auto"  # auto | cpu | cuda
    workers: int = 0
    seed: int = 42
    patience: int = 0  # stop after this many epochs without improvement (0 = never)


@dataclass(frozen=True)
class TrainResult:
    out: Path
    best_accuracy: float
    best_epoch: int
    epochs_run: int
    stopped: str  # "finished" | "early" | "cancelled"


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(
    task: Task,
    train_set: SizedDataset,
    val_set: SizedDataset,
    out: Path,
    config: TrainConfig | None = None,
    report: Report = lambda event: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> TrainResult:
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = pick_device(config.device)
    cuda = device.type == "cuda"
    loader_args: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "collate_fn": task.collate_fn(),
        "pin_memory": cuda,
        "persistent_workers": config.workers > 0,
    }
    train_loader = DataLoader(cast(Dataset[Any], train_set), shuffle=True, **loader_args)
    val_loader = DataLoader(cast(Dataset[Any], val_set), shuffle=False, **loader_args)

    model = task.model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    report(
        {
            "event": "start",
            "task": task.name,
            "device": str(device),
            "train": len(train_set),
            "val": len(val_set),
            "epochs": config.epochs,
        }
    )  # type: ignore[arg-type]

    best, best_epoch, since_best, stopped, epoch = -1.0, 0, 0, "finished", 0
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        total, batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=cuda):
                loss = task.loss(model, batch, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if math.isfinite(loss.item()):
                total += loss.item()
            batches += 1
            if cancelled():
                stopped = "cancelled"
                break
        model.eval()
        metrics = task.evaluate(model, val_loader, device)
        improved = metrics["accuracy"] > best
        if improved:
            best, best_epoch, since_best = metrics["accuracy"], epoch, 0
            meta = {**task.meta(metrics), "epoch": epoch, "trainSamples": len(train_set)}  # type: ignore[arg-type]
            checkpoint.save(model, out, meta, overwrite_meta=True)
        else:
            since_best += 1
        report(
            {
                "event": "epoch",
                "epoch": epoch,
                "epochs": config.epochs,
                "loss": total / max(batches, 1),
                **metrics,
                "best": best,
                "saved": improved,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        if stopped == "cancelled" or cancelled():
            stopped = "cancelled"
            break
        if config.patience and since_best >= config.patience:
            stopped = "early"
            break

    result = TrainResult(out, best, best_epoch, epoch, stopped)
    report(
        {
            "event": "done",
            "out": str(out),
            "best_accuracy": best,
            "best_epoch": best_epoch,
            "stopped": stopped,
        }
    )
    return result
