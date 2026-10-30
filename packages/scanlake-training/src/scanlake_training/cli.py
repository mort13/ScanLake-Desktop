"""`scanlake-train`: synthetic data, training, export and evaluation of the ScanLake models.

    scanlake-train synth strips OUT [--per-format 300] [--seed N]
    scanlake-train synth words OUT [--per-class 200] [--seed N]
    scanlake-train train crnn --data DIR [--val-data DIR] [--out STEM] [--epochs 60] [--json]
    scanlake-train train word --data DIR [--val-data DIR] [--out STEM] [--epochs 60] [--json]
    scanlake-train export STEM OUT_STEM            # safetensors -> ONNX (+ verify)
    scanlake-train evaluate MODEL --data DIR       # accuracy of a .onnx or .safetensors model
    scanlake-train checkpoint convert OLD.pth OUT_STEM
    scanlake-train checkpoint show STEM

Without --val-data, a fixed 15 % of the training folder (chosen by file name hash) validates.
Use --val-data with real captures (the desktop app's correction crops) for honest numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from scanlake_models import checkpoint


def _print_event(as_json: bool):  # type: ignore[no-untyped-def]
    def report(event: dict[str, Any]) -> None:
        if as_json:
            print(json.dumps(event), flush=True)
        elif event["event"] == "start":
            print(
                f"{event['task']}: {event['train']} train / {event['val']} val samples on "
                f"{event['device']}, {event['epochs']} epochs",
                flush=True,
            )
        elif event["event"] == "epoch":
            mark = "  saved" if event["saved"] else ""
            print(
                f"epoch {event['epoch']:3d}/{event['epochs']}  loss {event['loss']:.4f}  "
                f"val {event['accuracy']:.3f}  best {event['best']:.3f}  "
                f"({event['seconds']:.0f}s){mark}",
                flush=True,
            )
        elif event["event"] == "done":
            print(
                f"best val accuracy {event['best_accuracy']:.3f} (epoch {event['best_epoch']}) "
                f"-> {event['out']}.safetensors ({event['stopped']})",
                flush=True,
            )

    return report


def _synth(args: argparse.Namespace) -> int:
    from scanlake_training import synth

    if args.what == "strips":
        n = synth.generate_strips(Path(args.out), args.per_format, args.seed)
        print(f"wrote {n} strips to {args.out}")
    else:
        counts = synth.generate_words(Path(args.out), args.per_class, args.seed)
        print(f"wrote {sum(counts.values())} images in {len(counts)} classes to {args.out}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from scanlake_training import data
    from scanlake_training.tasks import CrnnTask, WordTask
    from scanlake_training.trainer import TrainConfig, train

    folder = Path(args.data)
    if args.task == "crnn":
        samples = data.strip_samples(folder)
        task: Any = CrnnTask()
        name = "digit_crnn"
    else:
        classes = data.word_classes(folder)
        if len(classes) < 2:
            raise SystemExit(f"need at least 2 class folders in {folder}")
        samples = data.word_samples(folder, classes)
        task = WordTask(classes)
        name = "word_model"
    if args.val_data:
        train_samples = samples
        val_samples = (
            data.strip_samples(Path(args.val_data))
            if args.task == "crnn"
            else data.word_samples(Path(args.val_data), task.classes)
        )
    else:
        train_samples, val_samples = data.split(samples, args.val_fraction)
    if not train_samples or not val_samples:
        raise SystemExit(f"not enough samples in {folder} ({len(samples)} found)")

    # Small folders are repeated so an epoch still has a useful number of updates
    oversample = max(1, (400 if args.task == "crnn" else 200) // len(train_samples))
    if args.task == "crnn":
        train_set: Any = data.StripDataset(train_samples, augment=True, oversample=oversample)
        val_set: Any = data.StripDataset(val_samples)
    else:
        train_set = data.WordDataset(
            train_samples, task.classes, augment=True, oversample=oversample
        )
        val_set = data.WordDataset(val_samples, task.classes)

    out = (
        Path(args.out)
        if args.out
        else Path("runs") / f"{args.task}-{time.strftime('%Y%m%d-%H%M%S')}" / name
    )
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
    )
    result = train(task, train_set, val_set, out, config, _print_event(args.json))
    return 0 if result.best_accuracy >= 0 else 1


def _export(args: argparse.Namespace) -> int:
    from scanlake_training.export import export_onnx, verify

    path = export_onnx(Path(args.stem), Path(args.out))
    check = verify(Path(args.stem), Path(args.out))
    print(f"wrote {path} (+ .onnx.data, .json)")
    print(
        f"verify: max |diff| {check.max_abs_diff:.2e}, same arg-max {check.same_argmax}, "
        f"batch-consistent {check.batch_consistent}"
    )
    return 0 if check.ok else 1


def _evaluate(args: argparse.Namespace) -> int:
    from scanlake_training.evaluate import evaluate_model

    result = evaluate_model(Path(args.model), Path(args.data))
    if args.json:
        print(json.dumps(result))
    else:
        print(
            f"{result['model']}: {result['correct']}/{result['total']} correct "
            f"= {result['accuracy']:.3f}"
        )
        for item in result["errors"][:15]:
            print(f"  {item['file']}: expected {item['expected']!r}, read {item['got']!r}")
    return 0


def _convert(args: argparse.Namespace) -> int:
    meta = checkpoint.convert_legacy(Path(args.source), Path(args.out))
    print(f"wrote {args.out}.safetensors (metadata: {args.out}.json)")
    print(json.dumps(meta, indent=2))
    return 0


def _show(args: argparse.Namespace) -> int:
    model, meta = checkpoint.load(Path(args.stem))
    print(json.dumps(meta, indent=2))
    print(f"{type(model).__name__}: {sum(p.numel() for p in model.parameters()):,} parameters")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scanlake-train", description="ScanLake model tooling.")
    sub = parser.add_subparsers(dest="group", required=True)

    s = sub.add_parser("synth", help="generate synthetic training data")
    s.add_argument("what", choices=["strips", "words"])
    s.add_argument("out")
    s.add_argument("--per-format", type=int, default=300, help="strips per format")
    s.add_argument("--per-class", type=int, default=200, help="word images per class")
    s.add_argument("--seed", type=int)
    s.set_defaults(run=_synth)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("task", choices=["crnn", "word"])
    t.add_argument("--data", required=True, help="training folder")
    t.add_argument("--val-data", help="separate validation folder (e.g. real captures)")
    t.add_argument("--val-fraction", type=float, default=0.15)
    t.add_argument("--out", help="output path without extension (default: runs/<task>-<time>/...)")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    t.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    t.add_argument("--patience", type=int, default=0, help="early stop after N flat epochs")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--json", action="store_true", help="print progress as JSON lines")
    t.set_defaults(run=_train)

    e = sub.add_parser("export", help="safetensors checkpoint -> ONNX, then verify")
    e.add_argument("stem")
    e.add_argument("out")
    e.set_defaults(run=_export)

    v = sub.add_parser("evaluate", help="accuracy of a model on a labelled folder")
    v.add_argument("model", help="model path without extension (.onnx or .safetensors)")
    v.add_argument("--data", required=True)
    v.add_argument("--json", action="store_true")
    v.set_defaults(run=_evaluate)

    c = sub.add_parser("checkpoint", help="convert and inspect checkpoints")
    csub = c.add_subparsers(dest="command", required=True)
    convert = csub.add_parser("convert", help="legacy Rock Capture CNN .pth -> safetensors")
    convert.add_argument("source")
    convert.add_argument("out", help="output path without extension")
    convert.set_defaults(run=_convert)
    show = csub.add_parser("show", help="print a checkpoint's metadata")
    show.add_argument("stem")
    show.set_defaults(run=_show)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
