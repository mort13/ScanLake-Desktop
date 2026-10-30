"""Entry point: `scanlake-desktop [--dev] [--replay FOLDER]`."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from pathlib import Path


def _setup_logging() -> None:
    from scanlake_desktop import paths

    handler = logging.handlers.RotatingFileHandler(
        paths.logs_dir() / "scanlake-desktop.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[handler, logging.StreamHandler()],
    )


def _install_crash_handler() -> None:
    """Unhandled errors are logged and shown, instead of the app vanishing silently."""
    import threading

    from scanlake_desktop import paths

    log = logging.getLogger("scanlake_desktop")

    def handle(kind: type[BaseException], error: BaseException, tb: object) -> None:
        log.critical("unhandled error", exc_info=(kind, error, tb))  # type: ignore[arg-type]
        from PySide6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is not None:
            message = f"Something went wrong: {error}\n\nDetails are in {paths.logs_dir()}."
            QMessageBox.critical(None, "ScanLake Desktop", message)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        info = (args.exc_type, args.exc_value, args.exc_traceback)
        log.critical("unhandled error in %s", args.thread, exc_info=info)  # type: ignore[arg-type]

    sys.excepthook = handle
    threading.excepthook = thread_hook


def self_test(report_path: Path) -> int:
    """Loads the assets and runs each available ONNX Runtime engine (packaging smoke test)."""
    import json
    import platform
    import traceback

    report: dict[str, object] = {
        "python": platform.python_version(),
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    ok = True
    try:
        from scanlake_core.format import parse_format
        from scanlake_desktop import paths
        from scanlake_desktop.assets import AssetSet
        from scanlake_inference.bench import time_engine
        from scanlake_inference.registry import available_backends, create_engine

        assets = AssetSet(paths.bundled_assets())
        report["assets"] = str(assets.root)
        report["ships"] = sorted(assets.ships())
        fmt = parse_format(json.loads(assets.bundled_format.read_text(encoding="utf-8")))
        report["format_version"] = fmt.version
        engines: dict[str, object] = {}
        for backend in available_backends():
            if not backend.available or not backend.id.startswith("ort"):
                engines[backend.id] = backend.reason or "skipped"
                continue
            try:
                engine = create_engine(backend.id, assets.models_dir)
                engines[backend.id] = f"{time_engine(engine, 5):.1f} ms/frame"
            except Exception as err:
                ok = False
                engines[backend.id] = f"FAILED: {err}"
        report["engines"] = engines
        from PySide6 import __version__ as pyside_version

        report["pyside6"] = pyside_version
    except Exception:
        ok = False
        report["error"] = traceback.format_exc()
    report["ok"] = ok
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    import multiprocessing

    # In the frozen app, DataLoader worker processes re-run this exe: let them run their task
    # instead of starting the app again (a no-op when not frozen)
    multiprocessing.freeze_support()
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--train-cli"]:  # dev mode runs training through the app (frozen too)
        from scanlake_training.cli import main as train_main

        return train_main(argv[1:])
    parser = argparse.ArgumentParser(prog="scanlake-desktop")
    parser.add_argument("--dev", action="store_true", help="enable developer mode")
    parser.add_argument("--replay", type=Path, help="read frames from a folder of images")
    parser.add_argument(
        "--self-test", type=Path, metavar="REPORT", help="check assets and engines, write a report"
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test(args.self_test)

    from scanlake_desktop.settings import peek_backend

    # PyTorch must be imported before Qt on Windows, or its DLLs fail to load
    if peek_backend().startswith("torch") or os.environ.get("SCANLAKE_PRELOAD_TORCH"):
        try:
            import torch  # noqa: F401
        except ImportError:
            pass

    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    from PySide6.QtWidgets import QApplication

    _setup_logging()
    _install_crash_handler()
    app = QApplication(sys.argv)
    app.setApplicationName("ScanLake Desktop")
    app.setOrganizationName("ScanLake")

    # One instance only: a second start just brings the first one to the front
    socket = QLocalSocket()
    socket.connectToServer("ScanLakeDesktop")
    if socket.waitForConnected(300):
        socket.write(b"show")
        socket.waitForBytesWritten(300)
        return 0
    QLocalServer.removeServer("ScanLakeDesktop")
    server = QLocalServer()
    server.listen("ScanLakeDesktop")

    from scanlake_desktop import paths
    from scanlake_desktop.app import DesktopApp
    from scanlake_desktop.settings import load_settings
    from scanlake_desktop.ui.style import STYLESHEET

    app.setStyleSheet(STYLESHEET)
    replay = sorted(args.replay.glob("*.png")) if args.replay else None
    first_run = not paths.settings_file().exists()
    desktop = DesktopApp(load_settings(), dev=args.dev, replay=replay, first_run=first_run)

    def show_existing() -> None:
        connection = server.nextPendingConnection()
        if connection is not None:
            connection.close()
        desktop.window.showNormal()
        desktop.window.raise_()
        desktop.window.activateWindow()

    server.newConnection.connect(show_existing)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
