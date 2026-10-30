"""Runs the uploader on its own thread: every few seconds, or right away when woken."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from scanlake_core.domain.capture import Identity
from scanlake_desktop.db.database import Database
from scanlake_desktop.upload.uploader import FormatHolder, Uploader
from scanlake_io.store_client import StoreClient

log = logging.getLogger(__name__)

POLL_S = 5.0


class UploadWorker(QThread):
    changed = Signal(object)  # UploadSummary
    finished_batch = Signal(object)  # UploadOutcome

    def __init__(
        self,
        db_path: Path,
        store: StoreClient,
        formats: FormatHolder,
        identity: Callable[[], Identity],
        export_dir: Callable[[], Path | None],
        enabled: Callable[[], bool],
    ) -> None:
        super().__init__()
        self._db_path = db_path
        self._store = store
        self._formats = formats
        self._identity = identity
        self._export_dir = export_dir
        self._enabled = enabled
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._force = False

    def wake(self, force: bool = False) -> None:
        """Check the queue now; `force` uploads even when automatic upload is off."""
        self._force = self._force or force
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self.wait(5000)

    def run(self) -> None:
        db = Database(self._db_path)  # sqlite connections stay in their thread
        uploader = Uploader(
            db, self._store, lambda: self._formats.fresh(), self._identity, self._export_dir
        )
        try:
            while not self._stop.is_set():
                force, self._force = self._force, False
                if force:
                    db.retry_failed()
                if self._enabled() or force:
                    for outcome in uploader.run_due():
                        self.finished_batch.emit(outcome)
                self.changed.emit(db.upload_summary())
                self._wake.wait(POLL_S)
                self._wake.clear()
        except Exception:
            log.exception("upload worker crashed")
        finally:
            db.close()
