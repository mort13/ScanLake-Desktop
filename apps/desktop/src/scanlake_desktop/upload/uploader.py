"""Upload queued batches to the Store, with retries and visible failures.

A batch is a list of (capture, revision). Its Parquet files are built when it is sent, from the
captures' current data and the current format, so a batch that waited through a format change
is still valid. After a successful upload each capture is marked uploaded up to the batched
revision; edits made meanwhile simply go out in the next batch (the Refinery keeps the latest).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scanlake_core.domain.capture import Identity
from scanlake_core.format import ParquetFormat, coverage_of_columns
from scanlake_desktop.db.database import BatchRow, Database
from scanlake_io.parquet import FormatMismatch, build_batch
from scanlake_io.store_client import StoreClient, StoreError

log = logging.getLogger(__name__)

FIRST_RETRY_S = 30.0
MAX_RETRY_S = 3600.0


@dataclass(frozen=True)
class UploadOutcome:
    batch_id: int
    session_id: str
    batch_number: int
    ok: bool
    message: str = ""
    violations: tuple[str, ...] = ()


def backoff(attempts: int) -> float:
    return min(MAX_RETRY_S, FIRST_RETRY_S * 2 ** max(attempts, 0))


@dataclass
class Uploader:
    db: Database
    store: StoreClient
    current_format: Callable[[], ParquetFormat]
    identity: Callable[[], Identity]
    export_dir: Callable[[], Path | None] = lambda: None

    def run_due(self, now: float | None = None) -> list[UploadOutcome]:
        return [self.send(batch) for batch in self.db.due_batches(now)]

    def send(self, batch: BatchRow) -> UploadOutcome:
        items = dict(self.db.batch_items(batch.batch_id))
        captures = self.db.live_captures(items)
        fmt = self.current_format()
        if not captures:  # everything in it was deleted before it went out
            self.db.mark_batch_done(batch.batch_id, fmt.version, [])
            return UploadOutcome(batch.batch_id, batch.session_id, batch.batch_number, True)

        scan_columns = {c for cap in captures for c in cap.scan}
        row_columns = sorted({c for cap in captures for row in cap.rows for c in row})
        cov = coverage_of_columns(fmt, scan_columns, row_columns)
        try:
            if not cov.complete:
                raise FormatMismatch(
                    "batch", [f"nothing fills required column {cov.describe_missing()}"]
                )
            identity = self.identity()
            files = build_batch(
                fmt, cov, captures, identity, tuple(row_columns),
                session_id=batch.session_id, batch_number=batch.batch_number,
            )  # fmt: skip
            receipt = self.store.upload(
                identity.user_id, batch.session_id, batch.batch_number, batch.is_final, files
            )
        except FormatMismatch as err:
            return self._failed(batch, f"Format v{fmt.version}: {err}", True, err.violations)
        except StoreError as err:
            delay = max(err.retry_after or 0.0, backoff(batch.attempts))
            return self._failed(batch, str(err), err.permanent, err.violations, delay)
        except Exception as err:  # never let one bad batch stop the queue
            log.exception("upload of batch %s failed", batch.batch_id)
            return self._failed(batch, f"{type(err).__name__}: {err}", False)

        self.db.mark_batch_done(batch.batch_id, fmt.version, receipt.keys)
        self._export(files)
        log.info("uploaded batch %s/%s (%d scans)", batch.session_id, batch.batch_number,
                 len(captures))  # fmt: skip
        return UploadOutcome(batch.batch_id, batch.session_id, batch.batch_number, True)

    def _failed(
        self,
        batch: BatchRow,
        message: str,
        permanent: bool,
        violations: tuple[str, ...] | list[str] = (),
        delay: float | None = None,
    ) -> UploadOutcome:
        next_at = time.time() + (delay if delay is not None else backoff(batch.attempts))
        self.db.mark_batch_failed(
            batch.batch_id, message, next_at, permanent=permanent, violations=violations
        )
        log.warning("upload of batch %s failed: %s", batch.batch_id, message)
        return UploadOutcome(
            batch.batch_id, batch.session_id, batch.batch_number, False, message, tuple(violations)
        )

    def _export(self, files: list) -> None:  # type: ignore[type-arg]
        target = self.export_dir()
        if target is None:
            return
        try:
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                (target / f.filename).write_bytes(f.data)
        except OSError as err:
            log.warning("could not copy batch to %s: %s", target, err)


class FormatHolder:
    """The current format, shared between threads and refreshed every few minutes."""

    def __init__(self, load: Callable[[], ParquetFormat], initial: ParquetFormat) -> None:
        self._load = load
        self._format = initial
        self._loaded_at = time.time()
        self._lock = threading.Lock()

    def get(self) -> ParquetFormat:
        with self._lock:
            return self._format

    def set(self, fmt: ParquetFormat) -> None:
        with self._lock:
            self._format, self._loaded_at = fmt, time.time()

    def fresh(self, max_age_s: float = 300) -> ParquetFormat:
        """The format, reloaded when older than `max_age_s` (call it off the GUI thread)."""
        if time.time() - self._loaded_at > max_age_s:
            try:
                self.set(self._load())
            except Exception as err:
                log.warning("could not refresh the upload format: %s", err)
                with self._lock:
                    self._loaded_at = time.time()
        return self.get()
