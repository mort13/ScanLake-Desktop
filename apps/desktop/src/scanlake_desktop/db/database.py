"""Local SQLite store: sessions, clusters, captures and the upload queue.

Capture values are stored per (kind, row, column), not in fixed columns, so a format that
gains a column needs no migration here. sqlite3 connections are per thread: the GUI and the
upload worker each open their own `Database` on the same file (WAL mode).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from scanlake_core.domain.capture import Capture, FieldValue, utc_timestamp
from scanlake_core.schema_map import COMPOSITIONS, SCANS

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    system       TEXT NOT NULL DEFAULT '',
    gravity_well TEXT NOT NULL DEFAULT '',
    region       TEXT NOT NULL DEFAULT '',
    ship         TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closed')),
    created_at   TEXT NOT NULL,
    closed_at    TEXT
);
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions (session_id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, seq)
);
CREATE TABLE IF NOT EXISTS captures (
    capture_id        TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions (session_id) ON DELETE CASCADE,
    cluster_id        TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    context           TEXT NOT NULL DEFAULT '{}',
    models            TEXT NOT NULL DEFAULT '',
    revision          INTEGER NOT NULL DEFAULT 1,
    uploaded_revision INTEGER NOT NULL DEFAULT 0,
    deleted           INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS captures_by_session ON captures (session_id, timestamp);
CREATE TABLE IF NOT EXISTS capture_values (
    capture_id  TEXT NOT NULL REFERENCES captures (capture_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    row_index   INTEGER NOT NULL,  -- -1 for per-capture (scans) values
    column_name TEXT NOT NULL,
    value       TEXT NOT NULL,
    confidence  REAL,
    source      TEXT NOT NULL,
    ocr_value   TEXT,
    PRIMARY KEY (capture_id, kind, row_index, column_name)
);
CREATE TABLE IF NOT EXISTS upload_batches (
    batch_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    batch_number    INTEGER NOT NULL,
    is_final        INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'done', 'failed')),
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    format_version  INTEGER,
    last_error      TEXT,
    violations      TEXT,
    r2_keys         TEXT,
    created_at      TEXT NOT NULL,
    uploaded_at     TEXT,
    UNIQUE (session_id, batch_number)
);
CREATE TABLE IF NOT EXISTS batch_items (
    batch_id   INTEGER NOT NULL REFERENCES upload_batches (batch_id) ON DELETE CASCADE,
    capture_id TEXT NOT NULL,
    revision   INTEGER NOT NULL,
    PRIMARY KEY (batch_id, capture_id)
);
"""


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    system: str
    gravity_well: str
    region: str
    ship: str
    status: str
    created_at: str
    closed_at: str | None = None
    scans: int = 0
    pending: int = 0  # captures not uploaded in their latest revision


@dataclass(frozen=True)
class BatchRow:
    batch_id: int
    session_id: str
    batch_number: int
    is_final: bool
    status: str
    attempts: int
    next_attempt_at: float
    last_error: str | None
    violations: tuple[str, ...]


@dataclass(frozen=True)
class UploadSummary:
    pending_batches: int
    failed_batches: int
    unbatched_captures: int
    last_error: str | None
    next_attempt_at: float | None


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # -- sessions -----------------------------------------------------------

    def create_session(self, system: str, gravity_well: str, region: str, ship: str) -> SessionRow:
        session_id = str(uuid.uuid4())
        now = utc_timestamp()
        with self._tx() as db:
            db.execute(
                "INSERT INTO sessions (session_id, system, gravity_well, region, ship, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, system, gravity_well, region, ship, now),
            )
            self._insert_cluster(db, session_id, 1)
        return self.session(session_id)  # type: ignore[return-value]

    def session(self, session_id: str) -> SessionRow | None:
        rows = self._sessions("WHERE s.session_id = ?", (session_id,))
        return rows[0] if rows else None

    def active_session(self) -> SessionRow | None:
        rows = self._sessions("WHERE s.status = 'active' ORDER BY s.created_at DESC LIMIT 1", ())
        return rows[0] if rows else None

    def sessions(self, limit: int = 50) -> list[SessionRow]:
        return self._sessions("ORDER BY s.created_at DESC LIMIT ?", (limit,))

    def _sessions(self, where: str, args: tuple[object, ...]) -> list[SessionRow]:
        query = f"""
            SELECT s.*,
                   (SELECT COUNT(*) FROM captures c
                     WHERE c.session_id = s.session_id AND c.deleted = 0) AS scans,
                   (SELECT COUNT(*) FROM captures c
                     WHERE c.session_id = s.session_id AND c.deleted = 0
                       AND c.revision > c.uploaded_revision) AS pending
            FROM sessions s {where}"""
        return [
            SessionRow(
                r["session_id"],
                r["system"],
                r["gravity_well"],
                r["region"],
                r["ship"],
                r["status"],
                r["created_at"],
                r["closed_at"],
                r["scans"],
                r["pending"],
            )  # fmt: skip
            for r in self._db.execute(query, args)
        ]

    def update_session_location(
        self, session_id: str, system: str, gravity_well: str, region: str
    ) -> None:
        self._db.execute(
            "UPDATE sessions SET system = ?, gravity_well = ?, region = ? WHERE session_id = ?",
            (system, gravity_well, region, session_id),
        )

    def close_session(self, session_id: str) -> None:
        self._db.execute(
            "UPDATE sessions SET status = 'closed', closed_at = ? WHERE session_id = ?",
            (utc_timestamp(), session_id),
        )

    # -- clusters -----------------------------------------------------------

    def clusters(self, session_id: str) -> list[str]:
        rows = self._db.execute(
            "SELECT cluster_id FROM clusters WHERE session_id = ? ORDER BY seq", (session_id,)
        )
        return [r["cluster_id"] for r in rows]

    def new_cluster(self, session_id: str) -> str:
        with self._tx() as db:
            seq = db.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM clusters WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            return self._insert_cluster(db, session_id, seq)

    @staticmethod
    def _insert_cluster(db: sqlite3.Connection, session_id: str, seq: int) -> str:
        cluster_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO clusters (cluster_id, session_id, seq, created_at) VALUES (?, ?, ?, ?)",
            (cluster_id, session_id, seq, utc_timestamp()),
        )
        return cluster_id

    def cluster_deposit(self, cluster_id: str, exclude_capture: str | None = None) -> str | None:
        """The deposit of the cluster's first saved scan (all scans of a rock share it)."""
        row = self._db.execute(
            """SELECT v.value FROM capture_values v JOIN captures c USING (capture_id)
               WHERE c.cluster_id = ? AND c.deleted = 0 AND c.capture_id IS NOT ?
                 AND v.kind = ? AND v.row_index = -1 AND v.column_name = 'deposit'
                 AND v.value != ''
               ORDER BY c.timestamp LIMIT 1""",
            (cluster_id, exclude_capture, SCANS),
        ).fetchone()
        return row["value"] if row else None

    # -- captures -----------------------------------------------------------

    def save_capture(self, capture: Capture, models: str = "") -> int:
        """Inserts or replaces `capture`; returns its new revision."""
        now = utc_timestamp()
        with self._tx() as db:
            existing = db.execute(
                "SELECT revision FROM captures WHERE capture_id = ?", (capture.capture_id,)
            ).fetchone()
            revision = (existing["revision"] + 1) if existing else 1
            if existing:
                db.execute(
                    "UPDATE captures SET cluster_id = ?, context = ?, revision = ?, deleted = 0,"
                    " updated_at = ? WHERE capture_id = ?",
                    (capture.cluster_id, json.dumps(capture.context), revision, now,
                     capture.capture_id),
                )  # fmt: skip
                db.execute("DELETE FROM capture_values WHERE capture_id = ?", (capture.capture_id,))
            else:
                db.execute(
                    "INSERT INTO captures (capture_id, session_id, cluster_id, timestamp, context,"
                    " models, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (capture.capture_id, capture.session_id, capture.cluster_id, capture.timestamp,
                     json.dumps(capture.context), models, now, now),
                )  # fmt: skip
            db.executemany(
                "INSERT INTO capture_values VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                list(_value_rows(capture)),
            )
        return revision

    def delete_capture(self, capture_id: str) -> bool:
        """Marks a capture deleted; returns True when an uploaded version stays on the server."""
        row = self._db.execute(
            "SELECT uploaded_revision FROM captures WHERE capture_id = ?", (capture_id,)
        ).fetchone()
        self._db.execute(
            "UPDATE captures SET deleted = 1, updated_at = ? WHERE capture_id = ?",
            (utc_timestamp(), capture_id),
        )
        return bool(row and row["uploaded_revision"] > 0)

    def capture(self, capture_id: str) -> Capture | None:
        found = self._captures("WHERE capture_id = ?", (capture_id,))
        return found[0] if found else None

    def live_captures(self, capture_ids: Iterable[str]) -> list[Capture]:
        """The not-deleted captures among `capture_ids`, oldest first."""
        ids = list(capture_ids)
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        return self._captures(
            f"WHERE capture_id IN ({marks}) AND deleted = 0 ORDER BY timestamp", tuple(ids)
        )

    def captures(self, session_id: str) -> list[Capture]:
        return self._captures(
            "WHERE session_id = ? AND deleted = 0 ORDER BY timestamp", (session_id,)
        )

    def _captures(self, where: str, args: tuple[object, ...]) -> list[Capture]:
        heads = self._db.execute(f"SELECT * FROM captures {where}", args).fetchall()
        if not heads:
            return []
        ids = [h["capture_id"] for h in heads]
        marks = ",".join("?" * len(ids))
        values: dict[str, list[sqlite3.Row]] = {i: [] for i in ids}
        for v in self._db.execute(
            f"SELECT * FROM capture_values WHERE capture_id IN ({marks})", ids
        ):
            values[v["capture_id"]].append(v)
        return [_capture_from(h, values[h["capture_id"]]) for h in heads]

    def upload_state(self, session_id: str) -> dict[str, str]:
        """capture_id -> 'uploaded' | 'pending' (latest revision not uploaded yet)."""
        rows = self._db.execute(
            "SELECT capture_id, revision, uploaded_revision FROM captures WHERE session_id = ?",
            (session_id,),
        )
        return {
            r["capture_id"]: "uploaded" if r["uploaded_revision"] >= r["revision"] else "pending"
            for r in rows
        }

    # -- upload queue -------------------------------------------------------

    def unbatched(self, session_id: str) -> list[tuple[str, int]]:
        """Captures whose latest revision is neither uploaded nor waiting in a pending batch."""
        rows = self._db.execute(
            """SELECT c.capture_id, c.revision FROM captures c
               WHERE c.session_id = ? AND c.deleted = 0 AND c.revision > c.uploaded_revision
                 AND NOT EXISTS (
                   SELECT 1 FROM batch_items i JOIN upload_batches b USING (batch_id)
                   WHERE i.capture_id = c.capture_id AND i.revision = c.revision
                     AND b.status = 'pending')
               ORDER BY c.timestamp""",
            (session_id,),
        )
        return [(r["capture_id"], r["revision"]) for r in rows]

    def create_batch(
        self, session_id: str, items: Iterable[tuple[str, int]], is_final: bool
    ) -> int | None:
        """Queues `items` as the session's next batch; returns its batch number."""
        items = list(items)
        if not items and not is_final:
            return None
        with self._tx() as db:
            number = db.execute(
                "SELECT COALESCE(MAX(batch_number), 0) + 1 FROM upload_batches"
                " WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            cur = db.execute(
                "INSERT INTO upload_batches (session_id, batch_number, is_final, created_at)"
                " VALUES (?, ?, ?, ?)",
                (session_id, number, int(is_final), utc_timestamp()),
            )
            db.executemany(
                "INSERT INTO batch_items VALUES (?, ?, ?)",
                [(cur.lastrowid, capture_id, rev) for capture_id, rev in items],
            )
        return number

    def due_batches(self, now: float | None = None) -> list[BatchRow]:
        rows = self._db.execute(
            "SELECT * FROM upload_batches WHERE status = 'pending' AND next_attempt_at <= ?"
            " ORDER BY batch_id",
            (now if now is not None else time.time(),),
        )
        return [_batch_from(r) for r in rows]

    def batch_items(self, batch_id: int) -> list[tuple[str, int]]:
        rows = self._db.execute(
            "SELECT capture_id, revision FROM batch_items WHERE batch_id = ?", (batch_id,)
        )
        return [(r["capture_id"], r["revision"]) for r in rows]

    def mark_batch_done(self, batch_id: int, format_version: int, keys: Iterable[str]) -> None:
        with self._tx() as db:
            db.execute(
                "UPDATE upload_batches SET status = 'done', uploaded_at = ?, format_version = ?,"
                " r2_keys = ?, last_error = NULL, violations = NULL,"
                " attempts = attempts + 1 WHERE batch_id = ?",
                (utc_timestamp(), format_version, json.dumps(list(keys)), batch_id),
            )
            db.execute(
                """UPDATE captures SET uploaded_revision = MAX(uploaded_revision, (
                     SELECT i.revision FROM batch_items i
                     WHERE i.batch_id = ? AND i.capture_id = captures.capture_id))
                   WHERE capture_id IN (SELECT capture_id FROM batch_items WHERE batch_id = ?)""",
                (batch_id, batch_id),
            )

    def mark_batch_failed(
        self,
        batch_id: int,
        error: str,
        next_attempt_at: float,
        *,
        permanent: bool,
        violations: Iterable[str] = (),
    ) -> None:
        self._db.execute(
            "UPDATE upload_batches SET status = ?, attempts = attempts + 1, last_error = ?,"
            " next_attempt_at = ?, violations = ? WHERE batch_id = ?",
            ("failed" if permanent else "pending", error, next_attempt_at,
             json.dumps(list(violations)), batch_id),
        )  # fmt: skip

    def retry_failed(self) -> None:
        """Puts permanently failed batches back in the queue (after the cause was fixed)."""
        self._db.execute(
            "UPDATE upload_batches SET status = 'pending', next_attempt_at = 0"
            " WHERE status = 'failed'"
        )

    def upload_summary(self) -> UploadSummary:
        row = self._db.execute(
            """SELECT
                 SUM(status = 'pending') AS pending, SUM(status = 'failed') AS failed,
                 MIN(CASE WHEN status = 'pending' THEN next_attempt_at END) AS next_at
               FROM upload_batches"""
        ).fetchone()
        last = self._db.execute(
            "SELECT last_error FROM upload_batches"
            " WHERE status != 'done' AND last_error IS NOT NULL ORDER BY batch_id DESC LIMIT 1"
        ).fetchone()
        unbatched = self._db.execute(
            """SELECT COUNT(*) FROM captures c WHERE c.deleted = 0
                 AND c.revision > c.uploaded_revision
                 AND NOT EXISTS (SELECT 1 FROM batch_items i JOIN upload_batches b
                   USING (batch_id) WHERE i.capture_id = c.capture_id
                   AND i.revision = c.revision AND b.status != 'done')"""
        ).fetchone()[0]
        return UploadSummary(
            pending_batches=row["pending"] or 0,
            failed_batches=row["failed"] or 0,
            unbatched_captures=unbatched,
            last_error=last["last_error"] if last else None,
            next_attempt_at=row["next_at"],
        )


def _value_rows(capture: Capture) -> Iterator[tuple[object, ...]]:
    for column, v in capture.scan.items():
        yield (capture.capture_id, SCANS, -1, column, v.text, v.confidence, v.source, v.ocr_text)
    for index, row in enumerate(capture.rows):
        for column, v in row.items():
            yield (
                capture.capture_id, COMPOSITIONS, index, column, v.text, v.confidence, v.source,
                v.ocr_text,
            )  # fmt: skip


def _capture_from(head: sqlite3.Row, values: list[sqlite3.Row]) -> Capture:
    capture = Capture(
        capture_id=head["capture_id"],
        session_id=head["session_id"],
        cluster_id=head["cluster_id"],
        timestamp=head["timestamp"],
        context=json.loads(head["context"] or "{}"),
    )
    n_rows = 1 + max((v["row_index"] for v in values), default=-1)
    capture.rows = [{} for _ in range(n_rows)]
    for v in values:
        value = FieldValue(v["value"], v["confidence"], v["source"], v["ocr_value"])
        if v["row_index"] < 0:
            capture.scan[v["column_name"]] = value
        else:
            capture.rows[v["row_index"]][v["column_name"]] = value
    return capture


def _batch_from(r: sqlite3.Row) -> BatchRow:
    return BatchRow(
        batch_id=r["batch_id"],
        session_id=r["session_id"],
        batch_number=r["batch_number"],
        is_final=bool(r["is_final"]),
        status=r["status"],
        attempts=r["attempts"],
        next_attempt_at=r["next_attempt_at"],
        last_error=r["last_error"],
        violations=tuple(json.loads(r["violations"] or "[]")),
    )
