"""Sessions, clusters and saving captures (runs on the GUI thread, owns its DB connection)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from scanlake_core.domain.capture import Capture, FieldValue, Row, utc_timestamp
from scanlake_core.domain.validation import Issue, is_duplicate, validate
from scanlake_core.format import Coverage, Source
from scanlake_core.schema_map import SCANS, SlotLayout
from scanlake_desktop.db.database import Database, SessionRow


@dataclass
class SaveResult:
    saved: bool
    issues: list[Issue] = field(default_factory=list)
    duplicate: bool = False
    capture_id: str = ""
    queued_batch: int | None = None


@dataclass
class SessionService:
    db: Database
    batch_size: int = 50
    session: SessionRow | None = None
    clusters: list[str] = field(default_factory=list)
    cluster_index: int = -1

    def __post_init__(self) -> None:
        active = self.db.active_session()
        if active:
            self._enter(active)

    # -- sessions ----------------------------------------------------------

    def start(self, system: str, gravity_well: str, region: str, ship: str) -> SessionRow:
        if self.session is not None:
            self.close()
        self._enter(self.db.create_session(system, gravity_well, region, ship))
        assert self.session is not None
        return self.session

    def close(self) -> int | None:
        """Closes the session and queues its final batch; returns that batch's number."""
        if self.session is None:
            return None
        session_id = self.session.session_id
        number = self.db.create_batch(session_id, self.db.unbatched(session_id), is_final=True)
        self.db.close_session(session_id)
        self.session, self.clusters, self.cluster_index = None, [], -1
        return number

    def set_location(self, system: str, gravity_well: str, region: str) -> None:
        if self.session is not None:
            self.db.update_session_location(self.session.session_id, system, gravity_well, region)
            self.refresh()

    def refresh(self) -> None:
        if self.session is not None:
            self.session = self.db.session(self.session.session_id)

    def _enter(self, session: SessionRow) -> None:
        self.session = session
        self.clusters = self.db.clusters(session.session_id)
        self.cluster_index = len(self.clusters) - 1

    # -- clusters ----------------------------------------------------------

    @property
    def cluster_id(self) -> str | None:
        return self.clusters[self.cluster_index] if self.clusters else None

    @property
    def cluster_number(self) -> int:
        return self.cluster_index + 1

    def new_cluster(self) -> str | None:
        if self.session is None:
            return None
        cluster_id = self.db.new_cluster(self.session.session_id)
        self.clusters.append(cluster_id)
        self.cluster_index = len(self.clusters) - 1
        return cluster_id

    def step_cluster(self, delta: int) -> None:
        if self.clusters:
            self.cluster_index = max(0, min(len(self.clusters) - 1, self.cluster_index + delta))

    def cluster_deposit(self, exclude_capture: str | None = None) -> str | None:
        cluster = self.cluster_id
        return self.db.cluster_deposit(cluster, exclude_capture) if cluster else None

    # -- captures ----------------------------------------------------------

    def captures(self) -> list[Capture]:
        return self.db.captures(self.session.session_id) if self.session else []

    def check(
        self,
        scan: dict[str, FieldValue],
        rows: list[Row],
        layout: SlotLayout,
        cov: Coverage,
        editing: Capture | None = None,
    ) -> list[Issue]:
        capture = self._draft(scan, rows, editing, ship="")
        required = [
            b.column
            for b in cov.bindings
            if b.kind == SCANS and b.source is Source.OCR and b.required
        ]
        return validate(
            capture,
            layout.scan_columns,
            layout.row_columns,
            required_scan_columns=required,
            cluster_deposit=self.cluster_deposit(editing.capture_id if editing else None),
        )

    def save(
        self,
        scan: dict[str, FieldValue],
        rows: list[Row],
        layout: SlotLayout,
        cov: Coverage,
        *,
        ship: str,
        models: str = "",
        editing: Capture | None = None,
    ) -> SaveResult:
        if self.session is None:
            return SaveResult(False, [Issue("Start a session first")])
        issues = self.check(scan, rows, layout, cov, editing)
        if issues:
            return SaveResult(False, issues)
        capture = self._draft(scan, rows, editing, ship)
        if editing is None and is_duplicate(capture, self.captures(), layout.row_columns):
            return SaveResult(False, duplicate=True)
        self.db.save_capture(capture, models)
        queued = self._maybe_queue_batch()
        self.refresh()
        return SaveResult(True, capture_id=capture.capture_id, queued_batch=queued)

    def delete(self, capture_id: str) -> bool:
        """Deletes a capture; True when an uploaded copy stays on the server."""
        kept_remotely = self.db.delete_capture(capture_id)
        self.refresh()
        return kept_remotely

    def queue_upload(self) -> int | None:
        if self.session is None:
            return None
        sid = self.session.session_id
        return self.db.create_batch(sid, self.db.unbatched(sid), is_final=False)

    def _maybe_queue_batch(self) -> int | None:
        assert self.session is not None
        pending = self.db.unbatched(self.session.session_id)
        if len(pending) >= self.batch_size:
            return self.db.create_batch(self.session.session_id, pending, is_final=False)
        return None

    def _draft(
        self, scan: dict[str, FieldValue], rows: list[Row], editing: Capture | None, ship: str
    ) -> Capture:
        assert self.session is not None or editing is not None
        session = self.session
        context = {
            "system": session.system if session else "",
            "gravity_well": session.gravity_well if session else "",
            "region": session.region if session else "",
            "ship": ship,
        }
        if editing is not None:
            context = {**editing.context, **({"ship": ship} if ship else {})}
        return Capture(
            capture_id=editing.capture_id if editing else str(uuid.uuid4()),
            session_id=editing.session_id if editing else session.session_id,  # type: ignore[union-attr]
            cluster_id=editing.cluster_id if editing else (self.cluster_id or ""),
            timestamp=editing.timestamp if editing else utc_timestamp(),
            context=context,
            scan=dict(scan),
            rows=[dict(r) for r in rows],
        )
