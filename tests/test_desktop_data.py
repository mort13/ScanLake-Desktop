"""Sessions, saving, the upload queue and the Store client, against a mocked Store."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest
import respx

from scanlake_core.domain.capture import FieldValue, Identity, Row
from scanlake_core.format import ParquetFormat, coverage
from scanlake_core.profile.model import HudProfile
from scanlake_core.schema_map import SlotLayout, slot_layout
from scanlake_desktop.db.database import Database
from scanlake_desktop.services.session_service import SessionService
from scanlake_desktop.upload.uploader import Uploader
from scanlake_io.store_client import StoreClient, StoreError

STORE = "https://store.test"
USER = "6f1c3c1e-6c1d-4c3a-9d6e-5b0d6a1e2f3a"


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "scanlake.db")


@pytest.fixture
def layout(profiles: dict[str, HudProfile]) -> SlotLayout:
    return slot_layout(profiles["mole"])


def _values(
    mass: str = "11826", deposit: str = "Quartz"
) -> tuple[dict[str, FieldValue], list[Row]]:
    ocr = FieldValue.from_ocr
    scan = {"deposit": ocr(deposit, 0.97), "mass": ocr(mass, 0.99), "resistance": ocr("12", 0.95),
            "instability": ocr("4.58", 0.9), "volume": ocr("42.10", 0.93)}  # fmt: skip
    empty = {"type": FieldValue(), "amount": FieldValue(), "quality": FieldValue()}
    rows = [
        {"type": ocr("Quartz", 0.98), "amount": ocr("58.40", 0.97), "quality": ocr("512", 0.96)},
        {"type": ocr("Iron", 0.9), "amount": ocr("41.60", 0.88), "quality": ocr("3", 0.92)},
        dict(empty), dict(empty), dict(empty),
    ]  # fmt: skip
    return scan, rows


def _service(db: Database, batch_size: int = 50) -> SessionService:
    service = SessionService(db, batch_size=batch_size)
    service.start("Stanton", "Calliope", "", "mole")
    return service


def test_save_validate_and_duplicates(
    db: Database, layout: SlotLayout, fmt: ParquetFormat, profiles: dict[str, HudProfile]
) -> None:
    cov = coverage(fmt, layout)
    service = _service(db)
    scan, rows = _values()
    result = service.save(scan, rows, layout, cov, ship="mole")
    assert result.saved, result.issues

    again = service.save(*_values(), layout, cov, ship="mole")
    assert not again.saved and again.duplicate

    bad_scan, bad_rows = _values(mass="")
    bad_rows[1]["amount"] = FieldValue(text="40.00")
    issues = service.save(bad_scan, bad_rows, layout, cov, ship="mole").issues
    assert {i.column for i in issues} >= {"mass", "amount"}

    other = service.save(*_values(mass="9000", deposit="Iron"), layout, cov, ship="mole")
    assert any("Cluster deposit" in i.message for i in other.issues)

    stored = service.captures()
    assert len(stored) == 1
    assert stored[0].scan["mass"].confidence == pytest.approx(0.99)
    assert stored[0].context["ship"] == "mole"


def test_session_survives_restart(db: Database, layout: SlotLayout, fmt: ParquetFormat) -> None:
    service = _service(db)
    service.new_cluster()
    restored = SessionService(db)
    assert restored.session is not None
    assert restored.session.session_id == service.session.session_id  # type: ignore[union-attr]
    assert restored.cluster_number == 2


def test_batches_queue_at_batch_size_and_on_close(
    db: Database, layout: SlotLayout, fmt: ParquetFormat
) -> None:
    cov = coverage(fmt, layout)
    service = _service(db, batch_size=2)
    assert service.save(*_values("5000"), layout, cov, ship="mole").queued_batch is None
    assert service.save(*_values("6000"), layout, cov, ship="mole").queued_batch == 1
    service.save(*_values("7000"), layout, cov, ship="mole")
    assert service.close() == 2
    assert [b.batch_number for b in db.due_batches()] == [1, 2]
    assert db.due_batches()[1].is_final


def _uploader(db: Database, fmt: ParquetFormat, tmp_path: Path) -> Uploader:
    return Uploader(
        db=db,
        store=StoreClient(STORE),
        current_format=lambda: fmt,
        identity=lambda: Identity(USER, "Mort13", "RoseCorp", "test"),
        export_dir=lambda: tmp_path / "export",
    )


def _mock_session(router: respx.Router) -> respx.Route:
    return router.post(f"{STORE}/api/session").mock(
        return_value=httpx.Response(
            200, json={"token": "tok", "sessionId": "x", "expiresAt": "2099-01-01T00:00:00.000Z"}
        )
    )


@respx.mock
def test_upload_success_and_reupload_after_edit(
    db: Database, layout: SlotLayout, fmt: ParquetFormat, tmp_path: Path
) -> None:
    cov = coverage(fmt, layout)
    service = _service(db)
    saved = service.save(*_values(), layout, cov, ship="mole")
    service.queue_upload()

    _mock_session(respx.mock)
    sent: list[httpx.Request] = []

    def upload(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"ok": True, "keys": ["k1", "k2", "k3"]})

    respx.post(f"{STORE}/api/upload").mock(side_effect=upload)
    [outcome] = _uploader(db, fmt, tmp_path).run_due()
    assert outcome.ok
    body = sent[0].content
    assert b'name="batchNumber"\r\n\r\n1' in body and b'name="isFinal"\r\n\r\nfalse' in body
    assert sent[0].headers["Authorization"] == "Bearer tok"
    exported = sorted(p.name for p in (tmp_path / "export").iterdir())
    assert exported[0] == f"{USER}_compositions_{service.session.session_id}_batch1.parquet"  # type: ignore[union-attr]
    scans = pq.read_table(io.BytesIO((tmp_path / "export" / exported[2]).read_bytes()))
    assert scans.column("mass").to_pylist() == [11826]
    assert db.upload_state(service.session.session_id) == {saved.capture_id: "uploaded"}  # type: ignore[union-attr]

    # Editing an uploaded scan queues it again
    editing = db.capture(saved.capture_id)
    scan, rows = _values(mass="11900")
    service.save(scan, rows, layout, cov, ship="mole", editing=editing)
    assert db.upload_state(service.session.session_id)[saved.capture_id] == "pending"  # type: ignore[union-attr]
    assert service.queue_upload() == 2


@respx.mock
def test_rejected_batch_keeps_violations(
    db: Database, layout: SlotLayout, fmt: ParquetFormat, tmp_path: Path
) -> None:
    service = _service(db)
    service.save(*_values(), layout, coverage(fmt, layout), ship="mole")
    service.queue_upload()
    _mock_session(respx.mock)
    respx.post(f"{STORE}/api/upload").mock(
        return_value=httpx.Response(
            422, json={"error": "scans: does not match", "violations": ['missing column "x"']}
        )
    )
    [outcome] = _uploader(db, fmt, tmp_path).run_due()
    assert not outcome.ok and outcome.violations == ('missing column "x"',)
    summary = db.upload_summary()
    assert summary.failed_batches == 1 and "422" in (summary.last_error or "")
    assert db.due_batches(time.time() + 10**6) == []  # permanent: not retried automatically


@respx.mock
def test_rate_limit_and_expired_token(
    db: Database, layout: SlotLayout, fmt: ParquetFormat, tmp_path: Path
) -> None:
    service = _service(db)
    service.save(*_values(), layout, coverage(fmt, layout), ship="mole")
    service.queue_upload()
    session_route = _mock_session(respx.mock)
    respx.post(f"{STORE}/api/upload").mock(
        side_effect=[
            httpx.Response(401, json={"error": "expired"}),
            httpx.Response(429, json={"error": "slow down"}, headers={"Retry-After": "900"}),
        ]
    )  # fmt: skip
    before = time.time()
    [outcome] = _uploader(db, fmt, tmp_path).run_due()
    assert not outcome.ok
    assert session_route.call_count == 2  # the 401 fetched a fresh token
    [batch] = db.due_batches(time.time() + 10**6)
    assert batch.attempts == 1 and batch.next_attempt_at >= before + 900


def test_store_error_classification() -> None:
    with respx.mock:
        respx.post(f"{STORE}/api/session").mock(
            return_value=httpx.Response(400, json={"error": "bad"})
        )
        with pytest.raises(StoreError) as err:
            StoreClient(STORE).session_token("not-a-uuid", "x")
    assert err.value.permanent and err.value.status == 400
    with respx.mock:
        respx.post(f"{STORE}/api/session").mock(side_effect=httpx.ConnectError("offline"))
        with pytest.raises(StoreError) as err:
            StoreClient(STORE).session_token(USER, USER)
    assert not err.value.permanent and err.value.status is None
    assert json.dumps({"ok": 1})
