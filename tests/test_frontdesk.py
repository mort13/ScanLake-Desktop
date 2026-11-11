"""FrontDesk session pings: the request, that no failure reaches the caller, and when the app
sends them (off the GUI thread)."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pytest
import respx

from scanlake_io.frontdesk_client import Kind, ping

if TYPE_CHECKING:
    from scanlake_desktop.app import DesktopApp

FRONTDESK = "https://frontdesk.test"
SESSION = "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("kind", ["open", "discard"])
def test_ping_posts_the_session(kind: Kind) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    assert ping(FRONTDESK + "/", SESSION, kind, http=_client(handler))
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "https://frontdesk.test/api/ping"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {"sessionId": SESSION, "source": "desktop", "kind": kind}
    assert request.extensions["timeout"]["read"] == 3.0


def test_ping_with_its_own_client() -> None:
    with respx.mock:
        route = respx.post(f"{FRONTDESK}/api/ping").mock(return_value=httpx.Response(204))
        assert ping(FRONTDESK, SESSION, "open", user_agent="ScanLakeDesktop/1.2.3")
    request = route.calls.last.request
    assert request.headers["User-Agent"] == "ScanLakeDesktop/1.2.3"
    assert request.extensions["timeout"] == {"connect": 3.0, "read": 3.0, "write": 3.0, "pool": 3.0}


@pytest.mark.parametrize("status", [429, 400, 404, 500, 503])
def test_ping_refused(status: int) -> None:
    assert not ping(FRONTDESK, SESSION, "open", http=_client(lambda _: httpx.Response(status)))


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("offline"), httpx.ReadTimeout("slow"), RuntimeError("anything else")],
)
def test_ping_swallows_errors(error: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    assert not ping(FRONTDESK, SESSION, "discard", http=_client(handler))


def test_ping_swallows_a_bad_url() -> None:
    assert not ping("scanlake.rocks", SESSION, "open")  # no scheme: httpx refuses before sending


def test_ping_off_without_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("nothing may be sent")

    assert not ping("", SESSION, "open", http=_client(handler))


# -- the app ----------------------------------------------------------------


class FakePing:
    """Stands in for `frontdesk_client.ping`: records the calls and where they ran."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.on_gui_thread = False

    def __call__(self, base_url: str, session_id: str, kind: str, **_: object) -> bool:
        self.on_gui_thread |= threading.current_thread() is threading.main_thread()
        self.calls.append((base_url, session_id, kind))
        return True


@pytest.fixture
def pings(app_env: Path, monkeypatch: pytest.MonkeyPatch) -> FakePing:
    from scanlake_io import frontdesk_client

    fake = FakePing()
    monkeypatch.setattr(frontdesk_client, "ping", fake)
    monkeypatch.setenv("SCANLAKE_FRONTDESK_URL", FRONTDESK + "/")
    return fake


def _start_app(app_env: Path) -> DesktopApp:
    from scanlake_desktop.app import DesktopApp
    from scanlake_desktop.capture.source import write_image
    from scanlake_desktop.settings import Settings

    frame = app_env / "replay" / "blank.png"
    write_image(frame, np.zeros((1440, 2560, 3), np.uint8))  # no HUD: the worker idles
    settings = Settings(backend="ort-cpu", ship="mole", beep=False)
    return DesktopApp(settings, replay=[frame])


def test_start_and_discard(qtbot, pings: FakePing, app_env: Path) -> None:  # type: ignore[no-untyped-def]
    app = _start_app(app_env)
    qtbot.addWidget(app.window)
    try:
        app.start_session()
        assert app.sessions.session is not None
        session_id = app.sessions.session.session_id
        qtbot.waitUntil(lambda: len(pings.calls) == 1)
        assert pings.calls == [(FRONTDESK, session_id, "open")]

        app.close_session()  # nothing saved: its empty final batch never reaches the Store
        qtbot.waitUntil(lambda: len(pings.calls) == 2)
        assert pings.calls[1] == (FRONTDESK, session_id, "discard")
        assert not pings.on_gui_thread
    finally:
        app.stop()


def test_close_with_scans_is_left_to_the_store(qtbot, pings: FakePing, app_env: Path) -> None:  # type: ignore[no-untyped-def]
    from scanlake_core.domain.capture import Capture, utc_timestamp

    app = _start_app(app_env)
    qtbot.addWidget(app.window)
    try:
        app.new_cluster()  # the hotkey starts a session when there is none
        session, cluster = app.sessions.session, app.sessions.cluster_id
        assert session is not None and cluster is not None
        capture = Capture(str(uuid.uuid4()), session.session_id, cluster, utc_timestamp())
        app.db.save_capture(capture)
        app.sessions.refresh()
        app.close_session()

        app.start_session()
        assert app.sessions.session is not None
        later = app.sessions.session.session_id
        qtbot.waitUntil(lambda: (FRONTDESK, later, "open") in pings.calls)
        assert [c for c in pings.calls if c[1] == session.session_id] == [
            (FRONTDESK, session.session_id, "open")
        ]
    finally:
        app.stop()


def test_resumed_session_reports_open(qtbot, pings: FakePing, app_env: Path) -> None:  # type: ignore[no-untyped-def]
    from scanlake_desktop import paths
    from scanlake_desktop.db.database import Database

    db = Database(paths.database_file())
    resumed = db.create_session("Stanton", "", "", "mole").session_id
    db.close()

    app = _start_app(app_env)
    qtbot.addWidget(app.window)
    try:
        assert app.sessions.session is not None
        assert app.sessions.session.session_id == resumed
        qtbot.waitUntil(lambda: len(pings.calls) == 1)
        assert pings.calls == [(FRONTDESK, resumed, "open")]
    finally:
        app.stop()
