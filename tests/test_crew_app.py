"""Shared sessions in the app: the room connection, the controller, the Crew tab, the Share
menu and Join dialog, and the wiring in DesktopApp (against a fake room)."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QObject, Signal

from scanlake_core.domain.capture import Capture, FieldValue, utc_timestamp
from scanlake_core.domain.crew import DONE, NEED_HELP, OPEN, Member, cluster_to_wire
from scanlake_core.domain.navigation import ClusterSummary
from scanlake_desktop.crew.controller import CrewController
from scanlake_desktop.crew.room_worker import GONE, OFFLINE, ONLINE, RoomWorker
from scanlake_desktop.db.database import Database
from scanlake_io import crew_client
from tests.test_crew import BOB, CODE, CREW, ME, _member, _room_cluster, _snapshot, _summary
from tests.test_navigation import _in_field
from tests.test_overview import _order, _start_app


class FakeWorker(QObject):
    """Stands in for RoomWorker: records what is sent; the test plays the room."""

    message = Signal(object)
    connection_changed = Signal(str)

    def __init__(self, base_url: str, code: str, hello: Any, user_agent: str = "") -> None:
        super().__init__()
        self.base_url, self.code, self.hello = base_url, code, hello
        self.sent: list[dict[str, Any]] = []
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    def connect_with(self, snapshot: dict[str, Any]) -> None:
        self.connection_changed.emit(ONLINE)
        self.message.emit(snapshot)

    def take(self) -> list[dict[str, Any]]:
        sent, self.sent = self.sent, []
        return sent


# -- database -----------------------------------------------------------------------------------


def test_shared_session_rows_and_batch_numbers(tmp_path: Path) -> None:
    db = Database(tmp_path / "scanlake.db")
    host = db.create_session("Pyro", "", "", "mole")
    share = db.set_share(host.session_id, CODE)
    assert share.batch_base == 0 and share.active and uuid.UUID(share.client_id)
    assert db.create_batch(host.session_id, [], is_final=True) == 1

    joined = db.create_session("Pyro", "", "", "mole", session_id="shared-id")
    assert joined.session_id == "shared-id"
    share = db.set_share("shared-id", CODE, batch_base=7000)
    assert db.create_batch("shared-id", [], is_final=True) == 7001  # its own number range
    db.close_session("shared-id")
    db.end_share("shared-id")
    assert db.share("shared-id") == replace(share, active=False)

    # Rejoining reopens the session; the share keeps its client id and base, batches go on
    assert db.reopen_session("shared-id", "golem").status == "active"  # type: ignore[union-attr]
    again = db.set_share("shared-id", "cd" * 32, batch_base=99000)
    assert again == replace(share, code="cd" * 32)
    assert db.create_batch("shared-id", [], is_final=True) == 7002
    assert db.reopen_session("unknown", "mole") is None and db.share("unknown") is None

    db.set_cluster_status("c1", DONE)
    db.set_cluster_status("c2", NEED_HELP, dirty=True)
    db.set_cluster_status("c1", OPEN)
    assert db.cluster_statuses() == {"c1": (OPEN, False), "c2": (NEED_HELP, True)}
    db.close()


def test_session_service_joins_and_rejoins_a_session_id(tmp_path: Path) -> None:
    from scanlake_desktop.services.session_service import SessionService

    db = Database(tmp_path / "scanlake.db")
    service = SessionService(db)
    first = service.start("Pyro", "Pyro V", "", "mole", session_id="shared-id")
    assert first.session_id == "shared-id" and len(service.clusters) == 1
    service.new_cluster()
    service.close()
    again = service.start("Stanton", "", "", "golem", session_id="shared-id")
    assert again.session_id == "shared-id" and again.status == "active"
    assert (again.system, again.ship) == ("Stanton", "golem") and len(service.clusters) == 2
    db.close()


# -- the connection -----------------------------------------------------------------------------


def _serve(handler: Any) -> tuple[Any, Any]:
    from websockets.sync.client import connect
    from websockets.sync.server import serve

    server = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.socket.getsockname()[1]
    return server, lambda base_url, code, user_agent="": connect(
        f"ws://127.0.0.1:{port}", legacy=True
    )


def test_room_worker_says_hello_relays_and_reconnects(qtbot: Any) -> None:
    received: list[dict[str, Any]] = []
    connections = 0

    def handler(socket: Any) -> None:
        nonlocal connections
        connections += 1
        received.append(json.loads(socket.recv()))
        socket.send(json.dumps({"t": "snapshot", "n": connections}))
        if connections == 1:
            socket.send("not json")
            received.append(json.loads(socket.recv()))
            return  # drops the connection: the worker comes back
        for raw in socket:
            received.append(json.loads(raw))

    server, connect = _serve(handler)
    worker = RoomWorker(CREW, CODE, lambda: {"t": "hello"}, connect=connect, backoff_s=0.05)
    messages: list[dict[str, Any]] = []
    states: list[str] = []
    worker.message.connect(messages.append)
    worker.connection_changed.connect(states.append)
    try:
        worker.send({"t": "stale"})  # queued before the connection: never sent
        worker.start()
        qtbot.waitUntil(lambda: messages == [{"t": "snapshot", "n": 1}], timeout=5000)
        worker.send({"t": "status"})
        qtbot.waitUntil(lambda: len(messages) == 2, timeout=5000)
        assert messages[1] == {"t": "snapshot", "n": 2}
        worker.send({"t": "enroute"})
        qtbot.waitUntil(lambda: len(received) == 4, timeout=5000)
        assert received == [{"t": "hello"}, {"t": "status"}, {"t": "hello"}, {"t": "enroute"}]
        assert states[:2] == ["connecting", ONLINE] and OFFLINE in states and states[-1] == ONLINE
    finally:
        worker.stop()
        server.shutdown()
    assert not worker.isRunning()


def test_room_worker_stops_when_the_room_is_gone(qtbot: Any) -> None:
    def gone(base_url: str, code: str, user_agent: str = "") -> Any:
        raise crew_client.RoomGone("gone", 404)

    worker = RoomWorker(CREW, CODE, lambda: {}, connect=gone)
    states: list[str] = []
    worker.connection_changed.connect(states.append)
    worker.start()
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=5000)
    qtbot.waitUntil(lambda: states == ["connecting", GONE], timeout=5000)

    def closed_as_gone(socket: Any) -> None:
        socket.recv()
        socket.close(4404, "ended")

    server, connect = _serve(closed_as_gone)
    worker = RoomWorker(CREW, CODE, lambda: {}, connect=connect)
    states = []
    worker.connection_changed.connect(states.append)
    worker.start()
    try:
        qtbot.waitUntil(lambda: states[-1:] == [GONE], timeout=5000)
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=5000)
    finally:
        worker.stop()
        server.shutdown()


# -- the controller -----------------------------------------------------------------------------


def _controller(
    tmp_path: Path, qtbot: Any
) -> tuple[CrewController, Database, list[tuple[str, bool]]]:
    db = Database(tmp_path / "scanlake.db")
    profile: dict[str, Any] = {"roles": ["pilot"], "ship": "MOLE", "crew": "Mole crew"}
    controller = CrewController(
        db, CREW, lambda: (ME, "Me"), lambda: (profile["roles"], profile["ship"], profile["crew"]),
        worker_factory=FakeWorker,
    )  # fmt: skip
    controller.profile_source = profile  # type: ignore[attr-defined]
    notices: list[tuple[str, bool]] = []
    controller.notice.connect(lambda text, urgent: notices.append((text, urgent)))
    return controller, db, notices


def test_controller_marks_clusters_without_a_room(tmp_path: Path, qtbot: Any) -> None:
    controller, db, _ = _controller(tmp_path, qtbot)
    controller.publish([_summary("a"), _summary("b", 2, last="11:00")])
    assert [c.cluster_id for c in controller.cards()] == ["b", "a"]
    controller.set_status("a", NEED_HELP)
    controller.set_status("b", "nonsense")
    controller.set_status("remote", DONE)  # not ours and no room: ignored
    assert [(c.cluster_id, c.status) for c in controller.cards()] == [("a", NEED_HELP), ("b", OPEN)]
    assert db.cluster_statuses() == {"a": (NEED_HELP, False)}
    controller.set_target("b")
    assert controller.cards()[1].enroute == ("You",)
    assert not controller.shared and controller.members() == []
    db.close()


def test_controller_syncs_with_the_room(tmp_path: Path, qtbot: Any) -> None:
    controller, db, notices = _controller(tmp_path, qtbot)
    a, b = _summary("a"), _summary("b", 2)
    db.set_cluster_status("b", DONE)
    controller.statuses = db.cluster_statuses()
    controller.publish([a, b])
    controller.set_target("a")
    share = db.set_share("sess", CODE)
    controller.open(share)
    room: FakeWorker = controller.worker
    assert room.running and room.code == CODE and room.hello() == {
        "t": "hello", "v": 1, "userId": ME, "clientId": share.client_id, "name": "Me",
        "roles": ["pilot"], "ship": "MOLE", "crew": "Mole crew", "target": "a",
    }  # fmt: skip
    assert controller.shared and not controller.online
    controller.set_status("a", NEED_HELP)  # before the snapshot: kept, sent afterwards
    assert room.sent == [] and db.cluster_statuses()["a"] == (NEED_HELP, True)

    # The room knows `a` (older, open) and Bob's `r`; `b` is new to it
    theirs = _summary("r", 7, last="11:00")
    room.connect_with(_snapshot(
        [_room_cluster(replace(a, rocks=1), ME), _room_cluster(theirs, BOB)],
        [{"userId": ME, "clusterId": "a", "since": 1}],
    ))  # fmt: skip
    assert controller.online
    assert room.take() == [
        {"t": "status", "clusterId": "a", "status": NEED_HELP},
        {"t": "clusters", "remove": [], "upsert": [
            {**cluster_to_wire(a), "status": NEED_HELP},
            {**cluster_to_wire(b), "status": DONE},
        ]},
    ]  # fmt: skip
    assert db.cluster_statuses()["a"] == (NEED_HELP, False)
    assert [(c.cluster_id, c.mine, c.status, c.enroute) for c in controller.cards()] == [
        ("a", True, NEED_HELP, ("You",)),
        ("r", False, OPEN, ()),
        ("b", True, DONE, ()),
    ]
    assert controller.target_of(ME) == "#1 · Gold" and controller.crews() == ["Mole crew"]

    controller.publish([a, b])  # nothing changed: nothing sent
    controller.publish([a])  # `b` lost its scans
    assert room.take() == [{"t": "clusters", "upsert": [], "remove": ["b"]}]

    # Changes from the room
    room.message.emit({"t": "enroute", "userId": BOB, "clusterId": "a", "at": 2})
    room.message.emit({"t": "status", "clusterId": "a", "status": DONE, "by": BOB, "at": 3})
    assert notices == [("Bob is on the way to your #1 · Gold", True),
                       ("Bob marked your #1 · Gold done", False)]  # fmt: skip
    assert db.cluster_statuses()["a"] == (DONE, False)
    assert controller.cards()[1].enroute == ("You", "Bob")
    assert controller.target_of(BOB) == "#1 · Gold"

    # Own actions go out and show at once
    controller.set_status("r", NEED_HELP)
    controller.set_target("r")
    controller.set_target("r")
    assert room.take() == [
        {"t": "status", "clusterId": "r", "status": NEED_HELP},
        {"t": "enroute", "clusterId": "r"},
    ]
    assert [(c.cluster_id, c.status, c.enroute) for c in controller.cards()][0] == (
        "r", NEED_HELP, ("You",))  # fmt: skip
    assert "r" not in db.cluster_statuses()  # another member's: only in the room

    controller.profile_source.update(roles=["scout", "hauler"], crew="Haulers")  # type: ignore[attr-defined]
    controller.set_profile()
    assert room.take() == [{"t": "profile", "name": "Me", "roles": ["scout", "hauler"],
                            "ship": "MOLE", "crew": "Haulers"}]  # fmt: skip
    me = next(m for m in controller.members() if m.user_id == ME)
    assert me.roles == ("scout", "hauler") and me.crew == "Haulers"

    # Connection lost: remote marks are refused, own ones wait
    room.connection_changed.emit(OFFLINE)
    assert not controller.online and controller.shared
    controller.set_status("r", DONE)
    controller.set_status("a", OPEN)
    assert room.take() == [] and db.cluster_statuses()["a"] == (OPEN, True)
    room.connect_with(_snapshot([_room_cluster(a, ME, DONE), _room_cluster(theirs, BOB)]))
    assert room.take() == [
        {"t": "status", "clusterId": "a", "status": OPEN},
        {"t": "enroute", "clusterId": "r"},
    ]

    controller.close()
    assert not room.running and not controller.shared
    assert [c.cluster_id for c in controller.cards()] == ["a"]
    db.close()


def test_controller_ends_when_the_room_is_gone(tmp_path: Path, qtbot: Any) -> None:
    controller, db, _ = _controller(tmp_path, qtbot)
    controller.open(db.set_share("sess", CODE))
    ended: list[str] = []
    controller.ended.connect(ended.append)
    controller.worker.connection_changed.emit(GONE)
    assert not controller.shared and "ended" in ended[0]
    assert db.share("sess").active is False  # type: ignore[union-attr]
    db.close()


# -- Crew tab, Share menu, Join dialog ----------------------------------------------------------


def test_crew_panel_groups_members_and_edits_the_own_profile(qtbot: Any) -> None:
    from scanlake_desktop.ui.crew_panel import CrewPanel

    panel = CrewPanel(["MOLE", "Prospector"])
    qtbot.addWidget(panel)
    changes: list[tuple[list[str], str, str]] = []
    panel.profile_changed.connect(lambda roles, ship, crew: changes.append((roles, ship, crew)))
    panel.set_profile(["pilot"], "MOLE", "")
    assert changes == [] and panel.roles["pilot"].isChecked() and not panel.leave.isEnabled()

    panel.roles["scout"].click()
    panel.crew.setCurrentText("Mole crew")
    edit = panel.crew.lineEdit()
    assert edit is not None
    edit.editingFinished.emit()
    edit.editingFinished.emit()  # unchanged: said once
    assert changes == [(["scout", "pilot"], "MOLE", ""), (["scout", "pilot"], "MOLE", "Mole crew")]

    members = [
        Member("1", "Zed", ("pilot",), "MOLE", "Mole crew", online=True),
        Member("2", "<i>Al</i>", ("scout", "laser"), "", "Mole crew", online=False),
        Member("3", "Cy", (), "C2", "haulers", online=True),
        Member("4", "Di", online=True),
    ]
    panel.show_room(members, "4", "https://crew.test/j/x", "Shared: 3 members online.",
                    {"1": "Bob #4 · Gold"}, shared=True, can_invite=True, enabled=True)  # fmt: skip
    assert panel.rows() == [
        ["haulers (1)"],
        ["Cy", "—", "C2", "online", ""],
        ["Mole crew (2)"],
        ["Zed", "Pilot", "MOLE", "online", "Bob #4 · Gold"],
        ["<i>Al</i>", "Scout, Laser Operator", "—", "offline", ""],
        ["No crew (1)"],
        ["Di (you)", "—", "—", "online", ""],
    ]
    assert panel.link.text() == "https://crew.test/j/x" and panel.invite.text() == "Copy link"
    assert panel.leave.isEnabled() and panel.crew.currentText() == "Mole crew"
    assert [panel.crew.itemText(i) for i in range(panel.crew.count())] == ["haulers", "Mole crew"]


def test_share_menu(qtbot: Any) -> None:
    from scanlake_desktop.ui.session_bar import SessionBar

    bar = SessionBar([("mole", "MOLE")])
    qtbot.addWidget(bar)
    clicked: list[str] = []
    bar.invite_clicked.connect(lambda: clicked.append("invite"))
    bar.join_clicked.connect(lambda: clicked.append("join"))
    bar.leave_clicked.connect(lambda: clicked.append("leave"))
    bar.set_shared(False)
    assert bar.share.text() == "Share" and not bar.leave_action.isEnabled()
    bar.invite_action.trigger()
    bar.join_action.trigger()
    bar.set_shared(True)
    assert bar.share.text() == "Shared" and bar.invite_action.text() == "Copy invite link"
    bar.leave_action.trigger()
    assert clicked == ["invite", "join", "leave"]
    bar.set_shared(False, enabled=False)
    assert not bar.share.isEnabled()


def test_join_dialog_previews_the_session(qtbot: Any) -> None:
    from PySide6.QtGui import QGuiApplication

    from scanlake_desktop.ui.join_dialog import JoinDialog

    info = crew_client.RoomInfo("sess", "Pyro", "Pyro V", "", "<b>Al</b>", 2)
    asked: list[str] = []

    def lookup(code: str) -> crew_client.RoomInfo:
        asked.append(code)
        if code != CODE:
            raise crew_client.RoomGone("This shared session does not exist any more")
        return info

    QGuiApplication.clipboard().setText("something else")
    dialog = JoinDialog(lookup, "Joining closes your current session.")
    qtbot.addWidget(dialog)
    assert dialog.link.text() == "" and not dialog.join.isEnabled()
    dialog.link.setText("https://example.org/")
    assert "not a ScanLake invite link" in dialog.preview.text() and asked == []

    dialog.link.setText(f"{CREW}/j/{'cd' * 32}")
    qtbot.waitUntil(lambda: "does not exist" in dialog.preview.text(), timeout=5000)
    assert not dialog.join.isEnabled() and dialog.info is None

    dialog.link.setText(f"{CREW}/j/{CODE}")
    qtbot.waitUntil(dialog.join.isEnabled, timeout=5000)
    assert dialog.code == CODE and dialog.info == info
    assert dialog.preview.text() == (
        "<b>Al</b>'s session · 2 members · Pyro / Pyro V\nJoining closes your current session."
    )
    assert dialog.preview.textFormat().name == "PlainText"

    QGuiApplication.clipboard().setText(f"  {CREW}/j/{CODE}\n")  # an invite: filled in
    prefilled = JoinDialog(lookup)
    qtbot.addWidget(prefilled)
    assert prefilled.link.text() == f"{CREW}/j/{CODE}"
    qtbot.waitUntil(prefilled.join.isEnabled, timeout=5000)


# -- the app ------------------------------------------------------------------------------------


def _shared_app(
    app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, list[tuple[str, str]]]:
    """An app whose shared sessions go to FakeWorker, with its FrontDesk pings recorded."""
    monkeypatch.setenv("SCANLAKE_CREW_URL", CREW)
    monkeypatch.setattr("scanlake_desktop.crew.controller.RoomWorker", FakeWorker)
    app = _start_app(app_env)
    pings: list[tuple[str, str]] = []
    monkeypatch.setattr(app, "_ping_frontdesk", lambda sid, kind: pings.append((sid, kind)))
    return app, pings


def _scan(app: Any, x: float = 1000.0) -> None:
    session = app.sessions.session
    app.db.save_capture(Capture(
        str(uuid.uuid4()), session.session_id, app.sessions.cluster_id, utc_timestamp(),
        scan={"deposit": FieldValue("Gold"), "volume": FieldValue("3.5")},
        position=_in_field(x, 0, 0),
    ))  # fmt: skip
    app._refresh_session()


def _remote(cluster_id: str = "r") -> ClusterSummary:
    return _summary(cluster_id, 7, last="09:00")


def test_app_invites_and_shows_the_room(
    qtbot: Any, app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtGui import QGuiApplication

    app, pings = _shared_app(app_env, monkeypatch)
    qtbot.addWidget(app.window)
    created: list[tuple[Any, ...]] = []

    def create_room(
        url: str, session_id: str, user_id: str, name: str, location: Any, **_: Any
    ) -> str:
        created.append((url, session_id, user_id, location))
        return CODE

    monkeypatch.setattr(crew_client, "create_room", create_room)
    try:
        window = app.window
        assert window.tabs.tabText(2) == "Crew" and window.session_bar.share.isEnabled()
        app.invite_session()  # starts a session when there is none
        qtbot.waitUntil(lambda: app.crew.shared, timeout=5000)
        session = app.sessions.session
        assert created == [(CREW, session.session_id, app.settings.user_id,
                            (session.system, session.gravity_well, session.region))]  # fmt: skip
        link = f"{CREW}/j/{CODE}"
        assert QGuiApplication.clipboard().text() == link and window.crew.link.text() == link
        assert window.session_bar.share.text() == "Shared"
        assert "Reconnecting" in window.crew.state.text()
        _scan(app)
        mine = app.sessions.cluster_id
        room: FakeWorker = app.crew.worker
        assert room.sent == []  # not connected yet

        room.connect_with({
            **_snapshot([_room_cluster(_remote(), BOB, NEED_HELP)]),
            "you": app.settings.user_id,
            "members": [_member(BOB, "Bob", crew="Mole crew"),
                        _member(app.settings.user_id, "Me")],
        })  # fmt: skip
        upserts = [m for m in room.take() if m["t"] == "clusters"]
        assert [c["clusterId"] for c in upserts[0]["upsert"]] == [mine]
        overview = window.overview
        assert _order(overview) == ["r", mine]  # need help first
        assert overview.cards["r"].title.text() == "Bob · #7 · Gold"
        assert overview.title.text() == "Clusters this session · shared with 1"
        assert window.crew.rows()[:2] == [["Mole crew (1)"], ["Bob", "—", "—", "online", ""]]

        # Navigating to Bob's cluster tells the room, and works with the live position
        overview.click("r")
        assert app.nav is not None and app.nav.target.owner == "Bob"
        assert room.take() == [{"t": "enroute", "clusterId": "r"}]
        assert overview.cards["r"].enroute.text() == "→ You"
        assert overview.target.text() == "Bob #7 · Gold"

        # Bob's cluster moves, then goes: the target follows, then navigation stops
        moved = replace(_remote(), rocks=9)
        room.message.emit({"t": "cluster", "cluster": _room_cluster(moved, BOB, NEED_HELP)})
        assert app.nav.target.rocks == 9
        room.message.emit({"t": "cluster_removed", "clusterId": "r"})
        assert app.nav is None and "r" not in overview.cards
        assert room.take() == [{"t": "enroute", "clusterId": None}]

        # A mark on an own card goes to the room; a notice from the room is flashed
        overview.cards[mine].help.click()
        assert room.take() == [{"t": "status", "clusterId": mine, "status": NEED_HELP}]
        room.message.emit({"t": "enroute", "userId": BOB, "clusterId": mine, "at": 1})
        assert "Bob is on the way to your #1 · Gold" in window.statusBar().currentMessage()
        assert overview.cards[mine].enroute.text() == "→ Bob"

        # The own profile, from the Crew tab
        window.crew.roles["laser"].click()
        assert app.settings.crew_roles == ["laser"]
        assert room.take()[0]["roles"] == ["laser"]

        # Closing a shared session with scans: no discard; an empty one neither (below)
        app.close_session()
        assert not app.crew.shared and not room.running
        assert pings == [(session.session_id, "open")]
        assert app.db.share(session.session_id).active is False
    finally:
        app.stop()


def test_app_joins_a_shared_session(
    qtbot: Any, app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, pings = _shared_app(app_env, monkeypatch)
    qtbot.addWidget(app.window)
    shared_id = str(uuid.uuid4())
    info = crew_client.RoomInfo(shared_id, "Pyro", "Pyro V", "", "Bob", 1)
    try:
        app.start_session()
        own = app.sessions.session.session_id
        _scan(app)
        app.join_room(CODE, info)  # closes the own session (it uploads), then joins
        session = app.sessions.session
        assert session.session_id == shared_id and (session.system, session.gravity_well) == (
            "Pyro",
            "Pyro V",
        )
        assert app.db.session(own).status == "closed"
        assert app.settings.system == "Pyro"
        share = app.db.share(shared_id)
        assert share.code == CODE and share.batch_base >= 1000 and share.batch_base % 1000 == 0
        assert (
            app.crew.worker.code == CODE and app.crew.worker.hello()["clientId"] == share.client_id
        )
        assert pings == [(own, "open"), (shared_id, "open")]

        app.join_room(CODE, info)  # already in it: just reconnects
        assert app.sessions.session.session_id == shared_id and len(pings) == 2

        # Leaving without scans must not discard the session for the others
        app.close_session()
        assert pings == [(own, "open"), (shared_id, "open")]

        # Joining again reopens the same local session, with the same client id
        app.join_room(CODE, info)
        assert app.sessions.session.session_id == shared_id
        assert app.db.share(shared_id).client_id == share.client_id
        assert app.db.share(shared_id).batch_base == share.batch_base

        app.leave_shared()  # stop sharing: the session goes on alone
        assert not app.crew.shared and app.sessions.session.session_id == shared_id
        assert app.window.session_bar.share.text() == "Share"
    finally:
        app.stop()


def test_app_resumes_a_shared_session(
    qtbot: Any, app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _ = _shared_app(app_env, monkeypatch)
    try:
        app.start_session()
        session_id = app.sessions.session.session_id
        app.share_session(session_id, CODE)
        client_id = app.crew.share.client_id
    finally:
        app.stop()

    again, _ = _shared_app(app_env, monkeypatch)
    qtbot.addWidget(again.window)
    try:
        assert again.crew.shared and again.crew.worker.code == CODE
        assert again.crew.worker.hello()["clientId"] == client_id
        again.crew.worker.connection_changed.emit(GONE)  # the room expired meanwhile
        assert not again.crew.shared and again.sessions.session.session_id == session_id
        assert "ended" in again.window.statusBar().currentMessage()
    finally:
        again.stop()


def test_app_without_a_crew_url_has_no_sharing(qtbot: Any, app_env: Path) -> None:
    app = _start_app(app_env)
    qtbot.addWidget(app.window)
    try:
        assert not app.crew.enabled and not app.window.session_bar.share.isEnabled()
        app.invite_session()
        app.join_session()
        assert app.sessions.session is None and "turned off" in app.window.crew.state.text()
    finally:
        app.stop()
