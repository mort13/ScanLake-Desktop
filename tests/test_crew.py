"""Shared sessions: the room's state, the wire form of a cluster, the cards and their order,
and the client for the ScanLake Crew service."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import httpx
import pytest
import respx

from scanlake_core.domain import crew
from scanlake_core.domain.crew import (
    DONE,
    NEED_HELP,
    OPEN,
    RoomState,
    SharedCluster,
    build_cards,
    cluster_to_wire,
    diff_clusters,
    notice_for,
    order_cards,
    summary_from_wire,
)
from scanlake_core.domain.navigation import ClusterSummary, nav_frame
from scanlake_io import crew_client
from tests.test_navigation import FIELD, _in_field

CREW = "https://crew.test"
CODE = "ab" * 32
ME, BOB = "me-id", "bob-id"


def _summary(
    cluster_id: str = "a", number: int = 1, last: str = "10:00", **extra: Any
) -> ClusterSummary:
    return ClusterSummary(
        cluster_id=cluster_id,
        number=number,
        deposit="Gold",
        rocks=3,
        total_scu=12.5,
        local_zone=FIELD,
        centres={FIELD: (1000.04, 0.0, 0.0), "Root": (1e9, 2e9, -1e7)},
        first_scan="2026-11-10T09:00:00.000Z",
        last_scan=f"2026-11-10T{last}:00.000Z",
        **extra,
    )


def _member(user_id: str, name: str, **extra: Any) -> dict[str, Any]:
    return {"userId": user_id, "name": name, "roles": [], "ship": "", "crew": "",
            "online": True, "joinedAt": 1, **extra}  # fmt: skip


def _room_cluster(summary: ClusterSummary, owner: str, status: str = OPEN) -> dict[str, Any]:
    return {**cluster_to_wire(summary), "ownerId": owner, "status": status,
            "statusBy": None, "statusAt": None}  # fmt: skip


def _snapshot(
    clusters: Sequence[dict[str, Any]] = (), enroute: Sequence[dict[str, Any]] = ()
) -> dict[str, Any]:
    return {
        "t": "snapshot", "v": 1, "now": 1_000, "you": ME,
        "room": {"sessionId": "s", "hostId": BOB,
                 "location": {"system": "Pyro", "gravityWell": "Pyro V", "region": "Belt"}},
        "members": [_member(BOB, "Bob", roles=["pilot", "bogus"], crew="Mole crew"),
                    _member(ME, "Me")],
        "clusters": list(clusters), "enroute": list(enroute),
    }  # fmt: skip


# -- wire ---------------------------------------------------------------------------------------


def test_cluster_round_trips_over_the_wire_and_navigates() -> None:
    summary = _summary()
    wire = cluster_to_wire(summary)
    assert wire["centres"][FIELD] == [1000.0, 0.0, 0.0]  # 0.1 m
    assert json.loads(json.dumps(wire)) == wire
    back = summary_from_wire({**wire, "ownerId": BOB}, owner="Bob")
    assert back == replace(summary, centres={FIELD: (1000.0, 0.0, 0.0), "Root": (1e9, 2e9, -1e7)},
                           owner="Bob")  # fmt: skip
    frame = nav_frame(_in_field(1000, 300, 400), back)  # the navigation takes it as it is
    assert frame is not None and frame.distance == 500

    junk = summary_from_wire({"clusterId": 5, "centres": {"z": [1, "x"]}, "rocks": True})
    assert junk == ClusterSummary("", 0, "", 0, 0.0, "")


def test_diff_sends_only_what_changed() -> None:
    a, b = _summary("a"), _summary("b", 2)
    upsert, remove = diff_clusters([a, b], {})
    assert [w["clusterId"] for w in upsert] == ["a", "b"] and remove == []
    sent = {"a": cluster_to_wire(a), "b": cluster_to_wire(b), "gone": cluster_to_wire(a)}
    upsert, remove = diff_clusters([a, replace(b, rocks=4)], sent)
    assert [w["clusterId"] for w in upsert] == ["b"] and remove == ["gone"]
    # A room's copy carries more keys than the owner published
    assert crew.payload(_room_cluster(a, ME)) == cluster_to_wire(a)


# -- cards --------------------------------------------------------------------------------------


def test_cards_order_need_help_on_the_way_open_done_then_newest() -> None:
    def card(
        cluster_id: str, status: str = OPEN, last: str = "10:00", enroute: tuple[str, ...] = ()
    ) -> SharedCluster:
        return SharedCluster(_summary(cluster_id, last=last), status=status, enroute=enroute)

    cards = [
        card("done-new", DONE, "12:00"),
        card("open-old", last="09:00"),
        card("open-new", last="11:00"),
        card("going", enroute=("Bob",), last="08:00"),
        card("help", NEED_HELP, "07:00"),
        card("help-going", NEED_HELP, "07:30", ("Bob",)),
        card("done-going", DONE, "06:00", ("Bob",)),
    ]
    assert [c.cluster_id for c in order_cards(cards)] == [
        "help-going", "help", "going", "open-new", "open-old", "done-new", "done-going",
    ]  # fmt: skip
    assert order_cards(reversed(cards)) == order_cards(cards)


def test_build_cards_merges_own_and_remote() -> None:
    room = RoomState()
    mine, theirs = _summary("a"), _summary("r", 4, last="11:00")
    room.apply(_snapshot(
        [_room_cluster(mine, ME, DONE), _room_cluster(theirs, BOB, NEED_HELP)],
        [{"userId": BOB, "clusterId": "a", "since": 1}],
    ))  # fmt: skip
    cards = build_cards([mine], {"a": OPEN}, room, target="r")
    assert [(c.cluster_id, c.mine, c.status, c.enroute) for c in cards] == [
        ("r", False, NEED_HELP, ("You",)),
        ("a", True, OPEN, ("Bob",)),  # own status comes from the local store
    ]
    assert cards[0].summary.owner == "Bob" and crew.label(cards[0].summary) == "Bob #4 · Gold"
    assert cards[1].summary.owner == "" and crew.label(cards[1].summary) == "#1 · Gold"
    assert [c.cluster_id for c in build_cards([mine], {})] == ["a"]  # alone


# -- the room's state ---------------------------------------------------------------------------


def test_room_state_follows_the_room() -> None:
    room = RoomState()
    room.apply(_snapshot([_room_cluster(_summary("r"), BOB)]))
    assert room.you == ME and room.session_id == "s" and room.location == ("Pyro", "Pyro V", "Belt")
    assert room.members[BOB].roles == ("pilot",) and room.members[BOB].role_names == "Pilot"
    assert room.name(BOB) == "Bob" and room.name(ME) == "You" and room.name("?") == "Someone"

    room.apply({"t": "enroute", "userId": BOB, "clusterId": "r", "at": 5})
    assert room.heading_to("r") == [BOB]
    room.apply({"t": "status", "clusterId": "r", "status": DONE, "by": BOB, "at": 6})
    assert room.clusters["r"]["status"] == DONE
    room.apply({"t": "status", "clusterId": "r", "status": "exploded", "by": BOB, "at": 6})
    assert room.clusters["r"]["status"] == DONE
    room.apply({"t": "member", "member": _member(BOB, "Bobby", ship="MOLE")})
    assert room.members[BOB].name == "Bobby" and room.members[BOB].ship == "MOLE"
    room.apply({"t": "presence", "userId": BOB, "online": False})
    assert not room.members[BOB].online and room.heading_to("r") == []  # offline: not heading

    room.apply({"t": "enroute", "userId": ME, "clusterId": "r", "at": 7})
    room.apply({"t": "cluster_removed", "clusterId": "r"})
    assert room.clusters == {} and room.enroute == {}
    room.apply({"t": "enroute", "userId": BOB, "clusterId": None, "at": 8})
    room.apply({"t": "something new"})  # ignored


def test_notices_say_who_did_what() -> None:
    room = RoomState()
    room.apply(_snapshot([_room_cluster(_summary("r", 4), BOB)]))
    own = {"a": _summary("a")}

    def note(message: dict[str, Any]) -> tuple[str, bool] | None:
        found = notice_for(message, room, own)
        return (found.text, found.urgent) if found else None

    assert note({"t": "enroute", "userId": BOB, "clusterId": "a"}) == (
        "Bob is on the way to your #1 · Gold", True)  # fmt: skip
    assert note({"t": "enroute", "userId": BOB, "clusterId": "r"}) == (
        "Bob is on the way to Bob #4 · Gold", False)  # fmt: skip
    assert note({"t": "enroute", "userId": BOB, "clusterId": None}) is None
    assert note({"t": "status", "clusterId": "a", "status": NEED_HELP, "by": BOB}) == (
        "Bob needs help at your #1 · Gold", True)  # fmt: skip
    assert note({"t": "status", "clusterId": "r", "status": DONE, "by": BOB}) == (
        "Bob marked Bob #4 · Gold done", False)  # fmt: skip
    assert note({"t": "status", "clusterId": "?", "status": DONE, "by": BOB}) is None
    assert note({"t": "member", "member": _member(BOB, "Bob", ship="C2")}) is None  # was online
    assert note({"t": "member", "member": _member("new", "Cy")}) == ("Cy joined the session", False)
    assert note({"t": "presence", "userId": BOB, "online": False}) == (
        "Bob left the session",
        False,
    )


# -- the Crew client ----------------------------------------------------------------------------


def test_parse_invite_takes_only_the_code() -> None:
    assert crew_client.parse_invite(f"https://crew.scanlake.rocks/j/{CODE}") == CODE
    assert crew_client.parse_invite(f"  {CODE.upper()}\n") == CODE
    # A link to another host still yields just the code: the app never connects there
    assert crew_client.parse_invite(f"https://evil.example/j/{CODE}?x=1") == CODE
    assert crew_client.parse_invite("https://crew.scanlake.rocks/j/abc") is None
    assert crew_client.parse_invite(CODE + "ab") is None  # no 64-hex run on its own
    assert crew_client.parse_invite("") is None
    assert crew_client.invite_link(CREW + "/", CODE) == f"{CREW}/j/{CODE}"
    assert crew_client.ws_url(CREW, CODE) == f"wss://crew.test/api/rooms/{CODE}/ws"
    assert crew_client.ws_url("http://127.0.0.1:8790", CODE).startswith("ws://127.0.0.1:8790/")


@respx.mock
def test_create_room_and_room_info() -> None:
    create = respx.post(f"{CREW}/api/rooms").mock(
        return_value=httpx.Response(201, json={"code": CODE})
    )
    code = crew_client.create_room(CREW, "sess", "user", "Al", ("Pyro", "Pyro V", ""))
    assert code == CODE
    assert json.loads(create.calls[0].request.content) == {
        "sessionId": "sess", "userId": "user", "name": "Al",
        "location": {"system": "Pyro", "gravityWell": "Pyro V", "region": ""},
    }  # fmt: skip

    respx.get(f"{CREW}/api/rooms/{CODE}").mock(return_value=httpx.Response(200, json={
        "sessionId": "sess", "hostName": "Al", "members": 2,
        "location": {"system": "Pyro", "gravityWell": "Pyro V", "region": "Belt"},
    }))  # fmt: skip
    info = crew_client.room_info(CREW, CODE)
    assert info == crew_client.RoomInfo("sess", "Pyro", "Pyro V", "Belt", "Al", 2)


@respx.mock
def test_crew_client_errors() -> None:
    respx.get(f"{CREW}/api/rooms/{CODE}").mock(return_value=httpx.Response(404))
    with pytest.raises(crew_client.RoomGone):
        crew_client.room_info(CREW, CODE)
    respx.post(f"{CREW}/api/rooms").mock(return_value=httpx.Response(429))
    with pytest.raises(crew_client.CrewError, match="try again"):
        crew_client.create_room(CREW, "s", "u", "n", ("", "", ""))
    respx.post(f"{CREW}/api/rooms").mock(side_effect=httpx.ConnectError("offline"))
    with pytest.raises(crew_client.CrewError, match="cannot be reached"):
        crew_client.create_room(CREW, "s", "u", "n", ("", "", ""))
    respx.post(f"{CREW}/api/rooms").mock(return_value=httpx.Response(201, json={"code": "x"}))
    with pytest.raises(crew_client.CrewError, match="no invite code"):
        crew_client.create_room(CREW, "s", "u", "n", ("", "", ""))


def test_connect_room_talks_to_a_socket_and_knows_a_gone_room() -> None:
    from websockets.http11 import Response
    from websockets.sync.server import serve

    def handler(socket: Any) -> None:
        socket.send(json.dumps({"echo": json.loads(socket.recv())}))

    def gate(socket: Any, request: Any) -> Response | None:
        return None if request.path == f"/api/rooms/{CODE}/ws" else socket.respond(404, "gone")

    with serve(handler, "127.0.0.1", 0, process_request=gate) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        with crew_client.connect_room(base, CODE) as socket:
            socket.send(json.dumps({"t": "hello"}))
            assert json.loads(socket.recv(timeout=5)) == {"echo": {"t": "hello"}}
        with pytest.raises(crew_client.RoomGone):
            crew_client.connect_room(base, "cd" * 32)
        server.shutdown()
