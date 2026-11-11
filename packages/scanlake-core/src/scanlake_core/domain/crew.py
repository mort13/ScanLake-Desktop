"""Shared sessions: what the members of one session know about each other.

Every member keeps their own scans and clusters and publishes one summary per cluster to the
session's room (ScanLake Crew). The room relays them and holds each cluster's status and who
is heading where. This module is the client's picture of the room (`RoomState`), the wire
form of a cluster, and the cards the Overview shows (own and other members' clusters, in the
order need help, on the way, open, done; newest scan first).

Wire protocol: `ScanLake Crew/shared/protocol.ts`. No Qt, no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from scanlake_core.domain.navigation import ClusterSummary

PROTOCOL_VERSION = 1

OPEN = "open"
DONE = "done"
NEED_HELP = "need_help"
STATUSES = (OPEN, DONE, NEED_HELP)

# Role id on the wire -> what the Crew tab shows, in this order
ROLES: dict[str, str] = {
    "scout": "Scout",
    "laser": "Laser Operator",
    "pilot": "Pilot",
    "hauler": "Cargo Hauler",
}

NAME_MAX = 40  # the room's limit for a name, ship or crew
YOU = "You"

Wire = dict[str, Any]


@dataclass(frozen=True)
class Member:
    user_id: str
    name: str
    roles: tuple[str, ...] = ()
    ship: str = ""
    crew: str = ""
    online: bool = False
    joined_at: float = 0.0

    @property
    def role_names(self) -> str:
        return ", ".join(ROLES[r] for r in self.roles if r in ROLES)


@dataclass(frozen=True)
class SharedCluster:
    """One card of the Overview: a cluster, whose it is, its status and who is heading there."""

    summary: ClusterSummary
    mine: bool = True
    owner_id: str = ""
    status: str = OPEN
    enroute: tuple[str, ...] = ()  # names; "You" for yourself

    @property
    def cluster_id(self) -> str:
        return self.summary.cluster_id

    @property
    def tier(self) -> int:
        """Need help, then open ones someone is heading to, then open, then done."""
        if self.status == NEED_HELP:
            return 0
        if self.status == DONE:
            return 3
        return 1 if self.enroute else 2


def order_cards(cards: Iterable[SharedCluster]) -> list[SharedCluster]:
    """By tier, then newest scan first (then by id, so the order never flickers)."""
    ordered = sorted(cards, key=lambda c: c.cluster_id)
    ordered.sort(key=lambda c: c.summary.last_scan, reverse=True)
    ordered.sort(key=lambda c: c.tier)
    return ordered


def label(summary: ClusterSummary) -> str:
    """ "#3 · Gold", or "Alice #3 · Gold" for another member's cluster."""
    who = f"{summary.owner} " if summary.owner else ""
    return f"{who}#{summary.number} · {summary.deposit or 'unknown deposit'}"


# -- wire ---------------------------------------------------------------------------------


def cluster_to_wire(summary: ClusterSummary) -> Wire:
    """What is published about an own cluster (positions to 0.1 m, so re-sends compare equal)."""
    return {
        "clusterId": summary.cluster_id,
        "number": summary.number,
        "deposit": summary.deposit,
        "rocks": summary.rocks,
        "totalScu": round(summary.total_scu, 3),
        "localZone": summary.local_zone,
        "centres": {
            zone: [round(c, 1) for c in centre] for zone, centre in summary.centres.items()
        },
        "firstScan": summary.first_scan,
        "lastScan": summary.last_scan,
    }


PAYLOAD_KEYS = tuple(cluster_to_wire(ClusterSummary("", 0, "", 0, 0.0, "")))


def payload(wire: Mapping[str, Any]) -> Wire:
    """The part of a room cluster its owner published (without owner and status)."""
    return {key: wire.get(key) for key in PAYLOAD_KEYS}


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def summary_from_wire(wire: Mapping[str, Any], owner: str = "") -> ClusterSummary:
    """Another member's cluster as a `ClusterSummary`, which the navigation takes as it is."""
    centres: dict[str, tuple[float, float, float]] = {}
    raw = wire.get("centres")
    if isinstance(raw, Mapping):
        for zone, value in raw.items():
            if isinstance(zone, str) and isinstance(value, Sequence) and len(value) == 3:
                centres[zone] = (_number(value[0]), _number(value[1]), _number(value[2]))
    return ClusterSummary(
        cluster_id=_text(wire.get("clusterId")),
        number=int(_number(wire.get("number"))),
        deposit=_text(wire.get("deposit")),
        rocks=int(_number(wire.get("rocks"))),
        total_scu=_number(wire.get("totalScu")),
        local_zone=_text(wire.get("localZone")),
        centres=centres,
        first_scan=_text(wire.get("firstScan")),
        last_scan=_text(wire.get("lastScan")),
        owner=owner,
    )


def diff_clusters(
    own: Sequence[ClusterSummary], sent: Mapping[str, Wire]
) -> tuple[list[Wire], list[str]]:
    """What to upsert and remove so the room's copy of the own clusters (`sent`) matches."""
    upsert = []
    for summary in own:
        wire = cluster_to_wire(summary)
        if sent.get(summary.cluster_id) != wire:
            upsert.append(wire)
    mine = {s.cluster_id for s in own}
    return upsert, [cluster_id for cluster_id in sent if cluster_id not in mine]


def _member(wire: Mapping[str, Any]) -> Member:
    roles = wire.get("roles")
    return Member(
        user_id=_text(wire.get("userId")),
        name=_text(wire.get("name")),
        roles=tuple(r for r in roles if r in ROLES) if isinstance(roles, list) else (),
        ship=_text(wire.get("ship")),
        crew=_text(wire.get("crew")),
        online=wire.get("online") is True,
        joined_at=_number(wire.get("joinedAt")),
    )


# -- the room, as this client knows it ----------------------------------------------------


@dataclass
class RoomState:
    you: str = ""  # your user id in the room
    session_id: str = ""
    host_id: str = ""
    location: tuple[str, str, str] = ("", "", "")  # system, gravity well, region
    members: dict[str, Member] = field(default_factory=dict)
    clusters: dict[str, Wire] = field(default_factory=dict)  # every cluster, own ones too
    enroute: dict[str, str] = field(default_factory=dict)  # user id -> cluster id
    server_now_ms: float = 0.0  # the room's clock at the last snapshot

    def name(self, user_id: str) -> str:
        if user_id == self.you:
            return YOU
        member = self.members.get(user_id)
        return (member.name if member else "") or "Someone"

    def heading_to(self, cluster_id: str) -> list[str]:
        """User ids of the members on the way to a cluster, in joining order."""
        return [user_id for user_id in self.members if self.enroute.get(user_id) == cluster_id]

    def apply(self, message: Mapping[str, Any]) -> None:
        """Takes a snapshot or one change from the room; anything unknown is ignored."""
        kind = message.get("t")
        if kind == "snapshot":
            room = message.get("room") or {}
            location = room.get("location") or {}
            self.you = _text(message.get("you"))
            self.session_id = _text(room.get("sessionId"))
            self.host_id = _text(room.get("hostId"))
            self.location = (
                _text(location.get("system")),
                _text(location.get("gravityWell")),
                _text(location.get("region")),
            )
            self.server_now_ms = _number(message.get("now"))
            members = [_member(m) for m in message.get("members") or []]
            self.members = {m.user_id: m for m in members}
            self.clusters = {
                _text(c.get("clusterId")): dict(c) for c in message.get("clusters") or []
            }
            self.enroute = {
                _text(e.get("userId")): _text(e.get("clusterId"))
                for e in message.get("enroute") or []
            }
        elif kind == "member":
            member = _member(message.get("member") or {})
            if member.user_id:
                self.members[member.user_id] = member
        elif kind == "presence":
            user_id = _text(message.get("userId"))
            member = self.members.get(user_id)
            online = message.get("online") is True
            if member is not None:
                self.members[user_id] = replace(member, online=online)
            if not online:
                self.enroute.pop(user_id, None)
        elif kind == "cluster":
            cluster = dict(message.get("cluster") or {})
            if cluster.get("clusterId"):
                self.clusters[_text(cluster.get("clusterId"))] = cluster
        elif kind == "cluster_removed":
            cluster_id = _text(message.get("clusterId"))
            self.clusters.pop(cluster_id, None)
            self.enroute = {u: c for u, c in self.enroute.items() if c != cluster_id}
        elif kind == "status":
            cluster = self.clusters.get(_text(message.get("clusterId")))
            if cluster is not None and message.get("status") in STATUSES:
                cluster["status"] = message.get("status")
                cluster["statusBy"] = message.get("by")
                cluster["statusAt"] = message.get("at")
        elif kind == "enroute":
            user_id = _text(message.get("userId"))
            cluster_id = message.get("clusterId")
            if isinstance(cluster_id, str) and cluster_id:
                self.enroute[user_id] = cluster_id
            else:
                self.enroute.pop(user_id, None)


def build_cards(
    own: Sequence[ClusterSummary],
    statuses: Mapping[str, str],
    room: RoomState | None = None,
    target: str | None = None,
) -> list[SharedCluster]:
    """The Overview's cards, in order: the own clusters (their status from `statuses`) and, in
    a shared session, the other members'. `target` is the cluster you are navigating to."""

    def heading(cluster_id: str) -> tuple[str, ...]:
        names = [YOU] if target == cluster_id else []
        if room is not None:
            names += [room.name(u) for u in room.heading_to(cluster_id) if u != room.you]
        return tuple(names)

    mine = {s.cluster_id for s in own}
    cards = [
        SharedCluster(
            summary=s,
            status=statuses.get(s.cluster_id, OPEN),
            enroute=heading(s.cluster_id),
        )
        for s in own
    ]
    if room is not None:
        for cluster_id, wire in room.clusters.items():
            if cluster_id in mine or not cluster_id:
                continue
            owner_id = _text(wire.get("ownerId"))
            status = wire.get("status")
            cards.append(
                SharedCluster(
                    summary=summary_from_wire(wire, room.name(owner_id)),
                    mine=False,
                    owner_id=owner_id,
                    status=status if status in STATUSES else OPEN,
                    enroute=heading(cluster_id),
                )
            )
    return order_cards(cards)


@dataclass(frozen=True)
class Notice:
    text: str
    urgent: bool = False  # someone needs help, or is heading to one of your clusters


def notice_for(
    message: Mapping[str, Any], room: RoomState, own: Mapping[str, ClusterSummary]
) -> Notice | None:
    """What to tell the player about a change from the room. Call it BEFORE `apply` (it
    compares with what was known), and never for a snapshot."""

    def name_of(cluster_id: str) -> str | None:
        if cluster_id in own:
            return "your " + label(own[cluster_id])
        wire = room.clusters.get(cluster_id)
        if wire is None:
            return None
        return label(summary_from_wire(wire, room.name(_text(wire.get("ownerId")))))

    kind = message.get("t")
    if kind == "enroute":
        cluster_id = message.get("clusterId")
        who = room.name(_text(message.get("userId")))
        if not isinstance(cluster_id, str) or not (where := name_of(cluster_id)):
            return None
        return Notice(f"{who} is on the way to {where}", urgent=cluster_id in own)
    if kind == "status":
        cluster_id = _text(message.get("clusterId"))
        who = room.name(_text(message.get("by")))
        where = name_of(cluster_id)
        if where is None:
            return None
        status = message.get("status")
        if status == NEED_HELP:
            return Notice(f"{who} needs help at {where}", urgent=True)
        if status == DONE:
            return Notice(f"{who} marked {where} done")
        return Notice(f"{who} reopened {where}")
    if kind == "member":
        member = _member(message.get("member") or {})
        before = room.members.get(member.user_id)
        if member.online and (before is None or not before.online):
            return Notice(f"{member.name or 'Someone'} joined the session")
        return None
    if kind == "presence" and message.get("online") is False:
        user_id = _text(message.get("userId"))
        if user_id in room.members:
            return Notice(f"{room.name(user_id)} left the session")
    return None
