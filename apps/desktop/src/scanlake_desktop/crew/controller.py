"""What the app knows and does about its shared session.

Without a shared session the controller still holds the clusters' done / need-help marks
(stored locally), so the Overview works the same alone. In a shared session it publishes the
own clusters, relays marks and the navigation target, and keeps the room's state for the
Overview's cards and the Crew tab.

The room does not echo a member's own changes back, so those are applied here at once.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from PySide6.QtCore import QObject, Signal

from scanlake_core.domain import crew
from scanlake_core.domain.crew import Member, RoomState, SharedCluster
from scanlake_core.domain.navigation import ClusterSummary
from scanlake_desktop.crew.room_worker import GONE, ONLINE, OUTDATED, RoomWorker
from scanlake_desktop.db.database import Database, ShareRow

log = logging.getLogger(__name__)


class CrewController(QObject):
    changed = Signal()  # cards, members or the connection changed
    notice = Signal(str, bool)  # text, urgent
    ended = Signal(str)  # the room is gone or too new: why (the session goes on alone)

    def __init__(
        self,
        db: Database,
        base_url: str,
        identity: Callable[[], tuple[str, str]],  # user id, display name
        profile: Callable[[], tuple[Sequence[str], str, str]],  # roles, ship, crew
        user_agent: str = "ScanLakeDesktop",
        worker_factory: Callable[..., Any] | None = None,  # RoomWorker
    ) -> None:
        super().__init__()
        self.db = db
        self.base_url = base_url
        self._identity = identity
        self._profile = profile
        self._user_agent = user_agent
        self._worker_factory = worker_factory
        self.worker: Any = None
        self.share: ShareRow | None = None
        self.room: RoomState | None = None  # None: the session is not shared
        self.connection = ""  # "" (not shared) or a room_worker state
        self.synced = False  # a snapshot arrived on this connection
        self.own: list[ClusterSummary] = []
        self.target: str | None = None
        self.statuses: dict[str, tuple[str, bool]] = db.cluster_statuses()
        self._sent: dict[str, crew.Wire] = {}  # the room's copy of the own clusters
        self._hello: dict[str, Any] = {}

    # -- state ------------------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """False when shared sessions are turned off (no Crew URL)."""
        return bool(self.base_url)

    @property
    def shared(self) -> bool:
        return self.share is not None

    @property
    def online(self) -> bool:
        return self.shared and self.connection == ONLINE and self.synced

    def cards(self) -> list[SharedCluster]:
        """The Overview's cards, in order."""
        statuses = {cluster_id: status for cluster_id, (status, _) in self.statuses.items()}
        return crew.build_cards(self.own, statuses, self.room, self.target)

    def members(self) -> list[Member]:
        return list(self.room.members.values()) if self.room is not None else []

    def crews(self) -> list[str]:
        """The crew names in use, for the Crew tab's suggestions."""
        return sorted({m.crew for m in self.members() if m.crew}, key=str.lower)

    def target_of(self, user_id: str) -> str:
        """ "#3 · Gold" a member is heading to, or ""."""
        if self.room is None:
            return ""
        cluster_id = self.target if user_id == self.room.you else self.room.enroute.get(user_id)
        card = next((c for c in self.cards() if c.cluster_id == cluster_id), None)
        return crew.label(card.summary) if card else ""

    # -- joining and leaving ----------------------------------------------------------------

    def open(self, share: ShareRow) -> None:
        """Connects to the room of a shared session (and reconnects for as long as it lasts)."""
        self.close()
        self.share = share
        self.room = RoomState()
        self._build_hello()
        self.worker = (self._worker_factory or RoomWorker)(
            self.base_url, share.code, lambda: self._hello, self._user_agent
        )
        self.worker.message.connect(self._on_message)
        self.worker.connection_changed.connect(self._on_connection)
        self.worker.start()
        self.changed.emit()

    def close(self) -> None:
        """Leaves the room (the session itself is untouched)."""
        if self.worker is not None:
            self.worker.message.disconnect(self._on_message)
            self.worker.connection_changed.disconnect(self._on_connection)
            self.worker.stop()
            self.worker = None
        had = self.share is not None
        self.share, self.room, self.connection, self.synced = None, None, "", False
        self._sent = {}
        if had:
            self.changed.emit()

    def rejoin(self) -> None:
        """The identity changed (Discord login): say hello again under the new user id."""
        if self.share is not None:
            self.open(self.share)

    # -- what this player does --------------------------------------------------------------

    def publish(self, own: Sequence[ClusterSummary]) -> None:
        """The own clusters as they are now; the room gets what changed."""
        self.own = list(own)
        if self.online:
            self._sync_clusters()

    def set_status(self, cluster_id: str, status: str) -> None:
        """Marks a cluster open, done or need help: an own one always, another member's only
        while connected."""
        if status not in crew.STATUSES:
            return
        mine = any(s.cluster_id == cluster_id for s in self.own)
        if not mine and not self.online:
            return
        if mine:
            dirty = self.shared and not self.online  # sent after the next snapshot
            self.db.set_cluster_status(cluster_id, status, dirty)
            self.statuses[cluster_id] = (status, dirty)
        if self.online and self.room is not None:
            known = self.room.clusters.get(cluster_id)
            if known is not None:
                known["status"] = status
                self._send({"t": "status", "clusterId": cluster_id, "status": status})
        self.changed.emit()

    def set_target(self, cluster_id: str | None) -> None:
        """The cluster this player navigates to (None: none), for the others to see."""
        if cluster_id == self.target:
            return
        self.target = cluster_id
        self._hello = {**self._hello, "target": cluster_id}
        if self.online:
            self._send({"t": "enroute", "clusterId": cluster_id})
        self.changed.emit()

    def set_profile(self) -> None:
        """Name, roles, ship or crew changed (the callables given at creation have them)."""
        self._build_hello()
        room = self.room
        if room is None:
            return
        hello = self._hello
        me = room.members.get(room.you)
        if me is not None:
            room.members[room.you] = replace(
                me,
                name=hello["name"],
                roles=tuple(hello["roles"]),
                ship=hello["ship"],
                crew=hello["crew"],
            )
        if self.online:
            self._send({"t": "profile", **{k: hello[k] for k in ("name", "roles", "ship", "crew")}})
        self.changed.emit()

    # -- from the room ----------------------------------------------------------------------

    def _on_connection(self, state: str) -> None:
        self.connection = state
        if state != ONLINE:
            self.synced = False
        if state in (GONE, OUTDATED):
            why = (
                "The shared session has ended; yours goes on alone."
                if state == GONE
                else "The shared session needs a newer ScanLake Desktop; yours goes on alone."
            )
            if self.share is not None:
                self.db.end_share(self.share.session_id)
            self.close()
            self.ended.emit(why)
            return
        self.changed.emit()

    def _on_message(self, message: dict[str, Any]) -> None:
        room = self.room
        if room is None:
            return
        kind = message.get("t")
        if kind == "error":
            log.info("shared session: %s", message.get("message"))
            return
        if kind == "snapshot":
            room.apply(message)
            self.synced = True
            self._reconcile()
            self.changed.emit()
            return
        if not self.synced:
            return
        note = crew.notice_for(message, room, {s.cluster_id: s for s in self.own})
        room.apply(message)
        if kind == "status":
            cluster_id = str(message.get("clusterId"))
            status = message.get("status")
            if any(s.cluster_id == cluster_id for s in self.own) and status in crew.STATUSES:
                self.db.set_cluster_status(cluster_id, status)
                self.statuses[cluster_id] = (status, False)
        if note is not None:
            self.notice.emit(note.text, note.urgent)
        self.changed.emit()

    def _reconcile(self) -> None:
        """After a snapshot: marks made while offline go to the room, the room's marks of the
        own clusters come here, and the room gets the own clusters as they are now."""
        room = self.room
        assert room is not None
        own_ids = {s.cluster_id for s in self.own}
        self._sent = {}
        for cluster_id, wire in room.clusters.items():
            if cluster_id not in own_ids:
                continue
            self._sent[cluster_id] = crew.payload(wire)
            local, dirty = self.statuses.get(cluster_id, (crew.OPEN, False))
            theirs = wire.get("status")
            if dirty:
                if theirs != local:
                    wire["status"] = local
                    self._send({"t": "status", "clusterId": cluster_id, "status": local})
                self.db.set_cluster_status(cluster_id, local)
                self.statuses[cluster_id] = (local, False)
            elif theirs in crew.STATUSES and theirs != local:
                self.db.set_cluster_status(cluster_id, theirs)
                self.statuses[cluster_id] = (theirs, False)
        self._sync_clusters()
        if self.target is not None and room.enroute.get(room.you) != self.target:
            self._send({"t": "enroute", "clusterId": self.target})
        room.enroute.pop(room.you, None)  # the own target is `self.target`

    def _sync_clusters(self) -> None:
        upsert, remove = crew.diff_clusters(self.own, self._sent)
        if not upsert and not remove:
            return
        room = self.room
        for wire in upsert:
            cluster_id = wire["clusterId"]
            self._sent[cluster_id] = wire
            status, dirty = self.statuses.get(cluster_id, (crew.OPEN, False))
            if dirty:  # a new cluster takes its mark along
                self.db.set_cluster_status(cluster_id, status)
                self.statuses[cluster_id] = (status, False)
            if room is not None:
                known = room.clusters.get(cluster_id, {})
                room.clusters[cluster_id] = {
                    **wire,
                    "ownerId": room.you,
                    "status": known.get("status", status),
                }
        for cluster_id in remove:
            self._sent.pop(cluster_id, None)
            if room is not None:
                room.clusters.pop(cluster_id, None)
        for start in range(0, max(len(upsert), 1), 40):  # the room takes 50 per message
            batch = [
                {**wire, "status": self.statuses.get(wire["clusterId"], (crew.OPEN, False))[0]}
                for wire in upsert[start : start + 40]
            ]
            self._send({"t": "clusters", "upsert": batch, "remove": remove if start == 0 else []})

    # -- internals --------------------------------------------------------------------------

    def _build_hello(self) -> None:
        user_id, name = self._identity()
        roles, ship, group = self._profile()
        self._hello = {
            "t": "hello",
            "v": crew.PROTOCOL_VERSION,
            "userId": user_id,
            "clientId": self.share.client_id if self.share else "",
            "name": (name.strip() or "Player")[: crew.NAME_MAX],
            "roles": [r for r in crew.ROLES if r in roles],
            "ship": ship.strip()[: crew.NAME_MAX],
            "crew": group.strip()[: crew.NAME_MAX],
            "target": self.target,
        }

    def _send(self, message: dict[str, Any]) -> None:
        if self.worker is not None:
            self.worker.send(message)
