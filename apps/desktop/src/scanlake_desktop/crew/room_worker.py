"""Keeps the connection to a shared session's room, on its own thread.

One thread owns the socket: it receives with a short timeout and sends what was queued in
between. After every (re)connect it says hello; what was queued before is dropped, because the
controller brings the room up to date from the snapshot that follows.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal
from websockets.exceptions import ConnectionClosed

from scanlake_io import crew_client

log = logging.getLogger(__name__)

CONNECTING = "connecting"
ONLINE = "online"
OFFLINE = "offline"  # lost or unreachable: retrying
GONE = "gone"  # the room no longer exists: not retrying
OUTDATED = "outdated"  # the room speaks a newer protocol: not retrying

RECV_S = 0.2
BACKOFF_MAX_S = 30.0
CLOSE_GONE = 4404
CLOSE_VERSION = 4409


class RoomWorker(QThread):
    message = Signal(object)  # dict: a snapshot or one change from the room
    connection_changed = Signal(str)  # CONNECTING | ONLINE | OFFLINE | GONE | OUTDATED

    def __init__(
        self,
        base_url: str,
        code: str,
        hello: Callable[[], dict[str, Any]],
        user_agent: str = "ScanLakeDesktop",
        connect: Callable[..., Any] = crew_client.connect_room,
        backoff_s: float = 1.0,
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._code = code
        self._hello = hello
        self._user_agent = user_agent
        self._connect = connect
        self._backoff_s = backoff_s
        self._outbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop = threading.Event()

    def send(self, message: dict[str, Any]) -> None:
        """Queues a message; it is dropped if the connection is lost before it went out."""
        self._outbox.put(message)

    def stop(self) -> None:
        self._stop.set()
        self.wait(5000)

    def run(self) -> None:
        backoff = self._backoff_s
        while not self._stop.is_set():
            self.connection_changed.emit(CONNECTING)
            try:
                socket = self._connect(self._base_url, self._code, user_agent=self._user_agent)
            except crew_client.RoomGone:
                self.connection_changed.emit(GONE)
                return
            except Exception as err:  # offline, DNS, TLS, a 5xx: try again later
                log.debug("shared session: connecting failed: %s", err)
                socket = None
            if socket is not None:
                backoff = self._backoff_s
                final = self._talk(socket)
                if final is not None:
                    self.connection_changed.emit(final)
                    return
            if self._stop.is_set():
                return
            self.connection_changed.emit(OFFLINE)
            self._stop.wait(backoff * random.uniform(0.8, 1.2))
            backoff = min(BACKOFF_MAX_S, backoff * 2)

    def _talk(self, socket: Any) -> str | None:
        """Runs one connection until it is lost; GONE / OUTDATED when retrying is pointless."""
        try:
            while not self._outbox.empty():  # from before this connection
                self._outbox.get_nowait()
            socket.send(json.dumps(self._hello()))
            self.connection_changed.emit(ONLINE)
            while not self._stop.is_set():
                try:
                    raw = socket.recv(timeout=RECV_S)
                except TimeoutError:
                    pass
                else:
                    self._received(raw)
                while True:
                    try:
                        out = self._outbox.get_nowait()
                    except queue.Empty:
                        break
                    socket.send(json.dumps(out))
        except ConnectionClosed as err:
            code = err.rcvd.code if err.rcvd is not None else None
            if code == CLOSE_GONE:
                return GONE
            if code == CLOSE_VERSION:
                return OUTDATED
            log.debug("shared session: connection closed (%s)", err)
        except Exception as err:
            log.warning("shared session: connection failed: %s", err)
        finally:
            try:
                socket.close()
            except Exception:
                pass
        return None

    def _received(self, raw: object) -> None:
        if not isinstance(raw, str):
            return
        try:
            message = json.loads(raw)
        except ValueError:
            return
        if isinstance(message, dict):
            self.message.emit(message)
