"""ScanLake Crew client: shared sessions (the room other players join by invite link).

    POST /api/rooms {sessionId, userId, name, location}  -> 201 {code}
    GET  /api/rooms/<code>                               -> {sessionId, location, hostName, members}
    GET  /api/rooms/<code>/ws                            -> WebSocket (scanlake_core.domain.crew)
    GET  /j/<code>                                       -> the invite page (what the link opens)

The code in an invite link is the only key to a room. The app always talks to its own
configured Crew URL and takes nothing but the code from a pasted link.
"""

from __future__ import annotations

import re
import ssl
from dataclasses import dataclass
from typing import Any

import httpx
from websockets.exceptions import InvalidStatus
from websockets.sync.client import ClientConnection, connect

DEFAULT_CREW = "https://crew.scanlake.rocks"
TIMEOUT_S = 10.0
MAX_MESSAGE_BYTES = 4 * 1024 * 1024  # a snapshot of a full room stays far below this

_CODE = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", re.IGNORECASE)


class CrewError(Exception):
    """The Crew service refused a request or could not be reached."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RoomGone(CrewError):
    """The shared session no longer exists (it expires a day after the last member left)."""


@dataclass(frozen=True)
class RoomInfo:
    session_id: str
    system: str
    gravity_well: str
    region: str
    host_name: str
    members: int


def parse_invite(text: str) -> str | None:
    """The room code in a pasted invite link (or a bare code); None when there is none."""
    found = _CODE.findall(text or "")
    return found[-1].lower() if found else None


def invite_link(base_url: str, code: str) -> str:
    return f"{base_url.rstrip('/')}/j/{code}"


def ws_url(base_url: str, code: str) -> str:
    base = base_url.rstrip("/")
    scheme, _, rest = base.partition("://")
    return f"{'wss' if scheme == 'https' else 'ws'}://{rest}/api/rooms/{code}/ws"


def _request(
    method: str,
    url: str,
    *,
    user_agent: str,
    http: httpx.Client | None,
    json: dict[str, Any] | None = None,
) -> httpx.Response:
    try:
        if http is None:
            with httpx.Client(timeout=TIMEOUT_S, headers={"User-Agent": user_agent}) as client:
                return client.request(method, url, json=json)
        return http.request(method, url, json=json, timeout=TIMEOUT_S)
    except httpx.HTTPError as err:
        raise CrewError(f"The shared-session service cannot be reached ({err})") from err


def create_room(
    base_url: str,
    session_id: str,
    user_id: str,
    name: str,
    location: tuple[str, str, str],
    *,
    user_agent: str = "ScanLakeDesktop",
    http: httpx.Client | None = None,
) -> str:
    """Creates the room for a session and returns its invite code."""
    system, gravity_well, region = location
    body = {
        "sessionId": session_id,
        "userId": user_id,
        "name": name,
        "location": {"system": system, "gravityWell": gravity_well, "region": region},
    }
    response = _request(
        "POST", base_url.rstrip("/") + "/api/rooms", user_agent=user_agent, http=http, json=body
    )
    if response.status_code == 429:
        raise CrewError("Too many shared sessions started; try again in a minute", 429)
    if not response.is_success:
        raise CrewError(f"Sharing failed (HTTP {response.status_code})", response.status_code)
    code = parse_invite(str(response.json().get("code", "")))
    if code is None:
        raise CrewError("Sharing failed (no invite code in the answer)")
    return code


def room_info(
    base_url: str,
    code: str,
    *,
    user_agent: str = "ScanLakeDesktop",
    http: httpx.Client | None = None,
) -> RoomInfo:
    """What a room says about itself before joining; RoomGone when it does not exist."""
    response = _request(
        "GET", f"{base_url.rstrip('/')}/api/rooms/{code}", user_agent=user_agent, http=http
    )
    if response.status_code == 404:
        raise RoomGone("This shared session does not exist any more", 404)
    if not response.is_success:
        raise CrewError(f"Lookup failed (HTTP {response.status_code})", response.status_code)
    data = response.json()
    location = data.get("location") or {}
    return RoomInfo(
        session_id=str(data.get("sessionId", "")),
        system=str(location.get("system", "")),
        gravity_well=str(location.get("gravityWell", "")),
        region=str(location.get("region", "")),
        host_name=str(data.get("hostName", "")),
        members=int(data.get("members", 0)),
    )


def connect_room(
    base_url: str, code: str, *, user_agent: str = "ScanLakeDesktop"
) -> ClientConnection:
    """Opens a room's socket; RoomGone when the room does not exist. The caller says hello."""
    url = ws_url(base_url, code)
    context = None
    if url.startswith("wss://"):
        import certifi  # the CA bundle httpx uses too (the frozen app has no system store path)

        context = ssl.create_default_context(cafile=certifi.where())
    try:
        return connect(
            url,
            ssl=context,
            user_agent_header=user_agent,
            open_timeout=TIMEOUT_S,
            close_timeout=2,
            max_size=MAX_MESSAGE_BYTES,
            legacy=True,  # a plain connection, closed by the caller (not a `with` block)
        )
    except InvalidStatus as err:
        if err.response.status_code == 404:
            raise RoomGone("This shared session does not exist any more", 404) from err
        raise CrewError(
            f"Joining failed (HTTP {err.response.status_code})", err.response.status_code
        ) from err
