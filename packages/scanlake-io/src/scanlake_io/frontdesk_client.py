"""FrontDesk client: tells the landing page (scanlake.rocks) about capture sessions.

    POST /api/ping {sessionId, source: "desktop", kind: "open"|"discard"}  -> 204 (429 when busy)

`open` is sent when a session starts or is resumed, `discard` when it ends without anything to
upload. Closing is not reported here: the Store tells FrontDesk when the final batch arrives.
FrontDesk keeps only a hash of the session id. A ping is best effort and never raises.
"""

from __future__ import annotations

import logging
from typing import Literal

import httpx

DEFAULT_FRONTDESK = "https://scanlake.rocks"
TIMEOUT_S = 3.0

Kind = Literal["open", "discard"]

log = logging.getLogger(__name__)


def ping(
    base_url: str,
    session_id: str,
    kind: Kind,
    *,
    user_agent: str = "ScanLakeDesktop",
    http: httpx.Client | None = None,
) -> bool:
    """Reports `kind` for `session_id`; True when FrontDesk took it. An empty URL turns it off."""
    if not base_url:
        return False
    url = base_url.rstrip("/") + "/api/ping"
    body = {"sessionId": session_id, "source": "desktop", "kind": kind}
    try:
        if http is None:
            with httpx.Client(timeout=TIMEOUT_S, headers={"User-Agent": user_agent}) as client:
                response = client.post(url, json=body)
        else:
            response = http.post(url, json=body, timeout=TIMEOUT_S)
    except Exception as err:  # offline, DNS, timeout, bad URL: the landing page can do without
        log.debug("FrontDesk ping (%s) failed: %s", kind, err)
        return False
    if response.status_code == 429:
        log.debug("FrontDesk ping (%s) rate-limited", kind)
    elif not response.is_success:
        log.warning("FrontDesk ping (%s): HTTP %d", kind, response.status_code)
    return response.is_success
