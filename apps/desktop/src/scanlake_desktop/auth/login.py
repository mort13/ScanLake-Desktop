"""Discord login through the system browser (RFC 8252 loopback redirect + RFC 7636 PKCE).

1. Make a PKCE verifier/challenge and a random state, listen on http://127.0.0.1:<free port>.
2. Open /desktop/authorize in the browser; the user logs in with Discord and clicks Allow.
3. The browser is redirected to http://127.0.0.1:<port>/callback?code=...&state=...
4. Check the state, exchange code + verifier for a bearer token.

The listener accepts exactly one valid callback, answers with a static page (it never echoes
parameters back) and gives up after five minutes or when cancelled.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from scanlake_desktop.auth.client import AuthClient, AuthError

TIMEOUT_S = 300

DONE_PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>ScanLake Desktop</title>
<style>body{font-family:system-ui,sans-serif;background:#16181d;color:#e6e6e6;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0}div{text-align:center}</style>
</head><body><div><h2>%s</h2><p>You can close this tab and go back to ScanLake Desktop.</p>
</div></body></html>"""


class LoginCancelled(AuthError):
    pass


def make_verifier() -> str:
    """43-128 characters of [A-Za-z0-9-._~] (token_urlsafe(64) gives 86)."""
    return secrets.token_urlsafe(64)


def challenge_of(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


@dataclass
class Callback:
    code: str | None = None
    error: str | None = None


class LoopbackReceiver:
    """A one-shot HTTP listener on 127.0.0.1 for the login redirect."""

    def __init__(self, state: str) -> None:
        self.state = state
        self.result: Callback | None = None
        self._done = threading.Event()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                url = urlparse(self.path)
                if url.path != "/callback":
                    self.send_error(404)
                    return
                params = {k: v[0] for k, v in parse_qs(url.query).items()}
                if params.get("state") != receiver.state:
                    self.send_error(400, "state mismatch")
                    return  # someone else's request: keep waiting for the real one
                if params.get("code"):
                    receiver._finish(Callback(code=params["code"]), "Logged in")
                else:
                    receiver._finish(
                        Callback(error=params.get("error", "unknown")), "Login cancelled"
                    )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(DONE_PAGE % receiver._title.encode())

            def log_message(self, format: str, *args: object) -> None:  # silence stderr
                pass

        self._title = ""
        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/callback"

    def _finish(self, result: Callback, title: str) -> None:
        if self.result is None:
            self.result = result
            self._title = title
            self._done.set()

    def wait(self, timeout: float, cancelled: threading.Event | None = None) -> Callback:
        deadline = timeout
        while deadline > 0:
            if self._done.wait(0.2):
                return self.result or Callback(error="unknown")
            if cancelled is not None and cancelled.is_set():
                raise LoginCancelled("login cancelled")
            deadline -= 0.2
        raise LoginCancelled("login timed out")

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def login(
    client: AuthClient,
    open_browser: Callable[[str], object],
    cancelled: threading.Event | None = None,
    timeout: float = TIMEOUT_S,
) -> str:
    """Runs the whole browser login (blocking; call it off the GUI thread). Returns the token."""
    verifier, state = make_verifier(), secrets.token_urlsafe(24)
    receiver = LoopbackReceiver(state)
    try:
        open_browser(client.authorize_url(receiver.redirect_uri, state, challenge_of(verifier)))
        callback = receiver.wait(timeout, cancelled)
        if callback.code is None:
            if callback.error == "access_denied":
                raise LoginCancelled("login was denied in the browser")
            raise AuthError(f"login failed in the browser ({callback.error})")
        return client.exchange(callback.code, verifier, receiver.redirect_uri)
    finally:
        receiver.close()
