"""Desktop login: PKCE, the loopback flow against a fake Auth service, reconcile, errors."""

from __future__ import annotations

import base64
import hashlib
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest
import respx

from scanlake_desktop.auth.client import (
    AuthClient,
    AuthUnavailable,
    Me,
    NotLoggedIn,
    UserIdConflict,
)
from scanlake_desktop.auth.login import LoginCancelled, LoopbackReceiver, challenge_of, login
from scanlake_desktop.auth.reconcile import LocalProfile, reconcile

LOCAL_ID = "11111111-1111-4111-8111-111111111111"
PRIMARY = "22222222-2222-4222-8222-222222222222"


def test_pkce_rfc7636_vector() -> None:
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert challenge_of(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


class FakeAuth:
    """Just enough of ScanLake Auth's desktop endpoints, with real PKCE and code checks."""

    def __init__(self, decision: str = "allow") -> None:
        self.codes: dict[str, tuple[str, str]] = {}  # code -> (challenge, redirect_uri)
        self.decision = decision
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                assert url.path == "/desktop/authorize"
                assert q["client_id"] == "scanlake-desktop" and q["code_challenge_method"] == "S256"
                if fake.decision == "allow":
                    fake.codes["code-1"] = (q["code_challenge"], q["redirect_uri"])
                    params = {"code": "code-1", "state": q["state"]}
                else:
                    params = {"error": "access_denied", "state": q["state"]}
                self.send_response(303)
                self.send_header("Location", f"{q['redirect_uri']}?{urlencode(params)}")
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"])).decode()
                form = {k: v[0] for k, v in parse_qs(body).items()}
                challenge, redirect = fake.codes.pop(form["code"], ("", ""))
                digest = hashlib.sha256(form["code_verifier"].encode()).digest()
                expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                ok = challenge == expected and redirect == form["redirect_uri"]
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = (
                    b'{"access_token":"T","token_type":"Bearer","expires_in":7776000}'
                    if ok
                    else b'{"error":"invalid_grant"}'
                )
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()


def _browser(url: str) -> None:
    """Stands in for the system browser: follows the redirects into the loopback listener."""
    threading.Thread(
        target=lambda: httpx.get(url, follow_redirects=True, timeout=10), daemon=True
    ).start()


def test_login_flow() -> None:
    fake = FakeAuth()
    try:
        assert login(AuthClient(fake.url), _browser, timeout=10) == "T"
    finally:
        fake.close()


def test_login_denied() -> None:
    fake = FakeAuth(decision="deny")
    try:
        with pytest.raises(LoginCancelled, match="denied"):
            login(AuthClient(fake.url), _browser, timeout=10)
    finally:
        fake.close()


def test_loopback_ignores_wrong_state_and_can_be_cancelled() -> None:
    receiver = LoopbackReceiver("right")
    try:
        r = httpx.get(f"{receiver.redirect_uri}?code=x&state=wrong")
        assert r.status_code == 400 and receiver.result is None
        assert receiver.redirect_uri.startswith("http://127.0.0.1:")
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(LoginCancelled):
            receiver.wait(5, cancel)
    finally:
        receiver.close()


ME = Me("d1", "mort", "Mort", "", None, (), "", "")


class StubClient(AuthClient):
    def __init__(self, me: Me, conflict: bool = False) -> None:
        self.me_value, self.conflict = me, conflict
        self.linked: list[str] = []
        self.pushed: list[tuple[str, str]] = []

    def link_user_id(self, token: str, user_id: str) -> Me:
        if self.conflict:
            raise UserIdConflict()
        self.linked.append(user_id)
        primary = self.me_value.primary_user_id or user_id
        self.me_value = replace(self.me_value, primary_user_id=primary,
                                user_ids=(*self.me_value.user_ids, user_id))  # fmt: skip
        return self.me_value

    def put_profile(self, token: str, user_name: str, org: str) -> Me:
        self.pushed.append((user_name, org))
        self.me_value = replace(self.me_value, user_name=user_name, org=org)
        return self.me_value


def test_reconcile_first_login_links_and_pushes_profile() -> None:
    client = StubClient(ME)
    local = LocalProfile(LOCAL_ID, "Mort13", "RoseCorp")
    result = reconcile(client, "T", ME, local, has_local_data=False)
    assert client.linked == [LOCAL_ID]
    assert result.profile == local  # the local ID became the primary
    assert client.pushed == [("Mort13", "RoseCorp")]


def test_reconcile_adopts_primary_and_skips_fresh_id() -> None:
    me = replace(ME, primary_user_id=PRIMARY, user_ids=(PRIMARY,), user_name="Mort", org="Rose")
    client = StubClient(me)
    result = reconcile(client, "T", me, LocalProfile(LOCAL_ID, "", ""), has_local_data=False)
    assert client.linked == []  # an unused fresh ID is not linked
    assert result.profile == LocalProfile(PRIMARY, "Mort", "Rose")
    assert result.changed(LocalProfile(LOCAL_ID, "", ""))


def test_reconcile_links_used_id_and_reports_conflict() -> None:
    me = replace(ME, primary_user_id=PRIMARY, user_ids=(PRIMARY,))
    client = StubClient(me, conflict=True)
    result = reconcile(client, "T", me, LocalProfile(LOCAL_ID, "A", "B"), has_local_data=True)
    assert result.notice and "another Discord account" in result.notice
    assert result.profile.user_id == PRIMARY


@respx.mock
def test_client_error_mapping() -> None:
    base = "https://auth.test"
    client = AuthClient(base)
    respx.get(f"{base}/api/me").mock(
        side_effect=[httpx.Response(401), httpx.Response(503), httpx.ConnectError("x")]
    )
    assert client.me("T") is None
    with pytest.raises(AuthUnavailable):
        client.me("T")
    with pytest.raises(AuthUnavailable):
        client.me("T")
    respx.post(f"{base}/api/me/user-ids").mock(return_value=httpx.Response(409))
    with pytest.raises(UserIdConflict):
        client.link_user_id("T", LOCAL_ID)
    respx.post(f"{base}/logout").mock(return_value=httpx.Response(401))
    client.logout("T")  # an expired token is already logged out
    respx.delete(f"{base}/api/me").mock(return_value=httpx.Response(401))
    with pytest.raises(NotLoggedIn):
        client.delete_account("T")
    route = respx.get(f"{base}/api/me").mock(
        return_value=httpx.Response(200, json={"discord": {"id": "1", "username": "m"},
                                               "primaryUserId": None, "userIds": [],
                                               "profile": {"userName": "", "org": ""}})
    )  # fmt: skip
    assert client.me("T").username == "m"  # type: ignore[union-attr]
    assert route.calls.last.request.headers["Authorization"] == "Bearer T"
