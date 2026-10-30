"""ScanLake Auth client for the desktop app (bearer token instead of the web cookie).

GET    /desktop/authorize   (opened in the browser, see login.py)
POST   /desktop/token       code + PKCE verifier -> {access_token, token_type, expires_in}
GET    /api/me              401 = not logged in (token unknown or expired)
POST   /api/me/user-ids     {userId}           409 = owned by another account, 422 = limit
PUT    /api/me/profile      {userName, org}
POST   /logout              ends this desktop session only
DELETE /api/me              unlinks Discord (account, IDs, all sessions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

CLIENT_ID = "scanlake-desktop"
DEFAULT_AUTH = "https://auth.scanlake.rocks"


class AuthError(Exception):
    """The Auth service refused or failed a request."""


class AuthUnavailable(AuthError):
    """Network error or 5xx: keep working with the local profile."""


class NotLoggedIn(AuthError):
    """401: the token is unknown or expired; log in again."""


class UserIdConflict(AuthError):
    def __init__(self) -> None:
        super().__init__("This ScanLake ID is already linked to another Discord account")


class UserIdLimit(AuthError):
    def __init__(self) -> None:
        super().__init__("This Discord account already has the maximum number of linked IDs")


@dataclass(frozen=True)
class Me:
    discord_id: str
    username: str
    global_name: str
    avatar_url: str
    primary_user_id: str | None
    user_ids: tuple[str, ...]
    user_name: str
    org: str

    @property
    def display_name(self) -> str:
        return self.global_name or self.username

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Me:
        discord = data.get("discord") or {}
        profile = data.get("profile") or {}
        return cls(
            discord_id=str(discord.get("id", "")),
            username=str(discord.get("username", "")),
            global_name=str(discord.get("globalName") or ""),
            avatar_url=str(discord.get("avatarUrl") or ""),
            primary_user_id=data.get("primaryUserId") or None,
            user_ids=tuple(data.get("userIds") or ()),
            user_name=str(profile.get("userName") or ""),
            org=str(profile.get("org") or ""),
        )


class AuthClient:
    def __init__(
        self, base_url: str = DEFAULT_AUTH, *, user_agent: str = "ScanLakeDesktop",
        http: httpx.Client | None = None,
    ) -> None:  # fmt: skip
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=15.0, headers={"User-Agent": user_agent})

    def authorize_url(self, redirect_uri: str, state: str, challenge: str) -> str:
        query = urlencode(
            {
                "response_type": "code",
                "client_id": CLIENT_ID,
                "redirect_uri": redirect_uri,
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        return f"{self.base_url}/desktop/authorize?{query}"

    def exchange(self, code: str, verifier: str, redirect_uri: str) -> str:
        """The bearer token for a one-time login code."""
        response = self._send(
            "POST",
            "/desktop/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "client_id": CLIENT_ID,
            },
        )
        token = response.json().get("access_token")
        if not isinstance(token, str) or not token:
            raise AuthError("the Auth service returned no token")
        return token

    def me(self, token: str) -> Me | None:
        """The account, or None when the token is no longer valid."""
        try:
            return Me.from_json(self._send("GET", "/api/me", token=token).json())
        except NotLoggedIn:
            return None

    def link_user_id(self, token: str, user_id: str) -> Me:
        return Me.from_json(
            self._send("POST", "/api/me/user-ids", token=token, json={"userId": user_id}).json()
        )

    def put_profile(self, token: str, user_name: str, org: str) -> Me:
        body = {"userName": user_name[:64], "org": org[:64]}
        return Me.from_json(self._send("PUT", "/api/me/profile", token=token, json=body).json())

    def logout(self, token: str) -> None:
        try:
            self._send("POST", "/logout", token=token)
        except NotLoggedIn:
            pass  # already gone

    def delete_account(self, token: str) -> None:
        self._send("DELETE", "/api/me", token=token)

    def _send(
        self, method: str, path: str, *, token: str | None = None, **kwargs: Any
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            response = self.http.request(method, self.base_url + path, headers=headers, **kwargs)
        except httpx.HTTPError as err:
            raise AuthUnavailable(f"cannot reach the login service: {err}") from err
        if response.is_success:
            return response
        status = response.status_code
        if status == 401:
            raise NotLoggedIn("not logged in")
        if status == 409:
            raise UserIdConflict()
        if status == 422:
            raise UserIdLimit()
        if status >= 500:
            raise AuthUnavailable(f"login service error {status}")
        try:
            detail = response.json().get("error", "")
        except ValueError:
            detail = ""
        raise AuthError(f"login service refused the request ({status} {detail})".strip())
