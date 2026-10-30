"""ScanLake Store client: session tokens and batch uploads (same requests as the web app).

    POST /api/session {userId, sessionId}      -> {token, expiresAt}   (10 per IP per hour)
    POST /api/upload  multipart, Bearer token  -> {ok, keys}           (100 per user per hour)

Tokens are cached per (userId, sessionId) until shortly before they expire, so a long session
costs one token. Every failure raises `StoreError` saying whether a retry can help.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from scanlake_io.parquet import BatchFile

DEFAULT_STORE = "https://scanlake-store.mo-55d.workers.dev"
TOKEN_MARGIN_S = 300


class StoreError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        permanent: bool = False,
        violations: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.permanent = permanent
        self.violations = tuple(violations)


@dataclass(frozen=True)
class UploadReceipt:
    keys: tuple[str, ...]


@dataclass
class StoreClient:
    base_url: str = DEFAULT_STORE
    user_agent: str = "ScanLakeDesktop"
    timeout: float = 60.0
    http: httpx.Client | None = None
    _tokens: dict[tuple[str, str], tuple[str, float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.http is None:
            self.http = httpx.Client(timeout=self.timeout, headers={"User-Agent": self.user_agent})

    def forget_tokens(self) -> None:
        self._tokens.clear()

    def session_token(self, user_id: str, session_id: str) -> str:
        cached = self._tokens.get((user_id, session_id))
        if cached and cached[1] - TOKEN_MARGIN_S > time.time():
            return cached[0]
        response = self._send(
            "POST", "/api/session", json={"userId": user_id, "sessionId": session_id}
        )
        body = response.json()
        expires = _parse_time(body.get("expiresAt")) or time.time() + 3600
        self._tokens[(user_id, session_id)] = (body["token"], expires)
        return body["token"]

    def upload(
        self,
        user_id: str,
        session_id: str,
        batch_number: int,
        is_final: bool,
        files: Sequence[BatchFile],
    ) -> UploadReceipt:
        data = {
            "userId": user_id,
            "sessionId": session_id,
            "batchNumber": str(batch_number),
            "isFinal": "true" if is_final else "false",
        }
        multipart = {f.kind: (f.filename, f.data, "application/vnd.apache.parquet") for f in files}
        for attempt in (1, 2):
            token = self.session_token(user_id, session_id)
            try:
                response = self._send(
                    "POST",
                    "/api/upload",
                    data=data,
                    files=multipart,
                    headers={"Authorization": f"Bearer {token}"},
                )
            except StoreError as err:
                if err.status == 401 and attempt == 1:  # expired or rotated secret
                    self._tokens.pop((user_id, session_id), None)
                    continue
                raise
            return UploadReceipt(tuple(response.json().get("keys", [])))
        raise StoreError("upload unauthorised", status=401, permanent=True)

    def _send(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        assert self.http is not None
        try:
            response = self.http.request(method, self.base_url + path, **kwargs)  # type: ignore[arg-type]
        except httpx.HTTPError as err:
            raise StoreError(f"cannot reach the Store: {err}") from err
        if response.is_success:
            return response
        try:
            body = response.json()
        except ValueError:
            body = {}
        message = str(body.get("error") or response.reason_phrase or "error")
        status = response.status_code
        retry_after = _retry_after(response.headers.get("Retry-After"))
        raise StoreError(
            f"Store {status}: {message}",
            status=status,
            retry_after=retry_after,
            permanent=status in (400, 403, 413, 422),
            violations=body.get("violations", []) if isinstance(body, dict) else [],
        )


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_time(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
