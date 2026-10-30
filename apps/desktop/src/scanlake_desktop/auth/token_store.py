"""Keeps the login token in the Windows Credential Manager (via keyring), never in a file or log.

One entry per Auth host, so a local dev login does not replace the production one. When no
keyring backend works (rare: headless Linux) the token lives in memory for this run only.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

log = logging.getLogger(__name__)

SERVICE = "ScanLake Desktop"


class TokenStore:
    def __init__(self, auth_url: str) -> None:
        self._account = f"auth:{urlparse(auth_url).netloc or auth_url}"
        self._memory: str | None = None

    def get(self) -> str | None:
        try:
            import keyring

            return keyring.get_password(SERVICE, self._account) or self._memory
        except Exception as err:
            log.warning("credential store unavailable: %s", type(err).__name__)
            return self._memory

    def set(self, token: str) -> None:
        self._memory = token
        try:
            import keyring

            keyring.set_password(SERVICE, self._account, token)
        except Exception as err:
            log.warning("token kept in memory only (credential store: %s)", type(err).__name__)

    def clear(self) -> None:
        self._memory = None
        try:
            import keyring

            keyring.delete_password(SERVICE, self._account)
        except Exception:
            pass
