"""The optional Discord login: state, background calls, and applying the account to settings."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from scanlake_desktop.auth.client import AuthClient, AuthError, AuthUnavailable, Me
from scanlake_desktop.auth.login import LoginCancelled, login
from scanlake_desktop.auth.reconcile import LocalProfile, Reconciled, reconcile
from scanlake_desktop.auth.token_store import TokenStore
from scanlake_desktop.tasks import run_in_background

log = logging.getLogger(__name__)


class AccountController(QObject):
    """Login state for the UI. Every network call runs off the GUI thread.

    `profile_changed` fires when the account changed the local ScanLake ID, name or org; the app
    then saves the settings (and forgets upload tokens if the ID changed).
    """

    changed = Signal()
    profile_changed = Signal(object, object)  # before: LocalProfile, after: LocalProfile
    message = Signal(str)

    def __init__(
        self,
        client: AuthClient,
        tokens: TokenStore,
        local_profile: Callable[[], LocalProfile],
        has_local_data: Callable[[], bool],
    ) -> None:
        super().__init__()
        self.client = client
        self.tokens = tokens
        self._local_profile = local_profile
        self._has_local_data = has_local_data
        self.me: Me | None = None
        self.notice: str | None = None
        self.busy = ""  # what is in progress, for the dialog
        self.login_url = ""
        self.offline = False
        self._cancel = threading.Event()

    @property
    def logged_in(self) -> bool:
        return self.me is not None

    # -- actions --------------------------------------------------------------

    def refresh(self) -> None:
        """Check the stored token and reconcile (on start-up and after logging in)."""
        token = self.tokens.get()
        if not token:
            self._set(me=None)
            return
        # SQLite and settings are read here, on the GUI thread, not inside the background call
        local, has_data = self._local_profile(), self._has_local_data()
        self._run(
            "Checking your login…",
            lambda: self._reconcile(token, local, has_data),
            self._reconciled,
        )

    def login(self) -> None:
        if self.busy:
            return
        self._cancel.clear()

        def open_browser(url: str) -> None:
            self.login_url = url
            QDesktopServices.openUrl(QUrl(url))

        def run() -> str:
            return login(self.client, open_browser, self._cancel)

        self._run("Waiting for the browser…", run, self._logged_in)

    def cancel_login(self) -> None:
        self._cancel.set()

    def logout(self) -> None:
        token = self.tokens.get()
        self.tokens.clear()
        self._set(me=None, notice=None)
        if token:
            run_in_background(lambda: self.client.logout(token), self._ignore)

    def unlink(self) -> None:
        token = self.tokens.get()
        if not token:
            return

        def done(result: Any) -> None:
            if isinstance(result, Exception):
                self._failed(result)
                return
            self.tokens.clear()
            self._set(me=None, notice=None)
            self.message.emit("Discord unlinked: the account and its logins were deleted")

        self._run("Unlinking…", lambda: self.client.delete_account(token), done)

    def push_profile(self, user_name: str, org: str) -> None:
        """Keep the account's name/org in step when they are edited here."""
        token = self.tokens.get()
        if token and self.me is not None:
            run_in_background(lambda: self.client.put_profile(token, user_name, org), self._pushed)

    # -- internals ------------------------------------------------------------

    def _reconcile(self, token: str, local: LocalProfile, has_data: bool) -> Reconciled | None:
        me = self.client.me(token)
        if me is None:
            return None
        return reconcile(self.client, token, me, local, has_data)

    def _reconciled(self, result: Any) -> None:
        if isinstance(result, AuthUnavailable):
            self.offline = True
            self._set(busy="")
            return
        if isinstance(result, Exception):
            self._failed(result)
            return
        self.offline = False
        if result is None:  # token expired or revoked
            self.tokens.clear()
            self._set(me=None, busy="")
            self.message.emit("Your Discord login expired; log in again to link this PC")
            return
        before = self._local_profile()
        if result.changed(before):
            self.profile_changed.emit(before, result.profile)
        self._set(me=result.me, notice=result.notice, busy="")

    def _logged_in(self, result: Any) -> None:
        self.login_url = ""
        if isinstance(result, LoginCancelled):
            self._set(busy="")
            self.message.emit(str(result).capitalize())
            return
        if isinstance(result, Exception):
            self._failed(result)
            return
        self.tokens.set(result)
        self._set(busy="")
        self.refresh()

    def _pushed(self, result: Any) -> None:
        if isinstance(result, Me):
            self._set(me=result)
        elif isinstance(result, Exception):
            log.warning("profile push failed: %s", result)

    def _failed(self, err: Exception) -> None:
        log.warning("account action failed: %s", err)
        self._set(busy="")
        self.message.emit(str(err) if isinstance(err, AuthError) else f"Login failed: {err}")

    def _ignore(self, result: Any) -> None:
        if isinstance(result, Exception):
            log.info("logout request failed (the token is deleted locally anyway): %s", result)

    def _run(self, busy: str, fn: Callable[[], Any], done: Callable[[Any], None]) -> None:
        self._set(busy=busy)
        run_in_background(fn, done)

    _UNSET: Any = object()

    def _set(self, me: Any = _UNSET, notice: Any = _UNSET, busy: Any = _UNSET) -> None:
        if me is not self._UNSET:
            self.me = me
        if notice is not self._UNSET:
            self.notice = notice
        if busy is not self._UNSET:
            self.busy = busy
        self.changed.emit()
