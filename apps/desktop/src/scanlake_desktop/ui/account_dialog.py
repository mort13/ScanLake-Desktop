"""Account: optional Discord login so the ScanLake ID follows you across PCs and reinstalls."""

from __future__ import annotations

import httpx
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from scanlake_desktop.auth.account import AccountController
from scanlake_desktop.tasks import run_in_background
from scanlake_desktop.ui.style import ERROR

ABOUT = (
    "Logging in with Discord is optional. It ties your anonymous ScanLake ID to your Discord "
    "account, so the same ID is used on every PC and comes back after a reinstall. Only your "
    "Discord name and avatar are used; no email."
)


class AccountDialog(QDialog):
    def __init__(self, account: AccountController, user_id: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Account")
        self.setMinimumWidth(420)
        self.account = account
        self.user_id = user_id
        self._avatar_url = ""

        self.avatar = QLabel()
        self.avatar.setFixedSize(64, 64)
        self.name = QLabel()
        self.name.setProperty("role", "title")
        self.detail = QLabel()
        self.detail.setProperty("role", "hint")
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail.setWordWrap(True)
        self.notice = QLabel()
        self.notice.setStyleSheet(f"color: {ERROR}")
        self.notice.setWordWrap(True)
        self.about = QLabel(ABOUT)
        self.about.setWordWrap(True)
        self.about.setProperty("role", "hint")

        self.login_button = QPushButton("Log in with Discord")
        self.cancel_button = QPushButton("Cancel")
        self.copy_button = QPushButton("Copy login link")
        self.logout_button = QPushButton("Log out")
        self.unlink_button = QPushButton("Unlink Discord…")
        self.login_button.clicked.connect(account.login)
        self.cancel_button.clicked.connect(account.cancel_login)
        self.copy_button.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(account.login_url)
        )
        self.logout_button.clicked.connect(account.logout)
        self.unlink_button.clicked.connect(self._unlink)

        head = QHBoxLayout()
        head.addWidget(self.avatar)
        names = QVBoxLayout()
        names.addWidget(self.name)
        names.addWidget(self.detail)
        head.addLayout(names, 1)
        buttons = QHBoxLayout()
        for b in (self.login_button, self.cancel_button, self.copy_button, self.logout_button,
                  self.unlink_button):  # fmt: skip
            buttons.addWidget(b)
        buttons.addStretch(1)
        box = QVBoxLayout(self)
        box.addLayout(head)
        box.addWidget(self.notice)
        box.addWidget(self.about)
        box.addLayout(buttons)

        account.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        a = self.account
        me = a.me
        busy = bool(a.busy)
        if me is not None:
            self.name.setText(me.display_name)
            self.detail.setText(
                f"@{me.username} · ScanLake ID {me.primary_user_id or self.user_id}"
                f" · {len(me.user_ids)} linked ID(s)"
            )
            self._load_avatar(me.avatar_url)
        else:
            self.name.setText(a.busy or "Not logged in")
            self.detail.setText(
                "Login service unreachable; working offline" if a.offline
                else f"ScanLake ID {self.user_id}"
            )  # fmt: skip
            self.avatar.clear()
        self.notice.setText(a.notice or "")
        self.notice.setVisible(bool(a.notice))
        self.about.setVisible(me is None)
        waiting = a.busy.startswith("Waiting")
        self.login_button.setVisible(me is None and not busy)
        self.cancel_button.setVisible(waiting)
        self.copy_button.setVisible(waiting)
        self.logout_button.setVisible(me is not None and not busy)
        self.unlink_button.setVisible(me is not None and not busy)

    def _unlink(self) -> None:
        answer = QMessageBox.question(
            self,
            "Unlink Discord",
            "Delete the Discord link? Your scans stay, but your ScanLake IDs are no longer tied "
            "to Discord, and you are logged out on every PC.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.account.unlink()

    def _load_avatar(self, url: str) -> None:
        if not url or url == self._avatar_url:
            return
        self._avatar_url = url

        def fetch() -> bytes:
            return httpx.get(url, timeout=10, follow_redirects=True).content

        def show(result: object) -> None:
            if isinstance(result, bytes):
                pixmap = QPixmap()
                if pixmap.loadFromData(result):
                    self.avatar.setPixmap(
                        pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
                    )  # fmt: skip

        run_in_background(fetch, show)
