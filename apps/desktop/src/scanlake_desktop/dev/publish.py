"""Dev mode: package the current profiles, anchors and models (with your edits) as an asset release.

Writes `<out>/<version>/` with the files, `manifest.json` and `upload-commands.txt`. Nothing is
uploaded: run the listed wrangler commands yourself once you have checked the release. Players
get it from the Gateway on their next start.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from scanlake_desktop.assets import AssetSet
from scanlake_io.manifest import Manifest, build_manifest, upload_commands


def prepare_release(
    assets: AssetSet, out_root: Path, version: str, min_app_version: str
) -> tuple[Path, Manifest, list[str]]:
    folder = out_root / version
    if folder.exists():
        shutil.rmtree(folder)
    for kind in ("profiles", "anchors"):
        shutil.copytree(assets.root / kind, folder / kind)
        if assets.overrides is not None and (assets.overrides / kind).exists():
            shutil.copytree(assets.overrides / kind, folder / kind, dirs_exist_ok=True)
    (folder / "models").mkdir(parents=True)
    for path in assets.models_dir.iterdir():  # the user's trained models when installed
        if path.suffix in (".onnx", ".data", ".json"):
            shutil.copyfile(path, folder / "models" / path.name)
    AssetSet(folder).ships()  # refuse to package a set the app cannot load
    manifest = build_manifest(folder, version, min_app_version)
    (folder / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    commands = upload_commands(manifest, folder)
    (folder / "upload-commands.txt").write_text(
        f"# Run from a folder with the Gateway's wrangler login, files first, manifest last:\n"
        f'cd "{folder}"\n' + "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    return folder, manifest, commands


class ReleaseDialog(QDialog):
    def __init__(self, assets: AssetSet, out_root: Path, app_version: str, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle("Prepare asset release")
        self.resize(760, 460)
        self.assets = assets
        self.out_root = out_root
        self.version = QLineEdit(time.strftime("%Y.%m.%d.%H%M"))
        self.min_app = QLineEdit("0.1.0" if app_version == "dev" else app_version)
        build = QPushButton("Build release")
        build.clicked.connect(self.build)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.copy_button = QPushButton("Copy commands")
        self.copy_button.setEnabled(False)
        self.open_button = QPushButton("Open folder")
        self.open_button.setEnabled(False)
        self.folder: Path | None = None
        self.copy_button.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.output.toPlainText())
        )
        self.open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder)))
        )
        hint = QLabel(
            "Packages the profiles, anchors and models this app uses now, including your dev-mode "
            "edits and installed models. Nothing is uploaded: check the release, then run the "
            "commands (files first, manifest last)."
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")
        form = QFormLayout()
        form.addRow("Release version", self.version)
        form.addRow("Needs app version ≥", self.min_app)
        buttons = QHBoxLayout()
        for b in (build, self.copy_button, self.open_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        box = QVBoxLayout(self)
        box.addWidget(hint)
        box.addLayout(form)
        box.addLayout(buttons)
        box.addWidget(self.output, 1)

    def build(self) -> None:
        try:
            folder, manifest, commands = prepare_release(
                self.assets, self.out_root, self.version.text().strip(), self.min_app.text().strip()
            )
        except Exception as err:  # show any packaging problem instead of crashing
            self.output.setPlainText(f"Release not built: {err}")
            return
        self.folder = folder
        size = sum(f.size for f in manifest.files) / 1e6
        self.output.setPlainText(
            f"# {len(manifest.files)} files, {size:.1f} MB, in {folder}\n"
            f'cd "{folder}"\n' + "\n".join(commands)
        )
        self.copy_button.setEnabled(True)
        self.open_button.setEnabled(True)
