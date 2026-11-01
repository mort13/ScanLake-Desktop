"""Wires settings, assets, the capture worker, the upload worker, hotkeys and the window."""

from __future__ import annotations

import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from scanlake_core.domain.capture import Identity
from scanlake_core.format import Coverage, ParquetFormat, coverage, parse_format
from scanlake_core.geometry import Rect
from scanlake_core.recognition import RoiReading
from scanlake_core.schema_map import SlotLayout, slot_layout
from scanlake_desktop import paths, updates
from scanlake_desktop.assets import AssetSet, Ship
from scanlake_desktop.auth.account import AccountController
from scanlake_desktop.auth.client import AuthClient
from scanlake_desktop.auth.reconcile import LocalProfile
from scanlake_desktop.auth.token_store import TokenStore
from scanlake_desktop.capture.source import Monitor, list_monitors
from scanlake_desktop.db.database import Database
from scanlake_desktop.hotkeys import HotkeyService
from scanlake_desktop.services.session_service import SessionService
from scanlake_desktop.services.staging import draft_values, needs_attention, staged_by_roi
from scanlake_desktop.settings import Endpoints, Settings, save_settings
from scanlake_desktop.tasks import run_in_background
from scanlake_desktop.ui.game_overlay import GameOverlay, Staged
from scanlake_desktop.ui.main_window import MainWindow
from scanlake_desktop.ui.settings_dialog import SettingsDialog
from scanlake_desktop.ui.style import ERROR
from scanlake_desktop.upload.upload_worker import UploadWorker
from scanlake_desktop.upload.uploader import FormatHolder
from scanlake_desktop.worker.pipeline_worker import LiveState, PipelineWorker, WorkerConfig
from scanlake_inference.registry import available_backends
from scanlake_io.format_source import load_format
from scanlake_io.store_client import StoreClient

log = logging.getLogger(__name__)


def app_version() -> str:
    try:
        return version("scanlake-desktop")
    except PackageNotFoundError:
        return "dev"


def beep(kind: str) -> None:
    if sys.platform != "win32":
        return
    import winsound

    sound = {"ok": "SystemAsterisk", "stage": "SystemDefault", "error": "SystemHand"}[kind]
    winsound.PlaySound(sound, winsound.SND_ALIAS | winsound.SND_ASYNC)


class DesktopApp(QObject):
    def __init__(
        self,
        settings: Settings,
        *,
        dev: bool = False,
        replay: list[Path] | None = None,
        first_run: bool = False,
    ):
        super().__init__()
        self.settings = settings
        self.dev = dev or settings.dev_mode
        self.replay = tuple(replay or ())
        self.version = app_version()
        self.endpoints = Endpoints.from_env()
        # Downloaded asset updates replace the bundled set; dev-mode edits override both
        root = updates.active_assets(paths.updates_dir(), paths.bundled_assets())
        self.assets = AssetSet(root, paths.user_assets_dir())
        self.bundled_format = paths.bundled_assets() / "format" / "format.json"
        self.ships: dict[str, Ship] = self.assets.ships()
        if settings.ship not in self.ships:
            settings.ship = next(iter(self.ships))
        self.monitors = list_monitors() or [Monitor(1, 0, 0, 2560, 1440)]
        self.db = Database(paths.database_file())
        self.sessions = SessionService(self.db)
        self.last_state: LiveState | None = None
        self.staged = False
        self.editing = None
        self.staged_readings: dict[str, RoiReading] = {}  # crops of what was staged
        self.staged_region = Rect(0, 0, 0, 0)  # the capture region when it was staged
        self.debug_window = None
        self.profile_editor = None

        fmt = self._offline_format()
        self.formats = FormatHolder(self._load_format_live, fmt)
        self.layout: SlotLayout = slot_layout(self.ship.hud)
        self.cov: Coverage = coverage(fmt, self.layout)

        ships = [(s.key, s.label) for s in self.ships.values()]
        self.window = MainWindow(ships, self.version)
        self._connect_window()
        self.overlay = GameOverlay()
        self.overlay.set_monitor(self.monitor)
        self.overlay.set_enabled(settings.game_overlay)

        self.stopped = False
        self.worker = PipelineWorker()
        self.worker.live.connect(self.on_live)
        self.worker.problem.connect(self._worker_problem)
        self.worker.backend_resolved.connect(self._remember_backend)
        self.worker.start()

        self.store = StoreClient(self.endpoints.store, user_agent=f"ScanLakeDesktop/{self.version}")
        self.uploader = UploadWorker(
            paths.database_file(),
            self.store,
            self.formats,
            self.identity,
            lambda: Path(self.settings.export_dir) if self.settings.export_dir else None,
        )
        self.uploader.changed.connect(self._show_upload_summary)
        self.uploader.finished_batch.connect(self._batch_finished)
        self.uploader.start()

        self.hotkeys = HotkeyService()
        self.hotkeys.stage.connect(self.stage)
        self.hotkeys.save.connect(self.save)
        self.hotkeys.new_cluster.connect(self.new_cluster)
        self._bind_hotkeys()

        self.account = AccountController(
            AuthClient(self.endpoints.auth, user_agent=f"ScanLakeDesktop/{self.version}"),
            TokenStore(self.endpoints.auth),
            lambda: LocalProfile(self.settings.user_id, self.settings.user_name, self.settings.org),
            lambda: any(s.scans for s in self.db.sessions()),
        )
        self.account.changed.connect(self._show_account)
        self.account.profile_changed.connect(self._apply_account_profile)
        self.account.message.connect(lambda text: self.window.flash(text, 10000))
        self.account_dialog = None

        self.tray = self._make_tray()
        self._apply_ship()
        self._refresh_session()
        self.window.set_dev_mode(self.dev)
        self._build_dev_menu()
        self.window.show()
        if first_run:
            self._welcome()
        self._refresh_format_in_background()
        self.account.refresh()
        self._check_updates()

    # -- setup ----------------------------------------------------------------

    @property
    def ship(self) -> Ship:
        return self.ships[self.settings.ship]

    @property
    def monitor(self) -> Monitor:
        return next(
            (m for m in self.monitors if m.index == self.settings.monitor), self.monitors[0]
        )

    def identity(self) -> Identity:
        s = self.settings
        return Identity(s.user_id, s.user_name, s.org, f"scanlake-desktop/{self.version}")

    def _offline_format(self) -> ParquetFormat:
        import json

        for path in (paths.format_cache_file(), self.bundled_format):
            try:
                return parse_format(json.loads(path.read_text(encoding="utf-8")), strict=False)
            except (OSError, ValueError):
                continue
        raise RuntimeError("no upload format available (bundled copy missing)")

    def _load_format_live(self) -> ParquetFormat:
        loaded = load_format(self.endpoints.format, paths.format_cache_file(), self.bundled_format)
        return loaded.format

    def _refresh_format_in_background(self) -> None:
        run_in_background(self._load_format_live, self._format_loaded)

    def _format_loaded(self, result: object) -> None:
        if isinstance(result, ParquetFormat):
            self.formats.set(result)
            self._update_coverage()
        else:
            log.warning("format refresh failed: %s", result)

    def _update_coverage(self) -> None:
        fmt = self.formats.get()
        self.cov = coverage(fmt, self.layout)
        text = f"Format v{fmt.version} ·"
        if not self.cov.complete:
            text += f" · cannot fill {self.cov.describe_missing()}"
            self.window.format_status.setStyleSheet(f"color: {ERROR}")
        else:
            self.window.format_status.setStyleSheet("")
        self.window.format_status.setText(text)

    def _welcome(self) -> None:
        from scanlake_desktop.ui.first_run import FirstRunDialog

        ships = [(s.key, s.label) for s in self.ships.values()]
        dialog = FirstRunDialog(self.settings, ships, self.monitors, self.window)
        dialog.exec()
        dialog.apply(self.settings)
        save_settings(self.settings)
        self.window.session_bar.set_ship(self.settings.ship)
        self._apply_ship()
        self._refresh_session()

    def _check_updates(self) -> None:
        """New HUD profiles/models from the Gateway (used from the next start), and app releases."""
        updater = updates.AssetUpdater(
            self.endpoints.gateway, paths.updates_dir(), paths.bundled_assets(), self.version
        )

        def assets_checked(result: object) -> None:
            if isinstance(result, updates.UpdateResult):
                if result.status == "updated":
                    self.window.flash(
                        f"New ship profiles/models ({result.version}) downloaded; "
                        "restart ScanLake Desktop to use them", 20000
                    )  # fmt: skip
                elif result.status == "failed":
                    log.info("asset update check: %s", result.message)

        def release_checked(result: object) -> None:
            if isinstance(result, tuple):
                version, url = result
                self.window.format_status.setText(
                    f'<a href="{url}">Update {version} available</a> · '
                    + self.window.format_status.text()
                )
                self.window.format_status.setOpenExternalLinks(True)

        run_in_background(updater.check, assets_checked)
        run_in_background(lambda: updates.latest_release(self.version), release_checked)

    def _connect_window(self) -> None:
        w = self.window
        w.stage_clicked.connect(self.stage)
        w.save_clicked.connect(self.save)
        w.clear_clicked.connect(self.clear)
        w.upload_clicked.connect(self.upload_now)
        w.settings_clicked.connect(self.open_settings)
        w.closing.connect(self.shutdown)
        bar = w.session_bar
        bar.ship_changed.connect(self.change_ship)
        bar.location_changed.connect(self.change_location)
        bar.start_clicked.connect(self.start_session)
        bar.close_clicked.connect(self.close_session)
        bar.cluster_step.connect(self.step_cluster)
        bar.new_cluster_clicked.connect(self.new_cluster)
        bar.account_clicked.connect(self.open_account)
        w.form.changed.connect(self._revalidate)
        w.scans.edit_requested.connect(self.edit_capture)
        w.scans.delete_requested.connect(self.delete_capture)
        w.session_bar.set_ship(self.settings.ship)
        w.session_bar.set_location(
            self.settings.system, self.settings.gravity_well, self.settings.region
        )

    def _make_tray(self) -> QSystemTrayIcon | None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return None
        tray = QSystemTrayIcon(self.window.windowIcon(), self.window)
        tray.setToolTip("ScanLake Desktop")
        tray.activated.connect(lambda _: (self.window.showNormal(), self.window.activateWindow()))
        tray.show()
        return tray

    def _build_dev_menu(self) -> None:
        menu = self.window.dev_menu
        menu.clear()
        menu.addAction("Pipeline debug window", self.open_debug_window)
        menu.addAction("Profile editor", self.open_profile_editor)
        menu.addAction("Labeler", self.open_labeler)
        menu.addAction("Training", self.open_training)
        menu.addSeparator()
        menu.addAction("Prepare asset release…", self.prepare_release)

    def _bind_hotkeys(self) -> None:
        h = self.settings.hotkeys
        self.hotkeys.bind(h.stage, h.save, h.new_cluster)
        self.window.stage_button.setText(f"Stage ({h.stage.upper()})")
        self.window.save_button.setText(f"Save ({h.save.upper()})")

    def _worker_config(self) -> WorkerConfig:
        s = self.settings
        monitor = self.monitor
        manual: Rect | None = None
        if s.region_mode == "manual" and s.manual_region is not None:
            r = s.manual_region
            manual = Rect(monitor.left + r.x, monitor.top + r.y, r.w, r.h)
        return WorkerConfig(
            hud=self.ship.hud,
            ship=self.ship.key,
            templates=self.ship.templates,
            models_dir=self.assets.models_dir,
            backend=s.backend,
            auto_backend=s.auto_backend,
            monitor=monitor,
            manual_region=manual,
            fps=s.fps,
            vote_window_s=s.vote_window_s,
            dml_device_id=s.dml_device_id,
            debug=self.debug_window is not None or self.profile_editor is not None,
            collect_crops=s.dev_mode and s.collect_crops,
            model_stamp=self.assets.model_stamp(),
            replay=self.replay,
        )

    def _apply_ship(self) -> None:
        self.layout = slot_layout(self.ship.hud)
        self._update_coverage()
        self.window.live.set_layout(self.layout)
        self.window.form.set_layout(self.layout)
        self.worker.configure(self._worker_config())

    def _remember_backend(self, backend: str) -> None:
        self.settings.auto_backend = backend
        save_settings(self.settings)

    # -- live -----------------------------------------------------------------

    def _worker_problem(self, text: str) -> None:
        if not self.stopped:
            self.window.flash(text, 15000)

    def on_live(self, state: LiveState) -> None:
        if self.stopped:
            return
        self.last_state = state
        if state.ship == self.settings.ship:
            self.window.live.update_state(state)
            self.overlay.update_state(state)
        if self.debug_window is not None:
            self.debug_window.update_state(state)

    # -- actions --------------------------------------------------------------

    def stage(self) -> None:
        """Stages the current values; again while staged: replaces them (Clear unstages)."""
        restage = self.staged and self.editing is None
        state = self.last_state
        if state is None or not state.locked:
            self._fail("The scan HUD is not visible")  # what is staged stays
            return
        scan, rows = draft_values(self.layout, state.votes)
        votes_by_cell: dict[tuple[int, str], object] = {}
        for slot in self.layout.scan:
            votes_by_cell[(-1, slot.column)] = state.votes.get(slot.roi)
        for i, row in enumerate(self.layout.rows):
            for slot in row:
                votes_by_cell[(i, slot.column)] = state.votes.get(slot.roi)
        attention = {
            cell
            for cell, vote in votes_by_cell.items()
            if needs_attention(
                scan.get(cell[1]) if cell[0] < 0 else rows[cell[0]].get(cell[1]),  # type: ignore[arg-type]
                vote,  # type: ignore[arg-type]
            )
        }
        unstable = sum(1 for v in state.votes.values() if not v.stable)
        settling = f" · {unstable} value{'s' if unstable > 1 else ''} still settling"
        title = ("Restaged scan" if restage else "Staged scan") + (settling if unstable else "")
        self.editing = None
        self.staged_readings = dict(state.readings)
        self.staged_region = state.region
        self.window.form.load(scan, rows, title, attention)
        self.staged = True
        self._revalidate()
        self.overlay.flash(title, "warn" if unstable else "info")
        self._beep("stage")

    def save(self) -> None:
        if not self.staged:
            self._fail("Nothing staged: press the stage key first")
            return
        if not self.cov.complete:
            self._fail(f"This ship profile cannot fill {self.cov.describe_missing()}")
            return
        if self.sessions.session is None:
            self.start_session()
        scan, rows = self.window.form.values()
        result = self.sessions.save(
            scan, rows, self.layout, self.cov, ship=self.ship.key,
            models=self.assets.model_versions(), editing=self.editing,
        )  # fmt: skip
        if result.duplicate:
            self._fail("This scan is already saved in this session")
            return
        if not result.saved:
            self.window.form.show_issues(result.issues)
            self._show_staged()  # the overlay marks the same values
            self._fail(result.issues[0].message if result.issues else "Cannot save")
            return
        self._beep("ok")
        self.window.flash("Saved" + (" (edited)" if self.editing else ""))
        self.overlay.flash(f"Saved · {len(self.sessions.captures())} scans this session", "ok")
        self._collect_crops(result.capture_id)
        deposit = scan.get("deposit")
        self.clear(keep_deposit=deposit.text if deposit else "")
        self._refresh_session()

    def _collect_crops(self, capture_id: str) -> None:
        """Dev mode: keep the staged ROI crops, labelled with the saved values, for training."""
        if not (self.settings.dev_mode and self.settings.collect_crops and self.staged_readings):
            return
        capture = self.db.capture(capture_id)
        if capture is None:
            return
        from scanlake_desktop.services.crops import collect

        try:
            n = collect(capture, self.layout, self.staged_readings, self.assets.word_classes(),
                        paths.training_dir())  # fmt: skip
            log.info("kept %d training crops of %s", n, capture_id)
        except OSError as err:
            log.warning("could not save training crops: %s", err)

    def clear(self, keep_deposit: str = "") -> None:
        self.staged = False
        self.editing = None
        self.staged_readings = {}
        self.window.form.clear({"deposit": keep_deposit} if keep_deposit else None)
        self.overlay.set_staged(None)

    def _revalidate(self) -> None:
        if not self.staged:
            return
        scan, rows = self.window.form.values()
        issues = (
            self.sessions.check(scan, rows, self.layout, self.cov, self.editing)
            if self.sessions.session is not None
            else []
        )
        self.window.form.show_issues(issues)
        self._show_staged()

    def _show_staged(self) -> None:
        """The overlay shows the staged values (with any edits) where they were read."""
        if not self.staged or self.editing is not None or not self.staged_readings:
            self.overlay.set_staged(None)  # an old scan being edited was not read here
            return
        keys = self.settings.hotkeys
        self.overlay.set_staged(
            Staged(
                values=staged_by_roi(self.layout, self.window.form.cell_states()),
                readings=self.staged_readings,
                region=self.staged_region,
                status=f"Staged · {keys.save.upper()} saves · {keys.stage.upper()} restages",
            )
        )

    def new_cluster(self) -> None:
        if self.sessions.session is None:
            self.start_session()
        self.sessions.new_cluster()
        self.worker.reset_votes()
        self._refresh_session()
        self._beep("stage")
        self.window.flash(f"New rock #{self.sessions.cluster_number}")
        self.overlay.flash(f"New rock #{self.sessions.cluster_number}")

    def step_cluster(self, delta: int) -> None:
        self.sessions.step_cluster(delta)
        self._refresh_session()

    def start_session(self) -> None:
        system, gravity_well, region = self.window.session_bar.location()
        self.sessions.start(system, gravity_well, region, self.ship.key)
        self._refresh_session()

    def close_session(self) -> None:
        number = self.sessions.close()
        self.clear()
        self._refresh_session()
        if number:
            self.uploader.wake()
        self.window.flash("Session closed" + (", uploading" if number else ""))

    def change_ship(self, key: str) -> None:
        if key in self.ships and key != self.settings.ship:
            self.settings.ship = key
            save_settings(self.settings)
            self.clear()
            self._apply_ship()

    def change_location(self, system: str, gravity_well: str, region: str) -> None:
        s = self.settings
        s.system, s.gravity_well, s.region = system, gravity_well, region
        save_settings(s)
        self.sessions.set_location(system, gravity_well, region)

    def edit_capture(self, capture_id: str) -> None:
        capture = self.db.capture(capture_id)
        if capture is None:
            return
        self.editing = capture
        self.staged = True
        self.staged_readings = {}
        self.window.form.load(
            capture.scan, capture.rows, f"Editing scan from {capture.timestamp[11:19]}"
        )
        self._revalidate()

    def delete_capture(self, capture_id: str) -> None:
        answer = QMessageBox.question(self.window, "Delete scan", "Delete this scan?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.sessions.delete(capture_id):
            self.window.flash("Deleted here; the copy already uploaded stays on the server", 8000)
        self._refresh_session()

    def upload_now(self) -> None:
        self.sessions.queue_upload()
        self.uploader.wake(force=True)
        self.window.flash("Uploading…")

    def open_settings(self) -> None:
        dialog = SettingsDialog(
            self.settings, self.monitors, available_backends(self.assets.models_dir),
            self.last_state.backend if self.last_state else "", self.window,
        )  # fmt: skip
        dialog.benchmark_requested.connect(self._rebenchmark)
        if not dialog.exec():
            return
        new = dialog.result_settings()
        if (new.user_name, new.org) != (self.settings.user_name, self.settings.org):
            self.account.push_profile(new.user_name, new.org)
        torch_switch = new.backend.startswith("torch") != self.settings.backend.startswith("torch")
        self.settings = new
        save_settings(new)
        self.dev = new.dev_mode
        self.window.set_dev_mode(self.dev)
        self._bind_hotkeys()
        self.overlay.set_monitor(self.monitor)
        self.overlay.set_enabled(new.game_overlay)
        self.worker.configure(self._worker_config())
        if torch_switch:
            QMessageBox.information(
                self.window, "Restart needed", "Restart ScanLake Desktop to switch to PyTorch."
            )

    def open_account(self) -> None:
        from scanlake_desktop.ui.account_dialog import AccountDialog

        if self.account_dialog is None:
            self.account_dialog = AccountDialog(self.account, self.settings.user_id, self.window)
        self.account_dialog.user_id = self.settings.user_id
        self.account_dialog.refresh()
        self.account_dialog.show()
        self.account_dialog.raise_()

    def _show_account(self) -> None:
        me = self.account.me
        if me is not None:
            text = me.display_name
        elif self.account.busy:
            text = self.account.busy
        else:
            text = "Log in with Discord"
        self.window.session_bar.account.setText(text)

    def _apply_account_profile(self, before: LocalProfile, after: LocalProfile) -> None:
        """The Discord account supplied a different ScanLake ID, name or org: use them here."""
        s = self.settings
        s.user_id, s.user_name, s.org = after.user_id, after.user_name, after.org
        save_settings(s)
        if after.user_id != before.user_id:
            self.store.forget_tokens()  # upload tokens belong to the previous ID
            self.window.flash("Using the ScanLake ID of your Discord account", 10000)
        if self.account_dialog is not None:
            self.account_dialog.user_id = after.user_id
        self._refresh_session()

    def _rebenchmark(self) -> None:
        self.settings.auto_backend = None
        self.settings.backend = "auto"
        save_settings(self.settings)
        self.worker.configure(self._worker_config())
        self.window.flash("Benchmarking the engines…")

    def open_profile_editor(self) -> None:
        from scanlake_desktop.dev.profile_editor.editor_window import ProfileEditor
        from scanlake_inference.registry import create_engine

        if self.profile_editor is None:
            ships = [(e.key, e.label, e.file) for e in self.assets.manifest()]
            self.profile_editor = ProfileEditor(
                self.assets,
                paths.user_assets_dir(),
                ships,
                self.formats.get,
                lambda: self.last_state.frame if self.last_state is not None else None,
                lambda: create_engine("ort-cpu", self.assets.models_dir),
            )
            self.profile_editor.applied.connect(self._profile_applied)
            self.profile_editor.destroyed.connect(self._profile_editor_closed)
            self.profile_editor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self.worker.configure(self._worker_config())  # start sending frames
        self.profile_editor.show()
        self.profile_editor.raise_()

    def _profile_editor_closed(self) -> None:
        self.profile_editor = None
        self.worker.configure(self._worker_config())

    def _profile_applied(self, key: str) -> None:
        """A profile was edited in dev mode: reload the ships and capture with the new one."""
        self.ships = self.assets.ships()
        if key == self.settings.ship:
            self._apply_ship()
        self.window.flash(f"Using the edited {self.ships[key].label} profile", 8000)

    def open_labeler(self) -> None:
        from scanlake_desktop.dev.labeler import LabelerWindow

        window = LabelerWindow(paths.training_dir(), self.assets.word_classes())
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.show()
        self._labeler = window

    def open_training(self) -> None:
        from scanlake_desktop.dev.training_window import TrainingWindow
        from scanlake_inference.registry import torch_installed

        window = TrainingWindow(
            paths.training_dir(), self.assets.models_dir, paths.user_assets_dir() / "models",
            torch_installed(),
        )  # fmt: skip
        window.model_installed.connect(self._model_installed)
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.show()
        self._training = window

    def prepare_release(self) -> None:
        from scanlake_desktop.dev.publish import ReleaseDialog

        dialog = ReleaseDialog(self.assets, paths.data_dir() / "publish", self.version, self.window)
        dialog.exec()

    def _model_installed(self, name: str) -> None:
        self.worker.configure(self._worker_config())  # new model files -> new engine
        self.window.flash(f"Capture now uses your trained {name}", 10000)

    def open_debug_window(self) -> None:
        from scanlake_desktop.dev.debug_window import DebugWindow

        if self.debug_window is None:
            self.debug_window = DebugWindow(paths.snapshots_dir(), self.monitor)
            self.debug_window.snapshot_saved.connect(
                lambda folder: self.window.flash(f"Snapshot saved to {folder}", 10000)
            )
            self.debug_window.destroyed.connect(self._debug_closed)
            self.debug_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self.worker.configure(self._worker_config())
        self.debug_window.show()
        self.debug_window.raise_()

    def _debug_closed(self) -> None:
        self.debug_window = None
        self.worker.configure(self._worker_config())

    # -- display --------------------------------------------------------------

    def _refresh_session(self) -> None:
        sessions = self.sessions
        sessions.refresh()
        active = sessions.session is not None
        self.window.session_bar.set_session(active, sessions.cluster_number, len(sessions.clusters))
        captures = sessions.captures()
        state = self.db.upload_state(sessions.session.session_id) if sessions.session else {}
        numbers = {cid: i + 1 for i, cid in enumerate(sessions.clusters)}
        self.window.scans.show_captures(
            captures, self.layout.scan_columns, self.layout.row_columns, state, numbers
        )
        s = self.settings
        who = s.user_name or "anonymous"
        self.window.user_status.setText(f"  {who} · {len(captures)} scans this session")

    def _show_upload_summary(self, summary: object) -> None:
        from scanlake_desktop.db.database import UploadSummary

        assert isinstance(summary, UploadSummary)
        if self.stopped:
            return
        parts = []
        if summary.failed_batches:
            parts.append(f"⚠ {summary.failed_batches} batch(es) rejected: {summary.last_error}")
        elif summary.pending_batches and summary.last_error:
            parts.append(f"Upload retrying: {summary.last_error}")
        elif summary.pending_batches:
            parts.append("Uploading…")
        elif summary.unbatched_captures:
            parts.append(f"{summary.unbatched_captures} scans upload when the session closes")
        else:
            parts.append("All scans uploaded")
        self.window.upload_status.setText(" · ".join(parts))

    def _batch_finished(self, outcome: object) -> None:
        if not self.stopped:  # (the database is closed after stop)
            self._refresh_session()

    def _fail(self, message: str) -> None:
        self.window.flash(message, 6000)
        self.overlay.flash(message, "error")
        self._beep("error")

    def _beep(self, kind: str) -> None:
        if self.settings.beep:
            beep(kind)

    def stop(self) -> None:
        # Updates the threads queued before they stopped must not reach a closed window or DB
        self.stopped = True
        self.hotkeys.stop()
        self.worker.stop()
        self.uploader.stop()
        self.overlay.close()
        save_settings(self.settings)
        self.db.close()

    def shutdown(self) -> None:
        self.stop()
        QApplication.quit()
