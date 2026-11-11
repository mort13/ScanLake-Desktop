"""Wires settings, assets, the capture worker, the upload worker, hotkeys and the window."""

from __future__ import annotations

import logging
import random
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from scanlake_core.domain.capture import Capture, Identity
from scanlake_core.domain.navigation import (
    CalibrationResult,
    CameraFrames,
    ClusterSummary,
    Convention,
    NavTracker,
    summarize_clusters,
)
from scanlake_core.domain.position import Position
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
from scanlake_desktop.crew.controller import CrewController
from scanlake_desktop.db.database import Database
from scanlake_desktop.hotkeys import HotkeyService
from scanlake_desktop.services.session_service import SessionService
from scanlake_desktop.services.staging import draft_values, needs_attention, staged_by_roi
from scanlake_desktop.settings import Endpoints, Settings, save_settings
from scanlake_desktop.tasks import run_in_background
from scanlake_desktop.ui.game_overlay import GameOverlay, Staged
from scanlake_desktop.ui.main_window import MainWindow
from scanlake_desktop.ui.nav_calibration import CalibrationDialog
from scanlake_desktop.ui.nav_overlay import NavOverlay
from scanlake_desktop.ui.settings_dialog import SettingsDialog
from scanlake_desktop.ui.style import ERROR
from scanlake_desktop.upload.upload_worker import UploadWorker
from scanlake_desktop.upload.uploader import FormatHolder
from scanlake_desktop.worker.pipeline_worker import (
    POSITION_EVERY_S,
    POSITION_FRESH_S,
    POSITION_NAV_S,
    LiveState,
    PipelineWorker,
    PositionRead,
    WorkerConfig,
)
from scanlake_inference.registry import available_backends
from scanlake_io import crew_client, frontdesk_client
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
        self.position_read: PositionRead | None = None  # the newest one sent on its own
        self.staged = False
        self.editing = None
        self.staged_readings: dict[str, RoiReading] = {}  # crops of what was staged
        self.staged_region = Rect(0, 0, 0, 0)  # the capture region when it was staged
        self.staged_position: Position | None = None  # the debug text when it was staged
        self.positions_seen = False  # the debug text was read at least once this run
        self.debug_window = None
        self.profile_editor = None
        # The Overview's clusters: the active session's own ones and, in a shared session,
        # the other members'
        self.clusters: list[ClusterSummary] = []
        self.nav: NavTracker | None = None  # the cluster being navigated to
        self.nav_position_t = 0.0  # the live position the Overview last saw
        self.camera_frames = CameraFrames()  # which frame CamDir is in, per zone (this run)
        self.calibration: CalibrationDialog | None = None  # open compass calibration

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
        self.nav_overlay = NavOverlay()
        self.nav_overlay.set_monitor(self.monitor)
        self.nav_overlay.set_anchor(settings.nav_compass_x, settings.nav_compass_y)
        self.nav_overlay.set_enabled(settings.nav_compass)
        self.nav_overlay.set_debug(settings.nav_debug)
        self.window.overview.set_debug(settings.nav_debug)
        self._show_compass_status()

        self.stopped = False
        self.worker = PipelineWorker()
        self.worker.live.connect(self.on_live)
        self.worker.position.connect(self.on_position)
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
            lambda: self.settings.upload_positions,
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

        self.crew = CrewController(
            self.db,
            self.endpoints.crew,
            self._crew_identity,
            self._crew_profile,
            f"ScanLakeDesktop/{self.version}",
        )
        self.crew.changed.connect(self._show_clusters)
        self.crew.notice.connect(self._crew_notice)
        self.crew.ended.connect(lambda why: self.window.flash(why, 15000))
        self.window.crew.set_profile(*self._crew_profile())

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
        if self.sessions.session is not None:  # resumed from the last run
            self._ping_frontdesk(self.sessions.session.session_id, "open")
            share = self.db.share(self.sessions.session.session_id)
            if share is not None and share.active and self.crew.enabled:
                self.crew.open(share)

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
        bar.invite_clicked.connect(self.invite_session)
        bar.join_clicked.connect(self.join_session)
        bar.leave_clicked.connect(self.leave_shared)
        w.crew.invite_clicked.connect(self.invite_session)
        w.crew.join_clicked.connect(self.join_session)
        w.crew.leave_clicked.connect(self.leave_shared)
        w.crew.profile_changed.connect(self._crew_profile_changed)
        w.overview.status_changed.connect(self._set_cluster_status)
        w.form.changed.connect(self._revalidate)
        w.scans.edit_requested.connect(self.edit_capture)
        w.scans.delete_requested.connect(self.delete_capture)
        w.overview.target_selected.connect(self.navigate_to)
        w.overview.target_cleared.connect(self.stop_navigation)
        w.overview.calibrate_clicked.connect(self.open_calibration)
        w.overview.debug_toggled.connect(self._nav_debug)
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
            position_every_s=POSITION_NAV_S if self._fast_positions() else POSITION_EVERY_S,
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
        self.positions_seen = self.positions_seen or state.position is not None
        if state.ship == self.settings.ship:
            self.window.live.update_state(state)
            self.overlay.update_state(state)
        if self.debug_window is not None:
            self.debug_window.update_state(state)
        self._update_navigation(state.position, state.position_t)

    def on_position(self, read: PositionRead) -> None:
        """A position the moment the debug text gave it (live capture), between LiveStates."""
        if self.stopped:
            return
        self.positions_seen = True
        self.position_read = read
        self._update_navigation(read.position, read.position_t)

    def _latest_position(self) -> tuple[Position | None, float]:
        """The newest position read, from the last LiveState or sent on its own."""
        state, read = self.last_state, self.position_read
        latest = (state.position, state.position_t) if state is not None else (None, 0.0)
        if read is not None and read.position_t > latest[1]:
            latest = (read.position, read.position_t)
        return latest

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
        position, position_t = self._latest_position()
        fresh = position is not None and state.t - position_t <= POSITION_FRESH_S
        self.staged_position = position if fresh else None
        self.window.form.load(scan, rows, title, attention)
        self.window.form.set_position(
            self.staged_position,
            "No position: the game's debug text was not readable" if self.positions_seen else "",
        )
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
            position=self.staged_position,
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
        self.staged_position = None
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
        session = self.sessions.start(system, gravity_well, region, self.ship.key)
        self._ping_frontdesk(session.session_id, "open")
        self._refresh_session()

    def close_session(self) -> None:
        session = self.sessions.session
        shared = session is not None and self.db.share(session.session_id) is not None
        self.crew.close()  # closing your session leaves a shared one
        if session is not None and shared:
            self.db.end_share(session.session_id)
        number = self.sessions.close()
        if session is not None and session.scans == 0 and not shared:
            # Its final batch is empty and never reaches the Store, which would report the close.
            # Never for a shared session: the others are still in it.
            self._ping_frontdesk(session.session_id, "discard")
        self.clear()
        self._refresh_session()
        if number:
            self.uploader.wake()
        self.window.flash("Session closed" + (", uploading" if number else ""))

    def _ping_frontdesk(self, session_id: str, kind: frontdesk_client.Kind) -> None:
        """Tells the landing page that a session is open or discarded (best effort, off-thread)."""
        url, agent = self.endpoints.frontdesk, f"ScanLakeDesktop/{self.version}"
        run_in_background(
            lambda: frontdesk_client.ping(url, session_id, kind, user_agent=agent), lambda _: None
        )

    # -- shared sessions (Share menu, Crew tab) ---------------------------------

    def _crew_identity(self) -> tuple[str, str]:
        """Who this player is to the other members: the ScanLake ID and a name."""
        me = self.account.me if hasattr(self, "account") else None
        name = self.settings.user_name or (me.display_name if me is not None else "")
        return self.settings.user_id, name

    def _crew_profile(self) -> tuple[list[str], str, str]:
        s = self.settings
        return list(s.crew_roles), s.crew_ship or self.ship.label, s.crew_group

    def _crew_profile_changed(self, roles: list[str], ship: str, group: str) -> None:
        s = self.settings
        s.crew_roles, s.crew_ship, s.crew_group = list(roles), ship, group
        save_settings(s)
        self.crew.set_profile()

    def _crew_notice(self, text: str, urgent: bool) -> None:
        self.window.flash(text, 15000 if urgent else 8000)
        if urgent:
            self.overlay.flash(text)
            self._beep("stage")

    def _set_cluster_status(self, cluster_id: str, status: str) -> None:
        self.crew.set_status(cluster_id, status)

    def invite_link(self) -> str:
        share = self.crew.share
        return crew_client.invite_link(self.endpoints.crew, share.code) if share else ""

    def _copy_invite(self) -> None:
        QGuiApplication.clipboard().setText(self.invite_link())
        self.window.flash("Invite link copied: the others paste it under Share → Join…", 10000)

    def invite_session(self) -> None:
        """Shares the active session (starting one if needed) and copies its invite link."""
        if not self.crew.enabled:
            return
        if self.crew.shared:
            self._copy_invite()
            return
        if self.sessions.session is None:
            self.start_session()
        session = self.sessions.session
        assert session is not None
        user_id, name = self._crew_identity()
        url, agent = self.endpoints.crew, f"ScanLakeDesktop/{self.version}"
        location = (session.system, session.gravity_well, session.region)
        session_id = session.session_id

        def created(result: object) -> None:
            if self.stopped:
                return
            if not isinstance(result, str):
                self.window.flash(f"Sharing failed: {result}", 10000)
                return
            self.share_session(session_id, result)
            if self.crew.shared:
                self._copy_invite()

        self.window.flash("Sharing the session…")
        run_in_background(
            lambda: crew_client.create_room(
                url, session_id, user_id, name, location, user_agent=agent
            ),
            created,
        )

    def share_session(self, session_id: str, code: str) -> None:
        """The active session now has a room: remember it and connect."""
        session = self.sessions.session
        if session is None or session.session_id != session_id:
            return  # closed meanwhile
        self.crew.open(self.db.set_share(session_id, code))
        self._refresh_session()

    def join_session(self) -> None:
        """Asks for an invite link and joins that shared session."""
        if not self.crew.enabled:
            return
        from scanlake_desktop.ui.join_dialog import JoinDialog

        url, agent = self.endpoints.crew, f"ScanLakeDesktop/{self.version}"
        session = self.sessions.session
        note = ""
        if session is not None:
            scans = "1 scan uploads" if session.scans == 1 else f"{session.scans} scans upload"
            note = f"Joining closes your current session ({scans} now), unless it is this one."
        dialog = JoinDialog(
            lambda code: crew_client.room_info(url, code, user_agent=agent), note, self.window
        )
        if dialog.exec() and dialog.code and dialog.info:
            self.join_room(dialog.code, dialog.info)

    def join_room(self, code: str, info: crew_client.RoomInfo) -> None:
        """Joins a shared session: the local session takes the room's session id (one that
        exists here is reopened), so every member's scans upload under the same session."""
        session = self.sessions.session
        if session is None or session.session_id != info.session_id:
            if session is not None:
                self.close_session()
            location = (info.system, info.gravity_well, info.region)
            if info.system:
                self.window.session_bar.set_location(*location)
                self.change_location(*location)
            else:
                location = self.window.session_bar.location()
            started = self.sessions.start(*location, self.ship.key, session_id=info.session_id)
            self._ping_frontdesk(started.session_id, "open")
        # A joiner's batches get their own numbers: the same user id joining from another PC
        # would otherwise write the same file names in the Store
        base = random.randrange(1, 2_000_000) * 1000
        self.crew.open(self.db.set_share(info.session_id, code, base))
        self._refresh_session()
        self.window.flash(f"Joined {info.host_name or 'the'} shared session".replace("  ", " "))

    def leave_shared(self) -> None:
        """Stops sharing; the session itself goes on alone."""
        share = self.crew.share
        if share is None:
            return
        self.crew.close()
        self.db.end_share(share.session_id)
        self.window.flash("Left the shared session; your session goes on alone")

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
        self.staged_position = None  # an edited capture keeps its own position
        self.window.form.load(
            capture.scan, capture.rows, f"Editing scan from {capture.timestamp[11:19]}"
        )
        self.window.form.set_position(capture.position)
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
        self.nav_overlay.set_monitor(self.monitor)
        self.nav_overlay.set_anchor(new.nav_compass_x, new.nav_compass_y)
        self.nav_overlay.set_enabled(new.nav_compass)
        self.worker.configure(self._worker_config())
        self.crew.set_profile()  # the name the others see may have changed
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
        if not self.settings.user_name:
            self.crew.set_profile()  # the others see the Discord name then

    def _apply_account_profile(self, before: LocalProfile, after: LocalProfile) -> None:
        """The Discord account supplied a different ScanLake ID, name or org: use them here."""
        s = self.settings
        s.user_id, s.user_name, s.org = after.user_id, after.user_name, after.org
        save_settings(s)
        if after.user_id != before.user_id:
            self.store.forget_tokens()  # upload tokens belong to the previous ID
            self.window.flash("Using the ScanLake ID of your Discord account", 10000)
            self.crew.rejoin()  # under the new ID; this install's clusters follow it
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

    # -- navigation (Overview tab) --------------------------------------------

    @property
    def convention(self) -> Convention | None:
        return Convention.parse(self.settings.nav_camdir_convention)

    def _fast_positions(self) -> bool:
        """Read the debug text often: a cluster is the target, or a calibration records."""
        recording = self.calibration is not None and self.calibration.fit_result is None
        return self.nav is not None or recording

    def _refresh_clusters(self, captures: list[Capture]) -> None:
        """The own clusters changed: the shared session hears of it, the Overview shows them."""
        self.crew.publish(summarize_clusters(captures, self.sessions.clusters))
        self._show_clusters()

    def _show_clusters(self) -> None:
        """Shows the Overview's cards (own clusters and other members') and the Crew tab;
        also runs whenever the shared session reports a change."""
        if self.stopped:
            return
        overview = self.window.overview
        cards = self.crew.cards()
        by_id = {card.cluster_id: card.summary for card in cards}
        if self.nav is not None:
            summary = by_id.get(self.nav.target.cluster_id)
            if summary is None or not summary.has_position:
                self.stop_navigation()
                cards = self.crew.cards()
            else:
                self.nav.target = summary  # new scans move its centre
        self.clusters = [card.summary for card in cards]
        overview.show_clusters(cards)  # clears a target that is gone (-> stop)
        if self.nav is not None:
            overview.set_target(self.nav.target.cluster_id)
        state = self.last_state
        overview.update_position(state.position if state else None)
        self._show_crew()

    def _show_crew(self) -> None:
        crew, room = self.crew, self.crew.room
        members = crew.members()
        online = sum(1 for m in members if m.online)
        if not crew.enabled:
            state, note = "Shared sessions are turned off (SCANLAKE_CREW_URL is empty).", ""
        elif room is None:
            state = "Not shared. Invite copies a link for the others; Join… takes a link you got."
            note = ""
        elif crew.online:
            people = "1 member" if online == 1 else f"{online} members"
            state = f"Shared: {people} online. Everyone here sees each other's clusters."
            note = f"shared with {online - 1}" if online > 1 else "shared"
        else:
            state = "Reconnecting to the shared session…"
            note = "shared · reconnecting…"
        self.window.overview.set_shared(note, offline=room is not None and not crew.online)
        self.window.session_bar.set_shared(room is not None, crew.enabled)
        self.window.crew.show_room(
            members,
            room.you if room else "",
            self.invite_link(),
            state,
            {m.user_id: crew.target_of(m.user_id) for m in members},
            shared=room is not None,
            can_invite=True,
            enabled=crew.enabled,
        )

    def navigate_to(self, cluster_id: str) -> None:
        summary = next((s for s in self.clusters if s.cluster_id == cluster_id), None)
        if summary is None or not summary.has_position:
            return
        self.nav = NavTracker(summary, self.convention, self.camera_frames)
        self.window.overview.set_target(cluster_id)
        self.crew.set_target(cluster_id)  # the others see who is on the way
        self._update_navigation(*self._latest_position())
        self.worker.configure(self._worker_config())

    def stop_navigation(self) -> None:
        if self.nav is None:
            return
        self.nav = None
        self.window.overview.set_target(None)
        self.crew.set_target(None)
        self.nav_overlay.show_fix(None)
        self.worker.configure(self._worker_config())

    def _update_navigation(self, position: Position | None, position_t: float) -> None:
        """Feeds a position to the Overview, calibration and compass; one the position thread
        sent already is not used again (a LiveState may carry it, or an older one, later)."""
        if position_t > self.nav_position_t:
            self.nav_position_t = position_t
            self.window.overview.update_position(position)
            if self.calibration is not None:
                self.calibration.add(position, position_t)
            convention = self.convention
            if position is not None and convention is not None:
                self.camera_frames.add(position, position_t, convention)
        if self.nav is not None:
            if position_t < self.nav_position_t:  # older than one used: only age the fix
                position, position_t = None, 0.0
            fix = self.nav.update(position, position_t, time.time())
            self.window.overview.show_fix(fix)
            self.nav_overlay.show_fix(fix)

    def _nav_debug(self, on: bool) -> None:
        self.settings.nav_debug = on
        save_settings(self.settings)
        self.nav_overlay.set_debug(on)

    def _show_compass_status(self) -> None:
        conv = self.convention
        self.window.overview.set_compass_status(conv is not None, str(conv) if conv else "")

    def open_calibration(self) -> None:
        if self.calibration is None:
            dialog = CalibrationDialog(paths.nav_calibration_dir, self.window)
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.calibrated.connect(self._calibrated)
            dialog.finished.connect(self._calibration_closed)
            self.calibration = dialog
        self.calibration.show()
        self.calibration.raise_()
        self.worker.configure(self._worker_config())

    def _calibrated(self, result: CalibrationResult) -> None:
        if result.convention is None:
            return
        self.settings.nav_camdir_convention = str(result.convention)
        save_settings(self.settings)
        if self.nav is not None:
            self.nav.convention = result.convention
        self._show_compass_status()
        self.worker.configure(self._worker_config())

    def _calibration_closed(self, _code: int) -> None:
        self.calibration = None
        if not self.stopped:
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
        self._refresh_clusters(captures if active else [])
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
        self.crew.close()
        self.worker.stop()
        self.uploader.stop()
        self.overlay.close()
        self.nav_overlay.close()
        if self.calibration is not None:
            self.calibration.close()
        save_settings(self.settings)
        self.db.close()

    def shutdown(self) -> None:
        self.stop()
        QApplication.quit()
