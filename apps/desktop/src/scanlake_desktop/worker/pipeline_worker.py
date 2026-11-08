"""The capture loop: grab -> recognize -> vote, on its own thread.

The GUI never waits for it. Configuration changes arrive through a queue and are applied between
frames, so switching profile, backend or region always takes effect on the next frame. Results
leave as `LiveState` objects (plain data) through a Qt signal, at most ~10 per second.

While the HUD is not found the loop slows to 2 frames per second, and every few seconds it
searches the whole monitor at the expected scale to find where the HUD actually is. A frame that
fails (screen not readable, bad region) is reported once and skipped; it never ends the loop.

The loop also reads the ship position from the game's debug text in the top-right corner
(r_DisplayInfo), HUD or not (the live panel previews it), about once a second, or every few
seconds while there is none.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from scanlake_core.domain.position import Position, parse_position
from scanlake_core.geometry import Rect, apply
from scanlake_core.preprocess import Gray, to_gray
from scanlake_core.profile.model import HudProfile
from scanlake_core.recognition import Engine, FrameRecognizer, RoiReading
from scanlake_core.vision.anchors import AnchorLocator
from scanlake_core.vision.debug_text import DebugTextReader, corner
from scanlake_core.voting import Observation, RoiVoter, Vote
from scanlake_desktop.capture.source import (
    CaptureSource,
    FileSource,
    Monitor,
    MssSource,
    auto_region,
    is_black,
)
from scanlake_inference.bench import benchmark, pick_fastest
from scanlake_inference.registry import AUTO, create_engine

log = logging.getLogger(__name__)

SEARCH_FPS = 2.0
WIDE_SEARCH_EVERY_S = 3.0
LOST_RESET_S = 1.0
POSITION_EVERY_S = 1.0  # debug-text reads while it is visible ...
POSITION_IDLE_S = 3.0  # ... and while it is not (r_DisplayInfo off)
POSITION_FRESH_S = 2.5  # a staged scan takes the debug-text position only if this fresh
EMIT_EVERY_S = 0.1
MIN_REGION = 32  # px: a smaller capture region cannot hold the HUD


@dataclass(frozen=True)
class WorkerConfig:
    hud: HudProfile
    ship: str
    templates: dict[str, Gray]
    models_dir: Path
    backend: str  # a backend id, or "auto" (resolved with `auto_backend` or a benchmark)
    auto_backend: str | None
    monitor: Monitor
    manual_region: Rect | None  # absolute; None = derive from the profile
    fps: int = 10
    vote_window_s: float = 1.5
    dml_device_id: int = 0
    debug: bool = False
    collect_crops: bool = False  # keep each ROI's crop for training data (dev mode)
    model_stamp: str = ""  # changes when the model files change -> new engine
    replay: tuple[Path, ...] = ()  # dev mode: read these images instead of the screen


@dataclass(frozen=True)
class LiveState:
    t: float
    ship: str
    locked: bool
    scale: float
    fps: float
    backend: str
    region: Rect
    black: bool
    votes: dict[str, Vote]
    readings: dict[str, RoiReading]
    timings: dict[str, float] = field(default_factory=dict)
    frame: np.ndarray | None = None  # debug only
    anchors: dict[str, tuple[float, float, int, int, float]] = field(default_factory=dict)
    position: Position | None = None  # the last ship position read from the debug text ...
    position_t: float = 0.0  # ... and when (time.time(); 0 = never)
    # (text, score) of the CamDir and Zone lines the latest read saw, parsed or not
    position_lines: tuple[tuple[str, float], ...] = ()


class PipelineWorker(QThread):
    live = Signal(object)  # LiveState
    problem = Signal(str)
    backend_resolved = Signal(str)  # what "auto" picked (to remember it)

    def __init__(self) -> None:
        super().__init__()
        self._commands: queue.Queue[tuple[str, object]] = queue.Queue()

    # -- commands (any thread) ----------------------------------------------

    def configure(self, config: WorkerConfig) -> None:
        self._commands.put(("config", config))

    def reset_votes(self) -> None:
        self._commands.put(("reset", None))

    def set_paused(self, paused: bool) -> None:
        self._commands.put(("pause", paused))

    def stop(self) -> None:
        self._commands.put(("stop", None))
        self.wait(3000)

    # -- loop ---------------------------------------------------------------

    def run(self) -> None:
        loop = _Loop(self)
        try:
            loop.run()
        except Exception as err:  # keep the app alive, tell the user
            log.exception("capture loop crashed")
            self.problem.emit(f"Capture stopped: {err}")
        finally:
            loop.close()


class _Loop:
    def __init__(self, worker: PipelineWorker) -> None:
        self.w = worker
        self.config: WorkerConfig | None = None
        self.paused = False
        self.source: CaptureSource | None = None
        self.engine: Engine | None = None
        self.engine_key: tuple[object, ...] = ()
        self.backend = ""
        self.recognizer: FrameRecognizer | None = None
        self.voters: dict[str, RoiVoter] = {}
        self.region = Rect(0, 0, 0, 0)
        self.found_region: Rect | None = None
        self.last_lock = 0.0
        self.last_wide = 0.0
        self.last_emit = 0.0
        self.frame_times: list[float] = []
        self.last_problem = ""
        self.position_reader: DebugTextReader | None = None  # loads the fonts on first use
        self.position: Position | None = None
        self.position_t = 0.0
        self.position_lines: tuple[tuple[str, float], ...] = ()
        self.last_position_try = 0.0
        self.position_failed = False

    def close(self) -> None:
        if self.source is not None:
            self.source.close()

    def run(self) -> None:
        while True:
            timeout = 0.2 if (self.paused or self.recognizer is None) else 0.0
            if not self._drain(timeout):
                return
            if self.paused or self.recognizer is None or self.config is None:
                continue
            started = time.perf_counter()
            try:
                locked = self._tick()
                self.last_problem = ""
            except Exception as err:  # one bad frame must not stop capturing
                locked = False
                self._frame_failed(err)
            fps = self.config.fps if locked else min(self.config.fps, SEARCH_FPS)
            wait = 1.0 / fps - (time.perf_counter() - started)
            if wait > 0 and not self._drain(wait):
                return

    def _drain(self, timeout: float) -> bool:
        """Applies queued commands; waits up to `timeout` for the first. False = stop."""
        deadline = time.perf_counter() + timeout
        while True:
            remaining = deadline - time.perf_counter()
            try:
                kind, payload = (
                    self.w._commands.get(timeout=remaining)
                    if remaining > 0
                    else self.w._commands.get_nowait()
                )
            except queue.Empty:
                return True
            if kind == "stop":
                return False
            if kind == "pause":
                self.paused = bool(payload)
            elif kind == "reset":
                for voter in self.voters.values():
                    voter.reset()
            elif kind == "config":
                self._apply(payload)  # type: ignore[arg-type]

    def _apply(self, config: WorkerConfig) -> None:
        old = self.config
        self.config = config
        if self.source is None or old is None or old.replay != config.replay:
            if self.source is not None:
                self.source.close()
            self.source = FileSource(list(config.replay)) if config.replay else MssSource()
        if old is None or old.monitor != config.monitor or old.hud is not config.hud:
            self.found_region = None

        key = (config.backend, config.auto_backend, config.models_dir, config.dml_device_id,
               config.model_stamp)  # fmt: skip
        if self.engine is None or key != self.engine_key:
            self._make_engine(config)
            self.engine_key = key
            self.recognizer = None
        if self.engine is None:
            return
        if (
            self.recognizer is None
            or old is None
            or old.hud is not config.hud
            or old.templates is not config.templates
            or old.debug != config.debug
            or old.collect_crops != config.collect_crops
        ):
            self.recognizer = FrameRecognizer(
                config.hud, config.templates, self.engine,
                keep_images=config.debug or config.collect_crops,
            )  # fmt: skip
            self.voters = {}
        if old is None or old.vote_window_s != config.vote_window_s or not self.voters:
            self.voters = {
                roi.name: RoiVoter(window_s=config.vote_window_s) for roi in config.hud.main.rois
            }

    def _make_engine(self, config: WorkerConfig) -> None:
        backend = config.backend
        if backend == AUTO:
            backend = config.auto_backend or ""
            if not backend:
                results = benchmark(config.models_dir, ["ort-cpu", "ort-dml"])
                backend = pick_fastest(results)
                log.info("backend benchmark: %s -> %s", results, backend)
                self.w.backend_resolved.emit(backend)
        try:
            self.engine = create_engine(backend, config.models_dir, device_id=config.dml_device_id)
            self.backend = backend
        except Exception as err:
            log.exception("backend %s failed", backend)
            self.w.problem.emit(f"{backend} is not usable ({err}); using ONNX Runtime on the CPU")
            self.engine = create_engine("ort-cpu", config.models_dir)
            self.backend = "ort-cpu"

    def _frame_failed(self, err: Exception) -> None:
        """Logs and reports a failing frame once (not every frame), and searches afresh."""
        self.found_region = None
        message = f"Capture problem: {err}"
        if message != self.last_problem:
            self.last_problem = message
            log.exception("capture frame failed")
            self.w.problem.emit(message)

    def _current_region(self) -> Rect:
        assert self.config is not None
        for region in (self.config.manual_region, self.found_region):
            if region is not None and region.w >= MIN_REGION and region.h >= MIN_REGION:
                return region
        return auto_region(self.config.hud.main.search_region, self.config.monitor)

    def _tick(self) -> bool:
        assert self.config is not None and self.recognizer is not None and self.source is not None
        now = time.time()
        self.region = self._current_region()
        frame = self.source.grab(self.region)
        black = is_black(frame)
        expected = self.config.monitor.scale
        result = self.recognizer.process(frame, expected_scale=expected)

        if result.locked:
            self.last_lock = now
            for name, reading in result.readings.items():
                voter = self.voters.get(name)
                if voter is not None:
                    voter.add(Observation(now, reading.text, reading.confidence))
        else:
            if now - self.last_lock > LOST_RESET_S:
                for voter in self.voters.values():
                    voter.reset()
            if (
                not self.config.replay
                and self.config.manual_region is None
                and now - self.last_wide > WIDE_SEARCH_EVERY_S
                and now - self.last_lock > 2.0
            ):
                self.last_wide = now
                self._wide_search(expected)
        self._read_position(now, frame)

        self.frame_times.append(now)
        self.frame_times = [t for t in self.frame_times if now - t < 2.0]
        if now - self.last_emit >= EMIT_EVERY_S:
            self.last_emit = now
            fix = result.fix
            anchors = {
                n: (m.x, m.y, m.w, m.h, m.confidence) for n, m in fix.matches.items() if m.found
            }
            fps = (len(self.frame_times) - 1) / max(now - self.frame_times[0], 1e-6)
            self.w.live.emit(
                LiveState(
                    t=now,
                    ship=self.config.ship,
                    locked=result.locked,
                    scale=fix.scale if result.locked else expected,
                    fps=fps if len(self.frame_times) > 1 else 0.0,
                    backend=self.backend,
                    region=self.region,
                    black=black,
                    votes={n: v.last for n, v in self.voters.items()},
                    readings=result.readings,
                    timings=result.timings,
                    frame=frame if self.config.debug else None,
                    anchors=anchors,
                    position=self.position,
                    position_t=self.position_t,
                    position_lines=self.position_lines,
                )
            )
        return result.locked

    def _read_position(self, now: float, frame: np.ndarray) -> None:
        """Reads the debug text now and then; keeps the last position that parsed.

        A replayed frame is a whole screen (or a region without the corner), so its corner is
        cut from it; the screen's corner is grabbed on its own. Failures never stop capturing.
        """
        assert self.config is not None and self.source is not None
        seen = self.position is not None and now - self.position_t < POSITION_IDLE_S
        if now - self.last_position_try < (POSITION_EVERY_S if seen else POSITION_IDLE_S):
            return
        self.last_position_try = now
        try:
            if self.position_reader is None:
                self.position_reader = DebugTextReader()
            if not self.position_reader.usable:
                return
            image = (
                corner(frame) if self.config.replay
                else self.source.grab(self.config.monitor.debug_text_rect)
            )  # fmt: skip
            lines = self.position_reader.read(image)
            self.position_lines = tuple((line.text, line.score) for line in lines)
            position = parse_position(self.position_lines)
            self.position_failed = False
        except Exception:
            if not self.position_failed:
                log.exception("reading the debug text failed")
            self.position_failed = True
            return
        if position is not None:
            self.position, self.position_t = position, now

    def _wide_search(self, expected: float) -> None:
        """Looks for the HUD on the whole monitor (only at the expected scale, to stay cheap)."""
        assert self.config is not None and self.source is not None
        monitor = self.config.monitor
        capture = self.config.hud.main
        frame = self.source.grab(monitor.rect)
        if is_black(frame):
            return
        locator = AnchorLocator(capture.anchors, self.config.templates)
        scales = [round(expected * f, 4) for f in (0.94, 1.0, 1.06)]
        fix = locator.locate(to_gray(frame), expected, only_scales=scales)
        if fix.transform is None:
            return
        r = capture.search_region
        x0, y0 = apply(fix.transform, 0, 0)
        x1, y1 = apply(fix.transform, r.w, r.h)
        local = Rect(round(x0), round(y0), round(x1 - x0), round(y1 - y0)).pad(0.25)
        local = local.clip(monitor.width, monitor.height)
        if local.w < MIN_REGION or local.h < MIN_REGION:  # (mostly) off this monitor
            return
        self.found_region = replace(local, x=local.x + monitor.left, y=local.y + monitor.top)
        log.info(
            "HUD found by full-monitor search at %s (scale %.3f)", self.found_region, fix.scale
        )
