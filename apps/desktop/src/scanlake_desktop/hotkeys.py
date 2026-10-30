"""Global hotkeys (they work while the game has focus), delivered as Qt signals.

pynput's listener runs on its own thread; its callback only emits a signal, which Qt queues to
the GUI thread. The keys are not swallowed, so the game still sees them.
"""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import QObject, Signal, SignalInstance

log = logging.getLogger(__name__)


def parse_key(name: str) -> object | None:
    """'f9' -> Key.f9, 'x' -> KeyCode('x'); None for an empty or unknown name."""
    from pynput import keyboard

    name = name.strip().lower()
    if not name:
        return None
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    return None


class HotkeyService(QObject):
    stage = Signal()
    save = Signal()
    new_cluster = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._listener: object | None = None
        self._bindings: dict[object, SignalInstance] = {}

    def bind(self, stage: str, save: str, new_cluster: str) -> None:
        self._bindings = {}
        for name, signal in (
            (stage, self.stage),
            (save, self.save),
            (new_cluster, self.new_cluster),
        ):
            key = parse_key(name)
            if key is not None:
                self._bindings[key] = signal
        if self._listener is None:
            self._start()

    def _start(self) -> None:
        if os.environ.get("SCANLAKE_NO_HOTKEYS"):
            return
        try:
            from pynput import keyboard
        except Exception as err:  # e.g. no display server on Linux CI
            log.warning("global hotkeys unavailable: %s", err)
            return

        def on_press(key: object) -> None:
            normalized = key
            char = getattr(key, "char", None)
            if isinstance(char, str):
                normalized = keyboard.KeyCode.from_char(char.lower())
            signal = self._bindings.get(normalized)
            if signal is not None:
                signal.emit()

        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        self._listener = listener

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()  # type: ignore[attr-defined]
            self._listener = None
