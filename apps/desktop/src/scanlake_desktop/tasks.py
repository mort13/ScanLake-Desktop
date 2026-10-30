"""Run a blocking function on Qt's thread pool and get its result back on the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Done(QObject):
    finished = Signal(object)  # the result, or the exception it raised


class _Task(QRunnable):
    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.done = _Done()

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as err:
            result = err
        self.done.finished.emit(result)


_running: set[_Task] = set()


def run_in_background(fn: Callable[[], Any], on_done: Callable[[Any], None]) -> None:
    """Calls `fn()` off the GUI thread, then `on_done(result_or_exception)` on the GUI thread."""
    task = _Task(fn)
    task.setAutoDelete(False)
    _running.add(task)

    def finished(result: Any) -> None:
        _running.discard(task)
        on_done(result)

    task.done.finished.connect(finished)
    QThreadPool.globalInstance().start(task)
