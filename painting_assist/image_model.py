# Copyright 2026 Vinay Williams

"""The image model: holds the immutable original RGB array and loads files.

Loading is offered two ways. :meth:`ImageModel.load_path` decodes synchronously
on the calling thread — Save/Export and the tests rely on it.
:meth:`ImageModel.load_path_async` runs the *same* decode on a
:class:`QThreadPool` worker and delivers the result on the GUI thread via a
queued signal, so opening a large image never freezes the UI. Rapid successive
async loads are latest-wins: only the most recent request's result is applied.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal


def _decode_rgb(path: str) -> np.ndarray:
    """Decode an image file to a C-contiguous uint8 RGB array.

    cv2 (BGR then BGR->RGB) is the fast path; Pillow is the fallback for the odd
    formats cv2 declines. Pure and Qt-free, so it runs safely on a worker thread.
    Raises on a missing/corrupt/unreadable file so callers can report the failure
    rather than silently loading nothing.
    """
    rgb: Optional[np.ndarray] = None

    try:
        import cv2

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = None

    if rgb is None:
        from PIL import Image

        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    return np.ascontiguousarray(rgb, dtype=np.uint8)


class _LoadSignals(QObject):
    """Signal holder for a load worker (a QRunnable cannot own signals).

    Each signal carries a monotonic ``serial`` so the model can drop the result
    of a load that a newer request has superseded (latest-wins).
    """

    loaded = Signal(int, str, object)  # (serial, path, rgb uint8 HxWx3)
    failed = Signal(int, str, str)  # (serial, path, message)


class _LoadTask(QRunnable):
    """Decode one image file off the GUI thread and report via ``signals``."""

    def __init__(self, path: str, serial: int, signals: _LoadSignals) -> None:
        super().__init__()
        self._path = path
        self._serial = serial
        self._signals = signals

    def run(self) -> None:  # noqa: D401 - QRunnable entry point
        """Decode and emit ``loaded``; emit ``failed`` on any error."""
        try:
            rgb = _decode_rgb(self._path)
        except Exception as exc:
            _emit(self._signals.failed, self._serial, self._path, str(exc))
            return
        _emit(self._signals.loaded, self._serial, self._path, rgb)


def _emit(signal, *args) -> None:
    """Emit from the worker, tolerating a holder Qt has already deleted.

    A decode that outlives the window (close during a slow load) must not
    raise on the pool thread; the result is simply dropped.
    """
    try:
        signal.emit(*args)
    except RuntimeError:
        pass


class ImageModel(QObject):
    """QObject holding ONLY the immutable original RGB array; emits on load."""

    image_loaded = Signal()  # emitted after a new original is set
    load_failed = Signal(str, str)  # (path, message) when an async load fails

    def __init__(self) -> None:
        super().__init__()
        self._original: Optional[np.ndarray] = None
        self._path: Optional[str] = None

        # Async loading: the shared thread pool, a signal holder marshalled back
        # onto this (GUI) thread, and a monotonic serial for latest-wins.
        self._pool = QThreadPool.globalInstance()
        self._load_serial = 0
        self._load_signals = _LoadSignals()
        self._load_signals.loaded.connect(self._on_async_loaded, Qt.QueuedConnection)
        self._load_signals.failed.connect(self._on_async_failed, Qt.QueuedConnection)

    def has_image(self) -> bool:
        """True iff an original image is currently loaded."""
        return self._original is not None

    def original(self) -> Optional[np.ndarray]:
        """The unmodified RGB uint8 HxWx3 source. Callers treat as read-only."""
        return self._original

    def path(self) -> Optional[str]:
        """Filesystem path of the loaded image, if known."""
        return self._path

    def set_image(self, rgb: np.ndarray, path: Optional[str] = None) -> None:
        """Store original (made C-contiguous uint8 RGB) + path, then emit image_loaded."""
        self._original = np.ascontiguousarray(rgb, dtype=np.uint8)
        self._path = path
        self.image_loaded.emit()

    def load_path(self, path: str) -> None:
        """Decode ``path`` synchronously on the calling thread and set it.

        The blocking decode path Save/Export and the tests rely on. GUI callers
        should prefer :meth:`load_path_async` so a large image does not freeze
        the UI. Falls back to Pillow if cv2 cannot read the file, and raises if
        neither can (matching the previous behaviour).
        """
        self.set_image(_decode_rgb(path), path)

    def load_path_async(self, path: str) -> None:
        """Decode ``path`` on a worker thread; deliver on the GUI thread.

        Emits :attr:`image_loaded` on success and :attr:`load_failed` on error,
        both on the GUI thread, so a caller can swap a synchronous
        :meth:`load_path` for this and keep its existing ``image_loaded`` wiring.
        Safe to call repeatedly: each call supersedes any still-running earlier
        one (latest-wins), so a burst of Opens applies only the last file.
        """
        self._load_serial += 1
        self._pool.start(_LoadTask(path, self._load_serial, self._load_signals))

    def shutdown(self) -> None:
        """Detach the async loader so a late decode cannot reach a dead model.

        Bumps the serial too, so any result that does arrive is ignored.
        """
        self._load_serial += 1
        for signal, slot in (
            (self._load_signals.loaded, self._on_async_loaded),
            (self._load_signals.failed, self._on_async_failed),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _on_async_loaded(self, serial: int, path: str, rgb: object) -> None:
        """GUI thread: apply a decoded image unless a newer load superseded it."""
        if serial != self._load_serial:
            return
        self.set_image(rgb, path)

    def _on_async_failed(self, serial: int, path: str, message: str) -> None:
        """GUI thread: report a failed load unless a newer load superseded it."""
        if serial != self._load_serial:
            return
        self.load_failed.emit(path, message)
