# Copyright 2026 Vinay Williams

"""The responsiveness layer — the single, decided rendering approach.

This module owns ALL of the responsiveness mechanics so that
:class:`~painting_assist.pipeline.ControlPipeline` stays thread-agnostic and
:class:`~painting_assist.main_window.MainWindow` stays thin. The four
mechanisms, implemented exactly one way:

1. **Debounce** — a single-shot :class:`QTimer` (``DEBOUNCE_MS``). Every
   :meth:`RenderController.request` restarts it; only the trailing tick
   dispatches a render, coalescing slider bursts.
2. **Worker thread** — :class:`QThreadPool` + a :class:`QRunnable`
   (``_RenderTask``). The heavy ``pipeline.process`` runs off the GUI thread.
3. **Downscale while dragging** — ``interactive=True`` renders at
   ``PREVIEW_SCALE``; the trailing full pass on release renders at 1.0.
4. **Stale-drop + single-worker discipline** — a monotonic ``generation`` int
   stamps each task; results with a stale generation are dropped. Only one task
   is in flight at a time, so the pipeline cache needs no locking.
"""

from __future__ import annotations

import logging

from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)

try:  # cv2 is the fast path for downscaling; degrade gracefully if absent.
    import cv2
except Exception:  # pragma: no cover - cv2 is a hard dependency in practice
    cv2 = None  # type: ignore[assignment]

from painting_assist.pipeline import ControlPipeline

_log = logging.getLogger(__name__)


class _TaskSignals(QObject):
    """Signal holder for a :class:`_RenderTask` (a QRunnable cannot have signals).

    ``done`` carries ``(image, generation, was_full, scale, metadata)`` and is
    connected with a :data:`Qt.QueuedConnection` so the result is marshalled back
    onto the GUI thread regardless of which worker thread emits it. ``image`` is
    ``None`` when the render raised, so the controller can clear its busy flag
    without displaying anything. ``metadata`` is the pipeline's side-channel dict
    (e.g. the quantize palette), or ``None`` on failure.
    """

    # (image|None, generation, was_full, scale, metadata|None)
    done = Signal(object, int, bool, float, object)


class _RenderTask(QRunnable):
    """Run ``pipeline.process`` on a worker thread and emit the result.

    Holds an immutable snapshot of everything it needs so it is fully decoupled
    from concurrent GUI edits: the source array, the per-control state snapshot,
    the chosen scale, the generation stamp, the pipeline, and a signal holder.
    """

    def __init__(
        self,
        pipeline: ControlPipeline,
        source: np.ndarray,
        states: dict,
        scale: float,
        token: int,
        generation: int,
        signals: _TaskSignals,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._source = source
        self._states = states
        self._scale = scale
        self._token = token
        self._generation = generation
        self._signals = signals

    def run(self) -> None:  # noqa: D401 - QRunnable entry point
        """Downscale if interactive, process, and emit ``done``.

        On any exception the frame is emitted with a ``None`` image (and logged),
        so the controller always clears its busy flag instead of wedging.
        """
        src = self._source
        was_full = self._scale >= 1.0
        metadata: dict = {}
        try:
            if not was_full and src is not None:
                src = self._downscale(src, self._scale)
            out = self._pipeline.process(
                src, self._states, self._token, metadata_out=metadata
            )
        except Exception:
            _log.exception("Render task failed (generation %d)", self._generation)
            self._signals.done.emit(None, self._generation, was_full, self._scale, None)
            return
        self._signals.done.emit(out, self._generation, was_full, self._scale, metadata)

    @staticmethod
    def _downscale(src: np.ndarray, scale: float) -> np.ndarray:
        """Return a downscaled copy of ``src`` for a fast interactive preview."""
        h, w = src.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if cv2 is not None:
            return cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Pure-numpy nearest-neighbour fallback (only if cv2 is unavailable).
        ys = (np.arange(new_h) * (h / new_h)).astype(np.intp)
        xs = (np.arange(new_w) * (w / new_w)).astype(np.intp)
        return np.ascontiguousarray(src[ys][:, xs])


class RenderController(QObject):
    """Debounced, threaded, stale-dropping renderer.

    Emits :attr:`rendered` ``(image, was_full_res)`` on the GUI thread whenever a
    fresh processed image is ready. Callers drive it solely via
    :meth:`request`.
    """

    # (image: np.ndarray RGB, was_full_res: bool, scale: float, metadata: dict)
    # ``scale`` is the factor the emitted image was rendered at (1.0 full-res,
    # PREVIEW_SCALE for an interactive preview) so the view can display a
    # downscaled preview at full on-screen size. ``metadata`` carries any
    # control side-channel outputs for this frame (e.g. the quantize palette).
    rendered = Signal(object, bool, float, object)

    PREVIEW_SCALE = 0.4
    # Interactive (dragging) frames debounce briefly for snappy feedback; the
    # trailing full-quality pass debounces longer to coalesce the burst.
    DEBOUNCE_MS = 24
    DEBOUNCE_FULL_MS = 60

    def __init__(
        self,
        pipeline: ControlPipeline,
        source_provider: Callable[[], Optional[np.ndarray]],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._source_provider = source_provider
        self._pool = QThreadPool.globalInstance()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.DEBOUNCE_MS)
        self._timer.timeout.connect(self._on_debounce)

        self._signals = _TaskSignals()
        self._signals.done.connect(self._on_task_done, Qt.QueuedConnection)

        self._generation = 0
        self._latest_interactive = False
        self._busy = False
        self._pending = False
        self._shutdown = False

        # Cache token: identifies the (source array, scale) combo so the pipeline
        # reuses its prefix cache across frames that render the same logical
        # source at the same scale (each interactive frame gets a fresh
        # downscaled array, so keying on array identity alone would never hit).
        self._source_token = 0
        self._source_key: Optional[tuple] = None

    # ------------------------------------------------------------------ #
    # Public API (GUI thread)
    # ------------------------------------------------------------------ #
    def request(self, interactive: bool) -> None:
        """(Re)start the debounce timer; remember the latest interactive flag."""
        if self._shutdown:
            return
        self._latest_interactive = bool(interactive)
        self._timer.setInterval(
            self.DEBOUNCE_MS if interactive else self.DEBOUNCE_FULL_MS
        )
        self._timer.start()

    def shutdown(self) -> None:
        """Stop scheduling, drop pending results, and drain in-flight workers.

        Called on window close so a worker thread cannot emit back into
        already-destroyed Qt objects (which segfaults during teardown).
        """
        self._shutdown = True
        self._timer.stop()
        try:
            self._signals.done.disconnect(self._on_task_done)
        except (RuntimeError, TypeError):  # already disconnected / no connection
            pass
        self._pool.waitForDone()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _on_debounce(self) -> None:
        """Debounce fired: dispatch a render if idle, else mark a pending one."""
        if self._shutdown:
            return
        if self._busy:
            self._pending = True
            return
        self._dispatch()

    def _dispatch(self) -> None:
        """Snapshot state and start one worker task (GUI thread)."""
        source = self._source_provider()
        if source is None:
            return
        self._generation += 1
        states = self._pipeline.snapshot_states()
        scale = self.PREVIEW_SCALE if self._latest_interactive else 1.0
        token = self._token_for(source, scale)
        self._busy = True
        task = _RenderTask(
            pipeline=self._pipeline,
            source=source,
            states=states,
            scale=scale,
            token=token,
            generation=self._generation,
            signals=self._signals,
        )
        self._pool.start(task)

    def _token_for(self, source: np.ndarray, scale: float) -> int:
        """Return a cache token, bumped whenever the (source, scale) combo changes."""
        key = (id(source), scale)
        if key != self._source_key:
            self._source_key = key
            self._source_token += 1
        return self._source_token

    def _on_task_done(
        self,
        image: object,
        generation: int,
        was_full: bool,
        scale: float,
        metadata: object,
    ) -> None:
        """GUI thread (QueuedConnection): emit fresh results, drop stale/failed ones."""
        self._busy = False
        if self._shutdown:
            return
        # image is None when the render raised; clear busy but display nothing.
        if image is not None and generation == self._generation:
            self.rendered.emit(image, was_full, scale, metadata)
        # If requests arrived while busy, dispatch the freshest snapshot now.
        if self._pending:
            self._pending = False
            self._dispatch()
