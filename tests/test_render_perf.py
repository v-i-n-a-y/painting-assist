# Copyright 2026 Vinay Williams

"""Responsiveness changes, exercised headlessly (QCoreApplication, no window):

* the memoised interactive-preview downscale in ``render_controller`` — the
  downscale runs once per source, not once per interactive frame; and
* the threaded image load in ``image_model`` — decode off the GUI thread,
  deliver on it, latest-wins, and report (not crash) on a bad file.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QThreadPool

from painting_assist import render_controller
from painting_assist.image_model import ImageModel
from painting_assist.render_controller import RenderController


@pytest.fixture(scope="module")
def qapp():
    """A minimal (non-GUI) Qt application so signals and the pool have a loop."""
    yield QCoreApplication.instance() or QCoreApplication([])


def _pump(app, predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Spin the event loop until ``predicate()`` holds or ``timeout`` s elapses."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QThreadPool.globalInstance().waitForDone(20)
        app.processEvents()
    return predicate()


class _RecordingPipeline:
    """Minimal ControlPipeline stand-in: records the source each process() sees."""

    def __init__(self) -> None:
        self.sources: list = []

    def snapshot_states(self) -> dict:
        return {}

    def process(self, source, states=None, token=None, metadata_out=None):
        self.sources.append(source)
        return source


def _counting_downscale(monkeypatch) -> dict:
    """Replace ``render_controller._downscale`` with a counting passthrough."""
    calls = {"n": 0}
    real = render_controller._downscale

    def wrapper(src, scale):
        calls["n"] += 1
        return real(src, scale)

    monkeypatch.setattr(render_controller, "_downscale", wrapper)
    return calls


# --------------------------------------------------------------------------- #
# Task 1: preview downscale memoisation
# --------------------------------------------------------------------------- #
def test_preview_source_memoises_and_reuses(qapp, monkeypatch):
    calls = _counting_downscale(monkeypatch)
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    ctrl = RenderController(_RecordingPipeline(), lambda: source)
    try:
        first = ctrl._preview_source(source, RenderController.PREVIEW_SCALE)
        second = ctrl._preview_source(source, RenderController.PREVIEW_SCALE)
        assert first is second  # the memoised array is reused, not recomputed
        assert calls["n"] == 1
        assert first.shape[:2] == (40, 40)  # 0.4 * 100
        # A full pass renders the source untouched and downscales nothing.
        assert ctrl._preview_source(source, 1.0) is source
        assert calls["n"] == 1
    finally:
        ctrl.shutdown()


def test_invalidate_source_drops_preview_cache(qapp, monkeypatch):
    calls = _counting_downscale(monkeypatch)
    source = np.zeros((80, 60, 3), dtype=np.uint8)
    ctrl = RenderController(_RecordingPipeline(), lambda: source)
    try:
        first = ctrl._preview_source(source, RenderController.PREVIEW_SCALE)
        assert calls["n"] == 1
        ctrl.invalidate_source()
        again = ctrl._preview_source(source, RenderController.PREVIEW_SCALE)
        assert calls["n"] == 2  # recomputed after the cache was invalidated
        assert again is not first
    finally:
        ctrl.shutdown()


def test_downscale_runs_once_across_interactive_dispatches(qapp, monkeypatch):
    calls = _counting_downscale(monkeypatch)
    source = np.zeros((200, 150, 3), dtype=np.uint8)
    pipe = _RecordingPipeline()
    ctrl = RenderController(pipe, lambda: source)
    ctrl._latest_interactive = True  # simulate an in-progress slider drag
    try:
        for _ in range(5):
            ctrl._dispatch()
        assert calls["n"] == 1  # one downscale across five interactive frames
    finally:
        ctrl.shutdown()  # drains the worker pool
    # Every frame was handed the very same downscaled array object.
    assert len(pipe.sources) == 5
    assert all(s is pipe.sources[0] for s in pipe.sources)
    assert pipe.sources[0].shape[:2] == (80, 60)  # 0.4 * (200, 150)


# --------------------------------------------------------------------------- #
# Task 2: threaded image load
# --------------------------------------------------------------------------- #
def _write_png(path, h: int, w: int) -> None:
    """Write a small, real, decodable PNG (content is irrelevant to the tests)."""
    import cv2

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 200
    cv2.imwrite(str(path), img)


def test_load_path_async_delivers_and_emits(qapp, tmp_path):
    path = tmp_path / "ref.png"
    _write_png(path, 8, 6)
    model = ImageModel()
    loaded: list = []
    model.image_loaded.connect(lambda: loaded.append(model.original()))

    model.load_path_async(str(path))
    assert _pump(qapp, lambda: bool(loaded)), "image_loaded never fired"

    assert model.has_image()
    got = model.original()
    assert got is not None
    assert got.shape == (8, 6, 3)
    assert got.dtype == np.uint8
    assert model.path() == str(path)


def test_load_path_async_reports_error_for_missing(qapp, tmp_path):
    missing = str(tmp_path / "does_not_exist.png")
    model = ImageModel()
    errors: list = []
    loaded: list = []
    model.load_failed.connect(lambda p, m: errors.append((p, m)))
    model.image_loaded.connect(lambda: loaded.append(True))

    model.load_path_async(missing)
    assert _pump(qapp, lambda: bool(errors)), "load_failed never fired"

    assert errors[0][0] == missing
    assert errors[0][1]  # a non-empty message
    assert not loaded
    assert not model.has_image()


def test_load_path_async_reports_error_for_corrupt(qapp, tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not a valid image file")
    model = ImageModel()
    errors: list = []
    model.load_failed.connect(lambda p, m: errors.append((p, m)))

    model.load_path_async(str(path))
    assert _pump(qapp, lambda: bool(errors)), "load_failed never fired"

    assert errors[0][0] == str(path)
    assert not model.has_image()


def test_load_path_async_latest_wins(qapp, tmp_path):
    p1 = tmp_path / "first.png"
    p2 = tmp_path / "second.png"
    _write_png(p1, 4, 4)
    _write_png(p2, 10, 12)
    model = ImageModel()
    seen: list = []
    model.image_loaded.connect(lambda: seen.append(model.path()))

    # Two loads in quick succession: only the latest should be applied.
    model.load_path_async(str(p1))
    model.load_path_async(str(p2))
    assert _pump(qapp, lambda: model.path() == str(p2)), "latest load never landed"

    assert str(p1) not in seen  # the superseded load was dropped, not shown
    assert model.original().shape == (10, 12, 3)


def test_load_path_sync_still_works(qapp, tmp_path):
    # The synchronous path Save/Export and the tests rely on must stay intact.
    path = tmp_path / "sync.png"
    _write_png(path, 5, 7)
    model = ImageModel()
    model.load_path(str(path))
    assert model.has_image()
    assert model.original().shape == (5, 7, 3)
    assert model.path() == str(path)
