# Copyright 2026 Vinay Williams

"""The compare-wipe "before" pixmap is smooth-scaled to the on-screen image
box on every paint. This exercises the pre-scaled cache added to
``ImageView._compare_scaled_pixmap``: it must be built once and reused across
repeated paints at the same size, and rebuilt when the on-screen image box
changes size, e.g. on zoom."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

from painting_assist.widgets.image_view import ImageView


def _img(w=200, h=150):
    rng = np.random.default_rng(3)
    return np.ascontiguousarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


@pytest.fixture
def app():
    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    return application


def test_compare_cache_built_once_and_rebuilt_on_zoom(app):
    view = ImageView()
    view.resize(320, 240)
    view.set_image(_img())
    view.set_compare_image(_img())
    view.set_compare_mode(True)

    view.grab()
    assert view._compare_scaled_cache is not None
    first_pixmap = view._compare_scaled_cache["pixmap"]
    first_key = view._compare_scaled_cache["key"]

    view.grab()
    assert view._compare_scaled_cache["key"] == first_key
    assert view._compare_scaled_cache["pixmap"] is first_pixmap

    view.scale(1.5, 1.5)
    view.grab()
    assert view._compare_scaled_cache["key"] != first_key
    assert view._compare_scaled_cache["pixmap"] is not first_pixmap

    view.deleteLater()
