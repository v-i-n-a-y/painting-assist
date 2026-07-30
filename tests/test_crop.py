# Copyright 2026 Vinay Williams

"""CropControl geometry: rect_norm clamping, process output shape, full-frame
detection, and degenerate-rect safety."""

from __future__ import annotations

import numpy as np

from painting_assist.controls.crop import CropControl


def _img(h=100, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_rect_norm_clamps_into_unit_square():
    c = CropControl()
    c.set("rx", 0.8)
    c.set("ry", 0.9)
    c.set("rw", 0.9)  # would exceed 1.0 with rx=0.8 -> clamped to 0.2
    c.set("rh", 0.5)  # would exceed 1.0 with ry=0.9 -> clamped to 0.1
    rx, ry, rw, rh = c.rect_norm()
    assert rx == 0.8 and ry == 0.9
    assert abs(rw - 0.2) < 1e-9
    assert abs(rh - 0.1) < 1e-9


def test_rect_norm_negative_clamped_to_zero():
    c = CropControl()
    c.set("rx", -1.0)
    c.set("ry", -1.0)
    rx, ry, _, _ = c.rect_norm()
    assert rx == 0.0 and ry == 0.0


def test_process_output_shape_for_known_rect():
    c = CropControl()
    # Crop the right half, bottom half.
    c.set("rx", 0.5)
    c.set("ry", 0.5)
    c.set("rw", 0.5)
    c.set("rh", 0.5)
    out = c.process(_img(h=100, w=200))
    assert out.shape == (50, 100, 3)
    assert out.dtype == np.uint8


def test_full_frame_is_not_active():
    c = CropControl()
    c.set_enabled(True)
    # Default rect is the full frame (rx=ry=0, rw=rh=1).
    assert c.is_active() is False


def test_partial_crop_is_active_when_enabled():
    c = CropControl()
    c.set_enabled(True)
    c.set("rw", 0.5)
    assert c.is_active() is True


def test_has_crop_tracks_applied_rect_independent_of_enabled():
    c = CropControl()
    # Full frame, disabled: no crop applied.
    assert c.has_crop() is False
    # A real crop is "applied" even while the control is disabled (unlike
    # is_active, which also requires enabled) — the editor uses this to know
    # the canvas aspect ratio is now fixed.
    c.set("rw", 0.5)
    assert c.has_crop() is True
    assert c.enabled is False


def test_degenerate_rect_never_empty():
    c = CropControl()
    # A zero-size rect must still yield at least a 1x1 region, never an empty
    # array (process floors width/height to >= 1 px).
    c.set("rx", 0.5)
    c.set("ry", 0.5)
    c.set("rw", 0.0)
    c.set("rh", 0.0)
    out = c.process(_img(h=100, w=200))
    assert out.size > 0
    assert out.shape[0] >= 1 and out.shape[1] >= 1
    assert out.shape[2] == 3
