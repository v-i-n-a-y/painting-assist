from __future__ import annotations

"""Headless tests for the control pipeline + BlurControl basics.

Runs without Qt: builds a ControlPipeline over a synthetic RGB image and
checks the array contract (shape/dtype) plus active vs. passthrough behaviour.
"""

import numpy as np

from painting_assist.controls import registry  # noqa: F401  -- populate registry
from painting_assist.controls.blur import BlurControl
from painting_assist.pipeline import ControlPipeline


def _make_image() -> np.ndarray:
    """A 256x256x3 uint8 image with structure so blur has something to do."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    # Add a hard edge so neighbouring-pixel blurring is unmistakable.
    img[:, :128] = 30
    img[:, 128:] = 220
    return np.ascontiguousarray(img)


def test_disabled_control_is_passthrough():
    img = _make_image()
    pipe = ControlPipeline([BlurControl()])
    out_off = pipe.process(img)
    assert out_off.shape == img.shape
    assert out_off.dtype == np.uint8
    assert np.array_equal(out_off, img)


def test_enabled_but_radius_zero_is_passthrough():
    img = _make_image()
    pipe = ControlPipeline([BlurControl()])
    pipe.set_enabled("blur", True)
    pipe.set_value("blur", "radius", 0)
    out_zero = pipe.process(img)
    assert np.array_equal(out_zero, img)


def test_enabled_radius_blurs_and_preserves_contract():
    img = _make_image()
    pipe = ControlPipeline([BlurControl()])
    pipe.set_enabled("blur", True)
    pipe.set_value("blur", "radius", 60)
    out_blur = pipe.process(img)
    assert out_blur.shape == img.shape
    assert out_blur.dtype == np.uint8
    assert not np.array_equal(out_blur, img)
    # The hard vertical edge must be softened: the seam column should no longer
    # be a pure step (intermediate values appear).
    seam = out_blur[128, 120:136, 0].astype(int)
    assert seam.min() > 30 and seam.max() < 220


def test_processing_does_not_mutate_input():
    img = _make_image()
    pipe = ControlPipeline([BlurControl()])
    pipe.set_enabled("blur", True)
    pipe.set_value("blur", "radius", 60)
    pipe.process(img)
    assert img[0, 0, 0] == 30


def test_toggle_off_returns_passthrough():
    img = _make_image()
    pipe = ControlPipeline([BlurControl()])
    pipe.set_enabled("blur", True)
    pipe.set_value("blur", "radius", 60)
    pipe.process(img)
    pipe.set_enabled("blur", False)
    out_off2 = pipe.process(img)
    assert np.array_equal(out_off2, img)


def test_snapshot_path_blurs():
    img = _make_image()
    pipe = ControlPipeline([BlurControl()])
    pipe.set_enabled("blur", True)
    pipe.set_value("blur", "radius", 40)
    states = pipe.snapshot_states()
    out_snap = pipe.process(img, states)
    assert out_snap.shape == img.shape
    assert not np.array_equal(out_snap, img)
