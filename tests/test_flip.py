# Copyright 2026 Vinay Williams

"""Tests for the Flip control."""

from __future__ import annotations

import numpy as np

from painting_assist.controls.flip import FlipControl


def _sample_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(7, 5, 3), dtype=np.uint8)


def test_disabled_is_identity() -> None:
    img = _sample_image()
    ctrl = FlipControl()
    ctrl.set_enabled(False)
    assert not ctrl.is_active()
    np.testing.assert_array_equal(ctrl.process(img), img)


def test_enabled_horizontal_mirrors() -> None:
    img = _sample_image()
    ctrl = FlipControl()
    ctrl.set_enabled(True)
    assert ctrl.is_active()
    np.testing.assert_array_equal(ctrl.process(img), np.fliplr(img))


def test_enabled_vertical_only() -> None:
    img = _sample_image()
    ctrl = FlipControl()
    ctrl.set_enabled(True)
    ctrl.set("horizontal", False)
    ctrl.set("vertical", True)
    assert ctrl.is_active()
    np.testing.assert_array_equal(ctrl.process(img), np.flipud(img))


def test_enabled_both_axes() -> None:
    img = _sample_image()
    ctrl = FlipControl()
    ctrl.set_enabled(True)
    ctrl.set("vertical", True)
    np.testing.assert_array_equal(ctrl.process(img), np.flipud(np.fliplr(img)))


def test_enabled_no_axis_is_inactive_and_identity() -> None:
    img = _sample_image()
    ctrl = FlipControl()
    ctrl.set_enabled(True)
    ctrl.set("horizontal", False)
    ctrl.set("vertical", False)
    assert not ctrl.is_active()
    np.testing.assert_array_equal(ctrl.process(img), img)


def test_does_not_mutate_input() -> None:
    img = _sample_image()
    original = img.copy()
    ctrl = FlipControl()
    ctrl.set_enabled(True)
    ctrl.set("vertical", True)
    out = ctrl.process(img)
    np.testing.assert_array_equal(img, original)
    out[0, 0, 0] = int(out[0, 0, 0]) ^ 0xFF
    np.testing.assert_array_equal(img, original)


def test_dtype_and_shape_preserved() -> None:
    img = _sample_image()
    ctrl = FlipControl()
    ctrl.set_enabled(True)
    ctrl.set("vertical", True)
    out = ctrl.process(img)
    assert out.dtype == img.dtype
    assert out.shape == img.shape


def test_registered_identity() -> None:
    assert FlipControl.id == "flip"
    assert FlipControl.order == 1
