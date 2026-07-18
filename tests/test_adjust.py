# Copyright 2026 Vinay Williams

"""ToneControl: contrast/saturation/temperature adjustments in RGB/Lab space."""

from __future__ import annotations

import cv2
import numpy as np

from painting_assist.controls.adjust import ToneControl


def _gradient(size: int = 64) -> np.ndarray:
    """A horizontal grey gradient spanning 0..255, centred near mid-grey."""
    row = np.linspace(0, 255, size, dtype=np.uint8)
    grey = np.tile(row, (size, 1))
    return np.repeat(grey[:, :, None], 3, axis=2)


def _colour_img(size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_identity_at_all_zero():
    img = _colour_img()
    c = ToneControl()
    out = c.process(img.copy())
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_inactive_at_defaults():
    c = ToneControl()
    c.set_enabled(True)
    assert c.is_active() is False
    c.set("contrast", 0.5)
    assert c.is_active() is True
    c.set("contrast", 0.0)
    c.set("saturation", -0.5)
    assert c.is_active() is True
    c.set("saturation", 0.0)
    c.set("temperature", 0.3)
    assert c.is_active() is True
    # Disabled always inactive.
    c.set_enabled(False)
    assert c.is_active() is False


def test_contrast_increases_spread():
    img = _gradient()
    c = ToneControl()
    c.set("contrast", 0.8)
    out = c.process(img.copy())
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.std() > img.std()


def test_contrast_lut_is_monotonic():
    for contrast in (-1.0, -0.4, 0.4, 1.0):
        lut = ToneControl._contrast_lut(contrast)
        assert lut.shape == (256,)
        assert lut.dtype == np.uint8
        assert np.all(np.diff(lut.astype(np.int16)) >= 0)


def test_saturation_minus_one_is_greyscale():
    img = _colour_img()
    c = ToneControl()
    c.set("saturation", -1.0)
    out = c.process(img.copy())
    # R, G, B collapse to (near) equal -> greyscale.
    max_channel_spread = np.abs(
        out.astype(np.int16).max(axis=2) - out.astype(np.int16).min(axis=2)
    ).max()
    assert max_channel_spread <= 3


def test_temperature_warm_raises_mean_b():
    img = _colour_img()
    c = ToneControl()
    c.set("temperature", 0.8)
    out = c.process(img.copy())
    lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)
    lab_out = cv2.cvtColor(out, cv2.COLOR_RGB2Lab).astype(np.float32)
    assert lab_out[:, :, 2].mean() > lab_in[:, :, 2].mean()


def test_dtype_and_shape_preserved_for_combined():
    img = _colour_img()
    c = ToneControl()
    c.set("contrast", 0.5)
    c.set("saturation", 0.5)
    c.set("temperature", -0.5)
    out = c.process(img.copy())
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_does_not_mutate_input():
    img = _colour_img()
    original = img.copy()
    c = ToneControl()
    c.set("contrast", 0.6)
    c.set("saturation", 0.4)
    c.process(img)
    assert np.array_equal(img, original)
