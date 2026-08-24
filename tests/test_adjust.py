# Copyright 2026 Vinay Williams

"""ToneControl: exposure/contrast/saturation/temperature in RGB/Lab space."""

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
    c.set("exposure", 0.4)
    assert c.is_active() is True
    c.set("exposure", 0.0)
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


def test_exposure_positive_brightens():
    img = _colour_img()
    c = ToneControl()
    c.set("exposure", 0.8)
    out = c.process(img.copy())
    lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)
    lab_out = cv2.cvtColor(out, cv2.COLOR_RGB2Lab).astype(np.float32)
    assert lab_out[:, :, 0].mean() > lab_in[:, :, 0].mean()


def test_exposure_negative_darkens():
    img = _colour_img()
    c = ToneControl()
    c.set("exposure", -0.8)
    out = c.process(img.copy())
    lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)
    lab_out = cv2.cvtColor(out, cv2.COLOR_RGB2Lab).astype(np.float32)
    assert lab_out[:, :, 0].mean() < lab_in[:, :, 0].mean()


def test_exposure_preserves_hue_and_chroma():
    img = _colour_img()
    c = ToneControl()
    c.set("exposure", 0.6)
    out = c.process(img.copy())
    lab_in = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)
    lab_out = cv2.cvtColor(out, cv2.COLOR_RGB2Lab).astype(np.float32)
    # Only L moves in Lab space, so a/b are preserved on every pixel where the
    # shift clipped nothing: L stayed in 0..255 AND no output RGB channel
    # saturated (a saturated channel shifts a/b on the RGB->Lab re-conversion).
    # Tolerance 2 covers the uint8 Lab<->RGB approximation.
    shift = 0.6 * ToneControl.EXPOSURE_L_SCALE
    safe = (lab_in[:, :, 0] + shift) <= 255.0
    safe &= (out.astype(np.int16) > 0).all(axis=2)
    safe &= (out.astype(np.int16) < 255).all(axis=2)
    assert safe.sum() > 100
    assert np.abs(lab_out[safe, 1] - lab_in[safe, 1]).max() <= 2
    assert np.abs(lab_out[safe, 2] - lab_in[safe, 2]).max() <= 2


def test_exposure_clips_at_extremes():
    # A bright grey pushed to +1.0 saturates at white...
    bright = np.full((16, 16, 3), 200, dtype=np.uint8)
    c = ToneControl()
    c.set("exposure", 1.0)
    out = c.process(bright.copy())
    assert out.min() == 255
    # ...and a dark grey pushed to -1.0 saturates at black.
    dark = np.full((16, 16, 3), 30, dtype=np.uint8)
    c2 = ToneControl()
    c2.set("exposure", -1.0)
    out2 = c2.process(dark.copy())
    assert out2.max() == 0


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
    c.set("exposure", 0.5)
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
