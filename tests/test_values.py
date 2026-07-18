# Copyright 2026 Vinay Williams

"""ValuesControl: greyscale is neutral, value-steps posterize the lightness
channel, keep_colour preserves hue, and isolate dims out-of-band pixels. All
work is headless (no GUI) and in CIELab, so tolerances allow for the RGB<->Lab
uint8 roundtrip."""

from __future__ import annotations

import cv2
import numpy as np

from painting_assist.controls.values import ValuesControl


def _img():
    """A deterministic, colourful test image spanning the value range."""
    rng = np.random.default_rng(11)
    return rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)


def _lab_L(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2Lab)[:, :, 0]


def test_identity_shape_and_dtype():
    img = _img()
    out = ValuesControl().process(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_does_not_mutate_input():
    img = _img()
    original = img.copy()
    c = ValuesControl()
    c.set("mode", "posterize")
    c.process(img)
    assert np.array_equal(img, original)


def test_greyscale_channels_equal():
    img = _img()
    c = ValuesControl()
    c.set("mode", "grey")
    out = c.process(img)
    # A neutral grey has R == G == B (within Lab roundtrip tolerance).
    r = out[:, :, 0].astype(np.int16)
    g = out[:, :, 1].astype(np.int16)
    b = out[:, :, 2].astype(np.int16)
    assert np.max(np.abs(r - g)) <= 2
    assert np.max(np.abs(g - b)) <= 2


def test_posterize_limits_value_count():
    img = _img()
    c = ValuesControl()
    c.set("mode", "posterize")
    c.set("steps", 3)
    out = c.process(img)
    unique_L = np.unique(_lab_L(out))
    # At most `steps` distinct levels, plus a little slack for Lab roundtrip.
    assert len(unique_L) <= 3 + 2


def test_determinism():
    img = _img()
    c = ValuesControl()
    c.set("mode", "posterize")
    c.set("steps", 4)
    out1 = c.process(img.copy())
    out2 = c.process(img.copy())
    assert np.array_equal(out1, out2)


def test_keep_colour_differs_from_neutral_posterize():
    img = _img()
    neutral = ValuesControl()
    neutral.set("mode", "posterize")
    neutral.set("steps", 4)
    neutral.set("keep_colour", False)
    out_neutral = neutral.process(img)

    coloured = ValuesControl()
    coloured.set("mode", "posterize")
    coloured.set("steps", 4)
    coloured.set("keep_colour", True)
    out_colour = coloured.process(img)

    assert not np.array_equal(out_neutral, out_colour)


def test_isolate_dims_out_of_band_pixels():
    img = _img()
    base = ValuesControl()
    base.set("mode", "posterize")
    base.set("steps", 4)
    out_base = base.process(img)

    iso = ValuesControl()
    iso.set("mode", "posterize")
    iso.set("steps", 4)
    iso.set("isolate", 1)  # keep darkest band, dim the rest
    out_iso = iso.process(img)

    # Isolation must change the image (some pixels get dimmed toward grey).
    assert not np.array_equal(out_base, out_iso)

    # Dimmed pixels move closer to flat mid-grey (128) than they were.
    changed = np.any(out_base != out_iso, axis=2)
    assert changed.any()
    before = np.abs(out_base[changed].astype(np.int16) - 128)
    after = np.abs(out_iso[changed].astype(np.int16) - 128)
    assert after.mean() < before.mean()
