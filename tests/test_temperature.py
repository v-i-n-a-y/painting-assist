# Copyright 2026 Vinay Williams

"""TemperatureMapControl: a warm/cool false-colour diagnostic. Warm patches tint
orange (R>B), cool patches tint blue (B>R), the tint follows lightness for
legibility, and Strength cross-fades back to greyscale (R==G==B at 0). Headless
and in CIELab, so tolerances allow for the RGB<->Lab uint8 roundtrip."""

from __future__ import annotations

import numpy as np

from painting_assist.controls.temperature import TemperatureMapControl

WARM = (240, 150, 40)  # orange -> strongly warm (high b*)
COOL = (40, 110, 210)  # blue -> strongly cool (low b*)


def _filled(colour, size=24):
    return np.full((size, size, 3), colour, dtype=np.uint8)


def _colour_img(size=32):
    rng = np.random.default_rng(5)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _run(img, strength=100):
    c = TemperatureMapControl()
    c.set("strength", strength)
    return c.process(img)


def test_shape_and_dtype_preserved():
    img = _colour_img()
    out = _run(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_is_active_when_enabled():
    c = TemperatureMapControl()
    assert c.is_active() is False
    c.set_enabled(True)
    assert c.is_active() is True


def test_enabled_changes_a_colour_image():
    img = _colour_img()
    out = _run(img)
    assert not np.array_equal(out, img)


def test_warm_patch_tints_orange():
    out = _run(_filled(WARM))
    # Orange: the red channel dominates blue.
    assert out[:, :, 0].mean() > out[:, :, 2].mean()


def test_cool_patch_tints_blue():
    out = _run(_filled(COOL))
    # Blue: the blue channel dominates red.
    assert out[:, :, 2].mean() > out[:, :, 0].mean()


def test_strength_zero_is_greyscale():
    img = _colour_img()
    out = _run(img, strength=0)
    r = out[:, :, 0].astype(np.int16)
    g = out[:, :, 1].astype(np.int16)
    b = out[:, :, 2].astype(np.int16)
    # At strength 0 the three channels come from the same lightness value.
    assert np.max(np.abs(r - g)) <= 1
    assert np.max(np.abs(g - b)) <= 1


def test_determinism():
    img = _colour_img()
    out1 = _run(img.copy())
    out2 = _run(img.copy())
    assert np.array_equal(out1, out2)


def test_does_not_mutate_input():
    img = _colour_img()
    original = img.copy()
    _run(img)
    assert np.array_equal(img, original)
