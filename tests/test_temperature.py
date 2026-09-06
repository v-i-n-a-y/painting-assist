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


def test_large_image_uses_proxy_and_correlates_with_small():
    # Above PROC_MAX_PX so process() must take the downscale-then-upscale
    # path. Build the large image by tiling a small patchwork so the proxy
    # (an area-averaged downscale) stays close to a plain resize of the
    # small original, letting us compare the two outputs directly.
    rng = np.random.default_rng(11)
    small = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    scale = 16  # 64*16 = 1024 -> 1024*1024 ~= 1.05M px, well above PROC_MAX_PX
    large = np.kron(small, np.ones((scale, scale, 1), dtype=np.uint8))
    assert large.shape[0] * large.shape[1] > TemperatureMapControl.PROC_MAX_PX

    out_large = _run(large)
    assert out_large.shape == large.shape
    assert out_large.dtype == np.uint8

    out_small = _run(small)

    # Downscale the large result back to the small image's size and compare;
    # a few Lab-uint8-roundtrip and resampling levels of slack are allowed.
    import cv2

    resized = cv2.resize(
        out_large, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA
    )
    diff = np.abs(resized.astype(np.int16) - out_small.astype(np.int16))
    assert diff.mean() < 6.0


def test_small_image_bypasses_proxy_unchanged():
    # At/below PROC_MAX_PX, no downscale should occur, so results must be
    # exactly reproducible run to run (same code path as before the proxy
    # was introduced).
    img = _colour_img(size=32)
    assert img.shape[0] * img.shape[1] <= TemperatureMapControl.PROC_MAX_PX
    out1 = _run(img)
    out2 = _run(img)
    assert np.array_equal(out1, out2)
