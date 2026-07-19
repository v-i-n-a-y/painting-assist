# Copyright 2026 Vinay Williams

"""WhiteBalanceControl: naming a patch that should be grey removes that colour
cast, landing the patch on the neutral Lab axis (a/b ~ 128). The default neutral
(128, 128, 128) is an exact no-op. Headless and in CIELab, so tolerances allow
for the RGB<->Lab uint8 roundtrip."""

from __future__ import annotations

import cv2
import numpy as np

from painting_assist.controls.white_balance import WhiteBalanceControl

WARM_GREY = (162, 150, 120)  # a grey with a clear yellow/warm cast (R>G>B)


def _filled(colour, size=24):
    return np.full((size, size, 3), colour, dtype=np.uint8)


def _colour_img(size=32):
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _lab_ab(rgb):
    lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2Lab)
    return lab[:, :, 1].astype(np.float32), lab[:, :, 2].astype(np.float32)


def _apply(img, neutral):
    c = WhiteBalanceControl()
    c.set("neutral_r", neutral[0])
    c.set("neutral_g", neutral[1])
    c.set("neutral_b", neutral[2])
    return c.process(img)


def test_shape_and_dtype_preserved():
    img = _colour_img()
    out = _apply(img, WARM_GREY)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_default_neutral_is_identity():
    img = _colour_img()
    out = WhiteBalanceControl().process(img.copy())
    # Default neutral (128, 128, 128) carries no cast: exact identity, no roundtrip.
    assert np.array_equal(out, img)


def test_picked_warm_grey_becomes_neutral():
    # An image that IS the warm-grey patch should correct to near-neutral a/b.
    img = _filled(WARM_GREY)
    before_a, before_b = _lab_ab(img)
    assert abs(before_a.mean() - 128) > 2 or abs(before_b.mean() - 128) > 2

    out = _apply(img, WARM_GREY)
    after_a, after_b = _lab_ab(out)
    assert abs(after_a.mean() - 128) <= 2
    assert abs(after_b.mean() - 128) <= 2


def test_warm_neutral_cools_the_image():
    # Correcting a warm (yellow) cast subtracts b*, cooling every pixel, so the
    # mean b* of a full colour image drops.
    img = _colour_img()
    out = _apply(img, WARM_GREY)
    _, in_b = _lab_ab(img)
    _, out_b = _lab_ab(out)
    assert out_b.mean() < in_b.mean()


def test_is_active_requires_a_real_cast():
    c = WhiteBalanceControl()
    c.set_enabled(True)
    # Default mid-grey neutral -> no cast -> inactive.
    assert c.is_active() is False
    # A lighter neutral grey is still castless (a/b stay at 128) -> inactive.
    c.set("neutral_r", 200)
    c.set("neutral_g", 200)
    c.set("neutral_b", 200)
    assert c.is_active() is False
    # A warm grey carries a cast -> active.
    c.set("neutral_r", WARM_GREY[0])
    c.set("neutral_g", WARM_GREY[1])
    c.set("neutral_b", WARM_GREY[2])
    assert c.is_active() is True
    # Disabled is always inactive.
    c.set_enabled(False)
    assert c.is_active() is False


def test_determinism():
    img = _colour_img()
    out1 = _apply(img.copy(), WARM_GREY)
    out2 = _apply(img.copy(), WARM_GREY)
    assert np.array_equal(out1, out2)


def test_does_not_mutate_input():
    img = _colour_img()
    original = img.copy()
    _apply(img, WARM_GREY)
    assert np.array_equal(img, original)
