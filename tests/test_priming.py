# Copyright 2026 Vinay Williams

"""Tests for painting_assist.priming: majority-colour detection and the
technique -> ground-colour maths.

Pure maths, no Qt — mirrors how the other colour helpers are tested. All
colour assertions go through CIELab (the space the maths works in).
"""

from __future__ import annotations

import cv2
import numpy as np

from painting_assist.priming import (
    DESCRIPTIONS,
    TECHNIQUES,
    PrimeResult,
    majority_colour,
    prime_colour,
)


def _lab(rgb) -> np.ndarray:
    """8-bit CIELab of an RGB triple (L, a, b) as floats."""
    return cv2.cvtColor(np.uint8([[[int(c) for c in rgb]]]), cv2.COLOR_RGB2Lab)[
        0, 0
    ].astype(float)


def _solid(rgb, size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = rgb
    return img


def _two_colour(frac_red: float = 2.0 / 3.0, size: int = 64) -> np.ndarray:
    """Left ``frac_red`` of the width red, the rest blue."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    split = int(size * frac_red)
    img[:, :split] = (200, 30, 30)
    img[:, split:] = (30, 30, 200)
    return img


# ---- majority colour ----
def test_majority_solid_image():
    rgb = majority_colour(_solid((100, 150, 200)))
    assert rgb is not None
    assert all(abs(a - b) <= 2 for a, b in zip(rgb, (100, 150, 200)))


def test_majority_picks_the_dominant_side():
    rgb = majority_colour(_two_colour())
    assert rgb is not None
    # Two-thirds red: the dominant cluster must read red, not blue.
    assert rgb[0] > rgb[2]


def test_majority_none_and_empty():
    assert majority_colour(None) is None
    assert majority_colour(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_majority_is_deterministic():
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    assert majority_colour(img) == majority_colour(img)


def test_majority_downscales_large_images():
    # 800x600 forces the proxy path; a solid image survives the round trip.
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:, :] = (20, 90, 160)
    rgb = majority_colour(img)
    assert rgb is not None
    assert all(abs(a - b) <= 2 for a, b in zip(rgb, (20, 90, 160)))


# ---- technique maths ----
def test_prime_none_for_empty_image():
    assert prime_colour(np.zeros((0, 0, 3), dtype=np.uint8), "midtone") is None


def test_prime_result_shape_and_hex():
    result = prime_colour(_solid((120, 80, 60)), "midtone")
    assert isinstance(result, PrimeResult)
    assert len(result.rgb) == 3 and all(0 <= c <= 255 for c in result.rgb)
    assert result.hex == "#{:02X}{:02X}{:02X}".format(*result.rgb)
    assert result.majority_hex.startswith("#") and len(result.majority_hex) == 7
    assert result.technique == "midtone"


def test_prime_midtone_sits_at_mid_value():
    result = prime_colour(_solid((200, 30, 30)), "midtone", strength=80)
    assert abs(_lab(result.rgb)[0] - 128.0) <= 2


def test_prime_zero_strength_is_neutral_grey():
    # No chroma: a neutral grey at the technique's value target. (Note: 8-bit
    # Lab L=128 is ~50% lightness, which is grey ~119 in sRGB, not 128.)
    for technique, target in (
        ("midtone", 128.0),
        ("complement", 128.0),
        ("light", 220.0),
        ("dark", 40.0),
    ):
        result = prime_colour(_solid((200, 30, 30)), technique, strength=0)
        r, g, b = result.rgb
        assert abs(r - g) <= 1 and abs(g - b) <= 1, technique
        assert abs(_lab(result.rgb)[0] - target) <= 2, technique


def test_prime_majority_full_strength_keeps_the_colour():
    img = _solid((180, 90, 40))
    majority = majority_colour(img)
    result = prime_colour(img, "majority", strength=100)
    assert all(abs(a - b) <= 2 for a, b in zip(result.rgb, majority))


def test_prime_complement_mirrors_chroma():
    # A muted majority: its complement fits in gamut, so a/b are mirrored
    # across the neutral axis: (a-128) -> -(a-128).
    img = _solid((180, 120, 100))
    result = prime_colour(img, "complement", strength=100)
    a_in = _lab(result.majority)[1]
    a_out = _lab(result.rgb)[1]
    assert abs((a_out - 128.0) + (a_in - 128.0)) <= 2
    # A warm majority yields a cool ground (a below neutral).
    assert a_out < 128.0


def test_prime_complement_of_saturated_colour_is_desaturated_not_clipped():
    # A saturated red's exact complement is out of sRGB gamut at mid-value;
    # the fit must desaturate toward neutral instead of clipping channels
    # (clipping would lose the hue).
    result = prime_colour(_solid((200, 30, 30)), "complement", strength=100)
    a_out = _lab(result.rgb)[1]
    # Still clearly cyan-leaning, but pulled back from the out-of-gamut a~65.
    assert 128.0 > a_out > 90.0


def test_prime_light_and_dark_value_targets():
    img = _solid((140, 100, 60))
    light = prime_colour(img, "light", strength=60)
    dark = prime_colour(img, "dark", strength=60)
    assert abs(_lab(light.rgb)[0] - 220.0) <= 2
    assert abs(_lab(dark.rgb)[0] - 40.0) <= 2
    # The light ground must be lighter than the dark one.
    assert sum(light.rgb) > sum(dark.rgb)


def test_prime_neutral_is_mid_grey_regardless_of_strength():
    for strength in (0, 50, 100):
        result = prime_colour(_solid((200, 30, 30)), "neutral", strength=strength)
        r, g, b = result.rgb
        assert abs(r - g) <= 1 and abs(g - b) <= 1
        assert abs(_lab(result.rgb)[0] - 128.0) <= 2


def test_technique_choices_all_have_descriptions():
    for value, _label in TECHNIQUES:
        assert value in DESCRIPTIONS
