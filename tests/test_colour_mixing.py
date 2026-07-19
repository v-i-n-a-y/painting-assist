# Copyright 2026 Vinay Williams

"""Tests for the colour-mixing helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from painting_assist.colour_mixing import (
    _HUE_WHEEL,
    BASE_PALETTES,
    describe_colour,
    suggest_mix,
)


@pytest.mark.parametrize("palette_key", list(BASE_PALETTES))
@pytest.mark.parametrize(
    "target",
    [(200, 55, 45), (120, 130, 60), (30, 30, 30), (240, 240, 235)],
)
def test_mix_proportions_valid(palette_key, target):
    mix = suggest_mix(target, palette_key)
    assert mix, "expected at least one base"
    props = [p for _, p in mix]
    assert all(p >= 0.0 for p in props)
    assert sum(props) == pytest.approx(1.0)
    # Sorted descending.
    assert props == sorted(props, reverse=True)


def test_pure_red_dominant_on_zorn():
    mix = suggest_mix((255, 0, 0), "zorn")
    assert mix[0][0] == "Cadmium Red"


def test_default_palette_is_zorn():
    # palette_key defaults to "zorn", so the single-argument call must work.
    assert suggest_mix((255, 0, 0)) == suggest_mix((255, 0, 0), "zorn")


def test_deterministic():
    a = suggest_mix((123, 80, 200), "split_primary")
    b = suggest_mix((123, 80, 200), "split_primary")
    assert a == b


def test_unknown_palette_raises():
    with pytest.raises(KeyError):
        suggest_mix((0, 0, 0), "nope")


def test_near_black_resolves_to_black_on_zorn():
    # A near-black target should resolve to the black base, not a scaled mix.
    mix = suggest_mix((10, 10, 10), "zorn")
    assert mix[0][0] == "Ivory Black"


def test_suggest_mix_custom_bases():
    # A painter's own tubes: mixing a purple from red + blue (+ white).
    bases = [
        ("My Red", (220, 20, 30)),
        ("My Blue", (20, 30, 200)),
        ("My White", (250, 250, 245)),
    ]
    mix = suggest_mix((130, 20, 130), bases=bases)
    assert mix
    names = {name for name, _ in mix}
    assert names <= {"My Red", "My Blue", "My White"}
    props = [p for _, p in mix]
    assert all(p >= 0.0 for p in props)
    assert sum(props) == pytest.approx(1.0)
    assert props == sorted(props, reverse=True)
    # Both chromatic tubes should carry weight for a red-blue purple.
    assert {"My Red", "My Blue"} <= names


def test_custom_bases_override_palette_key():
    # When bases are given, palette_key is ignored (even an unknown one).
    bases = [("A", (10, 10, 10)), ("B", (250, 250, 245))]
    mix = suggest_mix((20, 20, 20), palette_key="does-not-exist", bases=bases)
    assert {name for name, _ in mix} <= {"A", "B"}
    assert mix[0][0] == "A"


def test_empty_bases_raises():
    with pytest.raises(ValueError):
        suggest_mix((0, 0, 0), bases=[])


def test_describe_hex_formatting():
    assert describe_colour((255, 0, 0))["hex"] == "#ff0000"
    assert describe_colour((0, 0, 0))["hex"] == "#000000"
    assert describe_colour((16, 32, 48))["hex"] == "#102030"


def test_value_is_lab_lightness():
    # Value is Lab L * 100 / 255, so pure red sits near 53, not the ~30 the old
    # Rec. 601 luma gave. Black is 0 and white is 100.
    assert describe_colour((0, 0, 0))["value"] == pytest.approx(0.0, abs=0.5)
    assert describe_colour((255, 255, 255))["value"] == pytest.approx(100.0, abs=0.5)
    red_value = describe_colour((255, 0, 0))["value"]
    assert red_value == pytest.approx(53.3, abs=1.5)
    assert red_value > 45.0  # clearly not the old ~30


def test_chroma_is_lab_quantity():
    # Lab chroma of a saturated primary is well above the old 0-100 ceiling,
    # while a near-grey has almost none.
    assert describe_colour((255, 0, 0))["chroma"] > 100.0
    assert describe_colour((128, 128, 128))["chroma"] < 1.0


def test_warm_vs_cool():
    assert describe_colour((220, 40, 30))["temperature"] == "warm"
    assert describe_colour((30, 60, 200))["temperature"] == "cool"


def test_near_grey_neutral():
    assert describe_colour((128, 127, 129))["temperature"] == "neutral"


def test_value_monotonic():
    black = describe_colour((0, 0, 0))["value"]
    grey = describe_colour((128, 128, 128))["value"]
    white = describe_colour((255, 255, 255))["value"]
    assert black < grey < white


def test_hue_name_in_wheel():
    d = describe_colour((250, 220, 20))
    assert d["hue_name"] in {"yellow", "yellow-orange"}


@pytest.mark.parametrize(
    "rgb, expected",
    [
        ((255, 0, 0), "pure"),  # saturated red, mid value -> pure
        ((245, 200, 200), "tint"),  # pale pink -> tint
        ((20, 25, 60), "shade"),  # dark navy -> shade
        ((120, 110, 70), "tone"),  # muted olive -> tone
        ((128, 127, 129), "tone"),  # near-grey -> tone
    ],
)
def test_modifier_sensible(rgb, expected):
    assert describe_colour(rgb)["modifier"] == expected


def test_hue_wheel_matches_opencv():
    # The hardcoded Lab hue angles in _HUE_WHEEL must still agree with what
    # OpenCV produces for the twelve canonical fully-saturated sRGB hues, so
    # naming stays honest across OpenCV versions.
    cv2 = pytest.importorskip("cv2")
    canonical = {
        "red": (255, 0, 0),
        "orange": (255, 128, 0),
        "yellow-orange": (255, 191, 0),
        "yellow": (255, 255, 0),
        "yellow-green": (128, 255, 0),
        "green": (0, 255, 0),
        "cyan": (0, 255, 255),
        "azure": (0, 128, 255),
        "blue": (0, 0, 255),
        "violet": (128, 0, 255),
        "magenta": (255, 0, 255),
        "rose": (255, 0, 128),
    }
    reference = dict(_HUE_WHEEL)
    assert set(canonical) == set(reference)
    for name, rgb in canonical.items():
        pixel = np.array([[list(rgb)]], dtype=np.uint8)
        _, a, b = cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab)[0, 0].astype(float)
        hue = math.degrees(math.atan2(b - 128.0, a - 128.0)) % 360.0
        diff = abs(hue - reference[name]) % 360.0
        dist = min(diff, 360.0 - diff)
        assert dist < 1.0, f"{name}: cv2 hue {hue:.1f} vs wheel {reference[name]:.1f}"
        # Each canonical hue should also name itself.
        assert describe_colour(rgb)["hue_name"] == name
