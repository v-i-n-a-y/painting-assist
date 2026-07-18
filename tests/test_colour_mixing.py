# Copyright 2026 Vinay Williams

"""Tests for the colour-mixing helpers."""

from __future__ import annotations

import pytest

from painting_assist.colour_mixing import (
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


def test_deterministic():
    a = suggest_mix((123, 80, 200), "split_primary")
    b = suggest_mix((123, 80, 200), "split_primary")
    assert a == b


def test_unknown_palette_raises():
    with pytest.raises(KeyError):
        suggest_mix((0, 0, 0), "nope")


def test_describe_hex_formatting():
    assert describe_colour((255, 0, 0))["hex"] == "#ff0000"
    assert describe_colour((0, 0, 0))["hex"] == "#000000"
    assert describe_colour((16, 32, 48))["hex"] == "#102030"


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
