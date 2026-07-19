# Copyright 2026 Vinay Williams

"""Tests for the pigment-mixing engine and paint-matching suggestions."""

from __future__ import annotations

import pytest

from painting_assist.mixing import (
    MIXBOX_AVAILABLE,
    best_mix,
    deltae,
    suggest,
    tolerance_deltae,
)
from painting_assist.paints import DEFAULT_CATALOGUE

_CATALOGUE = dict(DEFAULT_CATALOGUE)


# --- deltae ----------------------------------------------------------------


def test_deltae_identical_is_zero():
    assert deltae((10, 20, 30), (10, 20, 30)) == 0.0
    assert deltae((255, 255, 255), (255, 255, 255)) == 0.0


def test_deltae_symmetric():
    assert deltae((200, 40, 40), (40, 50, 130)) == deltae((40, 50, 130), (200, 40, 40))


def test_deltae_large_for_clearly_different():
    # Black against white spans nearly the whole lightness axis.
    assert deltae((0, 0, 0), (255, 255, 255)) > 90.0


def test_deltae_positive():
    assert deltae((10, 20, 30), (40, 50, 60)) > 0.0


# --- mixbox engine sanity --------------------------------------------------


def test_mixbox_is_available():
    # mixbox is a declared dependency, so the pigment-mixing path is the default.
    assert MIXBOX_AVAILABLE is True


def test_mixbox_blue_and_yellow_make_green():
    # Proof the engine mixes pigment, not light: a convex mix of blue and yellow
    # latents lands green (additive RGB averaging would give a muddy grey).
    mixbox = pytest.importorskip("mixbox")
    blue = mixbox.rgb_to_latent((20, 30, 200))
    yellow = mixbox.rgb_to_latent((250, 230, 20))
    mixed = [0.5 * bl + 0.5 * ye for bl, ye in zip(blue, yellow)]
    r, g, b = mixbox.latent_to_rgb(mixed)
    assert g > r and g > b, f"expected green, got {(r, g, b)}"


# --- best_mix --------------------------------------------------------------


def test_best_mix_exact_tube_returns_that_tube():
    tubes = [
        ("White", (250, 250, 245)),
        ("Ultramarine", (40, 50, 130)),
        ("Cad Yellow", (255, 199, 27)),
    ]
    recipe, mixed_rgb, error = best_mix((40, 50, 130), tubes)
    assert recipe[0][0] == "Ultramarine"
    assert recipe[0][1] == pytest.approx(1.0, abs=1e-6)
    assert error < 1.0
    assert mixed_rgb == (40, 50, 130)


def test_best_mix_between_two_tubes_blends_both():
    tubes = [
        ("Ultramarine", (40, 50, 130)),
        ("Cad Yellow", (255, 199, 27)),
    ]
    recipe, mixed_rgb, error = best_mix((70, 120, 60), tubes)
    names = {name for name, _ in recipe}
    assert names == {"Ultramarine", "Cad Yellow"}
    # The blend is genuinely green (pigment mixing), and close to the target.
    r, g, b = mixed_rgb
    assert g > r and g > b
    assert error < 15.0


def test_best_mix_recipe_is_valid():
    tubes = [
        ("White", (250, 250, 245)),
        ("Ultramarine", (40, 50, 130)),
        ("Cad Yellow", (255, 199, 27)),
        ("Cad Red", (200, 40, 40)),
    ]
    recipe, _mixed, _error = best_mix((120, 90, 70), tubes)
    props = [p for _, p in recipe]
    assert props, "expected at least one tube"
    assert all(p >= 0.0 for p in props)
    assert sum(props) == pytest.approx(1.0)
    assert props == sorted(props, reverse=True)


def test_best_mix_error_matches_deltae():
    tubes = [("Ultramarine", (40, 50, 130)), ("Cad Yellow", (255, 199, 27))]
    _recipe, mixed_rgb, error = best_mix((70, 120, 60), tubes)
    assert error == pytest.approx(deltae(mixed_rgb, (70, 120, 60)))


def test_best_mix_empty_tubes():
    recipe, mixed_rgb, error = best_mix((123, 45, 67), [])
    assert recipe == []
    assert mixed_rgb == (123, 45, 67)
    assert error == 0.0


# --- tolerance mapping -----------------------------------------------------


def test_tolerance_deltae_endpoints():
    assert tolerance_deltae(0) == 0.0
    assert tolerance_deltae(25) == pytest.approx(10.0)
    assert tolerance_deltae(100) == pytest.approx(40.0)


def test_tolerance_deltae_clamps_out_of_range():
    assert tolerance_deltae(-10) == 0.0
    assert tolerance_deltae(200) == pytest.approx(40.0)


# --- suggest ---------------------------------------------------------------


def test_suggest_within_tolerance():
    tubes = [
        ("White", (250, 250, 245)),
        ("Ultramarine", (40, 50, 130)),
        ("Cad Yellow", (255, 199, 27)),
    ]
    result = suggest((70, 120, 60), tubes, tolerance_pct=25, on_miss="closest")
    assert result.within_tolerance is True
    assert result.buy is None
    assert result.recipe
    assert "tolerance" in result.message.lower()


def test_suggest_closest_miss_reports_gap():
    tubes = [("White", (250, 250, 245)), ("Black", (32, 30, 30))]
    result = suggest((0, 255, 255), tubes, tolerance_pct=25, on_miss="closest")
    assert result.within_tolerance is False
    assert result.buy is None
    assert result.recipe  # closest mix is still offered as a swatch
    assert "cannot be matched" in result.message


def test_suggest_buy_names_a_chromatic_paint():
    # An impoverished palette (white + black) cannot reach a saturated blue, but
    # adding the matching catalogue paint can.
    tubes = [("White", (250, 250, 245)), ("Black", (32, 30, 30))]
    target = _CATALOGUE["Ultramarine Blue"]
    result = suggest(target, tubes, tolerance_pct=25, on_miss="buy")
    assert result.buy == "Ultramarine Blue"
    assert result.within_tolerance is False
    assert "Ultramarine Blue" in result.message
    # The closest current-tube mix is still filled in for the UI swatch.
    assert result.recipe


def test_suggest_buy_reports_out_of_gamut():
    tubes = [("White", (250, 250, 245)), ("Black", (32, 30, 30))]
    result = suggest((0, 255, 255), tubes, tolerance_pct=25, on_miss="buy")
    assert result.buy is None
    assert result.within_tolerance is False
    assert "gamut" in result.message


def test_suggest_buy_skips_owned_paints():
    # A tube already owning a catalogue name (case-insensitively) is never the
    # buy suggestion; the suggested paint is genuinely new.
    tubes = [("White", (250, 250, 245)), ("ultramarine blue", (40, 50, 130))]
    target = _CATALOGUE["Cadmium Red"]
    result = suggest(target, tubes, tolerance_pct=25, on_miss="buy")
    if result.buy is not None:
        owned = {name.lower() for name, _ in tubes}
        assert result.buy.lower() not in owned


def test_suggest_no_tubes():
    result = suggest((10, 20, 30), [], on_miss="buy")
    assert result.recipe == []
    assert result.within_tolerance is False
    assert result.buy is None
    assert result.message


def test_suggest_defaults_to_catalogue():
    # Omitting the catalogue argument falls back to DEFAULT_CATALOGUE.
    tubes = [("White", (250, 250, 245)), ("Black", (32, 30, 30))]
    target = _CATALOGUE["Ultramarine Blue"]
    result = suggest(target, tubes, tolerance_pct=25, on_miss="buy")
    assert result.buy == "Ultramarine Blue"
