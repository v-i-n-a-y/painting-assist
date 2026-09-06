# Copyright 2026 Vinay Williams

"""Headless tests for the value-histogram maths (no QApplication / no QWidget).

Only the pure functions are exercised; the QWidget is never instantiated
because a headless Qt here cannot create a window.
"""

from __future__ import annotations

import numpy as np

from painting_assist.widgets.value_histogram import (
    _lab_l_channel,
    _value_histogram_from_l,
    _value_mass_split_from_l,
    value_histogram,
    value_mass_split,
)


def _solid(rgb, h: int = 8, w: int = 8) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :] = rgb
    return img


def test_sums_to_pixel_count() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(12, 20, 3), dtype=np.uint8)
    hist = value_histogram(img, bins=16)
    assert int(hist.sum()) == 12 * 20


def test_returns_ndarray_of_bin_count() -> None:
    img = _solid((120, 120, 120))
    for bins in (4, 8, 16, 32):
        hist = value_histogram(img, bins=bins)
        assert isinstance(hist, np.ndarray)
        assert hist.shape == (bins,)


def test_black_lands_in_lowest_bin() -> None:
    hist = value_histogram(_solid((0, 0, 0)), bins=16)
    assert hist[0] == 64
    assert int(hist[1:].sum()) == 0


def test_white_lands_in_highest_bin() -> None:
    hist = value_histogram(_solid((255, 255, 255)), bins=16)
    assert hist[-1] == 64
    assert int(hist[:-1].sum()) == 0


def test_deterministic() -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    first = value_histogram(img, bins=16)
    second = value_histogram(img, bins=16)
    assert np.array_equal(first, second)


def test_mass_split_black_is_dark_white_is_light() -> None:
    assert value_mass_split(_solid((0, 0, 0))) == (1.0, 0.0, 0.0)
    assert value_mass_split(_solid((255, 255, 255))) == (0.0, 0.0, 1.0)


def test_mass_split_sums_to_one() -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(9, 11, 3), dtype=np.uint8)
    dark, mid, light = value_mass_split(img)
    assert abs((dark + mid + light) - 1.0) < 1e-9


def test_shared_l_helper_matches_two_conversion_path() -> None:
    # The widget refresh now converts to Lab-L once and feeds both the
    # histogram and the mass split from that single array. Confirm this
    # shared-conversion path gives identical results to calling the public,
    # independently-converting functions.
    rng = np.random.default_rng(9)
    img = rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8)

    lab_l = _lab_l_channel(img)
    shared_hist = _value_histogram_from_l(lab_l, bins=16)
    shared_split = _value_mass_split_from_l(lab_l)

    direct_hist = value_histogram(img, bins=16)
    direct_split = value_mass_split(img)

    assert np.array_equal(shared_hist, direct_hist)
    for a, b in zip(shared_split, direct_split):
        assert abs(a - b) < 1e-6
