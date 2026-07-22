# Copyright 2026 Vinay Williams

"""Tests for the limited-palette / gamut simulation core (Qt-free)."""

from __future__ import annotations

import numpy as np

from painting_assist import palette_map


def _small_image(h: int = 12, w: int = 10) -> np.ndarray:
    """A deterministic HxWx3 uint8 RGB image."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _row_set(arr: np.ndarray) -> set[tuple[int, int, int]]:
    """The set of RGB rows present in an (N, 3) or (H, W, 3) array."""
    flat = arr.reshape(-1, 3)
    return {tuple(int(v) for v in row) for row in flat}


def test_build_gamut_single_tube_near_that_colour():
    tube = (200, 40, 60)
    gamut = palette_map.build_gamut([tube])
    assert gamut.dtype == np.uint8
    assert gamut.shape[1] == 3
    rows = _row_set(gamut)
    # The tube colour itself is present...
    assert tube in rows
    # ...and every candidate is close to it (a single tube can only mix with
    # itself, so there is no far-away colour).
    for row in gamut:
        assert np.max(np.abs(row.astype(int) - np.array(tube))) <= 4


def test_build_gamut_two_tubes_has_intermediate_mix():
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    gamut = palette_map.build_gamut([blue, yellow])
    rows = _row_set(gamut)
    # Endpoints are always present.
    assert blue in rows
    assert yellow in rows
    if palette_map._HAVE_MIXBOX:
        # At least one genuine intermediate mix that is neither endpoint.
        intermediates = [r for r in rows if r not in {blue, yellow}]
        assert intermediates, "expected a mixed colour between the two tubes"


def test_build_gamut_empty():
    gamut = palette_map.build_gamut([])
    assert gamut.shape == (0, 3)
    assert gamut.dtype == np.uint8


def test_build_gamut_respects_max_candidates_and_keeps_singles():
    tubes = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (30, 30, 30),
        (220, 220, 220),
    ]
    gamut = palette_map.build_gamut(tubes, max_candidates=20)
    assert len(gamut) <= 20
    rows = _row_set(gamut)
    if palette_map._HAVE_MIXBOX:
        # Single-tube colours must survive subsampling.
        for tube in tubes:
            assert tube in rows


def test_map_image_output_only_candidate_colours():
    img = _small_image()
    candidates = palette_map.build_gamut([(10, 20, 30), (240, 200, 50), (20, 120, 200)])
    out = palette_map.map_image(img, candidates)
    assert set(_row_set(out)).issubset(_row_set(candidates))


def test_map_image_empty_candidates_returns_unchanged():
    img = _small_image()
    out = palette_map.map_image(img, np.empty((0, 3), dtype=np.uint8))
    assert np.array_equal(out, img)


def test_map_image_does_not_mutate_input():
    img = _small_image()
    before = img.copy()
    candidates = palette_map.build_gamut([(10, 20, 30), (240, 200, 50)])
    palette_map.map_image(img, candidates)
    assert np.array_equal(img, before)


def test_map_image_shape_and_dtype_preserved():
    img = _small_image(9, 7)
    candidates = palette_map.build_gamut([(10, 20, 30), (240, 200, 50)])
    out = palette_map.map_image(img, candidates)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_simulate_deterministic():
    img = _small_image()
    tubes = [(0, 0, 255), (255, 255, 0), (200, 40, 60)]
    a = palette_map.simulate(img, tubes)
    b = palette_map.simulate(img, tubes)
    assert np.array_equal(a, b)


def test_zero_size_image_does_not_crash():
    img = np.empty((0, 0, 3), dtype=np.uint8)
    candidates = palette_map.build_gamut([(10, 20, 30), (240, 200, 50)])
    out = palette_map.map_image(img, candidates)
    assert out.shape == (0, 0, 3)
    assert out.dtype == np.uint8
