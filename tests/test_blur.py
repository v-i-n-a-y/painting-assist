# Copyright 2026 Vinay Williams

"""BlurControl: ladder parsing, effective_radius per mode, and the process()
array contract (shape/dtype preserved, input not mutated)."""

from __future__ import annotations

import numpy as np

from painting_assist.controls.blur import BlurControl


def _img():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    img[:, :64] = 30
    img[:, 64:] = 220
    return np.ascontiguousarray(img)


def _noise_img():
    """A high-frequency image (per-pixel noise) so blur has detail to remove."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    return np.ascontiguousarray(img)


def _detail(img):
    """Mean absolute neighbour difference: a proxy for local variance/detail."""
    a = img.astype(np.int64)
    dx = np.abs(np.diff(a, axis=1)).mean()
    dy = np.abs(np.diff(a, axis=0)).mean()
    return float(dx + dy)


# ---- stage_levels ----
def test_stage_levels_even_spacing():
    c = BlurControl()
    c.set("spacing", "even")
    c.set("stage_count", 5)
    c.set("radius", 100)
    levels = c.stage_levels()
    assert len(levels) == 5
    assert levels[0] == 100 and levels[-1] == 0
    # Monotonic non-increasing ladder.
    assert all(levels[i] >= levels[i + 1] for i in range(len(levels) - 1))


def test_stage_levels_manual_pads_when_too_few():
    c = BlurControl()
    c.set("spacing", "manual")
    c.set("stage_count", 5)
    c.set("manual_values", "80, 40")
    levels = c.stage_levels()
    assert len(levels) == 5
    assert levels[:2] == [80, 40]
    assert levels[2:] == [0, 0, 0]  # padded with 0


def test_stage_levels_manual_truncates_when_too_many():
    c = BlurControl()
    c.set("spacing", "manual")
    c.set("stage_count", 3)
    c.set("manual_values", "90, 70, 50, 30, 10")
    levels = c.stage_levels()
    assert levels == [90, 70, 50]  # truncated to stage_count


def test_stage_levels_manual_clamps_out_of_range():
    c = BlurControl()
    c.set("spacing", "manual")
    c.set("stage_count", 3)
    c.set("manual_values", "500, -20, 33")
    levels = c.stage_levels()
    assert levels == [100, 0, 33]  # clamped to 0..100


def test_parse_manual_skips_non_finite_entries():
    # "1e999" -> inf, "-1e999" -> -inf, "nan" -> nan: all parse without a
    # ValueError but must not reach int(round(...)) (which raises on inf/nan).
    c = BlurControl()
    c.set("manual_values", "10, 1e999, -1e999, nan, 5")
    assert c._parse_manual() == [10, 5]


def test_stage_levels_manual_survives_non_finite_entry():
    c = BlurControl()
    c.set("spacing", "manual")
    c.set("stage_count", 3)
    c.set("manual_values", "10, 1e999, 5")
    levels = c.stage_levels()  # must not raise (OverflowError on inf)
    assert levels == [10, 5, 0]  # inf dropped, remainder padded to length


# ---- effective_radius ----
def test_effective_radius_continuous():
    c = BlurControl()
    c.set("mode", "continuous")
    c.set("radius", 42)
    assert c.effective_radius() == 42


def test_effective_radius_stepped_picks_current_stage():
    c = BlurControl()
    c.set("mode", "stepped")
    c.set("spacing", "manual")
    c.set("stage_count", 3)
    c.set("manual_values", "90, 60, 30")
    c.set("stage", 2)
    assert c.effective_radius() == 60


# ---- process contract ----
def test_process_preserves_shape_dtype_and_input():
    c = BlurControl()
    c.set("radius", 60)
    img = _img()
    original = img.copy()
    out = c.process(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(img, original)  # input not mutated


def test_process_radius_zero_returns_input_unchanged():
    c = BlurControl()
    c.set("radius", 0)
    img = _img()
    out = c.process(img)
    assert np.array_equal(out, img)


def test_process_reduces_local_detail():
    c = BlurControl()
    c.set("radius", 60)
    img = _noise_img()
    out = c.process(img)
    # A heavy blur collapses per-pixel noise into masses: local detail must
    # drop sharply (well below half the original).
    assert _detail(out) < 0.5 * _detail(img)


def test_process_is_deterministic():
    c = BlurControl()
    c.set("radius", 45)
    img = _noise_img()
    assert np.array_equal(c.process(img), c.process(img))


def test_process_small_radius_is_near_identity():
    c = BlurControl()
    c.set("radius", 5)  # 128 px -> pixel_radius 1 -> a mild 3x3 kernel
    img = _img()
    out = c.process(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    # A tiny kernel only softens the single seam, leaving the flat masses close
    # to the input on average.
    assert np.abs(out.astype(np.int64) - img.astype(np.int64)).mean() < 2.0
