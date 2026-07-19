# Copyright 2026 Vinay Williams

"""GridControl is now a non-destructive viewer overlay: process() is the
identity and is_active() is always False. The pixel-drawing routine lives in the
module-level draw_grid(); overlay_spec() resolves the params (incl. the colour
preset RGB) for the viewer."""

from __future__ import annotations

import numpy as np
import pytest

from painting_assist.controls.grid import GridControl, draw_grid, _COLORS


def _img():
    rng = np.random.default_rng(7)
    return np.ascontiguousarray(
        rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    )


# ---- control is a no-op in the pipeline ----
def test_process_is_identity():
    c = GridControl()
    c.set_enabled(True)
    c.set("columns", 6)
    c.set("rows", 5)
    img = _img()
    out = c.process(img)
    assert out is img  # unchanged array returned as-is


def test_is_active_always_false():
    c = GridControl()
    c.set_enabled(True)
    c.set("columns", 8)
    c.set("diagonals", True)
    assert c.is_active() is False
    c.set_enabled(False)
    assert c.is_active() is False


# ---- overlay_spec resolution ----
def test_overlay_spec_resolves_color_and_opacity():
    c = GridControl()
    c.set_enabled(True)
    c.set("color", "cyan")
    c.set("opacity", 50)
    c.set("columns", 4)
    c.set("rows", 3)
    c.set("thickness", 3)
    c.set("diagonals", True)
    spec = c.overlay_spec()
    assert spec["color_rgb"] == _COLORS["cyan"]
    assert abs(spec["opacity"] - 0.5) < 1e-9
    assert spec["columns"] == 4 and spec["rows"] == 3
    assert spec["thickness"] == 3
    assert spec["diagonals"] is True
    assert spec["visible"] is True


def test_overlay_spec_visible_reflects_enabled_and_content():
    c = GridControl()
    # Disabled -> not visible even with divisions.
    c.set("columns", 4)
    assert c.overlay_spec()["visible"] is False
    # Enabled but nothing to draw (1x1, no diagonals) -> not visible.
    c.set_enabled(True)
    c.set("columns", 1)
    c.set("rows", 1)
    c.set("diagonals", False)
    assert c.overlay_spec()["visible"] is False
    # Enabled with a diagonal only -> visible.
    c.set("diagonals", True)
    assert c.overlay_spec()["visible"] is True


# ---- draw_grid free function ----
def test_draw_grid_changes_pixels_and_preserves_shape_dtype():
    img = _img()
    out = draw_grid(
        img,
        columns=5,
        rows=4,
        color_rgb=(255, 0, 0),
        opacity=1.0,
        thickness=2,
        diagonals=False,
    )
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert not np.array_equal(out, img)  # lines were drawn


def test_draw_grid_does_not_mutate_input():
    img = _img()
    before = img.copy()
    draw_grid(
        img,
        columns=6,
        rows=6,
        color_rgb=(0, 255, 0),
        opacity=0.7,
        thickness=3,
        diagonals=True,
    )
    assert np.array_equal(img, before)


def test_draw_grid_diagonals_flag_draws_more():
    img = _img()
    plain = draw_grid(
        img,
        columns=1,
        rows=1,
        color_rgb=(255, 255, 255),
        opacity=1.0,
        thickness=2,
        diagonals=False,
    )
    # With 1x1 and no diagonals, nothing is drawn.
    assert np.array_equal(plain, img)
    diag = draw_grid(
        img,
        columns=1,
        rows=1,
        color_rgb=(255, 255, 255),
        opacity=1.0,
        thickness=2,
        diagonals=True,
    )
    assert not np.array_equal(diag, img)


# ---- layout presets: overlay_spec fractions ----
def test_overlay_spec_layout_even_fractions_from_counts():
    c = GridControl()
    c.set("layout", "even")
    c.set("columns", 4)
    c.set("rows", 2)
    spec = c.overlay_spec()
    assert spec["layout"] == "even"
    assert spec["x_fractions"] == pytest.approx([0.25, 0.5, 0.75])
    assert spec["y_fractions"] == pytest.approx([0.5])
    assert spec["diagonal_lines"] == []


def test_overlay_spec_layout_thirds():
    c = GridControl()
    c.set("layout", "thirds")
    spec = c.overlay_spec()
    assert spec["x_fractions"] == pytest.approx([1 / 3, 2 / 3])
    assert spec["y_fractions"] == pytest.approx([1 / 3, 2 / 3])
    assert spec["diagonal_lines"] == []


def test_overlay_spec_layout_golden():
    c = GridControl()
    c.set("layout", "golden")
    spec = c.overlay_spec()
    assert spec["x_fractions"] == pytest.approx([0.382, 0.618])
    assert spec["y_fractions"] == pytest.approx([0.382, 0.618])


def test_overlay_spec_layout_quarters():
    c = GridControl()
    c.set("layout", "quarters")
    spec = c.overlay_spec()
    assert spec["x_fractions"] == pytest.approx([0.25, 0.5, 0.75])
    assert spec["y_fractions"] == pytest.approx([0.25, 0.5, 0.75])


def test_overlay_spec_layout_armature_has_diagonal_segments():
    c = GridControl()
    c.set("layout", "diagonals-armature")
    spec = c.overlay_spec()
    # The armature is purely diagonal: no axis-aligned lines.
    assert spec["x_fractions"] == []
    assert spec["y_fractions"] == []
    segments = spec["diagonal_lines"]
    assert len(segments) == 10  # two main diagonals + eight reciprocals
    for p0, p1 in segments:
        for fx, fy in (p0, p1):
            assert 0.0 <= fx <= 1.0
            assert 0.0 <= fy <= 1.0


def test_overlay_spec_ratio_layout_visible_without_counts():
    # A ratio layout draws its guides even at 1x1 with diagonals off...
    c = GridControl()
    c.set_enabled(True)
    c.set("columns", 1)
    c.set("rows", 1)
    c.set("diagonals", False)
    c.set("layout", "thirds")
    assert c.overlay_spec()["visible"] is True
    # ...whereas an empty even layout resolves to nothing to draw.
    c.set("layout", "even")
    assert c.overlay_spec()["visible"] is False


def test_unknown_layout_falls_back_to_even():
    # An invalid choice is clamped back to the default ("even").
    c = GridControl()
    c.set("columns", 3)
    c.set("rows", 1)
    c.set("layout", "bogus")
    spec = c.overlay_spec()
    assert spec["layout"] == "even"
    assert spec["x_fractions"] == pytest.approx([1 / 3, 2 / 3])


# ---- draw_grid honours the layout ----
def test_draw_grid_honours_non_even_layout():
    img = _img()
    # An even 1x1 grid with no diagonals draws nothing...
    even = draw_grid(
        img,
        columns=1,
        rows=1,
        color_rgb=(255, 0, 0),
        opacity=1.0,
        thickness=2,
        diagonals=False,
        layout="even",
    )
    assert np.array_equal(even, img)
    # ...but a thirds layout draws its guides regardless of columns/rows.
    thirds = draw_grid(
        img,
        columns=1,
        rows=1,
        color_rgb=(255, 0, 0),
        opacity=1.0,
        thickness=2,
        diagonals=False,
        layout="thirds",
    )
    assert thirds.shape == img.shape
    assert thirds.dtype == img.dtype
    assert not np.array_equal(thirds, img)


def test_draw_grid_armature_layout_draws_lines():
    img = _img()
    out = draw_grid(
        img,
        columns=1,
        rows=1,
        color_rgb=(255, 255, 255),
        opacity=1.0,
        thickness=2,
        diagonals=False,
        layout="diagonals-armature",
    )
    assert not np.array_equal(out, img)


def test_draw_grid_layout_does_not_mutate_input():
    img = _img()
    before = img.copy()
    draw_grid(
        img,
        columns=3,
        rows=3,
        color_rgb=(0, 255, 0),
        opacity=0.6,
        thickness=2,
        diagonals=False,
        layout="golden",
    )
    assert np.array_equal(img, before)


def test_draw_grid_default_layout_matches_explicit_even():
    # Omitting layout reproduces the original even-grid behaviour (compat).
    img = _img()
    explicit = draw_grid(
        img,
        columns=5,
        rows=4,
        color_rgb=(255, 0, 0),
        opacity=1.0,
        thickness=2,
        diagonals=False,
        layout="even",
    )
    default = draw_grid(
        img,
        columns=5,
        rows=4,
        color_rgb=(255, 0, 0),
        opacity=1.0,
        thickness=2,
        diagonals=False,
    )
    assert np.array_equal(explicit, default)
