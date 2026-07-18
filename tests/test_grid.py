# Copyright 2026 Vinay Williams

"""GridControl is now a non-destructive viewer overlay: process() is the
identity and is_active() is always False. The pixel-drawing routine lives in the
module-level draw_grid(); overlay_spec() resolves the params (incl. the colour
preset RGB) for the viewer."""

from __future__ import annotations

import numpy as np

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
