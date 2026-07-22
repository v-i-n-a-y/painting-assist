# Copyright 2026 Vinay Williams

"""Tests for the pure PDF tiling maths in :mod:`painting_assist.print_layout`."""

from __future__ import annotations

from painting_assist.print_layout import Tile, tile_grid

# A4 in mm and a typical margin, shared by several cases.
A4_W, A4_H = 210.0, 297.0
MARGIN = 10.0
PRINTABLE_W = A4_W - 2 * MARGIN  # 190 mm
PRINTABLE_H = A4_H - 2 * MARGIN  # 277 mm


def test_small_canvas_single_tile():
    """A canvas smaller than the printable area fits on exactly one page."""
    tiles = tile_grid(1000, 800, 100.0, 80.0, A4_W, A4_H, MARGIN)
    assert len(tiles) == 1
    tile = tiles[0]
    assert (tile.col, tile.row) == (0, 0)
    assert (tile.src_x, tile.src_y, tile.src_w, tile.src_h) == (0, 0, 1000, 800)
    assert tile.dst_mm_w == 100.0
    assert tile.dst_mm_h == 80.0


def test_two_and_a_half_pages_wide_gives_three_columns():
    """A canvas 2.5x the printable width needs three columns of pages."""
    canvas_w = PRINTABLE_W * 2.5
    tiles = tile_grid(2500, 400, canvas_w, PRINTABLE_H, A4_W, A4_H, MARGIN)
    cols = max(t.col for t in tiles) + 1
    rows = max(t.row for t in tiles) + 1
    assert cols == 3
    assert rows == 1


def test_source_rects_tile_full_image_without_gaps_or_overlaps():
    """Source rects partition the whole image exactly, no gaps or overlaps."""
    image_w, image_h = 3333, 2222
    canvas_w = PRINTABLE_W * 2.3
    canvas_h = PRINTABLE_H * 1.7
    tiles = tile_grid(image_w, image_h, canvas_w, canvas_h, A4_W, A4_H, MARGIN)
    cols = max(t.col for t in tiles) + 1
    rows = max(t.row for t in tiles) + 1

    # Column boundaries: each row repeats the same x partition; check the top row.
    top_row = sorted((t for t in tiles if t.row == 0), key=lambda t: t.col)
    x = 0
    for t in top_row:
        assert t.src_x == x
        assert t.src_w > 0
        x += t.src_w
    assert x == image_w

    # Row boundaries: each column repeats the same y partition; check first column.
    first_col = sorted((t for t in tiles if t.col == 0), key=lambda t: t.row)
    y = 0
    for t in first_col:
        assert t.src_y == y
        assert t.src_h > 0
        y += t.src_h
    assert y == image_h

    # Total covered pixel area equals the image area (no overlaps).
    covered = sum(t.src_w * t.src_h for t in tiles)
    assert covered == image_w * image_h
    assert len(tiles) == cols * rows


def test_physical_scale_is_true_within_rounding():
    """Summed destination mm equals the canvas size (pixel rounding aside)."""
    canvas_w = PRINTABLE_W * 2.5
    canvas_h = PRINTABLE_H * 1.5
    tiles = tile_grid(2000, 1500, canvas_w, canvas_h, A4_W, A4_H, MARGIN)
    top_row = [t for t in tiles if t.row == 0]
    first_col = [t for t in tiles if t.col == 0]
    assert abs(sum(t.dst_mm_w for t in top_row) - canvas_w) < 0.5
    assert abs(sum(t.dst_mm_h for t in first_col) - canvas_h) < 0.5


def test_degenerate_inputs_do_not_crash():
    """Zero/negative dimensions fall back to a single tile rather than raising."""
    for args in (
        (0, 0, 100.0, 100.0, A4_W, A4_H, MARGIN),
        (1000, 800, 0.0, 80.0, A4_W, A4_H, MARGIN),
        (1000, 800, -5.0, 80.0, A4_W, A4_H, MARGIN),
        (1000, 800, 100.0, 80.0, A4_W, A4_H, 200.0),  # margin swallows the page
    ):
        tiles = tile_grid(*args)
        assert len(tiles) == 1
        assert isinstance(tiles[0], Tile)
