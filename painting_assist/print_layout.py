# Copyright 2026 Vinay Williams

"""Pure tiling maths for the "Export Gridded Reference to PDF" feature.

Qt-free by design so the layout arithmetic can be unit-tested without a running
application. The Qt rendering (painting each tile onto a ``QPdfWriter`` page)
lives in the main window and simply consumes the tiles produced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Tile:
    """One page's worth of the gridded reference.

    ``col``/``row`` are the tile's position in the page grid (0-based).
    ``src_x``/``src_y``/``src_w``/``src_h`` describe the sub-region of the source
    image, in image pixels, that this page shows. ``dst_mm_w``/``dst_mm_h`` are
    the true physical size that sub-region must occupy on paper (millimetres), so
    that 1 mm on the printed page equals 1 mm of the real canvas.
    """

    col: int
    row: int
    src_x: int
    src_y: int
    src_w: int
    src_h: int
    dst_mm_w: float
    dst_mm_h: float


def _axis_boundaries(image_px: int, canvas_mm: float, printable_mm: float) -> List[int]:
    """Return pixel boundaries splitting ``image_px`` into printable-sized bands.

    The returned list has ``count + 1`` entries running from ``0`` to
    ``image_px`` with no gaps or overlaps, where ``count`` is the number of pages
    needed along this axis. Boundaries are placed by mapping each millimetre band
    edge into pixel space and rounding, then pinning the final edge exactly to
    ``image_px`` so the bands always cover the whole image.
    """
    count = max(1, int(math.ceil(canvas_mm / printable_mm)))
    px_per_mm = image_px / canvas_mm
    boundaries = [0]
    for i in range(1, count):
        edge = int(round(i * printable_mm * px_per_mm))
        edge = max(0, min(image_px, edge))
        boundaries.append(edge)
    boundaries.append(image_px)
    return boundaries


def tile_grid(
    image_w_px: int,
    image_h_px: int,
    canvas_mm_w: float,
    canvas_mm_h: float,
    page_mm_w: float,
    page_mm_h: float,
    margin_mm: float,
) -> List[Tile]:
    """Tile a gridded reference across pages at true physical scale.

    The source image (``image_w_px`` x ``image_h_px``) represents a physical
    canvas of ``canvas_mm_w`` x ``canvas_mm_h`` millimetres. The printable area
    of a page is ``page_mm_w`` x ``page_mm_h`` minus ``margin_mm`` on every side.
    The image is split into a grid of pages, each carrying the source pixel rect
    it shows and the true physical size (mm) that rect must occupy on paper.

    Degenerate inputs (any non-positive dimension) never raise: they fall back to
    a single tile covering the whole image, so the caller can still emit one page.
    """
    printable_mm_w = page_mm_w - 2.0 * margin_mm
    printable_mm_h = page_mm_h - 2.0 * margin_mm
    if (
        image_w_px <= 0
        or image_h_px <= 0
        or canvas_mm_w <= 0
        or canvas_mm_h <= 0
        or printable_mm_w <= 0
        or printable_mm_h <= 0
    ):
        return [
            Tile(
                col=0,
                row=0,
                src_x=0,
                src_y=0,
                src_w=max(0, int(image_w_px)),
                src_h=max(0, int(image_h_px)),
                dst_mm_w=max(0.0, float(canvas_mm_w)),
                dst_mm_h=max(0.0, float(canvas_mm_h)),
            )
        ]

    x_bounds = _axis_boundaries(int(image_w_px), canvas_mm_w, printable_mm_w)
    y_bounds = _axis_boundaries(int(image_h_px), canvas_mm_h, printable_mm_h)
    px_per_mm_x = image_w_px / canvas_mm_w
    px_per_mm_y = image_h_px / canvas_mm_h

    tiles: List[Tile] = []
    for row in range(len(y_bounds) - 1):
        y0, y1 = y_bounds[row], y_bounds[row + 1]
        for col in range(len(x_bounds) - 1):
            x0, x1 = x_bounds[col], x_bounds[col + 1]
            tiles.append(
                Tile(
                    col=col,
                    row=row,
                    src_x=x0,
                    src_y=y0,
                    src_w=x1 - x0,
                    src_h=y1 - y0,
                    dst_mm_w=(x1 - x0) / px_per_mm_x,
                    dst_mm_h=(y1 - y0) / px_per_mm_y,
                )
            )
    return tiles
