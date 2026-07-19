# Copyright 2026 Vinay Williams

"""Unit conversion and canvas calibration for the measuring overlays.

Qt-free so the maths is unit-testable without a GUI (like the geometry helpers
in :mod:`painting_assist.widgets.measure_items`). A :class:`Calibration` maps
scene geometry -- lengths and points expressed in *image pixels* -- to physical
canvas units, using the canvas size entered in the Canvas & Crop control plus
the pixel size of the displayed image.

The physical canvas width/height are taken as absolute (e.g. 40 x 30 cm), so a
scene length of ``L`` image-pixels across an ``image_w``-pixel-wide image is
``L * canvas_w / image_w`` canvas units. Horizontal and vertical scales are kept
separate so a freeform (unlocked) crop, whose pixel aspect need not match the
canvas, still measures correctly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# Centimetres per one unit of each supported physical length unit.
_CM_PER = {"cm": 1.0, "mm": 0.1, "in": 2.54}

# Length units the calipers/guides can display in. "px" is image pixels; the
# physical units need a physical canvas size to be meaningful.
DISPLAY_UNITS = ("cm", "mm", "in", "px")


def convert_from_cm(value_cm: float, unit: str) -> float:
    """Convert a length in centimetres to ``unit`` (cm/mm/in); passthrough else."""
    per = _CM_PER.get(unit)
    if per is None:
        return value_cm
    return value_cm / per


def format_measure(value: float, unit: str) -> str:
    """Format a measured length with its unit, e.g. ``"12.3 cm"`` or ``"120 px"``."""
    if unit == "px":
        return f"{value:.0f} px"
    if unit == "in":
        return f"{value:.2f} in"
    if unit == "mm":
        return f"{value:.0f} mm"
    return f"{value:.1f} cm"


@dataclass(frozen=True)
class Calibration:
    """Maps scene (image-pixel) geometry to physical canvas units.

    ``canvas_w``/``canvas_h`` are the physical canvas size in ``canvas_unit``
    (cm/mm/in, or px/ratio when no physical size is set). ``display_unit`` is the
    unit the calipers/guides report in; ``show_edges`` toggles the caliper's
    distance-to-nearest-edge readouts. The default calibration carries no
    physical size, so every length falls back to image pixels.
    """

    canvas_w: float = 0.0
    canvas_h: float = 0.0
    canvas_unit: str = "px"
    display_unit: str = "px"
    show_edges: bool = False

    def _cm_per_px(
        self, image_w: float, image_h: float
    ) -> Optional[Tuple[float, float]]:
        """Return ``(cx, cy)`` cm-per-image-pixel, or ``None`` with no physical size."""
        per = _CM_PER.get(self.canvas_unit)
        if per is None:
            return None
        if self.canvas_w <= 0 or self.canvas_h <= 0 or image_w <= 0 or image_h <= 0:
            return None
        return (self.canvas_w * per / image_w, self.canvas_h * per / image_h)

    def effective_unit(self, image_w: float, image_h: float) -> str:
        """The unit lengths actually resolve to: ``px`` with no physical size set."""
        if self.display_unit == "px" or self._cm_per_px(image_w, image_h) is None:
            return "px"
        return self.display_unit

    def is_physical(self, image_w: float, image_h: float) -> bool:
        """Whether a physical (non-pixel) reading is available for this image."""
        return self.effective_unit(image_w, image_h) != "px"

    def length_value(
        self, dx: float, dy: float, image_w: float, image_h: float
    ) -> float:
        """Numeric length of a scene vector in the effective unit (px if no size).

        Horizontal and vertical scales are applied separately so a diagonal is
        correct even when the pixel aspect differs from the canvas aspect (a
        freeform crop).
        """
        cmpp = self._cm_per_px(image_w, image_h)
        if self.display_unit == "px" or cmpp is None:
            return math.hypot(dx, dy)
        cx, cy = cmpp
        return convert_from_cm(math.hypot(dx * cx, dy * cy), self.display_unit)

    def length_str(self, dx: float, dy: float, image_w: float, image_h: float) -> str:
        """Format the length of a scene vector ``(dx, dy)`` in the display unit."""
        return format_measure(
            self.length_value(dx, dy, image_w, image_h),
            self.effective_unit(image_w, image_h),
        )

    def axis_str(
        self, distance_px: float, along: str, image_w: float, image_h: float
    ) -> str:
        """Format a 1-D distance (image px, along ``"x"`` or ``"y"``) in the unit."""
        cmpp = self._cm_per_px(image_w, image_h)
        if self.display_unit == "px" or cmpp is None:
            return format_measure(distance_px, "px")
        scale = cmpp[0] if along == "x" else cmpp[1]
        return format_measure(
            convert_from_cm(distance_px * scale, self.display_unit), self.display_unit
        )

    def edge_label(
        self,
        px: float,
        py: float,
        left: float,
        top: float,
        right: float,
        bottom: float,
        image_w: float,
        image_h: float,
    ) -> str:
        """Return a compact label of a point's distance to its nearest V+H edge.

        Picks the closer of the left/right edges and the closer of the top/bottom
        edges, labelling each with its side (L/R, T/B), e.g. ``"L 8.0 cm · T 5.0
        cm"``. Distances are in the display unit (image pixels with no physical
        size set).
        """
        d_left, d_right = px - left, right - px
        d_top, d_bottom = py - top, bottom - py
        v_dist, v_side = (d_left, "L") if d_left <= d_right else (d_right, "R")
        h_dist, h_side = (d_top, "T") if d_top <= d_bottom else (d_bottom, "B")
        v = self.axis_str(abs(v_dist), "x", image_w, image_h)
        h = self.axis_str(abs(h_dist), "y", image_w, image_h)
        return f"{v_side} {v} · {h_side} {h}"
