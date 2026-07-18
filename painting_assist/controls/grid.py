# Copyright 2026 Vinay Williams

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register

# Preset line colours (RGB, matching the app's array contract).
_COLORS = {
    "red": (220, 30, 30),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "cyan": (0, 200, 220),
    "magenta": (230, 40, 200),
    "yellow": (245, 220, 0),
    "green": (30, 200, 30),
    "blue": (40, 90, 235),
}


def draw_grid(
    img: np.ndarray,
    columns: int,
    rows: int,
    color_rgb: Tuple[int, int, int],
    opacity: float,
    thickness: int,
    diagonals: bool,
) -> np.ndarray:
    """Return a new RGB uint8 HxWx3 array with grid lines drawn over ``img``.

    Non-mutating: ``img`` is never modified. ``columns``/``rows`` are the number
    of even divisions, ``color_rgb`` an RGB triple, ``opacity`` a 0..1 blend
    factor, ``thickness`` a line width in pixels (scaled to the image's short
    side), and ``diagonals`` toggles the corner-to-corner centre-finding lines.
    Kept as a free function so the pixel-drawing routine can be shared by an
    export path independently of the (now non-destructive) control.
    """
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return img

    cols = max(1, int(columns))
    rows = max(1, int(rows))
    color = tuple(int(c) for c in color_rgb)
    opacity = max(0.0, min(1.0, float(opacity)))
    # Scale line width to the image so it reads at any resolution.
    short_side = min(h, w)
    line_w = max(1, int(round(int(thickness) * short_side / 1000.0)))

    overlay = img.copy()  # draw on a copy; never mutate the input
    for i in range(1, cols):
        x = int(round(i * w / cols))
        cv2.line(overlay, (x, 0), (x, h), color, line_w, cv2.LINE_AA)
    for j in range(1, rows):
        y = int(round(j * h / rows))
        cv2.line(overlay, (0, y), (w, y), color, line_w, cv2.LINE_AA)
    if bool(diagonals):
        cv2.line(overlay, (0, 0), (w - 1, h - 1), color, line_w, cv2.LINE_AA)
        cv2.line(overlay, (w - 1, 0), (0, h - 1), color, line_w, cv2.LINE_AA)

    if opacity >= 1.0:
        return overlay
    return cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0.0)


@register
class GridControl(Control):
    """A positioning grid drawn over the reference (the painter's "grid method").

    Divides the image into an even ``columns`` × ``rows`` lattice so you can
    transfer proportions and placement to a matching grid on your canvas.

    The grid is a non-destructive *viewer overlay*: it is not baked into the
    processed pixels. :meth:`process` is the identity and :meth:`is_active`
    always returns ``False`` so the pipeline treats the control as a no-op (and
    its params never churn the render cache). The control is retained only to
    own the params (for the control panel UI and session persistence); the view
    reads :meth:`overlay_spec` to draw the lines over the pixmap, and any export
    path can call the module-level :func:`draw_grid` to bake them if needed.
    """

    id = "grid"
    name = "Grid"
    order = 90  # drawn on top of every other control

    @classmethod
    def params(cls) -> List[Param]:
        """All generic widgets — no custom editor needed."""
        return [
            Param(
                name="columns",
                label="Columns",
                ptype=ParamType.INT,
                default=4,
                minimum=1,
                maximum=24,
                step=1,
                tooltip="Vertical divisions (number of columns).",
            ),
            Param(
                name="rows",
                label="Rows",
                ptype=ParamType.INT,
                default=4,
                minimum=1,
                maximum=24,
                step=1,
                tooltip="Horizontal divisions (number of rows).",
            ),
            Param(
                name="color",
                label="Colour",
                ptype=ParamType.CHOICE,
                default="red",
                choices=[(key, key.capitalize()) for key in _COLORS],
                tooltip="Line colour — pick one that contrasts with the image.",
            ),
            Param(
                name="opacity",
                label="Opacity",
                ptype=ParamType.INT,
                default=100,
                minimum=10,
                maximum=100,
                step=5,
                suffix=" %",
                tooltip="Line opacity; lower it to see the image through the grid.",
            ),
            Param(
                name="thickness",
                label="Line width",
                ptype=ParamType.INT,
                default=2,
                minimum=1,
                maximum=8,
                tooltip="Relative line width (scaled to the image size).",
            ),
            Param(
                name="diagonals",
                label="Diagonals",
                ptype=ParamType.BOOL,
                default=False,
                tooltip="Draw corner-to-corner diagonals to find the centre.",
            ),
        ]

    def is_active(self) -> bool:
        """Always ``False``: the grid is a viewer overlay, never a pixel change.

        Returning ``False`` unconditionally makes the pipeline skip this control
        entirely, so its params never invalidate the render cache. Whether the
        overlay actually shows anything is decided by the view from
        :meth:`overlay_spec`.
        """
        return False

    def process(self, img: np.ndarray) -> np.ndarray:
        """Identity: the grid is drawn by the viewer, not baked into pixels."""
        return img

    def overlay_spec(self) -> Dict[str, Any]:
        """Return the resolved overlay parameters for the viewer to draw.

        Resolves the colour preset to an actual RGB triple and normalises
        opacity to 0..1. ``visible`` reports whether the enabled control would
        draw anything (more than one division, or diagonals on); the view can
        use it to decide whether to show the overlay at all.
        """
        cols = max(1, int(self.get("columns")))
        rows = max(1, int(self.get("rows")))
        diagonals = bool(self.get("diagonals"))
        return {
            "columns": cols,
            "rows": rows,
            "color_rgb": _COLORS.get(str(self.get("color")), _COLORS["red"]),
            "opacity": max(0.0, min(1.0, int(self.get("opacity")) / 100.0)),
            "thickness": max(1, int(self.get("thickness"))),
            "diagonals": diagonals,
            "visible": self.enabled and (cols > 1 or rows > 1 or diagonals),
        }
