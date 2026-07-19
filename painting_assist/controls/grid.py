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

# Golden-ratio section positions (phi): the short and long divisions of a unit.
_GOLDEN = (0.382, 0.618)

# Selectable grid layouts (stored value -> shown label). The even layout uses
# the columns/rows counts; the ratio layouts ignore them for fixed proportional
# guides, and the armature draws the classic diagonal harmonic lines.
_LAYOUTS = [
    ("even", "Even (columns x rows)"),
    ("thirds", "Rule of thirds"),
    ("golden", "Golden sections"),
    ("quarters", "Quarters"),
    ("diagonals-armature", "Diagonal armature"),
]


def layout_fractions(
    layout: str, columns: int, rows: int
) -> Tuple[List[float], List[float]]:
    """Return ``(x_fractions, y_fractions)`` of interior line positions in 0..1.

    ``x_fractions`` are vertical-line x positions, ``y_fractions`` horizontal-line
    y positions, as fractions of the width/height. The named ratio layouts return
    fixed proportional guides independent of ``columns``/``rows``; ``even`` (also
    the fallback for any unknown name) spaces the requested number of divisions
    evenly. The ``diagonals-armature`` layout carries no axis-aligned lines (its
    guides are the diagonal segments from :func:`armature_segments`).
    """
    if layout == "thirds":
        return [1.0 / 3.0, 2.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]
    if layout == "golden":
        return [_GOLDEN[0], _GOLDEN[1]], [_GOLDEN[0], _GOLDEN[1]]
    if layout == "quarters":
        return [0.25, 0.5, 0.75], [0.25, 0.5, 0.75]
    if layout == "diagonals-armature":
        return [], []
    cols = max(1, int(columns))
    rows = max(1, int(rows))
    return (
        [i / cols for i in range(1, cols)],
        [j / rows for j in range(1, rows)],
    )


def armature_segments() -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return the armature-of-the-rectangle segments as 0..1 fraction endpoints.

    The two main diagonals plus the eight "reciprocal" lines from each corner to
    the midpoints of the two non-adjacent sides -- the classic painter's armature
    for locating harmonious divisions and the rabatment points. Each item is a
    ``((x0, y0), (x1, y1))`` pair in width/height fractions.
    """
    tl, tr, br, bl = (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
    top, right, bottom, left = (0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)
    return [
        (tl, br),
        (tr, bl),  # main diagonals
        (tl, right),
        (tl, bottom),  # from top-left
        (tr, left),
        (tr, bottom),  # from top-right
        (br, left),
        (br, top),  # from bottom-right
        (bl, right),
        (bl, top),  # from bottom-left
    ]


def draw_grid(
    img: np.ndarray,
    columns: int,
    rows: int,
    color_rgb: Tuple[int, int, int],
    opacity: float,
    thickness: int,
    diagonals: bool,
    layout: str = "even",
) -> np.ndarray:
    """Return a new RGB uint8 HxWx3 array with grid lines drawn over ``img``.

    Non-mutating: ``img`` is never modified. ``layout`` selects the line pattern
    (see :func:`layout_fractions`); ``columns``/``rows`` set the divisions of the
    even layout and are ignored by the ratio layouts. ``color_rgb`` is an RGB
    triple, ``opacity`` a 0..1 blend factor, ``thickness`` a line width in pixels
    (scaled to the image's short side), and ``diagonals`` toggles the
    corner-to-corner centre-finding lines (independent of ``layout``). Kept as a
    free function so the pixel-drawing routine can be shared by an export path
    independently of the (now non-destructive) control.
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
    x_fractions, y_fractions = layout_fractions(str(layout), cols, rows)
    for fx in x_fractions:
        x = int(round(fx * w))
        cv2.line(overlay, (x, 0), (x, h), color, line_w, cv2.LINE_AA)
    for fy in y_fractions:
        y = int(round(fy * h))
        cv2.line(overlay, (0, y), (w, y), color, line_w, cv2.LINE_AA)
    if str(layout) == "diagonals-armature":
        for (fx0, fy0), (fx1, fy1) in armature_segments():
            p0 = (int(round(fx0 * (w - 1))), int(round(fy0 * (h - 1))))
            p1 = (int(round(fx1 * (w - 1))), int(round(fy1 * (h - 1))))
            cv2.line(overlay, p0, p1, color, line_w, cv2.LINE_AA)
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
                name="layout",
                label="Layout",
                ptype=ParamType.CHOICE,
                default="even",
                choices=_LAYOUTS,
                tooltip=(
                    "Grid pattern: even divisions, rule of thirds, golden "
                    "sections, quarters, or the diagonal armature."
                ),
            ),
            Param(
                name="columns",
                label="Columns",
                ptype=ParamType.INT,
                default=4,
                minimum=1,
                maximum=24,
                step=1,
                tooltip="Vertical divisions (number of columns) — even layout.",
            ),
            Param(
                name="rows",
                label="Rows",
                ptype=ParamType.INT,
                default=4,
                minimum=1,
                maximum=24,
                step=1,
                tooltip="Horizontal divisions (number of rows) — even layout.",
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

    def create_editor(self, parent: object = None):
        """Build the grid editor (generic rows + canvas-position readout)."""
        from painting_assist.widgets.grid_editor import GridEditor

        return GridEditor(self, parent)

    def overlay_spec(self) -> Dict[str, Any]:
        """Return the resolved overlay parameters for the viewer to draw.

        Resolves the colour preset to an actual RGB triple and normalises
        opacity to 0..1. ``layout`` selects the pattern and drives the explicit
        ``x_fractions``/``y_fractions`` (vertical/horizontal line positions in
        0..1) plus ``diagonal_lines`` (armature segments as fraction-endpoint
        pairs) so the viewer and the export bake share one geometry contract.
        ``visible`` reports whether the enabled control would draw anything (any
        layout line, or diagonals on); the view can use it to decide whether to
        show the overlay at all.
        """
        cols = max(1, int(self.get("columns")))
        rows = max(1, int(self.get("rows")))
        layout = str(self.get("layout"))
        diagonals = bool(self.get("diagonals"))
        x_fractions, y_fractions = layout_fractions(layout, cols, rows)
        diagonal_lines = armature_segments() if layout == "diagonals-armature" else []
        has_lines = bool(x_fractions or y_fractions or diagonal_lines or diagonals)
        return {
            "columns": cols,
            "rows": rows,
            "layout": layout,
            "x_fractions": x_fractions,
            "y_fractions": y_fractions,
            "diagonal_lines": diagonal_lines,
            "color_rgb": _COLORS.get(str(self.get("color")), _COLORS["red"]),
            "opacity": max(0.0, min(1.0, int(self.get("opacity")) / 100.0)),
            "thickness": max(1, int(self.get("thickness"))),
            "diagonals": diagonals,
            "visible": self.enabled and has_lines,
        }
