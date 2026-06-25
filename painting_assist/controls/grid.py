from __future__ import annotations

from typing import List

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


@register
class GridControl(Control):
    """A positioning grid drawn over the reference (the painter's "grid method").

    Divides the image into an even ``columns`` × ``rows`` lattice so you can
    transfer proportions and placement to a matching grid on your canvas. Drawn
    last in the pipeline (on top of crop + blur), so the grid lines stay crisp
    over a blurred block-in and divide the *cropped* region — exactly the cells
    you'd rule onto your surface. It is non-destructive (the original is kept)
    and is included when you Save, so you can print a gridded reference.
    """

    id = "grid"
    name = "Grid"
    order = 90  # drawn on top of every other control

    @classmethod
    def params(cls) -> List[Param]:
        """All generic widgets — no custom editor needed."""
        return [
            Param(
                name="columns", label="Columns", ptype=ParamType.INT,
                default=4, minimum=1, maximum=24, step=1,
                tooltip="Vertical divisions (number of columns).",
            ),
            Param(
                name="rows", label="Rows", ptype=ParamType.INT,
                default=4, minimum=1, maximum=24, step=1,
                tooltip="Horizontal divisions (number of rows).",
            ),
            Param(
                name="color", label="Colour", ptype=ParamType.CHOICE,
                default="red",
                choices=[(key, key.capitalize()) for key in _COLORS],
                tooltip="Line colour — pick one that contrasts with the image.",
            ),
            Param(
                name="opacity", label="Opacity", ptype=ParamType.INT,
                default=100, minimum=10, maximum=100, step=5, suffix=" %",
                tooltip="Line opacity; lower it to see the image through the grid.",
            ),
            Param(
                name="thickness", label="Line width", ptype=ParamType.INT,
                default=2, minimum=1, maximum=8,
                tooltip="Relative line width (scaled to the image size).",
            ),
            Param(
                name="diagonals", label="Diagonals", ptype=ParamType.BOOL,
                default=False,
                tooltip="Draw corner-to-corner diagonals to find the centre.",
            ),
        ]

    def is_active(self) -> bool:
        """Active only when enabled AND something would actually be drawn."""
        if not self.enabled:
            return False
        return (
            int(self.get("columns")) > 1
            or int(self.get("rows")) > 1
            or bool(self.get("diagonals"))
        )

    def process(self, img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> the image with grid lines drawn over it (new array)."""
        h, w = img.shape[:2]
        if h < 2 or w < 2:
            return img

        cols = max(1, int(self.get("columns")))
        rows = max(1, int(self.get("rows")))
        color = _COLORS.get(str(self.get("color")), _COLORS["red"])
        opacity = max(0.0, min(1.0, int(self.get("opacity")) / 100.0))
        # Scale line width to the image so it reads at any resolution.
        short_side = min(h, w)
        thickness = max(1, int(round(int(self.get("thickness")) * short_side / 1000.0)))

        overlay = np.ascontiguousarray(img)  # copy we may draw on
        if overlay is img:
            overlay = img.copy()

        for i in range(1, cols):
            x = int(round(i * w / cols))
            cv2.line(overlay, (x, 0), (x, h), color, thickness, cv2.LINE_AA)
        for j in range(1, rows):
            y = int(round(j * h / rows))
            cv2.line(overlay, (0, y), (w, y), color, thickness, cv2.LINE_AA)
        if bool(self.get("diagonals")):
            cv2.line(overlay, (0, 0), (w - 1, h - 1), color, thickness, cv2.LINE_AA)
            cv2.line(overlay, (w - 1, 0), (0, h - 1), color, thickness, cv2.LINE_AA)

        if opacity >= 1.0:
            return overlay
        return cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0.0)
