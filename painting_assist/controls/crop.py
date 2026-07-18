# Copyright 2026 Vinay Williams

from __future__ import annotations

from typing import List, Optional

import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register


@register
class CropControl(Control):
    """Crop the reference to a canvas aspect ratio (or freeform).

    Enter the canvas' width and height and, with **Lock to canvas ratio** on,
    the interactive crop rectangle in the viewport is constrained to that aspect
    ratio so the reference matches the surface you will paint on. Turn the lock
    off for a freeform crop of any shape.

    The crop is stored as a *normalised* rectangle (``rx, ry, rw, rh`` as
    fractions 0..1 of the original image), so it is resolution-independent and
    fully non-destructive — the model keeps the untouched original and **Reset**
    restores the full frame. Runs first in the pipeline so every later control
    sees the cropped region.
    """

    id = "crop"
    name = "Canvas & Crop"
    order = 0  # crop before everything else

    @classmethod
    def params(cls) -> List[Param]:
        """Canvas dimensions + lock flag + the normalised crop rectangle."""
        return [
            Param(
                name="canvas_w",
                label="Width",
                ptype=ParamType.FLOAT,
                default=4.0,
                minimum=0.1,
                maximum=100000.0,
                step=0.1,
                tooltip="Canvas width (only the width:height ratio matters).",
            ),
            Param(
                name="canvas_h",
                label="Height",
                ptype=ParamType.FLOAT,
                default=3.0,
                minimum=0.1,
                maximum=100000.0,
                step=0.1,
                tooltip="Canvas height (only the width:height ratio matters).",
            ),
            Param(
                name="unit",
                label="Unit",
                ptype=ParamType.CHOICE,
                default="cm",
                choices=[
                    ("cm", "cm"),
                    ("in", "inch"),
                    ("mm", "mm"),
                    ("px", "px"),
                    ("ratio", "ratio"),
                ],
                tooltip="Display unit only — cropping uses the width:height ratio.",
            ),
            Param(
                name="lock_ratio",
                label="Lock to canvas ratio",
                ptype=ParamType.BOOL,
                default=True,
                tooltip="On: crop is locked to the canvas ratio. Off: freeform crop.",
            ),
            # Normalised crop rectangle (fractions of the original image).
            Param(
                name="rx",
                label="X",
                ptype=ParamType.FLOAT,
                default=0.0,
                minimum=0.0,
                maximum=1.0,
                step=0.0001,
            ),
            Param(
                name="ry",
                label="Y",
                ptype=ParamType.FLOAT,
                default=0.0,
                minimum=0.0,
                maximum=1.0,
                step=0.0001,
            ),
            Param(
                name="rw",
                label="W",
                ptype=ParamType.FLOAT,
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                step=0.0001,
            ),
            Param(
                name="rh",
                label="H",
                ptype=ParamType.FLOAT,
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                step=0.0001,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def aspect(self) -> Optional[float]:
        """Canvas width/height ratio when locked, else ``None`` (freeform)."""
        if not bool(self.get("lock_ratio")):
            return None
        w = float(self.get("canvas_w"))
        h = float(self.get("canvas_h"))
        if w <= 0 or h <= 0:
            return None
        return w / h

    def rect_norm(self) -> tuple:
        """Return the clamped normalised crop rect ``(rx, ry, rw, rh)``."""
        rx = float(self.get("rx"))
        ry = float(self.get("ry"))
        rw = float(self.get("rw"))
        rh = float(self.get("rh"))
        rx = min(max(rx, 0.0), 1.0)
        ry = min(max(ry, 0.0), 1.0)
        rw = min(max(rw, 0.0), 1.0 - rx)
        rh = min(max(rh, 0.0), 1.0 - ry)
        return (rx, ry, rw, rh)

    def _is_full_frame(self) -> bool:
        """True when the rect covers (essentially) the whole image."""
        rx, ry, rw, rh = self.rect_norm()
        return rx <= 1e-4 and ry <= 1e-4 and rw >= 1.0 - 1e-4 and rh >= 1.0 - 1e-4

    # ------------------------------------------------------------------ #
    # Control overrides
    # ------------------------------------------------------------------ #
    def is_active(self) -> bool:
        """Active only when enabled AND the rect actually trims the image."""
        return self.enabled and not self._is_full_frame()

    def process(self, img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> cropped RGB uint8 HxWx3 (fresh contiguous array)."""
        rx, ry, rw, rh = self.rect_norm()
        h, w = img.shape[:2]
        x0 = int(round(rx * w))
        y0 = int(round(ry * h))
        x1 = int(round((rx + rw) * w))
        y1 = int(round((ry + rh) * h))
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))
        if x0 == 0 and y0 == 0 and x1 == w and y1 == h:
            return img
        return np.ascontiguousarray(img[y0:y1, x0:x1])

    def create_editor(self, parent: Optional[object] = None):
        """Build the custom canvas-dimensions + crop-tool editor (lazy Qt import)."""
        from painting_assist.widgets.crop_editor import CropEditor

        return CropEditor(self, parent)
