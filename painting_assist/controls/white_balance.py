# Copyright 2026 Vinay Williams

"""White balance - neutralise a colour cast by naming a patch that should be grey.

The painter picks (with the eyedropper) a patch of the reference that they know
*ought* to be a neutral grey: a white wall in shadow, a grey card, a concrete
step. Whatever colour that patch actually is reveals the light's colour cast,
and this control removes it, shifting every pixel by the same amount so the
picked patch reads as truly neutral and the rest of the image is corrected in
step.

The work is done in CIELab (OpenCV's uint8 encoding: L in 0-255, a/b offset by
128). The neutral reference is converted to Lab to read its chromatic offset
(a-128, b-128); that offset is subtracted from every pixel's a and b, which
lands the picked colour on the neutral axis (a = b = 128) and cools or warms
everything else consistently. Lightness (L) is left untouched. Deterministic;
the output is uint8 RGB of the same shape as the input and ``img`` is never
mutated.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register


@register
class WhiteBalanceControl(Control):
    """White balance - grey-point neutralisation of a colour cast.

    Runs early (order 2, right after crop and flip) so every downstream colour-
    and value-oriented control sees the cast-corrected image. Neutral by
    default: the reference patch defaults to mid-grey (128, 128, 128), whose Lab
    a/b are already 128, so the control is a no-op until the painter picks a
    patch that carries an actual cast.
    """

    id = "white_balance"
    name = "White balance"
    # Runs after crop=0/flip=1 and before tone=5, so downstream colour/value
    # controls all see the cast-corrected image.
    order = 2

    NEUTRAL = 128  # Lab a/b of a chromatically neutral (grey) pixel, OpenCV uint8
    # A picked patch whose Lab a/b sit within this many units of neutral carries
    # no meaningful cast, so the control stays inactive and process is identity.
    EPS = 0.5

    @classmethod
    def params(cls) -> List[Param]:
        """Schema: the RGB of the patch that should read as neutral grey."""
        common = dict(
            ptype=ParamType.INT,
            default=128,
            minimum=0,
            maximum=255,
            step=1,
        )
        return [
            Param(
                name="neutral_r",
                label="Neutral R",
                tooltip=(
                    "Red of the patch that should be neutral grey. Pick it with "
                    "the eyedropper; whatever cast it carries is removed from the "
                    "whole image."
                ),
                **common,
            ),
            Param(
                name="neutral_g",
                label="Neutral G",
                tooltip="Green of the patch that should be neutral grey.",
                **common,
            ),
            Param(
                name="neutral_b",
                label="Neutral B",
                tooltip="Blue of the patch that should be neutral grey.",
                **common,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _neutral_offset(self) -> Tuple[float, float]:
        """Return the picked patch's Lab chromatic offset ``(a-128, b-128)``.

        The neutral RGB is converted through a 1x1 image so the offset matches,
        pixel for pixel, what the same colour becomes inside the full image
        (RGB->Lab is a pure per-pixel map). A neutral grey yields ``(0, 0)``.
        """
        r = int(self.get("neutral_r"))
        g = int(self.get("neutral_g"))
        b = int(self.get("neutral_b"))
        lab = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2Lab)
        return (
            float(lab[0, 0, 1]) - float(self.NEUTRAL),
            float(lab[0, 0, 2]) - float(self.NEUTRAL),
        )

    # ------------------------------------------------------------------ #
    # Control overrides
    # ------------------------------------------------------------------ #
    def is_active(self) -> bool:
        """Active only when enabled AND the picked patch carries a real cast."""
        if not self.enabled:
            return False
        offset_a, offset_b = self._neutral_offset()
        return abs(offset_a) > self.EPS or abs(offset_b) > self.EPS

    def process(self, img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> cast-corrected RGB uint8 HxWx3 (new array).

        Does not mutate ``img``. Subtracts the picked patch's Lab a/b offset from
        every pixel, landing that colour on the neutral axis. When the patch is
        already neutral (offset within ``EPS``) the transform is identity, so a
        fresh copy is returned without the lossy Lab roundtrip.
        """
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img.copy()

        offset_a, offset_b = self._neutral_offset()
        if abs(offset_a) <= self.EPS and abs(offset_b) <= self.EPS:
            return img.copy()

        lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2Lab)
        lab = lab.astype(np.float32)
        lab[:, :, 1] -= offset_a
        lab[:, :, 2] -= offset_b
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        return np.ascontiguousarray(out)
