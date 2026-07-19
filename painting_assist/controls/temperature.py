# Copyright 2026 Vinay Williams

"""Temperature map - a false-colour diagnostic of warm and cool zones.

Judging colour temperature by eye is hard: a warm shadow next to a cool one can
look identical until they sit side by side. This control repaints the reference
as a two-tone diagnostic so the temperature structure reads at a glance. Every
pixel is classed warm or cool from its position on the CIELab b* axis (the
yellow-blue axis, which is the painter's warm-cool axis) with a small a*
(red-green) contribution, then tinted toward orange (warm) or blue (cool). The
tint is modulated by the pixel's lightness so forms stay legible, and a Strength
knob cross-fades the whole thing back to a plain greyscale value study.

All work is in CIELab (OpenCV's uint8 encoding: L in 0-255, a/b offset by 128).
Deterministic; the output is uint8 RGB of the same shape as the input and
``img`` is never mutated.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register


@register
class TemperatureMapControl(Control):
    """Temperature map - repaint the reference as warm/cool false colour.

    A diagnostic view (order 25, after Values=20): when enabled it always
    replaces the image with the false-colour map. Neutral, near-greyscale pixels
    stay grey; only pixels with a real warm or cool bias pick up a tint, so the
    result reads as a value study with the temperature zones lit up.
    """

    id = "temp_map"
    name = "Temperature map"
    order = 25  # runs after values=20, a late diagnostic view

    NEUTRAL = 128.0  # Lab a/b of a chromatically neutral pixel (OpenCV uint8)
    # Warmth this many Lab units from neutral saturates the tint fully. Chosen so
    # ordinary casts read clearly without every pixel slamming to pure orange/blue.
    WARMTH_SCALE = 40.0
    # Two-tone targets (RGB, 0..1). Warm keeps R>B and cool keeps B>R by design.
    WARM_RGB = (1.0, 0.55, 0.15)  # orange
    COOL_RGB = (0.20, 0.50, 1.0)  # blue

    @classmethod
    def params(cls) -> List[Param]:
        """Schema: a single Strength cross-fade from greyscale to false colour."""
        return [
            Param(
                name="strength",
                label="Strength",
                ptype=ParamType.INT,
                default=100,
                minimum=0,
                maximum=100,
                step=1,
                suffix=" %",
                tooltip=(
                    "Cross-fade from a plain greyscale value study (0) to the "
                    "full warm/cool false-colour map (100)."
                ),
            ),
        ]

    # ------------------------------------------------------------------ #
    # Control overrides
    # ------------------------------------------------------------------ #
    def is_active(self) -> bool:
        """A diagnostic view: active whenever enabled."""
        return self.enabled

    def process(self, img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> warm/cool false-colour RGB uint8 HxWx3 (new array).

        Does not mutate ``img``. Converts to CIELab, derives a signed warmth from
        b* (plus a small a* term), tints each pixel toward orange or blue in
        proportion to ``|warmth|``, modulates by lightness for legibility, and
        cross-fades back toward greyscale by the Strength knob.
        """
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img.copy()

        strength = max(0, min(100, int(self.get("strength")))) / 100.0

        lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2Lab)
        lab = lab.astype(np.float32)
        lightness = lab[:, :, 0] / 255.0  # 0..1 brightness for legibility
        warmth = (lab[:, :, 2] - self.NEUTRAL) + 0.3 * (lab[:, :, 1] - self.NEUTRAL)
        signed = np.clip(warmth / self.WARMTH_SCALE, -1.0, 1.0)
        magnitude = np.abs(signed)

        warm = np.array(self.WARM_RGB, dtype=np.float32)
        cool = np.array(self.COOL_RGB, dtype=np.float32)
        tone = np.where((signed >= 0.0)[:, :, None], warm, cool)  # (h, w, 3)

        grey = np.repeat(lightness[:, :, None], 3, axis=2)
        tinted = lightness[:, :, None] * tone  # lightness-modulated tone

        blend = (strength * magnitude)[:, :, None]  # 0..1 per pixel
        out = grey * (1.0 - blend) + tinted * blend

        out = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out)
