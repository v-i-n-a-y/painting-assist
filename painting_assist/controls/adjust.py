# Copyright 2026 Vinay Williams

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register


@register
class ToneControl(Control):
    """Tone — global contrast, saturation and colour-temperature adjustment.

    Three independent knobs, each neutral at 0.0:

    * **Contrast** — steepens (or flattens) the tonal response about mid-grey via
      a monotonic 256-entry LUT applied to every RGB channel (``cv2.LUT``).
      ``+1`` strongly steepens (pushes toward pure black/white), ``-1`` strongly
      flattens (collapses toward a flat mid-grey).

    * **Saturation** — scales chroma in CIELab space about the neutral point
      (a,b = 128). ``-1`` desaturates fully to greyscale, ``+1`` doubles chroma.

    * **Temperature** — a warm/cool Lab shift: positive warms (raises b* toward
      yellow, nudges a* toward red), negative cools.

    The contrast LUT is cheap and runs alone when it is the only active knob; the
    (comparatively expensive) Lab round-trip is done once and only when either
    saturation or temperature is non-zero. Deterministic; output is uint8 RGB of
    the same shape as the input and ``img`` is never mutated.
    """

    id = "tone"
    name = "Tone"
    order = 5  # runs after crop=0, before blur=10

    @classmethod
    def params(cls) -> List[Param]:
        """Schema: three symmetric −1..1 knobs, all neutral at 0.0."""
        common = dict(
            ptype=ParamType.FLOAT,
            default=0.0,
            minimum=-1.0,
            maximum=1.0,
            step=0.05,
        )
        return [
            Param(
                name="contrast",
                label="Contrast",
                tooltip=(
                    "Steepen (right) or flatten (left) the tonal range about mid-grey."
                ),
                **common,
            ),
            Param(
                name="saturation",
                label="Saturation",
                tooltip=(
                    "Scale colour intensity: left toward greyscale, right toward "
                    "doubled chroma."
                ),
                **common,
            ),
            Param(
                name="temperature",
                label="Temperature",
                tooltip="Warm (right) or cool (left) the overall colour cast.",
                **common,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _contrast_lut(contrast: float) -> np.ndarray:
        """Return a 256-entry uint8 LUT for a monotonic gain about mid-grey.

        The gain is a linear factor about 127.5, clipped to 0..255. ``contrast``
        maps to a factor of 1 (identity) at 0, up to 3.0 at +1 (strong
        steepening) and down to 0.1 at −1 (strong flattening). Positive factors
        keep the LUT monotonic non-decreasing.
        """
        if contrast >= 0.0:
            factor = 1.0 + 2.0 * contrast  # 1 .. 3
        else:
            factor = 1.0 + 0.9 * contrast  # 1 .. 0.1
        x = np.arange(256, dtype=np.float32)
        y = (x - 127.5) * factor + 127.5
        return np.clip(np.round(y), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Control overrides
    # ------------------------------------------------------------------ #
    def is_active(self) -> bool:
        """Inactive (identity) when every knob sits at its 0.0 neutral."""
        if not self.enabled:
            return False
        return (
            float(self.get("contrast")) != 0.0
            or float(self.get("saturation")) != 0.0
            or float(self.get("temperature")) != 0.0
        )

    def process(self, img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> tone-adjusted RGB uint8 HxWx3 (new array)."""
        contrast = float(self.get("contrast"))
        saturation = float(self.get("saturation"))
        temperature = float(self.get("temperature"))

        out = img
        if contrast != 0.0:
            out = cv2.LUT(np.ascontiguousarray(out), self._contrast_lut(contrast))

        if saturation != 0.0 or temperature != 0.0:
            lab = cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_RGB2Lab)
            lab = lab.astype(np.float32)
            a = lab[:, :, 1]
            b = lab[:, :, 2]
            if saturation != 0.0:
                scale = 1.0 + saturation  # -1 -> 0 (grey), +1 -> 2x
                a[:] = (a - 128.0) * scale + 128.0
                b[:] = (b - 128.0) * scale + 128.0
            if temperature != 0.0:
                b[:] = b + temperature * 25.0  # warm -> +b* (yellow)
                a[:] = a + temperature * 8.0  # warm -> +a* (red)
            lab = np.clip(lab, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

        if out is img:
            return img.copy()
        return out
