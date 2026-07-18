# Copyright 2026 Vinay Williams

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register


@register
class BlurControl(Control):
    """Gaussian blur — the coarse-to-fine "blocking" control.

    Two modes share one blur engine:

    * **Continuous** — a single reversed "Detail" slider (``radius`` 0..100):
      LEFT is a heavy blur (block in big values/shapes first), RIGHT moves
      toward 0 (reveal fine detail). The stored value is the plain 0..100
      number; only the UI inverts the slider.

    * **Stepped** — a fixed ladder of discrete blur levels you step through as a
      painting progresses (Stage 1 = heaviest, last stage = sharp). The ladder
      is either **even** (``radius`` sets the heaviest level; the rest are
      evenly spaced down to 0 across ``stage_count`` stages) or **manual**
      (``manual_values`` is a comma-separated list of 0..100 radii you type in).

    Blur strength is interpreted as a fraction of the image's smaller dimension
    rather than an absolute pixel count, so a given radius blocks in roughly the
    same amount of structure on a 600 px thumbnail as on a 6000 px scan.
    """

    id = "blur"
    name = "Blur"
    order = 10  # runs after crop, before finer adjustments

    @classmethod
    def params(cls) -> List[Param]:
        """Schema covering both the continuous slider and the stepped ladder."""
        return [
            Param(
                name="mode",
                label="Mode",
                ptype=ParamType.CHOICE,
                default="continuous",
                choices=[("continuous", "Continuous"), ("stepped", "Stepped")],
                tooltip="Continuous slider, or a ladder of discrete blur stages.",
            ),
            Param(
                name="radius",
                label="Detail",
                ptype=ParamType.INT,
                default=0,
                minimum=0,
                maximum=100,
                step=1,
                reversed=True,  # LEFT = heavy blur (coarse), RIGHT = 0 (fine detail)
                suffix=" px",
                tooltip=(
                    "Continuous: left = heavy blur, right toward 0 reveals detail. "
                    "Stepped+even: the heaviest (Stage 1) blur level."
                ),
            ),
            Param(
                name="stage_count",
                label="Stages",
                ptype=ParamType.INT,
                default=5,
                minimum=2,
                maximum=12,
                step=1,
                tooltip="How many discrete blur set-points to step through.",
            ),
            Param(
                name="spacing",
                label="Spacing",
                ptype=ParamType.CHOICE,
                default="even",
                choices=[("even", "Even"), ("manual", "Manual")],
                tooltip="Even spacing down to sharp, or manually entered levels.",
            ),
            Param(
                name="manual_values",
                label="Levels",
                ptype=ParamType.TEXT,
                default="",
                tooltip="Comma-separated blur levels (0..100), e.g. 80, 55, 35, 18, 0.",
            ),
            Param(
                name="stage",
                label="Stage",
                ptype=ParamType.INT,
                default=1,
                minimum=1,
                maximum=12,
                step=1,
                tooltip="Which set-point is active (1 = heaviest blocking).",
            ),
        ]

    # ------------------------------------------------------------------ #
    # Stepped-mode helpers
    # ------------------------------------------------------------------ #
    def stage_count(self) -> int:
        """Current number of stages, clamped to the valid 2..12 range."""
        return max(2, min(12, int(self.get("stage_count"))))

    def stage_levels(self) -> List[int]:
        """Return the ladder of blur radii (length == stage_count).

        Even spacing runs from ``radius`` (heaviest, Stage 1) down to 0 (sharp)
        via :func:`numpy.linspace`. Manual spacing parses ``manual_values``;
        too few entries are padded with 0, too many are truncated, and each is
        clamped to 0..100.
        """
        n = self.stage_count()
        if str(self.get("spacing")) == "manual":
            levels = self._parse_manual()
            if len(levels) < n:
                levels = levels + [0] * (n - len(levels))
            return [int(max(0, min(100, v))) for v in levels[:n]]
        top = int(max(0, min(100, int(self.get("radius")))))
        return [int(round(v)) for v in np.linspace(top, 0, n)]

    def _parse_manual(self) -> List[int]:
        """Parse ``manual_values`` ("80, 55, 35") into a list of ints (lenient)."""
        raw = str(self.get("manual_values") or "")
        out: List[int] = []
        for chunk in raw.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                out.append(int(round(float(chunk))))
            except ValueError:
                continue
        return out

    def current_stage(self) -> int:
        """The active stage index (1-based), clamped to ``stage_count``."""
        return max(1, min(self.stage_count(), int(self.get("stage"))))

    def effective_radius(self) -> int:
        """The 0..100 blur radius actually applied, given the current mode."""
        if str(self.get("mode")) == "stepped":
            levels = self.stage_levels()
            return int(levels[self.current_stage() - 1])
        return int(max(0, min(100, int(self.get("radius")))))

    # ------------------------------------------------------------------ #
    # Control overrides
    # ------------------------------------------------------------------ #
    def is_active(self) -> bool:
        """Skip cheaply when disabled or when the effective radius is 0."""
        return self.enabled and self.effective_radius() > 0

    def process(self, img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> Gaussian-blurred RGB uint8 HxWx3 (new array).

        Does not mutate ``img``: returns it unchanged when there is nothing to
        do; otherwise ``cv2.GaussianBlur`` allocates a fresh array. The abstract
        0..100 radius is scaled by the image's smaller side so the effect is
        comparable across resolutions, then converted to an odd kernel size.
        """
        r = self.effective_radius()
        if r <= 0:
            return img

        h, w = img.shape[:2]
        short_side = min(h, w)
        if short_side <= 0:
            return img

        # r == 100 corresponds to ~12.5% of the smaller dimension.
        pixel_radius = int(round((r / 100.0) * 0.125 * short_side))
        if pixel_radius <= 0:
            return img

        k = 2 * pixel_radius + 1  # GaussianBlur needs an odd kernel
        return cv2.GaussianBlur(img, (k, k), 0)

    def create_editor(self, parent: Optional[object] = None):
        """Build the custom Continuous/Stepped editor (lazy Qt import)."""
        from painting_assist.widgets.blur_editor import BlurEditor

        return BlurEditor(self, parent)
