# Copyright 2026 Vinay Williams

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register


@register
class ValuesControl(Control):
    """Values — see the painting as value masses rather than colours.

    Value (lightness) is the structural backbone of a painting; hue and chroma
    ride on top of it. This control strips the reference back to its value
    story so the painter can judge the big dark/mid/light relationships without
    being distracted by colour.

    Two modes:

    * **Greyscale** replaces every pixel's colour with a neutral grey of the
      same perceptual lightness (the CIELab L channel), so the reference reads
      as a pure value study.
    * **Value steps** posterizes lightness into a handful of flat value bands
      — the classic "notan"/limited-value approach — optionally keeping the
      original colour within each band (value-grouped colour) rather than
      going neutral.

    The optional **Isolate** knob dims every value band except a chosen one
    toward flat mid-grey, letting the painter study one value mass at a time.

    All work is done in CIELab (OpenCV's uint8 encoding: L in 0–255, a/b
    offset by 128), which is perceptually uniform so equal steps in L look
    like equal steps in value to the eye.
    """

    id = "values"
    name = "Values"
    order = 20  # runs after quantize=15, before grid=90

    NEUTRAL = 128  # Lab a/b value for a chromatically-neutral (grey) pixel

    @classmethod
    def params(cls) -> List[Param]:
        """Schema for mode, value-step count, colour-keeping and isolation."""
        return [
            Param(
                name="mode",
                label="Mode",
                ptype=ParamType.CHOICE,
                default="grey",
                choices=[("grey", "Greyscale"), ("posterize", "Value steps")],
                tooltip=(
                    "Greyscale: replace colour with neutral grey of the same "
                    "lightness. Value steps: posterize lightness into flat "
                    "value bands (notan)."
                ),
            ),
            Param(
                name="steps",
                label="Value steps",
                ptype=ParamType.INT,
                default=3,
                minimum=2,
                maximum=8,
                step=1,
                tooltip=(
                    "Number of flat value bands (Value steps mode). Fewer "
                    "steps = a bolder, more abstract value pattern."
                ),
            ),
            Param(
                name="keep_colour",
                label="Keep colour",
                ptype=ParamType.BOOL,
                default=False,
                tooltip=(
                    "Value steps mode: posterize the lightness but keep each "
                    "pixel's original hue — value-grouped colour rather than "
                    "neutral grey."
                ),
            ),
            Param(
                name="isolate",
                label="Isolate band",
                ptype=ParamType.INT,
                default=0,
                minimum=0,
                maximum=8,
                step=1,
                tooltip=(
                    "0 = off. Otherwise keep the chosen value band (1 = "
                    "darkest) at full brightness and dim every other pixel "
                    "toward flat mid-grey so that band pops."
                ),
            ),
        ]

    # ------------------------------------------------------------------ #
    # Control overrides
    # ------------------------------------------------------------------ #
    def is_active(self) -> bool:
        """Always meaningful when enabled (greyscale changes any colour image)."""
        return self.enabled

    def process(self, img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> value-reduced RGB uint8 HxWx3 (new array).

        Does not mutate ``img``. Converts to CIELab, reduces the L channel
        according to the mode (neutral greyscale or posterized value steps,
        optionally keeping the original a/b), converts back to RGB, then
        applies the optional band isolation.
        """
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img.copy()

        mode = self.get("mode")
        steps = max(2, min(8, int(self.get("steps"))))
        keep_colour = bool(self.get("keep_colour"))
        isolate = max(0, min(8, int(self.get("isolate"))))

        lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2Lab)
        L = lab[:, :, 0].astype(np.float32)

        # Even bands over the full 0-255 lightness range. Interior edges feed
        # np.digitize; band index is 0..steps-1. Both posterize and grey-mode
        # isolation share this banding.
        edges = np.linspace(0.0, 255.0, steps + 1)
        band = np.digitize(L, edges[1:-1]).astype(np.int32)  # 0..steps-1
        band = np.clip(band, 0, steps - 1)

        if mode == "posterize":
            # Map each band to the mean L of its members (deterministic);
            # empty bands fall back to their geometric centre.
            centres = (edges[:-1] + edges[1:]) * 0.5
            sums = np.bincount(band.ravel(), weights=L.ravel(), minlength=steps)
            counts = np.bincount(band.ravel(), minlength=steps)
            means = np.where(counts > 0, sums / np.maximum(counts, 1), centres)
            new_L = means[band]
            out_lab = lab.copy()
            out_lab[:, :, 0] = np.clip(np.round(new_L), 0, 255).astype(np.uint8)
            if not keep_colour:
                out_lab[:, :, 1] = self.NEUTRAL
                out_lab[:, :, 2] = self.NEUTRAL
            out = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)
        else:  # "grey" (and any unknown value falls back to greyscale)
            out_lab = lab.copy()
            out_lab[:, :, 1] = self.NEUTRAL
            out_lab[:, :, 2] = self.NEUTRAL
            out = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)

        if isolate > 0:
            # Keep the chosen band (clamped to what's available), dim the rest
            # by blending 75% toward flat mid-grey so the band pops.
            keep_idx = min(isolate, steps) - 1
            mask = band == keep_idx
            grey = np.full_like(out, self.NEUTRAL)
            dimmed = np.round(
                0.25 * out.astype(np.float32) + 0.75 * grey.astype(np.float32)
            )
            dimmed = dimmed.astype(np.uint8)
            out = np.where(mask[:, :, None], out, dimmed)

        return np.ascontiguousarray(out)
