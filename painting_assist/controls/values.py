# Copyright 2026 Vinay Williams

from __future__ import annotations

import math
from typing import List, Tuple

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

    Three modes:

    * **Greyscale** replaces every pixel's colour with a neutral grey of the
      same perceptual lightness (the CIELab L channel), so the reference reads
      as a pure value study.
    * **Value steps** posterizes lightness into a handful of flat value bands
      — the classic "notan"/limited-value approach — optionally keeping the
      original colour within each band (value-grouped colour) rather than
      going neutral.
    * **Monochromatic** keeps the full value structure but stains it with a
      single pigment hue — a brunaille/imprimatura underpainting in one
      colour (burnt umber, Payne's grey, and so on). Value is preserved
      exactly; the chosen hue is strongest through the midtones and fades out
      toward the darkest darks and lightest lights (where real pigment can
      hold little chroma), so the painter can lay in a tonal underpainting in
      a single colour while still reading every value.

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

    # Full-strength chroma (CIELab a/b units) applied by Monochromatic mode at
    # the midtones; scaled down by the Tint knob and by a per-pixel taper that
    # falls to zero at pure black/white.
    MONO_CHROMA = 45.0

    # Built-in single-pigment underpainting colours, as (label, RGB). Only the
    # hue *direction* (the Lab a/b angle) is used, so exact values need not be
    # colorimetric — just a sensible warm/cool spread of the classic imprimatura
    # pigments. Burnt umber is the default (the traditional brunaille brown).
    # The painter can also pick any custom colour or one of their own paint
    # tubes; whatever the choice, the effective colour is stored as a hex string
    # in the ``mono_hex`` param so it travels with the worker snapshot self-
    # contained (no registry lookup on the worker thread).
    MONO_PRESETS = [
        ("Burnt umber", (110, 65, 40)),
        ("Raw umber", (112, 92, 58)),
        ("Burnt sienna", (150, 78, 45)),
        ("Payne's grey", (58, 74, 92)),
        ("Green earth", (92, 108, 80)),
        ("Indigo", (48, 60, 96)),
    ]
    DEFAULT_MONO_HEX = "#6e4128"  # burnt umber (110, 65, 40)

    @classmethod
    def params(cls) -> List[Param]:
        """Schema for mode, value-step count, colour-keeping and isolation."""
        return [
            Param(
                name="mode",
                label="Mode",
                ptype=ParamType.CHOICE,
                default="grey",
                choices=[
                    ("grey", "Greyscale"),
                    ("posterize", "Value steps"),
                    ("mono", "Monochromatic"),
                ],
                tooltip=(
                    "Greyscale: replace colour with neutral grey of the same "
                    "lightness. Value steps: posterize lightness into flat "
                    "value bands (notan). Monochromatic: keep the value "
                    "structure but stain it with a single pigment hue."
                ),
            ),
            Param(
                name="mono_hex",
                label="Mono colour",
                ptype=ParamType.TEXT,
                default=cls.DEFAULT_MONO_HEX,
                tooltip=(
                    "Monochromatic mode: the single pigment hue the value "
                    "study is stained with (a preset, one of your paints, or a "
                    "custom colour)."
                ),
            ),
            Param(
                name="tint",
                label="Tint strength",
                ptype=ParamType.FLOAT,
                default=0.7,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                tooltip=(
                    "Monochromatic mode: how saturated the pigment stain is. "
                    "0 = neutral grey, 1 = full pigment."
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

    def create_editor(self, parent=None):
        """Return the custom Values editor (colour picker for Monochromatic mode).

        Qt is imported lazily so this control module stays Qt-free for headless
        processing/testing; only building the editor pulls in the widget.
        """
        from painting_assist.widgets.values_editor import ValuesEditor

        return ValuesEditor(self, parent)

    @staticmethod
    def parse_hex(text: object) -> Tuple[int, int, int]:
        """Parse ``#rrggbb`` (or ``rrggbb``) to an (r, g, b) 0-255 triple.

        Robust to malformed input (used on session restore and free-form param
        values): anything that is not a valid 6-digit hex colour falls back to
        burnt umber, so :meth:`process` always has a usable pigment.
        """
        s = str(text).strip().lstrip("#")
        if len(s) == 6:
            try:
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
            except ValueError:
                pass
        return ValuesControl.MONO_PRESETS[0][1]

    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Return ``#rrggbb`` for an (r, g, b) triple (components clamped 0-255)."""
        r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
        return "#%02x%02x%02x" % (r, g, b)

    @classmethod
    def _mono_ab_luts(
        cls, rgb: Tuple[int, int, int], tint: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (a_lut, b_lut): 256-entry uint8 CIELab a/b maps keyed by L.

        The pigment fixes the hue *direction* (its Lab a/b angle); the chroma at
        each lightness is ``MONO_CHROMA * tint * sin(pi * L/255)`` so it peaks in
        the midtones and tapers to neutral at pure black and pure white, where
        real pigment holds no colour. A neutral/grey pigment yields flat grey.
        """
        pigment = np.array([[tuple(int(c) for c in rgb)]], dtype=np.uint8)
        lab = cv2.cvtColor(pigment, cv2.COLOR_RGB2Lab)[0, 0]
        da = float(lab[1]) - cls.NEUTRAL
        db = float(lab[2]) - cls.NEUTRAL
        norm = math.hypot(da, db)
        nx, ny = (0.0, 0.0) if norm < 1e-6 else (da / norm, db / norm)

        levels = np.arange(256.0)
        chroma = cls.MONO_CHROMA * tint * np.sin(np.pi * levels / 255.0)
        a_lut = np.clip(np.round(cls.NEUTRAL + nx * chroma), 0, 255).astype(np.uint8)
        b_lut = np.clip(np.round(cls.NEUTRAL + ny * chroma), 0, 255).astype(np.uint8)
        return a_lut, b_lut

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
        mono_rgb = self.parse_hex(self.get("mono_hex"))
        tint = max(0.0, min(1.0, float(self.get("tint"))))

        lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2Lab)
        L = np.ascontiguousarray(lab[:, :, 0])  # uint8 lightness, 0..255

        # Even bands over the full 0-255 lightness range. Interior edges feed
        # np.digitize; band index is 0..steps-1. Because L is uint8 there are
        # only 256 possible inputs, so the L->band map is a 256-entry lookup
        # table, shared by posterize value-mapping and (grey- or posterize-mode)
        # band isolation. Computing it once over the 256 levels avoids a
        # float digitize over every pixel.
        edges = np.linspace(0.0, 255.0, steps + 1)
        levels = np.arange(256.0)  # every possible L value
        band_lut = np.clip(np.digitize(levels, edges[1:-1]), 0, steps - 1).astype(
            np.int32
        )  # 256 entries: band index per L value

        if mode == "posterize":
            # Map each band to the mean L of its members (deterministic); empty
            # bands fall back to their geometric centre. The per-band sum/count
            # come from grouping the 256-bin L histogram by band, so no
            # full-image float pass is needed.
            centres = (edges[:-1] + edges[1:]) * 0.5
            hist = np.bincount(L.reshape(-1), minlength=256).astype(np.float64)
            sums = np.bincount(band_lut, weights=hist * levels, minlength=steps)
            counts = np.bincount(band_lut, weights=hist, minlength=steps)
            means = np.where(counts > 0, sums / np.maximum(counts, 1.0), centres)
            # L->new_L collapses to a 256-entry uint8 LUT applied with cv2.LUT.
            new_L_lut = np.clip(np.round(means[band_lut]), 0, 255).astype(np.uint8)
            out_lab = lab.copy()
            out_lab[:, :, 0] = cv2.LUT(L, new_L_lut)
            if not keep_colour:
                out_lab[:, :, 1] = self.NEUTRAL
                out_lab[:, :, 2] = self.NEUTRAL
            out = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)
        elif mode == "mono":
            # Preserve value exactly (L untouched) and drive the a/b channels
            # purely as a function of L, so every pixel of a given value gets
            # the same single hue. The hue direction comes from the pigment;
            # the amount is Tint * a midtone-peaked taper, so chroma vanishes at
            # pure black/white. Both channels collapse to 256-entry LUTs.
            a_lut, b_lut = self._mono_ab_luts(mono_rgb, tint)
            out_lab = lab.copy()
            out_lab[:, :, 1] = cv2.LUT(L, a_lut)
            out_lab[:, :, 2] = cv2.LUT(L, b_lut)
            out = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)
        else:  # "grey" (and any unknown value falls back to greyscale)
            out_lab = lab.copy()
            out_lab[:, :, 1] = self.NEUTRAL
            out_lab[:, :, 2] = self.NEUTRAL
            out = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)

        if isolate > 0:
            # Keep the chosen band (clamped to what's available), dim the rest
            # by blending 75% toward flat mid-grey so the band pops. Band
            # membership per pixel comes from the same L->band LUT, and the dim
            # is one saturating pass (cv2.addWeighted) rather than several
            # full-image float temporaries.
            keep_idx = min(isolate, steps) - 1
            keep_lut = (band_lut == keep_idx).astype(np.uint8)  # 256-entry 0/1
            mask = cv2.LUT(L, keep_lut).astype(bool)
            grey = np.full_like(out, self.NEUTRAL)
            dimmed = cv2.addWeighted(out, 0.25, grey, 0.75, 0.0)
            out = np.where(mask[:, :, None], out, dimmed)

        return np.ascontiguousarray(out)
