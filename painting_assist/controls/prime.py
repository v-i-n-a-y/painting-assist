# Copyright 2026 Vinay Williams

from __future__ import annotations

from typing import List

import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register
from painting_assist.priming import TECHNIQUES


@register
class PrimeControl(Control):
    """Priming — recommend a canvas ground colour from the reference.

    Analyses the processed reference for its majority colour and re-expresses
    it per painting technique (mid-tone dead colour, majority tint,
    complementary ground, light/dark ground, neutral grey) so the painter can
    prime the canvas with a colour that suits both the technique and the
    painting's dominant tone.

    Like the grid, this is an analysis tool, not a pixel change:
    :meth:`process` is the identity and :meth:`is_active` always returns
    ``False`` so the pipeline skips the control entirely (its params never
    churn the render cache). The window computes the recommendation from each
    rendered frame via :func:`painting_assist.priming.prime_colour` and pushes
    it to the editor.
    """

    id = "prime"
    name = "Priming"
    order = 95  # after grid=90; analysis tools sit at the bottom of the panel

    @classmethod
    def params(cls) -> List[Param]:
        """Technique choice plus how strongly the ground takes the tint."""
        return [
            Param(
                name="technique",
                label="Technique",
                ptype=ParamType.CHOICE,
                default="midtone",
                choices=TECHNIQUES,
                tooltip=(
                    "How the ground colour is derived from the reference's "
                    "majority colour."
                ),
            ),
            Param(
                name="strength",
                label="Tint strength",
                ptype=ParamType.INT,
                default=50,
                minimum=0,
                maximum=100,
                step=5,
                suffix=" %",
                tooltip=(
                    "How much of the majority colour's chroma the ground keeps "
                    "(0 = neutral grey, 100 = full majority chroma)."
                ),
            ),
        ]

    def is_active(self) -> bool:
        """Always ``False``: priming is a recommendation, never a pixel change.

        Returning ``False`` unconditionally makes the pipeline skip this
        control entirely, so its params never invalidate the render cache.
        """
        return False

    def process(self, img: np.ndarray) -> np.ndarray:
        """Identity: the recommendation is computed by the window, not baked."""
        return img

    def create_editor(self, parent: object = None):
        """Build the priming editor (technique, strength, swatch readout)."""
        from painting_assist.widgets.prime_editor import PrimeEditor

        return PrimeEditor(self, parent)
