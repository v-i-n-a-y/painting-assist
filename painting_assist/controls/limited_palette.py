# Copyright 2026 Vinay Williams

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np

from painting_assist import palette_map
from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register

RGB = Tuple[int, int, int]


def _hex_to_rgb(value: object) -> Optional[RGB]:
    """Parse a ``#rrggbb`` (or ``rrggbb``) string to an (r, g, b) triple, or None."""
    if not isinstance(value, str):
        return None
    s = value.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return None


def parse_hex_list(text: object) -> List[RGB]:
    """Parse a JSON array of hex colours to a list of (r, g, b), dropping bad entries."""
    try:
        data = json.loads(text) if text else []
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: List[RGB] = []
    for item in data:
        rgb = _hex_to_rgb(item)
        if rgb is not None:
            out.append(rgb)
    return out


@register
class LimitedPaletteControl(Control):
    """Limited palette - repaint the reference with only colours you can mix.

    Every pixel is snapped to the nearest colour that could actually be MIXED
    from a chosen set of paint tubes, using the physically-plausible pigment
    mixing model (mixbox / Kubelka-Munk). It answers the painter's real question
    before starting: "can I even mix this with the paints I have?", and turns a
    photo into a plan that respects a deliberately limited palette.

    The palette comes from one of three sources:

    * **Preset** - a classic limited palette (Zorn, earth, primary triad, split
      primary), built in.
    * **My paints** - the tubes recorded in My Paints (injected by the window
      into ``paints_json`` so the worker stays self-contained).
    * **Sampled** - colours picked from the image or a colour dialog, held in
      ``samples_json``.

    The heavy lifting (building the mixable gamut and the nearest-colour lookup)
    lives in :mod:`painting_assist.palette_map`; this control only resolves which
    tubes to use and calls it.
    """

    id = "limited_palette"
    name = "Limited palette"
    order = 16  # after Colour groups (15), before Values (20)

    # Built-in limited palettes as ordered lists of RGB tubes. Masstone values
    # mirror painting_assist.paints so a preset reads like real tubes.
    PRESETS = {
        "zorn": [(250, 250, 245), (196, 145, 72), (200, 40, 40), (32, 30, 30)],
        "earth": [
            (250, 250, 245),
            (196, 145, 72),
            (138, 54, 34),
            (78, 62, 40),
            (32, 30, 30),
        ],
        "primary": [
            (250, 250, 245),
            (255, 199, 27),
            (200, 40, 40),
            (40, 50, 130),
            (32, 30, 30),
        ],
        "split_primary": [
            (250, 250, 245),
            (32, 30, 30),
            (255, 199, 27),
            (245, 230, 80),
            (200, 40, 40),
            (180, 30, 90),
            (40, 50, 130),
            (15, 70, 130),
        ],
    }

    PRESET_CHOICES = [
        ("zorn", "Zorn (ochre / red / black / white)"),
        ("earth", "Earth"),
        ("primary", "Primary triad"),
        ("split_primary", "Split primary"),
    ]

    SOURCE_CHOICES = [
        ("preset", "Preset palette"),
        ("my_paints", "My Paints"),
        ("sampled", "Sampled colours"),
    ]

    @classmethod
    def params(cls) -> List[Param]:
        """Schema: palette source, the preset choice, and the two injected blobs."""
        return [
            Param(
                name="source",
                label="Palette from",
                ptype=ParamType.CHOICE,
                default="preset",
                choices=cls.SOURCE_CHOICES,
                tooltip=(
                    "Where the palette comes from: a built-in preset, your My "
                    "Paints tubes, or colours you sample from the image."
                ),
            ),
            Param(
                name="preset",
                label="Preset",
                ptype=ParamType.CHOICE,
                default="zorn",
                choices=cls.PRESET_CHOICES,
                tooltip="Which built-in limited palette to mix from (Preset source).",
            ),
            # Injected by the window from the My Paints inventory / the sampled
            # swatches; hidden from the generic UI (the editor manages them).
            Param(
                name="paints_json",
                label="My paints",
                ptype=ParamType.TEXT,
                default="[]",
            ),
            Param(
                name="samples_json",
                label="Sampled",
                ptype=ParamType.TEXT,
                default="[]",
            ),
        ]

    # ------------------------------------------------------------------ #
    def tubes(self) -> List[RGB]:
        """Resolve the active palette (list of RGB tubes) from the current source."""
        source = self.get("source")
        if source == "my_paints":
            return parse_hex_list(self.get("paints_json"))
        if source == "sampled":
            return parse_hex_list(self.get("samples_json"))
        return list(self.PRESETS.get(self.get("preset"), self.PRESETS["zorn"]))

    def is_active(self) -> bool:
        """Active only when enabled and the resolved palette has at least one tube."""
        return self.enabled and bool(self.tubes())

    def process(self, img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> a new array repainted from the mixable gamut."""
        tubes = self.tubes()
        if not tubes:
            return img.copy()
        return palette_map.simulate(img, tubes)

    def create_editor(self, parent=None):
        """Build the limited-palette editor (source picker + palette swatches)."""
        from painting_assist.widgets.limited_palette_editor import (
            LimitedPaletteEditor,
        )

        return LimitedPaletteEditor(self, parent)
