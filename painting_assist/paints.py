# Copyright 2026 Vinay Williams

"""A curated catalogue of common artist paints and pure serialisation helpers.

The catalogue is a plain list of ``(name, (r, g, b))`` tuples with representative
sRGB values for widely stocked tube colours. It is deliberately Qt-free and
side-effect-free so it can seed a palette, back a "paints I own" list, or serve
as the pool the mixing engine scans when suggesting a paint to buy.

The sRGB triples are approximations of each pigment's masstone, chosen to be
sensible mixing primaries rather than measured spectra. They are good enough for
the pigment-mixing engine in :mod:`painting_assist.mixing`, which reasons in a
Kubelka-Munk latent space, not for colour-critical reproduction.

:func:`paints_to_json` and :func:`paints_from_json` round-trip such a list
through a JSON string. Both are robust to malformed input: bad entries are
dropped rather than raised on, names are coerced to strings, and colour
components are clamped to integer 0-255 triples.
"""

from __future__ import annotations

import json

# A curated set of common artist pigments as (name, (r, g, b)) with
# representative sRGB masstone values. Ordered light-to-dark within each family
# (whites, yellows, earths, oranges/reds, blues, greens, violets, greys) so a UI
# can present them in a painterly sequence.
DEFAULT_CATALOGUE: list[tuple[str, tuple[int, int, int]]] = [
    ("Titanium White", (250, 250, 245)),
    ("Naples Yellow", (250, 218, 130)),
    ("Cadmium Lemon", (255, 241, 78)),
    ("Cadmium Yellow", (255, 199, 27)),
    ("Yellow Ochre", (196, 145, 72)),
    ("Raw Sienna", (150, 100, 45)),
    ("Raw Umber", (78, 62, 40)),
    ("Burnt Sienna", (138, 54, 34)),
    ("Burnt Umber", (72, 46, 34)),
    ("Cadmium Orange", (240, 110, 30)),
    ("Cadmium Red Light", (220, 60, 35)),
    ("Cadmium Red", (200, 40, 40)),
    ("Alizarin Crimson", (140, 30, 45)),
    ("Quinacridone Magenta", (160, 30, 85)),
    ("Cerulean Blue", (30, 120, 175)),
    ("Cobalt Blue", (30, 70, 150)),
    ("Ultramarine Blue", (40, 50, 130)),
    ("Phthalo Blue", (15, 70, 130)),
    ("Phthalo Green", (0, 110, 95)),
    ("Viridian", (30, 115, 95)),
    ("Sap Green", (75, 110, 45)),
    ("Cobalt Violet", (120, 70, 130)),
    ("Dioxazine Purple", (60, 30, 75)),
    ("Payne's Grey", (38, 52, 66)),
    ("Ivory Black", (32, 30, 30)),
]


def _coerce_rgb(value: object) -> tuple[int, int, int] | None:
    """Coerce ``value`` to an integer 0-255 sRGB triple, or ``None`` if it cannot.

    Accepts any length-3 sequence of number-like components. Each component is
    rounded to the nearest integer and clamped to 0-255. Non-numeric components
    (or a wrong-length value) make the whole triple invalid.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return None
    if len(value) != 3:
        return None
    out: list[int] = []
    for component in value:
        try:
            number = int(round(float(component)))
        except (TypeError, ValueError):
            return None
        out.append(max(0, min(255, number)))
    return (out[0], out[1], out[2])


def _coerce_entry(entry: object) -> tuple[str, tuple[int, int, int]] | None:
    """Coerce one serialised entry to ``(name, rgb)``, or ``None`` if malformed.

    Two shapes are accepted: the canonical ``{"name": str, "rgb": [r, g, b]}``
    object, and a bare ``[name, [r, g, b]]`` pair. The name is coerced to a
    non-empty string; the colour is passed through :func:`_coerce_rgb`.
    """
    if isinstance(entry, dict):
        name = entry.get("name")
        rgb = entry.get("rgb")
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        name, rgb = entry
    else:
        return None
    if name is None:
        return None
    name = str(name)
    if not name:
        return None
    coerced = _coerce_rgb(rgb)
    if coerced is None:
        return None
    return (name, coerced)


def paints_to_json(paints: list[tuple[str, tuple[int, int, int]]]) -> str:
    """Serialise a list of ``(name, (r, g, b))`` paints to a JSON string.

    Entries are coerced through :func:`_coerce_entry`, so a malformed entry is
    dropped rather than raised on and the output is always valid JSON: a list of
    ``{"name": str, "rgb": [r, g, b]}`` objects that :func:`paints_from_json`
    reads back.
    """
    out: list[dict[str, object]] = []
    for entry in paints:
        coerced = _coerce_entry(entry)
        if coerced is not None:
            name, rgb = coerced
            out.append({"name": name, "rgb": list(rgb)})
    return json.dumps(out)


def paints_from_json(text: str) -> list[tuple[str, tuple[int, int, int]]]:
    """Parse a JSON string back into a list of ``(name, (r, g, b))`` paints.

    Robust to malformed input: invalid JSON, a non-list top level, and any
    individual bad entry are handled by dropping the offending data and never
    raising. Names come back as strings and colours as clamped integer 0-255
    triples.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    result: list[tuple[str, tuple[int, int, int]]] = []
    for entry in data:
        coerced = _coerce_entry(entry)
        if coerced is not None:
            result.append(coerced)
    return result
