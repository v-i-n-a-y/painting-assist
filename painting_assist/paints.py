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

# A broad catalogue of artists' oil colours as (name, (r, g, b)) with
# representative sRGB masstone values (the full-strength tube colour, which is
# what the pigment-mixing engine blends). It spans the colours a supplier such as
# Jackson's carries across the major ranges (Winsor & Newton, Michael Harding,
# Gamblin, Old Holland, Sennelier, Williamsburg, Daniel Smith, Rembrandt,
# Schmincke), de-duplicated to the distinct pigment colours a painter mixes from
# rather than every near-identical brand SKU. Values come from well-known pigment
# masstones and standard references, not measured spectra, and are grouped by
# colour family. Titanium White stays first (a serialisation test checks it).
DEFAULT_CATALOGUE: list[tuple[str, tuple[int, int, int]]] = [
    # --- Whites and near-neutral tints -----------------------------------
    ("Titanium White", (250, 250, 245)),
    ("Zinc White", (250, 250, 248)),
    ("Flake White", (250, 248, 238)),
    ("Cremnitz White", (250, 247, 235)),
    ("Titanium Buff", (222, 205, 170)),
    ("Unbleached Titanium", (214, 195, 164)),
    # --- Yellows ---------------------------------------------------------
    ("Cadmium Lemon", (255, 241, 78)),
    ("Lemon Yellow", (245, 230, 80)),
    ("Bismuth Yellow", (250, 236, 90)),
    ("Hansa Yellow Light", (252, 222, 40)),
    ("Cadmium Yellow Pale", (255, 225, 40)),
    ("Cadmium Yellow", (255, 199, 27)),
    ("Hansa Yellow Medium", (252, 205, 30)),
    ("Cadmium Yellow Deep", (250, 170, 20)),
    ("Aureolin", (245, 200, 40)),
    ("Indian Yellow", (240, 160, 30)),
    ("Transparent Yellow", (235, 170, 20)),
    ("Nickel Titanate Yellow", (238, 220, 120)),
    ("Naples Yellow", (250, 218, 130)),
    ("Naples Yellow Deep", (240, 190, 110)),
    ("Chrome Yellow", (250, 200, 40)),
    # --- Ochres and earth yellows ----------------------------------------
    ("Yellow Ochre", (196, 145, 72)),
    ("Yellow Ochre Light", (210, 165, 90)),
    ("Gold Ochre", (190, 135, 55)),
    ("Mars Yellow", (200, 140, 60)),
    ("Raw Sienna", (150, 100, 45)),
    ("Transparent Oxide Yellow", (175, 120, 45)),
    # --- Oranges ---------------------------------------------------------
    ("Cadmium Orange", (240, 110, 30)),
    ("Cadmium Orange Deep", (230, 90, 25)),
    ("Pyrrole Orange", (240, 80, 35)),
    ("Perinone Orange", (235, 90, 40)),
    ("Transparent Orange", (230, 100, 25)),
    ("Mars Orange", (200, 90, 45)),
    # --- Cadmium and modern reds -----------------------------------------
    ("Cadmium Scarlet", (225, 60, 40)),
    ("Vermilion", (225, 60, 45)),
    ("Cadmium Red Light", (220, 60, 35)),
    ("Cadmium Red", (200, 40, 40)),
    ("Cadmium Red Deep", (170, 35, 40)),
    ("Pyrrole Red", (200, 40, 45)),
    ("Naphthol Red", (190, 35, 45)),
    ("Winsor Red", (200, 40, 42)),
    ("Permanent Red", (195, 40, 45)),
    ("Scarlet Lake", (210, 50, 40)),
    # --- Earth reds ------------------------------------------------------
    ("Light Red", (170, 80, 60)),
    ("English Red", (165, 75, 55)),
    ("Venetian Red", (150, 65, 50)),
    ("Indian Red", (120, 60, 55)),
    ("Terra Rosa", (160, 80, 65)),
    ("Red Ochre", (150, 70, 55)),
    ("Mars Red", (140, 55, 50)),
    ("Caput Mortuum", (95, 55, 55)),
    # --- Crimsons, magentas and roses ------------------------------------
    ("Alizarin Crimson", (140, 30, 45)),
    ("Permanent Alizarin Crimson", (145, 30, 48)),
    ("Carmine", (150, 25, 55)),
    ("Crimson Lake", (150, 25, 50)),
    ("Quinacridone Red", (170, 30, 60)),
    ("Quinacridone Rose", (180, 30, 90)),
    ("Quinacridone Magenta", (160, 30, 85)),
    ("Permanent Rose", (200, 40, 100)),
    ("Rose Madder", (170, 60, 80)),
    ("Magenta", (170, 30, 95)),
    ("Ruby Red", (150, 25, 70)),
    # --- Violets and purples ---------------------------------------------
    ("Cobalt Violet", (120, 70, 130)),
    ("Cobalt Violet Deep", (95, 55, 110)),
    ("Manganese Violet", (110, 55, 110)),
    ("Ultramarine Violet", (85, 60, 130)),
    ("Quinacridone Violet", (110, 35, 80)),
    ("Dioxazine Purple", (60, 30, 75)),
    ("Mauve", (95, 60, 120)),
    ("Mineral Violet", (110, 70, 135)),
    # --- Blues -----------------------------------------------------------
    ("King's Blue", (110, 150, 195)),
    ("Cerulean Blue", (30, 120, 175)),
    ("Manganese Blue", (25, 130, 170)),
    ("Cobalt Turquoise", (20, 130, 150)),
    ("Cobalt Blue", (30, 70, 150)),
    ("Cobalt Blue Deep", (25, 55, 130)),
    ("Ultramarine Blue", (40, 50, 130)),
    ("French Ultramarine", (35, 45, 120)),
    ("Ultramarine Blue Deep", (30, 40, 110)),
    ("Phthalo Blue", (15, 70, 130)),
    ("Phthalo Blue Red Shade", (30, 50, 125)),
    ("Prussian Blue", (20, 50, 75)),
    ("Indanthrone Blue", (35, 45, 90)),
    ("Indigo", (35, 45, 70)),
    # --- Greens ----------------------------------------------------------
    ("Phthalo Green", (0, 110, 95)),
    ("Phthalo Green Yellow Shade", (20, 120, 80)),
    ("Prussian Green", (20, 85, 75)),
    ("Viridian", (30, 115, 95)),
    ("Emerald Green", (30, 150, 110)),
    ("Cobalt Green", (60, 140, 110)),
    ("Cobalt Green Deep", (30, 110, 85)),
    ("Permanent Green Light", (90, 170, 80)),
    ("Permanent Green", (40, 130, 70)),
    ("Cadmium Green", (90, 150, 70)),
    ("Hooker's Green", (45, 95, 60)),
    ("Sap Green", (75, 110, 45)),
    ("Olive Green", (95, 100, 55)),
    ("Chromium Oxide Green", (95, 120, 80)),
    ("Terre Verte", (100, 120, 95)),
    ("Green Earth", (110, 125, 100)),
    # --- Browns and earths -----------------------------------------------
    ("Raw Umber", (78, 62, 40)),
    ("Raw Umber Light", (110, 90, 60)),
    ("Burnt Sienna", (138, 54, 34)),
    ("Burnt Umber", (72, 46, 34)),
    ("Burnt Umber Light", (110, 70, 50)),
    ("Brown Ochre", (135, 90, 50)),
    ("Transparent Oxide Red", (120, 55, 35)),
    ("Transparent Oxide Brown", (95, 55, 35)),
    ("Vandyke Brown", (65, 45, 35)),
    ("Sepia", (75, 55, 45)),
    ("Bistre", (90, 65, 45)),
    ("Mars Brown", (100, 60, 40)),
    # --- Greys and blacks ------------------------------------------------
    ("Davy's Grey", (110, 110, 100)),
    ("Neutral Grey", (128, 128, 128)),
    ("Ultramarine Grey", (120, 125, 135)),
    ("Payne's Grey", (38, 52, 66)),
    ("Ivory Black", (32, 30, 30)),
    ("Mars Black", (28, 28, 28)),
    ("Lamp Black", (25, 25, 28)),
    ("Blue Black", (30, 32, 38)),
    ("Vine Black", (35, 33, 33)),
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
        except (TypeError, ValueError, OverflowError):
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
