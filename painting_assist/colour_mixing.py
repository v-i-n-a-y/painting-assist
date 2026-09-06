# Copyright 2026 Vinay Williams

"""Colour-mixing helpers for matching a target colour from a limited palette.

These are pure, Qt-free functions intended to help a painter reason about how
to reach a target colour by mixing a small set of base paints, and to describe
a colour in painter's terms (value, hue, chroma, temperature, modifier).

Colour description works in OpenCV's 8-bit CIELAB space, the same convention
the Palette panel's readout uses, so the two never disagree about a colour. For
an 8-bit Lab pixel ``(L, a, b)`` with every channel on 0-255, we take

- ``value = L * 100 / 255`` (lightness on 0-100),
- ``hue = atan2(b - 128, a - 128)`` (degrees on the ``a*``-``b*`` plane), and
- ``chroma = hypot(a - 128, b - 128)`` (colourfulness, roughly 0-135).

The hue angle starts at 0 degrees along ``+a*`` (the red/pink axis) and
increases towards ``+b*`` (yellow), then ``-a*`` (green), then ``-b*`` (blue).

Important caveat on mixing: :func:`suggest_mix` uses an approximate additive
model in sRGB space. Real paint mixing is subtractive and non-linear, so treat
the suggested proportions as a rough starting guide rather than a physically
accurate recipe.
"""

from __future__ import annotations

import numpy as np

# A base palette maps a key to a human label and an ordered list of bases,
# where each base is a (name, (r, g, b)) tuple with 0-255 sRGB components.
BASE_PALETTES: dict[str, dict[str, object]] = {
    "zorn": {
        "label": "Zorn (limited earth)",
        "bases": [
            ("Titanium White", (250, 250, 245)),
            ("Yellow Ochre", (196, 145, 72)),
            ("Cadmium Red", (200, 55, 45)),
            ("Ivory Black", (32, 30, 30)),
        ],
    },
    "split_primary": {
        "label": "Split primary",
        "bases": [
            ("Titanium White", (250, 250, 245)),
            ("Lemon Yellow", (245, 230, 60)),
            ("Cadmium Yellow", (252, 196, 20)),
            ("Cadmium Red", (210, 50, 40)),
            ("Quinacridone Red", (150, 30, 55)),
            ("Ultramarine", (35, 45, 120)),
            ("Phthalo Blue", (20, 90, 140)),
        ],
    },
    "cmyw": {
        "label": "Cyan / Magenta / Yellow / White",
        "bases": [
            ("Cyan", (0, 160, 200)),
            ("Magenta", (200, 20, 120)),
            ("Yellow", (250, 220, 20)),
            ("White", (250, 250, 245)),
        ],
    },
}

# Proportions below this fraction are dropped before renormalising.
_MIX_EPSILON = 0.03

# Twelve-name hue wheel for CIELAB. Each name is paired with its hue angle in
# degrees on the OpenCV-Lab a*-b* plane (0 = +a*, red/pink; increasing towards
# +b*, yellow; then -a*, green; then -b*, blue). Lab hue is not linear in sRGB
# hue, so these reference angles are unevenly spaced and a colour is named by
# the nearest angle (measured around the circle), not by equal 30-degree
# wedges. The angles are the Lab hue of the twelve canonical fully-saturated
# sRGB hues spaced 30 degrees apart in HSV (red = (255, 0, 0),
# orange = (255, 128, 0), ... rose = (255, 0, 128)); regenerate by converting
# each with cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab). The suite checks that these
# still agree with OpenCV (test_hue_wheel_matches_opencv).
_HUE_WHEEL: list[tuple[str, float]] = [
    ("red", 39.9),
    ("orange", 59.8),
    ("yellow-orange", 83.1),
    ("yellow", 103.0),
    ("yellow-green", 128.3),
    ("green", 136.0),
    ("cyan", 196.3),
    ("azure", 285.0),
    ("blue", 306.2),
    ("violet", 311.7),
    ("magenta", 328.1),
    ("rose", 2.7),
]

# Chroma (Lab a*-b* radius) below which a colour reads as a near-neutral grey.
_NEUTRAL_CHROMA = 10.0
# Chroma at or above which a non-neutral colour reads as a pure, saturated hue.
_PURE_CHROMA = 45.0
# Value (Lab lightness, 0-100) cutoffs separating light tints from dark shades.
_TINT_VALUE = 70.0
_SHADE_VALUE = 35.0
# Warm hues run the red-orange-yellow arc, wrapping through rose and magenta;
# cool hues run green through blue to violet. The boundaries sit at the
# midpoints between yellow-green/green and violet/magenta on the Lab wheel.
_WARM_WRAP = 320.0  # magenta side; hues at or above this wrap round to warm
_WARM_END = 132.0  # yellow-green side; hues at or below this are still warm


def _as_rgb_array(rgb: tuple[int, int, int]) -> np.ndarray:
    """Return an ``(3,)`` float array of sRGB components clamped to 0-255."""
    arr = np.asarray(rgb, dtype=float)
    if arr.shape != (3,):
        raise ValueError("rgb must have exactly three components")
    return np.clip(arr, 0.0, 255.0)


def _nearest_hue_name(hue_deg: float) -> str:
    """Return the wheel name whose reference angle is closest to ``hue_deg``.

    Distance is measured around the circle, so 359 and 1 degree are two apart.
    """
    best_name = _HUE_WHEEL[0][0]
    best_dist = 360.0
    for name, angle in _HUE_WHEEL:
        diff = abs(hue_deg - angle) % 360.0
        dist = min(diff, 360.0 - diff)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _nnls(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Non-negative least squares: minimise ||matrix @ w - target|| with w >= 0.

    A compact deterministic Lawson-Hanson active-set solver. Unlike an ordinary
    least-squares fit that is then clipped, this respects the non-negativity
    constraint throughout, so a dark target resolves to the dark base rather
    than a distorted mix of light ones after clipping. Ties are broken by lowest
    index to keep the result deterministic.
    """
    n = matrix.shape[1]
    passive = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=float)
    tol = 1e-9
    # Iteration cap is a safety net; for a handful of bases this converges fast.
    for _outer in range(3 * n + 5):
        gradient = matrix.T @ (target - matrix @ weights)
        candidates = np.where(~passive)[0]
        if candidates.size == 0 or np.all(gradient[candidates] <= tol):
            break
        # Add the most promising currently-inactive base to the passive set.
        j = candidates[int(np.argmax(gradient[candidates]))]
        passive[j] = True
        for _inner in range(3 * n + 5):
            idx = np.where(passive)[0]
            sol, *_ = np.linalg.lstsq(matrix[:, idx], target, rcond=None)
            if np.all(sol > tol):
                weights[:] = 0.0
                weights[idx] = sol
                break
            # Some passive weights went non-positive; step toward the boundary.
            trial = np.zeros(n)
            trial[idx] = sol
            bad = idx[sol <= tol]
            denom = weights[bad] - trial[bad]
            safe = np.abs(denom) > tol
            if not np.any(safe):
                # Passive weight and its trial are both ~0 (collinear or
                # duplicate tube colours): nothing to step toward.
                weights[:] = 0.0
                weights[idx] = np.clip(sol, 0.0, None)
                break
            ratios = weights[bad][safe] / denom[safe]
            alpha = float(np.min(ratios))
            weights = weights + alpha * (trial - weights)
            passive[np.abs(weights) < tol] = False
    return np.clip(weights, 0.0, None)


def suggest_mix(
    target_rgb: tuple[int, int, int],
    palette_key: str = "zorn",
    bases: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> list[tuple[str, float]]:
    """Suggest an approximate mix of bases to reach ``target_rgb``.

    By default the bases come from the named built-in palette ``palette_key``
    (see :data:`BASE_PALETTES`). Pass ``bases`` as a list of
    ``(name, (r, g, b))`` tuples to mix against a painter's own tubes instead;
    when ``bases`` is given ``palette_key`` is ignored.

    The result is a list of ``(base_name, proportion)`` tuples sorted by
    descending proportion. Proportions are non-negative and sum to roughly
    1.0. Bases contributing less than a small epsilon are dropped and the
    remaining proportions renormalised.

    The method solves a non-negative least-squares fit of the target colour
    against the base-colour vectors, then renormalises the weights to sum to
    one. If every weight is zero, it falls back to the single nearest base. This
    is an APPROXIMATE additive-RGB guide, not a physically accurate subtractive
    paint recipe, and it is deterministic.

    Raises ``KeyError`` if ``palette_key`` is unknown (and no ``bases`` given),
    or ``ValueError`` if ``bases`` is given but empty.
    """
    if bases is not None:
        use_bases = list(bases)
        if not use_bases:
            raise ValueError("bases must contain at least one base colour")
    else:
        if palette_key not in BASE_PALETTES:
            raise KeyError(f"unknown palette key: {palette_key!r}")
        use_bases = BASE_PALETTES[palette_key]["bases"]  # type: ignore[assignment]

    names = [name for name, _ in use_bases]
    matrix = np.array([rgb for _, rgb in use_bases], dtype=float).T  # 3 x n
    target = _as_rgb_array(target_rgb)

    # Solve for a convex combination of the bases (weights non-negative and
    # summing to one) that best approximates the target. Mixing paints averages
    # their colours, so this is closer to reality than free scaling of a single
    # base: a near-black target then resolves to the black base rather than a
    # scaled-down white. The sum-to-one constraint is imposed as a heavily
    # weighted extra equation appended to the colour-fit system.
    penalty = 1000.0
    aug_matrix = np.vstack([matrix, np.full((1, matrix.shape[1]), penalty)])
    aug_target = np.append(target, penalty)
    weights = _nnls(aug_matrix, aug_target)

    total = weights.sum()
    if total <= 0.0:
        # Fall back to the single nearest base by Euclidean distance.
        dists = np.linalg.norm(matrix.T - target, axis=1)
        return [(names[int(np.argmin(dists))], 1.0)]

    weights = weights / total

    # Drop negligible contributions, then renormalise the survivors.
    keep = weights >= _MIX_EPSILON
    if not keep.any():
        keep = weights == weights.max()
    weights = np.where(keep, weights, 0.0)
    weights = weights / weights.sum()

    mix = [(names[i], float(weights[i])) for i in range(len(names)) if weights[i] > 0.0]
    mix.sort(key=lambda item: item[1], reverse=True)
    return mix


def describe_colour(rgb: tuple[int, int, int]) -> dict[str, object]:
    """Describe a colour in painter's terms, using OpenCV 8-bit CIELAB.

    The colour is converted to OpenCV's 8-bit Lab so the numbers match the
    Palette panel's readout exactly. Returns a dict with keys:

    - ``hex``: ``#rrggbb`` string.
    - ``value``: Lab lightness on 0-100, ``L * 100 / 255``.
    - ``hue_name``: nearest of the fixed twelve-name Lab hue wheel
      (see :data:`_HUE_WHEEL`).
    - ``chroma``: Lab colourfulness ``hypot(a - 128, b - 128)`` (roughly 0-135,
      near 0 for greys).
    - ``temperature``: ``"warm"``, ``"cool"`` or ``"neutral"`` (near-greys),
      from the Lab hue angle and a low-chroma neutral cutoff.
    - ``modifier``: ``"tint"``, ``"tone"``, ``"shade"`` or ``"pure"`` from the
      value and chroma.
    """
    import cv2

    r, g, b = _as_rgb_array(rgb)
    hex_str = "#{:02x}{:02x}{:02x}".format(int(round(r)), int(round(g)), int(round(b)))

    # One RGB pixel through OpenCV's 8-bit Lab: L on 0-255 (encodes 0-100),
    # a and b on 0-255 centred at 128. This mirrors how the panel samples Lab.
    pixel = np.array([[[r, g, b]]]).round().astype(np.uint8)
    lab_l, lab_a, lab_b = cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab)[0, 0].astype(float)

    value = float(lab_l * 100.0 / 255.0)
    da = lab_a - 128.0
    db = lab_b - 128.0
    chroma = float(np.hypot(da, db))
    hue_deg = float(np.degrees(np.arctan2(db, da)) % 360.0)

    hue_name = _nearest_hue_name(hue_deg)

    # Temperature: near-greys read neutral; otherwise warm across the
    # red-orange-yellow arc (wrapping through rose/magenta), cool elsewhere.
    if chroma < _NEUTRAL_CHROMA:
        temperature = "neutral"
    elif hue_deg >= _WARM_WRAP or hue_deg <= _WARM_END:
        temperature = "warm"
    else:
        temperature = "cool"

    # Modifier: a near-grey is a fully greyed tone; a strongly chromatic colour
    # is a pure hue; otherwise a light colour is a tint, a dark one a shade, and
    # a muted mid-value one a tone.
    if chroma < _NEUTRAL_CHROMA:
        modifier = "tone"
    elif chroma >= _PURE_CHROMA:
        modifier = "pure"
    elif value >= _TINT_VALUE:
        modifier = "tint"
    elif value <= _SHADE_VALUE:
        modifier = "shade"
    else:
        modifier = "tone"

    return {
        "hex": hex_str,
        "value": value,
        "hue_name": hue_name,
        "chroma": chroma,
        "temperature": temperature,
        "modifier": modifier,
    }
