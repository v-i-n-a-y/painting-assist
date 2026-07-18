# Copyright 2026 Vinay Williams

"""Colour-mixing helpers for matching a target colour from a limited palette.

These are pure, Qt-free functions intended to help a painter reason about how
to reach a target colour by mixing a small set of base paints, and to describe
a colour in painter's terms (value, hue, chroma, temperature, modifier).

Important caveat on mixing: :func:`suggest_mix` uses an approximate additive
model in sRGB space. Real paint mixing is subtractive and non-linear, so treat
the suggested proportions as a rough starting guide rather than a physically
accurate recipe.
"""

from __future__ import annotations

import colorsys

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

# Twelve-name hue wheel, ordered by increasing hue angle starting at red (0).
_HUE_NAMES = [
    "red",
    "orange",
    "yellow-orange",
    "yellow",
    "yellow-green",
    "green",
    "cyan",
    "azure",
    "blue",
    "violet",
    "magenta",
    "rose",
]


def _as_rgb_array(rgb: tuple[int, int, int]) -> np.ndarray:
    """Return an ``(3,)`` float array of sRGB components clamped to 0-255."""
    arr = np.asarray(rgb, dtype=float)
    if arr.shape != (3,):
        raise ValueError("rgb must have exactly three components")
    return np.clip(arr, 0.0, 255.0)


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
            ratios = weights[bad] / (weights[bad] - trial[bad])
            alpha = float(np.min(ratios))
            weights = weights + alpha * (trial - weights)
            passive[np.abs(weights) < tol] = False
    return np.clip(weights, 0.0, None)


def suggest_mix(
    target_rgb: tuple[int, int, int],
    palette_key: str,
) -> list[tuple[str, float]]:
    """Suggest an approximate mix of palette bases to reach ``target_rgb``.

    The result is a list of ``(base_name, proportion)`` tuples sorted by
    descending proportion. Proportions are non-negative and sum to roughly
    1.0. Bases contributing less than a small epsilon are dropped and the
    remaining proportions renormalised.

    The method solves a non-negative least-squares fit of the target colour
    against the base-colour vectors, then renormalises the weights to sum to
    one. If every weight is zero, it falls back to the single nearest base. This
    is an APPROXIMATE additive-RGB guide, not a physically accurate subtractive
    paint recipe, and it is deterministic.
    """
    if palette_key not in BASE_PALETTES:
        raise KeyError(f"unknown palette key: {palette_key!r}")

    bases = BASE_PALETTES[palette_key]["bases"]  # type: ignore[index]
    names = [name for name, _ in bases]
    matrix = np.array([rgb for _, rgb in bases], dtype=float).T  # 3 x n
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
    """Describe a colour in painter's terms.

    Returns a dict with keys:

    - ``hex``: ``#rrggbb`` string.
    - ``value``: perceptual luma on 0-100, from ``0.299r + 0.587g + 0.114b``
      (the Rec. 601 luma weights), scaled from 0-255 to 0-100.
    - ``hue_name``: nearest of a fixed twelve-name hue wheel.
    - ``chroma``: HSV saturation times value, on 0-100.
    - ``temperature``: ``"warm"``, ``"cool"`` or ``"neutral"`` (near-greys).
    - ``modifier``: ``"tint"``, ``"tone"``, ``"shade"`` or ``"pure"`` from
      value and chroma heuristics.
    """
    r, g, b = _as_rgb_array(rgb)
    hex_str = "#{:02x}{:02x}{:02x}".format(int(round(r)), int(round(g)), int(round(b)))

    value = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0 * 100.0

    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue_deg = h * 360.0
    chroma = s * v * 100.0

    # Nearest of twelve evenly spaced hue names (30 degrees apart).
    idx = int(round(hue_deg / 30.0)) % len(_HUE_NAMES)
    hue_name = _HUE_NAMES[idx]

    # Temperature: warm roughly from red through yellow into yellow-green,
    # cool otherwise; near-greys (low chroma) read as neutral.
    if chroma < 10.0:
        temperature = "neutral"
    elif hue_deg <= 150.0 or hue_deg >= 330.0:
        temperature = "warm"
    else:
        temperature = "cool"

    # Modifier heuristics from value and chroma.
    if chroma < 10.0:
        modifier = "tone"
    elif value >= 70.0:
        modifier = "tint"
    elif value <= 35.0:
        modifier = "shade"
    elif chroma >= 55.0:
        modifier = "pure"
    else:
        modifier = "tone"

    return {
        "hex": hex_str,
        "value": float(value),
        "hue_name": hue_name,
        "chroma": float(chroma),
        "temperature": temperature,
        "modifier": modifier,
    }
