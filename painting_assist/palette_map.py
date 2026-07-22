# Copyright 2026 Vinay Williams

"""Limited-palette / gamut simulation: re-render a photo in mixable colours.

This module answers "what would this photo look like if I could only use the
paints I own?". It builds the *gamut* a painter can reach from a chosen set of
paint tubes (their colours plus physically-plausible mixes of them), then maps
every pixel of a reference photo to the nearest reachable colour.

Pigment mixing uses `mixbox <https://github.com/scrtwpns/mixbox>`_, the same
Kubelka-Munk latent space :mod:`painting_assist.mixing` uses: a convex
combination of tube latents, mapped back to sRGB, is a physically-plausible mix
(blue plus yellow gives green, not grey). If mixbox is unavailable the gamut is
just the tube colours themselves, with no mixing.

The module is deliberately Qt-free and fully deterministic: it never calls any
random source, so the same tubes and image always give the same result. Nearest
colour is measured perceptually in true CIELAB via OpenCV's 8-bit Lab, matching
the rest of the app's colour readouts. Mapping goes through a coarse 3D RGB
lookup table rather than a per-pixel search so it stays fast on large images.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

try:
    import mixbox

    _HAVE_MIXBOX = True
except Exception:  # pragma: no cover - mixbox is a declared dependency
    mixbox = None
    _HAVE_MIXBOX = False

# Pairwise mixes are sampled at these convex blend fractions ``t`` (tube A at
# weight ``t``, tube B at ``1 - t``); 0.5 is the even mix, 0.25/0.75 the two
# lopsided ones. Triples are sampled only at equal thirds.
_PAIR_FRACTIONS = (0.25, 0.5, 0.75)

# Combinatorial caps that keep candidate generation from exploding on a big
# palette while staying deterministic:
#   * triples (O(N^3)) are only formed when the palette has at most this many
#     tubes;
#   * once the palette is larger than the pair cap, pairs (O(N^2)) are sampled
#     at the even fraction (0.5) only, instead of all three fractions.
_TRIPLE_TUBE_CAP = 8
_PAIR_FRACTION_CAP = 40

# Default lookup-table resolution: this many levels per RGB channel (bins of
# 8 across 0-255). L**3 = 33**3 ~ 36k bin centres, cheap to classify.
_DEFAULT_LUT_LEVELS = 33


def _clamp_round_rgb(rgb) -> tuple[int, int, int]:
    """Return ``rgb`` as an integer triple with each component clamped to 0-255."""
    r, g, b = (int(round(float(component))) for component in rgb)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _rgb_to_lab(colours: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 3)`` uint8 RGB array to ``(N, 3)`` float CIELAB.

    Uses OpenCV's 8-bit Lab (the convention :func:`painting_assist.mixing.deltae`
    uses) rescaled to true CIELAB units: ``L`` on 0-100 and ``a``/``b`` centred
    on 0. Squared Euclidean distance in this space is a CIE76 perceptual metric.
    """
    import cv2

    pixels = np.ascontiguousarray(colours, dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float64)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1] -= 128.0
    lab[:, 2] -= 128.0
    return lab


def build_gamut(tubes, max_candidates: int = 256) -> np.ndarray:
    """Build the palette of colours mixable from ``tubes``.

    ``tubes`` is a list of ``(r, g, b)`` integer triples. Returns a ``(C, 3)``
    uint8 array of candidate colours a painter could actually reach.

    With mixbox available the candidates are convex combinations in the
    Kubelka-Munk latent space:

    * every single tube (the tube colours themselves);
    * every unordered pair, mixed at the fractions in :data:`_PAIR_FRACTIONS`;
    * every unordered triple, mixed at equal thirds.

    Each mixed latent is mapped back to sRGB and clamped/rounded to a uint8
    triple. Duplicate rows are removed. To bound cost on large palettes:

    * triples are only formed when ``len(tubes) <= _TRIPLE_TUBE_CAP`` (8);
    * when ``len(tubes) > _PAIR_FRACTION_CAP`` (40) pairs are sampled at the even
      fraction (0.5) only.

    If the de-duplicated candidate count exceeds ``max_candidates`` the list is
    deterministically subsampled by an evenly-strided slice of a stable sort,
    with every single-tube colour always kept (no randomness is used anywhere).

    Without mixbox the gamut is simply the unique tube colours (no mixing).
    Empty ``tubes`` gives an empty ``(0, 3)`` array.
    """
    tube_rows = [_clamp_round_rgb(t) for t in tubes]
    if not tube_rows:
        return np.empty((0, 3), dtype=np.uint8)

    singles = _unique_rows(np.array(tube_rows, dtype=np.uint8))

    if not _HAVE_MIXBOX:  # pragma: no cover - mixbox is a declared dependency
        return singles

    n = len(tube_rows)
    latents = [np.asarray(mixbox.rgb_to_latent(rgb), dtype=float) for rgb in tube_rows]

    mixes: list[tuple[int, int, int]] = []

    pair_fractions = _PAIR_FRACTIONS if n <= _PAIR_FRACTION_CAP else (0.5,)
    for i, j in combinations(range(n), 2):
        for t in pair_fractions:
            mixed = t * latents[i] + (1.0 - t) * latents[j]
            mixes.append(_clamp_round_rgb(mixbox.latent_to_rgb(list(mixed))))

    if n <= _TRIPLE_TUBE_CAP:
        third = 1.0 / 3.0
        for i, j, k in combinations(range(n), 3):
            mixed = third * (latents[i] + latents[j] + latents[k])
            mixes.append(_clamp_round_rgb(mixbox.latent_to_rgb(list(mixed))))

    if mixes:
        candidates = _unique_rows(np.vstack([singles, np.array(mixes, dtype=np.uint8)]))
    else:
        candidates = singles

    if len(candidates) <= max_candidates:
        return candidates
    return _subsample(candidates, singles, max_candidates)


def _unique_rows(rows: np.ndarray) -> np.ndarray:
    """Return the unique rows of an ``(N, 3)`` uint8 array, in sorted order."""
    return np.unique(np.ascontiguousarray(rows, dtype=np.uint8), axis=0)


def _subsample(
    candidates: np.ndarray, singles: np.ndarray, max_candidates: int
) -> np.ndarray:
    """Deterministically cut ``candidates`` down to ``max_candidates`` rows.

    Every single-tube colour is kept; the remaining budget is filled from the
    other candidates by an evenly-strided slice of their stable (sorted) order,
    so the choice is reproducible and uses no randomness.
    """
    # Split candidates into single-tube rows (always kept) and the rest.
    single_set = {tuple(int(v) for v in row) for row in singles}
    keep_mask = np.array(
        [tuple(int(v) for v in row) in single_set for row in candidates], dtype=bool
    )
    kept = candidates[keep_mask]
    if len(kept) >= max_candidates:
        # More single tubes than the budget: keep an evenly-strided slice of them
        # (already sorted), still deterministic.
        stride = max(1, len(kept) // max_candidates)
        return kept[::stride][:max_candidates]

    others = candidates[~keep_mask]
    remaining = max_candidates - len(kept)
    if remaining <= 0 or len(others) == 0:
        chosen = kept
    else:
        stride = max(1, len(others) // remaining)
        chosen = np.vstack([kept, others[::stride][:remaining]])
    return _unique_rows(chosen)


def map_image(img: np.ndarray, candidates: np.ndarray, levels: int | None = None):
    """Re-render ``img`` using only the colours in ``candidates``.

    ``img`` is an ``H x W x 3`` uint8 RGB array; ``candidates`` is the ``(C, 3)``
    uint8 gamut from :func:`build_gamut`. Returns a new ``H x W x 3`` uint8 array
    where each pixel is replaced by the nearest candidate colour by CIELAB
    distance. ``img`` is never mutated.

    If ``candidates`` is empty the image is returned unchanged (a copy).

    Rather than a per-pixel-by-candidate distance matrix, the RGB cube is
    quantised into ``levels`` bins per channel (default
    :data:`_DEFAULT_LUT_LEVELS`, i.e. 33, bins of 8). Each bin centre's nearest
    candidate is found in Lab, giving an ``(L, L, L, 3)`` lookup table that every
    pixel is then indexed through, making the per-pixel cost O(1) and the build
    cost ``O(L**3 * C)``. Ties break to the lowest candidate index.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be an H x W x 3 RGB array")

    if candidates.size == 0 or img.size == 0:
        return img.copy()

    levels = _DEFAULT_LUT_LEVELS if levels is None else int(levels)

    lut = _build_lut(candidates, levels)

    src = img.astype(np.int64)
    idx = src * (levels - 1) // 255  # (H, W, 3) bin indices per channel
    out = lut[idx[..., 0], idx[..., 1], idx[..., 2]]
    return np.ascontiguousarray(out, dtype=np.uint8)


def _build_lut(candidates: np.ndarray, levels: int) -> np.ndarray:
    """Build the ``(L, L, L, 3)`` nearest-candidate lookup table.

    Bin ``k`` on a channel represents the RGB value ``round(255 * k / (L - 1))``.
    Every bin centre is classified to its nearest candidate in Lab by squared
    Euclidean distance (argmin, so ties go to the lowest candidate index).
    """
    axis = np.round(np.arange(levels) * 255.0 / (levels - 1)).astype(np.uint8)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    centres = grid.reshape(-1, 3)

    centre_lab = _rgb_to_lab(centres)
    cand_lab = _rgb_to_lab(candidates)

    # Squared Euclidean Lab distance from each bin centre to each candidate,
    # computed in chunks to keep the (bins x candidates) matrix modest.
    nearest = np.empty(len(centres), dtype=np.int64)
    chunk = 4096
    for start in range(0, len(centres), chunk):
        block = centre_lab[start : start + chunk]
        diff = block[:, None, :] - cand_lab[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        nearest[start : start + chunk] = np.argmin(dist2, axis=1)

    lut = candidates[nearest].reshape(levels, levels, levels, 3)
    return np.ascontiguousarray(lut, dtype=np.uint8)


def simulate(img: np.ndarray, tubes, max_candidates: int = 256):
    """Convenience: ``map_image(img, build_gamut(tubes, max_candidates))``.

    Builds the mixable gamut for ``tubes`` and re-renders ``img`` in it, so a
    caller can run the whole limited-palette simulation in one call.
    """
    return map_image(img, build_gamut(tubes, max_candidates))
