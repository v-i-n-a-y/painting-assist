# Copyright 2026 Vinay Williams

"""Canvas priming recommendations: majority colour + technique -> ground colour.

Qt-free so the maths is unit-testable without a GUI (like
:mod:`painting_assist.measure`). Given the processed reference image, the tool
finds its *majority colour* (the largest k-means cluster in CIELab space) and
re-expresses it per painting *technique* into a recommended canvas ground
colour:

* **Mid-tone (dead colour)** — the classic ground: a mid-value tone tinted with
  the majority colour, so the canvas starts at mid-value and the painter can
  judge both lights and darks against it.
* **Majority tint** — the majority colour itself (softened), so the ground
  harmonises with the painting's dominant tone.
* **Complementary ground** — the complement of the majority colour at
  mid-value: a muted ground the painting's colours sit on in harmony.
* **Light ground** — a light, low-chroma tint for light paintings and glazing.
* **Dark ground** — a dark, low-chroma tint for dark-to-light work.
* **Neutral grey** — a plain mid-grey, the safe untinted ground.

The *strength* (0..100) scales how much of the majority colour's chroma the
ground keeps: 0 is a neutral grey at the technique's value, 100 keeps the full
majority chroma. All colour work is in OpenCV 8-bit CIELab (L 0..255, a/b
offset 128), matching the rest of the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Selectable techniques (stored value -> shown label).
TECHNIQUES = [
    ("midtone", "Mid-tone (dead colour)"),
    ("majority", "Majority tint"),
    ("complement", "Complementary ground"),
    ("light", "Light ground"),
    ("dark", "Dark ground"),
    ("neutral", "Neutral grey"),
]

# One-line rationale shown in the editor for each technique.
DESCRIPTIONS = {
    "midtone": (
        "A mid-value ground tinted with the reference's majority colour. "
        "Starting at mid-value lets you judge both lights and darks against "
        "the canvas."
    ),
    "majority": (
        "The reference's majority colour, softened. The ground harmonises "
        "with the painting's dominant tone."
    ),
    "complement": (
        "The complement of the majority colour at mid-value — a muted ground "
        "the painting's colours sit on in harmony."
    ),
    "light": (
        "A light, low-chroma tint of the majority colour — for light "
        "paintings and glazing over a pale ground."
    ),
    "dark": (
        "A dark, low-chroma tint of the majority colour — for dark-to-light "
        "work; the lights will glow against it."
    ),
    "neutral": "A neutral mid-grey, the classic safe ground with no tint.",
}

# Target lightness (8-bit Lab L, 0..255) per technique. "majority" keeps the
# majority colour's own value; "neutral" is handled separately (zero chroma).
_VALUE_TARGETS = {
    "midtone": 128.0,
    "complement": 128.0,
    "light": 220.0,
    "dark": 40.0,
}


@dataclass(frozen=True)
class PrimeResult:
    """A recommended ground colour plus the majority colour it was derived from."""

    rgb: Tuple[int, int, int]
    majority: Tuple[int, int, int]
    technique: str

    @property
    def hex(self) -> str:
        """The ground colour as an uppercase ``#RRGGBB`` string."""
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)

    @property
    def majority_hex(self) -> str:
        """The majority colour as an uppercase ``#RRGGBB`` string."""
        return "#{:02X}{:02X}{:02X}".format(*self.majority)


def majority_colour(
    img: np.ndarray, k: int = 5, max_px: int = 4096
) -> Optional[Tuple[int, int, int]]:
    """Return the dominant colour of ``img`` as an RGB triple, or ``None``.

    The image is downscaled to a small proxy (at most ``max_px`` pixels) so the
    clustering is cheap enough to run on every render, then its colours are
    clustered in CIELab space (perceptual similarity) with k-means and the
    largest cluster's centroid is converted back to RGB. A flat image (one
    colour) short-circuits to its mean. The k-means seed is derived from the
    proxy's content (same pattern as the Colour groups control) so identical
    inputs always yield the same majority colour.
    """
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None

    px = h * w
    if px > max_px:
        scale = (max_px / px) ** 0.5
        pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(
            np.ascontiguousarray(img), (pw, ph), interpolation=cv2.INTER_AREA
        )
    else:
        small = np.ascontiguousarray(img)

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
    samples = lab.reshape(-1, 3).astype(np.float32)
    # A flat image has a single colour: skip the (degenerate) clustering.
    # (std across *pixels* per channel — a global std would see the
    # between-channel spread of any colour and never trigger.)
    if samples.shape[0] < 2 or float(samples.std(axis=0).max()) < 1e-3:
        mean = small.reshape(-1, 3).mean(axis=0)
        return tuple(int(round(c)) for c in mean)

    k_eff = min(max(2, int(k)), samples.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    sample_bytes = samples.view(np.uint8)[::997]
    seed = (int(sample_bytes.sum(dtype=np.uint64)) ^ (k_eff * 0x9E3779B1)) & 0x7FFFFFFF
    cv2.setRNGSeed(seed)
    _compactness, labels, centers = cv2.kmeans(
        samples, k_eff, None, criteria, 1, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.ravel(), minlength=k_eff)
    idx = int(np.argmax(counts))
    center = np.clip(np.round(centers[idx : idx + 1]), 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(center.reshape(1, 1, 3), cv2.COLOR_Lab2RGB)[0, 0]
    return tuple(int(c) for c in rgb)


def _lab_to_rgb(l: float, a: float, b: float) -> Tuple[int, int, int]:
    """Convert an 8-bit CIELab triple to an RGB triple (clipped to 0..255)."""
    rgb = cv2.cvtColor(
        np.uint8([[[int(round(l)), int(round(a)), int(round(b))]]]),
        cv2.COLOR_Lab2RGB,
    )[0, 0]
    return tuple(int(c) for c in rgb)


def _fit_to_gamut(
    l: float, a: float, b: float, max_iter: int = 16
) -> Tuple[float, float, float]:
    """Scale chroma toward neutral until the Lab colour fits in sRGB.

    A strongly chromatic colour at an extreme lightness (e.g. the complement
    of a saturated red at mid-value) can fall outside the sRGB gamut: the
    Lab->RGB conversion clips channels and the resulting colour loses its
    hue. Each pass checks the round trip and, when a/b drifted (clipping),
    scales the chroma by 0.8 toward neutral. Grounds are meant to be muted,
    so the fitted colour is the right one to recommend; a neutral grey at any
    lightness is always in gamut, so the loop always converges.
    """
    for _ in range(max_iter):
        rgb = _lab_to_rgb(l, a, b)
        back = cv2.cvtColor(
            np.uint8([[[rgb[0], rgb[1], rgb[2]]]]), cv2.COLOR_RGB2Lab
        )[0, 0]
        if abs(float(back[1]) - a) <= 3.0 and abs(float(back[2]) - b) <= 3.0:
            return l, a, b
        a = 128.0 + (a - 128.0) * 0.8
        b = 128.0 + (b - 128.0) * 0.8
    return l, a, b


def prime_colour(
    img: np.ndarray, technique: str, strength: int = 50
) -> Optional[PrimeResult]:
    """Recommend a priming colour for ``img`` using ``technique``.

    The image's majority colour is re-expressed in CIELab: the lightness is
    moved to the technique's target (mid-tone/complement -> mid, light ->
    high, dark -> low, majority -> unchanged, neutral -> mid with zero
    chroma), the complement mirrors a/b across the neutral axis, and the
    chroma is scaled by ``strength`` (0 = neutral, 100 = full majority
    chroma). The result is fitted back into the sRGB gamut (see
    :func:`_fit_to_gamut`) so the recommended colour is actually paintable.
    Returns ``None`` when ``img`` is empty.
    """
    majority = majority_colour(img)
    if majority is None:
        return None

    s = max(0.0, min(1.0, float(strength) / 100.0))
    lab = cv2.cvtColor(
        np.uint8([[[majority[0], majority[1], majority[2]]]]), cv2.COLOR_RGB2Lab
    )[0, 0]
    l, a, b = (float(c) for c in lab)

    if technique == "complement":
        # Mirror across the neutral axis: (a-128) -> -(a-128).
        a = 256.0 - a
        b = 256.0 - b
        l = _VALUE_TARGETS["complement"]
    elif technique == "neutral":
        a, b, l = 128.0, 128.0, 128.0
    elif technique in _VALUE_TARGETS:
        l = _VALUE_TARGETS[technique]
    # "majority" (and any unknown name) keeps the majority's own value.

    a = 128.0 + (a - 128.0) * s
    b = 128.0 + (b - 128.0) * s
    l, a, b = _fit_to_gamut(l, a, b)

    return PrimeResult(
        rgb=_lab_to_rgb(l, a, b),
        majority=majority,
        technique=technique,
    )
