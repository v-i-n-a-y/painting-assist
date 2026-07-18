# Copyright 2026 Vinay Williams

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register


@register
class ColourGroupsControl(Control):
    """Colour groups — coarse-to-fine colour-blocking via Lab k-means quantization.

    Mirrors the coarse→fine philosophy of the Blur control, but in colour space
    rather than spatial frequency. At the start of a painting you dial down to
    just a handful of flat colour masses — the big hue/value relationships — and
    slowly add more colour groups as the painting builds up detail.

    Internally, pixel colours are clustered in CIELab space (perceptual
    uniformity means the groupings match what the eye actually sees as "similar
    colour"). To stay responsive on large images, the grouping is computed on a
    bounded-size proxy: the image is downscaled so it has at most PROC_MAX_PX
    pixels, k-means clusters its colours and produces per-pixel labels directly,
    the proxy is recoloured from those labels, then the flat result is upscaled
    back to full size with nearest-neighbour interpolation — exact for flat colour
    masses. A pre-quantization median-blur pass (the "Smooth" slider) consolidates
    speckle into cleaner contiguous regions before the grouping runs.
    """

    id = "quantize"
    name = "Colour groups"
    order = 15  # runs after blur=10, before grid=90

    # Colour grouping is computed on a proxy no larger than this many pixels,
    # then the flat result is upscaled (nearest) to full size. This bounds the
    # cost independently of the source resolution and colour count, keeping
    # slider drags responsive.
    PROC_MAX_PX = 250_000

    @classmethod
    def params(cls) -> List[Param]:
        """Schema for the Colours count and the pre-quantization Smooth pass."""
        return [
            Param(
                name="colours",
                label="Colours",
                ptype=ParamType.INT,
                default=8,
                minimum=2,
                maximum=100,
                step=1,
                suffix="",
                tooltip=(
                    "Number of flat colour groups. "
                    "Drag left (fewer) to start with coarse colour masses; "
                    "drag right to add more colours as the painting progresses."
                ),
            ),
            Param(
                name="smooth",
                label="Smooth",
                ptype=ParamType.INT,
                default=0,
                minimum=0,
                maximum=100,
                step=1,
                tooltip=(
                    "Pre-quantization median blur (0 = off). "
                    "A higher value consolidates speckle into cleaner, more "
                    "contiguous colour regions before the grouping runs."
                ),
            ),
        ]

    # ------------------------------------------------------------------ #
    # Control overrides
    # ------------------------------------------------------------------ #
    def is_active(self) -> bool:
        """Always meaningful when enabled."""
        return self.enabled

    def process(self, img: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> colour-quantized RGB uint8 HxWx3 (new array).

        Does not mutate ``img``. To stay responsive on large images, the colour
        grouping is computed on a bounded-size proxy: the image is downscaled so
        it has at most ``PROC_MAX_PX`` pixels, k-means clusters its colours in
        CIELab space (perceptual similarity), each proxy pixel is recoloured with
        its cluster's Lab centroid via the labels k-means already produced, and
        the resulting flat image is upscaled back to full size with
        nearest-neighbour interpolation — which is exact for flat colour masses.
        Cost is therefore independent of the source resolution and colour count.
        """
        n = max(2, min(100, int(self.get("colours"))))
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img

        # Downscale to a bounded proxy for all the heavy work.
        px = h * w
        if px > self.PROC_MAX_PX:
            scale = (self.PROC_MAX_PX / px) ** 0.5
            pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
            small = cv2.resize(
                np.ascontiguousarray(img), (pw, ph), interpolation=cv2.INTER_AREA
            )
        else:
            small = np.ascontiguousarray(img)
        sh, sw = small.shape[:2]

        # Optional pre-quantization smoothing to consolidate speckle (cheap on
        # the proxy). Kernel is a fraction of the proxy's short side, capped so
        # medianBlur stays on its fast path.
        smooth = max(0, min(100, int(self.get("smooth"))))
        if smooth > 0:
            k = int(round((smooth / 100.0) * 0.05 * min(sh, sw)))
            k = min(k, 10)
            if k >= 1:
                small = cv2.medianBlur(small, 2 * k + 1)

        lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
        samples = lab.reshape(-1, 3).astype(np.float32)
        k_eff = min(n, samples.shape[0])
        if k_eff < 2:
            return img
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # KMEANS_PP_CENTERS seeds cluster centres from OpenCV's global RNG, so
        # without a fixed seed the same image quantizes to slightly different
        # masses on each re-render, making the flat colour blocks flicker as the
        # user drags unrelated sliders. Seed the RNG deterministically from the
        # proxy's own content (a strided sample of its bytes, folded together
        # with the colour count) so identical inputs always cluster identically,
        # while genuinely different images/settings still get different seeds.
        # cv2.setRNGSeed is global, which is fine here: a single worker thread
        # owns all processing, so there is no concurrent kmeans to interleave.
        sample_bytes = samples.view(np.uint8)[::997]
        seed = (
            int(sample_bytes.sum(dtype=np.uint64)) ^ (k_eff * 0x9E3779B1)
        ) & 0x7FFFFFFF
        cv2.setRNGSeed(seed)
        _compactness, labels, centers = cv2.kmeans(
            samples, k_eff, None, criteria, 1, cv2.KMEANS_PP_CENTERS
        )
        centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)  # (k_eff,3) Lab
        quant_lab = centers[labels.flatten()].reshape(sh, sw, 3)
        quant_rgb = cv2.cvtColor(quant_lab, cv2.COLOR_Lab2RGB)

        # Surface the cluster centroids as a palette for the side panel. Sort by
        # Lab lightness (ascending) so the strip reads dark -> light, and convert
        # each centroid from its Lab representation back to display RGB.
        order = np.argsort(centers[:, 0], kind="stable")
        palette_lab = centers[order].reshape(-1, 1, 3)
        palette_rgb = cv2.cvtColor(palette_lab, cv2.COLOR_Lab2RGB).reshape(-1, 3)
        self.emit_metadata("palette", [tuple(int(c) for c in px) for px in palette_rgb])

        # Upscale the flat masses back to full size (nearest = exact for flats).
        if (sh, sw) != (h, w):
            quant_rgb = cv2.resize(quant_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
        return quant_rgb
