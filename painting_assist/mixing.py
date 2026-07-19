# Copyright 2026 Vinay Williams

"""Physically-plausible pigment mixing and paint-matching for a painter's tubes.

Unlike the approximate additive-RGB guide in
:mod:`painting_assist.colour_mixing`, this module mixes paint the way paint
actually mixes. It uses `mixbox <https://github.com/scrtwpns/mixbox>`_, which
maps an sRGB colour into a seven-dimensional Kubelka-Munk latent space where
pigment mixing is *linear*: a convex combination of tube latents, mapped back to
sRGB, is a physically-plausible mix (blue plus yellow gives green, not grey).

Because mixing is linear in the latent space, finding the tube proportions that
best reach a target is a convex non-negative least-squares problem, solved once
per query with the Lawson-Hanson solver reused from
:mod:`painting_assist.colour_mixing` (a sum-to-one penalty row makes the weights
a convex combination). This is fast enough for an interactive click, and the
"which paint should I buy" scan is just one such solve per catalogue paint.

Match quality is judged with :func:`deltae`, a CIE76 distance in true CIELAB
derived from OpenCV's 8-bit Lab so it agrees with the rest of the app's colour
readouts.

If mixbox is ever unavailable, :data:`MIXBOX_AVAILABLE` is ``False`` and the
engine falls back to the additive model in
:func:`painting_assist.colour_mixing.suggest_mix`. mixbox is a declared
dependency, so the pigment-mixing path is the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from painting_assist import paints
from painting_assist.colour_mixing import _nnls, suggest_mix

try:
    import mixbox
except ImportError:  # pragma: no cover - mixbox is a declared dependency
    mixbox = None

MIXBOX_AVAILABLE = mixbox is not None

# Tube proportions below this fraction are dropped before renormalising, matching
# colour_mixing so recipes read consistently across the two engines.
_MIX_EPSILON = 0.03

# The sum-to-one constraint is imposed as a heavily weighted extra equation
# appended to the latent-fit system, exactly as colour_mixing.suggest_mix does.
_SUM_PENALTY = 1000.0

# A tolerance percentage maps linearly onto this many units of Lab deltaE, so the
# full 0-100% slider spans 0-40 deltaE. 40 is a generous upper bound: at that
# distance two colours are unmistakably different, so 100% tolerance accepts
# essentially any mix, while 25% ~ 10 deltaE is a reasonable painter tolerance.
_TOLERANCE_DELTAE_SPAN = 40.0


def _clamp_rgb(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return ``rgb`` as an integer triple with each component clamped to 0-255."""
    r, g, b = (int(round(float(component))) for component in rgb)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def deltae(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    """Perceptual CIE76 distance between two sRGB colours, in true CIELAB.

    Both colours are converted through OpenCV's 8-bit Lab (the same convention
    the Palette panel and :func:`painting_assist.colour_mixing.describe_colour`
    use) and rescaled to genuine CIELAB units: ``L`` on 0-100 and ``a``/``b``
    centred on 0. The result is the Euclidean distance in that space. It is 0 for
    identical colours, symmetric, and grows with visual difference.
    """
    import cv2

    pixels = np.array(
        [[list(_clamp_rgb(rgb1)), list(_clamp_rgb(rgb2))]], dtype=np.uint8
    )
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2Lab)[0].astype(float)
    lightness = lab[:, 0] * 100.0 / 255.0
    a = lab[:, 1] - 128.0
    b = lab[:, 2] - 128.0
    return float(
        np.sqrt(
            (lightness[0] - lightness[1]) ** 2 + (a[0] - a[1]) ** 2 + (b[0] - b[1]) ** 2
        )
    )


def tolerance_deltae(tolerance_pct: float) -> float:
    """Map a 0-100 tolerance percentage to a Lab deltaE budget.

    The mapping is linear: ``budget = tolerance_pct / 100 * 40``. So 0% demands
    an essentially exact match, 25% allows about 10 deltaE (a reasonable painter
    tolerance), and 100% accepts almost anything. The input is clamped to 0-100.
    """
    fraction = max(0.0, min(100.0, float(tolerance_pct))) / 100.0
    return fraction * _TOLERANCE_DELTAE_SPAN


def _best_mix_mixbox(
    target: tuple[int, int, int],
    tubes: list[tuple[str, tuple[int, int, int]]],
) -> tuple[list[tuple[str, float]], tuple[int, int, int]]:
    """Solve for convex tube weights in mixbox latent space toward ``target``.

    Each tube and the target are mapped to their seven-dimensional latent. A
    sum-to-one non-negative least-squares fit finds the convex combination of
    tube latents closest to the target latent; that combination is a valid latent
    and maps back to the achieved sRGB colour. Weights below :data:`_MIX_EPSILON`
    are dropped and the survivors renormalised, with the mixed colour recomputed
    from the survivors so recipe and swatch always agree.
    """
    names = [name for name, _ in tubes]
    latents = np.array(
        [mixbox.rgb_to_latent(rgb) for _, rgb in tubes], dtype=float
    ).T  # 7 x n
    target_latent = np.array(mixbox.rgb_to_latent(target), dtype=float)

    aug_matrix = np.vstack([latents, np.full((1, latents.shape[1]), _SUM_PENALTY)])
    aug_target = np.append(target_latent, _SUM_PENALTY)
    weights = _nnls(aug_matrix, aug_target)

    total = weights.sum()
    if total <= 0.0:
        # Degenerate fit: fall back to the single nearest tube by deltaE.
        nearest = min(range(len(tubes)), key=lambda i: deltae(target, tubes[i][1]))
        return [(names[nearest], 1.0)], _clamp_rgb(tubes[nearest][1])
    weights = weights / total

    keep = weights >= _MIX_EPSILON
    if not keep.any():
        keep = weights == weights.max()
    weights = np.where(keep, weights, 0.0)
    weights = weights / weights.sum()

    mixed_latent = latents @ weights
    mixed_rgb = _clamp_rgb(tuple(mixbox.latent_to_rgb(list(mixed_latent))))
    recipe = [
        (names[i], float(weights[i])) for i in range(len(names)) if weights[i] > 0.0
    ]
    recipe.sort(key=lambda item: item[1], reverse=True)
    return recipe, mixed_rgb


def _best_mix_additive(
    target: tuple[int, int, int],
    tubes: list[tuple[str, tuple[int, int, int]]],
) -> tuple[list[tuple[str, float]], tuple[int, int, int]]:
    """Fallback used when mixbox is unavailable: the additive-RGB guide.

    Delegates to :func:`painting_assist.colour_mixing.suggest_mix` for the
    proportions, then predicts the mixed colour as the proportion-weighted
    average of the tube sRGB values. This is an approximate additive model, not
    physically-plausible pigment mixing.
    """
    recipe = suggest_mix(target, bases=list(tubes))
    lookup = {name: np.array(rgb, dtype=float) for name, rgb in tubes}
    mixed = np.zeros(3)
    for name, proportion in recipe:
        mixed += proportion * lookup[name]
    return recipe, _clamp_rgb(tuple(mixed))


def best_mix(
    target_rgb: tuple[int, int, int],
    tubes: list[tuple[str, tuple[int, int, int]]],
) -> tuple[list[tuple[str, float]], tuple[int, int, int], float]:
    """Find the mix of ``tubes`` that best reaches ``target_rgb``.

    ``tubes`` is a list of ``(name, (r, g, b))``. Returns ``(recipe, mixed_rgb,
    error)`` where ``recipe`` is a list of ``(name, proportion)`` sorted by
    descending proportion (proportions non-negative, summing to about 1.0),
    ``mixed_rgb`` is the colour that recipe actually achieves, and ``error`` is
    the :func:`deltae` between ``mixed_rgb`` and the target.

    With no tubes there is nothing to mix, so the result is ``([], target,
    0.0)``. The pigment-mixing engine is used when :data:`MIXBOX_AVAILABLE`;
    otherwise the additive fallback is used.
    """
    target = _clamp_rgb(target_rgb)
    if not tubes:
        return [], target, 0.0
    if MIXBOX_AVAILABLE:
        recipe, mixed_rgb = _best_mix_mixbox(target, tubes)
    else:  # pragma: no cover - mixbox is a declared dependency
        recipe, mixed_rgb = _best_mix_additive(target, tubes)
    return recipe, mixed_rgb, deltae(mixed_rgb, target)


@dataclass(frozen=True)
class Suggestion:
    """The outcome of matching a target colour against a painter's tubes.

    ``recipe`` and ``mixed_rgb`` describe the closest mix from the *current*
    tubes (so the UI can always show a swatch), ``error`` is its deltaE from the
    target, and ``within_tolerance`` says whether that clears the deltaE budget.
    ``buy`` names a catalogue paint to add when ``on_miss="buy"`` finds one that
    would bring the mix within budget, otherwise ``None``. ``message`` is a
    human-readable summary suitable for display.
    """

    recipe: list[tuple[str, float]]
    mixed_rgb: tuple[int, int, int]
    error: float
    within_tolerance: bool
    buy: str | None
    message: str


def _best_paint_to_buy(
    target: tuple[int, int, int],
    tubes: list[tuple[str, tuple[int, int, int]]],
    catalogue: list[tuple[str, tuple[int, int, int]]],
) -> tuple[str | None, float]:
    """Return the catalogue paint whose addition best reaches ``target``.

    Every catalogue paint not already among ``tubes`` (compared by name,
    case-insensitively) is trialled by solving :func:`best_mix` over the tubes
    plus that one paint. The paint giving the lowest resulting error wins. Ties
    break towards the earlier catalogue entry. Returns ``(name, error)``, or
    ``(None, inf)`` if there is nothing new to try.
    """
    owned = {name.lower() for name, _ in tubes}
    best_name: str | None = None
    best_error = float("inf")
    for name, rgb in catalogue:
        if name.lower() in owned:
            continue
        _, _, error = best_mix(target, list(tubes) + [(name, rgb)])
        if error < best_error:
            best_error = error
            best_name = name
    return best_name, best_error


def suggest(
    target_rgb: tuple[int, int, int],
    tubes: list[tuple[str, tuple[int, int, int]]],
    tolerance_pct: float = 25.0,
    on_miss: str = "closest",
    catalogue: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> Suggestion:
    """Suggest how to reach ``target_rgb`` from the painter's ``tubes``.

    ``tolerance_pct`` (0-100) sets the deltaE budget via
    :func:`tolerance_deltae`. The closest mix from the current tubes is always
    computed so a swatch can be shown. When that mix clears the budget the
    suggestion is within tolerance. When it does not, ``on_miss`` decides:

    - ``"closest"`` reports the gap and offers the closest mix as-is.
    - ``"buy"`` scans ``catalogue`` (defaulting to
      :data:`painting_assist.paints.DEFAULT_CATALOGUE`) for a single paint whose
      addition would bring the mix within budget. If one is found, ``buy`` names
      it; if not, the colour is likely outside the paints' gamut and ``buy`` is
      ``None`` with a message saying so.

    With no tubes the recipe is empty and the message explains why.
    """
    target = _clamp_rgb(target_rgb)
    budget = tolerance_deltae(tolerance_pct)
    if catalogue is None:
        catalogue = paints.DEFAULT_CATALOGUE

    if not tubes:
        return Suggestion(
            recipe=[],
            mixed_rgb=target,
            error=0.0,
            within_tolerance=False,
            buy=None,
            message="Add at least one paint to your palette to mix a colour.",
        )

    recipe, mixed_rgb, error = best_mix(target, tubes)

    if error <= budget:
        return Suggestion(
            recipe=recipe,
            mixed_rgb=mixed_rgb,
            error=error,
            within_tolerance=True,
            buy=None,
            message=f"Mix within tolerance (off by dE {error:.1f}).",
        )

    if on_miss == "buy":
        buy_name, buy_error = _best_paint_to_buy(target, tubes, catalogue)
        if buy_name is not None and buy_error <= budget:
            return Suggestion(
                recipe=recipe,
                mixed_rgb=mixed_rgb,
                error=error,
                within_tolerance=False,
                buy=buy_name,
                message=f"Add {buy_name} to reach this colour.",
            )
        return Suggestion(
            recipe=recipe,
            mixed_rgb=mixed_rgb,
            error=error,
            within_tolerance=False,
            buy=None,
            message=(
                f"Even adding one paint will not reach this colour within "
                f"{tolerance_pct:.0f}% (likely outside the paints' gamut); "
                f"closest mix is off by dE {error:.1f}."
            ),
        )

    return Suggestion(
        recipe=recipe,
        mixed_rgb=mixed_rgb,
        error=error,
        within_tolerance=False,
        buy=None,
        message=(
            f"Closest mix is off by dE {error:.1f}; "
            f"cannot be matched within {tolerance_pct:.0f}%."
        ),
    )
