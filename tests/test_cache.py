# Copyright 2026 Vinay Williams

"""Prefix-cache behaviour of ControlPipeline, using spy controls that count
their process() invocations."""

from __future__ import annotations

from typing import List

import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.pipeline import MAX_CACHED_STAGES, ControlPipeline


class SpyControl(Control):
    """A minimal always-active control that counts process() calls and tags
    the image so a change is observable downstream."""

    def __init__(self, cid: str) -> None:
        self.id = cid
        self.name = cid
        self.order = 0
        self.calls = 0
        super().__init__()

    def params(self) -> List[Param]:  # instance method is fine for our use
        return [
            Param(
                name="amt",
                label="Amt",
                ptype=ParamType.INT,
                default=0,
                minimum=0,
                maximum=100,
            )
        ]

    def is_active(self) -> bool:
        return self.enabled

    def process(self, img: np.ndarray) -> np.ndarray:
        self.calls += 1
        return (img + int(self.get("amt"))).astype(np.uint8)


class AddMulControl(Control):
    """Order- AND value-sensitive stage: out = (img * mul + amt) mod 256.

    Distinct per-stage ``mul`` makes the chain non-commutative, so any incorrect
    prefix reuse (e.g. a stale cached array surviving eviction) yields a different
    final image than a fresh, correctly ordered pipeline would."""

    def __init__(self, cid: str, mul: int) -> None:
        self.id = cid
        self.name = cid
        self.order = 0
        self.mul = mul
        super().__init__()

    def params(self) -> List[Param]:
        return [
            Param(
                name="amt",
                label="Amt",
                ptype=ParamType.INT,
                default=0,
                minimum=0,
                maximum=255,
            )
        ]

    def is_active(self) -> bool:
        return self.enabled

    def process(self, img: np.ndarray) -> np.ndarray:
        # int64 widen keeps the multiply exact; astype copies, so img is untouched.
        return ((img.astype(np.int64) * self.mul + int(self.get("amt"))) % 256).astype(
            np.uint8
        )


def _img() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _rand_img() -> np.ndarray:
    """A small, spatially varying image so a corrupted prefix is observable."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)


def _retained(pipe: ControlPipeline) -> int:
    """Count populated cache slots (each holds one full-res array)."""
    return sum(1 for entry in pipe._cache if entry is not None)


def _pipe(controls):
    pipe = ControlPipeline(controls)
    for c in controls:
        pipe.set_enabled(c.id, True)
    return pipe


def test_same_source_and_token_hits_cache():
    a = SpyControl("a")
    pipe = _pipe([a])
    img = _img()
    pipe.process(img, token="T")
    assert a.calls == 1
    # A fresh (but equal) array with the same token must reuse the cache.
    pipe.process(_img(), token="T")
    assert a.calls == 1


def test_midchain_change_recomputes_stage_and_downstream_only():
    a, b, c = SpyControl("a"), SpyControl("b"), SpyControl("c")
    pipe = _pipe([a, b, c])
    img = _img()
    pipe.process(img, token="T")
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)

    # Change the middle control's value -> b recomputes, c recomputes, a does not.
    pipe.set_value("b", "amt", 5)
    pipe.process(img, token="T")
    assert a.calls == 1  # upstream untouched
    assert b.calls == 2  # changed stage recomputed
    assert c.calls == 2  # downstream recomputed


def test_token_change_invalidates_everything():
    a, b = SpyControl("a"), SpyControl("b")
    pipe = _pipe([a, b])
    img = _img()
    pipe.process(img, token="T1")
    assert (a.calls, b.calls) == (1, 1)
    pipe.process(img, token="T2")
    assert (a.calls, b.calls) == (2, 2)


# ---- memory cap (MAX_CACHED_STAGES) ----
def test_cap_not_triggered_at_or_below_threshold():
    # A chain of exactly the cap length leaves every slot populated: the number
    # of populated slots is not GREATER than the cap, so nothing is dropped.
    n = MAX_CACHED_STAGES
    muls = [2 * i + 1 for i in range(n)]
    controls = [AddMulControl(f"s{i}", muls[i]) for i in range(n)]
    pipe = _pipe(controls)
    for i, c in enumerate(controls):
        pipe.set_value(c.id, "amt", i + 1)
    pipe.process(_rand_img(), token="T")
    assert _retained(pipe) == n


def test_cap_bounds_memory_and_preserves_output():
    # A chain LONGER than the cap, mutated repeatedly downstream, must (a) never
    # retain more than the cap of full-res arrays and (b) still produce exactly
    # the image a fresh, isolated pipeline at the same final state produces.
    n = MAX_CACHED_STAGES + 3
    assert n > MAX_CACHED_STAGES  # the cap must actually engage
    muls = [2 * i + 1 for i in range(n)]
    img = _rand_img()

    controls = [AddMulControl(f"s{i}", muls[i]) for i in range(n)]
    pipe = _pipe(controls)
    for i, c in enumerate(controls):
        pipe.set_value(c.id, "amt", (i * 31 + 7) % 256)
    pipe.process(img, token="T")

    # Repeatedly mutate the most-downstream stage: each render re-fills the whole
    # chain and re-trims, exercising eviction + upstream recompute every time.
    for v in (10, 200, 55, 130, 99, 4):
        pipe.set_value("s%d" % (n - 1), "amt", v)
        pipe.process(img, token="T")
        assert _retained(pipe) <= MAX_CACHED_STAGES

    # A mid-chain mutation too, to move the changed stage upstream of cached tail.
    pipe.set_value("s%d" % (n // 2), "amt", 88)
    result = pipe.process(img, token="T")

    # (a) memory bound: the cap engaged and held.
    assert _retained(pipe) == MAX_CACHED_STAGES

    # (b) correctness: identical to a fresh pipeline built at the final state.
    final_amts = {c.id: c.get("amt") for c in controls}
    fresh_controls = [AddMulControl(f"s{i}", muls[i]) for i in range(n)]
    fresh = _pipe(fresh_controls)
    for c in fresh_controls:
        fresh.set_value(c.id, "amt", final_amts[c.id])
    fresh_result = fresh.process(img, token="U")

    assert np.array_equal(result, fresh_result)
