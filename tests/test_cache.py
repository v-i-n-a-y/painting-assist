from __future__ import annotations

"""Prefix-cache behaviour of ControlPipeline, using spy controls that count
their process() invocations."""

from typing import List

import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.pipeline import ControlPipeline


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
        return [Param(name="amt", label="Amt", ptype=ParamType.INT,
                      default=0, minimum=0, maximum=100)]

    def is_active(self) -> bool:
        return self.enabled

    def process(self, img: np.ndarray) -> np.ndarray:
        self.calls += 1
        return (img + int(self.get("amt"))).astype(np.uint8)


def _img() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


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
