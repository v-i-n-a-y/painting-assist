# Copyright 2026 Vinay Williams

"""Snapshot decoupling: processing against a states snapshot must never read or
mutate the live control, and live edits mid-render must not affect the snapshot
result."""

from __future__ import annotations

from typing import List

import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.pipeline import ControlPipeline


class AddControl(Control):
    def __init__(self, cid: str = "add") -> None:
        self.id = cid
        self.name = cid
        self.order = 0
        super().__init__()

    def params(self) -> List[Param]:
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
        return (img + int(self.get("amt"))).astype(np.uint8)


def _img() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_snapshot_leaves_live_values_untouched():
    ctrl = AddControl()
    pipe = ControlPipeline([ctrl])
    pipe.set_enabled("add", True)
    pipe.set_value("add", "amt", 10)
    states = pipe.snapshot_states()
    pipe.process(_img(), states)
    # Live control still holds its own value; process_snapshot ran on a clone.
    assert ctrl.get("amt") == 10
    assert ctrl.enabled is True


def test_live_edit_after_snapshot_does_not_change_snapshot_result():
    ctrl = AddControl()
    pipe = ControlPipeline([ctrl])
    pipe.set_enabled("add", True)
    pipe.set_value("add", "amt", 10)
    states = pipe.snapshot_states()

    # Mutate live values *after* taking the snapshot.
    pipe.set_value("add", "amt", 99)
    out = pipe.process(_img(), states, token="T")
    # Result reflects the snapshot (10), not the live edit (99).
    assert int(out[0, 0, 0]) == 10


def test_process_snapshot_does_not_mutate_live_control():
    ctrl = AddControl()
    before = ctrl.values()
    ctrl.process_snapshot(_img(), enabled=True, values={"amt": 42})
    # The clone carried the (42) snapshot; the live control is unchanged.
    assert ctrl.values() == before
    assert ctrl.enabled is False
