from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ParamType(Enum):
    """The kinds of tunable knob a control can declare."""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    CHOICE = "choice"
    TEXT = "text"


@dataclass(frozen=True)
class Param:
    """Immutable declarative descriptor for ONE tunable knob.

    control_panel.py builds a widget purely from this — no per-control UI code.
    """

    name: str                                            # machine key into the values dict
    label: str                                           # human label shown in the panel
    ptype: ParamType
    default: Any
    minimum: Optional[float] = None                      # FLOAT/INT only
    maximum: Optional[float] = None                      # FLOAT/INT only
    step: Optional[float] = None                         # slider granularity; default 1 (INT) / 0.01 (FLOAT)
    choices: Optional[Sequence[Tuple[Any, str]]] = None  # CHOICE only: (stored_value, shown_label) pairs
    reversed: bool = False                               # UX: invert slider so LEFT=max, RIGHT=min
    suffix: str = ""                                     # readout suffix, e.g. " px"
    tooltip: str = ""

    def effective_step(self) -> float:
        """Return step, defaulting to 1.0 for INT and 0.01 for FLOAT."""
        if self.step is not None:
            return float(self.step)
        if self.ptype is ParamType.INT:
            return 1.0
        if self.ptype is ParamType.FLOAT:
            return 0.01
        return 1.0

    def clamp(self, value: Any) -> Any:
        """Coerce + range-clamp an incoming value to a valid one (used on set/load)."""
        if self.ptype is ParamType.BOOL:
            return bool(value)

        if self.ptype is ParamType.TEXT:
            return "" if value is None else str(value)

        if self.ptype is ParamType.CHOICE:
            valid = [choice[0] for choice in (self.choices or ())]
            if value in valid:
                return value
            return self.default

        if self.ptype is ParamType.INT:
            try:
                coerced = int(round(float(value)))
            except (TypeError, ValueError):
                return self.default
            if self.minimum is not None:
                coerced = max(int(self.minimum), coerced)
            if self.maximum is not None:
                coerced = min(int(self.maximum), coerced)
            return coerced

        if self.ptype is ParamType.FLOAT:
            try:
                coerced = float(value)
            except (TypeError, ValueError):
                return self.default
            if self.minimum is not None:
                coerced = max(float(self.minimum), coerced)
            if self.maximum is not None:
                coerced = min(float(self.maximum), coerced)
            return coerced

        return value


class Control:
    """Base class for every image control. Subclass + @register = done.

    NOTE: not abc.ABC, to avoid metaclass friction; params()/process() raise
    NotImplementedError if not overridden.
    """

    # ---- class-level identity (subclass sets these) ----
    id: str = ""                  # stable unique registry key, e.g. "blur" (used in saved state)
    name: str = "Control"         # display name = panel section title, e.g. "Blur"
    order: int = 100              # pipeline + panel ordering; lower runs first

    def __init__(self) -> None:
        self._values: Dict[str, Any] = {p.name: p.default for p in self.params()}
        self.enabled: bool = False        # painter opts a control in

    # ---- the two things every subclass MUST declare ----
    @classmethod
    def params(cls) -> List[Param]:
        """Static schema. classmethod so UI/registry can introspect without an instance.

        Default raises; subclasses override.
        """
        raise NotImplementedError

    def process(self, img: np.ndarray) -> np.ndarray:
        """Pure transform: RGB uint8 HxWx3 in -> RGB uint8 HxWx3 out.

        Reads current values via self.get(...). MUST NOT mutate img.
        Called by the pipeline only when is_active() is True.
        """
        raise NotImplementedError

    # ---- optional custom UI (concrete controls may override) ----
    def create_editor(self, parent: Optional["object"] = None) -> Optional["object"]:
        """Return a custom editor widget for this control, or ``None``.

        Returning ``None`` (the default) means the panel builds the generic UI
        from :meth:`params` — sliders/spin boxes/checkboxes/combos/line-edits.
        Override to supply a richer, control-specific editor (e.g. a stepper or
        an interactive tool).

        A returned widget MUST follow the editor contract used by
        ``control_panel.ControlSection``:

        * expose a ``paramChanged(str, object)`` Qt signal carrying
          ``(param_name, value)`` for any knob the user changes;
        * expose an ``interaction(bool)`` Qt signal fired on the press/release
          of any continuous drag, so the renderer can drop to a fast preview;
        * provide a ``refresh()`` method that re-syncs the widget from this
          control's current state.

        Qt is intentionally *not* imported here: keeping :mod:`base` Qt-free
        lets controls be processed and unit-tested headlessly. Subclasses that
        build a widget import Qt lazily inside their own override.
        """
        return None

    # ---- generic state plumbing (concrete; subclasses rarely override) ----
    def _param_map(self) -> Dict[str, Param]:
        """Return a {name: Param} mapping for this control's schema."""
        return {p.name: p for p in self.params()}

    def get(self, name: str) -> Any:
        """Return current value of param `name`."""
        return self._values[name]

    def set(self, name: str, value: Any) -> None:
        """Set param `name`, clamped/coerced via its Param.clamp()."""
        spec = self._param_map().get(name)
        if spec is None:
            return
        self._values[name] = spec.clamp(value)

    def values(self) -> Dict[str, Any]:
        """Return a shallow COPY of the current values dict (snapshot for the worker)."""
        return dict(self._values)

    def set_enabled(self, value: bool) -> None:
        """Set self.enabled = bool(value)."""
        self.enabled = bool(value)

    def reset(self) -> None:
        """Reset enabled -> False and all params -> their defaults."""
        self.enabled = False
        self._values = {p.name: p.default for p in self.params()}

    def is_active(self) -> bool:
        """True iff enabled AND parameters actually change the image.

        Default: return self.enabled. Override to short-circuit identity
        (e.g. blur radius 0 -> False) so the pipeline skips it cheaply.
        """
        return self.enabled

    # ---- generic persistence (works for any param dict) ----
    def to_state(self) -> Dict[str, Any]:
        """Return {"id": self.id, "enabled": self.enabled, "values": <copy>}."""
        return {"id": self.id, "enabled": self.enabled, "values": dict(self._values)}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore enabled + values from a to_state() dict.

        Unknown keys ignored, each value run through Param.clamp().
        """
        if not isinstance(state, dict):
            return
        self.enabled = bool(state.get("enabled", False))
        values = state.get("values", {})
        if not isinstance(values, dict):
            return
        param_map = self._param_map()
        for name, value in values.items():
            spec = param_map.get(name)
            if spec is None:
                continue
            self._values[name] = spec.clamp(value)
