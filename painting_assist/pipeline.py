# Copyright 2026 Vinay Williams

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from painting_assist.controls.base import Control
from painting_assist.controls.registry import create_all

# Peak-memory bound. The prefix cache holds one full-resolution array per active
# stage, which over a long chain on a large image is otherwise unbounded (~1.25
# GB on a 50 MP full chain). After each render at most this many cached arrays
# are retained; see ControlPipeline._trim_cache for why dropping the rest is
# safe for correctness.
MAX_CACHED_STAGES = 4


@dataclass
class ControlState:
    """Immutable-ish snapshot of one control's (enabled, values) for the worker."""

    enabled: bool
    values: Dict[str, Any]

    def copy(self) -> "ControlState":
        """Return a deep-ish copy (values dict duplicated)."""
        return ControlState(enabled=self.enabled, values=dict(self.values))


@dataclass
class _CacheEntry:
    """One cached pipeline-prefix output, keyed by the cumulative chain key."""

    chain_key: Tuple[Any, ...]
    image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class ControlPipeline:
    """Owns the ordered control list and computes the processed image with a prefix cache.

    Qt-free and thread-agnostic: "output changed" is delivered via RenderController's
    callback path, not a Qt signal.
    """

    def __init__(self, controls: Optional[List[Control]] = None) -> None:
        """Default: registry.create_all(). Stores controls in given (registry) order,
        a per-stage cache (one slot per stage), and the id() of the base image.
        """
        self._controls: List[Control] = (
            controls if controls is not None else create_all()
        )
        self._by_id: Dict[str, Control] = {c.id: c for c in self._controls}
        self._cache: List[Optional[_CacheEntry]] = [None] * len(self._controls)
        self._base_token: Optional[Any] = None

    # ---- introspection (GUI thread; for the panel) ----
    def controls(self) -> List[Control]:
        """Return the ordered control list."""
        return self._controls

    def control(self, control_id: str) -> Control:
        """Return the control with the given id (raises KeyError if absent)."""
        return self._by_id[control_id]

    # ---- mutation (GUI thread; cheap, no processing) ----
    def set_value(self, control_id: str, name: str, value: Any) -> None:
        """Set a param value on a control (no processing)."""
        self._by_id[control_id].set(name, value)

    def set_enabled(self, control_id: str, enabled: bool) -> None:
        """Enable/disable a control (no processing)."""
        self._by_id[control_id].set_enabled(enabled)

    def reset(self) -> None:
        """reset() every control; clears cache."""
        for control in self._controls:
            control.reset()
        self._cache = [None] * len(self._controls)
        self._base_token = None

    # ---- snapshot for the worker ----
    def snapshot_states(self) -> Dict[str, ControlState]:
        """Deep-ish copy of each control's (enabled, values) -> immune to GUI edits."""
        return {
            control.id: ControlState(enabled=control.enabled, values=control.values())
            for control in self._controls
        }

    # ---- the heavy call (runs on the WORKER thread) ----
    def process(
        self,
        source: np.ndarray,
        states: Optional[Dict[str, ControlState]] = None,
        token: Optional[Any] = None,
        metadata_out: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Apply active controls in order over `source`, reusing cached prefixes.

        If states is given, each control's values/enabled are taken from the snapshot
        (so the worker is decoupled from concurrent GUI edits); else uses live control
        state. Always derives from `source` (non-destructive). Returns RGB uint8.

        `token` identifies the base image (source array + scale combo). The whole
        cache is invalidated when it changes. Callers that render the *same* logical
        source at the same scale should pass a stable token so the prefix cache is
        reused even when a fresh (e.g. downscaled) array is passed each frame. When
        omitted it falls back to ``id(source)``; this is only safe for one-shot
        callers (Save/Export on an isolated pipeline) because a freed array's id can
        be reused by a later, different array.

        If `metadata_out` is given, it is filled with the side-channel outputs
        (see :meth:`Control.emit_metadata`) of every active stage this call
        touches, including stages served from the prefix cache — the metadata is
        cached alongside each stage's image, so a palette does not vanish when
        the quantize stage is a cache hit.
        """
        if token is None:
            token = id(source)
        # If the base image changed, the whole cache is stale.
        if self._base_token != token:
            self._cache = [None] * len(self._controls)
            self._base_token = token

        current = source
        chain_key: Tuple[Any, ...] = ()

        for i, control in enumerate(self._controls):
            active, stage_key = self._stage_info(control, states)
            chain_key = chain_key + (control.id, stage_key)

            entry = self._cache[i]
            if entry is not None and entry.chain_key == chain_key:
                # Hit: reuse this stage's cached output as the input to the next.
                current = entry.image
                if metadata_out is not None and entry.metadata:
                    metadata_out.update(entry.metadata)
                continue

            # Miss: compute this stage from the current input.
            stage_meta: Dict[str, Any] = {}
            if active:
                output = self._run_stage(control, current, states, stage_meta)
            else:
                output = current

            self._cache[i] = _CacheEntry(
                chain_key=chain_key, image=output, metadata=stage_meta
            )
            if metadata_out is not None and stage_meta:
                metadata_out.update(stage_meta)
            # Downstream slots are left alone: their chain_key embeds every
            # upstream stage key, so they miss by themselves if this stage's
            # state changed, and hit again if this was merely a slot that
            # _trim_cache had evicted and we have just recomputed identically.
            current = output

        self._trim_cache()
        return current

    # ---- internals ----
    def _stage_info(
        self,
        control: Control,
        states: Optional[Dict[str, ControlState]],
    ) -> Tuple[bool, Tuple[Any, ...]]:
        """Return (active, stage_key) for a control under the given (or live) state.

        stage_key is ("off",) when inactive, else a stable rounded tuple of values.
        """
        if states is not None:
            state = states.get(control.id)
            if state is None:
                return False, ("off",)
            active = self._active_for_state(control, state)
            if not active:
                return False, ("off",)
            return True, self._values_key(state.values)

        # Live state.
        if not control.is_active():
            return False, ("off",)
        return True, self._values_key(control.values())

    def _active_for_state(self, control: Control, state: ControlState) -> bool:
        """Evaluate is_active() against a snapshot without touching live state.

        Uses ``is_active_snapshot`` so the worker thread never reads or mutates the
        live control (which the GUI thread may be editing concurrently).
        """
        if not state.enabled:
            return False
        return control.is_active_snapshot(state.enabled, state.values)

    @staticmethod
    def _values_key(values: Dict[str, Any]) -> Tuple[Any, ...]:
        """Build a stable, hashable key from a values dict (rounded floats)."""
        items = []
        for name in sorted(values.keys()):
            value = values[name]
            if isinstance(value, float):
                value = round(value, 6)
            items.append((name, value))
        return tuple(items)

    def _run_stage(
        self,
        control: Control,
        current: np.ndarray,
        states: Optional[Dict[str, ControlState]],
        stage_meta: Dict[str, Any],
    ) -> np.ndarray:
        """Run control.process(current) under the given snapshot (or live state).

        Any metadata the control emits during the run is collected into
        ``stage_meta``. With a snapshot, ``process_snapshot_meta`` runs against a
        throwaway copy so the live control is never mutated on the worker thread.
        """
        if states is None:
            control._metadata = {}
            out = control.process(current)
            if control._metadata:
                stage_meta.update(control._metadata)
            return out

        state = states[control.id]
        out, meta = control.process_snapshot_meta(current, state.enabled, state.values)
        if meta:
            stage_meta.update(meta)
        return out

    def _invalidate_from(self, i: int) -> None:
        """Null cache slots i..end (downstream stages stale)."""
        for j in range(i, len(self._cache)):
            self._cache[j] = None

    def _trim_cache(self) -> None:
        """Bound peak memory to at most MAX_CACHED_STAGES retained full-res arrays.

        Each populated cache slot holds one full-resolution array, so a long chain
        over a large image would otherwise pin one array per active stage. Once a
        render has finished we null the LOWEST-index populated slots down to the
        cap, keeping the most-downstream entries.

        This never changes what a later :meth:`process` returns: a dropped upstream
        prefix is simply recomputed from ``source`` on the next call, and any
        surviving entry is still validated by its ``chain_key`` (which encodes the
        upstream stage keys, not array identity) before it is reused, so it can
        only be served when it is genuinely correct.
        """
        populated = [i for i, entry in enumerate(self._cache) if entry is not None]
        excess = len(populated) - MAX_CACHED_STAGES
        # max(0, ...): a negative count would slice from the end (populated[:-1]),
        # nulling live upstream slots on a short chain instead of doing nothing.
        for i in populated[: max(0, excess)]:
            self._cache[i] = None
