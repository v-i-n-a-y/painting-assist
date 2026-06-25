from __future__ import annotations

from typing import Dict, List, Type

from painting_assist.controls.base import Control

_REGISTRY: Dict[str, Type[Control]] = {}


def register(cls: Type[Control]) -> Type[Control]:
    """Class decorator. Validates cls.id is non-empty and unique, registers, returns cls."""
    control_id = getattr(cls, "id", "")
    if not control_id:
        raise ValueError(
            "Control %r must declare a non-empty class attribute 'id'." % cls.__name__
        )
    if control_id in _REGISTRY and _REGISTRY[control_id] is not cls:
        raise ValueError(
            "Duplicate control id %r (already registered by %r)."
            % (control_id, _REGISTRY[control_id].__name__)
        )
    _REGISTRY[control_id] = cls
    return cls


def all_control_classes() -> List[Type[Control]]:
    """Registered classes sorted by (order, id) — the canonical pipeline/panel order."""
    return sorted(_REGISTRY.values(), key=lambda c: (c.order, c.id))


def get_control_class(control_id: str) -> Type[Control]:
    """Look up a registered class by id (raises KeyError if absent)."""
    return _REGISTRY[control_id]


def create_all() -> List[Control]:
    """Instantiate one of every registered control, in (order, id) order.

    The sole entry point MainWindow uses to build the pipeline + panel.
    """
    return [cls() for cls in all_control_classes()]


def registered_ids() -> List[str]:
    """List of registered ids in (order, id) order."""
    return [cls.id for cls in all_control_classes()]
