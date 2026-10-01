"""Shared translator helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import jax.numpy as jnp


def cast_like(value: Any, reference: Any):
    """Cast ``value`` to the same dtype as ``reference`` (reset-state leaf).

    Keeps OC→JAX overlays from introducing float32/int32 (or bool/int)
    mismatches that break ``lax.cond`` branches inside game step logic.
    """
    ref = jnp.asarray(reference)
    return jnp.asarray(value, dtype=ref.dtype)


def index_objects_by_category(
    objects: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    """Return the first object for each category (skips NoObject)."""
    by_cat: Dict[str, Mapping[str, Any]] = {}
    for obj in objects:
        cat = str(obj.get("category", "NoObject"))
        if cat == "NoObject":
            continue
        if cat not in by_cat:
            by_cat[cat] = obj
    return by_cat


def find_object(
    objects: Sequence[Mapping[str, Any]], category: str
) -> Optional[Mapping[str, Any]]:
    for obj in objects:
        if str(obj.get("category", "")) == category:
            return obj
    return None


def objects_as_dicts(objects: Iterable[Any]) -> List[Dict[str, Any]]:
    """Normalize live OCAtari GameObjects or already-serialized dicts."""
    out: List[Dict[str, Any]] = []
    for obj in objects:
        if isinstance(obj, Mapping):
            out.append(dict(obj))
            continue
        entry: Dict[str, Any] = {
            "category": getattr(obj, "category", "NoObject"),
            "x": int(getattr(obj, "x", 0)),
            "y": int(getattr(obj, "y", 0)),
            "w": int(getattr(obj, "w", 0)),
            "h": int(getattr(obj, "h", 0)),
            "prev_x": int(getattr(obj, "prev_x", getattr(obj, "x", 0))),
            "prev_y": int(getattr(obj, "prev_y", getattr(obj, "y", 0))),
            "dx": int(getattr(obj, "dx", 0)),
            "dy": int(getattr(obj, "dy", 0)),
            "visible": bool(getattr(obj, "visible", True)),
            "rgb": tuple(getattr(obj, "rgb", (0, 0, 0))),
            "orientation": getattr(obj, "orientation", None),
        }
        if hasattr(obj, "value"):
            entry["value"] = int(getattr(obj, "value"))
        if hasattr(obj, "ten"):
            entry["ten"] = int(getattr(obj, "ten"))
        # Kangaroo (and similar) motion flags exposed on GameObject.
        if hasattr(obj, "climbing"):
            entry["climbing"] = bool(getattr(obj, "climbing"))
        if hasattr(obj, "crashed"):
            entry["crashed"] = bool(getattr(obj, "crashed"))
        # MsPacman pill grid index when present.
        if hasattr(obj, "grid_ij"):
            entry["grid_ij"] = getattr(obj, "grid_ij")
        out.append(entry)
    return out


def collect_category(
    objects: Sequence[Mapping[str, Any]], category: str, *, require_visible: bool = True
) -> List[Mapping[str, Any]]:
    """All objects of ``category`` with optional w>0 filter."""
    out: List[Mapping[str, Any]] = []
    for obj in objects:
        if str(obj.get("category", "")) != category:
            continue
        if require_visible and int(obj.get("w", 0)) <= 0:
            continue
        out.append(obj)
    return out
