"""Lockstep comparison metrics: entities, score, pixels, survival.

Survival is intentionally *tolerant* of small timing/physics jitter:
paddle one frame late, ball bounce a few pixels off, etc. should not
immediately count as a dynamics failure.

Two layers:
  - **strict**: max entity L∞ > ``tau`` on a single frame (diagnostic / brittle)
  - **soft** (default headline): mean entity L1 > ``soft_tau`` for ``persist``
    consecutive frames (or a primary entity missing that long)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Entity keys that participate in survival / first-diverge (positions only).
PRIMARY_ENTITY_PREFIXES = (
    "player",
    "chicken",
    "skier",
    "bailey",
    "enemy",
    "ball",
    "bank_",
    "police_",
    "dynamite",
    "car_",
    "flag_",
    "tree_",
    "mogul_",
    "diver_",
    "shark_",
    "missile",
    "fruit_",
    "monkey_",
    "child",
    "bell",
    "falling_coco",
    "monster_",
    "hallmonster_",
    "ghost_",
    "powerpill_",
    "power_pill_",
    "ice_",
    "bird_",
    "obstacle_",
    "skull_",
    "key",
    "item_",
    "saucer_",
    "white_ufo_",
    "phoenix_",
    "enemy_missile_",
    "sam",
    "snake",
    "coily",
)

# Defaults tuned so 1-frame accel lag / small bounce offsets are not fatal.
DEFAULT_TAU_STRICT = 2.0
DEFAULT_SOFT_TAU = 12.0  # mean L1 over primary entities (px)
DEFAULT_PERSIST = 5  # consecutive soft-breach frames before "diverged"


def pixel_metrics(oc_frame: np.ndarray, jax_frame: np.ndarray) -> Dict[str, float]:
    """RGB MAE (0–255 scale and normalized) + exact-equal fraction."""
    a = np.asarray(oc_frame, dtype=np.float32)
    b = np.asarray(jax_frame, dtype=np.float32)
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        c = min(a.shape[2], b.shape[2]) if a.ndim == 3 else 1
        a = a[:h, :w, :c] if a.ndim == 3 else a[:h, :w]
        b = b[:h, :w, :c] if b.ndim == 3 else b[:h, :w]
    diff = np.abs(a - b)
    mae = float(diff.mean())
    equal = float(np.mean(a == b))
    return {
        "pixel_mae": mae,
        "pixel_mae_norm": mae / 255.0,
        "pixel_equal_frac": equal,
    }


def _is_primary_entity(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in PRIMARY_ENTITY_PREFIXES)


def entity_metrics(
    oc: Mapping[str, Any],
    jax: Mapping[str, Any],
    *,
    tau: float = DEFAULT_TAU_STRICT,
    soft_tau: float = DEFAULT_SOFT_TAU,
) -> Dict[str, Any]:
    """Compare entity xy dicts and score dicts.

    ``strict_breach``: any primary L∞ > ``tau`` or missing entity (brittle).
    ``soft_breach``: mean primary L1 > ``soft_tau`` or missing entity (dynamics).
    """
    oc_ent: Dict[str, Tuple[float, float]] = dict(oc.get("entities", {}))
    jax_ent: Dict[str, Tuple[float, float]] = dict(jax.get("entities", {}))
    oc_scores: Dict[str, float] = dict(oc.get("scores", {}))
    jax_scores: Dict[str, float] = dict(jax.get("scores", {}))

    keys = sorted(set(oc_ent) | set(jax_ent))
    primary_keys = [k for k in keys if _is_primary_entity(k)]

    per_entity: Dict[str, Dict[str, float]] = {}
    l1s: List[float] = []
    linfs: List[float] = []
    matched = 0
    missing = 0

    for k in primary_keys:
        if k not in oc_ent or k not in jax_ent:
            missing += 1
            per_entity[k] = {"l1": float("inf"), "linf": float("inf"), "present_both": 0.0}
            continue
        ox, oy = oc_ent[k]
        jx, jy = jax_ent[k]
        l1 = abs(ox - jx) + abs(oy - jy)
        linf = max(abs(ox - jx), abs(oy - jy))
        l1s.append(l1)
        linfs.append(linf)
        matched += 1
        per_entity[k] = {"l1": float(l1), "linf": float(linf), "present_both": 1.0}

    presence_denom = max(len(primary_keys), 1)
    presence = matched / presence_denom

    score_deltas: Dict[str, float] = {}
    score_keys = sorted(set(oc_scores) | set(jax_scores))
    for k in score_keys:
        if k not in oc_scores or k not in jax_scores:
            continue
        score_deltas[k] = abs(float(oc_scores[k]) - float(jax_scores[k]))

    mean_l1 = float(np.mean(l1s)) if l1s else (float("inf") if missing else 0.0)
    mean_linf = float(np.mean(linfs)) if linfs else (float("inf") if missing else 0.0)
    max_linf = float(np.max(linfs)) if linfs else (float("inf") if missing else 0.0)

    missing_breach = missing > 0
    strict_breach = bool(missing_breach or (linfs and max_linf > tau))
    soft_breach = bool(missing_breach or (np.isfinite(mean_l1) and mean_l1 > soft_tau))

    mean_score_delta = float(np.mean(list(score_deltas.values()))) if score_deltas else 0.0
    max_score_delta = float(np.max(list(score_deltas.values()))) if score_deltas else 0.0

    return {
        "entity_mean_l1": mean_l1,
        "entity_mean_linf": mean_linf,
        "entity_max_linf": max_linf,
        "entity_presence": float(presence),
        "entity_matched": int(matched),
        "entity_missing": int(missing),
        # Back-compat alias = strict single-frame breach
        "entity_diverged": strict_breach,
        "strict_breach": strict_breach,
        "soft_breach": soft_breach,
        "per_entity": per_entity,
        "score_deltas": score_deltas,
        "score_mean_abs": mean_score_delta,
        "score_max_abs": max_score_delta,
    }


def combine_frame_metrics(
    entity: Mapping[str, Any],
    pixel: Mapping[str, float],
) -> Dict[str, Any]:
    out = {k: v for k, v in entity.items() if k != "per_entity"}
    out.update(pixel)
    out["per_entity"] = entity.get("per_entity", {})
    out["score_deltas"] = entity.get("score_deltas", {})
    return out


def first_diverge_index(
    diverged_flags: Sequence[bool],
) -> Optional[int]:
    """Index of first True, or None if never diverged (survived)."""
    for i, d in enumerate(diverged_flags):
        if d:
            return i
    return None


def first_diverge_persistent(
    breach_flags: Sequence[bool],
    *,
    persist: int = DEFAULT_PERSIST,
) -> Optional[int]:
    """First index where ``persist`` consecutive breaches begin.

    Returns the start of that streak, or None if no such streak exists.
    ``persist <= 1`` reduces to ``first_diverge_index``.
    """
    if persist <= 1:
        return first_diverge_index(breach_flags)
    run = 0
    for i, d in enumerate(breach_flags):
        if d:
            run += 1
            if run >= persist:
                return i - persist + 1
        else:
            run = 0
    return None


def survival_rate(first_diverges: Sequence[Optional[int]], n: int) -> float:
    """Fraction of runs with no diverge in ``[0, n)`` (None counts as survived)."""
    if not first_diverges:
        return 0.0
    survived = sum(1 for fd in first_diverges if fd is None or fd >= n)
    return survived / len(first_diverges)
