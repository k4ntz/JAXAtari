"""Translate OCAtari Skiing object snapshots into JAXAtari SkiingState.

LOW COMPATIBILITY — best-effort soft survival only. OC Score/Clock .value are
often stuck at 0; pose/slot packing and piste scrolling diverge quickly.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_skiing import JaxSkiing, SkiingState

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers

# Observed OC orientation codes → nearest JAX skier_pos in 0..9 (best-effort).
_ORI_TO_POS = {
    8: 0,
    9: 1,
    10: 2,
    11: 3,
    12: 4,
    13: 5,
    14: 6,
    15: 7,
}


def _clip_speed(raw: float, max_speed: float = 4.0) -> float:
    return float(np.clip(raw, -max_speed, max_speed))


def _delta(
    obj: Mapping[str, Any],
    prev_xy: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    x = float(obj.get("x", 0))
    y = float(obj.get("y", 0))
    if prev_xy is not None:
        return x - float(prev_xy[0]), y - float(prev_xy[1])
    dx = float(obj.get("dx", 0))
    dy = float(obj.get("dy", 0))
    if dx == 0.0 and dy == 0.0:
        dx = x - float(obj.get("prev_x", x))
        dy = y - float(obj.get("prev_y", y))
    return dx, dy


def _pack_xy(
    base: np.ndarray,
    objects: Sequence[Mapping[str, Any]],
    *,
    n_slots: int,
) -> np.ndarray:
    """Overwrite first columns of base with OC xy; leave unused slots as reset."""
    out = np.array(base, dtype=np.float32).copy()
    for i, obj in enumerate(objects[:n_slots]):
        out[i, 0] = float(obj.get("x", 0))
        out[i, 1] = float(obj.get("y", 0))
    return out


def oc_frame_to_skiing_state(
    env: JaxSkiing,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    prev_player_xy: Optional[Tuple[float, float]] = None,
    print_assumptions: bool = False,
    **_ignored,
) -> SkiingState:
    if print_assumptions:
        print_disclaimers("skiing")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs
    updates: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        # Reset skier_x is float32; int32 breaks later lax.cond branches.
        updates["skier_x"] = cast_like(int(player["x"]), state.skier_x)
        ori = player.get("orientation")
        try:
            ori_i = int(ori) if ori is not None else None
        except (TypeError, ValueError):
            ori_i = None
        if ori_i is not None and ori_i in _ORI_TO_POS:
            updates["skier_pos"] = cast_like(_ORI_TO_POS[ori_i], state.skier_pos)
        # else leave reset pose (Q1≈B nearest map when known)

        dx, dy = _delta(player, prev_player_xy)
        updates["skier_x_speed"] = cast_like(_clip_speed(dx), state.skier_x_speed)
        updates["skier_y_speed"] = cast_like(_clip_speed(dy), state.skier_y_speed)

    flags = collect_category(objs, "Flag")
    if flags:
        # Left pole = min x; JAX stores left pole, right = x+flag_distance.
        left = min(flags, key=lambda o: float(o.get("x", 0)))
        flag_arr = np.array(state.flags, dtype=np.dtype(np.asarray(state.flags).dtype)).copy()
        flag_arr[0, 0] = float(left["x"])
        flag_arr[0, 1] = float(left["y"])
        updates["flags"] = cast_like(flag_arr, state.flags)

    trees = collect_category(objs, "Tree")
    if trees:
        tree_arr = _pack_xy(np.asarray(state.trees), trees, n_slots=int(state.trees.shape[0]))
        updates["trees"] = cast_like(tree_arr, state.trees)

    moguls = collect_category(objs, "Mogul")
    if moguls:
        mogul_arr = _pack_xy(np.asarray(state.moguls), moguls, n_slots=int(state.moguls.shape[0]))
        updates["moguls"] = cast_like(mogul_arr, state.moguls)

    score_obj = find_object(objs, "Score")
    if score_obj is not None and int(score_obj.get("value", 0)) > 0:
        updates["successful_gates"] = cast_like(
            int(score_obj["value"]), state.successful_gates
        )

    # Q4=C: step_count from frame_index (Clock.value unreliable).
    updates["step_count"] = cast_like(int(frame_index), state.step_count)

    if updates:
        state = state.replace(**updates)
    return state


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["skier"] = (float(player["x"]), float(player["y"]))

    for i, flag in enumerate(collect_category(objs, "Flag")):
        entities[f"flag_{i}"] = (float(flag["x"]), float(flag["y"]))
    for i, tree in enumerate(collect_category(objs, "Tree")):
        entities[f"tree_{i}"] = (float(tree["x"]), float(tree["y"]))
    for i, mogul in enumerate(collect_category(objs, "Mogul")):
        entities[f"mogul_{i}"] = (float(mogul["x"]), float(mogul["y"]))

    score_obj = find_object(objs, "Score")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        scores["successful_gates"] = float(score_obj["value"])

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: SkiingState, env: JaxSkiing) -> dict:
    entities = {
        "skier": (float(state.skier_x), float(env.consts.skier_y)),
    }
    flags = np.asarray(state.flags)
    for i in range(flags.shape[0]):
        if float(flags[i, 1]) < 250:  # crude on-screen filter
            entities[f"flag_{i}"] = (float(flags[i, 0]), float(flags[i, 1]))
    trees = np.asarray(state.trees)
    for i in range(trees.shape[0]):
        if float(trees[i, 1]) < 250:
            entities[f"tree_{i}"] = (float(trees[i, 0]), float(trees[i, 1]))
    moguls = np.asarray(state.moguls)
    for i in range(moguls.shape[0]):
        if float(moguls[i, 1]) < 250:
            entities[f"mogul_{i}"] = (float(moguls[i, 0]), float(moguls[i, 1]))
    scores = {"successful_gates": float(state.successful_gates)}
    return {"entities": entities, "scores": scores}
