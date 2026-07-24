"""Translate OCAtari Seaquest object snapshots into JAXAtari SeaquestState."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestState

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers


def _orientation_to_dir(orientation: Any, consts) -> int:
    """Map OC orientation to FACE_RIGHT(+1) / FACE_LEFT(-1)."""
    name = getattr(orientation, "name", None) or str(orientation)
    name = str(name).upper()
    if any(tok in name for tok in ("W", "LEFT", "270", "WEST")):
        return int(consts.FACE_LEFT)
    if any(tok in name for tok in ("E", "RIGHT", "90", "EAST")):
        return int(consts.FACE_RIGHT)
    # Numeric Orientation enums: E=4 often.
    try:
        val = int(orientation)
        if val in (3, 4, 5):  # SE/E/NE-ish
            return int(consts.FACE_RIGHT)
        if val in (1, 7, 8):  # SW/W/NW-ish depending on enum
            return int(consts.FACE_LEFT)
    except (TypeError, ValueError):
        pass
    return int(consts.FACE_RIGHT)


def _entity_direction(obj: Mapping[str, Any], default_dir: int) -> int:
    """Prefer dx; else OC orientation; else default.

    Shark/Sub dx is often 0 on skip frames — orientation E/W is the reliable signal.
    """
    dx = float(obj.get("dx", 0))
    if dx == 0.0:
        dx = float(obj.get("x", 0)) - float(obj.get("prev_x", obj.get("x", 0)))
    if dx < 0:
        return -1
    if dx > 0:
        return 1
    ori = obj.get("orientation")
    if ori is not None:
        name = str(getattr(ori, "name", ori)).upper()
        if any(tok in name for tok in ("W", "LEFT", "270", "WEST")):
            return -1
        if any(tok in name for tok in ("E", "RIGHT", "90", "EAST")):
            return 1
    return int(default_dir)


def _nearest_lane(y: float, lane_ys: np.ndarray) -> int:
    return int(np.argmin(np.abs(lane_ys.astype(np.float64) - float(y))))


def _pack_enemies_by_lane(
    base: Any,
    objects: Sequence[Mapping[str, Any]],
    lane_ys: np.ndarray,
    *,
    default_dir: int,
) -> jnp.ndarray:
    """Pack sharks/subs into 4 lanes × 3 slots (left→right in-lane).

    JAX ``move_enemies`` overwrites Y from ``SPAWN_POSITIONS_Y[slot // 3]`` every
    step, so list-order packing puts bottom-lane enemies into lane 0.
    """
    ref = jnp.asarray(base)
    out = np.zeros(ref.shape, dtype=np.dtype(ref.dtype))
    buckets: List[List[Mapping[str, Any]]] = [[] for _ in range(4)]
    for obj in objects:
        buckets[_nearest_lane(float(obj.get("y", 0)), lane_ys)].append(obj)

    for lane, bucket in enumerate(buckets):
        ordered = sorted(bucket, key=lambda o: float(o.get("x", 0)))[:3]
        for slot, obj in enumerate(ordered):
            idx = lane * 3 + slot
            out[idx, 0] = float(obj.get("x", 0))
            out[idx, 1] = float(lane_ys[lane])
            out[idx, 2] = float(_entity_direction(obj, default_dir))
    return cast_like(out, ref)


def _pack_divers_by_lane(
    base: Any,
    objects: Sequence[Mapping[str, Any]],
    lane_ys: np.ndarray,
    *,
    default_dir: int,
) -> jnp.ndarray:
    """One diver slot per lane (shape ``(4, 3)``); Y snapped to ``DIVER_SPAWN_POSITIONS``."""
    ref = jnp.asarray(base)
    out = np.zeros(ref.shape, dtype=np.dtype(ref.dtype))
    # If multiple OC divers map to the same lane, keep the left-most.
    chosen: dict = {}
    for obj in objects:
        lane = _nearest_lane(float(obj.get("y", 0)), lane_ys)
        prev = chosen.get(lane)
        if prev is None or float(obj.get("x", 0)) < float(prev.get("x", 0)):
            chosen[lane] = obj
    for lane, obj in chosen.items():
        out[lane, 0] = float(obj.get("x", 0))
        out[lane, 1] = float(lane_ys[lane])
        out[lane, 2] = float(_entity_direction(obj, default_dir))
    return cast_like(out, ref)


def _oxygen_from_oc(objs: Sequence[Mapping[str, Any]]) -> Optional[int]:
    bar = find_object(objs, "OxygenBar")
    depleted = find_object(objs, "OxygenBarDepleted")
    if bar is not None and "value" in bar and int(bar.get("value", -1)) >= 0:
        return int(np.clip(int(bar["value"]), 0, 64))
    # Brief fallback: width heuristic if value missing.
    if bar is not None and int(bar.get("w", 0)) > 0:
        w = float(bar["w"])
        dw = float(depleted.get("w", 0)) if depleted is not None else 0.0
        total = w + dw
        if total > 0:
            return int(np.clip(round(64.0 * w / total), 0, 64))
        return int(np.clip(round(w), 0, 64))
    return None


def oc_frame_to_seaquest_state(
    env: JaxSeaquest,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> SeaquestState:
    del frame_index
    if print_assumptions:
        print_disclaimers("seaquest")

    consts = env.consts
    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs
    updates: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        updates["player_x"] = cast_like(int(player["x"]), state.player_x)
        updates["player_y"] = cast_like(int(player["y"]), state.player_y)
        updates["player_direction"] = cast_like(
            _orientation_to_dir(player.get("orientation"), consts),
            state.player_direction,
        )

    oxygen = _oxygen_from_oc(objs)
    if oxygen is not None:
        updates["oxygen"] = cast_like(oxygen, state.oxygen)

    score_obj = find_object(objs, "PlayerScore")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        updates["score"] = cast_like(int(score_obj["value"]), state.score)
    lives_obj = find_object(objs, "Lives")
    if lives_obj is not None and int(lives_obj.get("value", -1)) >= 0:
        updates["lives"] = cast_like(int(lives_obj["value"]), state.lives)

    shark_lane_ys = np.asarray(consts.SPAWN_POSITIONS_Y)
    diver_lane_ys = np.asarray(consts.DIVER_SPAWN_POSITIONS)

    divers = collect_category(objs, "Diver")
    if divers:
        updates["diver_positions"] = _pack_divers_by_lane(
            state.diver_positions, divers, diver_lane_ys, default_dir=1
        )

    sharks = collect_category(objs, "Shark")
    if sharks:
        updates["shark_positions"] = _pack_enemies_by_lane(
            state.shark_positions, sharks, shark_lane_ys, default_dir=1
        )

    # Subs were previously left RESET (probe often missed them); map when OC exposes Submarine.
    subs = collect_category(objs, "Submarine")
    if not subs:
        # Some OC builds use shorter names.
        subs = collect_category(objs, "Sub")
    if subs:
        updates["sub_positions"] = _pack_enemies_by_lane(
            state.sub_positions, subs, shark_lane_ys, default_dir=1
        )

    missile = find_object(objs, "PlayerMissile")
    if missile is not None and int(missile.get("w", 0)) > 0:
        direction = _entity_direction(missile, 1)
        updates["player_missile_position"] = cast_like(
            [int(missile["x"]), int(missile["y"]), direction],
            state.player_missile_position,
        )

    # Soft-disable spawn timers when enemies placed.
    # prev_sub: 1 for lanes with subs, 0 for shark lanes (mixed → prefer what's present).
    if sharks or divers or subs:
        sp = state.spawn_state
        prev_sub = np.array(sp.prev_sub, copy=True)
        shark_packed = np.asarray(
            updates.get("shark_positions", state.shark_positions)
        )
        sub_packed = np.asarray(updates.get("sub_positions", state.sub_positions))
        for lane in range(4):
            sl = slice(lane * 3, (lane + 1) * 3)
            has_sub = np.any(sub_packed[sl, 2] != 0)
            has_shark = np.any(shark_packed[sl, 2] != 0)
            if has_sub and not has_shark:
                prev_sub[lane] = 1
            elif has_shark:
                prev_sub[lane] = 0
        updates["spawn_state"] = sp.replace(
            spawn_timers=jnp.full_like(sp.spawn_timers, 9999),
            to_be_spawned=jnp.zeros_like(sp.to_be_spawned),
            prev_sub=cast_like(prev_sub, sp.prev_sub),
        )

    # Q5=A: divers_collected RESET

    if updates:
        state = state.replace(**updates)
    return state


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["player"] = (float(player["x"]), float(player["y"]))
    for i, d in enumerate(collect_category(objs, "Diver")):
        entities[f"diver_{i}"] = (float(d["x"]), float(d["y"]))
    for i, s in enumerate(collect_category(objs, "Shark")):
        entities[f"shark_{i}"] = (float(s["x"]), float(s["y"]))
    for i, s in enumerate(collect_category(objs, "Submarine") or collect_category(objs, "Sub")):
        entities[f"sub_{i}"] = (float(s["x"]), float(s["y"]))
    missile = find_object(objs, "PlayerMissile")
    if missile is not None and int(missile.get("w", 0)) > 0:
        entities["missile"] = (float(missile["x"]), float(missile["y"]))

    oxygen = _oxygen_from_oc(objs)
    if oxygen is not None:
        scores["oxygen"] = float(oxygen)
    score_obj = find_object(objs, "PlayerScore")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        scores["score"] = float(score_obj["value"])
    lives_obj = find_object(objs, "Lives")
    if lives_obj is not None and int(lives_obj.get("value", -1)) >= 0:
        scores["lives"] = float(lives_obj["value"])

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: SeaquestState, env: JaxSeaquest) -> dict:
    del env
    entities = {
        "player": (float(state.player_x), float(state.player_y)),
    }
    divers = np.asarray(state.diver_positions)
    di = 0
    for i in range(divers.shape[0]):
        if int(divers[i, 2]) != 0:
            entities[f"diver_{di}"] = (float(divers[i, 0]), float(divers[i, 1]))
            di += 1
    sharks = np.asarray(state.shark_positions)
    si = 0
    for i in range(sharks.shape[0]):
        if int(sharks[i, 2]) != 0:
            entities[f"shark_{si}"] = (float(sharks[i, 0]), float(sharks[i, 1]))
            si += 1
    subs = np.asarray(state.sub_positions)
    sui = 0
    for i in range(subs.shape[0]):
        if int(subs[i, 2]) != 0:
            entities[f"sub_{sui}"] = (float(subs[i, 0]), float(subs[i, 1]))
            sui += 1
    missile = np.asarray(state.player_missile_position)
    if int(missile[2]) != 0:
        entities["missile"] = (float(missile[0]), float(missile[1]))
    scores = {
        "oxygen": float(state.oxygen),
        "score": float(state.score),
        "lives": float(state.lives),
    }
    return {"entities": entities, "scores": scores}
