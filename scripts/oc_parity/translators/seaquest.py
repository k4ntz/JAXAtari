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


def _entity_direction(
    obj: Mapping[str, Any],
    default_dir: int,
    *,
    prev_objects: Optional[Sequence[Mapping[str, Any]]] = None,
    lane_ys: Optional[np.ndarray] = None,
) -> int:
    """Prefer dx; else prev-frame Δx in-lane; else OC orientation; else default.

    Shark/Sub dx is often 0 on skip frames — orientation E/W or lookback is needed.
    """
    dx = float(obj.get("dx", 0))
    if dx == 0.0:
        dx = float(obj.get("x", 0)) - float(obj.get("prev_x", obj.get("x", 0)))
    if dx < 0:
        return -1
    if dx > 0:
        return 1
    # Same-frame prev_x often already equals x; match prev-frame object by lane.
    if prev_objects and lane_ys is not None:
        lane = _nearest_lane(float(obj.get("y", 0)), lane_ys)
        candidates = [
            p
            for p in prev_objects
            if _nearest_lane(float(p.get("y", 0)), lane_ys) == lane
        ]
        if candidates:
            # Nearest x in that lane.
            prev = min(
                candidates,
                key=lambda p: abs(float(p.get("x", 0)) - float(obj.get("x", 0))),
            )
            ddx = float(obj.get("x", 0)) - float(prev.get("x", 0))
            if ddx < 0:
                return -1
            if ddx > 0:
                return 1
            pdx = float(prev.get("dx", 0))
            if pdx < 0:
                return -1
            if pdx > 0:
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
    prev_objects: Optional[Sequence[Mapping[str, Any]]] = None,
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
            out[idx, 2] = float(
                _entity_direction(
                    obj,
                    default_dir,
                    prev_objects=prev_objects,
                    lane_ys=lane_ys,
                )
            )
    return cast_like(out, ref)


def _pack_divers_by_lane(
    base: Any,
    objects: Sequence[Mapping[str, Any]],
    lane_ys: np.ndarray,
    *,
    default_dir: int,
    prev_objects: Optional[Sequence[Mapping[str, Any]]] = None,
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
        out[lane, 2] = float(
            _entity_direction(
                obj,
                default_dir,
                prev_objects=prev_objects,
                lane_ys=lane_ys,
            )
        )
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
    prev_objects: Optional[Sequence[Any]] = None,
    **_ignored,
) -> SeaquestState:
    del frame_index
    if print_assumptions:
        print_disclaimers("seaquest")

    consts = env.consts
    objs = objects_as_dicts(objects)
    prev_objs = objects_as_dicts(prev_objects or [])
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
        updates["death_counter"] = cast_like(0, state.death_counter)
    else:
        # OC omits Player during death / blink — match JAX hide convention so
        # lockstep does not compare a leftover reset player at (76, 46).
        updates["player_x"] = cast_like(-100, state.player_x)
        updates["death_counter"] = cast_like(45, state.death_counter)
        updates["player_missile_position"] = cast_like(
            [0, 0, 0], state.player_missile_position
        )

    oxygen = _oxygen_from_oc(objs)
    if oxygen is not None:
        updates["oxygen"] = cast_like(oxygen, state.oxygen)

    # CollectedDiver HUD icons → divers_collected (was left at RESET=0).
    collected = [
        o for o in collect_category(objs, "CollectedDiver") if int(o.get("w", 0)) > 0
    ]
    updates["divers_collected"] = cast_like(len(collected), state.divers_collected)

    # just_surfaced=-1 is cold-start oxygen-fill mode. Mid-game underwater → 0.
    player_y_val = int(updates.get("player_y", state.player_y))
    if player_y_val > 52:
        updates["just_surfaced"] = cast_like(0, state.just_surfaced)

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
            state.diver_positions,
            divers,
            diver_lane_ys,
            default_dir=1,
            prev_objects=collect_category(prev_objs, "Diver"),
        )

    sharks = collect_category(objs, "Shark")
    if sharks:
        updates["shark_positions"] = _pack_enemies_by_lane(
            state.shark_positions,
            sharks,
            shark_lane_ys,
            default_dir=1,
            prev_objects=collect_category(prev_objs, "Shark"),
        )

    # Subs were previously left RESET (probe often missed them); map when OC exposes Submarine.
    subs = collect_category(objs, "Submarine")
    if not subs:
        # Some OC builds use shorter names.
        subs = collect_category(objs, "Sub")
    if subs:
        updates["sub_positions"] = _pack_enemies_by_lane(
            state.sub_positions,
            subs,
            shark_lane_ys,
            default_dir=1,
            prev_objects=collect_category(prev_objs, "Submarine")
            or collect_category(prev_objs, "Sub"),
        )

    missile = find_object(objs, "PlayerMissile")
    if missile is not None and int(missile.get("w", 0)) > 0:
        direction = _entity_direction(missile, 1)
        updates["player_missile_position"] = cast_like(
            [int(missile["x"]), int(missile["y"]), direction],
            state.player_missile_position,
        )

    # Soft-disable immediate re-spawns; clear RESET diver escort flags; align
    # lane_directions with packed enemy facing (JAX: 1=moving left, 0=right).
    sp = state.spawn_state
    prev_sub = np.array(sp.prev_sub, copy=True)
    shark_packed = np.asarray(updates.get("shark_positions", state.shark_positions))
    sub_packed = np.asarray(updates.get("sub_positions", state.sub_positions))
    for lane in range(4):
        sl = slice(lane * 3, (lane + 1) * 3)
        has_sub = np.any(sub_packed[sl, 2] != 0)
        has_shark = np.any(shark_packed[sl, 2] != 0)
        if has_sub and not has_shark:
            prev_sub[lane] = 1
        elif has_shark:
            prev_sub[lane] = 0
    lane_dirs = np.array(sp.lane_directions, copy=True)
    for lane in range(4):
        sl = slice(lane * 3, (lane + 1) * 3)
        for row in list(shark_packed[sl]) + list(sub_packed[sl]):
            if int(row[2]) != 0:
                lane_dirs[lane] = 1 if int(row[2]) < 0 else 0
                break
    # diver_array semantics in jax_seaquest:
    #   1 = spawn escort diver when timer hits 60
    #   0 = collected (and ALL-zero → force [-1]*4 → re-arm to 1)
    #  -1 = swam off, re-arm to 1 when lane empty
    # Use 2 as a soft "disabled" sentinel so we don't auto-rearm escorts.
    updates["spawn_state"] = sp.replace(
        spawn_timers=jnp.full_like(sp.spawn_timers, 9999),
        to_be_spawned=jnp.zeros_like(sp.to_be_spawned),
        prev_sub=cast_like(prev_sub, sp.prev_sub),
        diver_array=jnp.full_like(sp.diver_array, 2),
        lane_directions=cast_like(lane_dirs, sp.lane_directions),
    )

    if updates:
        state = state.replace(**updates)
    return state


def _lane_slot_xy_entities(
    objects: Sequence[Mapping[str, Any]],
    lane_ys: np.ndarray,
    *,
    prefix: str,
    slots_per_lane: int,
) -> dict:
    """Assign stable ``{prefix}_{lane*slots+slot}`` keys (same packing as inject)."""
    n_lanes = int(len(lane_ys))
    buckets: List[List[Mapping[str, Any]]] = [[] for _ in range(n_lanes)]
    for obj in objects:
        buckets[_nearest_lane(float(obj.get("y", 0)), lane_ys)].append(obj)
    entities: dict = {}
    for lane, bucket in enumerate(buckets):
        ordered = sorted(bucket, key=lambda o: float(o.get("x", 0)))[:slots_per_lane]
        for slot, obj in enumerate(ordered):
            idx = lane * slots_per_lane + slot
            entities[f"{prefix}_{idx}"] = (float(obj.get("x", 0)), float(obj.get("y", 0)))
    return entities


def extract_oc_compare_entities(objects: Sequence[Any], **kwargs) -> dict:
    """OC entities keyed by lane/slot so they align with JAX packed arrays.

    Previously used OC list order (``shark_0`` = first detected), while JAX used
    compacted packed order — pairing unrelated sharks and inflating L1.
    """
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    consts = kwargs.get("consts")
    shark_lane_ys = np.asarray(
        getattr(consts, "SPAWN_POSITIONS_Y", [71, 95, 119, 139]), dtype=np.float64
    )
    diver_lane_ys = np.asarray(
        getattr(consts, "DIVER_SPAWN_POSITIONS", [69, 93, 117, 141]), dtype=np.float64
    )

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["player"] = (float(player["x"]), float(player["y"]))

    entities.update(
        _lane_slot_xy_entities(
            collect_category(objs, "Diver"),
            diver_lane_ys,
            prefix="diver",
            slots_per_lane=1,
        )
    )
    entities.update(
        _lane_slot_xy_entities(
            collect_category(objs, "Shark"),
            shark_lane_ys,
            prefix="shark",
            slots_per_lane=3,
        )
    )
    subs = collect_category(objs, "Submarine") or collect_category(objs, "Sub")
    entities.update(
        _lane_slot_xy_entities(subs, shark_lane_ys, prefix="sub", slots_per_lane=3)
    )

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
    """Emit ``shark_{slot}`` / ``sub_{slot}`` / ``diver_{lane}`` (not compacted).

    Skip player while hidden (``player_x < 0`` / death blink), matching OC frames
    that omit the Player category.
    """
    del env
    entities: dict = {}
    if float(state.player_x) >= 0:
        entities["player"] = (float(state.player_x), float(state.player_y))
    divers = np.asarray(state.diver_positions)
    for i in range(divers.shape[0]):
        if int(divers[i, 2]) != 0:
            entities[f"diver_{i}"] = (float(divers[i, 0]), float(divers[i, 1]))
    sharks = np.asarray(state.shark_positions)
    for i in range(sharks.shape[0]):
        if int(sharks[i, 2]) != 0:
            entities[f"shark_{i}"] = (float(sharks[i, 0]), float(sharks[i, 1]))
    subs = np.asarray(state.sub_positions)
    for i in range(subs.shape[0]):
        if int(subs[i, 2]) != 0:
            entities[f"sub_{i}"] = (float(subs[i, 0]), float(subs[i, 1]))
    missile = np.asarray(state.player_missile_position)
    if int(missile[2]) != 0:
        entities["missile"] = (float(missile[0]), float(missile[1]))
    scores = {
        "oxygen": float(state.oxygen),
        "score": float(state.score),
        "lives": float(state.lives),
    }
    return {"entities": entities, "scores": scores}
