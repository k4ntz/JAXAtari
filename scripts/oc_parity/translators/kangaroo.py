"""Translate OCAtari Kangaroo object snapshots into JAXAtari KangarooState."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_kangaroo import JaxKangaroo, KangarooState, PlayerState

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers


def _orientation_pm1(orientation: Any) -> int:
    name = str(getattr(orientation, "name", orientation)).upper()
    if "W" in name or "LEFT" in name:
        return -1
    return 1


def _delta_xy(obj: Mapping[str, Any]) -> Tuple[float, float]:
    dx = float(obj.get("dx", 0))
    dy = float(obj.get("dy", 0))
    if dx == 0.0 and dy == 0.0:
        dx = float(obj.get("x", 0)) - float(obj.get("prev_x", obj.get("x", 0)))
        dy = float(obj.get("y", 0)) - float(obj.get("prev_y", obj.get("y", 0)))
    return dx, dy


def _monkey_state_from_delta(obj: Mapping[str, Any]) -> int:
    """Invent monkey FSM state from Δ.
    Codes: 0 nonexistent, 1 down, 2 left, 3 throwing, 4 right, 5 up.

    Prefer vertical motion (climbing) over horizontal so mid-climb monkeys
    don't switch to left/right and "walk" in the air.
    """
    dx, dy = _delta_xy(obj)
    # Climbing threshold: any clear vertical motion wins.
    if abs(dy) >= 1.0:
        return 1 if dy > 0 else 5  # down / up
    if abs(dx) >= 1.0:
        return 2 if dx < 0 else 4  # left / right
    if abs(dy) > abs(dx) and abs(dy) > 0:
        return 1 if dy > 0 else 5
    if abs(dx) > 0:
        return 2 if dx < 0 else 4
    return 1  # visible but idle: prefer hang/climb-down over lateral walk


def _jump_offset(counter: int) -> int:
    """Match jax_kangaroo._player_jump_controller offset_for."""
    if 0 < counter <= 8:
        return -1
    if 8 < counter < 16:
        return -8
    if 16 <= counter <= 24:
        return -8
    if 24 < counter <= 32:
        return -16
    if 32 < counter < 40:
        return -8
    return 0


def _jump_height(counter: int) -> int:
    """Match jax_kangaroo._player_height_controller while jumping."""
    if counter < 8:
        return 23
    if counter < 16:
        return 24
    if counter < 24:
        return 15
    if counter < 40:
        return 23
    return 24


def _platform_stand_y(
    env: JaxKangaroo,
    *,
    level: int,
    player_x: int,
    player_y: int,
    player_h: int,
) -> Optional[int]:
    """Standing Y on the closest platform under the player's feet, or None."""
    try:
        lc = env._get_level_constants(int(level))
    except Exception:
        return None
    positions = np.asarray(lc.platform_positions)
    sizes = np.asarray(lc.platform_sizes)
    ph = int(env.consts.PLAYER_HEIGHT)
    feet = player_y + player_h
    best_y: Optional[int] = None
    best_gap = 10_000
    for i in range(positions.shape[0]):
        px, py = int(positions[i, 0]), int(positions[i, 1])
        if px < 0:
            continue
        pw = int(sizes[i, 0])
        if (player_x + int(env.consts.PLAYER_WIDTH)) < px or player_x > (px + pw):
            continue
        if feet > py + 2:
            continue  # platform not below (allow tiny overlap)
        gap = py - feet
        if gap < best_gap:
            best_gap = gap
            best_y = py - ph
    return best_y


def _jump_counter_for_offset(offset: int, dy: float) -> int:
    """Pick a jump_counter whose offset matches, using dy for rising vs falling.

    Avoid counters 16–24 for the −8 hang: that band uses height 15 and the
    height controller then shifts y by +8, which looks like a ground snap.
    """
    if offset <= -14:
        return 28  # peak (-16), height 23
    if offset <= -5:
        if dy > 0.5:
            return 35  # falling hang (-8), height 23
        # Rising or apex: stay in 9–15 (height 24), not 16–24 (height 15).
        return 12
    if offset <= -1:
        return 6 if dy <= 0 else 4  # early (-1)
    return 2 if dy < 0 else 38


def _approx_player_airborne(
    player: Mapping[str, Any],
    base: PlayerState,
    *,
    climbing_flag: bool,
    env: JaxKangaroo,
    level: int,
) -> PlayerState:
    """Approximate jump/climb so mid-air Y isn't treated as standing on a platform."""
    x = int(player["x"])
    y = int(player["y"])
    dx, dy = _delta_xy(player)
    ori = _orientation_pm1(player.get("orientation"))
    h = int(player.get("h", base.height))

    p = base.replace(
        x=cast_like(x, base.x),
        y=cast_like(y, base.y),
        orientation=cast_like(ori, base.orientation),
        height=cast_like(h, base.height),
        vel_x=cast_like(int(np.clip(round(dx), -4, 4)), base.vel_x),
        is_climbing=cast_like(climbing_flag, base.is_climbing),
        is_crashing=cast_like(bool(player.get("crashed", False)), base.is_crashing),
    )

    if climbing_flag:
        # Mid-climb: keep climb bookkeeping so step doesn't snap to platforms.
        return p.replace(
            is_jumping=cast_like(False, base.is_jumping),
            climb_base_y=cast_like(y, base.climb_base_y),
            climb_counter=cast_like(max(1, abs(int(round(dy))) * 2), base.climb_counter),
            last_stood_on_platform_y=cast_like(y, base.last_stood_on_platform_y),
        )

    # OC has no "jumping" flag; dy is often 0 for several frames mid-arc.
    # Detect airborne via gap above the platform stand Y, not dy alone.
    stand_y = _platform_stand_y(env, level=level, player_x=x, player_y=y, player_h=h)
    gap_above = (stand_y - y) if stand_y is not None else 0
    airborne = gap_above >= 2 or abs(dy) >= 1.0
    if not airborne:
        return p.replace(
            is_jumping=cast_like(False, base.is_jumping),
            jump_base_y=cast_like(y, base.jump_base_y),
            landing_base_y=cast_like(y, base.landing_base_y),
            last_stood_on_platform_y=cast_like(y, base.last_stood_on_platform_y),
        )

    jump_base_y = int(stand_y) if stand_y is not None else (y - _jump_offset(12))
    offset = y - jump_base_y
    jump_counter = _jump_counter_for_offset(offset, dy)
    # Re-anchor base so y == jump_base_y + offset_for(counter) exactly.
    jump_base_y = y - _jump_offset(jump_counter)
    # Height must match the jump table or the next step's height Δ shifts y.
    jump_h = _jump_height(jump_counter)

    return p.replace(
        is_jumping=cast_like(True, base.is_jumping),
        jump_counter=cast_like(jump_counter, base.jump_counter),
        jump_base_y=cast_like(jump_base_y, base.jump_base_y),
        jump_orientation=cast_like(ori, base.jump_orientation),
        landing_base_y=cast_like(jump_base_y, base.landing_base_y),
        last_stood_on_platform_y=cast_like(jump_base_y, base.last_stood_on_platform_y),
        height=cast_like(jump_h, base.height),
    )


def oc_frame_to_kangaroo_state(
    env: JaxKangaroo,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> KangarooState:
    del frame_index
    if print_assumptions:
        print_disclaimers("kangaroo")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs

    # Force start layout: JAX reset uses current_level=1 (not 0).
    updates: dict = {"current_level": cast_like(1, state.current_level)}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        climbing = bool(player.get("climbing", False))
        # Also treat strong vertical Δ as climb when OC flag missing mid-ladder.
        dx, dy = _delta_xy(player)
        if not climbing and abs(dy) >= 2.0 and abs(dy) > abs(dx) + 0.5:
            # Likely jump rather than climb unless already flagged; keep jump path.
            climbing = False
        updates["player"] = _approx_player_airborne(
            player,
            state.player,
            climbing_flag=climbing,
            env=env,
            level=1,
        )

    level = state.level
    level_updates: dict = {}

    fruits = collect_category(objs, "Fruit")
    if fruits:
        pos = np.array(level.fruit_positions, dtype=np.dtype(np.asarray(level.fruit_positions).dtype)).copy()
        act = np.zeros(pos.shape[0], dtype=np.dtype(np.asarray(level.fruit_actives).dtype))
        for i, fr in enumerate(fruits[: pos.shape[0]]):
            pos[i, 0] = int(fr["x"])
            pos[i, 1] = int(fr["y"])
            act[i] = True
        level_updates["fruit_positions"] = cast_like(pos, level.fruit_positions)
        level_updates["fruit_actives"] = cast_like(act, level.fruit_actives)

    bell = find_object(objs, "Bell")
    if bell is not None and int(bell.get("w", 0)) > 0:
        level_updates["bell_position"] = cast_like(
            [int(bell["x"]), int(bell["y"])], level.bell_position
        )

    child = find_object(objs, "Child")
    if child is not None and int(child.get("w", 0)) > 0:
        level_updates["child_position"] = cast_like(
            [int(child["x"]), int(child["y"])], level.child_position
        )

    monkeys = collect_category(objs, "Monkey")
    if monkeys:
        mpos = np.array(level.monkey_positions, dtype=np.dtype(np.asarray(level.monkey_positions).dtype)).copy()
        mstates = np.zeros(mpos.shape[0], dtype=np.dtype(np.asarray(level.monkey_states).dtype))
        for i, m in enumerate(monkeys[: mpos.shape[0]]):
            mpos[i, 0] = int(m["x"])
            mpos[i, 1] = int(m["y"])
            mstates[i] = _monkey_state_from_delta(m)
        level_updates["monkey_positions"] = cast_like(mpos, level.monkey_positions)
        level_updates["monkey_states"] = cast_like(mstates, level.monkey_states)

    falling = find_object(objs, "FallingCoconut")
    if falling is not None and int(falling.get("w", 0)) > 0:
        level_updates["falling_coco_position"] = cast_like(
            [int(falling["x"]), int(falling["y"])], level.falling_coco_position
        )
        level_updates["falling_coco_dropping"] = cast_like(True, level.falling_coco_dropping)

    thrown = collect_category(objs, "ThrownCoconut")
    if thrown:
        cpos = np.array(level.coco_positions, dtype=np.dtype(np.asarray(level.coco_positions).dtype)).copy()
        cstates = np.zeros(cpos.shape[0], dtype=np.dtype(np.asarray(level.coco_states).dtype))
        for i, c in enumerate(thrown[: cpos.shape[0]]):
            cpos[i, 0] = int(c["x"])
            cpos[i, 1] = int(c["y"])
            cstates[i] = 2  # throwing / in flight
        level_updates["coco_positions"] = cast_like(cpos, level.coco_positions)
        level_updates["coco_states"] = cast_like(cstates, level.coco_states)

    time_obj = find_object(objs, "Time")
    if time_obj is not None and int(time_obj.get("value", -1)) >= 0:
        level_updates["timer"] = cast_like(int(time_obj["value"]), level.timer)

    # Q2=B: keep JAX platform/ladder constants (do not overlay).

    if level_updates:
        updates["level"] = level.replace(**level_updates)

    score_obj = find_object(objs, "Score")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        updates["score"] = cast_like(int(score_obj["value"]), state.score)
    lives_obj = find_object(objs, "Lives")
    if lives_obj is not None and int(lives_obj.get("value", -1)) >= 0:
        updates["lives"] = cast_like(int(lives_obj["value"]), state.lives)

    return state.replace(**updates)


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["player"] = (float(player["x"]), float(player["y"]))
    child = find_object(objs, "Child")
    if child is not None and int(child.get("w", 0)) > 0:
        entities["child"] = (float(child["x"]), float(child["y"]))
    bell = find_object(objs, "Bell")
    if bell is not None and int(bell.get("w", 0)) > 0:
        entities["bell"] = (float(bell["x"]), float(bell["y"]))
    for i, fr in enumerate(collect_category(objs, "Fruit")):
        entities[f"fruit_{i}"] = (float(fr["x"]), float(fr["y"]))
    for i, m in enumerate(collect_category(objs, "Monkey")):
        entities[f"monkey_{i}"] = (float(m["x"]), float(m["y"]))
    falling = find_object(objs, "FallingCoconut")
    if falling is not None and int(falling.get("w", 0)) > 0:
        entities["falling_coco"] = (float(falling["x"]), float(falling["y"]))

    score_obj = find_object(objs, "Score")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        scores["score"] = float(score_obj["value"])
    lives_obj = find_object(objs, "Lives")
    if lives_obj is not None and int(lives_obj.get("value", -1)) >= 0:
        scores["lives"] = float(lives_obj["value"])
    time_obj = find_object(objs, "Time")
    if time_obj is not None and int(time_obj.get("value", -1)) >= 0:
        scores["time"] = float(time_obj["value"])

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: KangarooState, env: JaxKangaroo) -> dict:
    del env
    entities = {
        "player": (float(state.player.x), float(state.player.y)),
        "child": (
            float(state.level.child_position[0]),
            float(state.level.child_position[1]),
        ),
        "bell": (
            float(state.level.bell_position[0]),
            float(state.level.bell_position[1]),
        ),
    }
    fruits = np.asarray(state.level.fruit_positions)
    act = np.asarray(state.level.fruit_actives)
    fi = 0
    for i in range(fruits.shape[0]):
        if int(act[i]) > 0:
            entities[f"fruit_{fi}"] = (float(fruits[i, 0]), float(fruits[i, 1]))
            fi += 1
    mpos = np.asarray(state.level.monkey_positions)
    mst = np.asarray(state.level.monkey_states)
    mi = 0
    for i in range(mpos.shape[0]):
        if int(mst[i]) != 0:
            entities[f"monkey_{mi}"] = (float(mpos[i, 0]), float(mpos[i, 1]))
            mi += 1
    if bool(state.level.falling_coco_dropping):
        fc = np.asarray(state.level.falling_coco_position)
        entities["falling_coco"] = (float(fc[0]), float(fc[1]))
    scores = {
        "score": float(state.score),
        "lives": float(state.lives),
        "time": float(state.level.timer),
    }
    return {"entities": entities, "scores": scores}
