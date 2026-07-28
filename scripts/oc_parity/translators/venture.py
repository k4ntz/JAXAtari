"""Translate OCAtari Venture object snapshots into JAXAtari GameState.

Infers main-map (level 0) vs room from OC object categories; world stays at 1.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_venture import GameState, JaxVenture

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers

_WORLD = 1
# Hall player: OC reset (67,177) vs JAX PLAYER_INITIAL (67,185).
_HALL_PLAYER_Y_SHIFT = 8.0


def _is_in_room(objs: Sequence[Mapping[str, Any]]) -> bool:
    """Room if Goblins (or large player sprite); hall otherwise."""
    if collect_category(objs, "Goblin"):
        return True
    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) >= 6:
        return True
    return False


def _infer_room_level(objs: Sequence[Mapping[str, Any]], chest_positions: np.ndarray) -> int:
    """Pick World-1 room 1–4 by nearest chest anchor (fallback: room 3)."""
    chest = find_object(objs, "Yellow_Collectable")
    # World-1 room chests live at global indices 1..4.
    anchors = np.asarray(chest_positions[1:5], dtype=np.float64)
    if chest is not None and int(chest.get("w", 0)) > 0:
        cx = float(chest["x"]) + float(chest.get("w", 0)) / 2.0
        cy = float(chest["y"]) + float(chest.get("h", 0)) / 2.0
        idx = int(np.argmin(np.sum((anchors - np.array([cx, cy])) ** 2, axis=1)))
        return idx + 1
    # No chest: use player position vs room chest anchors as a weak prior.
    player = find_object(objs, "Player")
    if player is not None:
        px = float(player["x"]) + float(player.get("w", 0)) / 2.0
        py = float(player["y"]) + float(player.get("h", 0)) / 2.0
        idx = int(np.argmin(np.sum((anchors - np.array([px, py])) ** 2, axis=1)))
        return idx + 1
    return 3


def _player_jax_xy(player: Mapping[str, Any], *, in_room: bool) -> Tuple[float, float]:
    x = float(player["x"])
    y = float(player["y"])
    w = float(player.get("w", 1))
    h = float(player.get("h", 2))
    if in_room:
        # Room sprites: JAX player.x/y are centers of the detailed sprite.
        return x + w / 2.0, y + h / 2.0
    # Hall: match OC top-left X (same as monsters); Y needs +8 vs ALE.
    return x, y + _HALL_PLAYER_Y_SHIFT


def _monster_slot_range(offsets: np.ndarray, level: int, world: int) -> Tuple[int, int]:
    """LEVEL_OFFSETS index = (world-1)*5 + level."""
    global_idx = (world - 1) * 5 + level
    start = int(offsets[global_idx]) if global_idx < len(offsets) else 0
    end = (
        int(offsets[global_idx + 1])
        if global_idx + 1 < len(offsets)
        else int(offsets[-1])
    )
    return start, end


def oc_frame_to_venture_state(
    env: JaxVenture,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    prev_player_xy: Optional[Tuple[float, float]] = None,
    print_assumptions: bool = False,
    **_ignored,
) -> GameState:
    del frame_index
    if print_assumptions:
        print_disclaimers("venture")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs

    in_room = _is_in_room(objs)
    if in_room:
        level = _infer_room_level(objs, np.asarray(env.consts.CHEST_POSITIONS))
    else:
        level = 0

    offsets = np.asarray(env.consts.LEVEL_OFFSETS)
    start, end = _monster_slot_range(offsets, level, _WORLD)

    updates: dict = {
        "current_level": cast_like(level, state.current_level),
        "world_level": cast_like(_WORLD, state.world_level),
        "last_level": cast_like(level, state.last_level),
    }

    player = find_object(objs, "Player")
    if player is not None:
        x, y = _player_jax_xy(player, in_room=in_room)
        dx, dy = 0.0, 0.0
        if prev_player_xy is not None:
            dx = float(player["x"]) - float(prev_player_xy[0])
            dy = float(player["y"]) - float(prev_player_xy[1])
        else:
            dx = float(player.get("dx", 0))
            dy = float(player.get("dy", 0))
        updates["player"] = state.player.replace(
            x=cast_like(x, state.player.x),
            y=cast_like(y, state.player.y),
            last_valid_x=cast_like(x, state.player.last_valid_x),
            last_valid_y=cast_like(y, state.player.last_valid_y),
            last_dx=cast_like(dx, state.player.last_dx),
            last_dy=cast_like(dy, state.player.last_dy),
        )

    if in_room:
        monsters: List[Mapping[str, Any]] = list(collect_category(objs, "Goblin"))
    else:
        monsters = list(collect_category(objs, "Hallmonsters"))

    if monsters and end > start:
        mx = np.array(state.monsters.x, dtype=np.dtype(np.asarray(state.monsters.x).dtype)).copy()
        my = np.array(state.monsters.y, dtype=np.dtype(np.asarray(state.monsters.y).dtype)).copy()
        mdx = np.array(state.monsters.dx, dtype=np.dtype(np.asarray(state.monsters.dx).dtype)).copy()
        mdy = np.array(state.monsters.dy, dtype=np.dtype(np.asarray(state.monsters.dy).dtype)).copy()
        active = np.array(
            state.monsters.active, dtype=np.dtype(np.asarray(state.monsters.active).dtype)
        ).copy()
        active[:] = False
        slots = list(range(start, end))
        for i, mon in enumerate(monsters[: len(slots)]):
            idx = slots[i]
            # Hall/room monster state matches OC top-left XY (verified on main map).
            mx[idx] = float(mon["x"])
            my[idx] = float(mon["y"])
            dx = float(mon.get("dx", 0))
            dy = float(mon.get("dy", 0))
            # OC often reports dx=dy=0; JAX moves as pos + dx * speed, so zero
            # velocity freezes monsters until rare MONSTER_CHANGE_DIR_PROB fires.
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                if "prev_x" in mon and "prev_y" in mon:
                    dx = float(mon["x"]) - float(mon["prev_x"])
                    dy = float(mon["y"]) - float(mon["prev_y"])
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                # Stable per-slot unit direction (mirrors reset cos/sin angles).
                angle = (idx * 2.399963229728653) % (2.0 * np.pi)  # golden-angle-ish
                dx = float(np.cos(angle))
                dy = float(np.sin(angle))
            mdx[idx] = dx
            mdy[idx] = dy
            active[idx] = True
        updates["monsters"] = state.monsters.replace(
            x=cast_like(mx, state.monsters.x),
            y=cast_like(my, state.monsters.y),
            dx=cast_like(mdx, state.monsters.dx),
            dy=cast_like(mdy, state.monsters.dy),
            active=cast_like(active, state.monsters.active),
        )

    shot = find_object(objs, "Shot")
    if shot is not None and int(shot.get("w", 0)) > 0:
        sx = float(shot["x"]) + float(shot.get("w", 0)) / 2.0
        sy = float(shot["y"]) + float(shot.get("h", 0)) / 2.0
        updates["projectile"] = state.projectile.replace(
            x=cast_like(sx, state.projectile.x),
            y=cast_like(sy, state.projectile.y),
            dx=cast_like(float(shot.get("dx", 0)), state.projectile.dx),
            dy=cast_like(float(shot.get("dy", 0)), state.projectile.dy),
            active=cast_like(True, state.projectile.active),
        )

    chest = find_object(objs, "Yellow_Collectable")
    if chest is not None and int(chest.get("w", 0)) > 0 and level >= 1:
        chests = np.array(
            state.chests_active, dtype=np.dtype(np.asarray(state.chests_active).dtype)
        ).copy()
        chest_idx = level - 1
        if 0 <= chest_idx < len(chests):
            chests[chest_idx] = True
            updates["chests_active"] = cast_like(chests, state.chests_active)

    score_obj = find_object(objs, "Score")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        updates["score"] = cast_like(int(score_obj["value"]), state.score)
    life_obj = find_object(objs, "Life")
    if life_obj is not None and int(life_obj.get("value", -1)) >= 0:
        updates["lives"] = cast_like(int(life_obj["value"]), state.lives)

    return state.replace(**updates)


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}
    in_room = _is_in_room(objs)

    player = find_object(objs, "Player")
    if player is not None:
        x, y = _player_jax_xy(player, in_room=in_room)
        entities["player"] = (float(x), float(y))
    monsters = (
        collect_category(objs, "Goblin")
        if in_room
        else collect_category(objs, "Hallmonsters")
    )
    for i, m in enumerate(monsters):
        entities[f"monster_{i}"] = (float(m["x"]), float(m["y"]))
    shot = find_object(objs, "Shot")
    if shot is not None and int(shot.get("w", 0)) > 0:
        entities["projectile"] = (
            float(shot["x"]) + float(shot.get("w", 0)) / 2.0,
            float(shot["y"]) + float(shot.get("h", 0)) / 2.0,
        )
    chest = find_object(objs, "Yellow_Collectable")
    if chest is not None and int(chest.get("w", 0)) > 0:
        entities["chest"] = (float(chest["x"]), float(chest["y"]))

    score_obj = find_object(objs, "Score")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        scores["score"] = float(score_obj["value"])
    life_obj = find_object(objs, "Life")
    if life_obj is not None and int(life_obj.get("value", -1)) >= 0:
        scores["lives"] = float(life_obj["value"])

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: GameState, env: JaxVenture) -> dict:
    del env
    entities = {
        "player": (float(state.player.x), float(state.player.y)),
    }
    active = np.asarray(state.monsters.active)
    xs = np.asarray(state.monsters.x)
    ys = np.asarray(state.monsters.y)
    mi = 0
    for i in range(len(active)):
        if bool(active[i]):
            entities[f"monster_{mi}"] = (float(xs[i]), float(ys[i]))
            mi += 1
    if bool(state.projectile.active):
        entities["projectile"] = (
            float(state.projectile.x),
            float(state.projectile.y),
        )
    scores = {"score": float(state.score), "lives": float(state.lives)}
    return {"entities": entities, "scores": scores}
