"""Translate OCAtari MontezumaRevenge snapshots into MontezumaRevengeState.

OC screen y = JAX y + 47 (renderer room_y offset).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_montezumarevenge import JaxMontezumaRevenge
from jaxatari.games.montezuma_revenge.core import MontezumaRevengeState

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers

_OC_Y_OFFSET = 47  # OC_y = JAX_y + 47


def _jax_y(oc_y: float) -> int:
    return int(oc_y) - _OC_Y_OFFSET


def oc_frame_to_montezumarevenge_state(
    env: JaxMontezumaRevenge,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    prev_player_xy: Optional[Tuple[float, float]] = None,
    print_assumptions: bool = False,
    **_ignored,
) -> MontezumaRevengeState:
    del frame_index
    if print_assumptions:
        print_disclaimers("montezumarevenge")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs

    # Q1=A: force start room (INITIAL_ROOM_ID). Reset already loads it.
    room_id = int(env.consts.INITIAL_ROOM_ID)
    updates: dict = {
        "room_id": cast_like(room_id, state.room_id),
    }

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        px = int(player["x"])
        py = _jax_y(float(player["y"]))
        if prev_player_xy is not None:
            vx = int(round(float(player["x"]) - float(prev_player_xy[0])))
            # prev xy is OC space; convert Δy directly.
            vy = int(round(float(player["y"]) - float(prev_player_xy[1])))
        else:
            vx = int(player.get("dx", 0))
            vy = int(player.get("dy", 0))
        updates["player_x"] = cast_like(px, state.player_x)
        updates["player_y"] = cast_like(py, state.player_y)
        updates["player_vx"] = cast_like(vx, state.player_vx)
        updates["player_vy"] = cast_like(vy, state.player_vy)
        if vx != 0:
            updates["player_dir"] = cast_like(1 if vx > 0 else -1, state.player_dir)

        # Climbing heuristic: overlap any Ladder in x and y.
        climbing = False
        for lad in collect_category(objs, "Ladder"):
            lx, ly, lw, lh = (
                int(lad["x"]),
                _jax_y(float(lad["y"])),
                int(lad["w"]),
                int(lad["h"]),
            )
            if lx - 4 <= px <= lx + lw + 4 and ly - 4 <= py <= ly + lh + 4:
                climbing = True
                break
        # Reset stores is_climbing as int32 (0/1), not bool.
        updates["is_climbing"] = cast_like(int(climbing), state.is_climbing)

    # Q2=A: keep JAX room geometry; overlay Skull / Key only.
    skulls = collect_category(objs, "Skull")
    if skulls:
        ex = np.array(state.enemies_x, dtype=np.dtype(np.asarray(state.enemies_x).dtype)).copy()
        ey = np.array(state.enemies_y, dtype=np.dtype(np.asarray(state.enemies_y).dtype)).copy()
        ea = np.array(state.enemies_active, dtype=np.dtype(np.asarray(state.enemies_active).dtype)).copy()
        ed = np.array(state.enemies_direction, dtype=np.dtype(np.asarray(state.enemies_direction).dtype)).copy()
        for i, sk in enumerate(skulls[: ex.shape[0]]):
            ex[i] = int(sk["x"])
            ey[i] = _jax_y(float(sk["y"]))
            ea[i] = 1
            dx = float(sk.get("dx", 0))
            if dx != 0:
                ed[i] = 1 if dx > 0 else -1
        updates["enemies_x"] = cast_like(ex, state.enemies_x)
        updates["enemies_y"] = cast_like(ey, state.enemies_y)
        updates["enemies_active"] = cast_like(ea, state.enemies_active)
        updates["enemies_direction"] = cast_like(ed, state.enemies_direction)

    key = find_object(objs, "Key")
    if key is not None and int(key.get("w", 0)) > 0:
        ix = np.array(state.items_x, dtype=np.dtype(np.asarray(state.items_x).dtype)).copy()
        iy = np.array(state.items_y, dtype=np.dtype(np.asarray(state.items_y).dtype)).copy()
        ia = np.array(state.items_active, dtype=np.dtype(np.asarray(state.items_active).dtype)).copy()
        it = np.array(state.items_type, dtype=np.dtype(np.asarray(state.items_type).dtype)).copy()
        ix[0] = int(key["x"])
        iy[0] = _jax_y(float(key["y"]))
        ia[0] = 1
        it[0] = 0  # key
        updates["items_x"] = cast_like(ix, state.items_x)
        updates["items_y"] = cast_like(iy, state.items_y)
        updates["items_active"] = cast_like(ia, state.items_active)
        updates["items_type"] = cast_like(it, state.items_type)

    life = find_object(objs, "Life")
    if life is not None and int(life.get("value", -1)) >= 0:
        updates["lives"] = cast_like(int(life["value"]), state.lives)
    score = find_object(objs, "Score")
    if score is not None and int(score.get("value", -1)) >= 0:
        updates["score"] = cast_like(int(score["value"]), state.score)

    return state.replace(**updates)


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["player"] = (float(player["x"]), float(_jax_y(float(player["y"]))))
    for i, sk in enumerate(collect_category(objs, "Skull")):
        entities[f"skull_{i}"] = (float(sk["x"]), float(_jax_y(float(sk["y"]))))
    key = find_object(objs, "Key")
    if key is not None and int(key.get("w", 0)) > 0:
        entities["key"] = (float(key["x"]), float(_jax_y(float(key["y"]))))

    life = find_object(objs, "Life")
    if life is not None and int(life.get("value", -1)) >= 0:
        scores["lives"] = float(life["value"])
    score = find_object(objs, "Score")
    if score is not None and int(score.get("value", -1)) >= 0:
        scores["score"] = float(score["value"])

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(
    state: MontezumaRevengeState, env: JaxMontezumaRevenge
) -> dict:
    del env
    entities = {
        "player": (float(state.player_x), float(state.player_y)),
    }
    ea = np.asarray(state.enemies_active)
    ex = np.asarray(state.enemies_x)
    ey = np.asarray(state.enemies_y)
    si = 0
    for i in range(len(ea)):
        if bool(ea[i]):
            entities[f"skull_{si}"] = (float(ex[i]), float(ey[i]))
            si += 1
    ia = np.asarray(state.items_active)
    ix = np.asarray(state.items_x)
    iy = np.asarray(state.items_y)
    it = np.asarray(state.items_type)
    for i in range(len(ia)):
        if bool(ia[i]) and int(it[i]) == 0:
            entities["key"] = (float(ix[i]), float(iy[i]))
            break
    scores = {"lives": float(state.lives), "score": float(state.score)}
    return {"entities": entities, "scores": scores}
