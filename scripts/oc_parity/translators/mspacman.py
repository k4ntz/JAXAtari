"""Translate OCAtari MsPacman object snapshots into JAXAtari PacmanState."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_mspacman import JaxPacman, PacmanState

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers

# Action enum used in PlayerState / GhostsState: 0 NOOP, 2 UP, 3 RIGHT, 4 LEFT, 5 DOWN
_ACT_UP, _ACT_RIGHT, _ACT_LEFT, _ACT_DOWN = 2, 3, 4, 5

# Calibrated at reset: OC Player (75,99) ↔ JAX (75,102). Same sprite height (10);
# JAX lattice snaps Y to ≡6 (mod 12). X matches 1:1.
_OC_TO_JAX_Y = 3


def _oc_xy_to_jax(x: float, y: float) -> Tuple[int, int]:
    return int(round(x)), int(round(y + _OC_TO_JAX_Y))


def _action_from_delta(dx: float, dy: float) -> int:
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return 0
    if abs(dx) >= abs(dy):
        return _ACT_RIGHT if dx > 0 else _ACT_LEFT
    return _ACT_DOWN if dy > 0 else _ACT_UP


def _pill_tile(obj: Mapping[str, Any], grid_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Return JAX ``pellets[tx, ty]`` indices.

    OC ``grid_ij`` is ``(row=y_tile, col=x_tile)``; JAX stores ``[x_tile, y_tile]``.
    """
    gij = obj.get("grid_ij")
    if gij is not None:
        try:
            row, col = int(gij[0]), int(gij[1])
            tx, ty = col, row
            if 0 <= tx < grid_shape[0] and 0 <= ty < grid_shape[1]:
                return tx, ty
        except Exception:
            pass
    # Inverse of eat_pellet, after OC→JAX Y shift.
    jx, jy = _oc_xy_to_jax(float(obj.get("x", 0)), float(obj.get("y", 0)))
    tx = int((jx - 2) // 8)
    ty = int((jy + 4) // 12)
    if 0 <= tx < grid_shape[0] and 0 <= ty < grid_shape[1]:
        return tx, ty
    return None


def _ghost_type_from_rgb(rgb: Any, fallback: int) -> int:
    """Best-effort Blinky/Pinky/Inky/Sue from rgb; else fallback index."""
    try:
        r, g, b = [int(c) for c in rgb[:3]]
    except Exception:
        return fallback
    if r > 180 and g < 100 and b < 100:
        return 0  # Blinky red
    if r > 180 and g > 120 and b > 150:
        return 1  # Pinky
    if b > 180 and r < 120:
        return 2  # Inky
    if r > 180 and g > 140 and b < 80:
        return 3  # Sue / orange
    return fallback


def oc_frame_to_mspacman_state(
    env: JaxPacman,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    prev_player_xy: Optional[Tuple[float, float]] = None,
    print_assumptions: bool = False,
    **_ignored,
) -> PacmanState:
    del frame_index
    if print_assumptions:
        print_disclaimers("mspacman")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs

    level = state.level
    player = state.player
    ghosts = state.ghosts

    pills = collect_category(objs, "Pill")
    if pills:
        grid = np.zeros(np.asarray(level.pellets).shape, dtype=bool)
        for pill in pills:
            tile = _pill_tile(pill, grid.shape)
            if tile is not None:
                grid[tile] = True
        level = level._replace(
            pellets=cast_like(grid, level.pellets),
            collected_pellets=cast_like(
                int(np.asarray(state.level.pellets).sum()) - int(grid.sum()),
                level.collected_pellets,
            ),
        )

    power = collect_category(objs, "PowerPill")
    if power:
        # Match by nearest JAX power-pellet screen position (tile*4+[4,6]).
        anchors = np.asarray(env.consts.POWER_PELLET_TILES) * 4 + np.array([4, 6])
        pp = np.zeros(4, dtype=bool)
        for p in power[:4]:
            jx, jy = _oc_xy_to_jax(float(p["x"]), float(p["y"]))
            # Ignore HUD fruit-like false positives far below the maze.
            if jy > 170:
                continue
            idx = int(np.argmin(np.sum((anchors - np.array([jx, jy])) ** 2, axis=1)))
            pp[idx] = True
        if pp.any():
            level = level._replace(power_pellets=cast_like(pp, level.power_pellets))

    poc_player = find_object(objs, "Player")
    if poc_player is not None and int(poc_player.get("w", 0)) > 0:
        ox = float(poc_player["x"])
        oy = float(poc_player["y"])
        jx, jy = _oc_xy_to_jax(ox, oy)
        if prev_player_xy is not None:
            dx = ox - float(prev_player_xy[0])
            dy = oy - float(prev_player_xy[1])
        else:
            dx = float(poc_player.get("dx", 0))
            dy = float(poc_player.get("dy", 0))
        act = _action_from_delta(dx, dy)
        # last_horiz_dir is an act_to_dir index: UP=0, RIGHT=1, LEFT=2, DOWN=3
        last_h = 2 if dx < 0 else (1 if dx > 0 else int(player.last_horiz_dir))
        player = player._replace(
            position=cast_like([jx, jy], player.position),
            action=cast_like(act, player.action),
            last_horiz_dir=cast_like(last_h, player.last_horiz_dir),
        )

    oc_ghosts = collect_category(objs, "Ghost")
    if oc_ghosts:
        positions = np.array(
            ghosts.positions, dtype=np.dtype(np.asarray(ghosts.positions).dtype)
        ).copy()
        types = np.array(
            ghosts.types, dtype=np.dtype(np.asarray(ghosts.types).dtype)
        ).copy()
        actions = np.array(
            ghosts.actions, dtype=np.dtype(np.asarray(ghosts.actions).dtype)
        ).copy()
        n = min(4, len(oc_ghosts))
        for i, g in enumerate(oc_ghosts[:n]):
            jx, jy = _oc_xy_to_jax(float(g["x"]), float(g["y"]))
            positions[i, 0] = jx
            positions[i, 1] = jy
            types[i] = _ghost_type_from_rgb(g.get("rgb"), i)
            actions[i] = _action_from_delta(float(g.get("dx", 0)), float(g.get("dy", 0)))
        order = list(range(n))
        uniq = {int(types[i]) for i in order}
        if len(uniq) == n:
            type_order = sorted(order, key=lambda i: int(types[i]))
            positions[:n] = positions[type_order]
            actions[:n] = actions[type_order]
            types[:n] = np.arange(n, dtype=types.dtype)
        ghosts = ghosts._replace(
            positions=cast_like(positions, ghosts.positions),
            types=cast_like(types, ghosts.types),
            actions=cast_like(actions, ghosts.actions),
        )

    updates: dict = {
        "level": level,
        "player": player,
        "ghosts": ghosts,
    }
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

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        jx, jy = _oc_xy_to_jax(float(player["x"]), float(player["y"]))
        entities["player"] = (float(jx), float(jy))
    for i, g in enumerate(collect_category(objs, "Ghost")[:4]):
        jx, jy = _oc_xy_to_jax(float(g["x"]), float(g["y"]))
        entities[f"ghost_{i}"] = (float(jx), float(jy))
    for i, p in enumerate(collect_category(objs, "PowerPill")[:4]):
        jx, jy = _oc_xy_to_jax(float(p["x"]), float(p["y"]))
        if jy <= 170:
            entities[f"powerpill_{i}"] = (float(jx), float(jy))

    score_obj = find_object(objs, "Score")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        scores["score"] = float(score_obj["value"])
    life_obj = find_object(objs, "Life")
    if life_obj is not None and int(life_obj.get("value", -1)) >= 0:
        scores["lives"] = float(life_obj["value"])
    pills = collect_category(objs, "Pill")
    scores["pellet_count"] = float(len(pills))

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: PacmanState, env: JaxPacman) -> dict:
    del env
    pos = np.asarray(state.player.position)
    entities = {"player": (float(pos[0]), float(pos[1]))}
    gpos = np.asarray(state.ghosts.positions)
    for i in range(gpos.shape[0]):
        entities[f"ghost_{i}"] = (float(gpos[i, 0]), float(gpos[i, 1]))
    scores = {
        "score": float(state.score),
        "lives": float(state.lives),
        "pellet_count": float(np.asarray(state.level.pellets).sum()),
    }
    return {"entities": entities, "scores": scores}
