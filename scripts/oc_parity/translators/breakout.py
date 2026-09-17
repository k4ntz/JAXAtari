"""Translate OCAtari Breakout snapshots into JAXAtari BreakoutState."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_breakout import BreakoutState, JaxBreakout

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers


def _blocks_from_oc(objs: Sequence[Mapping[str, Any]], consts) -> Optional[np.ndarray]:
    blocks_oc = collect_category(objs, "Block")
    if not blocks_oc:
        return None
    grid = np.zeros((int(consts.NUM_ROWS), int(consts.BLOCKS_PER_ROW)), dtype=np.int32)
    bx0 = int(consts.BLOCK_START_X)
    by0 = int(consts.BLOCK_START_Y)
    bw, bh = int(consts.BLOCK_SIZE[0]), int(consts.BLOCK_SIZE[1])
    for b in blocks_oc:
        x, y, w = float(b.get("x", 0)), float(b.get("y", 0)), float(b.get("w", 0))
        if w <= 0:
            continue
        # OC often merges horizontal runs (w multiple of cell width).
        n_cells = max(1, int(round(w / max(bw, 1))))
        col0 = int(round((x - bx0) / max(bw, 1)))
        row = int(round((y - by0) / max(bh, 1)))
        if row < 0 or row >= grid.shape[0]:
            continue
        for k in range(n_cells):
            col = col0 + k
            if 0 <= col < grid.shape[1]:
                grid[row, col] = 1
    return grid


def oc_frame_to_breakout_state(
    env: JaxBreakout,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> BreakoutState:
    del frame_index
    if print_assumptions:
        print_disclaimers("breakout")

    consts = env.consts
    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs
    updates: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        updates["player_x"] = cast_like(int(player["x"]), state.player_x)
        dx = float(player.get("dx", 0))
        if dx == 0.0:
            dx = float(player.get("x", 0)) - float(player.get("prev_x", player.get("x", 0)))
        updates["player_speed"] = cast_like(int(np.clip(dx, -3, 3)), state.player_speed)

    ball = find_object(objs, "Ball")
    if ball is not None and int(ball.get("w", 0)) > 0:
        updates["ball_x"] = cast_like(int(ball["x"]), state.ball_x)
        updates["ball_y"] = cast_like(int(ball["y"]), state.ball_y)
        updates["game_started"] = cast_like(1, state.game_started)
        dx = float(ball.get("dx", 0))
        dy = float(ball.get("dy", 0))
        if dx == 0.0 and dy == 0.0:
            dx = float(ball.get("x", 0)) - float(ball.get("prev_x", ball.get("x", 0)))
            dy = float(ball.get("y", 0)) - float(ball.get("prev_y", ball.get("y", 0)))
        if dx != 0.0 or dy != 0.0:
            updates["ball_vel_x"] = cast_like(int(np.sign(dx) or 1), state.ball_vel_x)
            updates["ball_vel_y"] = cast_like(int(np.sign(dy) or -1), state.ball_vel_y)

    grid = _blocks_from_oc(objs, consts)
    if grid is not None:
        updates["blocks"] = cast_like(grid, state.blocks)

    score_obj = find_object(objs, "PlayerScore")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        updates["score"] = cast_like(int(score_obj["value"]), state.score)
    lives_obj = find_object(objs, "Live") or find_object(objs, "Lives")
    if lives_obj is not None and int(lives_obj.get("value", -1)) >= 0:
        updates["lives"] = cast_like(int(lives_obj["value"]), state.lives)

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
    ball = find_object(objs, "Ball")
    if ball is not None and int(ball.get("w", 0)) > 0:
        entities["ball"] = (float(ball["x"]), float(ball["y"]))
    score_obj = find_object(objs, "PlayerScore")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        scores["score"] = float(score_obj["value"])
    lives_obj = find_object(objs, "Live") or find_object(objs, "Lives")
    if lives_obj is not None and int(lives_obj.get("value", -1)) >= 0:
        scores["lives"] = float(lives_obj["value"])
    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: BreakoutState, env: JaxBreakout) -> dict:
    entities = {"player": (float(state.player_x), float(env.consts.PLAYER_START_Y))}
    if int(state.game_started) != 0:
        entities["ball"] = (float(state.ball_x), float(state.ball_y))
    scores = {"score": float(state.score), "lives": float(state.lives)}
    return {"entities": entities, "scores": scores}
