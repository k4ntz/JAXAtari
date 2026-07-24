"""Translate OCAtari BeamRider object snapshots into JAXAtari BeamriderState.

OC coverage is thin (Player + Saucer only). Soft survival is mostly player-lane.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_beamrider import (
    BLUE_LINE_INIT_TABLE,
    BLUE_LINE_LOOP_TABLE,
    BeamriderState,
    JaxBeamrider,
)

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers

# OC mid-game Saucers are typically 2×2 at the horizon; treat that as live.
_SAUCER_MIN_W = 2
_SAUCER_Y_MAX = 150  # above playfield threshold (player ~y=167)
# Past init so shooting is unlocked (can_shoot = steps > 2000 | ufo_killed).
_POST_INTRO_STEPS = 2001


def _snap_lane(x: float, lanes: Sequence[float]) -> float:
    return float(min(lanes, key=lambda lx: abs(float(lx) - x)))


def _skip_intro_level(level, consts) -> Any:
    """Jump past the blue-line intro so enemies/player render in normal gameplay.

    Reset always starts at ``blue_line_counter=0``. While
    ``counter < len(BLUE_LINE_INIT_TABLE)`` (240 frames), observation/render force
    UFOs offscreen and only advance the beam animation — which is what mid-game
    overlay transfers were showing.
    """
    init_len = len(BLUE_LINE_INIT_TABLE)
    return level.replace(
        blue_line_counter=cast_like(init_len, level.blue_line_counter),
        line_positions=cast_like(BLUE_LINE_LOOP_TABLE[0], level.line_positions),
    )


def oc_frame_to_beamrider_state(
    env: JaxBeamrider,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> BeamriderState:
    if print_assumptions:
        print_disclaimers("beamrider")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs

    level = _skip_intro_level(state.level, env.consts)
    lanes = list(env.consts.BOTTOM_OF_LANES)

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        lane_x = _snap_lane(float(player["x"]), lanes)
        level = level.replace(player_pos=cast_like(lane_x, level.player_pos))

    saucers = [
        s
        for s in collect_category(objs, "Saucer")
        if int(s.get("w", 0)) >= _SAUCER_MIN_W and float(s.get("y", 999)) <= _SAUCER_Y_MAX
    ]
    if saucers:
        ufo = np.array(
            level.white_ufo_pos, dtype=np.dtype(np.asarray(level.white_ufo_pos).dtype)
        ).copy()
        # white_ufo_pos shape (2, 3) → [x row, y row] × up to 3 UFOs
        n = min(len(saucers), ufo.shape[1])
        for i in range(n):
            ufo[0, i] = float(saucers[i]["x"])
            ufo[1, i] = float(saucers[i]["y"])
        # Clear unused slots offscreen so reset leftovers don't linger.
        off = np.asarray(env.consts.ENEMY_OFFSCREEN_POS, dtype=float).reshape(-1)
        off_x, off_y = float(off[0]), float(off[-1] if off.size > 1 else off[0])
        for i in range(n, ufo.shape[1]):
            ufo[0, i] = off_x
            ufo[1, i] = off_y
        level = level.replace(white_ufo_pos=cast_like(ufo, level.white_ufo_pos))

    steps = max(int(frame_index), _POST_INTRO_STEPS)
    # Q4=A: score / sector / lives RESET
    return state.replace(
        level=level,
        steps=cast_like(steps, state.steps),
    )


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["player"] = (float(player["x"]), float(player["y"]))

    saucers = [
        s
        for s in collect_category(objs, "Saucer")
        if int(s.get("w", 0)) >= _SAUCER_MIN_W and float(s.get("y", 999)) <= _SAUCER_Y_MAX
    ]
    for i, s in enumerate(saucers):
        entities[f"saucer_{i}"] = (float(s["x"]), float(s["y"]))

    return {"entities": entities, "scores": {}}


def extract_jax_compare_entities(state: BeamriderState, env: JaxBeamrider) -> dict:
    entities = {
        "player": (
            float(_get_screen_x(state.level.player_pos, env)),
            float(env.consts.PLAYER_POS_Y),
        ),
    }
    ufo = np.asarray(state.level.white_ufo_pos)
    off = np.asarray(env.consts.ENEMY_OFFSCREEN_POS, dtype=float).reshape(-1)
    off_y = float(off[-1] if off.size > 1 else off[0])
    for i in range(ufo.shape[1]):
        x, y = float(ufo[0, i]), float(ufo[1, i])
        if y > 1 and y < _SAUCER_Y_MAX and x > 1 and abs(y - off_y) > 1:
            entities[f"saucer_{i}"] = (x, y)
    return {"entities": entities, "scores": {}}


def _get_screen_x(player_pos, env) -> float:
    try:
        from jaxatari.games.jax_beamrider import _get_player_screen_x

        return float(_get_player_screen_x(jnp.asarray(player_pos)))
    except Exception:
        return float(player_pos)
