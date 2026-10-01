"""Translate OCAtari Asteroids snapshots into JAXAtari AsteroidsState."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_asteroids import AsteroidsState, JaxAsteroids

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers

# OC rgb → JAX color index (see asteroids_colors / consts).
_RGB_TO_COLOR = {
    (180, 122, 48): 0,  # brown
    (104, 72, 198): 1,  # purple
    (136, 146, 62): 2,  # yellow-ish
}


def screen_to_subpixel(screen_xy: float) -> int:
    """Inverse of ``to_screen_pos``: ``sign(p)*(|p|//256)*2``."""
    s = int(round(screen_xy))
    return int(np.sign(s) * (abs(s) // 2) * 256)


def _size_from_wh(w: float, h: float, consts) -> int:
    area = float(w) * float(h)
    # LARGE ~16x28, MEDIUM ~8x15, SMALL ~4x8
    if area >= 16 * 20:
        return int(getattr(consts, "LARGE_1", 1))
    if area >= 8 * 10:
        return int(getattr(consts, "MEDIUM", 3))
    if area > 0:
        return int(getattr(consts, "SMALL", 4))
    return 0


def _color_from_rgb(rgb: Any) -> int:
    if rgb is None:
        return 0
    try:
        key = tuple(int(c) for c in rgb[:3])
    except Exception:
        return 0
    return int(_RGB_TO_COLOR.get(key, 0))


def oc_frame_to_asteroids_state(
    env: JaxAsteroids,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> AsteroidsState:
    del frame_index
    if print_assumptions:
        print_disclaimers("asteroids")

    consts = env.consts
    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs
    updates: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        updates["player_x"] = cast_like(screen_to_subpixel(player["x"]), state.player_x)
        updates["player_y"] = cast_like(screen_to_subpixel(player["y"]), state.player_y)
        updates["respawn_timer"] = cast_like(0, state.respawn_timer)
        ori = player.get("orientation")
        if ori is not None:
            try:
                rot = int(getattr(ori, "value", ori)) % 16
                updates["player_rotation"] = cast_like(rot, state.player_rotation)
            except (TypeError, ValueError):
                pass
    else:
        # Death / blink: keep player off compare by using respawn delay.
        updates["respawn_timer"] = cast_like(
            int(getattr(consts, "RESPAWN_DELAY", 136)), state.respawn_timer
        )

    rocks = [
        a
        for a in collect_category(objs, "Asteroid")
        if int(a.get("w", 0)) > 0 and int(a.get("h", 0)) > 0
    ]
    rocks = sorted(rocks, key=lambda o: (float(o.get("y", 0)), float(o.get("x", 0))))
    ast = np.array(state.asteroid_states, copy=True)
    ast[:, 3] = 0  # size=0 inactive
    max_n = min(len(rocks), ast.shape[0])
    for i in range(max_n):
        a = rocks[i]
        # Asteroid state uses screen-ish coords in many code paths; keep OC xy.
        ast[i, 0] = int(a["x"])
        ast[i, 1] = int(a["y"])
        ast[i, 2] = 0
        ast[i, 3] = _size_from_wh(a.get("w", 0), a.get("h", 0), consts)
        ast[i, 4] = _color_from_rgb(a.get("rgb"))
    updates["asteroid_states"] = cast_like(ast, state.asteroid_states)

    missiles = [
        m
        for m in collect_category(objs, "PlayerMissile")
        if int(m.get("w", 0)) > 0
    ][:2]
    miss = np.array(state.missile_states, copy=True)
    miss[:, 5] = 0
    life = int(getattr(consts, "MISSILE_LIFESPAN", 40))
    for i, m in enumerate(missiles):
        miss[i, 0] = int(m["x"])
        miss[i, 1] = int(m["y"])
        miss[i, 5] = life
    updates["missile_states"] = cast_like(miss, state.missile_states)

    score_obj = find_object(objs, "PlayerScore")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        updates["score"] = cast_like(int(score_obj["value"]), state.score)
    lives_obj = find_object(objs, "Lives")
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
    rocks = sorted(
        [a for a in collect_category(objs, "Asteroid") if int(a.get("w", 0)) > 0],
        key=lambda o: (float(o.get("y", 0)), float(o.get("x", 0))),
    )
    for i, a in enumerate(rocks[:17]):
        entities[f"asteroid_{i}"] = (float(a["x"]), float(a["y"]))
    for i, m in enumerate(
        [m for m in collect_category(objs, "PlayerMissile") if int(m.get("w", 0)) > 0][:2]
    ):
        entities[f"missile_{i}"] = (float(m["x"]), float(m["y"]))
    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: AsteroidsState, env: JaxAsteroids) -> dict:
    entities: dict = {}
    if int(state.respawn_timer) <= 0:
        sx = float(env.to_screen_pos(state.player_x))
        sy = float(env.to_screen_pos(state.player_y))
        entities["player"] = (sx, sy)
    rocks = np.asarray(state.asteroid_states)
    ai = 0
    for i in range(rocks.shape[0]):
        if int(rocks[i, 3]) != 0:
            entities[f"asteroid_{ai}"] = (float(rocks[i, 0]), float(rocks[i, 1]))
            ai += 1
    miss = np.asarray(state.missile_states)
    mi = 0
    for i in range(miss.shape[0]):
        if int(miss[i, 5]) > 0:
            entities[f"missile_{mi}"] = (float(miss[i, 0]), float(miss[i, 1]))
            mi += 1
    scores = {"score": float(state.score), "lives": float(state.lives)}
    return {"entities": entities, "scores": scores}
