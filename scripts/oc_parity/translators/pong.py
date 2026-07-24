"""Translate OCAtari Pong object snapshots into JAXAtari PongState."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_pong import JaxPong, PongConstants, PongState

from .base import find_object, objects_as_dicts


def _clip_speed(raw: float, max_speed: float) -> float:
    return float(np.clip(raw, -max_speed, max_speed))


def _estimate_paddle_speed(
    obj: Optional[Mapping[str, Any]],
    *,
    prev_y: Optional[float],
    max_speed: float,
) -> float:
    """Map OC paddle motion onto JAX analog speed via clipped Δy.

    Prefer lookback Δy when available; else nonzero object.dy; else 0.
    Large every-other-frame OC jumps are clipped to ±PADDLE_MAX_SPEED.
    """
    if obj is None:
        return 0.0

    y = float(obj.get("y", 0))
    dy = float(obj.get("dy", 0))

    if prev_y is not None:
        raw = y - float(prev_y)
        if raw != 0.0:
            return _clip_speed(raw, max_speed)

    if dy != 0.0:
        return _clip_speed(dy, max_speed)

    # Single-frame fallback: dy may be 0 on skip frames; use y - prev_y from OC.
    prev_obj_y = obj.get("prev_y", y)
    raw = y - float(prev_obj_y)
    return _clip_speed(raw, max_speed)


def oc_frame_to_pong_state(
    env: JaxPong,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    prev_player_y: Optional[float] = None,
    prev_enemy_y: Optional[float] = None,
) -> PongState:
    """Reset JAX Pong, then overlay fields derived from an OC object snapshot.

    Unmapped fields keep reset defaults. ``step_counter`` is forced to ≥60 when
    a live Ball is present so the post-goal freeze does not snap the ball away.
    """
    consts: PongConstants = env.consts
    objs = objects_as_dicts(objects)

    _, state = env.reset(jax.random.PRNGKey(int(seed)))

    player = find_object(objs, "Player")
    enemy = find_object(objs, "Enemy")
    ball = find_object(objs, "Ball")
    player_score_obj = find_object(objs, "PlayerScore")
    enemy_score_obj = find_object(objs, "EnemyScore")

    updates: dict = {}

    if player is not None:
        updates["player_y"] = jnp.array(float(player["y"]), dtype=jnp.float32)
        updates["player_speed"] = jnp.array(
            _estimate_paddle_speed(
                player, prev_y=prev_player_y, max_speed=float(consts.PADDLE_MAX_SPEED)
            ),
            dtype=jnp.float32,
        )

    if enemy is not None:
        updates["enemy_y"] = jnp.array(int(enemy["y"]), dtype=jnp.int32)
        updates["enemy_speed"] = jnp.array(
            int(
                round(
                    _estimate_paddle_speed(
                        enemy,
                        prev_y=prev_enemy_y,
                        max_speed=float(consts.PADDLE_MAX_SPEED),
                    )
                )
            ),
            dtype=jnp.int32,
        )

    ball_live = ball is not None and int(ball.get("w", 0)) > 0
    if ball_live:
        updates["ball_x"] = jnp.array(int(ball["x"]), dtype=jnp.int32)
        updates["ball_y"] = jnp.array(int(ball["y"]), dtype=jnp.int32)
        # Ball.dx/dy match integer velocities well when nonzero.
        bdx = int(ball.get("dx", 0))
        bdy = int(ball.get("dy", 0))
        if bdx == 0 and bdy == 0:
            # Fall back to prev delta if OC reported a skip-like zero.
            bdx = int(ball["x"]) - int(ball.get("prev_x", ball["x"]))
            bdy = int(ball["y"]) - int(ball.get("prev_y", ball["y"]))
        if bdx != 0 or bdy != 0:
            updates["ball_vel_x"] = jnp.array(bdx, dtype=jnp.int32)
            updates["ball_vel_y"] = jnp.array(bdy, dtype=jnp.int32)
        updates["step_counter"] = jnp.array(
            max(60, int(frame_index)), dtype=jnp.int32
        )

    if player_score_obj is not None and int(player_score_obj.get("value", -1)) >= 0:
        updates["player_score"] = jnp.array(
            int(player_score_obj["value"]), dtype=jnp.int32
        )
    if enemy_score_obj is not None and int(enemy_score_obj.get("value", -1)) >= 0:
        updates["enemy_score"] = jnp.array(
            int(enemy_score_obj["value"]), dtype=jnp.int32
        )

    if updates:
        state = state.replace(**updates)
    return state


def extract_oc_compare_entities(objects: Sequence[Any]) -> dict:
    """OC snapshot → comparable entity positions + scores for lockstep metrics."""
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["player"] = (float(player["x"]), float(player["y"]))

    enemy = find_object(objs, "Enemy")
    if enemy is not None and int(enemy.get("w", 0)) > 0:
        entities["enemy"] = (float(enemy["x"]), float(enemy["y"]))

    ball = find_object(objs, "Ball")
    if ball is not None and int(ball.get("w", 0)) > 0:
        entities["ball"] = (float(ball["x"]), float(ball["y"]))

    ps = find_object(objs, "PlayerScore")
    if ps is not None and int(ps.get("value", -1)) >= 0:
        scores["player"] = float(ps["value"])
    es = find_object(objs, "EnemyScore")
    if es is not None and int(es.get("value", -1)) >= 0:
        scores["enemy"] = float(es["value"])

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: PongState, env: JaxPong) -> dict:
    """JAX PongState → comparable entity positions + scores."""
    consts = env.consts
    entities = {
        "player": (float(consts.PLAYER_X), float(state.player_y)),
        "enemy": (float(consts.ENEMY_X), float(state.enemy_y)),
        "ball": (float(state.ball_x), float(state.ball_y)),
    }
    scores = {
        "player": float(state.player_score),
        "enemy": float(state.enemy_score),
    }
    return {"entities": entities, "scores": scores}


def trajectory_frame_to_pong_state(
    env: JaxPong,
    traj: Mapping[str, Any],
    t: int,
    *,
    seed: Optional[int] = None,
) -> PongState:
    """Translate trajectory frame ``t``, using ``t-1`` Y lookback for speeds."""
    objects = traj["objects"][t]
    prev_player_y = None
    prev_enemy_y = None
    if t > 0:
        prev_objs = objects_as_dicts(traj["objects"][t - 1])
        prev_player = find_object(prev_objs, "Player")
        prev_enemy = find_object(prev_objs, "Enemy")
        if prev_player is not None:
            prev_player_y = float(prev_player["y"])
        if prev_enemy is not None:
            prev_enemy_y = float(prev_enemy["y"])

    if seed is None:
        seed = int(traj.get("meta", {}).get("seed", 0))

    return oc_frame_to_pong_state(
        env,
        objects,
        seed=int(seed),
        frame_index=int(t),
        prev_player_y=prev_player_y,
        prev_enemy_y=prev_enemy_y,
    )
