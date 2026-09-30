"""Translate OCAtari Tennis object snapshots into JAXAtari TennisState."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_tennis import TennisJaxEnv, TennisState

from .base import cast_like, find_object, objects_as_dicts
from .registry import print_disclaimers

# JAX renders shadow at (ball_x, ball_y) and the ball sprite at (ball_x, ball_y - ball_z).
_MAX_BALL_Z = 40
# Hit path seeds z_fp≈140 / vz≈24; mid-rally we approximate from OC Δ.
_DEFAULT_HIT_VZ = 24.0


def _delta(obj: Mapping[str, Any], axis: str) -> float:
    """Prefer object.d*; fall back to x/y − prev_*."""
    dkey = f"d{axis}"
    d = float(obj.get(dkey, 0))
    if d != 0.0:
        return d
    cur = float(obj.get(axis, 0))
    prev = float(obj.get(f"prev_{axis}", cur))
    return cur - prev


def _infer_ball_z(ball: Mapping[str, Any], shadow: Mapping[str, Any]) -> float:
    """Screen y grows downward → height ≈ shadow.y − ball.y."""
    z = float(shadow.get("y", 0)) - float(ball.get("y", 0))
    return float(np.clip(z, 0.0, _MAX_BALL_Z))


def _approx_vz_fp(
    ball: Mapping[str, Any],
    shadow: Mapping[str, Any],
    z: float,
) -> float:
    """Approximate fixed-point vertical velocity from Δ(shadow.y − ball.y)."""
    dz = _delta(shadow, "y") - _delta(ball, "y")
    # z_fp ≈ 10*z; one frame of Δz → 10*Δz in fp units.
    vz = 10.0 * dz
    if abs(vz) < 1e-3 and z > 0:
        # Unknown ascent/descent: mild downward so gravity path is live.
        vz = -_DEFAULT_HIT_VZ * 0.25
    return float(np.clip(vz, -80.0, 80.0))


def _approx_move_xy(
    ground: Mapping[str, Any],
) -> Tuple[float, float]:
    """Per-frame ground-plane steps (JAX ball_x/y track the shadow)."""
    mx = _delta(ground, "x")
    my = _delta(ground, "y")
    # Clip wild OC skip frames.
    return float(np.clip(mx, -8.0, 8.0)), float(np.clip(my, -8.0, 8.0))


def oc_frame_to_tennis_state(
    env: TennisJaxEnv,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> TennisState:
    del frame_index
    if print_assumptions:
        print_disclaimers("tennis")

    consts = env.consts
    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs

    player = find_object(objs, "Player")
    enemy = find_object(objs, "Enemy")
    ball = find_object(objs, "Ball")
    shadow = find_object(objs, "BallShadow")
    ps = find_object(objs, "PlayerScore")
    es = find_object(objs, "EnemyScore")

    player_state = state.player_state
    enemy_state = state.enemy_state
    ball_state = state.ball_state
    game_state = state.game_state

    if player is not None and int(player.get("w", 0)) > 0:
        player_state = player_state.replace(
            player_x=cast_like(int(player["x"]), player_state.player_x),
            player_y=cast_like(int(player["y"]), player_state.player_y),
        )
    if enemy is not None and int(enemy.get("w", 0)) > 0:
        enemy_state = enemy_state.replace(
            enemy_x=cast_like(int(enemy["x"]), enemy_state.enemy_x),
            enemy_y=cast_like(int(enemy["y"]), enemy_state.enemy_y),
        )

    ball_live = ball is not None and int(ball.get("w", 0)) > 0
    shadow_live = shadow is not None and int(shadow.get("w", 0)) > 0

    if ball_live or shadow_live:
        # Ground / shadow plane = JAX (ball_x, ball_y). Prefer BallShadow.
        ground = shadow if shadow_live else ball
        assert ground is not None
        gx = float(ground["x"])
        gy = float(ground["y"])

        z = 0.0
        if ball_live and shadow_live:
            z = _infer_ball_z(ball, shadow)  # type: ignore[arg-type]
        elif ball_live and not shadow_live:
            # No shadow: treat Ball as ground (z unknown → 0).
            gx = float(ball["x"])  # type: ignore[index]
            gy = float(ball["y"])  # type: ignore[index]

        move_x, move_y = _approx_move_xy(ground)
        # Ground plane is the shadow. Do NOT blend Ball dy while the shadow is
        # live — serve z-bounce moves the ball sprite without ground velocity and
        # would falsely clear is_serving.
        if not shadow_live and ball_live:
            move_x = float(np.clip(_delta(ball, "x"), -8.0, 8.0))  # type: ignore[arg-type]
            move_y = float(np.clip(_delta(ball, "y"), -8.0, 8.0))  # type: ignore[arg-type]

        game_middle = float(np.asarray(consts.GAME_MIDDLE))
        player_on_top = True
        if player is not None and enemy is not None:
            player_on_top = float(player["y"]) < float(enemy["y"])

        # Serve/stand-off: shadow sits still while z bounces. Mid-flight always has
        # ground-plane velocity (OC dx/dy or prevΔ) — use that alone, not z or
        # "still on this half of the court" (fails after the ball crosses mid).
        in_rally = abs(move_x) >= 0.5 or abs(move_y) >= 0.5

        if not in_rally:
            # Keep serving/stand-off: zero ground velocity; z bounce via is_serving path.
            ball_state = ball_state.replace(
                ball_x=cast_like(gx, ball_state.ball_x),
                ball_y=cast_like(gy, ball_state.ball_y),
                ball_z=cast_like(z, ball_state.ball_z),
                ball_z_fp=cast_like(z * 10.0, ball_state.ball_z_fp),
                ball_velocity_z_fp=cast_like(0.0, ball_state.ball_velocity_z_fp),
                move_x=cast_like(0.0, ball_state.move_x),
                move_y=cast_like(0.0, ball_state.move_y),
                bounces=cast_like(0, ball_state.bounces),
                last_hit=cast_like(-1, ball_state.last_hit),
            )
            game_state = game_state.replace(
                is_serving=cast_like(True, game_state.is_serving)
            )
            if player is not None:
                # Ball on top half ⇒ top player is serving (when player is on top).
                serving_is_player = (gy <= game_middle) if player_on_top else (gy > game_middle)
                player_state = player_state.replace(
                    player_serving=cast_like(serving_is_player, player_state.player_serving)
                )
        else:
            vz_fp = (
                _approx_vz_fp(ball, shadow, z)  # type: ignore[arg-type]
                if (ball_live and shadow_live)
                else (-_DEFAULT_HIT_VZ * 0.25 if z > 0 else 0.0)
            )
            if move_y > 0.05:
                last_hit = (
                    int(consts.PLAYER_CONST) if player_on_top else int(consts.ENEMY_CONST)
                )
            elif move_y < -0.05:
                last_hit = (
                    int(consts.ENEMY_CONST) if player_on_top else int(consts.PLAYER_CONST)
                )
            else:
                last_hit = (
                    int(consts.PLAYER_CONST) if gy > game_middle else int(consts.ENEMY_CONST)
                )

            ahead = 40.0
            hit_tx = gx + move_x * ahead
            hit_ty = gy + move_y * ahead
            hit_sx = gx - move_x * 5.0
            hit_sy = gy - move_y * 5.0

            ball_state = ball_state.replace(
                ball_x=cast_like(gx, ball_state.ball_x),
                ball_y=cast_like(gy, ball_state.ball_y),
                ball_z=cast_like(z, ball_state.ball_z),
                ball_z_fp=cast_like(z * 10.0, ball_state.ball_z_fp),
                ball_velocity_z_fp=cast_like(vz_fp, ball_state.ball_velocity_z_fp),
                move_x=cast_like(move_x, ball_state.move_x),
                move_y=cast_like(move_y, ball_state.move_y),
                ball_hit_start_x=cast_like(hit_sx, ball_state.ball_hit_start_x),
                ball_hit_start_y=cast_like(hit_sy, ball_state.ball_hit_start_y),
                ball_hit_target_x=cast_like(hit_tx, ball_state.ball_hit_target_x),
                ball_hit_target_y=cast_like(hit_ty, ball_state.ball_hit_target_y),
                bounces=cast_like(0 if z > 1.0 else 1, ball_state.bounces),
                last_hit=cast_like(last_hit, ball_state.last_hit),
            )
            game_state = game_state.replace(
                is_serving=cast_like(False, game_state.is_serving)
            )

    # Q2=A: only copy scores when .value exists.
    if ps is not None and "value" in ps and int(ps.get("value", -1)) >= 0:
        game_state = game_state.replace(
            player_score=cast_like(int(ps["value"]), game_state.player_score)
        )
    if es is not None and "value" in es and int(es.get("value", -1)) >= 0:
        game_state = game_state.replace(
            enemy_score=cast_like(int(es["value"]), game_state.enemy_score)
        )

    # Q3=B: serving / field side stay at RESET (except is_serving cleared in-rally).

    return state.replace(
        player_state=player_state,
        enemy_state=enemy_state,
        ball_state=ball_state,
        game_state=game_state,
    )


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["player"] = (float(player["x"]), float(player["y"]))
    enemy = find_object(objs, "Enemy")
    if enemy is not None and int(enemy.get("w", 0)) > 0:
        entities["enemy"] = (float(enemy["x"]), float(enemy["y"]))
    # Compare visual ball (airborne), not shadow.
    ball = find_object(objs, "Ball")
    if ball is not None and int(ball.get("w", 0)) > 0:
        entities["ball"] = (float(ball["x"]), float(ball["y"]))

    ps = find_object(objs, "PlayerScore")
    if ps is not None and "value" in ps and int(ps.get("value", -1)) >= 0:
        scores["player"] = float(ps["value"])
    es = find_object(objs, "EnemyScore")
    if es is not None and "value" in es and int(es.get("value", -1)) >= 0:
        scores["enemy"] = float(es["value"])

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: TennisState, env: TennisJaxEnv) -> dict:
    del env
    # Match OC Ball: sprite is at (ball_x, ball_y - ball_z).
    bx = float(state.ball_state.ball_x)
    by = float(state.ball_state.ball_y) - float(state.ball_state.ball_z)
    entities = {
        "player": (float(state.player_state.player_x), float(state.player_state.player_y)),
        "enemy": (float(state.enemy_state.enemy_x), float(state.enemy_state.enemy_y)),
        "ball": (bx, by),
    }
    scores = {
        "player": float(state.game_state.player_score),
        "enemy": float(state.game_state.enemy_score),
    }
    return {"entities": entities, "scores": scores}
