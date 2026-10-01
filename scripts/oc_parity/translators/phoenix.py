"""Translate OCAtari Phoenix object snapshots into JAXAtari PhoenixState."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_phoenix import JaxPhoenix, PhoenixState

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers

_N_ENEMY_SLOTS = 8
_N_ENEMY_PROJ = 8
# Block fresh RNG dive rolls after transfer (can_attack requires cooldown==0).
_ATTACK_FREEZE_COOLDOWN = 10_000
# Bird this many px below formation home ⇒ treat as mid-dive (OC-observed).
_DIVE_Y_MARGIN = 12.0
_DIVE_CONTINUE = 28.0  # extend target when OC dy>0


def _group_horizontal_direction(birds: Sequence[Mapping[str, Any]]) -> float:
    """Phoenix wave moves as one group. JAX level-1 default is −1 (left)."""
    dxs = [float(b.get("dx", 0)) for b in birds]
    nonzero = [d for d in dxs if d != 0.0]
    if not nonzero:
        return -1.0
    return -1.0 if sum(nonzero) < 0 else 1.0


def _pack_enemies(
    base_x: Any,
    base_y: Any,
    birds: Sequence[Mapping[str, Any]],
) -> Tuple[Any, Any, Any, List[Tuple[int, Mapping[str, Any], float]]]:
    """Fill up to 8 slots; unused → x=-1. Returns (xs, ys, dirs, placements).

    placements: (slot, bird_dict, formation_home_y) for each packed bird.
    """
    ref_x = jnp.asarray(base_x)
    ref_y = jnp.asarray(base_y)
    xs = np.full(ref_x.shape, -1.0, dtype=np.float64)
    ys = np.full(ref_y.shape, -1.0, dtype=np.float64)
    group_dir = _group_horizontal_direction(birds)
    dirs = np.full(ref_x.shape, group_dir, dtype=np.float64)

    anchors = np.stack(
        [np.asarray(base_x, dtype=float), np.asarray(base_y, dtype=float)], axis=1
    )
    used: set = set()
    placements: List[Tuple[int, Mapping[str, Any], float]] = []
    ordered = sorted(birds, key=lambda o: (float(o.get("y", 0)), float(o.get("x", 0))))
    for bird in ordered[:_N_ENEMY_SLOTS]:
        bx, by = float(bird.get("x", 0)), float(bird.get("y", 0))
        dists = [
            (float(np.hypot(bx - anchors[i, 0], by - anchors[i, 1])), i)
            for i in range(_N_ENEMY_SLOTS)
            if i not in used
        ]
        if not dists:
            break
        _, slot = min(dists)
        used.add(slot)
        home_y = float(anchors[slot, 1])
        xs[slot] = bx
        ys[slot] = by
        placements.append((slot, bird, home_y))

    return (
        cast_like(xs, ref_x),
        cast_like(ys, ref_y),
        cast_like(dirs, ref_x),
        placements,
    )


def _apply_dive_from_oc(
    state: PhoenixState,
    placements: List[Tuple[int, Mapping[str, Any], float]],
    updates: dict,
) -> None:
    """Freeze RNG attacks; arm dive FSM only when OC bird is clearly below home."""
    n = int(np.asarray(state.phoenix_do_attack).shape[0])
    do_attack = np.zeros(n, dtype=bool)
    returning = np.zeros(n, dtype=bool)
    original_y = np.full(n, -1.0, dtype=np.float64)
    target_y = np.full(n, -1.0, dtype=np.float64)
    cooldown = np.full(n, _ATTACK_FREEZE_COOLDOWN, dtype=np.int32)
    drift = np.zeros(n, dtype=np.float64)

    for slot, bird, home_y in placements:
        by = float(bird.get("y", 0))
        dy = float(bird.get("dy", 0))
        if by < home_y + _DIVE_Y_MARGIN:
            continue  # still in formation → leave frozen
        # Mid-dive (or holding / returning) observed in OC.
        do_attack[slot] = True
        original_y[slot] = home_y
        cooldown[slot] = 0
        if dy > 0.5:
            target_y[slot] = by + _DIVE_CONTINUE
            returning[slot] = False
        elif dy < -0.5:
            # Climbing back toward formation.
            target_y[slot] = by
            returning[slot] = True
            do_attack[slot] = True
        else:
            # Holding near dive depth.
            target_y[slot] = by

    updates["phoenix_do_attack"] = cast_like(do_attack, state.phoenix_do_attack)
    updates["phoenix_returning"] = cast_like(returning, state.phoenix_returning)
    updates["phoenix_original_y"] = cast_like(original_y, state.phoenix_original_y)
    updates["phoenix_attack_target_y"] = cast_like(target_y, state.phoenix_attack_target_y)
    updates["phoenix_cooldown"] = cast_like(cooldown, state.phoenix_cooldown)
    updates["phoenix_drift"] = cast_like(drift, state.phoenix_drift)
    # Bat-wave dive timer: push out so bat levels don't RNG-dive immediately either.
    updates["bat_dive_timer"] = cast_like(_ATTACK_FREEZE_COOLDOWN, state.bat_dive_timer)
    updates["bat_dive_phase"] = cast_like(0, state.bat_dive_phase)


def oc_frame_to_phoenix_state(
    env: JaxPhoenix,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> PhoenixState:
    if print_assumptions:
        print_disclaimers("phoenix")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs
    updates: dict = {
        "level": cast_like(1, state.level),
        "level_transition_timer": cast_like(0, state.level_transition_timer),
        "step_counter": cast_like(max(0, int(frame_index)), state.step_counter),
    }

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        updates["player_x"] = cast_like(int(player["x"]), state.player_x)
        updates["player_y"] = cast_like(int(player["y"]), state.player_y)

    birds = collect_category(objs, "Phoenix")
    placements: List[Tuple[int, Mapping[str, Any], float]] = []
    if birds:
        ex, ey, hdir, placements = _pack_enemies(
            state.enemies_x, state.enemies_y, birds
        )
        updates["enemies_x"] = ex
        updates["enemies_y"] = ey
        updates["horizontal_direction_enemies"] = cast_like(
            hdir, state.horizontal_direction_enemies
        )
        dying = np.zeros(np.asarray(state.phoenix_dying).shape, dtype=bool)
        updates["phoenix_dying"] = cast_like(dying, state.phoenix_dying)
        updates["bat_dying"] = cast_like(dying, state.bat_dying)

    _apply_dive_from_oc(state, placements, updates)

    pp = find_object(objs, "Player_Projectile")
    if pp is not None and int(pp.get("w", 0)) > 0:
        updates["projectile_x"] = cast_like(int(pp["x"]), state.projectile_x)
        updates["projectile_y"] = cast_like(int(pp["y"]), state.projectile_y)
    else:
        updates["projectile_x"] = cast_like(-1, state.projectile_x)
        updates["projectile_y"] = cast_like(-1, state.projectile_y)

    eprojs = collect_category(objs, "Enemy_Projectile")
    epx = np.full((_N_ENEMY_PROJ,), -1, dtype=np.int32)
    epy = np.full((_N_ENEMY_PROJ,), -1, dtype=np.int32)
    for i, ep in enumerate(eprojs[:_N_ENEMY_PROJ]):
        epx[i] = int(ep["x"])
        epy[i] = int(ep["y"])
    updates["enemy_projectile_x"] = cast_like(epx, state.enemy_projectile_x)
    updates["enemy_projectile_y"] = cast_like(epy, state.enemy_projectile_y)

    score = find_object(objs, "Score")
    if score is not None and "value" in score and int(score.get("value", -1)) >= 0:
        updates["score"] = cast_like(int(score["value"]), state.score)

    lives = collect_category(objs, "Life")
    if lives:
        updates["lives"] = cast_like(len(lives), state.lives)

    return state.replace(**updates)


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["player"] = (float(player["x"]), float(player["y"]))
    for i, b in enumerate(collect_category(objs, "Phoenix")):
        entities[f"phoenix_{i}"] = (float(b["x"]), float(b["y"]))
    pp = find_object(objs, "Player_Projectile")
    if pp is not None and int(pp.get("w", 0)) > 0:
        entities["missile"] = (float(pp["x"]), float(pp["y"]))
    for i, ep in enumerate(collect_category(objs, "Enemy_Projectile")):
        entities[f"enemy_missile_{i}"] = (float(ep["x"]), float(ep["y"]))

    score = find_object(objs, "Score")
    if score is not None and "value" in score and int(score.get("value", -1)) >= 0:
        scores["score"] = float(score["value"])
    lives = collect_category(objs, "Life")
    if lives:
        scores["lives"] = float(len(lives))

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: PhoenixState, env: JaxPhoenix) -> dict:
    del env
    entities = {
        "player": (float(state.player_x), float(state.player_y)),
    }
    ex = np.asarray(state.enemies_x)
    ey = np.asarray(state.enemies_y)
    pi = 0
    for i in range(ex.shape[0]):
        if float(ex[i]) > -1:
            entities[f"phoenix_{pi}"] = (float(ex[i]), float(ey[i]))
            pi += 1
    if int(state.projectile_x) >= 0:
        entities["missile"] = (float(state.projectile_x), float(state.projectile_y))
    epx = np.asarray(state.enemy_projectile_x)
    epy = np.asarray(state.enemy_projectile_y)
    ei = 0
    for i in range(epx.shape[0]):
        if int(epx[i]) >= 0:
            entities[f"enemy_missile_{ei}"] = (float(epx[i]), float(epy[i]))
            ei += 1
    scores = {"score": float(state.score), "lives": float(state.lives)}
    return {"entities": entities, "scores": scores}
