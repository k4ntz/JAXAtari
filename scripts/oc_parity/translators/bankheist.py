"""Translate OCAtari BankHeist object snapshots into JAXAtari BankHeistState."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_bankheist import (
    BankHeistConstants,
    BankHeistState,
    Entity,
    JaxBankHeist,
    init_banks_or_police,
)

from .base import find_object, objects_as_dicts
from .registry import print_disclaimers


def _infer_direction(
    dx: float,
    dy: float,
    *,
    fallback: int,
) -> int:
    """Map displacement to BankHeist DIR_* codes (0 down, 1 up, 2 right, 3 left, 4 noop)."""
    if dx == 0.0 and dy == 0.0:
        return int(fallback)
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 3  # RIGHT / LEFT
    return 0 if dy > 0 else 1  # DOWN / UP


def _player_delta(
    player: Mapping[str, Any],
    prev_xy: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    x = float(player.get("x", 0))
    y = float(player.get("y", 0))
    if prev_xy is not None:
        return x - float(prev_xy[0]), y - float(prev_xy[1])
    dx = float(player.get("dx", 0))
    dy = float(player.get("dy", 0))
    if dx == 0.0 and dy == 0.0:
        dx = x - float(player.get("prev_x", x))
        dy = y - float(player.get("prev_y", y))
    return dx, dy


def _collect_category(objs: Sequence[Mapping[str, Any]], category: str) -> List[Mapping[str, Any]]:
    return [
        o
        for o in objs
        if str(o.get("category", "")) == category and int(o.get("w", 0)) > 0
    ]


def _entity_from_objects(
    objects: Sequence[Mapping[str, Any]],
    *,
    n_slots: int = 3,
    default_direction: int = 4,
) -> Entity:
    """Pack up to ``n_slots`` OC objects into a batched BankHeist Entity."""
    base = init_banks_or_police()
    positions = np.array(base.position, dtype=np.int32).copy()
    directions = np.array(base.direction, dtype=np.int32).copy()
    visibilities = np.array(base.visibility, dtype=np.int32).copy()

    for i in range(min(n_slots, len(objects))):
        obj = objects[i]
        positions[i, 0] = int(obj.get("x", 0))
        positions[i, 1] = int(obj.get("y", 0))
        dx = float(obj.get("dx", 0))
        dy = float(obj.get("dy", 0))
        if dx == 0.0 and dy == 0.0:
            dx = float(obj.get("x", 0)) - float(obj.get("prev_x", obj.get("x", 0)))
            dy = float(obj.get("y", 0)) - float(obj.get("prev_y", obj.get("y", 0)))
        directions[i] = _infer_direction(dx, dy, fallback=default_direction)
        visibilities[i] = 1

    return Entity(
        position=jnp.asarray(positions, dtype=jnp.int32),
        direction=jnp.asarray(directions, dtype=jnp.int32),
        visibility=jnp.asarray(visibilities, dtype=jnp.int32),
    )


def _fuel_from_gas_tank(
    gas: Optional[Mapping[str, Any]],
    consts: BankHeistConstants,
) -> Optional[float]:
    if gas is None:
        return None
    h = float(gas.get("h", 0))
    tank_h = float(consts.TANK_HEIGHT)
    if tank_h <= 0:
        return None
    frac = float(np.clip(h / tank_h, 0.0, 1.0))
    return frac * float(consts.FUEL_CAPACITY)


def oc_frame_to_bankheist_state(
    env: JaxBankHeist,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    prev_player_xy: Optional[Tuple[float, float]] = None,
    print_assumptions: bool = False,
    **_ignored,
) -> BankHeistState:
    """Reset JAX BankHeist (map 0), then overlay OC-derived fields.

    Forced assumptions (also in registry disclaimers):
    - Always ``level/map_id/difficulty_level = 0``
    - Speeds / latches / spawn timers internals → reset defaults (bank timers
      forced inactive when OC banks are placed so reset spawn does not clobber)
    - Player direction inferred from motion
    - Fuel scaled from ``Gas_Tank`` sprite height when present
    """
    del frame_index  # reserved for future use / API parity with pong
    if print_assumptions:
        print_disclaimers("bankheist")

    consts: BankHeistConstants = env.consts
    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs

    player = find_object(objs, "Player")
    banks = _collect_category(objs, "Bank")
    police = _collect_category(objs, "Police")
    dynamite = find_object(objs, "Dynamite")
    score_obj = find_object(objs, "Score")
    gas = find_object(objs, "Gas_Tank")
    lives = _collect_category(objs, "Life")

    updates: dict = {}

    # Explicit map/level lock (reset already 0; reaffirm for clarity).
    updates["level"] = jnp.array(0, dtype=jnp.int32)
    updates["map_id"] = jnp.array(0, dtype=jnp.int32)
    updates["difficulty_level"] = jnp.array(0, dtype=jnp.int32)
    updates["map_collision"] = env.city_collision_maps[0]
    updates["spawn_points"] = env.city_spawns[0]

    if player is not None:
        dx, dy = _player_delta(player, prev_player_xy)
        direction = _infer_direction(
            dx, dy, fallback=int(np.asarray(state.player.direction))
        )
        updates["player"] = Entity(
            position=jnp.array(
                [int(player["x"]), int(player["y"])], dtype=jnp.int32
            ),
            direction=jnp.array(direction, dtype=jnp.int32),
            visibility=jnp.array(1, dtype=jnp.int32),
        )
        updates["player_move_direction"] = jnp.array(direction, dtype=jnp.int32)

    if banks:
        updates["bank_positions"] = _entity_from_objects(banks)
        # Prevent reset spawn timers ([1,1,1]) from immediately overwriting OC banks.
        updates["bank_spawn_timers"] = jnp.array([-1, -1, -1], dtype=jnp.int32)

    if police:
        updates["enemy_positions"] = _entity_from_objects(police)

    if dynamite is not None and int(dynamite.get("w", 0)) > 0:
        updates["dynamite_position"] = jnp.array(
            [int(dynamite["x"]), int(dynamite["y"])], dtype=jnp.int32
        )
        # Mid-fuse so the stick is visible / will eventually explode.
        updates["dynamite_timer"] = jnp.array(
            [int(consts.DYNAMITE_EXPLOSION_DELAY)], dtype=jnp.int32
        )

    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        updates["money"] = jnp.array(int(score_obj["value"]), dtype=jnp.int32)

    if lives:
        updates["player_lives"] = jnp.array(len(lives), dtype=jnp.int32)

    fuel = _fuel_from_gas_tank(gas, consts)
    if fuel is not None:
        updates["fuel"] = jnp.array(fuel, dtype=jnp.float32)

    # Ensure we can move immediately after transfer.
    updates["game_paused"] = jnp.array(False, dtype=jnp.bool_)

    return state.replace(**updates)


def extract_oc_compare_entities(
    objects: Sequence[Any],
    *,
    consts: Optional[BankHeistConstants] = None,
) -> dict:
    """OC snapshot → comparable entity positions + score/lives/fuel."""
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["player"] = (float(player["x"]), float(player["y"]))

    for i, bank in enumerate(_collect_category(objs, "Bank")):
        entities[f"bank_{i}"] = (float(bank["x"]), float(bank["y"]))

    for i, police in enumerate(_collect_category(objs, "Police")):
        entities[f"police_{i}"] = (float(police["x"]), float(police["y"]))

    dynamite = find_object(objs, "Dynamite")
    if dynamite is not None and int(dynamite.get("w", 0)) > 0:
        entities["dynamite"] = (float(dynamite["x"]), float(dynamite["y"]))

    score_obj = find_object(objs, "Score")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        scores["money"] = float(score_obj["value"])

    lives = _collect_category(objs, "Life")
    if lives:
        scores["lives"] = float(len(lives))

    gas = find_object(objs, "Gas_Tank")
    if gas is not None and consts is not None:
        fuel = _fuel_from_gas_tank(gas, consts)
        if fuel is not None:
            scores["fuel"] = float(fuel)
    elif gas is not None:
        # Normalized tank fraction if consts unavailable.
        scores["fuel_frac"] = float(gas.get("h", 0)) / 25.0

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: BankHeistState, env: JaxBankHeist) -> dict:
    """JAX BankHeistState → comparable entity positions + score/lives/fuel."""
    entities: dict = {}
    pos = np.asarray(state.player.position)
    entities["player"] = (float(pos[0]), float(pos[1]))

    bank_pos = np.asarray(state.bank_positions.position)
    bank_vis = np.asarray(state.bank_positions.visibility)
    bi = 0
    for i in range(bank_vis.shape[0]):
        if int(bank_vis[i]) > 0:
            entities[f"bank_{bi}"] = (float(bank_pos[i, 0]), float(bank_pos[i, 1]))
            bi += 1

    pol_pos = np.asarray(state.enemy_positions.position)
    pol_vis = np.asarray(state.enemy_positions.visibility)
    pi = 0
    for i in range(pol_vis.shape[0]):
        if int(pol_vis[i]) > 0:
            entities[f"police_{pi}"] = (float(pol_pos[i, 0]), float(pol_pos[i, 1]))
            pi += 1

    dyn = np.asarray(state.dynamite_position)
    if not (int(dyn[0]) == -1 and int(dyn[1]) == -1):
        entities["dynamite"] = (float(dyn[0]), float(dyn[1]))

    scores = {
        "money": float(state.money),
        "lives": float(state.player_lives),
        "fuel": float(state.fuel),
    }
    return {"entities": entities, "scores": scores}
