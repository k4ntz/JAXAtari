"""Translate OCAtari BankHeist object snapshots into JAXAtari BankHeistState."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.games.jax_bankheist import (
    BankHeistConstants,
    BankHeistState,
    Entity,
    JaxBankHeist,
    init_banks_or_police,
)

from .base import find_object, objects_as_dicts
from .registry import print_disclaimers

# BankHeist DIR_* (matches BankHeistConstants).
_DIR_DOWN = 0
_DIR_UP = 1
_DIR_RIGHT = 2
_DIR_LEFT = 3
_DIR_NOOP = 4

# Stick → facing. Diagonals follow player_input_step (vertical component wins).
_ACTION_TO_DIR = {
    "UP": _DIR_UP,
    "DOWN": _DIR_DOWN,
    "LEFT": _DIR_LEFT,
    "RIGHT": _DIR_RIGHT,
    "UPFIRE": _DIR_UP,
    "DOWNFIRE": _DIR_DOWN,
    "LEFTFIRE": _DIR_LEFT,
    "RIGHTFIRE": _DIR_RIGHT,
    "UPRIGHT": _DIR_UP,
    "UPLEFT": _DIR_UP,
    "DOWNRIGHT": _DIR_DOWN,
    "DOWNLEFT": _DIR_DOWN,
    "UPRIGHTFIRE": _DIR_UP,
    "UPLEFTFIRE": _DIR_UP,
    "DOWNRIGHTFIRE": _DIR_DOWN,
    "DOWNLEFTFIRE": _DIR_DOWN,
}


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
        return _DIR_RIGHT if dx > 0 else _DIR_LEFT
    return _DIR_DOWN if dy > 0 else _DIR_UP


def _action_name(action: Optional[int], meanings: Optional[Sequence[str]]) -> str:
    if action is None:
        return "NOOP"
    a = int(action)
    if meanings is not None and 0 <= a < len(meanings):
        return str(meanings[a]).upper()
    # ALE / JAXAtariAction index names when meanings missing.
    names = [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE",
    ]
    return names[a] if 0 <= a < len(names) else "NOOP"


def _dir_from_action_name(name: str, *, fallback: int) -> int:
    return int(_ACTION_TO_DIR.get(str(name).upper(), fallback))


def _atari_action_value(action: Optional[int], meanings: Optional[Sequence[str]]) -> int:
    """Map OC action index → JAXAtariAction enum int (stored in latched_action)."""
    name = _action_name(action, meanings)
    return int(getattr(Action, name, Action.NOOP))


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


def _coasting_direction_from_lookback(
    xy_lookback: Sequence[Tuple[float, float]],
    *,
    fallback: int = _DIR_NOOP,
) -> int:
    """Most recent nonzero step in ``xy_lookback`` (oldest → newest, incl. current).

    BankHeist often moves on alternate frames (speed 0.5), so a 1-frame Δ is
    frequently (0,0) even while the car is still coasting after the stick is
    released — scan further back for the last real displacement.
    """
    if len(xy_lookback) < 2:
        return int(fallback)
    for i in range(len(xy_lookback) - 1, 0, -1):
        x1, y1 = xy_lookback[i]
        x0, y0 = xy_lookback[i - 1]
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        if dx != 0.0 or dy != 0.0:
            return _infer_direction(dx, dy, fallback=fallback)
    return int(fallback)

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
    player_xy_lookback: Optional[Sequence[Tuple[float, float]]] = None,
    oc_action: Optional[int] = None,
    prev_oc_action: Optional[int] = None,
    action_meanings: Optional[Sequence[str]] = None,
    print_assumptions: bool = False,
    **_ignored,
) -> BankHeistState:
    """Reset JAX BankHeist (map 0), then overlay OC-derived fields.

    Forced assumptions (also in registry disclaimers):
    - Always ``level/map_id/difficulty_level = 0``
    - Speeds / latches / spawn timers internals → reset defaults (bank timers
      forced inactive when OC banks are placed so reset spawn does not clobber)
    - ``player_move_direction`` from multi-frame position lookback (car coasts
      on NOOP; single-frame Δ is often 0 at half-speed)
    - Facing / stick from current (else prev) OC action when directional
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
        # Build oldest→newest xy history for coasting inference.
        lookback: List[Tuple[float, float]] = []
        if player_xy_lookback:
            lookback.extend(
                (float(x), float(y)) for x, y in player_xy_lookback if x is not None
            )
        cur_xy = (float(player["x"]), float(player["y"]))
        if not lookback or lookback[-1] != cur_xy:
            # Ensure current frame is last; seed with single-frame prev if needed.
            if not lookback and prev_player_xy is not None:
                lookback.append(
                    (float(prev_player_xy[0]), float(prev_player_xy[1]))
                )
            lookback.append(cur_xy)

        dx, dy = _player_delta(player, prev_player_xy)
        move_dir = _coasting_direction_from_lookback(
            lookback, fallback=_DIR_NOOP
        )
        if move_dir == _DIR_NOOP:
            # Fall back to OC dx/dy / 1-frame prev when lookback is flat.
            move_dir = _infer_direction(dx, dy, fallback=_DIR_NOOP)

        act_name = _action_name(oc_action, action_meanings)
        face_dir = _dir_from_action_name(act_name, fallback=_DIR_NOOP)
        if face_dir == _DIR_NOOP:
            face_dir = _dir_from_action_name(
                _action_name(prev_oc_action, action_meanings),
                fallback=move_dir if move_dir != _DIR_NOOP else _DIR_NOOP,
            )
        if face_dir == _DIR_NOOP and move_dir != _DIR_NOOP:
            face_dir = move_dir
        # If stick says move but we never saw motion, start moving that way.
        if move_dir == _DIR_NOOP and face_dir != _DIR_NOOP:
            move_dir = face_dir

        updates["player"] = Entity(
            position=jnp.array(
                [int(player["x"]), int(player["y"])], dtype=jnp.int32
            ),
            direction=jnp.array(face_dir, dtype=jnp.int32),
            visibility=jnp.array(1, dtype=jnp.int32),
        )
        updates["player_move_direction"] = jnp.array(move_dir, dtype=jnp.int32)

        # First post-inject step applies latched_action from the *previous* frame.
        latch_src = prev_oc_action if prev_oc_action is not None else oc_action
        updates["latched_action"] = jnp.array(
            _atari_action_value(latch_src, action_meanings), dtype=jnp.int32
        )

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
