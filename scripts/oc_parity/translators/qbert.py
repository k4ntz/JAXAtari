"""Translate OCAtari Qbert object snapshots into JAXAtari QbertState."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import jax
import numpy as np

from jaxatari.games.jax_qbert import JaxQbert, QbertState

from .base import cast_like, find_object, objects_as_dicts
from .registry import print_disclaimers

# Renderer table (row-major pyramid). Index = i*(i-1)/2 + (j-1), i=row 1..6, j=col 1..i.
_QBERT_POSITIONS = np.array(
    [
        [74, 18],
        [62, 47],
        [86, 47],
        [50, 76],
        [74, 76],
        [98, 76],
        [38, 105],
        [62, 105],
        [86, 105],
        [110, 105],
        [26, 134],
        [50, 134],
        [74, 134],
        [98, 134],
        [122, 134],
        [14, 163],
        [38, 163],
        [62, 163],
        [86, 163],
        [110, 163],
        [134, 163],
    ],
    dtype=np.float64,
)

_GRID_CELLS: List[Tuple[int, int]] = []
for _i in range(1, 7):
    for _j in range(1, _i + 1):
        _GRID_CELLS.append((_j, _i))

# Reject OC sprites that are mid-fall / off-pyramid (seen: Player y=-49, Coily x=0,y=9).
_MAX_SNAP_DIST = 28.0
_SCREEN_X = (0, 160)
_SCREEN_Y = (0, 210)


def _pixel_to_grid(
    x: float, y: float, *, max_dist: float = _MAX_SNAP_DIST
) -> Optional[Tuple[int, int]]:
    """Nearest pyramid cell [j,i]=[col,row], or None if too far / off-screen."""
    if not (_SCREEN_X[0] <= x < _SCREEN_X[1] and _SCREEN_Y[0] <= y < _SCREEN_Y[1]):
        return None
    best = None
    best_d = 1e9
    for (j, i), (px, py) in zip(_GRID_CELLS, _QBERT_POSITIONS):
        d = float(np.hypot(x - px, y - py))
        if d < best_d:
            best_d = d
            best = (int(j), int(i))
    if best is None or best_d > max_dist:
        return None
    return best


def _on_pyramid(state: QbertState, j: int, i: int) -> bool:
    """True if cell is a walkable cube (or disk)."""
    val = int(np.asarray(state.pyramid)[i, j])
    return val >= 0 or val == -2


def _enemy_category_to_field(category: str) -> Optional[str]:
    c = category.lower().replace("_", "")
    if c == "sam":
        return "sam_position"
    if c in ("coily", "snake"):
        return "snake_position"
    if c == "purpleball":
        return "purple_ball_position"
    if c == "greenball":
        return "green_ball_position"
    if c == "redball":
        return "red_ball_positions"
    return None


# Soft-survival window: enemies frozen (step != freeze) so JAX RNG hops don't fire.
_RNG_FREEZE_STEPS = 180

# prng_state slots used by move_* (see jax_qbert.move_purple_ball etc.).
_PRNG_PURPLE = 0
_PRNG_GREEN = 1
_PRNG_SAM = 2
_PRNG_RED = (3, 4, 5)


def _want_down_not_right(obj: Mapping[str, Any]) -> Optional[bool]:
    """Infer Purple/Sam/ball hop: True→DOWN, False→RIGHT, None→leave RNG."""
    dx = float(obj.get("dx", 0))
    dy = float(obj.get("dy", 0))
    if dx == 0.0 and dy == 0.0:
        px = float(obj.get("x", 0)) - float(obj.get("prev_x", obj.get("x", 0)))
        py = float(obj.get("y", 0)) - float(obj.get("prev_y", obj.get("y", 0)))
        dx, dy = px, py
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return None
    # Balls only choose DOWN vs RIGHT in JAX; map dominant motion.
    if abs(dy) >= abs(dx):
        return dy >= 0  # down/stay → DOWN branch
    return False if dx > 0 else True  # leftish → DOWN (only available vertical-ish)


def _prng_key_forcing_down(want_down: bool, salt: int) -> Any:
    """Find a PRNGKey whose next uniform(1..256) is </> 128 as needed."""
    import jax

    for i in range(4096):
        key = jax.random.PRNGKey((salt * 4096 + i) & 0xFFFFFFFF)
        val = int(jax.random.uniform(key, (), minval=1, maxval=256))
        if want_down and val < 128:
            return key
        if (not want_down) and val >= 128:
            return key
    return jax.random.PRNGKey(salt)


def oc_frame_to_qbert_state(
    env: JaxQbert,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> QbertState:
    if print_assumptions:
        print_disclaimers("qbert")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs

    # Soft-freeze RNG hops for _RNG_FREEZE_STEPS (step stays behind freeze).
    # After that, freeze catches up via normal +1 lockstep and PRNG bias applies
    # to the first Purple/Sam/ball DOWN-vs-RIGHT choice.
    updates: dict = {
        "level_number": cast_like(1, state.level_number),
        "round_number": cast_like(1, state.round_number),
        "step_counter": cast_like(0, state.step_counter),
        "green_ball_freeze_step": cast_like(
            _RNG_FREEZE_STEPS, state.green_ball_freeze_step
        ),
        "enemy_moving_counter": cast_like(1, state.enemy_moving_counter),
        "just_spawned": cast_like(0, state.just_spawned),
        "dead_animation_counter": cast_like(0, state.dead_animation_counter),
        "next_round_animation_counter": cast_like(
            0, state.next_round_animation_counter
        ),
        "is_player_moving": cast_like(0, state.is_player_moving),
        "player_moving_counter": cast_like(0, state.player_moving_counter),
        "player_position_category": cast_like(0, state.player_position_category),
        "same_cell_frames": cast_like(0, state.same_cell_frames),
        "snake_lock": cast_like(np.array([-1, -1], dtype=np.int32), state.snake_lock),
    }
    del frame_index

    prng = [np.array(k) for k in state.prng_state]

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        cell = _pixel_to_grid(float(player["x"]), float(player["y"]))
        if cell is not None and _on_pyramid(state, cell[0], cell[1]):
            pos = np.array(cell, dtype=np.int32)
            updates["player_position"] = cast_like(pos, state.player_position)
            updates["player_last_position"] = cast_like(pos, state.player_last_position)
            dx, dy = float(player.get("dx", 0)), float(player.get("dy", 0))
            if abs(dx) + abs(dy) > 0:
                if dx <= 0 and dy <= 0:
                    d = 0
                elif dx <= 0 and dy > 0:
                    d = 1
                elif dx > 0 and dy > 0:
                    d = 2
                else:
                    d = 3
                updates["player_direction"] = cast_like(d, state.player_direction)

    # Pyramid colors RESET (OC Cube RGB unreliable).
    red_slots = np.array([[-1, -1], [-1, -1], [-1, -1]], dtype=np.int32)
    red_i = 0
    for obj in objs:
        if int(obj.get("w", 0)) <= 0:
            continue
        field = _enemy_category_to_field(str(obj.get("category", "")))
        if field is None:
            continue
        cell = _pixel_to_grid(float(obj["x"]), float(obj["y"]))
        if cell is None or not _on_pyramid(state, cell[0], cell[1]):
            continue
        arr = np.array(cell, dtype=np.int32)
        want = _want_down_not_right(obj)
        if field == "red_ball_positions":
            if red_i < 3:
                red_slots[red_i] = arr
                if want is not None:
                    prng[_PRNG_RED[red_i]] = np.asarray(
                        _prng_key_forcing_down(want, salt=10 + red_i)
                    )
                red_i += 1
        else:
            updates[field] = cast_like(arr, getattr(state, field))
            if want is not None:
                if field == "purple_ball_position":
                    prng[_PRNG_PURPLE] = np.asarray(
                        _prng_key_forcing_down(want, salt=1)
                    )
                elif field == "green_ball_position":
                    prng[_PRNG_GREEN] = np.asarray(
                        _prng_key_forcing_down(want, salt=2)
                    )
                elif field == "sam_position":
                    prng[_PRNG_SAM] = np.asarray(
                        _prng_key_forcing_down(want, salt=3)
                    )
    if red_i:
        updates["red_ball_positions"] = cast_like(red_slots, state.red_ball_positions)

    updates["prng_state"] = cast_like(np.stack(prng, axis=0), state.prng_state)

    score = find_object(objs, "Score")
    if score is not None and "value" in score and int(score.get("value", -1)) >= 0:
        updates["player_score"] = cast_like(int(score["value"]), state.player_score)
    lives = find_object(objs, "Lives")
    if lives is not None and "value" in lives and int(lives.get("value", -1)) >= 0:
        updates["lives"] = cast_like(int(lives["value"]), state.lives)

    return state.replace(**updates)


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        cell = _pixel_to_grid(float(player["x"]), float(player["y"]))
        if cell is not None:
            entities["player"] = (float(cell[0]), float(cell[1]))
    for obj in objs:
        if int(obj.get("w", 0)) <= 0:
            continue
        field = _enemy_category_to_field(str(obj.get("category", "")))
        if field is None:
            continue
        cell = _pixel_to_grid(float(obj["x"]), float(obj["y"]))
        if cell is None:
            continue
        cat = str(obj.get("category", "")).lower()
        entities[cat] = (float(cell[0]), float(cell[1]))

    score = find_object(objs, "Score")
    if score is not None and "value" in score and int(score.get("value", -1)) >= 0:
        scores["score"] = float(score["value"])
    lives = find_object(objs, "Lives")
    if lives is not None and "value" in lives and int(lives.get("value", -1)) >= 0:
        scores["lives"] = float(lives["value"])

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: QbertState, env: JaxQbert) -> dict:
    del env
    entities = {
        "player": (float(state.player_position[0]), float(state.player_position[1])),
    }
    sam = np.asarray(state.sam_position)
    if int(sam[0]) != -1:
        entities["sam"] = (float(sam[0]), float(sam[1]))
    snake = np.asarray(state.snake_position)
    if int(snake[0]) != -1:
        entities["coily"] = (float(snake[0]), float(snake[1]))
    purple = np.asarray(state.purple_ball_position)
    if int(purple[0]) != -1:
        entities["purpleball"] = (float(purple[0]), float(purple[1]))
    scores = {
        "score": float(state.player_score),
        "lives": float(state.lives),
    }
    return {"entities": entities, "scores": scores}
