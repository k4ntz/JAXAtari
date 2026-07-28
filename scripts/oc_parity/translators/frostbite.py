"""Translate OCAtari Frostbite object snapshots into JAXAtari FrostbiteState."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_frostbite import FrostbiteState, JaxFrostbite

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers

# OC FloatingBlock rgbs observed in-game.
_OC_ICE_WHITE_RGB = (214, 214, 214)
_OC_ICE_BLUE_RGB = (84, 138, 210)


def _nearest_row(y: float, row_ys: Sequence[int]) -> int:
    return int(np.argmin([abs(y - ry) for ry in row_ys]))


def _rgb_tuple(obj: Mapping[str, Any]) -> Tuple[int, int, int]:
    rgb = obj.get("rgb", (0, 0, 0))
    try:
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    except Exception:
        return 0, 0, 0


def _is_blue_ice(obj: Mapping[str, Any]) -> bool:
    r, g, b = _rgb_tuple(obj)
    # Blue ice is distinctly cooler; white/gray is high luminance.
    if b > r + 20 and b > 120:
        return True
    # Distance to known swatches.
    dw = (r - _OC_ICE_WHITE_RGB[0]) ** 2 + (g - _OC_ICE_WHITE_RGB[1]) ** 2 + (
        b - _OC_ICE_WHITE_RGB[2]
    ) ** 2
    db = (r - _OC_ICE_BLUE_RGB[0]) ** 2 + (g - _OC_ICE_BLUE_RGB[1]) ** 2 + (
        b - _OC_ICE_BLUE_RGB[2]
    ) ** 2
    return db < dw


def _dir_from_dx(dx: float) -> int:
    """JAX: 0=right, 1=left — match OC screen Δx (positive x → right)."""
    if dx == 0.0:
        return 0
    return 0 if dx > 0 else 1


def _ice_row_below(y: float, row_ys: Sequence[int]) -> int:
    """Index of the first ice row at or below ``y`` (screen y grows downward)."""
    below = [i for i, ry in enumerate(row_ys) if float(ry) >= float(y) - 1.0]
    if below:
        return int(below[0])
    return int(len(row_ys) - 1)


def oc_frame_to_frostbite_state(
    env: JaxFrostbite,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> FrostbiteState:
    del frame_index
    if print_assumptions:
        print_disclaimers("frostbite")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs
    updates: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        updates["bailey_x"] = cast_like(int(player["x"]), state.bailey_x)
        updates["bailey_y"] = cast_like(int(player["y"]), state.bailey_y)
        dx = float(player.get("dx", 0))
        if dx == 0.0:
            dx = float(player.get("x", 0)) - float(player.get("prev_x", player.get("x", 0)))
        if dx != 0:
            updates["bailey_direction"] = cast_like(
                0 if dx > 0 else 1, state.bailey_direction
            )

    # Ice: bucket FloatingBlock by nearest ICE_ROW_Y, sort x into ≤6 slots.
    # Also transfer per-row color (white/blue) and travel direction.
    blocks = collect_category(objs, "FloatingBlock")
    if blocks:
        row_ys = list(env.consts.ICE_ROW_Y)
        positions = np.full(
            (4, 6), -512, dtype=np.dtype(np.asarray(state.ice_block_positions).dtype)
        )
        counts = np.zeros(4, dtype=np.dtype(np.asarray(state.ice_block_counts).dtype))
        colors = np.array(
            state.ice_colors, dtype=np.dtype(np.asarray(state.ice_colors).dtype)
        ).copy()
        dirs = np.array(
            state.ice_directions, dtype=np.dtype(np.asarray(state.ice_directions).dtype)
        ).copy()
        buckets: List[List[Mapping[str, Any]]] = [[] for _ in range(4)]
        for b in blocks:
            buckets[_nearest_row(float(b["y"]), row_ys)].append(b)
        for r in range(4):
            row = sorted(buckets[r], key=lambda o: float(o.get("x", 0)))
            for j, b in enumerate(row[:6]):
                positions[r, j] = int(b["x"])
            counts[r] = min(6, len(row))
            if row:
                # Majority blue → row blue.
                blue_votes = sum(1 for b in row if _is_blue_ice(b))
                colors[r] = (
                    env.consts.COLOR_ICE_BLUE
                    if blue_votes * 2 >= len(row)
                    else env.consts.COLOR_ICE_WHITE
                )
                # Direction from mean dx (0=right, 1=left). Keep alternating default if still.
                dxs = [float(b.get("dx", 0)) for b in row]
                dxs = [
                    (
                        d
                        if d != 0
                        else float(b.get("x", 0)) - float(b.get("prev_x", b.get("x", 0)))
                    )
                    for d, b in zip(dxs, row)
                ]
                mean_dx = float(np.mean(dxs)) if dxs else 0.0
                if abs(mean_dx) >= 0.5:
                    dirs[r] = 0 if mean_dx > 0 else 1
        updates["ice_block_positions"] = cast_like(positions, state.ice_block_positions)
        updates["ice_block_counts"] = cast_like(counts, state.ice_block_counts)
        updates["ice_colors"] = cast_like(colors, state.ice_colors)
        updates["ice_directions"] = cast_like(dirs, state.ice_directions)
        ice_x = np.array(
            state.ice_x, dtype=np.dtype(np.asarray(state.ice_x).dtype)
        ).copy()
        for r in range(4):
            if counts[r] > 0:
                ice_x[r] = positions[r, 0]
        updates["ice_x"] = cast_like(ice_x, state.ice_x)

    # Do NOT transfer House → building_igloo_idx: mapping mid-progress made the
    # igloo auto-complete as white ice was painted. Leave RESET (-1).

    birds = collect_category(objs, "Bird")
    obstacles = (
        birds
        + collect_category(objs, "Crab")
        + collect_category(objs, "Clam")
        + collect_category(objs, "GreenFish")
    )
    if obstacles:
        ox = np.array(state.obstacle_x, dtype=np.dtype(np.asarray(state.obstacle_x).dtype)).copy()
        oy = np.array(state.obstacle_y, dtype=np.dtype(np.asarray(state.obstacle_y).dtype)).copy()
        oactive = np.zeros(ox.shape[0], dtype=np.dtype(np.asarray(state.obstacle_active).dtype))
        otypes = np.array(
            state.obstacle_types, dtype=np.dtype(np.asarray(state.obstacle_types).dtype)
        ).copy()
        odir = np.array(
            state.obstacle_directions,
            dtype=np.dtype(np.asarray(state.obstacle_directions).dtype),
        ).copy()
        type_map = {"Bird": 0, "GreenFish": 1, "Crab": 2, "Clam": 3}
        # Birds fly opposite the ice floe beneath them; nearest-row fallback
        # wrongly sends the upper bird left when ice row 0 goes left.
        ice_dirs = np.asarray(updates.get("ice_directions", state.ice_directions))
        row_ys = list(env.consts.ICE_ROW_Y)
        for i, ob in enumerate(obstacles[: ox.shape[0]]):
            ox[i] = int(ob["x"])
            oy[i] = int(ob["y"])
            oactive[i] = 1
            otypes[i] = type_map.get(str(ob.get("category", "Bird")), 0)
            dx = float(ob.get("dx", 0))
            if dx == 0.0:
                dx = float(ob.get("x", 0)) - float(ob.get("prev_x", ob.get("x", 0)))
            if abs(dx) >= 0.5:
                odir[i] = _dir_from_dx(dx)
            else:
                row = _ice_row_below(float(ob["y"]), row_ys)
                ice_dir = int(ice_dirs[row]) if row < len(ice_dirs) else 0
                # Opposite of the ice below (Atari lane pairing).
                odir[i] = 1 - ice_dir
        updates["obstacle_x"] = cast_like(ox, state.obstacle_x)
        updates["obstacle_y"] = cast_like(oy, state.obstacle_y)
        updates["obstacle_active"] = cast_like(oactive, state.obstacle_active)
        updates["obstacle_types"] = cast_like(otypes, state.obstacle_types)
        updates["obstacle_directions"] = cast_like(odir, state.obstacle_directions)

    bear = find_object(objs, "Bear")
    if bear is not None and int(bear.get("w", 0)) > 0:
        updates["polar_grizzly_x"] = cast_like(int(bear["x"]), state.polar_grizzly_x)
        updates["polar_grizzly_active"] = cast_like(1, state.polar_grizzly_active)
        bdx = float(bear.get("dx", 0))
        if bdx != 0:
            updates["polar_grizzly_direction"] = cast_like(
                _dir_from_dx(bdx), state.polar_grizzly_direction
            )

    # score / temperature / lives RESET (no .value on OC HUD).

    if updates:
        state = state.replace(**updates)
    return state


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        entities["bailey"] = (float(player["x"]), float(player["y"]))

    row_ys = (98, 122, 147, 172)
    blocks = collect_category(objs, "FloatingBlock")
    buckets: List[List[Mapping[str, Any]]] = [[] for _ in range(4)]
    for b in blocks:
        buckets[_nearest_row(float(b["y"]), row_ys)].append(b)
    for r in range(4):
        row = sorted(buckets[r], key=lambda o: float(o.get("x", 0)))
        for j, b in enumerate(row):
            entities[f"ice_r{r}_b{j}"] = (float(b["x"]), float(b["y"]))

    for i, b in enumerate(collect_category(objs, "Bird")):
        entities[f"bird_{i}"] = (float(b["x"]), float(b["y"]))
    house = find_object(objs, "House")
    if house is not None and int(house.get("w", 0)) > 0:
        entities["house"] = (float(house["x"]), float(house["y"]))
    bear = find_object(objs, "Bear")
    if bear is not None and int(bear.get("w", 0)) > 0:
        entities["bear"] = (float(bear["x"]), float(bear["y"]))

    return {"entities": entities, "scores": {}}


def extract_jax_compare_entities(state: FrostbiteState, env: JaxFrostbite) -> dict:
    entities = {
        "bailey": (float(state.bailey_x), float(state.bailey_y)),
    }
    row_ys = list(env.consts.ICE_ROW_Y)
    positions = np.asarray(state.ice_block_positions)
    counts = np.asarray(state.ice_block_counts)
    for r in range(positions.shape[0]):
        for j in range(int(counts[r])):
            x = float(positions[r, j])
            if x > -500:
                entities[f"ice_r{r}_b{j}"] = (x, float(row_ys[r]))
    ox = np.asarray(state.obstacle_x)
    oy = np.asarray(state.obstacle_y)
    oa = np.asarray(state.obstacle_active)
    bi = 0
    for i in range(len(oa)):
        if int(oa[i]) > 0:
            entities[f"bird_{bi}"] = (float(ox[i]), float(oy[i]))
            bi += 1
    if int(state.polar_grizzly_active) > 0:
        entities["bear"] = (float(state.polar_grizzly_x), float(row_ys[0] - 20))
    return {"entities": entities, "scores": {}}
