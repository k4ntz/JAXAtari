"""Translate OCAtari Enduro snapshots into JAXAtari EnduroGameState."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_enduro import EnduroGameState, JaxEnduro

from .base import cast_like, collect_category, find_object, objects_as_dicts
from .registry import print_disclaimers


def oc_frame_to_enduro_state(
    env: JaxEnduro,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> EnduroGameState:
    del frame_index
    if print_assumptions:
        print_disclaimers("enduro")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs
    updates: dict = {}

    player = find_object(objs, "Player")
    if player is not None and int(player.get("w", 0)) > 0:
        updates["player_x"] = cast_like(float(player["x"]), state.player_x)
        updates["player_y"] = cast_like(float(player["y"]), state.player_y)

    # Near→far: larger y first into visible slots (player is near bottom).
    cars = sorted(
        [c for c in collect_category(objs, "Car") if int(c.get("w", 0)) > 0],
        key=lambda o: -float(o.get("y", 0)),
    )
    vis = np.array(state.visible_opponent_positions, copy=True)
    # Clear then pack.
    vis[:, 0] = -1
    n = min(len(cars), vis.shape[0])
    for i in range(n):
        vis[i, 0] = float(cars[i]["x"])
        vis[i, 1] = float(cars[i]["y"])
        # Keep color_idx from reset / default 0.
    updates["visible_opponent_positions"] = cast_like(vis, state.visible_opponent_positions)

    score_obj = find_object(objs, "PlayerScore")
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        # Enduro score is often "cars passed"; map loosely to cars_to_pass inverse if needed.
        pass
    level_obj = find_object(objs, "Level")
    if level_obj is not None and int(level_obj.get("value", -1)) >= 0:
        updates["level"] = cast_like(int(level_obj["value"]), state.level)

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
    cars = sorted(
        [c for c in collect_category(objs, "Car") if int(c.get("w", 0)) > 0],
        key=lambda o: -float(o.get("y", 0)),
    )
    for i, car in enumerate(cars[:7]):
        entities[f"car_{i}"] = (float(car["x"]), float(car["y"]))
    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: EnduroGameState, env: JaxEnduro) -> dict:
    del env
    entities = {"player": (float(state.player_x), float(state.player_y))}
    vis = np.asarray(state.visible_opponent_positions)
    ci = 0
    for i in range(vis.shape[0]):
        if float(vis[i, 0]) >= 0:
            entities[f"car_{ci}"] = (float(vis[i, 0]), float(vis[i, 1]))
            ci += 1
    return {"entities": entities, "scores": {}}
