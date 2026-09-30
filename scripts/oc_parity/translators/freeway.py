"""Translate OCAtari Freeway object snapshots into JAXAtari FreewayState."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games.jax_freeway import FreewayState, JaxFreeway

from .base import cast_like, collect_category, objects_as_dicts
from .registry import print_disclaimers


def _player_chicken(objs: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    chickens = collect_category(objs, "Chicken")
    if not chickens:
        return None
    # Q1=A: leftmost / x≈44 matches JAX chicken_x.
    return min(chickens, key=lambda o: float(o.get("x", 0)))


def _player_score(objs: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    scores = collect_category(objs, "Score", require_visible=False)
    scores = [s for s in scores if int(s.get("value", -1)) >= 0 or int(s.get("w", 0)) > 0]
    if not scores:
        return None
    # Q3=A: smaller x = left HUD (P1).
    return min(scores, key=lambda o: float(o.get("x", 0)))


def oc_frame_to_freeway_state(
    env: JaxFreeway,
    objects: Sequence[Any],
    *,
    seed: int = 0,
    frame_index: int = 0,
    print_assumptions: bool = False,
    **_ignored,
) -> FreewayState:
    """Reset JAX Freeway, then overlay OC chicken / cars / score."""
    del frame_index
    if print_assumptions:
        print_disclaimers("freeway")

    objs = objects_as_dicts(objects)
    _obs, state = env.reset(jax.random.PRNGKey(int(seed)))
    del _obs

    updates: dict = {}
    chicken = _player_chicken(objs)
    if chicken is not None:
        cy = cast_like(int(chicken["y"]), state.chicken_y)
        updates["chicken_y"] = cy
        updates["best_chicken_y"] = cast_like(cy, state.best_chicken_y)  # Q4=A

    # Q2=A verified: JAX lane ys are ascending [27..171]. Sort OC cars by y.
    cars = sorted(collect_category(objs, "Car"), key=lambda o: float(o.get("y", 0)))
    if cars:
        car_arr = np.array(state.cars, dtype=np.dtype(np.asarray(state.cars).dtype)).copy()
        n = min(len(cars), car_arr.shape[0])
        for i in range(n):
            car_arr[i, 0] = int(cars[i]["x"])
            # Keep JAX lane y (moving DOF is x).
        updates["cars"] = cast_like(car_arr, state.cars)

    score_obj = _player_score(objs)
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        updates["score"] = cast_like(int(score_obj["value"]), state.score)

    if updates:
        state = state.replace(**updates)
    return state


def extract_oc_compare_entities(objects: Sequence[Any], **_kwargs) -> dict:
    objs = objects_as_dicts(objects)
    entities: dict = {}
    scores: dict = {}

    chicken = _player_chicken(objs)
    if chicken is not None:
        entities["chicken"] = (float(chicken["x"]), float(chicken["y"]))

    cars = sorted(collect_category(objs, "Car"), key=lambda o: float(o.get("y", 0)))
    for i, car in enumerate(cars[:10]):
        entities[f"car_{i}"] = (float(car["x"]), float(car["y"]))

    score_obj = _player_score(objs)
    if score_obj is not None and int(score_obj.get("value", -1)) >= 0:
        scores["score"] = float(score_obj["value"])

    return {"entities": entities, "scores": scores}


def extract_jax_compare_entities(state: FreewayState, env: JaxFreeway) -> dict:
    entities = {
        "chicken": (float(env.consts.chicken_x), float(state.chicken_y)),
    }
    cars = np.asarray(state.cars)
    for i in range(cars.shape[0]):
        entities[f"car_{i}"] = (float(cars[i, 0]), float(cars[i, 1]))
    scores = {"score": float(state.score)}
    return {"entities": entities, "scores": scores}
