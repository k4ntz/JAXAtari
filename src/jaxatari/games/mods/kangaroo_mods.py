import functools
from typing import Any, Dict, Tuple, Union


import chex
import jax
from jaxatari.games.jax_kangaroo import KangarooState

from jaxatari.wrappers import GymnaxWrapper

class DisableThreadsWrapper(GymnaxWrapper):
    """Disable enemies in the environment."""
    @functools.partial(jax.jit, static_argnums=(0,))
    def disable_enemies(self, state: KangarooState) -> KangarooState:
        _, reset_state = self._env.reset()
        new_level = state.level._replace(
            falling_coco_position = reset_state.level.falling_coco_position,
            falling_coco_dropping = reset_state.level.falling_coco_dropping,
            falling_coco_counter = reset_state.level.falling_coco_counter,
            falling_coco_skip_update = reset_state.level.falling_coco_skip_update,
            monkey_states = reset_state.level.monkey_states,
            monkey_positions = reset_state.level.monkey_positions,
            monkey_throw_timers = reset_state.level.monkey_throw_timers,
            coco_positions = reset_state.level.coco_positions,
            coco_states = reset_state.level.coco_states,
        )
        new_state = state._replace(
            level = new_level, 
        )
        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: KangarooState, action: Union[int, float]) -> Tuple[chex.Array, KangarooState, float, bool, Dict[Any, Any]]:
        new_obs, state, reward, done, info = self._env.step(state, action)
        new_state = self.disable_enemies(state)

        return new_obs, new_state, reward, done, info