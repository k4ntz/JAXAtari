import os
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


class TurmoilConstants(NamedTuple):
 pass

class SpawnState(NamedTuple):
    pass

# Game state container
class TurmoilState(NamedTuple):
    pass


class PlayerEntity(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    o: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray


class TurmoilObservation(NamedTuple):
    pass

class TurmoilInfo(NamedTuple):
    pass



# RENDER CONSTANTS
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    pass

# Load sprites once at module level



class JaxTurmoil(JaxEnvironment[TurmoilState, TurmoilObservation, TurmoilInfo, TurmoilConstants]):
    def __init__(self, consts: TurmoilConstants = None, reward_funcs: list[callable] = None):
        consts = consts or TurmoilConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]
        self.frame_stack_size = 4
        self.obs_size = 6 + 12 * 5 + 12 * 5 + 4 * 5 + 4 * 5 + 5 + 5 + 4
        self.renderer = TurmoilRenderer(self.consts)

   
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TurmoilState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: TurmoilState) -> TurmoilObservation:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TurmoilState, all_rewards: jnp.ndarray) -> TurmoilInfo:
        pass


    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: TurmoilState, state: TurmoilState):
        pass


    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: TurmoilState, state: TurmoilState) -> jnp.ndarray:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TurmoilState) -> bool:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[TurmoilObservation, TurmoilState]:
        """Initialize game state"""
        pass

    @partial(jax.jit, static_argnums=(0, ))
    def step(
        self, state: TurmoilState, action: chex.Array
    ) -> Tuple[TurmoilObservation, TurmoilState, float, bool, TurmoilInfo]:
        pass

class TurmoilRenderer(JAXGameRenderer):
    def __init__(self, consts: TurmoilConstants = None):
        self.consts = consts or TurmoilConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        pass