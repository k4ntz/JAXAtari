from functools import partial
from typing import Tuple

import chex
import jax.lax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'procedural'},
        {'name': 'player', 'type': 'procedural'},
        {'name': 'demon', 'type': 'procedural'},
        {'name': 'projectile_player', 'type': 'procedural'},
        {'name': 'projectile_demon', 'type': 'procedural'},
        {'name': 'score_digits', 'type': 'procedural'},
    )

class DemonAttackConstants(struct.PyTreeNode):
    # Static Configuration
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=2)
    MAX_DEMONS: int = struct.field(pytree_node=False, default=3)

    # Coordinates & Sizes
    PLAYER_Y: int = struct.field(pytree_node=False, default=184)
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 12))
    DEMON_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 12))
    LASER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(1, 6))
    BOMB_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(2, 4))

    # Boundaries
    PLAYER_MIN_X: int = struct.field(pytree_node=False, default=16)
    PLAYER_MAX_X: int = struct.field(pytree_node=False, default=136)

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)

class DemonAttackState(struct.PyTreeNode):
    player_x: chex.Array
    laser_x: chex.Array
    laser_y: chex.Array
    laser_active: chex.Array

    demons_x: chex.Array
    demons_y: chex.Array  # Shape: (MAX_DEMONS,)
    demons_dir: chex.Array  # Shape: (MAX_DEMONS,) 1 for right, -1 for left
    demons_alive: chex.Array  # Shape: (MAX_DEMONS,) bool

    bomb_x: chex.Array
    bomb_y: chex.Array
    bomb_active: chex.Array

    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array
    key: chex.PRNGKey

class DemonAttackObservation(struct.PyTreeNode):
    player: ObjectObservation
    demons: ObjectObservation
    laser: ObjectObservation
    bomb: ObjectObservation
    score: jnp.ndarray
    lives: jnp.ndarray

class DemonAttackInfo(struct.PyTreeNode):
    time: jnp.ndarray

class JaxDemonAttack(JaxEnvironment[DemonAttackState, DemonAttackObservation, DemonAttackInfo, DemonAttackConstants]):

    def __init__(self, consts: DemonAttackConstants = None):
        consts = consts or DemonAttackConstants()
        super().__init__(consts)

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[DemonAttackObservation, DemonAttackState]:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DemonAttackState, action: chex.Array) -> Tuple[DemonAttackObservation, DemonAttackState, float, bool, DemonAttackInfo]:
        pass

    def render(self, state: DemonAttackState) -> jnp.ndarray:
        pass

    def _get_observation(self, state: DemonAttackState):
        pass

    def action_space(self) -> spaces.Discrete:
        pass

    def observation_space(self) -> spaces.Dict:
       pass

    def image_space(self) -> spaces.Box:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: DemonAttackState, ) -> DemonAttackInfo:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: DemonAttackState, state: DemonAttackState):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: DemonAttackState) -> bool:
        pass


class DemonAttackRenderer(JAXGameRenderer):
    def __init__(self, consts: DemonAttackConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or DemonAttackConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        pass