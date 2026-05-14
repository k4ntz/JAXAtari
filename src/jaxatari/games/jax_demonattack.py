from functools import partial
from typing import Tuple

import chex
import jax.lax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class DemonAttackConstants(struct.PyTreeNode):
    pass

class DemonAttackState(struct.PyTreeNode):
    pass

class DemonAttackObservation(struct.PyTreeNode):
    pass

class DemonAttackInfo(struct.PyTreeNode):
    pass

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