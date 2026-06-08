from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class JamesBondConstants(struct.PyTreeNode):
    pass


@struct.dataclass
class JamesBondState:
    pass


@struct.dataclass
class JamesBondObservation:
    pass


@struct.dataclass
class JamesBondInfo:
    pass


class JaxJamesBond(
    JaxEnvironment[JamesBondState, JamesBondObservation, JamesBondInfo, JamesBondConstants]
):
    # Compact agent action indices map to these ALE-style actions.
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP],
        dtype=jnp.int32,
    )

    def __init__(self, consts: JamesBondConstants = None):
        consts = consts or JamesBondConstants()
        super().__init__(consts)

    def reset(
        self, key: chex.PRNGKey = jax.random.PRNGKey(0)
    ) -> Tuple[JamesBondObservation, JamesBondState]:
        raise NotImplementedError("JamesBond reset is added in a later commit")

    def step(
        self, state: JamesBondState, action: chex.Array
    ) -> Tuple[JamesBondObservation, JamesBondState, chex.Array, chex.Array, JamesBondInfo]:
        raise NotImplementedError("JamesBond step is added in a later commit")

    def render(self, state: JamesBondState) -> jnp.ndarray:
        raise NotImplementedError("JamesBond renderer is added in a later commit")

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({})

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)
