from flax import struct

import jax
import jax.numpy as jnp

from jaxatari.spaces import spaces
from jaxatari.environment import JaxEnvironment

class CrazyClimberState(struct.PyTreeNode):
    pass

class CrazyClimberObservation(struct.PyTreeNode):
    pass

class CrazyClimberInfo(struct.PyTreeNode):
    pass

class CrazyClimberConstants(struct.PyTreeNode):
    pass

class JaxCrazyClimber(JaxEnvironment[CrazyClimberState, CrazyClimberObservation, CrazyClimberInfo, CrazyClimberConstants]):
    def __init__(self, consts: CrazyClimberConstants):
        super().__init__(consts)

    def reset(self, key: jax.random.PRNGKey) -> (CrazyClimberObservation, CrazyClimberState):
        pass

    def step(self, state: CrazyClimberState, action: int) -> (CrazyClimberObservation, CrazyClimberState, float, bool, CrazyClimberInfo):
        pass

    def render(self, state: CrazyClimberState) -> jnp.ndarray:
        pass

    # TODO: Returntype needs to be altered to match actual implementation
    def action_space(self) -> spaces.Discrete:
        pass