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

    # TODO: Returntype needs to be altered to match actual implementation
    def observation_space(self) -> spaces.Dict:
        pass

    # TODO: Returntype needs to be altered to match actual implementation
    def image_space(self) -> spaces.Box:
        pass 

    def _get_observation(self, state: CrazyClimberState) -> CrazyClimberObservation:
        pass
    
    def obs_to_flat_array(self, obs: CrazyClimberObservation) -> jnp.ndarray:
        pass

    def _get_info(self, state: CrazyClimberState) -> CrazyClimberInfo:
        pass

    def _get_reward(self, previous_state: CrazyClimberState, state: CrazyClimberState) -> float:
        pass

    def _get_done(self, state: CrazyClimberState) -> bool:
        pass