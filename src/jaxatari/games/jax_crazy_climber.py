from functools import partial

from flax import struct

import jax
import jax.numpy as jnp

from jaxatari.spaces import spaces
from jaxatari.environment import JaxEnvironment
from jaxatari.rendering import jax_rendering_utils as render_utils
from typing import Tuple
import jaxatari.spaces as spaces

class CrazyClimberState(struct.PyTreeNode):
    pass

class CrazyClimberObservation(struct.PyTreeNode):
    pass

class CrazyClimberInfo(struct.PyTreeNode):
    pass

class CrazyClimberConstants(struct.PyTreeNode):
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))
    PLAYER_X: int = struct.field(pytree_node=False, default=140)
    PLAYER_Y: int = struct.field(pytree_node=False, default=100)

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

    class CrazyClimberRenderer(JAXGameRenderer):
        def __init__(self, consts: CrazyClimberConstants = None, config: render_utils.RendererConfig = None):
            super().__init__(consts)
            self.consts = consts or CrazyClimberConstants()

            self.jr = render_utils.JAXRenderingUtils(self.config)

        @partial(jax.jit, static_argnums=(0,))
        def render(self, state: CrazyClimberState) -> jnp.ndarray:
            raster = self.jr.create_object_raster(self.BACKGROUND)

            player_mask = self.SHAPE_MASKS["player"]
            raster = self.jr.render_at(
                raster,
                self.consts.PLAYER_X,
                self.consts.PLAYER_Y, 
                jnp.round(state.player_y).astype(jnp.int32),
                player_mask,
            )