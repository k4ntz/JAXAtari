import os
import chex
import jax.lax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

from jax import random as jrandom
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvObs, EnvState
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces
class BoxingConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210

class BoxingState(NamedTuple):
    score: chex.Array

class EntityPositions(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray

class BoxingObservation(NamedTuple):
    player: EntityPositions
class BoxingInfo(NamedTuple):
    time: jnp.ndarray

class JaxBoxing(JaxEnvironment[BoxingState, BoxingObservation, BoxingInfo, BoxingConstants]):
    def __init__(self, consts: BoxingConstants = None, reward_funcs: list[callable] = None):
        consts = consts or BoxingConstants()
        super().__init__(consts)
        self.renderer = BoxingRenderer(consts)
        if reward_funcs is not None:
            self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP
        ]

    def reset(self, key=None) -> Tuple[BoxingObservation, BoxingState]:
        state = BoxingState(score=jnp.array(1))
        initial_obs = self._get_observation(state)
        return initial_obs, state

    def step(self, state: BoxingState, action: chex.Array) -> Tuple[BoxingObservation, BoxingState, float, bool, BoxingInfo]:
        new_state = BoxingState(score=state.score + 1)
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_rewards(new_state, action)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)
        return observation, new_state, env_reward, done, info

    def render(self, state: BoxingState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_done(self, state: EnvState) -> bool:
        return False
    def _get_reward(self, previous_state: EnvState, state: EnvState) -> float:
        return 0.0
    def _get_observation(self, state: BoxingState):
        return BoxingObservation(
            player=EntityPositions(
                x=jnp.array([0.0]),  # Placeholder for player x position
                y=jnp.array([0.0])   # Placeholder for player y position
            )
        )
    def _get_info(self, state: BoxingState, all_rewards: jnp.ndarray) -> BoxingInfo:
        return BoxingInfo(
            time=jnp.array([0])  # Placeholder for time information
        )
    def _get_all_rewards(self, state: BoxingState, action: chex.Array) -> jnp.ndarray:
        return jnp.array([0.0])
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

class BoxingRenderer(JAXGameRenderer):
    def __init__(self, consts: BoxingConstants = None):
        self.consts = consts or BoxingConstants()
        (
            self.SPRITE_BG,
        ) = self.load_sprites()
    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        background = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/boxing/background.npy"))
        SPRITE_BG = jnp.expand_dims(background, axis=0)

        return (SPRITE_BG,)
    def render(self, state):
        raster = jr.create_initial_frame(width=160, height=210)
        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0,0, frame_bg)
        return raster

