from encodings.punycode import digits
from functools import partial
import os

import chex
from flax import struct

from flax.nnx import state
import jax
import jax.numpy as jnp

from jaxatari.games.jax_pong import _create_wall_sprite
from jaxatari.renderers import JAXGameRenderer
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.rendering import jax_rendering_utils as render_utils
from typing import Tuple
import jaxatari.spaces as spaces

class CrazyClimberState(struct.PyTreeNode):
    key: chex.PRNGKey
    score: chex.Array
    step_counter: chex.Array

class CrazyClimberObservation(struct.PyTreeNode):
    pass

class CrazyClimberInfo(struct.PyTreeNode):
    pass

def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        {'name': 'life', 'type': 'life', 'file': 'life.npy'},
        {'name': 'life', 'type': 'life', 'pattern': 'player_{}.npy'},
    )

class CrazyClimberConstants(struct.PyTreeNode):
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0)),
    ASSET_CONFIG: tuple = _get_default_asset_config()

    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))

class JaxCrazyClimber(JaxEnvironment[CrazyClimberState, CrazyClimberObservation, CrazyClimberInfo, CrazyClimberConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN, Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT],
        dtype=jnp.int32,
    )

    def __init__(self, consts: CrazyClimberConstants = None):
        consts = consts or CrazyClimberConstants()
        super().__init__(consts)
        self.renderer = self.CrazyClimberRenderer(consts)

    def _score_and_reset(self, state: CrazyClimberState) -> CrazyClimberState:
        score_condition = jnp.array(True)

        score = jax.lax.cond(
            score_condition, # cond for score
            lambda s: s + jnp.array(100),
            lambda s: s,
            operand=state.score,
        )

        return state.replace(
            score=score,
        )

        initial_obs = self._get_observation(state)

        return initial_obs, state

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> (CrazyClimberObservation, CrazyClimberState):
        state_key, _step_key = jax.random.split(key)
        state = CrazyClimberState(
            key=state_key,
            score=jnp.array(0).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CrazyClimberState, action: chex.Array) -> (CrazyClimberObservation, CrazyClimberState, float, bool, CrazyClimberInfo):
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
         
        previous_state = state

        _, next_rng = jax.random.split(state.key)
        state = state.replace(key=next_rng)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info

    def render(self, state: CrazyClimberState) -> jnp.ndarray:
        return self.renderer.render(state)

    # TODO: Returntype needs to be altered to match actual implementation
    def action_space(self) -> spaces.Discrete:
        pass

    # TODO: Returntype needs to be altered to match actual implementation
    def observation_space(self) -> spaces.Dict:
        object_space = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "player": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
        })

    # TODO: Returntype needs to be altered to match actual implementation
    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        ) 

    def _get_observation(self, state: CrazyClimberState) -> CrazyClimberObservation:
        pass
    
    def obs_to_flat_array(self, obs: CrazyClimberObservation) -> jnp.ndarray:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: CrazyClimberState) -> CrazyClimberInfo:
        pass

    def _get_reward(self, previous_state: CrazyClimberState, state: CrazyClimberState) -> float:
        return 0

    def _get_done(self, state: CrazyClimberState) -> bool:
        pass

    class CrazyClimberRenderer(JAXGameRenderer):
        def __init__(self, consts: CrazyClimberConstants = None, config: render_utils.RendererConfig = None):
            self.consts = consts or CrazyClimberConstants()
            super().__init__(consts)
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
            self.jr = render_utils.JaxRenderingUtils(self.config)
            
            final_asset_config = list(self.consts.ASSET_CONFIG)

            sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "crazy_climber")

            (
                self.PALETTE,
                self.SHAPE_MASKS,
                self.BACKGROUND,
                self.COLOR_TO_ID,
                self.FLIP_OFFSETS
            ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        @partial(jax.jit, static_argnums=(0,))
        def render(self, state: CrazyClimberState) -> jnp.ndarray:
            raster = self.jr.create_object_raster(self.BACKGROUND)

            digits = self.jr.int_to_digits(state.score, max_digits=3)

            digit_masks = self.SHAPE_MASKS["digits"]

            is_player_single_digit = state.score < 10
            player_start_index = jax.lax.select(is_player_single_digit, 1, 0)
            player_num_to_render = jax.lax.select(is_player_single_digit, 1, 2)
            player_render_x = jax.lax.select(is_player_single_digit,
                                         120 + 16 // 2,
                                         120)

            raster = self.jr.render_label_selective(raster, player_render_x, 3, digits, digit_masks, player_start_index, player_num_to_render, spacing=16)

            return self.jr.render_from_palette(raster, self.PALETTE)