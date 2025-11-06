import gymnasium as gym
import ale_py
from gymnasium.utils import play
import matplotlib.pyplot as plt

import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action




#------------------------named Tuples---------------------------
class BattlezoneConstants(NamedTuple):
    WIDTH: int = 160  #rgb_array size
    HEIGHT: int = 210
    SCORE_COLOR: Tuple[int, int, int] = (236, 236, 236)#(45, 129, 105) but we need to change pallette first
    WALL_TOP_Y: int = 24
    WALL_TOP_HEIGHT: int = 10
    WALL_BOTTOM_Y: int = 194
    WALL_BOTTOM_HEIGHT: int = 16


# immutable state container
class BattlezoneState(NamedTuple):
    score: chex.Array
    step_counter: chex.Array


class BattlezoneObservation(NamedTuple):
    score: jnp.ndarray


class BattlezoneInfo(NamedTuple):
    time: jnp.ndarray


#----------------------------Battlezone Environment------------------------
class JaxBattlezone(JaxEnvironment[BattlezoneState, BattlezoneObservation, BattlezoneInfo, BattlezoneConstants]):
    def __init__(self, consts: BattlezoneConstants = None, reward_funcs: list[callable]=None):
        self.consts = consts or BattlezoneConstants()
        super().__init__(consts)
        self.renderer = BattlezoneRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set=[ #from https://ale.farama.org/environments/battle_zone/
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
            Action.DOWNLEFTFIRE]
        self.obs_size = 3*4+1+1  #?? change later from pong currently

    def _player_step(self, state: BattlezoneState, action: chex.Array) -> BattlezoneState:
        up = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)



        return BattlezoneState(
            score=jnp.array(0),
            step_counter=jnp.array(0),
        )


    def reset(self, key=None) -> Tuple[BattlezoneObservation, BattlezoneState]:
        state = BattlezoneState(
            score=jnp.array(0),
            step_counter=jnp.array(0),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BattlezoneState, action: chex.Array) -> Tuple[BattlezoneObservation, BattlezoneState,\
                float, bool, BattlezoneInfo]:
        previous_state = state
        state.step_counter +=1
        state = self._player_step(state, action)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: BattlezoneState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: BattlezoneState):

        return BattlezoneObservation(
            score=state.score,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: BattlezoneObservation) -> jnp.ndarray:
           return jnp.concatenate([
               #obs.player.x.flatten(),
               #obs.player.y.flatten(),
               #etc.
               obs.score.flatten(),
            ]
           )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({  #from pong currently
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "score": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BattlezoneState, ) -> BattlezoneInfo:
        return BattlezoneInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BattlezoneState, state: BattlezoneState):
        return state.score - previous_state.score  #temporary intuition change later

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BattlezoneState) -> bool:
        return False #if lives are < 0 change later



#-------------------------------------renderer-------------------------------------
class BattlezoneRenderer(JAXGameRenderer):
    def __init__(self, consts: BattlezoneConstants = None):
        super().__init__()
        self.consts = consts or BattlezoneConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            # downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        # 1. Create procedural assets for both walls
        wall_sprite_top = self._create_wall_sprite(self.consts.WALL_TOP_HEIGHT)
        wall_sprite_bottom = self._create_wall_sprite(self.consts.WALL_BOTTOM_HEIGHT)

        # 2. Update asset config to include both walls
        asset_config = self._get_asset_config(wall_sprite_top, wall_sprite_bottom)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/pong" #change later when we have sprites

        # 3. Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    def _create_wall_sprite(self, height: int) -> jnp.ndarray:
        """Procedurally creates an RGBA sprite for a wall of given height."""
        wall_color_rgba = (0, 0, 0, 255)  # black
        wall_shape = (height, self.consts.WIDTH, 4)
        wall_sprite = jnp.tile(jnp.array(wall_color_rgba, dtype=jnp.uint8), (*wall_shape[:2], 1))
        return wall_sprite

    def _get_asset_config(self, wall_sprite_top: jnp.ndarray, wall_sprite_bottom: jnp.ndarray) -> list:
        """Returns the declarative manifest of all assets for the game, including both wall sprites."""
        return [ #change later when we have assets
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'player', 'type': 'single', 'file': 'player.npy'},
            {'name': 'enemy', 'type': 'single', 'file': 'enemy.npy'},
            {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
            {'name': 'player_digits', 'type': 'digits', 'pattern': 'player_score_{}.npy'},
            {'name': 'enemy_digits', 'type': 'digits', 'pattern': 'enemy_score_{}.npy'},
            # Add the procedurally created sprites to the manifest
            {'name': 'wall_top', 'type': 'procedural', 'data': wall_sprite_top},
            {'name': 'wall_bottom', 'type': 'procedural', 'data': wall_sprite_bottom},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # --- Stamp Walls and Score (using the same color/ID) ---
        score_color_tuple = self.consts.SCORE_COLOR  # (236, 236, 236)
        score_id = self.COLOR_TO_ID[score_color_tuple]

        # Draw walls (using separate sprites for top and bottom)
        raster = self.jr.render_at(raster, 0, self.consts.WALL_TOP_Y, self.SHAPE_MASKS["wall_top"])
        raster = self.jr.render_at(raster, 0, self.consts.WALL_BOTTOM_Y, self.SHAPE_MASKS["wall_bottom"])


        #---------------------------player score--------------------change later cuz we have 3 digits
        # Stamp Score using the label utility
        player_digits = self.jr.int_to_digits(state.score, max_digits=2)
        # Note: The logic for single/double digits is complex for a jitted function.
        player_digit_masks = self.SHAPE_MASKS["player_digits"]  # Assumes single color

        is_player_single_digit = state.score < 10
        player_start_index = jax.lax.select(is_player_single_digit, 1, 0)
        player_num_to_render = jax.lax.select(is_player_single_digit, 1, 2)
        player_render_x = jax.lax.select(is_player_single_digit,
                                         120 + 16 // 2,
                                         120)

        raster = self.jr.render_label_selective(raster, player_render_x, 3, player_digits, player_digit_masks,
                                                player_start_index, player_num_to_render, spacing=16)
        #--------------------------------------------------------------------


        return self.jr.render_from_palette(raster, self.PALETTE)






#-----------------------------delete all following later----------------------------------------
def try_gym_battlezone():
    gym.register_envs(ale_py)
    env = gym.make("ALE/BattleZone-v5", render_mode="rgb_array", frameskip=1)
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    play.play(env, zoom=3, fps=60)  # , keys_to_action=keys_to_action)
    env.close()

def try_gym_battlezone_pixel():
    gym.register_envs(ale_py)
    env = gym.make("ALE/BattleZone-v5", render_mode="rgb_array", frameskip=1)
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    stepsize = 100
    for i in range(1000):
        action = 3
        obs, reward, terminated, truncated, info = env.step(action)
        if i%stepsize==0:
            im = plt.imshow(obs, interpolation='none', aspect='auto')
            plt.show()
    env.close()


if __name__ == "__main__":
    env = JaxBattlezone()
    initial_obs, state = env.reset()
    obs, state, env_reward, done, info = env.step(state, 0)