import gymnasium as gym
import ale_py
from gymnasium.utils import play

"""
Collecting necessary information about BattleZone.
===============================================================

Entities:
    - Player
    - Enemies
        - Tank
        - Flying Saucer
        - Supertank
        - Fighter
    - Projectiles

    
What do we need to research about each enemy (+player)?
- movement speed
- turn speed
- firing rate
- movement patterns
- score points worth
- sprite (extract from gym version)
- projectile speed and range

- how many enemies can be in game at once
- how does acceleration work? is there acceleration or do we just go straight into max speed? are there jerk/snap/crackle/pop?
- is there friction?
- is there a world size? 
- is there friendly fire for enemies?
- what is the radar range?
"""
import matplotlib.pyplot as plt
from enum import IntEnum, unique

import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnemyType




#------------------------named Tuples---------------------------
class BattlezoneConstants(NamedTuple):
    WIDTH: int = 160  #rgb_array size
    HEIGHT: int = 210
    SCORE_COLOR: Tuple[int, int, int] = (26,102,26)#(45, 129, 105) but we need to change pallette first
    WALL_TOP_Y: int = 0     #correct
    WALL_TOP_HEIGHT: int = 36   #correct
    WALL_BOTTOM_Y: int = 177    #correct
    WALL_BOTTOM_HEIGHT: int = 33    #correct
    TANK_SPRITE_POS_X: int = 43
    TANK_SPRITE_POS_Y: int = 140
    CHAINS_POS_Y:int = 158
    CHAINS_L_POS_X:int = 19
    CHAINS_R_POS_X: int = 109
    CHAINS_COL_1: Tuple[int, int, int] = (111,111,111)
    CHAINS_COL_2: Tuple[int, int, int] = (74, 74, 74)
    MOUNTAINS_Y: int = 36
    GRASS_BACK_Y: int = 95
    GRASS_FRONT_Y: int = 137


# immutable state container
class BattlezoneState(NamedTuple):
    score: chex.Array
    step_counter: chex.Array
    chains_l_anim_counter: chex.Array
    chains_r_anim_counter: chex.Array
    mountains_anim_counter:chex.Array
    grass_anim_counter:chex.Array
    enemies: chex.Array


class BattlezoneObservation(NamedTuple):
    score: jnp.ndarray


class BattlezoneInfo(NamedTuple):
    time: jnp.ndarray


class Enemy(NamedTuple):
    x: chex.Array
    z: chex.Array
    distance: chex.Array
    enemy_type: chex.Array
    orientation_angle: chex.Array


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
        up = jnp.logical_or(action==Action.UP,action==Action.UPFIRE)
        down = jnp.logical_or(action==Action.DOWN,action==Action.DOWNFIRE)
        right = jnp.logical_or(action==Action.RIGHT,action==Action.RIGHTFIRE)
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        upLeft = jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE)
        upRight = jnp.logical_or(action==Action.UPRIGHT,action==Action.UPRIGHTFIRE)
        downLeft = jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNLEFTFIRE)
        downRight = jnp.logical_or(action==Action.DOWNRIGHT,action==Action.DOWNRIGHTFIRE)
        fire = jnp.any(jnp.stack([action == Action.FIRE, action == Action.LEFTFIRE,
                                 action == Action.UPFIRE, action == Action.UPLEFTFIRE,
                                 action == Action.DOWNFIRE, action == Action.DOWNLEFTFIRE]), axis=0)
        #--------------------anims--------------------
        chain_r_offset = (-jnp.where(jnp.any(jnp.stack([upLeft, up, left])), 1.0, 0.0)
                          +jnp.where(jnp.any(jnp.stack([right, down, downRight])), 1.0, 0.0)
                          -jnp.where(upRight, 0.7, 0.0) + jnp.where(downLeft, 0.7, 0.0))
                            #i love magic numbers
        chain_l_offset = (-jnp.where(jnp.any(jnp.stack([upRight, up, right])), 1.0, 0.0)
                          + jnp.where(jnp.any(jnp.stack([left, down, downLeft])), 1.0, 0.0)
                          - jnp.where(upLeft, 0.7, 0.0) + jnp.where(downRight, 0.7, 0.0))
        mountains_offset = (jnp.where(jnp.any(jnp.stack([left, upLeft, downRight])), 1.0, 0.0)
                            -jnp.where(jnp.any(jnp.stack([right, upRight, downLeft])), 1.0, 0.0))
        grass_offset = (jnp.where(jnp.any(jnp.stack([up, upLeft, upRight])), 1.0, 0.0)
                        - jnp.where(jnp.any(jnp.stack([down, downRight, downLeft])), 1.0, 0.0))
        # --------------------enemies--------------------
        # updating enemy x and z


        return BattlezoneState(
            score=state.step_counter,
            step_counter=state.step_counter,
            enemies=state.enemies,
            chains_l_anim_counter=(state.chains_l_anim_counter + chain_l_offset)%32,
            chains_r_anim_counter=(state.chains_r_anim_counter + chain_r_offset)%32,
            mountains_anim_counter=(state.mountains_anim_counter + mountains_offset)%160,
            grass_anim_counter= (state.grass_anim_counter + grass_offset)%30
        )

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: BattlezoneState) -> BattlezoneState:
        update_all_enemies = jax.vmap(self._single_enemy_update, in_axes=(None, 0))(state, state.enemies)
        return state._replace(enemies=update_all_enemies)




    def reset(self, key=None) -> Tuple[BattlezoneObservation, BattlezoneState]:
        state = BattlezoneState(
            score=jnp.array(0),
            step_counter=jnp.array(0),
            chains_l_anim_counter=jnp.array(0),
            chains_r_anim_counter=jnp.array(0),
            mountains_anim_counter=jnp.array(0),
            grass_anim_counter=jnp.array(0),
            enemies=Enemy(
                x=jnp.empty((0,), dtype=jnp.float32),
                z=jnp.empty((0,), dtype=jnp.float32),
                distance=jnp.empty((0,), dtype=jnp.float32),
                enemy_type=jnp.empty((0,), dtype=jnp.float32),
                orientation_angle=jnp.empty((0,), dtype=jnp.float32),
            )
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BattlezoneState, action: chex.Array) -> Tuple[BattlezoneObservation, BattlezoneState,\
                float, bool, BattlezoneInfo]:
        previous_state = state
        new_state = state._replace(step_counter=state.step_counter+1)
        new_state = self._player_step(new_state, action)
        new_state = self._enemy_step(new_state)

        done = self._get_done(new_state)
        env_reward = self._get_reward(previous_state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def _single_enemy_update(self, state: BattlezoneState, enemy: Enemy) -> Enemy:
        new_distance = jnp.sqrt(enemy.x ** 2 + enemy.z ** 2)
        #Room for distance specific actions
        return enemy._replace(distance=new_distance)


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
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/battlezone" #change later when we have sprites

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
            {'name': 'tank', 'type': 'single', 'file': 'tank.npy'},
            {'name': 'chainsLeft', 'type': 'single', 'file': 'chainsLeft.npy'},
            {'name': 'chainsRight', 'type': 'single', 'file': 'chainsRight.npy'},
            {'name': 'mountains', 'type': 'single', 'file': 'mountains.npy'},
            {'name': 'grass_front_1', 'type': 'single', 'file': 'grass_front_1.npy'},
            {'name': 'grass_front_2', 'type': 'single', 'file': 'grass_front_2.npy'},
            {'name': 'grass_back', 'type': 'single', 'file': 'grass_back_1.npy'},
            {'name': 'player_digits', 'type': 'digits', 'pattern': 'player_score_{}.npy'},#todo change
            # Add the procedurally created sprites to the manifest
            {'name': 'wall_top', 'type': 'procedural', 'data': wall_sprite_top},
            {'name': 'wall_bottom', 'type': 'procedural', 'data': wall_sprite_bottom},
        ]

    def _scroll_chain_colors(self, chainMask, scroll):
        h, w = jnp.shape(chainMask)
        # create color pattern
        pattern = jnp.arange((h + 2) // 3) % 2 *3 # create enough 0/1 pairs
        row_values = jnp.repeat(pattern, 3)[:h]  # ensure exactly 19 rows
        # replace with actual color indices
        color_id1 = self.COLOR_TO_ID[self.consts.CHAINS_COL_1]
        color_id2 = self.COLOR_TO_ID[self.consts.CHAINS_COL_2]
        jnp.where(row_values == 0, color_id1, color_id2)
        # create and scroll array
        arr = jnp.broadcast_to(row_values[:, None], (h, w))
        scrolled = jnp.roll(arr, shift=scroll, axis=0)


        return jnp.where(chainMask==255, chainMask, scrolled)


    def _scroll_grass_back(self, grass_mask, scroll):
        grass_fill_color_id = grass_mask[0, 0]
        grass_back_shift = jnp.floor_divide(scroll, 2) % 4
        grass_back_scrolled_mask = jnp.roll(grass_mask, shift=grass_back_shift, axis=0)
        mask = jnp.arange(grass_back_scrolled_mask.shape[0]) < grass_back_shift
        mask = mask[:, None]  # broadcast across columns

        return jnp.where(mask, grass_fill_color_id, grass_back_scrolled_mask)


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        #-----------------background
        raster = self.jr.create_object_raster(self.BACKGROUND)

        mountains_mask = self.SHAPE_MASKS["mountains"]
        mountains_mask_scrolled = jnp.roll(mountains_mask, shift=state.mountains_anim_counter, axis=1)
        raster = self.jr.render_at(raster, 0,
                                   self.consts.MOUNTAINS_Y, mountains_mask_scrolled)

        grass_front_mask = jnp.where((state.grass_anim_counter % 30) < 15, self.SHAPE_MASKS["grass_front_1"],
                                      self.SHAPE_MASKS["grass_front_2"])
        raster = self.jr.render_at(raster, 0,
                                   self.consts.GRASS_FRONT_Y, grass_front_mask)

        grass_back_mask = self._scroll_grass_back(self.SHAPE_MASKS["grass_back"], state.grass_anim_counter)
        raster = self.jr.render_at(raster, 0,
                                   self.consts.GRASS_BACK_Y, grass_back_mask)


        tank_mask = self.SHAPE_MASKS["tank"]
        raster = self.jr.render_at(raster, self.consts.TANK_SPRITE_POS_X,
                                   self.consts.TANK_SPRITE_POS_Y, tank_mask)

        #--------------chains---------
        chains_l_mask = self.SHAPE_MASKS["chainsLeft"]
        color_shifted_chain_l = self._scroll_chain_colors(chains_l_mask, state.chains_l_anim_counter)
        raster = self.jr.render_at(raster, self.consts.CHAINS_L_POS_X,
                                   self.consts.CHAINS_POS_Y, color_shifted_chain_l)

        chains_r_mask = self.SHAPE_MASKS["chainsRight"]
        color_shifted_chain_r = self._scroll_chain_colors(chains_r_mask, state.chains_r_anim_counter)
        raster = self.jr.render_at(raster, self.consts.CHAINS_R_POS_X,
                                   self.consts.CHAINS_POS_Y, color_shifted_chain_r)


        # --- Stamp Walls and Score (using the same color/ID) ---
        score_color_tuple = self.consts.SCORE_COLOR  # (236, 236, 236)
        score_id = self.COLOR_TO_ID[score_color_tuple]


        #---------------------------player score--------------------change later cuz we have 3 digits
        # Stamp Score using the label utility
        score = jnp.array(state.grass_anim_counter, int) #:)
        player_digits = self.jr.int_to_digits(score, max_digits=2)
        # Note: The logic for single/double digits is complex for a jitted function.
        player_digit_masks = self.SHAPE_MASKS["player_digits"]  # Assumes single color

        is_player_single_digit = score < 10
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
    stepsize = 5
    for i in range(1000):
        action = 2
        obs, reward, terminated, truncated, info = env.step(action)
        #extract_sprite(obs, [[111,111,111], [74,74,74]])
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "observation.npy")
        np.save(save_path, obs)
        if i%stepsize==0 and i >= 500:
            im = plt.imshow(obs, interpolation='none', aspect='auto')
            plt.show()
    env.close()


import numpy as np


def extract_sprite(rgb_array, color_list, filename="output_rgba.npy"):
    h, w = rgb_array.shape[:2]
    if rgb_array.shape[-1] == 3: #we need rgba
        rgba = np.concatenate([rgb_array, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    else:
        rgba = rgb_array.copy()

    # Create a mask for all pixels matching any color in color_list
    mask = np.zeros((h, w), dtype=bool)
    for color in color_list:
        color = np.array(color)
        mask |= np.all(rgba[..., :3] == color, axis=-1)

    # Set alpha=0 where mask is False
    #rgba[~mask] = np.zeros((4), dtype=np.uint8)
    rgba[~mask, 3] = 0

    # Find the bounding box of the mask
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        print("No matching colors found.")
        return np.zeros((0, 0, 4), dtype=np.uint8)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Crop the region
    cropped = rgba[y_min:y_max + 1, x_min:x_max + 1]

    # Print the starting position
    print(f"New array begins at position (row={y_min}, col={x_min}) in the original array.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, filename)
    np.save(save_path, cropped)
    return cropped


if __name__ == "__main__":
    #env = JaxBattlezone()
    #initial_obs, state = env.reset()
    #for i in range(100):
        #obs, state, env_reward, done, info = env.step(state, 0)



    try_gym_battlezone()