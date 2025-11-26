import gymnasium as gym
import ale_py
from gymnasium.utils import play
from pygame.examples.scroll import zoom_factor

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
- how does acceleration work? is there acceleration or do we just go straight into max speed? are there jerk/snap/crackle/pop? -> no accel
- is there friction? no
- is there a world size? 
- is there friendly fire for enemies? yes
- what is the radar range?


------------------------------------------------------------------------

## World
- functionally toroidal, we work with signed 8bit ints for x and z coordinates (-128 to 127)
- coordinate system: player is always at (0,0), x increases to the right, z increases forward away from player, y is height on screen

## Misc
- first enemy in a game is always a tank which spawns at z=60.5547 x=6.8047 facing negative x
- at max two enemies in world at once
- there is always one enemy present. if the player kills the only enemy, a new one spawns immediately

## Player
- hitbox: a rectangle around the origin: (x: -3, z: 0) to (x: 3, z: 6)
- move speed: 0.25 units per frame
- turn speed: 2pi/270 rad/frame (270 frames for a full turn)
-

## Tank
- behaviour:
    - shoots when distance to player is 29.09 units
- score points: 1000
- move speed: 0.125 units per frame
- turn speed: 2pi/2048 rad/frame (512 frames for a quarter turn)

## Flying Saucer

## Supertank

## Fighter Jet

## Projectile
- speed: 0.5 units per frame

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
    RADAR_ROTATION_SPEED:float = -0.1
    RADAR_CENTER_X:int = 80
    RADAR_CENTER_Y:int = 18
    RADAR_RADIUS:int = 10
    RADAR_MAX_SCAN_RADIUS: int = 70
    RADAR_COLOR_1: Tuple[int, int, int] = (111,210,111)
    RADAR_COLOR_2: Tuple[int, int, int] = (236,236,236)
    LIFE_SCORE_COLOR: Tuple[int, int, int] = (45,129,105)
    LIFE_POS_X:int = 64
    LIFE_POS_Y:int = 189
    LIFE_X_OFFSET:int = 8
    SCORE_POS_X:int = 89
    SCORE_POS_Y:int = 179
    PLAYER_ROTATION_SPEED:float = 2*jnp.pi/270
    PLAYER_SPEED:float = 0.25

# immutable state container
class BattlezoneState(NamedTuple):
    score: chex.Array
    life: chex.Array
    step_counter: chex.Array
    chains_l_anim_counter: chex.Array
    chains_r_anim_counter: chex.Array
    mountains_anim_counter:chex.Array
    grass_anim_counter:chex.Array
    radar_rotation_counter:chex.Array
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
        super().__init__(self.consts)
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
        self.obs_size = 3*4+1+1  #?? TODO: change later from pong currently

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


        return BattlezoneState(
            score=state.score,
            life=state.life,
            step_counter=state.step_counter,
            enemies=state.enemies,
            chains_l_anim_counter=(state.chains_l_anim_counter + chain_l_offset)%32,
            chains_r_anim_counter=(state.chains_r_anim_counter + chain_r_offset)%32,
            mountains_anim_counter=(state.mountains_anim_counter + mountains_offset)%160,
            grass_anim_counter= (state.grass_anim_counter + grass_offset)%30,
            radar_rotation_counter=state.radar_rotation_counter
        )

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: BattlezoneState, action: chex.Array) -> BattlezoneState:
        #jax.debug.print("Action: {}",action)
        noop = (action==Action.NOOP)
        up = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)
        right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        upLeft = jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE)
        upRight = jnp.logical_or(action == Action.UPRIGHT, action == Action.UPRIGHTFIRE)
        downLeft = jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNLEFTFIRE)
        downRight = jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNRIGHTFIRE)
        direction = jnp.stack([noop, up, right, left, down, upRight, upLeft, downRight, downLeft, downRight])
        # updating enemy x and z based on player movement
        update_all_positions = jax.vmap(self._enemy_position_update, in_axes=(None, 0, None))(state, state.enemies, direction)
        update_all_distance = jax.vmap(self._enemy_distance_update, in_axes=(None, 0))(state, update_all_positions)
        return state._replace(enemies=update_all_distance)




    def reset(self, key=None) -> Tuple[BattlezoneObservation, BattlezoneState]:
        state = BattlezoneState(
            score=jnp.array(123400),
            life=jnp.array(5),
            step_counter=jnp.array(0),
            chains_l_anim_counter=jnp.array(0),
            chains_r_anim_counter=jnp.array(0),
            mountains_anim_counter=jnp.array(0),
            grass_anim_counter=jnp.array(0),
            radar_rotation_counter=jnp.array(0),
            enemies = Enemy(
                x=jnp.array([6.8047], dtype=jnp.float32),
                z=jnp.array([60.5547], dtype=jnp.float32),
                distance=jnp.array([60.93576], dtype=jnp.float32),
                enemy_type=jnp.array([EnemyType.TANK], dtype=jnp.float32),
                orientation_angle=jnp.array([1.57], dtype=jnp.float32)
            ),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BattlezoneState, action: chex.Array) -> Tuple[BattlezoneObservation, BattlezoneState,\
                float, bool, BattlezoneInfo]:
        previous_state = state
        new_state = state._replace(step_counter=state.step_counter+1)
        new_state = new_state._replace(radar_rotation_counter=(state.radar_rotation_counter
                                                           +self.consts.RADAR_ROTATION_SPEED)%360)
        new_state = self._player_step(new_state, action)
        new_state = self._enemy_step(new_state, action)

        done = self._get_done(new_state)
        env_reward = self._get_reward(previous_state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def _enemy_position_update(self, state: BattlezoneState, enemy: Enemy, direction) -> Enemy:
        #jax.debug.print("{}",direction)
        ###
        alpha = self.consts.PLAYER_ROTATION_SPEED
        speed = self.consts.PLAYER_SPEED
        ###
        offset_xz = jnp.array([
            [0, 0],     # Noop
            [0, -speed],  # Up
            [(enemy.x*jnp.cos(alpha)-enemy.z*jnp.sin(alpha))-enemy.x, (enemy.x*jnp.sin(alpha)+enemy.z*jnp.cos(alpha))-enemy.z],  # Right
            [(enemy.x*jnp.cos(-alpha)-enemy.z*jnp.sin(-alpha))-enemy.x, (enemy.x*jnp.sin(-alpha)+enemy.z*jnp.cos(-alpha))-enemy.z],  # Left
            [0, speed],  # Down
            [-speed, -speed],  # UpRight
            [speed, -speed],  # UpLeft
            [-speed, -speed],  # DownRight
            [speed, speed]  # DownLeft
        ])
        #jax.debug.print("{}",jnp.argmax(direction))
        # TODO: Enemies moving left and right but should rotate
        idx = jnp.argmax(direction)
        offset = offset_xz[idx]

        new_x = enemy.x + offset[0]
        new_z = enemy.z + offset[1]

        return enemy._replace(x=new_x, z=new_z)

    def _enemy_distance_update(self, state: BattlezoneState, enemy: Enemy) -> Enemy:
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
            {'name': 'life', 'type': 'single', 'file': 'life.npy'},
            {'name': 'player_digits', 'type': 'digits', 'pattern': 'player_score_{}.npy'},
            #enemies
            {'name': 'tank_enemy_front', 'type': 'single', 'file': 'tank_enemy_front.npy'}, #not sure if we can/should
            {'name': 'tank_enemy_left', 'type': 'single', 'file': 'tank_enemy_left.npy'},   #summarize them like digits
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

    @staticmethod
    def _draw_line(img, x0, y0, x1, y1, colorID, samples=256):
        #taken from experimental branch + some changes needs to be overworked maybe
        # Parametric line sampling (jit-friendly; Bresenham avoids floats but needs while loops)
        t = jnp.linspace(0.0, 1.0, samples)
        xs = jnp.round(x0 + (x1 - x0) * t).astype(jnp.int32)
        ys = jnp.round(y0 + (y1 - y0) * t).astype(jnp.int32)
        im = img
        im = im.at[ys.clip(0, im.shape[0] - 1), xs.clip(0, im.shape[1] - 1)].set(colorID)
        return im


    def _render_radar(self, img, state, center_x, center_y, radius, colorID_1, colorID_2):
        h, w = jnp.shape(img)
        #------------------draw line------------------
        alpha = state.radar_rotation_counter
        dir_x = jnp.sin(alpha)
        dir_y = jnp.cos(alpha)
        img = BattlezoneRenderer._draw_line(img, center_x, center_y,center_x+dir_x*radius,
                                            center_y+dir_y*radius, colorID_2)

        #-------------------draw circle-------------
        y = jnp.arange(h)[:, None]  #construct index coordinate mapping
        x = jnp.arange(w)[None, :]
        # Compute squared distance from center
        dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2
        extended_radius = radius+1
        mask = jnp.logical_and(dist_sq >= extended_radius ** 2, dist_sq < (extended_radius + 1) ** 2)
        img = jnp.where(mask, colorID_1, img)

        #------------------draw enemy dots----------------
        # Check if enemy in radar radius
        in_radar = jax.vmap(self.check_in_radar, in_axes=(0))(state.enemies)
        #jax.debug.print("in_radar: {}",in_radar)
        # Get raw player coords
        world_enemies_x = state.enemies.x
        world_enemies_z = state.enemies.z
        world_enemies_dist = state.enemies.distance
        #jax.debug.print("{}, {}",enemies_x, enemies_z)
        # Scale to radar size
        scale_val = radius / self.consts.RADAR_MAX_SCAN_RADIUS
        radar_enemies_x = world_enemies_x * scale_val
        radar_enemies_z = world_enemies_z * scale_val * (-1)
        # Offset to radar center
        radar_enemies_x = jnp.round(radar_enemies_x + center_x).astype(jnp.int32)
        radar_enemies_z = jnp.round(radar_enemies_z + center_y).astype(jnp.int32)
        # Only allow in range enemies
        radar_enemies_x = jnp.where(in_radar, radar_enemies_x, -1)
        radar_enemies_z = jnp.where(in_radar, radar_enemies_z, -1)
        # Draw point
        img = img.at[radar_enemies_z, radar_enemies_x].set(colorID_2)

        return img

    def check_in_radar(self, enemies: Enemy) -> chex.Array:
        bool_in_radar = enemies.distance <= self.consts.RADAR_MAX_SCAN_RADIUS
        return bool_in_radar



    def get_enemy_mask(self, enemy:Enemy): #todo change
        #selects the correct mask fo the given enemy
        return self.SHAPE_MASKS["tank_enemy_front"]


    def world_cords_to_viewport_cords(self, x, z): #todo change
        return 100, 80


    def zoom_mask(self, mask, zoom_factor:int):
        """
        scales the mask proportional to the zoom_factor
        at zoom_factor = 1 keeps everything the same
        every zoom factor above 1 makes the mask smaller by 2x2
        """
        if zoom_factor == 1:
            return mask

        x, y = mask.shape

        # compute new shape
        shrink = int(2 * (zoom_factor - 1))
        new_x = max(1, x - shrink)
        new_y = max(1, y - shrink)

        # create new grid indices for downsampling
        x_idx = jnp.floor(jnp.linspace(0, x - 1, new_x)).astype(int)
        y_idx = jnp.floor(jnp.linspace(0, y - 1, new_y)).astype(int)

        # index the original mask to create the zoomed mask
        zoomed_mask = mask[x_idx[:, None], y_idx[None, :]]

        return zoomed_mask


    def render_single_enemy(self, raster, enemy:Enemy): #todo change
        enemy_mask = self.get_enemy_mask(enemy)
        zoom_factor = 4     #change to distance * some_constant
        zoomed_mask = self.zoom_mask(enemy_mask, zoom_factor)
        x, y = self.world_cords_to_viewport_cords(enemy.x, enemy.z)

        return self.jr.render_at(raster, x+zoom_factor, y+zoom_factor, zoomed_mask)



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

        raster = self._render_radar(raster, state, self.consts.RADAR_CENTER_X, self.consts.RADAR_CENTER_Y,
                                    self.consts.RADAR_RADIUS, self.COLOR_TO_ID[self.consts.RADAR_COLOR_1],
                                    self.COLOR_TO_ID[self.consts.RADAR_COLOR_2])

        #--------------chains---------
        chains_l_mask = self.SHAPE_MASKS["chainsLeft"]
        color_shifted_chain_l = self._scroll_chain_colors(chains_l_mask, state.chains_l_anim_counter)
        raster = self.jr.render_at(raster, self.consts.CHAINS_L_POS_X,
                                   self.consts.CHAINS_POS_Y, color_shifted_chain_l)

        chains_r_mask = self.SHAPE_MASKS["chainsRight"]
        color_shifted_chain_r = self._scroll_chain_colors(chains_r_mask, state.chains_r_anim_counter)
        raster = self.jr.render_at(raster, self.consts.CHAINS_R_POS_X,
                                   self.consts.CHAINS_POS_Y, color_shifted_chain_r)

        #----------------life---------------------------
        life_mask = self.SHAPE_MASKS["life"]
        def render_single_life(i, raster):
            return self.jr.render_at(raster, self.consts.LIFE_POS_X+(self.consts.LIFE_X_OFFSET*i),
                                       self.consts.LIFE_POS_Y, life_mask)

        raster = jax.lax.fori_loop(0, state.life, render_single_life, raster)



        #---------------------------player score--------------------
        #primarily taken from pong + changes
        player_digit_masks = self.SHAPE_MASKS["player_digits"]  # Assumes single color
        player_digits = self.jr.int_to_digits(state.score, max_digits=8)
        #this does not correctly work currently (only when max_digits==exactly amount digits)

        raster = self.jr.render_label_selective(raster, self.consts.SCORE_POS_X, self.consts.SCORE_POS_Y, player_digits,
                                                player_digit_masks, 0, 8,
                                                spacing=6, max_digits_to_render=8)
                                                #best highscore i can find is 6 digits
        #--------------------------------------------------------------------

        #-------------------------------enemies-----------------
        def render_single_enemy_wrapped(raster, enemy): #so that i donw have to pass self
            return self.render_single_enemy(raster, enemy), None


        raster, _ = jax.lax.scan(render_single_enemy_wrapped, raster, state.enemies)



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