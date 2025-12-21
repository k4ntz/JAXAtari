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
- move speed: 0.24804691667 units per frame
- turn speed: 2pi/270 rad/frame (270 frames for a full turn)
-

## Tank
- behaviour:
    - shoots when distance to player is 29.09 units
- score points: 1000
- move speed: 0.125 units per frame
- turn speed: 2pi/2048 rad/frame (512 frames for a quarter turn)

## Flying Saucer
- score points: 5000

## Supertank
- score points: 3000

## Fighter Jet
- score points: 2000

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
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action



#------------------------named Tuples---------------------------
class EnemyType(IntEnum):
    TANK = 0
    SAUCER = 1
    # FIGHTER_JET = 2
    # SUPER_TANK = 3


class BattlezoneConstants(NamedTuple):

    # --- rendering: positions ---
    WIDTH: int = 160  # rgb_array size
    HEIGHT: int = 210
    WALL_TOP_Y: int = 0     # correct
    WALL_TOP_HEIGHT: int = 36   # correct
    WALL_BOTTOM_Y: int = 177    # correct
    WALL_BOTTOM_HEIGHT: int = 33    # correct
    TANK_SPRITE_POS_X: int = 43
    TANK_SPRITE_POS_Y: int = 140
    TARGET_INDICATOR_POS_X: int = 80
    TARGET_INDICATOR_POS_Y: int = 77
    CHAINS_POS_Y: int = 158
    CHAINS_L_POS_X: int = 19
    CHAINS_R_POS_X: int = 109
    MOUNTAINS_Y: int = 36
    GRASS_BACK_Y: int = 95
    HORIZON_Y: int = 92
    GRASS_FRONT_Y: int = 137
    RADAR_CENTER_X: int = 80
    RADAR_CENTER_Y: int = 18
    RADAR_RADIUS: int = 10
    LIFE_POS_X: int = 64
    LIFE_POS_Y: int = 189
    LIFE_X_OFFSET: int = 8
    SCORE_POS_X: int = 89
    SCORE_POS_Y: int = 179
    ENEMY_POS_Y: int = 85

    # --- rendering: colors ---
    SCORE_COLOR: Tuple[int, int, int] = (26,102,26)  # (45, 129, 105) but we need to change pallette first
    TARGET_INDICATOR_COLOR_ACTIVE: Tuple[int, int, int] = (255, 255, 0)
    TARGET_INDICATOR_COLOR_INACTIVE: Tuple[int, int, int] = (0, 0, 0)
    CHAINS_COL_1: Tuple[int, int, int] = (111,111,111)
    CHAINS_COL_2: Tuple[int, int, int] = (74, 74, 74)
    RADAR_COLOR_1: Tuple[int, int, int] = (111,210,111)
    RADAR_COLOR_2: Tuple[int, int, int] = (236,236,236)
    LIFE_SCORE_COLOR: Tuple[int, int, int] = (45,129,105)

    # --- world movement ---
    WORLD_SIZE_X: int = 256  # world size in x direction
    WORLD_SIZE_Z: int = 256  # world size in z direction
    PLAYER_ROTATION_SPEED: float = 2*jnp.pi/536
    PLAYER_SPEED: float = 0.25
    PROJECTILE_SPEED: float = 0.5
    ENEMY_SPEED: jnp.array = jnp.array([0.25, 0.25, 1.0, 1.0]) #todo change
    ENEMY_ROT_SPEED: jnp.array = jnp.array([2*jnp.pi/2048, 0.01, 0.01, 0.01]) #todo change

    # --- game mechanics ---
    HITBOX_SIZE: float = 6.0  # player
    ENEMY_HITBOX_SIZE: float = 4.0
    ENEMY_SPAWN_PROBS: jnp.array = jnp.array([
        # TANK, SAUCER, FIGHTER_JET, SUPER_TANK
        [1.0, 0.0],# 0.0, 0.0],   #1_000
        [0.8, 0.2],# 0.0, 0.0],   #2_000
        [0.6, 0.4],# 0.1, 0.0],   #7_000
        [0.5, 0.5]#, 0.2, 0.1]    #12_000
        ])
    RADAR_MAX_SCAN_RADIUS: int = 70

    # --- timing ---
    FIRE_CD: int = 114  # player
    PROJECTILE_TTL: int = 110
    DEATH_ANIM_LENGTH: int = 30
    ENEMY_DEATH_ANIM_LENGTH: int = 30
    ENEMY_SHOOT_CDS: jnp.array = jnp.array([400, 400, 400, 400]) #todo change

    # --- misc ---
    RADAR_ROTATION_SPEED: float = -0.05
    DISTANCE_TO_ZOOM_FACTOR_CONSTANT: float = 0.1
    ENEMY_SCORES: chex.Array = jnp.array([1000,5000,2000,3000], dtype=jnp.int32)


class Projectile(NamedTuple):
    """Class holding projectiles. properties are arrays."""
    x: chex.Array
    z: chex.Array
    orientation_angle: chex.Array
    active: chex.Array
    distance: chex.Array
    time_to_live: chex.Array


class Enemy(NamedTuple):
    x: chex.Array
    z: chex.Array
    distance: chex.Array
    enemy_type: chex.Array
    orientation_angle: chex.Array
    active: chex.Array
    death_anim_counter: chex.Array
    shoot_cd: chex.Array
    phase: chex.Array
    dist_moved_temp: chex.Array
    point_store_1_temp: chex.Array      # Especially for Saucer
    point_store_2_temp: chex.Array      # Especially for Saucer


# immutable state container
class BattlezoneState(NamedTuple):
    score: chex.Array
    life: chex.Array
    cur_fire_cd: chex.Array  # player current fire cooldown
    step_counter: chex.Array
    chains_l_anim_counter: chex.Array
    chains_r_anim_counter: chex.Array
    death_anim_counter: chex.Array
    mountains_anim_counter: chex.Array
    grass_anim_counter: chex.Array
    radar_rotation_counter: chex.Array
    enemies: Enemy
    player_projectile: Projectile # player can only fire 1 projectile
    enemy_projectiles: Projectile # per enemy 1 projectile
    random_key: chex.PRNGKey
    shot_spawn: chex.Array


class BattlezoneObservation(NamedTuple):
    score: jnp.ndarray


class BattlezoneInfo(NamedTuple):
    time: jnp.ndarray


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
        self.obs_size = 3*4+1+1  #?? TODO: change later, from pong currently


    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: BattlezoneState, action: chex.Array) -> BattlezoneState:

        #-------------------parse action--------------------
        noop = (action == Action.NOOP)
        up = jnp.logical_or(action==Action.UP, action==Action.UPFIRE)
        down = jnp.logical_or(action==Action.DOWN, action==Action.DOWNFIRE)
        right = jnp.logical_or(action==Action.RIGHT, action==Action.RIGHTFIRE)
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        upLeft = jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE)
        upRight = jnp.logical_or(action==Action.UPRIGHT, action==Action.UPRIGHTFIRE)
        downLeft = jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNLEFTFIRE)
        downRight = jnp.logical_or(action==Action.DOWNRIGHT, action==Action.DOWNRIGHTFIRE)
        
        wants_fire = jnp.any(jnp.stack([
            action == Action.FIRE, 
            action == Action.LEFTFIRE,
            action == Action.UPFIRE,
            action == Action.RIGHTFIRE,
            action == Action.DOWNFIRE, 
            action == Action.UPLEFTFIRE,  # TODO: why doesn't this work?
            action == Action.UPRIGHTFIRE,
            action == Action.DOWNLEFTFIRE,
            action == Action.DOWNRIGHTFIRE,  # TODO: why doesn't this work?
        ]), axis=0)

        jax.debug.print("{}", action)
        
        direction = jnp.stack([noop, up, right, left, down, upRight, upLeft, downRight, downLeft])

        #-------------------fire--------------
        will_fire = jnp.logical_and(wants_fire, state.cur_fire_cd <= 0)
        
        def fire_projectile(state: BattlezoneState):
            return state._replace(
                cur_fire_cd=jnp.array(self.consts.FIRE_CD, dtype=jnp.int32),
                player_projectile= Projectile(
                    x=jnp.array(-0.2, dtype=jnp.float32),
                    z=jnp.array(7, dtype=jnp.float32),
                    orientation_angle=jnp.array(jnp.pi, dtype=jnp.float32),
                    active=jnp.array(True, dtype=jnp.bool),
                    distance=jnp.array(0, dtype=jnp.float32),
                    time_to_live=jnp.array(self.consts.PROJECTILE_TTL, dtype=jnp.int32)
                )
            )

        def dont_fire(state:BattlezoneState):
            return state

        new_state = jax.lax.cond(will_fire, fire_projectile, dont_fire, state)


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


        #--------------------update positions based on player movement-------------------
        updated_enemies = jax.vmap(self._obj_player_position_update, in_axes=(0, None))(state.enemies, direction)
        updated_projectiles = (jax.vmap(self._obj_player_position_update, in_axes=(0, None))
                               (state.enemy_projectiles, direction))
        new_player_projectile = self._obj_player_position_update(new_state.player_projectile, direction)

        #--------------------update angles based on player movement-----------------------
        angle_change = (jnp.where(jnp.any(jnp.stack([left, upLeft, downRight])), 1.0, 0.0)
                        - jnp.where(jnp.any(jnp.stack([right, upRight, downLeft])), 1.0, 0.0))

        updated_enemies = jax.vmap(self._obj_player_rotation_update, in_axes=(0,None))(updated_enemies, angle_change)
        updated_projectiles = (jax.vmap(self._obj_player_rotation_update, in_axes=(0,None))
                               (updated_projectiles, angle_change))
        new_player_projectile = self._obj_player_rotation_update(new_player_projectile, angle_change)

        return new_state._replace(
            chains_l_anim_counter=(state.chains_l_anim_counter + chain_l_offset)%32,
            chains_r_anim_counter=(state.chains_r_anim_counter + chain_r_offset)%32,
            mountains_anim_counter=(state.mountains_anim_counter + mountains_offset)%160,
            grass_anim_counter= (state.grass_anim_counter + grass_offset)%30,
            radar_rotation_counter=state.radar_rotation_counter,
            enemies=updated_enemies,
            player_projectile=new_player_projectile,
            enemy_projectiles=updated_projectiles
        )


    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: BattlezoneState) -> BattlezoneState:
        d_anim_counter = state.enemies.death_anim_counter
        new_death_anim_counter = jnp.where(d_anim_counter > 0, d_anim_counter-1,d_anim_counter)
        new_enemies, new_projectiles = (jax.vmap(self.enemy_movement, in_axes=(0, 0))
                                        (state.enemies, state.enemy_projectiles))
        shoot_cd = new_enemies.shoot_cd
        new_shoot_cd = jnp.where(shoot_cd > 0, shoot_cd - 1, shoot_cd)
        return state._replace(enemies=new_enemies._replace(death_anim_counter=new_death_anim_counter,
                                                           shoot_cd=new_shoot_cd),
                              enemy_projectiles=new_projectiles)


    @partial(jax.jit, static_argnums=(0,))
    def _single_projectile_step(self, projectile: Projectile) -> Projectile:
        """implements movement of projectiles"""
        dir_x = -jnp.sin(projectile.orientation_angle)
        dir_z = jnp.cos(projectile.orientation_angle)
        new_x = projectile.x - dir_x*self.consts.PROJECTILE_SPEED
        new_z = projectile.z - dir_z*self.consts.PROJECTILE_SPEED

        return projectile._replace(
            x=new_x,
            z=new_z,
            time_to_live=jnp.where(projectile.time_to_live>0, projectile.time_to_live-1, 0),
            active=jnp.logical_and(projectile.active, projectile.time_to_live>0)
        )

    def _player_projectile_col_check(self, state: BattlezoneState):
        """"""
        hit_arr = (jax.vmap(self._player_projectile_collision_check, in_axes=(0, None))
                   (state.enemies, state.player_projectile))
        def _score_func(state1:BattlezoneState, in_tuple):
            enemy, hit = in_tuple
            new_score = state1.score + jnp.where(hit, self.consts.ENEMY_SCORES[enemy.enemy_type], 0)
            return state1._replace(score=new_score), None

        new_state, _ = jax.lax.scan(_score_func, state, (state.enemies, hit_arr))
        new_enemies_active = jnp.logical_and(new_state.enemies.active, jnp.invert(hit_arr))
        new_enemies_death_anim_counter = jnp.where(hit_arr,
                            self.consts.ENEMY_DEATH_ANIM_LENGTH, new_state.enemies.death_anim_counter)
        new_player_projectile_active = jnp.logical_and(new_state.player_projectile.active, jnp.invert(jnp.any(hit_arr)))
        return new_state._replace(
            enemies=new_state.enemies._replace(active=new_enemies_active,
                                               death_anim_counter=new_enemies_death_anim_counter),
            player_projectile=new_state.player_projectile._replace(active=new_player_projectile_active)
        )


    def reset(self, key=None) -> Tuple[BattlezoneObservation, BattlezoneState]:
        if key is None:
            key = jax.random.PRNGKey(0)
        state = BattlezoneState(
            score=jnp.array(0, dtype=jnp.int32),
            life=jnp.array(5, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            cur_fire_cd=jnp.array(0, dtype=jnp.int32),
            death_anim_counter=jnp.array(0, dtype=jnp.int32),
            chains_l_anim_counter=jnp.array(0, dtype=jnp.float32),
            chains_r_anim_counter=jnp.array(0, dtype=jnp.float32),
            mountains_anim_counter=jnp.array(0, dtype=jnp.float32),
            grass_anim_counter=jnp.array(0, dtype=jnp.float32),
            radar_rotation_counter=jnp.array(0, dtype=jnp.float32),
            enemies = Enemy(
                x=jnp.array([6.8047, 3.8047], dtype=jnp.float32),
                z=jnp.array([60.5547, 30.5547], dtype=jnp.float32),
                distance=jnp.array([60.93576, 30.93576], dtype=jnp.float32),
                enemy_type=jnp.array([EnemyType.TANK, EnemyType.TANK], dtype=jnp.int32),
                orientation_angle=jnp.array([1.57, 0.5], dtype=jnp.float32),
                active=jnp.array([True, False], dtype=jnp.bool),
                death_anim_counter=jnp.array([0,0], dtype=jnp.int32),
                shoot_cd=jnp.array([0, 0], dtype=jnp.int32),
                phase=jnp.array([0, 0], dtype=jnp.int32),
                dist_moved_temp=jnp.array([0, 0], dtype=jnp.float32),
                point_store_1_temp=jnp.array([[0, 0], [0, 0]], dtype=jnp.float32),
                point_store_2_temp=jnp.array([[0, 0], [0, 0]], dtype=jnp.float32),
            ),
            player_projectile=Projectile(
                x=jnp.array(0, dtype=jnp.float32),
                z=jnp.array(0, dtype=jnp.float32),
                orientation_angle=jnp.array(0, dtype=jnp.float32),
                active=jnp.array(False, dtype=jnp.bool),
                distance=jnp.array(0, dtype=jnp.float32),
                time_to_live=jnp.array(0, dtype=jnp.int32)
            ),
            enemy_projectiles=Projectile(
                x=jnp.array([0, 0], dtype=jnp.float32),
                z=jnp.array([0, 0], dtype=jnp.float32),
                orientation_angle=jnp.array([0, 0], dtype=jnp.float32),
                active=jnp.array([False, False], dtype=jnp.bool),
                distance=jnp.array([0, 0], dtype=jnp.float32),
                time_to_live=jnp.array([0, 0], dtype=jnp.int32)
            ),
            random_key=key,
            shot_spawn=jnp.array([False], dtype=jnp.bool)
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BattlezoneState, action: chex.Array) -> Tuple[BattlezoneObservation, BattlezoneState,\
                float, bool, BattlezoneInfo]:
        previous_state = state
        def normal_step(state):
            # update counters
            new_state = state._replace(step_counter=state.step_counter+1, cur_fire_cd=state.cur_fire_cd-1)
            new_state = new_state._replace(radar_rotation_counter=(state.radar_rotation_counter
                                                               +self.consts.RADAR_ROTATION_SPEED)%360)

            #-------------------projectiles-------------
            # move player projectile forwards
            new_player_projectile = self._single_projectile_step(state.player_projectile)
            new_state = new_state._replace(player_projectile=new_player_projectile)
            # check whether player projectile hit an enemy
            new_state = self._player_projectile_col_check(new_state)

            new_enemy_projectiles = jax.vmap(self._single_projectile_step, 0)(state.enemy_projectiles)
            new_state = new_state._replace(enemy_projectiles=new_enemy_projectiles)
            player_hit = jnp.any(jax.vmap(self._enemy_projectile_collision_check, 0)(new_state.enemy_projectiles))
            new_state = new_state._replace(death_anim_counter=jnp.where(player_hit,
                                                jnp.array(self.consts.DEATH_ANIM_LENGTH, dtype=jnp.int32),
                                                new_state.death_anim_counter))
            #------------------------------------------

            #-------------------spawn-------------------
            split_key, key = jax.random.split(new_state.random_key, 2)
            new_state = new_state._replace(random_key=key)
            new_state = new_state._replace(enemies=jax.vmap(self.spawn_enemy, in_axes=(0, 0, None, None))
                (jax.random.split(split_key,new_state.enemies.active.shape[0]), new_state.enemies, new_state.score, new_state))
            #-------------------------------------------

            new_state = self._player_step(new_state, action)
            new_state = self._enemy_step(new_state)

            return new_state

        def death_step(state):
            new_death_counter= state.death_anim_counter-1
            new_state = state._replace(death_anim_counter=new_death_counter)
            new_state = jax.lax.cond(new_death_counter <= 0, self.player_shot_reset, lambda x: x, new_state)
            return new_state

        new_state = jax.lax.cond(state.death_anim_counter <= 0, normal_step, death_step, state)

        done = self._get_done(new_state)
        env_reward = self._get_reward(previous_state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info


    def _obj_player_position_update(self, obj: NamedTuple, player_direction) -> Enemy:
        """
        _position_update version for named tuples that contain x, z, distance
        """
        new_x, new_z = self._position_update(obj.x, obj.z, obj.active, player_direction)
        return obj._replace(x=new_x, z=new_z, distance=self._get_distance(new_x, new_z))



    def _position_update(self, prev_x, prev_z, active, player_direction):
        """
        updates the x, z coordinates according to the current player movement
        """
        #jax.debug.print("{}",direction)
        ###
        alpha = self.consts.PLAYER_ROTATION_SPEED
        speed = self.consts.PLAYER_SPEED
        ###
        sin_alpha = jnp.sin(alpha)
        cos_alpha = jnp.cos(alpha)
        # Changing position based on player movement
        # TODO: it would be better if we only computed the offset for the active movement, right?
        offset_xz = jnp.array([
            [0, 0],     # Noop
            [0, -speed],  # Up
            [(prev_x*cos_alpha-prev_z*sin_alpha)-prev_x, (prev_x*sin_alpha+prev_z*cos_alpha)-prev_z],  # Right
            [(prev_x*jnp.cos(-alpha)-prev_z*jnp.sin(-alpha))-prev_x, (prev_x*jnp.sin(-alpha)+prev_z*jnp.cos(-alpha))-prev_z],  # Left
            [0, speed],  # Down
            [(prev_x * cos_alpha - (prev_z - speed) * sin_alpha) - prev_x, (prev_x * sin_alpha + (prev_z - speed) * cos_alpha) - prev_z], # UpRight
            [(prev_x * cos_alpha + (prev_z - speed) * sin_alpha) - prev_x, (-prev_x * sin_alpha + (prev_z - speed) * cos_alpha) - prev_z],  # UpLeft
            [(prev_x * cos_alpha + (prev_z + speed) * sin_alpha) - prev_x, (-prev_x * sin_alpha + (prev_z + speed) * cos_alpha) - prev_z],  # DownRight
            [(prev_x * cos_alpha - (prev_z + speed) * sin_alpha) - prev_x, (prev_x * sin_alpha + (prev_z + speed) * cos_alpha) - prev_z]  # DownLeft
        ])
        #jax.debug.print("{}",jnp.argmax(direction)
        idx = jnp.argmax(player_direction)
        offset = offset_xz[idx]

        new_x = prev_x + offset[0] 
        new_z = prev_z + offset[1] 

        # Wrap around world edges
        new_x = (new_x + self.consts.WORLD_SIZE_X/2) % self.consts.WORLD_SIZE_X - self.consts.WORLD_SIZE_X/2
        new_z = (new_z + self.consts.WORLD_SIZE_Z/2) % self.consts.WORLD_SIZE_Z - self.consts.WORLD_SIZE_Z/2
        
        return new_x, new_z


    def _obj_player_rotation_update(self, obj: NamedTuple, angle_change):
        alpha = self.consts.PLAYER_ROTATION_SPEED
        dist = self._get_distance(obj.x, obj.z)
        opp = jnp.tan(alpha)*dist
        beta = jnp.atan(dist/opp)
        angle = ((jnp.pi/2)-beta)*angle_change
        return obj._replace(orientation_angle=
                            (obj.orientation_angle-angle)%(2*jnp.pi))


    def _player_projectile_collision_check(self, enemies: Enemy, player_projectiles: Projectile) -> bool:
        """checks whether a player projectile hit an enemy. player_projectiles contains arrays of size 1"""
        s = self.consts.ENEMY_HITBOX_SIZE
        distx = jnp.abs((enemies.x+s) - player_projectiles.x) <= s
        distz = jnp.abs((enemies.z+s) - player_projectiles.z) <= s
        return jnp.all(jnp.stack([distx, distz, enemies.active, player_projectiles.active]))


    def _enemy_projectile_collision_check(self, obj: Projectile):
        s = self.consts.HITBOX_SIZE
        distx = jnp.abs(obj.x) <= s
        distz = jnp.abs(obj.z) <= s
        return jnp.all(jnp.stack([distx, distz, obj.active]))


    def _get_distance(self, x, z):
        distance = jnp.sqrt(x ** 2 + z ** 2)
        #Room for distance specific actions
        return distance

    @partial(jax.jit, static_argnums=(0,))
    def spawn_enemy(self, key, enemy:Enemy, score, state):
        def score_to_spawn_indx(score):
            threshold = jnp.array([1000, 2000, 7000, 12000])
            return jnp.sum(score >= threshold)
        def no_spawn(args):
            enemy, key, score = args
            return enemy
        def do_spawn(args):
            enemy, key, score = args
            # Enemy spawnprobs
            spawn_probs_index = score_to_spawn_indx(score)
            spawn_probs = self.consts.ENEMY_SPAWN_PROBS[spawn_probs_index]
            # Random keys
            key, k_dist, k_theta, k_type, k_orient = jax.random.split(key, 5)
            # Enemy position
            distance = jnp.sqrt(jax.random.uniform(k_dist, minval=.2, maxval=.9))*self.consts.RADAR_MAX_SCAN_RADIUS
            theta = jax.random.uniform(k_theta, minval=0.0, maxval=2*jnp.pi)
            enemy_type = jax.random.choice(k_type, a=len(EnemyType), p=spawn_probs)

            return enemy._replace(x=distance*jnp.cos(theta),
                                  z=distance*jnp.sin(theta),
                                  distance=distance,
                                  enemy_type=enemy_type,
                                  orientation_angle=jax.random.uniform(k_orient, minval=0.0, maxval=2*jnp.pi),
                                  active=True,
                                  shoot_cd=self.consts.ENEMY_SHOOT_CDS[enemy_type],
                                  phase=0,
                                  dist_moved_temp=0.0,
                                  point_store_1_temp=jnp.zeros((2,), dtype=jnp.float32),
                                  point_store_2_temp=jnp.zeros((2,), dtype=jnp.float32),
                                  )

        cond = jnp.logical_or(enemy.active, enemy.death_anim_counter>0)
        cond = jnp.logical_or(cond, jnp.logical_and(score<1000, state.shot_spawn==False))
        cond = jnp.any(cond)

        return jax.lax.cond(cond, no_spawn, do_spawn, (enemy, key, score))

    # -------------Enemy Movements-----------------
    @partial(jax.jit, static_argnums=(0,))
    def enemy_movement(self, enemy: Enemy, projectile: Projectile):
        perfect_angle = (2*jnp.pi - jnp.arctan2(enemy.x , enemy.z)) % (2*jnp.pi)
        angle_diff = (perfect_angle-enemy.orientation_angle+jnp.pi) % (2*jnp.pi)-jnp.pi
        speed = self.consts.ENEMY_SPEED[enemy.enemy_type]
        rot_speed = self.consts.ENEMY_ROT_SPEED[enemy.enemy_type]

        # ---------Helper Functions---------

        def move_to_player(enemy: Enemy, towards: int = 1) -> Enemy:  # Enemy towards player with 1 and away with -1
            k = (enemy.distance + towards * speed) / enemy.distance
            direction_x = -jnp.sin(perfect_angle)
            direction_z = jnp.cos(perfect_angle)
            return enemy._replace(x=enemy.x +direction_x*towards*speed, z=enemy.z +direction_z*towards*speed)

        def enemy_turn(enemy: Enemy) -> Enemy:
            return enemy._replace(orientation_angle=enemy.orientation_angle+jnp.sign(angle_diff)*rot_speed)

        def shoot_projectile(args) -> Tuple[Enemy, Projectile]:
            enemy, projectile= args
            new_enemy = enemy._replace(shoot_cd=self.consts.ENEMY_SHOOT_CDS[enemy.enemy_type])
            return new_enemy, projectile._replace(
                orientation_angle=perfect_angle,
                x = enemy.x+self.consts.ENEMY_HITBOX_SIZE,
                z = enemy.z,
                active=True,
                time_to_live=jnp.array(self.consts.PROJECTILE_TTL, dtype=jnp.int32)
            )


        # ---------------------------------
        ## Tank
        def tank_movement(tank: Enemy) -> Enemy:
            #jax.debug.print("{}",tank.distance)
            def player_spotted(tank):

                def in_range(tank):

                    def towards_player(tank):
                        #jax.debug.print("MOVE TO PLayer{}", tank.orientation_angle)
                        return move_to_player(tank)

                    def away_player(tank):
                        # jax.debug.print("{}", tank.orientation_angle)
                        return move_to_player(tank, -1)

                    return jax.lax.cond(tank.distance <= 10.0, away_player, towards_player, tank)    # Enemy keeps 10 metrics distance to player
                
                def out_range(tank):
                    return move_to_player(tank, -1)

                return jax.lax.cond(tank.distance <= 30.0, in_range, out_range, tank)
            
            def player_not_spotted(tank):
                return enemy_turn(tank)
                #return jax.lax.switch(distance_threshold_indx(tank.distance), (), tank)

            return jax.lax.cond(jnp.abs(angle_diff) <= rot_speed,
                                player_spotted, player_not_spotted, tank)


        def saucer_movement(saucer: Enemy) -> Enemy:
            min_dist = 27.

            def near_player(saucer):
                saucer = saucer._replace(phase=0)
                return move_to_player(saucer, -1)
            def far_player(saucer):

                def p0(saucer):
                    saucer = saucer._replace(dist_moved_temp=saucer.dist_moved_temp + speed)
                    saucer = move_to_player(saucer, -1)

                    def next_phase(saucer):
                        saucer = saucer._replace(dist_moved_temp=0.0)
                        saucer = saucer._replace(phase=1)
                        return saucer

                    def cont_move(saucer):
                        return saucer

                    return jax.lax.cond(saucer.dist_moved_temp>11.8, next_phase, cont_move, saucer)

                def p1(saucer: Enemy):
                    # Rotation center
                    ux = saucer.x/saucer.distance
                    uz = saucer.z/saucer.distance
                    rot_centre_x = saucer.x+(9.22*ux)+(4.33*uz)
                    rot_centre_z = saucer.z+(9.22*uz)-(4.33*ux)
                    # Intersection circle and line
                    ## Component for abc
                    rad = jnp.sqrt((saucer.x-rot_centre_x)*(saucer.x-rot_centre_x) + (saucer.z-rot_centre_z)*(saucer.z-rot_centre_z))
                    m = saucer.z/saucer.x
                    a = 1+m*m
                    b =(-2)*(rot_centre_x+m*rot_centre_z)
                    c = rot_centre_x*rot_centre_x+rot_centre_z*rot_centre_z-rad*rad
                    ## abc
                    x_1 = ((-b)+jnp.sqrt(b*b-4*a*c))/(2*a)
                    x_2 = ((-b)-jnp.sqrt(b*b-4*a*c))/(2*a)
                    ## z values
                    z_1 = m*x_1
                    z_2 = m*x_2
                    ## furdest point from origin
                    dist_sq1 = x_1*x_1 + z_1*z_1
                    dist_sq2 = x_2*x_2 + z_2*z_2
                    ## Set x and z
                    intersection_point_x = jnp.where(dist_sq1>dist_sq2, x_1, x_2)
                    intersection_point_y = jnp.where(dist_sq1>dist_sq2, z_1, z_2)
                    # Update saucer
                    saucer = saucer._replace(point_store_1_temp=jnp.array([rot_centre_x, rot_centre_z]))    # Centre of rotation
                    saucer = saucer._replace(point_store_2_temp=jnp.array([intersection_point_x, intersection_point_y]))    # Stop point for rotation
                    return saucer._replace(phase=2)

                def p2(saucer):
                    rot_centre_x = saucer.point_store_1_temp[0]
                    rot_centre_z = saucer.point_store_1_temp[1]
                    end_rot_point = saucer.point_store_2_temp

                    vx = saucer.x - rot_centre_x
                    vz = saucer.z - rot_centre_z
                    ## Radius
                    rad = jnp.sqrt(vx*vx + vz*vz)
                    ##
                    dtheta = speed/rad
                    ##
                    cos_dtheta = jnp.cos(dtheta)
                    sin_dtheta = jnp.sin(dtheta)
                    vx_new = cos_dtheta*vx-sin_dtheta*vz
                    vz_new = sin_dtheta*vx+cos_dtheta*vz
                    # Update saucer
                    saucer = saucer._replace(x = rot_centre_x+vx_new, z = rot_centre_z+vz_new)

                    def next_phase(saucer):
                        return saucer._replace(point_store_1_temp=jnp.zeros((2,), dtype=jnp.float32),
                                                 point_store_2_temp=jnp.zeros((2,), dtype=jnp.float32),
                                                 phase=0)

                    def cont_circular(saucer):
                        return saucer

                    reach_end_point = jnp.isclose(jnp.array([saucer.x, saucer.z]), end_rot_point, atol=1)
                    reach_end_point = jnp.all(reach_end_point)
                    return jax.lax.cond(reach_end_point, next_phase, cont_circular, saucer)

                return jax.lax.switch(saucer.phase, (p0, p1, p2), saucer)

            return jax.lax.cond(saucer.distance<min_dist, near_player, far_player, saucer)


        shoot_cond = jnp.all(jnp.array([enemy.enemy_type != EnemyType.SAUCER,
                                        enemy.shoot_cd <= 0,
                                        jnp.abs(angle_diff) <= rot_speed,
                                        enemy.distance <= 31.0, # TODO change
                                        enemy.active]))

        new_enemy, new_projectile = jax.lax.cond(shoot_cond, shoot_projectile, lambda x: x,
                                                 (enemy, projectile))
        return (jax.lax.switch(enemy.enemy_type, (tank_movement, saucer_movement), new_enemy),
                new_projectile)

    # ---------------------------------------------


    def render(self, state: BattlezoneState) -> jnp.ndarray:
        return self.renderer.render(state)

    def player_shot_reset(self, state:BattlezoneState) -> BattlezoneState:
        """reset function for when the player was shot but still has remaining lives"""

        split_key, key = jax.random.split(state.random_key, 2)
        # Set enemies to inactive
        inactive_enemies = state.enemies._replace(active=jnp.zeros_like(state.enemies.active))
        new_state = state._replace(enemies = inactive_enemies)
        new_state = new_state._replace(shot_spawn=jnp.ones_like(new_state.shot_spawn))
        new_state = new_state._replace(
            life=new_state.life-1,
            enemies=jax.vmap(self.spawn_enemy, in_axes=(0, 0, None, None))
                (jax.random.split(split_key,new_state.enemies.active.shape[0]), new_state.enemies, new_state.score, new_state),
            player_projectile=Projectile(
                x=jnp.array(0, dtype=jnp.float32),
                z=jnp.array(0, dtype=jnp.float32),
                orientation_angle=jnp.array(0, dtype=jnp.float32),
                active=jnp.array(False, dtype=jnp.bool),
                distance=jnp.array(0, dtype=jnp.float32),
                time_to_live=jnp.array(0, dtype=jnp.int32)
            ),
            enemy_projectiles=Projectile(
                x=jnp.array([0, 0], dtype=jnp.float32),
                z=jnp.array([0, 0], dtype=jnp.float32),
                orientation_angle=jnp.array([0, 0], dtype=jnp.float32),
                active=jnp.array([False, False], dtype=jnp.bool),
                distance=jnp.array([0, 0], dtype=jnp.float32),
                time_to_live=jnp.array([0, 0], dtype=jnp.int32)
            ),
            random_key=key,
            cur_fire_cd=jnp.array(0, dtype=jnp.int32),
        )

        return new_state

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
        player_dead = state.life == 0
        return player_dead  # if lives are < 0 change later



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
        wall_sprite_top = self._create_wall_sprite(self.consts.SCORE_POS_Y)
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

        #----------------------create padded enemy masks for uniform shape-----------------------
        pad = 140
        self.padded_enemy_masks = jnp.array([
            [self.pad_to_shape(self.SHAPE_MASKS["tank_enemy_01"],pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["tank_enemy_02"],pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["tank_enemy_03"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["tank_enemy_04"], pad, pad),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["tank_enemy_03"], axis=1), pad, pad),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["tank_enemy_02"], axis=1), pad, pad),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["tank_enemy_01"], axis=1), pad, pad),
             ],
            [self.pad_to_shape(self.SHAPE_MASKS["saucer_left"],pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["saucer_left"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["saucer_left"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["saucer_left"], pad, pad),
            self.pad_to_shape(self.SHAPE_MASKS["saucer_right"],pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["saucer_right"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["saucer_right"], pad, pad)
             ],

             ])
        self.enemy_explosion_mask = jnp.array([self.pad_to_shape(self.SHAPE_MASKS["enemy_explosion_1"], pad, pad),
                                self.pad_to_shape(self.SHAPE_MASKS["enemy_explosion_2"], pad, pad),
                                self.pad_to_shape(self.SHAPE_MASKS["enemy_explosion_3"], pad, pad)])
        self.projectile_masks = jnp.array([
            self.pad_to_shape(self.SHAPE_MASKS["projectile_big"], 3, 3),
            self.pad_to_shape(self.SHAPE_MASKS["projectile_small"], 3, 3)
        ])
        self.projectile_masks = jnp.array([
            self.pad_to_shape(self.SHAPE_MASKS["projectile_big"], 6, 3),
            self.pad_to_shape(self.SHAPE_MASKS["projectile_small"], 6, 3)
        ])



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
            {'name': 'tank_enemy_01', 'type': 'single', 'file': 'tank_enemy_01.npy'}, #not sure if we can/should
            {'name': 'tank_enemy_02', 'type': 'single', 'file': 'tank_enemy_02.npy'},   #summarize them like digits
            {'name': 'tank_enemy_03', 'type': 'single', 'file': 'tank_enemy_03.npy'},
            {'name': 'tank_enemy_04', 'type': 'single', 'file': 'tank_enemy_04.npy'},
            {'name': 'saucer_left', 'type': 'single', 'file': 'saucer_left.npy'},
            {'name': 'saucer_right', 'type': 'single', 'file': 'saucer_right.npy'},
            {'name': 'projectile_big', 'type': 'single', 'file': 'projectile_big.npy'},
            {'name': 'projectile_small', 'type': 'single', 'file': 'projectile_small.npy'},
            #anims
            {'name': 'enemy_explosion_1', 'type': 'single', 'file': 'enemy_explosion_1.npy'},
            {'name': 'enemy_explosion_2', 'type': 'single', 'file': 'enemy_explosion_2.npy'},
            {'name': 'enemy_explosion_3', 'type': 'single', 'file': 'enemy_explosion_3.npy'},
            # Add the procedurally created sprites to the manifest
            {'name': 'blackscreen', 'type': 'procedural', 'data': wall_sprite_top},
            {'name': 'wall_bottom', 'type': 'procedural', 'data': wall_sprite_bottom},
            {'name': 'target_indicator', 'type': 'single', 'file': 'yellow_pixel.npy'},
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
        return((enemies.distance <= self.consts.RADAR_MAX_SCAN_RADIUS)&enemies.active)



    def pad_to_shape(self, arr: jnp.ndarray, shape_target_x: int, shape_target_y: int) -> jnp.ndarray:
        x, y = arr.shape
        pad_x = shape_target_x - x
        pad_y = shape_target_y - y
        return jnp.pad(arr, ((0, pad_x), (0, pad_y)), mode='constant', constant_values=-1)

    def get_enemy_mask(self, enemy:Enemy):
        #selects the correct mask fo the given enemy
        selected_enemy_type = self.padded_enemy_masks[enemy.enemy_type] #this is still an array containing all rotations

        #----------------select sprite based on rotation------------
        n, _, _ = jnp.shape(selected_enemy_type)
        circle = 2*jnp.pi  #change to 2pi if radians
        #v = jnp.array([enemy.x, enemy.z])
        #w = jnp.array([0, 1])
        #to_screen_angle = (jnp.dot(v, w)/jnp.linalg.norm(v))
        #angle = (enemy.orientation_angle + to_screen_angle - (jnp.pi/2)) % circle
        angle = ((jnp.pi/2)-enemy.orientation_angle) % circle
        angle = jnp.where(angle<=jnp.pi, angle, jnp.where(angle <= jnp.pi+jnp.pi/2, jnp.pi, 0))
        index = jnp.round((angle/jnp.pi) * (n-1)).astype(int)
        rotated_sprite = selected_enemy_type[index]

        return rotated_sprite


    def world_cords_to_viewport_cords(self, x, z, f=60.0):
        #f = (screen_height / 2) / tan(FOVv / 2)
        def anchor(_):
            # Behind the camera or invalid
            return -100,-100

        def uvMap(_):
            u = ((f * (x / z))+self.consts.WIDTH/2).astype(int)
            vOffset = self.consts.HORIZON_Y
            v = ((f/(z-self.consts.HITBOX_SIZE)) + vOffset).astype(int)
            return u, v

        return jax.lax.cond(z<=0, anchor, uvMap, operand=None)


    def zoom_mask(self, mask, zoom_factor):
        """
        Scales the mask proportional to zoom_factor.
        - zoom_factor = 1 keeps everything the same
        - zoom_factor > 1 zooms in (mask elements appear smaller)
        Works with masks padded with -1.
        """

        def anchor(_):
            return mask

        def zoom(_):
            x, y = mask.shape

            # Create grid of coordinates in the output canvas
            xi = jnp.arange(x)
            yi = jnp.arange(y)
            xv, yv = jnp.meshgrid(xi, yi, indexing='ij')

            # Compute coordinates in the original mask to sample
            # Zoom center is at the middle of the **non-padded region**
            valid_rows = jnp.any(mask != -1, axis=1)
            valid_cols = jnp.any(mask != -1, axis=0)
            x_min, x_max = jnp.argmax(valid_rows), x - jnp.argmax(valid_rows[::-1]) - 1
            y_min, y_max = jnp.argmax(valid_cols), y - jnp.argmax(valid_cols[::-1]) - 1
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

            # Zoom coordinates
            x_orig = (xv - cx) * zoom_factor + cx
            y_orig = (yv - cy) * zoom_factor + cy

            # Clip to the **non-padded region**
            x_orig = jnp.clip(x_orig, x_min, x_max)
            y_orig = jnp.clip(y_orig, y_min, y_max)

            # Nearest neighbor sampling
            x0 = jnp.floor(x_orig).astype(int)
            y0 = jnp.floor(y_orig).astype(int)
            zoomed_mask = mask[x0, y0]

            # Zero edges: set padding outside zoomed area to -1
            rows = jnp.arange(x)[:, None]
            cols = jnp.arange(y)[None, :]
            #edge = (rows < x_min) | (rows > x_max) | (cols < y_min) | (cols > y_max)
            z = zoom_factor+1
            edge = (rows < x_min+z) | (rows >= x_max - z) | \
                   (cols < y_min+z) | (cols >= y_max - z)
            zoomed_mask = jnp.where(edge, -1, zoomed_mask)

            return zoomed_mask

        return jax.lax.cond(zoom_factor <= 1, anchor, zoom, operand=None)


    def render_single_enemy(self, raster, enemy:Enemy):

        def enemy_active(enemy):
            enemy_mask = self.get_enemy_mask(enemy)
            zoom_factor = ((jnp.sqrt(jnp.square(enemy.x) + jnp.square(enemy.z)) - 20.0) *
                           self.consts.DISTANCE_TO_ZOOM_FACTOR_CONSTANT).astype(int)
            zoomed_mask = self.zoom_mask(enemy_mask, zoom_factor)
            x, y = self.world_cords_to_viewport_cords(enemy.x, enemy.z)

            return self.jr.render_at_clipped(raster, x, self.consts.ENEMY_POS_Y, zoomed_mask)
        
        def enemy_inactive(enemy):

            def render_death(enemy):
                # choose frame of death animation based on death_anim_counter
                n = enemy.death_anim_counter
                index = jnp.where(
                    n >= 0.8 * self.consts.ENEMY_DEATH_ANIM_LENGTH, 
                    0, # index 0 for first 20% of the animation
                    jnp.where(
                        n >= 0.4 * self.consts.ENEMY_DEATH_ANIM_LENGTH, 
                        1, # index 1 for second 40% of the animation
                        2 # index 2 for last 40% of the animation
                    )
                ) # if it works it works
                mask = self.enemy_explosion_mask[index]

                # apply zoom based on distance
                zoom_factor = ((jnp.sqrt(jnp.square(enemy.x) + jnp.square(enemy.z)) - 20.0) *
                               self.consts.DISTANCE_TO_ZOOM_FACTOR_CONSTANT).astype(int)
                zoomed_mask = self.zoom_mask(mask, zoom_factor)
                x, y = self.world_cords_to_viewport_cords(enemy.x, enemy.z)

                return self.jr.render_at_clipped(raster, x, self.consts.ENEMY_POS_Y, zoomed_mask)
            
            def _pass(_):
                return raster

            # render death animation if death_anim_counter > 0, else render nothing
            return jax.lax.cond(enemy.death_anim_counter==0, _pass, render_death, enemy)

        return jax.lax.cond(enemy.active, enemy_active, enemy_inactive, enemy)


    def render_single_projectile(self, raster, projectile: Projectile):
        def projectile_active(projectile):
            projectile_mask_index = jnp.where(projectile.distance <= 15,0,1)
            projectile_mask = self.projectile_masks[projectile_mask_index]
            x, y = self.world_cords_to_viewport_cords(projectile.x, projectile.z)
            return self.jr.render_at_clipped(raster, x+projectile_mask_index, y, projectile_mask)
        def projectile_inactive(_):
            return raster

        render_condition = jnp.all(jnp.stack([projectile.active,
                                              projectile.z >= self.consts.HITBOX_SIZE,
                                              projectile.distance <= self.consts.RADAR_MAX_SCAN_RADIUS]))
        return jax.lax.cond(render_condition, projectile_active, projectile_inactive, projectile)


    def render_targeting_indicator(self, raster, state):

        # determine if pointing at enemy
        within_x = jnp.logical_and(state.enemies.x <= 0,
                       state.enemies.x >= -2*self.consts.ENEMY_HITBOX_SIZE)  # TODO: for blue tank this is too large and for saucer too small (enemy hitboxes don't fit sprites in general, doesn't scale with distance?)
        in_front = state.enemies.z > 0
        overlaps = jnp.logical_and(jnp.logical_and(state.enemies.active, within_x), in_front)
        pointing_at_enemy = jnp.any(overlaps)

        color_id = jnp.where(pointing_at_enemy,
                             jnp.array(self.COLOR_TO_ID[self.consts.TARGET_INDICATOR_COLOR_ACTIVE], dtype=jnp.int32),
                             jnp.array(self.COLOR_TO_ID[self.consts.TARGET_INDICATOR_COLOR_INACTIVE], dtype=jnp.int32)
                            )

        return BattlezoneRenderer._draw_line(
            raster, 
            self.consts.TARGET_INDICATOR_POS_X, 
            self.consts.TARGET_INDICATOR_POS_Y, 
            self.consts.TARGET_INDICATOR_POS_X, 
            self.consts.TARGET_INDICATOR_POS_Y + 6, 
            color_id
        )


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        #-----------------background
        raster = self.jr.create_object_raster(self.BACKGROUND)


        def normal_render(raster):
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


            #-------------------------------enemies-----------------
            def render_single_enemy_wrapped(raster, enemy): #so that i dont have to pass self
                return self.render_single_enemy(raster, enemy), None


            raster, _ = jax.lax.scan(render_single_enemy_wrapped, raster, state.enemies)

            #------------------------------projectiles---------------
            raster = self.render_single_projectile(raster, state.player_projectile)
            #raster = jax.lax.cond(state.step_counter%2==0, self.render_single_projectile, lambda r, _: r,
                         #raster, state.player_projectile)
            # probably more accurate but looks stoopid because different frame rates
            def render_single_projectile_wrapped(raster, projectile): #so that i dont have to pass self
                return self.render_single_projectile(raster, projectile), None

            raster, _ = jax.lax.scan(render_single_projectile_wrapped, raster, state.enemy_projectiles)





            # -------------------------foreground---------------------------------------------------------------
            tank_mask = self.SHAPE_MASKS["tank"]
            raster = self.jr.render_at(raster, self.consts.TANK_SPRITE_POS_X,
                                       self.consts.TANK_SPRITE_POS_Y, tank_mask)

            raster = self._render_radar(raster, state, self.consts.RADAR_CENTER_X, self.consts.RADAR_CENTER_Y,
                                        self.consts.RADAR_RADIUS, self.COLOR_TO_ID[self.consts.RADAR_COLOR_1],
                                        self.COLOR_TO_ID[self.consts.RADAR_COLOR_2])

            # --------------chains---------
            chains_l_mask = self.SHAPE_MASKS["chainsLeft"]
            color_shifted_chain_l = self._scroll_chain_colors(chains_l_mask, state.chains_l_anim_counter)
            raster = self.jr.render_at(raster, self.consts.CHAINS_L_POS_X,
                                       self.consts.CHAINS_POS_Y, color_shifted_chain_l)

            chains_r_mask = self.SHAPE_MASKS["chainsRight"]
            color_shifted_chain_r = self._scroll_chain_colors(chains_r_mask, state.chains_r_anim_counter)
            raster = self.jr.render_at(raster, self.consts.CHAINS_R_POS_X,
                                       self.consts.CHAINS_POS_Y, color_shifted_chain_r)
            
            # ------------------ target indicator -----------------
            raster = self.render_targeting_indicator(raster, state)

            return raster

        def death_render(raster):
            raster = self.jr.render_at(raster, 0, 0, self.SHAPE_MASKS["blackscreen"])
            return raster

        raster = jax.lax.cond(state.death_anim_counter<=0, normal_render, death_render, raster)


        # ----------------life---------------------------
        life_mask = self.SHAPE_MASKS["life"]

        def render_single_life(i, raster):
            return self.jr.render_at(raster, self.consts.LIFE_POS_X + (self.consts.LIFE_X_OFFSET * i),
                                     self.consts.LIFE_POS_Y, life_mask)

        raster = jax.lax.fori_loop(0, state.life, render_single_life, raster)

        # ---------------------------player score--------------------
        # primarily taken from pong + changes
        player_digit_masks = self.SHAPE_MASKS["player_digits"]  # Assumes single color
        player_digits = self.jr.int_to_digits(state.score, max_digits=8)
        # this does not correctly work currently (only when max_digits==exactly amount digits)

        raster = self.jr.render_label_selective(raster, self.consts.SCORE_POS_X, self.consts.SCORE_POS_Y, player_digits,
                                                player_digit_masks, 0, 8,
                                                spacing=6, max_digits_to_render=8)
        # best highscore i can find is 6 digits
        # --------------------------------------------------------------------
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