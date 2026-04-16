import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import os
import jax.lax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from enum import IntEnum, unique
from typing import NamedTuple, Tuple
from functools import partial
from flax import struct
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from gymnasium.utils import play

#------------------------named Tuples---------------------------
class EnemyType(IntEnum):
    TANK = 0
    SAUCER = 1
    FIGHTER_JET = 2
    SUPERTANK = 3


class BattlezoneConstants(struct.PyTreeNode):
    # --- rendering: positions ---
    WIDTH: int = struct.field(default=160, pytree_node=False)
    HEIGHT: int = struct.field(default=210, pytree_node=False)
    WALL_TOP_Y: int = struct.field(default=0, pytree_node=False)
    WALL_TOP_HEIGHT: int = struct.field(default=36, pytree_node=False)
    WALL_BOTTOM_Y: int = struct.field(default=177, pytree_node=False)
    WALL_BOTTOM_HEIGHT: int = struct.field(default=33, pytree_node=False)
    TANK_SPRITE_POS_X: int = struct.field(default=43, pytree_node=False)
    TANK_SPRITE_POS_Y: int = struct.field(default=140, pytree_node=False)
    TARGET_INDICATOR_POS_X: int = struct.field(default=80, pytree_node=False)
    TARGET_INDICATOR_POS_Y: int = struct.field(default=77, pytree_node=False)
    CHAINS_POS_Y: int = struct.field(default=158, pytree_node=False)
    CHAINS_L_POS_X: int = struct.field(default=19, pytree_node=False)
    CHAINS_R_POS_X: int = struct.field(default=109, pytree_node=False)
    MOUNTAINS_Y: int = struct.field(default=36, pytree_node=False)
    GRASS_BACK_Y: int = struct.field(default=95, pytree_node=False)
    HORIZON_Y: int = struct.field(default=92, pytree_node=False)
    GRASS_FRONT_Y: int = struct.field(default=137, pytree_node=False)
    RADAR_CENTER_X: int = struct.field(default=80, pytree_node=False)
    RADAR_CENTER_Y: int = struct.field(default=18, pytree_node=False)
    RADAR_RADIUS: int = struct.field(default=10, pytree_node=False)
    LIFE_POS_X: int = struct.field(default=64, pytree_node=False)
    LIFE_POS_Y: int = struct.field(default=189, pytree_node=False)
    LIFE_X_OFFSET: int = struct.field(default=8, pytree_node=False)
    SCORE_POS_X: int = struct.field(default=89, pytree_node=False)
    SCORE_POS_Y: int = struct.field(default=179, pytree_node=False)
    ENEMY_POS_Y: int = struct.field(default=85, pytree_node=False)

    # --- rendering: colors ---
    SCORE_COLOR: Tuple[int, int, int] = struct.field(default=(26,102,26), pytree_node=False)
    TARGET_INDICATOR_COLOR_ACTIVE: Tuple[int, int, int] = struct.field(default=(255,255,0), pytree_node=False)
    TARGET_INDICATOR_COLOR_INACTIVE: Tuple[int, int, int] = struct.field(default=(0,0,0), pytree_node=False)
    CHAINS_COL_1: Tuple[int, int, int] = struct.field(default=(111,111,111), pytree_node=False)
    CHAINS_COL_2: Tuple[int, int, int] = struct.field(default=(74,74,74), pytree_node=False)
    RADAR_COLOR_1: Tuple[int, int, int] = struct.field(default=(111,210,111), pytree_node=False)
    RADAR_COLOR_2: Tuple[int, int, int] = struct.field(default=(236,236,236), pytree_node=False)
    LIFE_SCORE_COLOR: Tuple[int, int, int] = struct.field(default=(45,129,105), pytree_node=False)

    # --- world movement ---
    WORLD_SIZE_X: int = struct.field(default=256, pytree_node=False)
    WORLD_SIZE_Z: int = struct.field(default=256, pytree_node=False)
    PLAYER_ROTATION_SPEED: float = struct.field(default=jnp.pi/134, pytree_node=False)
    PLAYER_SPEED: float = struct.field(default=0.5, pytree_node=False)
    PLAYER_SPEED_DRIVETURN: float = struct.field(default=0.115348, pytree_node=False)
    PROJECTILE_SPEED: float = struct.field(default=1, pytree_node=False)
    ENEMY_SPEED: chex.Array = struct.field(default_factory=lambda: jnp.array([0.5, 0.5, 2.0, 0.5]))
    ENEMY_ROT_SPEED: chex.Array = struct.field(default_factory=lambda: jnp.array([jnp.pi/512, 0.02, 0.02, 0.02]))

    # --- game mechanics ---
    HITBOX_SIZE: float = struct.field(default=6.0, pytree_node=False)
    ENEMY_HITBOX_SIZE: float = struct.field(default=4.5, pytree_node=False)
    ENEMY_SPAWN_PROBS: chex.Array = struct.field(default_factory=lambda: jnp.array([
        # T    S    F   ST
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.6, 0.3, 0.1, 0.0],
        [0.4, 0.2, 0.2, 0.2]
    ]))
    RADAR_MAX_SCAN_RADIUS: int = struct.field(default=110, pytree_node=False)
    SAUCER_MIN_DIST: float = struct.field(default=27.0, pytree_node=False)
    FIGHTER_AREA_X: Tuple[float, float] = struct.field(default=(-12.5, 12.5), pytree_node=False)
    FIGHTER_AREA_Z: Tuple[float, float] = struct.field(default=(75.0, 126.0), pytree_node=False)
    FIGHTER_SLOW_DOWN_DISTANCE: float = struct.field(default=48.0, pytree_node=False)
    FIGHTER_SHOOTING_DISTANCE: float = struct.field(default=30.0, pytree_node=False)
    TANKS_SHOOTING_DISTANCE: float = struct.field(default=31.0, pytree_node=False)
    FIGHTER_DESPAWN_FRAMES: int = struct.field(default=12, pytree_node=False)

    # --- timing ---
    FIRE_CD: int = struct.field(default=57, pytree_node=False)
    PROJECTILE_TTL: int = struct.field(default=55, pytree_node=False)
    DEATH_ANIM_LENGTH: int = struct.field(default=15, pytree_node=False)
    ENEMY_DEATH_ANIM_LENGTH: int = struct.field(default=15, pytree_node=False)
    ENEMY_SHOOT_CDS: chex.Array = struct.field(default_factory=lambda: jnp.array([200, 200, 200, 200]))

    # --- misc ---
    RADAR_ROTATION_SPEED: float = struct.field(default=-0.05, pytree_node=False)
    DISTANCE_TO_ZOOM_FACTOR_CONSTANT: float = struct.field(default=0.05, pytree_node=False)
    ENEMY_SCORES: chex.Array = struct.field(
            default_factory=lambda: jnp.array([1000, 5000, 2000, 3000], dtype=jnp.int32))
    CAMERA_FOCAL_LENGTH: float = struct.field(default=180, pytree_node=False)
    ENEMY_WIDTHS: chex.Array = struct.field(
            default_factory=lambda: jnp.array([24, 32, 32, 24], dtype=jnp.int32))
    ENEMY_HEIGHTS: chex.Array = struct.field(
            default_factory=lambda: jnp.array([14, 18, 17, 14], dtype=jnp.int32))


@struct.dataclass
class Projectile:
    """Class holding projectiles. properties are arrays."""
    x: chex.Array
    z: chex.Array
    orientation_angle: chex.Array
    active: chex.Array
    distance: chex.Array
    time_to_live: chex.Array


@struct.dataclass
class Enemy:
    x: chex.Array
    z: chex.Array
    distance: chex.Array
    enemy_type: chex.Array
    orientation_angle: chex.Array  # 0 = towards positive z
    active: chex.Array
    death_anim_counter: chex.Array
    shoot_cd: chex.Array
    phase: chex.Array
    dist_moved_temp: chex.Array
    # points used for movement behaviour
    point_store_1_temp: chex.Array 
    point_store_2_temp: chex.Array


# immutable state container
@struct.dataclass
class BattlezoneState:
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


@struct.dataclass
class BattlezoneObservation:
    enemies: ObjectObservation
    radar_dots: ObjectObservation
    projectiles: ObjectObservation
    score: jnp.ndarray
    life: jnp.ndarray
    enemy_types: jnp.ndarray


@struct.dataclass
class BattlezoneInfo:
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
            action == Action.UPLEFTFIRE,
            action == Action.UPRIGHTFIRE,
            action == Action.DOWNLEFTFIRE,
            action == Action.DOWNRIGHTFIRE,
        ]), axis=0)
        
        direction = jnp.stack([noop, up, right, left, down, upRight, upLeft, downRight, downLeft])  # leave order as is!

        #-------------------fire--------------
        will_fire = jnp.logical_and(wants_fire, state.cur_fire_cd <= 0)
        
        def fire_projectile(state: BattlezoneState):
            return state.replace(
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

        new_state = jax.lax.cond(will_fire, fire_projectile, lambda s: s, state)

        #--------------------anims--------------------
        chain_r_offset = (-jnp.where(jnp.any(jnp.stack([upLeft, up, left])), 1.0, 0.0)
                          +jnp.where(jnp.any(jnp.stack([right, down, downRight])), 1.0, 0.0)
                          -jnp.where(upRight, 0.7, 0.0) + jnp.where(downLeft, 0.7, 0.0))
                            # i love magic numbers
        chain_l_offset = (-jnp.where(jnp.any(jnp.stack([upRight, up, right])), 1.0, 0.0)
                          + jnp.where(jnp.any(jnp.stack([left, down, downLeft])), 1.0, 0.0)
                          - jnp.where(upLeft, 0.7, 0.0) + jnp.where(downRight, 0.7, 0.0))
        mountains_offset = (jnp.where(jnp.any(jnp.stack([left, upLeft, downRight])), 1.0, 0.0)
                            -jnp.where(jnp.any(jnp.stack([right, upRight, downLeft])), 1.0, 0.0))
        grass_offset = (jnp.where(jnp.any(jnp.stack([up, upLeft, upRight])), 1.0, 0.0)
                        - jnp.where(jnp.any(jnp.stack([down, downRight, downLeft])), 1.0, 0.0))

        #--------------------update positions based on player movement-------------------
        updated_enemies = jax.vmap(self._enemy_player_position_update, in_axes=(0, None))(state.enemies, direction)
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

        return new_state.replace(
            chains_l_anim_counter=(state.chains_l_anim_counter + chain_l_offset) % 32,
            chains_r_anim_counter=(state.chains_r_anim_counter + chain_r_offset) % 32,
            mountains_anim_counter=(state.mountains_anim_counter + mountains_offset * 5) % 160,
            grass_anim_counter=(state.grass_anim_counter + grass_offset) % 30,
            radar_rotation_counter=state.radar_rotation_counter,
            enemies=updated_enemies,
            player_projectile=new_player_projectile,
            enemy_projectiles=updated_projectiles
        )


    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: BattlezoneState) -> BattlezoneState:

        d_anim_counter = state.enemies.death_anim_counter
        new_death_anim_counter = jnp.where(d_anim_counter > 0, d_anim_counter - 1, d_anim_counter)
        new_enemies, new_projectiles = (jax.vmap(self.enemy_movement, in_axes=(0, 0))
                                        (state.enemies, state.enemy_projectiles))
        shoot_cd = new_enemies.shoot_cd
        new_shoot_cd = jnp.where(shoot_cd > 0, shoot_cd - 1, shoot_cd)

        return state.replace(
            enemies=new_enemies.replace(
                death_anim_counter=new_death_anim_counter, shoot_cd=new_shoot_cd
            ),
            enemy_projectiles=new_projectiles
        )


    @partial(jax.jit, static_argnums=(0,))
    def _single_projectile_step(self, projectile: Projectile) -> Projectile:
        """implements movement of projectiles"""

        dir_x = -jnp.sin(projectile.orientation_angle)
        dir_z = jnp.cos(projectile.orientation_angle)
        new_x = projectile.x - dir_x*self.consts.PROJECTILE_SPEED
        new_z = projectile.z - dir_z*self.consts.PROJECTILE_SPEED

        return projectile.replace(
            x=new_x,
            z=new_z,
            time_to_live=jnp.where(projectile.time_to_live>0, projectile.time_to_live-1, 0),
            active=jnp.logical_and(projectile.active, projectile.time_to_live>0)
        )


    def _player_projectile_collision_step(self, state: BattlezoneState):
        hit_arr = (jax.vmap(self._player_projectile_collision_check, in_axes=(0, None))
                   (state.enemies, state.player_projectile))
        
        def _score_func(state1: BattlezoneState, in_tuple):
            enemy, hit = in_tuple
            new_score = state1.score + jnp.where(hit, self.consts.ENEMY_SCORES[enemy.enemy_type], 0)

            return state1.replace(score=new_score), None

        new_state, _ = jax.lax.scan(_score_func, state, (state.enemies, hit_arr))
        new_enemies_active = jnp.logical_and(new_state.enemies.active, jnp.invert(hit_arr))
        new_enemies_death_anim_counter = jnp.where(hit_arr,
                            self.consts.ENEMY_DEATH_ANIM_LENGTH, new_state.enemies.death_anim_counter)
        new_player_projectile_active = jnp.logical_and(new_state.player_projectile.active, jnp.invert(jnp.any(hit_arr)))

        return new_state.replace(
            enemies=new_state.enemies.replace(
                active=new_enemies_active,
                death_anim_counter=new_enemies_death_anim_counter
            ),
            player_projectile=new_state.player_projectile.replace(
                active=new_player_projectile_active
            )
        )

    def _enemy_friendly_fire_step(self, state: BattlezoneState):
        hit_matrix = jax.vmap(
            lambda enemy: jax.vmap(
                self._player_projectile_collision_check,
                in_axes=(None, 0)
            )(enemy, state.enemy_projectiles)
        )(state.enemies)
        hit_arr = jnp.any(hit_matrix, axis=1)
        new_enemies_active = jnp.logical_and(state.enemies.active, jnp.invert(hit_arr))
        new_enemies_death_anim_counter = jnp.where(
            hit_arr,
            self.consts.ENEMY_DEATH_ANIM_LENGTH,
            state.enemies.death_anim_counter
        )
        projectile_hit_any_enemy = jnp.any(hit_matrix, axis=0)
        new_enemy_projectiles_active = jnp.logical_and(
            state.enemy_projectiles.active,
            jnp.invert(projectile_hit_any_enemy)
        )
        return state.replace(
            enemies=state.enemies.replace(
                active=new_enemies_active,
                death_anim_counter=new_enemies_death_anim_counter
            ),
            enemy_projectiles=state.enemy_projectiles.replace(
                active=new_enemy_projectiles_active
            )
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
    def step(
            self, 
            state: BattlezoneState, 
            action: chex.Array
        ) -> Tuple[BattlezoneObservation, BattlezoneState, float, bool, BattlezoneInfo]:

        previous_state = state

        def normal_step(state):
            # update counters
            new_state = state.replace(
                step_counter=state.step_counter + 1, 
                cur_fire_cd=state.cur_fire_cd - 1,
                radar_rotation_counter=(state.radar_rotation_counter + self.consts.RADAR_ROTATION_SPEED) % 360
            )

            #-------------------projectiles-------------
            # move player projectile forwards
            new_player_projectile = self._single_projectile_step(state.player_projectile)
            new_state = new_state.replace(player_projectile=new_player_projectile)
            # check whether player projectile hit an enemy
            new_state = self._player_projectile_collision_step(new_state)
            new_state = self._enemy_friendly_fire_step(new_state)

            new_enemy_projectiles = jax.vmap(self._single_projectile_step, 0)(state.enemy_projectiles)
            new_state = new_state.replace(enemy_projectiles=new_enemy_projectiles)
            player_hit = jnp.any(jax.vmap(self._enemy_projectile_collision_check, 0)(new_state.enemy_projectiles))
            new_state = new_state.replace(death_anim_counter=jnp.where(player_hit,
                                                jnp.array(self.consts.DEATH_ANIM_LENGTH, dtype=jnp.int32),
                                                new_state.death_anim_counter))
            #------------------------------------------

            #-------------------spawn-------------------
            split_key, key = jax.random.split(new_state.random_key, 2)
            new_state = new_state.replace(random_key=key)
            new_state = new_state.replace(enemies=jax.vmap(self.spawn_enemy, in_axes=(0, 0, None, None))
                (jax.random.split(split_key,new_state.enemies.active.shape[0]), new_state.enemies, new_state.score, new_state))
            #-------------------------------------------

            new_state = self._player_step(new_state, action)
            new_state = self._enemy_step(new_state)

            return new_state


        def death_step(state):
            new_death_counter = state.death_anim_counter - 1
            new_state = state.replace(death_anim_counter=new_death_counter)
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
        (now only for projectiles, enemies have their own version because of the stored points)
        """
        # get offset to add to coords to get updated position
        offset_x, offset_z = self._position_update(obj.x, obj.z, obj.active, player_direction)

        # update object position
        new_x = obj.x + offset_x
        new_z = obj.z + offset_z

        return obj.replace(
            x=self._wrap_coord(new_x, self.consts.WORLD_SIZE_X), 
            z=self._wrap_coord(new_z, self.consts.WORLD_SIZE_Z), 
            distance=self._get_distance(new_x, new_z)
        )
    

    def _enemy_player_position_update(self, obj: Enemy, player_direction) -> Enemy:
        """
        _position_update version for named tuples that contain x, z, distance
        """
        # get offset to add to coords to get updated position
        offset_x, offset_z = self._position_update(obj.x, obj.z, obj.active, player_direction)

        # update object position
        new_x = obj.x + offset_x
        new_z = obj.z + offset_z
        
        obj = obj.replace(
            x=new_x, 
            z=new_z, 
            distance=self._get_distance(new_x, new_z),
            # also need to adapt stored points for saucer movement (but not world-wrap these necessarily, only do that when obj position wraps)
            point_store_1_temp=obj.point_store_1_temp + jnp.array([offset_x, offset_z], dtype=jnp.float32),
            point_store_2_temp=obj.point_store_2_temp + jnp.array([offset_x, offset_z], dtype=jnp.float32)
        )

        obj = self._wrap_coords_and_stored_points(obj)

        return obj
    

    def _wrap_coord(self, coord, world_size):
        """Wraps a coordinate around the world edges."""
        return jax.lax.cond(
            coord < -world_size / 2.0, 
            lambda: coord + world_size,
            lambda: jax.lax.cond(
                coord > world_size / 2.0, 
                lambda: coord - world_size, 
                lambda: coord
            )
        )
        # maybe it's a numerical issue so try to avoid constant re-calculation
        # return ((coord + world_size / 2.0) % world_size) - (world_size / 2.0)


    def _wrap_coords_and_stored_points(self, enemy: Enemy):
        """
        world-wrap enemy coordinates. if wrap occured, also move stored points accordingly.
        """

        before_wrap_x = enemy.x
        before_wrap_z = enemy.z

        after_wrap_x = self._wrap_coord(before_wrap_x, self.consts.WORLD_SIZE_X)
        after_wrap_z = self._wrap_coord(before_wrap_z, self.consts.WORLD_SIZE_Z)

        # if the enemy position wraps, we need to wrap the stored points too
        def wrap_stored_points_x(enemy):
            wrap_direction = jnp.sign(before_wrap_x - after_wrap_x)  # + = right wrap, - = left wrap
            return enemy.replace(
                point_store_1_temp=enemy.point_store_1_temp.at[0].add(-wrap_direction * self.consts.WORLD_SIZE_X),
                point_store_2_temp=enemy.point_store_2_temp.at[0].add(-wrap_direction * self.consts.WORLD_SIZE_X)
            )

        def wrap_stored_points_z(enemy):
            wrap_direction = jnp.sign(before_wrap_z - after_wrap_z)  # + = up wrap, - = down wrap
            return enemy.replace(
                point_store_1_temp=enemy.point_store_1_temp.at[1].add(-wrap_direction * self.consts.WORLD_SIZE_Z),
                point_store_2_temp=enemy.point_store_2_temp.at[1].add(-wrap_direction * self.consts.WORLD_SIZE_Z)
            )

        # larger atol is fine since wrap creates difference of world_size
        enemy = jax.lax.cond(~jnp.isclose(before_wrap_x, after_wrap_x, atol=1.0), wrap_stored_points_x, lambda e: e, enemy)
        enemy = jax.lax.cond(~jnp.isclose(before_wrap_z, after_wrap_z, atol=1.0), wrap_stored_points_z, lambda e: e, enemy)
        enemy = enemy.replace(x=after_wrap_x, z=after_wrap_z, distance=self._get_distance(after_wrap_x, after_wrap_z))

        return enemy


    def _position_update(self, prev_x, prev_z, active, player_direction):
        """
        updates the x, z coordinates according to the current player movement
        """
        ###
        idx = jnp.argmax(player_direction)
        alpha = self.consts.PLAYER_ROTATION_SPEED
        speed = jax.lax.cond(idx > 4, lambda: self.consts.PLAYER_SPEED_DRIVETURN, lambda: self.consts.PLAYER_SPEED)
        ###
        sin_alpha = jnp.sin(alpha)
        cos_alpha = jnp.cos(alpha)
        # Changing position based on player movement
        # TODO: it would be better if we only computed the offset for the active movement, right?
        offset_xz = jnp.array([
            [0, 0],     # Noop
            [0, -speed],  # Up
            [(prev_x * cos_alpha - prev_z * sin_alpha) - prev_x, (prev_x * sin_alpha + prev_z * cos_alpha) - prev_z],  # Right
            [(prev_x * jnp.cos(-alpha) - prev_z * jnp.sin(-alpha)) - prev_x, (prev_x * jnp.sin(-alpha) + prev_z * jnp.cos(-alpha)) - prev_z],  # Left
            [0, speed],  # Down
            [(prev_x * cos_alpha - (prev_z - speed) * sin_alpha) - prev_x, (prev_x * sin_alpha + (prev_z - speed) * cos_alpha) - prev_z], # UpRight
            [(prev_x * cos_alpha + (prev_z - speed) * sin_alpha) - prev_x, (-prev_x * sin_alpha + (prev_z - speed) * cos_alpha) - prev_z],  # UpLeft
            [(prev_x * cos_alpha + (prev_z + speed) * sin_alpha) - prev_x, (-prev_x * sin_alpha + (prev_z + speed) * cos_alpha) - prev_z],  # DownRight
            [(prev_x * cos_alpha - (prev_z + speed) * sin_alpha) - prev_x, (prev_x * sin_alpha + (prev_z + speed) * cos_alpha) - prev_z]  # DownLeft
        ])
        
        offset = offset_xz[idx]

        return offset[0], offset[1]


    def _obj_player_rotation_update(self, obj: NamedTuple, angle_change):

        alpha = self.consts.PLAYER_ROTATION_SPEED
        dist = self._get_distance(obj.x, obj.z)
        opp = jnp.tan(alpha) * dist
        beta = jnp.atan(dist / opp)
        angle = ((jnp.pi/2) - beta) * angle_change

        return obj.replace(orientation_angle=(obj.orientation_angle - angle) % (2*jnp.pi))


    def _player_projectile_collision_check(self, enemies: Enemy, player_projectile: Projectile) -> bool:
        """checks whether a player projectile hit an enemy. player_projectiles contains arrays of size 1. uses square hitbox"""
        s = self.consts.ENEMY_HITBOX_SIZE
        distx = jnp.abs(enemies.x - player_projectile.x) <= s
        distz = jnp.abs(enemies.z - player_projectile.z) <= s

        return jnp.all(jnp.stack([distx, distz, enemies.active, player_projectile.active]))


    def _enemy_projectile_collision_check(self, obj: Projectile):
        """checks whether an enemy projectile hit the player. player hitbox is rectangle around origin: (x: -3, z: 0) to (x: 3, z: 6)"""
        s = self.consts.HITBOX_SIZE
        distx = jnp.abs(obj.x) <= s
        distz = (obj.z <= 2*s) & (obj.z >= 0)

        return jnp.all(jnp.stack([distx, distz, obj.active]))


    def _get_distance(self, x, z):
        distance = jnp.sqrt(x ** 2 + z ** 2)
        # Room for distance specific actions
        return distance


    @partial(jax.jit, static_argnums=(0,))
    def spawn_enemy(self, key, enemy: Enemy, score, state):

        def score_to_spawn_indx(score):
            threshold = jnp.array([1000, 2000, 7000, 12000])
            return jnp.sum(score >= threshold)

        def no_spawn(args):
            enemy, key, score = args
            return enemy


        def do_spawn(args):

            enemy, key, score = args
            spawn_probs_index = score_to_spawn_indx(score)
            spawn_probs = self.consts.ENEMY_SPAWN_PROBS[spawn_probs_index]
            key, k_dist, k_theta, k_type, k_orient = jax.random.split(key, 5)
            enemy_type = jax.random.choice(k_type, a=len(EnemyType), p=spawn_probs)

            def spawn_fighter():
                # Custom spawn logic for fighter jets
                # spawn in a specific area (FIGHTER_AREA_X, FIGHTER_AREA_Z)
                x = jax.random.uniform(k_dist, minval=self.consts.FIGHTER_AREA_X[0], maxval=self.consts.FIGHTER_AREA_X[1])
                z = jax.random.uniform(k_theta, minval=self.consts.FIGHTER_AREA_Z[0], maxval=self.consts.FIGHTER_AREA_Z[1])
                distance = self._get_distance(x, z)
                orientation_angle = jax.random.uniform(k_orient, minval=0.0, maxval=2*jnp.pi)

                return enemy.replace(
                    x=x,
                    z=z,
                    distance=distance,
                    enemy_type=enemy_type,
                    orientation_angle=orientation_angle,
                    active=True,
                    shoot_cd=self.consts.ENEMY_SHOOT_CDS[enemy_type],
                    phase=0,
                    dist_moved_temp=0.0,
                    point_store_1_temp=jnp.zeros((2,), dtype=jnp.float32),
                    point_store_2_temp=jnp.zeros((2,), dtype=jnp.float32),
                )

            def spawn_default():
                distance = jnp.sqrt(jax.random.uniform(k_dist, minval=.2, maxval=.9)) * self.consts.RADAR_MAX_SCAN_RADIUS
                theta = jax.random.uniform(k_theta, minval=0.0, maxval=2*jnp.pi)
                orientation_angle = jax.random.uniform(k_orient, minval=0.0, maxval=2*jnp.pi)
                return enemy.replace(
                    x=distance * jnp.cos(theta),
                    z=distance * jnp.sin(theta),
                    distance=distance,
                    enemy_type=enemy_type,
                    orientation_angle=orientation_angle,
                    active=True,
                    shoot_cd=self.consts.ENEMY_SHOOT_CDS[enemy_type],
                    phase=0,
                    dist_moved_temp=0.0,
                    point_store_1_temp=jnp.zeros((2,), dtype=jnp.float32),
                    point_store_2_temp=jnp.zeros((2,), dtype=jnp.float32),
                )

            # Use custom logic for fighter jets
            return jax.lax.cond(
                enemy_type == EnemyType.FIGHTER_JET,
                spawn_fighter,
                spawn_default
            )

        cond = jnp.logical_or(enemy.active, enemy.death_anim_counter > 0)
        cond = jnp.logical_or(cond, jnp.logical_and(score < 1000, state.shot_spawn == False))
        cond = jnp.any(cond)

        return jax.lax.cond(cond, no_spawn, do_spawn, (enemy, key, score))


    # -------------Enemy Movements-----------------
    @partial(jax.jit, static_argnums=(0,))
    def enemy_movement(self, enemy: Enemy, projectile: Projectile):

        perfect_angle = (2*jnp.pi - jnp.arctan2(enemy.x , enemy.z)) % (2*jnp.pi)
        angle_diff = (perfect_angle - enemy.orientation_angle + jnp.pi) % (2*jnp.pi) - jnp.pi
        speed = self.consts.ENEMY_SPEED[enemy.enemy_type]
        rot_speed = self.consts.ENEMY_ROT_SPEED[enemy.enemy_type]

        # ---------Helper Functions---------

        def move_to_direction(enemy: Enemy, angle: float, towards: int = -1) -> Enemy:
            """Enemy towards direction with -1 and away with 1"""

            direction_x = -jnp.sin(angle)
            direction_z = jnp.cos(angle)
            new_x = enemy.x + direction_x * towards * speed
            new_z = enemy.z + direction_z * towards * speed
            
            enemy = enemy.replace(x=new_x, z=new_z, distance=self._get_distance(new_x, new_z))
            enemy = self._wrap_coords_and_stored_points(enemy)

            return enemy
        
        def move_orthogonal_to_direction(enemy: Enemy, angle: float, direction: int = 1) -> Enemy:
            """Enemy right of direction with -1 and left with 1 (from direction's POV)"""

            ortho_angle = (angle + jnp.sign(direction) * (jnp.pi/2)) % (2*jnp.pi)
            direction_x = -jnp.sin(ortho_angle)
            direction_z = jnp.cos(ortho_angle)
            new_x = enemy.x + direction_x * speed * jnp.abs(direction)
            new_z = enemy.z + direction_z * speed * jnp.abs(direction)
            
            enemy = enemy.replace(x=new_x, z=new_z, distance=self._get_distance(new_x, new_z))
            enemy = self._wrap_coords_and_stored_points(enemy)

            return enemy

        def move_to_player(enemy: Enemy, towards: int = -1) -> Enemy:
            """Enemy towards player with -1 and away with 1"""            
            return move_to_direction(enemy, perfect_angle, towards)
        
        def move_orthogonal_to_player(enemy: Enemy, direction: int = 1) -> Enemy:
            """Enemy right of player with -1 and left with 1 (from player's POV)"""            
            return move_orthogonal_to_direction(enemy, perfect_angle, direction)

        def enemy_turn(enemy: Enemy) -> Enemy:
            return enemy.replace(orientation_angle=enemy.orientation_angle + jnp.sign(angle_diff) * rot_speed)

        def shoot_projectile(args) -> Tuple[Enemy, Projectile]:

            enemy, projectile = args
            new_enemy = enemy.replace(shoot_cd=self.consts.ENEMY_SHOOT_CDS[enemy.enemy_type])
            dist = jnp.sqrt(jnp.square(enemy.x)+ jnp.square(enemy.z))
            dir_x = enemy.x/dist
            dir_z = enemy.z / dist
            return new_enemy, projectile.replace(
                orientation_angle=perfect_angle,
                x = enemy.x - dir_x*(self.consts.ENEMY_HITBOX_SIZE+1),
                z = enemy.z - dir_z*(self.consts.ENEMY_HITBOX_SIZE+1),
                active=True,
                time_to_live=jnp.array(self.consts.PROJECTILE_TTL, dtype=jnp.int32)
            )


        # ---------------------------------
        ## Tank
        def tank_movement(tank: Enemy) -> Enemy:
            def player_spotted(tank):

                def too_close(tank):
                    return move_to_player(tank, 1)  # Enemy keeps 30 units distance to player
                
                def too_far(tank):
                    return move_to_player(tank, -1)

                return jax.lax.cond(tank.distance <= 30.0, too_close, too_far, tank)
            
            def player_not_spotted(tank):
                return enemy_turn(tank)

            return jax.lax.cond(jnp.abs(angle_diff) <= rot_speed,
                                player_spotted, player_not_spotted, tank)


        def saucer_movement(saucer: Enemy) -> Enemy:
            # TODO: try reflecting roattion center when going from + to - x etc or add a separate strafing mechanism that resets phases

            def near_player(saucer):
                """movement of the saucer when too near to the player"""
                saucer = saucer.replace(phase=0)

                # strafe speed is maximal when player points at middle of hitbox, minimal at edges, quadratic relation
                max_strafe_speed = 2.0 * speed * 2
                min_strafe_speed = 1.0 * speed * 2
                strafe_speed = (
                    (min_strafe_speed - max_strafe_speed) / (self.consts.ENEMY_HITBOX_SIZE ** 2)
                ) * (saucer.x ** 2) + max_strafe_speed

                # saucer tries to stay outside the player's aim, +-1 unit buffer
                saucer = jax.lax.cond(
                    (saucer.x > -self.consts.ENEMY_HITBOX_SIZE - 1.0) 
                    & (saucer.x < self.consts.ENEMY_HITBOX_SIZE + 1.0) 
                    & (saucer.z > 0.0), 
                    lambda x: move_orthogonal_to_player(x, -jnp.sign(saucer.x) * strafe_speed), 
                    lambda x: x,
                    saucer
                )

                # strafing is additional to moving away from player
                saucer = move_to_player(saucer, 1)
                
                return saucer
            
            def far_player(saucer):
                """movement of the saucer when far enough away from the player"""

                def p0(saucer):
                    saucer = saucer.replace(dist_moved_temp=saucer.dist_moved_temp + speed)

                    # saucer tries to stay outside the player's aim.
                    saucer = jax.lax.cond(
                        (saucer.x > -self.consts.ENEMY_HITBOX_SIZE - 1.0) 
                        & (saucer.x < self.consts.ENEMY_HITBOX_SIZE + 1.0) 
                        & (saucer.z > 0.0), 
                        lambda x: move_orthogonal_to_player(x, -jnp.sign(saucer.x) * 0.5), 
                        lambda x: move_to_player(x, 0.5),
                        saucer
                    )
                    saucer = move_to_player(saucer, 0.5)

                    def next_phase(saucer):
                        saucer = saucer.replace(dist_moved_temp=0.0)
                        saucer = saucer.replace(phase=1)
                        return saucer

                    def cont_move(saucer):
                        return saucer

                    return jax.lax.cond(saucer.dist_moved_temp > 11.8, next_phase, cont_move, saucer)

                def p1(saucer: Enemy):
                    # Rotation center
                    ux = saucer.x / saucer.distance
                    uz = saucer.z / saucer.distance
                    rot_centre_x = saucer.x + (9.22 * ux) + (4.33 * uz)
                    rot_centre_z = saucer.z + (9.22 * uz) - (4.33 * ux)
                    # Intersection circle and line
                    ## Component for abc
                    radius = jnp.sqrt((saucer.x - rot_centre_x)**2 + (saucer.z - rot_centre_z)**2)
                    m = saucer.z / saucer.x
                    a = 1 + m**2
                    b = (-2) * (rot_centre_x + m * rot_centre_z)
                    c = rot_centre_x**2 + rot_centre_z**2 - radius**2
                    ## abc
                    x_1 = ((-b) + jnp.sqrt(b**2 - 4*a*c)) / (2*a)
                    x_2 = ((-b) - jnp.sqrt(b**2 - 4*a*c)) / (2*a)
                    ## z values
                    z_1 = m*x_1
                    z_2 = m*x_2
                    ## furthest point from origin
                    dist_sq1 = x_1**2 + z_1**2
                    dist_sq2 = x_2**2 + z_2**2
                    ## Set x and z
                    intersection_point_x = jnp.where(dist_sq1 > dist_sq2, x_1, x_2)
                    intersection_point_y = jnp.where(dist_sq1 > dist_sq2, z_1, z_2)
                    # Update saucer
                    saucer = saucer.replace(point_store_1_temp=jnp.array([rot_centre_x, rot_centre_z]))    # Centre of rotation
                    saucer = saucer.replace(point_store_2_temp=jnp.array([intersection_point_x, intersection_point_y]))    # Stop point for rotation
                    
                    return saucer.replace(phase=2)

                def p2(saucer):
                    rot_centre_x = saucer.point_store_1_temp[0]
                    rot_centre_z = saucer.point_store_1_temp[1]
                    end_rot_point = saucer.point_store_2_temp

                    vx = saucer.x - rot_centre_x
                    vz = saucer.z - rot_centre_z
                    ## Radius
                    rad = jnp.sqrt(vx**2 + vz**2)
                    ##
                    dtheta = speed / rad
                    ##
                    cos_dtheta = jnp.cos(dtheta)
                    sin_dtheta = jnp.sin(dtheta)
                    vx_new = cos_dtheta * vx - sin_dtheta * vz
                    vz_new = sin_dtheta * vx + cos_dtheta * vz
                    # Update saucer
                    new_x = rot_centre_x + vx_new
                    new_z = rot_centre_z + vz_new
                    saucer = saucer.replace(x=new_x, z=new_z)

                    saucer = self._wrap_coords_and_stored_points(saucer)

                    def next_phase(saucer):
                        return saucer.replace(
                            point_store_1_temp=jnp.zeros((2,), dtype=jnp.float32),
                            point_store_2_temp=jnp.zeros((2,), dtype=jnp.float32),
                            dist_moved_temp=0.0,
                            phase=0
                        )

                    def cont_circular(saucer):
                        return saucer

                    reach_end_point = jnp.isclose(jnp.array([saucer.x, saucer.z]), end_rot_point, atol=1)
                    reach_end_point = jnp.all(reach_end_point)

                    return jax.lax.cond(reach_end_point, next_phase, cont_circular, saucer)

                return jax.lax.switch(saucer.phase, (p0, p1, p2), saucer)

            def despawn(saucer):
                return saucer.replace(active=False)
            
            # movement behaviour
            saucer = jax.lax.cond(saucer.distance < self.consts.SAUCER_MIN_DIST, near_player, far_player, saucer)

            # despawn if too far away
            saucer = jax.lax.cond(
                saucer.distance > jnp.sqrt((self.consts.WORLD_SIZE_X / 2)**2 + (self.consts.WORLD_SIZE_Z / 2)**2) * 0.88, 
                despawn, 
                lambda x: x, 
                saucer
            )

            return saucer


        def fighter_movement(fighter: Enemy) -> Enemy:
            
            def p0(fighter: Enemy):
                """right after spawning: setup"""
                return fighter.replace(
                    phase=1, 
                    # abuse dist_moved_temp as directional indicator
                    dist_moved_temp=jnp.sign(fighter.x),
                    point_store_1_temp=jnp.array([0.0, fighter.z]),
                    point_store_2_temp=jnp.array([0.0, 0.0]),
                    orientation_angle=jnp.pi,  # parallel to z axis facing player at spawn time
                    shoot_cd=0
                )

            def p1(fighter):
                """diagonal movement"""
                # do diagonal step in direction dependent on dist_moved_temp with speed dependent on distance
                speed_multiplier = jnp.where(fighter.distance < self.consts.FIGHTER_SLOW_DOWN_DISTANCE, 0.5, 1.0)
                # go 1/sqrt(2) in both directions to achieve diagonal movement of one unit
                fighter = move_to_direction(fighter, fighter.orientation_angle, speed_multiplier / jnp.sqrt(2.0))
                fighter = move_orthogonal_to_direction(fighter, fighter.orientation_angle, -fighter.dist_moved_temp * speed_multiplier / jnp.sqrt(2.0))
                # find orthogonal distance from fighter to the line connecting point_store_2_temp and point_store_1_temp
                # Let A = point_store_2_temp, B = point_store_1_temp, P = (fighter.x, fighter.z)
                A = fighter.point_store_2_temp
                B = fighter.point_store_1_temp
                P = jnp.array([fighter.x, fighter.z])
                BA = B - A
                PA = P - A
                # Compute orthogonal distance from P to line AB
                orth_dist = -(BA[0] * PA[1] - BA[1] * PA[0]) / jnp.sqrt(BA[0]**2 + BA[1]**2 + 1e-8)

                AB = B - A
                AP = P - A
                den = jnp.sqrt(AB[0]**2 + AB[1]**2 + 1e-8)
                # signed distance to the line through A perpendicular to AB
                vert_dist = (AP[0] * AB[0] + AP[1] * AB[1]) / den

                # check out of bounds orthogonal direction: negate dist_moved_temp
                fighter = jax.lax.cond(  # if in bounds...
                    (orth_dist > self.consts.FIGHTER_AREA_X[0]) 
                    & (orth_dist < self.consts.FIGHTER_AREA_X[1]), 
                    lambda f: f, 
                    lambda f: f.replace(dist_moved_temp=jnp.sign(orth_dist)),
                    fighter
                )
                # check shooting range: shoot, set phase as counter, (also= go to pX)
                fighter = jax.lax.cond(
                    (fighter.distance <= self.consts.FIGHTER_SHOOTING_DISTANCE)
                    | (jnp.abs(vert_dist) < 5.0),  # despawn fallback
                    lambda f: f.replace(phase=self.consts.FIGHTER_DESPAWN_FRAMES + 10),
                    lambda f: f,
                    fighter
                )

                return fighter
                

            def pX(fighter):
                """keep going and despawn"""
                # do diagonal step with slow speed in direction dependent on dist_moved_temp
                fighter = move_to_direction(fighter, fighter.orientation_angle, -0.5 / jnp.sqrt(2.0))
                fighter = move_orthogonal_to_direction(fighter, fighter.orientation_angle, fighter.dist_moved_temp * 0.5 / jnp.sqrt(2.0))
                # decrement phase
                fighter = fighter.replace(phase=fighter.phase - 1)
                # check counter elapsed: despawn
                fighter = jax.lax.cond(
                    jnp.logical_and(fighter.phase <= 10, fighter.death_anim_counter==0),
                    lambda f: f.replace(active=False),
                    lambda f: f,
                    fighter
                )

                return fighter
            
            # for all phases >=2, pX is chosen
            return jax.lax.switch(fighter.phase, (p0, p1, pX), fighter)

        def supertank_movement(supertank: Enemy) -> Enemy:
            def p0(supertank):
                """Closing in on player"""
                def too_close(supertank):
                    return move_to_player(supertank, 1)

                def too_far(supertank):
                    return move_to_player(supertank, -1)

                supertank = jax.lax.cond(supertank.distance <= 30.0, too_close, too_far, supertank)
                supertank = jax.lax.cond(jnp.abs(supertank.distance - 30.0) < 2.5,
                                         lambda supertank: supertank.replace(phase=1, dist_moved_temp=0.0),
                                         lambda supertank: supertank,
                                         supertank)
                return supertank

            def p1(supertank):
                """90° clockwise"""
                dtheta = -speed / supertank.distance
                cos_dtheta = jnp.cos(dtheta)
                sin_dtheta = jnp.sin(dtheta)
                # Update saucer
                scale = self.consts.TANKS_SHOOTING_DISTANCE / supertank.distance   # Reduce circle the further to player
                new_x = (cos_dtheta * supertank.x - sin_dtheta * supertank.z) * scale
                new_z = (sin_dtheta * supertank.x + cos_dtheta * supertank.z) * scale
                supertank = supertank.replace(x=new_x, z=new_z, dist_moved_temp=supertank.dist_moved_temp+jnp.abs(dtheta))
                supertank = self._wrap_coords_and_stored_points(supertank)
                supertank = jax.lax.cond(
                    supertank.dist_moved_temp >= (jnp.pi / 2),
                    lambda supertank: supertank.replace(phase=2, dist_moved_temp=0.0),
                    lambda supertank: supertank,
                    supertank
                )
                return supertank

            def p2(supertank):
                """180° counterclockwise"""
                dtheta = speed / supertank.distance
                cos_dtheta = jnp.cos(dtheta)
                sin_dtheta = jnp.sin(dtheta)
                # Update saucer
                scale = self.consts.TANKS_SHOOTING_DISTANCE / supertank.distance   # Reduce circle the further to player
                new_x = (cos_dtheta * supertank.x - sin_dtheta * supertank.z) * scale
                new_z = (sin_dtheta * supertank.x + cos_dtheta * supertank.z) * scale
                supertank = supertank.replace(x=new_x, z=new_z, dist_moved_temp=supertank.dist_moved_temp+jnp.abs(dtheta))
                supertank = self._wrap_coords_and_stored_points(supertank)
                supertank = jax.lax.cond(
                    supertank.dist_moved_temp >= (jnp.pi),
                    lambda supertank: supertank.replace(phase=3, dist_moved_temp=0.0),
                    lambda supertank: supertank,
                    supertank
                )
                return supertank

            def p3(supertank):
                """Taking a shot"""
                supertank = supertank.replace(phase=0)

                return supertank

            return jax.lax.switch(supertank.phase, (p0, p1, p2, p3), supertank)

        shoot_cond = (jnp.all(jnp.array([enemy.enemy_type == EnemyType.TANK,
                                        enemy.shoot_cd <= 0,
                                        jnp.abs(angle_diff) <= rot_speed,
                                        enemy.distance <= self.consts.TANKS_SHOOTING_DISTANCE,
                                        enemy.active]))
                    | jnp.all(jnp.array([enemy.enemy_type == EnemyType.SUPERTANK,
                                        enemy.shoot_cd <= 0,
                                        #jnp.abs(angle_diff) <= rot_speed,
                                        enemy.phase == 3,
                                        enemy.active]))
                    | jnp.all(jnp.array([enemy.enemy_type == EnemyType.FIGHTER_JET,
                                         enemy.shoot_cd <= 0,
                                         enemy.distance <= self.consts.FIGHTER_SHOOTING_DISTANCE,
                                         enemy.active]))                
                    )
        
        new_enemy, new_projectile = jax.lax.cond(shoot_cond, shoot_projectile, lambda x: x,
                                                 (enemy, projectile))
        return (jax.lax.switch(enemy.enemy_type, (tank_movement, saucer_movement, fighter_movement, supertank_movement), new_enemy),
                new_projectile)

    # ---------------------------------------------


    def render(self, state: BattlezoneState) -> jnp.ndarray:
        return self.renderer.render(state)

    def player_shot_reset(self, state:BattlezoneState) -> BattlezoneState:
        """reset function for when the player was shot but still has remaining lives"""

        split_key, key = jax.random.split(state.random_key, 2)
        # Set enemies to inactive
        inactive_enemies = state.enemies.replace(active=jnp.zeros_like(state.enemies.active))

        new_state = state.replace(
            shot_spawn=jnp.ones_like(state.shot_spawn),
            life=state.life - 1,
            enemies=jax.vmap(
                self.spawn_enemy, 
                in_axes=(0, 0, None, None)
            )(
                jax.random.split(split_key, inactive_enemies.active.shape[0]), 
                inactive_enemies, 
                state.score, 
                state
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
            cur_fire_cd=jnp.array(0, dtype=jnp.int32),
        )

        return new_state


    def world_cords_to_viewport_cords_arr(self, x, z, f):

        u = ((f * (x / z)) + self.consts.WIDTH/2).astype(jnp.int32)
        vOffset = self.consts.HORIZON_Y
        v = ((f / (z - self.consts.HITBOX_SIZE)) + vOffset).astype(jnp.int32)

        return u, v


    def check_in_radar(self, enemies: Enemy) -> chex.Array:
        return jnp.logical_and((enemies.distance <= self.consts.RADAR_MAX_SCAN_RADIUS), enemies.active)


    def _get_observation(self, state: BattlezoneState):

        #-------------------------------enemies----------------------------------------------
        enemies_u, _ = self.world_cords_to_viewport_cords_arr(state.enemies.x, state.enemies.z,
                                                                  self.consts.CAMERA_FOCAL_LENGTH)
        zoom_factor = jnp.clip(((-0.15 * (state.enemies.distance) + 21.0) / 20.0), 0.0, 1.0)
        pixels_deleted_due_to_zoom = (jnp.round(1.0 / zoom_factor) + 1)
        enemies_width = self.consts.ENEMY_WIDTHS[state.enemies.enemy_type] - pixels_deleted_due_to_zoom
        enemies_heights = self.consts.ENEMY_HEIGHTS[state.enemies.enemy_type] - pixels_deleted_due_to_zoom
        enemies_visible = jnp.logical_and(
            state.enemies.z > 0,
            jnp.logical_and(
                enemies_u < (self.consts.WIDTH + enemies_width // 2),
                enemies_u > (0 - enemies_width // 2)
            )
        )
        enemy_mask = jnp.logical_and(state.enemies.active, enemies_visible)
        enemies_u = enemies_u - (enemies_width / 2)
        enemies = ObjectObservation.create(
            x=jnp.clip(enemies_u, 0, self.consts.WIDTH),
            y=jnp.clip(
                jnp.full(
                    (len(enemies_u),),
                    self.consts.ENEMY_POS_Y - (enemies_heights / 2)
                ), 
                0, 
                self.consts.HEIGHT
            ),
            width = enemies_width,
            height = enemies_heights,
            active = enemy_mask
        )

        #---------------------------------projectiles------------------------------------------------
        enemy_projectiles_u, enemy_projectiles_v = self.world_cords_to_viewport_cords_arr(
            state.enemy_projectiles.x,
            state.enemy_projectiles.z, 
            self.consts.CAMERA_FOCAL_LENGTH
        )
        enemy_projectiles_visible = jnp.logical_and(
            state.enemies.z > 0,
            jnp.logical_and(
                enemies_u < self.consts.WIDTH,
                enemies_u > 0
            )
        )
        enemy_projectiles_mask = jnp.logical_and(enemy_projectiles_visible, state.enemy_projectiles.active)
        player_projectiles_u, player_projectiles_v = self.world_cords_to_viewport_cords_arr(
            state.player_projectile.x,
            state.player_projectile.z, 
            self.consts.CAMERA_FOCAL_LENGTH
        )
        projectiles_x = jnp.concatenate([
            jnp.atleast_1d(player_projectiles_u - 1),
            enemy_projectiles_u - 1,
        ])
        projectiles_y = jnp.concatenate([
            jnp.atleast_1d(player_projectiles_v - 1),
            enemy_projectiles_v - 1
        ])
        projectiles_active = jnp.concatenate([jnp.atleast_1d(state.player_projectile.active), enemy_projectiles_mask])
        projectiles = ObjectObservation.create(
            x=jnp.clip(projectiles_x, 0, self.consts.WIDTH),
            y=jnp.clip(projectiles_y, 0, self.consts.HEIGHT),
            width=jnp.full(
                (len(projectiles_x),),
                2
            ),
            height=jnp.full(
                (len(projectiles_x),),
                3
            ),
            active=projectiles_active
        )

        #-----------------------------radar----------------------------------------
        # Check if enemy in radar radius
        in_radar = jax.vmap(self.check_in_radar, in_axes=(0))(state.enemies)

        # Scale to radar size
        scale_val = self.consts.RADAR_RADIUS / self.consts.RADAR_MAX_SCAN_RADIUS
        radar_enemies_x = state.enemies.x * scale_val
        radar_enemies_z = state.enemies.z * scale_val * (-1)

        # Offset to radar center
        radar_enemies_x = jnp.round(radar_enemies_x + self.consts.RADAR_CENTER_X).astype(jnp.int32)
        radar_enemies_z = jnp.round(radar_enemies_z + self.consts.RADAR_CENTER_Y).astype(jnp.int32)

        # Only allow in range enemies
        radar_dots = ObjectObservation.create(
            x=jnp.clip(radar_enemies_x, 0, self.consts.WIDTH),
            y=jnp.clip(radar_enemies_z, 0, self.consts.HEIGHT),
            width=jnp.full(
                (len(radar_enemies_x),),
                1
            ),
            height=jnp.full(
                (len(radar_enemies_x),),
                1
            ),
            active=in_radar
        )
        #----------------------------------------------------------------------------

        return BattlezoneObservation(
            enemies=enemies,
            radar_dots=radar_dots,
            projectiles=projectiles,
            score=jnp.array(state.score),
            life=jnp.array(state.life),
            enemy_types=jnp.where(enemy_mask, jnp.array(state.enemies.enemy_type), -1),
        )


    def observation_space(self) -> spaces.Dict:
        """description of observation (must match)"""
        object_space = spaces.get_object_space(n=2, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))
        projectile_object_space = spaces.get_object_space(n=3, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "enemies": object_space,
            "radar_dots": object_space,
            "projectiles": projectile_object_space,
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "life": spaces.Box(low=0, high=5, shape=(), dtype=jnp.int32),
            "enemy_types": spaces.Box(low=jnp.array([-1, -1]), high=jnp.array([4, 4]), shape=(2,), dtype=jnp.int32)
        })


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)  # [Noop, Up, Right, Left, Down, UpRight, UpLeft, DownRight, DownLeft] all with and without Fire
    
    
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
        return (state.score+jnp.log(state.step_counter)) * state.life


    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BattlezoneState) -> bool:
        player_dead = state.life == 0
        return player_dead



#-------------------------------------renderer-------------------------------------
class BattlezoneRenderer(JAXGameRenderer):

    def __init__(self, consts: BattlezoneConstants = None, config=None):
        super().__init__()

        self.consts = consts or BattlezoneConstants()
        self.config = config or render_utils.RendererConfig(
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
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/battlezone" # TODO change later when we have sprites

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
            [self.pad_to_shape(self.SHAPE_MASKS["fighter_jet"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["fighter_jet"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["fighter_jet"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["fighter_jet"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["fighter_jet"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["fighter_jet"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["fighter_jet"], pad, pad),
            ],
            [self.pad_to_shape(self.SHAPE_MASKS["supertank_enemy_01"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["supertank_enemy_02"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["supertank_enemy_03"], pad, pad),
             self.pad_to_shape(self.SHAPE_MASKS["supertank_enemy_04"], pad, pad),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["supertank_enemy_03"], axis=1), pad, pad),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["supertank_enemy_02"], axis=1), pad, pad),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["supertank_enemy_01"], axis=1), pad, pad),
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
        wall_sprite = jnp.tile(
            jnp.array(wall_color_rgba, dtype=jnp.uint8), 
            (*wall_shape[:2], 1)
        )

        return wall_sprite


    def _get_asset_config(self, wall_sprite_top: jnp.ndarray, wall_sprite_bottom: jnp.ndarray) -> list:
        """Returns the declarative manifest of all assets for the game, including both wall sprites."""
        return [ # TODO change later when we have assets
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
            # enemies
            {'name': 'tank_enemy_01', 'type': 'single', 'file': 'tank_enemy_01.npy'}, # TODO not sure if we can/should
            {'name': 'tank_enemy_02', 'type': 'single', 'file': 'tank_enemy_02.npy'},   # summarize them like digits
            {'name': 'tank_enemy_03', 'type': 'single', 'file': 'tank_enemy_03.npy'},
            {'name': 'tank_enemy_04', 'type': 'single', 'file': 'tank_enemy_04.npy'},
            {'name': 'supertank_enemy_01', 'type': 'single', 'file': 'supertank_enemy_01.npy'},
            {'name': 'supertank_enemy_02', 'type': 'single', 'file': 'supertank_enemy_02.npy'},
            {'name': 'supertank_enemy_03', 'type': 'single', 'file': 'supertank_enemy_03.npy'},
            {'name': 'supertank_enemy_04', 'type': 'single', 'file': 'supertank_enemy_04.npy'},
            {'name': 'saucer_left', 'type': 'single', 'file': 'saucer_left.npy'},
            {'name': 'saucer_right', 'type': 'single', 'file': 'saucer_right.npy'},
            {'name': 'fighter_jet', 'type': 'single', 'file': 'fighter.npy'},
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
        pattern = jnp.arange((h + 2) // 3) % 2 * 3 # create enough 0/1 pairs
        row_values = jnp.repeat(pattern, 3)[:h]  # ensure exactly 19 rows

        # replace with actual color indices
        color_id1 = self.COLOR_TO_ID[self.consts.CHAINS_COL_1]
        color_id2 = self.COLOR_TO_ID[self.consts.CHAINS_COL_2]
        row_values = jnp.where(row_values == 0, color_id1, color_id2)

        # create and scroll array
        arr = jnp.broadcast_to(row_values[:, None], (h, w))
        scrolled = jnp.roll(arr, shift=scroll, axis=0)

        return jnp.where(chainMask==self.jr.TRANSPARENT_ID, chainMask, scrolled)


    def _scroll_grass_back(self, grass_mask, scroll):

        grass_fill_color_id = grass_mask[0, 0]
        grass_back_shift = jnp.floor_divide(scroll, 2) % 4
        grass_back_scrolled_mask = jnp.roll(grass_mask, shift=grass_back_shift, axis=0)
        mask = jnp.arange(grass_back_scrolled_mask.shape[0]) < grass_back_shift
        mask = mask[:, None]  # broadcast across columns

        return jnp.where(mask, grass_fill_color_id, grass_back_scrolled_mask)


    @staticmethod
    def _draw_line(img, x0, y0, x1, y1, colorID, samples=256):
        # taken from experimental branch + some changes needs to be overworked maybe
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
        img = BattlezoneRenderer._draw_line(
            img, 
            center_x, 
            center_y,
            center_x + dir_x * radius,
            center_y + dir_y * radius, 
            colorID_2
        )

        #-------------------draw circle-------------
        y = jnp.arange(h)[:, None]  # construct index coordinate mapping
        x = jnp.arange(w)[None, :]

        # Compute squared distance from center
        dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2
        extended_radius = radius + 1
        mask = jnp.logical_and(dist_sq >= extended_radius ** 2, dist_sq < (extended_radius + 1) ** 2)
        img = jnp.where(mask, colorID_1, img)

        #------------------draw enemy dots----------------
        # Check if enemy in radar radius
        in_radar = jax.vmap(self.check_in_radar, in_axes=(0))(state.enemies)
    
        # Get raw player coords
        world_enemies_x = state.enemies.x
        world_enemies_z = state.enemies.z
        world_enemies_dist = state.enemies.distance

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
        return((enemies.distance <= self.consts.RADAR_MAX_SCAN_RADIUS) & enemies.active)


    def pad_to_shape(self, arr: jnp.ndarray, shape_target_x: int, shape_target_y: int) -> jnp.ndarray:
        x, y = arr.shape
        pad_x = shape_target_x - x
        pad_y = shape_target_y - y

        return jnp.pad(arr, ((0, pad_x), (0, pad_y)), mode='constant', constant_values=self.jr.TRANSPARENT_ID)


    def get_enemy_mask(self, enemy:Enemy):
        # selects the correct mask fo the given enemy
        selected_enemy_type = self.padded_enemy_masks[enemy.enemy_type]  # this is still an array containing all rotations

        #----------------select sprite based on rotation------------
        n, _, _ = jnp.shape(selected_enemy_type)
        circle = 2*jnp.pi  # change to 2pi if radians
        # v = jnp.array([enemy.x, enemy.z])
        # w = jnp.array([0, 1])
        # to_screen_angle = (jnp.dot(v, w)/jnp.linalg.norm(v))
        # angle = (enemy.orientation_angle + to_screen_angle - (jnp.pi/2)) % circle
        angle = ((jnp.pi/2) - enemy.orientation_angle) % circle
        angle = jnp.where(
            angle <= jnp.pi, 
            angle, 
            jnp.where(
                angle <= jnp.pi + jnp.pi/2, 
                jnp.pi, 
                0
            )
        )
        index = jnp.round((angle / jnp.pi) * (n - 1)).astype(int)
        rotated_sprite = jnp.array(selected_enemy_type[index])

        # rotated_sprite = jnp.where(rotated_sprite!=255, rotated_sprite + jnp.uint8(enemy.phase), rotated_sprite)  # color by phase for debug

        return rotated_sprite


    def world_cords_to_viewport_cords(self, x, z, f):
        # f = (screen_height / 2) / tan(FOVv / 2)
        def anchor(_):
            # Behind the camera or invalid
            return -100, -100

        def uvMap(_):
            u = ((f * (x / z)) + self.consts.WIDTH / 2).astype(int)
            vOffset = self.consts.HORIZON_Y
            v = ((f / (z - self.consts.HITBOX_SIZE)) + vOffset).astype(int)
            
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
            valid_rows = jnp.any(mask != self.jr.TRANSPARENT_ID, axis=1)
            valid_cols = jnp.any(mask != self.jr.TRANSPARENT_ID, axis=0)
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

            # Zero edges: set padding outside zoomed area to self.transparent_id
            rows = jnp.arange(x)[:, None]
            cols = jnp.arange(y)[None, :]
            #edge = (rows < x_min) | (rows > x_max) | (cols < y_min) | (cols > y_max)
            z = zoom_factor + 1
            edge = (rows < x_min + z) | (rows >= x_max - z) | \
                   (cols < y_min + z) | (cols >= y_max - z)
            zoomed_mask = jnp.where(edge, self.jr.TRANSPARENT_ID, zoomed_mask)

            return zoomed_mask

        return jax.lax.cond(zoom_factor <= 1, anchor, zoom, operand=None)


    def render_single_enemy(self, raster, enemy: Enemy):

        def enemy_active(enemy):
            enemy_mask = self.get_enemy_mask(enemy)
            # zoom_factor = ((jnp.sqrt(jnp.square(enemy.x) + jnp.square(enemy.z)) - 20.0) *
            #                self.consts.DISTANCE_TO_ZOOM_FACTOR_CONSTANT).astype(int)
            zoom_factor = jnp.clip(((-0.15 * (enemy.distance) + 21.0) / 20.0), 0.0, 1.0)
            zoom_factor = jnp.round(1.0 / zoom_factor)
            zoomed_mask = self.zoom_mask(enemy_mask, zoom_factor)
            x, y = self.world_cords_to_viewport_cords(enemy.x, enemy.z, self.consts.CAMERA_FOCAL_LENGTH)

            rightmost_col = jnp.max(
                jnp.where(
                    jnp.any(zoomed_mask != self.jr.TRANSPARENT_ID, axis=0),
                    jnp.arange(zoomed_mask.shape[1]),
                    0
                )
            )

            return self.jr.render_at_clipped(raster, x - (rightmost_col // 2), self.consts.ENEMY_POS_Y, zoomed_mask)
        
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
                x, y = self.world_cords_to_viewport_cords(enemy.x, enemy.z, self.consts.CAMERA_FOCAL_LENGTH)

                rightmost_col = jnp.max(
                    jnp.where(
                        jnp.any(zoomed_mask != self.jr.TRANSPARENT_ID, axis=0),
                        jnp.arange(zoomed_mask.shape[1]),
                        0
                    )
                )

                return self.jr.render_at_clipped(raster, x - (rightmost_col // 2), self.consts.ENEMY_POS_Y, zoomed_mask)
            
            def _pass(_):
                return raster

            # render death animation if death_anim_counter > 0, else render nothing
            return jax.lax.cond(enemy.death_anim_counter==0, _pass, render_death, enemy)

        return jax.lax.cond(enemy.active, enemy_active, enemy_inactive, enemy)


    def render_single_projectile(self, raster, projectile: Projectile):

        def projectile_active(projectile):

            projectile_mask_index = jnp.where(projectile.distance <= 15, 0, 1)
            projectile_mask = self.projectile_masks[projectile_mask_index]
            x, y = self.world_cords_to_viewport_cords(projectile.x, projectile.z, self.consts.CAMERA_FOCAL_LENGTH)

            rightmost_col = jnp.max(
                jnp.where(
                    jnp.any(projectile_mask != self.jr.TRANSPARENT_ID, axis=0),
                    jnp.arange(projectile_mask.shape[1]),
                    0
                )
            )
            
            return self.jr.render_at_clipped(raster, x - (rightmost_col // 2), y, projectile_mask)
            

        def projectile_inactive(_):
            return raster

        render_condition = jnp.all(jnp.stack([projectile.active,
                                              projectile.z >= self.consts.HITBOX_SIZE,
                                              projectile.distance <= self.consts.RADAR_MAX_SCAN_RADIUS]))
        
        return jax.lax.cond(render_condition, projectile_active, projectile_inactive, projectile)


    def render_targeting_indicator(self, raster, state):

        # determine if pointing at enemy
        within_x = jnp.abs(state.enemies.x) <= self.consts.ENEMY_HITBOX_SIZE
        # within_x = jnp.logical_and(state.enemies.x <= 0,
        #                state.enemies.x >= -2*self.consts.ENEMY_HITBOX_SIZE)
        in_front = state.enemies.z > 0
        overlaps = state.enemies.active & within_x & in_front
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
            def render_single_enemy_wrapped(raster, enemy):  # so that i dont have to pass self
                return self.render_single_enemy(raster, enemy), None

            # most distant first
            order = jnp.argsort(state.enemies.distance)[::-1]
            enemies_sorted = jax.tree.map(lambda x: x[order], state.enemies)

            raster, _ = jax.lax.scan(render_single_enemy_wrapped, raster, enemies_sorted)

            #------------------------------projectiles---------------
            raster = self.render_single_projectile(raster, state.player_projectile)
            #raster = jax.lax.cond(state.step_counter%2==0, self.render_single_projectile, lambda r, _: r,
                         #raster, state.player_projectile)
            # probably more accurate but looks stoopid because different frame rates
            def render_single_projectile_wrapped(raster, projectile):  # so that i dont have to pass self
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

        raster = jax.lax.cond(state.death_anim_counter <= 0, normal_render, death_render, raster)

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
