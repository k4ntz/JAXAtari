"Authors: Lasse Reith, Benedikt Schwarz, Shir Nussbaum"
import os
from enum import IntEnum
from functools import partial
from typing import NamedTuple, Optional, Tuple

import chex
import jax
import jax.lax
import jax.numpy as jnp

import jaxatari.spaces as spaces

from jaxatari.environment import JAXAtariAction as Action, JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

class WhiteUFOPattern(IntEnum):
    """Behavioral patterns used by the white UFO enemies."""

    IDLE = 0
    DROP_STRAIGHT = 1
    DROP_RIGHT = 2
    DROP_LEFT = 3
    RETREAT = 4
    SHOOT = 5


class BeamriderConstants(NamedTuple):
    """All tunable constants for the Beamrider environment."""

    RENDER_SCALE_FACTOR: int = 4
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    PLAYER_WIDTH: int = 10
    PLAYER_HEIGHT: int = 10
    ENEMY_WIDTH: int = 4
    ENEMY_HEIGHT: int = 4
    PLAYER_COLOR: Tuple[int, int, int] = (223, 183, 85)
    LEFT_CLIP_PLAYER: int = 27
    RIGHT_CLIP_PLAYER: int = 137
    BOTTOM_OF_LANES: Tuple[int, int, int, int, int] = (27,52,77,102,127)
    TOP_OF_LANES: Tuple[int, int, int, int, int] = (38,61,71,81,91,102,123)  #lane 0,6 are connected to points in middle of the map, not to bottom lane points
    
    TOP_TO_BOTTOM_LANE_VECTORS: Tuple[Tuple[float, float],Tuple[float, float],Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2,4),(-1, 4), (-0.52, 4), (0,4), (0.52, 4), (1, 4),(2,4))


    MAX_LASER_Y: int = 67
    MIN_BULLET_Y:int =156
    MAX_TORPEDO_Y: int = 60
    BOTTOM_TO_TOP_LANE_VECTORS: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-1, 4), (-0.52, 4), (0,4), (0.52, 4), (1, 4))

    PLAYER_POS_Y: int = 165
    PLAYER_SPEED: float = 2.5

    BOTTOM_CLIP:int = 160
    TOP_CLIP:int=43
    LASER_HIT_RADIUS: Tuple[int, int] = (4, 2)
    TORPEDO_HIT_RADIUS: Tuple[int, int] = (3, 2)
    LASER_ID: int = 1
    TORPEDO_ID: int = 2
    BULLET_OFFSCREEN_POS: Tuple[int, int] = (800.0, 800.0)
    ENEMY_OFFSCREEN_POS: Tuple[int,int] = (2000,2000)
    MIN_BLUE_LINE_POS: int = 46
    MAX_BLUE_LINE_POS: int = 160
    WHITE_UFO_RETREAT_DURATION: int = 28
    WHITE_UFO_PATTERN_DURATIONS: Tuple[int, int, int, int, int, int] = (0, 42, 42, 42, 28, 12)
    WHITE_UFO_PATTERN_PROBS: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
    WHITE_UFO_SPEED_FACTOR: float = 0.1
    WHITE_UFO_SHOT_SPEED_FACTOR: float = 0.8
    WHITE_UFO_RETREAT_P_MIN: float = 0.05
    WHITE_UFO_RETREAT_P_MAX: float = 0.9
    WHITE_UFO_RETREAT_ALPHA: float = 0.01
    WHITE_UFO_RETREAT_SPEED_MULT: float = 1.5
    WHITE_UFO_TOP_LANE_MIN_SPEED: float = 0.3
    WHITE_UFO_TOP_LANE_TURN_SPEED: float = 0.5

    INIT_LINE_POS = jnp.array([118.08385, 90.88263, 156.90707, 49.115276, 58.471092, 71.82423 ])

    INIT_LINE_VEL = jnp.array([0.31581876, 0.22127658, 0.4507547, 0.07610765, 0.10862522, 0.15503621])
    NEW_LINE_VEL = 0.06528
    LINE_ACCELERATION = 1.007
    MAX_LINE_VEL = 0.6665
    BLUE_LINE_OFFSCREEN_Y = 500
    NEW_LINE_THRESHHOLD_BOTTOM_LINE = 54.0



class LevelState(NamedTuple):
    player_pos: chex.Array
    player_vel: chex.Array
    white_ufo_left: chex.Array
    comet_positions: chex.Array
    mothership_position: chex.Array
    player_shot_pos: chex.Array
    player_shot_vel: chex.Array
    torpedoes_left: chex.Array
    shooting_cooldown: chex.Array
    bullet_type: chex.Array

    # enemies
    enemy_type: chex.Array
    white_ufo_pos: chex.Array
    white_ufo_vel: chex.Array
    enemy_shot_pos: chex.Array
    enemy_shot_vel: chex.Array
    enemy_shot_timer: chex.Array
    white_ufo_time_on_lane: chex.Array
    white_ufo_attack_time: chex.Array
    white_ufo_time_allowed: chex.Array
    white_ufo_pattern_id: chex.Array
    white_ufo_pattern_timer: chex.Array

    line_positions: chex.Array
    line_velocities: chex.Array


class BeamriderState(NamedTuple):
    level: LevelState
    score: chex.Array
    sector: chex.Array
    level_finished: chex.Array
    reset_coords: chex.Array
    lives: chex.Array
    steps: chex.Array
    rng: chex.Array


class BeamriderInfo(NamedTuple):
    score: chex.Array
    sector: chex.Array


class BeamriderObservation(NamedTuple):
    pos: chex.Array
    shooting_cd: chex.Array
    torpedoes_left: chex.Array
    player_shots_pos: chex.Array
    player_shots_vel: chex.Array
    white_ufo_left: chex.Array

    # enemies
    enemy_type: chex.Array
    white_ufo_pos: chex.Array
    white_ufo_vel: chex.Array
    enemy_shot_pos: chex.Array
    enemy_shot_vel: chex.Array

class WhiteUFOUpdate(NamedTuple):
    """Aggregated quantities needed after updating all white UFOs."""

    pos: chex.Array
    vel: chex.Array
    time_on_lane: chex.Array
    attack_time: chex.Array
    pattern_id: chex.Array
    pattern_timer: chex.Array


class JaxBeamrider(JaxEnvironment[BeamriderState, BeamriderObservation, BeamriderInfo, BeamriderConstants]):
    """JAX implementation of the Beamrider environment."""

    def __init__(self, consts: Optional[BeamriderConstants] = None):
        super().__init__(consts)
        self.consts = consts or BeamriderConstants()
        self.renderer = BeamriderRenderer(self.consts)
        self.obs_size = 111
        self.action_set = [
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
            Action.DOWNLEFTFIRE,
        ]
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[BeamriderObservation, BeamriderState]:
        state = self.reset_level(1)
        observation = self._get_observation(state)
        return observation, state

    def reset_level(self, next_level=1) -> BeamriderState:
        enemy_shot_offscreen = jnp.tile(
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, 3),
        )

        level_state = LevelState(
            player_pos=jnp.array(77.0),
            player_vel=jnp.array(0.0),
            white_ufo_left=jnp.array(15),
            comet_positions=jnp.array(0),
            mothership_position=jnp.array(0),
            player_shot_pos=jnp.array(self.consts.BULLET_OFFSCREEN_POS),
            player_shot_vel=jnp.array([0, 0]),
            torpedoes_left=jnp.array(3),
            shooting_cooldown=jnp.array(0),
            bullet_type=jnp.array(self.consts.LASER_ID),
            enemy_type=jnp.array([0, 0, 0]),
            white_ufo_pos=jnp.array([[77.0, 77.0, 77.0], [43.0, 43.0, 43.0]]),
            white_ufo_vel=jnp.array([[-0.5, 0.5, 0.3], [0.0, 0.0, 0.0]]),
            enemy_shot_pos=enemy_shot_offscreen,
            enemy_shot_vel=jnp.zeros((3,), dtype=jnp.int32),
            enemy_shot_timer=jnp.zeros((3,), dtype=jnp.int32),
            white_ufo_time_on_lane=jnp.array([0, 0, 0]),
            white_ufo_attack_time=jnp.zeros((3,), dtype=jnp.int32),
            white_ufo_time_allowed=jnp.array([400, 600, 800]),
            white_ufo_pattern_id=jnp.zeros(3, dtype=jnp.int32),
            white_ufo_pattern_timer=jnp.zeros(3, dtype=jnp.int32),
            line_positions=self.consts.INIT_LINE_POS,
            line_velocities=self.consts.INIT_LINE_VEL,
        )

        return BeamriderState(
            level=level_state,
            score=jnp.array(0),
            sector=jnp.array(next_level),
            level_finished=jnp.array(0),
            reset_coords=jnp.array(False),
            lives=jnp.array(3),
            steps=jnp.array(0),
            rng=jnp.array(jax.random.key(42)),
        )
    

    def _get_observation(self, state: BeamriderState) -> BeamriderObservation:
        level = state.level
        return BeamriderObservation(
            pos=level.player_pos,
            shooting_cd=level.shooting_cooldown,
            torpedoes_left=level.torpedoes_left,
            player_shots_pos=level.player_shot_pos,
            player_shots_vel=level.player_shot_vel,
            white_ufo_left=level.white_ufo_left,
            enemy_type=level.enemy_type,
            white_ufo_pos=level.white_ufo_pos,
            white_ufo_vel=level.white_ufo_vel,
            enemy_shot_pos=level.enemy_shot_pos,
            enemy_shot_vel=level.enemy_shot_vel,
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step(
        self,
        state: BeamriderState,
        action: chex.Array,
    ) -> Tuple[BeamriderObservation, BeamriderState, float, bool, BeamriderInfo]:
        (
            player_x,
            vel_x,
            player_shot_position,
            player_shot_velocity,
            torpedos_left,
            bullet_type,
        ) = self._player_step(state, action)

        rngs = jax.random.split(state.rng, 4)
        next_rng = rngs[0]
        ufo_keys = rngs[1:]

        ufo_update = self._advance_white_ufos(state, ufo_keys)
        white_ufo_pos, player_shot_position = self._collision_handler(
            state, ufo_update.pos, player_shot_position, bullet_type
        )
        enemy_shot_pos, enemy_shot_lane, enemy_shot_timer, lives = self._enemy_shot_step(
            state,
            white_ufo_pos,
            ufo_update.pattern_id,
            ufo_update.pattern_timer,
        )
        
        line_positions, line_velocities = self._line_step(state)

        next_step = state.steps + 1
        new_level_state = LevelState(
            player_pos=player_x,
            player_vel=vel_x,
            white_ufo_left=jnp.array(15),
            comet_positions=jnp.array(0),
            mothership_position=jnp.array(0),
            player_shot_pos=player_shot_position,
            player_shot_vel=player_shot_velocity,
            torpedoes_left=torpedos_left,
            shooting_cooldown=jnp.array(0),
            bullet_type=bullet_type,
            enemy_type=jnp.array([0, 0, 0]),
            white_ufo_pos=white_ufo_pos,
            white_ufo_vel=ufo_update.vel,
            enemy_shot_pos=enemy_shot_pos,
            enemy_shot_vel=enemy_shot_lane,
            enemy_shot_timer=enemy_shot_timer,
            white_ufo_time_on_lane=ufo_update.time_on_lane,
            white_ufo_attack_time=ufo_update.attack_time,
            white_ufo_time_allowed=state.level.white_ufo_time_allowed,
            white_ufo_pattern_id=ufo_update.pattern_id,
            white_ufo_pattern_timer=ufo_update.pattern_timer,
            line_positions=line_positions,
            line_velocities=line_velocities,
        )

        new_state = BeamriderState(
            level=new_level_state,
            score=jnp.array(0),
            sector=jnp.array(1),
            level_finished=jnp.array(0),
            reset_coords=jnp.array(False),
            lives=lives,
            steps=next_step,
            rng=next_rng,
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)

        observation = self._get_observation(new_state)
        
        return observation, new_state, env_reward, done, info

    def _player_step(self, state: BeamriderState, action: chex.Array):
        #level_constants = self._get_level_constants(state.sector)
        x = state.level.player_pos
        v = state.level.player_vel

        # Get inputs
        press_right = jnp.any(
            jnp.array(
                [action == Action.RIGHT,action == Action.UPRIGHT,action == Action.DOWNRIGHT, action == Action.RIGHTFIRE]
            )
        )

        press_left = jnp.any(
            jnp.array(
                [action == Action.LEFT, action == Action.UPLEFT, action == Action.DOWNLEFT, action == Action.LEFTFIRE]
            )
        )

        press_up = jnp.any(
            jnp.array(
                [action == Action.UP, action == Action.UPRIGHT, action == Action.UPLEFT]
            )
        )
        fire_types = jnp.array([Action.FIRE, Action.DOWNFIRE, Action.LEFTFIRE, Action.RIGHTFIRE, Action.UPLEFTFIRE, Action.UPRIGHTFIRE, Action.DOWNLEFTFIRE, Action.DOWNRIGHTFIRE])
        press_fire = jnp.any(
            jnp.array(
                jnp.isin(action, fire_types)
            )
        )

        is_in_lane = jnp.isin(x, jnp.array(self.consts.BOTTOM_OF_LANES)) # predicate: x is one of LANES

        v = jax.lax.cond(
            is_in_lane,          
            lambda v_: jnp.zeros_like(v_),          # then -> 0
            lambda v_: v_,                          # else -> keep v
            v,                                      # operand
        )
        
        v = jax.lax.cond(
            jnp.logical_or(press_left,press_right),
            lambda v_: (press_right.astype(v.dtype) - press_left.astype(v.dtype)) * self.consts.PLAYER_SPEED,    
            lambda v_: v_,
            v,
        )
        x_before_change = x
        x = jnp.clip(x + v, self.consts.LEFT_CLIP_PLAYER, self.consts.RIGHT_CLIP_PLAYER - self.consts.PLAYER_WIDTH)

        ####### Ab hier von Lasse shot gedöns
        bullet_exists= self._bullet_infos(state)


        can_spawn_bullet = jnp.logical_and(jnp.logical_not(bullet_exists), is_in_lane)
        can_spawn_torpedo = jnp.logical_and(can_spawn_bullet, state.level.torpedoes_left >= 1)

        new_laser = jnp.logical_and(press_fire, can_spawn_bullet)
        new_torpedo = jnp.logical_and(press_up, can_spawn_torpedo)
        new_bullet = jnp.logical_or(new_torpedo, new_laser)

        lanes = jnp.array(self.consts.BOTTOM_OF_LANES)
        lane_velocities = jnp.array(self.consts.BOTTOM_TO_TOP_LANE_VECTORS)
        lane_index = jnp.argmax(x_before_change == lanes) 
        lane_velocity = lane_velocities[lane_index]

        shot_velocity = jnp.where(
            new_bullet,
            lane_velocity,
            state.level.player_shot_vel,
        )

 
        pos_if_no_new = jnp.where(bullet_exists, (state.level.player_shot_pos - shot_velocity), jnp.array(self.consts.BULLET_OFFSCREEN_POS))
        shot_position = jnp.where(new_bullet, jnp.array([state.level.player_pos +3 , self.consts.MIN_BULLET_Y]), pos_if_no_new)

        bullet_exists = jnp.any(jnp.array([new_laser, bullet_exists, new_torpedo]))
        torpedos_left = state.level.torpedoes_left - new_torpedo
        bullet_type_if_new = jnp.where(new_laser, self.consts.LASER_ID, self.consts.TORPEDO_ID)
        bullet_type = jnp.where(new_bullet, bullet_type_if_new, state.level.bullet_type)

        #####
        return(x,v,shot_position, shot_velocity, torpedos_left, bullet_type)

    def _collision_handler(self, state: BeamriderState, new_white_ufo_pos, new_shot_pos, new_bullet_type):
        enemies = new_white_ufo_pos.T
        distance_to_bullet = jnp.abs(enemies - new_shot_pos)
        bullet_type_is_laser = new_bullet_type == self.consts.LASER_ID
        bullet_radius = jnp.where(bullet_type_is_laser, jnp.array(self.consts.LASER_HIT_RADIUS),jnp.array(self.consts.TORPEDO_HIT_RADIUS))
        distance_bullet_radius = distance_to_bullet - bullet_radius
        mask = jnp.array((distance_bullet_radius[:, 0] <= 0) & (distance_bullet_radius[:, 1] <= 0))
        hit_index = jnp.argmax(mask)
        hit_exists = jnp.any(mask) 
        white_ufo_pos_after_hit = enemies.at[hit_index].set(jnp.array(self.consts.ENEMY_OFFSCREEN_POS)).T
        player_shot_pos = jnp.where(hit_exists, jnp.array(self.consts.BULLET_OFFSCREEN_POS), new_shot_pos)
        enemie_pos = jnp.where(hit_exists, white_ufo_pos_after_hit ,new_white_ufo_pos)
        return (enemie_pos, player_shot_pos)

    def entropy_heat_prob(self, steps_static, alpha=0.002, p_min=0.0002, p_max=0.8):
        steps =steps_static/10
        # steps_static: scalar integer or array
        heat = 1.0 - jnp.exp(-alpha * steps)
        p_swap = p_min + (p_max - p_min) * heat
        return p_swap
    
    def _advance_white_ufos(self, state: BeamriderState, keys: chex.Array) -> WhiteUFOUpdate:
        """Advance all white UFOs in lockstep for clearer logic inside step()."""

        updates = [self._white_ufo_step(state, idx, keys[idx]) for idx in range(3)]
        positions = jnp.stack([update[0] for update in updates], axis=1)
        vel_x = jnp.array([update[1] for update in updates])
        vel_y = jnp.array([update[2] for update in updates])
        velocities = jnp.stack([vel_x, vel_y])
        time_on_lane = jnp.array([update[3] for update in updates])
        attack_time = jnp.array([update[4] for update in updates])
        pattern_id = jnp.array([update[5] for update in updates], dtype=jnp.int32)
        pattern_timer = jnp.array([update[6] for update in updates], dtype=jnp.int32)
        return WhiteUFOUpdate(
            pos=positions,
            vel=velocities,
            time_on_lane=time_on_lane,
            attack_time=attack_time,
            pattern_id=pattern_id,
            pattern_timer=pattern_timer,
        )
    def _white_ufo_step(self, state: BeamriderState, index: int, key: chex.Array):
        white_ufo_position = jnp.array([state.level.white_ufo_pos[0][index], state.level.white_ufo_pos[1][index]])
        white_ufo_vel_x = state.level.white_ufo_vel[0][index]
        white_ufo_vel_y = state.level.white_ufo_vel[1][index]
        time_on_lane = state.level.white_ufo_time_on_lane[index]
        attack_time = state.level.white_ufo_attack_time[index]
        pattern_id = state.level.white_ufo_pattern_id[index]
        pattern_timer = state.level.white_ufo_pattern_timer[index]

        key_pattern, key_motion = jax.random.split(key)
        pattern_id, pattern_timer, time_on_lane, attack_time = self._white_ufo_update_pattern_state(
            white_ufo_position, time_on_lane, attack_time, pattern_id, pattern_timer, key_pattern
        )

        requires_lane_motion = self._white_ufo_pattern_requires_lane_motion(pattern_id)

        def follow_lane(_):
            return self._white_ufo_normal(white_ufo_position, white_ufo_vel_x, white_ufo_vel_y, pattern_id)

        def stay_on_top(_):
            return self._white_ufo_top_lane(white_ufo_position, white_ufo_vel_x, pattern_id, key_motion)

        white_ufo_vel_x, white_ufo_vel_y = jax.lax.cond(
            requires_lane_motion,
            follow_lane,
            stay_on_top,
            operand=None
        )

        new_x = white_ufo_position[0] + white_ufo_vel_x
        new_y = white_ufo_position[1] + white_ufo_vel_y
        new_x = jnp.clip(new_x, self.consts.LEFT_CLIP_PLAYER, self.consts.RIGHT_CLIP_PLAYER)
        new_y = jnp.clip(new_y, self.consts.TOP_CLIP, self.consts.BOTTOM_CLIP)
        white_ufo_position = jnp.array([new_x, new_y])

        return (
            white_ufo_position,
            white_ufo_vel_x,
            white_ufo_vel_y,
            time_on_lane,
            attack_time,
            pattern_id,
            pattern_timer,
        )

    def _white_ufo_pattern_requires_lane_motion(self, pattern_id: chex.Array) -> chex.Array:
        drop_straight = pattern_id == int(WhiteUFOPattern.DROP_STRAIGHT)
        drop_left = pattern_id == int(WhiteUFOPattern.DROP_LEFT)
        drop_right = pattern_id == int(WhiteUFOPattern.DROP_RIGHT)
        retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        return jnp.logical_or(
            drop_straight,
            jnp.logical_or(drop_left, jnp.logical_or(drop_right, retreat)),
        )

    def _white_ufo_update_pattern_state(
        self,
        position: chex.Array,
        time_on_lane: chex.Array,
        attack_time: chex.Array,
        pattern_id: chex.Array,
        pattern_timer: chex.Array,
        key: chex.Array,
    ):
        on_top_lane = position[1] <= self.consts.TOP_CLIP
        time_on_lane = jnp.where(on_top_lane, time_on_lane + 1, 0)
        attack_time = jnp.where(on_top_lane, 0, attack_time)
        pattern_timer = jnp.maximum(pattern_timer - 1, jnp.zeros_like(pattern_timer))

        closest_lane_id = self._white_ufo_closest_lane_id(position)
        shootable_lane = jnp.logical_and(closest_lane_id > 0, closest_lane_id < 6)
        allow_shoot = jnp.logical_and(jnp.logical_not(on_top_lane), shootable_lane)

        is_drop_pattern = jnp.logical_or(
            pattern_id == int(WhiteUFOPattern.DROP_STRAIGHT),
            jnp.logical_or(
                pattern_id == int(WhiteUFOPattern.DROP_LEFT),
                pattern_id == int(WhiteUFOPattern.DROP_RIGHT),
            ),
        )
        is_shoot_pattern = pattern_id == int(WhiteUFOPattern.SHOOT)
        is_engagement_pattern = jnp.logical_or(is_drop_pattern, is_shoot_pattern)
        attack_time = jnp.where(
            jnp.logical_and(jnp.logical_not(on_top_lane), is_engagement_pattern),
            attack_time + 1,
            attack_time,
        )

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        retreat_finished = jnp.logical_and(is_retreat, on_top_lane)
        pattern_id = jnp.where(retreat_finished, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(retreat_finished, 0, pattern_timer)
        attack_time = jnp.where(retreat_finished, 0, attack_time)

        pattern_finished_off_top = jnp.logical_and.reduce(jnp.array([
            jnp.logical_not(on_top_lane),
            is_engagement_pattern,
            pattern_timer == 0,
        ]))

        key_start_roll, key_start_choice, key_retreat_roll, key_chain_choice = jax.random.split(key, 4)
        retreat_roll = jax.random.uniform(key_retreat_roll)
        retreat_prob = self._white_ufo_retreat_prob(attack_time)
        retreat_now = jnp.logical_and(pattern_finished_off_top, retreat_roll < retreat_prob)
        pattern_id = jnp.where(retreat_now, int(WhiteUFOPattern.RETREAT), pattern_id)
        pattern_timer = jnp.where(retreat_now, self.consts.WHITE_UFO_RETREAT_DURATION, pattern_timer)
        attack_time = jnp.where(retreat_now, 0, attack_time)

        chain_next = jnp.logical_and(pattern_finished_off_top, jnp.logical_not(retreat_now))

        def choose_chain_pattern(_):
            pattern, duration = self._white_ufo_choose_pattern(key_chain_choice, allow_shoot=allow_shoot)
            return pattern, duration

        def keep_after_chain(_):
            return pattern_id, pattern_timer

        pattern_id, pattern_timer = jax.lax.cond(
            chain_next,
            choose_chain_pattern,
            keep_after_chain,
            operand=None,
        )

        should_choose_new = jnp.logical_and.reduce(jnp.array([
            on_top_lane,
            pattern_id == int(WhiteUFOPattern.IDLE),
            pattern_timer == 0,
        ]))
        p_start = self.entropy_heat_prob(time_on_lane)
        start_roll = jax.random.uniform(key_start_roll)
        start_attack = jnp.logical_and(should_choose_new, start_roll < p_start)

        def choose_new_pattern(_):
            pattern, duration = self._white_ufo_choose_pattern(key_start_choice, allow_shoot=jnp.array(False))
            return pattern, duration

        def keep_pattern(_):
            return pattern_id, pattern_timer

        pattern_id, pattern_timer = jax.lax.cond(
            start_attack,
            choose_new_pattern,
            keep_pattern,
            operand=None,
        )

        return pattern_id, pattern_timer, time_on_lane, attack_time

    def _white_ufo_closest_lane_id(self, position: chex.Array) -> chex.Array:
        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]

        ufo_x = position[0].astype(jnp.float32)
        ufo_y = position[1].astype(jnp.float32)
        lane_x_at_ufo_y = lanes_top_x + lane_dx_over_dy * (ufo_y - float(self.consts.TOP_CLIP))
        return jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x)).astype(jnp.int32)

    def _white_ufo_retreat_prob(self, attack_time: chex.Array) -> chex.Array:
        t = attack_time.astype(jnp.float32)
        alpha = jnp.array(self.consts.WHITE_UFO_RETREAT_ALPHA, dtype=jnp.float32)
        p_min = jnp.array(self.consts.WHITE_UFO_RETREAT_P_MIN, dtype=jnp.float32)
        p_max = jnp.array(self.consts.WHITE_UFO_RETREAT_P_MAX, dtype=jnp.float32)
        heat = 1.0 - jnp.exp(-alpha * t)
        return p_min + (p_max - p_min) * heat

    def _white_ufo_choose_pattern(self, key: chex.Array, *, allow_shoot: chex.Array):
        pattern_choices = jnp.array(
            [
                int(WhiteUFOPattern.DROP_STRAIGHT),
                int(WhiteUFOPattern.DROP_LEFT),
                int(WhiteUFOPattern.DROP_RIGHT),
                int(WhiteUFOPattern.SHOOT),
            ],
            dtype=jnp.int32,
        )
        pattern_probs = jnp.array(self.consts.WHITE_UFO_PATTERN_PROBS, dtype=jnp.float32)
        shoot_mask = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
        pattern_probs = jnp.where(allow_shoot, pattern_probs, pattern_probs * shoot_mask)
        pattern_probs = pattern_probs / jnp.sum(pattern_probs)
        pattern = jax.random.choice(key, pattern_choices, shape=(), p=pattern_probs)
        pattern_durations = jnp.array(self.consts.WHITE_UFO_PATTERN_DURATIONS)
        duration = pattern_durations[pattern]
        return pattern, duration

    def _white_ufo_top_lane(self, white_ufo_pos, white_ufo_vel_x, pattern_id, key: chex.Array):
        hold_position = pattern_id == int(WhiteUFOPattern.SHOOT)
        min_speed = float(self.consts.WHITE_UFO_TOP_LANE_MIN_SPEED)
        turn_speed = float(self.consts.WHITE_UFO_TOP_LANE_TURN_SPEED)

        vx = jnp.where(hold_position, 0.0, white_ufo_vel_x)
        need_boost = jnp.logical_and(jnp.logical_not(hold_position), jnp.abs(vx) < min_speed)
        random_sign = jnp.where(jax.random.uniform(key) < 0.5, -1.0, 1.0)
        direction = jnp.where(vx == 0.0, random_sign, jnp.sign(vx))
        vx = jnp.where(need_boost, direction * min_speed, vx)

        do_bounce = jnp.logical_not(hold_position)
        vx = jnp.where(
            jnp.logical_and(do_bounce, white_ufo_pos[0] >= self.consts.RIGHT_CLIP_PLAYER),
            -turn_speed,
            vx,
        )
        vx = jnp.where(
            jnp.logical_and(do_bounce, white_ufo_pos[0] <= self.consts.LEFT_CLIP_PLAYER),
            turn_speed,
            vx,
        )
        return vx, 0.0
    
    def _white_ufo_normal(self, white_ufo_pos, white_ufo_vel_x, white_ufo_vel_y, pattern_id):
        speed_factor = self.consts.WHITE_UFO_SPEED_FACTOR
        retreat_mult = self.consts.WHITE_UFO_RETREAT_SPEED_MULT
        x, y = white_ufo_pos[0], white_ufo_pos[1]
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)

        lane_dx = lane_vectors[:, 0]
        lane_dy = lane_vectors[:, 1]
        lane_x_at_y = lanes_top_x + (lane_dx / lane_dy) * (y - float(self.consts.TOP_CLIP))

        # 1. Identify the current lane
        closest_lane_id = jnp.argmin(jnp.abs(lane_x_at_y - x))

        # 2. Determine index offset based on pattern
        # DROP_RIGHT (+1), DROP_LEFT (-1), others (0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_LEFT), -1, lane_offset)

        # 3. Apply offset and clip to valid lane indices (0 to 6)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, 0, 6)

        lane_vector = lane_vectors[target_lane_id]
        target_lane_x = lane_x_at_y[target_lane_id]

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        cross_track = target_lane_x - x
        distance_to_lane = jnp.abs(cross_track)
        direction = jnp.sign(cross_track)

        def seek_lane(_):
            attack_vx = jnp.where(direction == 0, 0.0, direction * 0.5)
            retreat_vx = jnp.where(direction == 0, 0.0, direction * speed_factor * retreat_mult * 2.0)
            new_vx = jnp.where(is_retreat, retreat_vx, attack_vx)
            new_vy = jnp.where(is_retreat, -lane_vector[1] * speed_factor * retreat_mult, 0.25)
            return new_vx, new_vy

        def follow_lane(_):
            new_vx = jnp.where(is_retreat, -lane_vector[0] * speed_factor * retreat_mult, lane_vector[0] * speed_factor)
            new_vy = jnp.where(is_retreat, -lane_vector[1] * speed_factor * retreat_mult, lane_vector[1] * speed_factor)
            return new_vx, new_vy

        return jax.lax.cond(
            distance_to_lane <= 0.25,
            follow_lane,
            seek_lane,
            operand=None,
        )

    def _enemy_shot_step(
        self,
        state: BeamriderState,
        white_ufo_pos: chex.Array,
        white_ufo_pattern_id: chex.Array,
        white_ufo_pattern_timer: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]

        offscreen_xy = jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=jnp.float32)
        offscreen = jnp.tile(offscreen_xy.reshape(2, 1), (1, 3))

        shot_pos = state.level.enemy_shot_pos.astype(jnp.float32)
        shot_lane = state.level.enemy_shot_vel.astype(jnp.int32)
        shot_timer = state.level.enemy_shot_timer.astype(jnp.int32)

        shot_active = shot_pos[1] <= float(self.consts.BOTTOM_CLIP)
        shot_timer = jnp.where(shot_active, shot_timer + 1, 0)

        shoot_duration = jnp.array(self.consts.WHITE_UFO_PATTERN_DURATIONS, dtype=jnp.int32)[
            int(WhiteUFOPattern.SHOOT)
        ]
        wants_spawn = jnp.logical_and(
            white_ufo_pattern_id == int(WhiteUFOPattern.SHOOT),
            white_ufo_pattern_timer == shoot_duration,
        )
        ufo_on_screen = white_ufo_pos[1] <= float(self.consts.BOTTOM_CLIP)
        ufo_not_on_top_lane = white_ufo_pos[1] > float(self.consts.TOP_CLIP)
        ufo_x = white_ufo_pos[0].astype(jnp.float32)
        ufo_y = white_ufo_pos[1].astype(jnp.float32)
        lane_x_at_ufo_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (
            ufo_y[None, :] - float(self.consts.TOP_CLIP)
        )
        closest_lane = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x[None, :]), axis=0).astype(jnp.int32)
        allowed_shot_lane = jnp.logical_and(closest_lane > 0, closest_lane < 6)
        spawn = jnp.logical_and.reduce(
            jnp.array([
                wants_spawn,
                ufo_on_screen,
                ufo_not_on_top_lane,
                allowed_shot_lane,
                jnp.logical_not(shot_active),
            ])
        )

        spawn_y = jnp.clip(ufo_y + 4.0, float(self.consts.TOP_CLIP), float(self.consts.BOTTOM_CLIP))
        spawn_x = jnp.take(lanes_top_x, closest_lane) + jnp.take(lane_dx_over_dy, closest_lane) * (
            spawn_y - float(self.consts.TOP_CLIP)
        )
        spawn_pos = jnp.stack([spawn_x, spawn_y])
        shot_pos = jnp.where(spawn[None, :], spawn_pos, shot_pos)
        shot_lane = jnp.where(spawn, closest_lane, shot_lane)
        shot_timer = jnp.where(spawn, 0, shot_timer)
        shot_active = jnp.logical_or(shot_active, spawn)

        # Per-shot cadence (frame-by-frame):
        # sprite1 -> stand still -> move -> stand still -> sprite2 -> stand still -> move -> stand still -> ...
        should_move = jnp.logical_and(shot_active, (shot_timer % 4) == 2)
        speed = float(self.consts.WHITE_UFO_SHOT_SPEED_FACTOR)
        lane_dy = jnp.take(lane_vectors[:, 1], shot_lane)
        y_after = shot_pos[1] + jnp.where(should_move, lane_dy * speed, 0.0)
        x_after = jnp.take(lanes_top_x, shot_lane) + jnp.take(lane_dx_over_dy, shot_lane) * (
            y_after - float(self.consts.TOP_CLIP)
        )
        shot_pos = jnp.where(shot_active, jnp.stack([x_after, y_after]), shot_pos)

        moved_offscreen = shot_pos[1] > float(self.consts.BOTTOM_CLIP)
        shot_pos = jnp.where(moved_offscreen, offscreen, shot_pos)
        shot_timer = jnp.where(moved_offscreen, 0, shot_timer)
        shot_active = jnp.logical_and(shot_active, jnp.logical_not(moved_offscreen))

        player_xy = jnp.array([state.level.player_pos, float(self.consts.PLAYER_POS_Y)], dtype=jnp.float32)
        shot_xy = shot_pos.T
        delta = jnp.abs(shot_xy - player_xy[None, :])
        player_hit_radius = jnp.array(
            [float(self.consts.PLAYER_WIDTH) / 2.0, float(self.consts.PLAYER_HEIGHT) / 2.0],
            dtype=jnp.float32,
        )
        hits = jnp.logical_and.reduce(jnp.array([
            shot_active,
            delta[:, 0] <= player_hit_radius[0],
            delta[:, 1] <= player_hit_radius[1],
        ]))

        hit_count = jnp.sum(hits.astype(jnp.int32))
        lives = jnp.maximum(state.lives - hit_count, 0)

        shot_pos = jnp.where(hits[None, :], offscreen, shot_pos)
        shot_timer = jnp.where(hits, 0, shot_timer)
        return shot_pos, shot_lane, shot_timer, lives



    def _line_step(self, state: BeamriderState):

    
        velocities = state.level.line_velocities * self.consts.LINE_ACCELERATION
        velocities = jnp.clip(velocities, a_max= self.consts.MAX_LINE_VEL) 
        positions = state.level.line_positions + 2 * velocities #LINE accelerates twice as fast as constant
        positions = jnp.where(positions > self.consts.MAX_BLUE_LINE_POS, self.consts.BLUE_LINE_OFFSCREEN_Y, positions)
        trigger_reset = jnp.all(positions >= self.consts.NEW_LINE_THRESHHOLD_BOTTOM_LINE) & jnp.any(positions > self.consts.MAX_BLUE_LINE_POS)
        idx_to_reset = jnp.argmax(positions >= self.consts.BLUE_LINE_OFFSCREEN_Y)
        positions_with_reset = positions.at[idx_to_reset].set(46)
        velocities_with_reset = velocities.at[idx_to_reset].set(self.consts.NEW_LINE_VEL)
        positions = jnp.where(trigger_reset, positions_with_reset, positions)
        velocities = jnp.where(trigger_reset, velocities_with_reset, velocities)

        return (positions, velocities)

    def _bullet_infos(self, state: BeamriderState):
        shot_y = state.level.player_shot_pos[1]
        bullet_type = state.level.bullet_type

        laser_exists = jnp.all(jnp.array([shot_y >= self.consts.MAX_LASER_Y, 
                                          shot_y <= self.consts.MIN_BULLET_Y,
                                          bullet_type == self.consts.LASER_ID]))
        torpedo_exists = jnp.all(jnp.array([shot_y >= self.consts.MAX_TORPEDO_Y,
                                           shot_y <= self.consts.MIN_BULLET_Y,
                                           bullet_type == self.consts.TORPEDO_ID]))
        bullet_exists = jnp.logical_or(torpedo_exists, laser_exists)
        
        return(bullet_exists)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BeamriderState):
        # delegate to the renderer
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
    
    def _get_info(self, state: BeamriderState) -> BeamriderInfo:
        return BeamriderInfo(
            score=state.score,
            sector=state.sector,
        )

    def _get_reward(
        self, previous_state: BeamriderState, state: BeamriderState
    ) -> float:
        return state.score - previous_state.score

    def _get_done(self, state: BeamriderState) -> bool:
        return state.lives <= 0

class BeamriderRenderer(JAXGameRenderer):
    def __init__(self, consts=None):
        super().__init__()
        self.consts = consts or BeamriderConstants()
        self.rendering_config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )

        self.jr = render_utils.JaxRenderingUtils(self.rendering_config)

        # 1. Create procedural assets:
        # background_sprite = self._create_background_sprite()
        # player_sprite = self._create_player_sprite()

        #2 Update asset config to include sprites 
        asset_config = self._get_asset_config()
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/beamrider"

        # 3. Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    # def _create_player_sprite(self) -> jnp.ndarray:
    #     """Procedurally creates an RGBA sprite for the background"""
    #     player_color_rgba = (214, 239, 30, 0.7) # e.g., (236, 236, 236, 255)
    #     player_dimensions = (self.consts.PLAYER_HEIGHT, self.consts.PLAYER_WIDTH, 4)
    #     player_sprite = jnp.tile(jnp.array(player_color_rgba, dtype=jnp.uint8), (*player_dimensions[:2], 1))
    #     return player_sprite
    
    def _get_asset_config(self) -> list:
        """Returns the declarative manifest of all assets for the game, including both wall sprites."""
        return [
            {'name': 'background_sprite', 'type': 'background', 'file': 'background.npy'},
            {'name': 'player_sprite', 'type': 'single', 'file': 'player.npy'},
            {'name': 'white_ufo', 'type': 'group', 'files': ['Ufo_Player_Stage_1.npy', 'Ufo_Player_Stage_2.npy', 'Ufo_Player_Stage_3.npy', 'Ufo_Player_Stage_4.npy', 'Ufo_Player_Stage_5.npy', 'Ufo_Player_Stage_6.npy', 'Ufo_Player_Stage_7.npy']},
            {'name': 'laser_sprite', 'type': 'single', 'file': 'Laser.npy'},
            {'name': 'bullet_sprite', 'type': 'group', 'files': ['Laser.npy', 'Torpedo/Torpedo_3.npy', 'Torpedo/Torpedo_2.npy', 'Torpedo/Torpedo_1.npy']},
            {'name': 'enemy_shot', 'type': 'group', 'files': ['Enemy_Shot/Enemy_Shot_Vertical.npy', 'Enemy_Shot/Enemy_Shot_Horizontal.npy']},
            {'name': 'blue_line', 'type': 'single', 'file': 'blue_line.npy'},
            {'name': 'torpedos_left', 'type': 'single', 'file': 'torpedos_left.npy'}
        ]
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state:BeamriderState) -> chex.Array:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self._render_blue_lines(raster, state)
        raster = self._render_torpedo_icons(raster, state)
        raster = self._render_player_and_bullet(raster, state)
        raster = self._render_enemy_shots(raster, state)
        raster = self._render_white_ufos(raster, state)
        return self.jr.render_from_palette(raster, self.PALETTE)

    def _render_blue_lines(self, raster, state):
        """Draw the scrolling foreground lines."""

        blue_line_mask = self.SHAPE_MASKS["blue_line"]
        for idx in range(6):
            raster = self.jr.render_at_clipped(
                raster, 8, state.level.line_positions[idx], blue_line_mask
            )
        return raster

    def _render_torpedo_icons(self, raster, state):
        """Render the torpedo inventory indicator on the HUD."""

        torpedo_mask = self.SHAPE_MASKS["torpedos_left"]
        icon_config = [(3, 128), (2, 136), (1, 144)]
        for threshold, y in icon_config:
            y_pos = jnp.where(state.level.torpedoes_left >= threshold, y, 500)
            raster = self.jr.render_at_clipped(raster, y_pos, 32, torpedo_mask)
        return raster

    def _render_player_and_bullet(self, raster, state):
        player_mask = self.SHAPE_MASKS["player_sprite"]
        bullet_mask = self.SHAPE_MASKS["bullet_sprite"][
            self._get_index_bullet(state.level.player_shot_pos[1], state.level.bullet_type)
        ]
        raster = self.jr.render_at(raster, state.level.player_pos, self.consts.PLAYER_POS_Y, player_mask)
        raster = self.jr.render_at_clipped(
            raster,
            state.level.player_shot_pos[0] + self._get_bullet_alignment(state.level.player_shot_pos[1], state.level.bullet_type),
            state.level.player_shot_pos[1],
            bullet_mask,
        )
        return raster

    def _render_enemy_shots(self, raster, state):
        enemy_shot_masks = self.SHAPE_MASKS["enemy_shot"]
        for idx in range(3):
            timer = state.level.enemy_shot_timer[idx]
            sprite_idx = (jnp.floor_divide(timer, 4) % 2).astype(jnp.int32)
            y_pos = jnp.where(
                state.level.enemy_shot_pos[1][idx] <= self.consts.BOTTOM_CLIP,
                state.level.enemy_shot_pos[1][idx],
                500,
            )
            raster = self.jr.render_at_clipped(
                raster, state.level.enemy_shot_pos[0][idx], y_pos, enemy_shot_masks[sprite_idx]
            )
        return raster

    def _render_white_ufos(self, raster, state):
        white_ufo_masks = self.SHAPE_MASKS["white_ufo"]
        for idx in range(3):
            sprite_idx = self._get_index_ufo(state.level.white_ufo_pos[1][idx]) - 1
            raster = self.jr.render_at_clipped(
                raster,
                state.level.white_ufo_pos[0][idx] + self._get_ufo_alignment(state.level.white_ufo_pos[1][idx]),
                state.level.white_ufo_pos[1][idx],
                white_ufo_masks[sprite_idx],
            )
        return raster

    def _get_index_ufo(self,pos)->chex.Array:
        stage_1 = (pos >= 0).astype(jnp.int32)
        stage_2 = (pos >= 48).astype(jnp.int32)
        stage_3 = (pos >= 57).astype(jnp.int32) 
        stage_4 = (pos >= 62).astype(jnp.int32) #in reference game he chills there for a frame, only then switches
        stage_5 = (pos >= 69).astype(jnp.int32) #in reference game he chills there for a frame, only then switches
        stage_6 = (pos >= 86).astype(jnp.int32) #ab hier werden die schneller
        stage_7 = (pos >= 121).astype(jnp.int32)
        return stage_1 + stage_2 + stage_3 + stage_4 + stage_5 + stage_6 + stage_7
    
    def _get_ufo_alignment(self,pos)->chex.Array:
        stage_1 = (pos >= 0).astype(jnp.int32)
        stage_2 = (pos >= 48).astype(jnp.int32)
        stage_3 = (pos >= 57).astype(jnp.int32) 
        stage_4 = (pos >= 62).astype(jnp.int32) 
        stage_5 = (pos >= 69).astype(jnp.int32) 
        stage_6 = (pos >= 86).astype(jnp.int32) 
        stage_7 = (pos >= 121).astype(jnp.int32)
        return 4-(stage_1+stage_2+stage_3+stage_5+stage_7)
    
    def _get_index_bullet(self, pos, bullet_type) -> chex.Array:
        stage_1 = (pos >= 100).astype(jnp.int32)
        stage_2 = (pos >= 80).astype(jnp.int32)
        stage_3 = (pos >= 0).astype(jnp.int32) 
        result = jnp.where(bullet_type == self.consts.LASER_ID, 0, stage_1 + stage_2 + stage_3)
        return result
    
    def _get_bullet_alignment(self, pos, bullet_type) -> chex.Array:
        stage_1 = (pos >= 100).astype(jnp.int32)
        stage_2 = (pos >= 80).astype(jnp.int32)
        stage_3 = (pos >= 0).astype(jnp.int32) 
        #default alignment if smallest torpedo is +3
        #if bullet is laser, no offset
        result = jnp.where(bullet_type == self.consts.LASER_ID, 0, 4-(stage_1 + stage_2 + stage_3))
        return result
