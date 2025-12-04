"""
Authors: Lasse Reith, Benedikt Schwarz, Shir Nussbaum
"""
#active_development branch
from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax.lax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class BeamriderConstants(NamedTuple):
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
    LANES: Tuple[int, int, int, int, int] = (27,52,77,102,127)
    TOP_OF_LANES: Tuple[int, int, int, int, int] = (38,61,71,81,91,102,123)
    ENEMY_LANE_VECTORS: Tuple[Tuple[float, float],Tuple[float, float],Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-1.5,4),(-1, 4), (-0.52, 4), (0,4), (0.52, 4), (1, 4),(1.5,4))


    MAX_LASER_Y: int = 67
    MIN_BULLET_Y:int =156
    MAX_TORPEDO_Y: int = 60
    LANE_VECTORS: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-1, 4), (-0.52, 4), (0,4), (0.52, 4), (1, 4))

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


    INTERVAL_PER_MOVE = jnp.array([
    # Moves  0–5  -> ~8–9
    10, 9, 9, 9, 9, 9, 9,
    # Moves  6–11 -> ~8
    8, 8, 8, 8, 8, 8,
    # Moves 12–15 -> ~6
    6, 6, 6, 6,
    # Moves 16–23 -> ~5
    5, 5, 5, 5, 5, 5, 5, 5,
    # Moves 24–31 -> ~4
    4, 4, 4, 4, 4, 4, 4, 4,
    # Moves 32+   -> ~3
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3

])
    INIT_LINE_POS = jnp.array([118.08385, 90.88263, 156.90707, 49.115276, 58.471092, 71.82423 ])

    INIT_LINE_VEL = jnp.array([0.31581876, 0.22127658, 0.4507547, 0.07610765, 0.10862522, 0.15503621])
    NEW_LINE_VEL = 0.06528
    LINE_ACCELERATION = 1.007
    MAX_LINE_VEL = 0.6665
    BLUE_LINE_OFFSCREEN_Y = 500
    NEW_LINE_THRESHHOLD_BOTTOM_LINE = 54.0



class LevelState(NamedTuple):
    player_pos:chex.Array
    player_vel: chex.Array
    white_ufo_left: chex.Array           #int                   #int
    comet_positions: chex.Array          #int
    mothership_position: chex.Array      #int
    player_shot_pos:chex.Array          #intArray
    player_shot_vel:chex.Array          #intArray
    torpedoes_left: chex.Array
    shooting_cooldown: chex.Array
    bullet_type: chex.Array             # (0 = grad gibt's nix, 1 = laser, 2 = Torpedo)

    #enemies
    enemy_type : chex.Array              #int
    white_ufo_pos : chex.Array               #( (1,3) , (2,4) , ... )
    white_ufo_vel : chex.Array
    enemy_shot_pos : chex.Array          
    enemy_shot_vel : chex.Array
    white_ufo_time_on_lane: chex.Array
    white_ufo_time_allowed:chex.Array

    line_positions: chex.Array
    line_velocities: chex.Array 

class BeamriderState(NamedTuple):
    level : LevelState
    score: chex.Array
    sector: chex.Array                  #current level
    level_finished:chex.Array
    reset_coords:chex.Array
    lives:chex.Array
    steps: chex.Array
    rng: chex.Array


class BeamriderInfo(NamedTuple):
    score: chex.Array
    sector: chex.Array

class BeamriderObservation(NamedTuple):
    pos: chex.Array                      #player position
    shooting_cd:chex.Array               #int             
    torpedoes_left:chex.Array            #int
    player_shots_pos:chex.Array          #intArray
    player_shots_vel:chex.Array          #intArray
    white_ufo_left:chex.Array

    #enemies
    enemy_type: chex.Array
    white_ufo_pos:chex.Array
    white_ufo_vel :chex.Array
    enemy_shot_pos : chex.Array          
    enemy_shot_vel : chex.Array

    

class JaxBeamrider(JaxEnvironment[BeamriderState,BeamriderObservation,BeamriderInfo,BeamriderConstants]):
    def __init__(self,consts: BeamriderConstants = None ):
        super().__init__(consts)
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
            Action.DOWNLEFTFIRE
        ]
        self.consts = consts or BeamriderConstants()
        self.obs_size = 111
        self.renderer = BeamriderRenderer(self.consts)
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[BeamriderObservation, BeamriderState]:
        state = self.reset_level(1)
        #initial_obs = self._get_observation(state)  #not sure what this does, pong only?
        obs = self._get_observation(state)
        return obs, state

    def reset_level(self, next_level=1) -> BeamriderState:
        #next_level = jnp.clip(next_level, 1, 3)
        
        new_state = BeamriderState(
            level = LevelState(
                player_pos=jnp.array(77.0),
                player_vel=jnp.array(0.0),
                white_ufo_left=jnp.array(15),
                comet_positions=jnp.array(0),
                mothership_position=jnp.array(0),     
                player_shot_pos=jnp.array(self.consts.BULLET_OFFSCREEN_POS),
                player_shot_vel=jnp.array([0,0]),
                torpedoes_left=jnp.array(3),
                shooting_cooldown=jnp.array(0),
                bullet_type=jnp.array(self.consts.LASER_ID),

                #enemies
                enemy_type=jnp.array([0, 0, 0]),
                white_ufo_pos=jnp.array([[77.0,77.0,77.0],[43.0,43.0,43.0]]),
                white_ufo_vel=jnp.array([[-0.5,0.5,0.3],[0.0,0.0,0.0]]),
                enemy_shot_pos=jnp.array([[0,0,0],[0,0,0]]),       
                enemy_shot_vel=jnp.array([0, 0, 0]),
                white_ufo_time_on_lane=jnp.array([0,0,0]),
                white_ufo_time_allowed=jnp.array([400,600,800]),

                line_positions= self.consts.INIT_LINE_POS,
                line_velocities= self.consts.INIT_LINE_VEL
            ),
                score=jnp.array(0),
                sector=jnp.array(next_level),                #current level
                level_finished=jnp.array(0),
                reset_coords=jnp.array(False),
                lives=jnp.array(3), 
                steps= jnp.array(0),
                rng= jnp.array(jax.random.key(42))
        )
        return new_state
    

    def _get_observation(self, state: BeamriderState) -> BeamriderObservation:
        return BeamriderObservation(
            pos=state.level.player_pos,
            shooting_cd=state.level.shooting_cooldown,    
            torpedoes_left = state.level.torpedoes_left,
            player_shots_pos = state.level.player_shot_pos,
            player_shots_vel = state.level.player_shot_vel,
            white_ufo_left = state.level.white_ufo_left,

            #enemies
            enemy_type= state.level.enemy_type,
            white_ufo_pos=state.level.white_ufo_pos,
            white_ufo_vel =state.level.white_ufo_vel,
            enemy_shot_pos =state.level.enemy_shot_pos,        
            enemy_shot_vel =state.level.enemy_shot_vel,
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step(self, state: BeamriderState, action: chex.Array) -> Tuple[BeamriderObservation, BeamriderState, float, bool, BeamriderInfo]:
        #reset_cond = jnp.any(jnp.array([action == self.consts.RESET])) #not sure wtf this is supposed to do
        # return state of new player step || WIP
        (
            player_x,
            vel_x,
            player_shot_position,
            player_shot_velocity,
            torpedos_left,
            bullet_type
        ) = self._player_step(state, action)

        #ufo test
        (white_ufo_1_position,white_ufo_1_vel_x,white_ufo_1_vel_y,white_ufo_1_time_on_lane,key) = self._white_ufo_1_step(state)
        (white_ufo_2_position,white_ufo_2_vel_x,white_ufo_2_vel_y,white_ufo_2_time_on_lane) = self._white_ufo_2_step(state)
        (white_ufo_3_position,white_ufo_3_vel_x,white_ufo_3_vel_y,white_ufo_3_time_on_lane) = self._white_ufo_3_step(state)

        white_ufo_pos = jnp.array([[white_ufo_1_position[0],white_ufo_2_position[0],white_ufo_3_position[0]],
                               [white_ufo_1_position[1],white_ufo_2_position[1],white_ufo_3_position[1]]])
        (
            white_ufo_pos, player_shot_position
         ) = self._collision_handler(state, white_ufo_pos, player_shot_position, bullet_type)
        
        (line_positions, line_velocities) = self._line_step(state)

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

            #enemies
            enemy_type=jnp.array([0, 0, 0]),
            white_ufo_pos=white_ufo_pos,
            white_ufo_vel=jnp.array([[white_ufo_1_vel_x,white_ufo_2_vel_x,white_ufo_3_vel_x],[0.0,0.0,0.0]]),
            enemy_shot_pos=jnp.array([[0,0,0],[0,0,0]]),       
            enemy_shot_vel=jnp.array([0, 0, 0]),
            white_ufo_time_on_lane=jnp.array([white_ufo_1_time_on_lane,white_ufo_2_time_on_lane,white_ufo_3_time_on_lane]),
            white_ufo_time_allowed=state.level.white_ufo_time_allowed,

            line_positions= line_positions,
            line_velocities= line_velocities
        )

        new_state = BeamriderState(
            level=new_level_state,
            score=jnp.array(0),
            sector=jnp.array(1),                #current level
            level_finished=jnp.array(0),
            reset_coords=jnp.array(False),
            lives=jnp.array(3),
            steps=next_step,
            rng= key
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

        is_in_lane = jnp.isin(x, jnp.array(self.consts.LANES)) # predicate: x is one of LANES

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

        lanes = jnp.array(self.consts.LANES)
        lane_velocities = jnp.array(self.consts.LANE_VECTORS)
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
    



    def _white_ufo_1_step(self, state: BeamriderState):
        white_ufo_1_position= jnp.array([state.level.white_ufo_pos[0][0],state.level.white_ufo_pos[1][0]])
        
        white_ufo_1_vel_x = state.level.white_ufo_vel[0][0]
        white_ufo_1_vel_y = state.level.white_ufo_vel[1][0]
        time_on_lane_1 = state.level.white_ufo_time_on_lane[0]

        p_swap = self.entropy_heat_prob(time_on_lane_1)
        key, subkey = jax.random.split(state.rng)

        new_y = white_ufo_1_position[1]
        jax.debug.print("x: {}", state.level.white_ufo_pos[0][0])


        white_ufo_1_vel_x,white_ufo_1_vel_y = jax.lax.cond(
            new_y == 43.0,
            lambda vel_x, vel_y: self._white_ufo_top_lane(white_ufo_1_position,white_ufo_1_vel_x,state.level.white_ufo_time_on_lane[0]),
            lambda vel_x, vel_y: self._white_ufo_normal(white_ufo_1_position,white_ufo_1_vel_x,white_ufo_1_vel_y),
            white_ufo_1_vel_x,white_ufo_1_vel_y
        )
        
        new_y = jax.lax.cond(
            time_on_lane_1 == state.level.white_ufo_time_allowed[0],
            lambda pos_y : pos_y + 1,
            lambda pos_y: pos_y,
            new_y
        )
        time_on_lane_1 += 1

        white_ufo_1_position = white_ufo_1_position.at[0].add(white_ufo_1_vel_x)
        white_ufo_1_position = white_ufo_1_position.at[1].add(white_ufo_1_vel_y)
        return white_ufo_1_position,white_ufo_1_vel_x,white_ufo_1_vel_y,time_on_lane_1,key
    
    def _white_ufo_2_step(self, state: BeamriderState):
        white_ufo_2_position= jnp.array([state.level.white_ufo_pos[0][1],state.level.white_ufo_pos[1][1]])
        
        white_ufo_2_vel_x = state.level.white_ufo_vel[0][1]
        white_ufo_2_vel_y = state.level.white_ufo_vel[1][1]
        time_on_lane_2 = state.level.white_ufo_time_on_lane[1]

        white_ufo_2_vel_x,white_ufo_2_vel_y = jax.lax.cond(
            white_ufo_2_position[1] == 43.0,
            lambda vel_x, vel_y: self._white_ufo_top_lane(white_ufo_2_position,white_ufo_2_vel_x,state.level.white_ufo_time_on_lane[1]),
            lambda vel_x, vel_y: self._white_ufo_normal(white_ufo_2_position,white_ufo_2_vel_x,white_ufo_2_vel_y),
            white_ufo_2_vel_x,white_ufo_2_vel_y
        )

        time_on_lane_2 += 1
        white_ufo_2_position = white_ufo_2_position.at[0].add(white_ufo_2_vel_x)
        return white_ufo_2_position,white_ufo_2_vel_x,white_ufo_2_vel_y,time_on_lane_2
    
    def _white_ufo_3_step(self, state: BeamriderState):
        white_ufo_3_position= jnp.array([state.level.white_ufo_pos[0][2],state.level.white_ufo_pos[1][2]])

        white_ufo_3_vel_x = state.level.white_ufo_vel[0][2]
        white_ufo_3_vel_y = state.level.white_ufo_vel[1][2]
        time_on_lane_3 = state.level.white_ufo_time_on_lane[2]

        white_ufo_3_vel_x,white_ufo_3_vel_y = jax.lax.cond(
            white_ufo_3_position[1] == 43.0,
            lambda vel_x, vel_y: self._white_ufo_top_lane(white_ufo_3_position,white_ufo_3_vel_x,state.level.white_ufo_time_on_lane[2]),
            lambda vel_x, vel_y: self._white_ufo_normal(white_ufo_3_position,white_ufo_3_vel_x,white_ufo_3_vel_y),
            white_ufo_3_vel_x,white_ufo_3_vel_y
        )
        time_on_lane_3 += 1
        white_ufo_3_position = white_ufo_3_position.at[0].add(white_ufo_3_vel_x)
        

        return white_ufo_3_position,white_ufo_3_vel_x,white_ufo_3_vel_y, time_on_lane_3
    
    ####################benes code
    def _white_ufo_top_lane(self, white_ufo_pos, white_ufo_vel_x,time_on_lane):
        
        white_ufo_vel_x = jax.lax.cond(
            white_ufo_pos[0] >= self.consts.RIGHT_CLIP_PLAYER,
            lambda v: -0.5,
            lambda v: v,
            white_ufo_vel_x,
        )

        white_ufo_vel_x = jax.lax.cond(
            white_ufo_pos[0]<= self.consts.LEFT_CLIP_PLAYER,
            lambda v: 0.5,
            lambda v: v,
            white_ufo_vel_x,
        )
        return white_ufo_vel_x,0.0
    
    ########
    def _white_ufo_normal(self, white_ufo_pos, white_ufo_vel_x, white_ufo_vel_y):
        x=white_ufo_pos[0]
        lanes = jnp.array(self.consts.TOP_OF_LANES)
        distances = jnp.abs(lanes-x)
        lane_id = jnp.argmin(distances)
        lane_vector = jnp.array(self.consts.ENEMY_LANE_VECTORS)[lane_id]

        def seek_lane():
            direction = jnp.sign(lanes[lane_id]-x)
            new_vx = direction * 0.5
            return new_vx,0.25
        
        def follow_lane():
            return lane_vector[0],lane_vector[1]
        
        new_vx,new_vy = jax.lax.cond(
            distances[lane_id]<= 0.25,
            lambda vx,vy:follow_lane(),
            lambda vx,vy:seek_lane(),
            white_ufo_vel_x,white_ufo_vel_y
        )

        return new_vx,new_vy



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
            {'name': 'blue_line', 'type': 'single', 'file': 'blue_line.npy'},
            {'name': 'torpedos_left', 'type': 'single', 'file': 'torpedos_left.npy'}
        ]
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state:BeamriderState) -> chex.Array:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        blue_line_mask = self.SHAPE_MASKS["blue_line"]
        # Blue_Line Render:
        raster = self.jr.render_at_clipped(raster,8 ,state.level.line_positions[0],blue_line_mask)
        raster = self.jr.render_at_clipped(raster,8 ,state.level.line_positions[1],blue_line_mask)
        raster = self.jr.render_at_clipped(raster,8 ,state.level.line_positions[2],blue_line_mask)
        raster = self.jr.render_at_clipped(raster,8 ,state.level.line_positions[3],blue_line_mask)
        raster = self.jr.render_at_clipped(raster,8 ,state.level.line_positions[4],blue_line_mask)
        raster = self.jr.render_at_clipped(raster,8 ,state.level.line_positions[5],blue_line_mask)

        # ------------------
        torpedos_left = self.SHAPE_MASKS["torpedos_left"]
        torpedos_left_y_pos_3 = jnp.where(state.level.torpedoes_left >= 3, 128, 500)
        torpedos_left_y_pos_2 = jnp.where(state.level.torpedoes_left >= 2, 136, 500)
        torpedos_left_y_pos_1 = jnp.where(state.level.torpedoes_left >= 1, 144, 500)
        raster = self.jr.render_at_clipped(raster, torpedos_left_y_pos_3, 32, torpedos_left)
        raster = self.jr.render_at_clipped(raster, torpedos_left_y_pos_2, 32, torpedos_left)
        raster = self.jr.render_at_clipped(raster, torpedos_left_y_pos_1, 32, torpedos_left)
# 100* (state.level.torpedoes_left >=1).astype(jnp.int32) 

        player_mask = self.SHAPE_MASKS["player_sprite"]
        # laser_mask = self.SHAPE_MASKS["laser_sprite"]

        bullet_mask = self.SHAPE_MASKS["bullet_sprite"] [self._get_index_bullet(state.level.player_shot_pos[1], state.level.bullet_type)]
        raster = self.jr.render_at(raster, state.level.player_pos, self.consts.PLAYER_POS_Y, player_mask)

        # raster = self.jr.render_at_clipped(raster, state.level.player_shot_pos[0], state.level.player_shot_pos[1], laser_mask)
        raster = self.jr.render_at_clipped(raster, state.level.player_shot_pos[0]+self._get_bullet_alignment(state.level.player_shot_pos[1],state.level.bullet_type), state.level.player_shot_pos[1], bullet_mask)

        #white ufo masks
        white_ufo_1_mask = self.SHAPE_MASKS["white_ufo"][self._get_index_ufo(state.level.white_ufo_pos[1][0])-1]
        raster = self.jr.render_at_clipped(raster,state.level.white_ufo_pos[0][0]+self._get_ufo_alignment(state.level.white_ufo_pos[1][0]),
                                           state.level.white_ufo_pos[1][0], white_ufo_1_mask)

        white_ufo_2_mask = self.SHAPE_MASKS["white_ufo"][self._get_index_ufo(state.level.white_ufo_pos[1][1])-1]
        raster = self.jr.render_at_clipped(raster,state.level.white_ufo_pos[0][1]+self._get_ufo_alignment(state.level.white_ufo_pos[1][1]),
                                           state.level.white_ufo_pos[1][1], white_ufo_2_mask)
        
        white_ufo_3_mask = self.SHAPE_MASKS["white_ufo"][self._get_index_ufo(state.level.white_ufo_pos[1][2])-1]
        raster = self.jr.render_at_clipped(raster,state.level.white_ufo_pos[0][2]+self._get_ufo_alignment(state.level.white_ufo_pos[1][2]),
                                           state.level.white_ufo_pos[1][2], white_ufo_3_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)

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


