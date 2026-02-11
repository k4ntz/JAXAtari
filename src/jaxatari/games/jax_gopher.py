from argparse import Action
import os
from functools import partial
from typing import NamedTuple, Tuple, List
from enum import IntEnum

import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# ==========================================================================================
#  ASSET CONFIGURATION
# =======================================================================================
def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'group', 'files': [
            'farmer_default.npy', 
            'farmer_shovel_middle.npy', 
            'farmer_shovel_top.npy'
            ]},
        {'name': 'gopher', 'type': 'group', 'files': [
            'gopher_walk_left_legsclose.npy', 
            'gopher_walk_left_legsopen.npy', 
            'gopher_walk_right_legsclosed.npy', 
            'gopher_walk_right_legsopen.npy', 
            'gopher_stand_up.npy',
            'gopher_tongue_left_state1.npy',
            'gopher_tongue_left_state2.npy',
            'gopher_tongue_right_state1.npy',
            'gopher_tongue_right_state2.npy'
            ]},
        {'name': 'carrot', 'type': 'single', 'file': 'carrot.npy'},
        {'name': 'hole_tile_dug', 'type': 'digits', 'pattern': 'hole_tile_dug.npy'},
        {'name': 'tunnel_tile_dug', 'type': 'single', 'file': 'tunnel_tile_dug.npy'},
        {'name': 'bottom_ground', 'type': 'single', 'file': 'bottom_ground.npy'},
        {'name': 'player', 'type': 'group', 'files': [
            'farmer_default.npy', 
            'farmer_shovel_middle.npy', 
            'farmer_shovel_top.npy', 
            'farmer_seed_default.npy',
            'farmer_seed_shovel_middle.npy',
            'farmer_seed_shovel_up.npy',]},
        {'name': 'numbers', 'type': 'group', 'files': [
            'score_0.npy', 'score_1.npy', 'score_2.npy', 'score_3.npy', 
            'score_4.npy', 'score_5.npy', 'score_6.npy', 'score_7.npy', 
            'score_8.npy', 'score_9.npy']},
        {'name': 'duck', 'type': 'group', 'files': [
            'duck_left_middle.npy',
            'duck_left_up.npy',
            'duck_left_down.npy',
            'duck_right_middle.npy',
            'duck_right_up.npy',
            'duck_right_down.npy']},
        {'name': 'seed', 'type': 'single', 'file': 'seed.npy'}
    )

# ==========================================================================================
#  GAME CONSTANTS
# ==========================================================================================

class GopherConstants(NamedTuple):
    # --- Screen / Dimensions ---
    WIDTH: int = 160                                   
    HEIGHT: int = 210
    NUM_TILES: int = 40
    TILE_WIDTH: int = 4                                # Tunnel tile width
    

    # --- Entity Sizes ---
    PLAYER_SIZE: Tuple[int, int] = (13, 50)
    GOPHER_SIZE: Tuple[int, int] = (14, 12)
    CARROT_SIZE: Tuple[int, int] = (7, 15)
    SEED_SIZE: Tuple[int, int] = (1, 1)
    DUCK_SIZE:Tuple[int, int] = (15, 15)

    # --- Positions / Layout ---
    LEFT_WALL: int = 0
    RIGHT_WALL: int = 160       
    TUNNEL_TOP_Y: int = 182
    TUNNEL_BOTTOM_Y: int = 194
    WATER_Y_POS: int = 145
    L3_BOTTOM_Y: float = 176.0
    STEAL_Y_POS: float = 150.0
    PEEK_DECISION_Y: float = 168.0
    CEILING_Y: float = 148.0
    
    PLAYER_START_X: float = 74.0
    PLAYER_START_Y: float = 96.0
    GOPHER_START_X: float = 144.0
    GOPHER_START_Y: float = 183.0
    GOPHER_TOP_Y: float = 148.0
    DUCK_Y_POS: float = 30                            
    DUCK_SPAWN_MIN_X: float = 20.0          
    DUCK_SPAWN_MAX_X: float = 140.0         
    SCORE_Y_POS: int = 9
    CARROT_Y_POS: int = 151
    
    CARROT_X_POSITION: Tuple[int, int, int] = (60, 76, 92)
    HOLE_POSITION_X: Tuple = (12, 28, 44, 108, 124, 140)
    HOLE_POSITION_Y: Tuple = (175, 168, 161)                                    # 3 layers
    TUNNEL_POSITION: List = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36,               
                             40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 
                             80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 
                             120, 124, 128, 132, 136, 140, 144, 148, 152, 156}
    SCORE_X_POSITION: Tuple = (74, 81, 89, 97)
    
    # --- Speeds ---
    PLAYER_SPEED: float = 1.0
    GOPHER_SPEED_X: float = 4.0 / 3.0    
    GOPHER_SPEED_Y: float = 1.0
    GOPHER_SPEED_SMART_X: float = 8.0 / 3.0
    STEAL_SPEED_X: float = 1.8 
    DUCK_SPEED: float = 1.5 
    SEED_DROP_SPEED: float = 1.0                             
    
    # --- Setting ---
    NUM_CARROTS: int = 3

    # --- Offset / Threshold ---
    STAND_OFFSET: int = 8.0
    PUSH_DOWN_OFFSET: int = 3.0
    
    BONK_X_TOLERANCE: float = 13.0
    BONK_Y_THRESHOLD: float = 170.0
    SPRITE_OFFSET_PLAYER_SEED: int = 3

    # --- Timers(in frames) ---
    TIME_TO_PREPARE_CLIMB: int = 5   
    TIME_TO_RECOVER_DIG: int = 7            # Freeze after returning from digging up
    TIME_TO_PEEK: int = 24
    TIME_TO_PREPARE_STEAL: int = 5
    TIME_TO_START_DELAY: int = 60           # 60 Frames freeze at start of round
    TIME_DUCK_COOLDOWN: int = 600
    
    # --- Probabilities ---
    PROB_TURN_AFTER_DIG_NORMAL: float = 0.5
    PROB_DIG_L1: float = 0.4         
    PROB_DIG_L2: float = 0.2        
    PROB_DIG_L3: float = 0.6         
    PROB_CONTINUE_AFTER_L3: float = 0.8  
    PROB_PEEK_AT_SURFACE: float = 0.5    
    PROB_STEAL_NORMAL: float = 1.0          
    PROB_DUCK_SPAWN: float = 0.02
    # Smarter gopher probabilities
    PROB_STEAL_SMART: float = 0.9
    PROB_TURN_AFTER_DIG_SMART: float = 0.2

    SCORE_FOR_DUCK: int = 500               # Threshold for bonus duck to show up
    ASSET_CONFIG: tuple = _get_default_asset_config()

class GopherState(NamedTuple):
    # --- Player ---
    player_x: chex.Array  
    player_speed: chex.Array                        
    player_has_seed: chex.Array             # 0 = No, 1 = Yes
    bonk_timer: chex.Array
    
    # --- Gopher ---
    gopher_position: chex.Array            
    gopher_direction_x: chex.Array
    gopher_move_x_timer: chex.Array
    gopher_action: chex.Array
    gopher_timer: chex.Array
    gopher_target_idx: chex.Array   
    gopher_target_layer: chex.Array

    # --- Environment ---
    carrots_present: chex.Array             # shape(3, ), 1 for present, 0 for absent
    tunnel_layout: chex.Array               # shape (42,) 42 tunnel tiles, 1 for dug, 0 for undug
    hole_layout: chex.Array                 # shape (6, 3), 6 holes, each hole has 3 layers, 1 for dug, 0 for undug
    
    # --- Duck / Seed ---
    duck_active: chex.Array                 # 0 = Inactive, 1 = Active
    duck_x: chex.Array
    duck_dir: chex.Array                    # 1 = Right, -1 = Left
    duck_drop_x: chex.Array        
    duck_cool_down_timer: chex.Array        # For cool down after showing up
    duck_anim_timer: chex.Array

    seed_active: chex.Array                 # 0 = Inactive, 1 = Falling
    seed_x: chex.Array
    seed_y: chex.Array
    
    # --- Fields with Defaults ---
    score: chex.Array
    key: chex.Array                   
    frame_count: int = 0


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class GopherObservation(NamedTuple):
    player: EntityPosition
    gopher: EntityPosition
    duck: EntityPosition
    holes: jnp.ndarray
    carrots: jnp.ndarray
    seeds: jnp.ndarray  
    score_player: jnp.ndarray
    gopher_state: jnp.ndarray  
       
class GopherInfo(NamedTuple):
    difficulty_level: jnp.ndarray
    time:jnp.ndarray

class GopherAction(IntEnum): 
    IDLE = 0
    WALKING = 1
    DIGGING_TUNNEL = 2
    DIGGING_DOWN = 3    
    DIGGING_UP = 4 
    SEEING_CARROT = 5                           # Peek
    PREPARE_CLIMB = 6                           # Freeze 5 frames before Layer 2
    RECOVER_LAND = 7                            # Freeze 7 frames after returning
    PREPARE_STEAL = 8                           # Freeze 8 frames on surface after peek
    STEALING = 9        
    START_DELAY = 10                            # Freeze before start
    

    

# ==========================================================================================
#   MAIN GAME LOGIC
# ==========================================================================================

class JaxGopher(JaxEnvironment[GopherState, GopherObservation, GopherInfo, GopherConstants]):
    def __init__(self, consts: GopherConstants = None):
        consts = consts or GopherConstants()
        super().__init__(consts)     
        self.renderer = GopherRenderer(self.consts)
        self.action_set = [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
            Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE
        ]
    
    def _player_step(self, state: GopherState, action: chex.Array) -> GopherState:
        """
        Handles player horizontal movement and bonking input.
        """

        # --- Player walking ---
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        new_speed = jax.lax.select(left, -self.consts.PLAYER_SPEED, jax.lax.select(right, self.consts.PLAYER_SPEED, 0.0))

        # Wall Collision
        touch_left_wall = state.player_x <= self.consts.LEFT_WALL
        touch_right_wall = state.player_x + self.consts.PLAYER_SIZE[0] >= self.consts.RIGHT_WALL
        final_speed = jax.lax.cond(
            jnp.logical_or(jnp.logical_and(left, touch_left_wall), jnp.logical_and(right, touch_right_wall)),
            lambda _: 0.0, lambda _: new_speed, operand = None
        )

        # Freeze player if Game is in Start Delay
        is_frozen = state.gopher_action == GopherAction.START_DELAY
        final_speed = jax.lax.select(is_frozen, 0.0, final_speed)
        proposed_player_x = jnp.clip(
            state.player_x + final_speed,
            self.consts.LEFT_WALL,
            self.consts.RIGHT_WALL - self.consts.PLAYER_SIZE[0],
        )

        # --- Fire bonk logic ---
        fire_pressed = (action == Action.FIRE) | (action == Action.DOWNFIRE) | \
                      (action == Action.LEFTFIRE) | (action == Action.RIGHTFIRE)
        valid_fire = fire_pressed & jnp.logical_not(is_frozen)
        is_bonking = state.bonk_timer > 0
        
        # Timer increments up to 14 frames then resets
        def increment_timer(t): return jax.lax.select(t >= 14, 0 , t + 1)
        new_bonk_timer = jax.lax.cond(
            jnp.logical_or(valid_fire, is_bonking),
            increment_timer, lambda t: 0, operand = state.bonk_timer
        )

        new_state = state._replace(
            player_x = proposed_player_x, 
            player_speed = final_speed,
            bonk_timer = new_bonk_timer
        )
        return new_state, fire_pressed

    def _handle_repairs(self, state: GopherState) -> GopherState:
        """
        Handles filling in holes/tunnels if player bonks near them.
        """

        # Priority check: bonking over repairing when enemy's there
        is_hole_bonk, is_run_bonk = self._check_bonk_hit(state)
        is_bonking_gopher = is_hole_bonk | is_run_bonk
        
        # Setup player location
        player_center = state.player_x + (self.consts.PLAYER_SIZE[0] / 2.0)
        hole_positions = jnp.array(self.consts.HOLE_POSITION_X)
        dist_to_holes = jnp.abs(player_center - hole_positions)
        h_idx = jnp.argmin(dist_to_holes)
        is_near_hole = jnp.min(dist_to_holes) < 8.0 
        
        # Check if gopher blocks hole
        gopher_center = state.gopher_position[0] + (self.consts.GOPHER_SIZE[0] / 2.0)
        hole_center = hole_positions[h_idx]
        gopher_aligned_with_hole = jnp.abs(gopher_center - hole_center) < 8.0
        is_blocked_by_gopher = gopher_aligned_with_hole
        # Valid repair condition
        is_repairing = (state.bonk_timer == 1) & is_near_hole & \
                       jnp.logical_not(is_bonking_gopher) & \
                       jnp.logical_not(is_blocked_by_gopher)

        # Determine layer repair order(bottom -> top, l1 -> l2 -> l3)
        l3_dug = state.hole_layout[h_idx, 2] == 1
        l2_dug = state.hole_layout[h_idx, 1] == 1
        l1_dug = state.hole_layout[h_idx, 0] == 1
        
        # Map holes to tunnel tiles
        t1 = (hole_positions[h_idx] // 4).astype(jnp.int32)
        t2 = t1 + 1
        t_left_outer = jnp.maximum(0, t1 - 1)
        t_right_outer = jnp.minimum(self.consts.NUM_TILES - 1, t2 + 1)
        
        tunnel_dug = (state.tunnel_layout[t1] == 1) | (state.tunnel_layout[t2] == 1)
        
        # Repair lowest reachable hole layer
        reach_l3 = l3_dug                              
        reach_l2 = l2_dug & l3_dug                     
        reach_l1 = l1_dug & l2_dug & l3_dug            
        reach_tun = tunnel_dug & l1_dug & l2_dug & l3_dug               # Tunnel needs all open
        
        do_tun = reach_tun
        do_l1  = jnp.logical_not(do_tun) & reach_l1
        do_l2  = jnp.logical_not(do_tun) & jnp.logical_not(do_l1) & reach_l2
        do_l3  = jnp.logical_not(do_tun) & jnp.logical_not(do_l1) & jnp.logical_not(do_l2) & reach_l3
        
        # --- Apply Tunnel Repair ---
        new_tunnel_layout = jax.lax.cond(
            is_repairing & do_tun,
            lambda t: t.at[t1].set(0).at[t2].set(0).at[t_left_outer].set(0).at[t_right_outer].set(0),
            lambda t: t,
            operand=state.tunnel_layout
        )
        
        # --- Apply Hole Repair ---
        new_hole_layout = jax.lax.cond(
            is_repairing,
            lambda h: jax.lax.select(do_l1, h.at[h_idx, 0].set(0),
                      jax.lax.select(do_l2, h.at[h_idx, 1].set(0),
                      jax.lax.select(do_l3, h.at[h_idx, 2].set(0),
                      h))), 
            lambda h: h,
            operand=state.hole_layout
        )

        # Score calculation(20 points for each repair layer)
        did_repair = is_repairing & (do_tun | do_l1 | do_l2 | do_l3)
        score_increase = jax.lax.select(did_repair, 20, 0)
        return state._replace(tunnel_layout=new_tunnel_layout, hole_layout=new_hole_layout, score=state.score + score_increase)

    
    
    def _locate_gopher(self, state: GopherState):
        """
        Helper to find gopher related location (nearest hole, tile index).
        """
        current_y = state.gopher_position[1]
        current_x = state.gopher_position[0]
        is_at_floor = current_y >= self.consts.GOPHER_START_Y
        
        hole_x_pos = jnp.array(self.consts.HOLE_POSITION_X, dtype=jnp.float32)
        dist_to_holes = jnp.abs(current_x - hole_x_pos)
        h_idx = jnp.argmin(dist_to_holes)
        is_under_hole = dist_to_holes[h_idx] < 4.0 

        # Calculate head position
        head_x = jax.lax.select(state.gopher_direction_x == 1, current_x + 13.0, current_x)
        current_tile_idx = (head_x // self.consts.TILE_WIDTH).astype(jnp.int32)
        
        # Calculate alignment for climbing decision (center body)
        center_x = current_x + 6.0
        offset_x = center_x % float(self.consts.TILE_WIDTH)
        is_aligned_x = jnp.min(jnp.array([offset_x, 4.0 - offset_x])) < 1.0

        return {
            "is_at_floor": is_at_floor, 
            "current_tile_idx": current_tile_idx, 
            "h_idx": h_idx,
            "is_under_hole": is_under_hole, 
            "is_aligned_x": is_aligned_x
        }
    
    def _check_bonk_hit(self, state: GopherState) -> Tuple[bool, bool]:
        """ 
        Detects if Player hit Gopher.
        """
        # Check Player Attack Window 
        is_attacking = (state.bonk_timer > 0) & (state.bonk_timer < 10)
        
        # Check Horizontal Alignment
        p_center = state.player_x + (self.consts.PLAYER_SIZE[0] / 2.0)
        g_center = state.gopher_position[0] + (self.consts.GOPHER_SIZE[0] / 2.0)
        dist = jnp.abs(p_center - g_center)
        is_aligned = dist < self.consts.BONK_X_TOLERANCE
        
        hit_connects = is_attacking & is_aligned
        
        # Determine Hit Type based on Gopher State
        act = state.gopher_action
        y = state.gopher_position[1]
        
        # Type 1: Run Bonk 
        is_run_bonk = hit_connects & (act == GopherAction.STEALING)
        
        # Type 2: Hole Bonk (Peeking, Prep, or High Dig)
        # is_peeking = (act == GopherAction.SEEING_CARROT)
        # is_prepping = (act == GopherAction.PREPARE_STEAL)
        # is_climbing_exposed = (act == GopherAction.DIGGING_UP) & (y < self.consts.BONK_Y_THRESHOLD)
        
        # is_exposed = is_peeking | is_prepping | is_climbing_exposed
        is_exposed = (y < self.consts.PEEK_DECISION_Y)                     
        is_hole_bonk = hit_connects & is_exposed
        return is_hole_bonk, is_run_bonk
        
    def _update_timers(self, state: GopherState) -> int:
        """
        Simply decrements the main action timer.
        """
        is_waiting = state.gopher_timer > 0
        return jax.lax.select(is_waiting, state.gopher_timer - 1, 0)


    def _determine_next_state(self, state: GopherState, loc: dict, timer: int, rolls: chex.Array, fire_pressed: bool) -> Tuple[chex.Array, int, chex.Array, int]:
        """
        Handles digging l1, walking, climbing, stealing transition.
        """
        curr_act = state.gopher_action
        next_act = curr_act
        new_timer = timer
        new_holes = state.hole_layout
        dir_mod = 1 
        
        
        # Check difficulty(Normal Gopher / Smart Gopher)
        is_smart = state.score >= 1000
        
        # Probabilities
        # (smart gopher has higher probability to steal after peek
        # and less probability to turn around after digging tunnel)
        prob_steal = jax.lax.select(is_smart, self.consts.PROB_STEAL_SMART, self.consts.PROB_STEAL_NORMAL)
        prob_turn = jax.lax.select(is_smart, self.consts.PROB_TURN_AFTER_DIG_SMART, self.consts.PROB_TURN_AFTER_DIG_NORMAL)

        new_target_idx = state.gopher_target_idx
        new_target_idx = jax.lax.select(loc["is_under_hole"], new_target_idx, -1)
        r_l1, r_l2, r_l3, r_turn, r_steal = rolls[0], rolls[1], rolls[2], rolls[3], rolls[4]

        
        # --- Wake up from start delay ---
        finish_start_delay = (curr_act == GopherAction.START_DELAY) & (timer == 0) & fire_pressed
        next_act = jax.lax.select(finish_start_delay, jnp.array(GopherAction.IDLE), next_act)

        # --- Walking ---
        start_walk = (next_act == GopherAction.IDLE) & loc["is_at_floor"]
        next_act = jax.lax.select(start_walk, jnp.array(GopherAction.WALKING), next_act)

        # --- Dig Layer 1 ---
        is_fresh_hole = (state.hole_layout[loc["h_idx"], 0] == 0)
        can_dig_l1 = (curr_act == GopherAction.WALKING) & loc["is_under_hole"] & loc["is_at_floor"] & is_fresh_hole
        should_dig_l1 = can_dig_l1 & (r_l1 < self.consts.PROB_DIG_L1)
        
        new_holes = jax.lax.cond(should_dig_l1, lambda h: h.at[loc["h_idx"], 0].set(1), lambda h: h, operand=new_holes)
        
        # --- Decision: climb or continue walk (Layer 2) ---
        l1_is_open = (state.hole_layout[loc["h_idx"], 0] == 1) | should_dig_l1
        is_physically_ready = (curr_act == GopherAction.WALKING) & loc["is_under_hole"] & loc["is_at_floor"] & l1_is_open & loc["is_aligned_x"]
        is_new_encounter = (loc["h_idx"] != state.gopher_target_idx)
        can_attempt_climb = is_physically_ready & is_new_encounter
        
        start_l2_seq = can_attempt_climb & (r_l2 < self.consts.PROB_DIG_L2)

        # Update memory
        
        new_target_idx = jax.lax.select(can_attempt_climb, loc["h_idx"], new_target_idx)
        next_act = jax.lax.select(start_l2_seq, jnp.array(GopherAction.PREPARE_CLIMB), next_act)
        new_timer = jax.lax.select(start_l2_seq, jnp.array(self.consts.TIME_TO_PREPARE_CLIMB), new_timer)

        # --- Execution for climb(Layer 2)
        finish_prep = (curr_act == GopherAction.PREPARE_CLIMB) & (timer == 0)
        next_act = jax.lax.select(finish_prep, jnp.array(GopherAction.DIGGING_UP), next_act)
        new_timer = jax.lax.select(finish_prep, jnp.array(1), new_timer)
        new_holes = jax.lax.cond(finish_prep, lambda h: h.at[loc["h_idx"], 1].set(1), lambda h: new_holes, operand=new_holes)
        
        # --- Decision: Continue climb up or Recover ---
        check_climb = (curr_act == GopherAction.DIGGING_UP) & (timer == 0) & loc["is_at_floor"]
        wants_to_climb = r_l3 < self.consts.PROB_DIG_L3

        # --- Recover ---
        should_abort = check_climb & jnp.logical_not(wants_to_climb)
        next_act = jax.lax.select(should_abort, jnp.array(GopherAction.RECOVER_LAND), next_act)
        new_timer = jax.lax.select(should_abort, jnp.array(self.consts.TIME_TO_RECOVER_DIG), new_timer)

        # --- Decision: steal or retreat ---
        finish_peek = (curr_act == GopherAction.SEEING_CARROT) & (timer == 0)
        wants_steal = r_steal < prob_steal
        
        # Prepare steal
        do_steal = finish_peek & wants_steal
        next_act = jax.lax.select(do_steal, jnp.array(GopherAction.PREPARE_STEAL), next_act)
        new_timer = jax.lax.select(do_steal, jnp.array(self.consts.TIME_TO_PREPARE_STEAL), new_timer)
        
        # Retreat
        do_retreat = finish_peek & jnp.logical_not(wants_steal)
        next_act = jax.lax.select(do_retreat, jnp.array(GopherAction.DIGGING_DOWN), next_act)

        # --- Execution: Steal
        finish_steal_prep = (curr_act == GopherAction.PREPARE_STEAL) & (timer == 0)
        next_act = jax.lax.select(finish_steal_prep, jnp.array(GopherAction.STEALING), next_act)

        # --- Recover logic ---(7 frames freeze under hole before next action)
        landed = (curr_act == GopherAction.DIGGING_DOWN) & loc["is_at_floor"]
        next_act = jax.lax.select(landed, jnp.array(GopherAction.RECOVER_LAND), next_act)
        new_timer = jax.lax.select(landed, jnp.array(self.consts.TIME_TO_RECOVER_DIG), new_timer)
        
        # Finish Recover -> Walk
        finish_recover = (curr_act == GopherAction.RECOVER_LAND) & (timer == 0)
        next_act = jax.lax.select(finish_recover, jnp.array(GopherAction.WALKING), next_act)
        
        # Random turn after recover
        turn = finish_recover & (r_turn < prob_turn)
        dir_mod = jax.lax.select(turn, -1, 1)

        return next_act, new_timer, new_holes, dir_mod, new_target_idx
    
    def _process_horizontal_movement(self, state: GopherState, next_act: int, new_timer: int, roll_turn: float) -> Tuple[float, int, int, chex.Array, int]:
        """
        Handles walking, running (stealing), digging tunnels, and screen wrapping.
        """

        curr_x = state.gopher_position[0]
        curr_dir = state.gopher_direction_x
        
        # Calculate speed
        is_smart = state.score >= 1000
        base_speed = jax.lax.select(is_smart, self.consts.GOPHER_SPEED_SMART_X, self.consts.GOPHER_SPEED_X)


        # Steal Logic: Run towards closest carrot
        carrot_xs = jnp.array(self.consts.CARROT_X_POSITION)
        d0 = jax.lax.select(state.carrots_present[0] == 1, jnp.abs(curr_x - carrot_xs[0]), 1000.0)
        d1 = jax.lax.select(state.carrots_present[1] == 1, jnp.abs(curr_x - carrot_xs[1]), 1000.0)
        d2 = jax.lax.select(state.carrots_present[2] == 1, jnp.abs(curr_x - carrot_xs[2]), 1000.0)
        target_idx = jnp.argmin(jnp.array([d0, d1, d2]))
        target_x = carrot_xs[target_idx]

        is_stealing = (next_act == GopherAction.STEALING)

        # Calculate stealing speed
        current_steal_speed = jax.lax.select(is_smart, base_speed + 0.5, self.consts.STEAL_SPEED_X)
        speed_x = jax.lax.select(is_stealing, current_steal_speed, base_speed)
        steal_dir = jax.lax.select(target_x > curr_x, 1, -1)
        move_dir = jax.lax.select(is_stealing, steal_dir, curr_dir)
        
        # --- Walking in tunnel ---
        should_move = (next_act == GopherAction.WALKING) | (next_act == GopherAction.DIGGING_TUNNEL) | is_stealing
        dx = jax.lax.select(should_move, speed_x * move_dir, 0.0)
        new_x = (curr_x + dx) % self.consts.WIDTH
        
        # Wraparound
        proposed_x = curr_x + dx
        wrap_threshold_right = float(self.consts.WIDTH - self.consts.GOPHER_SIZE[0]) 
        new_x = jax.lax.select(proposed_x > wrap_threshold_right, 0.0, proposed_x)
        new_x = jax.lax.select(new_x < 0.0, wrap_threshold_right, new_x)

        # --- Dig tunel ---
        center_x = curr_x + 6.0
        current_center_idx = (center_x // 4.0).astype(jnp.int32) % 40
        is_edge_transition = (new_x < 2.0) | (new_x > 144.0)
        
        is_normal_move = (next_act == GopherAction.WALKING) | (next_act == GopherAction.DIGGING_TUNNEL)
        is_starting_climb = (next_act == GopherAction.PREPARE_CLIMB)
        
        should_dig_current = (is_normal_move | is_starting_climb) & \
                             (state.tunnel_layout[current_center_idx] == 0) & \
                             jnp.logical_not(is_edge_transition) 

        new_tunnels = jax.lax.cond(
            should_dig_current,
            lambda t: t.at[current_center_idx].set(1), 
            lambda t: t,
            operand=state.tunnel_layout
        )

        # Turning logic
        prob_turn = jax.lax.select(is_smart, self.consts.PROB_TURN_AFTER_DIG_SMART, self.consts.PROB_TURN_AFTER_DIG_NORMAL)
        is_near_edge = (curr_x < 10.0) | (curr_x > self.consts.WIDTH - 10.0)
        should_turn_tile = should_dig_current & (roll_turn < prob_turn) & jnp.logical_not(is_near_edge)
        
        new_dir = jax.lax.select(should_turn_tile, move_dir * -1, move_dir)
        final_dir = jax.lax.select(is_stealing, steal_dir, new_dir)
        final_act = jax.lax.select(should_move, jnp.array(GopherAction.WALKING), next_act)
        final_act = jax.lax.select(is_stealing, jnp.array(GopherAction.STEALING), final_act)
        return new_x, final_act, new_timer, new_tunnels, final_dir
    
    def _process_vertical_movement(self, state: GopherState, loc: dict, next_act: int, new_timer: int, current_x: float, rolls: chex.Array) -> Tuple[float, float, int, int, chex.Array]:
        """
        Handles movement UP/DOWN and L3 Digging logic.
        """
        curr_y = state.gopher_position[1]
        speed_y = self.consts.GOPHER_SPEED_Y
        final_act = next_act
        final_timer = new_timer
        new_holes = state.hole_layout

        # Unpack the two decision rolls
        r_cont_l3, r_peek = rolls[0], rolls[1]
        is_peeking = (state.gopher_action == GopherAction.SEEING_CARROT)

        # Determine direction
        should_up = (next_act == GopherAction.DIGGING_UP) & (new_timer == 0) & jnp.logical_not(is_peeking)
        should_down = (next_act == GopherAction.DIGGING_DOWN) & jnp.logical_not(is_peeking)
        
        dy = jax.lax.select(should_up, -speed_y, jax.lax.select(should_down, speed_y, 0.0))
        
        new_y = curr_y + dy
        new_y = jnp.clip(new_y, -100.0, self.consts.GOPHER_START_Y)

        # --- Execution: Dig L3  ---
        head_reached_l3_bottom = (new_y <= self.consts.L3_BOTTOM_Y) & (curr_y > self.consts.L3_BOTTOM_Y)
        dig_l3_moment = head_reached_l3_bottom & should_up & (state.hole_layout[loc["h_idx"], 2] == 0)
        new_holes = jax.lax.cond(dig_l3_moment, lambda h: h.at[loc["h_idx"], 2].set(1), lambda h: new_holes, operand=new_holes)

        # --- Decision: Continue or Retreat?
        wants_to_continue = r_cont_l3 < self.consts.PROB_CONTINUE_AFTER_L3
        should_abort_l3 = head_reached_l3_bottom & jnp.logical_not(wants_to_continue)
        
        # Decide: retreat
        final_act = jax.lax.select(should_abort_l3, jnp.array(GopherAction.DIGGING_DOWN), final_act)
        

        # --- Decision: peek or retreat(when reaching l3 top) ---
        passed_ground = (new_y <= self.consts.PEEK_DECISION_Y) & (curr_y > self.consts.PEEK_DECISION_Y)
        wants_to_peek = r_peek < self.consts.PROB_PEEK_AT_SURFACE
        should_retreat_early = passed_ground & should_up & jnp.logical_not(wants_to_peek)
        final_act = jax.lax.select(should_retreat_early, jnp.array(GopherAction.DIGGING_DOWN), final_act)
        new_y = jax.lax.select(should_retreat_early, self.consts.CEILING_Y + 20.0, new_y)
        
        # --- Execution: peek ---
        head_reach_top = (new_y <= self.consts.GOPHER_TOP_Y) & should_up
        should_peek_now = head_reach_top
        final_act = jax.lax.select(should_peek_now, jnp.array(GopherAction.SEEING_CARROT), final_act)
        final_timer = jax.lax.select(should_peek_now, jnp.array(self.consts.TIME_TO_PEEK), final_timer)
        new_y = jax.lax.select(should_peek_now, self.consts.CEILING_Y, new_y)
        
        # --- Landing on tunnel(Recover) ---
        hit_floor = (should_down) & (new_y >= self.consts.GOPHER_START_Y)
        final_act = jax.lax.select(hit_floor, jnp.array(GopherAction.RECOVER_LAND), final_act)
        final_timer = jax.lax.select(hit_floor, jnp.array(self.consts.TIME_TO_RECOVER_DIG), final_timer)

        # Snap X
        target_x = jnp.array(self.consts.HOLE_POSITION_X, dtype=jnp.float32)[loc["h_idx"]] - 1.0
        is_vert = should_up | should_down | (final_act == GopherAction.SEEING_CARROT) | (final_act == GopherAction.PREPARE_CLIMB)
        final_x = jax.lax.select(is_vert, target_x, current_x)
        final_y = jnp.clip(new_y, self.consts.CEILING_Y, self.consts.GOPHER_START_Y)
        return final_x, final_y, final_act, final_timer, new_holes
    
    def _resolve_collisions_and_reset(self, state: GopherState, is_stealing: bool, is_any_bonk: bool, current_x: float, current_y: float) -> GopherState:
        """
        Handles Carrot eating, Game Over, and Resets.
        """
        
        # Carrot Collision
        gx = current_x
        c_pos = jnp.array(self.consts.CARROT_X_POSITION)
        
        caught_0 = is_stealing & state.carrots_present[0] & (jnp.abs(gx - c_pos[0]) < 4.0)
        caught_1 = is_stealing & state.carrots_present[1] & (jnp.abs(gx - c_pos[1]) < 4.0)
        caught_2 = is_stealing & state.carrots_present[2] & (jnp.abs(gx - c_pos[2]) < 4.0)
        any_caught = caught_0 | caught_1 | caught_2
        
        # Update Carrot Data
        new_carrots = state.carrots_present
        new_carrots = jax.lax.select(caught_0, new_carrots.at[0].set(0), new_carrots)
        new_carrots = jax.lax.select(caught_1, new_carrots.at[1].set(0), new_carrots)
        new_carrots = jax.lax.select(caught_2, new_carrots.at[2].set(0), new_carrots)
        
        # Check Game Over
        carrots_left = jnp.sum(new_carrots)
        game_over = carrots_left == 0
        
        # Map tunnel as dug
        clean_tunnels = jnp.zeros(self.consts.NUM_TILES, dtype=jnp.int32)
        clean_tunnels = clean_tunnels.at[0].set(1).at[1].set(1).at[2].set(1).at[3].set(1)
        last = self.consts.NUM_TILES - 1
        clean_tunnels = clean_tunnels.at[last].set(1).at[last-1].set(1).at[last-2].set(1).at[last-3].set(1)
        
        # --- Round Reset (Carrot Lost) ---
        state_round_reset = state._replace(
            player_x=jnp.array(self.consts.PLAYER_START_X), 
            gopher_position=jnp.array([self.consts.GOPHER_START_X, self.consts.GOPHER_START_Y]),
            gopher_direction_x=jnp.array(-1, dtype=jnp.int32),
            gopher_action=jnp.array(GopherAction.START_DELAY),
            gopher_timer=jnp.array(self.consts.TIME_TO_START_DELAY),
            tunnel_layout=clean_tunnels,
            hole_layout=jnp.zeros((6, 3), dtype=jnp.int32),
            carrots_present=new_carrots, 
            score=state.score,
            gopher_target_idx=jnp.array(-1, dtype=jnp.int32),
            duck_active=jnp.array(0, dtype=jnp.int32),
            seed_active=jnp.array(0, dtype=jnp.int32),
            duck_cool_down_timer=jnp.array(self.consts.TIME_DUCK_COOLDOWN, dtype=jnp.int32)
            
        )

        # --- Hard Reset State (Reset All, Game over) ---
        state_hard_reset = state._replace(
            player_x=jnp.array(self.consts.PLAYER_START_X), 
            gopher_position=jnp.array([self.consts.GOPHER_START_X, self.consts.GOPHER_START_Y]),
            gopher_direction_x=jnp.array(-1, dtype=jnp.int32),
            gopher_action=jnp.array(GopherAction.START_DELAY),
            gopher_timer=jnp.array(self.consts.TIME_TO_START_DELAY),
            tunnel_layout=clean_tunnels,
            hole_layout=jnp.zeros((6, 3), dtype=jnp.int32),
            carrots_present=jnp.array([1, 1, 1]), 
            score=jnp.array(0),                   
            gopher_target_idx=jnp.array(-1, dtype=jnp.int32),
        )
        
        # --- Bonk Reset (Gopher bonked)---
        state_bonk_reset = state._replace(
            gopher_position=jnp.array([self.consts.GOPHER_START_X, self.consts.GOPHER_START_Y]),
            gopher_direction_x=jnp.array(-1, dtype=jnp.int32),
            gopher_action=jnp.array(GopherAction.IDLE),
            gopher_timer=jnp.array(0),
            gopher_target_idx=jnp.array(-1, dtype=jnp.int32),
            carrots_present=new_carrots, 
        )

        state_running = state._replace(carrots_present=new_carrots)

        # Final selection
        state_if_no_steal = jax.lax.cond(
            is_any_bonk,
            lambda _: state_bonk_reset._replace(score=state.score + 100),
            lambda _: state_running,      
            operand=None
        )
        state_if_steal = jax.lax.cond(
            game_over, 
            lambda _: state_hard_reset, 
            lambda _: state_round_reset, 
            operand=None
        )

        final_state = jax.lax.cond(
            any_caught,
            lambda _: state_if_steal,
            lambda _: state_if_no_steal,
            operand=None
        )

        return final_state
    
    def _gopher_step(self, state: GopherState, fire_pressed: bool) -> GopherState:
        '''
        Handle main Gopher move
        '''
        loc = self._locate_gopher(state)
        key, k1, k2, k3 = jax.random.split(state.key, 4)
        timer_step1 = self._update_timers(state)
        rolls = jax.random.uniform(k1, shape=(5,))
        
        # State decision
        act_step1, timer_step2, holes_step1, dir_mod, new_target_idx = self._determine_next_state(
            state, loc, timer_step1, rolls, fire_pressed
        )
        state = state._replace(hole_layout=holes_step1)

        # Bonk check
        temp_state_check = state._replace(gopher_action=act_step1)
        is_hole_bonk, is_run_bonk = self._check_bonk_hit(temp_state_check)
        is_any_bonk = is_hole_bonk | is_run_bonk
        
        # Horizontal physics
        state_pre_phys = state._replace(
            hole_layout=holes_step1, 
            gopher_action=act_step1
        )
        new_x, act_step2, timer_step3, new_tunnels, new_dir_raw = self._process_horizontal_movement(
            state_pre_phys, act_step1, timer_step2, jax.random.uniform(k2)
        )
        final_dir = new_dir_raw * dir_mod

        # Vertical 
        rolls_vert = jax.random.uniform(k3, shape=(2,))
        final_x, final_y, final_act, final_timer, new_holes = self._process_vertical_movement(
            state, loc, act_step2, timer_step3, new_x, rolls_vert
        )

        is_stealing = (final_act == GopherAction.STEALING)
        final_y = jax.lax.select(is_stealing, self.consts.STEAL_Y_POS, final_y)
        
        state_post_physics = state._replace(
            gopher_position=jnp.array([final_x, final_y]),
            gopher_direction_x=final_dir,
            gopher_action=final_act,
            gopher_timer=final_timer,
            tunnel_layout=new_tunnels,
            hole_layout=new_holes,
            gopher_move_x_timer=state.gopher_move_x_timer + 1,
            gopher_target_idx=new_target_idx,
            key=key
        )

        # Check game over condition and reset
        final_state = self._resolve_collisions_and_reset(
            state_post_physics, is_stealing, is_run_bonk, final_x, final_y
        )

        return final_state

    def _process_duck_and_seed(self, state: GopherState, key: chex.Array) -> GopherState:
        """
        Handles bonus duck spawning and seed mechanics.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Update timer
        current_duck_cool_down_timer = jax.lax.select(state.duck_cool_down_timer > 0, state.duck_cool_down_timer - 1, 0)
        new_anim_timer = jax.lax.select(state.duck_active == 1, state.duck_anim_timer + 1, 0)

        # Condition for duck to show up
        num_carrots_left = jnp.sum(state.carrots_present)
        carrot_missing = num_carrots_left < self.consts.NUM_CARROTS
        is_not_frozen = jnp.logical_not(state.gopher_action == GopherAction.START_DELAY)
        can_spawn = (state.score >= self.consts.SCORE_FOR_DUCK) & \
                    (state.duck_active == 0) & \
                    (state.seed_active == 0) & \
                    (current_duck_cool_down_timer == 0) & \
                    (state.player_has_seed == 0) & \
                    carrot_missing & \
                    is_not_frozen
        should_spawn = can_spawn & (jax.random.uniform(k1) < self.consts.PROB_DUCK_SPAWN)
        
        # Randomize Spawn Direction (Left->Right or Right->Left)
        spawn_dir = jax.lax.select(jax.random.bernoulli(k2), 1, -1)
        max_x = float(self.consts.WIDTH - self.consts.DUCK_SIZE[0])
        spawn_x = jax.lax.select(spawn_dir == 1, 0.0, max_x)
        
        # Randomize Drop Point (between 20 and 140)
        drop_target_x = jax.random.uniform(k3, minval=self.consts.DUCK_SPAWN_MIN_X, maxval=self.consts.DUCK_SPAWN_MAX_X)
        
        # Update State if spawning
        duck_active = jax.lax.select(should_spawn, 1, state.duck_active)
        duck_x = jax.lax.select(should_spawn, spawn_x, state.duck_x)
        duck_dir = jax.lax.select(should_spawn, spawn_dir, state.duck_dir)
        duck_drop_x = jax.lax.select(should_spawn, drop_target_x, state.duck_drop_x)
        
        # --- Duck movement ---
        duck_x = duck_x + (self.consts.DUCK_SPEED * duck_dir)
        hit_right = (duck_dir == 1) & (duck_x > max_x)
        hit_left = (duck_dir == -1) & (duck_x < 0.0)
        should_despawn = (duck_active == 1) & (hit_right | hit_left)
        
        # Reset if despawn
        new_duck_cool_down_timer = jax.lax.select(should_despawn, self.consts.TIME_DUCK_COOLDOWN, current_duck_cool_down_timer)
        duck_active = jax.lax.select(should_despawn, 0, duck_active)
        
        # --- Seed drop ---
        passed_drop = jnp.abs(duck_x - duck_drop_x) < 2.0
        trigger_drop = (state.duck_active == 1) & (state.seed_active == 0) & passed_drop
        seed_active = jax.lax.select(trigger_drop, 1, state.seed_active)
        seed_x = jax.lax.select(trigger_drop, duck_x, state.seed_x)
        seed_y = jax.lax.select(trigger_drop, float(self.consts.DUCK_Y_POS), state.seed_y)
        seed_y = seed_y + self.consts.SEED_DROP_SPEED
        
        # --- Catch seed ---
        p_x = state.player_x
        p_y = self.consts.PLAYER_START_Y
        caught = (seed_active == 1) & \
                 (seed_x >= p_x) & (seed_x <= p_x + 13.0) & \
                 (seed_y >= self.consts.PLAYER_START_Y) & (seed_y <= self.consts.WATER_Y_POS)
                 
        
        hit_water = seed_y >= self.consts.WATER_Y_POS               # when seed drops on ground, it disappears
        
        # Update Seed State
        seed_active = jax.lax.select(caught | hit_water, 0, seed_active)
        
        # Update Player Inventory
        player_has_seed = jax.lax.select(caught, 1, state.player_has_seed)
        
        return state._replace(
            duck_active=duck_active,
            duck_x=duck_x,
            duck_dir=duck_dir,
            duck_drop_x=duck_drop_x,
            seed_active=seed_active,
            seed_x=seed_x,
            seed_y=seed_y,
            player_has_seed=player_has_seed,
            duck_cool_down_timer=new_duck_cool_down_timer,
            duck_anim_timer=new_anim_timer
        )

    def _handle_planting(self, state: GopherState) -> GopherState:
        """
        Allows player to plant seed if holding one and hitting an empty slot.
        """
        is_planting_frame = (state.bonk_timer == 1) & (state.player_has_seed == 1)
        
        # Locate Carrot Slots
        p_center = state.player_x + 6.5
        c_pos = jnp.array(self.consts.CARROT_X_POSITION)
        
        # Check distance
        dist = jnp.abs(p_center - c_pos)
        near_0 = (dist[0] < 10.0) & (state.carrots_present[0] == 0)
        near_1 = (dist[1] < 10.0) & (state.carrots_present[1] == 0)
        near_2 = (dist[2] < 10.0) & (state.carrots_present[2] == 0)
        
        can_plant = is_planting_frame & (near_0 | near_1 | near_2)
        
        # Update Carrots
        new_carrots = state.carrots_present
        new_carrots = jax.lax.select(can_plant & near_0, new_carrots.at[0].set(1), new_carrots)
        new_carrots = jax.lax.select(can_plant & near_1, new_carrots.at[1].set(1), new_carrots)
        new_carrots = jax.lax.select(can_plant & near_2, new_carrots.at[2].set(1), new_carrots)
        
        # Remove Seed if planted
        new_has_seed = jax.lax.select(can_plant, 0, state.player_has_seed)
        
        return state._replace(
            carrots_present=new_carrots,
            player_has_seed=new_has_seed
        )
    
    def reset(self, key: chex.PRNGKey) -> tuple[chex.Array, GopherState]:
        key, subkey = jax.random.split(key)       
        tunnel_layout = jnp.zeros(self.consts.NUM_TILES, dtype=jnp.int32)
        tunnel_layout = tunnel_layout.at[0].set(1).at[1].set(1).at[2].set(1).at[3].set(1)
        last = self.consts.NUM_TILES - 1
        tunnel_layout = tunnel_layout.at[last].set(1).at[last-1].set(1).at[last-2].set(1).at[last-3].set(1)

        state = GopherState(
            # --- Player ---
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.float32), 
            player_speed=jnp.array(0.0),
            player_has_seed=jnp.array(0, dtype=jnp.int32),
            bonk_timer=jnp.array(0, dtype=jnp.int32),
            
            # --- Gopher ---
            gopher_position=jnp.array([self.consts.GOPHER_START_X, self.consts.GOPHER_START_Y], dtype=jnp.float32),
            gopher_direction_x=jnp.array(-1, dtype=jnp.int32),
            gopher_move_x_timer=jnp.array(0, dtype=jnp.int32),
            gopher_action=jnp.array(GopherAction.START_DELAY, dtype=jnp.int32),
            gopher_timer=jnp.array(self.consts.TIME_TO_START_DELAY, dtype=jnp.int32),
            gopher_target_idx=jnp.array(0, dtype=jnp.int32),
            gopher_target_layer=jnp.array(0, dtype=jnp.int32),
            
            # --- Environment ---
            carrots_present=jnp.array([1, 1, 1]),
            tunnel_layout=tunnel_layout,
            hole_layout=jnp.zeros((6, 3), dtype=jnp.int32),
            
            # --- Duck / Seed ---
            duck_active=jnp.array(0, dtype=jnp.int32),
            duck_x=jnp.array(0.0, dtype=jnp.float32),
            duck_dir=jnp.array(1, dtype=jnp.int32),
            duck_drop_x=jnp.array(0.0, dtype=jnp.float32),
            duck_cool_down_timer=jnp.array(0, dtype=jnp.int32),
            duck_anim_timer=jnp.array(0, dtype=jnp.int32),
            
            seed_active=jnp.array(0, dtype=jnp.int32),
            seed_x=jnp.array(0.0, dtype=jnp.float32),
            seed_y=jnp.array(0.0, dtype=jnp.float32),
            
            # --- Defaults ---
            score=jnp.array(0, dtype=jnp.int32),
            key=key,
            frame_count=jnp.array(0)
        )
        obs = self.render(state)
        return obs, state

    def step(self, state: GopherState, action: chex.Array) -> tuple[chex.Array, GopherState, jnp.ndarray, jnp.ndarray, dict]:
        key, k_duck = jax.random.split(state.key)
        state = state._replace(key=key) 

        state, fire_pressed = self._player_step(state, action)
        state = self._handle_repairs(state)
        state = self._handle_planting(state)           
        state = self._process_duck_and_seed(state, k_duck) 

        state = self._gopher_step(state, fire_pressed)
        state = state._replace(frame_count=state.frame_count + 1)
        obs = self.render(state)
        reward = jnp.array(0.0)
        done = jnp.array(False)
        info = self._get_info(state)
        return obs, state, reward, done, info
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GopherState) -> jnp.ndarray:
        return self.renderer.render(state)
    
    def action_space(self): 
        return spaces.Discrete(len(self.action_set))
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GopherState, ) -> GopherInfo:
        return GopherInfo(time=state.frame_count, difficulty_level=jnp.array(1, dtype=jnp.int32))

# ==========================================================================================
#   RENDERER
# ==========================================================================================

class GopherRenderer(JAXGameRenderer):
    def __init__(self, consts: GopherConstants = None):
        super().__init__(consts)
        self.consts = consts or GopherConstants()
        self.config = render_utils.RendererConfig(game_dimensions=(210, 160), channels=3)
        self.jr = render_utils.JaxRenderingUtils(self.config)
        final_asset_config = list(self.consts.ASSET_CONFIG)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/gopher"
        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND, self.COLOR_TO_ID, self.FLIP_OFFSETS) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GopherState):
        # --- Background ---
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # --- Duck ---
        anim_step = (state.duck_anim_timer // 8) % 4
        pose_offset = jax.lax.select(anim_step == 0, 0, 
                      jax.lax.select(anim_step == 1, 1,
                      jax.lax.select(anim_step == 2, 0, 2)))
        base_idx = jax.lax.select(state.duck_dir == 1, 3, 0)
        final_duck_idx = base_idx + pose_offset
        duck_mask = self.SHAPE_MASKS["duck"][final_duck_idx]
        
        is_safe_x = (state.duck_x > -1.0) & (state.duck_x < 146.0)
        should_draw_duck = (state.duck_active == 1) & is_safe_x
        raster = jax.lax.cond(
            should_draw_duck,
            lambda r: self.jr.render_at(r, jnp.int32(state.duck_x), jnp.int32(self.consts.DUCK_Y_POS), duck_mask),
            lambda r: r,
            operand=raster
        )

        # --- Seed ---
        seed_mask = self.SHAPE_MASKS["seed"]
        raster = jax.lax.cond(
            state.seed_active == 1,
            lambda r: self.jr.render_at(r, jnp.int32(state.seed_x), jnp.int32(state.seed_y), seed_mask),
            lambda r: r,
            operand=raster
        )

        # --- Carrot ---
        carrot_x = jnp.array(self.consts.CARROT_X_POSITION)
        carrot_y = self.consts.CARROT_Y_POS
        carrot_mask = self.SHAPE_MASKS["carrot"]
        def draw_one_carrot(i, current_raster):
            is_active = state.carrots_present[i] == 1
            x = carrot_x[i]
            y = carrot_y
            return jax.lax.cond(is_active, lambda r: self.jr.render_at(r, jnp.int32(x), jnp.int32(y), carrot_mask), lambda r: r, operand=current_raster)
        raster = jax.lax.fori_loop(0, 3, draw_one_carrot, raster)
        
        # --- Player ---
        base_idx = jax.lax.select(state.bonk_timer < 2, 0, jax.lax.select(state.bonk_timer < 4, 1, jax.lax.select(state.bonk_timer < 8, 2, 0)))
        final_player_idx = jax.lax.select(state.player_has_seed == 1, 
                                          base_idx + self.consts.SPRITE_OFFSET_PLAYER_SEED, 
                                          base_idx)
        
        player_mask = self.SHAPE_MASKS["player"][final_player_idx]
        raster = self.jr.render_at(raster, jnp.int32(state.player_x), jnp.int32(self.consts.PLAYER_START_Y), player_mask)

        # --- Tunnel dug ---
        tunnel_mask = self.SHAPE_MASKS["tunnel_tile_dug"]
        def draw_tunnel(i, r):
            x = self.consts.LEFT_WALL + (i * self.consts.TILE_WIDTH)
            y = self.consts.TUNNEL_TOP_Y 
            return jax.lax.cond(state.tunnel_layout[i] == 1, lambda _r: self.jr.render_at(_r, jnp.int32(x), jnp.int32(y), tunnel_mask), lambda _r: _r, operand=r)
        raster = jax.lax.fori_loop(0, self.consts.NUM_TILES, draw_tunnel, raster)

        # --- Hole dug ---
        hole_mask = self.SHAPE_MASKS["hole_tile_dug"][0]
        hole_x_coords = jnp.array(self.consts.HOLE_POSITION_X)     
        hole_y_coords = jnp.array(self.consts.HOLE_POSITION_Y) 
        def draw_hole_column(i, r):
            x = hole_x_coords[i]
            def draw_segment(depth_idx, current_r):
                is_dug = state.hole_layout[i, depth_idx] == 1
                segment_y = hole_y_coords[depth_idx]
                return jax.lax.cond(is_dug, lambda _r: self.jr.render_at(_r, jnp.int32(x), jnp.int32(segment_y), hole_mask), lambda _r: _r, operand=current_r)
            return jax.lax.fori_loop(0, 3, draw_segment, r)
        raster = jax.lax.fori_loop(0, 6, draw_hole_column, raster)
        
        # --- Gopher ---
        is_start_delay = state.gopher_action == GopherAction.START_DELAY
        anim_tick = (state.gopher_move_x_timer // 4) % 2
        gopher_leg_idx = jax.lax.select(is_start_delay, 0, anim_tick)
        
        dir_offset = jax.lax.select(state.gopher_direction_x == 1, 2, 0)
        walk_idx = dir_offset + gopher_leg_idx
        stand_idx = 4 
        
        is_left_side = state.gopher_position[0] < (self.consts.WIDTH / 2.0)
        seeing_base = jax.lax.select(is_left_side, 7, 5)
        seeing_idx = seeing_base + ((state.gopher_timer // 8) % 2)
        
        is_digging_up = state.gopher_action == GopherAction.DIGGING_UP
        is_digging_down = state.gopher_action == GopherAction.DIGGING_DOWN
        is_seeing = state.gopher_action == GopherAction.SEEING_CARROT
        
        # Default: Walk/Run Pose
        final_gopher_idx = walk_idx 
        # Stand up pose
        final_gopher_idx = jax.lax.select(is_digging_up | is_digging_down, stand_idx, final_gopher_idx)
        # Peek pose
        final_gopher_idx = jax.lax.select(is_seeing, seeing_idx, final_gopher_idx)
        
        # Height adjustment(difference in pose)
        draw_y = state.gopher_position[1]
        stand_offset = self.consts.STAND_OFFSET 
        draw_y = jax.lax.select(is_digging_up | is_digging_down, draw_y - stand_offset, draw_y)
        
        is_near_bottom = state.gopher_position[1] > 170.0
        push_down_offset = self.consts.PUSH_DOWN_OFFSET
        
        draw_y = jax.lax.select(
            is_digging_down & is_near_bottom, 
            draw_y + push_down_offset,
            draw_y
        )
        
        # Ceiling clamp for peek
        ceiling_y = self.consts.CEILING_Y
        draw_y = jax.lax.select(is_digging_up | is_digging_down, jnp.maximum(draw_y, ceiling_y), draw_y)
        draw_y = jax.lax.select(is_seeing, ceiling_y, draw_y)

        # Prepare steal: stand up pose
        is_prep_steal = state.gopher_action == GopherAction.PREPARE_STEAL
        final_gopher_idx = jax.lax.select(is_prep_steal, stand_idx, final_gopher_idx)
        draw_y = jax.lax.select(is_prep_steal, draw_y, draw_y)          
        raster = self.jr.render_at(raster, jnp.int32(state.gopher_position[0]), jnp.int32(draw_y), self.SHAPE_MASKS["gopher"][final_gopher_idx])
        
        
        # --- Gound bottom ---
        raster = self.jr.render_at(raster, jnp.int32(self.consts.LEFT_WALL), jnp.int32(self.consts.TUNNEL_BOTTOM_Y), self.SHAPE_MASKS["bottom_ground"])
        

       # --- Score ---
        score = state.score
        num_digits = 1
        num_digits = jax.lax.select(score > 9, 2, num_digits)
        num_digits = jax.lax.select(score > 99, 3, num_digits)
        num_digits = jax.lax.select(score > 999, 4, num_digits)
        
        x_coords = jnp.array(self.consts.SCORE_X_POSITION)
        def draw_digit(k, r):
            power_of_10 = jnp.power(10, k)
            digit_val = (score // power_of_10) % 10
            x_index = 3 - k
            x_pos = x_coords[x_index]
            
            digit_mask = self.SHAPE_MASKS["numbers"][digit_val]
            return self.jr.render_at(r, jnp.int32(x_pos), jnp.int32(self.consts.SCORE_Y_POS), digit_mask)

        raster = jax.lax.fori_loop(0, num_digits, draw_digit, raster)
        return self.jr.render_from_palette(raster, self.PALETTE)
