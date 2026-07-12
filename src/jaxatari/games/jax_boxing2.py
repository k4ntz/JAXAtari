"""
Boxing - JAXAtari Implementation (v2)
A high-fidelity, GPU-accelerated implementation of Atari 2600 Boxing.
"""

import os
from dataclasses import replace
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


# =============================================================================
# Difficulty Presets (Exposed Global Configuration)
# =============================================================================

DEFAULT_DIFFICULTY = "normal"

DIFFICULTY_PRESETS = {
    "easy": {
        "CPU_UPDATE_MASK": 7,         # Updates target every ~8 frames (slower reactions)
        "CPU_AGGR_WINNING": 20,       # Low punch rate when winning
        "CPU_AGGR_LOSING": 10,        # Very low punch rate when losing
        "CPU_DANCING_DURATION": 60,   # Retreats for a long time when hit
        "PLAYER_FACE_SHRINK_Y": -1.0,
    },
    "normal": {
        "CPU_UPDATE_MASK": 3,         # Updates target every ~4 frames
        "CPU_AGGR_WINNING": 55,       # Slightly higher aggressiveness
        "CPU_AGGR_LOSING": 35,        # Slightly higher aggressiveness
        "CPU_DANCING_DURATION": 30,   # Slightly shorter retreat duration
        "PLAYER_FACE_SHRINK_Y": 0.0,
    },
    "hard": {
        "CPU_UPDATE_MASK": 1,         # Updates target every ~2 frames (extremely fast)
        "CPU_AGGR_WINNING": 90,       # Very high pressure, constant punches
        "CPU_AGGR_LOSING": 70,        # Very high pressure
        "CPU_DANCING_DURATION": 10,   # Recovers and fights back almost instantly
        "PLAYER_FACE_SHRINK_Y": 0.0,
    },
    "impossible": {
        "CPU_UPDATE_MASK": 0,         # Updates target every frame (instantaneous reactions)
        "CPU_AGGR_WINNING": 120,      # Maximum pressure
        "CPU_AGGR_LOSING": 100,       # Aggressive pushback
        "CPU_DANCING_DURATION": 0,    # Strictly never retreats; fights back instantly
        "PLAYER_FACE_SHRINK_Y": 1.0,
    }
}


# =============================================================================
# Asset Config 
# =============================================================================

def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'white_main_body', 'type': 'single', 'file': 'body_white/main_body.npy'},
        {'name': 'white_top_arm_idle', 'type': 'single', 'file': 'body_white/top_arm_idle.npy'},
        {'name': 'white_top_arm_retracted', 'type': 'single', 'file': 'body_white/top_arm_retracted.npy'},
        {'name': 'white_top_arm_stretched', 'type': 'single', 'file': 'body_white/top_arm_stretched.npy'},
        {'name': 'white_top_arm_extended', 'type': 'single', 'file': 'body_white/top_arm_extended.npy'},
        {'name': 'white_bottom_arm_idle', 'type': 'single', 'file': 'body_white/bottom_arm_idle.npy'},
        {'name': 'white_bottom_arm_retracted', 'type': 'single', 'file': 'body_white/bottom_arm_retracted.npy'},
        {'name': 'white_bottom_arm_stretched', 'type': 'single', 'file': 'body_white/bottom_arm_stretched.npy'},
        {'name': 'white_bottom_arm_extended', 'type': 'single', 'file': 'body_white/bottom_arm_extended.npy'},
        {'name': 'black_main_body', 'type': 'single', 'file': 'body_black/main_body.npy'},
        {'name': 'black_top_arm_idle', 'type': 'single', 'file': 'body_black/top_arm_idle.npy'},
        {'name': 'black_top_arm_retracted', 'type': 'single', 'file': 'body_black/top_arm_retracted.npy'},
        {'name': 'black_top_arm_stretched', 'type': 'single', 'file': 'body_black/top_arm_stretched.npy'},
        {'name': 'black_top_arm_extended', 'type': 'single', 'file': 'body_black/top_arm_extended.npy'},
        {'name': 'black_bottom_arm_idle', 'type': 'single', 'file': 'body_black/bottom_arm_idle.npy'},
        {'name': 'black_bottom_arm_retracted', 'type': 'single', 'file': 'body_black/bottom_arm_retracted.npy'},
        {'name': 'black_bottom_arm_stretched', 'type': 'single', 'file': 'body_black/bottom_arm_stretched.npy'},
        {'name': 'black_bottom_arm_extended', 'type': 'single', 'file': 'body_black/bottom_arm_extended.npy'},
        {'name': 'white_idle', 'type': 'single', 'file': 'white_idle.npy'},
        {'name': 'black_idle', 'type': 'single', 'file': 'black_idle.npy'},
        {'name': 'white_stunned', 'type': 'single', 'file': 'white_stunned.npy'},
        {'name': 'black_stunned', 'type': 'single', 'file': 'black_stunned.npy'},
        # Animation frames
        {'name': 'white_punch_left_0', 'type': 'single', 'file': 'white_boxing_animation_left/0.npy'},
        {'name': 'white_punch_left_1', 'type': 'single', 'file': 'white_boxing_animation_left/1.npy'},
        {'name': 'white_punch_left_2', 'type': 'single', 'file': 'white_boxing_animation_left/2.npy'},
        {'name': 'white_punch_left_3', 'type': 'single', 'file': 'white_boxing_animation_left/3.npy'},
        {'name': 'white_punch_right_0', 'type': 'single', 'file': 'white_boxing_animation_right/0.npy'},
        {'name': 'white_punch_right_1', 'type': 'single', 'file': 'white_boxing_animation_right/1.npy'},
        {'name': 'white_punch_right_2', 'type': 'single', 'file': 'white_boxing_animation_right/2.npy'},
        {'name': 'white_punch_right_3', 'type': 'single', 'file': 'white_boxing_animation_right/3.npy'},
        {'name': 'black_punch_left_0', 'type': 'single', 'file': 'black_boxing_animation_left/0.npy'},
        {'name': 'black_punch_left_1', 'type': 'single', 'file': 'black_boxing_animation_left/1.npy'},
        {'name': 'black_punch_left_2', 'type': 'single', 'file': 'black_boxing_animation_left/2.npy'},
        {'name': 'black_punch_left_3', 'type': 'single', 'file': 'black_boxing_animation_left/3.npy'},
        {'name': 'black_punch_right_0', 'type': 'single', 'file': 'black_boxing_animation_right/0.npy'},
        {'name': 'black_punch_right_1', 'type': 'single', 'file': 'black_boxing_animation_right/1.npy'},
        {'name': 'black_punch_right_2', 'type': 'single', 'file': 'black_boxing_animation_right/2.npy'},
        {'name': 'black_punch_right_3', 'type': 'single', 'file': 'black_boxing_animation_right/3.npy'},
        # Arm sprites
        {'name': 'white_arm_left_0', 'type': 'single', 'file': 'arms/white_boxing_animation_left/0.npy'},
        {'name': 'white_arm_left_1', 'type': 'single', 'file': 'arms/white_boxing_animation_left/1.npy'},
        {'name': 'white_arm_left_2', 'type': 'single', 'file': 'arms/white_boxing_animation_left/2.npy'},
        {'name': 'white_arm_left_3', 'type': 'single', 'file': 'arms/white_boxing_animation_left/3.npy'},
        {'name': 'white_arm_right_0', 'type': 'single', 'file': 'arms/white_boxing_animation_right/0.npy'},
        {'name': 'white_arm_right_1', 'type': 'single', 'file': 'arms/white_boxing_animation_right/1.npy'},
        {'name': 'white_arm_right_2', 'type': 'single', 'file': 'arms/white_boxing_animation_right/2.npy'},
        {'name': 'white_arm_right_3', 'type': 'single', 'file': 'arms/white_boxing_animation_right/3.npy'},
        {'name': 'black_arm_left_0', 'type': 'single', 'file': 'arms/black_boxing_animation_left/0.npy'},
        {'name': 'black_arm_left_1', 'type': 'single', 'file': 'arms/black_boxing_animation_left/1.npy'},
        {'name': 'black_arm_left_2', 'type': 'single', 'file': 'arms/black_boxing_animation_left/2.npy'},
        {'name': 'black_arm_left_3', 'type': 'single', 'file': 'arms/black_boxing_animation_left/3.npy'},
        {'name': 'black_arm_right_0', 'type': 'single', 'file': 'arms/black_boxing_animation_right/0.npy'},
        {'name': 'black_arm_right_1', 'type': 'single', 'file': 'arms/black_boxing_animation_right/1.npy'},
        {'name': 'black_arm_right_2', 'type': 'single', 'file': 'arms/black_boxing_animation_right/2.npy'},
        {'name': 'black_arm_right_3', 'type': 'single', 'file': 'arms/black_boxing_animation_right/3.npy'},
        {'name': 'digits_white', 'type': 'digits', 'pattern': 'digits_white/{}.npy'},
        {'name': 'digits_black', 'type': 'digits', 'pattern': 'digits_black/{}.npy'},
        {'name': 'digits_time', 'type': 'digits', 'pattern': 'digits_time/{}.npy'},
    )


# =============================================================================
# Constants 
# =============================================================================

class BoxingConstants(struct.PyTreeNode):
    WIDTH: int = 160
    HEIGHT: int = 210
    
    # Ring boundaries
    XMIN: int = 32
    XMAX: int = 113
    YMIN: int = 34
    YMAX: int = 131
    
    # Boxer dimensions
    W_BOXER: int = 14
    H_BOXER: int = 47
    FACE_MIN_Y: int = 14
    FACE_MAX_Y: int = 32
    TOP_ARM_Y: int = 5
    BOT_ARM_Y: int = 39
    
    # Movement
    PLAYER_SPEED: int = 1
    ENEMY_SPEED: int = 1
    KNOCKBACK_DIST: int = 3
    STUN_DURATION: int = 12
    
    # Hit projection (knockback) animation constants
    HIT_ANIMATION_STEPS: int = 15
    KNOCKBACK_TOP_ARM_DY: int = -2  # Vertical move per step if hit by opponent top arm (up)
    KNOCKBACK_BOT_ARM_DY: int = 2   # Vertical move per step if hit by opponent bottom arm (down)
    KNOCKBACK_DX: int = 1           # Horizontal move (backward) per step
    
    # Punch Mechanics
    PUNCH_STATE_MAX: int = 4
    PUNCH_COOLDOWN: int = 8   # Delay between punches
    JAB_DIST: int = 27    # Distance for 1pt hit
    POWER_DIST: int = 16  # Distance for 2pt hit
    
    # Game rules
    MAX_SCORE: int = 100
    TOTAL_TIME: int = 7200 # 2 minutes at 60Hz
    
    # Starting positions
    P1_START_X: int = 95
    P2_START_X: int = 50
    START_Y: int = 82

    ASSET_CONFIG: tuple = _get_default_asset_config()

    # CPU Difficulty Parameters
    DIFFICULTY_PRESET: str = "normal"
    CPU_UPDATE_MASK: int = 3
    CPU_AGGR_WINNING: int = 40
    CPU_AGGR_LOSING: int = 20
    CPU_DANCING_DURATION: int = 40
    PLAYER_FACE_SHRINK_Y: int = 0
    ENEMY_PEACEFUL: bool = False
    SHOW_COLLISION_ZONE: bool = False


# =============================================================================
# State 
# =============================================================================

@struct.dataclass
class BoxingState:
    pos: chex.Array          # [2, 2] (P1/P2, X/Y)
    orientation: chex.Array  # [2] (0: Right, 1: Left)
    score: chex.Array        # [2]
    punch_state: chex.Array  # [2] (0-4)
    punch_arm: chex.Array    # [2] (0: left, 1: right)
    punch_cooldown: chex.Array # [2] (frames until next punch allowed)
    has_hit: chex.Array      # [2] (bool)
    stun_timer: chex.Array   # [2]
    timer: chex.Array        # int32
    done: chex.Array         # bool
    key: chex.PRNGKey
    cpu_target_x: chex.Array       # Target X position CPU is tracking
    cpu_target_y: chex.Array       # Target Y position CPU is tracking
    cpu_horiz_offset: chex.Array   # Random horizontal offset (0-31)
    cpu_vert_offset: chex.Array    # Random vertical offset (0-63)
    cpu_dancing_value: chex.Array  # Timer controlling CPU "dancing" behavior
    # Hit animation state
    hit_anim_timer: chex.Array     # [2] (int32) remaining steps
    hit_anim_dx: chex.Array        # [2] (int32) horizontal movement per step
    hit_anim_dy: chex.Array        # [2] (int32) vertical movement per step


@struct.dataclass
class BoxingObservation:
    """
    Observable game state for Boxing.
    """
    left_boxer: ObjectObservation
    right_boxer: ObjectObservation
    score_left: chex.Array
    score_right: chex.Array
    clock_minutes: chex.Array
    clock_seconds: chex.Array


@struct.dataclass
class BoxingInfo:
    time: chex.Array
    clock_minutes: chex.Array
    clock_seconds: chex.Array


# =============================================================================
# Environment
# =============================================================================

class JaxBoxing2(JaxEnvironment[BoxingState, BoxingObservation, BoxingInfo, BoxingConstants]):
    def __init__(self, consts: BoxingConstants | None = None, difficulty: str | None = None):
        if consts is None:
            consts = BoxingConstants()
            if difficulty is None:
                difficulty = DEFAULT_DIFFICULTY
        if difficulty is not None and difficulty != "custom":
            consts = self._apply_difficulty_preset(consts, difficulty)
        super().__init__(consts)
        self.renderer = BoxingRenderer(self.consts)
        self.action_set = [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
            Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
            Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
            Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
        ]

    def _apply_difficulty_preset(self, consts: BoxingConstants, difficulty: str) -> BoxingConstants:
        if difficulty not in DIFFICULTY_PRESETS:
            return consts
            
        params = DIFFICULTY_PRESETS[difficulty]
        return replace(
            consts,
            DIFFICULTY_PRESET=difficulty,
            CPU_UPDATE_MASK=params["CPU_UPDATE_MASK"],
            CPU_AGGR_WINNING=params["CPU_AGGR_WINNING"],
            CPU_AGGR_LOSING=params["CPU_AGGR_LOSING"],
            CPU_DANCING_DURATION=params["CPU_DANCING_DURATION"],
            PLAYER_FACE_SHRINK_Y=params.get("PLAYER_FACE_SHRINK_Y", 0.0),
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[BoxingObservation, BoxingState]:
        key, subkey = jax.random.split(key)
        pos = jnp.array([[self.consts.P1_START_X, self.consts.START_Y],
                         [self.consts.P2_START_X, self.consts.START_Y]], dtype=jnp.int32)
        orientation = jnp.array([
            (pos[0, 0] > pos[1, 0]).astype(jnp.int32),
            (pos[1, 0] > pos[0, 0]).astype(jnp.int32)
        ])
        state = BoxingState(
            pos=pos,
            orientation=orientation,
            score=jnp.array([0, 0], dtype=jnp.int32),
            punch_state=jnp.array([0, 0], dtype=jnp.int32),
            punch_arm=jnp.array([0, 0], dtype=jnp.int32),
            punch_cooldown=jnp.array([0, 0], dtype=jnp.int32),
            has_hit=jnp.array([False, False], dtype=jnp.bool_),
            stun_timer=jnp.array([0, 0], dtype=jnp.int32),
            timer=jnp.array(self.consts.TOTAL_TIME, dtype=jnp.int32),
            done=jnp.array(False),
            key=subkey,
            cpu_target_x=jnp.array(self.consts.P1_START_X, dtype=jnp.int32),
            cpu_target_y=jnp.array(self.consts.START_Y, dtype=jnp.int32),
            cpu_horiz_offset=jnp.array(0, dtype=jnp.int32),
            cpu_vert_offset=jnp.array(0, dtype=jnp.int32),
            cpu_dancing_value=jnp.array(0, dtype=jnp.int32),
            hit_anim_timer=jnp.array([0, 0], dtype=jnp.int32),
            hit_anim_dx=jnp.array([0, 0], dtype=jnp.int32),
            hit_anim_dy=jnp.array([0, 0], dtype=jnp.int32),
        )
        return self._get_observation(state), state

    def _get_observation(self, state: BoxingState) -> BoxingObservation:
        total_sec = state.timer // 60
        minutes = total_sec // 60
        seconds = total_sec % 60
        
        left_boxer = ObjectObservation(
            x=state.pos[0, 0].astype(jnp.int32),
            y=state.pos[0, 1].astype(jnp.int32),
            width=jnp.array(self.consts.W_BOXER),
            height=jnp.array(self.consts.H_BOXER),
            active=jnp.array(True),
            visual_id=jnp.array(0),
            state=state.punch_state[0],
            orientation=state.orientation[0],
        )
        
        right_boxer = ObjectObservation(
            x=state.pos[1, 0].astype(jnp.int32),
            y=state.pos[1, 1].astype(jnp.int32),
            width=jnp.array(self.consts.W_BOXER),
            height=jnp.array(self.consts.H_BOXER),
            active=jnp.array(True),
            visual_id=jnp.array(1),
            state=state.punch_state[1],
            orientation=state.orientation[1],
        )
        
        return BoxingObservation(
            left_boxer=left_boxer,
            right_boxer=right_boxer,
            score_left=state.score[0],
            score_right=state.score[1],
            clock_minutes=minutes.astype(jnp.int32),
            clock_seconds=seconds.astype(jnp.int32)
        )

    def _get_info(self, state: BoxingState) -> BoxingInfo:
        total_sec = state.timer // 60
        return BoxingInfo(
            time=self.consts.TOTAL_TIME - state.timer,
            clock_minutes=(total_sec // 60).astype(jnp.int32),
            clock_seconds=(total_sec % 60).astype(jnp.int32)
        )

    def _get_reward(self, state: BoxingState, new_state: BoxingState) -> jnp.ndarray:
        p1_points = new_state.score[0] - state.score[0]
        p2_points = new_state.score[1] - state.score[1]
        return p1_points - p2_points

    def _get_done(self, state: BoxingState) -> jnp.ndarray:
        return state.done

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        c = self.consts
        h = int(c.HEIGHT)
        w = int(c.WIDTH)
        screen_size = (h, w)
        single_obj = spaces.get_object_space(n=None, screen_size=screen_size)
        return spaces.Dict({
            "left_boxer": single_obj,
            "right_boxer": single_obj,
            "score_left": spaces.Box(low=0, high=100, shape=(), dtype=jnp.int32),
            "score_right": spaces.Box(low=0, high=100, shape=(), dtype=jnp.int32),
            "clock_minutes": spaces.Box(low=0, high=2, shape=(), dtype=jnp.int32),
            "clock_seconds": spaces.Box(low=0, high=59, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    def _move_boxer(self, state: BoxingState, idx: int, action: chex.Array):
        pos = state.pos[idx]
        hit_active = state.hit_anim_timer[idx] > 0
        
        # Decode action
        up = jnp.isin(action, jnp.array([Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE]))
        down = jnp.isin(action, jnp.array([Action.DOWN, Action.DOWNRIGHT, Action.DOWNLEFT, Action.DOWNFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE]))
        left = jnp.isin(action, jnp.array([Action.LEFT, Action.UPLEFT, Action.DOWNLEFT, Action.LEFTFIRE, Action.UPLEFTFIRE, Action.DOWNLEFTFIRE]))
        right = jnp.isin(action, jnp.array([Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT, Action.RIGHTFIRE, Action.UPRIGHTFIRE, Action.DOWNRIGHTFIRE]))
        
        dx = jnp.where(right, 1, jnp.where(left, -1, 0))
        dy = jnp.where(down, 1, jnp.where(up, -1, 0))
        
        # Calculate move velocities
        speed = jnp.where(idx == 0, self.consts.PLAYER_SPEED, self.consts.ENEMY_SPEED)
        move_dx = jnp.where(hit_active, state.hit_anim_dx[idx], dx * speed)
        move_dy = jnp.where(hit_active, state.hit_anim_dy[idx], dy * speed)
        
        can_move = jnp.logical_or(hit_active, state.stun_timer[idx] == 0)
        new_pos = pos + jnp.where(can_move, jnp.array([move_dx, move_dy]), 0)
        
        # Boundary clamping
        new_pos = jnp.array([
            jnp.clip(new_pos[0], self.consts.XMIN, self.consts.XMAX),
            jnp.clip(new_pos[1], self.consts.YMIN, self.consts.YMAX)
        ])
        return new_pos

    def _update_punch(self, state: BoxingState, action, idx, opponent_idx):
        # Explicitly convert action to int for comparison
        action_int = jnp.asarray(action, dtype=jnp.int32)
        fire_actions = jnp.array([
            Action.FIRE, Action.UPFIRE, Action.DOWNFIRE, Action.LEFTFIRE, Action.RIGHTFIRE,
            Action.UPLEFTFIRE, Action.UPRIGHTFIRE, Action.DOWNLEFTFIRE, Action.DOWNRIGHTFIRE
        ], dtype=jnp.int32)
        fire = jnp.any(action_int == fire_actions)
        fire = jnp.where(jnp.logical_and(idx == 1, self.consts.ENEMY_PEACEFUL), False, fire)
        
        curr_state = state.punch_state[idx]
        curr_cooldown = state.punch_cooldown[idx]
        is_stunned = jnp.logical_or(state.stun_timer[idx] > 0, state.hit_anim_timer[idx] > 0)
        
        # New cooldown (always decrement if > 0)
        dec_cooldown = jnp.maximum(curr_cooldown - 1, 0)
        
        # Determine target arm based on Y relative position:
        # Punch with top most arm (0) if opponent is above (opponent Y < player Y),
        # bottom arm (1) in the contrary case.
        target_arm = jnp.where(state.pos[opponent_idx, 1] < state.pos[idx, 1], 0, 1)
        
        def next_state_logic():
            # If idle and ready
            start_punch = jnp.logical_and(curr_state == 0, jnp.logical_and(dec_cooldown == 0, fire))
            
            # Progress punch state
            max_state = 17
            holding_extension = jnp.logical_and(curr_state == 12, fire)
            progressing = jnp.logical_and(jnp.logical_and(curr_state > 0, curr_state < max_state), ~holding_extension)
            finishing = curr_state == max_state
            
            new_s = jnp.where(start_punch, 1, 
                             jnp.where(holding_extension, 12,
                                       jnp.where(progressing, curr_state + 1, 0)))
            
            # Use the target arm computed based on the opponent's relative vertical position
            new_a = jnp.where(start_punch, target_arm, state.punch_arm[idx])
            
            new_h = jnp.where(start_punch, False, state.has_hit[idx])
            
            new_c = jnp.where(finishing, self.consts.PUNCH_COOLDOWN, dec_cooldown)
            
            return new_s, new_a, new_h, new_c

        # If stunned, reset state but keep decrementing cooldown
        new_s, new_a, new_h, new_c = jax.lax.cond(is_stunned, 
                                                 lambda: (0, state.punch_arm[idx], False, dec_cooldown), 
                                                 next_state_logic)
        
        return new_s, new_a, new_h, new_c

    def _update_cpu_state(self, state: BoxingState) -> BoxingState:
        # Split key for random decisions
        key, subkey1, subkey2, subkey3 = jax.random.split(state.key, 4)
        
        # Periodically update target position (every ~4 frames based on random for higher reaction speed)
        random_val = jax.random.randint(subkey1, (), 0, 256)
        update_target = (random_val & self.consts.CPU_UPDATE_MASK) == 0
        
        # Generate new random offsets
        new_horiz_offset = jax.random.randint(subkey2, (), 0, 32)  # 0-31
        new_vert_offset = jax.random.randint(subkey3, (), 0, 64)   # 0-63
        
        # Update target to track player position
        cpu_target_x = jnp.where(
            update_target,
            state.pos[0, 0],
            state.cpu_target_x
        ).astype(jnp.float32)
        cpu_target_y = jnp.where(
            update_target,
            state.pos[0, 1],
            state.cpu_target_y
        ).astype(jnp.float32)
        cpu_horiz_offset = jnp.where(
            update_target,
            new_horiz_offset,
            state.cpu_horiz_offset
        ).astype(jnp.int32)
        cpu_vert_offset = jnp.where(
            update_target,
            new_vert_offset,
            state.cpu_vert_offset
        ).astype(jnp.int32)
        
        # Decrement dancing value (moves towards 0)
        new_dancing = jnp.maximum(state.cpu_dancing_value - 1, 0).astype(jnp.int32)
        
        return replace(state,
            cpu_target_x=cpu_target_x,
            cpu_target_y=cpu_target_y,
            cpu_horiz_offset=cpu_horiz_offset,
            cpu_vert_offset=cpu_vert_offset,
            cpu_dancing_value=new_dancing,
            key=key
        )

    def _cpu_logic(self, state: BoxingState):
        p1_pos = state.pos[0]
        p2_pos = state.pos[1]
        
        # Calculate comfortable horizontal direction based on relative position
        is_cpu_on_right = p2_pos[0] >= p1_pos[0]
        sign_x = jnp.where(is_cpu_on_right, 1.0, -1.0)
        
        # Target position using the updated target coordinates and randomized offsets
        target_x = state.cpu_target_x + sign_x * (20.0 + (state.cpu_horiz_offset - 16.0))
        target_y = state.cpu_target_y + (state.cpu_vert_offset - 32.0)
        
        # Clamp target inside ring boundaries
        target_x = jnp.clip(target_x, self.consts.XMIN, self.consts.XMAX)
        target_y = jnp.clip(target_y, self.consts.YMIN, self.consts.YMAX)
        
        # Determine movement direction towards target
        move_right = target_x > p2_pos[0]
        move_left = target_x < p2_pos[0]
        move_down = target_y > p2_pos[1]
        move_up = target_y < p2_pos[1]
        
        # "Dancing" behavior: reverse horizontal movement when dancing and not hit
        dancing = state.cpu_dancing_value >= 16
        cpu_not_hit = state.stun_timer[1] == 0
        reverse_horiz = jnp.logical_and(dancing, cpu_not_hit)
        
        move_right_final = jnp.where(reverse_horiz, move_left, move_right)
        move_left_final = jnp.where(reverse_horiz, move_right, move_left)
        
        dx = jnp.where(move_right_final, 1, jnp.where(move_left_final, -1, 0))
        dy = jnp.where(move_down, 1, jnp.where(move_up, -1, 0))
        
        # Strike decision based on proximity and randomness
        horiz_dist = jnp.abs(p1_pos[0] - p2_pos[0])
        vert_dist = jnp.abs(p1_pos[1] - p2_pos[1])
        in_range = jnp.logical_and(horiz_dist <= self.consts.JAB_DIST, vert_dist <= self.consts.H_BOXER)
        
        # Don't start a punch while dancing (unless CPU was hit)
        punch_dancing = state.cpu_dancing_value > 0
        cpu_was_hit = state.stun_timer[1] > 0
        can_punch_dancing = jnp.logical_or(~punch_dancing, cpu_was_hit)
        
        score_diff = state.score[1] - state.score[0]
        aggressiveness = jnp.where(score_diff >= 0, self.consts.CPU_AGGR_WINNING, self.consts.CPU_AGGR_LOSING)
        
        # Split state.key to make random decision (without updating state.key here)
        _, subkey = jax.random.split(state.key)
        random_val = jax.random.randint(subkey, (), 0, 256)
        should_punch = random_val < aggressiveness
        
        is_idle = state.punch_state[1] == 0
        is_ready = state.punch_cooldown[1] == 0
        
        strike_decision = jnp.logical_and(
            jnp.logical_and(is_idle, is_ready),
            jnp.logical_and(jnp.logical_and(in_range, can_punch_dancing), should_punch)
        )
        
        # Combine movement and punching into a single action
        act = jnp.where(
            strike_decision,
            jnp.where(dy == -1,
                jnp.where(dx == 1, Action.UPRIGHTFIRE, jnp.where(dx == -1, Action.UPLEFTFIRE, Action.UPFIRE)),
                jnp.where(dy == 1,
                    jnp.where(dx == 1, Action.DOWNRIGHTFIRE, jnp.where(dx == -1, Action.DOWNLEFTFIRE, Action.DOWNFIRE)),
                    jnp.where(dx == 1, Action.RIGHTFIRE, jnp.where(dx == -1, Action.LEFTFIRE, Action.FIRE))
                )
            ),
            jnp.where(dy == -1,
                jnp.where(dx == 1, Action.UPRIGHT, jnp.where(dx == -1, Action.UPLEFT, Action.UP)),
                jnp.where(dy == 1,
                    jnp.where(dx == 1, Action.DOWNRIGHT, jnp.where(dx == -1, Action.DOWNLEFT, Action.DOWN)),
                    jnp.where(dx == 1, Action.RIGHT, jnp.where(dx == -1, Action.LEFT, Action.NOOP))
                )
            )
        )
        return act

    def step(self, state: BoxingState, action: chex.Array) -> Tuple[BoxingObservation, BoxingState, int, bool, BoxingInfo]:
        key, cpu_key = jax.random.split(state.key)
        state = replace(state, key=cpu_key)
        
        # Update CPU targeting, offsets, and dancing countdown
        state = self._update_cpu_state(state)
        
        # 1. CPU Action
        p2_action = self._cpu_logic(state)
        
        # 2. Movement
        new_p1_pos = self._move_boxer(state, 0, action)
        new_p2_pos = self._move_boxer(state, 1, p2_action)
        
        # 3. Collision (AABB push-out along the axis of minimum overlap)
        dx = new_p1_pos[0] - new_p2_pos[0]
        dy = new_p1_pos[1] - new_p2_pos[1]
        overlap_x = self.consts.W_BOXER - jnp.abs(dx)
        overlap_y = self.consts.H_BOXER - jnp.abs(dy)
        collision = jnp.logical_and(overlap_x > 0, overlap_y > 0)
        
        # Determine push direction based on the axis of minimum overlap
        sign_x = jnp.where(dx >= 0, 1, -1)
        sign_y = jnp.where(dy >= 0, 1, -1)
        
        push_x = jnp.where(overlap_x < overlap_y, sign_x * overlap_x, 0)
        push_y = jnp.where(overlap_x >= overlap_y, sign_y * overlap_y, 0)
        push = jnp.stack([push_x, push_y])
        
        new_p1_pos = jnp.where(collision, new_p1_pos + push // 2, new_p1_pos)
        new_p2_pos = jnp.where(collision, new_p2_pos - push // 2, new_p2_pos)
        
        # Clamp again after collision push
        new_p1_pos = jnp.clip(new_p1_pos, jnp.array([self.consts.XMIN, self.consts.YMIN]), jnp.array([self.consts.XMAX, self.consts.YMAX]))
        new_p2_pos = jnp.clip(new_p2_pos, jnp.array([self.consts.XMIN, self.consts.YMIN]), jnp.array([self.consts.XMAX, self.consts.YMAX]))
        
        pos = jnp.stack([new_p1_pos, new_p2_pos])
        orientation = jnp.array([
            (pos[0, 0] > pos[1, 0]).astype(jnp.int32),
            (pos[1, 0] > pos[0, 0]).astype(jnp.int32)
        ])
        state = replace(state, pos=pos, orientation=orientation)
        
        # 4. Punch State Update
        s0, a0, h0, c0 = self._update_punch(state, action, 0, 1)
        s1, a1, h1, c1 = self._update_punch(state, p2_action, 1, 0)
        
        def print_idle(operand):
            jax.debug.print("White Player: Idle")
 
        def print_punching(operand):
            state_val, arm_val = operand
            jax.debug.print("White Player: Punching - Arm (0=Top, 1=Bottom): {arm}, State: {state}", arm=arm_val, state=state_val)
 
        jax.lax.cond(s0 == 0, print_idle, print_punching, (s0, a0))
        
        state = replace(state, 
                        punch_state=jnp.array([s0, s1]), 
                        punch_arm=jnp.array([a0, a1]), 
                        has_hit=jnp.array([h0, h1]),
                        punch_cooldown=jnp.array([c0, c1]))
        
        # 5. Hit Detection & Scoring & Knockback
        def check_hit(attacker_idx, defender_idx, s):
            a_pos = s.pos[attacker_idx]
            d_pos = s.pos[defender_idx]
            
            p_state = s.punch_state[attacker_idx]
            not_hit_yet = jnp.logical_not(s.has_hit[attacker_idx])
            d_not_stunned = s.stun_timer[defender_idx] == 0
            
            punch_y = jnp.where(s.punch_arm[attacker_idx] == 0, a_pos[1] + self.consts.TOP_ARM_Y, a_pos[1] + self.consts.BOT_ARM_Y)
            face_shrink = jnp.where(defender_idx == 0, self.consts.PLAYER_FACE_SHRINK_Y, 0)
            min_y = d_pos[1] + self.consts.FACE_MIN_Y + face_shrink
            max_y = d_pos[1] + self.consts.FACE_MAX_Y - face_shrink
            
            in_power_vert_range = jnp.logical_and(punch_y >= min_y, punch_y <= max_y)
            in_jab_vert_range = jnp.logical_and(punch_y >= min_y + 6, punch_y <= max_y - 6)
            
            face_x_min = d_pos[0]
            face_x_max = d_pos[0] + self.consts.W_BOXER

            a_orient = s.orientation[attacker_idx]
            frame_map = jnp.array([0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 2, 2, 0, 0, 0])
            anim_frame = frame_map[p_state]
            
            start_r = jnp.array([10, 10, 14, 14])[anim_frame]
            start_l = jnp.array([0, 0, -8, -16])[anim_frame]
            glove_w = jnp.array([4, 4, 8, 16])[anim_frame]
            
            glove_x_min = jnp.where(a_orient == 0, a_pos[0] + start_r, a_pos[0] + start_l)
            glove_x_max = glove_x_min + glove_w
            
            in_power_horiz_range = jnp.logical_and(glove_x_max >= face_x_min, glove_x_min <= face_x_max)
            
            # Shrink face box horizontally by 4px on each side for Jab
            jab_face_x_min = face_x_min + 4
            jab_face_x_max = face_x_max - 4
            in_jab_horiz_range = jnp.logical_and(glove_x_max >= jab_face_x_min, glove_x_min <= jab_face_x_max)
            
            # Jab states: (8, 9)
            is_jab_state = jnp.isin(p_state, jnp.array([8, 9]))
            
            # Power states: (10, 11) - we remove 12 so that holding the arm extended doesn't register hits
            is_power_state = jnp.isin(p_state, jnp.array([10, 11]))
 
            is_jab = jnp.logical_and(is_jab_state, jnp.logical_and(in_jab_horiz_range, in_jab_vert_range))
            is_power = jnp.logical_and(is_power_state, jnp.logical_and(in_power_horiz_range, in_power_vert_range))
            
            valid_hit = jnp.logical_and(jnp.logical_or(is_jab, is_power), 
                                         jnp.logical_and(not_hit_yet, d_not_stunned))
            
            # Jab (short, deep hit) = 2 points, Power/Normal (long, extended hit) = 1 point
            points = jnp.where(valid_hit, jnp.where(is_jab, 2, 1), 0)
            
            return valid_hit, points
 
        # P1 hits P2
        p1_hit, p1_points = check_hit(0, 1, state)
        # P2 hits P1
        p2_hit, p2_points = check_hit(1, 0, state)
        
        # Apply hits
        new_scores = state.score + jnp.array([p1_points, p2_points])
        new_has_hit = state.has_hit.at[0].set(jnp.logical_or(state.has_hit[0], p1_hit)).at[1].set(jnp.logical_or(state.has_hit[1], p2_hit))
        
        # Apply Stun
        new_stun = jnp.maximum(state.stun_timer - 1, 0)
        new_stun = new_stun.at[1].set(jnp.where(p1_hit, self.consts.STUN_DURATION, new_stun[1]))
        new_stun = new_stun.at[0].set(jnp.where(p2_hit, self.consts.STUN_DURATION, new_stun[0]))
        
        # Trigger hit projection animation
        p1_arm = state.punch_arm[0]
        p2_arm = state.punch_arm[1]
 
        # Horizontal backward direction for each player if they get hit
        p1_back_dx = jnp.where(state.orientation[0] == 0, -self.consts.KNOCKBACK_DX, self.consts.KNOCKBACK_DX)
        p2_back_dx = jnp.where(state.orientation[1] == 0, -self.consts.KNOCKBACK_DX, self.consts.KNOCKBACK_DX)
 
        # Vertical direction depending on opponent's punch arm
        p1_kb_dy = jnp.where(p2_arm == 0, self.consts.KNOCKBACK_TOP_ARM_DY, self.consts.KNOCKBACK_BOT_ARM_DY)
        p2_kb_dy = jnp.where(p1_arm == 0, self.consts.KNOCKBACK_TOP_ARM_DY, self.consts.KNOCKBACK_BOT_ARM_DY)
 
        # Decrement hit animation timer and update with new hits if any
        new_hit_anim_timer = jnp.maximum(state.hit_anim_timer - 1, 0)
        new_hit_anim_timer = new_hit_anim_timer.at[0].set(jnp.where(p2_hit, self.consts.HIT_ANIMATION_STEPS, new_hit_anim_timer[0]))
        new_hit_anim_timer = new_hit_anim_timer.at[1].set(jnp.where(p1_hit, self.consts.HIT_ANIMATION_STEPS, new_hit_anim_timer[1]))
 
        new_hit_anim_dx = state.hit_anim_dx
        new_hit_anim_dx = new_hit_anim_dx.at[0].set(jnp.where(p2_hit, p1_back_dx, jnp.where(new_hit_anim_timer[0] > 0, state.hit_anim_dx[0], 0)))
        new_hit_anim_dx = new_hit_anim_dx.at[1].set(jnp.where(p1_hit, p2_back_dx, jnp.where(new_hit_anim_timer[1] > 0, state.hit_anim_dx[1], 0)))
 
        new_hit_anim_dy = state.hit_anim_dy
        new_hit_anim_dy = new_hit_anim_dy.at[0].set(jnp.where(p2_hit, p1_kb_dy, jnp.where(new_hit_anim_timer[0] > 0, state.hit_anim_dy[0], 0)))
        new_hit_anim_dy = new_hit_anim_dy.at[1].set(jnp.where(p1_hit, p2_kb_dy, jnp.where(new_hit_anim_timer[1] > 0, state.hit_anim_dy[1], 0)))
        
        new_dancing = jnp.where(p1_hit, self.consts.CPU_DANCING_DURATION, state.cpu_dancing_value).astype(jnp.int32)
        
        state = replace(state, 
                        score=new_scores,
                        has_hit=new_has_hit,
                        stun_timer=new_stun,
                        timer=state.timer - 1,
                        cpu_dancing_value=new_dancing,
                        hit_anim_timer=new_hit_anim_timer,
                        hit_anim_dx=new_hit_anim_dx,
                        hit_anim_dy=new_hit_anim_dy,
                        key=key)
        
        # 6. Termination
        done = jnp.logical_or(jnp.any(state.score >= self.consts.MAX_SCORE), state.timer <= 0)
        state = replace(state, done=done)
        
        return self._get_observation(state), state, (p1_points - p2_points).astype(jnp.float32), done, self._get_info(state)

    def render(self, state: BoxingState, debug: bool = False) -> jnp.ndarray:
        return self.renderer.render(state, debug=debug)


# =============================================================================
# Renderer
# =============================================================================

class BoxingRenderer(JAXGameRenderer):
    def __init__(self, consts: BoxingConstants | None = None, config: render_utils.RendererConfig | None = None):
        self.consts = consts or BoxingConstants()
        super().__init__(self.consts)
        self.config = config or render_utils.RendererConfig(game_dimensions=(210, 160), channels=3)
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/boxing"
        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND, self.COLOR_TO_ID, _) = self.jr.load_and_setup_assets(list(self.consts.ASSET_CONFIG), sprite_path)
        
        # Custom debug colors appended to palette
        self.DEBUG_RED_ID = self.PALETTE.shape[0]
        self.DEBUG_GREEN_ID = self.DEBUG_RED_ID + 1
        red_rgb = jnp.array([[255, 0, 0]], dtype=self.PALETTE.dtype)
        green_rgb = jnp.array([[0, 255, 0]], dtype=self.PALETTE.dtype)
        self.PALETTE = jnp.concatenate([self.PALETTE, red_rgb, green_rgb], axis=0)
        
        self.white_masks = {
            "body": self.SHAPE_MASKS["white_main_body"],
            "stunned": self.SHAPE_MASKS["white_stunned"],
            "top_arm": [
                self.SHAPE_MASKS["white_top_arm_idle"],
                self.SHAPE_MASKS["white_top_arm_retracted"],
                self.SHAPE_MASKS["white_top_arm_stretched"],
                self.SHAPE_MASKS["white_top_arm_extended"],
            ],
            "bottom_arm": [
                self.SHAPE_MASKS["white_bottom_arm_idle"],
                self.SHAPE_MASKS["white_bottom_arm_retracted"],
                self.SHAPE_MASKS["white_bottom_arm_stretched"],
                self.SHAPE_MASKS["white_bottom_arm_extended"],
            ],
        }
        self.black_masks = {
            "body": self.SHAPE_MASKS["black_main_body"],
            # black_stunned.npy is saved facing Left by default, so we pre-flip it horizontally
            # here to make it face Right, consistent with all other boxer assets.
            "stunned": self.SHAPE_MASKS["black_stunned"][:, ::-1],
            "top_arm": [
                self.SHAPE_MASKS["black_top_arm_idle"],
                self.SHAPE_MASKS["black_top_arm_retracted"],
                self.SHAPE_MASKS["black_top_arm_stretched"],
                self.SHAPE_MASKS["black_top_arm_extended"],
            ],
            "bottom_arm": [
                self.SHAPE_MASKS["black_bottom_arm_idle"],
                self.SHAPE_MASKS["black_bottom_arm_retracted"],
                self.SHAPE_MASKS["black_bottom_arm_stretched"],
                self.SHAPE_MASKS["black_bottom_arm_extended"],
            ],
        }

    def _render_boxer(self, raster, pos, is_stunned, p_state, arm_idx, masks, orientation):
        x = pos[0]
        y = pos[1]
        
        def render_stunned(r):
            mask = masks["stunned"]
            # If oriented Left (1), flip
            mask = jnp.where(orientation == 1, mask[:, ::-1], mask)
            return self.jr.render_at(r, x, y, mask)
            
        def render_normal(r):
            # Map punch state (0-17) to arm animation frame (0=idle, 1=retract, 2=stretch, 3=extend)
            frame_map = jnp.array([0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 2, 2, 0, 0, 0])
            anim_frame = frame_map[p_state]
            
            top_frame = jnp.where(jnp.logical_and(p_state > 0, arm_idx == 0), anim_frame, 0)
            bot_frame = jnp.where(jnp.logical_and(p_state > 0, arm_idx == 1), anim_frame, 0)
            
            def render_arm(op, i, is_top):
                mask_key = "top_arm" if is_top else "bottom_arm"
                mask = masks[mask_key][i]
                
                # Flip if facing Left (1)
                mask = jnp.where(orientation == 1, mask[:, ::-1], mask)
                
                arm_w = mask.shape[1]
                
                # Logic for Right (orientation 0)
                arm_x_right = x - (arm_w - 14)
                arm_x_right = jnp.where(i == 2, arm_x_right + 8, arm_x_right)
                arm_x_right = jnp.where(i == 3, arm_x_right + 16, arm_x_right)
                
                # Logic for Left (orientation 1)
                arm_x_left = jnp.where(i == 2, x - 8, x)
                arm_x_left = jnp.where(i == 3, x - 16, arm_x_left)
                
                arm_x = jnp.where(orientation == 0, arm_x_right, arm_x_left)
                arm_y = y if is_top else y + 34
                return self.jr.render_at(op, arm_x, arm_y, mask)
                
            r = jax.lax.switch(top_frame, [lambda op, i=i: render_arm(op, i, True) for i in range(4)], r)
            
            body_mask = masks["body"]
            # If oriented Left (1), flip
            body_mask = jnp.where(orientation == 1, body_mask[:, ::-1], body_mask)
            r = self.jr.render_at(r, x, y + 12, body_mask)
            
            r = jax.lax.switch(bot_frame, [lambda op, i=i: render_arm(op, i, False) for i in range(4)], r)
            return r

        return jax.lax.cond(is_stunned, render_stunned, render_normal, raster)

    @partial(jax.jit, static_argnums=(0, 2))
    def render(self, state: BoxingState, debug: bool = False) -> jnp.ndarray:
        raster_empty = self.jr.create_object_raster(self.BACKGROUND)
        
        raster = self._render_boxer(raster_empty, state.pos[0].astype(jnp.int32), 
                                    state.stun_timer[0] > 0, state.punch_state[0], 
                                    state.punch_arm[0], self.white_masks, state.orientation[0])
                                    
        raster = self._render_boxer(raster, state.pos[1].astype(jnp.int32), 
                                    state.stun_timer[1] > 0, state.punch_state[1], 
                                    state.punch_arm[1], self.black_masks, state.orientation[1])
        
        # HUD: Scores
        white_digits = self.jr.int_to_digits(state.score[0], max_digits=2)
        raster = self.jr.render_label(raster, 20, 5, white_digits, self.SHAPE_MASKS["digits_white"], spacing=8)
        
        black_digits = self.jr.int_to_digits(state.score[1], max_digits=2)
        raster = self.jr.render_label(raster, 130, 5, black_digits, self.SHAPE_MASKS["digits_black"], spacing=8)
        
        # HUD: Timer (M:SS)
        total_sec = jnp.maximum(state.timer, 0) // 60
        minutes = total_sec // 60
        seconds = total_sec % 60
        min_digit = self.jr.int_to_digits(minutes, max_digits=1)
        sec_digits = self.jr.int_to_digits(seconds, max_digits=2)
        raster = self.jr.render_label(raster, 70, 5, min_digit, self.SHAPE_MASKS["digits_time"], spacing=0)
        raster = self.jr.render_label(raster, 82, 5, sec_digits, self.SHAPE_MASKS["digits_time"], spacing=8)

        # Hitbox calculations (always computed, cheap)
        frame_map = jnp.array([0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 2, 2, 0, 0, 0])
        
        # Start offsets from pos[0] and widths for each frame (0=idle, 1=retract, 2=jab, 3=power)
        start_offsets_r = jnp.array([10.0, 10.0, 14.0, 14.0])
        widths = jnp.array([4.0, 4.0, 8.0, 16.0])
        start_offsets_l = jnp.array([0.0, 0.0, -8.0, -16.0])
        
        # Player 0
        p0_state = state.punch_state[0]
        anim_frame0 = frame_map[p0_state]
        top_frame0 = jnp.where(jnp.logical_and(p0_state > 0, state.punch_arm[0] == 0), anim_frame0, 0)
        bot_frame0 = jnp.where(jnp.logical_and(p0_state > 0, state.punch_arm[0] == 1), anim_frame0, 0)
        
        y_top0 = state.pos[0, 1] + self.consts.TOP_ARM_Y - 1.0
        y_bot0 = state.pos[0, 1] + self.consts.BOT_ARM_Y - 1.0
        
        w_top0 = widths[top_frame0]
        x_r_top0 = state.pos[0, 0] + start_offsets_r[top_frame0]
        x_l_top0 = state.pos[0, 0] + start_offsets_l[top_frame0]
        x_top0 = jnp.where(state.orientation[0] == 0, x_r_top0, x_l_top0)
        
        w_bot0 = widths[bot_frame0]
        x_r_bot0 = state.pos[0, 0] + start_offsets_r[bot_frame0]
        x_l_bot0 = state.pos[0, 0] + start_offsets_l[bot_frame0]
        x_bot0 = jnp.where(state.orientation[0] == 0, x_r_bot0, x_l_bot0)
        
        # Player 1
        p1_state = state.punch_state[1]
        anim_frame1 = frame_map[p1_state]
        top_frame1 = jnp.where(jnp.logical_and(p1_state > 0, state.punch_arm[1] == 0), anim_frame1, 0)
        bot_frame1 = jnp.where(jnp.logical_and(p1_state > 0, state.punch_arm[1] == 1), anim_frame1, 0)
        
        y_top1 = state.pos[1, 1] + self.consts.TOP_ARM_Y - 1.0
        y_bot1 = state.pos[1, 1] + self.consts.BOT_ARM_Y - 1.0
        
        w_top1 = widths[top_frame1]
        x_r_top1 = state.pos[1, 0] + start_offsets_r[top_frame1]
        x_l_top1 = state.pos[1, 0] + start_offsets_l[top_frame1]
        x_top1 = jnp.where(state.orientation[1] == 0, x_r_top1, x_l_top1)
        
        w_bot1 = widths[bot_frame1]
        x_r_bot1 = state.pos[1, 0] + start_offsets_r[bot_frame1]
        x_l_bot1 = state.pos[1, 0] + start_offsets_l[bot_frame1]
        x_bot1 = jnp.where(state.orientation[1] == 0, x_r_bot1, x_l_bot1)
        
        face0_shrink = self.consts.PLAYER_FACE_SHRINK_Y
        face1_shrink = 0.0
        face_positions = jnp.array([
            [state.pos[0, 0], state.pos[0, 1] + self.consts.FACE_MIN_Y + face0_shrink],
            [state.pos[1, 0], state.pos[1, 1] + self.consts.FACE_MIN_Y + face1_shrink],
        ])
        face_sizes = jnp.array([
            [self.consts.W_BOXER, self.consts.FACE_MAX_Y - self.consts.FACE_MIN_Y - 2 * face0_shrink],
            [self.consts.W_BOXER, self.consts.FACE_MAX_Y - self.consts.FACE_MIN_Y - 2 * face1_shrink],
        ])
        jab_face_positions = jnp.array([
            [state.pos[0, 0] + 4, state.pos[0, 1] + self.consts.FACE_MIN_Y + face0_shrink + 6],
            [state.pos[1, 0] + 4, state.pos[1, 1] + self.consts.FACE_MIN_Y + face1_shrink + 6],
        ])
        jab_face_sizes = jnp.array([
            [self.consts.W_BOXER - 8, self.consts.FACE_MAX_Y - self.consts.FACE_MIN_Y - 2 * face0_shrink - 12],
            [self.consts.W_BOXER - 8, self.consts.FACE_MAX_Y - self.consts.FACE_MIN_Y - 2 * face1_shrink - 12],
        ])
        glove_positions = jnp.array([
            [x_top0, y_top0], [x_bot0, y_bot0],
            [x_top1, y_top1], [x_bot1, y_bot1]
        ])
        glove_sizes = jnp.array([
            [w_top0, 3.0], [w_bot0, 3.0],
            [w_top1, 3.0], [w_bot1, 3.0]
        ])

        def apply_debug_solid(r):
            r = self.jr.draw_rects(r, face_positions, face_sizes, self.DEBUG_RED_ID)
            r = self.jr.draw_rects(r, glove_positions, glove_sizes, self.DEBUG_GREEN_ID)
            return r

        raster = jax.lax.cond(debug, apply_debug_solid, lambda r: r, raster)
        base_img = self.jr.render_from_palette(raster, self.PALETTE)
        
        def blend_hitboxes(img):
            # Create a blank raster to draw hitboxes
            hitbox_mask = jnp.zeros_like(raster)
            
            # Use 1 to mark outer face pixels and gloves (highly transparent)
            hitbox_mask = self.jr.draw_rects(hitbox_mask, face_positions, face_sizes, 1)
            hitbox_mask = self.jr.draw_rects(hitbox_mask, glove_positions, glove_sizes, 1)
            
            # Use 2 to mark inner face pixels for Jab (less transparent)
            hitbox_mask = self.jr.draw_rects(hitbox_mask, jab_face_positions, jab_face_sizes, 2)
            
            is_hitbox_1 = jnp.expand_dims(hitbox_mask == 1, axis=-1)
            is_hitbox_2 = jnp.expand_dims(hitbox_mask == 2, axis=-1)
            
            red_color = jnp.array([255, 0, 0], dtype=jnp.float32)
            # Highly transparent (25% opacity) for normal punch detection zone and gloves
            blended_1 = (img.astype(jnp.float32) * 0.75 + red_color * 0.25).astype(jnp.uint8)
            # Less transparent (70% opacity) for jab detection zone
            blended_2 = (img.astype(jnp.float32) * 0.3 + red_color * 0.7).astype(jnp.uint8)
            
            img = jnp.where(is_hitbox_1, blended_1, img)
            img = jnp.where(is_hitbox_2, blended_2, img)
            return img

        return jax.lax.cond(
            self.consts.SHOW_COLLISION_ZONE,
            blend_hitboxes,
            lambda x: x,
            base_img
        )
