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
# Asset Config 
# =============================================================================

def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
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
    
    # Movement
    MOVE_SPEED: float = 0.8
    KNOCKBACK_DIST: float = 3.0
    STUN_DURATION: int = 12
    
    # Punch Mechanics
    PUNCH_STATE_MAX: int = 4
    PUNCH_COOLDOWN: int = 8   # Delay between punches
    JAB_DIST: float = 28.0    # Distance for 1pt hit
    POWER_DIST: float = 16.0  # Distance for 2pt hit
    
    # Game rules
    MAX_SCORE: int = 100
    TOTAL_TIME: int = 7200 # 2 minutes at 60Hz
    
    # Starting positions
    P1_START_X: float = 50.0
    P2_START_X: float = 95.0
    START_Y: float = 82.0

    ASSET_CONFIG: tuple = _get_default_asset_config()


# =============================================================================
# State 
# =============================================================================

@struct.dataclass
class BoxingState:
    pos: chex.Array          # [2, 2] (P1/P2, X/Y)
    score: chex.Array        # [2]
    punch_state: chex.Array  # [2] (0-4)
    punch_arm: chex.Array    # [2] (0: left, 1: right)
    punch_cooldown: chex.Array # [2] (frames until next punch allowed)
    has_hit: chex.Array      # [2] (bool)
    stun_timer: chex.Array   # [2]
    timer: chex.Array        # int32
    done: chex.Array         # bool
    key: chex.PRNGKey


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
    def __init__(self, consts: BoxingConstants | None = None):
        consts = consts or BoxingConstants()
        super().__init__(consts)
        self.renderer = BoxingRenderer(self.consts)
        self.action_set = [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
            Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
            Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
            Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
        ]

    def reset(self, key: chex.PRNGKey) -> Tuple[BoxingObservation, BoxingState]:
        key, subkey = jax.random.split(key)
        state = BoxingState(
            pos=jnp.array([[self.consts.P1_START_X, self.consts.START_Y],
                           [self.consts.P2_START_X, self.consts.START_Y]], dtype=jnp.float32),
            score=jnp.array([0, 0], dtype=jnp.int32),
            punch_state=jnp.array([0, 0], dtype=jnp.int32),
            punch_arm=jnp.array([0, 0], dtype=jnp.int32),
            punch_cooldown=jnp.array([0, 0], dtype=jnp.int32),
            has_hit=jnp.array([False, False], dtype=jnp.bool_),
            stun_timer=jnp.array([0, 0], dtype=jnp.int32),
            timer=jnp.array(self.consts.TOTAL_TIME, dtype=jnp.int32),
            done=jnp.array(False),
            key=subkey
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
            orientation=jnp.array(0),
        )
        
        right_boxer = ObjectObservation(
            x=state.pos[1, 0].astype(jnp.int32),
            y=state.pos[1, 1].astype(jnp.int32),
            width=jnp.array(self.consts.W_BOXER),
            height=jnp.array(self.consts.H_BOXER),
            active=jnp.array(True),
            visual_id=jnp.array(1),
            state=state.punch_state[1],
            orientation=jnp.array(1),
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
        return (p1_points - p2_points).astype(jnp.float32)

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

    def _move_boxer(self, pos, action, stun_timer):
        # Decode action
        up = jnp.isin(action, jnp.array([Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE]))
        down = jnp.isin(action, jnp.array([Action.DOWN, Action.DOWNRIGHT, Action.DOWNLEFT, Action.DOWNFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE]))
        left = jnp.isin(action, jnp.array([Action.LEFT, Action.UPLEFT, Action.DOWNLEFT, Action.LEFTFIRE, Action.UPLEFTFIRE, Action.DOWNLEFTFIRE]))
        right = jnp.isin(action, jnp.array([Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT, Action.RIGHTFIRE, Action.UPRIGHTFIRE, Action.DOWNRIGHTFIRE]))
        
        dx = jnp.where(right, 1.0, jnp.where(left, -1.0, 0.0))
        dy = jnp.where(down, 1.0, jnp.where(up, -1.0, 0.0))
        
        # Normalize diagonal movement
        norm = jnp.sqrt(dx**2 + dy**2 + 1e-8)
        dx = jnp.where(norm > 1.0, dx / norm, dx)
        dy = jnp.where(norm > 1.0, dy / norm, dy)
        
        can_move = stun_timer == 0
        new_pos = pos + jnp.where(can_move, jnp.array([dx, dy]) * self.consts.MOVE_SPEED, 0.0)
        
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
        
        curr_state = state.punch_state[idx]
        curr_cooldown = state.punch_cooldown[idx]
        is_stunned = state.stun_timer[idx] > 0
        
        # New cooldown (always decrement if > 0)
        dec_cooldown = jnp.maximum(curr_cooldown - 1, 0)
        
        # Determine target arm if we start a punch based on Y relative position
        # For White (idx 0), facing Right: if above opponent, use bottom arm (1: right)
        # For Black (idx 1), facing Left: if above opponent, use bottom arm (0: left)
        # So we invert the arm selection for Black.
        target_arm = jnp.where(state.pos[idx, 1] < state.pos[opponent_idx, 1], 1, 0)
        target_arm = jnp.where(idx == 1, 1 - target_arm, target_arm)
        
        # Alternating logic: if holding fire, we might want to alternate.
        # But the original game also allows picking the best arm.
        # The user said "following the same logic as the other player".
        # Let's keep it simple and pick the best arm for now, but fix the black boxer's inversion.
        # To truly follow the original, we should probably toggle.
        # Let's implement toggling to match the description "automatically alternates".
        next_arm = 1 - state.punch_arm[idx]
        
        def next_state_logic():
            # If idle and ready
            start_punch = jnp.logical_and(curr_state == 0, jnp.logical_and(dec_cooldown == 0, fire))
            
            # Progress punch state
            progressing = jnp.logical_and(curr_state > 0, curr_state < self.consts.PUNCH_STATE_MAX)
            finishing = curr_state == self.consts.PUNCH_STATE_MAX
            
            new_s = jnp.where(start_punch, 1, 
                             jnp.where(progressing, curr_state + 1, 0))
            
            # Use next_arm for alternation if we just finished a punch and FIRE is still held,
            # or just use target_arm for the first punch.
            # Actually, the simplest alternating logic is to toggle every time a punch starts.
            new_a = jnp.where(start_punch, next_arm, state.punch_arm[idx])
            
            new_h = jnp.where(start_punch, False, state.has_hit[idx])
            
            new_c = jnp.where(finishing, self.consts.PUNCH_COOLDOWN, dec_cooldown)
            
            return new_s, new_a, new_h, new_c

        # If stunned, reset state but keep decrementing cooldown
        new_s, new_a, new_h, new_c = jax.lax.cond(is_stunned, 
                                                 lambda: (0, state.punch_arm[idx], False, dec_cooldown), 
                                                 next_state_logic)
        
        return new_s, new_a, new_h, new_c

    def _cpu_logic(self, state: BoxingState):
        p1_pos = state.pos[0]
        p2_pos = state.pos[1]
        
        # Simple AI: Track P1's Y, stay at distance on X
        target_x = p1_pos[0] + jnp.where(p2_pos[0] > p1_pos[0], 20.0, -20.0)
        target_y = p1_pos[1]
        
        dx = jnp.where(p2_pos[0] < target_x - 2, Action.RIGHT, jnp.where(p2_pos[0] > target_x + 2, Action.LEFT, Action.NOOP))
        dy = jnp.where(p2_pos[1] < target_y - 2, Action.DOWN, jnp.where(p2_pos[1] > target_y + 2, Action.UP, Action.NOOP))
        
        # Combine into action
        # Very simplified action mapping for CPU
        act = Action.NOOP
        act = jnp.where(jnp.logical_and(dx == Action.RIGHT, dy == Action.UP), Action.UPRIGHT, act)
        act = jnp.where(jnp.logical_and(dx == Action.LEFT, dy == Action.UP), Action.UPLEFT, act)
        act = jnp.where(jnp.logical_and(dx == Action.RIGHT, dy == Action.DOWN), Action.DOWNRIGHT, act)
        act = jnp.where(jnp.logical_and(dx == Action.LEFT, dy == Action.DOWN), Action.DOWNLEFT, act)
        act = jnp.where(jnp.logical_and(act == Action.NOOP, dx != Action.NOOP), dx, act)
        act = jnp.where(jnp.logical_and(act == Action.NOOP, dy != Action.NOOP), dy, act)
        
        # Punch if close
        dist = jnp.linalg.norm(p1_pos - p2_pos)
        should_punch = jnp.logical_and(dist < 30.0, jax.random.uniform(state.key, ()) < 0.1)
        act = jnp.where(should_punch, Action.FIRE, act)
        
        return act

    def step(self, state: BoxingState, action: chex.Array) -> Tuple[BoxingObservation, BoxingState, float, bool, BoxingInfo]:
        key, cpu_key = jax.random.split(state.key)
        state = replace(state, key=cpu_key)
        
        # 1. CPU Action
        p2_action = self._cpu_logic(state)
        
        # 2. Movement
        new_p1_pos = self._move_boxer(state.pos[0], action, state.stun_timer[0])
        new_p2_pos = self._move_boxer(state.pos[1], p2_action, state.stun_timer[1])
        
        # 3. Collision (simple push-out)
        dist = jnp.linalg.norm(new_p1_pos - new_p2_pos)
        min_dist = self.consts.W_BOXER
        overlap = min_dist - dist
        collision = overlap > 0
        push_dir = (new_p1_pos - new_p2_pos) / (dist + 1e-5)
        new_p1_pos = jnp.where(collision, new_p1_pos + push_dir * overlap * 0.5, new_p1_pos)
        new_p2_pos = jnp.where(collision, new_p2_pos - push_dir * overlap * 0.5, new_p2_pos)
        
        # Clamp again after collision push
        new_p1_pos = jnp.clip(new_p1_pos, jnp.array([self.consts.XMIN, self.consts.YMIN]), jnp.array([self.consts.XMAX, self.consts.YMAX]))
        new_p2_pos = jnp.clip(new_p2_pos, jnp.array([self.consts.XMIN, self.consts.YMIN]), jnp.array([self.consts.XMAX, self.consts.YMAX]))
        
        state = replace(state, pos=jnp.stack([new_p1_pos, new_p2_pos]))
        
        # 4. Punch State Update
        s0, a0, h0, c0 = self._update_punch(state, action, 0, 1)
        s1, a1, h1, c1 = self._update_punch(state, p2_action, 1, 0)
        
        state = replace(state, 
                        punch_state=jnp.array([s0, s1]), 
                        punch_arm=jnp.array([a0, a1]), 
                        has_hit=jnp.array([h0, h1]),
                        punch_cooldown=jnp.array([c0, c1]))
        
        # 5. Hit Detection & Scoring & Knockback
        def check_hit(attacker_idx, defender_idx, s):
            a_pos = s.pos[attacker_idx]
            d_pos = s.pos[defender_idx]
            dist = jnp.linalg.norm(a_pos - d_pos)
            
            p_state = s.punch_state[attacker_idx]
            not_hit_yet = jnp.logical_not(s.has_hit[attacker_idx])
            d_not_stunned = s.stun_timer[defender_idx] == 0
            
            # Per description:
            # State 3: Almost fully extended -> Long Jab (1pt)
            # State 4: Fully extended -> Power Punch (2pt)
            
            # We only register a Jab at State 3 if we are in the jab range 
            # (and not too close, as that would be a Power Punch opportunity at State 4)
            is_jab = jnp.logical_and(p_state == 3, 
                                     jnp.logical_and(dist < self.consts.JAB_DIST, dist >= self.consts.POWER_DIST))
            
            # We only register a Power Punch at State 4 if we are in power range
            is_power = jnp.logical_and(p_state == 4, dist < self.consts.POWER_DIST)
            
            valid_hit = jnp.logical_and(jnp.logical_or(is_jab, is_power), 
                                        jnp.logical_and(not_hit_yet, d_not_stunned))
            
            points = jnp.where(valid_hit, jnp.where(is_power, 2, 1), 0)
            
            return valid_hit, points

        # P1 hits P2
        p1_hit, p1_points = check_hit(0, 1, state)
        # P2 hits P1
        p2_hit, p2_points = check_hit(1, 0, state)
        
        # Apply hits
        new_scores = state.score + jnp.array([p1_points, p2_points])
        new_has_hit = state.has_hit.at[0].set(jnp.logical_or(state.has_hit[0], p1_hit)).at[1].set(jnp.logical_or(state.has_hit[1], p2_hit))
        
        # Apply Stun and Knockback
        new_stun = jnp.maximum(state.stun_timer - 1, 0)
        new_stun = new_stun.at[1].set(jnp.where(p1_hit, self.consts.STUN_DURATION, new_stun[1]))
        new_stun = new_stun.at[0].set(jnp.where(p2_hit, self.consts.STUN_DURATION, new_stun[0]))
        
        # Knockback logic
        kb_dir_p2 = (state.pos[1] - state.pos[0]) / (jnp.linalg.norm(state.pos[1] - state.pos[0]) + 1e-5)
        kb_dir_p1 = (state.pos[0] - state.pos[1]) / (jnp.linalg.norm(state.pos[0] - state.pos[1]) + 1e-5)
        
        new_p2_pos_kb = state.pos[1] + jnp.where(p1_hit, kb_dir_p2 * self.consts.KNOCKBACK_DIST, 0.0)
        new_p1_pos_kb = state.pos[0] + jnp.where(p2_hit, kb_dir_p1 * self.consts.KNOCKBACK_DIST, 0.0)
        
        # Clamp Knockback (Juggling mechanic: if at ropes, pos stays same)
        new_p2_pos_kb = jnp.clip(new_p2_pos_kb, jnp.array([self.consts.XMIN, self.consts.YMIN]), jnp.array([self.consts.XMAX, self.consts.YMAX]))
        new_p1_pos_kb = jnp.clip(new_p1_pos_kb, jnp.array([self.consts.XMIN, self.consts.YMIN]), jnp.array([self.consts.XMAX, self.consts.YMAX]))
        
        state = replace(state, 
                        pos=jnp.stack([new_p1_pos_kb, new_p2_pos_kb]),
                        score=new_scores,
                        has_hit=new_has_hit,
                        stun_timer=new_stun,
                        timer=state.timer - 1,
                        key=key)
        
        # 6. Termination
        done = jnp.logical_or(jnp.any(state.score >= self.consts.MAX_SCORE), state.timer <= 0)
        state = replace(state, done=done)
        
        return self._get_observation(state), state, (p1_points - p2_points).astype(jnp.float32), done, self._get_info(state)

    def render(self, state: BoxingState) -> jnp.ndarray:
        return self.renderer.render(state)


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
        
        # Pre-collect masks for white
        self.white_masks = {
            "body": self.SHAPE_MASKS["white_idle"],
            "stunned": self.SHAPE_MASKS["white_stunned"],
            "arm_left": [self.SHAPE_MASKS[f"white_arm_left_{i}"] for i in range(4)],
            "arm_right": [self.SHAPE_MASKS[f"white_arm_right_{i}"] for i in range(4)],
        }
        # Pre-collect masks for black
        self.black_masks = {
            "body": self.SHAPE_MASKS["black_idle"],
            "stunned": self.SHAPE_MASKS["black_stunned"],
            "arm_left": [self.SHAPE_MASKS[f"black_arm_left_{i}"] for i in range(4)],
            "arm_right": [self.SHAPE_MASKS[f"black_arm_right_{i}"] for i in range(4)],
        }

    def _render_body(self, raster, pos, masks, is_stunned):
        m = jax.lax.cond(is_stunned, lambda: masks["stunned"], lambda: masks["body"])
        return self.jr.render_at(raster, pos[0], pos[1], m)

    def _render_arms(self, raster, pos, opp_pos, masks, p_state, arm_idx):
        # Map punch_state (1-4) to animation frame (0-3)
        anim_frame = jax.lax.switch(jnp.clip(p_state - 1, 0, 3),
                                    [lambda: 0, lambda: 1, lambda: 2, lambda: 2])
        
        # Determine punch direction based on opponent's X
        is_punching_right = pos[0] < opp_pos[0]
        
        # Select arm mask set based on direction
        arm_masks = jax.lax.cond(is_punching_right,
                                    lambda: masks["arm_right"],
                                    lambda: masks["arm_left"])
        
        arm_mask = jax.lax.switch(anim_frame, [lambda: arm_masks[i] for i in range(4)])
        
        # Arm offsets
        arm_w = arm_mask.shape[1]
        x_offset = jax.lax.cond(is_punching_right,
                                lambda: 14,      # Punch right: arm starts after 14px body
                                lambda: -arm_w)  # Punch left: arm is to the left of x
        
        # Y offset: keep feet aligned (idle 47, arm 46)
        # and use arm_idx to distinguish between top/bottom arms
        y_offset = 1 + jax.lax.cond(arm_idx == 0, lambda: 0, lambda: 4)
        
        return self.jr.render_at(raster, pos[0] + x_offset, pos[1] + y_offset, arm_mask)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BoxingState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        def render_boxer(raster, idx, masks):
            p_pos = state.pos[idx].astype(jnp.int32)
            opp_pos = state.pos[1 - idx].astype(jnp.int32)
            is_stunned = state.stun_timer[idx] > 0
            p_state = state.punch_state[idx]
            arm_idx = state.punch_arm[idx]
            
            # Render body
            raster = self._render_body(raster, p_pos, masks, is_stunned)
            
            # Render arms if punching
            raster = jax.lax.cond(p_state > 0,
                                  lambda r: self._render_arms(r, p_pos, opp_pos, masks, p_state, arm_idx),
                                  lambda r: r,
                                  raster)
            return raster

        raster = render_boxer(raster, 0, self.white_masks)
        raster = render_boxer(raster, 1, self.black_masks)
        
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

        return self.jr.render_from_palette(raster, self.PALETTE)
