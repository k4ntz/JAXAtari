import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

class PitfallState(NamedTuple):
    screen_id: chex.Array

    # Player position & motion
    player_x: chex.Array   # scalar, e.g. jnp.array(10.0)
    player_y: chex.Array   # scalar
    player_vx: chex.Array  # scalar
    player_vy: chex.Array  # scalar
    on_ground: chex.Array  # bool scalar
    score: chex.Array
    timer_started: chex.Array


    # Resources / game status
    time_left: chex.Array  # int scalar (e.g. 2000)
    lives_left: chex.Array # int scalar (e.g. 3, even if not used yet)
    done: chex.Array       # bool scalar

    down_pressed: chex.Array
    on_ladder: chex.Array
    current_ground_y: chex.Array 


class PitfallConstants(NamedTuple):
    screen_width: int = 160     # Atari 2600 horizontal resolution
    screen_height: int = 210   # Atari vertical resolution used in ALE
    ground_y: int = 130         # approximate ground line in pixels
    underground_y: int = 180
    player_start_x: int = 20    # where Harry starts (left side)
    player_start_y: int = 130  # same as ground_y (standing on ground)

    player_speed: float = 3.5  # pixels per frame horizontally
    jump_velocity: float = -7.8  # initial upward velocity
    gravity: float = 1.0       # downward accel each frame

    fps: int = 30
    initial_time_seconds: int = 1200  # 20 minutes
    max_lives: int = 3          # Pitfall lives
    ladder_x: int = 80
    ladder_width: int = 10
    initial_score: int = 2000



class PitfallObservation(NamedTuple):
    # for now, can just mirror some fields from state
    player_x: chex.Array
    player_y: chex.Array
    time_left: chex.Array
    lives_left: chex.Array
    score: chex.Array
    timer_started: chex.Array

class PitfallInfo(NamedTuple):
    # extra logging. minimal for now
    time_left: chex.Array
    lives_left: chex.Array

class JaxPitfall(JaxEnvironment[PitfallState, PitfallObservation, PitfallInfo, PitfallConstants]):
    def __init__(self, consts: PitfallConstants | None = None):
        # If no constants are passed, use defaults
        if consts is None:
            consts = PitfallConstants()
        # Call base class constructor
        super().__init__(consts)
        # Store constants on self (the env instance)
        self.consts = consts
        self.num_screens = 3
        self.ladder_xpos_by_screen = jnp.array([80, 40, 120], dtype=jnp.int32)
        self.renderer = PitfallRenderer(self.consts, self.ladder_xpos_by_screen)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key = None
    ) -> tuple[PitfallObservation, PitfallState]:
        state = self._init_state()
        obs = self._get_observation(state)
        return obs, state
    
    def _apply_ladder(
        self,
        state: PitfallState,
        x: chex.Array,
        y: chex.Array,
        vy: chex.Array,
        down_pressed: chex.Array,
        move_jump: chex.Array,
        move_left: chex.Array,
        move_right: chex.Array,
        on_ground: chex.Array,
        current_ground_y: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Handles ladder enter/stay/exit.

        Returns:
            x, y, vy, on_ground, on_ladder, current_ground_y
        """
        consts = self.consts

        # ----- ladder horizontal span -----
        ladder_x = self.ladder_xpos_by_screen[state.screen_id]
        ladder_w = jnp.asarray(consts.ladder_width, dtype=jnp.int32)
        player_w = jnp.asarray(4, dtype=jnp.int32)  # matches renderer

        x_int = x.astype(jnp.int32)
        player_right = x_int + player_w
        ladder_right = ladder_x + ladder_w

        overlap_left = player_right > ladder_x
        overlap_right = x_int < ladder_right
        near_ladder = overlap_left & overlap_right

        # ----- vertical range of ladder -----
        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)

        on_upper = current_ground_y == upper_ground
        on_lower = current_ground_y == lower_ground

        # ============================================================
        # 1) ENTER LADDER (from ground)
        # ============================================================
        # From upper ground: press DOWN or UP while on ladder x-span
        enter_from_upper = (
            on_ground & on_upper & near_ladder & (down_pressed | move_jump)
        )
        # From lower ground: press UP while on ladder x-span
        enter_from_lower = (
            on_ground & on_lower & near_ladder & move_jump
        )

        entering_ladder = (~state.on_ladder) & (enter_from_upper | enter_from_lower)

        climb_speed = jnp.asarray(1.5, dtype=jnp.float32)

        # ============================================================
        # 2) STAY on LADDER (only if still aligned)
        # ============================================================
        # y here is *post-gravity* y (from step)
        ladder_vertical = (y >= upper_ground) & (y <= lower_ground)
        still_on_ladder = state.on_ladder & near_ladder & ladder_vertical

        on_ladder_now = entering_ladder | still_on_ladder

        # vertical input for climbing
        climb_delta = jnp.where(
            move_jump,          # UP
            -climb_speed,
            jnp.where(
                down_pressed,   # DOWN
                climb_speed,
                0.0,
            ),
        )

        # candidate y if climbing
        y_climb = y + climb_delta

        # clamp climbing between upper and lower ground
        y_climb = jnp.clip(y_climb, upper_ground, lower_ground)

        # while on ladder → use y_climb, zero vy, not on_ground
        y = jnp.where(on_ladder_now, y_climb, y)
        vy = jnp.where(on_ladder_now, 0.0, vy)
        on_ground = jnp.where(
            on_ladder_now,
            jnp.array(False, dtype=jnp.bool_),
            on_ground,   # keep whatever step computed if not on ladder
        )

        # ============================================================
        # 3) EXIT LADDER at top / bottom
        # ============================================================
        # Use a small tolerance for "at top" check (within 2 pixels)
        at_top = jnp.abs(y - upper_ground) <= 2.0
        at_bottom = jnp.abs(y - lower_ground) <= 2.0

        # ---- TOP EXIT: press LEFT or RIGHT when at top ----
        horiz_exit = move_left | move_right
        exit_top = on_ladder_now & at_top & horiz_exit

        # ---- BOTTOM EXIT: press DOWN at bottom ----
        exit_bottom = on_ladder_now & at_bottom & down_pressed

        exiting = exit_top | exit_bottom

        # ---- Directional hop for top exit ----
        exit_dir = jnp.where(
            move_left,
            jnp.array(-1.0, dtype=jnp.float32),
            jnp.where(
                move_right,
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
            ),
        )

        # horizontal push distance for the hop
        exit_step = jnp.asarray(8.0, dtype=jnp.float32)
        exit_dx = exit_dir * exit_step

        # small vertical hop above ground line
        hop_height = jnp.asarray(2.0, dtype=jnp.float32)
        jump_v = jnp.asarray(consts.jump_velocity, dtype=jnp.float32)  # FIXED: standard jump

        # apply top-exit transforms: sideways hop + jump
        x = jnp.where(exit_top, x + exit_dx, x)
        y = jnp.where(exit_top, upper_ground - hop_height, y)
        vy = jnp.where(exit_top, jump_v, vy)
        on_ground = jnp.where(
            exit_top,
            jnp.array(False, dtype=jnp.bool_),
            on_ground,
        )

        # clip x so we don't go outside the screen when hopping
        max_x = jnp.asarray(consts.screen_width - player_w, dtype=jnp.float32)
        x = jnp.clip(x, 0.0, max_x)

        # new ground y after exit
        new_ground_y = jnp.where(
            exit_top,
            upper_ground,
            jnp.where(exit_bottom, lower_ground, current_ground_y),
        )

        # bottom exit: snap to lower ground and stand
        y = jnp.where(exit_bottom, lower_ground, y)
        on_ground = jnp.where(
            exit_bottom,
            jnp.array(True, dtype=jnp.bool_),
            on_ground,
        )

        on_ladder = on_ladder_now & (~exiting)
        current_ground_y = jnp.where(exiting, new_ground_y, current_ground_y)

        return x, y, vy, on_ground, on_ladder, current_ground_y


    def step(
        self,
        state: PitfallState,
        action: int,
    ) -> tuple[PitfallObservation, PitfallState, float, bool, PitfallInfo]:
        consts = self.consts

        # Action is already 0..17 (full Atari action space)
        action = jnp.asarray(action, dtype=jnp.int32)

        # unpack
        x = state.player_x
        y = state.player_y
        vx = state.player_vx
        vy = state.player_vy
        on_ground = state.on_ground
        time_left = state.time_left
        lives_left = state.lives_left

        # --- Action parsing ---
        down_action = (
            (action == Action.DOWN) |
            (action == Action.DOWNLEFT) |
            (action == Action.DOWNRIGHT) |
            (action == Action.DOWNFIRE) |
            (action == Action.DOWNLEFTFIRE) |
            (action == Action.DOWNRIGHTFIRE)
        )       
        down_pressed = down_action
        move_left = (
            (action == Action.LEFT) |
            (action == Action.UPLEFT) |
            (action == Action.DOWNLEFT) |
            (action == Action.LEFTFIRE) |
            (action == Action.UPLEFTFIRE) |
            (action == Action.DOWNLEFTFIRE)
        )
        move_right = (
            (action == Action.RIGHT) |
            (action == Action.UPRIGHT) |
            (action == Action.DOWNRIGHT) |
            (action == Action.RIGHTFIRE) |
            (action == Action.UPRIGHTFIRE) |
            (action == Action.DOWNRIGHTFIRE)
        )
        move_jump = (
            (action == Action.UP) |
            (action == Action.UPLEFT) |
            (action == Action.UPRIGHT) |
            (action == Action.UPFIRE) |
            (action == Action.UPLEFTFIRE) |
            (action == Action.UPRIGHTFIRE)
        )

        # --- Timer logic: first input starts timer ---
        has_input = action != Action.NOOP
        timer_started = state.timer_started | has_input

        # decrement time only after started, clamp to 0
        time_left = state.time_left - timer_started.astype(jnp.int32)
        time_left = jnp.maximum(time_left, 0)

        # --- Setup ladder detection ---
        ladder_x = self.ladder_xpos_by_screen[state.screen_id]
        ladder_w = jnp.asarray(consts.ladder_width, dtype=jnp.int32)
        player_w = jnp.asarray(4, dtype=jnp.int32)

        x_int = x.astype(jnp.int32)
        player_right = x_int + player_w
        player_center = x_int + player_w // 2
        ladder_right = ladder_x + ladder_w

        overlap_left = player_right > ladder_x
        overlap_right = x_int < ladder_right
        near_ladder = overlap_left & overlap_right
        over_ladder = (player_center >= ladder_x) & (player_center < ladder_right)

        # --- Ground level definitions ---
        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)
        
        on_upper_level = state.current_ground_y == upper_ground
        on_lower_level = state.current_ground_y == lower_ground

        # Detect falling through hole: over ladder, on upper level, not on ground, not on ladder, and falling down
        falling_through_hole = over_ladder & on_upper_level & (~on_ground) & (~state.on_ladder) & (vy >= 0)

        # --- Horizontal movement ---
        speed = jnp.asarray(consts.player_speed, dtype=jnp.float32)
        vx = jnp.where(move_left, -speed, jnp.where(move_right, speed, 0.0))
        vx = jnp.where(state.on_ladder, 0.0, vx)  # disable horizontal movement on ladder
        vx = jnp.where(falling_through_hole, 0.0, vx)  # disable horizontal movement when falling through hole

        # --- Jump logic (prevent jumping when trying to enter ladder from below) ---
        trying_to_enter_ladder = near_ladder & on_lower_level & move_jump
        jump_mask = on_ground & move_jump & (~state.on_ladder) & (~trying_to_enter_ladder)
        vy = jnp.where(
            jump_mask,
            jnp.asarray(consts.jump_velocity, dtype=jnp.float32),
            vy,
        )

        # --- Gravity (only when NOT on ground and NOT on ladder) ---
        gravity = jnp.asarray(consts.gravity, dtype=jnp.float32)
        apply_gravity = (~on_ground) & (~state.on_ladder)
        vy = vy + gravity * apply_gravity.astype(jnp.float32)

        # --- Integrate position ---
        y = y + vy
        x = x + vx

        # --- Screen transition (insert BEFORE clamping x) ---
        x_after_move = x

        screen_width_px = jnp.asarray(consts.screen_width, dtype=jnp.float32)
        player_width_px = jnp.asarray(4.0, dtype=jnp.float32)

        left_edge = 0.0
        right_edge_for_left_of_player = screen_width_px - player_width_px

        exited_left  = x_after_move < left_edge
        exited_right = x_after_move > right_edge_for_left_of_player

        current_screen = state.screen_id

        screen_if_left_exit  = current_screen - 1
        screen_if_right_exit = current_screen + 1

        new_screen_id = jnp.where(
            exited_left,
            screen_if_left_exit,
            jnp.where(exited_right, screen_if_right_exit, current_screen)
        )

        num_screens = jnp.asarray(self.num_screens, dtype=jnp.int32)
        new_screen_id = jnp.mod(new_screen_id, num_screens)

        x_if_left_exit  = right_edge_for_left_of_player   # appear at right edge
        x_if_right_exit = left_edge                       # appear at left edge

        x = jnp.where(
            exited_left,
            x_if_left_exit,
            jnp.where(exited_right, x_if_right_exit, x_after_move)
        )

        # --- Clamp x to screen bounds ---
        player_w_f = jnp.asarray(4, dtype=jnp.float32)
        x = jnp.clip(x, 0.0, jnp.asarray(consts.screen_width, dtype=jnp.float32) - player_w_f)

        # --- Clamp to ground & reset velocity on landing ---
        previous_ground = state.current_ground_y
        clamp_mask = ~state.on_ladder  # only clamp if NOT on ladder

        # --- Check for ground collision with fallthrough detection ---
        # On UPPER ground → ladder behaves as a hole (no ground under ladder).
        raw_on_ground_upper = (y >= previous_ground) & (~over_ladder)

        # Check if falling through to lower ground
        falling_to_lower = on_upper_level & over_ladder & (y >= lower_ground)

        # --- Score handling ---
        score = state.score
        score = score + jnp.where(falling_to_lower, jnp.int32(-100), jnp.int32(0))
        score = jnp.maximum(score, jnp.int32(0))

        # On LOWER ground → continuous ground (no hole).
        raw_on_ground_lower = (y >= previous_ground)

        raw_on_ground = jnp.where(
            on_upper_level,
            raw_on_ground_upper | falling_to_lower,  # Can land on upper OR fall through to lower
            raw_on_ground_lower,
        )

        # Update current_ground_y when falling through
        current_ground_y = jnp.where(
            falling_to_lower,
            lower_ground,
            state.current_ground_y
        )

        # Update on_ground only when we are allowed to clamp
        on_ground = jnp.where(clamp_mask, raw_on_ground, on_ground)

        # Zero vy only when landing (falling down onto ground)
        vy = jnp.where(
            clamp_mask & (on_ground & (vy > 0)),
            0.0,
            vy,
        )

        # Snap y to appropriate ground level when landing
        landing_y = jnp.where(falling_to_lower, lower_ground, previous_ground)
        y = jnp.where(clamp_mask & on_ground, landing_y, y)

        # --- Ladder mechanics ---
        x, y, vy, on_ground, on_ladder, current_ground_y = self._apply_ladder(
            state=state,
            x=x,
            y=y,
            vy=vy,
            down_pressed=down_pressed,
            move_jump=move_jump,
            move_left=move_left,
            move_right=move_right,
            on_ground=on_ground,
            current_ground_y=current_ground_y,
        )

        done = (time_left <= 0) | (lives_left <= 0)

        # Build new state
        new_state = PitfallState(
            player_x=x,
            player_y=y,
            player_vx=vx,
            player_vy=vy,
            on_ground=on_ground,
            score=score,
            timer_started=timer_started,
            time_left=time_left,
            lives_left=lives_left,
            done=done,
            down_pressed=down_pressed,
            on_ladder=on_ladder,
            current_ground_y=current_ground_y,
            screen_id=new_screen_id
        )

        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PitfallState) -> PitfallObservation:
        return PitfallObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            time_left=state.time_left,
            lives_left=state.lives_left,
            score=state.score,
            timer_started=state.timer_started,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PitfallState) -> PitfallInfo:
        return PitfallInfo(
            time_left=state.time_left,
            lives_left=state.lives_left
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: PitfallState, new: PitfallState) -> float:
        # for now, no reward
        return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PitfallState) -> bool:
        return (state.time_left <= 0) | (state.lives_left <= 0)

    def action_space(self) -> spaces.Discrete:
        # Full Atari action space: 0..17
        return spaces.Discrete(18)

    def observation_space(self) -> spaces.Dict:
        # later we can make this more precise
        raise NotImplementedError

    def image_space(self) -> spaces.Box:
        # later, when we hook up the renderer
        raise NotImplementedError

    def render(self, state: PitfallState) -> jnp.ndarray:
        # play.py will JIT this 
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: PitfallObservation) -> jnp.ndarray:
        # pack obs into 1D array later
        raise NotImplementedError

    def _init_state(self) -> PitfallState:
        consts = self.consts
        state = PitfallState(
            player_x=jnp.array(consts.player_start_x, dtype=jnp.float32),
            player_y=jnp.array(consts.player_start_y, dtype=jnp.float32),
            player_vx=jnp.array(0.0, dtype=jnp.float32),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            on_ground=jnp.array(True, dtype=jnp.bool_),
            score=jnp.array(consts.initial_score, dtype=jnp.int32),
            timer_started=jnp.array(False, dtype=jnp.bool_),
            time_left=jnp.array(consts.initial_time_seconds * consts.fps, dtype=jnp.int32),
            lives_left=jnp.array(consts.max_lives, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_),
            down_pressed=jnp.array(False, dtype=jnp.bool_),
            on_ladder=jnp.array(False, dtype=jnp.bool_),
            current_ground_y=jnp.array(consts.ground_y, dtype=jnp.float32),
            screen_id=jnp.array(0, dtype=jnp.int32),
        )
        return state
    

##debug renderer
class PitfallRenderer(JAXGameRenderer):
    """Very simple renderer: black background, green ground, white player block."""

    def __init__(self, consts: PitfallConstants | None = None, ladder_xpos_by_screen=None):
        super().__init__()
        self.consts = consts or PitfallConstants()
        self.ladder_xpos_by_screen = ladder_xpos_by_screen

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PitfallState) -> jnp.ndarray:
        h = self.consts.screen_height
        w = self.consts.screen_width

        frame = jnp.zeros((h, w, 3), dtype=jnp.uint8)

        # --- tiny 3x5 digit font (10, 5, 3) ---
        digit_font = jnp.array([
            # 0
            [[1,1,1],
             [1,0,1],
             [1,0,1],
             [1,0,1],
             [1,1,1]],
            # 1
            [[0,1,0],
             [1,1,0],
             [0,1,0],
             [0,1,0],
             [1,1,1]],
            # 2
            [[1,1,1],
             [0,0,1],
             [1,1,1],
             [1,0,0],
             [1,1,1]],
            # 3
            [[1,1,1],
             [0,0,1],
             [1,1,1],
             [0,0,1],
             [1,1,1]],
            # 4
            [[1,0,1],
             [1,0,1],
             [1,1,1],
             [0,0,1],
             [0,0,1]],
            # 5
            [[1,1,1],
             [1,0,0],
             [1,1,1],
             [0,0,1],
             [1,1,1]],
            # 6
            [[1,1,1],
             [1,0,0],
             [1,1,1],
             [1,0,1],
             [1,1,1]],
            # 7
            [[1,1,1],
             [0,0,1],
             [0,1,0],
             [1,0,0],
             [1,0,0]],
            # 8
            [[1,1,1],
             [1,0,1],
             [1,1,1],
             [1,0,1],
             [1,1,1]],
            # 9
            [[1,1,1],
             [1,0,1],
             [1,1,1],
             [0,0,1],
             [1,1,1]],
        ], dtype=jnp.uint8)

        def int_to_digits(value: jnp.ndarray, width: int) -> jnp.ndarray:
            value = jnp.maximum(value, 0)
            digits = []
            v = value
            for _ in range(width):
                digits.append(v % 10)
                v = v // 10
            return jnp.stack(digits[::-1]).astype(jnp.int32)

        def draw_number(frame: jnp.ndarray, digits: jnp.ndarray, top: int, left: int, color: jnp.ndarray) -> jnp.ndarray:
            digit_w = 3
            digit_h = 5
            spacing = 1

            def draw_one(i, f):
                glyph = digit_font[digits[i]]  # (5,3)
                glyph_rgb = glyph[..., None] * color  # (5,3,3)
                return lax.dynamic_update_slice(f, glyph_rgb, (top, left + i * (digit_w + spacing), 0))

            return lax.fori_loop(0, digits.shape[0], draw_one, frame)

        # ----- static ground line -----
        ground = self.consts.ground_y
        underground = self.consts.underground_y
        frame = frame.at[ground:ground + 2, :, 1].set(200)  # green band
        frame = frame.at[underground:underground + 2, :, 1].set(120)

        ladder_x = self.ladder_xpos_by_screen[state.screen_id].astype(jnp.int32)
        ladder_w = jnp.asarray(self.consts.ladder_width, dtype=jnp.int32)

        ladder_top = jnp.asarray(ground - 1, dtype=jnp.int32)
        ladder_bottom = jnp.asarray(underground + 1, dtype=jnp.int32)

        ladder_h = ladder_bottom - ladder_top  # dynamic is OK as a value, but we need fixed shape
        # BUT ladder_h here is actually constant in your game (since ground/underground are constants).
        # So compute it as a Python int for a static shape:
        ladder_h_static = int((self.consts.underground_y + 1) - (self.consts.ground_y - 1))

        # Build a ladder patch (all zeros, blue channel = 255)
        ladder_patch = jnp.zeros((ladder_h_static, self.consts.ladder_width, 3), dtype=jnp.uint8)
        ladder_patch = ladder_patch.at[:, :, 2].set(255)

        # Paste ladder patch at dynamic x position
        frame = lax.dynamic_update_slice(frame, ladder_patch, (ladder_top, ladder_x, 0))

        # ----- HUD -----
        # Layout: Score on top row (above timer), Lives + Timer on second row
        score_row = 2
        timer_row = 9  # below score (5 pixel height + 2 gap)

        score_digits = int_to_digits(state.score.astype(jnp.int32), 4)  # 4 digits, no leading zero for typical scores
        lives_digits = int_to_digits(state.lives_left.astype(jnp.int32), 1)

        # Convert frames to MM:SS display (separate minutes and seconds)
        fps = jnp.int32(self.consts.fps)
        seconds_left = jnp.maximum(state.time_left // fps, 0).astype(jnp.int32)
        mm = seconds_left // 60
        ss = seconds_left % 60
        mm_digits = int_to_digits(mm, 2)
        ss_digits = int_to_digits(ss, 2)

        score_color = jnp.array([40, 220, 40], dtype=jnp.uint8)
        lives_color = jnp.array([240, 200, 40], dtype=jnp.uint8)
        time_color = jnp.array([40, 200, 240], dtype=jnp.uint8)
        colon_color = jnp.array([40, 200, 240], dtype=jnp.uint8)

        # Timer: MM:SS with colon
        timer_x = 20
        
        # Score positioned above timer
        frame = draw_number(frame, score_digits, score_row, timer_x, score_color)

        # Lives on left of timer row
        frame = draw_number(frame, lives_digits, timer_row, 4, lives_color)

        frame = draw_number(frame, mm_digits, timer_row, timer_x, time_color)
        # Draw colon (two dots)
        colon_x = timer_x + 2 * 4  # after 2 digits (each digit 3 wide + 1 spacing)
        frame = frame.at[timer_row + 1, colon_x, :].set(colon_color)
        frame = frame.at[timer_row + 3, colon_x, :].set(colon_color)
        # Seconds after colon
        frame = draw_number(frame, ss_digits, timer_row, colon_x + 2, time_color)

        # ----- dynamic player rect -----
        player_w, player_h = 4, 8

        # x position (clamped)
        x = jnp.clip(state.player_x.astype(jnp.int32), 0, w - player_w)

        # treat state.player_y as feet (bottom)
        bottom = jnp.clip(state.player_y.astype(jnp.int32), 0, h - 1)

        # we want the rect fully inside the screen: choose a valid top
        top = jnp.clip(bottom - player_h + 1, 0, h - player_h)

        # build a (player_h, player_w, 3) white rectangle
        color = jnp.array([255, 255, 255], dtype=jnp.uint8)
        rect = jnp.ones((player_h, player_w, 3), dtype=jnp.uint8) * color

        # dynamic insert: indices (top, x, 0) can be JAX values
        frame = lax.dynamic_update_slice(frame, rect, (top, x, 0))
        def add_down_banner(f):
            return f.at[0:5, :, 0].set(255)  # red strip at top

        frame = lax.cond(
            state.down_pressed,
            add_down_banner,   # then branch
            lambda f: f,       # else branch (no change)
            frame,
        )

        return frame
