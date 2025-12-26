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
    # Player position & motion
    player_x: chex.Array   # scalar, e.g. jnp.array(10.0)
    player_y: chex.Array   # scalar
    player_vx: chex.Array  # scalar
    player_vy: chex.Array  # scalar
    on_ground: chex.Array  # bool scalar

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

    initial_time: int = 2000    # Pitfall timer
    max_lives: int = 3          # Pitfall lives
    ladder_x: int = 80
    ladder_width: int = 8

class PitfallObservation(NamedTuple):
    # for now, can just mirror some fields from state
    player_x: chex.Array
    player_y: chex.Array
    time_left: chex.Array
    lives_left: chex.Array

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

        self.renderer = PitfallRenderer(self.consts)

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
        ladder_x = jnp.asarray(consts.ladder_x, dtype=jnp.int32)
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

        # --- Horizontal movement ---
        speed = jnp.asarray(consts.player_speed, dtype=jnp.float32)
        vx = jnp.where(move_left, -speed, jnp.where(move_right, speed, 0.0))
        vx = jnp.where(state.on_ladder, 0.0, vx)  # disable horizontal movement on ladder

        # --- Setup ladder detection (used for both jump and ground clamping) ---
        ladder_x = jnp.asarray(consts.ladder_x, dtype=jnp.int32)
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

        # --- Clamp to ground & reset velocity on landing ---
        previous_ground = state.current_ground_y
        clamp_mask = ~state.on_ladder  # only clamp if NOT on ladder

        # --- Check for ground collision with fallthrough detection ---
        # On UPPER ground → ladder behaves as a hole (no ground under ladder).
        raw_on_ground_upper = (y >= previous_ground) & (~over_ladder)

        # Check if falling through to lower ground
        falling_to_lower = on_upper_level & over_ladder & (y >= lower_ground)

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

        # --- Time & done ---   
        time_left = time_left - 1
        done = time_left <= 0

        # Build new state
        new_state = PitfallState(
            player_x=x,
            player_y=y,
            player_vx=vx,
            player_vy=vy,
            on_ground=on_ground,
            time_left=time_left,
            lives_left=lives_left,
            done=done,
            down_pressed=down_pressed,
            on_ladder=on_ladder,
            current_ground_y=current_ground_y,
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
            lives_left=state.lives_left
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
        return state.time_left <= 0

    def action_space(self) -> spaces.Discrete:
        # 0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP for example
        return spaces.Discrete(4)

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
            time_left=jnp.array(consts.initial_time, dtype=jnp.int32),
            lives_left=jnp.array(consts.max_lives, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_),
            down_pressed=jnp.array(False, dtype=jnp.bool_),
            current_ground_y=jnp.array(consts.ground_y, dtype=jnp.float32),
            on_ladder=jnp.array(False, dtype=jnp.bool_),
        )
        return state
    

##debug renderer
class PitfallRenderer(JAXGameRenderer):
    """Very simple renderer: black background, green ground, white player block."""

    def __init__(self, consts: PitfallConstants | None = None):
        super().__init__()
        self.consts = consts or PitfallConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PitfallState) -> jnp.ndarray:
        h = self.consts.screen_height
        w = self.consts.screen_width

        frame = jnp.zeros((h, w, 3), dtype=jnp.uint8)

        # ----- static ground line -----
        ground = self.consts.ground_y
        underground = self.consts.underground_y
        frame = frame.at[ground:ground + 2, :, 1].set(200)  # green band
        frame = frame.at[underground:underground + 2, :, 1].set(120)

        ladder_x = self.consts.ladder_x
        ladder_w = self.consts.ladder_width

        ladder_top = ground - 1
        ladder_bottom = underground + 1

        frame = frame.at[ladder_top:ladder_bottom,
                     ladder_x:ladder_x + ladder_w,
                     2].set(255)

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
