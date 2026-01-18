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


SEED = 0xC4


def pit_code_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bits 3..5 (uint8 0..7)."""
    return (room_byte.astype(jnp.uint8) >> jnp.uint8(3)) & jnp.uint8(0x7)


def obj_code_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bits 0..2 (uint8 0..7)."""
    return room_byte.astype(jnp.uint8) & jnp.uint8(0x7)


def wall_side_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bit 7 (uint8 0..1). 0=left, 1=right."""
    return (room_byte.astype(jnp.uint8) >> jnp.uint8(7)) & jnp.uint8(1)


def pit_type(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Pit type is bits 3..5 of the room byte (uint8 0..7)."""
    return pit_code_u8(room_byte)


def room_hazards_from_room_byte(room_byte: jnp.ndarray) -> tuple[
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
]:
    """Decode hazards/objects from the original Pitfall room byte.

    Mapping (bits 0..2) applies for non-croc (pit!=100b) and non-treasure (pit!=101b) rooms:
      000: 1 rolling log
      001: 2 rolling logs
      010: 2 rolling logs
      011: 3 rolling logs
      100: 1 stationary log
      101: 3 stationary logs
      110: fire
      111: snake

    Overrides:
      pit==100 (croc room): no objects/hazards
      pit==101 (treasure+tar): no hazards for now

    """
    rb = room_byte.astype(jnp.uint8)
    pit = pit_code_u8(rb)
    obj = obj_code_u8(rb)

    is_croc_room = pit == jnp.uint8(0b100)
    is_treasure_tar = pit == jnp.uint8(0b101)
    suppress = is_croc_room | is_treasure_tar

    log_count_u8 = jnp.where(
        obj == jnp.uint8(0),
        jnp.uint8(1),
        jnp.where(
            (obj == jnp.uint8(1)) | (obj == jnp.uint8(2)),
            jnp.uint8(2),
            jnp.where(
                obj == jnp.uint8(3),
                jnp.uint8(3),
                jnp.where(
                    obj == jnp.uint8(4),
                    jnp.uint8(1),
                    jnp.where(obj == jnp.uint8(5), jnp.uint8(3), jnp.uint8(0)),
                ),
            ),
        ),
    )

    has_logs = (log_count_u8 > jnp.uint8(0)) & (~suppress)
    logs_are_rolling = (obj <= jnp.uint8(3)) & has_logs
    log_count = has_logs.astype(jnp.int32) * log_count_u8.astype(jnp.int32)

    log_xs_1 = jnp.array([130, 0, 0], dtype=jnp.int32)
    log_xs_2 = jnp.array([40, 120, 0], dtype=jnp.int32)
    log_xs_3 = jnp.array([16, 90, 130], dtype=jnp.int32)

    log_xs = jnp.where(
        log_count == jnp.int32(1),
        log_xs_1,
        jnp.where(log_count == jnp.int32(2), log_xs_2, jnp.where(log_count == jnp.int32(3), log_xs_3, jnp.zeros((3,), dtype=jnp.int32))),
    )

    has_fire = (obj == jnp.uint8(0b110)) & (~suppress)
    has_snake = (obj == jnp.uint8(0b111)) & (~suppress)

    return has_logs, logs_are_rolling, log_count, log_xs, has_fire, has_snake


def debug_room_byte(room_byte: int) -> str:
    """Small non-JAX helper for quick printing in scripts."""
    rb = room_byte & 0xFF
    pit = (rb >> 3) & 0x7
    obj = rb & 0x7
    wall = (rb >> 7) & 0x1
    return f"room_byte=0x{rb:02X} pit={pit} obj={obj} wall_side={wall}"


def lfsr_right_u8(b: jnp.ndarray) -> jnp.ndarray:
    """Moving right: shift left; bit0 = XOR(bit3, bit4, bit5, bit7)."""
    b = b.astype(jnp.uint8)
    bit = ((b >> 3) ^ (b >> 4) ^ (b >> 5) ^ (b >> 7)) & jnp.uint8(1)
    return jnp.uint8(((b << 1) & jnp.uint8(0xFF)) | bit)


def lfsr_left_u8(b: jnp.ndarray) -> jnp.ndarray:
    """Moving left: shift right; bit7 = XOR(bit4, bit5, bit6, bit0)."""
    b = b.astype(jnp.uint8)
    bit = ((b >> 4) ^ (b >> 5) ^ (b >> 6) ^ b) & jnp.uint8(1)  # b includes bit0
    return jnp.uint8((b >> 1) | (bit << 7))


def step_lfsr(b: jnp.ndarray, fn, n_steps: jnp.ndarray) -> jnp.ndarray:
    """Apply an LFSR step function n_steps times (n_steps is typically 1 or 3)."""

    def body(_, bb):
        return fn(bb)

    return lax.fori_loop(0, n_steps.astype(jnp.int32), body, b)

class PitfallState(NamedTuple):
    screen_id: chex.Array
    room_byte: chex.Array

    player_x: chex.Array
    player_y: chex.Array
    player_vx: chex.Array
    player_vy: chex.Array
    on_ground: chex.Array
    score: chex.Array
    timer_started: chex.Array

    time_left: chex.Array
    lives_left: chex.Array
    done: chex.Array
    hurt_cooldown: chex.Array

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

    player_speed: float = 3.0  # pixels per frame horizontally
    jump_velocity: float = -7.8  # initial upward velocity
    gravity: float = 1.0       # downward accel each frame

    fps: int = 30
    initial_time_seconds: int = 1200  # 20 minutes
    max_lives: int = 3          # Pitfall lives
    ladder_x: int = 80
    ladder_width: int = 10
    initial_score: int = 2000
    tunnel_wall_width: int = 8

    # Side holes beside ladder (underground)
    hole_width: int = 20            # px
    hole_gap_from_ladder: int = 20   # px gap from ladder edge

    # Stationary wood logs (upper ground hazard)
    wood_drain_per_frame: int = 2  # score points drained each frame while touching any log
    wood_w: int = 10              # log diameter in px (circle radius ~ wood_w//2)
    wood_h: int = 10               # log diameter in px
    wood_y_offset: int = 0         # fine-tune vertical placement relative to ground

    # Fireplace hazard (upper ground)
    fire_w: int = 7
    fire_h: int = 7
    fire_y_offset: int = 0
    fire_hurt_cooldown_frames: int = 30  # ~1s at 30fps
    fire_respawn_y_offset: int = 20      # respawn above ground so gravity drops player

    # Snake hazard (upper ground)
    snake_w: int = 12
    snake_h: int = 6
    snake_hurt_cooldown_frames: int = 30



class PitfallObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    time_left: chex.Array
    lives_left: chex.Array
    score: chex.Array
    timer_started: chex.Array

class PitfallInfo(NamedTuple):
    time_left: chex.Array
    lives_left: chex.Array

class ScreenLayout(NamedTuple):
    has_ladder: chex.Array
    ladder_x: chex.Array
    has_wall: chex.Array
    wall_x: chex.Array
    wall_side: chex.Array

class JaxPitfall(JaxEnvironment[PitfallState, PitfallObservation, PitfallInfo, PitfallConstants]):
    def __init__(self, consts: PitfallConstants | None = None):
        if consts is None:
            consts = PitfallConstants()
        super().__init__(consts)
        self.consts = consts
        self.num_screens = 255
        W = self.consts.screen_width
        WW = self.consts.tunnel_wall_width

        LEFT_WALL_X = 11
        RIGHT_WALL_X = 144

        def clamp_wall_x(x: int) -> int:
            return max(0, min(W - WW, x))

        self.left_wall_x_px = jnp.array(clamp_wall_x(LEFT_WALL_X), dtype=jnp.int32)
        self.right_wall_x_px = jnp.array(clamp_wall_x(RIGHT_WALL_X), dtype=jnp.int32)

        ladder_x_px = int(round(140 * W / 300.0))
        ladder_x_px = max(0, min(W - consts.ladder_width, ladder_x_px))
        self.ladder_x_px = jnp.array(ladder_x_px, dtype=jnp.int32)

        self.renderer = PitfallRenderer(
            self.consts,
            self.ladder_x_px,
            self.left_wall_x_px,
            self.right_wall_x_px,
        )


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
        room_byte: chex.Array,
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
        layout = self._screen_layout(room_byte)
        ladder_x = layout.ladder_x
        has_ladder = layout.has_ladder
        ladder_w = jnp.asarray(consts.ladder_width, dtype=jnp.int32)
        player_w = jnp.asarray(4, dtype=jnp.int32)

        x_int = x.astype(jnp.int32)
        player_right = x_int + player_w
        ladder_right = ladder_x + ladder_w

        overlap_left = player_right > ladder_x
        overlap_right = x_int < ladder_right
        near_ladder = has_ladder & overlap_left & overlap_right

        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)

        on_upper = current_ground_y == upper_ground
        on_lower = current_ground_y == lower_ground

        enter_from_upper = (
            on_ground & on_upper & near_ladder & (down_pressed | move_jump)
        )
        enter_from_lower = (
            on_ground & on_lower & near_ladder & move_jump
        )

        entering_ladder = (~state.on_ladder) & (enter_from_upper | enter_from_lower)

        climb_speed = jnp.asarray(1.5, dtype=jnp.float32)

        ladder_vertical = (y >= upper_ground) & (y <= lower_ground)
        still_on_ladder = state.on_ladder & near_ladder & ladder_vertical

        on_ladder_now = entering_ladder | still_on_ladder

        climb_delta = jnp.where(
            move_jump,          # UP
            -climb_speed,
            jnp.where(
                down_pressed,   # DOWN
                climb_speed,
                0.0,
            ),
        )

        y_climb = y + climb_delta

        y_climb = jnp.clip(y_climb, upper_ground, lower_ground)

        y = jnp.where(on_ladder_now, y_climb, y)
        vy = jnp.where(on_ladder_now, 0.0, vy)
        on_ground = jnp.where(
            on_ladder_now,
            jnp.array(False, dtype=jnp.bool_),
            on_ground,   # keep whatever step computed if not on ladder
        )

        at_top = jnp.abs(y - upper_ground) <= 2.0
        at_bottom = jnp.abs(y - lower_ground) <= 2.0

        horiz_exit = move_left | move_right
        exit_top = on_ladder_now & at_top & horiz_exit

        exit_bottom = on_ladder_now & at_bottom & down_pressed

        exiting = exit_top | exit_bottom

        exit_dir = jnp.where(
            move_left,
            jnp.array(-1.0, dtype=jnp.float32),
            jnp.where(
                move_right,
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
            ),
        )

        exit_step = jnp.asarray(8.0, dtype=jnp.float32)
        exit_dx = exit_dir * exit_step

        hop_height = jnp.asarray(2.0, dtype=jnp.float32)
        jump_v = jnp.asarray(consts.jump_velocity, dtype=jnp.float32)

        x = jnp.where(exit_top, x + exit_dx, x)
        y = jnp.where(exit_top, upper_ground - hop_height, y)
        vy = jnp.where(exit_top, jump_v, vy)
        on_ground = jnp.where(
            exit_top,
            jnp.array(False, dtype=jnp.bool_),
            on_ground,
        )

        max_x = jnp.asarray(consts.screen_width - player_w, dtype=jnp.float32)
        x = jnp.clip(x, 0.0, max_x)

        new_ground_y = jnp.where(
            exit_top,
            upper_ground,
            jnp.where(exit_bottom, lower_ground, current_ground_y),
        )

        y = jnp.where(exit_bottom, lower_ground, y)
        on_ground = jnp.where(
            exit_bottom,
            jnp.array(True, dtype=jnp.bool_),
            on_ground,
        )

        on_ladder = on_ladder_now & (~exiting)
        current_ground_y = jnp.where(exiting, new_ground_y, current_ground_y)

        return x, y, vy, on_ground, on_ladder, current_ground_y

    def _screen_layout(self, room_byte: chex.Array) -> ScreenLayout:
        rb = room_byte.astype(jnp.uint8)
        pt = pit_type(rb)

        has_ladder = (pt == jnp.uint8(0)) | (pt == jnp.uint8(1))

        wall_side_bit = (rb >> jnp.uint8(7)) & jnp.uint8(1)
        wall_side = jnp.where(
            has_ladder,
            jnp.where(wall_side_bit == jnp.uint8(1), jnp.int32(1), jnp.int32(-1)),
            jnp.int32(0),
        )
        wall_x = jnp.where(wall_side_bit == jnp.uint8(1), self.right_wall_x_px, self.left_wall_x_px)

        has_wall = has_ladder
        ladder_x = self.ladder_x_px

        return ScreenLayout(
            has_ladder=has_ladder,
            ladder_x=ladder_x,
            has_wall=has_wall,
            wall_x=wall_x,
            wall_side=wall_side,
        )

    def _side_hole_info(
        self,
        room_byte: chex.Array,
        player_center_x: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Two side holes beside the ladder (symmetric about ladder):
          left hole ends gap px left of ladder
          right hole starts gap px right of ladder

        Returns:
          has_side_hole (bool),
          over_side_hole (bool),
          left_x (int32),
          right_x (int32),
          hole_w (int32)
        """
        rb = room_byte.astype(jnp.uint8)
        pt = pit_type(rb)

        has_side_hole = pt == jnp.uint8(1)

        W = jnp.int32(self.consts.screen_width)
        ladder_x = self.ladder_x_px.astype(jnp.int32)
        ladder_w = jnp.int32(self.consts.ladder_width)

        hole_w = jnp.int32(self.consts.hole_width)
        gap = jnp.int32(self.consts.hole_gap_from_ladder)

        left_x = ladder_x - gap - hole_w
        right_x = ladder_x + ladder_w + gap

        max_start = jnp.maximum(W - hole_w, jnp.int32(0))
        left_x = jnp.clip(left_x, 0, max_start)
        right_x = jnp.clip(right_x, 0, max_start)

        in_left = (player_center_x >= left_x) & (player_center_x < (left_x + hole_w))
        in_right = (player_center_x >= right_x) & (player_center_x < (right_x + hole_w))

        over_side_hole = has_side_hole & (in_left | in_right)
        return has_side_hole, over_side_hole, left_x, right_x, hole_w

    def step(
        self,
        state: PitfallState,
        action: int,
    ) -> tuple[PitfallObservation, PitfallState, float, bool, PitfallInfo]:
        consts = self.consts

        action = jnp.asarray(action, dtype=jnp.int32)

        x = state.player_x
        y = state.player_y
        vx = state.player_vx
        vy = state.player_vy
        on_ground = state.on_ground
        time_left = state.time_left
        lives_left = state.lives_left
        hurt_cooldown = state.hurt_cooldown

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

        has_input = action != Action.NOOP
        timer_started = state.timer_started | has_input

        time_left = state.time_left - timer_started.astype(jnp.int32)
        time_left = jnp.maximum(time_left, 0)

        layout = self._screen_layout(state.room_byte)
        ladder_x = layout.ladder_x
        has_ladder = layout.has_ladder
        ladder_w = jnp.asarray(consts.ladder_width, dtype=jnp.int32)
        player_w = jnp.asarray(4, dtype=jnp.int32)

        x_int = x.astype(jnp.int32)
        player_right = x_int + player_w
        player_center = x_int + player_w // 2
        ladder_right = ladder_x + ladder_w

        overlap_left = player_right > ladder_x
        overlap_right = x_int < ladder_right
        near_ladder = has_ladder & overlap_left & overlap_right
        over_ladder = has_ladder & (player_center >= ladder_x) & (player_center < ladder_right)

        has_side_hole, over_side_hole, hole_left_x, hole_right_x, hole_w = self._side_hole_info(
            room_byte=state.room_byte,
            player_center_x=player_center.astype(jnp.int32),
        )

        over_any_hole = over_ladder | over_side_hole

        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)
        
        on_upper_level = state.current_ground_y == upper_ground
        on_lower_level = state.current_ground_y == lower_ground

        falling_through_hole = over_any_hole & on_upper_level & (~on_ground) & (~state.on_ladder) & (vy >= 0)

        speed = jnp.asarray(consts.player_speed, dtype=jnp.float32)
        vx = jnp.where(move_left, -speed, jnp.where(move_right, speed, 0.0))
        vx = jnp.where(state.on_ladder, 0.0, vx)
        vx = jnp.where(falling_through_hole, 0.0, vx)

        trying_to_enter_ladder = near_ladder & on_lower_level & move_jump
        jump_mask = on_ground & move_jump & (~state.on_ladder) & (~trying_to_enter_ladder)
        vy = jnp.where(
            jump_mask,
            jnp.asarray(consts.jump_velocity, dtype=jnp.float32),
            vy,
        )

        gravity = jnp.asarray(consts.gravity, dtype=jnp.float32)
        apply_gravity = (~on_ground) & (~state.on_ladder)
        vy = vy + gravity * apply_gravity.astype(jnp.float32)

        y = y + vy
        x = x + vx

        wall_w = jnp.int32(consts.tunnel_wall_width)
        player_w_i = jnp.int32(4)

        block = layout.has_wall & on_lower_level

        wall_left  = layout.wall_x
        wall_right = layout.wall_x + wall_w

        x = jnp.where(
            block & (layout.wall_side == jnp.int32(1)),
            jnp.minimum(x, (wall_left - player_w_i).astype(x.dtype)),
            x,
        )

        x = jnp.where(
            block & (layout.wall_side == jnp.int32(-1)),
            jnp.maximum(x, wall_right.astype(x.dtype)),
            x,
        )

        x_after_move = x

        screen_width_px = jnp.asarray(consts.screen_width, dtype=jnp.float32)
        player_width_px = jnp.asarray(4.0, dtype=jnp.float32)

        left_edge = 0.0
        right_edge_for_left_of_player = screen_width_px - player_width_px

        exited_left  = x_after_move < left_edge
        exited_right = x_after_move > right_edge_for_left_of_player

        on_lower_level_for_stride = state.current_ground_y == jnp.asarray(consts.underground_y, jnp.float32)
        stride = jnp.where(on_lower_level_for_stride, jnp.int32(3), jnp.int32(1))

        room_byte = state.room_byte
        room_byte = jnp.where(
            exited_right,
            step_lfsr(room_byte, lfsr_right_u8, stride),
            room_byte,
        )
        room_byte = jnp.where(
            exited_left,
            step_lfsr(room_byte, lfsr_left_u8, stride),
            room_byte,
        )

        screen_id = state.screen_id
        screen_id = jnp.where(exited_right, jnp.mod(screen_id + stride, jnp.int32(255)), screen_id)
        screen_id = jnp.where(exited_left, jnp.mod(screen_id - stride, jnp.int32(255)), screen_id)

        new_screen_id = screen_id
        new_room_byte = room_byte

        x_if_left_exit = right_edge_for_left_of_player
        x_if_right_exit = left_edge

        x = jnp.where(
            exited_left,
            x_if_left_exit,
            jnp.where(exited_right, x_if_right_exit, x_after_move)
        )

        player_w_f = jnp.asarray(4, dtype=jnp.float32)
        x = jnp.clip(x, 0.0, jnp.asarray(consts.screen_width, dtype=jnp.float32) - player_w_f)

        previous_ground = state.current_ground_y
        clamp_mask = ~state.on_ladder

        raw_on_ground_upper = (y >= previous_ground) & (~over_any_hole)

        falling_to_lower = on_upper_level & over_any_hole & (y >= lower_ground)

        score = state.score
        score = score + jnp.where(falling_to_lower, jnp.int32(-100), jnp.int32(0))
        score = jnp.maximum(score, jnp.int32(0))

        raw_on_ground_lower = (y >= previous_ground)

        raw_on_ground = jnp.where(
            on_upper_level,
            raw_on_ground_upper | falling_to_lower,
            raw_on_ground_lower,
        )

        current_ground_y = jnp.where(
            falling_to_lower,
            lower_ground,
            state.current_ground_y
        )

        on_ground = jnp.where(clamp_mask, raw_on_ground, on_ground)

        vy = jnp.where(
            clamp_mask & (on_ground & (vy > 0)),
            0.0,
            vy,
        )

        landing_y = jnp.where(falling_to_lower, lower_ground, previous_ground)
        y = jnp.where(clamp_mask & on_ground, landing_y, y)

        x, y, vy, on_ground, on_ladder, current_ground_y = self._apply_ladder(
            state=state,
            room_byte=new_room_byte,
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

        has_logs, logs_are_rolling, log_count, log_xs, has_fireplace, has_snake = room_hazards_from_room_byte(new_room_byte)

        total_frames = jnp.int32(self.consts.initial_time_seconds * self.consts.fps)
        frames_elapsed = jnp.maximum(total_frames - time_left, jnp.int32(0))
        frames_elapsed = frames_elapsed * timer_started.astype(jnp.int32)

        screen_w_i = jnp.int32(consts.screen_width)
        speed = jnp.int32(1)
        direction = jnp.int32(-1)
        dx = jnp.mod(frames_elapsed * speed * direction, screen_w_i)
        moving_centers = jnp.mod(log_xs + dx, screen_w_i)
        log_centers = jnp.where(logs_are_rolling, moving_centers, log_xs)

        player_w_i = jnp.int32(4)
        player_h_i = jnp.int32(8)
        x0 = x.astype(jnp.int32)
        x1 = x0 + player_w_i

        player_bottom = y.astype(jnp.int32)
        y1 = player_bottom + jnp.int32(1)
        y0 = player_bottom - player_h_i + jnp.int32(1)

        wood_w = jnp.int32(consts.wood_w)
        wood_h = jnp.int32(consts.wood_h)
        wood_top = jnp.int32(consts.ground_y - consts.wood_h + consts.wood_y_offset)
        wood_y0 = wood_top
        wood_y1 = wood_top + wood_h

        overlap_y = (y1 > wood_y0) & (y0 < wood_y1)
        active = jnp.arange(3, dtype=jnp.int32) < log_count
        W = jnp.int32(consts.screen_width)
        half_w = wood_w // jnp.int32(2)

        x_left_raw = log_centers.astype(jnp.int32) - half_w
        x_left = jnp.mod(x_left_raw, W)

        seg1_x0 = x_left
        seg1_x1 = jnp.minimum(x_left + wood_w, W)
        wraps = (x_left + wood_w) > W

        seg2_x0 = jnp.zeros_like(seg1_x0)
        seg2_x1 = (x_left + wood_w) - W

        overlap_seg1 = (x1 > seg1_x0) & (x0 < seg1_x1)
        overlap_seg2 = wraps & (x1 > seg2_x0) & (x0 < seg2_x1)
        overlap_x = overlap_seg1 | overlap_seg2

        touching_any = jnp.any(active & overlap_x & overlap_y)
        touching_wood = has_logs & touching_any

        drain = jnp.int32(consts.wood_drain_per_frame)
        score = jnp.where(touching_wood, score - drain, score)
        score = jnp.maximum(score, jnp.int32(0))

        fire_x_center = jnp.int32(132)

        fire_w = jnp.int32(consts.fire_w)
        fire_h = jnp.int32(consts.fire_h)
        fire_top = jnp.int32(consts.ground_y - consts.fire_h + consts.fire_y_offset)
        fire_y0 = fire_top
        fire_y1 = fire_top + fire_h

        screen_w_i = jnp.int32(consts.screen_width)
        max_fire_start = jnp.maximum(screen_w_i - fire_w, jnp.int32(0))
        fire_left = jnp.clip(fire_x_center - (fire_w // jnp.int32(2)), 0, max_fire_start)
        fire_right = fire_left + fire_w

        overlap_fire = (x1 > fire_left) & (x0 < fire_right) & (y1 > fire_y0) & (y0 < fire_y1)
        can_hurt = hurt_cooldown == jnp.int32(0)
        hit_fire = has_fireplace & overlap_fire & can_hurt

        snake_count = has_snake.astype(jnp.int32)
        snake_x_center = jnp.int32(134)

        snake_w = jnp.int32(consts.snake_w)
        snake_h = jnp.int32(consts.snake_h)
        snake_top = jnp.int32(consts.ground_y - consts.snake_h)
        snake_y0 = snake_top
        snake_y1 = snake_top + snake_h

        max_snake_start = jnp.maximum(screen_w_i - snake_w, jnp.int32(0))
        snake_left = jnp.clip(snake_x_center - (snake_w // jnp.int32(2)), 0, max_snake_start)
        snake_right = snake_left + snake_w

        overlap_snake = (x1 > snake_left) & (x0 < snake_right) & (y1 > snake_y0) & (y0 < snake_y1)
        hit_snake = has_snake & (snake_count > jnp.int32(0)) & overlap_snake & can_hurt

        hit_hazard = hit_fire | hit_snake

        lives_left = jnp.where(hit_hazard, lives_left - jnp.int32(1), lives_left)
        lives_left = jnp.maximum(lives_left, jnp.int32(0))

        respawn_x = jnp.asarray(consts.player_start_x, dtype=jnp.float32)
        respawn_y = jnp.asarray(consts.player_start_y - consts.fire_respawn_y_offset, dtype=jnp.float32)

        x = jnp.where(hit_hazard, respawn_x, x)
        y = jnp.where(hit_hazard, respawn_y, y)
        vx = jnp.where(hit_hazard, jnp.asarray(0.0, dtype=jnp.float32), vx)
        vy = jnp.where(hit_hazard, jnp.asarray(0.0, dtype=jnp.float32), vy)
        on_ground = jnp.where(hit_hazard, jnp.array(False, dtype=jnp.bool_), on_ground)
        on_ladder = jnp.where(hit_hazard, jnp.array(False, dtype=jnp.bool_), on_ladder)
        current_ground_y = jnp.where(
            hit_hazard,
            jnp.asarray(consts.ground_y, dtype=jnp.float32),
            current_ground_y,
        )

        next_hurt_cooldown = jnp.maximum(hurt_cooldown - jnp.int32(1), jnp.int32(0))
        next_hurt_cooldown = jnp.where(
            hit_hazard,
            jnp.int32(max(consts.fire_hurt_cooldown_frames, consts.snake_hurt_cooldown_frames)),
            next_hurt_cooldown,
        )

        done = (time_left <= 0) | (lives_left <= 0)

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
            hurt_cooldown=next_hurt_cooldown,
            down_pressed=down_pressed,
            on_ladder=on_ladder,
            current_ground_y=current_ground_y,
            screen_id=new_screen_id,
            room_byte=new_room_byte,
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
        return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PitfallState) -> bool:
        return (state.time_left <= 0) | (state.lives_left <= 0)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)

    def observation_space(self) -> spaces.Dict:
        raise NotImplementedError

    def image_space(self) -> spaces.Box:
        raise NotImplementedError

    def render(self, state: PitfallState) -> jnp.ndarray:
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: PitfallObservation) -> jnp.ndarray:
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
            hurt_cooldown=jnp.array(0, dtype=jnp.int32),
            down_pressed=jnp.array(False, dtype=jnp.bool_),
            on_ladder=jnp.array(False, dtype=jnp.bool_),
            current_ground_y=jnp.array(consts.ground_y, dtype=jnp.float32),
            screen_id=jnp.array(0, dtype=jnp.int32),
            room_byte=jnp.array(SEED, dtype=jnp.uint8),
        )
        return state
    

class PitfallRenderer(JAXGameRenderer):
    """Very simple renderer: black background, green ground, white player block."""

    def __init__(
        self,
        consts,
        ladder_x_px,
        left_wall_x_px,
        right_wall_x_px,
    ):
        super().__init__()
        self.consts = consts or PitfallConstants()
        self.ladder_x_px = ladder_x_px
        self.left_wall_x_px = left_wall_x_px
        self.right_wall_x_px = right_wall_x_px

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PitfallState) -> jnp.ndarray:
        h = self.consts.screen_height
        w = self.consts.screen_width

        frame = jnp.zeros((h, w, 3), dtype=jnp.uint8)

        digit_font = jnp.array([
            [[1,1,1],
             [1,0,1],
             [1,0,1],
             [1,0,1],
             [1,1,1]],
            [[0,1,0],
             [1,1,0],
             [0,1,0],
             [0,1,0],
             [1,1,1]],
            [[1,1,1],
             [0,0,1],
             [1,1,1],
             [1,0,0],
             [1,1,1]],
            [[1,1,1],
             [0,0,1],
             [1,1,1],
             [0,0,1],
             [1,1,1]],
            [[1,0,1],
             [1,0,1],
             [1,1,1],
             [0,0,1],
             [0,0,1]],
            [[1,1,1],
             [1,0,0],
             [1,1,1],
             [0,0,1],
             [1,1,1]],
            [[1,1,1],
             [1,0,0],
             [1,1,1],
             [1,0,1],
             [1,1,1]],
            [[1,1,1],
             [0,0,1],
             [0,1,0],
             [1,0,0],
             [1,0,0]],
            [[1,1,1],
             [1,0,1],
             [1,1,1],
             [1,0,1],
             [1,1,1]],
            [[1,1,1],
             [1,0,1],
             [1,1,1],
             [0,0,1],
             [1,1,1]],
              [[1,1,1],
               [1,0,1],
               [1,1,1],
               [1,0,1],
               [1,0,1]],
              [[1,1,0],
               [1,0,1],
               [1,1,0],
               [1,0,1],
               [1,1,0]],
              [[1,1,1],
               [1,0,0],
               [1,0,0],
               [1,0,0],
               [1,1,1]],
              [[1,1,0],
               [1,0,1],
               [1,0,1],
               [1,0,1],
               [1,1,0]],
              [[1,1,1],
               [1,0,0],
               [1,1,1],
               [1,0,0],
               [1,1,1]],
              [[1,1,1],
               [1,0,0],
               [1,1,1],
               [1,0,0],
               [1,0,0]],
        ], dtype=jnp.uint8)

        def int_to_digits(value: jnp.ndarray, width: int) -> jnp.ndarray:
            value = jnp.maximum(value, 0)
            digits = []
            v = value
            for _ in range(width):
                digits.append(v % 10)
                v = v // 10
            return jnp.stack(digits[::-1]).astype(jnp.int32)

        def u8_to_hex2(value_u8: jnp.ndarray) -> jnp.ndarray:
            v = value_u8.astype(jnp.uint8)
            hi = (v >> jnp.uint8(4)) & jnp.uint8(0xF)
            lo = v & jnp.uint8(0xF)
            return jnp.stack([hi.astype(jnp.int32), lo.astype(jnp.int32)])

        def draw_number(frame: jnp.ndarray, digits: jnp.ndarray, top: int, left: int, color: jnp.ndarray) -> jnp.ndarray:
            digit_w = 3
            digit_h = 5
            spacing = 1

            def draw_one(i, f):
                glyph = digit_font[digits[i]]  # (5,3)
                glyph_rgb = glyph[..., None] * color  # (5,3,3)
                return lax.dynamic_update_slice(f, glyph_rgb, (top, left + i * (digit_w + spacing), 0))

            return lax.fori_loop(0, digits.shape[0], draw_one, frame)

        ground = self.consts.ground_y
        underground = self.consts.underground_y
        frame = frame.at[ground:ground + 2, :, 1].set(200)
        frame = frame.at[underground:underground + 2, :, 1].set(120)

        rb = state.room_byte.astype(jnp.uint8)
        pt = pit_code_u8(rb)
        obj = obj_code_u8(rb)

        has_ladder = (pt == jnp.uint8(0)) | (pt == jnp.uint8(1))
        has_side_hole = pt == jnp.uint8(1)
        has_wall = has_ladder

        wall_side_bit = wall_side_u8(rb)
        wall_side = jnp.where(
            has_wall,
            jnp.where(wall_side_bit == jnp.uint8(1), jnp.int32(1), jnp.int32(-1)),
            jnp.int32(0),
        )
        wall_x = jnp.where(wall_side_bit == jnp.uint8(1), self.right_wall_x_px, self.left_wall_x_px)

        ladder_x = self.ladder_x_px

        ladder_top = jnp.asarray(ground - 1, dtype=jnp.int32)
        ladder_bottom = jnp.asarray(underground + 1, dtype=jnp.int32)

        ladder_h = ladder_bottom - ladder_top  # dynamic is OK as a value, but we need fixed shape
        ladder_h_static = int((self.consts.underground_y + 1) - (self.consts.ground_y - 1))

        ladder_patch = jnp.zeros((ladder_h_static, self.consts.ladder_width, 3), dtype=jnp.uint8)
        ladder_patch = ladder_patch.at[:, :, 2].set(255)

        frame = lax.cond(
            has_ladder,
            lambda f: lax.dynamic_update_slice(f, ladder_patch, (ladder_top, ladder_x, 0)),
            lambda f: f,
            frame,
        )

        ladder_w_static = int(self.consts.ladder_width)
        hole_w_static = int(self.consts.hole_width)
        gap_static = int(self.consts.hole_gap_from_ladder)

        W_static = int(self.consts.screen_width)
        max_start_static = max(0, W_static - hole_w_static)

        left_x = jnp.clip(
            ladder_x - jnp.int32(gap_static) - jnp.int32(hole_w_static),
            0,
            jnp.int32(max_start_static),
        )
        right_x = jnp.clip(
            ladder_x + jnp.int32(ladder_w_static) + jnp.int32(gap_static),
            0,
            jnp.int32(max_start_static),
        )

        ground_band_h = 2
        ground_top = jnp.asarray(ground, dtype=jnp.int32)
        ladder_clear = jnp.zeros((ground_band_h, ladder_w_static, 3), dtype=jnp.uint8)
        hole_clear = jnp.zeros((ground_band_h, hole_w_static, 3), dtype=jnp.uint8)

        frame = lax.cond(
            has_ladder,
            lambda f: lax.dynamic_update_slice(f, ladder_clear, (ground_top, ladder_x, 0)),
            lambda f: f,
            frame,
        )

        def clear_side_openings(f):
            f = lax.dynamic_update_slice(f, hole_clear, (ground_top, left_x, 0))
            f = lax.dynamic_update_slice(f, hole_clear, (ground_top, right_x, 0))
            return f

        frame = lax.cond(has_side_hole, clear_side_openings, lambda f: f, frame)

        hole_top_py = int(self.consts.ground_y)
        hole_bottom_py = int(self.consts.underground_y)
        hole_h_static = max(1, hole_bottom_py - hole_top_py)
        hole_top = jnp.int32(hole_top_py)

        hole_patch = jnp.zeros((hole_h_static, hole_w_static, 3), dtype=jnp.uint8)

        def draw_side_holes(f):
            f = lax.dynamic_update_slice(f, hole_patch, (hole_top, left_x, 0))
            f = lax.dynamic_update_slice(f, hole_patch, (hole_top, right_x, 0))
            return f

        frame = lax.cond(has_side_hole, draw_side_holes, lambda f: f, frame)

        has_wood, logs_are_rolling, wood_count, wood_xs, has_fireplace, has_snake = room_hazards_from_room_byte(rb)

        total_frames = jnp.int32(self.consts.initial_time_seconds * self.consts.fps)
        frames_elapsed = jnp.maximum(total_frames - state.time_left.astype(jnp.int32), jnp.int32(0))
        frames_elapsed = frames_elapsed * state.timer_started.astype(jnp.int32)

        screen_w_i = jnp.int32(self.consts.screen_width)
        speed = jnp.int32(1)
        direction = jnp.int32(-1)
        dx = jnp.mod(frames_elapsed * speed * direction, screen_w_i)
        moving_centers = jnp.mod(wood_xs + dx, screen_w_i)
        log_centers = jnp.where(logs_are_rolling, moving_centers, wood_xs)

        wood_w_static = int(self.consts.wood_w)
        wood_h_static = int(self.consts.wood_h)
        wood_top_py = int(self.consts.ground_y - self.consts.wood_h + self.consts.wood_y_offset)
        wood_top = jnp.int32(wood_top_py)
        wood_color = jnp.array([110, 70, 25], dtype=jnp.uint8)

        y_idx = jnp.arange(wood_h_static, dtype=jnp.float32)[:, None]      # (H,1)
        x_idx = jnp.arange(W_static, dtype=jnp.float32)[None, :]           # (1,W)
        cy = (jnp.float32(wood_h_static) - 1.0) / 2.0
        r = (jnp.minimum(jnp.float32(wood_w_static), jnp.float32(wood_h_static)) - 1.0) / 2.0

        def draw_one_log_region(region: jnp.ndarray, center_x: jnp.ndarray) -> jnp.ndarray:
            cx = center_x.astype(jnp.float32)
            dx = jnp.abs(x_idx - cx)
            dx = jnp.minimum(dx, jnp.float32(W_static) - dx)
            dy = jnp.abs(y_idx - cy)
            mask = (dx * dx + dy * dy) <= (r * r)
            return jnp.where(mask[..., None], wood_color, region)

        def draw_wood_logs(f: jnp.ndarray) -> jnp.ndarray:
            region = lax.dynamic_slice(f, (wood_top, 0, 0), (wood_h_static, W_static, 3))

            def apply_logs(i, reg):
                active_i = jnp.int32(i) < wood_count
                cx = log_centers[i]
                return lax.cond(
                    active_i,
                    lambda r_in: draw_one_log_region(r_in, cx),
                    lambda r_in: r_in,
                    reg,
                )

            region = lax.fori_loop(0, 3, apply_logs, region)
            return lax.dynamic_update_slice(f, region, (wood_top, 0, 0))

        frame = lax.cond(has_wood, draw_wood_logs, lambda f: f, frame)

        fire_x_center = jnp.int32(132)

        fire_w_static = int(self.consts.fire_w)
        fire_h_static = int(self.consts.fire_h)
        fire_top_py = int(self.consts.ground_y - self.consts.fire_h + self.consts.fire_y_offset)
        fire_top = jnp.int32(fire_top_py)

        max_fire_start_static = max(0, W_static - fire_w_static)
        half_fire_w_static = fire_w_static // 2
        fire_left = jnp.clip(
            fire_x_center - jnp.int32(half_fire_w_static),
            0,
            jnp.int32(max_fire_start_static),
        )

        fire_patch = jnp.zeros((fire_h_static, fire_w_static, 3), dtype=jnp.uint8)
        fire_patch = fire_patch.at[:, :, 0].set(255)
        fire_patch = fire_patch.at[:, :, 1].set(120)

        frame = lax.cond(
            has_fireplace,
            lambda f: lax.dynamic_update_slice(f, fire_patch, (fire_top, fire_left, 0)),
            lambda f: f,
            frame,
        )

        snake_count = has_snake.astype(jnp.int32)
        snake_x_center = jnp.int32(134)

        snake_w_static = int(self.consts.snake_w)
        snake_h_static = int(self.consts.snake_h)
        snake_top_py = int(self.consts.ground_y - self.consts.snake_h)
        snake_top = jnp.int32(snake_top_py)

        max_snake_start_static = max(0, W_static - snake_w_static)
        half_snake_w_static = snake_w_static // 2
        snake_left = jnp.clip(
            snake_x_center - jnp.int32(half_snake_w_static),
            0,
            jnp.int32(max_snake_start_static),
        )

        snake_patch = jnp.zeros((snake_h_static, snake_w_static, 3), dtype=jnp.uint8)
        snake_patch = snake_patch.at[:, :, 1].set(200)
        snake_patch = snake_patch.at[:, :, 0].set(20)

        draw_snake = has_snake & (snake_count > jnp.int32(0))
        frame = lax.cond(
            draw_snake,
            lambda f: lax.dynamic_update_slice(f, snake_patch, (snake_top, snake_left, 0)),
            lambda f: f,
            frame,
        )

        # Wall uses static Python ints for shapes to avoid JAX concretization
        top_pad = 0
        bot_pad = 0
        wall_top_py = int(self.consts.ground_y + top_pad)
        wall_bottom_py = int(self.consts.underground_y - bot_pad)
        wall_h_static = max(0, wall_bottom_py - wall_top_py)
        wall_w_static = int(self.consts.tunnel_wall_width)

        wall_top = jnp.int32(wall_top_py)

        wall_patch = jnp.zeros((wall_h_static, wall_w_static, 3), dtype=jnp.uint8)
        wall_patch = wall_patch.at[:, :, 0].set(180)
        wall_patch = wall_patch.at[:, :, 1].set(40)

        frame = lax.cond(
            has_wall,
            lambda f: lax.dynamic_update_slice(f, wall_patch, (wall_top, wall_x, 0)),
            lambda f: f,
            frame,
        )

        score_row = 2
        timer_row = 9

        score_digits = int_to_digits(state.score.astype(jnp.int32), 4)
        lives_digits = int_to_digits(state.lives_left.astype(jnp.int32), 1)

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

        timer_x = 20
        
        frame = draw_number(frame, score_digits, score_row, timer_x, score_color)

        frame = draw_number(frame, lives_digits, timer_row, 4, lives_color)

        frame = draw_number(frame, mm_digits, timer_row, timer_x, time_color)
        colon_x = timer_x + 2 * 4
        frame = frame.at[timer_row + 1, colon_x, :].set(colon_color)
        frame = frame.at[timer_row + 3, colon_x, :].set(colon_color)
        frame = draw_number(frame, ss_digits, timer_row, colon_x + 2, time_color)

        screen_digits = int_to_digits(state.screen_id.astype(jnp.int32), 3)
        frame = draw_number(frame, screen_digits, score_row, 120, jnp.array([200, 200, 200], dtype=jnp.uint8))

        rb_hex = u8_to_hex2(rb)
        debug_color = jnp.array([200, 200, 200], dtype=jnp.uint8)
        frame = draw_number(frame, rb_hex, score_row, 90, debug_color)
        frame = draw_number(frame, pt.astype(jnp.int32)[None], timer_row, 90, debug_color)
        frame = draw_number(frame, obj.astype(jnp.int32)[None], timer_row, 98, debug_color)
        frame = draw_number(frame, wall_side_bit.astype(jnp.int32)[None], timer_row, 106, debug_color)

        player_w, player_h = 4, 8

        x = jnp.clip(state.player_x.astype(jnp.int32), 0, w - player_w)

        bottom = jnp.clip(state.player_y.astype(jnp.int32), 0, h - 1)

        top = jnp.clip(bottom - player_h + 1, 0, h - player_h)

        color = jnp.array([255, 255, 255], dtype=jnp.uint8)
        rect = jnp.ones((player_h, player_w, 3), dtype=jnp.uint8) * color

        frame = lax.dynamic_update_slice(frame, rect, (top, x, 0))
        def add_down_banner(f):
            return f.at[0:5, :, 0].set(255)

        frame = lax.cond(
            state.down_pressed,
            add_down_banner,
            lambda f: f,
            frame,
        )

        return frame
