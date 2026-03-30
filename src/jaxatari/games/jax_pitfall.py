import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax
from typing import Tuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


SEED = 0xC4


HUD_FONT_16 = jnp.array(
    [
        # 0
        [[1, 1, 1],
         [1, 0, 1],
         [1, 0, 1],
         [1, 0, 1],
         [1, 1, 1]],
        # 1
        [[0, 1, 0],
         [1, 1, 0],
         [0, 1, 0],
         [0, 1, 0],
         [1, 1, 1]],
        # 2
        [[1, 1, 1],
         [0, 0, 1],
         [1, 1, 1],
         [1, 0, 0],
         [1, 1, 1]],
        # 3
        [[1, 1, 1],
         [0, 0, 1],
         [1, 1, 1],
         [0, 0, 1],
         [1, 1, 1]],
        # 4
        [[1, 0, 1],
         [1, 0, 1],
         [1, 1, 1],
         [0, 0, 1],
         [0, 0, 1]],
        # 5
        [[1, 1, 1],
         [1, 0, 0],
         [1, 1, 1],
         [0, 0, 1],
         [1, 1, 1]],
        # 6
        [[1, 1, 1],
         [1, 0, 0],
         [1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        # 7
        [[1, 1, 1],
         [0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [0, 0, 1]],
        # 8
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        # 9
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1],
         [0, 0, 1],
         [1, 1, 1]],
        # A
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1],
         [1, 0, 1],
         [1, 0, 1]],
        # B
        [[1, 1, 0],
         [1, 0, 1],
         [1, 1, 0],
         [1, 0, 1],
         [1, 1, 0]],
        # C
        [[1, 1, 1],
         [1, 0, 0],
         [1, 0, 0],
         [1, 0, 0],
         [1, 1, 1]],
        # D
        [[1, 1, 0],
         [1, 0, 1],
         [1, 0, 1],
         [1, 0, 1],
         [1, 1, 0]],
        # E
        [[1, 1, 1],
         [1, 0, 0],
         [1, 1, 1],
         [1, 0, 0],
         [1, 1, 1]],
        # F
        [[1, 1, 1],
         [1, 0, 0],
         [1, 1, 1],
         [1, 0, 0],
         [1, 0, 0]],
    ],
    dtype=jnp.uint8,
)


def _get_default_pitfall_asset_config() -> tuple:
    """Default declarative asset manifest for Pitfall."""
    return (
        {'name': 'background', 'type': 'background'},
        {
            'name': 'background_tree_variant_0',
            'type': 'single',
            'file': 'background_tree_variant_0.npy',
        },
        {
            'name': 'background_tree_variant_1',
            'type': 'single',
            'file': 'background_tree_variant_1.npy',
        },
        {
            'name': 'background_tree_variant_2',
            'type': 'single',
            'file': 'background_tree_variant_2.npy',
        },
        {
            'name': 'background_tree_variant_3',
            'type': 'single',
            'file': 'background_tree_variant_3.npy',
        },
        {
            'name': 'backdrop_crocodilepit_and_rope',
            'type': 'single',
            'file': 'backdrop_crocodilepit_and_rope.npy',
        },
        {
            'name': 'wall',
            'type': 'single',
            'file': 'wall.npy',
        },
        {
            'name': 'harry_idle',
            'type': 'group',
            'files': ['harryidle1.npy'],
        },
        {
            'name': 'harry_run',
            'type': 'group',
            'files': [
                'harryrunning1.npy',
                'harryrunning2.npy',
                'harryrunning3.npy',
                'harryrunning4.npy',
                'harryrunning5.npy',
            ],
        },
        {
            'name': 'harry_climb',
            'type': 'group',
            'files': ['harryclimb1.npy', 'harryclimb2.npy'],
        },
        {
            'name': 'harry_jump',
            'type': 'group',
            'files': ['harryjumping1.npy', 'harryjumping2.npy'],
        },
        {
            'name': 'scorpion_left',
            'type': 'group',
            'files': ['scorpion_left1_alpha.npy', 'scorpion_left2_alpha.npy'],
        },
        {
            'name': 'scorpion_right',
            'type': 'group',
            'files': ['scorpion_right1_alpha.npy', 'scorpion_right2_alpha.npy'],
        },
    )


def pit_code_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bits 3..5 (uint8 0..7)."""
    return (room_byte.astype(jnp.uint8) >> jnp.uint8(3)) & jnp.uint8(0x7)


def obj_code_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bits 0..2 (uint8 0..7)."""
    return room_byte.astype(jnp.uint8) & jnp.uint8(0x7)


def wall_side_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bit 7 (uint8 0..1). 0=left, 1=right."""
    return (room_byte.astype(jnp.uint8) >> jnp.uint8(7)) & jnp.uint8(1)


def tree_variant_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bits 6..7 (uint8 0..3)."""
    return (room_byte.astype(jnp.uint8) >> jnp.uint8(6)) & jnp.uint8(0x3)


def pit_type(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Pit type is bits 3..5 of the room byte (uint8 0..7)."""
    return pit_code_u8(room_byte)


def has_scorpion_from_room_byte(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Scorpion is present when pit type does not include a ladder (pit_type not in {0,1})."""
    pt = pit_code_u8(room_byte.astype(jnp.uint8))
    has_ladder = (pt == jnp.uint8(0)) | (pt == jnp.uint8(1))
    return ~has_ladder


def has_vine_from_room_byte(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Vine/rope presence from room-byte decode rules."""
    rb = room_byte.astype(jnp.uint8)
    pt = pit_code_u8(rb)
    obj = obj_code_u8(rb)

    is_croc_pit = pt == jnp.uint8(0b100)
    is_croc_vine_obj = (
        (obj == jnp.uint8(0b010)) |
        (obj == jnp.uint8(0b011)) |
        (obj == jnp.uint8(0b110)) |
        (obj == jnp.uint8(0b111))
    )

    is_shifting_tar = pt == jnp.uint8(0b110)
    return (is_croc_pit & is_croc_vine_obj) | is_shifting_tar


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

@struct.dataclass
class PitfallState:
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
    scorpion_x: chex.Array
    scorpion_anim_idx: chex.Array
    scorpion_anim_timer: chex.Array
    scorpion_facing_right: chex.Array
    touching_wood: chex.Array
    climb_active: chex.Array
    ladder_step_idx: chex.Array
    ladder_step_timer: chex.Array
    run_startup_timer: chex.Array
    run_anim_phase: chex.Array
    run_anim_timer: chex.Array
    facing_left: chex.Array
    log_push_remaining: chex.Array
    ladder_exit_frames: chex.Array
    respawn_phase: chex.Array
    respawn_timer: chex.Array
    respawn_target_x: chex.Array
    respawn_target_y: chex.Array
    respawn_target_ground_y: chex.Array

class PitfallConstants(struct.PyTreeNode):
    screen_width: int = 160     # Atari 2600 horizontal resolution
    screen_height: int = 210   # Atari vertical resolution used in ALE
    ground_y: int = 130         # approximate ground line in pixels
    underground_y: int = 180
    player_start_x: int = 20    # where Harry starts (left side)
    player_start_y: int = 130  # same as ground_y (standing on ground)

    player_speed: float = 3.0  # pixels per frame horizontally
    jump_velocity: float = -4.0  # softer launch for floatier arcs
    gravity: float = 0.55       # symmetric gravity for longer hang-time
    fall_speed: float = 3.0    # terminal velocity cap on descent

    fps: int = 30
    initial_time_seconds: int = 1200  # 20 minutes
    max_lives: int = 3          # Pitfall lives
    ladder_x: int = 80
    ladder_width: int = 16
    ladder_hole_fall_width: int = 10  # narrower than full ladder sprite for fall-through detection
    ladder_entry_width: int = 10  # require Harry to be centered over the opening before upper ladder entry
    ladder_top_peek_offset: int = 6  # keep most of Harry in the hole; face/shoulders emerge at top
    initial_score: int = 2000
    tunnel_wall_width: int = 8
    wall_contact_overlap: int = 0  # keep Harry fully outside tunnel walls on both sides
    right_wall_block_player_width: int = 9  # lets Harry stand closer when approaching a right-side wall

    # Side holes beside ladder (underground)
    hole_width: int = 12            # px (from ladder_with_pits sprite, cols 2-13 and 54-65)
    hole_gap_from_ladder: int = 12   # px floor bridge between ladder edge and hole

    # Stationary wood logs (upper ground hazard)
    wood_drain_per_frame: int = 2  # score points drained each frame while touching any log
    wood_w: int = 6               # log width in px (from log sprite)
    wood_h: int = 14              # log height in px (from log sprite)
    wood_y_offset: int = 0         # fine-tune vertical placement relative to ground
    wood_visual_contact_pad_x: int = 3  # start log interaction pose slightly before full overlap
    wood_visual_contact_shift_x: int = -3  # shift visual slide trigger slightly left

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

    # Scorpion hazard (underground)
    scorpion_spawn_x: int = 80
    scorpion_w: int = 12
    scorpion_h: int = 6
    scorpion_y_offset: int = 0
    scorpion_speed: float = 0.6
    scorpion_anim_period: int = 6
    scorpion_hurt_cooldown_frames: int = 30
    # Additional hitbox shrink (pixels) applied to the tight visible-pixel bbox.
    # Increasing these values makes scorpion deaths require more explicit contact.
    scorpion_hit_inset_x: int = 2
    scorpion_hit_inset_y: int = 2

    # Log pushback when climbing a ladder
    log_push_amount: float = 6.0    # total px to push Harry down on log hit
    log_push_speed: float = 2.0     # px/frame push rate (≈3 frames for full push)
    ladder_exit_grace_frames: int = 6  # preserve jump-out motion and block instant ladder recapture
    ladder_exit_jump_speed: float = 1.35  # tuned so the exit arc lands on the inner ground rims near each hole
    ladder_exit_initial_hop: float = 1.0  # minimal visual nudge to clear the ladder lip without doing the heavy lifting
    normal_jump_horizontal_scale: float = 0.7  # keep standard jumps to ~70% of run-speed carry
    death_pause_frames: int = 45  # ~1.5 seconds at 30 FPS
    respawn_drop_speed: float = 1.0  # gentle deterministic fall during respawn animation
    respawn_drop_spawn_y_offset: int = 20  # spawn this far above the upper ground line before drop-in
    underground_respawn_x: int = 20  # left opening area for underground re-entry
    underground_respawn_spawn_above_reveal: int = 0  # start at reveal edge (avoid visible drop-in)
    underground_respawn_reveal_from_ground: int = 17  # reveal boundary is this many px below ground
    underground_respawn_reveal_y_offset: int = 1  # clip reveal just below the upper ledge
    underground_respawn_wall_clearance: int = 2  # start just below the wall top rather than high above ground

    # Rendering tune: negative moves sprite up (player_y is treated as bottom/feet).
    harry_y_tune: int = -2
    underground_harry_y_tune: int = 2  # push Harry slightly down only on lower ground

    ASSET_CONFIG: tuple = _get_default_pitfall_asset_config()



@struct.dataclass
class PitfallObservation:
    player_x: chex.Array
    player_y: chex.Array
    screen_id: chex.Array
    room_byte: chex.Array
    current_ground_y: chex.Array
    on_ground: chex.Array
    on_ladder: chex.Array
    facing_left: chex.Array

    scorpion_x: chex.Array
    has_scorpion: chex.Array
    has_fire: chex.Array
    has_snake: chex.Array
    has_logs: chex.Array
    log_count: chex.Array
    log_xs: chex.Array
    logs_are_rolling: chex.Array

    has_ladder: chex.Array
    ladder_x: chex.Array
    has_wall: chex.Array
    wall_x: chex.Array
    wall_side: chex.Array

    time_left: chex.Array
    lives_left: chex.Array
    score: chex.Array

@struct.dataclass
class PitfallInfo:
    time_left: chex.Array
    lives_left: chex.Array

@struct.dataclass
class ScreenLayout:
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

        LEFT_WALL_X = 23
        RIGHT_WALL_X = 132

        def clamp_wall_x(x: int) -> int:
            return max(0, min(W - WW, x))

        self.left_wall_x_px = jnp.array(clamp_wall_x(LEFT_WALL_X), dtype=jnp.int32)
        self.right_wall_x_px = jnp.array(clamp_wall_x(RIGHT_WALL_X), dtype=jnp.int32)

        ladder_x_px = int(round(140 * W / 300.0))
        ladder_x_px = max(0, min(W - consts.ladder_width, ladder_x_px))
        self.ladder_x_px = jnp.array(ladder_x_px, dtype=jnp.int32)

        self.renderer = PitfallRenderer(
            consts=self.consts,
            ladder_x_px=self.ladder_x_px,
            left_wall_x_px=self.left_wall_x_px,
            right_wall_x_px=self.right_wall_x_px,
        )

        # Scorpion sprite size used for collision/clip. This is derived from the final
        # padded scorpion masks to keep render and gameplay boxes consistent.
        self.scorpion_w_px = jnp.array(int(self.renderer.SCORPION_RIGHT_MASKS.shape[2]), dtype=jnp.int32)
        self.scorpion_h_px = jnp.array(int(self.renderer.SCORPION_RIGHT_MASKS.shape[1]), dtype=jnp.int32)

        # Scorpion hitbox: use a tight bbox of non-transparent pixels (union across
        # both directions + both animation frames). This avoids collisions being
        # driven by the padded render canvas.
        _tid = int(self.renderer.jr.TRANSPARENT_ID)
        _sc_l = np.array(self.renderer.SCORPION_LEFT_MASKS)
        _sc_r = np.array(self.renderer.SCORPION_RIGHT_MASKS)
        _sc_combined = np.concatenate([_sc_l, _sc_r], axis=0)
        _occupied = np.any(_sc_combined != _tid, axis=0)
        _ys, _xs = np.where(_occupied)
        if _ys.size == 0 or _xs.size == 0:
            _hit_y0 = 0
            _hit_x0 = 0
            _hit_h = int(self.scorpion_h_px)
            _hit_w = int(self.scorpion_w_px)
        else:
            _hit_y0 = int(_ys.min())
            _hit_x0 = int(_xs.min())
            _hit_h = int(_ys.max()) - _hit_y0 + 1
            _hit_w = int(_xs.max()) - _hit_x0 + 1

        _inset_x = max(0, int(self.consts.scorpion_hit_inset_x))
        _inset_y = max(0, int(self.consts.scorpion_hit_inset_y))
        if _hit_w > 1 and _inset_x > 0:
            _hit_x0 = min(_hit_x0 + _inset_x, int(self.scorpion_w_px) - 1)
            _hit_w = max(1, _hit_w - 2 * _inset_x)
        if _hit_h > 1 and _inset_y > 0:
            _hit_y0 = min(_hit_y0 + _inset_y, int(self.scorpion_h_px) - 1)
            _hit_h = max(1, _hit_h - 2 * _inset_y)

        self.scorpion_hit_x0_px = jnp.array(_hit_x0, dtype=jnp.int32)
        self.scorpion_hit_y0_px = jnp.array(_hit_y0, dtype=jnp.int32)
        self.scorpion_hit_w_px = jnp.array(_hit_w, dtype=jnp.int32)
        self.scorpion_hit_h_px = jnp.array(_hit_h, dtype=jnp.int32)
        self.wall_block_player_width_px = jnp.array(
            max(
                int(self.renderer.HARRY_IDLE_MASKS.shape[2]),
                int(self.renderer.HARRY_RUN_MASKS.shape[2]),
                int(self.renderer.HARRY_CLIMB_MASKS.shape[2]),
                int(self.renderer.HARRY_JUMP_MASKS.shape[2]),
            ),
            dtype=jnp.int32,
        )
        self.wall_render_height_px = jnp.array(int(self.renderer.WALL_RENDER_MASK.shape[0]), dtype=jnp.int32)


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
        vx: chex.Array,
        y: chex.Array,
        vy: chex.Array,
        down_pressed: chex.Array,
        move_jump: chex.Array,
        move_left: chex.Array,
        move_right: chex.Array,
        on_ground: chex.Array,
        current_ground_y: chex.Array,
    ) -> tuple[
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ]:
        """
        Handles ladder enter/stay/exit.

        Returns:
            x, vx, y, vy, on_ground, on_ladder, current_ground_y, climb_active, exit_top_jump, ladder_step_idx, ladder_step_timer
        """
        consts = self.consts
        layout = self._screen_layout(room_byte)
        ladder_x = layout.ladder_x
        has_ladder = layout.has_ladder
        ladder_w = jnp.asarray(consts.ladder_width, dtype=jnp.int32)
        ladder_entry_w = jnp.asarray(consts.ladder_entry_width, dtype=jnp.int32)
        player_w = jnp.asarray(4, dtype=jnp.int32)
        sprite_w = self.wall_block_player_width_px.astype(jnp.int32)  # actual sprite width (13)

        x_int = x.astype(jnp.int32)
        player_right = x_int + player_w
        player_center = x_int + sprite_w // jnp.int32(2)
        ladder_right = ladder_x + ladder_w

        overlap_left = player_right > ladder_x
        overlap_right = x_int < ladder_right
        near_ladder = has_ladder & overlap_left & overlap_right
        entry_inset = (ladder_w - ladder_entry_w) // jnp.int32(2)
        entry_x0 = ladder_x + entry_inset
        entry_x1 = entry_x0 + ladder_entry_w
        centered_on_ladder = has_ladder & (player_center >= entry_x0) & (player_center < entry_x1)

        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)
        ladder_top_y = upper_ground + jnp.asarray(consts.ladder_top_peek_offset, dtype=jnp.float32)

        on_upper = current_ground_y == upper_ground
        on_lower = current_ground_y == lower_ground
        ladder_exit_active = state.ladder_exit_frames > jnp.int32(0)

        enter_from_upper = (
            on_ground & on_upper & centered_on_ladder & (down_pressed | move_jump) & (~ladder_exit_active)
        )
        enter_from_lower = (
            on_ground & on_lower & near_ladder & move_jump & (~ladder_exit_active)
        )

        entering_ladder = (~state.on_ladder) & (enter_from_upper | enter_from_lower)

        # Snap Harry to the centre of the ladder on entry (+1px right)
        ladder_center_x = (ladder_x + ladder_w // jnp.int32(2) - sprite_w // jnp.int32(2) + jnp.int32(1)).astype(x.dtype)
        x = jnp.where(entering_ladder, ladder_center_x, x)

        ladder_vertical = (y >= upper_ground) & (y <= lower_ground)
        still_on_ladder = state.on_ladder & near_ladder & ladder_vertical

        on_ladder_now = entering_ladder | still_on_ladder

        ladder_steps = jnp.int32(11)
        ladder_step_period = jnp.int32(4)
        step_size = (lower_ground - ladder_top_y) / ladder_steps.astype(jnp.float32)

        ladder_step_idx = state.ladder_step_idx.astype(jnp.int32)
        ladder_step_timer = state.ladder_step_timer.astype(jnp.int32)

        ladder_step_idx = jnp.where(
            entering_ladder,
            jnp.where(enter_from_upper, ladder_steps, jnp.int32(0)),
            ladder_step_idx,
        )
        ladder_step_timer = jnp.where(entering_ladder, jnp.int32(0), ladder_step_timer)

        climb_dir = jnp.where(
            move_jump,
            jnp.int32(1),
            jnp.where(down_pressed, jnp.int32(-1), jnp.int32(0)),
        )

        attempt_step = on_ladder_now & (climb_dir != jnp.int32(0))
        timer_next = jnp.where(attempt_step, ladder_step_timer + jnp.int32(1), jnp.int32(0))

        ladder_step_candidate = jnp.clip(
            ladder_step_idx + climb_dir,
            jnp.int32(0),
            ladder_steps,
        )
        moved = ladder_step_candidate != ladder_step_idx
        step_now = attempt_step & (timer_next >= ladder_step_period) & moved

        ladder_step_idx = jnp.where(step_now, ladder_step_candidate, ladder_step_idx)
        ladder_step_timer = jnp.where(
            on_ladder_now,
            jnp.where(timer_next >= ladder_step_period, jnp.int32(0), timer_next),
            jnp.int32(0),
        )

        y_step = lower_ground - ladder_step_idx.astype(jnp.float32) * step_size
        y = jnp.where(on_ladder_now, y_step, y)
        vy = jnp.where(on_ladder_now, 0.0, vy)
        on_ground = jnp.where(
            on_ladder_now,
            jnp.array(False, dtype=jnp.bool_),
            on_ground,   # keep whatever step computed if not on ladder
        )

        at_top = on_ladder_now & (ladder_step_idx == ladder_steps)
        at_bottom = on_ladder_now & (ladder_step_idx == jnp.int32(0))

        jump_dir_input = move_left | move_right
        exit_top_jump = on_ladder_now & at_top & jump_dir_input

        exit_bottom = on_ladder_now & at_bottom & down_pressed

        exiting = exit_top_jump | exit_bottom

        exit_dir = jnp.where(
            move_left,
            jnp.array(-1.0, dtype=jnp.float32),
            jnp.where(
                move_right,
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
            ),
        )

        jump_v = jnp.asarray(consts.jump_velocity, dtype=jnp.float32)
        jump_vx = jnp.asarray(consts.ladder_exit_jump_speed, dtype=jnp.float32) * exit_dir

        # Small visual nudge so Harry clears the ladder lip immediately, while
        # the arc itself is still carried primarily by the jump velocity.
        initial_hop_x = jnp.asarray(consts.ladder_exit_initial_hop, dtype=jnp.float32) * exit_dir
        x = jnp.where(exit_top_jump, x + initial_hop_x, x)

        y = jnp.where(exit_top_jump, ladder_top_y, y)
        vy = jnp.where(exit_top_jump, jump_v, vy)
        vx = jnp.where(exit_top_jump, jump_vx, vx)
        on_ground = jnp.where(
            exit_top_jump,
            jnp.array(False, dtype=jnp.bool_),
            on_ground,
        )

        max_x = jnp.asarray(consts.screen_width - player_w, dtype=jnp.float32)
        x = jnp.clip(x, 0.0, max_x)

        new_ground_y = jnp.where(
            exit_top_jump,
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

        climb_active = on_ladder & step_now

        ladder_step_idx = jnp.where(on_ladder, ladder_step_idx, jnp.int32(0))
        ladder_step_timer = jnp.where(on_ladder, ladder_step_timer, jnp.int32(0))

        return x, vx, y, vy, on_ground, on_ladder, current_ground_y, climb_active, exit_top_jump, ladder_step_idx, ladder_step_timer

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
        ladder_exit_frames = state.ladder_exit_frames
        transition_active = state.respawn_phase != jnp.int32(0)
        gameplay_active = ~transition_active
        time_left = state.time_left
        lives_left = state.lives_left
        hurt_cooldown = state.hurt_cooldown
        scorpion_x_prev = state.scorpion_x
        scorpion_x = scorpion_x_prev
        scorpion_facing_right = state.scorpion_facing_right

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

        down_pressed = down_pressed & gameplay_active
        move_left = move_left & gameplay_active
        move_right = move_right & gameplay_active
        move_jump = move_jump & gameplay_active

        has_input = action != Action.NOOP
        timer_started = state.timer_started | (has_input & gameplay_active)

        time_left = state.time_left - (timer_started & gameplay_active).astype(jnp.int32)
        time_left = jnp.maximum(time_left, 0)

        layout = self._screen_layout(state.room_byte)
        ladder_x = layout.ladder_x
        has_ladder = layout.has_ladder
        ladder_w = jnp.asarray(consts.ladder_width, dtype=jnp.int32)
        ladder_fall_w = jnp.asarray(consts.ladder_hole_fall_width, dtype=jnp.int32)
        player_w = jnp.asarray(4, dtype=jnp.int32)
        sprite_w = self.wall_block_player_width_px.astype(jnp.int32)  # actual sprite width (13)

        x_int = x.astype(jnp.int32)
        player_right = x_int + player_w
        # Collision center: visual sprite center, shifted 1px left when idle
        is_moving = move_left | move_right
        idle_shift = jnp.where(is_moving, jnp.int32(0), jnp.int32(-1))
        player_center = x_int + sprite_w // jnp.int32(2) + idle_shift
        ladder_right = ladder_x + ladder_w

        overlap_left = player_right > ladder_x
        overlap_right = x_int < ladder_right
        near_ladder = has_ladder & overlap_left & overlap_right
        # Fall-through uses the narrower ladder_hole_fall_width
        fall_inset = (ladder_w - ladder_fall_w) // jnp.int32(2)
        fall_x0 = ladder_x + fall_inset
        fall_x1 = fall_x0 + ladder_fall_w
        over_ladder = has_ladder & (player_center >= fall_x0) & (player_center < fall_x1)

        has_side_hole, over_side_hole, hole_left_x, hole_right_x, hole_w = self._side_hole_info(
            room_byte=state.room_byte,
            player_center_x=player_center.astype(jnp.int32),
        )

        over_any_hole = over_ladder | over_side_hole

        prev_x_int = state.player_x.astype(jnp.int32)
        prev_is_moving = jnp.abs(state.player_vx) > jnp.asarray(0.0, dtype=jnp.float32)
        prev_idle_shift = jnp.where(prev_is_moving, jnp.int32(0), jnp.int32(-1))
        prev_player_center = prev_x_int + sprite_w // jnp.int32(2) + prev_idle_shift
        prev_over_ladder = has_ladder & (prev_player_center >= fall_x0) & (prev_player_center < fall_x1)
        _, prev_over_side_hole, _, _, _ = self._side_hole_info(
            room_byte=state.room_byte,
            player_center_x=prev_player_center.astype(jnp.int32),
        )
        prev_over_any_hole = prev_over_ladder | prev_over_side_hole

        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)
        
        on_upper_level = state.current_ground_y == upper_ground
        on_lower_level = state.current_ground_y == lower_ground
        ladder_exit_active = ladder_exit_frames > jnp.int32(0)

        falling_through_hole = over_any_hole & on_upper_level & (~on_ground) & (~state.on_ladder) & (vy >= 0)

        speed = jnp.asarray(consts.player_speed, dtype=jnp.float32)
        jump_horiz_speed = speed * jnp.asarray(consts.normal_jump_horizontal_scale, dtype=jnp.float32)
        vx = jnp.where(move_left, -speed, jnp.where(move_right, speed, 0.0))
        # When airborne, preserve launch momentum (no mid-air steering) so
        # regular jumps still have enough carry while being shorter than full-speed runs.
        airborne = (~on_ground) & (~state.on_ladder)
        air_vx = jnp.where(
            state.player_vx < 0,
            -jnp.minimum(jnp.abs(state.player_vx), jump_horiz_speed),
            jnp.where(state.player_vx > 0, jnp.minimum(jnp.abs(state.player_vx), jump_horiz_speed), 0.0),
        )
        vx = jnp.where(airborne, air_vx, vx)
        vx = jnp.where(airborne & ladder_exit_active, state.player_vx, vx)
        vx = jnp.where(state.on_ladder, 0.0, vx)
        vx = jnp.where(falling_through_hole & (~ladder_exit_active), 0.0, vx)

        trying_to_enter_ladder = near_ladder & on_lower_level & move_jump
        jump_mask = on_ground & move_jump & (~state.on_ladder) & (~trying_to_enter_ladder) & (~ladder_exit_active)
        jump_launch_vx = jnp.where(move_left, -jump_horiz_speed, jnp.where(move_right, jump_horiz_speed, 0.0))
        vx = jnp.where(jump_mask, jump_launch_vx, vx)
        vy = jnp.where(
            jump_mask,
            jnp.asarray(consts.jump_velocity, dtype=jnp.float32),
            vy,
        )

        gravity = jnp.asarray(consts.gravity, dtype=jnp.float32)
        fall_speed = jnp.asarray(consts.fall_speed, dtype=jnp.float32)
        apply_gravity = (~on_ground) & (~state.on_ladder)
        # Symmetric gravity on both ascent and descent (capped at fall_speed)
        vy = jnp.where(
            apply_gravity,
            jnp.minimum(vy + gravity, fall_speed),
            vy,
        )

        y = y + vy
        x = x + vx

        wall_w = jnp.int32(consts.tunnel_wall_width)
        wall_block_player_w = self.wall_block_player_width_px.astype(jnp.int32)
        right_wall_block_player_w = jnp.int32(consts.right_wall_block_player_width)

        block = layout.has_wall & on_lower_level

        wall_left  = layout.wall_x
        wall_right = layout.wall_x + wall_w

        x = jnp.where(
            block & (layout.wall_side == jnp.int32(1)),
            jnp.minimum(x, (wall_left - right_wall_block_player_w).astype(x.dtype)),
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

        entered_new_room = exited_left | exited_right
        scorpion_spawn_x = jnp.asarray(consts.scorpion_spawn_x, dtype=jnp.float32)
        scorpion_x = jnp.where(entered_new_room, scorpion_spawn_x, scorpion_x)

        player_w_f = jnp.asarray(4, dtype=jnp.float32)
        x = jnp.clip(x, 0.0, jnp.asarray(consts.screen_width, dtype=jnp.float32) - player_w_f)

        previous_ground = state.current_ground_y
        clamp_mask = ~state.on_ladder

        raw_on_ground_upper = (y >= previous_ground) & (~over_any_hole)
        # One-frame edge grace: if Harry was grounded and just crossed from
        # solid ground into a hole this frame, keep him grounded for that
        # frame only. Next frame he can fall normally.
        just_entered_hole = over_any_hole & (~prev_over_any_hole)
        held_by_ground = state.on_ground & (y >= previous_ground) & on_upper_level & just_entered_hole
        raw_on_ground_upper = raw_on_ground_upper | held_by_ground

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

        x, vx, y, vy, on_ground, on_ladder, current_ground_y, climb_active, started_ladder_exit, ladder_step_idx, ladder_step_timer = self._apply_ladder(
            state=state,
            room_byte=new_room_byte,
            x=x,
            vx=vx,
            y=y,
            vy=vy,
            down_pressed=down_pressed,
            move_jump=move_jump,
            move_left=move_left,
            move_right=move_right,
            on_ground=on_ground,
            current_ground_y=current_ground_y,
        )

        ladder_exit_frames = jnp.maximum(ladder_exit_frames - jnp.int32(1), jnp.int32(0))
        ladder_exit_frames = jnp.where(
            started_ladder_exit,
            jnp.int32(consts.ladder_exit_grace_frames),
            ladder_exit_frames,
        )
        ladder_exit_frames = jnp.where(on_ladder, jnp.int32(0), ladder_exit_frames)

        has_logs, logs_are_rolling, log_count, log_xs, has_fireplace, has_snake = room_hazards_from_room_byte(new_room_byte)

        has_scorpion = has_scorpion_from_room_byte(new_room_byte)
        player_is_underground = current_ground_y == lower_ground

        scorpion_speed = jnp.asarray(consts.scorpion_speed, dtype=jnp.float32)
        player_center_x_f = x + jnp.asarray(2.0, dtype=jnp.float32)
        dx_scorpion = player_center_x_f - scorpion_x
        scorpion_step = jnp.clip(dx_scorpion, -scorpion_speed, scorpion_speed)
        scorpion_x = jnp.where(
            has_scorpion & gameplay_active,
            scorpion_x + scorpion_step,
            scorpion_x,
        )
        scorpion_max_x = jnp.asarray(
            max(0, consts.screen_width - int(self.scorpion_w_px)),
            dtype=jnp.float32,
        )
        scorpion_x = jnp.clip(scorpion_x, 0.0, scorpion_max_x)

        # Persist scorpion facing direction from its own last nonzero movement.
        # Must not depend on Harry's position; stopping preserves direction.
        scorpion_dx = scorpion_x - scorpion_x_prev
        scorpion_facing_right = jnp.where(
            scorpion_dx > jnp.asarray(0.0, dtype=jnp.float32),
            jnp.array(True, dtype=jnp.bool_),
            jnp.where(
                scorpion_dx < jnp.asarray(0.0, dtype=jnp.float32),
                jnp.array(False, dtype=jnp.bool_),
                scorpion_facing_right,
            ),
        )

        # Scorpion animation: advance only while the scorpion is actually moving.
        scorpion_anim_period = jnp.int32(max(1, int(consts.scorpion_anim_period)))
        scorpion_moved = (scorpion_x != scorpion_x_prev) & has_scorpion & gameplay_active
        scorpion_anim_timer = jnp.where(
            scorpion_moved,
            state.scorpion_anim_timer.astype(jnp.int32) + jnp.int32(1),
            jnp.int32(0),
        )
        scorpion_advance = scorpion_moved & (scorpion_anim_timer >= scorpion_anim_period)
        scorpion_anim_idx = jnp.where(
            scorpion_advance,
            jnp.int32(1) - state.scorpion_anim_idx.astype(jnp.int32),
            state.scorpion_anim_idx.astype(jnp.int32),
        )
        scorpion_anim_timer = jnp.where(scorpion_advance, jnp.int32(0), scorpion_anim_timer)
        scorpion_anim_idx = jnp.where(has_scorpion, scorpion_anim_idx, jnp.int32(0))
        scorpion_anim_timer = jnp.where(has_scorpion, scorpion_anim_timer, jnp.int32(0))

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

        wood_visual_pad_x = jnp.int32(consts.wood_visual_contact_pad_x)
        wood_visual_shift_x = jnp.int32(consts.wood_visual_contact_shift_x)
        visual_seg1_x0 = seg1_x0 + wood_visual_shift_x
        visual_seg1_x1 = seg1_x1 + wood_visual_shift_x
        visual_seg2_x0 = seg2_x0 + wood_visual_shift_x
        visual_seg2_x1 = seg2_x1 + wood_visual_shift_x
        visual_overlap_seg1 = (x1 > (visual_seg1_x0 - wood_visual_pad_x)) & (x0 < (visual_seg1_x1 + wood_visual_pad_x))
        visual_overlap_seg2 = wraps & (x1 > (visual_seg2_x0 - wood_visual_pad_x)) & (x0 < (visual_seg2_x1 + wood_visual_pad_x))
        visual_overlap_x = visual_overlap_seg1 | visual_overlap_seg2

        touching_any = jnp.any(active & overlap_x & overlap_y)
        touching_wood = has_logs & jnp.any(active & visual_overlap_x & overlap_y)
        scoring_touching_wood = has_logs & touching_any & gameplay_active

        drain = jnp.int32(consts.wood_drain_per_frame)
        score = jnp.where(scoring_touching_wood, score - drain, score)
        score = jnp.maximum(score, jnp.int32(0))

        # --- Log pushback on ladder -------------------------------------------
        # Rising-edge: only trigger when touching_wood transitions False→True
        log_hit_on_ladder = on_ladder & touching_wood & (~state.touching_wood)
        log_push_remaining = jnp.where(
            log_hit_on_ladder,
            jnp.asarray(consts.log_push_amount, dtype=jnp.float32),
            state.log_push_remaining,
        )
        push_speed = jnp.asarray(consts.log_push_speed, dtype=jnp.float32)
        push_this_frame = jnp.minimum(log_push_remaining, push_speed)
        push_this_frame = jnp.where(
            on_ladder & (log_push_remaining > 0), push_this_frame, jnp.float32(0.0)
        )
        pushed_y = y + push_this_frame
        y = jnp.where(
            on_ladder,
            jnp.clip(
                pushed_y,
                jnp.asarray(consts.ground_y, dtype=jnp.float32),
                jnp.asarray(consts.underground_y, dtype=jnp.float32),
            ),
            y,
        )
        log_push_remaining = jnp.maximum(log_push_remaining - push_this_frame, jnp.float32(0.0))
        # Reset push state when not on ladder
        log_push_remaining = jnp.where(on_ladder, log_push_remaining, jnp.float32(0.0))
        # Animate climb sprite during the push
        climb_active = climb_active | (on_ladder & (push_this_frame > 0))

        # Keep ladder position aligned to discrete steps even if other ladder-only
        # effects (like log pushback) adjust y.
        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)
        ladder_top_y = upper_ground + jnp.asarray(consts.ladder_top_peek_offset, dtype=jnp.float32)
        ladder_steps_f = jnp.asarray(11.0, dtype=jnp.float32)
        step_size = (lower_ground - ladder_top_y) / ladder_steps_f
        ladder_step_idx_from_y = jnp.round((lower_ground - y) / step_size).astype(jnp.int32)
        ladder_step_idx_from_y = jnp.clip(ladder_step_idx_from_y, jnp.int32(0), jnp.int32(11))
        y_quant = lower_ground - ladder_step_idx_from_y.astype(jnp.float32) * step_size
        y = jnp.where(on_ladder, y_quant, y)
        ladder_step_idx = jnp.where(on_ladder, ladder_step_idx_from_y, jnp.int32(0))
        ladder_step_timer = jnp.where(on_ladder, ladder_step_timer, jnp.int32(0))
        # ----------------------------------------------------------------------

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
        can_hurt = (hurt_cooldown == jnp.int32(0)) & gameplay_active
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

        scorpion_w = self.scorpion_w_px.astype(jnp.int32)
        scorpion_h = self.scorpion_h_px.astype(jnp.int32)
        scorpion_top_render = jnp.int32(consts.underground_y) - scorpion_h + jnp.int32(1) + jnp.int32(consts.scorpion_y_offset)

        hit_x0 = self.scorpion_hit_x0_px.astype(jnp.int32)
        hit_y0 = self.scorpion_hit_y0_px.astype(jnp.int32)
        hit_w = self.scorpion_hit_w_px.astype(jnp.int32)
        hit_h = self.scorpion_hit_h_px.astype(jnp.int32)

        scorpion_y0 = scorpion_top_render + hit_y0
        scorpion_y1 = scorpion_y0 + hit_h

        max_scorpion_start = jnp.maximum(screen_w_i - hit_w, jnp.int32(0))
        scorpion_left = jnp.clip(scorpion_x.astype(jnp.int32) + hit_x0, jnp.int32(0), max_scorpion_start)
        scorpion_right = scorpion_left + hit_w

        overlap_scorpion = (x1 > scorpion_left) & (x0 < scorpion_right) & (y1 > scorpion_y0) & (y0 < scorpion_y1)
        hit_scorpion = has_scorpion & player_is_underground & overlap_scorpion & can_hurt
        scorpion_x_before_reset = scorpion_x

        hit_other_hazard = hit_fire | hit_snake
        hit_hazard = hit_scorpion | hit_other_hazard

        lives_left = jnp.where(hit_hazard, lives_left - jnp.int32(1), lives_left)
        lives_left = jnp.maximum(lives_left, jnp.int32(0))
        lost_final_life = hit_hazard & (lives_left <= jnp.int32(0))

        respawn_x = jnp.asarray(consts.player_start_x, dtype=jnp.float32)
        scorpion_respawn_x = jnp.asarray(consts.underground_respawn_x, dtype=jnp.float32)
        respawn_ground_y = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        respawn_underground_y = jnp.asarray(consts.underground_y, dtype=jnp.float32)
        respawn_target_x = jnp.where(
            lost_final_life,
            respawn_x,
            jnp.where(hit_scorpion, scorpion_respawn_x, respawn_x),
        )
        respawn_target_y = jnp.where(
            lost_final_life,
            respawn_ground_y,
            jnp.where(hit_scorpion, respawn_underground_y, respawn_ground_y),
        )
        respawn_target_ground_y = respawn_target_y

        # --- Run animation state (render-only) ------------------------------
        running_now = gameplay_active & (~on_ladder) & on_ground & (jnp.abs(vx) > jnp.asarray(0.0, dtype=jnp.float32))
        running_prev = (~state.on_ladder) & state.on_ground & (jnp.abs(state.player_vx) > jnp.asarray(0.0, dtype=jnp.float32))
        started_running = running_now & (~running_prev)

        # Startup lasts 4 frames total: running3 then running4.
        run_startup_timer = jnp.where(
            started_running,
            jnp.int32(4),
            jnp.where(
                running_now,
                jnp.maximum(state.run_startup_timer - jnp.int32(1), jnp.int32(0)),
                jnp.int32(0),
            ),
        )

        steady_running = running_now & (run_startup_timer == jnp.int32(0))
        run_anim_timer = jnp.where(
            steady_running,
            state.run_anim_timer + jnp.int32(1),
            jnp.int32(0),
        )
        hold_frames = jnp.int32(3)
        advance_phase = steady_running & (run_anim_timer >= hold_frames)
        run_anim_timer = jnp.where(advance_phase, jnp.int32(0), run_anim_timer)
        run_anim_phase = jnp.where(
            steady_running,
            jnp.where(
                advance_phase,
                jnp.mod(state.run_anim_phase + jnp.int32(1), jnp.int32(4)),
                state.run_anim_phase,
            ),
            jnp.int32(0),
        )
        # --------------------------------------------------------------------

        vx = jnp.where(hit_hazard, jnp.asarray(0.0, dtype=jnp.float32), vx)
        vy = jnp.where(hit_hazard, jnp.asarray(0.0, dtype=jnp.float32), vy)
        on_ladder = jnp.where(hit_hazard, jnp.array(False, dtype=jnp.bool_), on_ladder)
        ladder_exit_frames = jnp.where(hit_hazard, jnp.int32(0), ladder_exit_frames)

        next_hurt_cooldown = jnp.maximum(hurt_cooldown - jnp.int32(1), jnp.int32(0))
        next_hurt_cooldown = jnp.where(
            hit_scorpion,
            jnp.int32(consts.scorpion_hurt_cooldown_frames),
            next_hurt_cooldown,
        )
        next_hurt_cooldown = jnp.where(
            hit_other_hazard,
            jnp.int32(max(consts.fire_hurt_cooldown_frames, consts.snake_hurt_cooldown_frames)),
            next_hurt_cooldown,
        )

        respawn_phase = jnp.where(hit_hazard, jnp.int32(1), state.respawn_phase)
        respawn_timer = jnp.where(hit_hazard, jnp.int32(consts.death_pause_frames), state.respawn_timer)
        stored_respawn_target_x = jnp.where(hit_hazard, respawn_target_x, state.respawn_target_x)
        stored_respawn_target_y = jnp.where(hit_hazard, respawn_target_y, state.respawn_target_y)
        stored_respawn_target_ground_y = jnp.where(hit_hazard, respawn_target_ground_y, state.respawn_target_ground_y)

        done = (time_left <= 0)

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
            scorpion_x=scorpion_x,
            scorpion_anim_idx=scorpion_anim_idx,
            scorpion_anim_timer=scorpion_anim_timer,
            scorpion_facing_right=scorpion_facing_right,
            touching_wood=touching_wood,
            climb_active=climb_active,
            ladder_step_idx=ladder_step_idx,
            ladder_step_timer=ladder_step_timer,
            run_startup_timer=run_startup_timer,
            run_anim_phase=run_anim_phase,
            run_anim_timer=run_anim_timer,
            facing_left=jnp.where(
                move_left, jnp.array(True, dtype=jnp.bool_),
                jnp.where(move_right, jnp.array(False, dtype=jnp.bool_), state.facing_left),
            ),
            log_push_remaining=log_push_remaining,
            ladder_exit_frames=ladder_exit_frames,
            respawn_phase=respawn_phase,
            respawn_timer=respawn_timer,
            respawn_target_x=stored_respawn_target_x,
            respawn_target_y=stored_respawn_target_y,
            respawn_target_ground_y=stored_respawn_target_ground_y,
            screen_id=new_screen_id,
            room_byte=new_room_byte,
        )

        is_pause_phase = state.respawn_phase == jnp.int32(1)
        is_drop_phase = state.respawn_phase == jnp.int32(2)
        start_drop = is_pause_phase & (state.respawn_timer <= jnp.int32(1))
        pause_timer_next = jnp.maximum(state.respawn_timer - jnp.int32(1), jnp.int32(0))
        underground_respawn = state.respawn_target_ground_y == jnp.asarray(consts.underground_y, dtype=jnp.float32)
        wall_top_y = jnp.asarray(
            consts.underground_y - self.wall_render_height_px.astype(jnp.int32) + consts.underground_respawn_wall_clearance,
            dtype=jnp.float32,
        )
        # Render uses a fixed reveal boundary at ground_y (+ offset). Anchor the underground
        # respawn spawn position to this same boundary so Harry emerges from the tunnel edge
        # rather than dropping in from above it.
        underground_reveal_y = jnp.asarray(
            consts.ground_y + consts.underground_respawn_reveal_from_ground + consts.underground_respawn_reveal_y_offset,
            dtype=jnp.float32,
        )
        respawn_spawn_y = jnp.where(
            underground_respawn,
            jnp.asarray(
                underground_reveal_y - consts.underground_respawn_spawn_above_reveal,
                dtype=jnp.float32,
            ),
            jnp.asarray(
                consts.ground_y - consts.respawn_drop_spawn_y_offset,
                dtype=jnp.float32,
            ),
        )
        respawn_drop_speed = jnp.asarray(consts.respawn_drop_speed, dtype=jnp.float32)
        drop_y = jnp.minimum(state.player_y + respawn_drop_speed, state.respawn_target_y)
        landed = is_drop_phase & (drop_y >= state.respawn_target_y)

        transition_state = PitfallState(
            player_x=jnp.where(start_drop | is_drop_phase, state.respawn_target_x, state.player_x),
            player_y=jnp.where(
                start_drop,
                respawn_spawn_y,
                jnp.where(is_drop_phase, drop_y, state.player_y),
            ),
            player_vx=jnp.array(0.0, dtype=jnp.float32),
            player_vy=jnp.where(
                start_drop | (is_drop_phase & (~landed)),
                respawn_drop_speed,
                jnp.array(0.0, dtype=jnp.float32),
            ),
            on_ground=jnp.where(
                start_drop,
                jnp.array(False, dtype=jnp.bool_),
                jnp.where(is_drop_phase, landed, state.on_ground),
            ),
            score=state.score,
            timer_started=state.timer_started,
            time_left=state.time_left,
            lives_left=state.lives_left,
            done=state.done,
            hurt_cooldown=state.hurt_cooldown,
            down_pressed=jnp.array(False, dtype=jnp.bool_),
            on_ladder=jnp.array(False, dtype=jnp.bool_),
            current_ground_y=jnp.where(
                start_drop | is_drop_phase,
                state.respawn_target_ground_y,
                state.current_ground_y,
            ),
            scorpion_x=jnp.where(
                start_drop & underground_respawn,
                scorpion_spawn_x,
                state.scorpion_x,
            ),
            scorpion_anim_idx=jnp.int32(0),
            scorpion_anim_timer=jnp.int32(0),
            scorpion_facing_right=state.scorpion_facing_right,
            touching_wood=jnp.array(False, dtype=jnp.bool_),
            climb_active=jnp.array(False, dtype=jnp.bool_),
            ladder_step_idx=jnp.int32(0),
            ladder_step_timer=jnp.int32(0),
            run_startup_timer=jnp.int32(0),
            run_anim_phase=jnp.int32(0),
            run_anim_timer=jnp.int32(0),
            facing_left=state.facing_left,
            log_push_remaining=jnp.array(0.0, dtype=jnp.float32),
            ladder_exit_frames=jnp.int32(0),
            respawn_phase=jnp.where(
                is_pause_phase,
                jnp.where(start_drop, jnp.int32(2), jnp.int32(1)),
                jnp.where(is_drop_phase, jnp.where(landed, jnp.int32(0), jnp.int32(2)), jnp.int32(0)),
            ),
            respawn_timer=jnp.where(is_pause_phase & (~start_drop), pause_timer_next, jnp.int32(0)),
            respawn_target_x=state.respawn_target_x,
            respawn_target_y=state.respawn_target_y,
            respawn_target_ground_y=state.respawn_target_ground_y,
            screen_id=state.screen_id,
            room_byte=state.room_byte,
        )

        final_state = jax.tree.map(
            lambda normal_value, transition_value: jnp.where(transition_active, transition_value, normal_value),
            new_state,
            transition_state,
        )
        final_done = (final_state.time_left <= 0) | ((final_state.lives_left <= 0) & (final_state.respawn_phase == jnp.int32(0)))
        final_state = final_state.replace(done=final_done)

        obs = self._get_observation(final_state)
        reward = self._get_reward(state, final_state)
        info = self._get_info(final_state)

        return obs, final_state, reward, final_state.done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PitfallState) -> PitfallObservation:
        has_logs, logs_are_rolling, log_count, log_xs, has_fire, has_snake = room_hazards_from_room_byte(state.room_byte)
        has_scorpion = has_scorpion_from_room_byte(state.room_byte)
        layout = self._screen_layout(state.room_byte)
        return PitfallObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            screen_id=state.screen_id,
            room_byte=state.room_byte,
            current_ground_y=state.current_ground_y,
            on_ground=state.on_ground,
            on_ladder=state.on_ladder,
            facing_left=state.facing_left,
            scorpion_x=state.scorpion_x,
            has_scorpion=has_scorpion,
            has_fire=has_fire,
            has_snake=has_snake,
            has_logs=has_logs,
            log_count=log_count,
            log_xs=log_xs,
            logs_are_rolling=logs_are_rolling,
            has_ladder=layout.has_ladder,
            ladder_x=layout.ladder_x,
            has_wall=layout.has_wall,
            wall_x=layout.wall_x,
            wall_side=layout.wall_side,
            time_left=state.time_left,
            lives_left=state.lives_left,
            score=state.score,
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
        return spaces.Dict(
            {
                "player_x": spaces.Box(low=0.0, high=float(self.consts.screen_width - 1), shape=(), dtype=jnp.float32),
                "player_y": spaces.Box(low=0.0, high=float(self.consts.screen_height - 1), shape=(), dtype=jnp.float32),
                "screen_id": spaces.Box(low=0, high=254, shape=(), dtype=jnp.int32),
                "room_byte": spaces.Box(low=0, high=255, shape=(), dtype=jnp.uint8),
                "current_ground_y": spaces.Box(low=0.0, high=float(self.consts.screen_height - 1), shape=(), dtype=jnp.float32),
                "on_ground": spaces.Discrete(2),
                "on_ladder": spaces.Discrete(2),
                "facing_left": spaces.Discrete(2),
                "scorpion_x": spaces.Box(low=0.0, high=float(self.consts.screen_width - 1), shape=(), dtype=jnp.float32),
                "has_scorpion": spaces.Discrete(2),
                "has_fire": spaces.Discrete(2),
                "has_snake": spaces.Discrete(2),
                "has_logs": spaces.Discrete(2),
                "log_count": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
                "log_xs": spaces.Box(low=0, high=int(self.consts.screen_width - 1), shape=(3,), dtype=jnp.int32),
                "logs_are_rolling": spaces.Discrete(2),
                "has_ladder": spaces.Discrete(2),
                "ladder_x": spaces.Box(low=0, high=int(self.consts.screen_width - 1), shape=(), dtype=jnp.int32),
                "has_wall": spaces.Discrete(2),
                "wall_x": spaces.Box(low=0, high=int(self.consts.screen_width - 1), shape=(), dtype=jnp.int32),
                "wall_side": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
                "time_left": spaces.Box(
                    low=0,
                    high=int(self.consts.initial_time_seconds * self.consts.fps),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "lives_left": spaces.Box(low=0, high=int(self.consts.max_lives), shape=(), dtype=jnp.int32),
                "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(
                int(self.consts.screen_height),
                int(self.consts.screen_width),
                3,
            ),
            dtype=jnp.uint8,
        )

    def render(self, state: PitfallState) -> jnp.ndarray:
        return self.renderer.render(state)

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
            scorpion_x=jnp.array(consts.scorpion_spawn_x, dtype=jnp.float32),
            scorpion_anim_idx=jnp.int32(0),
            scorpion_anim_timer=jnp.int32(0),
            scorpion_facing_right=jnp.array(True, dtype=jnp.bool_),
            touching_wood=jnp.array(False, dtype=jnp.bool_),
            climb_active=jnp.array(False, dtype=jnp.bool_),
            ladder_step_idx=jnp.int32(0),
            ladder_step_timer=jnp.int32(0),
            run_startup_timer=jnp.int32(0),
            run_anim_phase=jnp.int32(0),
            run_anim_timer=jnp.int32(0),
            facing_left=jnp.array(False, dtype=jnp.bool_),
            log_push_remaining=jnp.array(0.0, dtype=jnp.float32),
            ladder_exit_frames=jnp.array(0, dtype=jnp.int32),
            respawn_phase=jnp.array(0, dtype=jnp.int32),
            respawn_timer=jnp.array(0, dtype=jnp.int32),
            respawn_target_x=jnp.array(consts.player_start_x, dtype=jnp.float32),
            respawn_target_y=jnp.array(consts.ground_y, dtype=jnp.float32),
            respawn_target_ground_y=jnp.array(consts.ground_y, dtype=jnp.float32),
            screen_id=jnp.array(0, dtype=jnp.int32),
            room_byte=jnp.array(SEED, dtype=jnp.uint8),
        )
        return state
    

class PitfallRenderer(JAXGameRenderer):
    """Pitfall renderer using the shared raster+palette pipeline."""

    def __init__(
        self,
        consts: PitfallConstants | None = None,
        config: render_utils.RendererConfig | None = None,
        ladder_x_px: chex.Array | None = None,
        left_wall_x_px: chex.Array | None = None,
        right_wall_x_px: chex.Array | None = None,
    ):
        super().__init__()
        self.consts = consts or PitfallConstants()

        screen_w = int(self.consts.screen_width)
        wall_w = int(self.consts.tunnel_wall_width)

        def _clamp_wall_x(x: int) -> int:
            return max(0, min(screen_w - wall_w, x))

        if left_wall_x_px is None:
            left_wall_x_px = jnp.array(_clamp_wall_x(23), dtype=jnp.int32)
        else:
            left_wall_x_px = jnp.asarray(left_wall_x_px, dtype=jnp.int32)

        if right_wall_x_px is None:
            right_wall_x_px = jnp.array(_clamp_wall_x(132), dtype=jnp.int32)
        else:
            right_wall_x_px = jnp.asarray(right_wall_x_px, dtype=jnp.int32)

        if ladder_x_px is None:
            ladder_x_default = int(round(140 * screen_w / 300.0))
            ladder_x_default = max(0, min(screen_w - int(self.consts.ladder_width), ladder_x_default))
            ladder_x_px = jnp.array(ladder_x_default, dtype=jnp.int32)
        else:
            ladder_x_px = jnp.asarray(ladder_x_px, dtype=jnp.int32)

        self.ladder_x_px = ladder_x_px
        self.left_wall_x_px = left_wall_x_px
        self.right_wall_x_px = right_wall_x_px

        self.config = config or render_utils.RendererConfig(
            game_dimensions=(self.consts.screen_height, self.consts.screen_width),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sprites', 'pitfall')
        asset_config = list(self.consts.ASSET_CONFIG)

        h = int(self.consts.screen_height)
        w = int(self.consts.screen_width)

        bg = jnp.zeros((h, w, 4), dtype=jnp.uint8)
        bg = bg.at[:, :, 3].set(255)
        ground = int(self.consts.ground_y)
        underground = int(self.consts.underground_y)
        bg = bg.at[ground:ground + 2, :, 1].set(200)
        bg = bg.at[underground:underground + 2, :, 1].set(120)

        asset_config = [
            {'name': 'background', 'type': 'background', 'data': bg},
            *[a for a in asset_config if a.get('type') != 'background'],
        ]

        def _load_alpha_group(file_names: list[str], rgb: tuple[int, int, int]) -> list[jnp.ndarray]:
            def _trim_rgba(frame_rgba: jnp.ndarray) -> jnp.ndarray:
                frame_np = np.array(frame_rgba)
                alpha = frame_np[:, :, 3] > 0
                ys, xs = np.where(alpha)
                if ys.size == 0 or xs.size == 0:
                    return frame_rgba
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                return jnp.asarray(frame_np[y0:y1 + 1, x0:x1 + 1, :], dtype=jnp.uint8)

            frames = []
            max_h = 1
            max_w = 1
            for file_name in file_names:
                file_path = os.path.join(sprite_path, file_name)
                frame_np = np.load(file_path)
                frame = jnp.asarray(frame_np)

                if frame.ndim == 2:
                    alpha = jnp.where(frame > 0, jnp.uint8(255), jnp.uint8(0))
                    h_frame, w_frame = alpha.shape
                    color_rgb = jnp.stack(
                        [
                            jnp.full((h_frame, w_frame), jnp.uint8(rgb[0]), dtype=jnp.uint8),
                            jnp.full((h_frame, w_frame), jnp.uint8(rgb[1]), dtype=jnp.uint8),
                            jnp.full((h_frame, w_frame), jnp.uint8(rgb[2]), dtype=jnp.uint8),
                        ],
                        axis=2,
                    )
                    frame_rgba = jnp.concatenate([color_rgb, alpha[:, :, None]], axis=2)
                elif frame.ndim == 3 and frame.shape[2] in (3, 4):
                    frame_u8 = frame.astype(jnp.uint8)
                    if frame_u8.shape[2] == 3:
                        alpha = jnp.where(jnp.any(frame_u8 != 0, axis=2), jnp.uint8(255), jnp.uint8(0))
                        frame_rgba = jnp.concatenate([frame_u8, alpha[:, :, None]], axis=2)
                    else:
                        frame_rgba = frame_u8
                else:
                    raise ValueError(f"Unsupported scorpion frame format for {file_name}: shape={frame.shape}")

                frame_rgba = _trim_rgba(frame_rgba)
                # IMPORTANT: do not downscale here.
                # Scorpion masks are downscaled later alongside other sprites; doing it here as
                # well effectively double-downscales and can delete thin features (tail tip).
                frames.append(frame_rgba)

                max_h = max(max_h, int(frame_rgba.shape[0]))
                max_w = max(max_w, int(frame_rgba.shape[1]))

            # Pad to a common size so asset loading can stack frames.
            # Padding is transparent (alpha=0), so it never crops visible pixels.
            padded_frames = []
            for fr in frames:
                pad_h = max_h - int(fr.shape[0])
                pad_w = max_w - int(fr.shape[1])
                padded_frames.append(
                    jnp.pad(
                        fr,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode='constant',
                        constant_values=jnp.uint8(0),
                    )
                )
            return padded_frames

        def _normalize_to_rgba_u8(image: np.ndarray, asset_name: str) -> np.ndarray:
            if image.ndim == 2:
                alpha = np.where(image > 0, np.uint8(255), np.uint8(0))
                rgb = np.stack([image, image, image], axis=2).astype(np.uint8)
                return np.concatenate([rgb, alpha[:, :, None]], axis=2).astype(np.uint8)
            if image.ndim == 3 and image.shape[2] == 3:
                image_u8 = image.astype(np.uint8)
                alpha = np.where(np.any(image_u8 != 0, axis=2), np.uint8(255), np.uint8(0))
                return np.concatenate([image_u8, alpha[:, :, None]], axis=2).astype(np.uint8)
            if image.ndim == 3 and image.shape[2] == 4:
                return image.astype(np.uint8)
            raise ValueError(f"Unsupported backdrop format for {asset_name}: shape={image.shape}")

        def _cleanup_leading_black_edges(backdrop_rgba: np.ndarray, max_strip: int = 16) -> np.ndarray:
            rgb = backdrop_rgba[:, :, :3]
            h_img, w_img = rgb.shape[:2]

            top_limit = min(max_strip, h_img)
            left_limit = min(max_strip, w_img)

            top_strip = 0
            for y in range(top_limit):
                if np.all(rgb[y] == 0):
                    top_strip += 1
                else:
                    break

            left_strip = 0
            for x in range(left_limit):
                if np.all(rgb[:, x] == 0):
                    left_strip += 1
                else:
                    break

            cleaned = backdrop_rgba.copy()
            if 0 < top_strip < h_img:
                cleaned[:top_strip, :, :] = cleaned[top_strip:top_strip + 1, :, :]
            if 0 < left_strip < w_img:
                cleaned[:, :left_strip, :] = cleaned[:, left_strip:left_strip + 1, :]
            return cleaned

        def _load_fullscreen_backdrop(asset_name: str, file_name: str) -> dict | None:
            file_path = os.path.join(sprite_path, file_name)
            if not os.path.exists(file_path):
                return None
            backdrop_np = np.load(file_path)
            backdrop_rgba = _normalize_to_rgba_u8(backdrop_np, asset_name)

            bh, bw = int(backdrop_rgba.shape[0]), int(backdrop_rgba.shape[1])
            if (bh, bw) != (h, w):
                raise ValueError(f"{asset_name} has unexpected size {(bh, bw)}; expected {(h, w)}")

            backdrop_rgba = _cleanup_leading_black_edges(backdrop_rgba)
            return {
                'name': asset_name,
                'type': 'single',
                'data': jnp.asarray(backdrop_rgba, dtype=jnp.uint8),
            }

        def _load_trimmed_sprite(
            file_name: str,
            crop_box: tuple[int, int, int, int],
            black_transparent: bool = True,
        ) -> jnp.ndarray:
            """Load a full-screen sprite .npy, crop to *crop_box* (y0,x0,y1,x1),
            optionally mark black pixels transparent, and return RGBA."""
            file_path = os.path.join(sprite_path, file_name)
            img_np = np.load(file_path)
            img_rgba = _normalize_to_rgba_u8(img_np, file_name)
            y0, x0, y1, x1 = crop_box
            cropped = img_rgba[y0:y1, x0:x1].copy()
            if black_transparent:
                is_black = np.all(cropped[:, :, :3] == 0, axis=2)
                cropped[is_black, 3] = 0
            return jnp.asarray(cropped, dtype=jnp.uint8)

        # Load trimmed log / ladder sprites (bounding boxes measured from sprite files)
        log_left_rgba = _load_trimmed_sprite('log_left.npy', (118, 25, 132, 31))
        log_right_rgba = _load_trimmed_sprite('log_right.npy', (119, 24, 133, 30))
        ladder_rgba = _load_trimmed_sprite('ladder.npy', (117, 72, 178, 88), black_transparent=False)
        ladder_with_pits_rgba = _load_trimmed_sprite(
            'ladder_with_pits.npy', (119, 46, 178, 114), black_transparent=False,
        )

        converted_asset_config = []
        for asset in asset_config:
            if (
                asset.get('type') == 'single'
                and asset.get('name') in {
                    'background_tree_variant_0',
                    'background_tree_variant_1',
                    'background_tree_variant_2',
                    'background_tree_variant_3',
                    'backdrop_crocodilepit_and_rope',
                }
                and 'file' in asset
            ):
                converted_backdrop = _load_fullscreen_backdrop(asset['name'], asset['file'])
                if converted_backdrop is not None:
                    converted_asset_config.append(converted_backdrop)
            elif asset.get('name') == 'scorpion_left' and asset.get('type') == 'group' and 'files' in asset:
                converted_asset_config.append(
                    {
                        'name': 'scorpion_left',
                        'type': 'group',
                        'data': _load_alpha_group(asset['files'], (255, 255, 255)),
                    }
                )
            elif asset.get('name') == 'scorpion_right' and asset.get('type') == 'group' and 'files' in asset:
                converted_asset_config.append(
                    {
                        'name': 'scorpion_right',
                        'type': 'group',
                        'data': _load_alpha_group(asset['files'], (255, 255, 255)),
                    }
                )
            else:
                converted_asset_config.append(asset)
        asset_config = converted_asset_config

        def _color_swatch(rgb: tuple[int, int, int]) -> jnp.ndarray:
            return jnp.array([rgb[0], rgb[1], rgb[2], 255], dtype=jnp.uint8).reshape(1, 1, 4)

        asset_config.extend(
            [
                {'name': 'color_wood', 'type': 'procedural', 'data': _color_swatch((110, 70, 25))},
                {'name': 'color_fire', 'type': 'procedural', 'data': _color_swatch((255, 120, 0))},
                {'name': 'color_snake', 'type': 'procedural', 'data': _color_swatch((20, 200, 0))},
                {'name': 'color_hole', 'type': 'procedural', 'data': _color_swatch((0, 0, 0))},
                {'name': 'log_left_sprite', 'type': 'single', 'data': log_left_rgba},
                {'name': 'log_right_sprite', 'type': 'single', 'data': log_right_rgba},
                {'name': 'ladder_sprite', 'type': 'single', 'data': ladder_rgba},
                {'name': 'ladder_with_pits_sprite', 'type': 'single', 'data': ladder_with_pits_rgba},
            ]
        )

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        self.WOOD_ID = self.SHAPE_MASKS['color_wood'][0, 0].astype(self.BACKGROUND.dtype)
        self.FIRE_ID = self.SHAPE_MASKS['color_fire'][0, 0].astype(self.BACKGROUND.dtype)
        self.SNAKE_ID = self.SHAPE_MASKS['color_snake'][0, 0].astype(self.BACKGROUND.dtype)
        self.HOLE_ID = self.SHAPE_MASKS['color_hole'][0, 0].astype(self.BACKGROUND.dtype)

        transparent_pixel = jnp.full((1, 1), int(self.jr.TRANSPARENT_ID), dtype=self.BACKGROUND.dtype)
        self.BACKGROUND_TREE_VARIANT_0 = self.SHAPE_MASKS.get('background_tree_variant_0', transparent_pixel)
        self.BACKGROUND_TREE_VARIANT_1 = self.SHAPE_MASKS.get('background_tree_variant_1', self.BACKGROUND_TREE_VARIANT_0)
        self.BACKGROUND_TREE_VARIANT_2 = self.SHAPE_MASKS.get('background_tree_variant_2', self.BACKGROUND_TREE_VARIANT_1)
        self.BACKGROUND_TREE_VARIANT_3 = self.SHAPE_MASKS.get('background_tree_variant_3', self.BACKGROUND_TREE_VARIANT_2)
        self.BACKDROP_CROCODILEPIT_AND_ROPE = self.SHAPE_MASKS.get('backdrop_crocodilepit_and_rope', transparent_pixel)
        wall_mask = self.SHAPE_MASKS.get('wall', transparent_pixel)
        self.WALL_MASK = wall_mask[0] if wall_mask.ndim == 3 else wall_mask
        wall_extension_rows = min(3, int(self.WALL_MASK.shape[0]))
        self.WALL_RENDER_MASK = jnp.concatenate(
            [self.WALL_MASK[:wall_extension_rows], self.WALL_MASK],
            axis=0,
        )
        wall_red_id = jnp.asarray(self.COLOR_TO_ID.get((167, 26, 26), int(self.WOOD_ID)), dtype=self.WALL_RENDER_MASK.dtype)
        wall_top_row = self.WALL_RENDER_MASK[0]
        self.WALL_RENDER_MASK = self.WALL_RENDER_MASK.at[0].set(
            jnp.where(
                wall_top_row != jnp.asarray(self.jr.TRANSPARENT_ID, dtype=self.WALL_RENDER_MASK.dtype),
                wall_red_id,
                wall_top_row,
            )
        )

        # Log / ladder sprite masks (trimmed from full-screen captures)
        _log_left = self.SHAPE_MASKS.get('log_left_sprite', transparent_pixel)
        self.LOG_LEFT_MASK = _log_left[0] if _log_left.ndim == 3 else _log_left
        _log_right = self.SHAPE_MASKS.get('log_right_sprite', transparent_pixel)
        self.LOG_RIGHT_MASK = _log_right[0] if _log_right.ndim == 3 else _log_right
        _ladder = self.SHAPE_MASKS.get('ladder_sprite', transparent_pixel)
        self.LADDER_SPRITE_MASK = _ladder[0] if _ladder.ndim == 3 else _ladder
        _lwp = self.SHAPE_MASKS.get('ladder_with_pits_sprite', transparent_pixel)
        self.LADDER_WITH_PITS_MASK = _lwp[0] if _lwp.ndim == 3 else _lwp

        self.HAS_BACKDROP_CROCODILEPIT_AND_ROPE = jnp.array(
            'backdrop_crocodilepit_and_rope' in self.SHAPE_MASKS, dtype=jnp.bool_
        )

        def _ensure_3d(mask_stack: jnp.ndarray) -> jnp.ndarray:
            return mask_stack[None, :, :] if mask_stack.ndim == 2 else mask_stack

        def _pad_to(mask_stack: jnp.ndarray, target_h: int, target_w: int) -> jnp.ndarray:
            mask_stack = _ensure_3d(mask_stack)
            pad_h = max(0, target_h - int(mask_stack.shape[1]))
            pad_w = max(0, target_w - int(mask_stack.shape[2]))
            return jnp.pad(
                mask_stack,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=int(self.jr.TRANSPARENT_ID),
            )

        def _downscale_mask_stack(mask_stack: jnp.ndarray, scale: float) -> jnp.ndarray:
            masks_np = np.array(_ensure_3d(mask_stack))
            frames_out = []
            for frame in masks_np:
                h_frame, w_frame = frame.shape
                new_h = max(1, int(round(h_frame * scale)))
                new_w = max(1, int(round(w_frame * scale)))
                y_idx = np.clip(np.round(np.arange(new_h) / scale).astype(np.int32), 0, h_frame - 1)
                x_idx = np.clip(np.round(np.arange(new_w) / scale).astype(np.int32), 0, w_frame - 1)
                resized = frame[y_idx][:, x_idx]
                frames_out.append(resized.astype(np.uint8))
            return jnp.asarray(np.stack(frames_out, axis=0), dtype=jnp.uint8)

        def _shrink_mask_stack(mask_stack: jnp.ndarray, border: int = 1) -> jnp.ndarray:
            mask_stack = _ensure_3d(mask_stack)
            if int(mask_stack.shape[1]) <= border * 2 or int(mask_stack.shape[2]) <= border * 2:
                return mask_stack
            return mask_stack[:, border:-border, border:-border]

        def _trim_transparent_border_stack(mask_stack: jnp.ndarray) -> jnp.ndarray:
            """Trim only fully-transparent borders per frame (safe, non-destructive).

            This avoids cropping real scorpion pixels (which the fixed 1px shrink can do)
            while still normalizing padding so frames/directions align.
            """
            masks_np = np.array(_ensure_3d(mask_stack))
            tid = int(self.jr.TRANSPARENT_ID)
            trimmed = []
            max_h = 1
            max_w = 1
            for frame in masks_np:
                ys, xs = np.where(frame != tid)
                if ys.size == 0:
                    cropped = frame
                else:
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    x0, x1 = int(xs.min()), int(xs.max()) + 1
                    cropped = frame[y0:y1, x0:x1]
                cropped = cropped.astype(np.uint8)
                trimmed.append(cropped)
                max_h = max(max_h, int(cropped.shape[0]))
                max_w = max(max_w, int(cropped.shape[1]))

            padded = []
            for cropped in trimmed:
                pad_h = max_h - int(cropped.shape[0])
                pad_w = max_w - int(cropped.shape[1])
                padded.append(
                    np.pad(
                        cropped,
                        ((0, pad_h), (0, pad_w)),
                        mode='constant',
                        constant_values=tid,
                    )
                )
            return jnp.asarray(np.stack(padded, axis=0), dtype=jnp.uint8)

        sprite_scale = 0.80
        scorpion_scale = 0.60

        def _crop_and_pad_scorpion_stacks(
            left_stack: jnp.ndarray,
            right_stack: jnp.ndarray,
            margin: int = 1,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Crop both scorpion directions to the same union bbox (plus margin), then pad.

            This keeps left/right + both animation frames anchored identically and avoids
            tail loss from over-trimming.
            """
            left_np = np.array(_ensure_3d(left_stack))
            right_np = np.array(_ensure_3d(right_stack))
            tid = int(self.jr.TRANSPARENT_ID)

            # Ensure both stacks have identical spatial dimensions before union bbox.
            max_h = max(int(left_np.shape[1]), int(right_np.shape[1]))
            max_w = max(int(left_np.shape[2]), int(right_np.shape[2]))
            if int(left_np.shape[1]) != max_h or int(left_np.shape[2]) != max_w:
                left_np = np.pad(
                    left_np,
                    ((0, 0), (0, max_h - int(left_np.shape[1])), (0, max_w - int(left_np.shape[2]))),
                    mode='constant',
                    constant_values=tid,
                )
            if int(right_np.shape[1]) != max_h or int(right_np.shape[2]) != max_w:
                right_np = np.pad(
                    right_np,
                    ((0, 0), (0, max_h - int(right_np.shape[1])), (0, max_w - int(right_np.shape[2]))),
                    mode='constant',
                    constant_values=tid,
                )
            combined = np.concatenate([left_np, right_np], axis=0)
            occupied = np.any(combined != tid, axis=0)
            ys, xs = np.where(occupied)

            if ys.size == 0 or xs.size == 0:
                left_out = jnp.asarray(left_np.astype(np.uint8), dtype=jnp.uint8)
                right_out = jnp.asarray(right_np.astype(np.uint8), dtype=jnp.uint8)
                h = max(int(left_out.shape[1]), int(right_out.shape[1]))
                w = max(int(left_out.shape[2]), int(right_out.shape[2]))
                return _pad_to(left_out, h, w), _pad_to(right_out, h, w)

            h_full = int(combined.shape[1])
            w_full = int(combined.shape[2])
            y0 = max(int(ys.min()) - margin, 0)
            y1 = min(int(ys.max()) + margin + 1, h_full)
            x0 = max(int(xs.min()) - margin, 0)
            x1 = min(int(xs.max()) + margin + 1, w_full)

            left_crop = left_np[:, y0:y1, x0:x1].astype(np.uint8)
            right_crop = right_np[:, y0:y1, x0:x1].astype(np.uint8)
            left_out = jnp.asarray(left_crop, dtype=jnp.uint8)
            right_out = jnp.asarray(right_crop, dtype=jnp.uint8)
            h = max(int(left_out.shape[1]), int(right_out.shape[1]))
            w = max(int(left_out.shape[2]), int(right_out.shape[2]))
            return _pad_to(left_out, h, w), _pad_to(right_out, h, w)

        harry_idle = _downscale_mask_stack(self.SHAPE_MASKS['harry_idle'], sprite_scale)
        harry_run = _downscale_mask_stack(self.SHAPE_MASKS['harry_run'], sprite_scale)
        harry_climb = _downscale_mask_stack(self.SHAPE_MASKS['harry_climb'], sprite_scale)
        harry_jump = _downscale_mask_stack(self.SHAPE_MASKS['harry_jump'], sprite_scale)

        max_h = max(int(harry_idle.shape[1]), int(harry_run.shape[1]), int(harry_climb.shape[1]), int(harry_jump.shape[1]))
        max_w = max(int(harry_idle.shape[2]), int(harry_run.shape[2]), int(harry_climb.shape[2]), int(harry_jump.shape[2]))

        def _pad_and_offset(name: str, masks: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            masks_3d = _ensure_3d(masks)
            extra_pad_w = jnp.int32(max_w - int(masks_3d.shape[2]))
            extra_pad_h = jnp.int32(max_h - int(masks_3d.shape[1]))
            base_offset = self.FLIP_OFFSETS.get(name, jnp.array([0, 0], dtype=jnp.int32))
            base_offset = jnp.round(base_offset.astype(jnp.float32) * jnp.asarray(sprite_scale, dtype=jnp.float32)).astype(jnp.int32)
            return _pad_to(masks_3d, max_h, max_w), base_offset + jnp.array([extra_pad_w, extra_pad_h], dtype=jnp.int32)

        self.HARRY_IDLE_MASKS, self.HARRY_IDLE_FLIP_OFFSET = _pad_and_offset('harry_idle', harry_idle)
        self.HARRY_RUN_MASKS, self.HARRY_RUN_FLIP_OFFSET = _pad_and_offset('harry_run', harry_run)
        self.HARRY_CLIMB_MASKS, self.HARRY_CLIMB_FLIP_OFFSET = _pad_and_offset('harry_climb', harry_climb)
        self.HARRY_JUMP_MASKS, self.HARRY_JUMP_FLIP_OFFSET = _pad_and_offset('harry_jump', harry_jump)
        scorpion_left = _downscale_mask_stack(self.SHAPE_MASKS['scorpion_left'], scorpion_scale)
        scorpion_right = _downscale_mask_stack(self.SHAPE_MASKS['scorpion_right'], scorpion_scale)
        scorpion_left, scorpion_right = _crop_and_pad_scorpion_stacks(scorpion_left, scorpion_right, margin=1)
        scorpion_h = max(int(scorpion_left.shape[1]), int(scorpion_right.shape[1]))
        scorpion_w = max(int(scorpion_left.shape[2]), int(scorpion_right.shape[2]))
        self.SCORPION_LEFT_MASKS = _pad_to(scorpion_left, scorpion_h, scorpion_w)
        self.SCORPION_RIGHT_MASKS = _pad_to(scorpion_right, scorpion_h, scorpion_w)
        self.TREE_VARIANT_TO_ASSET_IDX = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PitfallState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        rb = state.room_byte.astype(jnp.uint8)
        pt = pit_code_u8(rb)
        tree_variant = tree_variant_u8(rb)

        tree_bg_asset_idx = self.TREE_VARIANT_TO_ASSET_IDX[tree_variant.astype(jnp.int32)]

        def _render_tree_variant_0(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                jnp.int32(0),
                jnp.int32(0),
                self.BACKGROUND_TREE_VARIANT_0,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            )

        def _render_tree_variant_1(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                jnp.int32(0),
                jnp.int32(0),
                self.BACKGROUND_TREE_VARIANT_1,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            )

        def _render_tree_variant_2(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                jnp.int32(0),
                jnp.int32(0),
                self.BACKGROUND_TREE_VARIANT_2,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            )

        def _render_tree_variant_3(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                jnp.int32(0),
                jnp.int32(0),
                self.BACKGROUND_TREE_VARIANT_3,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            )

        raster = lax.switch(
            tree_bg_asset_idx,
            (
                _render_tree_variant_0,
                _render_tree_variant_1,
                _render_tree_variant_2,
                _render_tree_variant_3,
            ),
            raster,
        )

        has_ladder = (pt == jnp.uint8(0)) | (pt == jnp.uint8(1))
        has_scorpion = has_scorpion_from_room_byte(rb)
        has_croc_rope_backdrop = (
            (pt == jnp.uint8(0b100))
            & has_vine_from_room_byte(rb)
            & self.HAS_BACKDROP_CROCODILEPIT_AND_ROPE
        )

        raster = lax.cond(
            has_croc_rope_backdrop,
            lambda r: self.jr.render_at_clipped(
                r,
                jnp.int32(0),
                jnp.int32(0),
                self.BACKDROP_CROCODILEPIT_AND_ROPE,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        # ---- Underground elements: holes + ladder sprites ----
        has_simple_ladder = pt == jnp.uint8(0)
        has_ladder_with_pits = pt == jnp.uint8(1)

        ladder_x = self.ladder_x_px.astype(jnp.int32)
        ladder_w = jnp.int32(self.consts.ladder_width)
        hole_w = jnp.int32(self.consts.hole_width)
        hole_top = jnp.int32(int(self.consts.ground_y))
        hole_h = jnp.int32(max(0, int(self.consts.underground_y) - int(self.consts.ground_y)))

        # Center hole for standalone ladder – fill shaft below ground,
        # then the ladder sprite overlays with rungs + hole opening.
        ladder_hole_pos = jnp.where(
            has_simple_ladder,
            jnp.array([ladder_x, hole_top], dtype=jnp.int32),
            jnp.array([-1, -1], dtype=jnp.int32),
        )
        ladder_hole_size = jnp.array([ladder_w, hole_h], dtype=jnp.int32)
        raster = self.jr.draw_rects(raster, ladder_hole_pos[None, :], ladder_hole_size[None, :], int(self.HOLE_ID))

        # Simple ladder sprite (center hole drawn above, sprite only adds the ladder)
        ladder_sprite_top = jnp.int32(int(self.consts.ground_y) - 13)
        raster = lax.cond(
            has_simple_ladder,
            lambda r: self.jr.render_at_clipped(
                r,
                ladder_x,
                ladder_sprite_top,
                self.LADDER_SPRITE_MASK,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        # Ladder-with-pits sprite (includes left hole + ladder + right hole)
        # Fixed offset 26: ladder structure starts at sprite col 26
        lwp_x = ladder_x - jnp.int32(26)
        lwp_top = jnp.int32(int(self.consts.ground_y) - 11)
        raster = lax.cond(
            has_ladder_with_pits,
            lambda r: self.jr.render_at_clipped(
                r,
                lwp_x,
                lwp_top,
                self.LADDER_WITH_PITS_MASK,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        wall_side_bit = (rb >> jnp.uint8(7)) & jnp.uint8(1)
        has_wall = has_simple_ladder | has_ladder_with_pits
        wall_x = jnp.where(wall_side_bit == jnp.uint8(1), self.right_wall_x_px, self.left_wall_x_px).astype(jnp.int32)
        wall_h = jnp.int32(int(self.WALL_RENDER_MASK.shape[0]))
        wall_top = jnp.int32(int(self.consts.underground_y)) - wall_h
        draw_wall_sprite = has_wall

        has_logs, logs_are_rolling, log_count, log_xs, has_fireplace, has_snake = room_hazards_from_room_byte(rb)

        total_frames = jnp.int32(self.consts.initial_time_seconds * self.consts.fps)
        frames_elapsed = jnp.maximum(total_frames - state.time_left.astype(jnp.int32), jnp.int32(0))
        frames_elapsed = frames_elapsed * state.timer_started.astype(jnp.int32)

        W = jnp.int32(self.consts.screen_width)
        speed = jnp.int32(1)
        direction = jnp.int32(-1)
        dx = jnp.mod(frames_elapsed * speed * direction, W)
        moving_centers = jnp.mod(log_xs + dx, W)
        log_centers = jnp.where(logs_are_rolling, moving_centers, log_xs)

        wood_top_static = int(self.consts.ground_y - self.consts.wood_h + self.consts.wood_y_offset)
        wood_top = jnp.int32(wood_top_static)
        log_half_w = jnp.int32(self.consts.wood_w // 2)

        # Moving-log animation: alternate sprite + 1px vertical bobble
        log_anim_phase = jnp.mod(frames_elapsed // jnp.int32(8), jnp.int32(2))
        use_right_frame = logs_are_rolling & (log_anim_phase == jnp.int32(1))
        bobble_y = jnp.where(logs_are_rolling, log_anim_phase, jnp.int32(0))

        log_mask = jnp.where(use_right_frame, self.LOG_RIGHT_MASK, self.LOG_LEFT_MASK)

        def _draw_logs(r: jnp.ndarray) -> jnp.ndarray:
            def body(i, rr):
                active_i = jnp.int32(i) < log_count
                cx = log_centers[i].astype(jnp.int32)
                x = cx - log_half_w
                y = wood_top + bobble_y
                return lax.cond(
                    active_i,
                    lambda rr: self.jr.render_at_clipped(
                        rr, x, y, log_mask,
                        flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                        flip_offset=jnp.array([0, 0], dtype=jnp.int32),
                    ),
                    lambda rr: rr,
                    rr,
                )
            return lax.fori_loop(0, 3, body, r)

        fire_x_center = jnp.int32(132)
        fire_w = jnp.int32(self.consts.fire_w)
        fire_h = jnp.int32(self.consts.fire_h)
        fire_top = jnp.int32(int(self.consts.ground_y - self.consts.fire_h + self.consts.fire_y_offset))
        fire_left = jnp.mod(fire_x_center - (fire_w // jnp.int32(2)), W)
        fire_pos = jnp.where(has_fireplace, jnp.array([fire_left, fire_top], dtype=jnp.int32), jnp.array([-1, -1], dtype=jnp.int32))
        fire_size = jnp.array([fire_w, fire_h], dtype=jnp.int32)
        raster = self.jr.draw_rects(raster, fire_pos[None, :], fire_size[None, :], int(self.FIRE_ID))

        snake_x_center = jnp.int32(134)
        snake_w = jnp.int32(self.consts.snake_w)
        snake_h = jnp.int32(self.consts.snake_h)
        snake_top = jnp.int32(int(self.consts.ground_y - self.consts.snake_h))
        snake_left = jnp.mod(snake_x_center - (snake_w // jnp.int32(2)), W)
        snake_pos = jnp.where(has_snake, jnp.array([snake_left, snake_top], dtype=jnp.int32), jnp.array([-1, -1], dtype=jnp.int32))
        snake_size = jnp.array([snake_w, snake_h], dtype=jnp.int32)
        raster = self.jr.draw_rects(raster, snake_pos[None, :], snake_size[None, :], int(self.SNAKE_ID))

        scorpion_anim_idx = jnp.mod(state.scorpion_anim_idx.astype(jnp.int32), jnp.int32(2)).astype(jnp.int32)
        scorpion_facing_right = state.scorpion_facing_right.astype(jnp.bool_)
        scorpion_mask = jnp.where(
            scorpion_facing_right,
            self.SCORPION_RIGHT_MASKS[scorpion_anim_idx],
            self.SCORPION_LEFT_MASKS[scorpion_anim_idx],
        )
        scorpion_h_sprite = jnp.int32(scorpion_mask.shape[0])
        scorpion_top = jnp.int32(self.consts.underground_y) - scorpion_h_sprite + jnp.int32(1) + jnp.int32(self.consts.scorpion_y_offset)

        raster = lax.cond(
            has_scorpion,
            lambda r: self.jr.render_at_clipped(
                r,
                state.scorpion_x.astype(jnp.int32),
                scorpion_top,
                scorpion_mask,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        moving = jnp.abs(state.player_vx) > jnp.asarray(0.0, dtype=jnp.float32)
        # Use facing_left for idle/jump sprites so Harry remembers direction
        flip = jnp.where(moving, state.player_vx < 0.0, state.facing_left)

        steady_cycle = jnp.array([1, 0, 1, 3], dtype=jnp.int32)  # running2 dominates; avoids running3; includes running4
        steady_run_idx = steady_cycle[jnp.mod(state.run_anim_phase.astype(jnp.int32), jnp.int32(4))]
        startup_run_idx = jnp.where(
            state.run_startup_timer > jnp.int32(2),
            jnp.int32(2),  # running3
            jnp.int32(3),  # running4
        )
        run_idx = jnp.where(
            state.run_startup_timer > jnp.int32(0),
            startup_run_idx,
            steady_run_idx,
        ).astype(jnp.int32)
        step_idx = state.ladder_step_idx.astype(jnp.int32)
        climb_idx = jnp.mod(step_idx, jnp.int32(2)).astype(jnp.int32)
        jump_idx = jnp.where(state.player_vy < jnp.asarray(0.0, dtype=jnp.float32), jnp.int32(0), jnp.int32(1))

        def _use_climb(_):
            return self.HARRY_CLIMB_MASKS[climb_idx], self.HARRY_CLIMB_FLIP_OFFSET

        def _use_jump(_):
            return self.HARRY_JUMP_MASKS[jump_idx], self.HARRY_JUMP_FLIP_OFFSET

        def _use_fall(_):
            return self.HARRY_JUMP_MASKS[jnp.int32(1)], self.HARRY_JUMP_FLIP_OFFSET

        def _use_run(_):
            return self.HARRY_RUN_MASKS[run_idx], self.HARRY_RUN_FLIP_OFFSET

        def _use_run1(_):
            return self.HARRY_RUN_MASKS[jnp.int32(0)], self.HARRY_RUN_FLIP_OFFSET

        def _use_idle(_):
            return self.HARRY_IDLE_MASKS[jnp.int32(0)], self.HARRY_IDLE_FLIP_OFFSET

        def _non_ladder(_):
            # During airtime (self-initiated jump or any non-ladder airborne state),
            # lock Harry to harryrunning1.
            return lax.cond(
                ~state.on_ground,
                _use_run1,
                lambda __: lax.cond(
                    state.touching_wood,
                    _use_fall,
                    lambda ___: lax.cond(moving, _use_run, _use_idle, None),
                    None,
                ),
                None,
            )

        harry_mask, flip_offset = lax.cond(state.on_ladder, _use_climb, _non_ladder, None)

        harry_h = jnp.int32(harry_mask.shape[0])
        y_top = state.player_y.astype(jnp.int32) - harry_h + jnp.int32(1)
        underground_tune = jnp.where(
            state.current_ground_y == jnp.asarray(self.consts.underground_y, dtype=jnp.float32),
            jnp.int32(int(self.consts.underground_harry_y_tune)),
            jnp.int32(0),
        )
        y_top = y_top + jnp.int32(int(self.consts.harry_y_tune)) + underground_tune

        def _draw_harry(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                state.player_x.astype(jnp.int32),
                y_top,
                harry_mask,
                flip_horizontal=flip,
                flip_offset=flip_offset,
            )

        # Save raster before Harry+logs for lip occlusion (lip only covers
        # ladder area so logs outside the lip bbox are unaffected).
        raster_base = raster

        raster = lax.cond(
            state.touching_wood,
            lambda r: _draw_harry(lax.cond(has_logs, _draw_logs, lambda rr: rr, r)),
            lambda r: lax.cond(has_logs, _draw_logs, lambda rr: rr, _draw_harry(r)),
            raster,
        )

        underground_respawn_reveal = (
            (state.respawn_phase == jnp.int32(2))
            & (state.respawn_target_ground_y == jnp.asarray(self.consts.underground_y, dtype=jnp.float32))
        )
        reveal_y = jnp.int32(
            int(
                self.consts.ground_y
                + self.consts.underground_respawn_reveal_from_ground
                + self.consts.underground_respawn_reveal_y_offset
            )
        )

        def _clip_underground_respawn(r: jnp.ndarray) -> jnp.ndarray:
            H, W = r.shape
            yy = jnp.arange(H, dtype=jnp.int32)[:, None]
            xx = jnp.arange(W, dtype=jnp.int32)[None, :]
            harry_w = jnp.int32(harry_mask.shape[1])
            harry_x0 = state.player_x.astype(jnp.int32)
            harry_x1 = harry_x0 + harry_w
            hidden_mask = (
                (xx >= harry_x0)
                & (xx < harry_x1)
                & (yy >= y_top)
                & (yy < jnp.minimum(y_top + harry_h, reveal_y))
            )
            return jnp.where(hidden_mask, raster_base, r)

        raster = lax.cond(
            underground_respawn_reveal,
            _clip_underground_respawn,
            lambda r: r,
            raster,
        )

        raster = lax.cond(
            draw_wall_sprite,
            lambda r: self.jr.render_at_clipped(
                r,
                wall_x,
                wall_top,
                self.WALL_RENDER_MASK,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        # Re-stamp logs into lip region so lip occlusion doesn't erase them
        raster_base = lax.cond(has_logs, _draw_logs, lambda r: r, raster_base)

        # Lip occlusion: restore backdrop pixels in a thin strip around the
        # ladder/pit opening so ladder rungs appear in front of Harry.
        lip_y0 = jnp.int32(int(self.consts.ground_y) - 4)
        lip_y1 = jnp.int32(int(self.consts.ground_y) + 6)
        has_any_ladder = has_simple_ladder | has_ladder_with_pits

        # For simple ladder, lip covers just the ladder width.
        # For ladder_with_pits, lip covers the full sprite.
        lip_x0 = jnp.where(has_ladder_with_pits, lwp_x, ladder_x)
        lip_x1 = jnp.where(
            has_ladder_with_pits,
            lwp_x + jnp.int32(self.LADDER_WITH_PITS_MASK.shape[1]),
            ladder_x + jnp.int32(self.consts.ladder_width),
        )
        transparent_id = jnp.asarray(self.jr.TRANSPARENT_ID, dtype=raster.dtype)

        def _apply_lip(r: jnp.ndarray) -> jnp.ndarray:
            # For each pixel in the lip bbox, if the base raster (before Harry)
            # is non-transparent, restore it on top of Harry.
            H, W = r.shape
            yy = jnp.arange(H, dtype=jnp.int32)[:, None]
            xx = jnp.arange(W, dtype=jnp.int32)[None, :]
            in_lip = (yy >= lip_y0) & (yy < lip_y1) & (xx >= lip_x0) & (xx < lip_x1)
            base_not_transparent = raster_base != transparent_id
            mask = in_lip & base_not_transparent
            return jnp.where(mask, raster_base, r)

        raster = lax.cond(has_any_ladder, _apply_lip, lambda r: r, raster)

        frame = self.jr.render_from_palette(raster, self.PALETTE)

        font = HUD_FONT_16

        def _draw_digits(
            f: jnp.ndarray,
            digits: jnp.ndarray,
            top: int,
            left: int,
            spacing: int,
            color: jnp.ndarray,
        ) -> jnp.ndarray:
            top_i32 = jnp.int32(top)
            left0_i32 = jnp.int32(left)
            spacing_i32 = jnp.int32(spacing)
            channels = f.shape[2]

            color_u8 = color.astype(jnp.uint8)
            if channels == 1:
                gray = jnp.mean(color_u8.astype(jnp.float32)).astype(jnp.uint8)
                draw_color = jnp.array([gray], dtype=jnp.uint8)
            else:
                draw_color = color_u8

            def body(i, frame_in):
                d = digits[i].astype(jnp.int32)
                glyph = font[d].astype(jnp.bool_)
                start = (top_i32, left0_i32 + jnp.int32(i) * spacing_i32, jnp.int32(0))
                region = lax.dynamic_slice(frame_in, start, (5, 3, channels))
                new_region = jnp.where(glyph[:, :, None], draw_color[None, None, :], region)
                return lax.dynamic_update_slice(frame_in, new_region, start)

            return lax.fori_loop(0, digits.shape[0], body, f)

        digit_spacing = 4

        score_color = jnp.array([40, 220, 40], dtype=jnp.uint8)
        lives_color = jnp.array([240, 200, 40], dtype=jnp.uint8)
        time_color = jnp.array([40, 200, 240], dtype=jnp.uint8)
        debug_color = jnp.array([180, 180, 180], dtype=jnp.uint8)

        score_row = 2
        timer_row = 9
        timer_x = 20

        score_digits = self.jr.int_to_digits(state.score.astype(jnp.int32), max_digits=4)
        lives_digits = self.jr.int_to_digits(state.lives_left.astype(jnp.int32), max_digits=1)
        screen_digits = self.jr.int_to_digits(state.screen_id.astype(jnp.int32), max_digits=3)

        time_seconds = state.time_left.astype(jnp.int32) // jnp.int32(self.consts.fps)
        minutes = time_seconds // jnp.int32(60)
        seconds = time_seconds - minutes * jnp.int32(60)
        mm_digits = self.jr.int_to_digits(minutes, max_digits=2)
        ss_digits = self.jr.int_to_digits(seconds, max_digits=2)

        rb_u8 = state.room_byte.astype(jnp.uint8)
        rb_hi = ((rb_u8 >> jnp.uint8(4)) & jnp.uint8(0xF)).astype(jnp.int32)
        rb_lo = (rb_u8 & jnp.uint8(0xF)).astype(jnp.int32)
        rb_hex = jnp.stack([rb_hi, rb_lo]).astype(jnp.int32)

        pit_d = pit_code_u8(rb_u8).astype(jnp.int32)
        obj_d = obj_code_u8(rb_u8).astype(jnp.int32)
        wall_d = wall_side_u8(rb_u8).astype(jnp.int32)
        tree_d = tree_variant_u8(rb_u8).astype(jnp.int32)
        pit_digits = self.jr.int_to_digits(pit_d, max_digits=1)
        obj_digits = self.jr.int_to_digits(obj_d, max_digits=1)
        wall_digits = self.jr.int_to_digits(wall_d, max_digits=1)
        tree_digits = self.jr.int_to_digits(tree_d, max_digits=1)

        # Old layout:
        # score at (2, 20); lives at (9, 4); time at (9, 20)
        # screen_id at (2, 120); room_byte hex at (2, 90)
        # pit/obj/wall/tree at (9, 90/98/106/114)

        frame = _draw_digits(frame, score_digits, score_row, timer_x, digit_spacing, score_color)
        frame = _draw_digits(frame, lives_digits, timer_row, 4, digit_spacing, lives_color)

        frame = _draw_digits(frame, mm_digits, timer_row, timer_x, digit_spacing, time_color)
        if frame.shape[2] == 1:
            time_dot_color = jnp.array([jnp.mean(time_color.astype(jnp.float32)).astype(jnp.uint8)], dtype=jnp.uint8)
        else:
            time_dot_color = time_color
        colon_x = timer_x + 2 * digit_spacing - 1
        frame = frame.at[timer_row + 1, colon_x, :].set(time_dot_color)
        frame = frame.at[timer_row + 3, colon_x, :].set(time_dot_color)
        frame = _draw_digits(frame, ss_digits, timer_row, timer_x + 2 * digit_spacing + 2, digit_spacing, time_color)

        frame = _draw_digits(frame, screen_digits, score_row, 120, digit_spacing, debug_color)
        frame = _draw_digits(frame, rb_hex, score_row, 90, digit_spacing, debug_color)

        frame = _draw_digits(frame, pit_digits, timer_row, 90, digit_spacing, debug_color)
        frame = _draw_digits(frame, obj_digits, timer_row, 98, digit_spacing, debug_color)
        frame = _draw_digits(frame, wall_digits, timer_row, 106, digit_spacing, debug_color)
        frame = _draw_digits(frame, tree_digits, timer_row, 114, digit_spacing, debug_color)

        return frame
