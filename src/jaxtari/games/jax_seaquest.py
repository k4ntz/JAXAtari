import os
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct

import jaxtari.spaces as spaces
from jaxtari.environment import JaxEnvironment, JaxtariAction as Action, ObjectObservation
from jaxtari.renderers import JAXGameRenderer
from jaxtari.rendering import jax_rendering_utils as render_utils
from jaxtari.modification import AutoDerivedConstants

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Seaquest.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'bg/1.npy'},
        {'name': 'player_sub', 'type': 'group', 'files': ['player_sub/1.npy', 'player_sub/2.npy', 'player_sub/3.npy']},
        {'name': 'diver', 'type': 'group', 'files': ['diver/1.npy', 'diver/2.npy']},
        {'name': 'shark_base', 'type': 'group', 'files': ['shark/1.npy', 'shark/2.npy']},
        {'name': 'enemy_sub', 'type': 'group', 'files': ['enemy_sub/1.npy', 'enemy_sub/2.npy', 'enemy_sub/3.npy']},
        {'name': 'player_torp', 'type': 'single', 'file': 'player_torp/1.npy'},
        {'name': 'enemy_torp', 'type': 'single', 'file': 'enemy_torp/1.npy'},
        {'name': 'life_indicator', 'type': 'single', 'file': 'life_indicator/1.npy'},
        {'name': 'diver_indicator', 'type': 'single', 'file': 'diver_indicator/1.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/{}.npy'},
    )


class SeaquestConstants(AutoDerivedConstants):
    SCREEN_WIDTH: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(160))
    SCREEN_HEIGHT: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(210))
    # Colors
    BACKGROUND_COLOR: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 0, 139]))  # Dark blue for water
    PLAYER_COLOR: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([187, 187, 53]))  # Yellow for player sub
    DIVER_COLOR: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([66, 72, 200]))  # Pink for divers
    SHARK_DIFFICULTY_COLORS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(
        [
            [92, 186, 92],  # Level 0: Base green
            # Level 1: yellow-olive from ALE frames after first surface
            [160, 171, 79],
            [
                170,
                92,
                170,
            ],  # Level 2: Purple (adjusted from original ROM COLOR_KILLER_SHARK_02)
            [213, 92, 130],  # Level 3: Pink (adjusted from original ROM)
            [186, 92, 92],  # Level 4: Red (adjusted from original ROM)
        ]
    ))
    ENEMY_SUB_COLOR: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([170, 170, 170]))  # Gray for enemy subs
    OXYGEN_BAR_COLOR: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([214, 214, 214, 255]))  # White for oxygen
    OXYGEN_BAR_BG_COLOR: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([163, 57, 21, 255]))  # Reddish background
    # Low-O₂ warning flash (ALE lockstep frames): when oxygen <= threshold the
    # white fill blinks to black 16f off / 16f on (period 32). Flash starts with
    # BLACK on the drain tick that reaches the threshold (step%32 < 16). Measured
    # on seaquest_play01.npz. Disabled while refilling at the surface / init fill.
    # Threshold 16 is 25% of the 64-unit tank (ALE fill width hits 16 then blacks).
    OXYGEN_FLASH_THRESHOLD: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(16, dtype=jnp.int32)
    )
    OXYGEN_FLASH_PERIOD: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(32, dtype=jnp.int32)
    )
    SCORE_COLOR: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([210, 210, 64]))  # Score color
    OXYGEN_TEXT_COLOR: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 0, 0]))  # Black for oxygen text

    # Object sizes and initial positions from RAM state
    PLAYER_SIZE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([16, 11]))  # Width, Height
    # Shrink the *damage* hitbox horizontally only (not missile kills / render).
    # ALE TIA ignores grazing side overlaps with bobbing sharks in the lane gap;
    # vertical size stays full so real overlaps still kill.
    PLAYER_COLLISION_INSET_X: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(3, dtype=jnp.int32)
    )
    DIVER_SIZE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([8, 11]))
    SHARK_SIZE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([8, 7]))
    ENEMY_SUB_SIZE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([8, 11]))
    MISSILE_SIZE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([8, 1]))
    # Missile y = player_y + OFFSET. ALE OC sits at +7; +4 was over-hitting
    # sharks on low bob (false L2 kill → early diver co-spawn / phantom pair).
    PLAYER_MISSILE_Y_OFFSET: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(7, dtype=jnp.int32)
    )
    SHARK_BOB_PHASE_OFFSET: jnp.ndarray = struct.field(
        # Locked to ALE OC on seaquest_play01 (episode frame = step_counter):
        # residual Y matches wave[(step+8)%64]-4 with MAE 0.
        pytree_node=False, default_factory=lambda: jnp.array(8, dtype=jnp.int32)
    )

    PLAYER_START_X: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(76))
    PLAYER_START_Y: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(46))

    X_BORDERS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 160]))
    PLAYER_BOUNDS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[21, 134], [46, 141]]))

    # Maximum number of objects (from MAX_NB_OBJECTS)
    MAX_DIVERS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(4))
    MAX_SHARKS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(12))
    MAX_SUBS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(12))
    MAX_ENEMY_MISSILES: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(4))
    MAX_PLAYER_TORPS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1))
    MAX_SURFACE_SUBS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1))
    MAX_COLLECTED_DIVERS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(6))

    # define object orientations
    FACE_LEFT: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(-1))
    FACE_RIGHT: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1))

    # Lane Y bases from ALE OC (seaquest_play_fresh): shark bob midpoints sit at
    # 68.5/92.5/116.5/140.5 with SHARK_BOB_WAVE mean offset -0.5 → bases 69/93/117/141.
    # Subs and divers share the same fixed lane Ys (no bob). Old [71,95,119,139]
    # was ~+2px high and made stacked sharks look "out of phase" vs nearest-lane residuals.
    SPAWN_POSITIONS_Y: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([69, 93, 117, 141])
    )
    # Subs sit on the lane base (ALE sub y ∈ {69,93,117,141}). Previously 2 with the
    # old shark bases to land near 69/93/117 — keep 0 now that bases match ALE.
    SUBMARINE_Y_OFFSET: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(0))
    # Enemy missiles fire at sub_y+4 → {73,97,121,145} (ALE recording; was 141 on lane 3).
    ENEMY_MISSILE_Y: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([73, 97, 121, 145])
    )
    DIVER_SPAWN_POSITIONS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([69, 93, 117, 141]))

    # Off-screen spawn. Despawn is a timer after on→off (see
    # ENEMY_OFFSCREEN_DESPAWN_FRAMES), not an X threshold — pre-entry spawns at
    # ENEMY_SPAWN_X_* stay alive until they have been on-screen and left again.
    #
    # Symmetric L/R: the lane's *leading* (native) slot always sits
    # ENEMY_SPAWN_EDGE_MARGIN past the entry edge, with other slots packed
    # away from the screen. Same approach time both directions — direction
    # flips no longer change the gap. (Absolute ALE bases -88/215 made the
    # short side depend on which form bit was on.)
    ENEMY_SPAWN_EDGE_MARGIN: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(56, dtype=jnp.int32)
    )
    # Convenience lead X for single-slot helpers (margin past 0 / 160).
    ENEMY_SPAWN_X_FROM_LEFT: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(-56, dtype=jnp.int32)
    )
    ENEMY_SPAWN_X_FROM_RIGHT: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(216, dtype=jnp.int32)
    )
    # Legacy names (unused by movement).
    ENEMY_DESPAWN_X_LEFT: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(0, dtype=jnp.int32)
    )
    ENEMY_DESPAWN_X_RIGHT: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(160, dtype=jnp.int32)
    )
    # Frames to keep a slot after it leaves [0,160]. With lead-symmetric spawn,
    # T=88 matches ALE AFK shark-to-shark mean (~1339f); periods stay flat L/R.
    ENEMY_OFFSCREEN_DESPAWN_FRAMES: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(88, dtype=jnp.int32)
    )
    # Collision only once enough of the sprite is on-screen. ALE recording: 0 kills at
    # x<=2; earliest left-edge kills at kill_x>=4 after ~7-9 frames visible.
    ENEMY_HITTABLE_X_MIN: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(4, dtype=jnp.int32)
    )
    ENEMY_HITTABLE_X_MAX: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(160, dtype=jnp.int32)
    )
    # Wave spacing: ALE in-lane gaps are always 16 (or 32 for 1-0-1). Never 0.
    ENEMY_WAVE_SPACING: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(16, dtype=jnp.int32)
    )
    ENEMY_WAVE_SPACING_GAP: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(32, dtype=jnp.int32)
    )

    # Shark vertical bob from ALE ram[93] over a recorded cycle (dwell 4, 8 at peaks):
    # offset displayed as ram[93]-4. Indexed by step_counter % 64.
    SHARK_BOB_WAVE: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
             5, 5, 5, 5, 6, 6, 6, 6,
             7, 7, 7, 7, 7, 7, 7, 7,
             6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3,
             2, 2, 2, 2, 1, 1, 1, 1],
            dtype=jnp.int32,
        ),
    )

    MISSILE_SPAWN_POSITIONS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([39, 126]))  # Right, Left

    # First wave: ALE lane forms are fixed absolute slots (not direction-flipped).
    # Bottom lane (ALE i=0 / JAX li=3) is always form=4 → slot 0; the other three
    # are form=1 → slot 2. That permanent 32px gap is what makes the bottom lane
    # look "offset" under pure survive.
    FIRST_WAVE_DIRS: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([False, False, False, True])
    )
    # JAX lanes top→bottom (y 69,93,117,141). Slot masks match ALE forms 1/1/1/4.
    LANE_SLOT_MASKS: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [
                [0, 0, 1],  # top    ALE i=3 form=1
                [0, 0, 1],  #        ALE i=2 form=1
                [0, 0, 1],  #        ALE i=1 form=1
                [1, 0, 0],  # bottom ALE i=0 form=4
            ],
            dtype=jnp.int32,
        ),
    )

    # --- SCORING CONSTANTS ---
    # Enemy Scoring: Starts at 20, +10 per rescue, max 90
    SCORE_ENEMY_BASE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(20))
    SCORE_ENEMY_STEP: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(10))
    SCORE_ENEMY_MAX: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(90))

    # Diver Scoring: Starts at 50, +50 per rescue, max 1000
    SCORE_DIVER_BASE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(50))
    SCORE_DIVER_STEP: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(50))
    SCORE_DIVER_MAX: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1000))

    # Oxygen Scoring: Same scaling as Enemies (per user request)
    # Starts at 20 per unit, +10 per rescue, max 90 per unit
    SCORE_OXYGEN_BASE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(20))
    SCORE_OXYGEN_STEP: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(10))
    SCORE_OXYGEN_MAX: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(90))

    # --- SPAWN / DIVER CADENCE ---
    # empty→spawn countdown:
    #   reload = SPAWN_TIMER_RELOAD(+_AFTER_SURVIVE) + DIVER_SPAWN_TIMER_TRIGGER
    #   fire escorts (+ divers on shark waves) when timer == TRIGGER.
    # Kill-clear RELOAD tuned from seed45 policy meter (emulated frames):
    #   Confirmed NOT shooting lag: hittable→clear TTK ALE≈31f / JAX≈33f.
    #   ALE clear→next-hittable mean≈84 / med≈109; JAX@RELOAD=6 was ≈48 / 52.
    #   Global kill-clear only (survive-off keeps AFTER_SURVIVE=6). Sweep:
    #   6→~5920, 38→~4540, 44→~4360, 50→~3160, 63→~3040 (ALE seed45≈3680).
    #   RELOAD=50 ≈ ALE farm gap without overshooting the old fast recycle.
    SPAWN_TIMER_RELOAD: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(50, dtype=jnp.int32)
    )
    # Survive-off / idle lane recycle — independent of kill-clear RELOAD.
    SPAWN_TIMER_RELOAD_AFTER_SURVIVE: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(6, dtype=jnp.int32)
    )
    # Kept for soft-reset / episode_frame bookkeeping; wave arming no longer
    # schedules off this clock (fixed RELOAD above).
    SURVIVE_WAVE_PERIOD: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(661, dtype=jnp.int32)
    )
    SURVIVE_WAVE_EPOCH: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(126, dtype=jnp.int32)
    )
    ENEMY_SPAWN_TIMER_RELOAD: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(50, dtype=jnp.int32)
    )
    DIVER_WAVE_SPAWN_TIMER_RELOAD: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(50, dtype=jnp.int32)
    )
    # Timer value at which diver+escort co-spawn. Opening timers
    # (INITIAL_SPAWN_TIMER_BASE + FIRST_WAVE_LANE_DELAY) should land the first
    # co-spawn with on-screen divers aligned to ALE.
    DIVER_SPAWN_TIMER_TRIGGER: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(128, dtype=jnp.int32)
    )
    # Opening wave only *places* divers in the upper two lanes; all four stay
    # armed and bottom lanes drop on a later shark co-spawn.
    FIRST_WAVE_DIVER_LANES: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1, 1, 0, 0], dtype=jnp.int32),
    )
    # After death / successful surface: each still-armed lane (da==1) rolls this
    # chance to co-spawn on the immediate next shark-trigger tick; losers wait
    # for their lane's next regular clear→shark wave (and get diver_suppress_next).
    DIVER_INSTANT_SPAWN_PROB: float = struct.field(pytree_node=False, default=0.5)
    # Opening countdown from reset. Spawn when timer hits TRIGGER.
    # Shared base for all lanes; per-lane first-only stagger is
    # FIRST_WAVE_LANE_DELAY (not baked into spawn X).
    INITIAL_SPAWN_TIMER_BASE: jnp.ndarray = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array(243, dtype=jnp.int32)
    )
    # Extra frames before each lane's *first* place only (reset / soft-reset).
    # Emulates TIA not lighting all four enemy objects on the same beat —
    # independent of L/R geometry. Bottom (JAX li=3) +60 matches the old
    # INITIAL [277,…,337] / SURFACE_FREEZE [80,…,120] stagger.
    FIRST_WAVE_LANE_DELAY: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([0, 0, 0, 60], dtype=jnp.int32),
    )
    # Mid-game surface refill only — must not apply during opening oxygen fill.
    SURFACE_FREEZE_SPAWN_TIMERS: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([80, 80, 80, 120], dtype=jnp.int32),
    )

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=lambda: _get_default_asset_config())

@struct.dataclass
class SpawnState:
    difficulty: chex.Array  # Current difficulty level (0-7)
    lane_dependent_pattern: chex.Array  # Track waves independently per lane [4 lanes]
    to_be_spawned: (
        chex.Array
    )  # tracks which enemies are still in the spawning cycle [4 lanes * 3 slots] -> necessary due to the spaced out spawning of multiple enemies
    survived: (
        chex.Array
    )  # track if last enemy survived [4 lanes * 3 slots] -> 1 if survived whilst going right, 0 if not, -1 if survived whilst going left
    prev_sub: chex.Array  # Track previous entity type for each lane [4 lanes]
    spawn_timers: chex.Array  # Individual spawn timers per lane [4 lanes]
    diver_array: (
        chex.Array
    )  # Per-lane dedicated diver: 1=armed, 0=collected, -1=swam off (→1 when empty).
    # A collected lane (0) stays dark until *all four* are collected, then every
    # lane rearms together and each drops in front of its next clean shark wave.
    # After death/surface: armed lanes that won the instant-spawn coin flip.
    diver_instant_pending: chex.Array  # (4,) bool/int — force next shark-tick place
    # Remaining shark-diver skips per lane. Instant-roll losers (and death
    # re-arms) set this so the next escort wave is shark-only for that lane.
    diver_suppress_next: chex.Array  # (4,) int
    lane_directions: (
        chex.Array
    )  # Track lane directions for each wave [4 lanes] -> 0 = right, 1 = left
    # Monotonic frame count for survive-wave scheduling. ``step_counter`` wraps
    # at 1024 (anim/bob), so ALE's ~661f global wave beat cannot use it.
    episode_frame: chex.Array
    # Absolute episode_frame when the next survive-off wave should co-spawn.
    next_survive_wave_at: chex.Array
    # Per-slot off-screen despawn countdowns (12 sharks + 12 subs). 0 = idle /
    # not counting; armed to ENEMY_OFFSCREEN_DESPAWN_FRAMES on on→off.
    enemy_offscreen_timers: chex.Array

# Game state container
@struct.dataclass
class SeaquestState:
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array  # 0 for right, 1 for left
    oxygen: chex.Array
    divers_collected: chex.Array
    score: chex.Array
    lives: chex.Array
    spawn_state: SpawnState
    diver_positions: chex.Array  # (4, 3) array for divers
    shark_positions: (
        chex.Array
    )  # (12, 3) array for sharks - separated into 4 lanes, 3 slots per lane [left to right]
    sub_positions: (
        chex.Array
    )  # (12, 3) array for enemy subs - separated into 4 lanes, 3 slots per lane [left to right]
    enemy_missile_positions: (
        chex.Array
    )  # (4, 3) array for enemy missiles (only the front boats can shoot)
    surface_sub_position: chex.Array  # (1, 3) array for surface submarine
    player_missile_position: (
        chex.Array
    )  # (1, 3) array for player missile (x, y, direction)
    step_counter: chex.Array
    just_surfaced: chex.Array  # Flag for tracking actual surfacing moment
    successful_rescues: (
        chex.Array
    )  # Number of times the player has surfaced with all six divers
    death_counter: chex.Array  # Counter for tracking death animation
    rng_key: chex.PRNGKey


@struct.dataclass
class SeaquestObservation:
    player: ObjectObservation
    divers: ObjectObservation # n=4
    enemies: ObjectObservation  # n=25 (Sharks, Subs, Surface Sub)
    projectiles: ObjectObservation # n=5 (Player & Enemy)
    oxygen_level: jnp.ndarray
    player_score: jnp.ndarray
    lives: jnp.ndarray
    collected_divers: jnp.ndarray

@struct.dataclass
class SeaquestInfo:
    difficulty: jnp.ndarray  # Current difficulty level
    successful_rescues: jnp.ndarray  # Number of successful rescues
    step_counter: jnp.ndarray  # Current step count


@struct.dataclass
class CarryState:
    missile_pos: chex.Array
    shark_pos: chex.Array
    sub_pos: chex.Array
    score: chex.Array


# RENDER CONSTANTS
def get_shark_color_index(difficulty: chex.Array) -> chex.Array:
    """
    Determine which shark color to use based on difficulty level.
    Color cycle: Green -> Yellow -> Pink -> Orange -> Green -> Yellow -> Green -> Orange -> back to start

    Args:
        difficulty: Current difficulty level (0-7)

    Returns:
        Color index: 0=Green, 1=Yellow, 2=Pink, 3=Orange
    """
    # Map difficulty to color index using the specific 8-level pattern
    # Pattern: Green -> Yellow -> Pink -> Orange -> Green -> Yellow -> Green -> Orange -> back to start
    color_mapping = jnp.array([0, 1, 2, 3, 0, 1, 0, 3])  # 0=Green, 1=Yellow, 2=Pink, 3=Orange
    color_index = jnp.take(color_mapping, difficulty % 8)
    return color_index

class JaxSeaquest(JaxEnvironment[SeaquestState, SeaquestObservation, SeaquestInfo, SeaquestConstants]):
    def initialize_spawn_state(self) -> SpawnState:
        """Initialize spawn state with first wave matching original game."""
        return SpawnState(
            difficulty=jnp.array(0),
            lane_dependent_pattern=jnp.zeros(
                4, dtype=jnp.int32
            ),  # Each lane starts at wave 0
            to_be_spawned=jnp.zeros(
                12, dtype=jnp.int32
            ),  # Track which enemies are still in the spawning cycle
            survived=jnp.zeros(12, dtype=jnp.int32),  # Track which enemies survived
            prev_sub=jnp.full(
                4, -1, dtype=jnp.int32
            ),  # -1 = no prior wave (first spawn = sharks); then 0=shark / 1=sub
            spawn_timers=self.initial_spawn_timers(),
            diver_array=jnp.ones(4, dtype=jnp.int32),  # all lanes armed; opening places only top 2
            diver_instant_pending=jnp.zeros(4, dtype=jnp.int32),
            diver_suppress_next=jnp.zeros(4, dtype=jnp.int32),
            lane_directions=self.consts.FIRST_WAVE_DIRS.astype(jnp.int32),  # First wave directions
            episode_frame=jnp.int32(0),
            # First shark→sub flip on ALE's clock (126 + 662).
            next_survive_wave_at=jnp.int32(788),
            enemy_offscreen_timers=jnp.full(24, -1, dtype=jnp.int32),
        )

    def initial_spawn_timers(self) -> chex.Array:
        """Opening / soft-reset per-lane timers: shared base + first-wave delay knob."""
        return (
            self.consts.INITIAL_SPAWN_TIMER_BASE.astype(jnp.int32)
            + self.consts.FIRST_WAVE_LANE_DELAY.astype(jnp.int32)
        )

    def soft_reset_spawn_state(
        self,
        spawn_state: SpawnState,
        diver_positions: chex.Array | None = None,
        *,
        rearm_divers: bool = False,
        rng: chex.PRNGKey | None = None,
    ) -> SpawnState:
        """Reset spawn bookkeeping after a life / rescue stage reset.

        ``diver_array`` is the dedicated per-lane slot:
          - death: preserve collected (0) / armed (1) / swam-off (-1). Do not
            pass ``diver_positions`` — dismissing live divers as collected
            would collapse remaining armed lanes into all-0 and falsely re-arm
            the whole set (including already-bagged top lanes).
          - successful surface: preserve collected (0); dismiss live world
            divers to 0; if every lane is then collected, re-arm all four.

        Each armed lane rolls ``DIVER_INSTANT_SPAWN_PROB`` (50%) to co-spawn with
        the next escort wave. Losers get ``diver_suppress_next`` so they do not
        also place on that same shark wave a few frames later.
        """
        da = spawn_state.diver_array
        if diver_positions is not None:
            da = jnp.where(diver_positions[:, 2] != 0, jnp.int32(0), da)
        da = jnp.where(rearm_divers, jnp.ones_like(da), da)
        # Full-collect / all-dismissed → rearm all four before the instant roll.
        da = jnp.where(jnp.all(da == 0), jnp.ones_like(da), da)

        timers = self.initial_spawn_timers()
        pending = jnp.zeros(4, dtype=jnp.int32)
        suppress = jnp.zeros(4, dtype=jnp.int32)

        if rng is not None:
            p = jnp.float32(self.consts.DIVER_INSTANT_SPAWN_PROB)
            hits = jax.random.bernoulli(rng, p=p, shape=(4,))
            pending = jnp.logical_and(da == 1, hits).astype(jnp.int32)
            trigger_p1 = (
                self.consts.DIVER_SPAWN_TIMER_TRIGGER.astype(jnp.int32) + jnp.int32(1)
            )
            timers = jnp.where(pending.astype(bool), trigger_p1, timers)
            # Armed instant-roll losers skip the next shark-diver opportunity.
            suppress = jnp.where(
                jnp.logical_and(da == 1, jnp.logical_not(pending.astype(bool))),
                jnp.int32(1),
                jnp.int32(0),
            )

        prev_sub = jnp.where(
            rearm_divers,
            jnp.full_like(spawn_state.prev_sub, -1),
            spawn_state.prev_sub,
        )
        lane_dirs = jnp.where(
            rearm_divers,
            self.consts.FIRST_WAVE_DIRS.astype(jnp.int32),
            spawn_state.lane_directions,
        )

        return spawn_state.replace(
            spawn_timers=timers,
            survived=jnp.zeros_like(spawn_state.survived),
            diver_array=da,
            diver_instant_pending=pending,
            diver_suppress_next=suppress,
            prev_sub=prev_sub,
            lane_directions=lane_dirs,
            next_survive_wave_at=spawn_state.episode_frame
            + self.consts.SURVIVE_WAVE_PERIOD.astype(jnp.int32)
            + jnp.int32(1),
        )

    def spawn_timer_reload(
        self, diver_array: chex.Array, *, after_survive: bool = False
    ) -> chex.Array:
        """Per-lane reload after kill-clear or survive-off.

        Kill-clear uses ``SPAWN_TIMER_RELOAD + TRIGGER``; survive-off uses
        ``SPAWN_TIMER_RELOAD_AFTER_SURVIVE + TRIGGER`` (kept short so AFK
        spacing stays despawn/travel-driven).
        """
        trigger = self.consts.DIVER_SPAWN_TIMER_TRIGGER.astype(jnp.int32)
        reload = jnp.where(
            after_survive,
            self.consts.SPAWN_TIMER_RELOAD_AFTER_SURVIVE.astype(jnp.int32),
            self.consts.SPAWN_TIMER_RELOAD.astype(jnp.int32),
        )
        return jnp.broadcast_to(reload + trigger, diver_array.shape)

    @partial(jax.jit, static_argnums=(0,))
    def check_collision_single(self, pos1, size1, pos2, size2):
        """Check collision between two single entities"""
        # Calculate edges for rectangle 1
        rect1_left = pos1[0]
        rect1_right = pos1[0] + size1[0]
        rect1_top = pos1[1]
        rect1_bottom = pos1[1] + size1[1]

        # Calculate edges for rectangle 2
        rect2_left = pos2[0]
        rect2_right = pos2[0] + size2[0]
        rect2_top = pos2[1]
        rect2_bottom = pos2[1] + size2[1]

        # Check overlap
        horizontal_overlap = jnp.logical_and(
            rect1_left < rect2_right,
            rect1_right > rect2_left
        )

        vertical_overlap = jnp.logical_and(
            rect1_top < rect2_bottom,
            rect1_bottom > rect2_top
        )

        return jnp.logical_and(horizontal_overlap, vertical_overlap)

    @partial(jax.jit, static_argnums=(0,))
    def check_collision_batch(self, pos1, size1, pos2_array, size2):
        """Check collision between one entity and an array of entities"""
        # Calculate edges for rectangle 1
        rect1_left = pos1[0]
        rect1_right = pos1[0] + size1[0]
        rect1_top = pos1[1]
        rect1_bottom = pos1[1] + size1[1]

        # Calculate edges for all rectangles in pos2_array
        rect2_left = pos2_array[:, 0]
        rect2_right = pos2_array[:, 0] + size2[0]
        rect2_top = pos2_array[:, 1]
        rect2_bottom = pos2_array[:, 1] + size2[1]

        # Check overlap for all entities
        horizontal_overlaps = jnp.logical_and(
            rect1_left < rect2_right,
            rect1_right > rect2_left
        )

        vertical_overlaps = jnp.logical_and(
            rect1_top < rect2_bottom,
            rect1_bottom > rect2_top
        )

        # Combine checks for each entity
        collisions = jnp.logical_and(horizontal_overlaps, vertical_overlaps)

        # Return true if any collision detected
        return jnp.any(collisions)


    @partial(jax.jit, static_argnums=(0,))
    def check_missile_collisions(
        self,
        missile_pos: chex.Array,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        diver_positions: chex.Array,
        score: chex.Array,
        successful_rescues: chex.Array,
        spawn_state: SpawnState,
        rng_key: chex.PRNGKey,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, SpawnState, chex.PRNGKey]:
        """
        Check for collisions between player missile and enemies using a vectorized approach.
        """
        missile_rect_pos = missile_pos[:2]
        missile_active = missile_pos[2] != 0

        # --- 1. Vectorized Collision Detection ---
        all_enemies = jnp.concatenate([shark_positions, sub_positions], axis=0)
        enemy_sizes = jnp.concatenate([
            jnp.repeat(jnp.array(self.consts.SHARK_SIZE)[None, :], shark_positions.shape[0], axis=0),
            jnp.repeat(jnp.array(self.consts.ENEMY_SUB_SIZE)[None, :], sub_positions.shape[0], axis=0)
        ], axis=0)

        def check_single_enemy(enemy_pos, enemy_size):
            hit = self.check_collision_single(
                missile_rect_pos, self.consts.MISSILE_SIZE, enemy_pos[:2], enemy_size
            )
            # Off-screen entities (ALE-invisible / pre-entry) cannot be shot.
            return jnp.logical_and(hit, self.enemy_is_hittable(enemy_pos))

        all_collision_mask = jax.vmap(check_single_enemy, in_axes=(0, 0))(all_enemies, enemy_sizes)
        all_collision_mask = jnp.logical_and(missile_active, all_collision_mask)

        shark_collision_mask = all_collision_mask[:shark_positions.shape[0]]
        sub_collision_mask = all_collision_mask[shark_positions.shape[0]:]

        # --- 2. Update Game State Based on Collision Masks ---
        points_per_kill = self.calculate_kill_points(successful_rescues)
        score_increase = jnp.sum(all_collision_mask * points_per_kill)
        new_score = score + score_increase

        zeros = jnp.zeros_like(shark_positions[0])
        new_shark_positions = jnp.where(shark_collision_mask[:, None], zeros, shark_positions)
        new_sub_positions = jnp.where(sub_collision_mask[:, None], zeros, sub_positions)

        missile_was_destroyed = jnp.any(all_collision_mask)
        new_missile_pos = jnp.where(missile_was_destroyed, jnp.zeros(3), missile_pos)

        # --- 3. Update SpawnState Based on Collision Masks ---
        # The survived array is (12,), so we need to merge collision results from both
        # shark and sub slots.
        is_sub_mask_slots = jnp.repeat(spawn_state.prev_sub == 1, 3) # (4,) -> (12,)
        final_collision_mask = jnp.where(is_sub_mask_slots, sub_collision_mask, shark_collision_mask)
        new_survived = jnp.where(final_collision_mask, 0, spawn_state.survived)

        # Determine which *lanes* had a collision across all 8 virtual lanes.
        lane_had_collision_8 = jnp.any(all_collision_mask.reshape(8, 3), axis=1) # Shape (8,)

        # Merge the 8-lane collision results into a 4-lane mask
        shark_lanes_hit, sub_lanes_hit = lane_had_collision_8[:4], lane_had_collision_8[4:]
        lane_had_collision_4 = jnp.where(spawn_state.prev_sub == 1, sub_lanes_hit, shark_lanes_hit) # Shape (4,)

        # Next-wave heading / timer reload when the lane is fully cleared (last kill),
        # not on partial hits. Diver heading is not independent: it always snaps to
        # that same next-escort direction (ALE). The old 25% "reverse" RNG was wrong.
        sharks_remain = jnp.any(new_shark_positions.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        subs_remain = jnp.any(new_sub_positions.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        enemies_remain = jnp.logical_or(sharks_remain, subs_remain)
        lane_fully_cleared = jnp.logical_and(lane_had_collision_4, jnp.logical_not(enemies_remain))

        # Per-lane reload only — do NOT wipe sibling lanes. Same shared
        # SPAWN_TIMER_RELOAD + TRIGGER counter as survive-off.
        kill_reload = self.spawn_timer_reload(spawn_state.diver_array)
        new_spawn_timers = jnp.where(
            lane_fully_cleared,
            kill_reload,
            spawn_state.spawn_timers,
        )

        # Kill-clear must wipe the whole lane's survived flags. Otherwise a
        # trailing formation member that was falsely marked "survived" while
        # still parking off-screen would force the next wave to be a submarine.
        new_survived = jnp.where(
            jnp.repeat(lane_fully_cleared, 3),
            jnp.int32(0),
            new_survived,
        )

        # Kill-clear: 50/50 re-roll next-wave heading per cleared lane. (We used
        # to keep the old dir when L/R approach times differed — that looked like
        # a timer bug. Lead-symmetric spawn makes both sides the same length.)
        rng_key, dir_rng = jax.random.split(rng_key)
        random_directions = jax.random.bernoulli(dir_rng, 0.5, (4,)).astype(jnp.int32)
        new_lane_directions = jnp.where(
            lane_fully_cleared,
            random_directions,
            spawn_state.lane_directions,
        )

        # Snap active divers to the current escort heading after the roll.
        diver_active = diver_positions[:, 2] != 0
        snap_diver = jnp.logical_and(lane_fully_cleared, diver_active)
        matched_diver_dirs = jnp.where(new_lane_directions == 1, jnp.int32(-1), jnp.int32(1))
        new_diver_positions = diver_positions.at[:, 2].set(
            jnp.where(snap_diver, matched_diver_dirs, diver_positions[:, 2])
        )

        new_spawn_state = spawn_state.replace(
            survived=new_survived,
            spawn_timers=new_spawn_timers,
            lane_directions=new_lane_directions,
        )

        return (
            new_missile_pos, new_shark_positions, new_sub_positions, new_diver_positions,
            new_score, new_spawn_state, rng_key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def check_player_collision(
        self,
        player_x,
        player_y,
        submarine_list,
        shark_list,
        surface_sub_pos,
        enemy_projectile_list,
        score,
        successful_rescues,
    ) -> Tuple[chex.Array, chex.Array]:
        # check if the player has collided with any of the three given lists
        # the player is a 16x11 rectangle
        # the submarine is a 8x11 rectangle
        # the shark is a 8x7 rectangle
        # the missile is a 8x1 rectangle
        # the surface submarine is 8x11 as well

        # Damage hitbox: horizontal inset only so grazing side overlaps with
        # bobbing sharks do not count (ALE TIA); vertical extent stays full.
        inset_x = self.consts.PLAYER_COLLISION_INSET_X.astype(jnp.int32)
        player_pos = jnp.array([player_x + inset_x, player_y])
        player_size = jnp.array(
            [self.consts.PLAYER_SIZE[0] - 2 * inset_x, self.consts.PLAYER_SIZE[1]]
        )

        # check if the player has collided with any of the submarines
        # (off-screen / pre-entry enemies cannot collide — same gate as missile hits)
        hittable_subs = jax.vmap(self.enemy_is_hittable)(submarine_list)
        sub_pos_gated = jnp.where(hittable_subs[:, None], submarine_list, jnp.zeros_like(submarine_list))
        submarine_collisions = jnp.any(
            self.check_collision_batch(
                player_pos, player_size, sub_pos_gated, self.consts.ENEMY_SUB_SIZE
            )
        )

        # check if the player has collided with any of the sharks
        hittable_sharks = jax.vmap(self.enemy_is_hittable)(shark_list)
        shark_pos_gated = jnp.where(hittable_sharks[:, None], shark_list, jnp.zeros_like(shark_list))
        shark_collisions = jnp.any(
            self.check_collision_batch(
                player_pos, player_size, shark_pos_gated, self.consts.SHARK_SIZE
            )
        )

        # check if the player collided with the surface submarine
        surface_collision = self.check_collision_single(
            player_pos,
            player_size,
            surface_sub_pos,
            self.consts.ENEMY_SUB_SIZE
        )

        # check if the player has collided with any of the enemy projectiles
        missile_collisions = jnp.any(
            self.check_collision_batch(
                player_pos,
                player_size,
                enemy_projectile_list,
                self.consts.MISSILE_SIZE
            )
        )

        # Calculate points for collisions.
        # When colliding with a shark or submarine the player gains points similar to killing the object
        collision_points = jnp.where(
            shark_collisions,
            self.calculate_kill_points(successful_rescues),
            jnp.where(
                submarine_collisions,
                self.calculate_kill_points(successful_rescues),
                jnp.where(surface_collision, self.calculate_kill_points(successful_rescues), 0),
            ),
        )

        return (
            jnp.any(
                jnp.array(
                    [
                        submarine_collisions,
                        shark_collisions,
                        missile_collisions,
                        surface_collision,
                    ]
                )
            ),
            collision_points,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_spawn_position(self, moving_left: chex.Array, slot: chex.Array) -> chex.Array:
        """Get spawn position based on movement direction and slot number.

        Spawns at the lead-edge margin (ENEMY_SPAWN_X_FROM_LEFT/RIGHT) so
        entities are not missile-hittable until they enter the visible band.
        """
        base_y = jnp.array(self.consts.SPAWN_POSITIONS_Y[slot])
        x_pos = jnp.where(
            moving_left,
            self.consts.ENEMY_SPAWN_X_FROM_RIGHT.astype(jnp.int32),
            self.consts.ENEMY_SPAWN_X_FROM_LEFT.astype(jnp.int32),
        )
        direction = jnp.where(moving_left, -1, 1)  # -1 for left, 1 for right
        return jnp.array([x_pos, base_y, direction], dtype=jnp.int32)

    def enemy_is_hittable(self, enemy_pos: chex.Array) -> chex.Array:
        """True if enemy is active and far enough on-screen to take missile hits."""
        active = enemy_pos[2] != 0
        onscreen = jnp.logical_and(
            enemy_pos[0] >= self.consts.ENEMY_HITTABLE_X_MIN,
            enemy_pos[0] < self.consts.ENEMY_HITTABLE_X_MAX,
        )
        return jnp.logical_and(active, onscreen)

    @partial(jax.jit, static_argnums=(0,))
    def is_slot_empty(self, pos: chex.Array) -> chex.Array:
        """Check if a position slot is empty (0,0,ß)"""
        return pos[2] == 0

    @partial(jax.jit, static_argnums=(0,))
    def get_front_entity(self, i, lane_positions):
        # check on the first submarine in the lane which direction they are going
        direction = lane_positions[0][2]

        direction = jnp.where(
            lane_positions[0][2] == 0,
            jnp.where(
                lane_positions[1][2] == 0, lane_positions[2][2], lane_positions[1][2]
            ),
            lane_positions[0][2],
        )

        # if direction is 1, go from right to left until an active entity is found
        # if direction is -1, go from left to right until an active entity is found
        front_entity = jnp.where(
            direction == -1,
            jnp.where(
                lane_positions[0][2] != 0,
                lane_positions[0],
                jnp.where(
                    lane_positions[1][2] != 0,
                    lane_positions[1],
                    jnp.where(lane_positions[2][2] != 0, lane_positions[2], jnp.zeros(3)),
                ),
            ),
            jnp.where(
                lane_positions[2][2] != 0,
                lane_positions[2],
                jnp.where(
                    lane_positions[1][2] != 0,
                    lane_positions[1],
                    jnp.where(lane_positions[0][2] != 0, lane_positions[0], jnp.zeros(3)),
                ),
            ),
        )

        return front_entity

    @partial(jax.jit, static_argnums=(0,))
    def get_pattern_for_difficulty(
        self, current_pattern: chex.Array, moving_left: chex.Array
    ) -> chex.Array:
        """Returns spawn pattern based on the lane's current wave/pattern number

        Pattern meanings:
        0: Single enemy (initial pattern)
        1: Two adjacent enemies
        2: Two enemies with gap
        3: Three enemies in a row
        """
        # Basic pattern arrays for different formations (absolute slot bits:
        # index 0 = left, 1 = middle, 2 = right — same as ALE form layout).
        PATTERNS = jnp.array(
            [
                [0, 0, 1],  # wave 0: Single enemy (top-lane native; bottom uses mask)
                [1, 1, 0],  # wave 1: Two adjacent — left + middle (ALE)
                [1, 0, 1],  # wave 2: Two with gap
                [1, 1, 1],  # wave 3: Three in row
            ]
        )

        # Reverse pattern if moving left
        base_pattern = PATTERNS[current_pattern]

        return base_pattern

    @partial(jax.jit, static_argnums=(0,))
    def update_enemy_spawns(
        self,
        spawn_state: SpawnState,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        diver_positions: chex.Array,
        step_counter: chex.Array,
        rng: chex.PRNGKey = None,
    ) -> Tuple[SpawnState, chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
        """Update enemy spawns using pattern-based system matching original game.
        Args:
            spawn_state: Current spawn state
            shark_positions: Current shark positions
            sub_positions: Current submarine positions
            diver_positions: Current diver positions
            step_counter: Current step counter
            rng: Optional random key for direction randomization

        Returns:
            Tuple of updated spawn state, shark/sub/diver positions, and RNG key
        """


        # Timers are decremented once in spawn_step (shared with diver spawn).
        new_state = spawn_state


        # --- START of new vectorized calculation ---
        # 1. Vectorized check for empty lanes across all 4 lanes
        sharks_active = shark_positions.reshape(4, 3, 3)[:, :, 2] != 0
        subs_active = sub_positions.reshape(4, 3, 3)[:, :, 2] != 0
        all_lanes_empty = jnp.all(~sharks_active & ~subs_active, axis=1)  # Shape (4,)

        # 2. Vectorized check for entities that still need to be spawned
        to_be_spawned_lanes = spawn_state.to_be_spawned.reshape(4, 3)
        any_to_be_spawned = jnp.any(to_be_spawned_lanes != 0, axis=1)  # Shape (4,)

        # 3. Lanes that need an update: empty, trickle-spawning, OR timer hit
        # (overwrite stragglers still mid-crossing — skipping them left mixed
        # prev_sub / orphaned enemies on screen).
        timers_ready = (
            spawn_state.spawn_timers == self.consts.DIVER_SPAWN_TIMER_TRIGGER
        )
        all_lanes_need_update = jnp.logical_or(
            jnp.logical_or(all_lanes_empty, any_to_be_spawned),
            timers_ready,
        )
        # The scan_lanes function is now much simpler
        def scan_lanes(carry, lane_idx):
            curr_state, curr_shark_positions, curr_sub_positions, curr_diver_positions, curr_rng = carry

            # Use the pre-computed mask to check if this lane needs an update
            needs_update = all_lanes_need_update[lane_idx]

            # The rest of the function proceeds as before
            new_carry = jax.lax.cond(
                needs_update,
                lambda x: process_lane(lane_idx, x), # process_lane is unchanged
                lambda x: x,
                (curr_state, curr_shark_positions, curr_sub_positions, curr_diver_positions, curr_rng),
            )

            return new_carry, None

        def initialize_new_spawn_cycle(i, carry):
            spawn_state, shark_positions, sub_positions, diver_positions, rng = carry

            # Split RNG key for this lane
            rng, lane_rng = jax.random.split(rng)

            # Formation: ALE uses fixed absolute slot bits per lane (bottom
            # form=4 = slot 0, others form=1 = slot 2 at diff-0). Do NOT mirror
            # on direction — that erased the permanent bottom-lane 32px offset.
            clipped_difficulty = spawn_state.difficulty % 8
            density = jnp.where(
                clipped_difficulty < 2,
                0,
                jnp.where(
                    clipped_difficulty < 4,
                    1,
                    jnp.where(clipped_difficulty < 6, 2, 3),
                ),
            )
            lane_mask = self.consts.LANE_SLOT_MASKS[i]
            # density 0: lane's native single. Higher: shared ALE formation bits.
            # Do NOT OR/maximum with lane_mask — that turned bottom ([1,0,0]) +
            # two-enemy ([1,1,0]) into three ([1,1,1]) at the 2-enemy difficulty.
            formation = self.get_pattern_for_difficulty(density, False)
            current_pattern = jnp.where(density == 0, lane_mask, formation)
            lane_specific_pattern = density

            # Escort heading comes from lane_directions (chosen when the previous
            # wave cleared). Divers follow that heading — they do not drive it.
            moving_left = spawn_state.lane_directions[i] == 1

            # Subs only after *this lane's* shark wave survived. Kill-clear wipes
            # that lane's survived flags so it respawns sharks while siblings that
            # swam off can still schedule submarines (ALE top-snipe behaviour).
            lane_had_survivor = jnp.any(
                spawn_state.survived.reshape(4, 3) != 0, axis=1
            )[i]
            is_sub = jnp.logical_and(
                lane_had_survivor, spawn_state.prev_sub[i] == 0
            )

            # Symmetric lead-centric formation: native slot (lane mask) sits
            # ENEMY_SPAWN_EDGE_MARGIN past the entry edge; other slots pack
            # away from the screen. Same approach both directions.
            dir_sign = jnp.where(moving_left, jnp.int32(-1), jnp.int32(1))
            margin = self.consts.ENEMY_SPAWN_EDGE_MARGIN.astype(jnp.int32)
            lead_x = jnp.where(
                moving_left,
                jnp.int32(160) + margin,
                jnp.int32(0) - margin,
            )
            lead_slot = jnp.argmax(lane_mask).astype(jnp.int32)
            away = -dir_sign
            slot_idx = jnp.arange(3, dtype=jnp.int32)
            slot_xs = lead_x + (slot_idx - lead_slot) * (
                self.consts.ENEMY_WAVE_SPACING.astype(jnp.int32) * away
            )
            slot_active = current_pattern != 0

            base_y = self.consts.SPAWN_POSITIONS_Y[i]
            lane_positions = jnp.stack(
                [slot_xs, jnp.full(3, base_y, dtype=jnp.int32), jnp.full(3, dir_sign, dtype=jnp.int32)],
                axis=1,
            )
            lane_positions = jnp.where(slot_active[:, None], lane_positions, jnp.zeros((3, 3), dtype=jnp.int32))

            indices = jnp.array([i * 3, i * 3 + 1, i * 3 + 2])
            # Full lane replace: clear the opposite type so a period-based
            # mid-crossing swap cannot leave stale sharks/subs behind.
            cleared = jnp.zeros((3, 3), dtype=jnp.int32)
            new_shark_positions = jnp.where(
                is_sub,
                shark_positions.at[indices].set(cleared),
                shark_positions.at[indices].set(lane_positions),
            )
            new_sub_positions = jnp.where(
                is_sub,
                sub_positions.at[indices].set(lane_positions),
                sub_positions.at[indices].set(cleared),
            )

            # wipe the survived status for this lane (since we are starting a new wave)
            new_survived_full = spawn_state.survived.at[indices].set(
                jnp.zeros(3, dtype=jnp.int32)
            )

            # Formation is fully placed — nothing left to trickle-spawn.
            new_full_to_be_spawned = spawn_state.to_be_spawned.at[indices].set(
                jnp.zeros(3, dtype=jnp.int32)
            )

            # Divers co-spawn only with sharks (spawn_divers). Sub waves are
            # gated in process_lane so they never start while a diver is live.
            new_spawn_state = SpawnState(
                difficulty=spawn_state.difficulty,
                lane_dependent_pattern=spawn_state.lane_dependent_pattern.at[i].set(
                    lane_specific_pattern
                ),
                to_be_spawned=new_full_to_be_spawned,
                survived=new_survived_full,
                prev_sub=spawn_state.prev_sub.at[i].set(is_sub.astype(jnp.int32)),
                # Wave already placed at the diver-trigger tick — park timer at 0
                # until the next clear reloads it.
                spawn_timers=spawn_state.spawn_timers.at[i].set(jnp.int32(0)),
                diver_array=spawn_state.diver_array,
                diver_instant_pending=spawn_state.diver_instant_pending,
                diver_suppress_next=spawn_state.diver_suppress_next,
                lane_directions=spawn_state.lane_directions,
                episode_frame=spawn_state.episode_frame,
                next_survive_wave_at=spawn_state.next_survive_wave_at,
                enemy_offscreen_timers=(
                    spawn_state.enemy_offscreen_timers.at[indices].set(
                        jnp.full(3, -1, dtype=jnp.int32)
                    ).at[indices + 12].set(jnp.full(3, -1, dtype=jnp.int32))
                ),
            )

            return new_spawn_state, new_shark_positions, new_sub_positions, diver_positions, rng

        # Modified continue_spawn_cycle to handle RNG
        def continue_spawn_cycle(i: int, carry):
            spawn_state, shark_positions, sub_positions, diver_positions, rng = carry

            # Rest of function remains the same, just pass along the RNG
            # get the relevant missing entities for this lane from the to_be_spawned array
            relevant_to_be_spawned = jax.lax.dynamic_slice(
                spawn_state.to_be_spawned, (i * 3,), (3,)
            )

            # check in which direction we are moving by finding the first non-zero value in the missing_entities array
            moving_left = jnp.where(
                relevant_to_be_spawned[0] == 0,
                jnp.where(
                    relevant_to_be_spawned[1] == 0,
                    jnp.where(relevant_to_be_spawned[2] == -1, True, False),
                    jnp.where(relevant_to_be_spawned[1] == -1, True, False),
                ),
                jnp.where(relevant_to_be_spawned[0] == -1, True, False),
            )

            # Find the index of the first non-zero value based on direction
            def scan_right_to_left(j, val):
                return jnp.where(relevant_to_be_spawned[2 - j] != 0, 2 - j, val)

            def scan_left_to_right(j, val):
                return jnp.where(relevant_to_be_spawned[j] != 0, j, val)

            # Use fori_loop to scan array in appropriate direction
            spawn_idx = jax.lax.cond(
                moving_left,
                lambda _: jax.lax.fori_loop(0, 3, scan_left_to_right, -1),
                lambda _: jax.lax.fori_loop(0, 3, scan_right_to_left, -1),
                operand=None,
            )

            spawn_idx = spawn_idx.astype(jnp.int32)

            # Get reference x position from neighboring entity
            # For moving right, look at entity to the right (spawn_idx + 1)
            # For moving left, look at entity to the left (spawn_idx - 1)
            reference_idx = jnp.where(moving_left, spawn_idx - 1, spawn_idx + 1)
            reference_idx = reference_idx.astype(jnp.int32)
            base_idx = i * 3  # Base index for this lane's entities

            # Get position from either shark or sub position arrays
            # We'll need to check both since we don't know which type exists
            reference_shark_pos = shark_positions[base_idx + reference_idx]
            reference_sub_pos = sub_positions[base_idx + reference_idx]

            # Active = nonzero direction. Do NOT use x!=0: spawn-from-left uses x=-8/0,
            # and the old `reference_x == 0 → force spawn` stacked the whole wave on top
            # of itself (JAX play log: 10k+ in-lane gaps < 8px; ALE: 0).
            reference_active = jnp.logical_or(
                reference_shark_pos[2] != 0, reference_sub_pos[2] != 0
            )
            reference_x = jnp.where(
                reference_shark_pos[2] != 0, reference_shark_pos[0], reference_sub_pos[0]
            )

            # Edge Case: third option exists for the pattern 1 0 1, then check the next entity
            edge_case_reference_idx = jnp.where(moving_left, spawn_idx - 2, spawn_idx + 2)

            edge_case_reference_idx = edge_case_reference_idx.astype(jnp.int32)

            edge_shark = shark_positions[base_idx + edge_case_reference_idx]
            edge_sub = sub_positions[base_idx + edge_case_reference_idx]
            edge_active = jnp.logical_or(edge_shark[2] != 0, edge_sub[2] != 0)
            edge_x = jnp.where(edge_shark[2] != 0, edge_shark[0], edge_sub[0])

            # Gap pattern (1 0 1): reference slot empty but the far entity is active.
            use_gap_reference = jnp.logical_and(jnp.logical_not(reference_active), edge_active)
            spacing_ref_x = jnp.where(use_gap_reference, edge_x, reference_x)
            spacing_ref_active = jnp.logical_or(reference_active, use_gap_reference)

            # Get base spawn position for this lane
            base_spawn_pos = self.get_spawn_position(moving_left, jnp.array(i))

            # ALE in-lane gaps are 16px (32px for 1-0-1). Spawn next member at the edge
            # only once the reference has moved far enough.
            offset = jnp.where(
                use_gap_reference,
                self.consts.ENEMY_WAVE_SPACING_GAP,
                self.consts.ENEMY_WAVE_SPACING,
            )
            far_enough = jnp.abs(base_spawn_pos[0] - spacing_ref_x) >= offset
            # If the leading entity was destroyed (no active reference), spawn immediately.
            should_spawn = jnp.where(spacing_ref_active, far_enough, True)

            spawn_pos = jnp.where(should_spawn, base_spawn_pos, jnp.zeros(3))

            # Update positions based on enemy type
            new_shark_positions = shark_positions.at[base_idx + spawn_idx].set(
                jnp.where(
                    spawn_state.prev_sub[i] != 1,
                    spawn_pos,
                    shark_positions[base_idx + spawn_idx],
                )
            )
            new_sub_positions = sub_positions.at[base_idx + spawn_idx].set(
                jnp.where(
                    spawn_state.prev_sub[i] == 1, spawn_pos, sub_positions[base_idx + spawn_idx]
                )
            )

            # Update the to_be_spawned array
            new_to_be_spawned = spawn_state.to_be_spawned.at[base_idx + spawn_idx].set(
                jnp.where(
                    should_spawn,
                    jnp.array(0),  # Single value
                    spawn_state.to_be_spawned[base_idx + spawn_idx],
                )
            )

            # Then create the new spawn state with the updated array
            new_spawn_state = SpawnState(
                difficulty=spawn_state.difficulty,
                lane_dependent_pattern=spawn_state.lane_dependent_pattern,
                to_be_spawned=new_to_be_spawned,
                survived=spawn_state.survived,
                prev_sub=spawn_state.prev_sub,
                spawn_timers=spawn_state.spawn_timers,
                diver_array=spawn_state.diver_array,
                diver_instant_pending=spawn_state.diver_instant_pending,
                diver_suppress_next=spawn_state.diver_suppress_next,
                lane_directions=spawn_state.lane_directions,
                episode_frame=spawn_state.episode_frame,
                next_survive_wave_at=spawn_state.next_survive_wave_at,
                enemy_offscreen_timers=spawn_state.enemy_offscreen_timers,
            )

            return new_spawn_state, new_shark_positions, new_sub_positions, diver_positions, rng

        # Per-lane: hold this lane's sub spawn while *its* diver is still live.
        # (A global any-diver gate let a late opposite-side L3 diver stretch
        # L0/L1/L2 gaps by hundreds of frames after a direction flip.)
        survived_lanes = jnp.any(
            spawn_state.survived.reshape(4, 3) != 0, axis=1
        )
        lane_wants_sub = jnp.logical_and(
            survived_lanes, spawn_state.prev_sub == 0
        )
        lane_diver_live = diver_positions[:, 2] != 0
        hold_sub_wave = jnp.logical_and(lane_wants_sub, lane_diver_live)

        # Modified process_lane to handle RNG
        def process_lane(i, carry):
            loc_spawn_state, shark_positions, sub_positions, diver_positions, rng = carry
            base_idx = i * 3  # Base index for this lane's slots

            # determine if we need to initialize a new pattern or keep spawning for the current one
            # do this by checking in the relevant part of the to_be_spawned array if there are still 1s
            relevant_to_be_spawned = jax.lax.dynamic_slice(
                loc_spawn_state.to_be_spawned, (base_idx,), (3,)
            )

            # if there are still 1s in the relevant part of the to_be_spawned array, keep spawning
            keep_spawning = jnp.any(relevant_to_be_spawned)

            # check the lane spawn timer (use decremented timers from carry)
            lane_timer = loc_spawn_state.spawn_timers[i]

            base_idx = i * 3
            # Get the sharks and subs for the current lane `i`
            lane_sharks = jax.lax.dynamic_slice(shark_positions, (base_idx, 0), (3, 3))
            lane_subs = jax.lax.dynamic_slice(sub_positions, (base_idx, 0), (3, 3))

            # Vectorized check for active entities in the lane
            sharks_active = lane_sharks[:, 2] != 0
            subs_active = lane_subs[:, 2] != 0
            lane_empty = jnp.all(~sharks_active & ~subs_active)

            # Co-spawn at TRIGGER; overwrite stragglers. If this lane wants a
            # sub wave and any diver is still live, pin timer at TRIGGER+1 so
            # the next tick re-arms at TRIGGER (decrement would miss the window).
            at_trigger = lane_timer == self.consts.DIVER_SPAWN_TIMER_TRIGGER
            hold_this = hold_sub_wave[i]
            allow_new_initialization = jnp.logical_and(
                at_trigger, jnp.logical_not(hold_this)
            )

            def hold_timer_for_diver(x):
                spawn_state, shark_positions, sub_positions, diver_positions, rng = x
                pinned = spawn_state.replace(
                    spawn_timers=spawn_state.spawn_timers.at[i].set(
                        self.consts.DIVER_SPAWN_TIMER_TRIGGER.astype(jnp.int32)
                        + jnp.int32(1)
                    )
                )
                return pinned, shark_positions, sub_positions, diver_positions, rng

            def handle_no_spawning(x):
                spawn_state, shark_positions, sub_positions, diver_positions, rng = x
                return jax.lax.cond(
                    allow_new_initialization,
                    lambda y: initialize_new_spawn_cycle(i, y),
                    lambda y: jax.lax.cond(
                        jnp.logical_and(at_trigger, hold_this),
                        hold_timer_for_diver,
                        lambda z: z,
                        y,
                    ),
                    (spawn_state, shark_positions, sub_positions, diver_positions, rng),
                )

            new_spawn_state, new_shark_positions, new_sub_positions, new_diver_positions, new_rng = jax.lax.cond(
                keep_spawning,
                lambda x: continue_spawn_cycle(i, x),
                handle_no_spawning,
                (loc_spawn_state, shark_positions, sub_positions, diver_positions, rng),
            )

            return new_spawn_state, new_shark_positions, new_sub_positions, new_diver_positions, new_rng

        # Replace the manual loop with lax.scan
        lane_indices = jnp.arange(4)
        (final_state, final_shark_positions, final_sub_positions, final_diver_positions, final_rng), _ = jax.lax.scan(
            scan_lanes,
            (new_state, shark_positions, sub_positions, diver_positions, rng if rng is not None else jax.random.PRNGKey(42)),
            lane_indices
        )

        return final_state, final_shark_positions, final_sub_positions, final_diver_positions, final_rng

    @partial(jax.jit, static_argnums=(0,))
    def step_enemy_movement(
        self,
        spawn_state: SpawnState,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        step_counter: chex.Array,
        rng: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, SpawnState, chex.PRNGKey]:
        """Update enemy positions based on their patterns"""
        # Keep rng in the signature/return for call-site compatibility.
        direction_rng = rng  # unused; survive-off no longer re-rolls direction

        def get_shark_offset(step_counter):
            """Vertical bob from ALE ram[93] waveform (recorded 64-frame cycle).

            Phase is global (shared ``step_counter``): with correct lane bases,
            concurrent ALE sharks have residual-std 0 and lane-lane resid corr ≈ 1.
            """
            wave = self.consts.SHARK_BOB_WAVE
            phase = self.consts.SHARK_BOB_PHASE_OFFSET.astype(jnp.int32)
            return wave[(step_counter + phase) % wave.shape[0]] - 4

        def calculate_movement_speed(step_counter, difficulty):
            """
            Calculates movement speed based on difficulty. This function is vectorized
            and uses jnp.select for efficient conditional logic.
            """
            # Ensure difficulty is non-negative and wraps at 256 for consistent logic
            safe_difficulty = jnp.maximum(0, difficulty % 256)

            # --- Speed for difficulties 0-9 ---
            diff_lt_10 = safe_difficulty < 10
            cycle_pos = step_counter % 12
            # Diff 0: ALE surface-noop move_frac ≈ 0.374 on residues t%8 ∈ {2,4,7}
            # (exactly 3/8). Older (t%3==0) at 0.333 made crossings ~12% slow and
            # compounded the late wave-2 gap.
            step8 = step_counter % 8
            move_375 = jnp.logical_or(step8 == 2, jnp.logical_or(step8 == 4, step8 == 7))
            should_move_patterns = jnp.array([
                move_375,             # 37.5% (ALE opening)
                (cycle_pos % 2) == 0,  # 50%
                (cycle_pos % 3) != 2,  # 67%
                (cycle_pos % 4) != 3,  # 75%
                (cycle_pos % 6) != 5,  # 83%
                cycle_pos != 11,      # 92%
            ])
            indices = jnp.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
            should_move = should_move_patterns[indices[safe_difficulty]]
            speed_for_diff_0_9 = jnp.where(should_move, 1, 0)

            # --- Speed for difficulties 10+ ---
            diff_above_threshold = jnp.maximum(0, safe_difficulty - 10)
            base_speed = 1 + (diff_above_threshold // 16)
            position_in_tier = diff_above_threshold % 16

            # Probabilities for gaining +1 speed within a tier
            higher_speed_patterns = jnp.array([
                (step_counter % 16) == 0, # 6.25%
                (step_counter % 8) == 0,  # 12.5%
                (step_counter % 4) == 0,  # 25%
                (step_counter % 2) == 0,  # 50%
                (step_counter % 4) != 0,  # 75%
                (step_counter % 8) != 0,  # 87.5%
                (step_counter % 16) != 0, # 93.75%
            ])
            # Indices to select the correct probability pattern
            tier_indices = jnp.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6])
            use_higher_speed = higher_speed_patterns[tier_indices[position_in_tier]]

            speed_for_diff_10_plus = jnp.where(use_higher_speed, base_speed + 1, base_speed)

            # Select speed based on difficulty bracket
            return jnp.where(diff_lt_10, speed_for_diff_0_9, speed_for_diff_10_plus)

        def move_single_enemy(pos, timer, is_shark, slot_idx, movement_speed, shark_bob):
            """Move one enemy; arm/decrement off-screen despawn timer.

            Pre-entry spawns sit past the edge with timer=-1 (not counting).
            Only an on→off transition arms ``ENEMY_OFFSCREEN_DESPAWN_FRAMES``.
            """
            is_active = jnp.logical_not(self.is_slot_empty(pos))
            not_counting = jnp.int32(-1)
            hold = self.consts.ENEMY_OFFSCREEN_DESPAWN_FRAMES.astype(jnp.int32)

            velocity_x = pos[2] * movement_speed
            lane_idx = (slot_idx // 3) % 4
            base_y = self.consts.SPAWN_POSITIONS_Y[lane_idx]
            y_offset = jnp.where(
                is_shark,
                shark_bob,
                -self.consts.SUBMARINE_Y_OFFSET,
            )
            y_position = base_y + y_offset
            new_x = pos[0] + velocity_x
            new_pos = jnp.array([new_x, y_position, pos[2]], dtype=pos.dtype)
            new_pos = jnp.where(is_active, new_pos, pos)

            was_on = jnp.logical_and(
                is_active,
                jnp.logical_and(pos[0] >= 0, pos[0] <= 160),
            )
            now_off = jnp.logical_and(
                is_active,
                jnp.logical_or(new_pos[0] < 0, new_pos[0] > 160),
            )

            new_timer = jnp.where(is_active, timer, not_counting)
            new_timer = jnp.where(
                jnp.logical_and(is_active, jnp.logical_not(now_off)),
                not_counting,
                new_timer,
            )
            new_timer = jnp.where(jnp.logical_and(was_on, now_off), hold, new_timer)
            # Decrement on subsequent off-screen frames only (not the arm frame).
            new_timer = jnp.where(
                jnp.logical_and(
                    now_off,
                    jnp.logical_and(new_timer > 0, jnp.logical_not(was_on)),
                ),
                new_timer - jnp.int32(1),
                new_timer,
            )

            despawn = jnp.logical_and(
                now_off, jnp.logical_and(is_active, new_timer == 0)
            )
            final_pos = jnp.where(despawn, jnp.zeros_like(pos), new_pos)
            new_timer = jnp.where(despawn, not_counting, new_timer)
            return final_pos, despawn, new_timer

        # Speed + shark bob depend only on (step, difficulty) — compute once,
        # not 24× inside the vmap (same values for every slot).
        movement_speed = calculate_movement_speed(step_counter, spawn_state.difficulty)
        shark_bob = get_shark_offset(step_counter)

        # 1. Combine sharks and subs into a single array for vectorized processing
        shark_positions_int = shark_positions.astype(jnp.int32)
        sub_positions_int = sub_positions.astype(jnp.int32)
        all_positions = jnp.concatenate([shark_positions_int, sub_positions_int], axis=0)
        is_shark_array = jnp.concatenate(
            [jnp.ones(12, dtype=bool), jnp.zeros(12, dtype=bool)]
        )
        all_slot_indices = jnp.arange(24)
        all_timers = spawn_state.enemy_offscreen_timers.astype(jnp.int32)

        # 2. Apply movement + off-screen timers to all 24 slots in parallel
        vmap_move = jax.vmap(move_single_enemy, in_axes=(0, 0, 0, 0, None, None))
        new_all_positions, enemies_survived_mask, new_offscreen_timers = vmap_move(
            all_positions,
            all_timers,
            is_shark_array,
            all_slot_indices,
            movement_speed,
            shark_bob,
        )

        # 3. Handle lane-based logic by padding the 4-lane state to an 8-lane structure
        #    and then merging the results back down.

        # Pad the original 4-lane state for comparison purposes
        old_survived_padded = jnp.pad(spawn_state.survived, (0, 12))

        # Perform logic in the temporary 8-lane structure
        num_lanes = 8 # 4 for sharks, 4 for subs
        survived_mask_lanes = enemies_survived_mask.reshape(num_lanes, 3)
        old_survived_lanes = old_survived_padded.reshape(num_lanes, 3)

        any_newly_survived_in_lane = jnp.any(
            jnp.logical_and(survived_mask_lanes, old_survived_lanes == 0), axis=1
        )

        all_pos_lanes = all_positions.reshape(num_lanes, 3, 3)
        dir0 = all_pos_lanes[:, 0, 2]
        dir1 = all_pos_lanes[:, 1, 2]
        dir2 = all_pos_lanes[:, 2, 2]
        lane_base_direction = jnp.where(
            dir0 != 0, dir0, jnp.where(dir1 != 0, dir1, jnp.where(dir2 != 0, dir2, 1))
        )

        survived_direction_per_slot = jnp.repeat(lane_base_direction, 3, axis=0)
        temp_survived = jnp.where(
            enemies_survived_mask, survived_direction_per_slot, old_survived_padded
        )

        temp_survived_lanes = temp_survived.reshape(num_lanes, 3)
        lanes_to_flip = lane_base_direction == -1
        flipped_survived = jnp.flip(temp_survived_lanes, axis=1)
        temp_survived_lanes = jnp.where(
            lanes_to_flip[:, None],
            flipped_survived,
            temp_survived_lanes,
        )

        # Merge 8-lane results back into 4-lane state (prev_sub: -1/0/1).
        is_sub_mask = (spawn_state.prev_sub == 1)[:, None]
        shark_survived_res = temp_survived_lanes[:4]
        sub_survived_res = temp_survived_lanes[4:]
        final_survived_lanes = jnp.where(is_sub_mask, sub_survived_res, shark_survived_res)
        new_survived = final_survived_lanes.flatten()

        any_newly_survived_sharks = any_newly_survived_in_lane[:4]
        any_newly_survived_subs = any_newly_survived_in_lane[4:]
        any_newly_survived_final = jnp.where(
            spawn_state.prev_sub == 1, any_newly_survived_subs, any_newly_survived_sharks
        )

        # Per-lane survive-clear only. Lanes are independent: each arms its own
        # reload when *it* empties by swim-off. AFK sync is emergent (same start
        # + same cross time), not forced by arming siblings together — that was
        # what stole a kill-lane's shark cycle into the others' sub wave.
        sharks_still = jnp.any(new_all_positions[:12].reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        subs_still = jnp.any(new_all_positions[12:].reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        still_live = jnp.where(spawn_state.prev_sub == 1, subs_still, sharks_still)
        lane_cleared_by_survive = jnp.logical_and(
            any_newly_survived_final, jnp.logical_not(still_live)
        )

        survive_reload = self.spawn_timer_reload(
            spawn_state.diver_array, after_survive=True
        )
        arm_lane = jnp.logical_and(
            spawn_state.spawn_timers == 0, lane_cleared_by_survive
        )
        new_spawn_timers = jnp.where(
            arm_lane,
            survive_reload,
            spawn_state.spawn_timers,
        )

        # Flip heading after a *sub* wave survives off — not after sharks.
        # ALE keeps the same approach side for shark→sub (and often the next
        # shark); flipping on shark-clear made JAX sub enter opposite to ALE,
        # swapping long/short gaps and desyncing AFK within a few waves.
        flip_lane = jnp.logical_and(
            lane_cleared_by_survive, spawn_state.prev_sub == 1
        )
        new_lane_directions = jnp.where(
            flip_lane,
            1 - spawn_state.lane_directions,
            spawn_state.lane_directions,
        )

        # Survived flags already come from per-slot swim-off above — no
        # whole-wave ones-fill (that forced empty kill-lanes onto subs).
        new_survived = final_survived_lanes.flatten()

        new_spawn_state = spawn_state.replace(
            survived=new_survived,
            lane_directions=new_lane_directions,
            spawn_timers=new_spawn_timers,
            enemy_offscreen_timers=new_offscreen_timers,
        )

        new_shark_positions, new_sub_positions = jnp.split(new_all_positions, 2, axis=0)
        new_shark_positions = new_shark_positions.astype(shark_positions.dtype)
        new_sub_positions = new_sub_positions.astype(sub_positions.dtype)

        return new_shark_positions, new_sub_positions, new_spawn_state, rng

    @partial(jax.jit, static_argnums=(0,))
    def spawn_divers(
        self,
        spawn_state: SpawnState,
        diver_positions: chex.Array,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        step_counter: chex.Array,
    ) -> tuple[chex.Array, SpawnState]:
        """
        Vectorized function to spawn divers according to pattern that depends on collection state.
        """
        # --- 1. Vectorized Pre-computation and Checks (for all 4 lanes at once) ---

        # Divers only place on the shark co-spawn trigger (same tick as escorts).
        # Death/surface "instant" rolls set diver_instant_pending; spawn_step /
        # surface-unfreeze snap those lanes onto TRIGGER so they co-spawn with
        # the next wave — never alone ahead of escorts.
        instant_pending = spawn_state.diver_instant_pending.astype(bool)
        timers_ready_mask = (
            spawn_state.spawn_timers == self.consts.DIVER_SPAWN_TIMER_TRIGGER
        )

        # Condition: A diver must not already exist in the lane.
        diver_exists_mask = diver_positions[:, 2] != 0  # Shape: (4,)

        # Condition: The enemy lane must be empty.
        sharks_active_per_lane = jnp.any(shark_positions.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        subs_active_per_lane = jnp.any(sub_positions.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        lanes_are_empty_mask = jnp.logical_not(
            jnp.logical_or(sharks_active_per_lane, subs_active_per_lane)
        ) # Shape: (4,)

        # Condition: The lane must be marked as available for spawning (value of 1).
        lanes_ready_to_spawn_mask = spawn_state.diver_array == 1  # Shape: (4,)

        # Opening / first shark wave per lane: only upper two lanes place divers.
        # Use per-lane prev_sub < 0 — a global "all lanes still opening" check
        # failed when FIRST_WAVE_LANE_DELAY held the bottom lane back (siblings
        # already had prev_sub=0, so bottom got a diver on its delayed first place).
        is_lane_first_escort = spawn_state.prev_sub < 0
        opening_lane_mask = self.consts.FIRST_WAVE_DIVER_LANES.astype(bool)
        lane_place_mask = jnp.where(
            is_lane_first_escort, opening_lane_mask, jnp.ones(4, dtype=bool)
        )
        lane_place_mask = jnp.logical_or(lane_place_mask, instant_pending)

        # Condition: Do not spawn a diver if *this lane's* next escort is a sub.
        survived_lanes = jnp.any(
            spawn_state.survived.reshape(4, 3) != 0, axis=1
        )
        next_is_sub_mask = jnp.logical_and(
            survived_lanes, spawn_state.prev_sub == 0
        )

        # --- 2. Final spawn mask ---
        # Instant-roll losers carry diver_suppress_next > 0 and skip this
        # shark-diver opportunity (consumed below even if they did not place).
        suppress_mask = spawn_state.diver_suppress_next > 0
        eligible_mask = jnp.logical_and.reduce(
            jnp.array([
                timers_ready_mask,
                jnp.logical_not(diver_exists_mask),
                lanes_ready_to_spawn_mask,
                lane_place_mask,
                jnp.logical_not(next_is_sub_mask),
            ])
        )
        should_spawn_mask = jnp.logical_and(
            eligible_mask, jnp.logical_not(suppress_mask)
        )
        new_suppress = jnp.where(
            eligible_mask,
            jnp.maximum(spawn_state.diver_suppress_next - jnp.int32(1), jnp.int32(0)),
            spawn_state.diver_suppress_next,
        )

        # --- 3. Calculate New Positions and State ---

        # Opening: ALE OC shows divers at x=1 / x=159. We place at 0 / 160 so the
        # same-frame diver-move tick (diff-0 schedule) advances them onto 1 / 159
        # and stays lockstep with ALE; placing at 1/159 then moving overshoots.
        moving_left_mask = spawn_state.lane_directions == 1
        x_positions = jnp.where(
            moving_left_mask,
            jnp.int32(160),
            jnp.int32(0),
        )
        directions = jnp.where(moving_left_mask, -1, 1)

        # Create the potential new diver data for all 4 lanes.
        potential_new_divers = jnp.stack(
            [x_positions, self.consts.DIVER_SPAWN_POSITIONS, directions], axis=1
        ) # Shape: (4, 3)

        # Use the final spawn mask to decide whether to use the new diver data or keep the old.
        # The `[:, None]` broadcasts the (4,) mask to the (4, 3) diver_positions array.
        new_diver_positions = jnp.where(
            should_spawn_mask[:, None],
            potential_new_divers,
            diver_positions
        )

        # Update the diver_array: swam-off (-1) re-arms when the *diver slot*
        # is empty — not when the escort lane is empty. Continuous shark farming
        # otherwise leaves da stuck at -1 forever; collected top lanes (0) then
        # never see rearm_all, so divers vanish for the rest of the dive.
        spawn_next_cycle_mask = jnp.logical_and(
            spawn_state.diver_array == -1,
            jnp.logical_not(diver_exists_mask),
        )
        new_diver_array = jnp.where(
            spawn_next_cycle_mask,
            1,
            spawn_state.diver_array
        )

        # Consume instant-pending on the reserved co-spawn tick (whether or not
        # a diver actually placed — e.g. blocked by next-is-sub), and whenever
        # we successfully place. Drop pending if the lane is no longer armed.
        new_pending = jnp.where(
            jnp.logical_or(should_spawn_mask, timers_ready_mask),
            jnp.int32(0),
            spawn_state.diver_instant_pending,
        )
        new_pending = jnp.where(new_diver_array == 1, new_pending, jnp.int32(0))

        return new_diver_positions, spawn_state.replace(
            diver_array=new_diver_array,
            diver_instant_pending=new_pending,
            diver_suppress_next=new_suppress,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_diver_movement(
        self,
        diver_positions: chex.Array,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        state_player_x: chex.Array,
        state_player_y: chex.Array,
        state_divers_collected: chex.Array,
        spawn_state: SpawnState,
        step_counter: chex.Array,
        rng: chex.PRNGKey,
    ) -> tuple[chex.Array, chex.Array, SpawnState, chex.PRNGKey]:
        """Move divers according to their pattern and handle collisions.
        Returns updated diver positions, number of collected divers, updated spawn state, and updated RNG key.
        """
        new_diver_array = spawn_state.diver_array

        def calculate_enemy_movement_speed(step_counter, difficulty):
            """Identical schedule to ``step_enemy_movement`` (escorts drag divers)."""
            safe_difficulty = jnp.maximum(0, difficulty % 256)
            diff_lt_10 = safe_difficulty < 10
            cycle_pos = step_counter % 12
            # Diff 0 must use the same 3/8 schedule as sharks — the old
            # (cycle_pos%3)==0 path left divers at 1/3 while sharks ran at
            # 3/8, so escorts walked through divers.
            step8 = step_counter % 8
            move_375 = jnp.logical_or(
                step8 == 2, jnp.logical_or(step8 == 4, step8 == 7)
            )
            should_move_patterns = jnp.array(
                [
                    move_375,  # 37.5% (ALE opening / shark lockstep)
                    (cycle_pos % 2) == 0,  # 50%
                    (cycle_pos % 3) != 2,  # 67%
                    (cycle_pos % 4) != 3,  # 75%
                    (cycle_pos % 6) != 5,  # 83%
                    cycle_pos != 11,  # 92%
                ]
            )
            indices = jnp.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
            should_move = should_move_patterns[indices[jnp.minimum(safe_difficulty, 9)]]
            speed_for_diff_0_9 = jnp.where(should_move, 1, 0)

            diff_above_threshold = jnp.maximum(0, safe_difficulty - 10)
            base_speed = 1 + (diff_above_threshold // 16)
            position_in_tier = diff_above_threshold % 16
            higher_speed_patterns = jnp.array(
                [
                    (step_counter % 16) == 0,
                    (step_counter % 8) == 0,
                    (step_counter % 4) == 0,
                    (step_counter % 2) == 0,
                    (step_counter % 4) != 0,
                    (step_counter % 8) != 0,
                    (step_counter % 16) != 0,
                ]
            )
            tier_indices = jnp.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6])
            use_higher_speed = higher_speed_patterns[tier_indices[position_in_tier]]
            speed_for_diff_10_plus = jnp.where(
                use_higher_speed, base_speed + 1, base_speed
            )
            return jnp.where(diff_lt_10, speed_for_diff_0_9, speed_for_diff_10_plus)

        def calculate_diver_movement(step_counter, difficulty):
            """Calculate diver movement based on difficulty level.

            Args:
                step_counter: Current step counter (frame number)
                difficulty: Current difficulty level (0-255)

            Returns:
                Movement speed for the current frame (0, 1, or 2+)
                0 = no movement, 1 = normal speed, 2+ = higher speeds
            """
            # Ensure difficulty is non-negative and handle wrapping
            safe_difficulty = jnp.clip(difficulty % 256, 0, 255)

            # For difficulties 0-27, we have specific movement patterns
            is_high_difficulty = safe_difficulty >= 28

            # For difficulties 0-27, determine if we should move and use speed 1
            low_diff_should_move = determine_low_difficulty_movement(
                step_counter, safe_difficulty
            )
            low_diff_speed = jnp.where(low_diff_should_move, 1, 0)

            # For difficulties 28+, always move but with varying speed
            high_diff_speed = determine_high_difficulty_speed(step_counter, safe_difficulty)

            # Return appropriate speed based on difficulty
            return jnp.where(is_high_difficulty, high_diff_speed, low_diff_speed)

        def determine_low_difficulty_movement(step_counter, difficulty):
            """Determine if the diver should move for difficulties 0-27.

            Brackets are pairs (0-1, 2-3, …); index with difficulty//2 instead of
            a 14-way jnp.select (same result, far cheaper in XLA).
            """
            cycle_6_7 = step_counter % 8
            move_6_7 = jnp.logical_or(
                cycle_6_7 == 0, jnp.logical_or(cycle_6_7 == 3, cycle_6_7 == 5)
            )
            cycle_8_9 = step_counter % 10
            move_8_9 = (
                (cycle_8_9 == 0)
                | (cycle_8_9 == 2)
                | (cycle_8_9 == 4)
                | (cycle_8_9 == 7)
                | (cycle_8_9 == 9)
            )
            cycle_12_13 = step_counter % 8
            move_12_13 = (
                (cycle_12_13 == 0)
                | (cycle_12_13 == 2)
                | (cycle_12_13 == 4)
                | (cycle_12_13 == 6)
                | (cycle_12_13 == 7)
            )
            cycle_14_15 = step_counter % 7
            move_14_15 = (
                (cycle_14_15 == 0)
                | (cycle_14_15 == 1)
                | (cycle_14_15 == 3)
                | (cycle_14_15 == 5)
                | (cycle_14_15 == 6)
            )
            cycle_16_17 = step_counter % 10
            move_16_17 = (
                (cycle_16_17 == 0)
                | (cycle_16_17 == 1)
                | (cycle_16_17 == 3)
                | (cycle_16_17 == 4)
                | (cycle_16_17 == 6)
                | (cycle_16_17 == 8)
                | (cycle_16_17 == 9)
            )
            moves = jnp.array(
                [
                    # Diff 0-1: ALE opening divers advance on a 5/5/6 cadence
                    # (3 moves / 16 frames). Residues locked on seaquest_play01
                    # (move when pre-increment step%16 ∈ {4,9,14}). Old (%5==0)
                    # was ~2f early. Opening spawn uses x=0/160 so the co-timed
                    # first move lands on ALE's x=1/159 the same frame.
                    jnp.logical_or(
                        (step_counter % 16) == 4,
                        jnp.logical_or(
                            (step_counter % 16) == 9,
                            (step_counter % 16) == 14,
                        ),
                    ),
                    (step_counter % 4) == 0,  # 2-3
                    (step_counter % 3) == 0,  # 4-5
                    move_6_7,  # 6-7
                    move_8_9,  # 8-9
                    (step_counter % 2) == 0,  # 10-11
                    move_12_13,  # 12-13
                    move_14_15,  # 14-15
                    move_16_17,  # 16-17
                    (step_counter % 4) != 3,  # 18-19
                    (step_counter % 5) != 4,  # 20-21
                    (step_counter % 8) != 7,  # 22-23
                    (step_counter % 16) != 15,  # 24-25
                    True,  # 26-27
                ]
            )
            return moves[jnp.minimum(difficulty // 2, 13)]

        def determine_high_difficulty_speed(step_counter, difficulty):
            """Determine the speed (1 or 2+) for difficulties 28+."""
            diff_above_27 = difficulty - 28
            tier = diff_above_27 // 16
            position_in_tier = diff_above_27 % 16
            base_speed = tier + 1
            higher_speed = tier + 2
            # Same tier-position → frequency table as enemy movement (index, not select).
            higher_speed_patterns = jnp.array(
                [
                    (step_counter % 16) == 15,
                    (step_counter % 8) == 7,
                    (step_counter % 4) == 3,
                    (step_counter % 2) == 1,
                    (step_counter % 4) != 0,
                    (step_counter % 8) != 0,
                    (step_counter % 16) != 0,
                ]
            )
            # pos 0→0; 1-3→1; 4-6→2; 7-9→3; 10-12→4; 13-14→5; 15→6
            tier_indices = jnp.array(
                [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6]
            )
            use_higher_speed = higher_speed_patterns[tier_indices[position_in_tier]]
            return jnp.where(use_higher_speed, higher_speed, base_speed)

        # Speeds depend only on (step, difficulty) — hoist out of the per-diver loop.
        diver_speed = calculate_diver_movement(step_counter, spawn_state.difficulty)
        escort_speed = calculate_enemy_movement_speed(
            step_counter, spawn_state.difficulty
        )

        def move_single_diver(i, carry):
            # Unpack carry state - (positions, collected_count, diver_array)
            positions, collected, diver_array = carry
            diver_pos = positions[i]

            # Only process active divers (direction != 0)
            is_active = diver_pos[2] != 0

            # Check for collision with player first if diver is active
            player_collision = jnp.logical_and(
                is_active,
                self.check_collision_single(
                    jnp.array([state_player_x, state_player_y]),
                    self.consts.PLAYER_SIZE,
                    jnp.array([diver_pos[0], diver_pos[1]]),
                    self.consts.DIVER_SIZE,
                ),
            )

            # Only collect if we haven't reached max divers
            can_collect = state_divers_collected < 6
            should_collect = jnp.logical_and(player_collision, can_collect)

            # Escorts in the same lane (sharks only — divers never share with subs).
            all_shark_lane_pos = jax.lax.dynamic_slice(shark_positions, (i * 3, 0), (3, 3))

            # Escort spacing: ALE locks ~5px. Soft-attach within 6px (≤1px nudge);
            # outside that band divers keep their normal solo schedule.
            all_escorts = all_shark_lane_pos
            escort_alive = all_escorts[:, 2] != 0
            dists = jnp.where(
                escort_alive,
                jnp.abs(all_escorts[:, 0] - diver_pos[0]),
                jnp.int32(10_000),
            )
            nearest_idx = jnp.argmin(dists)
            escort = all_escorts[nearest_idx]
            escort_active = escort_alive[nearest_idx]
            horiz_gap = jnp.abs(escort[0] - diver_pos[0])
            escort_gap_target = jnp.int32(5)
            escort_lock = jnp.logical_and(
                is_active,
                jnp.logical_and(escort_active, horiz_gap <= jnp.int32(6)),
            )

            # Heading: follow live escort when locked; otherwise match the
            # planned next-wave direction (lane_directions). Divers are not
            # independently random — they track current/next escort heading.
            planned_dir = jnp.where(
                spawn_state.lane_directions[i] == 1, jnp.int32(-1), jnp.int32(1)
            )
            any_escort = jnp.any(escort_alive)
            move_dir = jnp.where(
                escort_lock,
                escort[2],
                jnp.where(
                    jnp.logical_and(is_active, jnp.logical_not(any_escort)),
                    planned_dir,
                    diver_pos[2],
                ),
            )

            movement_speed = jnp.where(escort_lock, escort_speed, diver_speed)
            should_move = movement_speed > 0

            movement_x = move_dir * movement_speed

            attached_x = escort[0] + move_dir * escort_gap_target
            new_x = jnp.where(
                escort_lock,
                attached_x,
                jnp.where(
                    should_move,
                    diver_pos[0] + movement_x,
                    diver_pos[0],
                ),
            )

            # Divers despawn past the visible edges. Allow x=0 and x=160 for one
            # frame so left-entry / right-entry co-spawns (placed at 0 / 160) are
            # not wiped before the first move tick advances them to 1 / 159.
            out_of_bounds = jnp.logical_or(
                new_x < self.consts.X_BORDERS[0],
                new_x > self.consts.X_BORDERS[1],
            )

            # Create new position array - handle collection and bounds
            new_pos = jnp.where(
                jnp.logical_or(~is_active, jnp.logical_or(out_of_bounds, should_collect)),
                jnp.zeros(3),  # Reset if out of bounds or collected
                jnp.array([new_x, self.consts.DIVER_SPAWN_POSITIONS[i], move_dir]),
            )

            # Update collection count if collected
            new_collected = collected + jnp.where(should_collect, 1, 0)

            # Update diver collection tracking - mark lane as collected when diver is collected
            updated_diver_array = diver_array.at[i].set(
                jnp.where(should_collect, 0, diver_array[i])
            )

            # if the diver went out of bounds set the entry to -1
            updated_diver_array = updated_diver_array.at[i].set(
                jnp.where(out_of_bounds, -1, updated_diver_array[i])
            )

            # Update the diver position, collection count and diver_array
            return positions.at[i].set(new_pos), new_collected, updated_diver_array

        # Update all diver positions and track collections
        initial_carry = (diver_positions, state_divers_collected, new_diver_array)
        final_positions, final_collected, final_diver_array = jax.lax.fori_loop(
            0, diver_positions.shape[0], move_single_diver, initial_carry
        )

        # Cycle complete only when every lane was *collected* (da==0). A swim-off
        # (-1→1 when the diver slot is empty) must NOT re-open previously
        # collected lanes — each lane stays dark until the full set is bagged.
        rearm_all = jnp.all(final_diver_array == 0)
        reset_array = jnp.where(
            rearm_all,
            jnp.ones(4, dtype=jnp.int32),
            final_diver_array,
        )
        # After a full-collect rearm, skip one shark-diver opportunity so the
        # next wave is not an immediate 4-diver dump (surface soft_reset also
        # rolls instant + suppress; this covers mid-dive rearm).
        new_suppress = jnp.where(
            rearm_all,
            jnp.ones(4, dtype=jnp.int32),
            spawn_state.diver_suppress_next,
        )

        return (
            final_positions,
            final_collected,
            spawn_state.replace(
                diver_array=reset_array,
                diver_suppress_next=new_suppress,
            ),
            rng,
        )

    @partial(jax.jit, static_argnums=(0,))
    def spawn_step(
        self,
        state,
        spawn_state: SpawnState,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        diver_positions: chex.Array,
        rng_key: chex.PRNGKey,
    ) -> Tuple[SpawnState, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Main spawn handling function to be called in game step"""
        # Move existing enemies
        new_shark_positions, new_sub_positions, spawn_state_after_movement, new_key = (
            self.step_enemy_movement(
                spawn_state, shark_positions, sub_positions, state.step_counter, rng_key
            )
        )

        # Shared countdown: divers + escorts both fire at DIVER_SPAWN_TIMER_TRIGGER.
        decremented_timers = jnp.where(
            spawn_state_after_movement.spawn_timers > 0,
            spawn_state_after_movement.spawn_timers - 1,
            spawn_state_after_movement.spawn_timers,
        )
        spawn_state_ticked = spawn_state_after_movement.replace(
            spawn_timers=decremented_timers,
            episode_frame=spawn_state_after_movement.episode_frame + jnp.int32(1),
        )

        # Divers first (lane still empty), then escorts same tick (far off-screen).
        new_diver_positions, spawn_state_after_divers = self.spawn_divers(
            spawn_state_ticked,
            diver_positions,
            new_shark_positions,
            new_sub_positions,
            state.step_counter,
        )

        new_spawn_state, new_shark_positions, new_sub_positions, new_diver_positions, new_key = (
            self.update_enemy_spawns(
                spawn_state_after_divers,
                new_shark_positions,
                new_sub_positions,
                new_diver_positions,
                state.step_counter,
                new_key,
            )
        )

        return (
            new_spawn_state,
            new_shark_positions,
            new_sub_positions,
            new_diver_positions,
            new_key,
        )


    def surface_sub_step(self, state: SeaquestState) -> chex.Array:
        # Check direction value specifically to get scalar boolean
        sub_exists = state.surface_sub_position[2] != 0

        def spawn_sub(_):
            return jnp.array([159, 45, -1])  # Always spawns right facing left

        def move_sub(carry):
            sub_pos = carry
            new_x = jnp.where(
                state.step_counter % 4 == 0,
                sub_pos[0] - 1,  # Direction always -1
                sub_pos[0],
            )

            # Return either zeros or new position
            return jnp.where(
                jnp.logical_or(new_x < -8, sub_pos[2] == 0),
                jnp.zeros(3),
                jnp.array([new_x, 45, -1]),
            )

        # Each condition needs to be scalar
        enough_rescues = state.successful_rescues >= 2
        enough_divers = state.divers_collected >= 1
        correct_timing = jnp.logical_and(
            state.step_counter % 256 == 0, state.step_counter != 0
        )

        # check if the submarine should spawn
        should_spawn = jnp.logical_and(
            jnp.logical_and(enough_rescues, enough_divers),
            jnp.logical_and(correct_timing, ~sub_exists),
        )

        temp1 = spawn_sub(state.surface_sub_position)
        temp2 = move_sub(state.surface_sub_position)

        return jnp.where(should_spawn, temp1, temp2)

    @partial(jax.jit, static_argnums=(0,))
    def enemy_missiles_step(
        self, curr_sub_positions, curr_enemy_missile_positions, step_counter, difficulty
    ) -> chex.Array:

        def calculate_missile_speed(step_counter, difficulty):
            """JAX-compatible missile speed calculation function"""
            # Base tier size is 16 difficulty levels
            tier_size = 16

            # Determine base speed (1, 2, 3, etc.) based on difficulty tier
            base_speed = 1 + (difficulty // tier_size)

            # Calculate position within the current tier (0-15)
            position_in_tier = difficulty % tier_size

            # Special case for difficulty 0
            is_diff_0 = difficulty == 0

            # Create position bracket array for each pattern
            pos_brackets = jnp.array(
                [
                    jnp.logical_and(
                        position_in_tier >= 0, position_in_tier <= 2
                    ),  # 0-2: 6.25%
                    jnp.logical_and(
                        position_in_tier >= 3, position_in_tier <= 4
                    ),  # 3-4: 12.5%
                    jnp.logical_and(
                        position_in_tier >= 5, position_in_tier <= 6
                    ),  # 5-6: 25%
                    jnp.logical_and(
                        position_in_tier >= 7, position_in_tier <= 8
                    ),  # 7-8: 50%
                    jnp.logical_and(
                        position_in_tier >= 9, position_in_tier <= 10
                    ),  # 9-10: 75%
                    jnp.logical_and(
                        position_in_tier >= 11, position_in_tier <= 12
                    ),  # 11-12: 87.5%
                    jnp.logical_and(
                        position_in_tier >= 13, position_in_tier <= 14
                    ),  # 13-14: 93.75%
                    position_in_tier == 15,  # 15: 100%
                ]
            )

            # Create array of higher speed patterns
            higher_speed_patterns = jnp.array(
                [
                    (step_counter % 16) == 0,  # 6.25%
                    (step_counter % 8) == 0,  # 12.5%
                    (step_counter % 4) == 0,  # 25%
                    (step_counter % 2) == 0,  # 50%
                    (step_counter % 4) != 0,  # 75%
                    (step_counter % 8) != 0,  # 87.5%
                    (step_counter % 16) != 0,  # 93.75%
                    True,  # 100%
                ]
            )

            # Use jnp.select to choose the pattern
            use_higher_speed = jnp.select(
                pos_brackets, higher_speed_patterns, default=False
            )

            # Higher speed is base_speed + 1
            higher_speed = base_speed + 1

            # Handle difficulty 0 special case
            return jnp.where(
                is_diff_0, 1, jnp.where(use_higher_speed, higher_speed, base_speed)
            )

        # 1. Define a function that operates on a SINGLE missile and its corresponding lane.
        #    It no longer needs an index `i` or a `carry` argument.
        def vmapped_missile_update(missile_pos, lane_subs, lane_y_pos):
            # Get the front submarine for this specific lane
            sub_pos = self.get_front_entity(0, lane_subs) # Index 0 is fine since it only looks at the 3 subs passed in

            # Check if the missile should be spawned
            missile_exists = missile_pos[2] != 0
            should_spawn = jnp.logical_and(
                ~missile_exists,
                (sub_pos[0] >= self.consts.MISSILE_SPAWN_POSITIONS[0]) &
                (sub_pos[0] <= self.consts.MISSILE_SPAWN_POSITIONS[1])
            )

            # Calculate new missile position
            new_missile_x = sub_pos[0] + 4 * sub_pos[2]
            spawned_missile = jnp.array([new_missile_x, lane_y_pos, sub_pos[2]])
            new_missile = jnp.where(should_spawn, spawned_missile, missile_pos)

            # Move the missile if it exists
            movement_speed = calculate_missile_speed(step_counter, difficulty)
            velocity = movement_speed * new_missile[2]
            moved_missile = new_missile.at[0].add(velocity)
            new_missile = jnp.where(missile_exists, moved_missile, new_missile)

            # Check bounds and return
            is_out_of_bounds = (new_missile[0] < self.consts.X_BORDERS[0]) | (new_missile[0] > self.consts.X_BORDERS[1])
            return jnp.where(is_out_of_bounds, jnp.zeros(3), new_missile)

        # 2. Prepare the inputs for vmap
        # Reshape subs into a per-lane format: (4 lanes, 3 subs per lane, 3 coords)
        all_lane_subs = curr_sub_positions.reshape(4, 3, 3)

        # 3. Use jax.vmap to apply the update function in parallel
        new_missile_positions = jax.vmap(
            vmapped_missile_update, in_axes=(0, 0, 0) # Map over missiles, sub-lanes, and y-positions
        )(curr_enemy_missile_positions, all_lane_subs, self.consts.ENEMY_MISSILE_Y)

        return new_missile_positions

    @partial(jax.jit, static_argnums=(0,))
    def player_missile_step(
        self, state: SeaquestState, curr_player_x, curr_player_y, action: chex.Array
    ) -> chex.Array:
        # check if the player shot this frame
        fire = jnp.any(
            jnp.array(
                [
                    action == Action.FIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                    action == Action.RIGHTFIRE,
                    action == Action.LEFTFIRE,
                    action == Action.UPFIRE,
                ]
            )
        )

        # IMPORTANT: do not change the order of this check, since the missile does not move in its first frame!!
        # also check if there is currently a missile in frame by checking if the player_missile_position is empty
        missile_exists = state.player_missile_position[2] != 0

        # if the player shot and there is no missile in frame, then we can shoot a missile
        # Missile y = player_y + PLAYER_MISSILE_Y_OFFSET; x depends on facing.
        y_off = self.consts.PLAYER_MISSILE_Y_OFFSET.astype(jnp.int32)
        new_missile = jnp.where(
            jnp.logical_and(fire, jnp.logical_not(missile_exists)),
            jnp.where(
                state.player_direction == -1,
                jnp.array([curr_player_x + 3, curr_player_y + y_off, -1]),
                jnp.array([curr_player_x + 13, curr_player_y + y_off, 1]),
            ),
            state.player_missile_position,
        )

        # if a missile is in frame and exists, we move the missile further in the specified direction (5 per tick), also always put the missile at the current player y position
        new_missile = jnp.where(
            missile_exists,
            jnp.array(
                [new_missile[0] + new_missile[2] * 5, curr_player_y + y_off, new_missile[2]]
            ),
            new_missile,
        )

        # check if the new positions are still in bounds
        new_missile = jnp.where(
            new_missile[0] < self.consts.X_BORDERS[0],
            jnp.array([0, 0, 0]),
            jnp.where(new_missile[0] > self.consts.X_BORDERS[1], jnp.array([0, 0, 0]), new_missile),
        )

        return new_missile

    @partial(jax.jit, static_argnums=(0,))
    def update_oxygen(self, state, player_x, player_y, player_missile_position):
        """Update oxygen levels and handle surfacing mechanics with proper surfacing detection"""
        PLAYER_BREATHING_Y = [47, 52]  # Range where oxygen neither increases nor decreases

        # Detect actual surfacing moment
        at_surface = player_y == 46
        was_underwater = player_y > 46
        just_surfaced = jnp.logical_and(at_surface, state.just_surfaced == 0)

        # Check player state
        decrease_ox = player_y > PLAYER_BREATHING_Y[1]
        has_divers = state.divers_collected >= 0  # Changed to > 0 instead of >= 0
        has_all_divers = state.divers_collected >= 6
        needs_oxygen = state.oxygen < 64

        # Special handling for initialization state
        in_init_state = state.just_surfaced == -1
        started_diving = player_y > self.consts.PLAYER_START_Y
        filling_init_oxygen = jnp.logical_and(in_init_state, state.oxygen < 64)

        # Surfacing conditions
        increase_ox = jnp.logical_and(at_surface, needs_oxygen)
        stay_same = jnp.logical_and(
            player_y >= PLAYER_BREATHING_Y[0], player_y <= PLAYER_BREATHING_Y[1]
        )

        # Calculate new divers count before other logic
        # Mid-surface with 1–5 divers: deposit one immediately. Full bag (6):
        # keep all six for the score-freeze animation (ALE pays oxygen then all
        # six diver bonuses; decrementing here dropped one +50 payout).
        new_divers_collected = jnp.where(
            jnp.logical_and(just_surfaced, has_divers),
            jnp.where(
                in_init_state,
                state.divers_collected,
                jnp.where(
                    has_all_divers,
                    state.divers_collected,
                    state.divers_collected - 1,
                ),
            ),
            state.divers_collected,
        )

        # Handle surfacing without divers - prevent during init
        # Only lose life if we started with no divers
        lose_life = jnp.logical_and(
            jnp.logical_and(just_surfaced, new_divers_collected < 0),
            jnp.logical_not(in_init_state),
        )

        # Handle surfacing with all divers
        should_reset = jnp.logical_and(just_surfaced, has_all_divers)

        # Update surfacing flag with consideration for remaining divers
        new_just_surfaced = jnp.where(
            in_init_state,
            jnp.where(
                jnp.logical_and(started_diving, state.oxygen >= 63),
                jnp.array(0),
                jnp.array(-1),
            ),
            jnp.where(
                was_underwater,
                jnp.array(0),
                jnp.where(at_surface, jnp.array(1), state.just_surfaced),
            ),
        )

        # Handle oxygen changes
        new_oxygen = jnp.where(
            filling_init_oxygen,
            jnp.where(state.step_counter % 2 == 0, state.oxygen + 1, state.oxygen),
            jnp.where(
                decrease_ox,
                jnp.where(state.step_counter % 32 == 0, state.oxygen - 1, state.oxygen),
                state.oxygen,
            ),
        )

        # Important: Base blocking decision on has_divers instead of still_has_divers
        can_refill = jnp.logical_and(increase_ox, has_divers)
        new_oxygen = jnp.where(
            jnp.logical_and(can_refill, jnp.logical_not(in_init_state)),
            jnp.where(
                state.oxygen < 64,
                jnp.where(state.step_counter % 2 == 0, state.oxygen + 1, state.oxygen),
                state.oxygen,
            ),
            new_oxygen,
        )

        # Increase difficulty when reaching max oxygen after surfacing
        old_difficulty = state.spawn_state.difficulty
        reached_max = jnp.logical_and(
            jnp.logical_and(new_oxygen >= 64, state.oxygen < 64),
            jnp.logical_not(in_init_state),
        )
        new_difficulty = jnp.where(reached_max, old_difficulty + 1, old_difficulty)

        new_oxygen = jnp.where(stay_same, state.oxygen, new_oxygen)

        # Use has_divers for blocking decision and combine with oxygen check
        should_block = jnp.logical_and(at_surface, needs_oxygen)

        player_x = jnp.where(should_block, state.player_x, player_x)

        player_y = jnp.where(
            should_block,
            jnp.array(46, dtype=jnp.int32),  # Force to exact surface position
            player_y,
        )

        player_missile_position = jnp.where(
            should_block, jnp.zeros(3), player_missile_position
        )

        # Prevent oxygen depletion during init
        oxygen_depleted = jnp.logical_and(
            new_oxygen <= jnp.array(0), jnp.logical_not(in_init_state)
        )

        return (
            new_oxygen,
            player_x,
            player_y,
            player_missile_position,
            oxygen_depleted,
            lose_life,
            new_divers_collected,
            should_reset,
            new_just_surfaced,
            new_difficulty,
        )

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self, state: SeaquestState, action: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        # implement all the possible movement directions for the player, the mapping is:
        # anything with left in it, add -1 to the x position
        # anything with right in it, add 1 to the x position
        # anything with up in it, add -1 to the y position
        # anything with down in it, add 1 to the y position
        up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                    action == Action.UPRIGHT,
                    action == Action.UPLEFT,
                    action == Action.UPFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                    action == Action.LEFTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )

        player_x = jnp.where(
            right, state.player_x + 1, jnp.where(left, state.player_x - 1, state.player_x)
        )

        player_y = jnp.where(
            down, state.player_y + 1, jnp.where(up, state.player_y - 1, state.player_y)
        )

        # set the direction according to the movement
        player_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_direction))

        # perform out of bounds checks
        player_x = jnp.where(
            player_x < self.consts.PLAYER_BOUNDS[0][0],
            self.consts.PLAYER_BOUNDS[0][0],  # Clamp to min player bound
            jnp.where(
                player_x > self.consts.PLAYER_BOUNDS[0][1],
                self.consts.PLAYER_BOUNDS[0][1],  # Clamp to max player bound
                player_x,
            ),
        )

        player_y = jnp.where(
            player_y < self.consts.PLAYER_BOUNDS[1][0],
            self.consts.PLAYER_BOUNDS[1][0],
            jnp.where(player_y > self.consts.PLAYER_BOUNDS[1][1], self.consts.PLAYER_BOUNDS[1][1], player_y),
        )

        return player_x, player_y, player_direction

    @partial(jax.jit, static_argnums=(0,))
    def calculate_kill_points(self, successful_rescues: chex.Array) -> chex.Array:
        """
        Calculate the points awarded for killing a shark or submarine.
        Scales based on successful rescues using defined constants.
        """
        bonus = self.consts.SCORE_ENEMY_STEP * successful_rescues
        points = self.consts.SCORE_ENEMY_BASE + bonus
        return jnp.minimum(points, self.consts.SCORE_ENEMY_MAX)


    # Minimal ALE action set for Seaquest (from scripts/action_space_helper.py)
    ACTION_SET: jnp.ndarray = jnp.array(
        [
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
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: SeaquestConstants = None):
        consts = consts or SeaquestConstants()
        super().__init__(consts)
        self.obs_size = 6 + 12 * 5 + 12 * 5 + 4 * 5 + 4 * 5 + 5 + 5 + 4
        self.renderer = SeaquestRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SeaquestState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        h = int(self.consts.SCREEN_HEIGHT)
        w = int(self.consts.SCREEN_WIDTH)
        screen_size = (h, w)

        single_obj = spaces.get_object_space(n=None, screen_size=screen_size)

        return spaces.Dict({
            "player": single_obj,
            "divers": spaces.get_object_space(n=self.consts.MAX_DIVERS, screen_size=screen_size),
            # Enemies: 12 Sharks + 12 Subs + 1 Surface Sub = 25
            "enemies": spaces.get_object_space(n=25, screen_size=screen_size),
            # Projectiles: 1 Player + 4 Enemy = 5
            "projectiles": spaces.get_object_space(n=5, screen_size=screen_size),

            "oxygen_level": spaces.Box(low=0, high=255, shape=(), dtype=jnp.int32),
            "player_score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
            "collected_divers": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for Seaquest.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SeaquestState) -> SeaquestObservation:
        c = self.consts
        w, h = int(c.SCREEN_WIDTH), int(c.SCREEN_HEIGHT)

        # --- Helper for orientation ---
        def get_orientation(direction):
            # 1 -> 90.0 (Right), -1 -> 270.0 (Left), 0 -> 0.0 (Inactive/None)
            return jnp.select(
                [direction == 1, direction == -1],
                [90.0, 270.0],
                0.0
            ).astype(jnp.float32)

        # --- Player ---
        player = ObjectObservation.create(
            x=jnp.clip(jnp.array(state.player_x, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(state.player_y, dtype=jnp.int32), 0, h),
            width=jnp.array(c.PLAYER_SIZE[0], dtype=jnp.int32),
            height=jnp.array(c.PLAYER_SIZE[1], dtype=jnp.int32),
            active=jnp.array(1, dtype=jnp.int32),
            orientation=get_orientation(state.player_direction)
        )

        # --- Divers ---
        # Clip x into the playfield for consumers that expect screen coords, but
        # mark off-screen divers inactive so heuristics / skills ignore them.
        d_pos = state.diver_positions
        d_alive = d_pos[:, 2] != 0
        d_w = jnp.int32(c.DIVER_SIZE[0])
        d_onscreen = jnp.logical_and(d_pos[:, 0] < w, (d_pos[:, 0] + d_w) > 0)
        d_active = jnp.logical_and(d_alive, d_onscreen).astype(jnp.int32)
        divers = ObjectObservation.create(
            x=jnp.clip(d_pos[:, 0].astype(jnp.int32), 0, w),
            y=jnp.clip(d_pos[:, 1].astype(jnp.int32), 0, h),
            width=jnp.full((4,), c.DIVER_SIZE[0], dtype=jnp.int32),
            height=jnp.full((4,), c.DIVER_SIZE[1], dtype=jnp.int32),
            active=d_active,
            orientation=get_orientation(d_pos[:, 2])
        )

        # --- Enemies (Grouped) ---
        # 1. Sharks (12) - IDs 0-3 based on difficulty color
        sharks_pos = state.shark_positions
        shark_color_idx = get_shark_color_index(state.spawn_state.difficulty)
        sharks_vid = jnp.full((12,), shark_color_idx, dtype=jnp.int32)
        sharks_w = jnp.full((12,), c.SHARK_SIZE[0], dtype=jnp.int32)
        sharks_h = jnp.full((12,), c.SHARK_SIZE[1], dtype=jnp.int32)

        # 2. Submarines (12) - ID 4
        subs_pos = state.sub_positions
        subs_vid = jnp.full((12,), 4, dtype=jnp.int32)
        subs_w = jnp.full((12,), c.ENEMY_SUB_SIZE[0], dtype=jnp.int32)
        subs_h = jnp.full((12,), c.ENEMY_SUB_SIZE[1], dtype=jnp.int32)

        # 3. Surface Submarine (1) - ID 5
        surf_pos = state.surface_sub_position[None, :]
        surf_vid = jnp.array([5], dtype=jnp.int32)
        surf_w = jnp.array([c.ENEMY_SUB_SIZE[0]], dtype=jnp.int32)
        surf_h = jnp.array([c.ENEMY_SUB_SIZE[1]], dtype=jnp.int32)

        # Concatenate all enemies
        e_pos = jnp.concatenate([sharks_pos, subs_pos, surf_pos])
        e_vid = jnp.concatenate([sharks_vid, subs_vid, surf_vid])
        e_w = jnp.concatenate([sharks_w, subs_w, surf_w])
        e_h = jnp.concatenate([sharks_h, subs_h, surf_h])
        e_alive = e_pos[:, 2] != 0
        # Same gate as enemy_is_hittable: alive off-screen stays in state, but
        # observation.active=0 so clipped x cannot fake an on-screen target.
        e_onscreen = jnp.logical_and(
            e_pos[:, 0] >= c.ENEMY_HITTABLE_X_MIN,
            e_pos[:, 0] < c.ENEMY_HITTABLE_X_MAX,
        )
        e_active = jnp.logical_and(e_alive, e_onscreen).astype(jnp.int32)

        enemies = ObjectObservation.create(
            x=jnp.clip(e_pos[:, 0].astype(jnp.int32), 0, w),
            y=jnp.clip(e_pos[:, 1].astype(jnp.int32), 0, h),
            width=e_w,
            height=e_h,
            active=e_active,
            visual_id=e_vid,
            orientation=get_orientation(e_pos[:, 2])
        )

        # --- Projectiles (Grouped) ---
        # 1. Player Missile (1) - ID 0
        pm_pos = state.player_missile_position[None, :]
        pm_vid = jnp.array([0], dtype=jnp.int32)

        # 2. Enemy Missiles (4) - ID 1
        em_pos = state.enemy_missile_positions
        em_vid = jnp.full((4,), 1, dtype=jnp.int32)

        # Concatenate projectiles
        p_pos = jnp.concatenate([pm_pos, em_pos])
        p_vid = jnp.concatenate([pm_vid, em_vid])
        p_alive = p_pos[:, 2] != 0
        p_onscreen = jnp.logical_and(p_pos[:, 0] < w, (p_pos[:, 0] + c.MISSILE_SIZE[0]) > 0)
        p_active = jnp.logical_and(p_alive, p_onscreen).astype(jnp.int32)

        projectiles = ObjectObservation.create(
            x=jnp.clip(p_pos[:, 0].astype(jnp.int32), 0, w),
            y=jnp.clip(p_pos[:, 1].astype(jnp.int32), 0, h),
            width=jnp.full((5,), c.MISSILE_SIZE[0], dtype=jnp.int32),
            height=jnp.full((5,), c.MISSILE_SIZE[1], dtype=jnp.int32),
            active=p_active,
            visual_id=p_vid,
            orientation=get_orientation(p_pos[:, 2])
        )

        return SeaquestObservation(
            player=player,
            divers=divers,
            enemies=enemies,
            projectiles=projectiles,
            oxygen_level=state.oxygen.astype(jnp.int32),
            player_score=state.score.astype(jnp.int32),
            lives=state.lives.astype(jnp.int32),
            collected_divers=state.divers_collected.astype(jnp.int32)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SeaquestState) -> SeaquestInfo:
        return SeaquestInfo(
            successful_rescues=state.successful_rescues,
            difficulty=state.spawn_state.difficulty,
            step_counter=state.step_counter,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: SeaquestState, state: SeaquestState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SeaquestState) -> bool:
        return state.lives < 0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[SeaquestObservation, SeaquestState]:
        """Initialize game state"""
        reset_state = SeaquestState(
            player_x=jnp.array(self.consts.PLAYER_START_X),
            player_y=jnp.array(self.consts.PLAYER_START_Y),
            player_direction=jnp.array(0),
            oxygen=jnp.array(0),  # Full oxygen
            divers_collected=jnp.array(0),
            score=jnp.array(0),
            lives=jnp.array(3),
            spawn_state=self.initialize_spawn_state(),
            diver_positions=jnp.zeros((self.consts.MAX_DIVERS, 3)),  # 4 divers
            shark_positions=jnp.zeros((self.consts.MAX_SHARKS, 3)),
            sub_positions=jnp.zeros((self.consts.MAX_SUBS, 3)),  # x, y, direction
            enemy_missile_positions=jnp.zeros((self.consts.MAX_ENEMY_MISSILES, 3)),  # 4 missiles
            surface_sub_position=jnp.zeros(3),  # 1 surface sub
            player_missile_position=jnp.zeros(3),  # x,y,direction
            step_counter=jnp.array(0),
            just_surfaced=jnp.array(-1),
            successful_rescues=jnp.array(0),
            death_counter=jnp.array(0),
            rng_key=key,
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0, ))
    def step(
        self, state: SeaquestState, action: chex.Array
    ) -> Tuple[SeaquestObservation, SeaquestState, float, bool, SeaquestInfo]:
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        previous_state = state
        _, reset_state = self.reset(state.rng_key)

        # First handle death animation if active
        def handle_death_animation():
            # This outer conditional remains the same.
            # It decides if the animation is over or still running.
            is_animation_over = state.death_counter <= 1

            def on_animation_continue():
                # This is the original logic for when the animation is still running.
                # It correctly updates the Y-positions and the player visibility.
                shark_y_positions, _, _, _ = self.step_enemy_movement(
                    state.spawn_state,
                    state.shark_positions,
                    state.sub_positions,
                    state.step_counter,
                    state.rng_key,
                )
                new_shark_positions = state.shark_positions.at[:, 1].set(
                    shark_y_positions[:, 1]
                )
                should_hide_player = state.death_counter <= 45
                return state.replace(
                    death_counter=state.death_counter - 1,
                    shark_positions=new_shark_positions,
                    sub_positions=state.sub_positions,
                    enemy_missile_positions=state.enemy_missile_positions,
                    player_missile_position=jnp.zeros(3),
                    player_x=jnp.where(should_hide_player, -100, state.player_x),
                    step_counter=state.step_counter + 1,
                )

            def on_animation_over():
                # This is the new, more precise logic for when the animation ends.
                # We add a check to see if this is the absolute final life.
                is_final_life = state.lives <= 0

                def handle_game_over():
                    # If this is the final life, return the TRUE final state of the game
                    # while setting lives to -1 to trigger done=True.
                    # This is what your snapshot test needs to see.
                    return state.replace(
                        lives=state.lives - 1,
                        death_counter=0,
                    )

                def handle_stage_reset():
                    # If the player still has lives left, perform the original stage reset.
                    # This preserves the mechanic of resetting the level after losing a life.
                    # Keep per-lane collected bits — ALE does not re-arm bagged
                    # diver lanes on oxygen / collision death.
                    return reset_state.replace(
                        lives=state.lives - 1,
                        score=state.score,
                        successful_rescues=state.successful_rescues,
                        divers_collected=jnp.maximum(state.divers_collected - 1, 0),
                        spawn_state=self.soft_reset_spawn_state(
                            state.spawn_state,
                            rng=jax.random.fold_in(state.rng_key, state.step_counter),
                        ),
                    )

                # Use the new nested conditional to choose the correct outcome.
                return jax.lax.cond(
                    is_final_life,
                    lambda: handle_game_over(),
                    lambda: handle_stage_reset(),
                )

            # This is the main conditional call.
            return jax.lax.cond(
                is_animation_over,
                lambda: on_animation_over(),
                lambda: on_animation_continue(),
            )

        def handle_score_freeze():
            # Calculate points logic based on the rescues prior to the recent increment
            rescues_for_calc = state.successful_rescues - 1

            diver_bonus_val = self.consts.SCORE_DIVER_STEP * rescues_for_calc
            points_per_diver = jnp.minimum(
                self.consts.SCORE_DIVER_BASE + diver_bonus_val,
                self.consts.SCORE_DIVER_MAX,
            )

            oxygen_bonus_val = self.consts.SCORE_OXYGEN_STEP * rescues_for_calc
            points_per_oxygen_unit = jnp.minimum(
                self.consts.SCORE_OXYGEN_BASE + oxygen_bonus_val,
                self.consts.SCORE_OXYGEN_MAX
            )

            # Keep X positions from original state, only update Y for sharks
            shark_y_positions, _, _, _ = self.step_enemy_movement(
                state.spawn_state,
                state.shark_positions,
                state.sub_positions,
                state.step_counter,
                state.rng_key,
            )
            new_shark_positions = state.shark_positions.at[:, 1].set(
                shark_y_positions[:, 1]
            )

            # Animation has two phases: Oxygen phase (< -96) and Diver phase (>= -96)
            is_oxygen_phase = state.death_counter < -96
            is_diver_phase = state.death_counter >= -96

            # Phase 1: Drain oxygen (1 unit every 2 ticks)
            drain_this_tick = jnp.logical_and(is_oxygen_phase, state.death_counter % 2 == 0)
            has_oxygen = state.oxygen > 0
            actually_drain = jnp.logical_and(drain_this_tick, has_oxygen)

            new_ox = jnp.where(
                actually_drain,
                state.oxygen - 1,
                state.oxygen
            )
            oxygen_points = jnp.where(actually_drain, points_per_oxygen_unit, 0)

            # Phase 2: Free divers (1 diver every 16 ticks)
            free_diver_this_tick = jnp.logical_and(is_diver_phase, state.death_counter % 16 == 0)
            has_divers = state.divers_collected > 0
            actually_free = jnp.logical_and(free_diver_this_tick, has_divers)

            new_divers_collected = jnp.where(
                actually_free,
                jnp.maximum(state.divers_collected - 1, jnp.int32(0)),
                state.divers_collected
            )
            diver_points = jnp.where(actually_free, points_per_diver, 0)

            new_score = state.score + oxygen_points + diver_points

            # Return either final reset or animation frame
            return jax.lax.cond(
                state.death_counter >= -1,
                lambda _: reset_state.replace(
                    player_x=state.player_x,
                    player_y=state.player_y,
                    player_direction=state.player_direction,
                    score=new_score,
                    lives=state.lives,
                    successful_rescues=state.successful_rescues,
                    divers_collected=jnp.array(0),
                    spawn_state=self.soft_reset_spawn_state(
                        state.spawn_state,
                        rng=jax.random.fold_in(state.rng_key, state.step_counter),
                    ),
                    surface_sub_position=state.surface_sub_position,
                    oxygen=jnp.array(0),  # This triggers the oxygen refill mechanism
                ),
                lambda _: state.replace(
                    death_counter=state.death_counter + 1,
                    shark_positions=new_shark_positions,
                    sub_positions=state.sub_positions,
                    enemy_missile_positions=state.enemy_missile_positions,
                    player_missile_position=jnp.zeros(3),
                    step_counter=state.step_counter + 1,
                    oxygen=new_ox,
                    divers_collected=new_divers_collected,
                    score=new_score,
                ),
                operand=None,
            )

        # Normal game logic starts here
        def normal_game_step():
            # First check if player should be frozen for oxygen refill
            at_surface = state.player_y == 46
            needs_oxygen = state.oxygen < 64
            should_block = jnp.logical_and(at_surface, needs_oxygen)

            # Mid-game surface refill: hold spawn timers at SURFACE_FREEZE values.
            # Opening oxygen fill (just_surfaced == -1) must keep counting INITIAL timers
            # so the first wave lands near ALE frame ~260 instead of ~205.
            # After freeze: snap death/surface instant-pending lanes onto the
            # co-spawn trigger so they place *with* escorts, not alone early.
            in_init = state.just_surfaced == -1
            should_freeze_spawns = jnp.logical_and(should_block, jnp.logical_not(in_init))
            trigger_p1 = (
                self.consts.DIVER_SPAWN_TIMER_TRIGGER.astype(jnp.int32) + jnp.int32(1)
            )

            def _freeze_timers():
                return state.spawn_state.replace(
                    spawn_timers=jnp.array(
                        self.consts.SURFACE_FREEZE_SPAWN_TIMERS, dtype=jnp.int32
                    )
                )

            def _snap_instant_pending():
                ss = state.spawn_state
                pending = ss.diver_instant_pending.astype(bool)
                return ss.replace(
                    spawn_timers=jnp.where(pending, trigger_p1, ss.spawn_timers)
                )

            new_spawn_state = jax.lax.cond(
                should_freeze_spawns,
                _freeze_timers,
                _snap_instant_pending,
            )

            state_updated = state.replace(spawn_state=new_spawn_state)

            # If blocked, force position and disable actions
            player_x = jnp.where(should_block, state.player_x, state.player_x)
            player_y = jnp.where(
                should_block, jnp.array(46, dtype=jnp.int32), state.player_y
            )
            action_mod = jnp.where(should_block, jnp.array(Action.NOOP), atari_action)

            # Now calculate movement using potentially modified positions and action
            next_x, next_y, player_direction = self.player_step(
                state.replace(player_x=player_x, player_y=player_y), action_mod
            )
            player_missile_position = self.player_missile_step(
                state, next_x, next_y, action_mod
            )

            # Rest of oxygen handling and game logic
            (
                new_oxygen,
                player_x,
                player_y,
                player_missile_position,
                oxygen_depleted,
                lose_life_surfacing,
                new_divers_collected,
                should_reset,
                new_just_surfaced,
                new_difficulty,
            ) = self.update_oxygen(state, next_x, next_y, player_missile_position)

            # Update divers collected count from oxygen mechanics
            state_updated = state_updated.replace(
                divers_collected=new_divers_collected
            )

            # update the spawn state with the new difficulty
            new_spawn_state = state_updated.spawn_state.replace(
                difficulty=new_difficulty
            )

            # Check missile collisions
            (
                player_missile_position,
                new_shark_positions,
                new_sub_positions,
                new_diver_positions_after_kills,
                new_score,
                updated_spawn_state,
                new_rng_key,
            ) = self.check_missile_collisions(
                player_missile_position,
                state_updated.shark_positions,
                state_updated.sub_positions,
                state.diver_positions,
                state_updated.score,
                state_updated.successful_rescues,
                new_spawn_state,
                state.rng_key,
            )

            # perform all necessary spawn steps
            (
                new_spawn_state,
                new_shark_positions,
                new_sub_positions,
                new_diver_positions,
                new_rng_key,
            ) = self.spawn_step(
                state_updated,
                updated_spawn_state,
                new_shark_positions,
                new_sub_positions,
                new_diver_positions_after_kills,
                new_rng_key,
            )

            new_diver_positions, new_divers_collected, new_spawn_state, new_rng_key = (
                self.step_diver_movement(
                    new_diver_positions,
                    new_shark_positions,
                    new_sub_positions,
                    player_x,
                    player_y,
                    state_updated.divers_collected,
                    new_spawn_state,
                    state_updated.step_counter,
                    new_rng_key,
                )
            )

            new_surface_sub_pos = self.surface_sub_step(state_updated)

            state_updated.replace(surface_sub_position=new_surface_sub_pos)

            # update the enemy missile positions
            new_enemy_missile_positions = self.enemy_missiles_step(
                new_sub_positions,
                state_updated.enemy_missile_positions,
                state_updated.step_counter,
                state_updated.spawn_state.difficulty,
            )

            # append the surface submarine to the other submarines for the collision check
            # check if the player has collided with any of the enemies
            player_collision, collision_points = self.check_player_collision(
                player_x,
                player_y,
                new_sub_positions,
                new_shark_positions,
                new_surface_sub_pos,
                new_enemy_missile_positions,
                new_score,
                state_updated.successful_rescues,
            )

            lose_life = jnp.any(
                jnp.array([oxygen_depleted, player_collision, lose_life_surfacing])
            )

            # Start death animation; dismiss world divers (ALE clears them on
            # death) so they cannot keep swimming as phantoms through the anim.
            # Preserve diver_array collected bits — do not pass positions into
            # soft_reset (live→0 would all-zero then re-arm already-bagged lanes).
            death_animation_state = state_updated.replace(
                score=state.score + collision_points,
                death_counter=jnp.array(90),
                diver_positions=jnp.zeros_like(state_updated.diver_positions),
                spawn_state=self.soft_reset_spawn_state(
                    state_updated.spawn_state,
                    rng=jax.random.fold_in(
                        state_updated.rng_key, state_updated.step_counter
                    ),
                ),
            )

            # Create the scoring state. Dismiss any uncollected world divers so
            # their stale arm bit cannot place a diver on the wrong lane after
            # the freeze (ALE keeps collected lanes at 0 across the surface).
            scoring_state = state_updated.replace(
                player_x=player_x,
                player_y=player_y,
                player_direction=player_direction,
                lives=state_updated.lives,
                score=state_updated.score,
                successful_rescues=state_updated.successful_rescues + 1,
                diver_positions=jnp.zeros_like(state_updated.diver_positions),
                spawn_state=self.soft_reset_spawn_state(
                    state_updated.spawn_state,
                    state_updated.diver_positions,
                    rng=jax.random.fold_in(
                        state_updated.rng_key, state_updated.step_counter + 1
                    ),
                ).replace(
                    difficulty=state_updated.spawn_state.difficulty + 1,
                ),
                death_counter=jnp.array(-(96 + state_updated.oxygen * 2)),
            )

            # cap the step counter to 1024
            new_step_counter = jnp.where(
                state_updated.step_counter == 1024,
                jnp.array(0),
                state_updated.step_counter + 1,
            )

            # Create the normal returned state
            normal_returned_state = SeaquestState(
                player_x=player_x,
                player_y=player_y,
                player_direction=player_direction,
                oxygen=new_oxygen,
                divers_collected=new_divers_collected,
                score=new_score,
                lives=state_updated.lives,
                spawn_state=new_spawn_state.replace(
                    survived=new_spawn_state.survived.astype(jnp.int32)
                ),
                diver_positions=new_diver_positions,
                shark_positions=new_shark_positions,
                sub_positions=new_sub_positions,
                enemy_missile_positions=new_enemy_missile_positions,
                surface_sub_position=new_surface_sub_pos,
                player_missile_position=player_missile_position,
                step_counter=new_step_counter,
                just_surfaced=new_just_surfaced,
                successful_rescues=state_updated.successful_rescues,
                death_counter=jnp.array(0),
                rng_key=new_rng_key,
            )

            # First handle surfacing with all divers (scoring)
            intermediate_state = jax.lax.cond(
                should_reset,
                lambda _: scoring_state,
                lambda _: normal_returned_state,
                operand=None,
            )

            # Then handle life loss - start death animation instead of immediate reset
            final_state = jax.lax.cond(
                lose_life,
                lambda _: death_animation_state,
                lambda _: intermediate_state,
                operand=None,
            )

            # Check for additional life every 10,000 points
            additional_lives = (final_state.score // 10000) - (state.score // 10000)
            new_lives = jnp.minimum(final_state.lives + additional_lives, 6) # max 6 lives possible

            # Update the final state with new lives
            final_state = final_state.replace(lives=new_lives)

            # Check if the game is over
            game_over = final_state.lives <= -1

            # Handle game over state
            return jax.lax.cond(
                game_over,
                lambda _: state.replace(
                    score=final_state.score,
                    lives=jnp.array(-1),
                    death_counter=jnp.array(0),
                ),
                lambda _: final_state,
                operand=None,
            )

        return_state = jax.lax.cond(
            state.death_counter > 0,
            lambda _: handle_death_animation(),
            lambda _: jax.lax.cond(
                state.death_counter < 0,
                lambda _: handle_score_freeze(),
                lambda _: normal_game_step(),
                operand=None,
            ),
            operand=None,
        )

        # Get observation and info
        observation = self._get_observation(return_state)

        done = self._get_done(return_state)
        env_reward = self._get_reward(previous_state, return_state)
        info = self._get_info(return_state)

        # Choose between death animation and normal game step
        return observation, return_state, env_reward, done, info


class SeaquestRenderer(JAXGameRenderer):
    def __init__(self, consts: SeaquestConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or SeaquestConstants()
        super().__init__(self.consts)

        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 2. Create procedural assets using modded constants
        procedural_sprites = self._create_procedural_sprites()

        # 3. Append procedural assets
        for name, data in procedural_sprites.items():
            final_asset_config.append({'name': name, 'type': 'procedural', 'data': data})

        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "seaquest")

        # 4. Load all assets, create palette, and generate ID masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        # Load packed surface-wave background frames (bg/1.npy + bg/2.npy).
        # Done once at init — render only indexes.
        # Live ALE from reset: hold frame 0 for 6 frames, then advance every 8
        # (t=0..5 → 0, t=6..13 → 1, t=14..21 → 2, ...).
        self.SURFACE_WAVE_FIRST_HOLD = 6
        self.SURFACE_WAVE_HOLD = 8
        # Horizon + surface strip re-blitted after the player for ALE-style occlusion.
        # ALE at player_y=46: yellow shows on rows 46-52; rows 53-56 fully cover the
        # hull (two dark-blue wave lines, black waterline, one deep-blue row).
        self.SURFACE_OCCLUSION_Y0 = 53
        self.SURFACE_OCCLUSION_Y1 = 57
        self.BACKGROUND_FRAMES = self._load_surface_wave_backgrounds(sprite_path)
        # Keep BACKGROUND / BACKGROUND_ALT aliases for mods that still reference them.
        self.BACKGROUND = self.BACKGROUND_FRAMES[0]
        alt_i = 1 if self.BACKGROUND_FRAMES.shape[0] > 1 else 0
        self.BACKGROUND_ALT = self.BACKGROUND_FRAMES[alt_i]

        # Pre-compute oxygen bar color ID (convert JAX array to numpy, then tuple for dict lookup)
        oxygen_color_rgb = np.asarray(self.consts.OXYGEN_BAR_COLOR[:3])
        self.OXYGEN_COLOR_ID = self.COLOR_TO_ID.get(tuple(oxygen_color_rgb), 0)
        oxygen_bar_bg_color_rgb = np.asarray(self.consts.OXYGEN_BAR_BG_COLOR[:3])
        self.OXYGEN_BAR_BG_COLOR_ID = self.COLOR_TO_ID.get(tuple(oxygen_bar_bg_color_rgb), 0)
        # Flash-off fill is black (ALE leaves the drained segment black, not red).
        self.OXYGEN_FLASH_OFF_COLOR_ID = self.COLOR_TO_ID.get((0, 0, 0), 0)

        self.SHARK_COLOR_MAP = self._precompute_shark_color_map()

    def _load_surface_wave_backgrounds(self, sprite_path: str) -> jnp.ndarray:
        """Stack packed background color-ID rasters for surface-wave flicker.

        Uses the base background from asset loading (``bg/1.npy``) plus an
        optional second packed frame at ``bg/2.npy``.
        """
        frames = [np.asarray(self.BACKGROUND)]
        bg2_path = os.path.join(sprite_path, "bg", "2.npy")
        if os.path.exists(bg2_path):
            bg2_rgba = self.jr.loadFrame(bg2_path)
            frames.append(
                np.asarray(self.jr._create_background_raster(bg2_rgba, self.COLOR_TO_ID))
            )
        return jnp.asarray(np.stack(frames, axis=0))

    def _create_procedural_sprites(self) -> dict:
        """Creates 1x1 pixel sprites to ensure colors are in the palette."""
        procedural_sprites = {}
        for i, color in enumerate(self.consts.SHARK_DIFFICULTY_COLORS):
            rgba = jnp.array(list(color) + [255], dtype=jnp.uint8).reshape(1, 1, 4)
            procedural_sprites[f'shark_color_{i}'] = rgba

        rgba_oxy = jnp.array(list(self.consts.OXYGEN_BAR_COLOR[:3]) + [255], dtype=jnp.uint8).reshape(1, 1, 4)
        procedural_sprites['oxygen_bar_color'] = rgba_oxy
        rgba_oxy_bg = jnp.array(list(self.consts.OXYGEN_BAR_BG_COLOR[:3]) + [255], dtype=jnp.uint8).reshape(1, 1, 4)
        procedural_sprites['oxygen_bar_bg_color'] = rgba_oxy_bg
        procedural_sprites['oxygen_flash_off_color'] = jnp.array(
            [0, 0, 0, 255], dtype=jnp.uint8
        ).reshape(1, 1, 4)
        return procedural_sprites

    def _precompute_shark_color_map(self) -> jnp.ndarray:
        """Creates a lookup table mapping difficulty (0-7) to a shark color ID."""
        color_cycle_indices = jnp.array([0, 1, 2, 3, 0, 1, 0, 3])
        cycle_rgb_colors = self.consts.SHARK_DIFFICULTY_COLORS[color_cycle_indices]
        return jnp.array([self.COLOR_TO_ID[tuple(rgb)] for rgb in np.array(cycle_rgb_colors)])

    # --- Sequential Rendering for Batched Objects ---
    def render_object_sequentially(self, current_raster, pos, shape_masks, flip_offsets, anim_idx):
        """Helper to render a single object with a pre-calculated animation index."""
        is_active = pos[2] != 0

        return jax.lax.cond(
            is_active,
            lambda r: self.jr.render_at_clipped(r, pos[0], pos[1], shape_masks[anim_idx],
                                                flip_horizontal=pos[2] == self.consts.FACE_LEFT,
                                                flip_offset=flip_offsets),
            lambda r: r,
            current_raster
        )

    # --- Render Divers ---
    # Original cycle: Frame 0 for 16 steps, Frame 1 for 4 steps. Total = 20 steps.
    def _draw_divers(self, raster, state):
        diver_anim_idx = jax.lax.select((state.step_counter % 20) < 16, 0, 1)
        raster = jax.lax.fori_loop(
            0, state.diver_positions.shape[0],
            lambda i, r: self.render_object_sequentially(r, state.diver_positions[i], self.SHAPE_MASKS['diver'], self.FLIP_OFFSETS['diver'], diver_anim_idx),
            raster
        )

        # --- Render Enemy Subs ---
        # Original cycle: 3 frames, each for 4 steps. Total = 12 steps.
        enemy_sub_anim_idx = (state.step_counter % 12) // 4
        all_subs = jnp.concatenate([state.sub_positions, state.surface_sub_position[None, :]])
        raster = jax.lax.fori_loop(
            0, all_subs.shape[0],
            lambda i, r: self.render_object_sequentially(r, all_subs[i], self.SHAPE_MASKS['enemy_sub'], self.FLIP_OFFSETS['enemy_sub'], enemy_sub_anim_idx),
            raster
        )
        return raster

    @partial(jax.jit, static_argnames=['self'])
    def render(self, state: SeaquestState) -> jnp.ndarray:
        # Cycle pre-baked surface-wave backgrounds.
        # Live ALE from reset: first band holds SURFACE_WAVE_FIRST_HOLD frames,
        # then advances every SURFACE_WAVE_HOLD (indices at t=0,6,14,22,...).
        n_bg = self.BACKGROUND_FRAMES.shape[0]
        hold = jnp.int32(self.SURFACE_WAVE_HOLD)
        first_hold = jnp.int32(self.SURFACE_WAVE_FIRST_HOLD)
        t = state.step_counter.astype(jnp.int32)
        bg_idx = jnp.where(
            t < first_hold,
            jnp.int32(0),
            jnp.int32(1) + (t - first_hold) // hold,
        ) % n_bg
        raster = self.BACKGROUND_FRAMES[bg_idx]

        # Use the raw step_counter for precise animation control
        step_counter = state.step_counter

        # --- Player & Player Torpedo ---
        # 3 frames × 4 steps. Live ALE from reset changes at t=2,6,10,14,...
        # i.e. ((step+2)%12)//4 (first period is only 2 frames on frame 0).
        player_anim_idx = ((step_counter + 2) % 12) // 4
        raster = self.jr.render_at(
            raster, state.player_x, state.player_y,
            self.SHAPE_MASKS['player_sub'][player_anim_idx],
            flip_horizontal=state.player_direction == self.consts.FACE_LEFT,
            flip_offset=self.FLIP_OFFSETS['player_sub']
        )

        # Re-blit the 4 waterline occlusion rows (53-56) so the player hull is
        # partially covered while the conning tower (46-52) stays visible.
        # Colors come from the active packed bg frame (wave flicker).
        y0 = self.SURFACE_OCCLUSION_Y0
        y1 = self.SURFACE_OCCLUSION_Y1
        surface_strip = self.BACKGROUND_FRAMES[bg_idx, y0:y1, :]
        raster = raster.at[y0:y1, :].set(surface_strip)

        torp = state.player_missile_position
        raster = jax.lax.cond(
            torp[2] != 0,
            lambda r: self.jr.render_at_clipped(r, torp[0], torp[1], self.SHAPE_MASKS['player_torp'],
                                        flip_horizontal=torp[2] == self.consts.FACE_LEFT),
            lambda r: r,
            raster
        )

        raster = self._draw_divers(raster, state)

        # Surface sub is drawn with enemy subs above; re-apply occlusion so it
        # clips under the waterline the same way as the player.
        raster = raster.at[y0:y1, :].set(surface_strip)

        # --- Render Enemy Torpedoes ---
        # No animation, so index is always 0
        raster = jax.lax.fori_loop(
            0, state.enemy_missile_positions.shape[0],
            lambda i, r: self.render_object_sequentially(r, state.enemy_missile_positions[i], self.SHAPE_MASKS['enemy_torp'][None, ...], jnp.zeros(2, dtype=jnp.int32), 0),
            raster
        )

        # --- Render Sharks ---
        # Original cycle: Frame 0 for 16 steps, Frame 1 for 8 steps. Total = 24 steps.
        shark_anim_idx = jax.lax.select((step_counter % 24) < 16, 0, 1)
        difficulty_idx = state.spawn_state.difficulty % 8
        shark_color_id = self.SHARK_COLOR_MAP[difficulty_idx]

        base_shark_masks = self.SHAPE_MASKS['shark_base']
        recolored_shark_masks = jnp.where(base_shark_masks != self.jr.TRANSPARENT_ID, shark_color_id, base_shark_masks)
        raster = jax.lax.fori_loop(
            0, state.shark_positions.shape[0],
            lambda i, r: self.render_object_sequentially(r, state.shark_positions[i], recolored_shark_masks, self.FLIP_OFFSETS['shark_base'], shark_anim_idx),
            raster
        )

        # --- UI Elements (Unchanged) ---
        max_score_digits = 6
        score_digits = self.jr.int_to_digits(state.score, max_digits=max_score_digits)
        clamped_score = jnp.minimum(jnp.maximum(state.score, 0), 10**max_score_digits - 1)
        score_digit_thresholds = jnp.array([1, 10, 100, 1000, 10000, 100000], dtype=clamped_score.dtype)
        num_score_digits = jnp.maximum(1, jnp.sum(clamped_score >= score_digit_thresholds))
        score_start_index = max_score_digits - num_score_digits
        score_x = 59 + score_start_index * 8
        raster = self.jr.render_label_selective(
            raster,
            score_x,
            9,
            score_digits,
            self.SHAPE_MASKS['digits'],
            score_start_index,
            num_score_digits,
            spacing=8,
            max_digits_to_render=max_score_digits,
        )

        raster = self.jr.render_indicator(raster, 58, 22, state.lives, self.SHAPE_MASKS['life_indicator'], spacing=8, max_value=3)

        # Collected divers blink when there are 6 of them
        visible_divers = jax.lax.select(
            jnp.logical_and(state.divers_collected == 6, (state.step_counter % 16) < 8),
            0,
            state.divers_collected
        )
        # ALE OC CollectedDiver at x=58,66,74,... (pitch 8); y=178.
        raster = self.jr.render_indicator(
            raster, 58, 178, visible_divers, self.SHAPE_MASKS['diver_indicator'],
            spacing=8, max_value=6,
        )

        # Low-oxygen warning: white fill blinks to black (ALE ≤16 O₂, 16f off then
        # 16f on). Suppress while refilling (surface / opening fill) — ALE does
        # not flash during oxygen refill.
        flash_period = self.consts.OXYGEN_FLASH_PERIOD.astype(jnp.int32)
        at_surface = state.player_y <= jnp.int32(46)
        in_init_fill = state.just_surfaced == -1
        refilling = jnp.logical_or(
            jnp.logical_and(at_surface, state.oxygen < 64),
            jnp.logical_and(in_init_fill, state.oxygen < 64),
        )
        # Black for the first half of the 32-frame period (aligned with drain ticks).
        flash_off = jnp.logical_and(
            jnp.logical_and(
                state.oxygen <= self.consts.OXYGEN_FLASH_THRESHOLD,
                jnp.logical_not(refilling),
            ),
            (state.step_counter % flash_period) < (flash_period // 2),
        )
        oxy_fill_id = jax.lax.select(
            flash_off, self.OXYGEN_FLASH_OFF_COLOR_ID, self.OXYGEN_COLOR_ID
        )
        raster = self.jr.render_bar(
            raster, 49, 170, state.oxygen, 64, 63, 5, oxy_fill_id, self.OXYGEN_BAR_BG_COLOR_ID
        )

        raster = self.jr.draw_rects(
            raster,
            positions=jnp.array([[0, 0]]),
            sizes=jnp.array([[8, self.config.game_dimensions[0]]]),
            color_id=self.BACKGROUND[0, 0]
        )

        return self.jr.render_from_palette(raster, self.PALETTE)
