import os
from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


# --- Level Configuration ---
class RoadSectionConfig(NamedTuple):
    scroll_start: int
    scroll_end: int
    road_width: int
    road_top: int
    road_height: int
    road_pattern_style: int = 0


class OfframpConfig(NamedTuple):
    """Configuration for an offramp section in a level."""
    enabled: bool = False
    scroll_start: int = 0  # Scroll step when the split first appears on the left screen edge
    scroll_end: int = 0    # Scroll step when the merge first appears on the left screen edge;
                           # the offramp remains active until the merge sprite exits the screen
    bridges: Tuple[int, ...] = ()  # Scroll steps at which vertical bridges appear across the median


class LevelConfig(NamedTuple):
    level_number: int
    scroll_distance_to_complete: int
    road_sections: Tuple[RoadSectionConfig, ...]
    spawn_seeds: bool
    spawn_trucks: bool
    spawn_ravines: bool = False
    decorations: Tuple[Tuple[int, int, int, int], ...] = ()
    seed_spawn_config: Optional[Tuple[int, int]] = None
    truck_spawn_config: Optional[Tuple[int, int]] = None
    ravine_spawn_config: Optional[Tuple[int, int]] = None
    spawn_landmines: bool = False
    landmine_spawn_config: Optional[Tuple[int, int]] = None
    spawn_cannons: bool = False
    cannon_spawn_config: Optional[Tuple[int, int]] = None
    # Fixed scroll-step positions at which cannons appear (deterministic placement).
    # When non-empty, these override the random cannon_spawn_config interval.
    # Cannons alternate normal/mirrored starting with normal (right-facing).
    cannon_scroll_steps: Tuple[int, ...] = ()
    future_entity_types: Dict[str, Any] = {}
    render_road_stripes: bool = True
    # Dynamic road height configuration (for alternating heights)
    dynamic_road_heights: Optional[Tuple[int, int]] = None  # (height_a, height_b)
    dynamic_road_interval: int = 400  # scroll distance between height switches
    dynamic_road_transition_length: int = 10  # transition zone length in scroll units
    # Phase offset added to world_x before the modulo so the level starts inside zone A
    dynamic_road_scroll_offset: int = 0
    offramps: Tuple[OfframpConfig, ...] = ()
    # Ravine-linked entity spawning: when True, seeds/mines are scheduled to appear
    # just ahead of each ravine instead of on their own random timer.
    ravine_linked_seed: bool = False
    ravine_linked_mine: bool = False


# --- Constants ---
class RoadRunnerConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    PLAYER_MOVE_SPEED: int = 3
    PLAYER_ANIMATION_SPEED: int = 2
    # If the players x coordinate would be below this value after applying movement, we move everything one to the right to simulate movement.
    X_SCROLL_THRESHOLD: int = 70
    ENEMY_MOVE_SPEED: int = 2
    ENEMY_REACTION_DELAY: int = 6
    PLAYER_START_X: int = 70
    PLAYER_START_Y: int = 120
    ENEMY_X: int = 140
    ENEMY_Y: int = 120
    PLAYER_SIZE: Tuple[int, int] = (8, 32)
    ENEMY_SIZE: Tuple[int, int] = (4, 4)
    SEED_SIZE: Tuple[int, int] = (5, 5)
    PLAYER_PICKUP_OFFSET: int = PLAYER_SIZE[1] * 3 // 4  # Bottom 25% of player height
    PLAYER_ROAD_TOP_OFFSET: int = 10
    ROAD_HEIGHT: int = 70
    ROAD_TOP_Y: int = 110
    ROAD_DASH_LENGTH: int = 5
    ROAD_GAP_HEIGHT: int = 14
    ROAD_PATTERN_WIDTH: int = ROAD_DASH_LENGTH * 4
    SPAWN_Y_RANDOM_OFFSET_MIN: int = -20
    SPAWN_Y_RANDOM_OFFSET_MAX: int = 20
    PLAYER_COLOR: Tuple[int, int, int] = (92, 186, 92)
    ENEMY_COLOR: Tuple[int, int, int] = (213, 130, 74)
    SEED_SPAWN_MIN_INTERVAL: int = 5
    SEED_SPAWN_MAX_INTERVAL: int = 20
    MAX_STREAK: int = 10
    SEED_BASE_VALUE: int = 100
    TRUCK_SIZE: Tuple[int, int] = (15, 15)
    TRUCK_COLLISION_OFFSET: int = TRUCK_SIZE[1] // 2  # Bottom half of truck height
    TRUCK_COLOR: Tuple[int, int, int] = (255, 0, 0)
    TRUCK_SPEED: int = 2
    TRUCK_SPAWN_MIN_INTERVAL: int = 120
    TRUCK_SPAWN_MAX_INTERVAL: int = 240
    LEVEL_TRANSITION_DURATION: int = 30
    LEVEL_COMPLETE_SCROLL_DISTANCE: int = 3000
    STARTING_LIVES: int = 3
    JUMP_TIME_DURATION: int = 20  # Jump duration in steps (~0.33 seconds at 60 FPS)
    FALL_ANIMATION_DURATION: int = 10  # Fall animation duration in frames
    SIDE_MARGIN: int = 8
    RAVINE_SIZE: Tuple[int, int] = (13, 32)
    RAVINE_SPAWN_MIN_INTERVAL: int = 30
    RAVINE_SPAWN_MAX_INTERVAL: int = 60
    LANDMINE_SIZE: Tuple[int, int] = (4, 4)
    LANDMINE_SPAWN_MIN_INTERVAL: int = 120
    LANDMINE_SPAWN_MAX_INTERVAL: int = 240
    CANNON_SIZE: Tuple[int, int] = (5, 12)
    BULLET_SIZE: Tuple[int, int] = (2, 2)
    BULLET_SPEED: int = 2
    CANNON_SPAWN_MIN_INTERVAL: int = 120
    CANNON_SPAWN_MAX_INTERVAL: int = 240
    DEATH_ANIMATION_DURATION: int = 60  # 1 second at 60 FPS
    # Enemy speed variation - speeds as offsets from PLAYER_MOVE_SPEED
    ENEMY_SLOW_SPEED_OFFSET: int = -1        # Speed = PLAYER_MOVE_SPEED - 1 = 2
    ENEMY_FAST_SPEED_OFFSET: int = 1         # Speed = PLAYER_MOVE_SPEED + 1 = 4
    ENEMY_SAME_SPEED_OFFSET: int = 0         # Speed = PLAYER_MOVE_SPEED = 3
    # Enemy speed cycle durations (in scroll distance units)
    ENEMY_SLOW_DURATION: int = 60            # Default slow phase duration
    ENEMY_FAST_DURATION: int = 60            # ~1 second at 60 FPS  
    ENEMY_SAME_DURATION: int = 300           # ~5 seconds at 60 FPS
    # Enemy approach slowdown/reversal multiplier (when player moves right)
    # Positive values slow down (0.5 = half speed), negative values reverse direction (-0.5 = move away at half speed)
    ENEMY_APPROACH_SLOWDOWN: float = -0.5     # Moves backwards at half speed when player approaches
    # Enemy Flattened (Run Over) State
    ENEMY_FLATTENED_DURATION: int = 120  # 2 seconds at 60 FPS
    ENEMY_FLATTENED_SCORE: int = 1000
    # --- Offramp Constants ---
    OFFRAMP_HEIGHT: int = 12   # Height of offramp road in pixels (narrow "one lane")
    OFFRAMP_GAP: int = 8       # Gap (median) between offramp bottom and main road top
    OFFRAMP_RAMP_WIDTH: int = 24  # Width of the diagonal split/merge transition in pixels
    OFFRAMP_BRIDGE_WIDTH: int = 16  # Width of a bridge segment crossing the median
    # --- Decoration Type Constants ---
    DECO_CACTUS = 0
    DECO_SIGN_THIS_WAY = 1
    DECO_SIGN_BIRD_SEED = 2
    DECO_SIGN_CARS_AHEAD = 3
    DECO_SIGN_EXIT = 4
    DECO_TUMBLEWEED = 5
    DECO_SIGN_ACME_MINES = 6
    DECO_SIGN_STEEL_SHOT = 7  # placeholder sprite; replace with final asset when available
    # --- Ravine-linked entity spawn constants ---
    # How many scroll steps *before* the next ravine spawns that seeds/mines should appear.
    # At PLAYER_MOVE_SPEED=3 px/step: 15 steps → 45 px ahead, 12 steps → 36 px ahead.
    RAVINE_SEED_AHEAD_SCROLL_STEPS: int = 8
    RAVINE_MINE_AHEAD_SCROLL_STEPS: int = 8
    # Probability thresholds for picking which entity (if any) spawns with each ravine.
    # A single uniform draw r is used so seeds and mines can't both appear for the same ravine.
    # r < SEED_PROB → seed linked; SEED_PROB ≤ r < SEED_PROB+MINE_PROB → mine linked; else nothing.
    RAVINE_SEED_LINK_PROB: float = 0.52   # ≈ 12 seeds / 23 ravines for Level 2
    RAVINE_MINE_LINK_PROB: float = 0.17   # ≈  4 mines / 23 ravines for Level 2
    levels: Tuple[LevelConfig, ...] = ()


_BASE_CONSTS = RoadRunnerConstants()
_DEFAULT_ROAD_HEIGHT = _BASE_CONSTS.ROAD_HEIGHT


def _centered_top(height: int) -> int:
    return max((_DEFAULT_ROAD_HEIGHT - height) // 2, 0)


RoadRunner_Level_1 = LevelConfig(
    level_number=1,
    scroll_distance_to_complete=1500,
    road_sections=(
        RoadSectionConfig(
            scroll_start=0,
            scroll_end=1500,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=0,
            road_height=_DEFAULT_ROAD_HEIGHT,
            road_pattern_style=0,
        ),
    ),
    spawn_seeds=True,
    spawn_trucks=True,
    seed_spawn_config=(
        15,
        45,
    ),
    truck_spawn_config=(
        240,
        360,
    ),
    decorations=(
        # --- INTRO (0-6s) ---
        (50, 60, 1, _BASE_CONSTS.DECO_SIGN_THIS_WAY),
        (0, 45, 2, _BASE_CONSTS.DECO_CACTUS),
        (30, 55, 1, _BASE_CONSTS.DECO_CACTUS),
        (180, 70, 1, _BASE_CONSTS.DECO_SIGN_BIRD_SEED),
        (350, 60, 1, _BASE_CONSTS.DECO_SIGN_CARS_AHEAD),

        # --- THE DESERT RUN (13 Cacti) ---
        (420, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (500, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (580, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (660, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (740, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (820, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (900, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (980, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (1060, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (1140, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (1220, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (1300, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (1380, 45, 3, _BASE_CONSTS.DECO_CACTUS),

        # --- OUTRO ---
        (2000, 60, 1, _BASE_CONSTS.DECO_SIGN_EXIT),
    ),
)

RoadRunner_Level_2 = LevelConfig(
    level_number=2,
    scroll_distance_to_complete=1500,
    road_sections=(
        RoadSectionConfig(
            scroll_start=0,
            scroll_end=1500,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=_centered_top(32),
            road_height=32,
        ),
    ),
    spawn_seeds=True,
    spawn_trucks=False,
    spawn_ravines=True,
    # Large fallback interval: seeds only appear via ravine-linked scheduling
    seed_spawn_config=(10000, 10001),
    # avg 65 scroll steps → ~23 ravines per level (first 15 evenly spaced, last 8 denser)
    ravine_spawn_config=(50, 80),
    spawn_landmines=True,
    # Large fallback interval: mines only appear via ravine-linked scheduling
    landmine_spawn_config=(10000, 10001),
    render_road_stripes=False,
    # Seeds and mines are scheduled just ahead of each ravine
    ravine_linked_seed=True,
    ravine_linked_mine=True,
    decorations=(
        # --- Pairs of cactus + tumbleweed throughout the level ---
        # Each pair: cactus on one side of the road, tumbleweed on the other.
        # Decoration format: (d_x, y, d_slowdown, type)
        # Screen appearance at scroll step T ≈ d_x * 2 * d_slowdown / 3
        # (slowdown=2: foreground layer, appears ~step d_x*4/3; slowdown=3: background, ~step d_x*2)
        (75, 45, 2, _BASE_CONSTS.DECO_CACTUS),        # pair 1: appears ~step 100
        (50, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),    # pair 1
        (225, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 2: appears ~step 300
        (150, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 2
        (375, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 3: appears ~step 500
        (250, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 3
        (525, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 4: appears ~step 700
        (350, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 4
        (675, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 5: appears ~step 900
        (450, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 5
        (825, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 6: appears ~step 1100
        (550, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 6
        (975, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 7: appears ~step 1300
        (650, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 7

        # --- Exit sign (same position as Level 1) ---
        (2000, 60, 1, _BASE_CONSTS.DECO_SIGN_EXIT),
    ),
)

RoadRunner_Level_3 = LevelConfig(
    level_number=3,
    scroll_distance_to_complete=1500,
    # Single full-level road section; narrowing/widening is handled by dynamic_road_heights
    # which produces smooth per-column transitions instead of abrupt jumps.
    road_sections=(
        RoadSectionConfig(
            scroll_start=0,
            scroll_end=1500,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=_centered_top(_DEFAULT_ROAD_HEIGHT),
            road_height=_DEFAULT_ROAD_HEIGHT,
        ),
    ),
    # Road alternates between wide (70 px) and narrower (50 px) with a sharp diagonal
    # transition edge spanning 10 world pixels, matching the visual style of the original game.
    dynamic_road_heights=(_DEFAULT_ROAD_HEIGHT, 50),
    dynamic_road_interval=400,
    dynamic_road_transition_length=10,
    # Shift phase by half_trans so world_x=0 lands at the start of zone A (not in a
    # wrap-around B→A transition), ensuring the road starts at full width.
    dynamic_road_scroll_offset=5,
    spawn_seeds=True,
    spawn_trucks=True,
    spawn_landmines=True,
    # Seeds at similar frequency to Level 1
    seed_spawn_config=(15, 45),
    # ~5 trucks over the level; first appear around the first narrow section
    truck_spawn_config=(200, 350),
    # ~6-7 mines over the level, matching video density
    landmine_spawn_config=(150, 300),
    render_road_stripes=True,
    decorations=(
        # --- Intro sign ---
        (50, 55, 1, _BASE_CONSTS.DECO_SIGN_STEEL_SHOT),     # "STEEL SHOT" placeholder at level start
        # --- Lots of cacti (more than Level 1) ---
        # Decoration formula: appears at scroll step T = d_x * 2 * d_slowdown / PLAYER_MOVE_SPEED
        # (PLAYER_MOVE_SPEED=3, so T = d_x * 2 * d_slowdown / 3)
        (75,  45, 2, _BASE_CONSTS.DECO_CACTUS),             # T≈100
        (88,  55, 3, _BASE_CONSTS.DECO_CACTUS),             # T≈176
        (188, 45, 2, _BASE_CONSTS.DECO_CACTUS),             # T≈251
        (163, 60, 3, _BASE_CONSTS.DECO_CACTUS),             # T≈326
        (300, 45, 2, _BASE_CONSTS.DECO_CACTUS),             # T≈400
        (238, 55, 3, _BASE_CONSTS.DECO_CACTUS),             # T≈476
        # --- ACME MINES sign (~02:25, T≈450) ---
        (675, 60, 1, _BASE_CONSTS.DECO_SIGN_ACME_MINES),    # T≈450
        (413, 45, 2, _BASE_CONSTS.DECO_CACTUS),             # T≈551
        (313, 55, 3, _BASE_CONSTS.DECO_CACTUS),             # T≈626
        (525, 45, 2, _BASE_CONSTS.DECO_CACTUS),             # T≈700
        (388, 60, 3, _BASE_CONSTS.DECO_CACTUS),             # T≈776
        (638, 45, 2, _BASE_CONSTS.DECO_CACTUS),             # T≈851
        (463, 55, 3, _BASE_CONSTS.DECO_CACTUS),             # T≈926
        (750, 45, 2, _BASE_CONSTS.DECO_CACTUS),             # T≈1000
        (538, 60, 3, _BASE_CONSTS.DECO_CACTUS),             # T≈1076
        (863, 45, 2, _BASE_CONSTS.DECO_CACTUS),             # T≈1151
        (625, 55, 3, _BASE_CONSTS.DECO_CACTUS),             # T≈1250
        (1013, 45, 2, _BASE_CONSTS.DECO_CACTUS),            # T≈1351
        (725, 60, 3, _BASE_CONSTS.DECO_CACTUS),             # T≈1450
        # --- Exit sign ---
        # d_x=2000 with d_slowdown=1 appears at scroll step T = 2000*2*1/3 ≈ 1333, well within 1500
        (2000, 60, 1, _BASE_CONSTS.DECO_SIGN_EXIT),         # T≈1333
    ),
)

RoadRunner_Level_4 = LevelConfig(
    level_number=4,
    scroll_distance_to_complete=1500,
    road_sections=(
        RoadSectionConfig(
            scroll_start=0,
            scroll_end=1500,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=_centered_top(30),
            road_height=30,
            road_pattern_style=0,
        ),
    ),
    spawn_seeds=True,
    spawn_trucks=False,
    spawn_landmines=True,
    spawn_cannons=True,
    # ~4 seeds over 1500 scroll steps → avg 375 steps between seeds
    seed_spawn_config=(300, 450),
    # ~5 mines over 1500 scroll steps → avg 300 steps between mines
    landmine_spawn_config=(220, 380),
    # ~7 cannons over 1500 scroll steps → avg 215 steps between cannons
    cannon_spawn_config=(150, 280),
    # Fixed cannon positions flanking each offramp.
    # Pattern per offramp: left-facing cannon → split → bridges → merge → right-facing cannon
    # Each position is ~10 scroll steps before scroll_start / after scroll_end.
    cannon_scroll_steps=(
        120,   # Left-facing cannon before offramp 1 split   (scroll_start=130)
        345,   # Right-facing cannon after offramp 1 merge   (scroll_end=335)
        535,   # Left-facing cannon before offramp 2 split   (scroll_start=545)
        760,   # Right-facing cannon after offramp 2 merge   (scroll_end=750)
        845,   # Left-facing cannon before offramp 3 split   (scroll_start=855)
        1070,  # Right-facing cannon after offramp 3 merge   (scroll_end=1060)
        1180,  # Left-facing cannon before offramp 4 split   (scroll_start=1190)
        1405,  # Right-facing cannon after offramp 4 merge   (scroll_end=1395)
    ),
    render_road_stripes=True,
    # Four offramps, each sandwiched between left-facing and right-facing cannons.
    # Pattern: left cannon → split → bridge → bridge → merge → right cannon
    # All offramps have the same internal structure (length ≈ 205 scroll steps).
    # Video timestamps (level starts at 03:09, rate ≈ 25.86 steps/s):
    #   Offramp 1: 03:14–03:22, Offramp 2: 03:30–03:38,
    #   Offramp 3: 03:42–03:51, Offramp 4: 03:55–04:04
    offramps=(
        OfframpConfig(enabled=True, scroll_start=130, scroll_end=335,
                      bridges=(180, 260)),
        OfframpConfig(enabled=True, scroll_start=545, scroll_end=750,
                      bridges=(595, 675)),
        OfframpConfig(enabled=True, scroll_start=855, scroll_end=1060,
                      bridges=(905, 985)),
        OfframpConfig(enabled=True, scroll_start=1190, scroll_end=1395,
                      bridges=(1240, 1320)),
    ),
    decorations=(
        # --- Pairs of cactus + tumbleweed throughout the level (like Level 2) ---
        # Each pair: cactus on one side of the road, tumbleweed on the other.
        # Decoration format: (d_x, y, d_slowdown, type)
        # Screen appearance at scroll step T ≈ d_x * 2 * d_slowdown / 3
        (75, 45, 2, _BASE_CONSTS.DECO_CACTUS),        # pair 1: appears ~step 100
        (50, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),    # pair 1
        (225, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 2: appears ~step 300
        (150, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 2
        (375, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 3: appears ~step 500
        (250, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 3
        (525, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 4: appears ~step 700
        (350, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 4
        (675, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 5: appears ~step 900
        (450, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 5
        (825, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 6: appears ~step 1100
        (550, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 6
        (975, 45, 2, _BASE_CONSTS.DECO_CACTUS),       # pair 7: appears ~step 1300
        (650, 55, 3, _BASE_CONSTS.DECO_TUMBLEWEED),   # pair 7

        # --- Exit sign (same position as Level 1) ---
        (2000, 60, 1, _BASE_CONSTS.DECO_SIGN_EXIT),
    ),
)

DEFAULT_LEVELS: Tuple[LevelConfig, ...] = (
    RoadRunner_Level_1,
    RoadRunner_Level_2,
    RoadRunner_Level_3,
    RoadRunner_Level_4,
)


# --- Helper Functions ---
def _build_road_section_arrays(
    levels: Tuple[LevelConfig, ...], consts: RoadRunnerConstants
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Build road section data arrays from level configs.
    Returns (max_road_sections, road_section_data, road_section_counts).
    """
    if not levels:
        return 0, jnp.array([], dtype=jnp.int32).reshape(0, 0, 6), jnp.array([], dtype=jnp.int32)

    max_road_sections = max(len(cfg.road_sections) for cfg in levels)


    road_sections_data = []
    road_section_counts = []
    default_section = [
        0,
        consts.LEVEL_COMPLETE_SCROLL_DISTANCE,
        consts.WIDTH,
        0,
        0,
        consts.ROAD_HEIGHT,
    ]
    for cfg in levels:
        # Determine pattern style based on render_road_stripes
        # 0 = Default (Stripes), 1 = No Stripes
        pattern_style_override = 0 if cfg.render_road_stripes else 1

        rows = [
            [
                section.scroll_start,
                section.scroll_end,
                section.road_width,
                pattern_style_override,
                section.road_top,
                section.road_height,
            ]
            for section in cfg.road_sections
        ]
        if not rows:
            rows = [default_section[:]]
        road_section_counts.append(len(rows))
        while len(rows) < max_road_sections:
            rows.append(rows[-1][:])
        road_sections_data.append(rows)

    return (
        max_road_sections,
        jnp.array(road_sections_data, dtype=jnp.int32),
        jnp.array(road_section_counts, dtype=jnp.int32),
    )


def _build_dynamic_road_config_arrays(
    levels: Tuple[LevelConfig, ...]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    enabled = jnp.array(
        [cfg.dynamic_road_heights is not None for cfg in levels],
        dtype=jnp.bool_
    ) if levels else jnp.array([], dtype=jnp.bool_)

    heights = jnp.array(
        [cfg.dynamic_road_heights if cfg.dynamic_road_heights else (70, 70) for cfg in levels],
        dtype=jnp.int32
    ) if levels else jnp.zeros((0, 2), dtype=jnp.int32)

    intervals = jnp.array(
        [cfg.dynamic_road_interval for cfg in levels],
        dtype=jnp.int32
    ) if levels else jnp.array([], dtype=jnp.int32)

    transition_lengths = jnp.array(
        [cfg.dynamic_road_transition_length for cfg in levels],
        dtype=jnp.int32
    ) if levels else jnp.array([], dtype=jnp.int32)

    scroll_offsets = jnp.array(
        [cfg.dynamic_road_scroll_offset for cfg in levels],
        dtype=jnp.int32
    ) if levels else jnp.array([], dtype=jnp.int32)

    return enabled, heights, intervals, transition_lengths, scroll_offsets


def _build_spawn_interval_array(
    levels: Tuple[LevelConfig, ...],
    config_attr: str,
    default_min: int,
    default_max: int,
) -> jnp.ndarray:
    """
    Build spawn interval array from level configs.

    Args:
        levels: Tuple of level configurations
        config_attr: Name of the config attribute (e.g., 'seed_spawn_config')
        default_min: Default minimum interval
        default_max: Default maximum interval

    Returns:
        Array of shape (num_levels, 2) with [min, max] intervals per level
    """
    if not levels:
        return jnp.array([], dtype=jnp.int32).reshape(0, 2)

    return jnp.array(
        [
            [
                (getattr(cfg, config_attr) or (default_min, default_max))[0],
                (getattr(cfg, config_attr) or (default_min, default_max))[1],
            ]
            for cfg in levels
        ],
        dtype=jnp.int32,
    )


def _build_spawn_enabled_array(
    levels: Tuple[LevelConfig, ...],
    spawn_attr: str,
) -> jnp.ndarray:
    """
    Build spawn enabled boolean array from level configs.

    Args:
        levels: Tuple of level configurations
        spawn_attr: Name of the spawn attribute (e.g., 'spawn_seeds')

    Returns:
        Boolean array indicating if spawning is enabled per level
    """
    if not levels:
        return jnp.array([], dtype=jnp.bool_)

    return jnp.array(
        [getattr(cfg, spawn_attr) for cfg in levels], dtype=jnp.bool_
    )


MAX_OFFRAMPS: int = 4  # Maximum number of offramp sections per level
MAX_OFFRAMP_BRIDGES: int = 8  # Maximum number of bridges per offramp section


def _build_offramp_arrays(
    levels: Tuple[LevelConfig, ...],
) -> jnp.ndarray:
    """
    Build offramp data arrays from level configs.

    Returns array of shape (num_levels, MAX_OFFRAMPS, 3 + MAX_OFFRAMP_BRIDGES) with columns:
      [enabled (0/1), scroll_start, scroll_end, bridge_0, bridge_1, ..., bridge_N]
    Bridge columns hold the scroll-step position of each bridge, or -1 if absent.
    Each level may contain up to MAX_OFFRAMPS offramp sections.
    """
    cols = 3 + MAX_OFFRAMP_BRIDGES
    if not levels:
        return jnp.zeros((0, MAX_OFFRAMPS, cols), dtype=jnp.int32)

    disabled_row = [0, 0, 0] + [-1] * MAX_OFFRAMP_BRIDGES
    level_rows = []
    for cfg in levels:
        offramps = getattr(cfg, 'offramps', ())
        offramp_rows = []
        for offramp in offramps[:MAX_OFFRAMPS]:
            bridges = list(offramp.bridges)[:MAX_OFFRAMP_BRIDGES]
            # Pad to fixed length with -1
            bridges += [-1] * (MAX_OFFRAMP_BRIDGES - len(bridges))
            offramp_rows.append([
                1 if offramp.enabled else 0,
                offramp.scroll_start,
                offramp.scroll_end,
            ] + bridges)
        # Pad with disabled offramp rows
        while len(offramp_rows) < MAX_OFFRAMPS:
            offramp_rows.append(list(disabled_row))
        level_rows.append(offramp_rows)
    return jnp.array(level_rows, dtype=jnp.int32)


MAX_FIXED_CANNONS: int = 16  # Maximum number of fixed cannon positions per level
_NO_CANNON_STEP: int = 999999  # Sentinel: no more fixed cannon positions


def _build_cannon_fixed_steps_array(
    levels: Tuple[LevelConfig, ...],
) -> jnp.ndarray:
    """Build fixed cannon scroll-step arrays from level configs.

    Returns array of shape (num_levels, MAX_FIXED_CANNONS).
    Each row holds the sorted scroll steps at which cannons should appear,
    padded with -1 for unused slots.
    """
    if not levels:
        return jnp.zeros((0, MAX_FIXED_CANNONS), dtype=jnp.int32)

    rows = []
    for cfg in levels:
        steps = list(getattr(cfg, 'cannon_scroll_steps', ()))[:MAX_FIXED_CANNONS]
        steps.sort()
        steps += [-1] * (MAX_FIXED_CANNONS - len(steps))
        rows.append(steps)
    return jnp.array(rows, dtype=jnp.int32)


def _find_active_offramp_row(
    offramp_data: jnp.ndarray,
    state,
    level_count: int,
    consts,
) -> tuple:
    """Find the currently active offramp for this level.

    Works as a standalone function so both JaxRoadRunner and RoadRunnerRenderer
    can share the same logic.

    Args:
        offramp_data: shape (num_levels, MAX_OFFRAMPS, 3 + MAX_OFFRAMP_BRIDGES)
        state: game state with scrolling_step_counter and current_level
        level_count: number of configured levels
        consts: RoadRunnerConstants (needs PLAYER_MOVE_SPEED, OFFRAMP_RAMP_WIDTH, WIDTH)

    Returns:
        (any_active, row) where:
          any_active: scalar bool — True when an offramp is on-screen
          row: shape (3 + MAX_OFFRAMP_BRIDGES,) — data for the active offramp
    """
    SPEED = consts.PLAYER_MOVE_SPEED
    RAMP_W = consts.OFFRAMP_RAMP_WIDTH
    W = consts.WIDTH
    cols = 3 + MAX_OFFRAMP_BRIDGES

    if level_count == 0:
        return jnp.array(False), jnp.zeros(cols, dtype=jnp.int32)

    level_idx = jnp.clip(state.current_level, 0, level_count - 1).astype(jnp.int32)
    all_rows = offramp_data[level_idx]  # (MAX_OFFRAMPS, cols)

    enabled = all_rows[:, 0] > 0
    scroll_starts = all_rows[:, 1]
    scroll_ends = all_rows[:, 2]

    counter = state.scrolling_step_counter
    merge_xs = (counter - scroll_ends) * SPEED
    active_mask = enabled & (counter >= scroll_starts) & (merge_xs < W + RAMP_W)

    any_active = jnp.any(active_mask)
    # argmax returns the index of the first True; if none, returns 0 (disabled row)
    active_idx = jnp.argmax(active_mask)
    row = all_rows[active_idx]
    return any_active, row


def _check_aabb_collision(
    x1: chex.Array, y1: chex.Array, w1: int, h1: int,
    x2: chex.Array, y2: chex.Array, w2: int, h2: int,
) -> chex.Array:
    """
    Check if two axis-aligned bounding boxes overlap.

    Args:
        x1, y1: Top-left corner of first box
        w1, h1: Width and height of first box
        x2, y2: Top-left corner of second box
        w2, h2: Width and height of second box

    Returns:
        Boolean indicating if boxes overlap
    """
    overlap_x = (x1 < x2 + w2) & (x1 + w1 > x2)
    overlap_y = (y1 < y2 + h2) & (y1 + h1 > y2)
    return overlap_x & overlap_y


def _update_orientation(
    vel_x: chex.Array,
    current_looks_right: chex.Array,
) -> chex.Array:
    """
    Update orientation based on horizontal velocity.

    Args:
        vel_x: Horizontal velocity
        current_looks_right: Current orientation

    Returns:
        Updated orientation (True = right, False = left)
    """
    return jnp.where(vel_x > 0, True, jnp.where(vel_x < 0, False, current_looks_right))


def _get_road_section_for_scroll(
    state: "RoadRunnerState",
    level_count: int,
    max_road_sections: int,
    road_section_data: jnp.ndarray,
    road_section_counts: jnp.ndarray,
    consts: RoadRunnerConstants,
) -> RoadSectionConfig:
    """
    Get the current road section based on scroll position.

    Args:
        state: Current game state
        level_count: Number of levels
        max_road_sections: Maximum number of road sections across all levels
        road_section_data: Array of road section data per level
        road_section_counts: Array of section counts per level
        consts: Game constants

    Returns:
        The current RoadSectionConfig
    """
    if level_count == 0 or max_road_sections == 0:
        return RoadSectionConfig(
            0,
            consts.LEVEL_COMPLETE_SCROLL_DISTANCE,
            consts.WIDTH,
            0,
            consts.ROAD_HEIGHT,
            0,
        )

    max_index = level_count - 1
    level_idx = jnp.clip(state.current_level, 0, max_index).astype(jnp.int32)
    sections = road_section_data[level_idx]
    section_count = road_section_counts[level_idx]
    indices = jnp.arange(max_road_sections, dtype=jnp.int32)
    valid_section = indices < section_count
    counter = state.scrolling_step_counter
    in_section = (counter >= sections[:, 0]) & (counter < sections[:, 1])
    active_sections = jnp.where(valid_section, in_section, False)
    no_match_value = jnp.array(max_road_sections, dtype=jnp.int32)
    candidate_indices = jnp.where(active_sections, indices, no_match_value)
    match_idx = jnp.min(candidate_indices)
    fallback_idx = jnp.maximum(section_count - 1, 0)
    section_idx = jnp.where(match_idx < max_road_sections, match_idx, fallback_idx)
    selected = sections[section_idx]
    return RoadSectionConfig(
        selected[0],
        selected[1],
        selected[2],
        selected[4],
        selected[5],
        selected[3],
    )


def _get_dynamic_road_height(
    scroll_pos: chex.Array,
    height_a: int,
    height_b: int,
    interval: int,
    transition_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Calculate road height at a given scroll position with transition zones.

    The road height alternates between height_a and height_b every `interval` scroll units.
    Transitions between heights are smooth over `transition_length` scroll units.

    Args:
        scroll_pos: Current scroll position
        height_a: First height value
        height_b: Second height value
        interval: Scroll distance between height switches
        transition_length: Length of transition zone in scroll units

    Returns:
        (current_height, is_in_transition, transition_progress)
        - current_height: The road height at this position (int32)
        - is_in_transition: Boolean, True if in a transition zone
        - transition_progress: 0.0-1.0 progress through transition (only valid if is_in_transition)
    """
    cycle_length = interval * 2  # Full A→B→A cycle
    pos_in_cycle = scroll_pos % cycle_length

    # Transition zones are centered at interval and cycle_length (0)
    # Zone A: [half_trans, interval - half_trans)
    # Transition A→B: [interval - half_trans, interval + half_trans)
    # Zone B: [interval + half_trans, cycle_length - half_trans)
    # Transition B→A: [cycle_length - half_trans, cycle_length) AND [0, half_trans)

    half_trans = transition_length // 2

    trans_a_to_b_start = interval - half_trans
    trans_a_to_b_end = interval + half_trans
    trans_b_to_a_start = cycle_length - half_trans

    # Zone A: past B→A transition end, before A→B transition start
    in_zone_a = (pos_in_cycle >= half_trans) & (pos_in_cycle < trans_a_to_b_start)
    in_trans_a_to_b = (pos_in_cycle >= trans_a_to_b_start) & (pos_in_cycle < trans_a_to_b_end)
    in_zone_b = (pos_in_cycle >= trans_a_to_b_end) & (pos_in_cycle < trans_b_to_a_start)
    # B→A transition wraps around: [trans_b_to_a_start, cycle_length) OR [0, half_trans)
    in_trans_b_to_a = (pos_in_cycle >= trans_b_to_a_start) | (pos_in_cycle < half_trans)

    # Calculate transition progress for interpolation
    trans_a_to_b_progress = (pos_in_cycle - trans_a_to_b_start) / transition_length

    # B→A progress needs special handling for wrap-around
    # First half of B→A: [trans_b_to_a_start, cycle_length) -> progress 0 to 0.5
    # Second half of B→A: [0, half_trans) -> progress 0.5 to 1.0
    trans_b_to_a_progress = jnp.where(
        pos_in_cycle >= trans_b_to_a_start,
        (pos_in_cycle - trans_b_to_a_start) / transition_length,
        (pos_in_cycle + half_trans) / transition_length
    )

    # Calculate heights with interpolation in transition zones
    height = jnp.where(
        in_zone_a,
        jnp.float32(height_a),
        jnp.where(
            in_zone_b,
            jnp.float32(height_b),
            jnp.where(
                in_trans_a_to_b,
                # Interpolate A→B
                jnp.float32(height_a) + (height_b - height_a) * trans_a_to_b_progress,
                # Interpolate B→A
                jnp.float32(height_b) + (height_a - height_b) * trans_b_to_a_progress
            )
        )
    )

    is_transition = in_trans_a_to_b | in_trans_b_to_a
    transition_progress = jnp.where(in_trans_a_to_b, trans_a_to_b_progress, trans_b_to_a_progress)

    return height.astype(jnp.int32), is_transition, transition_progress



# --- State and Observation ---
class RoadRunnerState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_x_history: chex.Array
    player_y_history: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    step_counter: chex.Array
    player_is_moving: chex.Array
    player_looks_right: chex.Array
    enemy_is_moving: chex.Array
    enemy_looks_right: chex.Array
    score: chex.Array
    is_scrolling: chex.Array
    scrolling_step_counter: chex.Array
    is_round_over: chex.Array
    seeds: chex.Array # 2D array of shape (4, 3)
    next_seed_spawn_scroll_step: chex.Array # Scrolling step counter value at which to spawn next seed
    rng: chex.Array # PRNG state
    seed_pickup_streak: chex.Array
    last_picked_up_seed_id: chex.Array
    next_seed_id: chex.Array
    truck_x: chex.Array
    truck_y: chex.Array
    next_truck_spawn_step: chex.Array
    current_level: chex.Array
    level_transition_timer: chex.Array
    is_in_transition: chex.Array
    lives: chex.Array
    jump_timer: chex.Array  # Countdown timer for jump (0 when not jumping)
    is_jumping: chex.Array  # Boolean flag indicating if player is currently jumping
    ravines: chex.Array 
    next_ravine_spawn_scroll_step: chex.Array
    landmine_x: chex.Array
    landmine_y: chex.Array
    next_landmine_spawn_step: chex.Array
    cannon_x: chex.Array
    cannon_y: chex.Array
    next_cannon_spawn_step: chex.Array
    cannon_has_fired: chex.Array
    cannon_is_mirrored: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    death_timer: chex.Array
    instant_death: chex.Array # Boolean, if true, skip death animation/delay
    is_falling: chex.Array  # Boolean flag indicating if player is falling into ravine
    fall_timer: chex.Array  # Countdown timer for fall animation (0 when not falling)
    fall_clip_y: chex.Array  # Y coordinate below which player sprite should be clipped during fall
    enemy_speed_phase_start: chex.Array  # Scroll step when current speed phase cycle began
    enemy_flattened_timer: chex.Array  # Timer for enemy being run over
    player_on_offramp: chex.Array  # Boolean, whether the player is currently on the offramp

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class RoadRunnerObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition
    score: jnp.ndarray
    ravine: EntityPosition


class RoadRunnerInfo(NamedTuple):
    score: jnp.ndarray
    lives: jnp.ndarray
    step: jnp.ndarray


# --- Main Environment Class ---
class JaxRoadRunner(
    JaxEnvironment[RoadRunnerState, RoadRunnerObservation, RoadRunnerInfo, RoadRunnerConstants]
):
    def __init__(self, consts: RoadRunnerConstants = None):
        if consts is None:
            consts = RoadRunnerConstants(levels=DEFAULT_LEVELS)
        elif len(consts.levels) == 0:
            consts = consts._replace(levels=DEFAULT_LEVELS)
        super().__init__(consts)
        self.renderer = RoadRunnerRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
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
        self.obs_size = 2 * 4  # Simplified

        # Pre-calculate normalized velocities
        sqrt2_inv = 1 / jnp.sqrt(2)
        self._velocities = (
            jnp.array(
                [
                    [0, 0],  # NOOP
                    [0, 0],  # FIRE (jump handled separately)
                    [0, -1],  # UP
                    [0, 1],  # DOWN
                    [-1, 0],  # LEFT
                    [1, 0],  # RIGHT
                    [sqrt2_inv, -sqrt2_inv],  # UPRIGHT
                    [-sqrt2_inv, -sqrt2_inv],  # UPLEFT
                    [sqrt2_inv, sqrt2_inv],  # DOWNRIGHT
                    [-sqrt2_inv, sqrt2_inv],  # DOWNLEFT
                    [0, -1],  # UPFIRE (jump + up)
                    [1, 0],  # RIGHTFIRE (jump + right)
                    [-1, 0],  # LEFTFIRE (jump + left)
                    [0, 1],  # DOWNFIRE (jump + down)
                    [sqrt2_inv, -sqrt2_inv],  # UPRIGHTFIRE (jump + upright)
                    [-sqrt2_inv, -sqrt2_inv],  # UPLEFTFIRE (jump + upleft)
                    [sqrt2_inv, sqrt2_inv],  # DOWNRIGHTFIRE (jump + downright)
                    [-sqrt2_inv, sqrt2_inv],  # DOWNLEFTFIRE (jump + downleft)
                ]
            )
            * self.consts.PLAYER_MOVE_SPEED
        )
        self._level_count = len(self.consts.levels)
        levels = self.consts.levels

        # Build spawn enabled arrays
        self._level_spawn_seeds = _build_spawn_enabled_array(levels, 'spawn_seeds')
        self._level_spawn_trucks = _build_spawn_enabled_array(levels, 'spawn_trucks')
        self._level_spawn_ravines = _build_spawn_enabled_array(levels, 'spawn_ravines')
        self._level_spawn_landmines = _build_spawn_enabled_array(levels, 'spawn_landmines')
        self._level_spawn_cannons = _build_spawn_enabled_array(levels, 'spawn_cannons')
        self._level_ravine_linked_seed = _build_spawn_enabled_array(levels, 'ravine_linked_seed')
        self._level_ravine_linked_mine = _build_spawn_enabled_array(levels, 'ravine_linked_mine')

        # Build spawn interval arrays
        self._seed_spawn_intervals = _build_spawn_interval_array(
            levels, 'seed_spawn_config',
            self.consts.SEED_SPAWN_MIN_INTERVAL, self.consts.SEED_SPAWN_MAX_INTERVAL
        )
        self._truck_spawn_intervals = _build_spawn_interval_array(
            levels, 'truck_spawn_config',
            self.consts.TRUCK_SPAWN_MIN_INTERVAL, self.consts.TRUCK_SPAWN_MAX_INTERVAL
        )
        self._ravine_spawn_intervals = _build_spawn_interval_array(
            levels, 'ravine_spawn_config',
            self.consts.RAVINE_SPAWN_MIN_INTERVAL, self.consts.RAVINE_SPAWN_MAX_INTERVAL
        )
        self._landmine_spawn_intervals = _build_spawn_interval_array(
            levels, 'landmine_spawn_config',
            self.consts.LANDMINE_SPAWN_MIN_INTERVAL, self.consts.LANDMINE_SPAWN_MAX_INTERVAL
        )
        self._cannon_spawn_intervals = _build_spawn_interval_array(
            levels, 'cannon_spawn_config',
            self.consts.CANNON_SPAWN_MIN_INTERVAL, self.consts.CANNON_SPAWN_MAX_INTERVAL
        )

        # Build road section arrays
        (
            self._max_road_sections,
            self._road_section_data,
            self._road_section_counts,
        ) = _build_road_section_arrays(levels, self.consts)

        # Build offramp data array: shape (num_levels, MAX_OFFRAMPS, 3 + MAX_OFFRAMP_BRIDGES)
        self._offramp_data = _build_offramp_arrays(levels)

        # Build fixed cannon scroll-step arrays: shape (num_levels, MAX_FIXED_CANNONS)
        self._cannon_fixed_steps = _build_cannon_fixed_steps_array(levels)

        # Build per-level scroll distances array
        self._level_scroll_distances = jnp.array(
            [cfg.scroll_distance_to_complete for cfg in levels],
            dtype=jnp.int32,
        ) if levels else jnp.array([self.consts.LEVEL_COMPLETE_SCROLL_DISTANCE], dtype=jnp.int32)

        (
            self._dynamic_road_enabled,
            self._dynamic_road_heights,
            self._dynamic_road_intervals,
            self._dynamic_road_transition_lengths,
            self._dynamic_road_scroll_offsets
        ) = _build_dynamic_road_config_arrays(levels)

    def _handle_input(self, action: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
        """Handles user input to determine player velocity and jump action."""
        # Map action to the corresponding index in the action_set
        action_idx = jnp.argmax(jnp.array(self.action_set) == action)
        vel = self._velocities[action_idx]
        # Check if action involves FIRE (jump): FIRE (1) or any *FIRE action (10-17)
        is_fire_action = (action == Action.FIRE) | ((action >= Action.UPFIRE) & (action <= Action.DOWNLEFTFIRE))
        return vel[0], vel[1], is_fire_action

    def _get_active_offramp_row(
        self, state: "RoadRunnerState"
    ) -> tuple[chex.Array, jnp.ndarray]:
        """Find the currently active offramp for this level.

        Returns (any_active, row) where:
          any_active: scalar bool — True when an offramp is on-screen
          row: shape (3 + MAX_OFFRAMP_BRIDGES,) — data for the active offramp
               (or the first row if none is active; callers gate on any_active).
        """
        return _find_active_offramp_row(
            self._offramp_data, state, self._level_count, self.consts)

    def _get_offramp_info(
        self, state: "RoadRunnerState"
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Return (offramp_active, split_x, merge_x, offramp_top, offramp_bottom).

        split_x / merge_x are the screen X coordinates of the leading edges of the
        split and merge diagonal sprites respectively.
        offramp_top / offramp_bottom are the screen Y coordinates of the offramp road.
        """
        SPEED = self.consts.PLAYER_MOVE_SPEED
        RAMP_W = self.consts.OFFRAMP_RAMP_WIDTH
        W = self.consts.WIDTH

        any_active, row = self._get_active_offramp_row(state)
        scroll_start = row[1]
        scroll_end = row[2]

        counter = state.scrolling_step_counter

        # Screen x of split leading edge: 0 at scroll_start, grows right each step
        split_x = (counter - scroll_start) * SPEED
        # Screen x of merge leading edge: 0 at scroll_end, grows right each step
        merge_x = (counter - scroll_end) * SPEED

        offramp_active = any_active

        road_top, _, _ = self._get_road_bounds(state)
        offramp_bottom = (road_top - self.consts.OFFRAMP_GAP).astype(jnp.int32)
        offramp_top = (offramp_bottom - self.consts.OFFRAMP_HEIGHT).astype(jnp.int32)
        return offramp_active, split_x, merge_x, offramp_top, offramp_bottom

    def _get_bridge_screen_xs(self, state: "RoadRunnerState") -> jnp.ndarray:
        """Return array of shape (MAX_OFFRAMP_BRIDGES,) with the screen-X left edge of each
        bridge.  A value of -9999 means the bridge slot is unused (scroll_step == -1)."""
        SPEED = self.consts.PLAYER_MOVE_SPEED
        counter = state.scrolling_step_counter

        if self._level_count == 0:
            return jnp.full((MAX_OFFRAMP_BRIDGES,), -9999, dtype=jnp.int32)

        _, row = self._get_active_offramp_row(state)
        bridge_scroll_steps = row[3:]  # shape (MAX_OFFRAMP_BRIDGES,)

        # Screen-X: bridge_scroll_step gives the step at which the bridge's left edge
        # reaches x=0 (same convention as split_x / merge_x).
        # A bridge scrolls leftward: screen_x = (counter - bridge_step) * SPEED
        # bridge_step == -1 means unused; return a sentinel off-screen.
        bridge_xs = (counter - bridge_scroll_steps) * SPEED
        # Replace unused slots (bridge_scroll_steps == -1) with off-screen sentinel
        bridge_xs = jnp.where(bridge_scroll_steps >= 0, bridge_xs, jnp.full_like(bridge_xs, -9999))
        return bridge_xs

    def _player_at_bridge(
        self, state: "RoadRunnerState", x_pos: chex.Array
    ) -> chex.Array:
        """Return True if the player (left edge at x_pos) overlaps any active bridge."""
        BRIDGE_W = self.consts.OFFRAMP_BRIDGE_WIDTH
        PLAYER_W = self.consts.PLAYER_SIZE[0]
        W = self.consts.WIDTH

        offramp_active, _, _, _, _ = self._get_offramp_info(state)
        bridge_xs = self._get_bridge_screen_xs(state)

        # Overlap: player right > bridge left AND player left < bridge right, AND bridge on screen
        def overlaps_bridge(bx):
            on_screen = (bx >= 0) & (bx < W)
            overlaps = (x_pos + PLAYER_W > bx) & (x_pos < bx + BRIDGE_W)
            return on_screen & overlaps

        # OR together all bridge overlaps
        any_bridge = jnp.any(
            jnp.array([overlaps_bridge(bridge_xs[i]) for i in range(MAX_OFFRAMP_BRIDGES)])
        )
        return offramp_active & any_bridge

    def _check_player_bounds(
        self, state: RoadRunnerState, x_pos: chex.Array, y_pos: chex.Array,
        road_top: chex.Array, road_bottom: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        main_min_y = road_top - (self.consts.PLAYER_SIZE[1] - 5)
        main_max_y = road_bottom - self.consts.PLAYER_SIZE[1]

        offramp_active, split_x, merge_x, offramp_top, offramp_bottom = \
            self._get_offramp_info(state)
        off_min_y = offramp_top - (self.consts.PLAYER_SIZE[1] - 5)
        off_max_y = offramp_bottom - self.consts.PLAYER_SIZE[1]

        RAMP_W = self.consts.OFFRAMP_RAMP_WIDTH
        PLAYER_W = self.consts.PLAYER_SIZE[0]

        # The player can only cross between roads when they are physically at one of the
        # diagonal connecting sections (split or merge) OR at a bridge.
        at_split = (x_pos + PLAYER_W > split_x - RAMP_W) & (x_pos < split_x)
        at_merge = (x_pos + PLAYER_W > merge_x) & (x_pos < merge_x + RAMP_W)
        at_bridge = self._player_at_bridge(state, x_pos)
        # The merge is the END of the offramp — it only allows descent (offramp → main road).
        # A player already on the main road must not be able to re-enter the offramp via the
        # merge; the road has ended there and there is nothing above the merge to the right.
        at_merge_descending = at_merge & state.player_on_offramp
        in_transition = offramp_active & (at_split | at_merge_descending | at_bridge)

        # During a transition (split, merge, or bridge), the player can cross between roads.
        # When the proposed y falls in the gap zone (off_max_y < y < main_min_y), complete
        # the crossing in one step to the destination road — no parking in the median.
        y_in_gap = (y_pos > off_max_y) & (y_pos < main_min_y)
        y_cross = jnp.where(
            state.player_on_offramp,
            main_min_y,   # was on offramp, moving down → land on main road
            off_max_y,    # was on main road, moving up → land on offramp
        )
        y_transition = jnp.where(
            y_in_gap,
            y_cross,
            jnp.clip(y_pos, off_min_y, main_max_y),
        )

        # Outside a transition the player is constrained to their current road.
        # The main road uses its full vertical range including the top lane — hitbox
        # separation in _check_game_over handles enemy-through-median concerns.
        #
        # Once the merge sprite has scrolled past the player's position the offramp road
        # strip no longer covers them, so we treat it as ended regardless of player_on_offramp.
        # Bridges are an exception: if a bridge is active at the player's x the player can
        # still cross, so the merge-has-passed rule is suppressed in that case.
        # De Morgan: ~(merge_has_passed & ~at_bridge) = ~merge_has_passed | at_bridge
        merge_has_passed = merge_x > x_pos + PLAYER_W
        on_offramp_road = state.player_on_offramp & offramp_active & (~merge_has_passed | at_bridge)
        checked_y = jnp.where(
            in_transition,
            y_transition,
            jnp.where(
                on_offramp_road,
                jnp.clip(y_pos, off_min_y, off_max_y),
                jnp.clip(y_pos, main_min_y, main_max_y),
            ),
        )
        checked_x = jnp.clip(
            x_pos,
            self.consts.SIDE_MARGIN,
            self.consts.WIDTH - self.consts.PLAYER_SIZE[0] - self.consts.SIDE_MARGIN,
        )
        return (checked_x, checked_y)

    def _check_enemy_bounds(
        self, x_pos: chex.Array, y_pos: chex.Array,
        road_top: chex.Array, road_bottom: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        min_y = road_top - (self.consts.PLAYER_SIZE[1] // 3)
        max_y = road_bottom - self.consts.PLAYER_SIZE[1]
        checked_y = jnp.clip(y_pos, min_y, max_y)

        # Only clip x on the left side
        checked_x = jnp.maximum(x_pos, self.consts.SIDE_MARGIN)

        return (checked_x, checked_y)

    def _handle_scrolling(self, state: RoadRunnerState, x_pos: chex.Array):
        return jnp.where(state.is_scrolling, x_pos + self.consts.PLAYER_MOVE_SPEED, x_pos)

    def _player_step(
        self, state: RoadRunnerState, action: chex.Array,
        road_top: chex.Array, road_bottom: chex.Array,
    ) -> RoadRunnerState:

        # --- Update Player Position ---
        input_vel_x, input_vel_y, is_fire_action = self._handle_input(action)

        # Handle jump logic (simple boolean state - no position checking)
        # If FIRE is pressed and not already jumping, start jump
        # Otherwise, count down the jump timer
        can_start_jump = (state.jump_timer == 0) & jnp.logical_not(state.is_round_over)
        should_start_jump = is_fire_action & can_start_jump

        new_jump_timer = jnp.where(
            should_start_jump,
            jnp.array(self.consts.JUMP_TIME_DURATION, dtype=jnp.int32),
            jnp.maximum(state.jump_timer - 1, 0),
        )

        # Determine if currently jumping
        is_jumping = new_jump_timer > 0

        # If round is over, player is forced to move right.
        vel_x = jnp.where(
            state.is_round_over,
            jnp.array(self.consts.PLAYER_MOVE_SPEED, dtype=jnp.float32),
            input_vel_x,
        )
        vel_y = jnp.where(state.is_round_over, 0.0, input_vel_y)

        # Determine if scrolling should happen based on the potential next position.
        tentative_player_x = state.player_x + vel_x
        is_scrolling = tentative_player_x < self.consts.X_SCROLL_THRESHOLD

        # When scrolling, the player's horizontal velocity should counteract the scroll.
        # We use the original vel_x for non-scrolling movement.
        final_vel_x = jnp.where(
            is_scrolling,
            -float(self.consts.PLAYER_MOVE_SPEED),
            vel_x,
        )

        player_x = state.player_x + final_vel_x
        player_y = state.player_y + vel_y

        player_x, player_y = self._check_player_bounds(state, player_x, player_y, road_top, road_bottom)

        is_moving = (vel_x != 0) | (vel_y != 0)

        # Update player orientation based on horizontal movement
        player_looks_right = _update_orientation(vel_x, state.player_looks_right)

        # Update the state with the scrolling flag for other parts of the game (e.g., rendering).
        state = state._replace(
            is_scrolling=is_scrolling,
            scrolling_step_counter=state.scrolling_step_counter + jnp.where(is_scrolling, 1, 0),
        )

        # Apply the scroll offset to the player's final position.
        player_x = self._handle_scrolling(state, player_x)

        # Update player position history for enemy AI
        new_x_history = jnp.roll(state.player_x_history, shift=1)
        new_x_history = new_x_history.at[0].set(state.player_x)
        new_y_history = jnp.roll(state.player_y_history, shift=1)
        new_y_history = new_y_history.at[0].set(state.player_y)

        # Determine which road the player is on after movement.
        # Use the player's proposed X to check whether they are at a diagonal section or bridge.
        # Outside those crossing points the current road is preserved (bounds already enforce it).
        offramp_active, split_x, merge_x, offramp_top, offramp_bottom = \
            self._get_offramp_info(state)
        road_top_after, _, _ = self._get_road_bounds(state)
        RAMP_W = self.consts.OFFRAMP_RAMP_WIDTH
        PLAYER_W = self.consts.PLAYER_SIZE[0]
        at_split = (player_x + PLAYER_W > split_x - RAMP_W) & (player_x < split_x)
        at_merge = (player_x + PLAYER_W > merge_x) & (player_x < merge_x + RAMP_W)
        at_bridge = self._player_at_bridge(state, player_x)
        # Merge is unidirectional: only offramp → main road.  A main-road player must not
        # be able to snap back to the offramp via the merge (phantom-extension bug).
        at_merge_descending = at_merge & state.player_on_offramp
        in_transition = offramp_active & (at_split | at_merge_descending | at_bridge)
        # Player is "on the offramp" only when their top edge is within the offramp band.
        # off_max_y is the lowest valid top-edge position on the offramp road.
        # Once the player descends even one pixel below it they are committed to the
        # main road.  The old midpoint-of-gap threshold left a wide dead-zone (y in
        # [off_max_y+1 .. midpoint-1]) where the player was still flagged "on offramp"
        # while physically in the gap, causing a snap back to the offramp the moment
        # the bridge/merge diagonal scrolled away and in_transition turned False.
        off_max_y_int = offramp_bottom - self.consts.PLAYER_SIZE[1]
        on_offramp_by_y = player_y.astype(jnp.int32) <= off_max_y_int
        # After the merge has scrolled past the player's position there is no more offramp
        # road beneath them.  Clear player_on_offramp so the main-road bounds take over.
        # Bridges override this: if the player is at a bridge they are still crossing.
        merge_has_passed = merge_x > player_x + PLAYER_W
        new_on_offramp = jnp.where(
            in_transition,
            on_offramp_by_y,
            # De Morgan: ~(merge_has_passed & ~at_bridge) = ~merge_has_passed | at_bridge
            state.player_on_offramp & offramp_active & (~merge_has_passed | at_bridge),
        )

        return state._replace(
            player_x=player_x.astype(jnp.int32),
            player_y=player_y.astype(jnp.int32),
            player_is_moving=is_moving,
            player_looks_right=player_looks_right,
            player_x_history=new_x_history,
            player_y_history=new_y_history,
            jump_timer=new_jump_timer,
            is_jumping=is_jumping,
            player_on_offramp=new_on_offramp,
        )

    def _enemy_step(self, state: RoadRunnerState, road_top: chex.Array, road_bottom: chex.Array) -> RoadRunnerState:
        def game_over_logic(st: RoadRunnerState) -> RoadRunnerState:
            new_enemy_x = st.enemy_x + self.consts.PLAYER_MOVE_SPEED
            new_enemy_x, new_enemy_y = self._check_enemy_bounds(new_enemy_x, st.enemy_y, road_top, road_bottom)
            return st._replace(
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_is_moving=True,
                enemy_looks_right=True,
            )

        def flattened_logic(st: RoadRunnerState) -> RoadRunnerState:
            new_timer = st.enemy_flattened_timer - 1
            # Update position only based on scrolling (stuck to road)
            new_enemy_x = self._handle_scrolling(st, st.enemy_x)
            new_enemy_x, new_enemy_y = self._check_enemy_bounds(new_enemy_x, st.enemy_y, road_top, road_bottom)

            return st._replace(
                enemy_x=new_enemy_x.astype(jnp.int32),
                enemy_y=new_enemy_y.astype(jnp.int32),
                enemy_is_moving=False,
                enemy_flattened_timer=new_timer
            )

        def normal_logic(st: RoadRunnerState) -> RoadRunnerState:
            # Calculate current speed phase based on scroll distance
            total_cycle = (self.consts.ENEMY_SLOW_DURATION +
                           self.consts.ENEMY_FAST_DURATION +
                           self.consts.ENEMY_SAME_DURATION)
            
            cycle_progress = (st.scrolling_step_counter - st.enemy_speed_phase_start) % total_cycle
            
            # Determine current phase speed offset
            in_slow = cycle_progress < self.consts.ENEMY_SLOW_DURATION
            in_fast = (cycle_progress >= self.consts.ENEMY_SLOW_DURATION) & \
                      (cycle_progress < self.consts.ENEMY_SLOW_DURATION + self.consts.ENEMY_FAST_DURATION)
            
            speed_offset = jnp.where(
                in_slow,
                self.consts.ENEMY_SLOW_SPEED_OFFSET,
                jnp.where(
                    in_fast,
                    self.consts.ENEMY_FAST_SPEED_OFFSET,
                    self.consts.ENEMY_SAME_SPEED_OFFSET
                )
            )
            
            base_speed = self.consts.PLAYER_MOVE_SPEED + speed_offset
            
            # Check if player is moving right (approaching enemy)
            # Player velocity is the difference between current position and previous position
            player_vel_x = st.player_x - st.player_x_history[0]
            is_approaching = player_vel_x > 0
            
            slowdown = jnp.where(
                is_approaching,
                self.consts.ENEMY_APPROACH_SLOWDOWN,
                1.0
            )
            
            # Calculate final speed (absolute value for clipping bounds)
            # Negative slowdown will reverse direction via the multiplier on delta
            final_speed = (base_speed * jnp.abs(slowdown)).astype(jnp.int32)
            final_speed = jnp.maximum(final_speed, 1)

            # Get the distance to the player, with a configurable frame delay.
            delayed_player_x = st.player_x_history[
                self.consts.ENEMY_REACTION_DELAY - 1
            ]
            delayed_player_y = st.player_y_history[
                self.consts.ENEMY_REACTION_DELAY - 1
            ]
            delta_x = delayed_player_x - st.enemy_x
            delta_y = delayed_player_y - st.enemy_y
            
            # Apply direction modifier (negative slowdown reverses direction)
            direction_modifier = jnp.where(slowdown < 0, -1.0, 1.0)
            modified_delta_x = delta_x * direction_modifier
            modified_delta_y = delta_y * direction_modifier

            # Determine enemy movement and orientation
            enemy_is_moving = (delta_x != 0) | (delta_y != 0)
            enemy_looks_right = _update_orientation(modified_delta_x, st.enemy_looks_right)

            # Update enemy position, clipping movement to final_speed to prevent jittering
            new_enemy_x = st.enemy_x + jnp.clip(
                modified_delta_x, -final_speed, final_speed
            )
            new_enemy_y = st.enemy_y + jnp.clip(
                modified_delta_y, -final_speed, final_speed
            )

            new_enemy_x = self._handle_scrolling(st, new_enemy_x)

            new_enemy_x, new_enemy_y = self._check_enemy_bounds(new_enemy_x, new_enemy_y, road_top, road_bottom)
            return st._replace(
                enemy_x=new_enemy_x.astype(jnp.int32),
                enemy_y=new_enemy_y.astype(jnp.int32),
                enemy_is_moving=enemy_is_moving,
                enemy_looks_right=enemy_looks_right,
            )

        # Use switch instead of nested conds: 0=normal, 1=flattened, 2=game_over
        branch_idx = jnp.where(
            state.is_round_over, 2,
            jnp.where(state.enemy_flattened_timer > 0, 1, 0)
        )
        return jax.lax.switch(branch_idx, [normal_logic, flattened_logic, game_over_logic], state)

    def _check_game_over(self, state: RoadRunnerState) -> RoadRunnerState:
        # Check if the enemy and the player overlap
        collision = _check_aabb_collision(
            state.player_x, state.player_y,
            self.consts.PLAYER_SIZE[0], self.consts.PLAYER_SIZE[1],
            state.enemy_x, state.enemy_y,
            self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1],
        )

        # Don't trigger game over if enemy is flattened
        collision = collision & (state.enemy_flattened_timer == 0)

        # Don't trigger game over if the player is separated from the enemy by the median.
        # The enemy always stays on the main road.  A collision across the median is only
        # possible if both sprites overlap AND are on the same road surface.
        # We suppress the collision when the offramp is active and:
        #   1. player_on_offramp — player is on the upper road (offramp band), OR
        #   2. player_in_gap — player's top edge is in the forbidden gap zone while
        #      traversing a crossing point (split/merge/bridge).
        # Crucially we do NOT suppress when the player is simply in the top lane of the
        # main road (y ∈ [main_min_y, road_top-1]) — both player and enemy can occupy
        # the top lane and a collision there is valid.
        offramp_active, _, _, _, offramp_bottom = self._get_offramp_info(state)
        road_top, _, _ = self._get_road_bounds(state)
        off_max_y = offramp_bottom - self.consts.PLAYER_SIZE[1]
        main_min_y = road_top - (self.consts.PLAYER_SIZE[1] - 5)
        player_in_gap = (state.player_y > off_max_y) & (state.player_y < main_min_y)
        collision = collision & ~(offramp_active & (state.player_on_offramp | player_in_gap))

        return state._replace(
            is_round_over=state.is_round_over | collision,
            player_x=jnp.where(collision, (state.enemy_x + self.consts.ENEMY_SIZE[0] + 2).astype(jnp.int32), state.player_x),
            player_y=jnp.where(collision, state.enemy_y.astype(jnp.int32), state.player_y),
        )

    def update_streak(self, state: RoadRunnerState, seed_idx: int, max_streak: int) -> RoadRunnerState:
        last_picked_up_seed_id = state.last_picked_up_seed_id
        state = state._replace(last_picked_up_seed_id=state.seeds[seed_idx, 2])
        is_consecutive = state.seeds[seed_idx, 2] == last_picked_up_seed_id + 1
        new_streak = jnp.where(
            is_consecutive,
            jnp.minimum(state.seed_pickup_streak + 1, max_streak),
            1,
        )
        return state._replace(seed_pickup_streak=new_streak)

    def _seed_picked_up(self, state: RoadRunnerState, seed_idx: int) -> RoadRunnerState:
        state = self.update_streak(state, seed_idx, self.consts.MAX_STREAK)

        # Set seed to inactive (-1, -1)
        updated_seeds = state.seeds.at[seed_idx].set(
            jnp.array([-1, -1, -1], dtype=jnp.int32)
        )
        # Increment score by 100 Placeholder value
        new_score = state.score + self.consts.SEED_BASE_VALUE * state.seed_pickup_streak
        return state._replace(
            seeds=updated_seeds,
            score=new_score.astype(jnp.int32),
        )

    def _check_seed_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Check for collisions between player and all active seeds.
        Uses AABB (Axis-Aligned Bounding Box) collision detection.
        """
        # Player pickup area starts at PLAYER_PICKUP_OFFSET from top
        player_pickup_y = state.player_y + self.consts.PLAYER_PICKUP_OFFSET
        pickup_height = self.consts.PLAYER_SIZE[1] - self.consts.PLAYER_PICKUP_OFFSET

        def check_and_pickup_seed(i: int, st: RoadRunnerState) -> RoadRunnerState:
            """Check collision for seed at index i and pick it up if colliding."""
            seed_x = st.seeds[i, 0]
            seed_y = st.seeds[i, 1]
            is_active = seed_x >= 0

            collision = is_active & _check_aabb_collision(
                state.player_x, player_pickup_y,
                self.consts.PLAYER_SIZE[0], pickup_height,
                seed_x, seed_y,
                self.consts.SEED_SIZE[0], self.consts.SEED_SIZE[1],
            )

            return jax.lax.cond(
                collision,
                lambda s: self._seed_picked_up(s, i),
                lambda s: s,
                st,
            )

        # Check all seeds using fori_loop
        return jax.lax.fori_loop(
            0,
            state.seeds.shape[0],
            check_and_pickup_seed,
            state,
        )

    def _update_and_spawn_seeds(self, state: RoadRunnerState,
                                spawn_road_top: chex.Array, spawn_road_bottom: chex.Array) -> RoadRunnerState:
        """
        Update seed positions (apply scrolling, despawn off-screen) and spawn new seeds.
        Combined function for efficiency - seeds update and spawn logic together.
        """
        consts = self.consts
        level_idx = self._get_level_index(state)
        road_top = spawn_road_top
        road_bottom = spawn_road_bottom
        if self._level_count > 0:
            spawn_seeds_enabled = self._level_spawn_seeds[level_idx]
            seed_spawn_bounds = self._seed_spawn_intervals[level_idx]
        else:
            spawn_seeds_enabled = jnp.array(True, dtype=jnp.bool_)
            seed_spawn_bounds = jnp.array(
                [consts.SEED_SPAWN_MIN_INTERVAL, consts.SEED_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )
        # Update seed positions: apply scrolling and despawn off-screen seeds
        seed_x = state.seeds[:, 0]
        # Move active seeds when scrolling, then despawn off-screen ones
        scroll_offset = jnp.where(state.is_scrolling, consts.PLAYER_MOVE_SPEED, 0)
        updated_x = jnp.where(seed_x >= 0, seed_x + scroll_offset, seed_x)
        # Mark seeds as inactive if they moved off-screen (x >= WIDTH)
        seed_active = (updated_x >= 0) & (updated_x < consts.WIDTH)
        updated_x = jnp.where(seed_active, updated_x, -1)
        updated_seeds = (
            state.seeds.at[:, 0]
            .set(updated_x)
            .at[:, 1]
            .set(jnp.where(seed_active, state.seeds[:, 1], -1))
        )

        # Prepare for spawning: split RNG and check conditions
        rng_road, rng_spawn_y, rng_interval, rng_after = jax.random.split(state.rng, 4)
        available_slots = updated_x == -1
        should_spawn = (
            state.is_scrolling
            & (state.scrolling_step_counter >= state.next_seed_spawn_scroll_step)
            & jnp.any(available_slots)
            & spawn_seeds_enabled
        )

        # Determine whether to spawn on offramp or main road.
        # Only spawn on the offramp while the merge hasn't entered the screen yet
        # (merge_x <= 0).  Once the merge appears, the offramp road band shrinks from the
        # left and new seeds spawned at x=0 would land outside it on the median/background.
        offramp_active, _, merge_x_spawn, offramp_top_y, offramp_bottom_y = self._get_offramp_info(state)
        use_offramp = offramp_active & (merge_x_spawn <= 0) & (jax.random.uniform(rng_road) > 0.5)
        spawn_min_y = jnp.where(use_offramp, offramp_top_y.astype(jnp.int32), road_top)
        spawn_max_y = jnp.where(
            use_offramp,
            (offramp_bottom_y - consts.SEED_SIZE[1]).astype(jnp.int32),
            road_bottom - consts.SEED_SIZE[1],
        )

        def _spawn(st: RoadRunnerState) -> RoadRunnerState:
            slot_idx = jnp.argmax(available_slots)
            # Generate random Y position within selected road bounds
            seed_y = jax.random.randint(
                rng_spawn_y,
                (),
                spawn_min_y,
                jnp.maximum(spawn_min_y + 1, spawn_max_y + 1),
                dtype=jnp.int32,
            )
            # Spawn at x=0, update next spawn step
            next_spawn_step = state.scrolling_step_counter + jax.random.randint(
                rng_interval,
                (),
                seed_spawn_bounds[0],
                seed_spawn_bounds[1] + 1,
                dtype=jnp.int32,
            )
            # Get the seeds id, then increment the next id in the state
            seed_id = st.next_seed_id

            # Build spawned seeds array
            spawned_seeds = updated_seeds.at[slot_idx].set(
                jnp.array([0, seed_y, seed_id], dtype=jnp.int32)
            )
            return st._replace(
                seeds=spawned_seeds,
                next_seed_spawn_scroll_step=next_spawn_step,
                next_seed_id=seed_id + 1,
                rng=rng_after,
            )

        return jax.lax.cond(
            should_spawn,
            _spawn,
            lambda st: st._replace(seeds=updated_seeds, rng=rng_after),
            state,
        )

    def _update_and_spawn_truck(self, state: RoadRunnerState,
                                spawn_road_top: chex.Array, spawn_road_bottom: chex.Array,
                                spawn_road_height: chex.Array) -> RoadRunnerState:
        """
        Update truck position (move right at TRUCK_SPEED + scroll offset) and spawn new truck.
        Trucks spawn regardless of scrolling state, using step_counter.
        Trucks are affected by road scrolling - they move with the scroll offset.
        """
        consts = self.consts
        level_idx = self._get_level_index(state)
        road_top = spawn_road_top
        road_bottom = spawn_road_bottom
        road_height_at_spawn = spawn_road_height
        if self._level_count > 0:
            spawn_trucks_enabled = self._level_spawn_trucks[level_idx]
            truck_spawn_bounds = self._truck_spawn_intervals[level_idx]
        else:
            spawn_trucks_enabled = jnp.array(True, dtype=jnp.bool_)
            truck_spawn_bounds = jnp.array(
                [consts.TRUCK_SPAWN_MIN_INTERVAL, consts.TRUCK_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )

        # Update truck position: move active truck right, apply scrolling offset
        # Move by TRUCK_SPEED, plus scroll offset when scrolling is active
        scroll_offset = jnp.where(state.is_scrolling, consts.PLAYER_MOVE_SPEED, 0)
        updated_truck_x = jnp.where(
            state.truck_x >= 0,
            state.truck_x + consts.TRUCK_SPEED + scroll_offset,
            state.truck_x
        )
        # Despawn if off-screen
        truck_active = (updated_truck_x >= 0) & (updated_truck_x < consts.WIDTH)
        updated_truck_x = jnp.where(truck_active, updated_truck_x, -1)
        updated_truck_y = jnp.where(truck_active, state.truck_y, -1)

        # Prepare for spawning: split RNG and check conditions
        rng_spawn_y, rng_interval, rng_after = jax.random.split(state.rng, 3)

        # Trucks only spawn if the road is wide enough (>= 70)
        road_wide_enough = road_height_at_spawn >= 70

        should_spawn = (
            (updated_truck_x < 0)  # No truck currently active
            & (state.step_counter >= state.next_truck_spawn_step)
            & spawn_trucks_enabled
            & road_wide_enough
        )

        # Compute spawn values unconditionally
        spawn_min = road_top
        spawn_max = road_bottom - consts.TRUCK_SIZE[1]
        truck_y_spawn = jax.random.randint(
            rng_spawn_y, (), spawn_min,
            jnp.maximum(spawn_min + 1, spawn_max + 1), dtype=jnp.int32,
        )
        next_spawn_step = state.step_counter + jax.random.randint(
            rng_interval, (), truck_spawn_bounds[0],
            truck_spawn_bounds[1] + 1, dtype=jnp.int32,
        )

        return state._replace(
            truck_x=jnp.where(should_spawn, jnp.array(0, dtype=jnp.int32), updated_truck_x),
            truck_y=jnp.where(should_spawn, truck_y_spawn, updated_truck_y),
            next_truck_spawn_step=jnp.where(should_spawn, next_spawn_step, state.next_truck_spawn_step),
            rng=rng_after,
        )

    def _check_truck_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Check for collisions between truck and player/enemy.
        Uses AABB (Axis-Aligned Bounding Box) collision detection.
        All branches use jnp.where to avoid tracing multiple cond paths.
        """
        truck_active = state.truck_x >= 0

        # Truck collision area for player (lower half, using TRUCK_COLLISION_OFFSET)
        truck_collision_y_player = state.truck_y + self.consts.TRUCK_COLLISION_OFFSET
        truck_collision_height_player = self.consts.TRUCK_SIZE[1] - self.consts.TRUCK_COLLISION_OFFSET

        # Player pickup area (lower portion)
        player_pickup_y = state.player_y + self.consts.PLAYER_PICKUP_OFFSET
        pickup_height = self.consts.PLAYER_SIZE[1] - self.consts.PLAYER_PICKUP_OFFSET

        # Check player-truck collision (only if truck active)
        player_collision = truck_active & _check_aabb_collision(
            state.player_x, player_pickup_y,
            self.consts.PLAYER_SIZE[0], pickup_height,
            state.truck_x, truck_collision_y_player,
            self.consts.TRUCK_SIZE[0], truck_collision_height_player,
        )

        # --- Enemy Collision Logic (Forgiving) ---
        hit_buffer = 4
        e_x = state.enemy_x - hit_buffer
        e_y = state.enemy_y - hit_buffer
        e_w = self.consts.ENEMY_SIZE[0] + (hit_buffer * 2)
        e_h = self.consts.ENEMY_SIZE[1] + (hit_buffer * 2)

        enemy_collision = truck_active & _check_aabb_collision(
            e_x, e_y, e_w, e_h,
            state.truck_x, state.truck_y,
            self.consts.TRUCK_SIZE[0], self.consts.TRUCK_SIZE[1],
        )

        # Handle player collision with jnp.where
        state = state._replace(
            is_round_over=state.is_round_over | player_collision,
            player_x=jnp.where(player_collision, (state.truck_x + self.consts.TRUCK_SIZE[0] + 2).astype(jnp.int32), state.player_x),
        )

        # Handle enemy collision with jnp.where (only if not already flattened)
        should_flatten = enemy_collision & (state.enemy_flattened_timer == 0)
        state = state._replace(
            enemy_flattened_timer=jnp.where(should_flatten, jnp.array(self.consts.ENEMY_FLATTENED_DURATION, dtype=jnp.int32), state.enemy_flattened_timer),
            score=jnp.where(should_flatten, state.score + self.consts.ENEMY_FLATTENED_SCORE, state.score),
        )

        return state
    
    def _update_and_spawn_ravines(self, state: RoadRunnerState,
                                  center_road_top: chex.Array, center_road_bottom: chex.Array,
                                  center_road_height: chex.Array) -> RoadRunnerState:
        """
        Update ravine positions (move left with scroll speed) and spawn new ravines.
        Ravines are fixed to the road, so they move exactly with the scroll speed.
        """
        consts = self.consts
        level_idx = self._get_level_index(state)
        if self._level_count > 0:
            spawn_ravines_enabled = self._level_spawn_ravines[level_idx]
            ravine_spawn_bounds = self._ravine_spawn_intervals[level_idx]
        else:
            spawn_ravines_enabled = jnp.array(False, dtype=jnp.bool_)
            ravine_spawn_bounds = jnp.array(
                [consts.RAVINE_SPAWN_MIN_INTERVAL, consts.RAVINE_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )

        road_top = center_road_top
        road_bottom = center_road_bottom
        road_height = center_road_height
        
        # Only spawn if road height is compatible (== 32)
        height_compatible = road_height == 32
        
        should_spawn_active = spawn_ravines_enabled & height_compatible
        
        # Update ravine positions: move active ravines.
        # Ravines move ONLY when scrolling happens.
        ravine_x = state.ravines[:, 0]
        
        scroll_offset = jnp.where(state.is_scrolling, consts.PLAYER_MOVE_SPEED, 0)
        
        # Update positions
        ravine_x = jnp.where(
            state.ravines[:, 0] >= 0,
            state.ravines[:, 0] + scroll_offset,
            state.ravines[:, 0]
        )
        
        # Despawn if off-screen (>= WIDTH)
        ravine_active = (ravine_x >= 0) & (ravine_x < consts.WIDTH)
        updated_x = jnp.where(ravine_active, ravine_x, -1)
        updated_y = jnp.where(ravine_active, state.ravines[:, 1], -1)
        
        updated_ravines = jnp.stack([updated_x, updated_y], axis=-1)
        
        # Spawning Logic
        rng_spawn, rng_interval, rng_link, rng_after = jax.random.split(state.rng, 4)
        available_slots = updated_x == -1
        
        should_spawn = (
            state.is_scrolling
            & (state.scrolling_step_counter >= state.next_ravine_spawn_scroll_step)
            & jnp.any(available_slots)
            & should_spawn_active
        )
        
        # Compute spawn values unconditionally
        slot_idx = jnp.argmax(available_slots)
        spawn_y = road_top
        next_spawn_step = state.scrolling_step_counter + jax.random.randint(
            rng_interval, (), ravine_spawn_bounds[0],
            ravine_spawn_bounds[1] + 1, dtype=jnp.int32,
        )
        new_ravine = jnp.array([0, spawn_y], dtype=jnp.int32)
        spawned_ravines = updated_ravines.at[slot_idx].set(new_ravine)

        # --- Ravine-linked entity scheduling ---
        # When a ravine spawns, optionally pre-schedule a seed or mine to appear
        # just ahead of it so the player encounters the entity before the ravine.
        # A single random draw determines which (if any) entity is linked, ensuring
        # seeds and mines never appear before the same ravine.
        if self._level_count > 0:
            ravine_linked_seed = self._level_ravine_linked_seed[level_idx]
            ravine_linked_mine = self._level_ravine_linked_mine[level_idx]
        else:
            ravine_linked_seed = jnp.array(False, dtype=jnp.bool_)
            ravine_linked_mine = jnp.array(False, dtype=jnp.bool_)

        link_roll = jax.random.uniform(rng_link)
        seed_threshold = jnp.float32(consts.RAVINE_SEED_LINK_PROB)
        mine_threshold = jnp.float32(consts.RAVINE_SEED_LINK_PROB + consts.RAVINE_MINE_LINK_PROB)

        should_link_seed = ravine_linked_seed & (link_roll < seed_threshold)
        should_link_mine = ravine_linked_mine & (link_roll >= seed_threshold) & (link_roll < mine_threshold)

        # Seed: schedule it RAVINE_SEED_AHEAD_SCROLL_STEPS before the next ravine spawns.
        # next_spawn_step is the scrolling step at which the NEXT ravine will appear;
        # the seed should spawn that many steps earlier (i.e. arrive at the player ahead of it).
        linked_seed_scroll_step = next_spawn_step - jnp.int32(consts.RAVINE_SEED_AHEAD_SCROLL_STEPS)
        new_next_seed_spawn_scroll_step = jnp.where(
            should_spawn & should_link_seed,
            linked_seed_scroll_step,
            state.next_seed_spawn_scroll_step,
        )

        # Mine: same idea but using step_counter (mine timer is step-based, not scroll-based).
        # When should_spawn is True, next_spawn_step = scrolling_step_counter + interval,
        # so remaining is always positive (== the drawn interval).  The approximation
        # remaining_scroll_steps ≈ remaining_step_counter_steps holds because the player
        # is continuously scrolling when running; any small deviation is not significant.
        remaining_scroll_to_next_ravine = next_spawn_step - state.scrolling_step_counter
        mine_steps_from_now = remaining_scroll_to_next_ravine - jnp.int32(consts.RAVINE_MINE_AHEAD_SCROLL_STEPS)
        # Only schedule if there is enough lead time to appear ahead of the ravine
        enough_lead_time = mine_steps_from_now > jnp.int32(0)
        should_link_mine = should_link_mine & enough_lead_time
        linked_mine_step = state.step_counter + mine_steps_from_now
        new_next_landmine_spawn_step = jnp.where(
            should_spawn & should_link_mine,
            linked_mine_step,
            state.next_landmine_spawn_step,
        )

        return state._replace(
            ravines=jnp.where(should_spawn, spawned_ravines, updated_ravines),
            next_ravine_spawn_scroll_step=jnp.where(should_spawn, next_spawn_step, state.next_ravine_spawn_scroll_step),
            next_seed_spawn_scroll_step=new_next_seed_spawn_scroll_step,
            next_landmine_spawn_step=new_next_landmine_spawn_step,
            rng=rng_after,
        )

    def _check_ravine_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Check collision with ravines.
        If player overlaps with ravine AND is NOT jumping, they start falling animation.
        """
        # Player feet area (bottom 4 pixels)
        player_feet_y = state.player_y + self.consts.PLAYER_SIZE[1] - 4
        feet_height = 4

        def check_ravine(i, st):
            r_x = st.ravines[i, 0]
            r_y = st.ravines[i, 1]
            active = r_x >= 0

            overlap = _check_aabb_collision(
                state.player_x, player_feet_y,
                self.consts.PLAYER_SIZE[0], feet_height,
                r_x, r_y,
                self.consts.RAVINE_SIZE[0], self.consts.RAVINE_SIZE[1],
            )

            collision = active & overlap & jnp.logical_not(st.is_jumping)

            # Center player on ravine when collision first occurs
            ravine_center_x = r_x + (self.consts.RAVINE_SIZE[0] - self.consts.PLAYER_SIZE[0]) // 2
            new_player_x = jnp.where(
                collision & jnp.logical_not(st.is_falling),
                ravine_center_x,
                st.player_x
            )

            # Calculate clip boundary (bottom of road/ravine)
            ravine_bottom_y = r_y + self.consts.RAVINE_SIZE[1]
            new_fall_clip_y = jnp.where(
                collision & jnp.logical_not(st.is_falling),
                ravine_bottom_y,
                st.fall_clip_y
            )

            # Start fall animation instead of instant death
            new_is_falling = st.is_falling | collision
            new_fall_timer = jnp.where(
                collision & jnp.logical_not(st.is_falling),
                jnp.array(self.consts.FALL_ANIMATION_DURATION, dtype=jnp.int32),
                st.fall_timer
            )

            return st._replace(
                player_x=new_player_x,
                is_falling=new_is_falling,
                fall_timer=new_fall_timer,
                fall_clip_y=new_fall_clip_y
            )

        return jax.lax.fori_loop(0, 3, check_ravine, state)

    def _update_and_spawn_landmines(self, state: RoadRunnerState,
                                    spawn_road_top: chex.Array, spawn_road_bottom: chex.Array) -> RoadRunnerState:
        """
        Update landmine positions (move with scroll) and spawn new landmines.
        Only one landmine active at a time.
        """
        consts = self.consts
        level_idx = self._get_level_index(state)
        road_top = spawn_road_top
        road_bottom = spawn_road_bottom
        
        if self._level_count > 0:
            spawn_landmines_enabled = self._level_spawn_landmines[level_idx]
            landmine_spawn_bounds = self._landmine_spawn_intervals[level_idx]
        else:
            spawn_landmines_enabled = jnp.array(False, dtype=jnp.bool_)
            landmine_spawn_bounds = jnp.array(
                [consts.LANDMINE_SPAWN_MIN_INTERVAL, consts.LANDMINE_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )

        # Update landmine position
        # Move by scroll offset when scrolling is active
        scroll_offset = jnp.where(state.is_scrolling, consts.PLAYER_MOVE_SPEED, 0)
        
        updated_landmine_x = jnp.where(
            state.landmine_x >= 0,
            state.landmine_x + scroll_offset,
            state.landmine_x
        )
        
        # Despawn if off-screen
        landmine_active = (updated_landmine_x >= 0) & (updated_landmine_x < consts.WIDTH)
        updated_landmine_x = jnp.where(landmine_active, updated_landmine_x, -1)
        updated_landmine_y = jnp.where(landmine_active, state.landmine_y, -1)

        # Prepare for spawning
        rng_road, rng_spawn_y, rng_interval, rng_after = jax.random.split(state.rng, 4)
        
        should_spawn = (
            (updated_landmine_x < 0)  # No landmine currently active
            & (state.step_counter >= state.next_landmine_spawn_step)
            & spawn_landmines_enabled
            & (state.death_timer == 0) # Don't spawn if dying
            & jnp.logical_not(state.is_in_transition)
        )

        # Determine whether to spawn on offramp or main road.
        # Only spawn on the offramp while the merge hasn't entered the screen yet
        # (merge_x <= 0).  Once the merge appears, the offramp road band shrinks from the
        # left and new landmines spawned at x=0 would land outside it on the median/background.
        offramp_active, _, merge_x_spawn, offramp_top_y, offramp_bottom_y = self._get_offramp_info(state)
        use_offramp = offramp_active & (merge_x_spawn <= 0) & (jax.random.uniform(rng_road) > 0.5)
        spawn_min_y = jnp.where(use_offramp, offramp_top_y.astype(jnp.int32), road_top)
        spawn_max_y = jnp.where(
            use_offramp,
            (offramp_bottom_y - consts.LANDMINE_SIZE[1]).astype(jnp.int32),
            road_bottom - consts.LANDMINE_SIZE[1],
        )

        def _spawn(st: RoadRunnerState) -> RoadRunnerState:
            # Generate random Y position within selected road bounds
            landmine_y = jax.random.randint(
                rng_spawn_y,
                (),
                spawn_min_y,
                jnp.maximum(spawn_min_y + 1, spawn_max_y + 1),
                dtype=jnp.int32,
            )

            next_spawn_step = st.step_counter + jax.random.randint(
                rng_interval,
                (),
                landmine_spawn_bounds[0],
                landmine_spawn_bounds[1] + 1,
                dtype=jnp.int32,
            )
            return st._replace(
                landmine_x=jnp.array(0, dtype=jnp.int32),
                landmine_y=landmine_y,
                next_landmine_spawn_step=next_spawn_step,
                rng=rng_after,
            )

        return jax.lax.cond(
            should_spawn,
            _spawn,
            lambda st: st._replace(
                landmine_x=updated_landmine_x,
                landmine_y=updated_landmine_y,
                rng=rng_after,
            ),
            state,
        )

    def _check_landmine_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Check collision between player and landmine.
        """
        active = state.landmine_x >= 0

        # Player pickup area (lower portion)
        player_pickup_y = state.player_y + self.consts.PLAYER_PICKUP_OFFSET
        pickup_height = self.consts.PLAYER_SIZE[1] - self.consts.PLAYER_PICKUP_OFFSET

        overlap = _check_aabb_collision(
            state.player_x, player_pickup_y,
            self.consts.PLAYER_SIZE[0], pickup_height,
            state.landmine_x, state.landmine_y,
            self.consts.LANDMINE_SIZE[0], self.consts.LANDMINE_SIZE[1],
        )
        collision = active & overlap & (state.death_timer == 0)

        return state._replace(
            death_timer=jnp.where(collision, jnp.array(self.consts.DEATH_ANIMATION_DURATION, dtype=jnp.int32), state.death_timer),
            landmine_x=jnp.where(collision, jnp.array(-1, dtype=jnp.int32), state.landmine_x)
        )

    def _update_and_spawn_cannon_and_bullets(self, state: RoadRunnerState,
                                             spawn_road_top: chex.Array) -> RoadRunnerState:
        """
        Update cannon position (moves with scroll_offset like ravines),
        and spawn new bullets from cannons.
        Bullets spawn from active cannons and move rightwards.
        """
        consts = self.consts
        level_idx = self._get_level_index(state)
        if self._level_count > 0:
            spawn_cannons_enabled = self._level_spawn_cannons[level_idx]
            cannon_spawn_bounds = self._cannon_spawn_intervals[level_idx]
        else:
            spawn_cannons_enabled = jnp.array(False, dtype=jnp.bool_)
            cannon_spawn_bounds = jnp.array(
                [consts.CANNON_SPAWN_MIN_INTERVAL, consts.CANNON_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )

        scroll_offset = jnp.where(state.is_scrolling, consts.PLAYER_MOVE_SPEED, 0)

        # Cannon Movement (acts like a stationary object that scrolls with the environment)
        updated_cannon_x = jnp.where(
            state.cannon_x >= 0,
            state.cannon_x + scroll_offset,
            state.cannon_x
        )

        # Despawn cannon if off-screen (moved past right edge, or too far)
        cannon_active = (updated_cannon_x >= 0) & (updated_cannon_x < consts.WIDTH)
        updated_cannon_x = jnp.where(cannon_active, updated_cannon_x, -1)
        # Cannon y is fixed to spawn_road_top - CANNON_SIZE[1]
        updated_cannon_y = jnp.where(cannon_active, state.cannon_y, -1)

        # Cannon Spawning Logic
        rng_interval, rng_after = jax.random.split(state.rng, 2)

        should_spawn_cannon = (
            (updated_cannon_x < 0)  # No active cannon
            & state.is_scrolling
            & (state.scrolling_step_counter >= state.next_cannon_spawn_step)
            & spawn_cannons_enabled
        )

        spawn_y = spawn_road_top - consts.CANNON_SIZE[1] # Keep the subtraction, as the road_top is technically the top pixel of the road surface. If it spawns too high, we need to adjust the subtraction. Wait, previously it said "it needs to sit right on top of the road". The road is at Y=road_top. Things are drawn from top-left, so Y=road_top-height puts it exactly on top of the road. But maybe it needs to overlap, so let's adjust it by adding CANNON_SIZE[1]//2. Actually, let's just make spawn_y = spawn_road_top - consts.CANNON_SIZE[1] + 3. Or just spawn_road_top - 12. Let's make it spawn_y = spawn_road_top - consts.CANNON_SIZE[1] + 6

        spawn_y = spawn_road_top - consts.CANNON_SIZE[1] + 2

        # Compute next cannon spawn step: use fixed positions when available,
        # otherwise fall back to random interval.
        random_next = state.scrolling_step_counter + jax.random.randint(
            rng_interval, (), cannon_spawn_bounds[0],
            cannon_spawn_bounds[1] + 1, dtype=jnp.int32,
        )
        if self._level_count > 0:
            fixed_steps = self._cannon_fixed_steps[level_idx]
            valid_next = (fixed_steps > state.scrolling_step_counter) & (fixed_steps >= 0)
            fixed_candidates = jnp.where(valid_next, fixed_steps, jnp.int32(_NO_CANNON_STEP))
            fixed_next = jnp.min(fixed_candidates)
            has_fixed = jnp.any(self._cannon_fixed_steps[level_idx] >= 0)
            next_cannon_spawn_step = jnp.where(has_fixed, fixed_next, random_next)
        else:
            next_cannon_spawn_step = random_next

        updated_cannon_x = jnp.where(should_spawn_cannon, jnp.array(0, dtype=jnp.int32), updated_cannon_x)
        updated_cannon_y = jnp.where(should_spawn_cannon, spawn_y, updated_cannon_y)
        next_cannon_spawn_step = jnp.where(should_spawn_cannon, next_cannon_spawn_step, state.next_cannon_spawn_step)

        # Reset has_fired flag if a new cannon spawns
        updated_cannon_has_fired = jnp.where(should_spawn_cannon, jnp.array(False, dtype=jnp.bool_), state.cannon_has_fired)

        # Toggle mirrored state when spawning a new cannon
        updated_cannon_is_mirrored = jnp.where(should_spawn_cannon, ~state.cannon_is_mirrored, state.cannon_is_mirrored)

        # Bullet Movement
        # Normal cannons: bullet moves right at BULLET_SPEED
        # Mirrored cannons: bullet moves left at -BULLET_SPEED * 2.5 (faster to compensate for scroll)
        bullet_velocity = jnp.where(
            state.cannon_is_mirrored,
            jnp.int32(-consts.BULLET_SPEED * 2.5),  # Leftward for mirrored (faster)
            consts.BULLET_SPEED                      # Rightward for normal
        )
        updated_bullet_x = jnp.where(
            state.bullet_x >= 0,
            state.bullet_x + bullet_velocity + scroll_offset,
            state.bullet_x
        )

        bullet_active = (updated_bullet_x >= 0) & (updated_bullet_x < consts.WIDTH)
        updated_bullet_x = jnp.where(bullet_active, updated_bullet_x, -1)
        updated_bullet_y = jnp.where(bullet_active, state.bullet_y, -1)

        # Bullet Spawning Logic
        # A bullet spawns if cannon is active, it hasn't fired yet, and no bullet is currently active.
        cannon_is_active = updated_cannon_x >= 0

        # For mirrored cannons, only shoot after reaching the right half of screen
        cannon_can_shoot = jnp.where(
            updated_cannon_is_mirrored,
            updated_cannon_x > ((consts.WIDTH // 4) * 3),  # Mirrored: wait until past middle (x > WIDTH/2)
            jnp.array(True, dtype=jnp.bool_)  # Normal: can always shoot
        )

        should_spawn_bullet = cannon_is_active & (updated_bullet_x < 0) & ~updated_cannon_has_fired & cannon_can_shoot

        # Bullet spawn position depends on cannon direction
        # Normal: spawn at right side of cannon
        # Mirrored: spawn at left side of cannon
        b_spawn_x = jnp.where(
            updated_cannon_is_mirrored,
            updated_cannon_x,  # Left side for mirrored
            updated_cannon_x + consts.CANNON_SIZE[0]  # Right side for normal
        )
        # Cannon bullet holes usually near middle:
        b_spawn_y = updated_cannon_y + (consts.CANNON_SIZE[1] // 2) - (consts.BULLET_SIZE[1] // 2)

        updated_bullet_x = jnp.where(should_spawn_bullet, b_spawn_x, updated_bullet_x)
        updated_bullet_y = jnp.where(should_spawn_bullet, b_spawn_y, updated_bullet_y)

        updated_cannon_has_fired = updated_cannon_has_fired | should_spawn_bullet

        return state._replace(
            cannon_x=updated_cannon_x,
            cannon_y=updated_cannon_y,
            next_cannon_spawn_step=next_cannon_spawn_step,
            cannon_has_fired=updated_cannon_has_fired,
            cannon_is_mirrored=updated_cannon_is_mirrored,
            bullet_x=updated_bullet_x,
            bullet_y=updated_bullet_y,
            rng=rng_after,
        )

    def _check_bullet_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        active = state.bullet_x >= 0

        # Expand player hitbox vertically so bullets don't pass underneath due to cannon visual offset
        player_hit_y = state.player_y
        hit_height = self.consts.PLAYER_SIZE[1] + 8

        overlap = _check_aabb_collision(
            state.player_x, player_hit_y,
            self.consts.PLAYER_SIZE[0], hit_height,
            state.bullet_x, state.bullet_y,
            self.consts.BULLET_SIZE[0], self.consts.BULLET_SIZE[1],
        )
        collision = active & overlap & (state.death_timer == 0) & jnp.logical_not(state.is_jumping)

        return state._replace(
            death_timer=jnp.where(collision, jnp.array(self.consts.DEATH_ANIMATION_DURATION, dtype=jnp.int32), state.death_timer),
            bullet_x=jnp.where(collision, jnp.array(-1, dtype=jnp.int32), state.bullet_x),
            is_round_over=state.is_round_over | collision,
        )

    def reset(self, key=None) -> Tuple[RoadRunnerObservation, RoadRunnerState]:
        # Initialize RNG key
        if key is None:
            key = jax.random.PRNGKey(42)

        state = RoadRunnerState(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            player_x_history=jnp.array(
                [self.consts.PLAYER_START_X] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            player_y_history=jnp.array(
                [self.consts.PLAYER_START_Y] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            enemy_x=jnp.array(self.consts.ENEMY_X, dtype=jnp.int32),
            enemy_y=jnp.array(self.consts.ENEMY_Y, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            player_is_moving=jnp.array(False, dtype=jnp.bool_),
            player_looks_right=jnp.array(False, dtype=jnp.bool_),
            enemy_is_moving=jnp.array(False, dtype=jnp.bool_),
            enemy_looks_right=jnp.array(False, dtype=jnp.bool_),
            score=jnp.array(0, dtype=jnp.int32),
            is_scrolling=jnp.array(False, dtype=jnp.bool_),
            scrolling_step_counter=jnp.array(0, dtype=jnp.int32),
            is_round_over=jnp.array(False, dtype=jnp.bool_),
            seeds=jnp.full((4, 3), -1, dtype=jnp.int32),  # Initialize all seeds as inactive (-1, -1)
            next_seed_spawn_scroll_step=jnp.array(0, dtype=jnp.int32),
            rng=key,
            seed_pickup_streak=jnp.array(0, dtype=jnp.int32),
            next_seed_id=jnp.array(0, dtype=jnp.int32),
            last_picked_up_seed_id=jnp.array(0, dtype=jnp.int32),
            truck_x=jnp.array(-1, dtype=jnp.int32),
            truck_y=jnp.array(-1, dtype=jnp.int32),
            next_truck_spawn_step=jnp.array(
                self.consts.TRUCK_SPAWN_MIN_INTERVAL, dtype=jnp.int32
            ),
            current_level=jnp.array(0, dtype=jnp.int32),
            level_transition_timer=jnp.array(0, dtype=jnp.int32),
            is_in_transition=jnp.array(False, dtype=jnp.bool_),
            lives=jnp.array(self.consts.STARTING_LIVES, dtype=jnp.int32),
            jump_timer=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
            ravines=jnp.full((3, 2), -1, dtype=jnp.int32),
            next_ravine_spawn_scroll_step=jnp.array(0, dtype=jnp.int32),
            landmine_x=jnp.array(-1, dtype=jnp.int32),
            landmine_y=jnp.array(-1, dtype=jnp.int32),
            next_landmine_spawn_step=jnp.array(
                self.consts.LANDMINE_SPAWN_MIN_INTERVAL, dtype=jnp.int32
            ),
            cannon_x=jnp.array(-1, dtype=jnp.int32),
            cannon_y=jnp.array(-1, dtype=jnp.int32),
            next_cannon_spawn_step=jnp.array(
                self.consts.CANNON_SPAWN_MIN_INTERVAL, dtype=jnp.int32
            ),
            cannon_has_fired=jnp.array(False, dtype=jnp.bool_),
            # Start mirrored=False so the first toggle (~False → True) produces a
            # left-facing cannon, matching the pre-split position in the offramp pattern.
            cannon_is_mirrored=jnp.array(False, dtype=jnp.bool_),
            bullet_x=jnp.array(-1, dtype=jnp.int32),
            bullet_y=jnp.array(-1, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            enemy_speed_phase_start=jnp.array(0, dtype=jnp.int32),
            enemy_flattened_timer=jnp.array(0, dtype=jnp.int32),
            player_on_offramp=jnp.array(False, dtype=jnp.bool_),
            instant_death=jnp.array(False, dtype=jnp.bool_),
            is_falling=jnp.array(False, dtype=jnp.bool_),
            fall_timer=jnp.array(0, dtype=jnp.int32),
            fall_clip_y=jnp.array(0, dtype=jnp.int32),
        )
        state = self._initialize_spawn_timers(state, jnp.array(0, dtype=jnp.int32))
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> Tuple[RoadRunnerObservation, RoadRunnerState, float, bool, RoadRunnerInfo]:
        state = self._handle_level_transition(state)
        operand = (state, action)

        def _gameplay_branch(data):
            st, act = data

            # Compute road bounds once for the entire step
            center_top, center_bottom, center_height = self._get_road_bounds(st)
            spawn_top, spawn_bottom, spawn_height = self._get_road_bounds_at_x(
                st, jnp.array(0, dtype=jnp.int32))
            spawn_top = spawn_top.astype(jnp.int32)
            spawn_bottom = spawn_bottom.astype(jnp.int32)

            st = self._player_step(st, act, center_top, center_bottom)
            st = self._enemy_step(st, center_top, center_bottom)
            st = self._check_game_over(st)
            st = self._update_and_spawn_seeds(st, spawn_top, spawn_bottom)
            st = self._check_seed_collisions(st)
            st = self._update_and_spawn_truck(st, spawn_top, spawn_bottom, spawn_height)
            st = self._check_truck_collisions(st)
            st = self._update_and_spawn_ravines(st, center_top, center_bottom, center_height)
            st = self._check_ravine_collisions(st)
            st = self._update_and_spawn_landmines(st, spawn_top, spawn_bottom)
            st = self._check_landmine_collisions(st)
            # Add cannon update and bullet collision check
            st = self._update_and_spawn_cannon_and_bullets(st, spawn_top)
            st = self._check_bullet_collisions(st)
            st = self._check_level_completion(st)

            reward = (st.score - state.score).astype(jnp.float32)

            player_at_end = st.player_x >= self.consts.WIDTH - self.consts.PLAYER_SIZE[0]
            should_reset = st.instant_death | (st.is_round_over & player_at_end)

            return st, reward, should_reset

        def _fall_timer_branch(data):
            """Handle falling into ravine animation"""
            st, _ = data
            new_timer = jnp.maximum(st.fall_timer - 1, 0)
            timer_expired = new_timer == 0

            # Move player downward during fall animation (4 pixels per frame for faster fall)
            new_player_y = st.player_y + 4

            st = st._replace(
                fall_timer=new_timer,
                player_y=new_player_y,
                instant_death=st.instant_death | timer_expired,
                is_falling=jnp.logical_not(timer_expired) & st.is_falling,
            )
            return st, jnp.float32(0.0), st.instant_death

        def _death_timer_branch(data):
             st, _ = data
             new_timer = jnp.maximum(st.death_timer - 1, 0)
             timer_expired = new_timer == 0
             st = st._replace(
                 death_timer=new_timer,
                 instant_death=st.instant_death | timer_expired,
             )
             return st, jnp.float32(0.0), st.instant_death

        def _transition_branch(data):
            st, _ = data
            return st, jnp.float32(0.0), jnp.array(False, dtype=jnp.bool_)

        # Use switch instead of nested conds: 0=gameplay, 1=fall_timer, 2=death_timer, 3=transition
        branch_idx = jnp.where(
            state.is_in_transition, 3,
            jnp.where(state.death_timer > 0, 2,
                jnp.where(state.fall_timer > 0, 1, 0)
            )
        )
        state, reward, should_reset = jax.lax.switch(
            branch_idx,
            [_gameplay_branch, _fall_timer_branch, _death_timer_branch, _transition_branch],
            operand
        )

        # Handle round end ONCE (instead of duplicated in gameplay + death branches)
        state = jax.lax.cond(
            should_reset, self._handle_round_end, lambda inner: inner, state
        )

        state = state._replace(step_counter=state.step_counter + 1)
        observation = self._get_observation(state)
        info = self._get_info(state)

        return observation, state, reward, False, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: RoadRunnerState, state: RoadRunnerState) -> float:
        diff = state.score - previous_state.score
        # If score decreased (reset), we return 0.0.
        return jax.lax.select(diff < 0, 0.0, diff.astype(jnp.float32))

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: RoadRunnerState) -> bool:
        return state.is_round_over & (state.lives == 0)

    def _handle_round_end(self, state: RoadRunnerState) -> RoadRunnerState:
        """Handle end of round - merged next_life and game_over into one path."""
        is_game_over = state.lives <= 1
        # For game over: reset score, lives, level. For next life: keep them.
        rng, new_key = jax.random.split(state.rng)
        reset_state = state._replace(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            player_x_history=jnp.array(
                [self.consts.PLAYER_START_X] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            player_y_history=jnp.array(
                [self.consts.PLAYER_START_Y] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            enemy_x=jnp.array(self.consts.ENEMY_X, dtype=jnp.int32),
            enemy_y=jnp.array(self.consts.ENEMY_Y, dtype=jnp.int32),
            is_round_over=jnp.array(False, dtype=jnp.bool_),
            seeds=jnp.full((4, 3), -1, dtype=jnp.int32),
            scrolling_step_counter=jnp.array(0, dtype=jnp.int32),
            seed_pickup_streak=jnp.array(0, dtype=jnp.int32),
            last_picked_up_seed_id=jnp.array(0, dtype=jnp.int32),
            next_seed_id=jnp.array(0, dtype=jnp.int32),
            truck_x=jnp.array(-1, dtype=jnp.int32),
            truck_y=jnp.array(-1, dtype=jnp.int32),
            next_seed_spawn_scroll_step=jnp.array(0, dtype=jnp.int32),
            next_truck_spawn_step=jnp.array(0, dtype=jnp.int32),
            jump_timer=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
            instant_death=jnp.array(False, dtype=jnp.bool_),
            is_falling=jnp.array(False, dtype=jnp.bool_),
            fall_timer=jnp.array(0, dtype=jnp.int32),
            fall_clip_y=jnp.array(0, dtype=jnp.int32),
            ravines=jnp.full((3, 2), -1, dtype=jnp.int32),
            landmine_x=jnp.array(-1, dtype=jnp.int32),
            landmine_y=jnp.array(-1, dtype=jnp.int32),
            next_landmine_spawn_step=jnp.array(0, dtype=jnp.int32),
            cannon_x=jnp.array(-1, dtype=jnp.int32),
            cannon_y=jnp.array(-1, dtype=jnp.int32),
            next_cannon_spawn_step=jnp.array(0, dtype=jnp.int32),
            cannon_has_fired=jnp.array(False, dtype=jnp.bool_),
            # Start mirrored=False so first toggle gives left-facing (pre-split pattern)
            cannon_is_mirrored=jnp.array(False, dtype=jnp.bool_),
            bullet_x=jnp.array(-1, dtype=jnp.int32),
            bullet_y=jnp.array(-1, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            enemy_speed_phase_start=jnp.array(0, dtype=jnp.int32),
            enemy_flattened_timer=jnp.array(0, dtype=jnp.int32),
            # Conditionally reset score, lives, level, step_counter, rng
            score=jnp.where(is_game_over, jnp.array(0, dtype=jnp.int32), state.score),
            lives=jnp.where(is_game_over, jnp.array(self.consts.STARTING_LIVES, dtype=jnp.int32), state.lives - 1),
            current_level=jnp.where(is_game_over, jnp.array(0, dtype=jnp.int32), state.current_level),
            step_counter=jnp.where(is_game_over, jnp.array(0, dtype=jnp.int32), state.step_counter),
            player_is_moving=jnp.array(False, dtype=jnp.bool_),
            player_looks_right=jnp.array(False, dtype=jnp.bool_),
            enemy_is_moving=jnp.array(False, dtype=jnp.bool_),
            enemy_looks_right=jnp.array(False, dtype=jnp.bool_),
            is_scrolling=jnp.array(False, dtype=jnp.bool_),
            is_in_transition=jnp.array(False, dtype=jnp.bool_),
            level_transition_timer=jnp.array(0, dtype=jnp.int32),
            rng=jnp.where(is_game_over, new_key, rng),
            next_ravine_spawn_scroll_step=jnp.array(0, dtype=jnp.int32),
            player_on_offramp=jnp.array(False, dtype=jnp.bool_),
        )
        level_idx = self._get_level_index(reset_state)
        return self._initialize_spawn_timers(reset_state, level_idx)

    def _check_level_completion(self, state: RoadRunnerState) -> RoadRunnerState:
        level_idx = self._get_level_index(state)
        target_distance = self._level_scroll_distances[level_idx]
        level_complete = state.scrolling_step_counter >= target_distance
        max_level_index = max(self._level_count - 1, 0)
        has_next_level = state.current_level < max_level_index
        ready_for_transition = (
            level_complete & has_next_level & jnp.logical_not(state.is_in_transition)
        )

        def _start_transition(st: RoadRunnerState) -> RoadRunnerState:
            jax.debug.print(
                "Level {lvl} reached scroll {scroll} → preparing transition",
                lvl=st.current_level + 1,
                scroll=st.scrolling_step_counter,
            )
            return st._replace(
                is_in_transition=jnp.array(True, dtype=jnp.bool_),
                level_transition_timer=jnp.array(
                    self.consts.LEVEL_TRANSITION_DURATION, dtype=jnp.int32
                ),
                current_level=jnp.array(
                    jnp.minimum(st.current_level + 1, max_level_index), dtype=jnp.int32
                ),
                scrolling_step_counter=jnp.array(0, dtype=jnp.int32),
            )

        return jax.lax.cond(ready_for_transition, _start_transition, lambda st: st, state)

    def _handle_level_transition(self, state: RoadRunnerState) -> RoadRunnerState:
        def _no_transition(st: RoadRunnerState) -> RoadRunnerState:
            return st

        def _process_transition(st: RoadRunnerState) -> RoadRunnerState:
            new_timer = jnp.maximum(st.level_transition_timer - 1, 0)
            st = st._replace(level_transition_timer=new_timer)
            transition_complete = new_timer == 0

            def _complete(s: RoadRunnerState) -> RoadRunnerState:
                reset_state = self._reset_level_entities(s)
                level_idx = self._get_level_index(reset_state)
                reset_state = self._initialize_spawn_timers(reset_state, level_idx)
                return reset_state._replace(
                    is_in_transition=jnp.array(False, dtype=jnp.bool_),
                    level_transition_timer=jnp.array(0, dtype=jnp.int32),
                )

            return jax.lax.cond(transition_complete, _complete, lambda s: s, st)

        return jax.lax.cond(
            state.is_in_transition, _process_transition, _no_transition, state
        )

    def _reset_level_entities(self, state: RoadRunnerState) -> RoadRunnerState:
        history_x = jnp.full(
            (self.consts.ENEMY_REACTION_DELAY,),
            self.consts.PLAYER_START_X,
            dtype=jnp.int32,
        )
        history_y = jnp.full(
            (self.consts.ENEMY_REACTION_DELAY,),
            self.consts.PLAYER_START_Y,
            dtype=jnp.int32,
        )
        cleared_seeds = jnp.full_like(state.seeds, -1)
        return state._replace(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            player_x_history=history_x,
            player_y_history=history_y,
            player_is_moving=jnp.array(False, dtype=jnp.bool_),
            player_looks_right=jnp.array(False, dtype=jnp.bool_),
            enemy_x=jnp.array(self.consts.ENEMY_X, dtype=jnp.int32),
            enemy_y=jnp.array(self.consts.ENEMY_Y, dtype=jnp.int32),
            enemy_is_moving=jnp.array(False, dtype=jnp.bool_),
            enemy_looks_right=jnp.array(False, dtype=jnp.bool_),
            seeds=cleared_seeds,
            seed_pickup_streak=jnp.array(0, dtype=jnp.int32),
            last_picked_up_seed_id=jnp.array(0, dtype=jnp.int32),
            next_seed_id=jnp.array(0, dtype=jnp.int32),
            truck_x=jnp.array(-1, dtype=jnp.int32),
            truck_y=jnp.array(-1, dtype=jnp.int32),
            is_round_over=jnp.array(False, dtype=jnp.bool_),
            is_scrolling=jnp.array(False, dtype=jnp.bool_),
            jump_timer=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
            ravines=jnp.full((3, 2), -1, dtype=jnp.int32),
            next_ravine_spawn_scroll_step=jnp.array(0, dtype=jnp.int32),
            instant_death=jnp.array(False, dtype=jnp.bool_),
            is_falling=jnp.array(False, dtype=jnp.bool_),
            fall_timer=jnp.array(0, dtype=jnp.int32),
            fall_clip_y=jnp.array(0, dtype=jnp.int32),
            enemy_speed_phase_start=jnp.array(0, dtype=jnp.int32),
            enemy_flattened_timer=jnp.array(0, dtype=jnp.int32),
            player_on_offramp=jnp.array(False, dtype=jnp.bool_),
        )

    def _get_level_index(self, state: RoadRunnerState) -> jnp.ndarray:
        if self._level_count == 0:
            return jnp.array(0, dtype=jnp.int32)
        max_index = self._level_count - 1
        return jnp.clip(state.current_level, 0, max_index).astype(jnp.int32)

    def _get_current_level_config(self, state: RoadRunnerState) -> LevelConfig:
        if not self.consts.levels:
            return RoadRunner_Level_1
        max_index = len(self.consts.levels) - 1
        level_idx = jnp.clip(state.current_level, 0, max_index).astype(jnp.int32)
        branches = tuple(lambda cfg=cfg: cfg for cfg in self.consts.levels)
        return jax.lax.switch(level_idx, branches)

    def _initialize_spawn_timers(
        self, state: RoadRunnerState, level_idx: jnp.ndarray
    ) -> RoadRunnerState:
        if self._level_count > 0:
            seed_bounds = self._seed_spawn_intervals[level_idx]
            truck_bounds = self._truck_spawn_intervals[level_idx]
            ravine_bounds = self._ravine_spawn_intervals[level_idx]
            landmine_bounds = self._landmine_spawn_intervals[level_idx]
            cannon_bounds = self._cannon_spawn_intervals[level_idx]
        else:
            seed_bounds = jnp.array(
                [self.consts.SEED_SPAWN_MIN_INTERVAL, self.consts.SEED_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )
            truck_bounds = jnp.array(
                [self.consts.TRUCK_SPAWN_MIN_INTERVAL, self.consts.TRUCK_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )
            ravine_bounds = jnp.array(
                [
                    self.consts.RAVINE_SPAWN_MIN_INTERVAL,
                    self.consts.RAVINE_SPAWN_MAX_INTERVAL,
                ],
                dtype=jnp.int32,
            )
            landmine_bounds = jnp.array(
                [self.consts.LANDMINE_SPAWN_MIN_INTERVAL, self.consts.LANDMINE_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )
            cannon_bounds = jnp.array(
                [self.consts.CANNON_SPAWN_MIN_INTERVAL, self.consts.CANNON_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )

        rng, seed_key = jax.random.split(state.rng)
        next_seed_spawn_scroll_step = state.scrolling_step_counter + jax.random.randint(
            seed_key,
            (),
            seed_bounds[0],
            seed_bounds[1] + 1,
            dtype=jnp.int32,
        )
        rng, truck_key = jax.random.split(rng)
        next_truck_spawn_step = state.step_counter + jax.random.randint(
            truck_key,
            (),
            truck_bounds[0],
            truck_bounds[1] + 1,
            dtype=jnp.int32,
        )
        rng, ravine_key = jax.random.split(rng)
        next_ravine_spawn_scroll_step = state.scrolling_step_counter + jax.random.randint(
            ravine_key,
            (),
            ravine_bounds[0],
            ravine_bounds[1] + 1,
            dtype=jnp.int32,
        )
        rng, landmine_key = jax.random.split(rng)
        next_landmine_spawn_step = state.step_counter + jax.random.randint(
            landmine_key,
            (),
            landmine_bounds[0],
            landmine_bounds[1] + 1,
            dtype=jnp.int32,
        )
        rng, cannon_key = jax.random.split(rng)
        random_cannon_step = state.scrolling_step_counter + jax.random.randint(
            cannon_key,
            (),
            cannon_bounds[0],
            cannon_bounds[1] + 1,
            dtype=jnp.int32,
        )
        # Use fixed cannon positions when available for the level.
        if self._level_count > 0:
            fixed_steps = self._cannon_fixed_steps[level_idx]
            valid = (fixed_steps >= state.scrolling_step_counter) & (fixed_steps >= 0)
            candidates = jnp.where(valid, fixed_steps, jnp.int32(_NO_CANNON_STEP))
            fixed_first = jnp.min(candidates)
            has_fixed = jnp.any(fixed_steps >= 0)
            next_cannon_spawn_step = jnp.where(has_fixed, fixed_first, random_cannon_step)
        else:
            next_cannon_spawn_step = random_cannon_step
        return state._replace(
            rng=rng,
            next_seed_spawn_scroll_step=next_seed_spawn_scroll_step,
            next_truck_spawn_step=next_truck_spawn_step,
            next_ravine_spawn_scroll_step=next_ravine_spawn_scroll_step,
            next_landmine_spawn_step=next_landmine_spawn_step,
            next_cannon_spawn_step=next_cannon_spawn_step,
        )

    def _get_current_road_section(self, state: RoadRunnerState) -> RoadSectionConfig:
        return _get_road_section_for_scroll(
            state,
            self._level_count,
            self._max_road_sections,
            self._road_section_data,
            self._road_section_counts,
            self.consts,
        )

    def _get_road_bounds(self, state: RoadRunnerState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        section = self._get_current_road_section(state)

        # Check if dynamic road heights are enabled for this level
        level_idx = self._get_level_index(state)

        if self._level_count > 0:
            dynamic_enabled = self._dynamic_road_enabled[level_idx]
            heights = self._dynamic_road_heights[level_idx]
            interval = self._dynamic_road_intervals[level_idx]
            trans_len = self._dynamic_road_transition_lengths[level_idx]
            scroll_offset = self._dynamic_road_scroll_offsets[level_idx]

            # Get dynamic height if enabled
            # Use scroll_pos * PLAYER_MOVE_SPEED to match rendering speed
            # Add road width offset to match rendering inversion
            # Rendering uses: world_scroll + (road_width - 1 - col_idx)
            # For collision, use the center of the road (player position)
            static_road_width = self.consts.WIDTH - 2 * self.consts.SIDE_MARGIN
            player_road_offset = static_road_width // 2  # Player is roughly at center
            collision_world_x = (
                state.scrolling_step_counter * self.consts.PLAYER_MOVE_SPEED
                + (static_road_width - 1 - player_road_offset)
                + scroll_offset
            )
            dynamic_height, _, _ = _get_dynamic_road_height(
                collision_world_x,
                heights[0],
                heights[1],
                interval,
                trans_len,
            )

            # Use dynamic height if enabled, otherwise use section height
            base_road_height = jnp.where(dynamic_enabled, dynamic_height, section.road_height)
        else:
            base_road_height = section.road_height

        road_height = jnp.clip(base_road_height, 1, self.consts.ROAD_HEIGHT)

        # Center the road vertically within the road area when height changes
        # This keeps the road centered as height changes, similar to _centered_top
        height_diff = self.consts.ROAD_HEIGHT - road_height
        section_top = height_diff // 2

        road_top = self.consts.ROAD_TOP_Y + section_top
        road_bottom = road_top + road_height
        return road_top, road_bottom, road_height

    def _get_road_bounds_at_x(
        self, state: RoadRunnerState, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get road bounds at a specific screen X coordinate.
        Useful for checking spawn validity (e.g., at x=0).
        """
        section = self._get_current_road_section(state)

        # Check if dynamic road heights are enabled for this level
        level_idx = self._get_level_index(state)

        if self._level_count > 0:
            dynamic_enabled = self._dynamic_road_enabled[level_idx]
            heights = self._dynamic_road_heights[level_idx]
            interval = self._dynamic_road_intervals[level_idx]
            trans_len = self._dynamic_road_transition_lengths[level_idx]
            scroll_offset = self._dynamic_road_scroll_offsets[level_idx]

            # Map screen X to world X conceptually used for road generation
            # Rendering: world_scroll + (road_width - 1 - col_idx)
            # col_idx corresponds to x.
            static_road_width = self.consts.WIDTH - 2 * self.consts.SIDE_MARGIN

            # Note: Spawning usually happens at x=0 (left side).
            # If x=0, world_x is larger (further ahead in scroll).
            # world_x = scroll + width - 1 - x

            world_x = (
                state.scrolling_step_counter * self.consts.PLAYER_MOVE_SPEED
                + (static_road_width - 1 - x)
                + scroll_offset
            )

            dynamic_height, _, _ = _get_dynamic_road_height(
                world_x,
                heights[0],
                heights[1],
                interval,
                trans_len,
            )

            # Use dynamic height if enabled, otherwise use section height
            base_road_height = jnp.where(dynamic_enabled, dynamic_height, section.road_height)
        else:
            base_road_height = section.road_height

        road_height = jnp.clip(base_road_height, 1, self.consts.ROAD_HEIGHT)

        # Center the road vertically
        height_diff = self.consts.ROAD_HEIGHT - road_height
        section_top = height_diff // 2

        road_top = self.consts.ROAD_TOP_Y + section_top
        road_bottom = road_top + road_height
        return road_top, road_bottom, road_height


    def render(self, state: RoadRunnerState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: RoadRunnerState) -> RoadRunnerObservation:
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )
        enemy = EntityPosition(
            x=state.enemy_x,
            y=state.enemy_y,
            width=jnp.array(self.consts.ENEMY_SIZE[0]),
            height=jnp.array(self.consts.ENEMY_SIZE[1]),
        )

        # Valid ravines have x >= 0 (strictly active in our logic, although we set to -1 when inactive)
        active_ravines_mask = state.ravines[:, 0] >= 0

        # Let's find the ravine with smallest x >= 0
        ravine_x = state.ravines[:, 0]
        ravine_y = state.ravines[:, 1]
        
        # Mask out inactive ones with a large value
        masked_x = jnp.where(active_ravines_mask, ravine_x, self.consts.WIDTH * 2)
        idx = jnp.argmin(masked_x)
        
        nearest_ravine_x = ravine_x[idx]
        nearest_ravine_y = ravine_y[idx]
        
        is_active = active_ravines_mask[idx]
        
        ravine_obs = EntityPosition(
            x=jnp.where(is_active, nearest_ravine_x, jnp.array(0, dtype=jnp.int32)),
            y=jnp.where(is_active, nearest_ravine_y, jnp.array(0, dtype=jnp.int32)),
            width=jnp.where(is_active, jnp.array(self.consts.RAVINE_SIZE[0]), jnp.array(0)),
            height=jnp.where(is_active, jnp.array(self.consts.RAVINE_SIZE[1]), jnp.array(0)),
        )

        return RoadRunnerObservation(
             player=player, 
             enemy=enemy, 
             score=state.score, 
             ravine=ravine_obs
        )

    def _get_info(self, state: RoadRunnerState) -> RoadRunnerInfo:
        return RoadRunnerInfo(
            score=state.score,
            lives=state.lives,
            step=state.step_counter
        )

    def obs_to_flat_array(self, obs: RoadRunnerObservation) -> jnp.ndarray:
        """Convert observation to a flat array."""
        player_arr = jnp.array([obs.player.x, obs.player.y, obs.player.width, obs.player.height])
        enemy_arr = jnp.array([obs.enemy.x, obs.enemy.y, obs.enemy.width, obs.enemy.height])
        ravine_arr = jnp.array([obs.ravine.x, obs.ravine.y, obs.ravine.width, obs.ravine.height])
        score_arr = jnp.array([obs.score])

        # Flatten and concatenate
        return jnp.concatenate([
            player_arr.reshape(-1),
            enemy_arr.reshape(-1),
            ravine_arr.reshape(-1),
            score_arr.reshape(-1)
        ]).astype(jnp.int32)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        # Simplified observation space
        return spaces.Dict(
            {
                "player": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "y": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                        "width": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "height": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "enemy": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "y": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                        "width": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "height": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "score": spaces.Box(
                    low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32
                ),
                "ravine": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "y": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                        "width": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "height": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )


# --- Renderer Class (Simplified) ---
class RoadRunnerRenderer(JAXGameRenderer):
    def __init__(self, consts: RoadRunnerConstants = None, config: render_utils.RendererConfig = None):
        super().__init__()
        self.consts = consts or RoadRunnerConstants()
        self.deco_id_to_sprite = {
            0: "cactus",
            1: "sign_this_way",
            2: "sign_birdseed",
            3: "sign_cars_ahead",
            4: "sign_exit",
            5: "tumbleweed",
            6: "sign_acme_mines",
            7: "sign_steel_shot",
        }

        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        road_sprite = self._create_road_sprite(stripes=True)
        road_no_stripes_sprite = self._create_road_sprite(stripes=False)
        life_sprite = self._create_life_sprite()
        offramp_road_sprite = self._create_offramp_road_sprite()
        offramp_bridge_sprite = self._create_offramp_bridge_sprite()
        asset_config = self._get_asset_config(
            road_sprite, road_no_stripes_sprite, life_sprite, offramp_road_sprite,
            offramp_bridge_sprite,
        )
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/roadrunner"

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        self._level_count = len(self.consts.levels)
        (
            self._max_road_sections,
            self._road_section_data,
            self._road_section_counts,
        ) = _build_road_section_arrays(self.consts.levels, self.consts)
        # Build offramp data for the renderer
        self._offramp_data = _build_offramp_arrays(self.consts.levels)

        # Build dynamic road height config arrays for renderer
        levels = self.consts.levels
        (
            self._dynamic_road_enabled,
            self._dynamic_road_heights,
            self._dynamic_road_intervals,
            self._dynamic_road_transition_lengths,
            self._dynamic_road_scroll_offsets
        ) = _build_dynamic_road_config_arrays(levels)

        # Pre-calculate unique road dimensions for rendering optimization
        unique_heights_list, unique_widths_list = self._get_unique_road_dims()
        self._unique_heights_list = unique_heights_list
        self._unique_widths_list = unique_widths_list
        # Convert to JAX arrays for runtime matching
        self._unique_heights_arr = jnp.array(unique_heights_list, dtype=jnp.int32)
        self._unique_widths_arr = jnp.array(unique_widths_list, dtype=jnp.int32)

        # Pre-build decoration data arrays for vectorized rendering
        self._deco_sprites, self._deco_data, self._num_decos = self._build_decoration_arrays()

    def _build_decoration_arrays(self):
        """Pre-compute flat arrays for all decorations across all levels."""
        levels = self.consts.levels

        # Collect all (level_idx, d_x, d_y, d_slowdown, d_type) tuples
        all_decos = []
        for level_idx, level_cfg in enumerate(levels):
            for deco in level_cfg.decorations:
                d_x, d_y, d_slowdown, d_type = deco
                all_decos.append((level_idx, d_x, d_y, d_slowdown, d_type))

        num_decos = len(all_decos)
        if num_decos == 0:
            # No decorations at all - create dummy arrays
            dummy_sprite = jnp.zeros((1, 1, 1), dtype=jnp.uint8)
            dummy_data = jnp.zeros((1, 4), dtype=jnp.int32)
            return dummy_sprite, dummy_data, 0

        # Get unique decoration sprite names and find max dimensions
        sprite_names = [self.deco_id_to_sprite[d_type] for _, _, _, _, d_type in all_decos]
        sprites = [self.SHAPE_MASKS[name] for name in sprite_names]
        max_h = max(s.shape[0] for s in sprites)
        max_w = max(s.shape[1] for s in sprites)

        # Pad all sprites to (max_h, max_w) with TRANSPARENT_ID and stack
        padded = []
        transparent_id = self.jr.TRANSPARENT_ID
        for s in sprites:
            pad_h = max_h - s.shape[0]
            pad_w = max_w - s.shape[1]
            if pad_h > 0 or pad_w > 0:
                p = jnp.pad(s, ((0, pad_h), (0, pad_w)),
                            constant_values=transparent_id)
            else:
                p = s
            padded.append(p)

        # stacked_sprites: (num_decos, max_h, max_w)
        stacked_sprites = jnp.stack(padded, axis=0)

        # deco_data: (num_decos, 4) = [level_idx, d_x, d_y, d_slowdown]
        deco_data = jnp.array(
            [(level_idx, d_x, d_y, d_slowdown) for level_idx, d_x, d_y, d_slowdown, _ in all_decos],
            dtype=jnp.int32,
        )

        return stacked_sprites, deco_data, num_decos

    def _create_road_sprite(self, stripes: bool = True) -> jnp.ndarray:
        ROAD_HEIGHT = self.consts.ROAD_HEIGHT
        WIDTH = self.consts.WIDTH
        DASH_LENGTH = self.consts.ROAD_DASH_LENGTH
        GAP_HEIGHT = self.consts.ROAD_GAP_HEIGHT
        PATTERN_WIDTH = self.consts.ROAD_PATTERN_WIDTH

        # Create a wider road for scrolling
        SCROLL_WIDTH = WIDTH + PATTERN_WIDTH

        road_color_rgba = jnp.array([0, 0, 0, 255], dtype=jnp.uint8)
        marking_color_rgba = jnp.array([255, 255, 255, 255], dtype=jnp.uint8)

        # Create a coordinate grid for the wider sprite
        y, x = jnp.indices((ROAD_HEIGHT, SCROLL_WIDTH))

        # Define the pattern using modular arithmetic
        is_marking_col = (x % PATTERN_WIDTH) >= (3 * DASH_LENGTH)
        is_marking_row = (y % (GAP_HEIGHT + 1)) == GAP_HEIGHT
        is_not_last_row = y < (ROAD_HEIGHT - 1)
        is_marking = is_marking_col & is_marking_row & is_not_last_row

        # Use jnp.where to create the sprite from the pattern
        if stripes:
             road_sprite = jnp.where(
                is_marking[:, :, jnp.newaxis],
                marking_color_rgba,
                road_color_rgba,
            )
        else:
             road_sprite = jnp.tile(road_color_rgba, (ROAD_HEIGHT, SCROLL_WIDTH, 1))

        return road_sprite

    def _create_offramp_road_sprite(self) -> jnp.ndarray:
        """Create the offramp road sprite: solid black, no lane markings."""
        H = self.consts.OFFRAMP_HEIGHT
        WIDTH = self.consts.WIDTH
        PATTERN_WIDTH = self.consts.ROAD_PATTERN_WIDTH
        SCROLL_WIDTH = WIDTH + PATTERN_WIDTH

        road_color_rgba = jnp.array([0, 0, 0, 255], dtype=jnp.uint8)
        return jnp.broadcast_to(road_color_rgba, (H, SCROLL_WIDTH, 4)).copy()

    def _create_offramp_split_sprite(self) -> jnp.ndarray:
        """Create the diagonal transition sprite for the split/merge connecting sections.

        The sprite covers the vertical span from offramp_top to the main road top
        (OFFRAMP_HEIGHT + OFFRAMP_GAP rows) and is OFFRAMP_RAMP_WIDTH columns wide.

        Column 0  (left  = fully separated): only the offramp rows are road.
        Column W-1 (right = still merged):   the entire sprite height is road.

        The formula fills from the diagonal line DOWN to the bottom of the sprite,
        ensuring the full bottom row is always road (flush connection to the main road
        at every column).  The diagonal runs through the gap area from top-right to
        bottom-left, making the connecting section look like a proper ramp rather
        than a triangle with a single point of contact.
        """
        RAMP_W = self.consts.OFFRAMP_RAMP_WIDTH
        OFFRAMP_H = self.consts.OFFRAMP_HEIGHT
        GAP_H = self.consts.OFFRAMP_GAP
        total_h = OFFRAMP_H + GAP_H

        road_color_rgba = jnp.array([0, 0, 0, 255], dtype=jnp.uint8)
        # Use alpha=0 for non-road pixels: the rendering pipeline treats pixels with
        # alpha <= 128 as TRANSPARENT_ID, so they show the background through naturally.
        bg_color_rgba = jnp.array([0, 0, 0, 0], dtype=jnp.uint8)

        y, x = jnp.indices((total_h, RAMP_W))
        # The diagonal line in the gap area runs from (x=RAMP_W-1, y=OFFRAMP_H) on the
        # right (merged) to (x=0, y=total_h-1) on the left (separated bottom).
        # Road fills:
        #   • always the offramp band (y < OFFRAMP_H)
        #   • the gap region BELOW the diagonal (y >= diagonal_y(x))
        # This guarantees the bottom row (y=total_h-1) is road at every column,
        # creating a full-width flush connection to the main road.
        # Using RAMP_W (not RAMP_W-1) as denominator ensures diagonal_y(x=0) = total_h-1,
        # so the leftmost column's bottom row is road; RAMP_W-1 would give total_h, missing it.
        safe_denom = jnp.maximum(RAMP_W, 1)
        diagonal_y = OFFRAMP_H + GAP_H * (RAMP_W - 1 - x) // safe_denom
        is_road = (y < OFFRAMP_H) | (y >= diagonal_y)
        return jnp.where(is_road[:, :, jnp.newaxis], road_color_rgba, bg_color_rgba)

    def _create_offramp_bridge_sprite(self) -> jnp.ndarray:
        """Create the bridge sprite: a solid black vertical strip filling the gap (median).

        The bridge connects the bottom of the offramp road to the top of the main road,
        spanning exactly OFFRAMP_GAP rows and OFFRAMP_BRIDGE_WIDTH columns.
        """
        GAP_H = self.consts.OFFRAMP_GAP
        BRIDGE_W = self.consts.OFFRAMP_BRIDGE_WIDTH
        road_color_rgba = jnp.array([0, 0, 0, 255], dtype=jnp.uint8)
        return jnp.broadcast_to(road_color_rgba, (GAP_H, BRIDGE_W, 4)).copy()

    def _create_seed_sprite(self) -> jnp.ndarray:
        seed_color_rgba = (0, 0, 255, 255)
        seed_shape = (self.consts.SEED_SIZE[0], self.consts.SEED_SIZE[1], 4)
        return jnp.tile(
            jnp.array(seed_color_rgba, dtype=jnp.uint8), (*seed_shape[:2], 1)
        )

    def _create_life_sprite(self) -> jnp.ndarray:
        # Green square for lives
        life_color_rgba = (*self.consts.PLAYER_COLOR, 255)
        life_shape = (6, 6, 4) # 6x6 square
        return jnp.tile(
            jnp.array(life_color_rgba, dtype=jnp.uint8), (*life_shape[:2], 1)
        )

    def _get_render_section(self, state: RoadRunnerState) -> RoadSectionConfig:
        return _get_road_section_for_scroll(
            state,
            self._level_count,
            self._max_road_sections,
            self._road_section_data,
            self._road_section_counts,
            self.consts,
        )

    def _get_asset_config(
        self,
        road_sprite: jnp.ndarray,
        road_no_stripes_sprite: jnp.ndarray,
        life_sprite: jnp.ndarray,
        offramp_road_sprite: jnp.ndarray,
        offramp_bridge_sprite: jnp.ndarray,
    ) -> list:
        asset_config = [
            {"name": "background", "type": "background", "file": "background.npy"},
            {"name": "player", "type": "single", "file": "roadrunner_stand.npy"},
            {"name": "player_run1", "type": "single", "file": "roadrunner_run1.npy"},
            {"name": "player_run2", "type": "single", "file": "roadrunner_run2.npy"},
            {"name": "player_jump", "type": "single", "file": "roadrunner_jump.npy"},
            {"name": "enemy", "type": "single", "file": "enemy_stand.npy"},
            {"name": "enemy_run1", "type": "single", "file": "enemy_run1.npy"},
            {"name": "enemy_run2", "type": "single", "file": "enemy_run2.npy"},
            {"name": "enemy_run_over", "type": "single", "file": "enemy_run_over.npy"},
            {"name": "road", "type": "procedural", "data": road_sprite},
            {"name": "road_no_stripes", "type": "procedural", "data": road_no_stripes_sprite},
            {"name": "score_digits", "type": "digits", "pattern": "score_{}.npy"},
            {"name": "score_blank", "type": "single", "file": "score_10.npy"},
            {"name": "seed", "type": "single", "file": "birdseed.npy"},
            {"name": "truck", "type": "single", "file": "truck.npy"},
            {"name": "life", "type": "single", "file": "lives.npy"},
            {"name": "ravine", "type": "single", "file": "ravine.npy"},
            {"name": "landmine", "type": "single", "file": "landmine.npy"},
            {"name": "player_burnt", "type": "single", "file": "roadrunner_burnt.npy"},
            {"name": "end_of_level_1", "type": "single", "file": "end_of_level_1.npy"},
            {"name": "cactus", "type": "single", "file": "cactus.npy"},
            {"name": "tumbleweed", "type": "single", "file": "tumbleweed.npy"},
            {"name": "sign_this_way", "type": "single", "file": "sign_this_way.npy"},
            {"name": "sign_birdseed", "type": "single", "file": "sign_birdseed.npy"},
            {"name": "sign_cars_ahead", "type": "single", "file": "sign_cars_ahead.npy"},
            {"name": "sign_exit", "type": "single", "file": "sign_exit.npy"},
            {"name": "sign_acme_mines", "type": "single", "file": "sign_acme_mines.npy"},
            {"name": "sign_steel_shot", "type": "single", "file": "sign_steel_shot.npy"},
            {"name": "canon", "type": "single", "file": "canon.npy"},
            {"name": "bullet", "type": "single", "file": "bullet.npy"},
            # Offramp sprites
            {"name": "offramp_road", "type": "procedural", "data": offramp_road_sprite},
            {"name": "offramp_split", "type": "single", "file": "offramp_split.npy"},
            {"name": "offramp_merge", "type": "single", "file": "offramp_merge.npy"},
            {"name": "offramp_bridge", "type": "procedural", "data": offramp_bridge_sprite},
        ]

        return asset_config

    def _render_score(self, canvas: jnp.ndarray, score: jnp.ndarray) -> jnp.ndarray:
        MIN_DIGITS = 2
        MAX_DIGITS = 6

        safe_score = jnp.where(score == 0, 1, score)
        score_digit_count = (jnp.floor(jnp.log10(safe_score)) + 1).astype(int)
        visible_count = jnp.clip(score_digit_count, MIN_DIGITS, MAX_DIGITS)

        raw_digits = self.jr.int_to_digits(score, max_digits=MAX_DIGITS)

        digit_indices = jnp.arange(MAX_DIGITS)
        cutoff_index = MAX_DIGITS - visible_count
        is_visible_mask = digit_indices >= cutoff_index

        EMPTY_INDEX = 10
        score_digits = jnp.where(is_visible_mask, raw_digits, EMPTY_INDEX)

        original_masks = self.SHAPE_MASKS["score_digits"]
        blank_sprite = self.SHAPE_MASKS["score_blank"]
        score_digit_masks = jnp.concatenate([original_masks, blank_sprite[None, ...]], axis=0)

        score_x = (self.consts.WIDTH // 2 - (MAX_DIGITS * 6) // 2)
        score_y = 4

        canvas = self.jr.render_label_selective(
            canvas,
            score_x,
            score_y,
            score_digits,
            score_digit_masks,
            0,
            MAX_DIGITS,
            spacing=8, # offset, so width of digit + space
            max_digits_to_render=MAX_DIGITS,
        )
        return canvas

    def _render_lives(self, canvas: jnp.ndarray, lives: jnp.ndarray) -> jnp.ndarray:
        num_squares = jnp.maximum(lives - 1, 0)

        start_y = 24
        square_size = 6
        spacing = 2

        start_x = (self.consts.WIDTH // 3) + 2

        def render_square(i, r):
            x = start_x + i * (square_size + spacing)
            return self.jr.render_at(r, x, start_y, self.SHAPE_MASKS["life"])

        return jax.lax.fori_loop(0, num_squares, render_square, canvas)

    def _render_seeds(self, canvas: jnp.ndarray, seeds: jnp.ndarray) -> jnp.ndarray:
        # Only render active seeds (x >= 0)
        def render_seed(i, c):
            seed_x = seeds[i, 0]
            seed_y = seeds[i, 1]
            # Only render if seed is active (x >= 0)
            return jax.lax.cond(
                seed_x >= 0,
                lambda can: self.jr.render_at(
                    can, seed_x, seed_y, self.SHAPE_MASKS["seed"]
                ),
                lambda can: can,
                c,
            )

        # Use fori_loop to render all seeds
        return jax.lax.fori_loop(0, seeds.shape[0], render_seed, canvas)

    def _render_truck(self, canvas: jnp.ndarray, truck_x: chex.Array, truck_y: chex.Array) -> jnp.ndarray:
        # Only render if truck is active (x >= 0)
        return jax.lax.cond(
            truck_x >= 0,
            lambda can: self.jr.render_at(can, truck_x, truck_y, self.SHAPE_MASKS["truck"]),
            lambda can: can,
            canvas,
        )

    def _render_decorations(self, canvas: jnp.ndarray, state: RoadRunnerState) -> jnp.ndarray:
        if self._num_decos == 0:
            return canvas

        scroll_base = state.scrolling_step_counter * self.consts.PLAYER_MOVE_SPEED

        def render_one_deco(i, c):
            level_idx = self._deco_data[i, 0]
            d_x = self._deco_data[i, 1]
            d_y = self._deco_data[i, 2]
            d_slowdown = self._deco_data[i, 3]
            sprite = self._deco_sprites[i]

            screen_x = scroll_base / (2 * d_slowdown) - d_x
            is_active_level = state.current_level == level_idx
            is_visible = (screen_x > 0) & (screen_x < self.consts.WIDTH - 16)
            should_render = is_active_level & is_visible

            return jax.lax.cond(
                should_render,
                lambda can: self.jr.render_at(can, screen_x, d_y, sprite),
                lambda can: can,
                c,
            )

        return jax.lax.fori_loop(0, self._num_decos, render_one_deco, canvas)
    
    def _render_ravines(self, canvas: jnp.ndarray, ravines: jnp.ndarray) -> jnp.ndarray:
        # Only render active ravines (x >= 0)
        def render_ravine(i, c):
            r_x = ravines[i, 0]
            r_y = ravines[i, 1]
            # Only render if active
            return jax.lax.cond(
                r_x >= 0,
                lambda can: self.jr.render_at(
                    can, r_x, r_y, self.SHAPE_MASKS["ravine"]
                ),
                lambda can: can,
                c,
            )
        
        return jax.lax.fori_loop(0, ravines.shape[0], render_ravine, canvas)

    def _render_landmine(self, canvas: jnp.ndarray, landmine_x: chex.Array, landmine_y: chex.Array) -> jnp.ndarray:
        # Only render if active
        return jax.lax.cond(
            landmine_x >= 0,
            lambda can: self.jr.render_at(
                can, landmine_x, landmine_y, self.SHAPE_MASKS["landmine"]
            ),
            lambda can: can,
            canvas,
        )

    def _render_offramp(
        self, canvas: jnp.ndarray, state: RoadRunnerState, road_offset: jnp.ndarray
    ) -> jnp.ndarray:
        """Render the offramp road band and its split/merge transition sprites."""
        if self._level_count == 0:
            return canvas

        any_active, row = _find_active_offramp_row(
            self._offramp_data, state, self._level_count, self.consts)
        scroll_start = row[1]
        scroll_end = row[2]

        counter = state.scrolling_step_counter
        SPEED = self.consts.PLAYER_MOVE_SPEED
        RAMP_W = self.consts.OFFRAMP_RAMP_WIDTH
        OFFRAMP_H = self.consts.OFFRAMP_HEIGHT
        W = self.consts.WIDTH
        MARGIN = self.consts.SIDE_MARGIN
        PATTERN_WIDTH = self.consts.ROAD_PATTERN_WIDTH
        SCROLL_W = W + PATTERN_WIDTH  # unscaled total sprite width

        # Compute scaled dimensions from the actual (possibly downscaled) sprite.
        # road_mask is already in scaled space; all dynamic_slice sizes must match it.
        offramp_sprite = self.SHAPE_MASKS["offramp_road"]
        ofr_sprite_h = offramp_sprite.shape[0]  # Python int — static for dynamic_slice
        ofr_sprite_w = offramp_sprite.shape[1]
        # Width of the visible portion of the sprite (covers the canvas width)
        ofr_slice_w = ofr_sprite_w - int(round(PATTERN_WIDTH * ofr_sprite_w / SCROLL_W))
        # Scale the dynamic road_offset into sprite column space
        scaled_road_offset = (road_offset * ofr_sprite_w // SCROLL_W).astype(jnp.int32)
        # Scale x coordinates for column masking
        x_scale = ofr_sprite_w / SCROLL_W  # float, used for left_x / right_x scaling

        # Active status already computed by _get_active_offramp_row
        offramp_active = any_active

        # Compute Y position of the offramp road, flush above the main road.
        # For narrow roads (e.g. Level 4, road_height=30), the road is centered
        # within the ROAD_HEIGHT area, so the visible road top is offset down from
        # ROAD_TOP_Y.  The offramp must be positioned relative to this actual top.
        section = self._get_render_section(state)
        current_road_height = jnp.clip(section.road_height, 1, self.consts.ROAD_HEIGHT)
        road_top_offset = (self.consts.ROAD_HEIGHT - current_road_height) // 2
        actual_road_top = self.consts.ROAD_TOP_Y + road_top_offset
        offramp_top = actual_road_top - self.consts.OFFRAMP_GAP - OFFRAMP_H

        def _render_active(c: jnp.ndarray) -> jnp.ndarray:
            # Screen x of split leading edge (0 at scroll_start, grows rightward)
            split_x = (counter - scroll_start) * SPEED
            # Screen x of merge leading edge (0 at scroll_end, grows rightward)
            merge_x = (counter - scroll_end) * SPEED

            # Offramp road band: visible between the merge and the split leading edges
            left_x = jnp.maximum(MARGIN, merge_x)
            right_x = jnp.minimum(W - MARGIN, split_x)

            # Slice the scrolling offramp road pattern using scaled dimensions
            offramp_road = jax.lax.dynamic_slice(
                offramp_sprite,
                (0, scaled_road_offset),
                (ofr_sprite_h, ofr_slice_w),
            )

            x_coords = jnp.arange(ofr_slice_w)
            # Scale left_x / right_x into sprite column space for the column mask
            col_visible = (x_coords >= left_x * x_scale) & (x_coords < right_x * x_scale)
            # Use TRANSPARENT_ID for masked-out columns so they show the background through.
            col_mask = jnp.broadcast_to(col_visible[jnp.newaxis, :], (ofr_sprite_h, ofr_slice_w))
            offramp_road_masked = jnp.where(col_mask, offramp_road, self.jr.TRANSPARENT_ID)
            c = self.jr.render_at(c, 0, offramp_top, offramp_road_masked)

            # --- Split sprite: rendered just right of split_x ---
            # The sprite spans [split_x, split_x + RAMP_W] in screen x.
            s_x = split_x
            s_x_clamped = jnp.clip(s_x, 0, W).astype(jnp.int32)
            split_on_screen = (split_x > MARGIN - RAMP_W) & (s_x < W - MARGIN)
            c = jax.lax.cond(
                split_on_screen,
                lambda cv: self.jr.render_at(
                    cv, s_x_clamped, offramp_top, self.SHAPE_MASKS["offramp_split"]
                ),
                lambda cv: cv,
                c,
            )

            # --- Merge sprite: rendered just left of merge_x ---
            # The sprite spans [merge_x - RAMP_W, merge_x] in screen x.
            m_x = merge_x - RAMP_W
            m_x_clamped = jnp.clip(m_x, 0, W).astype(jnp.int32)
            merge_on_screen = (merge_x > MARGIN) & (m_x < W - MARGIN)
            c = jax.lax.cond(
                merge_on_screen,
                lambda cv: self.jr.render_at(
                    cv, m_x_clamped, offramp_top, self.SHAPE_MASKS["offramp_merge"]
                ),
                lambda cv: cv,
                c,
            )

            # --- Bridge sprites: one per configured bridge ---
            # Each bridge is a solid vertical strip filling the gap (median).
            # bridge Y is just below the offramp road band.
            bridge_y = offramp_top + OFFRAMP_H
            bridge_xs = row[3:]  # shape (MAX_OFFRAMP_BRIDGES,)
            BRIDGE_W = self.consts.OFFRAMP_BRIDGE_WIDTH

            def render_one_bridge(i, cv):
                bstep = bridge_xs[i]
                bx = (counter - bstep) * SPEED
                bx_clamped = jnp.clip(bx, 0, W - BRIDGE_W).astype(jnp.int32)
                on_screen = (bstep >= 0) & (bx >= 0) & (bx < W)
                return jax.lax.cond(
                    on_screen,
                    lambda can: self.jr.render_at(
                        can, bx_clamped, bridge_y, self.SHAPE_MASKS["offramp_bridge"]
                    ),
                    lambda can: can,
                    cv,
                )

            c = jax.lax.fori_loop(0, MAX_OFFRAMP_BRIDGES, render_one_bridge, c)
            return c

        return jax.lax.cond(offramp_active, _render_active, lambda c: c, canvas)

    def _get_animated_sprite(
        self,
        is_moving: chex.Array,
        looks_right: chex.Array,
        step_counter: chex.Array,
        animation_speed: int,
        stand_sprite: jnp.ndarray,
        run1_sprite: jnp.ndarray,
        run2_sprite: jnp.ndarray,
    ) -> jnp.ndarray:
        run_frame_idx = (step_counter // animation_speed % 2) + 1
        sprite_idx = jnp.where(is_moving, run_frame_idx, 0)
        # Stack sprites and index into them
        stacked = jnp.stack([stand_sprite, run1_sprite, run2_sprite], axis=0)
        mask = stacked[sprite_idx]
        return jnp.where(looks_right, jnp.fliplr(mask), mask)

    def _get_unique_road_dims(self) -> Tuple[list, list]:
        """
        Extract unique (height, width) pairs from level configurations.
        Includes both static road section heights and dynamic road heights.
        Returns tuple of (heights, widths) lists.
        """
        dims = set()
        for level in self.consts.levels:
            # Get width from road sections (common width for the level)
            for section in level.road_sections:
                dims.add((int(section.road_height), int(section.road_width)))

            # Add dynamic road heights if enabled
            if level.dynamic_road_heights is not None:
                height_a, height_b = level.dynamic_road_heights
                # Use the same width as the first section
                if level.road_sections:
                    width = level.road_sections[0].road_width
                else:
                    width = self.consts.WIDTH - 2 * self.consts.SIDE_MARGIN
                dims.add((int(height_a), int(width)))
                dims.add((int(height_b), int(width)))

        # If no dims found (empty levels?), default to base consts
        if not dims:
            dims.add((int(self.consts.ROAD_HEIGHT), int(self.consts.WIDTH)))

        sorted_dims = sorted(list(dims))
        heights = [d[0] for d in sorted_dims]
        widths = [d[1] for d in sorted_dims]
        return heights, widths

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: RoadRunnerState) -> jnp.ndarray:
        canvas = self.jr.create_object_raster(self.BACKGROUND)

        # --- Animate Road ---
        PATTERN_WIDTH = self.consts.ROAD_PATTERN_WIDTH
        section = self._get_render_section(state)

        # Check if dynamic road heights are enabled for this level
        level_idx = jnp.clip(state.current_level, 0, max(self._level_count - 1, 0)).astype(jnp.int32)

        # Use static road width from constants (all levels use same width)
        static_road_width = self.consts.WIDTH - 2 * self.consts.SIDE_MARGIN
        left_padding = self.consts.SIDE_MARGIN

        # Select the appropriate road mask based on pattern style
        road_mask = jax.lax.cond(
            section.road_pattern_style == 1,
            lambda: self.SHAPE_MASKS["road_no_stripes"],
            lambda: self.SHAPE_MASKS["road"]
        )

        # Scroll offset for road pattern animation
        scroll_offset = PATTERN_WIDTH - (
            (state.scrolling_step_counter * self.consts.PLAYER_MOVE_SPEED)
            % PATTERN_WIDTH
        )

        # Compute slice dimensions from the actual sprite shape so this works
        # correctly even when the renderer has been hot-swapped with a downscaled config.
        # road_mask has shape (sprite_h, sprite_w) where sprite_w = scaled SCROLL_WIDTH.
        SCROLL_W = self.consts.WIDTH + PATTERN_WIDTH   # total sprite width (unscaled)
        sprite_h = road_mask.shape[0]                  # Python int — static for dynamic_slice
        sprite_w = road_mask.shape[1]
        # Scale static_road_width and left_padding proportionally
        scaled_static_width = sprite_w * static_road_width // SCROLL_W
        scaled_left_pad     = sprite_w * left_padding     // SCROLL_W
        # Scale the dynamic scroll offset
        scaled_scroll = (scroll_offset * sprite_w // SCROLL_W).astype(jnp.int32)
        scaled_src_x  = scaled_scroll + scaled_left_pad

        if self._level_count > 0:
            dynamic_enabled = self._dynamic_road_enabled[level_idx]
            heights_config = self._dynamic_road_heights[level_idx]
            height_a = heights_config[0]
            height_b = heights_config[1]
            interval = self._dynamic_road_intervals[level_idx]
            trans_len = self._dynamic_road_transition_lengths[level_idx]
            dynamic_scroll_offset = self._dynamic_road_scroll_offsets[level_idx]

            # For dynamic roads, render per-column based on world position
            def render_dynamic_road(c):
                """Render road with per-column height calculation using vectorized masking."""
                # Calculate world x for each column
                world_scroll = state.scrolling_step_counter * self.consts.PLAYER_MOVE_SPEED
                col_indices = jnp.arange(scaled_static_width, dtype=jnp.int32)
                world_x = world_scroll + (scaled_static_width - 1 - col_indices) + dynamic_scroll_offset

                dynamic_height, _, _ = _get_dynamic_road_height(
                    world_x,
                    height_a,
                    height_b,
                    interval,
                    trans_len,
                )
                col_heights = dynamic_height.astype(jnp.int32)

                col_heights = jnp.clip(col_heights, 1, sprite_h)

                # Calculate vertical offsets for centering (per column)
                height_diffs = sprite_h - col_heights
                road_top_offsets = height_diffs // 2

                # Slice road at max (sprite) height
                road_slice = jax.lax.dynamic_slice(
                    road_mask,
                    (0, scaled_src_x),
                    (sprite_h, scaled_static_width)
                )

                # Create row indices for the road slice
                row_indices = jnp.arange(sprite_h, dtype=jnp.int32)

                row_grid = row_indices[:, None]
                top_grid = road_top_offsets[None, :]
                height_grid = col_heights[None, :]

                # Pixel is visible if: row >= top_offset AND row < top_offset + height
                visible_mask = (row_grid >= top_grid) & (row_grid < top_grid + height_grid)

                masked_slice = jnp.where(
                    visible_mask,
                    road_slice,
                    self.jr.TRANSPARENT_ID
                )

                return self.jr.render_at(c, left_padding, self.consts.ROAD_TOP_Y, masked_slice)

            def render_static_road(c):
                """Render road with single uniform height using max-height slicing."""
                current_road_height = jnp.clip(section.road_height, 1, self.consts.ROAD_HEIGHT)
                height_diff = self.consts.ROAD_HEIGHT - current_road_height
                centered_road_top = height_diff // 2

                # Slice at sprite height
                road_slice = jax.lax.dynamic_slice(
                    road_mask,
                    (0, scaled_src_x),
                    (sprite_h, scaled_static_width)
                )

                # Create visibility mask for centered road
                row_indices = jnp.arange(sprite_h, dtype=jnp.int32)
                visible_mask = (row_indices >= centered_road_top) & (row_indices < centered_road_top + current_road_height)

                masked_slice = jnp.where(
                    visible_mask[:, None],
                    road_slice,
                    self.jr.TRANSPARENT_ID
                )

                return self.jr.render_at(c, left_padding, self.consts.ROAD_TOP_Y, masked_slice)

            # Choose rendering path based on whether dynamic heights are enabled
            canvas = jax.lax.cond(dynamic_enabled, render_dynamic_road, render_static_road, canvas)
        else:
            # No levels configured, use default full-height road
            road_slice = jax.lax.dynamic_slice(
                road_mask,
                (0, scaled_src_x),
                (sprite_h, scaled_static_width)
            )

            canvas = self.jr.render_at(canvas, left_padding, self.consts.ROAD_TOP_Y, road_slice)


        # Render Offramp (above the main road)
        canvas = self._render_offramp(canvas, state, scroll_offset)

        # Render Ravines
        canvas = self._render_ravines(canvas, state.ravines)

        # Render Seeds
        canvas = self._render_seeds(canvas, state.seeds)
        
        # Render Landmine
        canvas = self._render_landmine(canvas, state.landmine_x, state.landmine_y)

        # Render score
        canvas = self._render_score(canvas, state.score)

        # Render Lives
        canvas = self._render_lives(canvas, state.lives)

        canvas = self._render_decorations(canvas, state)

        # Render Player
        def _clip_sprite_bottom(sprite_mask, player_y, clip_y):
            """Clip sprite vertically at clip_y - only show pixels above this y coordinate"""
            # Calculate how many rows to keep (from top of sprite)
            # If player_y is above clip_y, we see the full sprite
            # As player moves down, we see less of the sprite
            sprite_height = sprite_mask.shape[0]
            visible_height = jnp.maximum(0, clip_y - player_y)
            visible_height = jnp.minimum(visible_height, sprite_height)

            # Create row mask
            row_indices = jnp.arange(sprite_height)
            row_mask = row_indices < visible_height

            # Apply mask - set invisible pixels to transparent
            clipped_mask = jnp.where(
                row_mask[:, None],
                sprite_mask,
                self.jr.TRANSPARENT_ID
            )
            return clipped_mask

        def _render_burnt_player(c):
             return self.jr.render_at(c, state.player_x, state.player_y, self.SHAPE_MASKS["player_burnt"])

        def _render_normal_player(c):
            player_mask = self._get_animated_sprite(
                state.player_is_moving,
                state.player_looks_right,
                state.step_counter,
                self.consts.PLAYER_ANIMATION_SPEED,
                self.SHAPE_MASKS["player"],
                self.SHAPE_MASKS["player_run1"],
                self.SHAPE_MASKS["player_run2"],
            )
            # Clip sprite if falling
            player_mask = jax.lax.cond(
                state.is_falling,
                lambda m: _clip_sprite_bottom(m, state.player_y, state.fall_clip_y),
                lambda m: m,
                player_mask,
            )
            return self.jr.render_at(c, state.player_x, state.player_y, player_mask)

        def _render_jumping_player(c):
            jump_mask = self.SHAPE_MASKS["player_jump"]
            # Flip jump sprite if player looks right
            jump_mask = jax.lax.cond(
                state.player_looks_right,
                lambda: jnp.fliplr(jump_mask),
                lambda: jump_mask,
            )
            # Clip sprite if falling (shouldn't happen during jump, but for safety)
            jump_mask = jax.lax.cond(
                state.is_falling,
                lambda m: _clip_sprite_bottom(m, state.player_y, state.fall_clip_y),
                lambda m: m,
                jump_mask,
            )
            return self.jr.render_at(c, state.player_x, state.player_y, jump_mask)

        # Switch between burnt, jumping, Normal
        # Priority: Burnt (Death) > Jump > Normal

        def _render_alive_player(c):
            return jax.lax.cond(
                state.is_jumping,
                _render_jumping_player,
                _render_normal_player,
                c,
            )

        canvas = jax.lax.cond(
            state.death_timer > 0,
            _render_burnt_player,
            _render_alive_player,
            canvas,
        )

        # Render Enemy
        def _render_enemy(c):
            enemy_mask = self._get_animated_sprite(
                state.enemy_is_moving,
                state.enemy_looks_right,
                state.step_counter,
                self.consts.PLAYER_ANIMATION_SPEED,  # Assuming same speed for now
                self.SHAPE_MASKS["enemy"],
                self.SHAPE_MASKS["enemy_run1"],
                self.SHAPE_MASKS["enemy_run2"],
            )

            flattened_mask = self.SHAPE_MASKS["enemy_run_over"]
            # Apply flip to run-over sprite as well if needed
            flattened_mask = jax.lax.cond(
                state.enemy_looks_right,
                lambda: jnp.fliplr(flattened_mask),
                lambda: flattened_mask
            )

            final_mask = jax.lax.cond(
                state.enemy_flattened_timer > 0,
                lambda: flattened_mask,
                lambda: enemy_mask
            )

            return self.jr.render_at(c, state.enemy_x, state.enemy_y, final_mask)

        # Render the enemy only if it is on screen
        # else return the canvas unchanged
        canvas = jax.lax.cond(
            state.enemy_x < self.consts.WIDTH,
            _render_enemy,
            lambda c: c,
            canvas,
        )

        def _render_transition(_):
            t_canvas = self.jr.create_object_raster(self.BACKGROUND)

            t_canvas = self.jr.render_at(
                t_canvas,
                0,
                0,
                self.SHAPE_MASKS["end_of_level_1"]
            )

            return self.jr.render_from_palette(t_canvas, self.PALETTE)

        # Render Truck
        canvas = self._render_truck(canvas, state.truck_x, state.truck_y)

        # Render Cannon and Bullet
        def render_cannon(can):
            cannon_sprite = self.SHAPE_MASKS["canon"]
            # Flip sprite horizontally if mirrored
            flipped_sprite = jnp.where(
                state.cannon_is_mirrored,
                jnp.fliplr(cannon_sprite),
                cannon_sprite
            )
            return self.jr.render_at(can, state.cannon_x, state.cannon_y, flipped_sprite)

        canvas = jax.lax.cond(
            state.cannon_x >= 0,
            render_cannon,
            lambda can: can,
            canvas,
        )
        canvas = jax.lax.cond(
            state.bullet_x >= 0,
            lambda can: self.jr.render_at(can, state.bullet_x, state.bullet_y, self.SHAPE_MASKS["bullet"]),
            lambda can: can,
            canvas,
        )

        final_frame = self.jr.render_from_palette(canvas, self.PALETTE)

        # --- Mask Side Margins ---
        # Force pixels in the side margins to be black.
        # Use the actual frame dimensions (handles downscaling correctly).
        actual_h, actual_w, actual_c = final_frame.shape
        margin = self.consts.SIDE_MARGIN
        width = self.consts.WIDTH
        # Scale the margin to match the actual frame width
        scaled_margin = int(round(margin * actual_w / width))

        # Create a mask for valid gameplay area (True for valid, False for margin)
        col_indices = jnp.arange(actual_w)
        valid_cols = (col_indices >= scaled_margin) & (col_indices < (actual_w - scaled_margin))
        # Broadcast to full image shape (H, W, C)
        margin_mask = jnp.broadcast_to(valid_cols[None, :, None], final_frame.shape)

        # Use black for the margins
        final_frame = jnp.where(margin_mask, final_frame, jnp.zeros_like(final_frame))

        return jax.lax.cond(
            state.is_in_transition,
            _render_transition,
            lambda _: final_frame,
            operand=None,
        )
