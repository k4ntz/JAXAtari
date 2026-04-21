'''
TODO: debug super tank. Movement way off
TODO: check why the bomber is behaving weirdly, bullet should come straight at me
'''

import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import os
import jax.lax
import jax.image
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from enum import IntEnum, unique
from typing import NamedTuple, Tuple
from functools import partial
from flax import struct
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.modification import AutoDerivedConstants
from gymnasium.utils import play

#------------------------named Tuples---------------------------
class EnemyType(IntEnum):
    TANK = 0
    SAUCER = 1
    FIGHTER_JET = 2
    SUPERTANK = 3


class BattlezoneConstants(AutoDerivedConstants):
    # --- rendering: positions ---
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    WALL_TOP_Y: int = struct.field(pytree_node=False, default=0)
    WALL_TOP_HEIGHT: int = struct.field(pytree_node=False, default=36)
    WALL_BOTTOM_Y: int = struct.field(pytree_node=False, default=177)
    WALL_BOTTOM_HEIGHT: int = struct.field(pytree_node=False, default=33)
    TANK_SPRITE_POS_X: int = struct.field(pytree_node=False, default=43)
    TANK_SPRITE_POS_Y: int = struct.field(pytree_node=False, default=140)
    TARGET_INDICATOR_POS_X: int = struct.field(pytree_node=False, default=80)
    TARGET_INDICATOR_POS_Y: int = struct.field(pytree_node=False, default=77)
    CHAINS_POS_Y: int = struct.field(pytree_node=False, default=158)
    CHAINS_L_POS_X: int = struct.field(pytree_node=False, default=19)
    CHAINS_R_POS_X: int = struct.field(pytree_node=False, default=109)
    MOUNTAINS_Y: int = struct.field(pytree_node=False, default=36)
    GRASS_BACK_Y: int = struct.field(pytree_node=False, default=95)
    HORIZON_Y: int = struct.field(pytree_node=False, default=92)
    GRASS_FRONT_Y: int = struct.field(pytree_node=False, default=137)
    RADAR_CENTER_X: int = struct.field(pytree_node=False, default=80)
    RADAR_CENTER_Y: int = struct.field(pytree_node=False, default=18)
    RADAR_RADIUS: int = struct.field(pytree_node=False, default=10)
    LIFE_POS_X: int = struct.field(pytree_node=False, default=64)
    LIFE_POS_Y: int = struct.field(pytree_node=False, default=189)
    LIFE_X_OFFSET: int = struct.field(pytree_node=False, default=8)
    SCORE_POS_X: int = struct.field(pytree_node=False, default=89)
    SCORE_POS_Y: int = struct.field(pytree_node=False, default=179)
    ENEMY_POS_Y: int = struct.field(pytree_node=False, default=85)

    # --- rendering: colors ---
    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(26, 102, 26))
    TARGET_INDICATOR_COLOR_ACTIVE: Tuple[int, int, int] = struct.field(pytree_node=False, default=(255, 255, 0))
    TARGET_INDICATOR_COLOR_INACTIVE: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))
    CHAINS_COL_1: Tuple[int, int, int] = struct.field(pytree_node=False, default=(111, 111, 111))
    CHAINS_COL_2: Tuple[int, int, int] = struct.field(pytree_node=False, default=(74, 74, 74))
    RADAR_COLOR_1: Tuple[int, int, int] = struct.field(pytree_node=False, default=(111, 210, 111))
    RADAR_COLOR_2: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))
    LIFE_SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(45, 129, 105))

    # --- world movement ---
    WORLD_SIZE_X: int = struct.field(pytree_node=False, default=256)
    WORLD_SIZE_Z: int = struct.field(pytree_node=False, default=256)
    PLAYER_ROTATION_SPEED: float = struct.field(pytree_node=False, default=jnp.pi / 134)
    PLAYER_SPEED: float = struct.field(pytree_node=False, default=0.5)
    PLAYER_SPEED_DRIVETURN: float = struct.field(pytree_node=False, default=0.115348)
    PROJECTILE_SPEED: float = struct.field(pytree_node=False, default=0.4)
    ENEMY_SPEED: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([0.264, 0.5, 2.0, 0.5])
    )
    ENEMY_ROT_SPEED: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([jnp.pi / 571, 0.02, 0.02, 0.02])
    )

    # --- game mechanics ---
    # World z of the player "plane": vertical screen mapping uses f/(z − this), player shots
    # only draw when z ≥ this; enemy shot vs player uses |x| ≤ this/2 (same float as x width).
    PLAYER_Z_PLANE: float = struct.field(pytree_node=False, default=6.0)

    # --- NEW HITBOX VARS ---
    # Decoupled from the visual plane, this gives the player a tight 2x2 collision box
    # so enemy bullets can be consistently dodged by sidestepping.
    PLAYER_HITBOX_X: float = struct.field(pytree_node=False, default=2.0)
    PLAYER_HITBOX_Z: float = struct.field(pytree_node=False, default=2.0)
    # World-space aim / geometry center for the player hitbox (enemy shots & turning toward player).
    PLAYER_HITBOX_CENTER_X: float = struct.field(pytree_node=False, default=0.0)
    PLAYER_HITBOX_CENTER_Z: float = struct.field(pytree_node=False, default=0.0)

    # Edge length of the axis-aligned world box for enemy–enemy overlap (and similar).
    HITBOX_SIZE: float = struct.field(pytree_node=False, default=4.0)
    # Player (and friendly-fire) shots vs saucer & fighter only; tanks/supertanks use HITBOX_SIZE.
    PLAYER_SHOT_VS_AIR_ENEMY_HITBOX_SIZE: float = struct.field(pytree_node=False, default=1.5)
    # Enemy shots spawn this far along the enemy→player direction (world units) from the enemy.
    ENEMY_PROJECTILE_SPAWN_OFFSET: float = struct.field(pytree_node=False, default=3)

    # 9 Discrete Distance Stages
    ENEMY_STAGE_THRESHOLDS: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0])
    )
    # Screen Y anchor per distance stage (rows: TANK, SAUCER, FIGHTER_JET, SUPERTANK).
    ENEMY_STAGE_Y_POS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [
                [95, 95, 95, 95, 95, 95, 95, 95, 95],  # TANK
                [95, 95, 95, 95, 95, 95, 95, 95, 95],  # SAUCER
                # Fighter dives from high up (Y=40) down to the horizon (Y=95)
                [110, 105, 100, 95, 95, 95, 95, 95, 95],  # FIGHTER_JET
                [95, 95, 95, 95, 95, 95, 95, 95, 95],  # SUPERTANK
            ],
            dtype=jnp.int32,
        ),
    )
    ENEMY_STAGE_WIDTHS: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([
            [32, 32, 32, 24, 20, 16, 12, 8, 4],  # TANK
            [24, 24, 24, 20, 16, 12, 8, 6, 4],  # SAUCER
            [32, 28, 24, 20, 16, 12, 8, 6, 4],  # FIGHTER_JET
            [48, 40, 32, 24, 20, 16, 12, 8, 4],  # SUPERTANK
        ], dtype=jnp.int32)
    )
    ENEMY_STAGE_HEIGHTS: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([
            [20, 20, 20, 16, 12, 10, 8, 6, 4],  # TANK
            [14, 12, 10, 8, 6, 5, 4, 3, 2],  # SAUCER
            [18, 16, 14, 12, 10, 8, 6, 4, 2],  # FIGHTER_JET
            [28, 24, 20, 16, 12, 10, 8, 6, 4],  # SUPERTANK
        ],             dtype=jnp.int32)
    )
    # Ground spawns (tank / saucer / supertank): radial distance band vs ENEMY_STAGE_THRESHOLDS.
    # Stage index = sum(distance >= thresholds). Require at least N stages from the player (near)
    # and leave the outermost M stages empty (far edge), matching ALE-style spawn rings.
    SPAWN_STAGE_OFFSET_FROM_NEAR: int = struct.field(pytree_node=False, default=2)
    SPAWN_STAGE_OFFSET_FROM_FAR: int = struct.field(pytree_node=False, default=4)
    # Tank/supertank azimuth (theta in x=cosθ, z=sinθ; θ=pi/2 is +Z “in front” / FOV center).
    # SPAWN_TANK_FOV_PROB: chance to spawn inside forward cone; else uniform on full circle (sides/behind).
    SPAWN_TANK_FOV_PROB: float = struct.field(pytree_node=False, default=0.4)
    # Half-width of forward cone (radians), centered on pi/2.
    SPAWN_TANK_FOV_ARC_RAD: float = struct.field(pytree_node=False, default=0.85)
    # ---------------- DIFFICULTY SCALING ----------------
    # Score thresholds that trigger a difficulty increase.
    # Defines 6 tiers: [0-10k, 10k-20k, 20k-30k, 30k-50k, 50k-80k, 80k+]
    DIFFICULTY_THRESHOLDS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([10000, 20000, 30000, 50000, 80000], dtype=jnp.int32)
    )

    # Global multipliers for enemy drive speed based on the tier
    DIFFICULTY_SPEED_MULT: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1.0, 1.0, 1.2, 1.5, 1.8, 1.8], dtype=jnp.float32)
    )

    # Global multipliers for enemy rotation/turning speed based on the tier
    DIFFICULTY_ROT_SPEED_MULT: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0], dtype=jnp.float32)
    )

    # Spawn probabilities: [Tank, Saucer, Fighter, Supertank]
    DIFFICULTY_SPAWN_WEIGHTS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([
            [0.90, 0.10, 0.00, 0.00],  # Tier 0: 0 - 10k
            [0.80, 0.10, 0.10, 0.00],  # Tier 1: 10k - 20k
            [0.70, 0.10, 0.20, 0.00],  # Tier 2: 20k - 30k
            [0.60, 0.10, 0.20, 0.10],  # Tier 3: 30k - 50k
            [0.40, 0.10, 0.10, 0.40],  # Tier 4: 50k - 80k (Supertanks spawn heavily)
            [0.35, 0.05, 0.10, 0.50],  # Tier 5: 80k+ (50% Supertanks)
        ], dtype=jnp.float32)
    )
    # Dodge parameters for Tanks and Supertanks (player bullet near-miss raycast)
    ENEMY_DODGE_MIN_DIST: float = struct.field(pytree_node=False, default=3.0)
    ENEMY_DODGE_MAX_DIST: float = struct.field(pytree_node=False, default=15.0)

    # Debug override: force enemy type for the first spawn of each batch (matches spawn_idx ==
    # total_spawned_count at spawn time, not literal 0 after the opening wave). -1 disables.
    FIRST_SPAWN_TYPE: int = struct.field(pytree_node=False, default=-1) 
    RADAR_MAX_SCAN_RADIUS: int = struct.field(pytree_node=False, default=110)
    SAUCER_MIN_DIST: float = struct.field(pytree_node=False, default=27.0)
    # Saucer flips strafe when the yellow target column aligns with it; min frames between flips.
    SAUCER_HOVER_DIRECTION_FLIP_COOLDOWN_FRAMES: int = struct.field(pytree_node=False, default=30)
    FIGHTER_AREA_X: Tuple[float, float] = struct.field(pytree_node=False, default=(-12.5, 12.5))
    FIGHTER_AREA_Z: Tuple[float, float] = struct.field(pytree_node=False, default=(75.0, 126.0))
    # Baseline deterministic sequence for Fighter Jet (115 samples; sub-frame advance via jnp.interp).
    # 1.0 = ALE pace; scale up/down without stutter.
    FIGHTER_ANIM_SPEED: float = struct.field(pytree_node=False, default=1.0)
    # Tight lateral scale avoids x/z blow-up when the jet is very close.
    FIGHTER_DRIFT_SCALE: float = struct.field(pytree_node=False, default=0.04)
    # Triggers at logic frame 108 per ALE CSV log alignment
    FIGHTER_SHOOTING_DISTANCE: float = struct.field(pytree_node=False, default=16.5)
    FIGHTER_DIST_SEQ: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [
                96.00, 94.90, 92.40, 89.90, 87.40, 84.90, 84.28, 83.65, 83.03, 82.40, 81.78, 81.15, 80.53, 79.90, 79.28, 78.65, 78.03, 77.40, 76.78, 76.15, 75.53, 74.90, 74.23, 73.57, 72.90, 72.23, 71.57, 70.90, 70.23, 69.57, 68.90, 68.23, 67.57, 66.90, 66.23, 65.57, 64.90, 64.37, 63.85, 63.32, 62.79, 62.27, 61.74, 61.22, 60.69, 60.16, 59.64, 59.11, 58.58, 58.06, 57.53, 57.01, 56.48, 55.95, 55.43, 54.90, 54.27, 53.65, 53.02, 52.40, 51.77, 51.15, 50.52, 49.90, 49.27, 48.65, 48.02, 47.40, 46.77, 46.15, 45.52, 44.90, 44.34, 43.79, 43.23, 42.68, 42.12, 41.57, 41.01, 40.46, 39.90, 39.34, 38.79, 38.23, 37.68, 37.12, 36.57, 36.01, 35.46, 34.90, 33.99, 33.08, 32.17, 31.26, 30.35, 29.45, 28.54, 27.63, 26.72, 25.81, 24.90, 23.84, 22.77, 21.71, 20.64, 19.58, 18.51, 17.45, 16.39, 15.32, 14.26, 13.19, 12.13, 11.06, 10.00,
            ],
            dtype=jnp.float32,
        ),
    )
    FIGHTER_LATERAL_SEQ: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [
                0.00, 0.00, 0.00, -1.00, -1.00, -1.00, -2.00, -2.00, -2.00, -2.00, -3.00, -3.00, -3.00, -4.00, -5.00, -5.00, -5.00, -6.00, -6.00, -6.00, -6.00, -6.00, -5.00, -4.00, -4.00, -3.00, -2.00, -2.00, -1.00, -1.00, -1.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, -1.00, -1.00, -2.00, -3.00, -4.00, -4.00, -5.00, -6.00, -6.00, -7.00, -8.00, -8.00, -9.00, -9.00, -8.00, -8.00, -7.00, -6.00, -5.00, -4.00, -3.00, -3.00, -2.00, -1.00, 0.00, 0.00, 1.00, 1.00, 2.00, 2.00, 4.00, 4.00, 5.00, 5.00, 4.00, 3.00, 2.00, 1.00, 1.00, 0.00, -1.00, -2.00, -3.00, -4.00, -5.00, -6.00, -7.00, -7.00, -8.00, -9.00, -10.00, -9.00, -8.00, -7.00, -5.00, -3.00, -1.00, 1.00, 3.00, 5.00, 7.00, 9.00, 11.00, 13.00, 15.00, 17.00, 19.00, 21.00, 23.00, 25.00, 27.00, 29.00, 31.00,
            ],
            dtype=jnp.float32,
        ),
    )
    TANKS_SHOOTING_DISTANCE: float = struct.field(pytree_node=False, default=20.0)
    # Atari ram_depth never exceeded ~47; jax_dist = -1.875*ram_depth + 106.25 => min ~18.125.
    # Clamping continuous world distance avoids perspective divide blow-up (x/z) when z -> 0.
    ENEMY_MIN_WORLD_DISTANCE: float = struct.field(pytree_node=False, default=18.125)
    # Lateral drift along the world tangent when clamped while driving forward; tune for ~2–3 px/frame.
    ENEMY_NEAR_DRIFT_SPEED: float = struct.field(pytree_node=False, default=0.05)
    # --- timing ---
    # Player: one shot per this many simulation steps; bullet lives this many sim steps (same units as _single_projectile_step).
    PLAYER_BULLET_Y_SEQ: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([
            129, 126, 121, 119, 115, 113, 111, 109, 107, 105,
            103, 101, 100, 99, 98, 98, 98, 97, 97, 97,
            97, 97, 96, 96, 96, 96, 96, 96, 96, 96,
            95, 95, 95, 95, 95, 95, 95, 95, 95, 95,
            94, 94, 93, 93, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,
        ], dtype=jnp.int32),
    )
    FIRE_CD: int = struct.field(pytree_node=False, default=55)
    PLAYER_PROJECTILE_TTL: int = struct.field(pytree_node=False, default=55)
    PROJECTILE_TTL: int = struct.field(pytree_node=False, default=55)  # enemy projectiles only
    # ALE tuning: visual-only player-shot lateral drift scale during turns.
    # Physics/collision should remain unscaled to avoid steerable bullets.
    PLAYER_PROJECTILE_TURN_SCALE: float = struct.field(pytree_node=False, default=1)
    # Ramp visual drift from a milder early-flight scale to the target scale.
    PLAYER_PROJECTILE_TURN_SCALE_MIN: float = struct.field(pytree_node=False, default=1)
    PLAYER_PROJECTILE_TURN_SCALE_RAMP_STEPS: int = struct.field(pytree_node=False, default=14)
    DEATH_ANIM_LENGTH: int = struct.field(pytree_node=False, default=15)
    ENEMY_DEATH_ANIM_LENGTH: int = struct.field(pytree_node=False, default=15)
    ENEMY_SHOOT_CDS: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([200, 200, 200, 200])
    )

    # --- misc ---
    RADAR_ROTATION_SPEED: float = struct.field(pytree_node=False, default=-0.05)
    DISTANCE_TO_ZOOM_FACTOR_CONSTANT: float = struct.field(pytree_node=False, default=0.05)
    ENEMY_SCORES: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1000, 3000, 2000, 5000], dtype=jnp.int32),
    )
    # Extra life milestones: 50k and 100k only.
    BONUS_LIFE_EVERY_SCORE: int = struct.field(pytree_node=False, default=50_000)
    BONUS_LIFE_MAX_SCORE: int = struct.field(pytree_node=False, default=100_000)
    MAX_LIVES: int = struct.field(pytree_node=False, default=6)
    CAMERA_FOCAL_LENGTH: float = struct.field(pytree_node=False, default=180)
    ENEMY_WIDTHS: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([24, 32, 32, 24], dtype=jnp.int32)
    )
    ENEMY_HEIGHTS: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([14, 18, 17, 14], dtype=jnp.int32)
    )


@struct.dataclass
class Projectile:
    """Class holding projectiles. properties are arrays."""
    x: chex.Array
    z: chex.Array
    orientation_angle: chex.Array
    active: chex.Array
    distance: chex.Array
    time_to_live: chex.Array


@struct.dataclass
class Enemy:
    x: chex.Array
    z: chex.Array
    distance: chex.Array
    enemy_type: chex.Array
    orientation_angle: chex.Array  # 0 = towards positive z
    active: chex.Array
    death_anim_counter: chex.Array
    shoot_cd: chex.Array
    phase: chex.Array
    dist_moved_temp: chex.Array
    # points used for movement behaviour
    point_store_1_temp: chex.Array
    point_store_2_temp: chex.Array


# immutable state container
@struct.dataclass
class BattlezoneState:
    score: chex.Array
    life: chex.Array
    cur_fire_cd: chex.Array  # player current fire cooldown
    step_counter: chex.Array
    chains_l_anim_counter: chex.Array
    chains_r_anim_counter: chex.Array
    death_anim_counter: chex.Array
    mountains_anim_counter: chex.Array
    grass_anim_counter: chex.Array
    radar_rotation_counter: chex.Array
    enemies: Enemy
    player_projectile: Projectile # player can only fire 1 projectile
    enemy_projectiles: Projectile # per enemy 1 projectile
    random_key: chex.PRNGKey
    shot_spawn: chex.Array
    total_spawned_count: chex.Array


@struct.dataclass
class BattlezoneObservation:
    enemies: ObjectObservation
    radar_dots: ObjectObservation
    projectiles: ObjectObservation
    score: jnp.ndarray
    life: jnp.ndarray
    enemy_types: jnp.ndarray


@struct.dataclass
class BattlezoneInfo:
    time: jnp.ndarray


#----------------------------Battlezone Environment------------------------
class JaxBattlezone(JaxEnvironment[BattlezoneState, BattlezoneObservation, BattlezoneInfo, BattlezoneConstants]):

    def __init__(self, consts: BattlezoneConstants = None):

        self.consts = consts or BattlezoneConstants()
        super().__init__(self.consts)

        self.renderer = BattlezoneRenderer(self.consts)

        self.action_set=[ #from https://ale.farama.org/environments/battle_zone/
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
            Action.DOWNLEFTFIRE]


    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: BattlezoneState, action: chex.Array) -> BattlezoneState:

        #-------------------parse action--------------------
        wants_fire = jnp.any(jnp.stack([
            action == Action.FIRE,
            action == Action.LEFTFIRE,
            action == Action.UPFIRE,
            action == Action.RIGHTFIRE,
            action == Action.DOWNFIRE,
            action == Action.UPLEFTFIRE,
            action == Action.UPRIGHTFIRE,
            action == Action.DOWNLEFTFIRE,
            action == Action.DOWNRIGHTFIRE,
        ]), axis=0)

        # Emulate Atari-style "every other frame" simulation: movement/scrolling only
        # happens on even steps, while fire input is still registered every step.
        do_move = ((state.step_counter + 1) % 2) == 0
        move_action = jnp.where(do_move, action, Action.NOOP)

        noop = (move_action == Action.NOOP)
        up = jnp.logical_or(move_action == Action.UP, move_action == Action.UPFIRE)
        down = jnp.logical_or(move_action == Action.DOWN, move_action == Action.DOWNFIRE)
        right = jnp.logical_or(move_action == Action.RIGHT, move_action == Action.RIGHTFIRE)
        left = jnp.logical_or(move_action == Action.LEFT, move_action == Action.LEFTFIRE)
        upLeft = jnp.logical_or(move_action == Action.UPLEFT, move_action == Action.UPLEFTFIRE)
        upRight = jnp.logical_or(move_action == Action.UPRIGHT, move_action == Action.UPRIGHTFIRE)
        downLeft = jnp.logical_or(move_action == Action.DOWNLEFT, move_action == Action.DOWNLEFTFIRE)
        downRight = jnp.logical_or(move_action == Action.DOWNRIGHT, move_action == Action.DOWNRIGHTFIRE)

        direction = jnp.stack([noop, up, right, left, down, upRight, upLeft, downRight, downLeft])  # leave order as is!

        #-------------------fire--------------
        will_fire = jnp.logical_and(wants_fire, jnp.invert(state.player_projectile.active))

        def fire_projectile(state: BattlezoneState):
            return state.replace(
                player_projectile= Projectile(
                    x=jnp.array(0.0, dtype=jnp.float32),
                    z=jnp.array(10.0, dtype=jnp.float32),
                    orientation_angle=jnp.array(jnp.pi, dtype=jnp.float32),
                    active=jnp.array(True, dtype=jnp.bool),
                    distance=jnp.array(0, dtype=jnp.float32),
                    time_to_live=jnp.array(self.consts.PLAYER_PROJECTILE_TTL, dtype=jnp.int32)
                )
            )

        new_state = jax.lax.cond(will_fire, fire_projectile, lambda s: s, state)

        #--------------------anims--------------------
        # Keep visual turn cues in sync with world rotation:
        # use a single signed turn signal for all turn states (standstill/forward/reverse).
        turn_signal = (jnp.where(jnp.any(jnp.stack([left, upLeft, downLeft])), 1.0, 0.0)
                       - jnp.where(jnp.any(jnp.stack([right, upRight, downRight])), 1.0, 0.0))
        move_signal = (jnp.where(jnp.any(jnp.stack([up, upLeft, upRight])), 1.0, 0.0)
                       - jnp.where(jnp.any(jnp.stack([down, downRight, downLeft])), 1.0, 0.0))

        chain_r_offset = turn_signal + move_signal
        chain_l_offset = turn_signal - move_signal
        mountains_offset = turn_signal
        grass_offset = move_signal

        #--------------------update positions based on player movement-------------------
        updated_enemies = jax.vmap(self._enemy_player_position_update, in_axes=(0, None))(state.enemies, direction)
        updated_projectiles = (jax.vmap(self._obj_player_position_update, in_axes=(0, None))
                               (state.enemy_projectiles, direction))
        new_player_projectile = self._obj_player_position_update(new_state.player_projectile, direction)

        #--------------------update angles based on player movement-----------------------
        # Keep reverse-turn signs aligned with _position_update rotational branches:
        # DOWNLEFT follows LEFT's rotation, DOWNRIGHT follows RIGHT's rotation.
        angle_change = (jnp.where(jnp.any(jnp.stack([left, upLeft, downLeft])), 1.0, 0.0)
                        - jnp.where(jnp.any(jnp.stack([right, upRight, downRight])), 1.0, 0.0))

        updated_enemies = jax.vmap(self._obj_player_rotation_update, in_axes=(0,None))(updated_enemies, angle_change)
        updated_projectiles = (jax.vmap(self._obj_player_rotation_update, in_axes=(0,None))
                               (updated_projectiles, angle_change))
        new_player_projectile = self._obj_player_rotation_update(new_player_projectile, angle_change)
        return new_state.replace(
            chains_l_anim_counter=(state.chains_l_anim_counter + chain_l_offset) % 32,
            chains_r_anim_counter=(state.chains_r_anim_counter + chain_r_offset) % 32,
            mountains_anim_counter=(state.mountains_anim_counter + mountains_offset * 2.5) % 160,
            grass_anim_counter=(state.grass_anim_counter + grass_offset) % 30,
            radar_rotation_counter=new_state.radar_rotation_counter,
            enemies=updated_enemies,
            player_projectile=new_player_projectile,
            enemy_projectiles=updated_projectiles
        )


    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: BattlezoneState) -> BattlezoneState:

        d_anim_counter = state.enemies.death_anim_counter
        new_death_anim_counter = jnp.where(d_anim_counter > 0, d_anim_counter - 1, d_anim_counter)
        new_enemies, new_projectiles = jax.vmap(
            self.enemy_movement, in_axes=(0, 0, None, None)
        )(state.enemies, state.enemy_projectiles, state.score, state.player_projectile)
        zero_dir = jnp.zeros((9,), dtype=jnp.bool_)
        new_enemies = jax.vmap(self._apply_enemy_min_distance_boundary, in_axes=(0, None, None))(
            new_enemies, zero_dir, jnp.float32(0.0)
        )
        shoot_cd = new_enemies.shoot_cd
        new_shoot_cd = jnp.where(shoot_cd > 0, shoot_cd - 1, shoot_cd)

        return state.replace(
            enemies=new_enemies.replace(
                death_anim_counter=new_death_anim_counter, shoot_cd=new_shoot_cd
            ),
            enemy_projectiles=new_projectiles
        )


    @partial(jax.jit, static_argnums=(0,))
    def _single_projectile_step(self, projectile: Projectile) -> Projectile:
        """implements movement of projectiles"""

        dir_x = -jnp.sin(projectile.orientation_angle)
        dir_z = jnp.cos(projectile.orientation_angle)
        new_x = projectile.x - dir_x*self.consts.PROJECTILE_SPEED
        new_z = projectile.z - dir_z*self.consts.PROJECTILE_SPEED

        return projectile.replace(
            x=new_x,
            z=new_z,
            time_to_live=jnp.where(projectile.time_to_live>0, projectile.time_to_live-1, 0),
            active=jnp.logical_and(projectile.active, projectile.time_to_live>0)
        )


    @partial(jax.jit, static_argnums=(0,))
    def _player_projectile_step(self, projectile: Projectile) -> Projectile:
        """Smooth 3D movement decoupled from projected pixel trajectory."""
        speed = 110.0 / self.consts.PLAYER_PROJECTILE_TTL

        dir_x = -jnp.sin(projectile.orientation_angle)
        dir_z = jnp.cos(projectile.orientation_angle)

        new_x = projectile.x - dir_x * speed
        new_z = projectile.z - dir_z * speed
        new_distance = jnp.sqrt(new_x**2 + new_z**2)

        return projectile.replace(
            x=new_x,
            z=new_z,
            distance=new_distance,
            time_to_live=jnp.where(projectile.time_to_live > 0, projectile.time_to_live - 1, 0),
            active=jnp.logical_and(projectile.active, projectile.time_to_live > 1),
        )


    def _player_projectile_collision_step(self, state: BattlezoneState):
        hit_arr = (jax.vmap(self._player_projectile_collision_check, in_axes=(0, None))
                   (state.enemies, state.player_projectile))

        score_increases = jnp.where(
            hit_arr,
            self.consts.ENEMY_SCORES[state.enemies.enemy_type],
            0
        )
        new_score = state.score + jnp.sum(score_increases, dtype=state.score.dtype)
        capped_new_score = jnp.minimum(new_score, jnp.int32(self.consts.BONUS_LIFE_MAX_SCORE))
        capped_old_score = jnp.minimum(state.score, jnp.int32(self.consts.BONUS_LIFE_MAX_SCORE))
        bonus_life_steps = (
            capped_new_score // self.consts.BONUS_LIFE_EVERY_SCORE
            - capped_old_score // self.consts.BONUS_LIFE_EVERY_SCORE
        )
        new_life = jnp.minimum(
            jnp.int32(self.consts.MAX_LIVES),
            state.life + bonus_life_steps.astype(state.life.dtype),
        )

        new_enemies_active = jnp.logical_and(state.enemies.active, jnp.invert(hit_arr))
        new_enemies_death_anim_counter = jnp.where(hit_arr,
                            self.consts.ENEMY_DEATH_ANIM_LENGTH, state.enemies.death_anim_counter)
        new_player_projectile_active = jnp.logical_and(state.player_projectile.active, jnp.invert(jnp.any(hit_arr)))

        return state.replace(
            score=new_score,
            life=new_life,
            enemies=state.enemies.replace(
                active=new_enemies_active,
                death_anim_counter=new_enemies_death_anim_counter
            ),
            player_projectile=state.player_projectile.replace(
                active=new_player_projectile_active
            )
        )

    def _enemy_friendly_fire_step(self, state: BattlezoneState):
        hit_matrix = jax.vmap(
            lambda enemy: jax.vmap(
                self._player_projectile_collision_check,
                in_axes=(None, 0)
            )(enemy, state.enemy_projectiles)
        )(state.enemies)
        # One projectile slot exists per enemy slot. Ignore self-collisions so an
        # enemy cannot instantly "friendly-fire" itself with its own shot.
        n_enemies = hit_matrix.shape[0]
        self_hits = jnp.eye(n_enemies, dtype=jnp.bool_)
        hit_matrix = jnp.logical_and(hit_matrix, jnp.invert(self_hits))
        hit_arr = jnp.any(hit_matrix, axis=1)
        new_enemies_active = jnp.logical_and(state.enemies.active, jnp.invert(hit_arr))
        new_enemies_death_anim_counter = jnp.where(
            hit_arr,
            self.consts.ENEMY_DEATH_ANIM_LENGTH,
            state.enemies.death_anim_counter
        )
        projectile_hit_any_enemy = jnp.any(hit_matrix, axis=0)
        new_enemy_projectiles_active = jnp.logical_and(
            state.enemy_projectiles.active,
            jnp.invert(projectile_hit_any_enemy)
        )
        return state.replace(
            enemies=state.enemies.replace(
                active=new_enemies_active,
                death_anim_counter=new_enemies_death_anim_counter
            ),
            enemy_projectiles=state.enemy_projectiles.replace(
                active=new_enemy_projectiles_active
            )
        )

    def _enemy_enemy_collision_step(self, state: BattlezoneState):
        """
        Checks if any active enemy is colliding with any other active enemy.
        If their 3D coordinates overlap, both are destroyed.
        """
        def single_enemy_check(enemy1):
            def pair_check(enemy2):
                s = self.consts.HITBOX_SIZE
                hit_x = jnp.abs(enemy1.x - enemy2.x) < s
                hit_z = jnp.abs(enemy1.z - enemy2.z) < s
                # Fighter uses a separate flight path; 3D AABB vs ground units caused false
                # overlaps mid-dive (both units destroyed = bomber "respawns" when slot refills).
                has_fighter = jnp.logical_or(
                    enemy1.enemy_type == EnemyType.FIGHTER_JET,
                    enemy2.enemy_type == EnemyType.FIGHTER_JET,
                )
                overlap = hit_x & hit_z & enemy1.active & enemy2.active & jnp.logical_not(has_fighter)
                return overlap
            return jax.vmap(pair_check)(state.enemies)

        hit_matrix = jax.vmap(single_enemy_check)(state.enemies)

        n_enemies = hit_matrix.shape[0]
        self_hits = jnp.eye(n_enemies, dtype=jnp.bool_)
        hit_matrix = jnp.logical_and(hit_matrix, jnp.invert(self_hits))

        hit_arr = jnp.any(hit_matrix, axis=1)

        new_enemies_active = jnp.logical_and(state.enemies.active, jnp.invert(hit_arr))
        new_enemies_death_anim_counter = jnp.where(
            hit_arr,
            self.consts.ENEMY_DEATH_ANIM_LENGTH,
            state.enemies.death_anim_counter
        )

        return state.replace(
            enemies=state.enemies.replace(
                active=new_enemies_active,
                death_anim_counter=new_enemies_death_anim_counter
            )
        )

    def reset(self, key=None) -> Tuple[BattlezoneObservation, BattlezoneState]:

        if key is None:
            key = jax.random.PRNGKey(0)

        state = BattlezoneState(
            score=jnp.array(0, dtype=jnp.int32),
            life=jnp.array(5, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            cur_fire_cd=jnp.array(0, dtype=jnp.int32),
            death_anim_counter=jnp.array(0, dtype=jnp.int32),
            chains_l_anim_counter=jnp.array(0, dtype=jnp.float32),
            chains_r_anim_counter=jnp.array(0, dtype=jnp.float32),
            mountains_anim_counter=jnp.array(0, dtype=jnp.float32),
            grass_anim_counter=jnp.array(0, dtype=jnp.float32),
            radar_rotation_counter=jnp.array(0, dtype=jnp.float32),
            enemies = Enemy(
                x=jnp.array([0.0, 0.0], dtype=jnp.float32),
                z=jnp.array([50.0, 0.0], dtype=jnp.float32),
                distance=jnp.array([50.0, 0.0], dtype=jnp.float32),
                enemy_type=jnp.array([EnemyType.TANK, EnemyType.TANK], dtype=jnp.int32),
                orientation_angle=jnp.array([jnp.pi / 2, 0.0], dtype=jnp.float32),
                active=jnp.array([False, False], dtype=jnp.bool),
                death_anim_counter=jnp.array([0,0], dtype=jnp.int32),
                shoot_cd=jnp.array([0, 0], dtype=jnp.int32),
                phase=jnp.array([0, 0], dtype=jnp.int32),
                dist_moved_temp=jnp.array([0, 0], dtype=jnp.float32),
                point_store_1_temp=jnp.array([[0, 0], [0, 0]], dtype=jnp.float32),
                point_store_2_temp=jnp.array([[0, 0], [0, 0]], dtype=jnp.float32),
            ),
            player_projectile=Projectile(
                x=jnp.array(0, dtype=jnp.float32),
                z=jnp.array(0, dtype=jnp.float32),
                orientation_angle=jnp.array(0, dtype=jnp.float32),
                active=jnp.array(False, dtype=jnp.bool),
                distance=jnp.array(0, dtype=jnp.float32),
                time_to_live=jnp.array(0, dtype=jnp.int32)
            ),
            enemy_projectiles=Projectile(
                x=jnp.array([0, 0], dtype=jnp.float32),
                z=jnp.array([0, 0], dtype=jnp.float32),
                orientation_angle=jnp.array([0, 0], dtype=jnp.float32),
                active=jnp.array([False, False], dtype=jnp.bool),
                distance=jnp.array([0, 0], dtype=jnp.float32),
                time_to_live=jnp.array([0, 0], dtype=jnp.int32)
            ),
            random_key=key,
            shot_spawn=jnp.array([False], dtype=jnp.bool),
            total_spawned_count=jnp.array(0, dtype=jnp.int32),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state


    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            state: BattlezoneState,
            action: chex.Array
        ) -> Tuple[BattlezoneObservation, BattlezoneState, float, bool, BattlezoneInfo]:

        previous_state = state

        def normal_step(state):
            # "Simulate on even frames" (frameskip=2 style): game logic updates only on
            # every other call; the other call mostly registers inputs / returns obs.
            next_step_counter = state.step_counter + 1
            do_sim = (next_step_counter % 2) == 0

            new_state = state.replace(step_counter=next_step_counter)

            # Update time-based counters only on simulation frames.
            new_state = jax.lax.cond(
                do_sim,
                lambda s: s.replace(
                    radar_rotation_counter=(s.radar_rotation_counter + self.consts.RADAR_ROTATION_SPEED) % 360
                ),
                lambda s: s,
                new_state
            )

            # Register player action every frame (movement itself is gated in _player_step).
            new_state = self._player_step(new_state, action)

            def sim_step(s: BattlezoneState) -> BattlezoneState:
                #-------------------projectiles-------------
                # Collision check before movement avoids tunneling when a shot spawns or
                # starts very close to an enemy between discrete simulation steps.
                s = self._player_projectile_collision_step(s)
                s = s.replace(player_projectile=self._player_projectile_step(s.player_projectile))
                s = self._player_projectile_collision_step(s)
                s = self._enemy_friendly_fire_step(s)
                s = self._enemy_enemy_collision_step(s)

                s = s.replace(enemy_projectiles=jax.vmap(self._single_projectile_step, 0)(s.enemy_projectiles))
                player_hit = jnp.any(jax.vmap(self._enemy_projectile_collision_check, 0)(s.enemy_projectiles))
                s = s.replace(death_anim_counter=jnp.where(
                    player_hit,
                    jnp.array(self.consts.DEATH_ANIM_LENGTH, dtype=jnp.int32),
                    s.death_anim_counter
                ))
                #------------------------------------------

                #-------------------spawn-------------------
                split_key, key = jax.random.split(s.random_key, 2)
                s = s.replace(random_key=key)

                is_slot_ready = jnp.logical_and(~s.enemies.active, s.enemies.death_anim_counter <= 0)
                num_spawning = jnp.sum(is_slot_ready.astype(jnp.int32))
                spawn_indices = s.total_spawned_count + jnp.cumsum(is_slot_ready.astype(jnp.int32)) - 1

                new_enemies = self._spawn_all_ready_slots(
                    split_key,
                    s.enemies,
                    s.score,
                    spawn_indices,
                    s.total_spawned_count,
                    is_slot_ready,
                )
                s = s.replace(
                    enemies=new_enemies,
                    total_spawned_count=s.total_spawned_count + num_spawning
                )
                #-------------------------------------------

                s = self._enemy_step(s)
                return s

            new_state = jax.lax.cond(do_sim, sim_step, lambda s: s, new_state)
            return new_state


        def death_step(state):
            new_death_counter = state.death_anim_counter - 1
            new_state = state.replace(death_anim_counter=new_death_counter)
            new_state = jax.lax.cond(new_death_counter <= 0, self.player_shot_reset, lambda x: x, new_state)
            return new_state

        new_state = jax.lax.cond(state.death_anim_counter <= 0, normal_step, death_step, state)

        done = self._get_done(new_state)
        env_reward = self._get_reward(previous_state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info


    def _obj_player_position_update(self, obj: NamedTuple, player_direction) -> Enemy:
        """
        _position_update version for named tuples that contain x, z, distance
        (now only for projectiles, enemies have their own version because of the stored points)
        """
        # get offset to add to coords to get updated position
        offset_x, offset_z = self._position_update(obj.x, obj.z, obj.active, player_direction)

        # update object position
        new_x = obj.x + offset_x
        new_z = obj.z + offset_z

        return obj.replace(
            x=self._wrap_coord(new_x, self.consts.WORLD_SIZE_X),
            z=self._wrap_coord(new_z, self.consts.WORLD_SIZE_Z),
            distance=self._get_distance(new_x, new_z)
        )


    def _enemy_player_position_update(self, obj: Enemy, player_direction) -> Enemy:
        """
        _position_update version for named tuples that contain x, z, distance
        """
        # get offset to add to coords to get updated position
        offset_x, offset_z = self._position_update(obj.x, obj.z, obj.active, player_direction)

        # update object position
        new_x = obj.x + offset_x
        new_z = obj.z + offset_z

        obj = obj.replace(
            x=new_x,
            z=new_z,
            distance=self._get_distance(new_x, new_z),
            # also need to adapt stored points for saucer movement (but not world-wrap these necessarily, only do that when obj position wraps)
            point_store_1_temp=obj.point_store_1_temp + jnp.array([offset_x, offset_z], dtype=jnp.float32),
            point_store_2_temp=obj.point_store_2_temp + jnp.array([offset_x, offset_z], dtype=jnp.float32)
        )

        obj = self._wrap_coords_and_stored_points(obj)

        obj = self._apply_enemy_min_distance_boundary(obj, player_direction, jnp.float32(1.0))

        return obj


    def _wrap_coord(self, coord, world_size):
        """Wraps a coordinate around the world edges."""
        half_size = world_size / 2.0
        return coord - world_size * jnp.floor((coord + half_size) / world_size)


    def _wrap_coords_and_stored_points(self, enemy: Enemy):
        """
        world-wrap enemy coordinates. if wrap occured, also move stored points accordingly.
        """

        before_wrap_x = enemy.x
        before_wrap_z = enemy.z

        after_wrap_x = self._wrap_coord(before_wrap_x, self.consts.WORLD_SIZE_X)
        after_wrap_z = self._wrap_coord(before_wrap_z, self.consts.WORLD_SIZE_Z)

        # if the enemy position wraps, we need to wrap the stored points too
        def wrap_stored_points_x(enemy):
            wrap_direction = jnp.sign(before_wrap_x - after_wrap_x)  # + = right wrap, - = left wrap
            return enemy.replace(
                point_store_1_temp=enemy.point_store_1_temp.at[0].add(-wrap_direction * self.consts.WORLD_SIZE_X),
                point_store_2_temp=enemy.point_store_2_temp.at[0].add(-wrap_direction * self.consts.WORLD_SIZE_X)
            )

        def wrap_stored_points_z(enemy):
            wrap_direction = jnp.sign(before_wrap_z - after_wrap_z)  # + = up wrap, - = down wrap
            return enemy.replace(
                point_store_1_temp=enemy.point_store_1_temp.at[1].add(-wrap_direction * self.consts.WORLD_SIZE_Z),
                point_store_2_temp=enemy.point_store_2_temp.at[1].add(-wrap_direction * self.consts.WORLD_SIZE_Z)
            )

        # larger atol is fine since wrap creates difference of world_size
        enemy = jax.lax.cond(~jnp.isclose(before_wrap_x, after_wrap_x, atol=1.0), wrap_stored_points_x, lambda e: e, enemy)
        enemy = jax.lax.cond(~jnp.isclose(before_wrap_z, after_wrap_z, atol=1.0), wrap_stored_points_z, lambda e: e, enemy)
        enemy = enemy.replace(x=after_wrap_x, z=after_wrap_z, distance=self._get_distance(after_wrap_x, after_wrap_z))

        return enemy


    def _position_update(self, prev_x, prev_z, active, player_direction):
        """
        updates the x, z coordinates according to the current player movement
        """
        ###
        idx = jnp.argmax(player_direction)
        alpha = self.consts.PLAYER_ROTATION_SPEED
        speed = jax.lax.cond(idx > 4, lambda: self.consts.PLAYER_SPEED_DRIVETURN, lambda: self.consts.PLAYER_SPEED)
        ###
        sin_alpha = jnp.sin(alpha)
        cos_alpha = jnp.cos(alpha)
        sin_neg_alpha = -sin_alpha
        cos_neg_alpha = cos_alpha

        branches = (
            lambda: jnp.array([0.0, 0.0], dtype=jnp.float32),  # Noop
            lambda: jnp.array([0.0, -speed], dtype=jnp.float32),  # Up
            lambda: jnp.array([
                (prev_x * cos_alpha - prev_z * sin_alpha) - prev_x,
                (prev_x * sin_alpha + prev_z * cos_alpha) - prev_z
            ], dtype=jnp.float32),  # Right
            lambda: jnp.array([
                (prev_x * cos_neg_alpha - prev_z * sin_neg_alpha) - prev_x,
                (prev_x * sin_neg_alpha + prev_z * cos_neg_alpha) - prev_z
            ], dtype=jnp.float32),  # Left
            lambda: jnp.array([0.0, speed], dtype=jnp.float32),  # Down
            lambda: jnp.array([
                (prev_x * cos_alpha - (prev_z - speed) * sin_alpha) - prev_x,
                (prev_x * sin_alpha + (prev_z - speed) * cos_alpha) - prev_z
            ], dtype=jnp.float32),  # UpRight
            lambda: jnp.array([
                (prev_x * cos_alpha + (prev_z - speed) * sin_alpha) - prev_x,
                (-prev_x * sin_alpha + (prev_z - speed) * cos_alpha) - prev_z
            ], dtype=jnp.float32),  # UpLeft
            lambda: jnp.array([
                (prev_x * cos_alpha - (prev_z + speed) * sin_alpha) - prev_x,
                (prev_x * sin_alpha + (prev_z + speed) * cos_alpha) - prev_z
            ], dtype=jnp.float32),  # DownRight
            lambda: jnp.array([
                (prev_x * cos_alpha + (prev_z + speed) * sin_alpha) - prev_x,
                (-prev_x * sin_alpha + (prev_z + speed) * cos_alpha) - prev_z
            ], dtype=jnp.float32),  # DownLeft
        )

        offset = jax.lax.switch(idx, branches)

        return offset[0], offset[1]


    def _obj_player_rotation_update(self, obj: NamedTuple, angle_change):
        """
        Applies a uniform coordinate rotation to the object's orientation so objects
        move in straight lines in player-centric world space when the camera turns.
        """
        alpha = self.consts.PLAYER_ROTATION_SPEED
        angle = alpha * angle_change

        return obj.replace(orientation_angle=(obj.orientation_angle - angle) % (2 * jnp.pi))


    def _player_projectile_visual_turn_scale(self, projectile_ttl: chex.Array):
        age = (self.consts.PLAYER_PROJECTILE_TTL - projectile_ttl).astype(jnp.float32)
        ramp = jnp.float32(max(1, self.consts.PLAYER_PROJECTILE_TURN_SCALE_RAMP_STEPS))
        blend = jnp.clip(age / ramp, 0.0, 1.0)
        return (
            self.consts.PLAYER_PROJECTILE_TURN_SCALE_MIN
            + (self.consts.PLAYER_PROJECTILE_TURN_SCALE - self.consts.PLAYER_PROJECTILE_TURN_SCALE_MIN) * blend
        )


    def _scale_player_projectile_screen_x(self, x: chex.Array, projectile_ttl: chex.Array):
        x = x.astype(jnp.float32)
        scale = self._player_projectile_visual_turn_scale(projectile_ttl)
        return self.consts.TARGET_INDICATOR_POS_X + (x - self.consts.TARGET_INDICATOR_POS_X) * scale


    def _player_projectile_collision_check(self, enemies: Enemy, player_projectile: Projectile) -> bool:
        """
        Pure 3D axis-aligned box around the enemy vs projectile position (no screen projection).
        """
        is_air = jnp.logical_or(
            enemies.enemy_type == EnemyType.SAUCER,
            enemies.enemy_type == EnemyType.FIGHTER_JET,
        )
        s = jnp.where(
            is_air,
            jnp.float32(self.consts.PLAYER_SHOT_VS_AIR_ENEMY_HITBOX_SIZE / 2.0),
            jnp.float32(self.consts.HITBOX_SIZE / 2.0),
        )

        min_x = enemies.x - s
        max_x = enemies.x + s
        min_z = enemies.z - s
        max_z = enemies.z + s

        hit_x = (player_projectile.x >= min_x) & (player_projectile.x <= max_x)
        hit_z = (player_projectile.z >= min_z) & (player_projectile.z <= max_z)

        return jnp.all(
            jnp.stack(
                [
                    hit_x,
                    hit_z,
                    enemies.active,
                    player_projectile.active,
                    enemies.z > 0,
                ]
            )
        )


    def _enemy_projectile_collision_check(self, obj: Projectile):
        """
        Axis-aligned check for enemy shot vs player (not pixel masks).
        Uses the tight 2x2 player hitbox for consistent dodging.
        """
        half_w = self.consts.PLAYER_HITBOX_X / 2.0
        half_d = self.consts.PLAYER_HITBOX_Z / 2.0

        # Player is always physically located at x=0, z=0 in absolute world space
        distx = jnp.abs(obj.x) <= half_w
        distz = (obj.z <= half_d) & (obj.z >= -half_d)

        return jnp.all(jnp.stack([distx, distz, obj.active]))


    def _get_distance(self, x, z):
        distance = jnp.sqrt(x ** 2 + z ** 2)
        # Room for distance specific actions
        return distance

    def _get_distance_stage(self, distance: chex.Array) -> chex.Array:
        # Returns 0 (closest) up to 8 (farthest)
        return jnp.sum(distance >= self.consts.ENEMY_STAGE_THRESHOLDS)

    def _apply_enemy_min_distance_boundary(
        self, enemy: Enemy, player_direction: chex.Array, drift_enabled: chex.Array
    ) -> Enemy:
        """
        When radial distance drops below ENEMY_MIN_WORLD_DISTANCE, clamp onto that circle and
        optionally add a small tangent drift (player driving forward into the boundary).
        Keeps world z away from 0 so projection u = f*x/z stays stable.
        """
        rel_x = enemy.x
        rel_z = enemy.z
        dist = self._get_distance(rel_x, rel_z)
        eps = jnp.float32(1e-6)
        safe_dist = jnp.maximum(dist, eps)
        ux = rel_x / safe_dist
        uz = rel_z / safe_dist
        perp_x = uz
        perp_z = -ux

        too_close = jnp.logical_and(
            jnp.logical_and(enemy.active, dist < self.consts.ENEMY_MIN_WORLD_DISTANCE),
            enemy.enemy_type != EnemyType.FIGHTER_JET,
        )

        forward = jnp.logical_and(
            drift_enabled > 0,
            jnp.any(
                jnp.stack(
                    [
                        player_direction[1],
                        player_direction[5],
                        player_direction[6],
                    ]
                )
            ),
        )
        drift_mag = jnp.where(too_close & forward, self.consts.ENEMY_NEAR_DRIFT_SPEED, jnp.float32(0.0))

        clamp_x = ux * self.consts.ENEMY_MIN_WORLD_DISTANCE + perp_x * drift_mag
        clamp_z = uz * self.consts.ENEMY_MIN_WORLD_DISTANCE + perp_z * drift_mag

        new_x = jnp.where(too_close, clamp_x, rel_x)
        new_z = jnp.where(too_close, clamp_z, rel_z)
        dpx = jnp.where(too_close, new_x - rel_x, jnp.float32(0.0))
        dpz = jnp.where(too_close, new_z - rel_z, jnp.float32(0.0))
        delta = jnp.array([dpx, dpz], dtype=jnp.float32)

        return enemy.replace(
            x=new_x,
            z=new_z,
            distance=self._get_distance(new_x, new_z),
            point_store_1_temp=enemy.point_store_1_temp + delta,
            point_store_2_temp=enemy.point_store_2_temp + delta,
        )


    def _mask_spawn_weights_for_singletons(
        self,
        weights: chex.Array,
        has_saucer: chex.Array,
        has_fighter: chex.Array,
        has_supertank: chex.Array,
    ) -> chex.Array:
        """Zero out weights for enemy classes already present; renormalize (fallback: TANK)."""
        w = weights
        w = w.at[EnemyType.SAUCER].set(jnp.where(has_saucer, jnp.float32(0.0), w[EnemyType.SAUCER]))
        w = w.at[EnemyType.FIGHTER_JET].set(jnp.where(has_fighter, jnp.float32(0.0), w[EnemyType.FIGHTER_JET]))
        w = w.at[EnemyType.SUPERTANK].set(jnp.where(has_supertank, jnp.float32(0.0), w[EnemyType.SUPERTANK]))
        s = jnp.sum(w)
        w_norm = jnp.where(s > jnp.float32(0.0), w / s, jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32))
        return w_norm

    @partial(jax.jit, static_argnums=(0,))
    def spawn_enemy(
        self,
        key,
        enemy: Enemy,
        score,
        spawn_idx,
        total_spawned_count,
        has_saucer: chex.Array,
        has_fighter: chex.Array,
        has_supertank: chex.Array,
    ):

        def do_spawn(args):
            enemy, key, score, spawn_idx, total_spawned_count, has_saucer, has_fighter, has_supertank = args
            tier_idx = jnp.sum(score >= self.consts.DIFFICULTY_THRESHOLDS)
            weights = self._mask_spawn_weights_for_singletons(
                self.consts.DIFFICULTY_SPAWN_WEIGHTS[tier_idx],
                has_saucer,
                has_fighter,
                has_supertank,
            )
            key, k_type, k_dist, k_theta, k_orient, k_saucer = jax.random.split(key, 6)
            enemy_type = jax.random.choice(k_type, a=len(EnemyType), p=weights)
            is_3rd_or_4th = jnp.logical_or(spawn_idx == 2, spawn_idx == 3)
            override_saucer = jnp.logical_and(
                jnp.logical_and(is_3rd_or_4th, jax.random.uniform(k_saucer) < 0.75),
                jnp.logical_not(has_saucer),
            )
            enemy_type = jnp.where(override_saucer, EnemyType.SAUCER, enemy_type)
            first_type = jnp.int32(self.consts.FIRST_SPAWN_TYPE)
            first_type_valid = jnp.logical_and(first_type >= 0, first_type < len(EnemyType))
            ft_first = first_type
            ft_first = jnp.where(
                jnp.logical_and(ft_first == EnemyType.SAUCER, has_saucer),
                jnp.int32(EnemyType.TANK),
                ft_first,
            )
            ft_first = jnp.where(
                jnp.logical_and(ft_first == EnemyType.FIGHTER_JET, has_fighter),
                jnp.int32(EnemyType.TANK),
                ft_first,
            )
            ft_first = jnp.where(
                jnp.logical_and(ft_first == EnemyType.SUPERTANK, has_supertank),
                jnp.int32(EnemyType.TANK),
                ft_first,
            )
            enemy_type = jnp.where(
                jnp.logical_and(spawn_idx == total_spawned_count, first_type_valid),
                ft_first,
                enemy_type,
            )

            def spawn_fighter():
                tcx = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_X)
                tcz = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_Z)
                x = jax.random.uniform(k_dist, minval=self.consts.FIGHTER_AREA_X[0], maxval=self.consts.FIGHTER_AREA_X[1])
                z = jax.random.uniform(k_theta, minval=self.consts.FIGHTER_AREA_Z[0], maxval=self.consts.FIGHTER_AREA_Z[1])
                distance = self._get_distance(x - tcx, z - tcz)
                perfect_angle = (2 * jnp.pi - jnp.arctan2(x - tcx, z - tcz)) % (2 * jnp.pi)

                return enemy.replace(
                    x=x,
                    z=z,
                    distance=distance,
                    enemy_type=enemy_type,
                    orientation_angle=perfect_angle,
                    active=True,
                    shoot_cd=self.consts.ENEMY_SHOOT_CDS[enemy_type],
                    phase=0,
                    dist_moved_temp=0.0,
                    point_store_1_temp=jnp.zeros((2,), dtype=jnp.float32),
                    point_store_2_temp=jnp.zeros((2,), dtype=jnp.float32),
                )

            def spawn_default():
                is_first = spawn_idx == 0

                th = self.consts.ENEMY_STAGE_THRESHOLDS
                near = self.consts.SPAWN_STAGE_OFFSET_FROM_NEAR
                far = self.consts.SPAWN_STAGE_OFFSET_FROM_FAR
                min_d = th[near - 1]
                max_d = th[th.shape[0] - far]
                u = jax.random.uniform(k_dist, minval=0.0, maxval=1.0)
                rand_dist = min_d + u * (max_d - min_d)
                distance = jnp.where(is_first, 50.0, rand_dist)

                k_az_fov_bernoulli, k_az_fov_jit, k_az_else, k_az_saucer = jax.random.split(k_theta, 4)
                rand_full = jax.random.uniform(k_az_saucer, minval=0.0, maxval=2 * jnp.pi)
                in_fov = jax.random.uniform(k_az_fov_bernoulli, minval=0.0, maxval=1.0) < jnp.float32(
                    self.consts.SPAWN_TANK_FOV_PROB
                )
                forward = jnp.float32(jnp.pi / 2.0)
                theta_fov = (
                    forward
                    + jax.random.uniform(
                        k_az_fov_jit,
                        minval=-self.consts.SPAWN_TANK_FOV_ARC_RAD,
                        maxval=self.consts.SPAWN_TANK_FOV_ARC_RAD,
                    )
                ) % (2 * jnp.pi)
                theta_elsewhere = jax.random.uniform(k_az_else, minval=0.0, maxval=2 * jnp.pi)
                is_tank_like = jnp.logical_or(
                    enemy_type == EnemyType.TANK,
                    enemy_type == EnemyType.SUPERTANK,
                )
                theta_tank = jnp.where(in_fov, theta_fov, theta_elsewhere)
                theta_pick = jnp.where(is_tank_like, theta_tank, rand_full)
                theta = jnp.where(is_first, jnp.pi / 2, theta_pick)

                # Calculate coordinates first
                tcx = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_X)
                tcz = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_Z)
                new_x = distance * jnp.cos(theta)
                new_z = distance * jnp.sin(theta)

                perfect_angle = (2 * jnp.pi - jnp.arctan2(new_x - tcx, new_z - tcz)) % (2 * jnp.pi)

                # Randomly pick left or right for sideways spawns
                sideways_dir = jnp.where(jax.random.uniform(k_orient) > 0.5, jnp.pi / 2, -jnp.pi / 2)
                sideways_angle = (perfect_angle + sideways_dir) % (2 * jnp.pi)

                # Every 3rd enemy (e.g., indices 2, 5, 8...) spawns directly facing the player
                is_facing_player = (spawn_idx % 3) == 2

                orientation_angle = jnp.where(
                    is_first,
                    jnp.pi / 2,  # First enemy always spawns predictably
                    jnp.where(is_facing_player, perfect_angle, sideways_angle),
                )

                return enemy.replace(
                    x=new_x,
                    z=new_z,
                    distance=distance,
                    enemy_type=enemy_type,
                    orientation_angle=orientation_angle,
                    active=True,
                    shoot_cd=self.consts.ENEMY_SHOOT_CDS[enemy_type],
                    phase=0,
                    dist_moved_temp=0.0,
                    point_store_1_temp=jnp.zeros((2,), dtype=jnp.float32),
                    point_store_2_temp=jnp.zeros((2,), dtype=jnp.float32),
                )

            return jax.lax.cond(
                enemy_type == EnemyType.FIGHTER_JET,
                spawn_fighter,
                spawn_default
            )

        is_slot_ready = jnp.logical_and(~enemy.active, enemy.death_anim_counter <= 0)
        return jax.lax.cond(
            is_slot_ready,
            do_spawn,
            lambda x: x[0],
            (enemy, key, score, spawn_idx, total_spawned_count, has_saucer, has_fighter, has_supertank),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_all_ready_slots(
        self,
        key: chex.PRNGKey,
        enemies: Enemy,
        score: chex.Array,
        spawn_indices: chex.Array,
        total_spawned_count: chex.Array,
        is_slot_ready: chex.Array,
    ) -> Enemy:
        """Spawn into ready slots in slot index order so at most one SAUCER / FIGHTER_JET / SUPERTANK exists."""
        has_saucer = jnp.any(jnp.logical_and(enemies.active, enemies.enemy_type == EnemyType.SAUCER))
        has_fighter = jnp.any(jnp.logical_and(enemies.active, enemies.enemy_type == EnemyType.FIGHTER_JET))
        has_supertank = jnp.any(jnp.logical_and(enemies.active, enemies.enemy_type == EnemyType.SUPERTANK))
        n = enemies.active.shape[0]

        def body(carry, i):
            hs, hf, hst, k = carry
            ki, k = jax.random.split(k)
            ei = jax.tree.map(lambda a: a[i], enemies)
            new_e = self.spawn_enemy(
                ki, ei, score, spawn_indices[i], total_spawned_count, hs, hf, hst
            )
            ready = is_slot_ready[i]
            hs = jnp.logical_or(hs, jnp.logical_and(ready, new_e.enemy_type == EnemyType.SAUCER))
            hf = jnp.logical_or(hf, jnp.logical_and(ready, new_e.enemy_type == EnemyType.FIGHTER_JET))
            hst = jnp.logical_or(hst, jnp.logical_and(ready, new_e.enemy_type == EnemyType.SUPERTANK))
            return (hs, hf, hst, k), new_e

        init_carry = (has_saucer, has_fighter, has_supertank, key)
        _, stacked = jax.lax.scan(body, init_carry, jnp.arange(n, dtype=jnp.int32))
        return stacked


    @partial(jax.jit, static_argnums=(0,))
    def _enemy_under_target_crosshair(self, enemy: Enemy) -> chex.Array:
        """True when render_targeting_indicator would use the active (yellow) color for this enemy."""
        e_u, _ = self.world_cords_to_viewport_cords_arr(
            enemy.x, enemy.z, self.consts.CAMERA_FOCAL_LENGTH
        )
        stage = jnp.sum(enemy.distance >= self.consts.ENEMY_STAGE_THRESHOLDS)
        stage = jnp.clip(stage, 0, 8)

        n = self.renderer.padded_enemy_masks.shape[1]
        angle = self.renderer._view_angle_for_enemy_sprite(enemy)
        rot_index = jnp.round((angle / jnp.pi) * (n - 1)).astype(jnp.int32)
        rot_index = jnp.clip(rot_index, 0, n - 1)

        mask = self.renderer.PRECOMPUTED_SCALES[enemy.enemy_type, rot_index, stage]
        _, mask_w = mask.shape

        target_x = self.consts.TARGET_INDICATOR_POS_X
        draw_x = e_u - (mask_w // 2)

        local_x = target_x - draw_x.astype(jnp.int32)
        in_bounds_x = (local_x >= 0) & (local_x < mask_w)

        safe_x = jnp.clip(local_x, 0, mask_w - 1)
        column_has_solid = jnp.any(mask[:, safe_x] != self.renderer.jr.TRANSPARENT_ID)

        return enemy.active & (enemy.z > 0) & in_bounds_x & column_has_solid

    # -------------Enemy Movements-----------------
    @partial(jax.jit, static_argnums=(0,))
    def enemy_movement(
        self,
        enemy: Enemy,
        projectile: Projectile,
        score: jnp.int32,
        player_projectile: Projectile,
    ):

        tcx = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_X)
        tcz = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_Z)
        perfect_angle = (2 * jnp.pi - jnp.arctan2(enemy.x - tcx, enemy.z - tcz)) % (2 * jnp.pi)
        angle_diff = (perfect_angle - enemy.orientation_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

        tier_idx = jnp.sum(score >= self.consts.DIFFICULTY_THRESHOLDS)
        speed = self.consts.ENEMY_SPEED[enemy.enemy_type] * self.consts.DIFFICULTY_SPEED_MULT[tier_idx]
        rot_speed = self.consts.ENEMY_ROT_SPEED[enemy.enemy_type] * self.consts.DIFFICULTY_ROT_SPEED_MULT[tier_idx]

        # --- DODGE DETECTION (player bullet vs this enemy) ---
        proj_dx = -jnp.sin(player_projectile.orientation_angle)
        proj_dz = jnp.cos(player_projectile.orientation_angle)
        vec_x = enemy.x - player_projectile.x
        vec_z = enemy.z - player_projectile.z
        forward_dist = proj_dx * vec_x + proj_dz * vec_z
        orth_dist = jnp.abs(proj_dx * vec_z - proj_dz * vec_x)
        is_dodging = jnp.all(
            jnp.stack(
                [
                    player_projectile.active,
                    forward_dist > 0.0,
                    forward_dist < enemy.distance + 10.0,
                    orth_dist > self.consts.ENEMY_DODGE_MIN_DIST,
                    orth_dist < self.consts.ENEMY_DODGE_MAX_DIST,
                ]
            )
        )
        dodge_dir = jnp.sign(proj_dx * vec_z - proj_dz * vec_x)

        pointing_at = self._enemy_under_target_crosshair(enemy)

        # Determine if the enemy is looking straight at the player (with slight tolerance)
        is_facing_straight = jnp.abs(angle_diff) <= rot_speed * 1.5
        enemy_stage = self._get_distance_stage(enemy.distance)

        # ---------Helper Functions---------

        def move_to_direction(enemy: Enemy, angle: float, towards: int = -1) -> Enemy:
            """Enemy towards direction with -1 and away with 1"""

            direction_x = -jnp.sin(angle)
            direction_z = jnp.cos(angle)
            new_x = enemy.x + direction_x * towards * speed
            new_z = enemy.z + direction_z * towards * speed

            enemy = enemy.replace(x=new_x, z=new_z, distance=self._get_distance(new_x, new_z))
            enemy = self._wrap_coords_and_stored_points(enemy)

            return enemy

        def move_orthogonal_to_direction(enemy: Enemy, angle: float, direction: int = 1) -> Enemy:
            """Enemy right of direction with -1 and left with 1 (from direction's POV)"""

            ortho_angle = (angle + jnp.sign(direction) * (jnp.pi/2)) % (2*jnp.pi)
            direction_x = -jnp.sin(ortho_angle)
            direction_z = jnp.cos(ortho_angle)
            new_x = enemy.x + direction_x * speed * jnp.abs(direction)
            new_z = enemy.z + direction_z * speed * jnp.abs(direction)

            enemy = enemy.replace(x=new_x, z=new_z, distance=self._get_distance(new_x, new_z))
            enemy = self._wrap_coords_and_stored_points(enemy)

            return enemy

        def move_to_player(enemy: Enemy, towards: int = -1) -> Enemy:
            """Enemy towards player with -1 and away with 1"""
            return move_to_direction(enemy, perfect_angle, towards)

        def move_orthogonal_to_player(enemy: Enemy, direction: int = 1) -> Enemy:
            """Enemy right of player with -1 and left with 1 (from player's POV)"""
            return move_orthogonal_to_direction(enemy, perfect_angle, direction)

        def enemy_turn(enemy: Enemy) -> Enemy:
            return enemy.replace(orientation_angle=enemy.orientation_angle + jnp.sign(angle_diff) * rot_speed)

        def shoot_projectile(args) -> Tuple[Enemy, Projectile]:

            enemy, projectile = args
            new_enemy = enemy.replace(shoot_cd=self.consts.ENEMY_SHOOT_CDS[enemy.enemy_type])
            tcx = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_X)
            tcz = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_Z)
            dx = tcx - enemy.x
            dz = tcz - enemy.z
            dist_to_target = jnp.sqrt(dx * dx + dz * dz + jnp.float32(1e-8))
            spawn = jnp.float32(self.consts.ENEMY_PROJECTILE_SPAWN_OFFSET + 1.0)
            return new_enemy, projectile.replace(
                orientation_angle=perfect_angle,
                x=enemy.x + (dx / dist_to_target) * spawn,
                z=enemy.z + (dz / dist_to_target) * spawn,
                active=True,
                time_to_live=jnp.array(self.consts.PROJECTILE_TTL, dtype=jnp.int32),
            )


        # ---------------------------------
        ## Tank
        def tank_movement(tank: Enemy) -> Enemy:
            tank = jax.lax.cond(
                is_dodging & (tank.phase != 4),
                lambda t: t.replace(phase=4, dist_moved_temp=dodge_dir),
                lambda t: t,
                tank,
            )

            def p0(tank):
                return tank.replace(phase=1)

            def p1(tank):
                tank = enemy_turn(tank)
                return jax.lax.cond(is_facing_straight, lambda t: t.replace(phase=2), lambda t: t, tank)

            def p2(tank):
                return jax.lax.cond(
                    is_facing_straight,
                    lambda t: jax.lax.cond(
                        t.distance > self.consts.TANKS_SHOOTING_DISTANCE,
                        lambda t2: move_to_player(t2, -1),
                        lambda t2: t2.replace(phase=3),
                        t,
                    ),
                    lambda t: t.replace(phase=1),
                    tank,
                )

            def p3(tank):
                return jax.lax.cond(tank.shoot_cd <= 0, lambda t: t.replace(phase=1), lambda t: t, tank)

            def p4(tank):
                evade_angle = (player_projectile.orientation_angle + tank.dist_moved_temp * jnp.pi / 2) % (
                    2 * jnp.pi
                )
                angle_diff_ev = (evade_angle - tank.orientation_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
                tank = tank.replace(
                    orientation_angle=tank.orientation_angle + jnp.sign(angle_diff_ev) * rot_speed * 2.0
                )
                tank = jax.lax.cond(
                    jnp.abs(angle_diff_ev) < 0.5,
                    lambda t: move_to_direction(t, t.orientation_angle, -1),
                    lambda t: t,
                    tank,
                )
                return jax.lax.cond(is_dodging, lambda t: t, lambda t: t.replace(phase=1), tank)

            return jax.lax.switch(tank.phase, (p0, p1, p2, p3, p4), tank)


        def saucer_movement(saucer: Enemy) -> Enemy:
            # Stable periodic strafing: move left/right in fixed intervals.
            # Unlike tanks, the saucer actively flees the player and despawns.
            strafe_duration = jnp.float32(14.0)

            def p0(s):
                init_dir = jnp.where(jnp.abs(s.x) < 1e-3, jnp.float32(1.0), jnp.sign(s.x))
                return s.replace(
                    phase=1,
                    dist_moved_temp=0.0,
                    point_store_1_temp=jnp.array([init_dir, 0.0], dtype=jnp.float32),
                    point_store_2_temp=jnp.zeros((2,), dtype=jnp.float32),
                )

            def p1(s):
                cd = s.point_store_2_temp[0]
                should_flip = jnp.logical_and(
                    pointing_at,
                    cd <= 0.0,
                )
                new_cd = jnp.where(
                    should_flip,
                    jnp.float32(self.consts.SAUCER_HOVER_DIRECTION_FLIP_COOLDOWN_FRAMES),
                    jnp.maximum(cd - 1.0, 0.0),
                )
                s = s.replace(point_store_2_temp=s.point_store_2_temp.at[0].set(new_cd))
                s = jax.lax.cond(
                    should_flip,
                    lambda t: t.replace(point_store_1_temp=t.point_store_1_temp.at[0].set(-t.point_store_1_temp[0])),
                    lambda t: t,
                    s,
                )

                strafe_dir = jnp.where(
                    jnp.abs(s.point_store_1_temp[0]) < 1e-3,
                    jnp.float32(1.0),
                    jnp.sign(s.point_store_1_temp[0]),
                )

                s = move_orthogonal_to_player(s, strafe_dir)

                # --- Saucer actively tries to get away ---
                # A positive value passed to move_to_player pushes the enemy AWAY from the origin
                s = move_to_player(s, 0.4)

                # Despawn if it exceeds the radar scan radius so the game can spawn a new enemy
                s = jax.lax.cond(
                    s.distance > self.consts.RADAR_MAX_SCAN_RADIUS + 5.0,
                    lambda t: t.replace(active=False),
                    lambda t: t,
                    s,
                )

                s = s.replace(dist_moved_temp=s.dist_moved_temp + speed)
                return jax.lax.cond(
                    s.dist_moved_temp >= strafe_duration,
                    lambda t: t.replace(
                        dist_moved_temp=0.0,
                        point_store_1_temp=t.point_store_1_temp.at[0].set(-strafe_dir),
                    ),
                    lambda t: t,
                    s,
                )

            return jax.lax.switch(saucer.phase, (p0, p1), saucer)


        def fighter_movement(fighter: Enemy) -> Enemy:
            def p0(fighter: Enemy):
                """right after spawning: setup"""
                return fighter.replace(
                    phase=1,
                    dist_moved_temp=0.0,  # float index along the 115-sample ALE sequence
                    point_store_1_temp=jnp.array([0.0, 0.0]),
                    point_store_2_temp=jnp.array([0.0, 0.0]),
                    shoot_cd=0,
                )

            def p1(fighter):
                """Deterministic movement with linear interpolation, switching to chase if outrun."""
                current_idx = fighter.dist_moved_temp
                bomber_speed = self.consts.FIGHTER_ANIM_SPEED * self.consts.DIFFICULTY_SPEED_MULT[tier_idx]
                next_idx = current_idx + bomber_speed

                indices = jnp.arange(115, dtype=jnp.float32)

                # 1. Clamp indices so we don't read out of bounds if the sequence is exhausted
                safe_current_idx = jnp.minimum(current_idx, 114.0)
                safe_next_idx = jnp.minimum(next_idx, 114.0)

                dist_current = jnp.interp(safe_next_idx, indices, self.consts.FIGHTER_DIST_SEQ)
                dist_prev = jnp.interp(safe_current_idx, indices, self.consts.FIGHTER_DIST_SEQ)

                lateral_current = jnp.interp(safe_next_idx, indices, self.consts.FIGHTER_LATERAL_SEQ)
                lateral_prev = jnp.interp(safe_current_idx, indices, self.consts.FIGHTER_LATERAL_SEQ)

                seq_forward_move = dist_prev - dist_current
                seq_lateral_move = (lateral_current - lateral_prev) * self.consts.FIGHTER_DRIFT_SCALE

                # 2. Catch-up: sequence exhausted → fly forward at base speed, no lateral from seq
                is_seq_exhausted = current_idx >= 114.0
                constant_forward_speed = self.consts.ENEMY_SPEED[EnemyType.FIGHTER_JET] * self.consts.DIFFICULTY_SPEED_MULT[tier_idx]

                forward_move = jnp.where(is_seq_exhausted, constant_forward_speed, seq_forward_move)
                lateral_move = jnp.where(is_seq_exhausted, jnp.float32(0.0), seq_lateral_move)

                dir_x = -jnp.sin(fighter.orientation_angle)
                dir_z = jnp.cos(fighter.orientation_angle)

                ortho_angle = (fighter.orientation_angle + jnp.pi / 2) % (2 * jnp.pi)
                ortho_x = -jnp.sin(ortho_angle)
                ortho_z = jnp.cos(ortho_angle)

                new_x = fighter.x - dir_x * forward_move + ortho_x * lateral_move
                new_z = fighter.z - dir_z * forward_move + ortho_z * lateral_move

                fighter = fighter.replace(
                    x=new_x,
                    z=new_z,
                    distance=self._get_distance(new_x, new_z),
                    dist_moved_temp=next_idx,
                )
                fighter = self._wrap_coords_and_stored_points(fighter)

                # 3. Despawn only after passing close enough, or after a long chase (safety)
                despawn_distance = jnp.maximum(jnp.float32(0.0), self.consts.FIGHTER_SHOOTING_DISTANCE - 10.0)
                should_despawn = (fighter.distance <= despawn_distance) | (fighter.dist_moved_temp > 400.0)

                fighter = jax.lax.cond(
                    should_despawn,
                    lambda f: f.replace(active=False),
                    lambda f: f,
                    fighter,
                )

                return fighter

            return jax.lax.cond(fighter.phase == 0, p0, p1, fighter)

        def supertank_movement(supertank: Enemy) -> Enemy:
            supertank = jax.lax.cond(
                is_dodging & (supertank.phase != 4),
                lambda t: t.replace(phase=4, dist_moved_temp=dodge_dir),
                lambda t: t,
                supertank,
            )

            def p0(st):
                dir_val = jnp.where(jnp.abs(st.x) < 1e-3, jnp.float32(1.0), jnp.sign(st.x))
                return st.replace(phase=1, dist_moved_temp=dir_val)

            def p1(st):
                strafe_angle = (perfect_angle + st.dist_moved_temp * jnp.pi / 2) % (2 * jnp.pi)
                angle_diff_st = (strafe_angle - st.orientation_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
                st = st.replace(orientation_angle=st.orientation_angle + jnp.sign(angle_diff_st) * rot_speed)
                return jax.lax.cond(
                    jnp.abs(angle_diff_st) <= rot_speed * 1.5,
                    lambda t: t.replace(phase=2),
                    lambda t: t,
                    st,
                )

            def p2(st):
                st = move_to_direction(st, st.orientation_angle, -1)
                return jax.lax.cond(
                    st.shoot_cd <= 0,
                    lambda t: t.replace(
                        phase=5,
                        point_store_2_temp=t.point_store_2_temp.at[1].set(jnp.float32(100.0)),
                    ),
                    lambda t: t,
                    st,
                )

            def p3(st):
                st = st.replace(orientation_angle=perfect_angle)
                return jax.lax.cond(
                    st.shoot_cd > 0,
                    lambda t: t.replace(phase=1, dist_moved_temp=-t.dist_moved_temp),
                    lambda t: t,
                    st,
                )

            def p4(st):
                evade_angle = (player_projectile.orientation_angle + st.dist_moved_temp * jnp.pi / 2) % (
                    2 * jnp.pi
                )
                angle_diff_ev = (evade_angle - st.orientation_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
                st = st.replace(
                    orientation_angle=st.orientation_angle + jnp.sign(angle_diff_ev) * rot_speed * 2.0
                )
                st = jax.lax.cond(
                    jnp.abs(angle_diff_ev) < 0.5,
                    lambda t: move_to_direction(t, t.orientation_angle, -1),
                    lambda t: t,
                    st,
                )
                return jax.lax.cond(is_dodging, lambda t: t, lambda t: t.replace(phase=1), st)

            def p5(st):
                angle_diff_aim = (perfect_angle - st.orientation_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
                st = st.replace(
                    orientation_angle=st.orientation_angle + jnp.sign(angle_diff_aim) * rot_speed
                )
                return jax.lax.cond(
                    jnp.abs(angle_diff_aim) <= rot_speed * 1.5,
                    lambda t: t.replace(phase=6),
                    lambda t: t,
                    st,
                )

            def p6(st):
                angle_diff_ch = (perfect_angle - st.orientation_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
                st = st.replace(
                    orientation_angle=st.orientation_angle + jnp.sign(angle_diff_ch) * rot_speed
                )
                charge_speed = speed * jnp.float32(0.6)
                direction_x = -jnp.sin(st.orientation_angle)
                direction_z = jnp.cos(st.orientation_angle)
                new_x = st.x + direction_x * (-1) * charge_speed
                new_z = st.z + direction_z * (-1) * charge_speed
                st = st.replace(x=new_x, z=new_z, distance=self._get_distance(new_x, new_z))
                st = self._wrap_coords_and_stored_points(st)
                counter = st.point_store_2_temp[1] - 1.0
                st = st.replace(point_store_2_temp=st.point_store_2_temp.at[1].set(counter))
                return jax.lax.cond(counter <= 0, lambda t: t.replace(phase=3), lambda t: t, st)

            return jax.lax.switch(supertank.phase, (p0, p1, p2, p3, p4, p5, p6), supertank)

        shoot_cond = jnp.all(
            jnp.array(
                [
                    enemy.enemy_type != EnemyType.SAUCER,
                    enemy.shoot_cd <= 0,
                    enemy.active,
                    enemy_stage == 0,
                ]
            )
        ) & (
            jnp.all(
                jnp.array(
                    [
                        enemy.enemy_type == EnemyType.TANK,
                        is_facing_straight,
                        enemy.distance <= self.consts.TANKS_SHOOTING_DISTANCE,
                    ]
                )
            )
            | jnp.all(
                jnp.array(
                    [
                        enemy.enemy_type == EnemyType.SUPERTANK,
                        enemy.phase == 3,
                    ]
                )
            )
            | jnp.all(
                jnp.array(
                    [
                        enemy.enemy_type == EnemyType.FIGHTER_JET,
                        enemy.distance <= self.consts.FIGHTER_SHOOTING_DISTANCE,
                    ]
                )
            )
        )

        new_enemy, new_projectile = jax.lax.cond(shoot_cond, shoot_projectile, lambda x: x, (enemy, projectile))
        return (
            jax.lax.switch(
                enemy.enemy_type,
                (tank_movement, saucer_movement, fighter_movement, supertank_movement),
                new_enemy,
            ),
            new_projectile,
        )

    # ---------------------------------------------


    def render(self, state: BattlezoneState) -> jnp.ndarray:
        return self.renderer.render(state)

    def player_shot_reset(self, state:BattlezoneState) -> BattlezoneState:
        """reset function for when the player was shot but still has remaining lives"""

        split_key, key = jax.random.split(state.random_key, 2)
        # Set enemies to inactive
        inactive_enemies = state.enemies.replace(active=jnp.zeros_like(state.enemies.active))
        is_slot_ready = jnp.logical_and(~inactive_enemies.active, inactive_enemies.death_anim_counter <= 0)
        num_spawning = jnp.sum(is_slot_ready.astype(jnp.int32))
        spawn_indices = state.total_spawned_count + jnp.cumsum(is_slot_ready.astype(jnp.int32)) - 1

        new_state = state.replace(
            shot_spawn=jnp.ones_like(state.shot_spawn),
            life=state.life - 1,
            enemies=self._spawn_all_ready_slots(
                split_key,
                inactive_enemies,
                state.score,
                spawn_indices,
                state.total_spawned_count,
                is_slot_ready,
            ),
            player_projectile=Projectile(
                x=jnp.array(0, dtype=jnp.float32),
                z=jnp.array(0, dtype=jnp.float32),
                orientation_angle=jnp.array(0, dtype=jnp.float32),
                active=jnp.array(False, dtype=jnp.bool),
                distance=jnp.array(0, dtype=jnp.float32),
                time_to_live=jnp.array(0, dtype=jnp.int32)
            ),
            enemy_projectiles=Projectile(
                x=jnp.array([0, 0], dtype=jnp.float32),
                z=jnp.array([0, 0], dtype=jnp.float32),
                orientation_angle=jnp.array([0, 0], dtype=jnp.float32),
                active=jnp.array([False, False], dtype=jnp.bool),
                distance=jnp.array([0, 0], dtype=jnp.float32),
                time_to_live=jnp.array([0, 0], dtype=jnp.int32)
            ),
            random_key=key,
            cur_fire_cd=jnp.array(0, dtype=jnp.int32),
            total_spawned_count=state.total_spawned_count + num_spawning,
        )

        return new_state


    def world_cords_to_viewport_cords_arr(self, x, z, f):

        u = ((f * (x / z)) + self.consts.WIDTH/2).astype(jnp.int32)
        vOffset = self.consts.HORIZON_Y
        v = ((f / (z - self.consts.PLAYER_Z_PLANE)) + vOffset).astype(jnp.int32)

        return u, v


    def check_in_radar(self, enemies: Enemy) -> chex.Array:
        return jnp.logical_and((enemies.distance <= self.consts.RADAR_MAX_SCAN_RADIUS), enemies.active)


    def _get_observation(self, state: BattlezoneState):

        #-------------------------------enemies----------------------------------------------
        enemies_u, _ = self.world_cords_to_viewport_cords_arr(state.enemies.x, state.enemies.z,
                                                                  self.consts.CAMERA_FOCAL_LENGTH)
        zoom_factor = jnp.clip(((-0.15 * (state.enemies.distance) + 21.0) / 20.0), 0.0, 1.0)
        pixels_deleted_due_to_zoom = (jnp.round(1.0 / zoom_factor) + 1)
        enemies_width = self.consts.ENEMY_WIDTHS[state.enemies.enemy_type] - pixels_deleted_due_to_zoom
        enemies_heights = self.consts.ENEMY_HEIGHTS[state.enemies.enemy_type] - pixels_deleted_due_to_zoom
        enemies_visible = jnp.logical_and(
            state.enemies.z > 0,
            jnp.logical_and(
                enemies_u < (self.consts.WIDTH + enemies_width // 2),
                enemies_u > (0 - enemies_width // 2)
            )
        )
        enemy_mask = jnp.logical_and(state.enemies.active, enemies_visible)
        enemies_u = enemies_u - (enemies_width / 2)
        enemies = ObjectObservation.create(
            x=jnp.clip(enemies_u, 0, self.consts.WIDTH),
            y=jnp.clip(
                jnp.full(
                    (len(enemies_u),),
                    self.consts.ENEMY_POS_Y - (enemies_heights / 2)
                ),
                0,
                self.consts.HEIGHT
            ),
            width = enemies_width,
            height = enemies_heights,
            active = enemy_mask
        )

        #---------------------------------projectiles------------------------------------------------
        enemy_projectiles_u, enemy_projectiles_v = self.world_cords_to_viewport_cords_arr(
            state.enemy_projectiles.x,
            state.enemy_projectiles.z,
            self.consts.CAMERA_FOCAL_LENGTH
        )
        enemy_projectiles_visible = jnp.logical_and(
            state.enemies.z > 0,
            jnp.logical_and(
                enemies_u < self.consts.WIDTH,
                enemies_u > 0
            )
        )
        enemy_projectiles_mask = jnp.logical_and(enemy_projectiles_visible, state.enemy_projectiles.active)
        player_projectiles_u, _ = self.world_cords_to_viewport_cords_arr(
            state.player_projectile.x,
            state.player_projectile.z,
            self.consts.CAMERA_FOCAL_LENGTH
        )
        # Visual-only drift tuning: compress lateral displacement around reticle
        # to match ALE without changing projectile world physics/collision.
        player_projectiles_u = self._scale_player_projectile_screen_x(
            player_projectiles_u, state.player_projectile.time_to_live
        )
        idx = jnp.clip(
            self.consts.PLAYER_PROJECTILE_TTL - state.player_projectile.time_to_live,
            0,
            self.consts.PLAYER_BULLET_Y_SEQ.shape[0] - 1,
        )
        player_projectiles_v = self.consts.PLAYER_BULLET_Y_SEQ[idx]
        projectiles_x = jnp.concatenate([
            jnp.atleast_1d(player_projectiles_u - 1),
            enemy_projectiles_u - 1,
        ])
        projectiles_y = jnp.concatenate([
            jnp.atleast_1d(player_projectiles_v - 1),
            enemy_projectiles_v - 1
        ])
        player_width = jnp.where(player_projectiles_v <= 105, 1, 2)
        projectiles_w = jnp.concatenate([
            jnp.atleast_1d(player_width),
            jnp.full((len(enemy_projectiles_u),), 2),
        ])
        projectiles_active = jnp.concatenate([jnp.atleast_1d(state.player_projectile.active), enemy_projectiles_mask])
        projectiles = ObjectObservation.create(
            x=jnp.clip(projectiles_x, 0, self.consts.WIDTH),
            y=jnp.clip(projectiles_y, 0, self.consts.HEIGHT),
            width=projectiles_w,
            height=jnp.full(
                (len(projectiles_x),),
                3
            ),
            active=projectiles_active
        )

        #-----------------------------radar----------------------------------------
        # Check if enemy in radar radius
        in_radar = jax.vmap(self.check_in_radar, in_axes=(0))(state.enemies)

        # Scale to radar size
        scale_val = self.consts.RADAR_RADIUS / self.consts.RADAR_MAX_SCAN_RADIUS
        radar_enemies_x = state.enemies.x * scale_val
        radar_enemies_z = state.enemies.z * scale_val * (-1)

        # Offset to radar center
        radar_enemies_x = jnp.round(radar_enemies_x + self.consts.RADAR_CENTER_X).astype(jnp.int32)
        radar_enemies_z = jnp.round(radar_enemies_z + self.consts.RADAR_CENTER_Y).astype(jnp.int32)

        # Only allow in range enemies
        radar_dots = ObjectObservation.create(
            x=jnp.clip(radar_enemies_x, 0, self.consts.WIDTH),
            y=jnp.clip(radar_enemies_z, 0, self.consts.HEIGHT),
            width=jnp.full(
                (len(radar_enemies_x),),
                1
            ),
            height=jnp.full(
                (len(radar_enemies_x),),
                1
            ),
            active=in_radar
        )
        #----------------------------------------------------------------------------

        return BattlezoneObservation(
            enemies=enemies,
            radar_dots=radar_dots,
            projectiles=projectiles,
            score=jnp.array(state.score),
            life=jnp.array(state.life),
            enemy_types=jnp.where(enemy_mask, jnp.array(state.enemies.enemy_type), -1),
        )


    def observation_space(self) -> spaces.Dict:
        """description of observation (must match)"""
        object_space = spaces.get_object_space(n=2, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))
        projectile_object_space = spaces.get_object_space(n=3, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "enemies": object_space,
            "radar_dots": object_space,
            "projectiles": projectile_object_space,
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "life": spaces.Box(low=0, high=self.consts.MAX_LIVES, shape=(), dtype=jnp.int32),
            "enemy_types": spaces.Box(low=jnp.array([-1, -1]), high=jnp.array([4, 4]), shape=(2,), dtype=jnp.int32)
        })


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)  # [Noop, Up, Right, Left, Down, UpRight, UpLeft, DownRight, DownLeft] all with and without Fire


    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BattlezoneState, ) -> BattlezoneInfo:
        return BattlezoneInfo(time=state.step_counter)


    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BattlezoneState, state: BattlezoneState):
        return state.score - previous_state.score


    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BattlezoneState) -> bool:
        player_dead = state.life == 0
        return player_dead



#-------------------------------------renderer-------------------------------------
class BattlezoneRenderer(JAXGameRenderer):

    def __init__(self, consts: BattlezoneConstants = None, config=None):
        super().__init__()

        self.consts = consts or BattlezoneConstants()
        self.config = config or render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            # downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Create procedural assets for both walls
        wall_sprite_top = self._create_wall_sprite(self.consts.SCORE_POS_Y)
        wall_sprite_bottom = self._create_wall_sprite(self.consts.WALL_BOTTOM_HEIGHT)

        # 2. Update asset config to include both walls
        asset_config = self._get_asset_config(wall_sprite_top, wall_sprite_bottom)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/battlezone" # TODO change later when we have sprites

        # 3. Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        #----------------------create padded enemy masks for uniform shape-----------------------
        enemy_sprites = [
            self.SHAPE_MASKS["tank_enemy_01"],
            self.SHAPE_MASKS["tank_enemy_02"],
            self.SHAPE_MASKS["tank_enemy_03"],
            self.SHAPE_MASKS["tank_enemy_04"],
            self.SHAPE_MASKS["saucer_left"],
            self.SHAPE_MASKS["saucer_right"],
            self.SHAPE_MASKS["fighter_jet"],
            self.SHAPE_MASKS["supertank_enemy_01"],
            self.SHAPE_MASKS["supertank_enemy_02"],
            self.SHAPE_MASKS["supertank_enemy_03"],
            self.SHAPE_MASKS["supertank_enemy_04"],
        ]
        enemy_pad_x = max(sprite.shape[0] for sprite in enemy_sprites)
        enemy_pad_y = max(sprite.shape[1] for sprite in enemy_sprites)

        self.padded_enemy_masks = jnp.array([
            [self.pad_to_shape(self.SHAPE_MASKS["tank_enemy_01"], enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(self.SHAPE_MASKS["tank_enemy_02"], enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(self.SHAPE_MASKS["tank_enemy_03"], enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(self.SHAPE_MASKS["tank_enemy_03"], enemy_pad_x, enemy_pad_y),  # Reusing 3rd for the 4th turn phase
             self.pad_to_shape(self.SHAPE_MASKS["tank_enemy_04"], enemy_pad_x, enemy_pad_y),  # 4th is now the front-facing phase
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["tank_enemy_03"], axis=1), enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["tank_enemy_03"], axis=1), enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["tank_enemy_02"], axis=1), enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["tank_enemy_01"], axis=1), enemy_pad_x, enemy_pad_y),
             ],
            # Duplicate the saucer/fighter to have 9 elements to match indexing
            [self.pad_to_shape(self.SHAPE_MASKS["saucer_left"], enemy_pad_x, enemy_pad_y)] * 4 + \
            [self.pad_to_shape(self.SHAPE_MASKS["saucer_left"], enemy_pad_x, enemy_pad_y)] + \
            [self.pad_to_shape(self.SHAPE_MASKS["saucer_right"], enemy_pad_x, enemy_pad_y)] * 4,

            [self.pad_to_shape(self.SHAPE_MASKS["fighter_jet"], enemy_pad_x, enemy_pad_y)] * 9,

            [self.pad_to_shape(self.SHAPE_MASKS["supertank_enemy_01"], enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(self.SHAPE_MASKS["supertank_enemy_02"], enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(self.SHAPE_MASKS["supertank_enemy_03"], enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(self.SHAPE_MASKS["supertank_enemy_03"], enemy_pad_x, enemy_pad_y),  # Reusing 3rd
             self.pad_to_shape(self.SHAPE_MASKS["supertank_enemy_04"], enemy_pad_x, enemy_pad_y),  # Front-facing
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["supertank_enemy_03"], axis=1), enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["supertank_enemy_03"], axis=1), enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["supertank_enemy_02"], axis=1), enemy_pad_x, enemy_pad_y),
             self.pad_to_shape(jnp.flip(self.SHAPE_MASKS["supertank_enemy_01"], axis=1), enemy_pad_x, enemy_pad_y),
            ],
        ])
        self.PRECOMPUTED_SCALES = self._precalculate_stage_masks()
        explosion_sprites = [
            self.SHAPE_MASKS["enemy_explosion_1"],
            self.SHAPE_MASKS["enemy_explosion_2"],
            self.SHAPE_MASKS["enemy_explosion_3"],
        ]
        explosion_pad_x = max(sprite.shape[0] for sprite in explosion_sprites)
        explosion_pad_y = max(sprite.shape[1] for sprite in explosion_sprites)
        self.enemy_explosion_mask = jnp.array([
            self.pad_to_shape(self.SHAPE_MASKS["enemy_explosion_1"], explosion_pad_x, explosion_pad_y),
            self.pad_to_shape(self.SHAPE_MASKS["enemy_explosion_2"], explosion_pad_x, explosion_pad_y),
            self.pad_to_shape(self.SHAPE_MASKS["enemy_explosion_3"], explosion_pad_x, explosion_pad_y)
        ])
        self.projectile_masks = jnp.array([
            self.pad_to_shape(self.SHAPE_MASKS["projectile_big"], 6, 3),
            self.pad_to_shape(self.SHAPE_MASKS["projectile_small"], 6, 3)
        ])

        # Precompute tiny render sprites to avoid runtime line tracing/scatters.
        self.target_indicator_active_mask, self.target_indicator_inactive_mask = self._build_target_indicator_masks()
        self.radar_num_angles = 360
        self.radar_line_masks = self._precompute_radar_line_masks(self.consts.RADAR_RADIUS, self.radar_num_angles)


    def _create_wall_sprite(self, height: int) -> jnp.ndarray:
        """Procedurally creates an RGBA sprite for a wall of given height."""
        wall_color_rgba = (0, 0, 0, 255)  # black
        wall_shape = (height, self.consts.WIDTH, 4)
        wall_sprite = jnp.tile(
            jnp.array(wall_color_rgba, dtype=jnp.uint8),
            (*wall_shape[:2], 1)
        )

        return wall_sprite

    def _build_target_indicator_masks(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        height = 7
        width = 1
        active_color = jnp.array(self.COLOR_TO_ID[self.consts.TARGET_INDICATOR_COLOR_ACTIVE], dtype=jnp.int32)
        inactive_color = jnp.array(self.COLOR_TO_ID[self.consts.TARGET_INDICATOR_COLOR_INACTIVE], dtype=jnp.int32)
        active_mask = jnp.full((height, width), active_color, dtype=jnp.int32)
        inactive_mask = jnp.full((height, width), inactive_color, dtype=jnp.int32)
        return active_mask, inactive_mask

    def _bresenham_points(self, x0: int, y0: int, x1: int, y1: int):
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0

        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

        return points

    def _precompute_radar_line_masks(self, radius: int, num_angles: int = 360) -> jnp.ndarray:
        size = radius * 2 + 1
        center = radius
        transparent = int(self.jr.TRANSPARENT_ID)
        masks = []

        for i in range(num_angles):
            angle = (2.0 * jnp.pi * i) / num_angles
            x1 = int(round(center + float(jnp.sin(angle) * radius)))
            y1 = int(round(center + float(jnp.cos(angle) * radius)))
            mask = [[transparent for _ in range(size)] for _ in range(size)]
            for x, y in self._bresenham_points(center, center, x1, y1):
                if 0 <= x < size and 0 <= y < size:
                    mask[y][x] = 1
            masks.append(mask)

        return jnp.array(masks, dtype=jnp.int32)

    def _get_asset_config(self, wall_sprite_top: jnp.ndarray, wall_sprite_bottom: jnp.ndarray) -> list:
        """Returns the declarative manifest of all assets for the game, including both wall sprites."""
        return [ # TODO change later when we have assets
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'tank', 'type': 'single', 'file': 'tank.npy'},
            {'name': 'chainsLeft', 'type': 'single', 'file': 'chainsLeft.npy'},
            {'name': 'chainsRight', 'type': 'single', 'file': 'chainsRight.npy'},
            {'name': 'mountains', 'type': 'single', 'file': 'mountains.npy'},
            {'name': 'grass_front_1', 'type': 'single', 'file': 'grass_front_1.npy'},
            {'name': 'grass_front_2', 'type': 'single', 'file': 'grass_front_2.npy'},
            {'name': 'grass_back', 'type': 'single', 'file': 'grass_back_1.npy'},
            {'name': 'life', 'type': 'single', 'file': 'life.npy'},
            {'name': 'player_digits', 'type': 'digits', 'pattern': 'player_score_{}.npy'},
            # enemies
            {'name': 'tank_enemy_01', 'type': 'single', 'file': 'tank_enemy_01.npy'}, # TODO not sure if we can/should
            {'name': 'tank_enemy_02', 'type': 'single', 'file': 'tank_enemy_02.npy'},   # summarize them like digits
            {'name': 'tank_enemy_03', 'type': 'single', 'file': 'tank_enemy_03.npy'},
            {'name': 'tank_enemy_04', 'type': 'single', 'file': 'tank_enemy_04.npy'},
            {'name': 'supertank_enemy_01', 'type': 'single', 'file': 'supertank_enemy_01.npy'},
            {'name': 'supertank_enemy_02', 'type': 'single', 'file': 'supertank_enemy_02.npy'},
            {'name': 'supertank_enemy_03', 'type': 'single', 'file': 'supertank_enemy_03.npy'},
            {'name': 'supertank_enemy_04', 'type': 'single', 'file': 'supertank_enemy_04.npy'},
            {'name': 'saucer_left', 'type': 'single', 'file': 'saucer_left.npy'},
            {'name': 'saucer_right', 'type': 'single', 'file': 'saucer_right.npy'},
            {'name': 'fighter_jet', 'type': 'single', 'file': 'fighter.npy'},
            {'name': 'projectile_big', 'type': 'single', 'file': 'projectile_big.npy'},
            {'name': 'projectile_small', 'type': 'single', 'file': 'projectile_small.npy'},
            #anims
            {'name': 'enemy_explosion_1', 'type': 'single', 'file': 'enemy_explosion_1.npy'},
            {'name': 'enemy_explosion_2', 'type': 'single', 'file': 'enemy_explosion_2.npy'},
            {'name': 'enemy_explosion_3', 'type': 'single', 'file': 'enemy_explosion_3.npy'},
            # Add the procedurally created sprites to the manifest
            {'name': 'blackscreen', 'type': 'procedural', 'data': wall_sprite_top},
            {'name': 'wall_bottom', 'type': 'procedural', 'data': wall_sprite_bottom},
            {'name': 'target_indicator', 'type': 'single', 'file': 'yellow_pixel.npy'},
        ]


    def _scroll_chain_colors(self, chainMask, scroll):

        h, w = jnp.shape(chainMask)

        # create color pattern
        pattern = jnp.arange((h + 2) // 3) % 2 * 3 # create enough 0/1 pairs
        row_values = jnp.repeat(pattern, 3)[:h]  # ensure exactly 19 rows

        # replace with actual color indices
        color_id1 = self.COLOR_TO_ID[self.consts.CHAINS_COL_1]
        color_id2 = self.COLOR_TO_ID[self.consts.CHAINS_COL_2]
        row_values = jnp.where(row_values == 0, color_id1, color_id2)

        # create and scroll array
        arr = jnp.broadcast_to(row_values[:, None], (h, w))
        scrolled = jnp.roll(arr, shift=scroll, axis=0)

        return jnp.where(chainMask==self.jr.TRANSPARENT_ID, chainMask, scrolled)


    def _scroll_grass_back(self, grass_mask, scroll):

        grass_fill_color_id = grass_mask[0, 0]
        grass_back_shift = jnp.floor_divide(scroll, 2) % 4
        grass_back_scrolled_mask = jnp.roll(grass_mask, shift=grass_back_shift, axis=0)
        mask = jnp.arange(grass_back_scrolled_mask.shape[0]) < grass_back_shift
        mask = mask[:, None]  # broadcast across columns

        return jnp.where(mask, grass_fill_color_id, grass_back_scrolled_mask)


    @staticmethod
    def _draw_line(img, x0, y0, x1, y1, colorID, samples=256):
        # taken from experimental branch + some changes needs to be overworked maybe
        # Parametric line sampling (jit-friendly; Bresenham avoids floats but needs while loops)
        # Keep sample count static for JIT shape stability.
        t = jnp.linspace(0.0, 1.0, samples)
        xs = jnp.round(x0 + (x1 - x0) * t).astype(jnp.int32)
        ys = jnp.round(y0 + (y1 - y0) * t).astype(jnp.int32)
        im = img
        im = im.at[ys.clip(0, im.shape[0] - 1), xs.clip(0, im.shape[1] - 1)].set(colorID)

        return im


    def _render_radar(self, img, state, center_x, center_y, radius, colorID_1, colorID_2):

        h, w = jnp.shape(img)

        #------------------draw line------------------
        normalized_angle = jnp.mod(state.radar_rotation_counter, 2.0 * jnp.pi)
        angle_idx = jnp.round(normalized_angle * (self.radar_num_angles / (2.0 * jnp.pi))).astype(jnp.int32)
        angle_idx = jnp.mod(angle_idx, self.radar_num_angles)
        line_mask = self.radar_line_masks[angle_idx]
        colored_line_mask = jnp.where(line_mask == 1, colorID_2, self.jr.TRANSPARENT_ID)
        img = self.jr.render_at_clipped(img, center_x - radius, center_y - radius, colored_line_mask)

        #-------------------draw circle-------------
        y = jnp.arange(h)[:, None]  # construct index coordinate mapping
        x = jnp.arange(w)[None, :]

        # Compute squared distance from center
        dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2
        extended_radius = radius + 1
        mask = jnp.logical_and(dist_sq >= extended_radius ** 2, dist_sq < (extended_radius + 1) ** 2)
        img = jnp.where(mask, colorID_1, img)

        #------------------draw enemy dots----------------
        # Check if enemy in radar radius
        in_radar = jax.vmap(self.check_in_radar, in_axes=(0))(state.enemies)

        # Get raw player coords
        world_enemies_x = state.enemies.x
        world_enemies_z = state.enemies.z
        world_enemies_dist = state.enemies.distance

        # Scale to radar size
        scale_val = radius / self.consts.RADAR_MAX_SCAN_RADIUS
        radar_enemies_x = world_enemies_x * scale_val
        radar_enemies_z = world_enemies_z * scale_val * (-1)

        # Offset to radar center
        radar_enemies_x = jnp.round(radar_enemies_x + center_x).astype(jnp.int32)
        radar_enemies_z = jnp.round(radar_enemies_z + center_y).astype(jnp.int32)

        # Only allow in range enemies
        radar_enemies_x = jnp.where(in_radar, radar_enemies_x, -1)
        radar_enemies_z = jnp.where(in_radar, radar_enemies_z, -1)

        # Draw point
        img = img.at[radar_enemies_z, radar_enemies_x].set(colorID_2)

        return img


    def check_in_radar(self, enemies: Enemy) -> chex.Array:
        return((enemies.distance <= self.consts.RADAR_MAX_SCAN_RADIUS) & enemies.active)


    def pad_to_shape(self, arr: jnp.ndarray, shape_target_x: int, shape_target_y: int) -> jnp.ndarray:
        x, y = arr.shape
        pad_x = shape_target_x - x
        pad_y = shape_target_y - y

        return jnp.pad(arr, ((0, pad_x), (0, pad_y)), mode='constant', constant_values=self.jr.TRANSPARENT_ID)


    def _view_angle_for_enemy_sprite(self, enemy: Enemy) -> jnp.ndarray:
        """Angle in [0, pi] for sprite column: from angle_diff vs player, not camera/orientation alone."""
        tcx = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_X)
        tcz = jnp.float32(self.consts.PLAYER_HITBOX_CENTER_Z)
        perfect_angle = (2 * jnp.pi - jnp.arctan2(enemy.x - tcx, enemy.z - tcz)) % (2 * jnp.pi)
        angle_diff = (perfect_angle - enemy.orientation_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
        angle = (jnp.pi / 2) - angle_diff
        return jnp.clip(angle, 0.0, jnp.pi)

    def get_enemy_mask(self, enemy: Enemy):
        selected_enemy_type = self.padded_enemy_masks[enemy.enemy_type]

        angle = self._view_angle_for_enemy_sprite(enemy)

        n, _, _ = jnp.shape(selected_enemy_type)
        index = jnp.round((angle / jnp.pi) * (n - 1)).astype(jnp.int32)
        index = jnp.clip(index, 0, n - 1)

        return jnp.array(selected_enemy_type[index])


    def world_cords_to_viewport_cords(self, x, z, f):
        # f = (screen_height / 2) / tan(FOVv / 2)
        def anchor(_):
            # Behind the camera or invalid
            return -100, -100

        def uvMap(_):
            u = ((f * (x / z)) + self.consts.WIDTH / 2).astype(int)
            vOffset = self.consts.HORIZON_Y
            v = ((f / (z - self.consts.PLAYER_Z_PLANE)) + vOffset).astype(int)

            return u, v

        return jax.lax.cond(z<=0, anchor, uvMap, operand=None)


    def _player_projectile_visual_turn_scale(self, projectile_ttl: chex.Array):
        age = (self.consts.PLAYER_PROJECTILE_TTL - projectile_ttl).astype(jnp.float32)
        ramp = jnp.float32(max(1, self.consts.PLAYER_PROJECTILE_TURN_SCALE_RAMP_STEPS))
        blend = jnp.clip(age / ramp, 0.0, 1.0)
        return (
            self.consts.PLAYER_PROJECTILE_TURN_SCALE_MIN
            + (self.consts.PLAYER_PROJECTILE_TURN_SCALE - self.consts.PLAYER_PROJECTILE_TURN_SCALE_MIN) * blend
        )


    def _scale_player_projectile_screen_x(self, x: chex.Array, projectile_ttl: chex.Array):
        x = x.astype(jnp.float32)
        scale = self._player_projectile_visual_turn_scale(projectile_ttl)
        return self.consts.TARGET_INDICATOR_POS_X + (x - self.consts.TARGET_INDICATOR_POS_X) * scale


    def _precalculate_stage_masks(self) -> jnp.ndarray:
        num_types, num_rotations, max_h, max_w = self.padded_enemy_masks.shape
        num_stages = 9

        type_blocks = []
        for e_type in range(num_types):
            rot_blocks = []
            for rot in range(num_rotations):
                stage_blocks = []
                for stage in range(num_stages):
                    mask = self.padded_enemy_masks[e_type, rot]
                    target_w = int(self.consts.ENEMY_STAGE_WIDTHS[e_type, stage])
                    target_h = int(self.consts.ENEMY_STAGE_HEIGHTS[e_type, stage])

                    resized = jax.image.resize(mask, (target_h, target_w), method="nearest")

                    pad_top = max(0, (max_h - target_h) // 2)
                    pad_bottom = max(0, max_h - target_h - pad_top)
                    pad_left = max(0, (max_w - target_w) // 2)
                    pad_right = max(0, max_w - target_w - pad_left)

                    padded = jnp.pad(
                        resized,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="constant",
                        constant_values=self.jr.TRANSPARENT_ID,
                    )
                    stage_blocks.append(padded[:max_h, :max_w])
                rot_blocks.append(jnp.stack(stage_blocks, axis=0))
            type_blocks.append(jnp.stack(rot_blocks, axis=0))
        return jnp.stack(type_blocks, axis=0)

    def get_enemy_mask_precomputed(self, enemy: Enemy) -> jnp.ndarray:
        n = self.padded_enemy_masks.shape[1]

        angle = self._view_angle_for_enemy_sprite(enemy)

        rot_index = jnp.round((angle / jnp.pi) * (n - 1)).astype(jnp.int32)
        rot_index = jnp.clip(rot_index, 0, n - 1)

        stage = jnp.sum(enemy.distance >= self.consts.ENEMY_STAGE_THRESHOLDS)
        stage = jnp.clip(stage, 0, 8)

        return self.PRECOMPUTED_SCALES[enemy.enemy_type, rot_index, stage]


    def zoom_mask(self, mask, zoom_factor):
        """
        Scales the mask proportional to zoom_factor.
        - zoom_factor = 1 keeps everything the same
        - zoom_factor > 1 zooms in (mask elements appear smaller)
        Works with masks padded with -1.
        """

        def anchor(_):
            return mask

        def zoom(_):
            x, y = mask.shape

            # Create grid of coordinates in the output canvas
            xi = jnp.arange(x)
            yi = jnp.arange(y)
            xv, yv = jnp.meshgrid(xi, yi, indexing='ij')

            # Compute coordinates in the original mask to sample
            # Zoom center is at the middle of the **non-padded region**
            valid_rows = jnp.any(mask != self.jr.TRANSPARENT_ID, axis=1)
            valid_cols = jnp.any(mask != self.jr.TRANSPARENT_ID, axis=0)
            x_min, x_max = jnp.argmax(valid_rows), x - jnp.argmax(valid_rows[::-1]) - 1
            y_min, y_max = jnp.argmax(valid_cols), y - jnp.argmax(valid_cols[::-1]) - 1
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

            # Zoom coordinates
            x_orig = (xv - cx) * zoom_factor + cx
            y_orig = (yv - cy) * zoom_factor + cy

            # Clip to the **non-padded region**
            x_orig = jnp.clip(x_orig, x_min, x_max)
            y_orig = jnp.clip(y_orig, y_min, y_max)

            # Nearest neighbor sampling
            x0 = jnp.floor(x_orig).astype(int)
            y0 = jnp.floor(y_orig).astype(int)
            zoomed_mask = mask[x0, y0]

            # Zero edges: set padding outside zoomed area to self.transparent_id
            rows = jnp.arange(x)[:, None]
            cols = jnp.arange(y)[None, :]
            z = zoom_factor + 1
            edge = (rows < x_min + z) | (rows >= x_max - z) | \
                   (cols < y_min + z) | (cols >= y_max - z)
            zoomed_mask = jnp.where(edge, self.jr.TRANSPARENT_ID, zoomed_mask)

            return zoomed_mask

        return jax.lax.cond(zoom_factor <= 1, anchor, zoom, operand=None)


    def render_single_enemy(self, raster, enemy: Enemy):

        def enemy_active(enemy):
            zoomed_mask = self.get_enemy_mask_precomputed(enemy)
            x, _ = self.world_cords_to_viewport_cords(enemy.x, enemy.z, self.consts.CAMERA_FOCAL_LENGTH)

            stage = jnp.sum(enemy.distance >= self.consts.ENEMY_STAGE_THRESHOLDS)
            stage = jnp.clip(stage, 0, 8)
            draw_y = self.consts.ENEMY_STAGE_Y_POS[enemy.enemy_type, stage] - (
                self.padded_enemy_masks.shape[2] // 2
            )

            return self.jr.render_at_clipped(raster, x - (zoomed_mask.shape[1] // 2), draw_y, zoomed_mask)

        def enemy_inactive(enemy):

            def render_death(enemy):
                # choose frame of death animation based on death_anim_counter
                n = enemy.death_anim_counter
                index = jnp.where(
                    n >= 0.8 * self.consts.ENEMY_DEATH_ANIM_LENGTH,
                    0, # index 0 for first 20% of the animation
                    jnp.where(
                        n >= 0.4 * self.consts.ENEMY_DEATH_ANIM_LENGTH,
                        1, # index 1 for second 40% of the animation
                        2 # index 2 for last 40% of the animation
                    )
                ) # if it works it works
                mask = self.enemy_explosion_mask[index]

                # apply zoom based on distance
                zoom_factor = ((jnp.sqrt(jnp.square(enemy.x) + jnp.square(enemy.z)) - 20.0) *
                               self.consts.DISTANCE_TO_ZOOM_FACTOR_CONSTANT).astype(int)
                zoomed_mask = self.zoom_mask(mask, zoom_factor)
                x, y = self.world_cords_to_viewport_cords(enemy.x, enemy.z, self.consts.CAMERA_FOCAL_LENGTH)
                del y

                rightmost_col = jnp.max(
                    jnp.where(
                        jnp.any(zoomed_mask != self.jr.TRANSPARENT_ID, axis=0),
                        jnp.arange(zoomed_mask.shape[1]),
                        0
                    )
                )

                return self.jr.render_at_clipped(raster, x - (rightmost_col // 2), self.consts.ENEMY_POS_Y, zoomed_mask)

            def _pass(_):
                return raster

            # render death animation if death_anim_counter > 0, else render nothing
            return jax.lax.cond(enemy.death_anim_counter==0, _pass, render_death, enemy)

        return jax.lax.cond(enemy.active, enemy_active, enemy_inactive, enemy)


    def render_single_projectile(self, raster, projectile: Projectile, is_player: bool = False):

        def projectile_active(projectile):
            x, true_y = self.world_cords_to_viewport_cords(projectile.x, projectile.z, self.consts.CAMERA_FOCAL_LENGTH)
            x = jax.lax.cond(
                is_player,
                lambda v: self._scale_player_projectile_screen_x(v, projectile.time_to_live),
                lambda v: v.astype(jnp.float32),
                x,
            )

            def get_player_y(_):
                idx = jnp.clip(
                    self.consts.PLAYER_PROJECTILE_TTL - projectile.time_to_live,
                    0,
                    self.consts.PLAYER_BULLET_Y_SEQ.shape[0] - 1,
                )
                return self.consts.PLAYER_BULLET_Y_SEQ[idx]

            y = jax.lax.cond(is_player, get_player_y, lambda _: true_y, operand=None)
            projectile_mask_index = jnp.where(y <= 105, 1, 0)
            projectile_mask = self.projectile_masks[projectile_mask_index]
            centered_x = x - (projectile_mask.shape[1] // 2)
            return self.jr.render_at_clipped(raster, centered_x, y, projectile_mask)


        def projectile_inactive(_):
            return raster

        # Player bullet lifetime is governed by TTL and should not be clipped by
        # radar distance (distance includes spawn depth offset and can truncate
        # the last visual frames early).
        within_player_render_range = jax.lax.cond(
            is_player,
            lambda _: jnp.array(True, dtype=jnp.bool_),
            lambda _: projectile.distance <= self.consts.RADAR_MAX_SCAN_RADIUS,
            operand=None,
        )
        render_condition = jnp.all(jnp.stack([
            projectile.active,
            projectile.z >= self.consts.PLAYER_Z_PLANE,
            within_player_render_range,
        ]))

        return jax.lax.cond(render_condition, projectile_active, projectile_inactive, projectile)


    def render_targeting_indicator(self, raster, state):
        target_x = self.consts.TARGET_INDICATOR_POS_X

        def check_enemy_target(enemy):
            e_u, _ = self.world_cords_to_viewport_cords(
                enemy.x, enemy.z, self.consts.CAMERA_FOCAL_LENGTH
            )
            stage = jnp.sum(enemy.distance >= self.consts.ENEMY_STAGE_THRESHOLDS)
            stage = jnp.clip(stage, 0, 8)

            n = self.padded_enemy_masks.shape[1]
            angle = self._view_angle_for_enemy_sprite(enemy)
            rot_index = jnp.round((angle / jnp.pi) * (n - 1)).astype(jnp.int32)
            rot_index = jnp.clip(rot_index, 0, n - 1)

            mask = self.PRECOMPUTED_SCALES[enemy.enemy_type, rot_index, stage]
            _, mask_w = mask.shape

            draw_x = e_u - (mask_w // 2)

            local_x = target_x - draw_x.astype(jnp.int32)
            in_bounds_x = (local_x >= 0) & (local_x < mask_w)

            safe_x = jnp.clip(local_x, 0, mask_w - 1)
            column_has_solid = jnp.any(mask[:, safe_x] != self.jr.TRANSPARENT_ID)

            return enemy.active & (enemy.z > 0) & in_bounds_x & column_has_solid

        pointing_at_enemy = jnp.any(jax.vmap(check_enemy_target)(state.enemies))

        color_id = jnp.where(
            pointing_at_enemy,
            jnp.array(self.COLOR_TO_ID[self.consts.TARGET_INDICATOR_COLOR_ACTIVE], dtype=jnp.int32),
            jnp.array(self.COLOR_TO_ID[self.consts.TARGET_INDICATOR_COLOR_INACTIVE], dtype=jnp.int32),
        )

        indicator_mask = jax.lax.cond(
            color_id == self.COLOR_TO_ID[self.consts.TARGET_INDICATOR_COLOR_ACTIVE],
            lambda _: self.target_indicator_active_mask,
            lambda _: self.target_indicator_inactive_mask,
            operand=None,
        )

        return self.jr.render_at(
            raster,
            self.consts.TARGET_INDICATOR_POS_X,
            self.consts.TARGET_INDICATOR_POS_Y,
            indicator_mask,
        )


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        #-----------------background
        raster = self.jr.create_object_raster(self.BACKGROUND)

        def normal_render(raster):
            mountains_mask = self.SHAPE_MASKS["mountains"]
            mountains_mask_scrolled = jnp.roll(mountains_mask, shift=state.mountains_anim_counter, axis=1)
            raster = self.jr.render_at(raster, 0,
                                       self.consts.MOUNTAINS_Y, mountains_mask_scrolled)

            grass_front_mask = jnp.where((state.grass_anim_counter % 30) < 15, self.SHAPE_MASKS["grass_front_1"],
                                          self.SHAPE_MASKS["grass_front_2"])
            raster = self.jr.render_at(raster, 0,
                                       self.consts.GRASS_FRONT_Y, grass_front_mask)

            grass_back_mask = self._scroll_grass_back(self.SHAPE_MASKS["grass_back"], state.grass_anim_counter)
            raster = self.jr.render_at(raster, 0,
                                       self.consts.GRASS_BACK_Y, grass_back_mask)

            #-------------------------------enemies-----------------
            def render_single_enemy_wrapped(raster, enemy):  # so that i dont have to pass self
                return self.render_single_enemy(raster, enemy), None

            # most distant first
            order = jnp.argsort(state.enemies.distance)[::-1]
            enemies_sorted = jax.tree.map(lambda x: x[order], state.enemies)

            raster, _ = jax.lax.scan(render_single_enemy_wrapped, raster, enemies_sorted)

            #------------------------------projectiles---------------
            raster = self.render_single_projectile(raster, state.player_projectile, is_player=True)
            #raster = jax.lax.cond(state.step_counter%2==0, self.render_single_projectile, lambda r, _: r,
                         #raster, state.player_projectile)
            # probably more accurate but looks stoopid because different frame rates
            def render_single_projectile_wrapped(raster, projectile):  # so that i dont have to pass self
                return self.render_single_projectile(raster, projectile, is_player=False), None

            raster, _ = jax.lax.scan(render_single_projectile_wrapped, raster, state.enemy_projectiles)

            # -------------------------foreground---------------------------------------------------------------
            tank_mask = self.SHAPE_MASKS["tank"]
            raster = self.jr.render_at(raster, self.consts.TANK_SPRITE_POS_X,
                                       self.consts.TANK_SPRITE_POS_Y, tank_mask)

            raster = self._render_radar(raster, state, self.consts.RADAR_CENTER_X, self.consts.RADAR_CENTER_Y,
                                        self.consts.RADAR_RADIUS, self.COLOR_TO_ID[self.consts.RADAR_COLOR_1],
                                        self.COLOR_TO_ID[self.consts.RADAR_COLOR_2])

            # --------------chains---------
            chains_l_mask = self.SHAPE_MASKS["chainsLeft"]
            color_shifted_chain_l = self._scroll_chain_colors(chains_l_mask, state.chains_l_anim_counter)
            raster = self.jr.render_at(raster, self.consts.CHAINS_L_POS_X,
                                       self.consts.CHAINS_POS_Y, color_shifted_chain_l)

            chains_r_mask = self.SHAPE_MASKS["chainsRight"]
            color_shifted_chain_r = self._scroll_chain_colors(chains_r_mask, state.chains_r_anim_counter)
            raster = self.jr.render_at(raster, self.consts.CHAINS_R_POS_X,
                                       self.consts.CHAINS_POS_Y, color_shifted_chain_r)

            # ------------------ target indicator -----------------
            raster = self.render_targeting_indicator(raster, state)

            return raster

        def death_render(raster):
            raster = self.jr.render_at(raster, 0, 0, self.SHAPE_MASKS["blackscreen"])
            return raster

        raster = jax.lax.cond(state.death_anim_counter <= 0, normal_render, death_render, raster)

        # ----------------life---------------------------
        life_mask = self.SHAPE_MASKS["life"]

        def render_single_life(i, raster):
            return self.jr.render_at(raster, self.consts.LIFE_POS_X + (self.consts.LIFE_X_OFFSET * i),
                                     self.consts.LIFE_POS_Y, life_mask)

        raster = jax.lax.fori_loop(0, state.life, render_single_life, raster)

        # ---------------------------player score--------------------
        # primarily taken from pong + changes
        player_digit_masks = self.SHAPE_MASKS["player_digits"]  # Assumes single color
        player_digits = self.jr.int_to_digits(state.score, max_digits=8)
        # this does not correctly work currently (only when max_digits==exactly amount digits)

        raster = self.jr.render_label_selective(raster, self.consts.SCORE_POS_X, self.consts.SCORE_POS_Y, player_digits,
                                                player_digit_masks, 0, 8,
                                                spacing=6, max_digits_to_render=8)
        # best highscore i can find is 6 digits
        # --------------------------------------------------------------------
        return self.jr.render_from_palette(raster, self.PALETTE)
