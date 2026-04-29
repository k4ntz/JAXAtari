import os
import jax
import jax.image as jim
import jax.numpy as jnp
from functools import partial

from jax.scipy import ndimage
import numpy as np
from flax import struct
import jaxatari.spaces as spaces
from jaxatari.core import JaxEnvironment
from jaxatari.environment import ObjectObservation
from typing import Tuple, Optional
from enum import IntEnum
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
import jax.debug

"""
    Group members of the Gravitar base: Xusong Yin, Elizaveta Kuznetsova, Li Dai

    Gravitar Rework by Tiago Soares
"""


def _get_default_ship_angles():
    """Ship discrete rotation system (16 angles like original ALE)"""
    return jnp.array([
        -jnp.pi/2,              # 0: N (270° or -90°)
        -jnp.pi/2 + jnp.pi/8,   # 1: NNE (292.5°)
        -jnp.pi/2 + jnp.pi/4,   # 2: NE (315°)
        -jnp.pi/2 + 3*jnp.pi/8, # 3: ENE (337.5°)
        0.0,                    # 4: E (0°)
        jnp.pi/8,               # 5: ESE (22.5°)
        jnp.pi/4,               # 6: SE (45°)
        3*jnp.pi/8,             # 7: SSE (67.5°)
        jnp.pi/2,               # 8: S (90°)
        jnp.pi/2 + jnp.pi/8,    # 9: SSW (112.5°)
        jnp.pi/2 + jnp.pi/4,    # 10: SW (135°)
        jnp.pi/2 + 3*jnp.pi/8,  # 11: WSW (157.5°)
        jnp.pi,                 # 12: W (180°)
        jnp.pi + jnp.pi/8,      # 13: WNW (202.5°)
        jnp.pi + jnp.pi/4,      # 14: NW (225°)
        jnp.pi + 3*jnp.pi/8,    # 15: NNW (247.5°)
    ], dtype=jnp.float32)


class GravitarConstants(struct.PyTreeNode):
    """Constants for Gravitar game configuration."""
    
    # World scaling
    WORLD_SCALE: float = struct.field(pytree_node=False, default=3.0)
    FORCE_SPRITES: bool = struct.field(pytree_node=False, default=True)
    SCALE: int = struct.field(pytree_node=False, default=1)
    
    # Object limits
    MAX_BULLETS: int = struct.field(pytree_node=False, default=16) # reduced from 64 for faster compilation
    MAX_ENEMIES: int = struct.field(pytree_node=False, default=4) # reduced from 16 for faster compilation
    
    # Action constants
    NOOP: int = struct.field(pytree_node=False, default=0)
    FIRE: int = struct.field(pytree_node=False, default=1)
    UP: int = struct.field(pytree_node=False, default=2)
    RIGHT: int = struct.field(pytree_node=False, default=3)
    LEFT: int = struct.field(pytree_node=False, default=4)
    DOWN: int = struct.field(pytree_node=False, default=5)
    UPRIGHT: int = struct.field(pytree_node=False, default=6)
    UPLEFT: int = struct.field(pytree_node=False, default=7)
    DOWNRIGHT: int = struct.field(pytree_node=False, default=8)
    DOWNLEFT: int = struct.field(pytree_node=False, default=9)
    UPFIRE: int = struct.field(pytree_node=False, default=10)
    RIGHTFIRE: int = struct.field(pytree_node=False, default=11)
    LEFTFIRE: int = struct.field(pytree_node=False, default=12)
    DOWNFIRE: int = struct.field(pytree_node=False, default=13)
    UPRIGHTFIRE: int = struct.field(pytree_node=False, default=14)
    UPLEFTFIRE: int = struct.field(pytree_node=False, default=15)
    DOWNRIGHTFIRE: int = struct.field(pytree_node=False, default=16)
    DOWNLEFTFIRE: int = struct.field(pytree_node=False, default=17)
    
    # HUD settings
    HUD_HEIGHT: int = struct.field(pytree_node=False, default=24)
    MAX_LIVES: int = struct.field(pytree_node=False, default=6)
    HUD_PADDING: int = struct.field(pytree_node=False, default=5)
    HUD_SHIP_WIDTH: int = struct.field(pytree_node=False, default=10)
    HUD_SHIP_HEIGHT: int = struct.field(pytree_node=False, default=12)
    HUD_SHIP_SPACING: int = struct.field(pytree_node=False, default=12)
    
    # Window dimensions
    WINDOW_WIDTH: int = struct.field(pytree_node=False, default=160)
    WINDOW_HEIGHT: int = struct.field(pytree_node=False, default=210)
    
    # Spawn and respawn timing
    SAUCER_SPAWN_DELAY_FRAMES: int = struct.field(pytree_node=False, default=200)
    SAUCER_RESPAWN_DELAY_FRAMES: int = struct.field(pytree_node=False, default=180 * 3)
    UFO_SPAWN_DELAY_FRAMES: int = struct.field(pytree_node=False, default=180 * 2)
    UFO_RESPAWN_DELAY_FRAMES: int = struct.field(pytree_node=False, default=180 * 2)
    UFO_SPAWN_Y_THRESHOLD: float = struct.field(pytree_node=False, default=50.0)
    
    # Movement speeds and physics
    SAUCER_SPEED_MAP: float = struct.field(pytree_node=False, default=0.18)
    SAUCER_SPEED_ARENA: float = struct.field(pytree_node=False, default=0.36)
    SAUCER_RADIUS: float = struct.field(pytree_node=False, default=3.0)
    SHIP_RADIUS: float = struct.field(pytree_node=False, default=2.0)
    TRACTOR_BEAM_RANGE: float = struct.field(pytree_node=False, default=15.0)
    PLAYER_BULLET_SPEED: float = struct.field(pytree_node=False, default=1.3)
    SAUCER_BULLET_SPEED: float = struct.field(pytree_node=False, default=2.0)
    ENEMY_BULLET_SPEED: float = struct.field(pytree_node=False, default=1.3)
    UFO_HIT_RADIUS: float = struct.field(pytree_node=False, default=3.0)
    
    # HP and damage
    SAUCER_INIT_HP: int = struct.field(pytree_node=False, default=1)
    
    # Animation timing
    SAUCER_EXPLOSION_FRAMES: int = struct.field(pytree_node=False, default=60)
    SAUCER_FIRE_INTERVAL_FRAMES: int = struct.field(pytree_node=False, default=8)
    ENEMY_EXPLOSION_FRAMES: int = struct.field(pytree_node=False, default=60)
    ENEMY_FIRE_COOLDOWN_FRAMES: int = struct.field(pytree_node=False, default=10)
    PLAYER_FIRE_COOLDOWN_FRAMES: int = struct.field(pytree_node=False, default=8)

    # Bullet caps (moddable)
    MAX_ACTIVE_PLAYER_BULLETS_MAP: int = struct.field(pytree_node=False, default=1)
    MAX_ACTIVE_PLAYER_BULLETS_LEVEL: int = struct.field(pytree_node=False, default=2)
    MAX_ACTIVE_PLAYER_BULLETS_ARENA: int = struct.field(pytree_node=False, default=2)
    MAX_ACTIVE_SAUCER_BULLETS: int = struct.field(pytree_node=False, default=2)
    MAX_ACTIVE_ENEMY_BULLETS: int = struct.field(pytree_node=False, default=2)

    # Physics moddable
    SOLAR_GRAVITY: float = struct.field(pytree_node=False, default=0.044)
    PLANETARY_GRAVITY: float = struct.field(pytree_node=False, default=0.0032)
    REACTOR_GRAVITY: float = struct.field(pytree_node=False, default=0.0001)
    THRUST_POWER: float = struct.field(pytree_node=False, default=0.035)
    MAX_SPEED: float = struct.field(pytree_node=False, default=3.0)
    FUEL_CONSUME_THRUST: float = struct.field(pytree_node=False, default=4.0)
    FUEL_CONSUME_SHIELD_TRACTOR: float = struct.field(pytree_node=False, default=10.0)
    STARTING_FUEL: float = struct.field(pytree_node=False, default=10000.0)
    ALLOW_TRACTOR_IN_REACTOR: bool = struct.field(pytree_node=False, default=False)
    ENEMY_KILL_SCORE: float = struct.field(pytree_node=False, default=250.0)
    LEVEL_CLEAR_SCORE: float = struct.field(pytree_node=False, default=1000.0)
    UFO_KILL_SCORE: float = struct.field(pytree_node=False, default=100.0)
    SAUCER_KILL_SCORE: float = struct.field(pytree_node=False, default=100.0)

    # Bonuses
    SOLAR_SYSTEM_BONUS_FUEL: float = struct.field(pytree_node=False, default=7000.0)
    SOLAR_SYSTEM_BONUS_LIVES: int = struct.field(pytree_node=False, default=2)
    SOLAR_SYSTEM_BONUS_SCORE: float = struct.field(pytree_node=False, default=4000.0)
    
    # Ship rotation
    SHIP_ANGLES: jnp.ndarray = struct.field(pytree_node=False, default_factory=_get_default_ship_angles)
    ROTATION_COOLDOWN_FRAMES: int = struct.field(pytree_node=False, default=5)
    
    # Debug settings
    SHIP_ANCHOR_X: Optional[float] = struct.field(pytree_node=False, default=None)
    SHIP_ANCHOR_Y: Optional[float] = struct.field(pytree_node=False, default=None)
    DEBUG_DRAW_SHIP_ORIGIN: bool = struct.field(pytree_node=False, default=True)
    
    # Reactor physics
    REACTOR_START_Y: float = struct.field(pytree_node=False, default=30.0)
    # Optional per-object layout override for reactor level (level 4).
    # Each entry supports either:
    # - {'type': <SpriteIdx/int>, 'coords': (<x>, <y>)}
    # - (<SpriteIdx/int>, <x>, <y>)
    REACTOR_LEVEL_LAYOUT: tuple = struct.field(pytree_node=False, default_factory=tuple)


# Module-level constants used by free functions (not methods)
# These cannot use self.consts since they're not class methods
_DEFAULT_CONSTS = GravitarConstants()
WINDOW_WIDTH = _DEFAULT_CONSTS.WINDOW_WIDTH
WINDOW_HEIGHT = _DEFAULT_CONSTS.WINDOW_HEIGHT  
HUD_HEIGHT = _DEFAULT_CONSTS.HUD_HEIGHT
WORLD_SCALE = _DEFAULT_CONSTS.WORLD_SCALE
SHIP_RADIUS = _DEFAULT_CONSTS.SHIP_RADIUS
SHIP_ANGLES = _DEFAULT_CONSTS.SHIP_ANGLES
ROTATION_COOLDOWN_FRAMES = _DEFAULT_CONSTS.ROTATION_COOLDOWN_FRAMES
MAX_BULLETS = _DEFAULT_CONSTS.MAX_BULLETS
MAX_ENEMIES = _DEFAULT_CONSTS.MAX_ENEMIES
NOOP = _DEFAULT_CONSTS.NOOP
REACTOR_START_Y = _DEFAULT_CONSTS.REACTOR_START_Y
MAX_LIVES = _DEFAULT_CONSTS.MAX_LIVES
SAUCER_SPAWN_DELAY_FRAMES = _DEFAULT_CONSTS.SAUCER_SPAWN_DELAY_FRAMES
SAUCER_RESPAWN_DELAY_FRAMES = _DEFAULT_CONSTS.SAUCER_RESPAWN_DELAY_FRAMES
UFO_RESPAWN_DELAY_FRAMES = _DEFAULT_CONSTS.UFO_RESPAWN_DELAY_FRAMES
UFO_SPAWN_Y_THRESHOLD = _DEFAULT_CONSTS.UFO_SPAWN_Y_THRESHOLD
SAUCER_SPEED_MAP = _DEFAULT_CONSTS.SAUCER_SPEED_MAP
SAUCER_SPEED_ARENA = _DEFAULT_CONSTS.SAUCER_SPEED_ARENA
PLAYER_BULLET_SPEED = _DEFAULT_CONSTS.PLAYER_BULLET_SPEED
SAUCER_BULLET_SPEED = _DEFAULT_CONSTS.SAUCER_BULLET_SPEED
ENEMY_BULLET_SPEED = _DEFAULT_CONSTS.ENEMY_BULLET_SPEED
SAUCER_RADIUS = _DEFAULT_CONSTS.SAUCER_RADIUS
UFO_HIT_RADIUS = _DEFAULT_CONSTS.UFO_HIT_RADIUS
TRACTOR_BEAM_RANGE = _DEFAULT_CONSTS.TRACTOR_BEAM_RANGE
SAUCER_INIT_HP = _DEFAULT_CONSTS.SAUCER_INIT_HP
SAUCER_EXPLOSION_FRAMES = _DEFAULT_CONSTS.SAUCER_EXPLOSION_FRAMES
SAUCER_FIRE_INTERVAL_FRAMES = _DEFAULT_CONSTS.SAUCER_FIRE_INTERVAL_FRAMES
ENEMY_EXPLOSION_FRAMES = _DEFAULT_CONSTS.ENEMY_EXPLOSION_FRAMES
ENEMY_FIRE_COOLDOWN_FRAMES = _DEFAULT_CONSTS.ENEMY_FIRE_COOLDOWN_FRAMES
PLAYER_FIRE_COOLDOWN_FRAMES = _DEFAULT_CONSTS.PLAYER_FIRE_COOLDOWN_FRAMES
SOLAR_SYSTEM_BONUS_FUEL = _DEFAULT_CONSTS.SOLAR_SYSTEM_BONUS_FUEL
SOLAR_SYSTEM_BONUS_LIVES = _DEFAULT_CONSTS.SOLAR_SYSTEM_BONUS_LIVES
SOLAR_SYSTEM_BONUS_SCORE = _DEFAULT_CONSTS.SOLAR_SYSTEM_BONUS_SCORE
MAX_ACTIVE_PLAYER_BULLETS_MAP = _DEFAULT_CONSTS.MAX_ACTIVE_PLAYER_BULLETS_MAP
MAX_ACTIVE_PLAYER_BULLETS_LEVEL = _DEFAULT_CONSTS.MAX_ACTIVE_PLAYER_BULLETS_LEVEL
MAX_ACTIVE_PLAYER_BULLETS_ARENA = _DEFAULT_CONSTS.MAX_ACTIVE_PLAYER_BULLETS_ARENA
MAX_ACTIVE_SAUCER_BULLETS = _DEFAULT_CONSTS.MAX_ACTIVE_SAUCER_BULLETS
MAX_ACTIVE_ENEMY_BULLETS = _DEFAULT_CONSTS.MAX_ACTIVE_ENEMY_BULLETS
FUEL_CONSUME_THRUST = _DEFAULT_CONSTS.FUEL_CONSUME_THRUST
FUEL_CONSUME_SHIELD_TRACTOR = _DEFAULT_CONSTS.FUEL_CONSUME_SHIELD_TRACTOR
STARTING_FUEL = _DEFAULT_CONSTS.STARTING_FUEL
ENEMY_KILL_SCORE = _DEFAULT_CONSTS.ENEMY_KILL_SCORE
LEVEL_CLEAR_SCORE = _DEFAULT_CONSTS.LEVEL_CLEAR_SCORE
UFO_KILL_SCORE = _DEFAULT_CONSTS.UFO_KILL_SCORE
SAUCER_KILL_SCORE = _DEFAULT_CONSTS.SAUCER_KILL_SCORE

# Precomputed neighborhood grid for terrain_hit_mask hot path.
_TERRAIN_MASK_R_MAX = 8
_TERRAIN_MASK_DX_FULL = jnp.arange(-_TERRAIN_MASK_R_MAX, _TERRAIN_MASK_R_MAX + 1, dtype=jnp.int32)
_TERRAIN_MASK_DY_FULL = jnp.arange(-_TERRAIN_MASK_R_MAX, _TERRAIN_MASK_R_MAX + 1, dtype=jnp.int32)
_TERRAIN_MASK_DX, _TERRAIN_MASK_DY = jnp.meshgrid(_TERRAIN_MASK_DX_FULL, _TERRAIN_MASK_DY_FULL, indexing='xy')
_TERRAIN_MASK_DIST2 = _TERRAIN_MASK_DX * _TERRAIN_MASK_DX + _TERRAIN_MASK_DY * _TERRAIN_MASK_DY

# Precomputed grids for terrain_hit hot path
_TERRAIN_HIT_RMAX = 16
_TERRAIN_HIT_DX = jnp.arange(-_TERRAIN_HIT_RMAX, _TERRAIN_HIT_RMAX + 1, dtype=jnp.int32)
_TERRAIN_HIT_DY = jnp.arange(-_TERRAIN_HIT_RMAX, _TERRAIN_HIT_RMAX + 1, dtype=jnp.int32)
_TERRAIN_HIT_DIST2 = _TERRAIN_HIT_DY[:, None] ** 2 + _TERRAIN_HIT_DX[None, :] ** 2

# Precomputed trigonometry to avoid jnp.arctan2 in hot loops
_SHIP_ANGLES_COS = jnp.cos(_DEFAULT_CONSTS.SHIP_ANGLES)
_SHIP_ANGLES_SIN = jnp.sin(_DEFAULT_CONSTS.SHIP_ANGLES)


@jax.jit
def snap_angle_to_discrete(angle: jnp.ndarray) -> jnp.ndarray:
    """Snap a continuous angle to the nearest of 16 discrete ship angles."""
    # Fast squared distance in 2D space avoids slow arctan2/sin/cos
    cost = (jnp.cos(angle) - _SHIP_ANGLES_COS) ** 2 + (jnp.sin(angle) - _SHIP_ANGLES_SIN) ** 2
    closest_idx = jnp.argmin(cost)
    return SHIP_ANGLES[closest_idx]


@jax.jit
def get_ship_sprite_idx(angle: jnp.ndarray) -> jnp.ndarray:
    """Get the sprite index for a given ship angle.
    
    Args:
        angle: Angle in radians (should be one of the discrete angles)
    Returns:
        Sprite index from SHIP_SPRITE_INDICES
    """
    cost = (jnp.cos(angle) - _SHIP_ANGLES_COS) ** 2 + (jnp.sin(angle) - _SHIP_ANGLES_SIN) ** 2
    closest_idx = jnp.argmin(cost)
    return SHIP_SPRITE_INDICES[closest_idx]


def _jax_rotate(image, angle_deg, reshape=False, order=1, mode='constant', cval=0):
    angle_rad = jnp.deg2rad(angle_deg)
    height, width = image.shape[:2]
    center_y, center_x = height / 2, width / 2
    y_coords, x_coords = jnp.mgrid[0:height, 0:width]
    y_centered, x_centered = y_coords - center_y, x_coords - center_x
    cos_angle, sin_angle = jnp.cos(-angle_rad), jnp.sin(-angle_rad)
    source_x = center_x + x_centered * cos_angle - y_centered * sin_angle
    source_y = center_y + x_centered * sin_angle + y_centered * cos_angle
    source_coords = jnp.stack([source_y, source_x])
    rotated_channels = []
    for i in range(image.shape[2]):
        rotated_channel = ndimage.map_coordinates(
            image[..., i], source_coords, order=order, mode=mode, cval=cval
        )
        rotated_channels.append(rotated_channel)
    return jnp.stack(rotated_channels, axis=-1).astype(image.dtype)

_OBS_MAX_PLANETS = 7
_OBS_HUD_DIM = 11


class SpriteIdx(IntEnum):
    # Ship & bullets
    SHIP_IDLE = 0  # spaceship.npy (points north)
    SHIP_THRUST = 1  # ship_thrust.npy
    SHIP_BULLET = 2  # ship_bullet.npy
    
    # Ship orientations (16 discrete angles like original ALE)
    SHIP_N = 0     # North - reuses SHIP_IDLE (spaceship.npy)
    SHIP_NNE = 50  # spaceship_nne.npy
    SHIP_NE = 51   # spaceship_ne.npy
    SHIP_NEE = 52  # spaceship_nee.npy
    SHIP_E = 53    # spaceship_e.npy
    SHIP_SEE = 54  # spaceship_see.npy
    SHIP_SE = 55   # spaceship_se.npy
    SHIP_SSE = 56  # spaceship_sse.npy
    SHIP_S = 57    # spaceship_s.npy
    SHIP_SSW = 58  # spaceship_ssw.npy
    SHIP_SW = 59   # spaceship_sw.npy
    SHIP_SWW = 60  # spaceship_sww.npy
    SHIP_W = 61    # spaceship_w.npy
    SHIP_NWW = 62  # spaceship_nww.npy
    SHIP_NW = 63   # spaceship_nw.npy
    SHIP_NNW = 64  # spaceship_nnw.npy

    # Enemy bullets
    ENEMY_BULLET = 3  # enemy_bullet.npy
    ENEMY_GREEN_BULLET = 4  # enemy_green_bullet.npy

    # Enemies
    ENEMY_ORANGE = 5  # enemy_orange.npy
    ENEMY_GREEN = 6  # enemy_green.npy
    ENEMY_SAUCER = 7  # saucer.npy
    ENEMY_UFO = 8  # UFO.npy

    # Explosions / crashes
    ENEMY_CRASH = 9  # enemy_crash.npy
    SAUCER_CRASH = 10  # saucer_crash.npy
    SHIP_CRASH = 11  # ship_crash.npy

    # World objects
    FUEL_TANK = 12  # fuel_tank.npy
    OBSTACLE = 13  # obstacle.npy
    SPAWN_LOC = 14  # spawn_location.npy

    # Reactor & terrain
    REACTOR = 15  # reactor.npy
    REACTOR_TERR = 16  # reactor_terrain.npy
    TERRAIN1 = 17  # terrain1.npy
    TERRAIN2 = 18  # terrain2.npy
    TERRAIN3 = 19  # terrain3.npy
    TERRAIN4 = 20  # terrain4.npy

    # Planets & UI
    PLANET1 = 21  # planet1.npy
    PLANET2 = 22  # planet2.npy
    PLANET3 = 23  # planet3.npy
    PLANET4 = 24  # planet4.npy
    REACTOR_DEST = 25  # reactor_destination.npy
    SCORE_UI = 26  # score.npy
    HP_UI = 27  # HP.npy
    SHIP_THRUST_BACK = 28
    # Score digits
    DIGIT_0 = 29
    DIGIT_1 = 30
    DIGIT_2 = 31
    DIGIT_3 = 32
    DIGIT_4 = 33
    DIGIT_5 = 34
    DIGIT_6 = 35
    DIGIT_7 = 36
    DIGIT_8 = 37
    DIGIT_9 = 38
    ENEMY_ORANGE_FLIPPED = 39
    SHIELD = 40
    REACTOR_DEST_HIT = 41  # reactor_destination_hit.npy


# Map angle index to sprite index (defined after SpriteIdx enum)
SHIP_SPRITE_INDICES = jnp.array([
    int(SpriteIdx.SHIP_N),    # 0: N
    int(SpriteIdx.SHIP_NNE),  # 1: NNE
    int(SpriteIdx.SHIP_NE),   # 2: NE
    int(SpriteIdx.SHIP_NEE),  # 3: NEE
    int(SpriteIdx.SHIP_E),    # 4: E
    int(SpriteIdx.SHIP_SEE),  # 5: SEE
    int(SpriteIdx.SHIP_SE),   # 6: SE
    int(SpriteIdx.SHIP_SSE),  # 7: SSE
    int(SpriteIdx.SHIP_S),    # 8: S
    int(SpriteIdx.SHIP_SSW),  # 9: SSW
    int(SpriteIdx.SHIP_SW),   # 10: SW
    int(SpriteIdx.SHIP_SWW),  # 11: SWW
    int(SpriteIdx.SHIP_W),    # 12: W
    int(SpriteIdx.SHIP_NWW),  # 13: NWW
    int(SpriteIdx.SHIP_NW),   # 14: NW
    int(SpriteIdx.SHIP_NNW),  # 15: NNW
], dtype=jnp.int32)

TERRAIN_SCALE_OVERRIDES = {
    SpriteIdx.TERRAIN2: 1,
}

LEVEL_LAYOUTS = {
    # Level 0 (Planet 1)
    0: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (37, 43)},
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (82, 31)},
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (152, -3)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (22, 71)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (104, 59)},
    ],
    # Level 1 (Planet 2)
    1: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (124, 18)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (83, 78)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (40, 38)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (44, 58)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (61, -2)},
    ],
    # Level 2 (Planet 3)
    2: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (24, 38)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (43, 82)},
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (60, -2)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (108, 22)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (135, 68)},
    ],
    # Level 3 (Planet 4)
    3: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (88, 27)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (116, 11)},
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (122, 113)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (76, 59)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (19, 95)},
    ],
    # Level 4 (Reactor)
    4: [],
}

LEVEL_OFFSETS = {
    0: (0, 50),
    1: (0, 7),
    2: (0, 44),
    3: (0, 26),
    4: (0, 14),
}

SPRITE_TO_LEVEL_ID = {
    int(SpriteIdx.PLANET1): 0,
    int(SpriteIdx.PLANET2): 1,
    int(SpriteIdx.PLANET3): 2,
    int(SpriteIdx.PLANET4): 3,
    int(SpriteIdx.REACTOR): 4,
}

LEVEL_ID_TO_TERRAIN_SPRITE = {
    0: SpriteIdx.TERRAIN1,
    1: SpriteIdx.TERRAIN2,
    2: SpriteIdx.TERRAIN3,
    3: SpriteIdx.TERRAIN4,
    4: SpriteIdx.REACTOR_TERR,
}

# 2. Maps the Level ID to the Terrain Bank Index (0=empty, 1=T1, 2=T2, etc.)
LEVEL_ID_TO_BANK_IDX = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
}


# ========== Bullet State ==========
# Defines the state of bullets
@struct.dataclass
class Bullets:
    x: jnp.ndarray  # shape(MAX_BULLETS, )
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    alive: jnp.ndarray  # boolean array
    sprite_idx: jnp.ndarray  # sprite index for each bullet (for different bullet types)

    def _replace(self, **kwargs):
        return self.replace(**kwargs)


# ========== Enemies States ==========
# Initializes the state of enemies
@struct.dataclass
class Enemies:
    x: jnp.ndarray  # shape (MAX_ENEMIES,)
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    vx: jnp.ndarray
    sprite_idx: jnp.ndarray
    death_timer: jnp.ndarray
    hp: jnp.ndarray

    def _replace(self, **kwargs):
        return self.replace(**kwargs)


# ========== Ship State ==========
@struct.dataclass
class ShipState:
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    angle: jnp.ndarray
    is_thrusting: jnp.ndarray  # Boolean flag to track if ship is actively thrusting
    rotation_cooldown: jnp.ndarray  # Frames until next rotation is allowed

    def _replace(self, **kwargs):
        return self.replace(**kwargs)


# ========== Saucer State ==========
@struct.dataclass
class SaucerState:
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    hp: jnp.ndarray
    alive: jnp.ndarray
    death_timer: jnp.ndarray

    def _replace(self, **kwargs):
        return self.replace(**kwargs)


# ========== UFO State ==========
@struct.dataclass
class UFOState:
    x: jnp.ndarray  # f32
    y: jnp.ndarray  # f32
    vx: jnp.ndarray  # f32
    vy: jnp.ndarray  # f32
    hp: jnp.ndarray  # i32
    alive: jnp.ndarray  # bool
    death_timer: jnp.ndarray

    def _replace(self, **kwargs):
        return self.replace(**kwargs)


# ========== FuelTanks State ==========
@struct.dataclass
class FuelTanks:
    x: jnp.ndarray  # (MAX_ENEMIES,)
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    sprite_idx: jnp.ndarray
    active: jnp.ndarray  # A boolean array to indicate if it's still active

    def _replace(self, **kwargs):
        return self.replace(**kwargs)


# ========== Env State ==========
@struct.dataclass
class EnvState:
    mode: jnp.ndarray
    state: ShipState
    bullets: Bullets
    cooldown: jnp.ndarray
    enemies: Enemies
    fuel_tanks: FuelTanks
    shield_active: jnp.ndarray
    enemy_bullets: Bullets
    fire_cooldown: jnp.ndarray
    key: jnp.ndarray
    key_alt: jnp.ndarray
    score: jnp.ndarray
    done: jnp.ndarray
    lives: jnp.ndarray
    crash_timer: jnp.ndarray
    planets_pi: jnp.ndarray
    planets_px: jnp.ndarray
    planets_py: jnp.ndarray
    planets_pr: jnp.ndarray
    planets_id: jnp.ndarray  # The ID of the entered level (int32)

    fuel: jnp.ndarray

    current_level: jnp.ndarray  # int32, current level ID (typically -1 in map mode)
    terrain_sprite_idx: jnp.ndarray  # int32, terrain sprite for the current level (TERRAIN* / REACTOR_TERR)
    terrain_mask: jnp.ndarray  # (Hmask, Wmask) uint8
    terrain_scale: jnp.ndarray  # float32, rendering scale factor
    terrain_offset: jnp.ndarray  # (2,) float32, screen-top-left offset [ox, oy]

    terrain_bank: jnp.ndarray  # int32，shape (B, H, W)
    terrain_bank_idx: jnp.ndarray  # int32, index of the currently used bank (0 = no terrain)
    respawn_shift_x: jnp.ndarray  # float32
    reactor_dest_active: jnp.ndarray  # bool
    reactor_dest_x: jnp.ndarray  # float32, world coordinates
    reactor_dest_y: jnp.ndarray  # float32
    reactor_dest_radius: jnp.ndarray  # float32, world coordinate reach radius

    # --- saucer / arena ---
    mode_timer: jnp.ndarray  # int32, cumulative frames in the current mode
    saucer: SaucerState
    map_return_x: jnp.ndarray  # float32
    map_return_y: jnp.ndarray  # float32
    map_return_vx: jnp.ndarray  # float32
    map_return_vy: jnp.ndarray  # float32
    map_return_angle: jnp.ndarray  # float32
    saucer_spawn_timer: jnp.ndarray  # Tracks if a saucer has spawned in the current level

    ufo: UFOState
    ufo_spawn_timer: jnp.ndarray  # int32, cooldown timer before UFO can spawn again
    ufo_home_x: jnp.ndarray  # f32
    ufo_home_y: jnp.ndarray  # f32
    ufo_bullets: Bullets
    level_offset: jnp.ndarray
    reactor_destroyed: jnp.ndarray
    planets_cleared_mask: jnp.ndarray

    reactor_timer: jnp.ndarray 
    reactor_activated: jnp.ndarray 
    exit_allowed: jnp.ndarray  # bool, whether ship can exit through top of level
    max_active_player_bullets_map: jnp.ndarray  # int32
    max_active_player_bullets_level: jnp.ndarray  # int32
    max_active_player_bullets_arena: jnp.ndarray  # int32
    max_active_saucer_bullets: jnp.ndarray  # int32
    max_active_enemy_bullets: jnp.ndarray  # int32
    enemy_fire_cooldown_frames: jnp.ndarray  # int32
    solar_gravity: jnp.ndarray  # float32
    planetary_gravity: jnp.ndarray  # float32
    reactor_gravity: jnp.ndarray  # float32
    fuel_consume_thrust: jnp.ndarray  # float32
    fuel_consume_shield_tractor: jnp.ndarray  # float32
    allow_tractor_in_reactor: jnp.ndarray  # bool
    enemy_kill_score: jnp.ndarray  # float32
    level_clear_score: jnp.ndarray  # float32
    ufo_kill_score: jnp.ndarray  # float32
    saucer_kill_score: jnp.ndarray  # float32
    thrust_power: jnp.ndarray  # float32 (unscaled; divided by WORLD_SCALE in physics)
    max_speed: jnp.ndarray  # float32 (unscaled; divided by WORLD_SCALE in physics)
    prev_action: jnp.ndarray  # int32, previous action taken

    def _replace(self, **kwargs):
        return self.replace(**kwargs)


@struct.dataclass
class GravitarObservation:
    ship: ObjectObservation
    enemies: ObjectObservation  # n = MAX_ENEMIES (turrets)
    fuel_tanks: ObjectObservation  # n = MAX_ENEMIES (planet pickups)
    saucer: ObjectObservation  # scalar
    ufo: ObjectObservation  # scalar
    planets: ObjectObservation  # n = _OBS_MAX_PLANETS (solar map objects)
    projectiles: ObjectObservation  # n = MAX_ENEMIES (enemy bulletspool)
    terrain: ObjectObservation  # scalar
    reactor_destination: ObjectObservation  # scalar
    lives: jnp.ndarray  # scalar int32
    fuel: jnp.ndarray  # scalar float32


@jax.jit
def _clip_xy_to_screen(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    cx = jnp.clip(x, 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    cy = jnp.clip(y, 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    return cx, cy


def _sprite_wh_scalar(sprite_idx: jnp.ndarray, fallback_w: int = 0, fallback_h: int = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = sprite_idx.astype(jnp.int32)
    max_idx = _OBS_SPRITE_DIMS.shape[0] - 1
    safe_idx = jnp.clip(idx, 0, max_idx)
    wh = _OBS_SPRITE_DIMS[safe_idx]
    valid = (idx >= 0) & (idx <= max_idx)
    w = jnp.where(valid, wh[0], jnp.array(fallback_w, dtype=jnp.int16))
    h = jnp.where(valid, wh[1], jnp.array(fallback_h, dtype=jnp.int16))
    return w, h


def _sprite_wh_vector(sprite_idx: jnp.ndarray, fallback_w: int = 0, fallback_h: int = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = sprite_idx.astype(jnp.int32)
    max_idx = _OBS_SPRITE_DIMS.shape[0] - 1
    safe_idx = jnp.clip(idx, 0, max_idx)
    wh = _OBS_SPRITE_DIMS[safe_idx]
    valid = (idx >= 0) & (idx <= max_idx)
    w = jnp.where(valid, wh[:, 0], jnp.full(idx.shape, fallback_w, dtype=jnp.int16))
    h = jnp.where(valid, wh[:, 1], jnp.full(idx.shape, fallback_h, dtype=jnp.int16))
    return w, h


@jax.jit
def _get_observation_from_state(state: EnvState) -> GravitarObservation:
    ship: ShipState = state.state
    enemies: Enemies = state.enemies
    fuel_tanks: FuelTanks = state.fuel_tanks
    saucer: SaucerState = state.saucer
    ufo: UFOState = state.ufo
    enemy_bullets: Bullets = state.enemy_bullets

    # --- Ship ---
    sx, sy = _clip_xy_to_screen(ship.x, ship.y)
    ship_active = jnp.array(1, dtype=jnp.int32)
    ship_visual_id = get_ship_sprite_idx(ship.angle).astype(jnp.int16)
    ship_orientation = ship.angle.astype(jnp.float32)
    ship_w, ship_h = _sprite_wh_scalar(ship_visual_id, fallback_w=3, fallback_h=7)

    ship_obj = ObjectObservation.create(
        x=sx,
        y=sy,
        width=ship_w,
        height=ship_h,
        active=ship_active,
        visual_id=ship_visual_id,
        orientation=ship_orientation,
        state=jnp.array(0, dtype=jnp.int32),
    )

    # --- Enemies (turrets) ---
    enemy_present = (enemies.hp > 0) | (enemies.death_timer > 0)
    enemy_present_i = enemy_present.astype(jnp.int32)
    ex = jnp.clip(enemies.x, 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    ey = jnp.clip(enemies.y, 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    ew = jnp.clip(enemies.w, 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    eh = jnp.clip(enemies.h, 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    e_visual = jnp.where(enemy_present, enemies.sprite_idx, jnp.int32(0)).astype(jnp.int16)

    enemies_obj = ObjectObservation.create(
        x=ex,
        y=ey,
        width=ew,
        height=eh,
        active=enemy_present_i,
        visual_id=e_visual,
        orientation=jnp.zeros_like(enemies.x, dtype=jnp.float32),
        state=jnp.zeros_like(enemies.x, dtype=jnp.int32),
    )

    # --- Fuel tanks (planet pickups) ---
    tank_present = fuel_tanks.active
    tx = jnp.clip(fuel_tanks.x, 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    ty = jnp.clip(fuel_tanks.y, 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    tw = jnp.clip(fuel_tanks.w, 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    th = jnp.clip(fuel_tanks.h, 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    t_visual = jnp.where(tank_present, fuel_tanks.sprite_idx, jnp.int32(0)).astype(jnp.int16)

    fuel_tanks_obj = ObjectObservation.create(
        x=tx,
        y=ty,
        width=tw,
        height=th,
        active=tank_present.astype(jnp.int32),
        visual_id=t_visual,
        orientation=jnp.zeros_like(fuel_tanks.x, dtype=jnp.float32),
        state=jnp.zeros_like(fuel_tanks.x, dtype=jnp.int32),
    )

    # --- Saucer (single) ---
    saucer_present = (saucer.alive | (saucer.death_timer > 0))
    saucer_active_i = saucer_present.astype(jnp.int32)
    sax, say = _clip_xy_to_screen(saucer.x, saucer.y)
    saucer_visual_id = jnp.array(int(SpriteIdx.ENEMY_SAUCER), dtype=jnp.int16)
    saucer_w, saucer_h = _sprite_wh_scalar(saucer_visual_id, fallback_w=8, fallback_h=7)
    saucer_obj = ObjectObservation.create(
        x=sax,
        y=say,
        width=saucer_w,
        height=saucer_h,
        active=saucer_active_i,
        visual_id=saucer_visual_id,
        orientation=jnp.array(0.0, dtype=jnp.float32),
        state=jnp.array(1, dtype=jnp.int32),
    )

    # --- UFO (single) ---
    ufo_present = (ufo.alive | (ufo.death_timer > 0))
    ufo_active_i = ufo_present.astype(jnp.int32)
    uax, uay = _clip_xy_to_screen(ufo.x, ufo.y)
    ufo_visual_id = jnp.array(int(SpriteIdx.ENEMY_UFO), dtype=jnp.int16)
    ufo_w, ufo_h = _sprite_wh_scalar(ufo_visual_id, fallback_w=7, fallback_h=6)
    ufo_obj = ObjectObservation.create(
        x=uax,
        y=uay,
        width=ufo_w,
        height=ufo_h,
        active=ufo_active_i,
        visual_id=ufo_visual_id,
        orientation=jnp.array(0.0, dtype=jnp.float32),
        state=jnp.array(0, dtype=jnp.int32),
    )

    # --- Enemy bullets (pool size MAX_ENEMIES) ---
    pb_alive = enemy_bullets.alive
    px = jnp.clip(enemy_bullets.x, 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    py = jnp.clip(enemy_bullets.y, 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    p_visual = jnp.where(pb_alive, enemy_bullets.sprite_idx, jnp.int32(0)).astype(jnp.int16)
    bullet_w, bullet_h = _sprite_wh_vector(p_visual, fallback_w=1, fallback_h=2)

    # --- Solar system objects (planets/reactor/obstacle/spawn marker) ---
    planets_active = (state.planets_pi >= 0).astype(jnp.int32)
    planet_x = jnp.clip(state.planets_px, 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    planet_y = jnp.clip(state.planets_py, 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    planet_visual = jnp.where(state.planets_pi >= 0, state.planets_pi, jnp.int32(0)).astype(jnp.int16)
    planet_w, planet_h = _sprite_wh_vector(planet_visual, fallback_w=0, fallback_h=0)

    planets_obj = ObjectObservation.create(
        x=planet_x,
        y=planet_y,
        width=planet_w,
        height=planet_h,
        active=planets_active,
        visual_id=planet_visual,
        orientation=jnp.zeros_like(state.planets_px, dtype=jnp.float32),
        state=state.planets_cleared_mask.astype(jnp.int32),
    )

    terrain_active = (state.terrain_sprite_idx >= 0).astype(jnp.int32)
    terrain_x = jnp.clip(state.terrain_offset[0], 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    terrain_y = jnp.clip(state.terrain_offset[1], 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    terrain_visual = jnp.where(state.terrain_sprite_idx >= 0, state.terrain_sprite_idx, jnp.int32(0)).astype(jnp.int16)
    terrain_w, terrain_h = _sprite_wh_scalar(terrain_visual, fallback_w=WINDOW_WIDTH, fallback_h=WINDOW_HEIGHT)

    terrain_obj = ObjectObservation.create(
        x=terrain_x,
        y=terrain_y,
        width=terrain_w,
        height=terrain_h,
        active=terrain_active,
        visual_id=terrain_visual,
        orientation=jnp.array(0.0, dtype=jnp.float32),
        state=state.terrain_bank_idx.astype(jnp.int32),
    )

    reactor_dest_active = state.reactor_dest_active.astype(jnp.int32)
    reactor_dest_x = jnp.clip(state.reactor_dest_x, 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    reactor_dest_y = jnp.clip(state.reactor_dest_y, 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    reactor_dest_visual = jnp.where(
        state.reactor_destroyed,
        jnp.int32(int(SpriteIdx.REACTOR_DEST_HIT)),
        jnp.int32(int(SpriteIdx.REACTOR_DEST)),
    ).astype(jnp.int16)
    reactor_dest_w, reactor_dest_h = _sprite_wh_scalar(reactor_dest_visual, fallback_w=5, fallback_h=5)

    reactor_destination_obj = ObjectObservation.create(
        x=reactor_dest_x,
        y=reactor_dest_y,
        width=reactor_dest_w,
        height=reactor_dest_h,
        active=reactor_dest_active,
        visual_id=reactor_dest_visual,
        orientation=jnp.array(0.0, dtype=jnp.float32),
        state=state.reactor_activated.astype(jnp.int32),
    )

    projectiles_obj = ObjectObservation.create(
        x=px,
        y=py,
        width=bullet_w,
        height=bullet_h,
        active=pb_alive.astype(jnp.int32),
        visual_id=p_visual,
        orientation=jnp.zeros_like(px, dtype=jnp.float32),
        state=jnp.zeros_like(px, dtype=jnp.int32),
    )

    return GravitarObservation(
        ship=ship_obj,
        enemies=enemies_obj,
        fuel_tanks=fuel_tanks_obj,
        saucer=saucer_obj,
        ufo=ufo_obj,
        planets=planets_obj,
        projectiles=projectiles_obj,
        terrain=terrain_obj,
        reactor_destination=reactor_destination_obj,
        lives=jnp.clip(state.lives, 0, MAX_LIVES).astype(jnp.int32),
        fuel=jnp.maximum(state.fuel, 0.0).astype(jnp.float32),
    )


@jax.jit
def _get_observation_from_ship_state(ship: ShipState) -> GravitarObservation:
    sx, sy = _clip_xy_to_screen(ship.x, ship.y)
    ship_visual_id = get_ship_sprite_idx(ship.angle).astype(jnp.int16)
    ship_w, ship_h = _sprite_wh_scalar(ship_visual_id, fallback_w=3, fallback_h=7)

    ship_obj = ObjectObservation.create(
        x=sx,
        y=sy,
        width=ship_w,
        height=ship_h,
        active=jnp.array(1, dtype=jnp.int32),
        visual_id=ship_visual_id,
        orientation=ship.angle.astype(jnp.float32),
        state=jnp.array(0, dtype=jnp.int32),
    )

    inactive_scalar = jnp.array(0, dtype=jnp.int32)
    inactive_scalar16 = jnp.array(0, dtype=jnp.int16)
    inactive_pool = jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32)
    inactive_pool16 = jnp.zeros((MAX_ENEMIES,), dtype=jnp.int16)
    inactive_bullets = jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32)
    inactive_bullets16 = jnp.zeros((MAX_ENEMIES,), dtype=jnp.int16)
    zero_orientation_pool = jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32)
    zero_orientation_bullets = jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32)
    inactive_planets = jnp.zeros((_OBS_MAX_PLANETS,), dtype=jnp.int32)
    inactive_planets16 = jnp.zeros((_OBS_MAX_PLANETS,), dtype=jnp.int16)
    zero_orientation_planets = jnp.zeros((_OBS_MAX_PLANETS,), dtype=jnp.float32)

    saucer_visual_id = jnp.array(int(SpriteIdx.ENEMY_SAUCER), dtype=jnp.int16)
    saucer_w, saucer_h = _sprite_wh_scalar(saucer_visual_id, fallback_w=8, fallback_h=7)
    ufo_visual_id = jnp.array(int(SpriteIdx.ENEMY_UFO), dtype=jnp.int16)
    ufo_w, ufo_h = _sprite_wh_scalar(ufo_visual_id, fallback_w=7, fallback_h=6)
    bullet_visual_id = jnp.full((MAX_ENEMIES,), int(SpriteIdx.ENEMY_BULLET), dtype=jnp.int16)
    bullet_w, bullet_h = _sprite_wh_vector(bullet_visual_id, fallback_w=1, fallback_h=2)

    enemies_obj = ObjectObservation.create(
        x=inactive_pool16,
        y=inactive_pool16,
        width=inactive_pool16,
        height=inactive_pool16,
        active=inactive_pool,
        visual_id=inactive_pool16,
        orientation=zero_orientation_pool,
        state=inactive_pool,
    )

    fuel_tanks_obj = ObjectObservation.create(
        x=inactive_pool16,
        y=inactive_pool16,
        width=inactive_pool16,
        height=inactive_pool16,
        active=inactive_pool,
        visual_id=inactive_pool16,
        orientation=zero_orientation_pool,
        state=inactive_pool,
    )

    saucer_obj = ObjectObservation.create(
        x=inactive_scalar16,
        y=inactive_scalar16,
        width=saucer_w,
        height=saucer_h,
        active=inactive_scalar,
        visual_id=saucer_visual_id,
        orientation=jnp.array(0.0, dtype=jnp.float32),
        state=inactive_scalar,
    )

    ufo_obj = ObjectObservation.create(
        x=inactive_scalar16,
        y=inactive_scalar16,
        width=ufo_w,
        height=ufo_h,
        active=inactive_scalar,
        visual_id=ufo_visual_id,
        orientation=jnp.array(0.0, dtype=jnp.float32),
        state=inactive_scalar,
    )

    planets_obj = ObjectObservation.create(
        x=inactive_planets16,
        y=inactive_planets16,
        width=inactive_planets16,
        height=inactive_planets16,
        active=inactive_planets,
        visual_id=inactive_planets16,
        orientation=zero_orientation_planets,
        state=inactive_planets,
    )

    terrain_obj = ObjectObservation.create(
        x=inactive_scalar16,
        y=inactive_scalar16,
        width=jnp.array(WINDOW_WIDTH, dtype=jnp.int16),
        height=jnp.array(WINDOW_HEIGHT, dtype=jnp.int16),
        active=inactive_scalar,
        visual_id=inactive_scalar16,
        orientation=jnp.array(0.0, dtype=jnp.float32),
        state=inactive_scalar,
    )

    reactor_destination_obj = ObjectObservation.create(
        x=inactive_scalar16,
        y=inactive_scalar16,
        width=jnp.array(5, dtype=jnp.int16),
        height=jnp.array(5, dtype=jnp.int16),
        active=inactive_scalar,
        visual_id=jnp.array(int(SpriteIdx.REACTOR_DEST), dtype=jnp.int16),
        orientation=jnp.array(0.0, dtype=jnp.float32),
        state=inactive_scalar,
    )

    projectiles_obj = ObjectObservation.create(
        x=inactive_bullets16,
        y=inactive_bullets16,
        width=bullet_w,
        height=bullet_h,
        active=inactive_bullets,
        visual_id=inactive_bullets16,
        orientation=zero_orientation_bullets,
        state=inactive_bullets,
    )

    return GravitarObservation(
        ship=ship_obj,
        enemies=enemies_obj,
        fuel_tanks=fuel_tanks_obj,
        saucer=saucer_obj,
        ufo=ufo_obj,
        planets=planets_obj,
        projectiles=projectiles_obj,
        terrain=terrain_obj,
        reactor_destination=reactor_destination_obj,
        lives=jnp.array(0, dtype=jnp.int32),
        fuel=jnp.array(0.0, dtype=jnp.float32),
    )


@struct.dataclass
class GravitarInfo:
    lives: jnp.ndarray
    score: jnp.ndarray
    fuel: jnp.ndarray
    mode: jnp.ndarray
    crash_timer: jnp.ndarray
    done: jnp.ndarray
    current_level: jnp.ndarray
    crash: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(False))
    hit_by_bullet: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(False))
    reactor_crash_exit: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(False))
    all_rewards: jnp.ndarray = struct.field(default_factory=lambda: jnp.zeros((5,), dtype=jnp.float32))
    level_cleared: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(False))

    def _replace(self, **kwargs):
        return self.replace(**kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)

# ========== Init Function ==========

def make_empty_ufo() -> UFOState:
    f32 = jnp.float32
    i32 = jnp.int32
    return UFOState(
        x=f32(0.0), y=f32(0.0),
        vx=f32(0.0), vy=f32(0.0),
        hp=i32(0),
        alive=jnp.array(False),
        death_timer=i32(0)
    )


def make_default_saucer() -> SaucerState:
    return SaucerState(
        x=jnp.float32(-999.0), y=jnp.float32(-999.0),
        vx=jnp.float32(0.0), vy=jnp.float32(0.0),
        hp=jnp.int32(0),
        alive=jnp.array(False),
        death_timer=jnp.int32(0),
    )


# Maps planet sprite indices to terrain bank indices (0=empty, 1..4 correspond to TERRAIN1..4)
@jax.jit
def planet_to_bank_idx(psi: jnp.ndarray) -> jnp.ndarray:
    b = jnp.int32(0)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.PLANET1)), jnp.int32(1), b)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.PLANET2)), jnp.int32(2), b)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.PLANET3)), jnp.int32(3), b)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.PLANET4)), jnp.int32(4), b)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.REACTOR)), jnp.int32(5), b)
    return b


@jax.jit
def map_planet_to_terrain(planet_sprite_idx: jnp.ndarray) -> jnp.ndarray:
    P1 = jnp.int32(int(SpriteIdx.PLANET1))
    P2 = jnp.int32(int(SpriteIdx.PLANET2))
    P3 = jnp.int32(int(SpriteIdx.PLANET3))
    P4 = jnp.int32(int(SpriteIdx.PLANET4))
    PR = jnp.int32(int(SpriteIdx.REACTOR))

    T1 = jnp.int32(int(SpriteIdx.TERRAIN1))
    T2 = jnp.int32(int(SpriteIdx.TERRAIN2))
    T3 = jnp.int32(int(SpriteIdx.TERRAIN3))
    T4 = jnp.int32(int(SpriteIdx.TERRAIN4))
    TR = jnp.int32(int(SpriteIdx.REACTOR_TERR))

    invalid = jnp.int32(-1)
    out = invalid
    out = jnp.where(planet_sprite_idx == P1, T1, out)
    out = jnp.where(planet_sprite_idx == P2, T2, out)
    out = jnp.where(planet_sprite_idx == P3, T3, out)
    out = jnp.where(planet_sprite_idx == P4, T4, out)
    out = jnp.where(planet_sprite_idx == PR, TR, out)
    return out


def _opt(name_wo_ext: str):
    sprite_dir = os.path.join(os.path.dirname(__file__), "sprites", "gravitar")
    path = os.path.join(sprite_dir, f"{name_wo_ext}.npy")
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        # Normalize to uint8; scale float 0-1 or uint8 0/1 to 0-255
        if np.issubdtype(arr.dtype, np.floating):
            if 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0:
                arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype == np.uint8 and arr.max() <= 1:
            arr = (arr * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Convert black background and add alpha channel
        rgb = arr[..., :3]
        alpha = (rgb.max(axis=-1) >= 1).astype(np.uint8) * 255
        rgba = np.dstack([rgb, alpha])
        return rgba
    except Exception as e:
        return None


def _load_and_convert_sprites():
    numpy_sprites = load_sprites_tuple()

    def array_to_jax(arr):
        if arr is None:
            return None
        return jnp.array(arr).astype(jnp.uint8)

    jax_sprites = {}
    for i, arr in enumerate(numpy_sprites):
        if arr is not None:
            jax_sprites[i] = array_to_jax(arr)
    return jax_sprites


def load_sprites_tuple() -> tuple:
    # 1. Create a list large enough to hold all sprites, filled with None
    #    This ensures the index corresponds perfectly with the SpriteIdx value.
    num_sprites = max(int(e) for e in SpriteIdx) + 1
    sprites = [None] * num_sprites

    # 2. Define a dictionary to map SpriteIdx enums to their .npy filenames (without extension).
    #    This is the central place for managing all sprites. Adding a new one just requires one new line here.
    sprite_map = {
        SpriteIdx.SHIP_IDLE: "spaceship",  # N (north)
        SpriteIdx.SHIP_THRUST: "ship_thrust",
        SpriteIdx.SHIP_BULLET: "ship_bullet",
        # 16 discrete ship orientations
        SpriteIdx.SHIP_NNE: "spaceship_nne",
        SpriteIdx.SHIP_NE: "spaceship_ne",
        SpriteIdx.SHIP_NEE: "spaceship_nee",
        SpriteIdx.SHIP_E: "spaceship_e",
        SpriteIdx.SHIP_SEE: "spaceship_see",
        SpriteIdx.SHIP_SE: "spaceship_se",
        SpriteIdx.SHIP_SSE: "spaceship_sse",
        SpriteIdx.SHIP_S: "spaceship_s",
        SpriteIdx.SHIP_SSW: "spaceship_ssw",
        SpriteIdx.SHIP_SW: "spaceship_sw",
        SpriteIdx.SHIP_SWW: "spaceship_sww",
        SpriteIdx.SHIP_W: "spaceship_w",
        SpriteIdx.SHIP_NWW: "spaceship_nww",
        SpriteIdx.SHIP_NW: "spaceship_nw",
        SpriteIdx.SHIP_NNW: "spaceship_nnw",
        SpriteIdx.ENEMY_BULLET: "enemy_bullet",
        SpriteIdx.ENEMY_GREEN_BULLET: "enemy_green_bullet",
        SpriteIdx.ENEMY_ORANGE: "enemy_orange",
        # SpriteIdx.ENEMY_ORANGE_FLIPPED: "enemy_orange_flipped",
        SpriteIdx.ENEMY_GREEN: "enemy_green",
        SpriteIdx.ENEMY_SAUCER: "saucer",
        SpriteIdx.ENEMY_UFO: "UFO",
        SpriteIdx.ENEMY_CRASH: "enemy_crash",
        SpriteIdx.SAUCER_CRASH: "saucer_crash",
        SpriteIdx.SHIP_CRASH: "ship_crash",
        SpriteIdx.FUEL_TANK: "fuel_tank",
        SpriteIdx.OBSTACLE: "obstacle",
        SpriteIdx.SPAWN_LOC: "spawn_location",
        SpriteIdx.REACTOR: "reactor",
        SpriteIdx.REACTOR_TERR: "reactor_terrain",
        SpriteIdx.TERRAIN1: "terrain1",
        SpriteIdx.TERRAIN2: "terrain2",
        SpriteIdx.TERRAIN3: "terrain3",
        SpriteIdx.TERRAIN4: "terrain4",
        SpriteIdx.PLANET1: "planet1",
        SpriteIdx.PLANET2: "planet2",
        SpriteIdx.PLANET3: "planet3",
        SpriteIdx.PLANET4: "planet4",
        SpriteIdx.REACTOR_DEST: "reactor_destination",
        SpriteIdx.REACTOR_DEST_HIT: "reactor_destination_hit",
        SpriteIdx.SCORE_UI: "score",
        SpriteIdx.HP_UI: "HP",
        SpriteIdx.SHIP_THRUST_BACK: "ship_thrust_back",
        SpriteIdx.DIGIT_0: "score_0",
        SpriteIdx.DIGIT_1: "score_1",
        SpriteIdx.DIGIT_2: "score_2",
        SpriteIdx.DIGIT_3: "score_3",
        SpriteIdx.DIGIT_4: "score_4",
        SpriteIdx.DIGIT_5: "score_5",
        SpriteIdx.DIGIT_6: "score_6",
        SpriteIdx.DIGIT_7: "score_7",
        SpriteIdx.DIGIT_8: "score_8",
        SpriteIdx.DIGIT_9: "score_9",
        SpriteIdx.SHIELD: "shield",
    }

    # 3. Iterate through the map dictionary, calling _opt to load all base sprites.
    for idx_enum, name in sprite_map.items():
        sprites[int(idx_enum)] = _opt(name)

    # 4. Manually create flipped versions of sprites in memory.
    orange_surf = sprites[int(SpriteIdx.ENEMY_ORANGE)]
    if orange_surf is not None:
        sprites[int(SpriteIdx.ENEMY_ORANGE_FLIPPED)] = np.flip(orange_surf, axis=0)

    # 5. Convert the final list to a tuple and return.
    return tuple(sprites)


def _build_obs_sprite_dims(sprites: tuple) -> jnp.ndarray:
    max_sprite_id = max(int(e) for e in SpriteIdx)
    dims = np.zeros((max_sprite_id + 1, 2), dtype=np.int16)
    for sprite_idx in range(len(sprites)):
        surf = sprites[sprite_idx]
        if surf is not None:
            dims[sprite_idx, 0] = np.int16(surf.shape[1])
            dims[sprite_idx, 1] = np.int16(surf.shape[0])
    return jnp.array(dims, dtype=jnp.int16)


_OBS_SPRITES = load_sprites_tuple()
_OBS_SPRITE_DIMS = _build_obs_sprite_dims(_OBS_SPRITES)


# Initializes an empty bullet pool
def create_empty_bullets_fixed(size: int) -> Bullets:
    return Bullets(
        x=jnp.zeros((size,), dtype=jnp.float32),
        y=jnp.zeros((size,), dtype=jnp.float32),
        vx=jnp.zeros((size,), dtype=jnp.float32),
        vy=jnp.zeros((size,), dtype=jnp.float32),
        alive=jnp.zeros((size,), dtype=bool),
        sprite_idx=jnp.full((size,), int(SpriteIdx.ENEMY_BULLET), dtype=jnp.int32)
    )


def create_empty_bullets_64():
    return create_empty_bullets_fixed(MAX_BULLETS)


def create_empty_bullets_16():
    return create_empty_bullets_fixed(MAX_ENEMIES)


@jax.jit
def create_empty_enemies():
    return Enemies(
        x=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        y=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        w=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        h=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        vx=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        sprite_idx=jnp.full((MAX_ENEMIES,), -1, dtype=jnp.int32),
        death_timer=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
        hp=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32)
    )


@jax.jit
def create_env_state(rng: jnp.ndarray) -> EnvState:
    # ALE spawn location coordinates: (75, 131)
    spawn_x = jnp.array(76.0, dtype=jnp.float32)
    spawn_y = jnp.array(130.0, dtype=jnp.float32)

    return EnvState(
        mode=jnp.int32(1),
        state=ShipState(
            x=spawn_x,
            y=spawn_y,
            vx=jnp.array(0.0),
            vy=jnp.array(0.0),
            angle=jnp.array(-jnp.pi / 2),
            is_thrusting=jnp.array(False),
            rotation_cooldown=jnp.int32(0)
        ),
        bullets=create_empty_bullets_64(),
        cooldown=jnp.array(0, dtype=jnp.int32),
        enemies=create_empty_enemies(),
        fuel_tanks=FuelTanks(
            x=jnp.zeros(MAX_ENEMIES), y=jnp.zeros(MAX_ENEMIES), w=jnp.zeros(MAX_ENEMIES),
            h=jnp.zeros(MAX_ENEMIES), sprite_idx=jnp.full(MAX_ENEMIES, -1),
            active=jnp.zeros(MAX_ENEMIES, dtype=bool)
        ),
        shield_active=jnp.array(False),
        enemy_bullets=create_empty_bullets_16(),
        fire_cooldown=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
        key=rng,
        key_alt=rng,
        score=jnp.array(0.0),
        done=jnp.array(False),
        lives=jnp.array(3, dtype=jnp.int32),
        crash_timer=jnp.int32(0),
        planets_pi=jnp.zeros(7, dtype=jnp.int32), 
        planets_px=jnp.zeros(7, dtype=jnp.float32),
        planets_py=jnp.zeros(7, dtype=jnp.float32),
        planets_pr=jnp.zeros(7, dtype=jnp.float32),
        planets_id=jnp.zeros(7, dtype=jnp.int32),
        fuel=jnp.array(STARTING_FUEL, dtype=jnp.float32),
        current_level=jnp.int32(-1),
        terrain_sprite_idx=jnp.int32(-1),
        terrain_mask=jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=jnp.int32),
        terrain_scale=jnp.array(1.0),
        terrain_offset=jnp.array([0.0, 0.0]),
        terrain_bank=jnp.zeros((6, WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=jnp.int32),
        terrain_bank_idx=jnp.int32(0),
        reactor_timer=jnp.int32(0),
        reactor_activated=jnp.array(False),
        respawn_shift_x=jnp.float32(0.0),
        reactor_dest_active=jnp.array(False),
        reactor_dest_x=jnp.float32(0.0),
        reactor_dest_y=jnp.float32(0.0),
        reactor_dest_radius=jnp.float32(0.4),
        mode_timer=jnp.int32(0),
        saucer=make_default_saucer(),
        map_return_x=jnp.float32(0.0),
        map_return_y=jnp.float32(0.0),
        map_return_vx=jnp.float32(0.0),
        map_return_vy=jnp.float32(0.0),
        map_return_angle=jnp.float32(-jnp.pi / 2),
        saucer_spawn_timer=jnp.int32(SAUCER_SPAWN_DELAY_FRAMES),
        ufo=make_empty_ufo(),
        ufo_spawn_timer=jnp.int32(0),
        ufo_home_x=jnp.float32(0.0),
        ufo_home_y=jnp.float32(0.0),
        ufo_bullets=create_empty_bullets_16(),
        level_offset=jnp.array([0, 0], dtype=jnp.float32),
        reactor_destroyed=jnp.array(False),
        planets_cleared_mask=jnp.zeros(7, dtype=bool),
        exit_allowed=jnp.array(False),
        max_active_player_bullets_map=jnp.int32(MAX_ACTIVE_PLAYER_BULLETS_MAP),
        max_active_player_bullets_level=jnp.int32(MAX_ACTIVE_PLAYER_BULLETS_LEVEL),
        max_active_player_bullets_arena=jnp.int32(MAX_ACTIVE_PLAYER_BULLETS_ARENA),
        max_active_saucer_bullets=jnp.int32(MAX_ACTIVE_SAUCER_BULLETS),
        max_active_enemy_bullets=jnp.int32(MAX_ACTIVE_ENEMY_BULLETS),
        enemy_fire_cooldown_frames=jnp.int32(ENEMY_FIRE_COOLDOWN_FRAMES),
        solar_gravity=jnp.float32(_DEFAULT_CONSTS.SOLAR_GRAVITY),
        planetary_gravity=jnp.float32(_DEFAULT_CONSTS.PLANETARY_GRAVITY),
        reactor_gravity=jnp.float32(_DEFAULT_CONSTS.REACTOR_GRAVITY),
        thrust_power=jnp.float32(_DEFAULT_CONSTS.THRUST_POWER),
        max_speed=jnp.float32(_DEFAULT_CONSTS.MAX_SPEED),
        fuel_consume_thrust=jnp.float32(_DEFAULT_CONSTS.FUEL_CONSUME_THRUST),
        fuel_consume_shield_tractor=jnp.float32(_DEFAULT_CONSTS.FUEL_CONSUME_SHIELD_TRACTOR),
        allow_tractor_in_reactor=jnp.array(_DEFAULT_CONSTS.ALLOW_TRACTOR_IN_REACTOR),
        enemy_kill_score=jnp.float32(ENEMY_KILL_SCORE),
        level_clear_score=jnp.float32(LEVEL_CLEAR_SCORE),
        ufo_kill_score=jnp.float32(UFO_KILL_SCORE),
        saucer_kill_score=jnp.float32(SAUCER_KILL_SCORE),
        prev_action=jnp.int32(0),
    )


@jax.jit
def make_level_start_state(level_id: int) -> ShipState:
    START_Y = jnp.float32(44.0)
    REACTOR_START_Y = jnp.float32(68.0)  # Lower spawn point for reactor

    x = jnp.array(WINDOW_WIDTH / 2 + 5.0, dtype=jnp.float32)
    y = jnp.array(START_Y, dtype=jnp.float32)

    angle = jnp.array(-jnp.pi / 2, dtype=jnp.float32)  # Pointing up for normal levels
    angle_down = jnp.array(jnp.pi / 2, dtype=jnp.float32)  # Pointing down for reactor

    is_reactor = (jnp.asarray(level_id, dtype=jnp.int32) == 4)
    x = jnp.where(is_reactor, x - 55.0, x)
    y = jnp.where(is_reactor, REACTOR_START_Y, y)
    angle = jnp.where(is_reactor, angle_down, angle)

    return ShipState(x=x, y=y, vx=jnp.float32(0.0), vy=jnp.float32(0.0), angle=angle, is_thrusting=jnp.array(False), rotation_cooldown=jnp.int32(0))


# ========== Update Bullets ==========
@jax.jit
def update_bullets(bullets: Bullets) -> Bullets:
    new_x = bullets.x + bullets.vx
    new_y = bullets.y + bullets.vy

    valid_x = (new_x >= 0) & (new_x < WINDOW_WIDTH)
    valid_y = (new_y >= HUD_HEIGHT) & (new_y < WINDOW_HEIGHT)
    valid = valid_x & valid_y & bullets.alive

    return Bullets(
        x=new_x,
        y=new_y,
        vx=bullets.vx,
        vy=bullets.vy,
        alive=valid,
        sprite_idx=bullets.sprite_idx
    )


# ========== Merge Bullets ==========
@jax.jit
def merge_bullets(prev: Bullets, add: Bullets, max_len: int | None = None) -> Bullets:
    # provided, default to the capacity of the previous pool (static shape)
    cap = prev.x.shape[0] if (max_len is None) else max_len
    cap_i = jnp.int32(cap)

    # First, enforce the cap on the old bullet pool to prioritize keeping older bullets
    prev0 = _enforce_cap_keep_old(prev, cap_i)
    used = jnp.minimum(_bullets_alive_count(prev0), cap_i)
    space_left0 = cap_i - used

    def place_one(carry, i):
        b, space_left = carry
        take = add.alive[i] & (space_left > 0)

        # Find the first available empty slot
        dead_mask = ~b.alive
        has_slot = jnp.any(dead_mask)

        take = take & has_slot
        idx = jnp.argmax(dead_mask.astype(jnp.int32))

        b2 = jax.lax.cond(
            take,
            lambda _: Bullets(
                x=b.x.at[idx].set(add.x[i]),
                y=b.y.at[idx].set(add.y[i]),
                vx=b.vx.at[idx].set(add.vx[i]),
                vy=b.vy.at[idx].set(add.vy[i]),
                alive=b.alive.at[idx].set(True),
                sprite_idx=b.sprite_idx.at[idx].set(add.sprite_idx[i]),
            ),
            lambda _: b,
            operand=None
        )
        space2 = jnp.where(take, space_left - 1, space_left)

        return (b2, space2), None

    n_add = add.x.shape[0]  # Static length
    (merged, _), _ = jax.lax.scan(place_one, (prev0, space_left0), jnp.arange(n_add))

    # Re-enforce the cap to ensure the total number of alive bullets does not exceed the capacity
    merged = _enforce_cap_keep_old(merged, cap_i)

    return merged


@jax.jit
def _bullets_alive_count(bullets: Bullets):
    return jnp.asarray(jnp.sum(bullets.alive.astype(jnp.int32)), dtype=jnp.int32)


@jax.jit
def _enforce_cap_keep_old(b: Bullets, cap: int) -> Bullets:
    cap_i = jnp.int32(cap)
    rank = jnp.cumsum(b.alive.astype(jnp.int32)) - 1  # Sequential number for each alive bullet (0,1,2,...)
    keep = b.alive & (rank < cap_i)

    return Bullets(x=b.x, y=b.y, vx=b.vx, vy=b.vy, alive=keep, sprite_idx=b.sprite_idx)


# ========== Fire Bullet ==========
# Fires a new bullet
@jax.jit
def fire_bullet(bullets: Bullets, ship_x, ship_y, ship_angle, bullet_speed):
    def add_bullet(_):
        idx = jnp.argmax(bullets.alive == False)
        new_vx = jnp.cos(ship_angle) * bullet_speed
        new_vy = jnp.sin(ship_angle) * bullet_speed

        return Bullets(
            x=bullets.x.at[idx].set(ship_x),
            y=bullets.y.at[idx].set(ship_y),
            vx=bullets.vx.at[idx].set(new_vx),
            vy=bullets.vy.at[idx].set(new_vy),
            alive=bullets.alive.at[idx].set(True),
            sprite_idx=bullets.sprite_idx.at[idx].set(int(SpriteIdx.SHIP_BULLET))
        )

    def skip_bullet(_):
        return bullets

    can_fire = jnp.any(bullets.alive == False)

    return jax.lax.cond(can_fire, add_bullet, skip_bullet, operand=None)


@jax.jit
def _fire_single_from_to(bullets: Bullets, sx, sy, tx, ty, speed=jnp.float32(0.7)) -> Bullets:
    dx = tx - sx
    dy = ty - sy
    d = jnp.maximum(jnp.sqrt(dx * dx + dy * dy), 1e-3)
    vx = speed * dx / d
    vy = speed * dy / d

    one = Bullets(
        x=jnp.array([sx], dtype=jnp.float32),
        y=jnp.array([sy], dtype=jnp.float32),
        vx=jnp.array([vx], dtype=jnp.float32),
        vy=jnp.array([vy], dtype=jnp.float32),
        alive=jnp.array([True]),
        sprite_idx=jnp.array([int(SpriteIdx.ENEMY_BULLET)], dtype=jnp.int32)
    )

    return merge_bullets(bullets, one, max_len=16)


# ========== Ship Collision Utilities ==========
# Ship collision logic
@jax.jit
def check_ship_crash(state: ShipState, enemies: Enemies, hitbox_size: float) -> bool:
    sx1 = state.x - hitbox_size
    sx2 = state.x + hitbox_size
    sy1 = state.y - hitbox_size
    sy2 = state.y + hitbox_size

    ex1 = enemies.x - enemies.w / 2
    ex2 = enemies.x + enemies.w / 2
    ey1 = enemies.y - enemies.h / 2
    ey2 = enemies.y + enemies.h / 2

    overlap_x = (sx1 <= ex2) & (sx2 >= ex1)
    overlap_y = (sy1 <= ey2) & (sy2 >= ey1)

    return jnp.any(overlap_x & overlap_y)


@jax.jit
def check_ship_enemy_collisions(ship: ShipState, enemies: Enemies, ship_radius: float) -> jnp.ndarray:
    # Treat enemy coordinates as the center of a rectangle
    enemy_half_w = enemies.w / 2
    enemy_half_h = enemies.h / 2

    # Calculate the distance between the ship's center and each enemy's center
    delta_x = ship.x - enemies.x
    delta_y = ship.y - enemies.y

    # Clamp the distance to the enemy's rectangular bounds
    clamped_x = jnp.clip(delta_x, -enemy_half_w, enemy_half_w)
    clamped_y = jnp.clip(delta_y, -enemy_half_h, enemy_half_h)

    # Find the vector from the ship's center to the closest point on the rectangle
    closest_point_dx = delta_x - clamped_x
    closest_point_dy = delta_y - clamped_y

    # Calculate the squared distance
    distance_sq = closest_point_dx ** 2 + closest_point_dy ** 2

    # A collision occurs if the distance is less than the ship's radius
    # Also ensure the enemy is "alive" (width > 0)）
    collided_mask = (distance_sq < ship_radius ** 2) & (enemies.w > 0.0)

    return collided_mask


@jax.jit
def check_ship_hit(state: ShipState, bullets: Bullets, hitbox_size: float) -> bool:
    sx1 = state.x - hitbox_size
    sx2 = state.x + hitbox_size
    sy1 = state.y - hitbox_size
    sy2 = state.y + hitbox_size

    within_x = (bullets.x >= sx1) & (bullets.x <= sx2)
    within_y = (bullets.y >= sy1) & (bullets.y <= sy2)

    return jnp.any(within_x & within_y & bullets.alive)


@jax.jit
def check_enemy_hit(bullets: Bullets, enemies: Enemies) -> Tuple[Bullets, Enemies]:
    # 1. Perform all collision detection calculations first
    padding = 0.2
    ex1 = enemies.x - enemies.w / 2 - padding
    ex2 = enemies.x + enemies.w / 2 + padding
    ey1 = enemies.y - enemies.h / 2 - padding
    ey2 = enemies.y + enemies.h / 2

    bx = bullets.x[:, None]
    by = bullets.y[:, None]

    cond_x = (bx >= ex1) & (bx <= ex2)
    cond_y = (by >= ey1) & (by <= ey2)

    hit_matrix = cond_x & cond_y & bullets.alive[:, None] & (enemies.w > 0)[:, None].T

    bullet_hit = jnp.any(hit_matrix, axis=1)
    enemy_hit = jnp.any(hit_matrix, axis=0)

    # 2. Update bullet states
    new_bullets = Bullets(
        x=bullets.x,
        y=bullets.y,
        vx=bullets.vx,
        vy=bullets.vy,
        alive=bullets.alive & (~bullet_hit),
        sprite_idx=bullets.sprite_idx
    )

    # 3. Calculate all the new values for the enemies to be updated externally
    # a) Calculate the new HP after being hit
    hp_after_hit = enemies.hp - jnp.where(enemy_hit, 1, 0)

    # b) Determine which enemies have "just died"
    was_alive = (enemies.hp > 0)
    is_dead_now = (hp_after_hit <= 0)
    just_died = was_alive & is_dead_now

    # c) Calculate the updated death timer
    death_timer_after_hit = jnp.where(
        just_died,
        ENEMY_EXPLOSION_FRAMES,
        enemies.death_timer
    )

    # 4. Finally, create a new Enemies object with the pre-calculated new values in a single step
    new_enemies = Enemies(
        x=enemies.x,
        y=enemies.y,
        w=enemies.w,  # Width and height remain unchanged here
        h=enemies.h,
        vx=enemies.vx,
        sprite_idx=enemies.sprite_idx,
        death_timer=death_timer_after_hit,
        hp=hp_after_hit
    )

    return new_bullets, new_enemies


@jax.jit
def terrain_hit(env_state: EnvState, x: jnp.ndarray, y: jnp.ndarray, radius=jnp.float32(0.3)) -> jnp.ndarray:
    adjusted_x, adjusted_y = x, y
    H, W = env_state.terrain_bank.shape[1], env_state.terrain_bank.shape[2]

    xi = jnp.clip(jnp.round(adjusted_x).astype(jnp.int32), 0, W - 1)
    yi = jnp.clip(jnp.round(adjusted_y).astype(jnp.int32), 0, H - 1)

    xs = jnp.clip(xi + _TERRAIN_HIT_DX, 0, W - 1)
    ys = jnp.clip(yi + _TERRAIN_HIT_DY, 0, H - 1)

    bi = jnp.clip(env_state.terrain_bank_idx, 0, env_state.terrain_bank.shape[0] - 1)
    page = env_state.terrain_bank[bi]

    patch = page[ys[:, None], xs[None, :]]

    r_eff = jnp.minimum(jnp.float32(radius), jnp.float32(_TERRAIN_HIT_RMAX))
    mask = _TERRAIN_HIT_DIST2 <= (r_eff ** 2)
    bg_val = env_state.terrain_bank[0, 0, 0]
    is_not_black = patch != bg_val

    return jnp.any(is_not_black & mask)


@jax.jit
def consume_ship_hits(state, bullets, hitbox_size):
    # Ship's collision radius
    hs = jnp.asarray(hitbox_size, dtype=jnp.float32)
    eff_r = hs + jnp.float32(0.04)

    hit_mask = bullets.alive & _segment_hits_circle(
        bullets.x, bullets.y, bullets.vx, bullets.vy,
        state.x, state.y, eff_r
    )

    any_hit = jnp.any(hit_mask)

    new_bullets = Bullets(
        x=bullets.x, y=bullets.y,
        vx=bullets.vx, vy=bullets.vy,
        alive=bullets.alive & (~hit_mask),  # Eliminate hit bullets
        sprite_idx=bullets.sprite_idx
    )

    return new_bullets, any_hit


@jax.jit
def kill_bullets_hit_terrain_segment(prev: Bullets, nxt: Bullets, terrain_mask: jnp.ndarray,
                                     samples: int = 4) -> Bullets:
    H, W = terrain_mask.shape

    def body(i, acc_hit):
        t = jnp.float32(i) / jnp.float32(samples - 1)  # 0..1
        xs = prev.x + t * (nxt.x - prev.x)
        ys = prev.y + t * (nxt.y - prev.y)

        xi = jnp.clip(xs.astype(jnp.int32), 0, jnp.int32(W - 1))
        yi = jnp.clip(ys.astype(jnp.int32), 0, jnp.int32(H - 1))

        hit_i = terrain_mask[yi, xi] > 0

        return acc_hit | hit_i

    init = jnp.zeros_like(prev.alive, dtype=jnp.bool_)
    hits = jax.lax.fori_loop(0, samples, body, init)
    alive = nxt.alive & (~hits) & prev.alive  # Only active bullets are considered

    return Bullets(x=nxt.x, y=nxt.y, vx=nxt.vx, vy=nxt.vy, alive=alive, sprite_idx=nxt.sprite_idx)


# ========== Ship Step ==========
# Ship movement
@jax.jit
def ship_step(state: ShipState,
              action: int,
              window_size: tuple[int, int],
              hud_height: int,
              fuel: jnp.ndarray,
              terrain_bank_idx: jnp.ndarray = jnp.int32(0),
              allow_exit_top: jnp.ndarray = jnp.bool_(False),
              thrust_power: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.THRUST_POWER),
              max_speed: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.MAX_SPEED),
              solar_gravity: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.SOLAR_GRAVITY),
              planetary_gravity: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.PLANETARY_GRAVITY),
              reactor_gravity: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.REACTOR_GRAVITY)) -> ShipState:
    # --- Track thrusting state for rendering ---
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])  # UP, UPRIGHT, UPLEFT, UPFIRE, UPRIGHTFIRE, UPLEFTFIRE
    is_thrusting_now = jnp.isin(action, thrust_actions) & (fuel > 0.0)
    
    # --- Physics Parameters ---
    THRUST_POWER = jnp.asarray(thrust_power, dtype=jnp.float32) / WORLD_SCALE
    scaled_solar_gravity = solar_gravity / WORLD_SCALE
    scaled_planetary_gravity = planetary_gravity / WORLD_SCALE
    scaled_reactor_gravity = reactor_gravity / WORLD_SCALE
    MAX_SPEED = jnp.asarray(max_speed, dtype=jnp.float32) / WORLD_SCALE

    # 0.0 = full stop on collision (inelastic)
    # 1.0 = perfect bounce (elastic)
    bounce_damping = 0.75

    # --- 1. Initialize velocity variables for this frame ---
    #     we get the initial velocity from the state
    vx = state.vx
    vy = state.vy

    # --- 2. Rotation Logic (Discrete 12-angle system) ---
    rotate_right_actions = jnp.array([3, 6, 8, 11, 14, 16])
    rotate_left_actions = jnp.array([4, 7, 9, 12, 15, 17])
    right = jnp.isin(action, rotate_right_actions)
    left = jnp.isin(action, rotate_left_actions)

    # Find current angle index in the discrete angle array
    cost = (jnp.cos(state.angle) - _SHIP_ANGLES_COS) ** 2 + (jnp.sin(state.angle) - _SHIP_ANGLES_SIN) ** 2
    current_idx = jnp.argmin(cost)
    
    # Only allow rotation if cooldown has expired
    can_rotate = state.rotation_cooldown <= 0
    wants_to_rotate = right | left
    
    # Rotate to next/previous discrete angle (wrapping around) only if allowed
    next_idx = jnp.where(can_rotate & right, (current_idx + 1) % 16, current_idx)
    next_idx = jnp.where(can_rotate & left, (current_idx - 1) % 16, next_idx)
    
    # Update angle and cooldown
    angle = SHIP_ANGLES[next_idx]
    did_rotate = (next_idx != current_idx)
    new_rotation_cooldown = jnp.where(
        did_rotate,
        jnp.int32(ROTATION_COOLDOWN_FRAMES),
        jnp.maximum(jnp.int32(0), state.rotation_cooldown - 1)
    )

    # --- 3. Thrust Calculation ---
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])
    # DOWN-family actions are shield/tractor controls in Gravitar and should
    # not apply any directional thrust.
    down_thrust_actions = jnp.array([], dtype=jnp.int32)

    thrust_pressed = jnp.isin(action, thrust_actions)
    down_pressed = jnp.isin(action, down_thrust_actions)

    can_thrust = fuel > 0.0

    # Forward thrust (vector addition), controlled by the UP key
    vx = jnp.where(thrust_pressed & can_thrust, vx + jnp.cos(angle) * THRUST_POWER, vx)
    vy = jnp.where(thrust_pressed & can_thrust, vy + jnp.sin(angle) * THRUST_POWER, vy)

    # Shield/tractor actions intentionally do not affect ship velocity.
    vx = jnp.where(down_pressed & can_thrust, vx - jnp.cos(angle) * THRUST_POWER, vx)
    vy = jnp.where(down_pressed & can_thrust, vy - jnp.sin(angle) * THRUST_POWER, vy)

    # Apply gravity based on mode and terrain
    # Map mode (terrain_bank_idx == 0): pull toward sun
    # Terrain2 (bank_idx == 2) and Reactor (bank_idx == 5): pull toward center (radial gravity)
    # Other planets: pull downward
    is_map_mode = (terrain_bank_idx == 0)
    is_planet = (terrain_bank_idx == 1) | (terrain_bank_idx == 2) | (terrain_bank_idx == 3) | (terrain_bank_idx == 4)
    is_reactor = (terrain_bank_idx == 5)
    is_central_gravity = (terrain_bank_idx == 2) | (terrain_bank_idx == 5)

    gravity = jnp.where(is_map_mode, scaled_solar_gravity, jnp.where(is_planet, scaled_planetary_gravity, scaled_reactor_gravity))

    # Sun position (the OBSTACLE sprite) - ALE center coordinates: (82, 86)
    sun_x = 82.0
    sun_y = 86.0
    
    # Calculate direction to sun (for map mode)
    dx_to_sun = sun_x - state.x
    dy_to_sun = sun_y - state.y
    dist_to_sun = jnp.sqrt(dx_to_sun**2 + dy_to_sun**2)
    dist_to_sun = jnp.maximum(dist_to_sun, 1.0)  # Avoid division by zero
    
    # Gravity magnitude (stronger when closer to sun)
    gravity_strength = gravity * (3.2 / dist_to_sun)  # Inverse distance law
    gravity_strength = jnp.clip(gravity_strength, 0.0, gravity * 5.0)  # Cap maximum gravity
    
    # Level center for terrain2's radial gravity
    level_center_x = window_size[0] / 2.0 + 5.0  # Slightly right of center to match ALE's layout
    level_center_y = window_size[1] / 2.0
    
    # Calculate direction to level center (for terrain2)
    dx_to_center = level_center_x - state.x
    dy_to_center = level_center_y - state.y
    dist_to_center = jnp.sqrt(dx_to_center**2 + dy_to_center**2)
    dist_to_center = jnp.maximum(dist_to_center, 1.0)
    
    # Radial gravity strength for terrain2
    radial_gravity_strength = gravity * (50.0 / dist_to_center)
    radial_gravity_strength = jnp.clip(radial_gravity_strength, 0.0, gravity * 2.0)
    
    # Apply gravitational pull based on terrain type
    # Map mode: toward sun
    # Terrain2: toward center (radial)
    # Other planets: downward only
    vx = jnp.where(is_map_mode, 
                   vx + (dx_to_sun / dist_to_sun) * gravity_strength,
                   jnp.where(is_central_gravity,
                            vx + (dx_to_center / dist_to_center) * radial_gravity_strength,
                            vx))
    vy = jnp.where(is_map_mode,
                   vy + (dy_to_sun / dist_to_sun) * gravity_strength,
                   jnp.where(is_central_gravity,
                            vy + (dy_to_center / dist_to_center) * radial_gravity_strength,
                            vy + gravity))


    # --- 4. Apply maximum speed limit ---
    speed_sq = vx ** 2 + vy ** 2
    # Debug: print current speed magnitude each physics step
    #jax.debug.print("current speed: {x}", x=jnp.sqrt(speed_sq))

    def cap_velocity(v_tuple):
        v_x, v_y, spd_sq = v_tuple
        speed = jnp.sqrt(spd_sq)
        scale = MAX_SPEED / speed

        return v_x * scale, v_y * scale

    def no_op(v_tuple):
        return v_tuple[0], v_tuple[1]

    vx, vy = jax.lax.cond(
        speed_sq > MAX_SPEED ** 2,
        cap_velocity,
        no_op,
        (vx, vy, speed_sq)
    )

    # --- 5. Position and Boundary Collision ---
    window_width, window_height = window_size
    
    # Define boundaries to prevent sprite overflow
    # Left/right use wider margins (8.0) to prevent rendering overflow
    # Top/bottom use smaller margins (ship radius)
    HORIZONTAL_MARGIN = 8.0
    VERTICAL_MARGIN = SHIP_RADIUS
    min_x = HORIZONTAL_MARGIN
    max_x = window_width - HORIZONTAL_MARGIN
    min_y_base = hud_height + VERTICAL_MARGIN
    max_y = window_height - VERTICAL_MARGIN
    
    # Calculate next position
    next_x = state.x + vx
    next_y = state.y + vy
    
    # Check if next position would cross boundaries
    will_hit_left = next_x < min_x
    will_hit_right = next_x > max_x
    # Top boundary: only enforce when exit is NOT allowed
    will_hit_top = (next_y < min_y_base) & (~allow_exit_top)
    # Bottom boundary: always enforced
    will_hit_bottom = next_y > max_y
    
    # For bouncing: reverse velocity and apply damping when hitting boundary
    # This creates an "energy field" bounce effect
    bounced_vx = jnp.where(will_hit_left | will_hit_right, -vx * bounce_damping, vx)
    bounced_vy = jnp.where(will_hit_top | will_hit_bottom, -vy * bounce_damping, vy)
    
    # Calculate final position: if we would hit a boundary, clamp to the boundary
    # and use the bounced velocity for next frame
    clamped_x = jnp.clip(next_x, min_x, max_x)
    # For Y: when exit is allowed, don't clamp the top; otherwise enforce min_y_base
    min_y_effective = jnp.where(allow_exit_top, jnp.float32(-1000.0), min_y_base)
    clamped_y = jnp.clip(next_y, min_y_effective, max_y)
    
    # Use bounced velocities if we hit something, otherwise keep original velocities
    final_vx = bounced_vx
    final_vy = bounced_vy
    final_x = clamped_x
    final_y = clamped_y

    # e. Normalize the angle (remains unchanged)
    normalized_angle = (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # f. Return the new state with corrected position and velocity
    return ShipState(x=final_x, y=final_y, vx=final_vx, vy=final_vy, angle=normalized_angle, is_thrusting=is_thrusting_now, rotation_cooldown=new_rotation_cooldown)


# ========== Logic about saucer ==========
@jax.jit
def _get_reactor_center(px, py, pi) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    REACTOR = jnp.int32(int(SpriteIdx.REACTOR))
    mask = (pi == REACTOR)

    any_reactor = jnp.any(mask)
    idx = jnp.argmax(mask.astype(jnp.int32))

    rx = jax.lax.cond(any_reactor, lambda _: px[idx], lambda _: jnp.float32(WINDOW_WIDTH * 0.18), operand=None)
    ry = jax.lax.cond(any_reactor, lambda _: py[idx], lambda _: jnp.float32(WINDOW_HEIGHT * 0.43), operand=None)

    return rx, ry, any_reactor


@jax.jit
def _spawn_saucer_at(x, y, towards_x, towards_y, speed=jnp.float32(0.8)) -> SaucerState:
    dx = towards_x - x
    dy = towards_y - y

    d = jnp.maximum(jnp.sqrt(dx * dx + dy * dy), 1e-3)

    vx = speed * dx / d
    vy = speed * dy / d

    return SaucerState(
        x=jnp.float32(x), y=jnp.float32(y),
        vx=vx, vy=vy,
        hp=jnp.int32(SAUCER_INIT_HP),
        alive=jnp.array(True),
        death_timer=jnp.int32(0),
    )


@jax.jit
def _update_saucer_seek(s: SaucerState, target_x, target_y, speed) -> SaucerState:
    dx = target_x - s.x
    dy = target_y - s.y
    d = jnp.maximum(jnp.sqrt(dx * dx + dy * dy), 1e-3)

    vx = speed * dx / d
    vy = speed * dy / d

    return s._replace(x=s.x + vx, y=s.y + vy, vx=vx, vy=vy)


@jax.jit
def _update_saucer_horizontal(s: SaucerState, target_x, reactor_y, speed) -> SaucerState:
    """Update saucer to move only horizontally at reactor height, following the ship's x position"""
    # Move horizontally towards target_x
    dx = target_x - s.x
    # Determine direction: move left or right
    vx = jnp.where(dx > 0, speed, -speed)
    vx = jnp.where(jnp.abs(dx) < speed, jnp.float32(0.0), vx)  # Stop if close enough
    
    # Keep y fixed at reactor height
    return s._replace(x=s.x + vx, y=jnp.float32(reactor_y), vx=vx, vy=jnp.float32(0.0))


@jax.jit
def _saucer_fire_one(sauc: SaucerState,
                     ship_x: jnp.ndarray,
                     ship_y: jnp.ndarray,
                     prev_enemy_bullets: Bullets,
                     mode_timer: jnp.ndarray,
               max_active_bullets: jnp.ndarray = jnp.int32(1),
                     ) -> Bullets:
    can_fire = sauc.alive & ((mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0) \
           & (_bullets_alive_count(prev_enemy_bullets) < max_active_bullets)

    def do_fire(_):
        merged = _fire_single_from_to(
            prev_enemy_bullets,
            sauc.x, sauc.y,
            ship_x, ship_y,
            SAUCER_BULLET_SPEED
        )

        return _enforce_cap_keep_old(merged, cap=max_active_bullets)

    return jax.lax.cond(can_fire, do_fire, lambda _: prev_enemy_bullets, operand=None)


@jax.jit
def _saucer_fire_random(sauc: SaucerState,
                        prev_enemy_bullets: Bullets,
                        mode_timer: jnp.ndarray,
                        key: jnp.ndarray,
                        max_active_bullets: jnp.ndarray = jnp.int32(2),
                        ) -> Bullets:
    """Saucer fires in random directions with max 2 bullets"""
    can_fire = sauc.alive & ((mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0) \
               & (_bullets_alive_count(prev_enemy_bullets) < max_active_bullets)

    def do_fire(_):
        # Generate random angle (0 to 2*pi)
        angle = jax.random.uniform(key, minval=0.0, maxval=2.0 * jnp.pi)
        
        # Calculate velocity components from random angle
        vx = SAUCER_BULLET_SPEED * jnp.cos(angle)
        vy = SAUCER_BULLET_SPEED * jnp.sin(angle)
        
        one = Bullets(
            x=jnp.array([sauc.x], dtype=jnp.float32),
            y=jnp.array([sauc.y], dtype=jnp.float32),
            vx=jnp.array([vx], dtype=jnp.float32),
            vy=jnp.array([vy], dtype=jnp.float32),
            alive=jnp.array([True]),
            sprite_idx=jnp.array([int(SpriteIdx.ENEMY_BULLET)], dtype=jnp.int32)
        )
        
        merged = merge_bullets(prev_enemy_bullets, one, max_len=16)
        return _enforce_cap_keep_old(merged, cap=max_active_bullets)

    return jax.lax.cond(can_fire, do_fire, lambda _: prev_enemy_bullets, operand=None)


@jax.jit
def _circle_hit(ax, ay, ar, bx, by, br) -> jnp.ndarray:
    dx = ax - bx
    dy = ay - by

    return (dx * dx + dy * dy) <= (ar + br) * (ar + br)


@jax.jit
def _segment_hits_circle(bx, by, vx, vy, cx, cy, r):
    # Bullet's previous position p0 = p1 - v
    px0 = bx - vx
    py0 = by - vy
    dx = vx
    dy = vy
    # Find the point on the line segment with parameter t* ∈ [0,1] that is closest to the circle's center
    a = dx * dx + dy * dy + 1e-6
    t = jnp.clip(-(((px0 - cx) * dx + (py0 - cy) * dy) / a), 0.0, 1.0)

    qx = px0 + t * dx
    qy = py0 + t * dy
    d2 = (qx - cx) * (qx - cx) + (qy - cy) * (qy - cy)

    return d2 <= (r * r)


@jax.jit
def _bullets_hit_saucer(bullets: Bullets, sauc: SaucerState):
    eff_r = SAUCER_RADIUS

    hit_mask = bullets.alive & _segment_hits_circle(
        bullets.x, bullets.y, bullets.vx, bullets.vy,
        sauc.x, sauc.y, eff_r
    )
    any_hit = jnp.any(hit_mask)

    new_bullets = Bullets(
        x=bullets.x, y=bullets.y,
        vx=bullets.vx, vy=bullets.vy,
        alive=bullets.alive & (~hit_mask),  # Eliminate hit bullets
        sprite_idx=bullets.sprite_idx
    )

    return new_bullets, any_hit


@jax.jit
def _bullets_hit_ufo(bullets: Bullets, ufo) -> Tuple[Bullets, jnp.ndarray]:
    eff_r = SAUCER_RADIUS

    hit_mask = bullets.alive & _segment_hits_circle(
        bullets.x, bullets.y, bullets.vx, bullets.vy,
        ufo.x, ufo.y, eff_r
    )

    any_hit = jnp.any(hit_mask)

    new_bullets = Bullets(
        x=bullets.x, y=bullets.y,
        vx=bullets.vx, vy=bullets.vy,
        alive=bullets.alive & (~hit_mask),
        sprite_idx=bullets.sprite_idx
    )

    return new_bullets, any_hit


# ========== Enemy Step ==========
# Enemy Movement
@jax.jit
def enemy_step(enemies: Enemies, window_width: int) -> Enemies:
    x = enemies.x + enemies.vx
    left_hit = x <= 0
    right_hit = (x + enemies.w) >= window_width

    hit_edge = left_hit | right_hit
    vx = jnp.where(hit_edge, -enemies.vx, enemies.vx)

    return Enemies(x=x, y=enemies.y, w=enemies.w, h=enemies.h, vx=vx, sprite_idx=enemies.sprite_idx,
                   death_timer=enemies.death_timer, hp=enemies.hp)


# ========== Enemy Fire ==========
@jax.jit
def enemy_fire(enemies: Enemies,
               ship_x: float,
               ship_y: float,
               enemy_bullet_speed: float,
               fire_cooldown: jnp.ndarray,  # shape should match len(enemies.x)
               fire_interval: int,
               key: jax.random.PRNGKey
               ) -> tuple[Bullets, jnp.ndarray, jax.random.PRNGKey]:
    ex_center = enemies.x + enemies.w / 2
    ey_center = enemies.y - enemies.h / 2

    dx = ship_x - ex_center  # shape=(N,)
    dy = ship_y - ey_center  # shape=(N,)

    dist = jnp.sqrt(dx ** 2 + dy ** 2)
    dist = jnp.where(dist < 1e-3, 1.0, dist)

    vx = dx / dist * enemy_bullet_speed
    vy = dy / dist * enemy_bullet_speed

    # 1. Determine which turrets "should" fire in this frame
    alive_mask = (enemies.w > 0.0) & (enemies.death_timer == 0)
    should_fire = (fire_cooldown == 0) & alive_mask

    # 2. Calculate the cooldown for the "next frame"
    # - If a turret fired this frame (`should_fire` is True), its cooldown is reset to `fire_interval`
    # - Otherwise, the cooldown remains unchanged (since the decrement happens in `_step_level_core`)
    new_fire_cooldown = jnp.where(should_fire, fire_interval, fire_cooldown)
    x_out = jnp.where(should_fire, ex_center, -1.0)
    y_out = jnp.where(should_fire, ey_center, -1.0)

    vx_out = jnp.where(should_fire, vx, 0.0)
    vy_out = jnp.where(should_fire, vy, 0.0)

    bullets_out = Bullets(
        x=x_out,
        y=y_out,
        vx=vx_out,
        vy=vy_out,
        alive=should_fire
    )

    return bullets_out, new_fire_cooldown, key


# ========== Step Core Map ==========
@jax.jit
def step_core_map(state: ShipState,
                  action: int,
                  window_size: Tuple[int, int],
                  hud_height: int,
                  fuel: jnp.ndarray = jnp.float32(0.0),
                  terrain_bank_idx: jnp.ndarray = jnp.int32(0),
                  thrust_power: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.THRUST_POWER),
                  max_speed: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.MAX_SPEED),
                  solar_gravity: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.SOLAR_GRAVITY),
                  planetary_gravity: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.PLANETARY_GRAVITY),
                  reactor_gravity: jnp.ndarray = jnp.float32(_DEFAULT_CONSTS.REACTOR_GRAVITY)
                  ) -> Tuple[GravitarObservation, ShipState, float, bool, GravitarInfo, bool, int]:
    new_state = ship_step(state, action, window_size, hud_height, fuel=fuel, terrain_bank_idx=terrain_bank_idx,
                          thrust_power=thrust_power,
                          max_speed=max_speed,
                          solar_gravity=solar_gravity,
                          planetary_gravity=planetary_gravity,
                          reactor_gravity=reactor_gravity)

    obs = _get_observation_from_ship_state(new_state)

    reward = 0.0
    done = jnp.array(False)
    info = GravitarInfo(
        lives=jnp.int32(0),
        score=jnp.float32(0.0),
        fuel=fuel,
        mode=jnp.int32(0),
        crash_timer=jnp.int32(0),
        done=done,
        current_level=jnp.int32(-1),
    )
    planet_x = jnp.array([60.0, 120.0, 200.0])
    planet_y = jnp.array([120.0, 200.0, 80.0])
    planet_r = jnp.array([3, 3, 3])
    level_ids = jnp.array([0, 1, 2])
    dx = planet_x - new_state.x
    dy = planet_y - new_state.y
    dists = jnp.sqrt(dx ** 2 + dy ** 2)
    within_planet = dists < planet_r
    reset = jnp.any(within_planet)
    level_idx = jnp.argmax(within_planet)
    level = jnp.where(reset, level_ids[level_idx], -1)

    return obs, new_state, reward, done, info, reset, level


# ========== Step Core Level Skeleton ==========
@jax.jit
def terrain_hit_mask(mask: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, radius: float = 2) -> jnp.ndarray:
    H, W = mask.shape
    r = jnp.int32(jnp.clip(radius, 1.0, float(_TERRAIN_MASK_R_MAX)))

    valid = (_TERRAIN_MASK_DIST2 <= (r * r))

    mx = jnp.floor(x).astype(jnp.int32)
    my = jnp.floor(y).astype(jnp.int32)

    sx = jnp.clip(mx + _TERRAIN_MASK_DX, 0, W - 1)
    sy = jnp.clip(my + _TERRAIN_MASK_DY, 0, H - 1)

    samples = mask[sy, sx]
    bg_val = mask[0, 0]
    samples = jnp.where(valid, samples, bg_val)

    return jnp.any(samples != bg_val)


@jax.jit
def step_map(env_state: EnvState, action: int):
    # --- 1. State Preparation ---
    # Check if the ship was crashing in the previous frame
    was_crashing = env_state.crash_timer > 0

    # --- 2. Ship Movement and Player Firing ---
    # If the ship is crashing, ignore player input and set velocity to zero to "freeze" the ship
    ship_state_before_move = env_state.state._replace(
        vx=jnp.where(was_crashing, 0.0, env_state.state.vx),
        vy=jnp.where(was_crashing, 0.0, env_state.state.vy)
    )

    actual_action = jnp.where(was_crashing, NOOP, action)
    ship_after_move = ship_step(ship_state_before_move, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT, env_state.fuel, env_state.terrain_bank_idx,
                                thrust_power=env_state.thrust_power,
                                max_speed=env_state.max_speed,
                                solar_gravity=env_state.solar_gravity,
                                planetary_gravity=env_state.planetary_gravity,
                                reactor_gravity=env_state.reactor_gravity)
    
    # Calculate fuel consumption in map mode
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])
    shield_tractor_actions = jnp.array([5, 8, 9, 13, 16, 17])
    is_thrusting = jnp.isin(actual_action, thrust_actions)
    is_using_shield_tractor = jnp.isin(actual_action, shield_tractor_actions)
    
    fuel_consumed = jnp.where(is_thrusting, env_state.fuel_consume_thrust, 0.0)
    fuel_consumed += jnp.where(is_using_shield_tractor, env_state.fuel_consume_shield_tractor, 0.0)
    new_fuel = jnp.maximum(0.0, env_state.fuel - fuel_consumed)
    
    # Detect fire button press (not hold) - only fire on transition from not-pressed to pressed
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_fire_pressed = jnp.isin(action, fire_actions)
    was_fire_pressed = jnp.isin(env_state.prev_action, fire_actions)
    fire_just_pressed = is_fire_pressed & (~was_fire_pressed)
    
    can_fire = fire_just_pressed & (env_state.cooldown == 0) & (
                _bullets_alive_count(env_state.bullets) < env_state.max_active_player_bullets_map)

    bullets = jax.lax.cond(
        can_fire,
        lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, PLAYER_BULLET_SPEED),
        lambda b: b,
        env_state.bullets
    )

    bullets = update_bullets(bullets)
    cooldown = jnp.where(can_fire, PLAYER_FIRE_COOLDOWN_FRAMES, jnp.maximum(env_state.cooldown - 1, 0))

    # Initialize a temporary `env_state` for subsequent chained updates
    new_env = env_state._replace(state=ship_after_move, bullets=bullets, cooldown=cooldown, fuel=new_fuel, prev_action=action)

    # --- 3. Saucer Logic ---
    saucer = new_env.saucer
    timer = new_env.saucer_spawn_timer

    should_tick_timer = (new_env.mode == 0) & (~saucer.alive) & (saucer.death_timer == 0)
    timer = jnp.where(should_tick_timer, jnp.maximum(timer - 1, 0), timer)

    rx, ry, has_reactor = _get_reactor_center(new_env.planets_px, new_env.planets_py, new_env.planets_pi)
    should_spawn = (timer == 0) & (~saucer.alive) & has_reactor

    saucer = jax.lax.cond(should_spawn,
                          lambda: _spawn_saucer_at(rx, ry, new_env.state.x, new_env.state.y, SAUCER_SPEED_MAP),
                          lambda: saucer)
    timer = jnp.where(should_spawn, 99999, timer)

    # Saucer moves horizontally at reactor height, following ship's x position
    saucer_after_move = jax.lax.cond(saucer.alive, 
                                     lambda s: _update_saucer_horizontal(s, new_env.state.x, ry, SAUCER_SPEED_MAP), 
                                     lambda s: s,
                                     operand=saucer)
    bullets_after_hit, hit_any_bullet = _bullets_hit_saucer(new_env.bullets, saucer_after_move)

    sauc_after_hp = saucer_after_move._replace(hp=saucer_after_move.hp - jnp.where(hit_any_bullet, 1, 0))
    just_died = (saucer_after_move.hp > 0) & (sauc_after_hp.hp <= 0) & saucer_after_move.alive

    timer = jnp.where(just_died, SAUCER_RESPAWN_DELAY_FRAMES, timer)

    sauc_final = sauc_after_hp._replace(
        alive=sauc_after_hp.hp > 0,
        death_timer=jnp.where(just_died, SAUCER_EXPLOSION_FRAMES, jnp.maximum(saucer_after_move.death_timer - 1, 0))
    )

    mode_timer = jnp.where(new_env.mode == 0, new_env.mode_timer + 1, 0)

    # Saucer fires in random directions (360 degrees) with max 2 bullets
    fire_key, new_main_key = jax.random.split(new_env.key)
    enemy_bullets = _saucer_fire_random(
        sauc_final,
        new_env.enemy_bullets,
        mode_timer,
        fire_key,
        new_env.max_active_saucer_bullets,
    )
    enemy_bullets = update_bullets(enemy_bullets)

    new_env = new_env._replace(
        bullets=bullets_after_hit, saucer=sauc_final, saucer_spawn_timer=timer,
        enemy_bullets=enemy_bullets, mode_timer=mode_timer, key=new_main_key
    )

    # --- 4. Collision and State Finalization ---
    # a) Saucer bullet hits ship
    enemy_bullets_after_hit, hit_ship_by_bullet = consume_ship_hits(new_env.state, new_env.enemy_bullets, SHIP_RADIUS)
    new_env = new_env._replace(enemy_bullets=enemy_bullets_after_hit)

    # b) Ship collides with an obstacle
    px, py, pr, pi, pid = new_env.planets_px, new_env.planets_py, new_env.planets_pr, new_env.planets_pi, new_env.planets_id
    dx, dy = px - new_env.state.x, py - new_env.state.y

    dist2 = dx * dx + dy * dy
    hit_obstacle = jnp.any((pi == SpriteIdx.OBSTACLE) & (dist2 <= (pr + SHIP_RADIUS) ** 2))

    # c) Unify the crash conditions
    ship_should_crash = hit_ship_by_bullet | hit_obstacle

    # d) Unify the crash timer logic
    start_crash = ship_should_crash & (~was_crashing)
    crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(new_env.crash_timer - 1, 0))

    new_env = new_env._replace(crash_timer=crash_timer_next)
    is_crashing_now = new_env.crash_timer > 0

    # e) Disable other collisions during a crash ("ghost" state)
    allowed = jnp.any(jnp.stack(
        [pi == SpriteIdx.PLANET1, pi == SpriteIdx.PLANET2, pi == SpriteIdx.PLANET3, pi == SpriteIdx.PLANET4,
         pi == SpriteIdx.REACTOR], 0), axis=0)
    allowed = allowed & (~new_env.planets_cleared_mask)

    is_reactor_and_destroyed = (pi == int(SpriteIdx.REACTOR)) & new_env.reactor_destroyed
    allowed = allowed & (~is_reactor_and_destroyed)

    hit_planet = allowed & (dist2 <= (pr * 0.85 + SHIP_RADIUS) ** 2)
    can_enter_planet = jnp.any(hit_planet) & (~is_crashing_now)
    
    # Save current position when entering a planet level (will be used on level completion)
    def _save_map_position(env):
        return env._replace(
            map_return_x=env.state.x,
            map_return_y=env.state.y,
            map_return_vx=env.state.vx,
            map_return_vy=env.state.vy,
            map_return_angle=env.state.angle,
        )
    
    new_env = jax.lax.cond(can_enter_planet, _save_map_position, lambda e: e, new_env)

    # Check if ship is close enough to saucer to trigger arena battle
    # Use a larger trigger radius (3x the collision radius) to match "fly near" behavior
    ARENA_TRIGGER_RADIUS = SAUCER_RADIUS * 3.0
    hit_to_arena = sauc_final.alive & _circle_hit(new_env.state.x, new_env.state.y, SHIP_RADIUS, sauc_final.x,
                                                  sauc_final.y, ARENA_TRIGGER_RADIUS) & (~is_crashing_now)

    def _enter_arena(env):
        W, H = jnp.float32(WINDOW_WIDTH), jnp.float32(WINDOW_HEIGHT)
        # Save current position BEFORE modifying state
        return_x, return_y = env.state.x, env.state.y
        ship_approached_from_above = env.state.y < sauc_final.y

        ship_spawn_y = jnp.where(ship_approached_from_above, H * 0.20, H * 0.80)
        saucer_spawn_y = jnp.where(ship_approached_from_above, H * 0.80, H * 0.20)
        return env._replace(
            mode=jnp.int32(2), mode_timer=jnp.int32(0),
            state=env.state._replace(x=W * 0.80, y=ship_spawn_y, vx=env.state.vx, vy=env.state.vy),
            saucer=sauc_final._replace(
                x=W * 0.20,
                y=saucer_spawn_y,
                vx=jnp.float32(SAUCER_SPEED_ARENA),
                vy=jnp.float32(0.0),
                hp=jnp.int32(SAUCER_INIT_HP),
                alive=jnp.array(True),
                death_timer=jnp.int32(0),
            ),
            # Clear all bullets when entering arena to prevent rogue bullets
            bullets=create_empty_bullets_64(),
            enemy_bullets=create_empty_bullets_16(),
            # Save the position for restoration after arena
            map_return_x=return_x,
            map_return_y=return_y,
            map_return_vx=env.state.vx,
            map_return_vy=env.state.vy,
            map_return_angle=env.state.angle,
        )

    new_env = jax.lax.cond(hit_to_arena, _enter_arena, lambda e: e, new_env)

    # f) Only signal a Reset when the animation is finished
    reset_signal_from_crash = (env_state.crash_timer > 0) & (crash_timer_next == 0)
    # --- 5. Final Return Values ---
    hit_idx = jnp.argmax(hit_planet.astype(jnp.int32))
    level_id = jax.lax.cond(can_enter_planet, lambda: pid[hit_idx], lambda: -1)
    should_reset = can_enter_planet | reset_signal_from_crash
    final_level_id = jnp.where(reset_signal_from_crash, -2, level_id)

    obs = _get_observation_from_state(new_env)

    reward_saucer = jnp.where(just_died, new_env.saucer_kill_score, jnp.float32(0.0))
    reward = reward_saucer
    done = jnp.array(False)
    info = GravitarInfo(
        lives=new_env.lives,
        score=new_env.score,
        fuel=new_env.fuel,
        mode=new_env.mode,
        crash_timer=new_env.crash_timer,
        done=done,
        current_level=new_env.current_level,
        crash=start_crash,
        hit_by_bullet=hit_ship_by_bullet,
        reactor_crash_exit=jnp.array(False),
        all_rewards=jnp.array([
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            reward_saucer,
            jnp.float32(0.0),
        ], dtype=jnp.float32),
    )

    new_env = new_env._replace(score=new_env.score + reward, shield_active=is_using_shield_tractor)

    return obs, new_env, reward, done, info, should_reset, final_level_id


@jax.jit
def _step_level_core(env_state: EnvState, action: int):
    # --- 1. UFO Spawn ---
    def _spawn_ufo_once(env):
        W, H = WINDOW_WIDTH, WINDOW_HEIGHT
        b = env.terrain_bank_idx

        # 1. Determine UFO's spawn X coordinate and initial velocity
        # UFO should spawn OFF-SCREEN (outside visible area) and fly in
        is_born_on_left = (b == 2) | (b == 4)
        x0 = jnp.where(is_born_on_left, -30.0, W + 30.0)  # Spawn outside visible area
        vx = jnp.where(is_born_on_left, 0.6 / WORLD_SCALE, -0.6 / WORLD_SCALE)

        # 2. Check for the safe altitude across the entire future path
        bank_idx = jnp.clip(env.terrain_bank_idx, 0, env.terrain_bank.shape[0] - 1)
        terrain_page = env.terrain_bank[bank_idx]

        # We no longer slice, but compute over the entire terrain map
        is_ground = jnp.sum(terrain_page, axis=-1) > 0
        y_indices = jnp.arange(H, dtype=jnp.int32)[:, None]  # Convert to column vector for broadcasting

        # Find the y-coordinates of all ground points in the terrain
        ground_indices = jnp.where(is_ground, y_indices, H)

        # Among all these points, find the highest one (with the smallest y-value)
        highest_point_on_map = jnp.min(ground_indices)

        # 3. Calculate the final spawn Y coordinate (logic unchanged)
        safe_y = jnp.float32(highest_point_on_map) - 10.0
        final_y0 = jnp.clip(safe_y, HUD_HEIGHT + 20.0, H - 20.0)

        # 4. Return the updated environment state with spawn timer set to respawn delay
        return env._replace(
            ufo=UFOState(
                x=jnp.float32(x0),
                y=jnp.float32(final_y0),
                vx=jnp.float32(vx),
                vy=jnp.float32(0.0),
                hp=jnp.int32(1),
                alive=jnp.array(True),
                death_timer=jnp.int32(0),
            ),
            ufo_spawn_timer=UFO_RESPAWN_DELAY_FRAMES, ufo_home_x=x0, ufo_home_y=final_y0,
            ufo_bullets=create_empty_bullets_16(),
        )

    # UFO spawns in planet levels when timer is 0 and ship has descended low enough, but not in reactor (5) and Planet 2 (2)
    ship_low_enough_for_ufo = env_state.state.y >= jnp.float32(UFO_SPAWN_Y_THRESHOLD)
    can_spawn_ufo = (
        (env_state.mode == 1)
        & (env_state.ufo_spawn_timer == 0)
        & (~env_state.ufo.alive)
        & (env_state.terrain_bank_idx != 5)
        & (env_state.terrain_bank_idx != 2)
        & ship_low_enough_for_ufo
    )
    state_after_spawn = jax.lax.cond(can_spawn_ufo, _spawn_ufo_once, lambda e: e, env_state)

    is_in_reactor = (env_state.current_level == 4)

    timer_after_tick = env_state.reactor_timer - 1
    new_reactor_timer = jnp.where(is_in_reactor, timer_after_tick, env_state.reactor_timer)

    timer_ran_out = is_in_reactor & (new_reactor_timer <= 0)
    
    # Use exit_allowed from previous frame to determine if ship can exit through top
    # (This was set based on the previous frame's enemy status)
    allow_exit_top = state_after_spawn.exit_allowed
    
    # --- 2. State Update (Movement & Player Firing) ---
    was_crashing = state_after_spawn.crash_timer > 0
    ship_state_before_move = state_after_spawn.state._replace(
        vx=jnp.where(was_crashing, 0.0, state_after_spawn.state.vx),
        vy=jnp.where(was_crashing, 0.0, state_after_spawn.state.vy)
    )
    actual_action = jnp.where(was_crashing, NOOP, action)

    ship_after_move = ship_step(ship_state_before_move, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT, state_after_spawn.fuel, state_after_spawn.terrain_bank_idx, allow_exit_top,
                                thrust_power=state_after_spawn.thrust_power,
                                max_speed=state_after_spawn.max_speed,
                                solar_gravity=state_after_spawn.solar_gravity,
                                planetary_gravity=state_after_spawn.planetary_gravity,
                                reactor_gravity=state_after_spawn.reactor_gravity)
    
    # Detect fire button press (not hold) - only fire on transition from not-pressed to pressed
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_fire_pressed = jnp.isin(action, fire_actions)
    was_fire_pressed = jnp.isin(state_after_spawn.prev_action, fire_actions)
    fire_just_pressed = is_fire_pressed & (~was_fire_pressed)
    
    can_fire_player = fire_just_pressed & (
                state_after_spawn.cooldown == 0) & (_bullets_alive_count(state_after_spawn.bullets) < state_after_spawn.max_active_player_bullets_level)

    bullets = jax.lax.cond(
        can_fire_player,
        lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, PLAYER_BULLET_SPEED),
        lambda b: b,
        state_after_spawn.bullets
    )

    cooldown = jnp.where(can_fire_player, PLAYER_FIRE_COOLDOWN_FRAMES, jnp.maximum(state_after_spawn.cooldown - 1, 0))

    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])
    shield_tractor_actions = jnp.array([5, 8, 9, 13, 16, 17])
    
    is_thrusting = jnp.isin(actual_action, thrust_actions)
    is_using_shield_tractor = jnp.isin(actual_action, shield_tractor_actions)

    fuel_consumed = jnp.where(is_thrusting, state_after_spawn.fuel_consume_thrust, 0.0)
    fuel_consumed += jnp.where(is_using_shield_tractor, state_after_spawn.fuel_consume_shield_tractor, 0.0)

    ship = ship_after_move
    tanks = state_after_spawn.fuel_tanks
    ship_left = ship.x - SHIP_RADIUS
    ship_right = ship.x + SHIP_RADIUS
    ship_top = ship.y - SHIP_RADIUS
    ship_bottom = ship.y + SHIP_RADIUS

    tank_half_w = tanks.w / 2
    tank_half_h = tanks.h / 2
    tank_left = tanks.x - tank_half_w
    tank_right = tanks.x + tank_half_w
    tank_top = tanks.y - tank_half_h
    tank_bottom = tanks.y + tank_half_h

    # Direct collision with fuel tank
    overlap_x = (ship_right > tank_left) & (ship_left < tank_right)
    overlap_y = (ship_bottom > tank_top) & (ship_top < tank_bottom)
    direct_collision = tanks.active & overlap_x & overlap_y
    
    # Tractor beam pickup: when shield/tractor is active in planet levels (not reactor)
    is_planet_level = state_after_spawn.mode == 1
    is_reactor = state_after_spawn.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)
    can_use_tractor = is_planet_level & ((~is_reactor) | state_after_spawn.allow_tractor_in_reactor)
    
    # Calculate distance from ship to each tank
    dx = tanks.x - ship.x
    dy = tanks.y - ship.y
    distance_sq = dx * dx + dy * dy
    in_tractor_range = distance_sq <= (TRACTOR_BEAM_RANGE ** 2)
    
    # While crashing, freeze interactions so revealed hidden tanks from green enemies
    # are not auto-collected during crash animation frames.
    can_collect_tanks = ~was_crashing

    # Pickup happens on direct collision OR when using tractor beam and in range (planet levels only)
    tractor_pickup = can_collect_tanks & can_use_tractor & is_using_shield_tractor & in_tractor_range & tanks.active
    collision_mask = can_collect_tanks & (direct_collision | tractor_pickup)
    
    new_tanks_active = tanks.active & ~collision_mask
    new_fuel_tanks = tanks._replace(active=new_tanks_active)

    num_tanks_collected = jnp.sum(collision_mask)
    fuel_gained = num_tanks_collected * 5000.0
    
    fuel_after_actions = state_after_spawn.fuel - fuel_consumed + fuel_gained

    # --- 3. Integrate UFO Logic ---
    # Call the new helper function that returns an `env_state` with partially updated UFO-related states
    state_after_ufo = _update_ufo(state_after_spawn, ship_after_move, bullets)
    # Extract the updated state from the return value
    ufo = state_after_ufo.ufo
    bullets = state_after_ufo.bullets
    # Note: ufo_bullets always empty since UFOs don't shoot

    # --- 4. Ground Enemy (Turret) Logic ---
    enemies = enemy_step(state_after_ufo.enemies, WINDOW_WIDTH)
    is_exploding = enemies.death_timer > 0

    enemies = enemies._replace(
        death_timer=jnp.maximum(enemies.death_timer - 1, 0),
        w=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.w),
        h=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.h)
    )

    # === Enemy LOGIC ===
    # 1. Prepare the state for the current frame
    current_fire_cooldown = state_after_ufo.fire_cooldown
    current_key = state_after_ufo.key
    current_enemy_bullets = state_after_ufo.enemy_bullets

    # 2. Decide which turrets "can" fire now
    # Allow multiple turrets to fire (remove global limit)
    is_turret = (enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE)) | \
                (enemies.sprite_idx == int(SpriteIdx.ENEMY_GREEN)) | \
                (enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE_FLIPPED))

    # Turrets ready to fire: active, cooldown expired, not exploding, and is a turret
    turrets_ready_mask = (enemies.w > 0) & (current_fire_cooldown == 0) & (enemies.death_timer == 0) & is_turret

    current_enemy_bullets_alive = _bullets_alive_count(current_enemy_bullets)
    space_left_enemy_bullets = jnp.maximum(state_after_ufo.max_active_enemy_bullets - current_enemy_bullets_alive, 0)
    
    # Randomly select which ready turrets get to fire if they exceed space_left
    current_key, rank_key = jax.random.split(current_key)
    rand_vals = jax.random.uniform(rank_key, shape=turrets_ready_mask.shape)
    # Assign -1 to non-ready turrets so they rank lowest
    scores = jnp.where(turrets_ready_mask, rand_vals, -1.0)
    # Rank: 0 is highest score. Add 1 for each strictly greater score
    ranks = jnp.sum(scores[:, None] > scores[None, :], axis=0)

    should_fire_mask = turrets_ready_mask & (ranks < space_left_enemy_bullets)
    any_turret_firing = jnp.any(should_fire_mask)

    # 3. Calculate the cooldown for the "next frame"
    # First, decrement the cooldown for all turrets
    next_frame_cooldown = jnp.maximum(current_fire_cooldown - 1, 0)
    # Then, for turrets that "just" fired, reset their cooldown to random interval (60-120 frames)
    # Use deterministic approach: vary based on position
    base_interval = state_after_ufo.enemy_fire_cooldown_frames
    variance_max = jnp.maximum(jnp.int32(1), base_interval)
    varied_interval = base_interval + jnp.int32((enemies.x * 0.5) % variance_max)  # Varies by position
    next_frame_cooldown = jnp.where(should_fire_mask, varied_interval, next_frame_cooldown)

    # 4. If any turrets are firing, generate new bullets
    def _generate_bullets(_):
        # Spawn bullets at the CENTER of the enemy sprite (not offset)
        ex_center = enemies.x
        ey_center = enemies.y

        # Generate random angles for bullets in a slightly wider arc than 180 degrees
        # Normal bunkers: shoot upward (away from ground below)
        # Flipped bunkers: shoot downward (away from ground above)
        # Use position + frame counter to create pseudo-random angles
        angle_seed = (enemies.x + enemies.y + jnp.float32(state_after_ufo.mode_timer)) * 0.1
        fire_arc = jnp.pi + jnp.deg2rad(20.0)  # 200° total arc (was 180°)
        half_arc = fire_arc * 0.5
        random_offset = (angle_seed % 1.0) * fire_arc
        
        # Check if enemy is flipped
        is_flipped = (enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE_FLIPPED))
        
        # Normal: centered upward (-π/2), widened to 200°
        # Flipped: centered downward (+π/2), widened to 200°
        normal_start = -jnp.pi * 0.5 - half_arc
        flipped_start = jnp.pi * 0.5 - half_arc
        random_angle = jnp.where(is_flipped, flipped_start + random_offset, normal_start + random_offset)

        # Bullet speed
        vx = jnp.cos(random_angle) * ENEMY_BULLET_SPEED
        vy = jnp.sin(random_angle) * ENEMY_BULLET_SPEED

        x_out = jnp.where(should_fire_mask, ex_center, -1.0)
        y_out = jnp.where(should_fire_mask, ey_center, -1.0)

        vx_out = jnp.where(should_fire_mask, vx, 0.0)
        vy_out = jnp.where(should_fire_mask, vy, 0.0)
        
        # Determine bullet sprite: green enemies use green bullets, orange/flipped use orange bullets
        is_green_enemy = (enemies.sprite_idx == int(SpriteIdx.ENEMY_GREEN))
        bullet_sprite = jnp.where(is_green_enemy, int(SpriteIdx.ENEMY_GREEN_BULLET), int(SpriteIdx.ENEMY_BULLET))

        return Bullets(x=x_out, y=y_out, vx=vx_out, vy=vy_out, alive=should_fire_mask, sprite_idx=bullet_sprite)

    def _get_empty_bullets(_):
        return create_empty_bullets_16()

    new_enemy_bullets = jax.lax.cond(
        any_turret_firing,
        _generate_bullets,
        _get_empty_bullets,
        operand=None
    )

    # 5. Merge bullets and assign the state to final variables
    enemy_bullets = merge_bullets(current_enemy_bullets, new_enemy_bullets)
    enemy_bullets = _enforce_cap_keep_old(enemy_bullets, cap=state_after_ufo.max_active_enemy_bullets)
    fire_cooldown = next_frame_cooldown
    key = current_key
    # === Enemy LOGIC Over===

    # --- 5. Advance All Bullets ---
    bullets = update_bullets(bullets)
    enemy_bullets = update_bullets(enemy_bullets)

    # --- 6. Collision Detection ---
    bullets = _bullets_hit_terrain(state_after_ufo, bullets)
    enemy_bullets = _bullets_hit_terrain(state_after_ufo, enemy_bullets)
    enemies_before_hits = enemies
    bullets, enemies = check_enemy_hit(bullets, enemies)

    hit_enemy_mask = check_ship_enemy_collisions(ship_after_move, enemies, SHIP_RADIUS)
    enemies = enemies._replace(death_timer=jnp.where(hit_enemy_mask, ENEMY_EXPLOSION_FRAMES, enemies.death_timer))

    # Green enemies carry hidden tanks at the same coordinates; reveal those
    # tanks once the corresponding green enemy is killed.
    green_sprite_idx = int(SpriteIdx.ENEMY_GREEN)
    was_green_alive = (state_after_ufo.enemies.sprite_idx == green_sprite_idx) & (state_after_ufo.enemies.hp > 0)
    is_green_dead_now = (enemies.sprite_idx == green_sprite_idx) & (enemies.hp <= 0)
    green_killed_by_hp = was_green_alive & is_green_dead_now
    green_killed_by_contact = (enemies.sprite_idx == green_sprite_idx) & hit_enemy_mask
    green_just_killed = green_killed_by_hp | green_killed_by_contact

    same_x = jnp.abs(new_fuel_tanks.x[:, None] - enemies.x[None, :]) < 0.51
    same_y = jnp.abs(new_fuel_tanks.y[:, None] - enemies.y[None, :]) < 0.51
    same_position = same_x & same_y
    tank_matches_killed_green = jnp.any(same_position & green_just_killed[None, :], axis=1)

    hidden_tank_to_reveal = (~new_fuel_tanks.active) & (new_fuel_tanks.sprite_idx == int(SpriteIdx.FUEL_TANK)) & tank_matches_killed_green
    new_fuel_tanks = new_fuel_tanks._replace(active=new_fuel_tanks.active | hidden_tank_to_reveal)

    enemy_bullets, hit_by_enemy_bullet = consume_ship_hits(ship_after_move, enemy_bullets, SHIP_RADIUS)
    hit_terr = terrain_hit(state_after_ufo, ship_after_move.x, ship_after_move.y, 2)
    
    # Check for collision with UFO (rammer) - use state BEFORE UFO update to detect collision
    # (UFO may already be marked dead after update, but collision still happened)
    ufo_before_update = state_after_spawn.ufo
    hit_by_ufo = _circle_hit(ship_after_move.x, ship_after_move.y, SHIP_RADIUS, 
                              ufo_before_update.x, ufo_before_update.y, UFO_HIT_RADIUS) & ufo_before_update.alive

    # Check for collision with reactor destination core (fatal crash)
    dx_reactor = ship_after_move.x - state_after_ufo.reactor_dest_x
    dy_reactor = ship_after_move.y - state_after_ufo.reactor_dest_y
    dist_sq_reactor = dx_reactor**2 + dy_reactor**2
    hit_reactor_dest = is_in_reactor & state_after_ufo.reactor_dest_active & (dist_sq_reactor < (state_after_ufo.reactor_dest_radius + SHIP_RADIUS)**2)

    # --- 7. State Finalization ---
    # a) Initial check for ship death
    hit_enemy_types = jnp.where(hit_enemy_mask, enemies.sprite_idx, -1)
    is_orange_turret_type = (hit_enemy_types == int(SpriteIdx.ENEMY_ORANGE)) | \
                            (hit_enemy_types == int(SpriteIdx.ENEMY_ORANGE_FLIPPED))
    crashed_on_turret = jnp.any(
        is_orange_turret_type | (hit_enemy_types == int(SpriteIdx.ENEMY_GREEN)))
    bullet_hit_kills = hit_by_enemy_bullet & ~is_using_shield_tractor
    
    # UFO collision and reactor destination collision are always fatal (shield doesn't protect)
    dead = crashed_on_turret | bullet_hit_kills | hit_terr | timer_ran_out | hit_by_ufo | hit_reactor_dest

    # b) Special rules for the Reactor level 
    def check_reactor_hit(b: Bullets):
        dx = b.x - env_state.reactor_dest_x
        dy = b.y - env_state.reactor_dest_y
        dist_sq = dx**2 + dy**2
        hit_mask = b.alive & (dist_sq < (env_state.reactor_dest_radius + 5.0)**2)
        return jnp.any(hit_mask), b._replace(alive=b.alive & ~hit_mask)

    can_activate = is_in_reactor & ~env_state.reactor_activated
    hit_reactor_core, bullets_after_reactor_hit = jax.lax.cond(
        can_activate,
        check_reactor_hit,
        lambda b: (jnp.array(False), b),
        bullets
    )
    
    bullets = bullets_after_reactor_hit
    new_reactor_activated = env_state.reactor_activated | hit_reactor_core

    exited_top = ship_after_move.y < (HUD_HEIGHT + SHIP_RADIUS)
    win_reactor = is_in_reactor & new_reactor_activated & exited_top

    # UFO (rammer) doesn't count as an enemy that must be killed to clear a level
    all_enemies_gone = jnp.all(enemies.w == 0)
    reset_level_win = all_enemies_gone & (~is_in_reactor) & exited_top
    level_cleared_now = reset_level_win | win_reactor

    # c) Score calculation
    just_started_exploding = (
        (enemies_before_hits.w > 0)
        & (enemies_before_hits.death_timer == 0)
        & (enemies.death_timer > 0)
    )
    is_orange = (enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_ORANGE))) | \
                (enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_ORANGE_FLIPPED)))
    is_green = enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_GREEN))

    k_orange = jnp.sum(just_started_exploding & is_orange).astype(jnp.float32)
    k_green = jnp.sum(just_started_exploding & is_green).astype(jnp.float32)

    score_from_enemies = state_after_ufo.enemy_kill_score * (k_orange + k_green)
    score_from_level_clear = jnp.where(level_cleared_now, state_after_ufo.level_clear_score, 0.0)
    ufo_just_died = (state_after_ufo.ufo.alive == False) & (state_after_spawn.ufo.alive == True) & (
                state_after_ufo.ufo.death_timer > 0)
    score_from_ufo = jnp.where(ufo_just_died, state_after_ufo.ufo_kill_score, 0.0)
    score_delta = score_from_enemies + score_from_level_clear + score_from_ufo

    all_rewards = jnp.array([
        score_from_enemies,
        score_from_level_clear,
        score_from_ufo,
        jnp.float32(0.0),  # saucer_kill
        jnp.float32(0.0),  # penalty
    ], dtype=jnp.float32)
    
    # d) Crash and respawn logic
    was_crashing = state_after_ufo.crash_timer > 0
    start_crash = dead & (~was_crashing)
    crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(state_after_ufo.crash_timer - 1, 0))
    crash_animation_finished = (state_after_ufo.crash_timer == 1)

    respawn_now = crash_animation_finished & (~is_in_reactor)
    reset_from_reactor_crash = crash_animation_finished & is_in_reactor
    death_event = respawn_now | reset_from_reactor_crash

    lives_after_death = state_after_ufo.lives - jnp.where(death_event, 1, 0)
    
    score_before_update = state_after_ufo.score
    score_after_update = score_before_update + score_delta
    
    bonus_life_threshold_crossed = (score_after_update // 10000) > (score_before_update // 10000)
    lives_gained_from_score = jnp.where(bonus_life_threshold_crossed, 1, 0)
    
    final_lives = lives_after_death + lives_gained_from_score
    
    # Don't reset fuel on respawn - keep current fuel level
    final_fuel = jnp.maximum(0.0, fuel_after_actions)
    
    # e) The final Reset signal   # UFO (rammer) doesn't count as an enemy that must be killed to win
    all_enemies_gone = jnp.all(enemies.w == 0)
    
    # Planet level win: all enemies destroyed AND player exits through top
    # Remove has_meaningful_enemies check - if you can exit (exit_allowed was True), you should be able to leave
    reset_level_win = all_enemies_gone & (~is_in_reactor) & exited_top
    
    # Reactor early exit: allow leaving reactor without destroying it (no rewards)
    reset_reactor_early_exit = is_in_reactor & (~new_reactor_activated) & exited_top

    reset = reset_level_win | win_reactor | reset_from_reactor_crash | reset_reactor_early_exit

    # f) Is the game over?
    game_over = (death_event & (final_lives <= 0)) 

    # --- 8. Respawn State Transition ---
    def _respawn_level_state(operands):
        s, b, eb, fc, cd = operands
        ship_respawn = make_level_start_state(s.current_level)
        ship_respawn = ship_respawn._replace(x=ship_respawn.x + s.respawn_shift_x)
        return (
            ship_respawn,
            create_empty_bullets_64(),
            create_empty_bullets_16(),
            jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
            0,
            make_empty_ufo(),
            jnp.int32(UFO_RESPAWN_DELAY_FRAMES),
        )

    def _keep_state_no_respawn(operands):
        return (
            ship_after_move,
            bullets,
            enemy_bullets,
            fire_cooldown,
            cooldown,
            ufo,
            state_after_ufo.ufo_spawn_timer,
        )

    state, bullets, enemy_bullets, fire_cooldown, cooldown, ufo, ufo_spawn_timer = jax.lax.cond(
        respawn_now & ~game_over,
        _respawn_level_state,
        _keep_state_no_respawn,
        operand=(state_after_ufo, bullets, enemy_bullets, fire_cooldown, cooldown)
    )

    # --- 9. Assemble and Return Final State ---
    reactor_destroyed_next = state_after_ufo.reactor_destroyed | win_reactor

    # Find which planet index corresponds to the current level
    # planets_id contains level IDs, we need to find which planet has current_level
    current_level_id = state_after_ufo.current_level
    planet_indices = jnp.arange(state_after_ufo.planets_id.shape[0])
    matches_current_level = (state_after_ufo.planets_id == current_level_id)
    # Use argmax to get first matching index (will be 0 if no match, but that's ok since we guard with reset_level_win)
    current_planet_idx = jnp.argmax(matches_current_level.astype(jnp.int32))
    
    cleared_mask_next = jnp.where(
        reset_level_win,
        state_after_ufo.planets_cleared_mask.at[current_planet_idx].set(True),
        state_after_ufo.planets_cleared_mask
    )

    # Update exit_allowed flag for next frame:
    # - Reactor level: always allowed (can exit early or after destroying reactor)
    # - Other levels: only after all enemies destroyed
    new_exit_allowed = is_in_reactor | new_reactor_activated | (all_enemies_gone & (~is_in_reactor))

    final_env_state = state_after_ufo._replace(
        state=state, bullets=bullets, cooldown=cooldown, enemies=enemies,
        enemy_bullets=enemy_bullets, fire_cooldown=fire_cooldown, key=key,
        ufo=ufo, ufo_bullets=create_empty_bullets_16(),
        ufo_spawn_timer=ufo_spawn_timer,
        fuel_tanks=new_fuel_tanks,
        fuel=final_fuel,
        shield_active=is_using_shield_tractor,
        reactor_timer=new_reactor_timer,
        reactor_activated=new_reactor_activated,
        score=score_after_update, 
        crash_timer=crash_timer_next,
        lives=final_lives,
        done=game_over,
        reactor_destroyed=reactor_destroyed_next,
        planets_cleared_mask=cleared_mask_next,
        mode_timer=state_after_ufo.mode_timer + 1,
        exit_allowed=new_exit_allowed,
        prev_action=action,
    )

    obs = _get_observation_from_state(final_env_state)

    reward = score_delta
    info = GravitarInfo(
        lives=final_env_state.lives,
        score=final_env_state.score,
        fuel=final_env_state.fuel,
        mode=final_env_state.mode,
        crash_timer=final_env_state.crash_timer,
        done=final_env_state.done,
        current_level=final_env_state.current_level,
        crash=start_crash,
        hit_by_bullet=hit_by_enemy_bullet,
        reactor_crash_exit=reset_from_reactor_crash,
        all_rewards=all_rewards,
    )

    return obs, final_env_state, reward, game_over, info, reset, jnp.int32(-1)

batched_terrain_hit = jax.vmap(terrain_hit, in_axes=(None, 0, 0, None))


# ========== Arena Step Core ==========
@jax.jit
def step_arena(env_state: EnvState, action: int):
    # --- 1. Setup ---
    ship = env_state.state
    saucer = env_state.saucer
    is_crashing = env_state.crash_timer > 0

    # --- 2. Ship Movement and Player Firing ---
    # If the ship is crashing, ignore player input and force no movement
    actual_action = jnp.where(is_crashing, NOOP, action)
    ship_after_move = ship_step(ship, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT, env_state.fuel, env_state.terrain_bank_idx,
                                thrust_power=env_state.thrust_power,
                                max_speed=env_state.max_speed,
                                solar_gravity=env_state.solar_gravity,
                                planetary_gravity=env_state.planetary_gravity,
                                reactor_gravity=env_state.reactor_gravity)

    # Detect fire button press (not hold) - only fire on transition
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_fire_pressed = jnp.isin(action, fire_actions)
    was_fire_pressed = jnp.isin(env_state.prev_action, fire_actions)
    fire_just_pressed = is_fire_pressed & (~was_fire_pressed)
    
    can_fire = fire_just_pressed & (env_state.cooldown == 0) & (
                _bullets_alive_count(env_state.bullets) < env_state.max_active_player_bullets_arena)

    bullets = jax.lax.cond(
        can_fire,
        lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, PLAYER_BULLET_SPEED),
        lambda b: b,
        env_state.bullets
    )

    bullets = update_bullets(bullets)
    cooldown = jnp.where(can_fire, PLAYER_FIRE_COOLDOWN_FRAMES, jnp.maximum(env_state.cooldown - 1, 0))

    # --- Calculate fuel consumption ---
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])
    is_thrusting = jnp.isin(action, thrust_actions)
    fuel_consumed = jnp.where(is_thrusting, env_state.fuel_consume_thrust, 0.0)
    fuel_after_actions = jnp.maximum(0.0, env_state.fuel - fuel_consumed)

    # --- 3. Saucer Movement and Firing ---
    fire_key, new_main_key = jax.random.split(env_state.key)
    saucer_after_move = jax.lax.cond(saucer.alive,
                                     lambda s: _update_saucer_seek(s, ship_after_move.x, ship_after_move.y,
                                                                   SAUCER_SPEED_ARENA), lambda s: s, operand=saucer)
    enemy_bullets = _saucer_fire_random(
        saucer_after_move,
        env_state.enemy_bullets,
        env_state.mode_timer,
        fire_key,
        env_state.max_active_saucer_bullets,
    )
    enemy_bullets = update_bullets(enemy_bullets)

    # --- 4. Collision Detection ---
    # a) Player bullet hits Saucer
    bullets, hit_saucer_by_bullet = _bullets_hit_saucer(bullets, saucer_after_move)

    # b) Ship and Saucer collide directly
    hit_saucer_by_contact = _circle_hit(ship_after_move.x, ship_after_move.y, SHIP_RADIUS, saucer_after_move.x,
                                        saucer_after_move.y, SAUCER_RADIUS) & saucer_after_move.alive

    # c) Saucer bullet hits ship
    enemy_bullets, hit_ship_by_bullet = consume_ship_hits(ship_after_move, enemy_bullets, SHIP_RADIUS)

    # --- 5. State Finalization ---
    # a) Is the Saucer dead?
    # Death conditions: Hit by bullet ONLY (collision with ship doesn't destroy saucer)
    saucer_is_hit = hit_saucer_by_bullet
    hp_after_hit = saucer_after_move.hp - jnp.where(saucer_is_hit, 1, 0)  # Simplified: 1 HP is lost per hit
    was_alive = saucer_after_move.alive

    is_dead_now = hp_after_hit <= 0
    just_died = was_alive & is_dead_now

    saucer_final = saucer_after_move._replace(
        hp=hp_after_hit,
        alive=was_alive & (~is_dead_now),
        death_timer=jnp.where(just_died, SAUCER_EXPLOSION_FRAMES, jnp.maximum(saucer_after_move.death_timer - 1, 0))
    )

    # b) Is the ship dead?
    # Death conditions: Hit by Saucer bullet OR collided with Saucer
    ship_is_hit = hit_ship_by_bullet | hit_saucer_by_contact

    # c) Update the ship's crash timer
    start_crash = ship_is_hit & (~is_crashing)  # A crash is only initiated if hit while not already crashing
    crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(env_state.crash_timer - 1, 0))

    # d) Determine if a reset signal should be sent
    # Signal condition: The ship's crash animation has just finished playing (timer goes from 1 to 0)
    reset_signal = (env_state.crash_timer == 1)
    # If the ship didn't die but the saucer's explosion animation is finished, also exit the Arena
    back_to_map_signal = (~ship_is_hit) & (~saucer_final.alive) & (saucer_final.death_timer == 0)

    # --- 6. Assemble and Return ---
    obs = _get_observation_from_state(env_state._replace(
        state=ship_after_move,
        bullets=bullets,
        cooldown=cooldown,
        saucer=saucer_final,
        enemy_bullets=enemy_bullets,
        fire_cooldown=env_state.fire_cooldown,
        fuel=fuel_after_actions,
        crash_timer=crash_timer_next,
    ))
    reward = jnp.where(just_died, env_state.saucer_kill_score, jnp.float32(0.0))
    done = jnp.array(False)  # The Arena itself never ends the episode; we return to the map instead
    info = GravitarInfo(
        lives=env_state.lives,
        score=env_state.score,
        fuel=fuel_after_actions,
        mode=env_state.mode,
        crash_timer=crash_timer_next,
        done=done,
        current_level=env_state.current_level,
        crash=start_crash,
        hit_by_bullet=hit_ship_by_bullet,
        reactor_crash_exit=jnp.array(False),
        all_rewards=jnp.array([
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            reward,
            jnp.float32(0.0),
        ], dtype=jnp.float32),
    )

    final_env_state = env_state._replace(
        state=ship_after_move,
        bullets=bullets,
        cooldown=cooldown,
        saucer=saucer_final,
        enemy_bullets=enemy_bullets,
        key=new_main_key,
        crash_timer=crash_timer_next,
        mode_timer=env_state.mode_timer + 1,
        score=env_state.score + reward,
        fuel=fuel_after_actions,
        prev_action=action,
    )

    # If it's a "win" exit, return directly to the map and restore ship position
    def _go_to_map_win(env):
        # Restore the ship to its position before entering the arena
        # Keep the ship's original angle and thrusting state
        restored_ship = ShipState(
            x=env.map_return_x,
            y=env.map_return_y,
            vx=env.map_return_vx,
            vy=env.map_return_vy,
            angle=env.map_return_angle,
            is_thrusting=jnp.array(False),
            rotation_cooldown=jnp.int32(0)
        )
        return env._replace(
            # Keep arena mode for this transition frame so step_full can
            # correctly take the arena-return branch and preserve trajectory.
            mode=jnp.int32(2), 
            state=restored_ship,
            saucer=make_default_saucer(),
            saucer_spawn_timer=jnp.int32(SAUCER_RESPAWN_DELAY_FRAMES)
        )

    final_env_state = jax.lax.cond(back_to_map_signal, _go_to_map_win, lambda e: e, final_env_state)

    # The final `reset` signal is either "crash finished" or "win exit"
    return obs, final_env_state, reward, done, info, reset_signal | back_to_map_signal, jnp.int32(-1)


@jax.jit
def _bullets_hit_terrain(env_state: EnvState, bullets: Bullets) -> Bullets:
    H, W = env_state.terrain_bank.shape[1], env_state.terrain_bank.shape[2]

    bank_idx = jnp.clip(env_state.terrain_bank_idx, 0, env_state.terrain_bank.shape[0] - 1)
    terrain_map = env_state.terrain_bank[bank_idx]

    xi = jnp.clip(jnp.round(bullets.x).astype(jnp.int32), 0, W - 1)
    yi = jnp.clip(jnp.round(bullets.y).astype(jnp.int32), 0, H - 1)

    pixel_colors = terrain_map[yi, xi]
    bg_val = env_state.terrain_bank[0, 0, 0]
    hit_terrain_mask = pixel_colors != bg_val

    final_hit_mask = bullets.alive & hit_terrain_mask

    return bullets._replace(alive=bullets.alive & ~final_hit_mask)


@jax.jit
def _ufo_ground_safe_y_at(terrain_bank, terrain_bank_idx, xf):
    W, H = WINDOW_WIDTH, WINDOW_HEIGHT
    bank_idx = jnp.clip(terrain_bank_idx, 0, terrain_bank.shape[0] - 1)

    terrain_page = terrain_bank[bank_idx]
    col_x = jnp.clip(xf.astype(jnp.int32), 0, W - 1)
    bg_val = terrain_bank[0, 0, 0]
    is_ground_in_col = terrain_page[:, col_x] != bg_val
    y_indices = jnp.arange(H, dtype=jnp.int32)

    ground_indices = jnp.where(is_ground_in_col, y_indices, H)
    ground_y = jnp.min(ground_indices)

    return jnp.float32(ground_y) - 20.0  # CLEARANCE


# --- Logic for when the UFO is alive ---
@jax.jit
def _ufo_alive_step(e, ship, bullets):
    u = e.ufo
    # --- 1. Define physics and boundary constants ---
    LEFT_BOUNDARY = 8.0
    RIGHT_BOUNDARY = WINDOW_WIDTH - 8.0
    MIN_ALTITUDE = jnp.float32(HUD_HEIGHT + 20.0)
    VERTICAL_ADJUST_SPEED = 0.5 / WORLD_SCALE  # Max vertical speed for the UFO

    # --- 2. Horizontal Movement Logic ---
    # a. Calculate the theoretical next X position
    next_x = u.x + u.vx

    # b. Check for collision with horizontal boundaries
    hit_left_wall = (next_x <= LEFT_BOUNDARY)
    hit_right_wall = (next_x >= RIGHT_BOUNDARY)
    hit_horizontal_boundary = hit_left_wall | hit_right_wall

    # c. Calculate the final horizontal velocity
    final_vx = jnp.where(hit_horizontal_boundary, -u.vx, u.vx)

    # d. Update horizontal position and clamp it for safety
    final_x = jnp.clip(u.x + final_vx, LEFT_BOUNDARY, RIGHT_BOUNDARY)

    # --- 3. Vertical Movement Logic (New smooth version) ---
    # a. Find the safe Y coordinate below the current X position
    safe_y_here = _ufo_ground_safe_y_at(e.terrain_bank, e.terrain_bank_idx, final_x)

    # b. Determine the UFO's vertical target position
    #    Target = 20 pixels above ground, but not lower than the minimum altitude
    target_y = jnp.maximum(safe_y_here - 20.0, MIN_ALTITUDE)

    # c. Calculate the vertical velocity towards the target
    #    If UFO is above target (u.y < target_y), vy should be positive (move down)
    #    If UFO is below target (u.y > target_y), vy should be negative (move up)
    y_difference = target_y - u.y

    #    Clip the vertical velocity to achieve smooth movement
    final_vy = jnp.clip(y_difference, -VERTICAL_ADJUST_SPEED, VERTICAL_ADJUST_SPEED)

    # d. Update vertical position based on the calculated velocity
    final_y = u.y + final_vy

    # e. Final safety clamp to ensure the UFO never leaves the safe zone
    final_y = jnp.clip(final_y, MIN_ALTITUDE, WINDOW_HEIGHT - 20.0)

    # --- 4. Assemble the updated UFO state after all calculations ---
    u_after_move = u._replace(x=final_x, y=final_y, vx=final_vx, vy=final_vy)

    # --- 5. Subsequent logic (collision, firing, etc.) remains unchanged ---

    # a. Collision Detection
    hit_by_ship = _circle_hit(ship.x, ship.y, SHIP_RADIUS, u_after_move.x, u_after_move.y,
                              UFO_HIT_RADIUS) & u_after_move.alive
    bullets_after_hit, hit_by_bullet = _bullets_hit_ufo(bullets, u_after_move)

    # b. State Update
    hp_after_hit = u_after_move.hp - jnp.where(hit_by_bullet, 1, 0)
    was_alive = u_after_move.alive

    is_dead_now = (hp_after_hit <= 0)
    just_died = was_alive & is_dead_now

    u_final = u_after_move._replace(
        hp=hp_after_hit,
        alive=was_alive & (~is_dead_now),
        death_timer=jnp.where(just_died, SAUCER_EXPLOSION_FRAMES, u_after_move.death_timer)
    )

    # c. UFOs should NOT fire - remove firing logic
    # According to Atari manual, UFOs (rammers) only chase the ship, they don't shoot
    
    # 6. Return the final environment state with all updates
    # Keep ufo_bullets empty since UFOs don't shoot
    return e._replace(ufo=u_final, bullets=bullets_after_hit, ufo_bullets=create_empty_bullets_16())


@jax.jit
# --- Logic for when the UFO is dead ---
def _ufo_dead_step(e, ship, bullets):
    u = e.ufo
    u2 = u._replace(death_timer=jnp.maximum(u.death_timer - 1, 0))
    # Tick down spawn timer when death animation is finished
    new_spawn_timer = jnp.where(
        u.death_timer == 0,
        jnp.maximum(e.ufo_spawn_timer - 1, 0),
        e.ufo_spawn_timer
    )
    # Return the COMPLETE environment state
    return e._replace(ufo=u2, ufo_spawn_timer=new_spawn_timer, ufo_bullets=create_empty_bullets_16(), bullets=bullets)


@jax.jit
def _update_ufo(env: EnvState, ship: ShipState, bullets: Bullets) -> EnvState:
    return jax.lax.cond(
        env.ufo.alive,
        _ufo_alive_step,
        _ufo_dead_step,
        env, ship, bullets
    )


# ========== Step Core ==========
@jax.jit
def step_core(env_state: EnvState, action: int):
    # `jax.lax.switch` selects a function from the list based on the value of `mode` (0, 1, or 2)
    # It automatically passes the `(env_state, action)` operands to the chosen function.
    def _game_is_over(state, _):
        # If the game is over (state.done is True), do nothing and return the current state.
        # This effectively freezes the game.

        # 1. Create an info dictionary with the same structure as the _game_is_running branch
        #    to satisfy the type requirements of jax.lax.cond.
        info = GravitarInfo(
            lives=state.lives,
            score=state.score,
            fuel=state.fuel,
            mode=state.mode,
            crash_timer=state.crash_timer,
            done=state.done,
            current_level=state.current_level,
            crash=jnp.array(False),
            hit_by_bullet=jnp.array(False),
            reactor_crash_exit=jnp.array(False),
            all_rewards=jnp.array([
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(0.0),
            ], dtype=jnp.float32),
        )

        # 2. Return a tuple with the same pytree structure as the other branch.
        obs = _get_observation_from_state(state)

        return obs, state, 0.0, jnp.array(True), info, jnp.array(False), jnp.int32(-1)

    def _game_is_running(state, act):
        # If the game is still running, use jax.lax.switch to call the correct
        # step function based on the game mode (0: map, 1: level, 2: arena).
        return jax.lax.switch(
            jnp.clip(state.mode, 0, 2),
            [step_map, _step_level_core, step_arena],  
            state,
            act
        )

    return jax.lax.cond(
        env_state.done,
        _game_is_over,
        _game_is_running,
        env_state,
        action
    )


@partial(jax.jit, static_argnums=(2,))
def step_full(env_state: EnvState, action: int, env_instance: 'JaxGravitar'):
    """
    Executes a full step of the game logic, including handling resets.
    This function is designed to be JIT-compiled and contains the main
    state transition logic for the entire game.
    """

    def _handle_reset(operands):
        """
        This branch is executed only when the `reset` flag from step_core is True.
        It handles all state transitions, such as entering a level or returning to the map.
        """
        obs, current_state, reward, done, info, reset, level = operands

        # === BRANCH 1: ENTER A LEVEL ===
        def _enter_level(_):
            """Handles the transition from the map into a level with proper terrain and entity loading."""
            new_main_key, subkey_for_reset = jax.random.split(current_state.key)

            # Properly load level with terrain, enemies, and fuel tanks
            obs_reset, next_state = env_instance.reset_level(subkey_for_reset, level, current_state)
            next_state = next_state._replace(key=new_main_key)

            enter_info = info.replace(level_cleared=jnp.array(False))
            return obs_reset, next_state, reward, jnp.array(False), enter_info, jnp.array(True), level

        # === BRANCH 2: RETURN TO THE MAP ===
        def _return_to_map(_):
            """Handles the transition from a level back to the map (due to win, loss, or crash)."""
            # Check if returning from arena (mode=2) vs planet level (mode=1)
            # Both return with level=-1, so we need to use mode to distinguish
            is_from_arena = (current_state.mode == 2)
            
            # Death event detection:
            # For arena (mode=2, level=-1): it's a death if saucer is still alive or death animation playing
            # A win only occurs if saucer is dead (alive=False) AND death animation finished (death_timer=0)
            is_arena_death = is_from_arena & (level == -1) & (current_state.saucer.alive | (current_state.saucer.death_timer > 0))
            fuel_depleted = current_state.fuel <= 0.0
            # In level mode, `current_state.done` is raised by `_step_level_core` on terminal death
            # (e.g., losing last life). Treat it as an explicit death-event signal here.
            terminal_done_death = current_state.done
            is_a_death_event = (level == -2) | info.get("crash", False) | info.get("hit_by_bullet", False) | info.get(
                "reactor_crash_exit", False) | is_arena_death | fuel_depleted | terminal_done_death

            def _on_win(_):
                new_main_key, subkey_for_reset = jax.random.split(current_state.key)
                
                # Check if solar system is complete (reactor destroyed OR all planets cleared)
                all_planets_cleared = jnp.all(current_state.planets_cleared_mask)
                solar_system_complete = current_state.reactor_destroyed | all_planets_cleared
                
                # When solar system is complete: add bonuses and reset planets and reactor
                bonus_fuel = jnp.where(solar_system_complete, SOLAR_SYSTEM_BONUS_FUEL, 0.0)
                bonus_lives = jnp.where(solar_system_complete, SOLAR_SYSTEM_BONUS_LIVES, 0)
                bonus_score = jnp.where(solar_system_complete, SOLAR_SYSTEM_BONUS_SCORE, 0.0)
                final_fuel = current_state.fuel + bonus_fuel
                final_lives = current_state.lives + bonus_lives
                final_score = current_state.score + bonus_score
                
                # Reset reactor and planets if solar system was completed, otherwise keep state
                final_reactor_destroyed = jnp.where(solar_system_complete, jnp.array(False), current_state.reactor_destroyed)
                final_planets_cleared = jnp.where(solar_system_complete, 
                                                   jnp.zeros_like(current_state.planets_cleared_mask),
                                                   current_state.planets_cleared_mask)
                
                # If returning from arena, preserve current ship position; otherwise reset to spawn
                def _arena_return_state():
                    # Keep the ship exactly as it was restored in step_arena
                    return current_state._replace(
                        mode=jnp.int32(0),
                        key=new_main_key,
                        done=jnp.array(False),
                        fuel=final_fuel,
                        lives=final_lives,
                        score=final_score,
                        reactor_destroyed=final_reactor_destroyed,
                        planets_cleared_mask=final_planets_cleared
                    )
                
                def _level_return_state():
                    # Check if returning from reactor level (level 4)
                    is_from_reactor = (current_state.current_level == 4)
                    
                    # Reactor: spawn at original spawn point
                    # Planets: restore to position where player entered the level
                    return_x = jnp.where(is_from_reactor, jnp.float32(77.0), current_state.map_return_x)
                    return_y = jnp.where(is_from_reactor, jnp.float32(131.0), current_state.map_return_y)
                    
                    restored_ship = ShipState(
                        x=return_x,
                        y=return_y,
                        vx=jnp.float32(0.0),
                        vy=jnp.float32(0.0),
                        angle=jnp.float32(-jnp.pi / 2),
                        is_thrusting=jnp.array(False),
                        rotation_cooldown=jnp.int32(0)
                    )
                    # Call reset_map to properly initialize all map state, then override ship position
                    obs_reset, map_state = env_instance.reset_map(
                        subkey_for_reset,
                        lives=final_lives,
                        score=final_score,
                        fuel=final_fuel,
                        reactor_destroyed=final_reactor_destroyed,
                        planets_cleared_mask=final_planets_cleared
                    )
                    # Override ship position to restore where the player entered from (or spawn for reactor)
                    # Preserve map_return coordinates in case they're needed again
                    return map_state._replace(
                        key=new_main_key,
                        state=restored_ship,
                        map_return_x=current_state.map_return_x,
                        map_return_y=current_state.map_return_y
                    )
                
                final_state = jax.lax.cond(is_from_arena, _arena_return_state, _level_return_state)
                obs_out = env_instance._get_observation(final_state)
                win_info = info.replace(level_cleared=jnp.array(True))
                return obs_out, final_state, reward, jnp.array(False), win_info, jnp.array(True), level

            def _on_death(_):
                # Level mode (1) already decrements lives in `_step_level_core`.
                # Map/Arena modes (0/2) rely on this reset handler to decrement.
                is_level_mode = (current_state.mode == jnp.int32(1))
                is_reactor_level = (current_state.current_level == jnp.int32(4))
                is_planet_level = is_level_mode & (~is_reactor_level)
                fuel_depleted_now = current_state.fuel <= 0.0
                force_full_reset = is_planet_level | fuel_depleted_now

                lives_after_death = jnp.where(
                    force_full_reset,
                    current_state.lives,
                    jnp.where(
                    current_state.mode == jnp.int32(1),
                    current_state.lives,
                    current_state.lives - 1,
                    ),
                )
                death_info = info.replace(level_cleared=jnp.array(False))
                is_game_over = force_full_reset | (lives_after_death <= 0)

                # In level mode, keep in-level respawn when lives remain,
                # except for reactor level (4), which should return to the
                # solar-system spawn on death.
                stay_in_level = is_level_mode & (~is_game_over) & (~is_reactor_level)

                def _continue_level_respawn():
                    # Rebuild the current level from its start configuration while
                    # preserving already-updated meta state (lives/score/fuel/etc.)
                    # from `current_state`.
                    new_main_key, subkey_for_reset = jax.random.split(current_state.key)
                    respawn_level = current_state.current_level
                    obs_reset, next_state = env_instance.reset_level(
                        subkey_for_reset,
                        respawn_level,
                        current_state,
                    )
                    next_state = next_state._replace(key=new_main_key, done=jnp.array(False))
                    return obs_reset, next_state, reward, jnp.array(False), death_info, jnp.array(False), respawn_level

                def _reset_to_map_after_death():
                    new_main_key, subkey_for_reset = jax.random.split(current_state.key)
                    obs_reset, map_state = jax.lax.cond(
                        is_game_over,
                        # FULL RESET
                        lambda: env_instance.reset_map(subkey_for_reset),
                        # NORMAL DEATH RESET
                        lambda: env_instance.reset_map(
                            subkey_for_reset,
                            lives=lives_after_death,
                            score=current_state.score,
                            fuel=current_state.fuel,
                            reactor_destroyed=current_state.reactor_destroyed,
                            planets_cleared_mask=current_state.planets_cleared_mask
                        )
                    )
                    final_map_state = map_state._replace(
                        key=new_main_key,
                        done=jnp.array(False, dtype=bool)
                    )
                    return obs_reset, final_map_state, reward, is_game_over, death_info, jnp.array(True), level

                return jax.lax.cond(
                    stay_in_level,
                    lambda: _continue_level_respawn(),
                    lambda: _reset_to_map_after_death(),
                )

            return jax.lax.cond(is_a_death_event, _on_death, _on_win, operand=None)

        return jax.lax.cond(level >= 0, _enter_level, _return_to_map, operand=None)

    def _no_reset(operands):
        obs, new_env_state, reward, done, info, reset, level = operands
        no_reset_info = info.replace(level_cleared=jnp.array(False))
        new_env_state = new_env_state._replace(done=jnp.array(False, dtype=bool))
        return obs, new_env_state, reward, done, no_reset_info, reset, level

    obs, new_env_state, reward, done, info, reset, level = step_core(env_state, action)

    # Ensure life-loss transitions go through reset handling only outside level
    # mode. In level mode, `_step_level_core` already handles in-level respawn
    # and should not be redirected back to the solar-system map.
    life_lost = new_env_state.lives < env_state.lives
    fuel_depleted = new_env_state.fuel <= 0.0
    is_level_mode = env_state.mode == jnp.int32(1)
    forced_death_reset = life_lost & (~reset) & (~is_level_mode) & (~done)
    forced_fuel_reset = fuel_depleted & (~reset) & (~done)

    effective_reset = reset | forced_death_reset | forced_fuel_reset | done
    effective_level = jnp.where(forced_death_reset | forced_fuel_reset, jnp.int32(-2), level)

    operands = (obs, new_env_state, reward, done, info, effective_reset, effective_level)
    #return jax.lax.cond(effective_reset, _handle_reset, _no_reset, operands)

    obs, new_env_state, reward, done, info, reset, level = jax.lax.cond(effective_reset, _handle_reset, _no_reset, operands)

    # Ensure API consistency
    #done = new_env_state.done
    
    return obs, new_env_state, reward, done, info, reset, level


def get_action_from_key():
    """Placeholder function for key input - returns NOOP since we don't use Pygame input in benchmarks"""
    return 0  # NOOP

class JaxGravitar(JaxEnvironment):
    def __init__(self, consts: GravitarConstants = None):
        consts = consts or GravitarConstants()
        super().__init__(consts)
        self.obs_shape = (5,)
        self.num_actions = 18

        # ---- Resource Loading and JAX Renderer Initialization ----
        self.sprites = _OBS_SPRITES
        self.renderer = GravitarRenderer(width=self.consts.WINDOW_WIDTH, height=self.consts.WINDOW_HEIGHT, consts=self.consts)

        # --- Store original sprite dimensions ---
        self.sprite_dims = {}
        sprites_to_measure = [
            SpriteIdx.ENEMY_ORANGE, SpriteIdx.ENEMY_GREEN,
            SpriteIdx.FUEL_TANK, SpriteIdx.ENEMY_UFO,
            SpriteIdx.ENEMY_ORANGE_FLIPPED,
        ]
        for sprite_idx in sprites_to_measure:
            sprite_surf = self.sprites[sprite_idx]
            if sprite_surf is not None:
                # NumPy arrays: height, width
                self.sprite_dims[int(sprite_idx)] = (sprite_surf.shape[1], sprite_surf.shape[0])

        # --- Map layout and collision radii ---
        MAP_SCALE = 3
        HITBOX_SCALE = 0.90
        # Layout stores exact ALE center pixel coordinates directly (no fraction conversion)
        layout = [
            (SpriteIdx.PLANET1, 126, 46),
            (SpriteIdx.PLANET2, 37, 67),
            (SpriteIdx.REACTOR, 22, 107),
            (SpriteIdx.SPAWN_LOC, 75, 131),
            (SpriteIdx.OBSTACLE, 82, 86),
            (SpriteIdx.PLANET3, 142, 157),
            (SpriteIdx.PLANET4, 30, 177),
        ]
        px, py, pr, pi = [], [], [], []
        for idx, center_x, center_y in layout:
            cx, cy = float(center_x), float(center_y)
            spr = self.sprites[idx]
            if spr is not None:
                # SPAWN_LOC is just a visual marker, no collision needed
                if idx == SpriteIdx.SPAWN_LOC:
                    r = 0.0
                # OBSTACLE (13x8 sprite) needs smaller hitbox than planets
                elif idx == SpriteIdx.OBSTACLE:
                    r = 0.15 * max(spr.shape[1], spr.shape[0]) * MAP_SCALE * HITBOX_SCALE
                else:
                    # Planets and reactor use larger circular hitbox
                    r = 0.3 * max(spr.shape[1], spr.shape[0]) * MAP_SCALE * HITBOX_SCALE
            else:
                r = 4
            px.append(cx)
            py.append(cy)
            pr.append(r)
            pi.append(int(idx))
        self.planets = (np.array(px, dtype=np.float32), np.array(py, dtype=np.float32), np.array(pr, dtype=np.float32),
                        np.array(pi, dtype=np.int32))
        self.terrain_bank = self._build_terrain_bank()

        # --- Convert all level data to JAX arrays ---
        # Reactor layout can be overridden through constants to support mods that
        # add enemies/fuel tanks in level 4 without patching base code.
        num_levels = max(LEVEL_LAYOUTS.keys()) + 1
        reactor_override = tuple(self.consts.REACTOR_LEVEL_LAYOUT)
        max_default_objects = max(len(v) for v in LEVEL_LAYOUTS.values()) if LEVEL_LAYOUTS else 0
        max_objects = max(max_default_objects, len(reactor_override))
        layout_types = np.full((num_levels, max_objects), -1, dtype=np.int32)
        layout_coords_x = np.zeros((num_levels, max_objects), dtype=np.float32)
        layout_coords_y = np.zeros((num_levels, max_objects), dtype=np.float32)
        for level_id, layout_data in LEVEL_LAYOUTS.items():
            level_layout_data = reactor_override if (level_id == 4 and len(reactor_override) > 0) else layout_data
            for i, obj in enumerate(level_layout_data):
                if isinstance(obj, dict):
                    obj_type = obj["type"]
                    coord_x, coord_y = obj["coords"]
                else:
                    obj_type, coord_x, coord_y = obj
                layout_types[level_id, i] = int(obj_type)
                layout_coords_x[level_id, i] = coord_x
                layout_coords_y[level_id, i] = coord_y
        self.jax_layout = {"types": jnp.array(layout_types), "coords_x": jnp.array(layout_coords_x),
                           "coords_y": jnp.array(layout_coords_y)}

        max_sprite_id = max(int(e) for e in SpriteIdx)
        dims_array = np.zeros((max_sprite_id + 1, 2), dtype=np.float32)
        for k, v in self.sprite_dims.items():
            dims_array[k] = v
        self.jax_sprite_dims = jnp.array(dims_array)

        level_ids_sorted = sorted(LEVEL_ID_TO_TERRAIN_SPRITE.keys())
        self.jax_level_to_terrain = jnp.array([LEVEL_ID_TO_TERRAIN_SPRITE[k] for k in level_ids_sorted])
        self.jax_level_to_bank = jnp.array([LEVEL_ID_TO_BANK_IDX[k] for k in level_ids_sorted])
        self.jax_level_offsets = jnp.array([LEVEL_OFFSETS[k] for k in level_ids_sorted])

        level_transforms = np.zeros((num_levels, 3), dtype=np.float32)  # scale, ox, oy
        for level_id in level_ids_sorted:
            terrain_sprite_enum = LEVEL_ID_TO_TERRAIN_SPRITE[level_id]
            terr_surf = self.sprites[terrain_sprite_enum]
            th, tw = terr_surf.shape[0], terr_surf.shape[1]
            scale = min(WINDOW_WIDTH / tw, WINDOW_HEIGHT / th)
            extra = TERRAIN_SCALE_OVERRIDES.get(terrain_sprite_enum, 1.0)
            scale *= float(extra)
            sw, sh = int(tw * scale), int(th * scale)
            level_offset = LEVEL_OFFSETS.get(level_id, (0, 0))
            ox = (WINDOW_WIDTH - sw) // 2 + level_offset[0]
            oy = (WINDOW_HEIGHT - sh) // 2 + level_offset[1]
            level_transforms[level_id] = [scale, ox, oy]
        self.jax_level_transforms = jnp.array(level_transforms)

        # ---- JIT Helper Initialization ----
        dummy_key = jax.random.PRNGKey(0)
        _obs_dummy, dummy_state = self.reset(dummy_key)
        tmp_obs, tmp_state = self.reset_level(dummy_key, jnp.int32(0), dummy_state)

        obs_struct = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            tmp_obs
        )
        state_struct = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            tmp_state
        )

        self.reset_level_out_struct = (obs_struct, state_struct)

    def _get_reward(self, previous_state: EnvState, state: EnvState) -> jnp.ndarray:
        """
        Calculates the reward based on the change in score between two states.
        Args:
            previous_state: The environment state before the last action.
            state: The environment state after the last action.

        Returns: The reward for the last transition.
        """
        reward = state.score - previous_state.score
        return reward

    def _get_done(self, state: EnvState) -> jnp.ndarray:
        """
        Determines if the episode has terminated.
        Args:
            state: The current environment state.

        Returns: A boolean JAX array indicating termination.
        """
        return state.done

    def _get_observation(
        self,
        state: EnvState,
    ) -> GravitarObservation:
        """
        Extracts the structured observation from the environment state.
        Args:
            state: The current environment state.

        Returns: A structured observation dataclass containing the vector observation.
        """
        return _get_observation_from_state(state)
    

    def _get_info(self, state: EnvState, all_rewards: Optional[jnp.ndarray] = None) -> GravitarInfo:
        """
        Extracts debugging information from the environment state.
        Args:
            state: The current environment state.
            all_rewards: Optional array of rewards from the last step, if available.

        Returns: A structured info dataclass.
        """
        rewards = all_rewards if all_rewards is not None else jnp.zeros((5,), dtype=jnp.float32)
        return GravitarInfo(
            lives=state.lives,
            score=state.score,
            fuel=state.fuel,
            mode=state.mode,
            crash_timer=state.crash_timer,
            done=state.done,
            current_level=state.current_level,
            all_rewards=rewards,
        )

    # === Implement all required abstract methods ===
    def reset(self, key: jnp.ndarray) -> tuple[GravitarObservation, EnvState]:
        """Implements the main reset entry point of the environment."""
        return self.reset_map(key)

    def step(self, env_state: EnvState, action: int):
        """Implements the main step entry point of the environment."""
        obs, ns, reward, done, info, _reset, _level = step_full(env_state, action, self)
        return obs, ns, reward, done, info

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self) -> spaces.Dict: 
        screen_size = (WINDOW_HEIGHT, WINDOW_WIDTH)
        orientation_range = (-jnp.pi, jnp.pi)

        return spaces.Dict({
            'ship': spaces.get_object_space(n=None, screen_size=screen_size, orientation_range=orientation_range),
            'enemies': spaces.get_object_space(n=MAX_ENEMIES, screen_size=screen_size, orientation_range=orientation_range),
            'fuel_tanks': spaces.get_object_space(n=MAX_ENEMIES, screen_size=screen_size, orientation_range=orientation_range),
            'saucer': spaces.get_object_space(n=None, screen_size=screen_size, orientation_range=orientation_range),
            'ufo': spaces.get_object_space(n=None, screen_size=screen_size, orientation_range=orientation_range),
            'planets': spaces.get_object_space(n=_OBS_MAX_PLANETS, screen_size=screen_size, orientation_range=orientation_range),
            'projectiles': spaces.get_object_space(n=MAX_ENEMIES, screen_size=screen_size, orientation_range=orientation_range),
            'terrain': spaces.get_object_space(n=None, screen_size=screen_size, orientation_range=orientation_range),
            'reactor_destination': spaces.get_object_space(n=None, screen_size=screen_size, orientation_range=orientation_range),
            'lives': spaces.Box(low=0, high=MAX_LIVES, shape=(), dtype=jnp.int32),
            'fuel': spaces.Box(low=0.0, high=1000000.0, shape=(), dtype=jnp.float32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=jnp.uint8)

    def get_ram(self, state: EnvState) -> jnp.ndarray:
        return jnp.zeros(128, dtype=jnp.uint8)

    def get_ale_lives(self, state: EnvState) -> jnp.ndarray:
        return state.lives

    def render(self, env_state: EnvState) -> Tuple[jnp.ndarray]:
        """Renders the state using the pure JAX renderer."""
        frame = self.renderer.render(env_state)
        return frame

    # ===  Ensure all reset functions return JAX arrays ===
    def reset_map(self, key: jnp.ndarray,
                  lives: Optional[int] = None,
                  score: Optional[float] = None,
                  fuel: Optional[float] = None,
                  reactor_destroyed: Optional[jnp.ndarray] = None,
                  planets_cleared_mask: Optional[jnp.ndarray] = None
                  ) -> tuple[GravitarObservation, EnvState]:
        # ALE spawn location coordinates: (75, 131)
        spawn_x = jnp.array(76.0, dtype=jnp.float32)
        spawn_y = jnp.array(131.0, dtype=jnp.float32)

        # INITIAL SPEED
        ship_state = ShipState(
            x=spawn_x,
            y=spawn_y,
            vx=jnp.array(jnp.cos(-jnp.pi / 4) * 0.075, dtype=jnp.float32),
            vy=jnp.array(jnp.sin(-jnp.pi / 4) * 0.02, dtype=jnp.float32),
            angle=jnp.array(-jnp.pi / 2, dtype=jnp.float32),
            is_thrusting=jnp.array(False),
            rotation_cooldown=jnp.int32(0)
        )
        px_np, py_np, pr_np, pi_np = self.planets
        ids_np = [SPRITE_TO_LEVEL_ID.get(sprite_idx, -1) for sprite_idx in pi_np]
        final_reactor_destroyed = reactor_destroyed if reactor_destroyed is not None else jnp.array(False)
        final_cleared_mask = planets_cleared_mask if planets_cleared_mask is not None else jnp.zeros_like(
            self.planets[0], dtype=bool)

        env_state = EnvState(
            mode=jnp.int32(0), state=ship_state, bullets=create_empty_bullets_64(),
            cooldown=jnp.array(0, dtype=jnp.int32), enemies=create_empty_enemies(),
            fuel_tanks=FuelTanks(x=jnp.zeros(self.consts.MAX_ENEMIES), y=jnp.zeros(self.consts.MAX_ENEMIES), w=jnp.zeros(self.consts.MAX_ENEMIES),
                                 h=jnp.zeros(self.consts.MAX_ENEMIES), sprite_idx=jnp.full(self.consts.MAX_ENEMIES, -1),
                                 active=jnp.zeros(self.consts.MAX_ENEMIES, dtype=bool)),
            enemy_bullets=create_empty_bullets_16(), fire_cooldown=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.int32),
            key=key, key_alt=key, score=jnp.array(score if score is not None else 0.0, dtype=jnp.float32),
            done=jnp.array(False), lives=jnp.array(lives if lives is not None else self.consts.MAX_LIVES, dtype=jnp.int32),
            fuel=jnp.array(fuel if fuel is not None else self.consts.STARTING_FUEL, dtype=jnp.float32),
            shield_active=jnp.array(False),
            reactor_timer=jnp.int32(0),
            reactor_activated=jnp.array(False),
            crash_timer=jnp.int32(0), planets_px=jnp.array(px_np), planets_py=jnp.array(py_np),
            planets_pr=jnp.array(pr_np), planets_pi=jnp.array(pi_np), planets_id=jnp.array(ids_np),
            current_level=jnp.int32(-1), terrain_sprite_idx=jnp.int32(-1),
            terrain_mask=jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=jnp.int32),
            terrain_scale=jnp.array(1.0), 
            terrain_offset=jnp.array([0.0, 0.0]),
            terrain_bank=self.terrain_bank, 
            terrain_bank_idx=jnp.int32(0), 
            respawn_shift_x=jnp.float32(0.0),
            reactor_dest_active=jnp.array(False), 
            reactor_dest_x=jnp.float32(0.0), 
            reactor_dest_y=jnp.float32(0.0),
            reactor_dest_radius=jnp.float32(0.4), 
            mode_timer=jnp.int32(0), 
            saucer=make_default_saucer(),
            saucer_spawn_timer=jnp.int32(SAUCER_SPAWN_DELAY_FRAMES), 
            map_return_x=jnp.float32(0.0),
            map_return_y=jnp.float32(0.0),
            map_return_vx=jnp.float32(0.0),
            map_return_vy=jnp.float32(0.0),
            map_return_angle=jnp.float32(-jnp.pi / 2),
            ufo=make_empty_ufo(), ufo_spawn_timer=jnp.int32(0), 
            ufo_home_x=jnp.float32(0.0), 
            ufo_home_y=jnp.float32(0.0),
            ufo_bullets=create_empty_bullets_16(), 
            level_offset=jnp.array([0, 0], dtype=jnp.float32),
            reactor_destroyed=final_reactor_destroyed, 
            planets_cleared_mask=final_cleared_mask,
            exit_allowed=jnp.array(False),
            max_active_player_bullets_map=jnp.int32(self.consts.MAX_ACTIVE_PLAYER_BULLETS_MAP),
            max_active_player_bullets_level=jnp.int32(self.consts.MAX_ACTIVE_PLAYER_BULLETS_LEVEL),
            max_active_player_bullets_arena=jnp.int32(self.consts.MAX_ACTIVE_PLAYER_BULLETS_ARENA),
            max_active_saucer_bullets=jnp.int32(self.consts.MAX_ACTIVE_SAUCER_BULLETS),
            max_active_enemy_bullets=jnp.int32(self.consts.MAX_ACTIVE_ENEMY_BULLETS),
            enemy_fire_cooldown_frames=jnp.int32(self.consts.ENEMY_FIRE_COOLDOWN_FRAMES),
            solar_gravity=jnp.float32(self.consts.SOLAR_GRAVITY),
            planetary_gravity=jnp.float32(self.consts.PLANETARY_GRAVITY),
            reactor_gravity=jnp.float32(self.consts.REACTOR_GRAVITY),
            thrust_power=jnp.float32(self.consts.THRUST_POWER),
            max_speed=jnp.float32(self.consts.MAX_SPEED),
            fuel_consume_thrust=jnp.float32(self.consts.FUEL_CONSUME_THRUST),
            fuel_consume_shield_tractor=jnp.float32(self.consts.FUEL_CONSUME_SHIELD_TRACTOR),
            allow_tractor_in_reactor=jnp.array(self.consts.ALLOW_TRACTOR_IN_REACTOR),
            enemy_kill_score=jnp.float32(self.consts.ENEMY_KILL_SCORE),
            level_clear_score=jnp.float32(self.consts.LEVEL_CLEAR_SCORE),
            ufo_kill_score=jnp.float32(self.consts.UFO_KILL_SCORE),
            saucer_kill_score=jnp.float32(self.consts.SAUCER_KILL_SCORE),
            prev_action=jnp.int32(0),
        )

        return self._get_observation(env_state), env_state

    def reset_level(self, key: jnp.ndarray, level_id: jnp.ndarray, prev_env_state: EnvState):
        level_id = jnp.asarray(level_id, dtype=jnp.int32)
        level_offset = self.jax_level_offsets[level_id]
        terrain_sprite_idx = self.jax_level_to_terrain[level_id]
        bank_idx = self.jax_level_to_bank[level_id]
        transform = self.jax_level_transforms[level_id]
        scale, ox, oy = transform[0], transform[1], transform[2]

        def loop_body(i, carry):
            enemies, tanks, e_idx, t_idx = carry
            obj_type = self.jax_layout["types"][level_id, i]

            def place_obj(val):
                enemies_in, tanks_in, e_idx_in, t_idx_in = val
                orig_idx = jnp.where(obj_type == SpriteIdx.ENEMY_ORANGE_FLIPPED, SpriteIdx.ENEMY_ORANGE, obj_type)
                w, h = self.jax_sprite_dims[orig_idx]
                tank_w, tank_h = self.jax_sprite_dims[int(SpriteIdx.FUEL_TANK)]
                
                # For terrain2 (level 1), coordinates are designed for 160-width but sprite is 96-width
                # Scale coordinates to account for this: multiply by (96/160) for x coords
                terrain_sprite = self.jax_level_to_terrain[level_id]
                is_terrain2 = (terrain_sprite == int(SpriteIdx.TERRAIN2))
                coord_x = self.jax_layout["coords_x"][level_id, i]
                coord_y = self.jax_layout["coords_y"][level_id, i]
                # Adjust x coordinate for terrain2's narrower width
                adjusted_coord_x = jnp.where(is_terrain2, coord_x * 0.6, coord_x)  # 96/160 = 0.6
                
                x = ox + coord_x * scale
                y = oy + coord_y * scale
                is_tank = (obj_type == SpriteIdx.FUEL_TANK).astype(jnp.int32)
                is_green_enemy = (obj_type == SpriteIdx.ENEMY_GREEN).astype(jnp.int32)
                spawn_tank = jnp.maximum(is_tank, is_green_enemy)
                spawned_tank_active = is_tank  # explicit tank active, green-linked tank hidden

                new_enemies = enemies_in._replace(x=enemies_in.x.at[e_idx_in].set(jnp.where(is_tank, -1.0, x)),
                                                  y=enemies_in.y.at[e_idx_in].set(jnp.where(is_tank, -1.0, y)),
                                                  w=enemies_in.w.at[e_idx_in].set(jnp.where(is_tank, 0.0, w)),
                                                  h=enemies_in.h.at[e_idx_in].set(jnp.where(is_tank, 0.0, h)),
                                                  sprite_idx=enemies_in.sprite_idx.at[e_idx_in].set(
                                                      jnp.where(is_tank, -1, obj_type)),
                                                  hp=enemies_in.hp.at[e_idx_in].set(jnp.where(is_tank, 0, 1)), )
                new_tanks = tanks_in._replace(x=tanks_in.x.at[t_idx_in].set(jnp.where(spawn_tank, x, -1.0)),
                                              y=tanks_in.y.at[t_idx_in].set(jnp.where(spawn_tank, y, -1.0)),
                                              w=tanks_in.w.at[t_idx_in].set(jnp.where(spawn_tank, tank_w, 0.0)),
                                              h=tanks_in.h.at[t_idx_in].set(jnp.where(spawn_tank, tank_h, 0.0)),
                                              sprite_idx=tanks_in.sprite_idx.at[t_idx_in].set(
                                                  jnp.where(spawn_tank, int(SpriteIdx.FUEL_TANK), -1)),
                                              active=tanks_in.active.at[t_idx_in].set(
                                                  jnp.where(spawn_tank, spawned_tank_active.astype(bool), False)), )
                return new_enemies, new_tanks, e_idx_in + (1 - is_tank), t_idx_in + spawn_tank

            return jax.lax.cond(obj_type != -1, place_obj, lambda x: x, (enemies, tanks, e_idx, t_idx))

        init_enemies = create_empty_enemies()
        init_tanks = FuelTanks(x=jnp.full((MAX_ENEMIES,), -1.0), y=jnp.full((MAX_ENEMIES,), -1.0),
                               w=jnp.zeros((MAX_ENEMIES,)), h=jnp.zeros((MAX_ENEMIES,)),
                               sprite_idx=jnp.full((MAX_ENEMIES,), -1), active=jnp.zeros((MAX_ENEMIES,), dtype=bool))
        enemies, fuel_tanks, _, _ = jax.lax.fori_loop(0, self.jax_layout["types"].shape[1], loop_body,
                                                      (init_enemies, init_tanks, 0, 0))

        ship_state = make_level_start_state(level_id)
        initial_timer = jnp.where(level_id == 4, 60 * 60, 0)

        env_state = prev_env_state._replace(
            mode=jnp.int32(1), state=ship_state,
            bullets=create_empty_bullets_64(), 
            cooldown=jnp.int32(0),
            enemies=enemies,
            fuel_tanks=fuel_tanks,
            shield_active=jnp.array(False),
            enemy_bullets=create_empty_bullets_16(),
            fire_cooldown=jnp.full((MAX_ENEMIES,), 60, dtype=jnp.int32),
            key=key, crash_timer=jnp.int32(0), current_level=level_id,
            done=jnp.array(False),
            terrain_sprite_idx=terrain_sprite_idx,
            terrain_mask=jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=jnp.int32),
            terrain_scale=scale,
            terrain_offset=jnp.array([ox, oy]),
            terrain_bank_idx=bank_idx,
            reactor_dest_active=(level_id == 4),
            reactor_dest_x=jnp.float32(96.0),
            reactor_dest_y=jnp.float32(125.0),
            mode_timer=jnp.int32(0), ufo=make_empty_ufo(),
            ufo_spawn_timer=jnp.int32(UFO_RESPAWN_DELAY_FRAMES),
            level_offset=jnp.array(level_offset, dtype=jnp.float32),
            reactor_timer=initial_timer.astype(jnp.int32),
            reactor_activated=jnp.array(False),
            exit_allowed=(level_id == 4),  # Allow exit from start in reactor level
        )

        return self._get_observation(env_state), env_state

    # --- Helper Methods ---
    def _build_terrain_bank(self) -> jnp.ndarray:
        W, H = WINDOW_WIDTH, WINDOW_HEIGHT
        bank = [np.full((H, W), self.renderer.jr.TRANSPARENT_ID, dtype=np.int32)]
        BANK_IDX_TO_LEVEL_ID = {v: k for k, v in LEVEL_ID_TO_BANK_IDX.items()}

        def sprite_to_mask(idx: int, bank_idx: int) -> np.ndarray:
            surf = self.sprites[SpriteIdx(idx)]
            th, tw = surf.shape[0], surf.shape[1]
            
            # Calculate scale with overrides
            scale = min(W / tw, H / th)
            extra = TERRAIN_SCALE_OVERRIDES.get(SpriteIdx(idx), 1.0)
            scale *= float(extra)
            
            # Calculate scaled dimensions
            sw, sh = int(tw * scale), int(th * scale)
            
            # For terrain2 (narrow sprite), we need to center it within the full window width
            # The offset should place the scaled sprite in the center
            ox, oy = (W - sw) // 2, (H - sh) // 2
            
            # Apply level-specific offset after centering
            level_id = BANK_IDX_TO_LEVEL_ID.get(bank_idx)
            if level_id is not None:
                level_offset = LEVEL_OFFSETS.get(level_id, (0, 0))
                ox += level_offset[0]
                oy += level_offset[1]

            # Scale using NumPy operations
            if surf.shape[0] != sh or surf.shape[1] != sw:
                scale_h = max(1, int(round(sh / surf.shape[0])))
                scale_w = max(1, int(round(sw / surf.shape[1])))
                rgba_array = np.repeat(np.repeat(surf, scale_h, axis=0), scale_w, axis=1)[:sh, :sw]
            else:
                rgba_array = surf

            id_mask = np.array(self.renderer.jr._create_id_mask(rgba_array, self.renderer.COLOR_TO_ID))

            color_map = np.full((H, W), self.renderer.jr.TRANSPARENT_ID, dtype=np.int32)
            src_w, src_h = id_mask.shape[1], id_mask.shape[0]
            dst_x, dst_y = max(ox, 0), max(oy, 0)
            src_x = abs(min(ox, 0))
            src_y = abs(min(oy, 0))
            copy_w = min(W - dst_x, src_w - src_x)
            copy_h = min(H - dst_y, src_h - src_y)
            if copy_w > 0 and copy_h > 0:
                color_map[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = id_mask[
                    src_y:src_y + copy_h, src_x:src_x + copy_w]
            return color_map

        terrains_to_build = [
            (SpriteIdx.TERRAIN1, 1), (SpriteIdx.TERRAIN2, 2), (SpriteIdx.TERRAIN3, 3),
            (SpriteIdx.TERRAIN4, 4), (SpriteIdx.REACTOR_TERR, 5),
        ]
        for sprite_idx, bank_idx in terrains_to_build:
            bank.append(sprite_to_mask(int(sprite_idx), bank_idx))
        return jnp.array(np.stack(bank, axis=0), dtype=jnp.int32)


class GravitarRenderer(JAXGameRenderer):
    def __init__(self, width: int = None, height: int = None, consts: GravitarConstants = None,
                 config: render_utils.RendererConfig = None):
        super().__init__()
        self.consts = consts or GravitarConstants()
        self.width = width if width is not None else self.consts.WINDOW_WIDTH
        self.height = height if height is not None else self.consts.WINDOW_HEIGHT
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.height, self.width),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        jax_sprites = _load_and_convert_sprites()
        
        # Convert sprites dict to tuple for JIT-compatible indexing
        max_sprite_idx = max(jax_sprites.keys()) if jax_sprites else -1
        sprites_list = []
        for i in range(max_sprite_idx + 1):
            sprite = jax_sprites.get(i)
            if sprite is not None:
                sprites_list.append(sprite)
            else:
                # Create a placeholder 1x1 transparent sprite for missing indices
                sprites_list.append(jnp.zeros((1, 1, 4), dtype=jnp.uint8))

        ui_colors = jnp.array([[[255, 255, 0, 255], [50, 50, 50, 255], [0, 0, 0, 255]]], dtype=jnp.uint8)
        all_assets = sprites_list + [ui_colors]

        self.PALETTE, self.COLOR_TO_ID = self.jr._create_palette(all_assets)

        if self.COLOR_TO_ID:
            self.jr.TRANSPARENT_ID = max(self.COLOR_TO_ID.values()) + 1
        else:
            self.jr.TRANSPARENT_ID = 255

        palette_size = self.PALETTE.shape[0]
        required_size = self.jr.TRANSPARENT_ID + 1
        if palette_size < required_size:
            if self.config.channels == 1:
                padding = jnp.zeros((required_size - palette_size, 1), dtype=self.PALETTE.dtype)
            else:
                padding = jnp.zeros((required_size - palette_size, 3), dtype=self.PALETTE.dtype)
            self.PALETTE = jnp.concatenate([self.PALETTE, padding], axis=0)

        self.color_yellow = self.COLOR_TO_ID.get((255, 255, 0), 0)
        self.color_gray = self.COLOR_TO_ID.get((50, 50, 50), 0)

        masks_list = []
        for s in sprites_list:
            masks_list.append(self.jr._create_id_mask(s, self.COLOR_TO_ID))
        self.sprites = tuple(masks_list)


        self.digit_masks = jnp.stack([self.sprites[int(SpriteIdx.DIGIT_0) + i] for i in range(10)])

        idle_sprite = jax_sprites.get(int(SpriteIdx.SHIP_IDLE))
        crash_sprite = jax_sprites.get(int(SpriteIdx.SHIP_CRASH))
        thrust_sprite = jax_sprites.get(int(SpriteIdx.SHIP_THRUST))

        orientation_sprites = [
            jax_sprites.get(int(SpriteIdx.SHIP_N)), jax_sprites.get(int(SpriteIdx.SHIP_NNE)),
            jax_sprites.get(int(SpriteIdx.SHIP_NE)), jax_sprites.get(int(SpriteIdx.SHIP_NEE)),
            jax_sprites.get(int(SpriteIdx.SHIP_E)), jax_sprites.get(int(SpriteIdx.SHIP_SEE)),
            jax_sprites.get(int(SpriteIdx.SHIP_SE)), jax_sprites.get(int(SpriteIdx.SHIP_SSE)),
            jax_sprites.get(int(SpriteIdx.SHIP_S)), jax_sprites.get(int(SpriteIdx.SHIP_SSW)),
            jax_sprites.get(int(SpriteIdx.SHIP_SW)), jax_sprites.get(int(SpriteIdx.SHIP_SWW)),
            jax_sprites.get(int(SpriteIdx.SHIP_W)), jax_sprites.get(int(SpriteIdx.SHIP_NWW)),
            jax_sprites.get(int(SpriteIdx.SHIP_NW)), jax_sprites.get(int(SpriteIdx.SHIP_NNW)),
        ]

        all_ship_sprites = [idle_sprite, crash_sprite, thrust_sprite] + orientation_sprites
        valid_ship_sprites = [s for s in all_ship_sprites if s is not None]

        if valid_ship_sprites:
            max_h = max(s.shape[0] for s in valid_ship_sprites)
            max_w = max(s.shape[1] for s in valid_ship_sprites)

            def pad_mask(sprite_rgba, h, w):
                if sprite_rgba is None:
                    return jnp.full((h, w), self.jr.TRANSPARENT_ID, dtype=jnp.int32)
                mask = self.jr._create_id_mask(sprite_rgba, self.COLOR_TO_ID)
                pad_h = (h - mask.shape[0]) // 2
                pad_w = (w - mask.shape[1]) // 2
                return jnp.pad(mask,
                               ((pad_h, h - mask.shape[0] - pad_h), (pad_w, w - mask.shape[1] - pad_w)),
                               constant_values=self.jr.TRANSPARENT_ID)

            self.padded_ship_idle = pad_mask(idle_sprite, max_h, max_w)
            self.padded_ship_crash = pad_mask(crash_sprite, max_h, max_w)
            self.padded_ship_thrust = pad_mask(thrust_sprite, max_h, max_w)
            self.padded_ship_orientations = tuple(pad_mask(sprite, max_h, max_w) for sprite in orientation_sprites)
            self.ship_orientations_array = jnp.stack(self.padded_ship_orientations)
        else:
            default_mask = jnp.full((1, 1), self.jr.TRANSPARENT_ID, dtype=jnp.int32)
            self.padded_ship_idle = default_mask
            self.padded_ship_crash = default_mask
            self.padded_ship_thrust = default_mask
            self.padded_ship_orientations = tuple(default_mask for _ in range(16))
            self.ship_orientations_array = jnp.stack(self.padded_ship_orientations)

    @partial(jax.jit, static_argnames=('self',))
    def render(self, state: EnvState) -> jnp.ndarray:
        H, W = self.height, self.width
        frame = jnp.full((H, W), self.jr.TRANSPARENT_ID, dtype=jnp.int32)

        def render_centered(f_in, x, y, sprite_arr):
            h, w = sprite_arr.shape[0], sprite_arr.shape[1]
            cx = jnp.round(x - w / 2).astype(jnp.int32)
            cy = jnp.round(y - h / 2).astype(jnp.int32)
            return self.jr.render_at(f_in, cx, cy, sprite_arr)

        # === 1. Draw Map Elements ===
        def draw_map_elements(f):
            def draw_one_planet(i, fc):
                is_cleared = (i < state.planets_cleared_mask.shape[0]) & state.planets_cleared_mask[i]
                pid = state.planets_pi[i]
                is_reactor_destroyed = (pid == int(SpriteIdx.REACTOR)) & state.reactor_destroyed
                should_draw = (pid >= 0) & ~(is_cleared | is_reactor_destroyed)

                def draw_p(r):
                    r = jax.lax.cond(pid == int(SpriteIdx.PLANET1),
                                     lambda rr: render_centered(rr, state.planets_px[i], state.planets_py[i],
                                                                self.sprites[int(SpriteIdx.PLANET1)]), lambda rr: rr, r)
                    r = jax.lax.cond(pid == int(SpriteIdx.PLANET2),
                                     lambda rr: render_centered(rr, state.planets_px[i], state.planets_py[i],
                                                                self.sprites[int(SpriteIdx.PLANET2)]), lambda rr: rr, r)
                    r = jax.lax.cond(pid == int(SpriteIdx.PLANET3),
                                     lambda rr: render_centered(rr, state.planets_px[i], state.planets_py[i],
                                                                self.sprites[int(SpriteIdx.PLANET3)]), lambda rr: rr, r)
                    r = jax.lax.cond(pid == int(SpriteIdx.PLANET4),
                                     lambda rr: render_centered(rr, state.planets_px[i], state.planets_py[i],
                                                                self.sprites[int(SpriteIdx.PLANET4)]), lambda rr: rr, r)
                    r = jax.lax.cond(pid == int(SpriteIdx.REACTOR),
                                     lambda rr: render_centered(rr, state.planets_px[i], state.planets_py[i],
                                                                self.sprites[int(SpriteIdx.REACTOR)]), lambda rr: rr, r)
                    r = jax.lax.cond(pid == int(SpriteIdx.OBSTACLE),
                                     lambda rr: render_centered(rr, state.planets_px[i], state.planets_py[i],
                                                                self.sprites[int(SpriteIdx.OBSTACLE)]), lambda rr: rr,
                                     r)
                    r = jax.lax.cond(pid == int(SpriteIdx.SPAWN_LOC),
                                     lambda rr: render_centered(rr, state.planets_px[i], state.planets_py[i],
                                                                self.sprites[int(SpriteIdx.SPAWN_LOC)]), lambda rr: rr,
                                     r)
                    return r

                return jax.lax.cond(should_draw, draw_p, lambda r: r, fc)

            return jax.lax.fori_loop(0, state.planets_pi.shape[0], draw_one_planet, f)

        frame = jax.lax.cond(state.mode == 0, draw_map_elements, lambda f: f, frame)

        # === 2. Draw Terrain (only in Level Modus) ===
        def draw_level_terrain(f):
            bank_idx = jnp.clip(state.terrain_bank_idx, 0, state.terrain_bank.shape[0] - 1)
            terrain_map = state.terrain_bank[bank_idx]
            is_terrain_pixel = terrain_map != self.jr.TRANSPARENT_ID
            return jnp.where(is_terrain_pixel, terrain_map, f)

        frame = jax.lax.cond(state.mode == 1, draw_level_terrain, lambda f: f, frame)

        # === 3. Draw Level Actors (Enemies & Tanks) ===
        def draw_level_actors(f):
            # Fuel Tanks
            sprite_arr_tank = self.sprites[int(SpriteIdx.FUEL_TANK)]

            def draw_one_tank(i, fc):
                return jax.lax.cond(
                    state.fuel_tanks.active[i],
                    lambda r: render_centered(r, state.fuel_tanks.x[i], state.fuel_tanks.y[i], sprite_arr_tank),
                    lambda r: r, fc
                )

            f_tanks = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES, draw_one_tank, f)

            def draw_one_enemy(i, fc):
                w = state.enemies.w[i]
                death_timer = state.enemies.death_timer[i]
                sprite_id = state.enemies.sprite_idx[i]

                is_alive = w > 0
                is_exploding = death_timer > 0
                is_active = is_alive & ~is_exploding

                fc = jax.lax.cond(is_active & (sprite_id == int(SpriteIdx.ENEMY_ORANGE)),
                                  lambda r: render_centered(r, state.enemies.x[i], state.enemies.y[i],
                                                            self.sprites[int(SpriteIdx.ENEMY_ORANGE)]),
                                  lambda r: r, fc)
                fc = jax.lax.cond(is_active & (sprite_id == int(SpriteIdx.ENEMY_GREEN)),
                                  lambda r: render_centered(r, state.enemies.x[i], state.enemies.y[i],
                                                            self.sprites[int(SpriteIdx.ENEMY_GREEN)]),
                                  lambda r: r, fc)
                fc = jax.lax.cond(is_active & (sprite_id == int(SpriteIdx.ENEMY_ORANGE_FLIPPED)),
                                  lambda r: render_centered(r, state.enemies.x[i], state.enemies.y[i],
                                                            self.sprites[int(SpriteIdx.ENEMY_ORANGE_FLIPPED)]),
                                  lambda r: r, fc)
                fc = jax.lax.cond(is_exploding,
                                  lambda r: render_centered(r, state.enemies.x[i], state.enemies.y[i],
                                                            self.sprites[int(SpriteIdx.ENEMY_CRASH)]),
                                  lambda r: r, fc)
                return fc

            f_enemies = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES, draw_one_enemy, f_tanks)

            # UFO
            ufo = state.ufo
            f_ufo = jax.lax.cond(ufo.alive,
                                 lambda r: render_centered(r, ufo.x, ufo.y, self.sprites[int(SpriteIdx.ENEMY_UFO)]),
                                 lambda r: r, f_enemies)
            f_ufo = jax.lax.cond(ufo.death_timer > 0,
                                 lambda r: render_centered(r, ufo.x, ufo.y, self.sprites[int(SpriteIdx.ENEMY_CRASH)]),
                                 lambda r: r, f_ufo)
            return f_ufo

        frame = jax.lax.cond(state.mode == 1, draw_level_actors, lambda f: f, frame)

        # === 3.5. Draw Saucer and Reactor Destination ===
        def draw_saucer(f):
            saucer = state.saucer
            f = jax.lax.cond(saucer.alive,
                             lambda r: render_centered(r, saucer.x, saucer.y,
                                                       self.sprites[int(SpriteIdx.ENEMY_SAUCER)]),
                             lambda r: r, f)
            f = jax.lax.cond(saucer.death_timer > 0,
                             lambda r: render_centered(r, saucer.x, saucer.y,
                                                       self.sprites[int(SpriteIdx.SAUCER_CRASH)]),
                             lambda r: r, f)
            return f

        frame = jax.lax.cond((state.mode == 0) | (state.mode == 2), draw_saucer, lambda f: f, frame)

        def draw_reactor_destination(f):
            sprite_arr = jax.lax.select(
                state.reactor_activated,
                self.sprites[int(SpriteIdx.REACTOR_DEST_HIT)],
                self.sprites[int(SpriteIdx.REACTOR_DEST)]
            )
            return render_centered(f, state.reactor_dest_x, state.reactor_dest_y, sprite_arr)

        should_draw_destination = (state.mode == 1) & (
                state.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)) & state.reactor_dest_active
        frame = jax.lax.cond(should_draw_destination, draw_reactor_destination, lambda f: f, frame)

        # === 4. Bullets ===
        def draw_player_bullets(f):
            sprite_arr = self.sprites[int(SpriteIdx.SHIP_BULLET)]

            def draw_one(i, fc):
                return jax.lax.cond(state.bullets.alive[i],
                                    lambda r: render_centered(r, state.bullets.x[i], state.bullets.y[i], sprite_arr),
                                    lambda r: r, fc)

            return jax.lax.fori_loop(0, state.bullets.x.shape[0], draw_one, f)

        def draw_enemy_bullets(f):
            def draw_one(i, fc):
                active = state.enemy_bullets.alive[i]
                sprite_id = state.enemy_bullets.sprite_idx[i]

                fc = jax.lax.cond(active & (sprite_id == int(SpriteIdx.ENEMY_BULLET)),
                                  lambda r: render_centered(r, state.enemy_bullets.x[i], state.enemy_bullets.y[i],
                                                            self.sprites[int(SpriteIdx.ENEMY_BULLET)]),
                                  lambda r: r, fc)
                fc = jax.lax.cond(active & (sprite_id == int(SpriteIdx.ENEMY_GREEN_BULLET)),
                                  lambda r: render_centered(r, state.enemy_bullets.x[i], state.enemy_bullets.y[i],
                                                            self.sprites[int(SpriteIdx.ENEMY_GREEN_BULLET)]),
                                  lambda r: r, fc)
                return fc

            return jax.lax.fori_loop(0, state.enemy_bullets.x.shape[0], draw_one, f)

        frame = draw_player_bullets(frame)
        frame = draw_enemy_bullets(frame)

        # === 5. Draw the ship ===
        ship_state = state.state
        is_crashing = state.crash_timer > 0
        is_thrusting = ship_state.is_thrusting

        angle_diffs = jnp.abs(jnp.arctan2(
            jnp.sin(ship_state.angle - self.consts.SHIP_ANGLES),
            jnp.cos(ship_state.angle - self.consts.SHIP_ANGLES)
        ))
        angle_idx = jnp.argmin(angle_diffs)

        oriented_ship_mask = self.ship_orientations_array[angle_idx]
        ship_sprite = jax.lax.select(is_crashing, self.padded_ship_crash, oriented_ship_mask)
        frame = render_centered(frame, ship_state.x, ship_state.y, ship_sprite)

        def draw_thrust_flame(f):
            THRUST_OFFSET = 5.0
            thrust_x = ship_state.x - jnp.cos(ship_state.angle) * THRUST_OFFSET
            thrust_y = ship_state.y - jnp.sin(ship_state.angle) * THRUST_OFFSET
            return render_centered(f, thrust_x, thrust_y, self.padded_ship_thrust)

        frame = jax.lax.cond(is_thrusting & (~is_crashing), draw_thrust_flame, lambda f: f, frame)

        def draw_shield_and_tractor(f):
            f_with_shield = render_centered(f, ship_state.x, ship_state.y, self.sprites[int(SpriteIdx.SHIELD)])

            is_planet_level = state.mode == 1
            is_reactor = state.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)
            can_show_tractor = is_planet_level & ((~is_reactor) | state.allow_tractor_in_reactor)

            def draw_tractor(frame_in):
                TRACTOR_OFFSET = 8.0
                tractor_x = ship_state.x - jnp.cos(ship_state.angle) * TRACTOR_OFFSET
                tractor_y = ship_state.y - jnp.sin(ship_state.angle) * TRACTOR_OFFSET
                return render_centered(frame_in, tractor_x, tractor_y, self.sprites[int(SpriteIdx.SHIP_THRUST_BACK)])

            return jax.lax.cond(can_show_tractor, draw_tractor, lambda frame_in: frame_in, f_with_shield)

        frame = jax.lax.cond(state.shield_active, draw_shield_and_tractor, lambda f: f, frame)

        # === 6. Draw the HUD ===
        def draw_hud(f):
            W, H = self.width, self.height

            # Fuel
            f = self.jr.render_label(
                f, x=10, y=5,
                digits=self.jr.int_to_digits(state.fuel.astype(jnp.int32), max_digits=5),
                digit_masks=self.digit_masks, spacing=8, max_digits=5
            )
            
            # Score
            f = self.jr.render_label(
                f, x=W - 55, y=5,
                digits=self.jr.int_to_digits(state.score.astype(jnp.int32), max_digits=6),
                digit_masks=self.digit_masks, spacing=8, max_digits=6
            )
            
            # Lives
            f = self.jr.render_indicator(
                f, x=W - 50, y=17,
                value=state.lives, shape_mask=self.sprites[int(SpriteIdx.HP_UI)],
                spacing=8, max_value=self.consts.MAX_LIVES
            )

            # Reactor Timer
            def draw_reactor_timer(frame_carry):
                seconds_left = state.reactor_timer // 60
                return self.jr.render_label(
                    frame_carry, x=W // 2 - 8, y=5,
                    digits=self.jr.int_to_digits(seconds_left, max_digits=2),
                    digit_masks=self.digit_masks, spacing=8, max_digits=2
                )

            is_in_reactor = (state.mode == 1) & (state.current_level == 4)
            return jax.lax.cond(is_in_reactor, draw_reactor_timer, lambda fc: fc, f)

        frame = draw_hud(frame)

        frame_rgb = self.jr.render_from_palette(frame, self.PALETTE)
        return frame_rgb

__all__ = ["JaxGravitar", "get_env_and_renderer"]


def get_env_and_renderer():
    env = JaxGravitar()
    # Just instantiate it, or pass in your game resolution as parameters
    renderer = GravitarRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    return env, renderer

