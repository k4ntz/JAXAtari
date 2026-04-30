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
from typing import Optional, Tuple
from enum import IntEnum
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

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
    THRUST_POWER: float = struct.field(pytree_node=False, default=0.030)
    MAX_SPEED: float = struct.field(pytree_node=False, default=2.5)
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

    # Optional full sprite table (same layout as load_sprites_tuple()). None = bundled defaults.
    SPRITES_TUPLE: Optional[tuple] = struct.field(pytree_node=False, default=None)


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

# Specialized precomputed mask for ship collision (radius=2, 5x5 patch).
# Using a 5x5 window instead of the full 33x33 cuts dynamic_slice cost by ~43x.
_TERRAIN_HIT_SHIP_R = 2
_TERRAIN_HIT_SHIP_SIZE = 2 * _TERRAIN_HIT_SHIP_R + 1  # 5
_dy_ship = jnp.arange(-_TERRAIN_HIT_SHIP_R, _TERRAIN_HIT_SHIP_R + 1, dtype=jnp.int32)
_dx_ship = jnp.arange(-_TERRAIN_HIT_SHIP_R, _TERRAIN_HIT_SHIP_R + 1, dtype=jnp.int32)
_TERRAIN_HIT_MASK_SHIP = (_dy_ship[:, None] ** 2 + _dx_ship[None, :] ** 2) <= (_TERRAIN_HIT_SHIP_R ** 2)

# Precomputed trigonometry to avoid jnp.arctan2 in hot loops
_SHIP_ANGLES_COS = jnp.cos(_DEFAULT_CONSTS.SHIP_ANGLES)
_SHIP_ANGLES_SIN = jnp.sin(_DEFAULT_CONSTS.SHIP_ANGLES)

# --- Precomputed per-action lookup tables ---
# Replaces all jnp.isin(action, constant_array) calls with O(1) table lookups.
# Actions: NOOP=0 FIRE=1 UP=2 RIGHT=3 LEFT=4 DOWN=5 UPRIGHT=6 UPLEFT=7
#          DOWNRIGHT=8 DOWNLEFT=9 UPFIRE=10 RIGHTFIRE=11 LEFTFIRE=12 DOWNFIRE=13
#          UPRIGHTFIRE=14 UPLEFTFIRE=15 DOWNRIGHTFIRE=16 DOWNLEFTFIRE=17
_N_ACTIONS = 18
_THRUST_SET          = {2, 6, 7, 10, 14, 15}
_ROTATE_RIGHT_SET    = {3, 6, 8, 11, 14, 16}
_ROTATE_LEFT_SET     = {4, 7, 9, 12, 15, 17}
_FIRE_SET            = {1, 10, 11, 12, 13, 14, 15, 16, 17}
_SHIELD_TRACTOR_SET  = {5, 8, 9, 13, 16, 17}

_ACTION_IS_THRUST        = jnp.array([a in _THRUST_SET         for a in range(_N_ACTIONS)], dtype=jnp.bool_)
_ACTION_IS_ROTATE_RIGHT  = jnp.array([a in _ROTATE_RIGHT_SET   for a in range(_N_ACTIONS)], dtype=jnp.bool_)
_ACTION_IS_ROTATE_LEFT   = jnp.array([a in _ROTATE_LEFT_SET    for a in range(_N_ACTIONS)], dtype=jnp.bool_)
_ACTION_IS_FIRE          = jnp.array([a in _FIRE_SET           for a in range(_N_ACTIONS)], dtype=jnp.bool_)
_ACTION_IS_SHIELD        = jnp.array([a in _SHIELD_TRACTOR_SET for a in range(_N_ACTIONS)], dtype=jnp.bool_)


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
    # All coords are global screen pixels (x=0 left edge, y=0 top of screen).
    # Level 0 (Planet 1)
    0: [
        {'type': SpriteIdx.ENEMY_ORANGE,         'coords': (37,  156)},
        {'type': SpriteIdx.ENEMY_ORANGE,         'coords': (82,  144)},
        {'type': SpriteIdx.ENEMY_ORANGE,         'coords': (152, 110)},
        {'type': SpriteIdx.ENEMY_GREEN,          'coords': (22,  184)},
        {'type': SpriteIdx.FUEL_TANK,            'coords': (104, 172)},
    ],
    # Level 1 (Planet 2)
    1: [
        {'type': SpriteIdx.ENEMY_ORANGE,         'coords': (124,  92)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (83,  152)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (40,  112)},
        {'type': SpriteIdx.ENEMY_GREEN,          'coords': (44, 132)},
        {'type': SpriteIdx.FUEL_TANK,            'coords': (61,  72)},
    ],
    # Level 2 (Planet 3)
    2: [
        {'type': SpriteIdx.ENEMY_ORANGE,         'coords': (24,  140)},
        {'type': SpriteIdx.ENEMY_GREEN,          'coords': (43,  184)},
        {'type': SpriteIdx.ENEMY_ORANGE,         'coords': (60,  100)},
        {'type': SpriteIdx.ENEMY_GREEN,          'coords': (108, 124)},
        {'type': SpriteIdx.FUEL_TANK,            'coords': (135, 170)},
    ],
    # Level 3 (Planet 4)
    3: [
        {'type': SpriteIdx.ENEMY_ORANGE,         'coords': (88,   92)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (116,  76)},
        {'type': SpriteIdx.ENEMY_ORANGE,         'coords': (122, 178)},
        {'type': SpriteIdx.ENEMY_GREEN,          'coords': (76,  124)},
        {'type': SpriteIdx.FUEL_TANK,            'coords': (19,  160)},
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
    angle_idx: jnp.ndarray

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
    player_bullet_idx: jnp.ndarray
    cooldown: jnp.ndarray
    enemies: Enemies
    fuel_tanks: FuelTanks
    shield_active: jnp.ndarray
    enemy_bullets: Bullets
    enemy_bullet_idx: jnp.ndarray
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
    terrain_scale: jnp.ndarray  # float32, rendering scale factor
    terrain_offset: jnp.ndarray  # (2,) float32, screen-top-left offset [ox, oy]

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
    map_return_angle_idx: jnp.ndarray
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


def _sprite_wh_scalar(
    sprite_dims: jnp.ndarray, sprite_idx: jnp.ndarray, fallback_w: int = 0, fallback_h: int = 0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = sprite_idx.astype(jnp.int32)
    max_idx = sprite_dims.shape[0] - 1
    safe_idx = jnp.clip(idx, 0, max_idx)
    wh = sprite_dims[safe_idx]
    valid = (idx >= 0) & (idx <= max_idx)
    w = jnp.where(valid, wh[0], jnp.array(fallback_w, dtype=jnp.int16))
    h = jnp.where(valid, wh[1], jnp.array(fallback_h, dtype=jnp.int16))
    return w, h


def _sprite_wh_vector(
    sprite_dims: jnp.ndarray, sprite_idx: jnp.ndarray, fallback_w: int = 0, fallback_h: int = 0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = sprite_idx.astype(jnp.int32)
    max_idx = sprite_dims.shape[0] - 1
    safe_idx = jnp.clip(idx, 0, max_idx)
    wh = sprite_dims[safe_idx]
    valid = (idx >= 0) & (idx <= max_idx)
    w = jnp.where(valid, wh[:, 0], jnp.full(idx.shape, fallback_w, dtype=jnp.int16))
    h = jnp.where(valid, wh[:, 1], jnp.full(idx.shape, fallback_h, dtype=jnp.int16))
    return w, h


@jax.jit
def _get_observation_from_state(state: EnvState, sprite_dims: jnp.ndarray) -> GravitarObservation:
    ship: ShipState = state.state
    enemies: Enemies = state.enemies
    fuel_tanks: FuelTanks = state.fuel_tanks
    saucer: SaucerState = state.saucer
    ufo: UFOState = state.ufo
    enemy_bullets: Bullets = state.enemy_bullets

    # --- Ship ---
    sx, sy = _clip_xy_to_screen(ship.x, ship.y)
    ship_active = jnp.array(1, dtype=jnp.int32)
    ship_visual_id = SHIP_SPRITE_INDICES[ship.angle_idx].astype(jnp.int16)
    ship_orientation = ship.angle.astype(jnp.float32)
    ship_w, ship_h = _sprite_wh_scalar(sprite_dims, ship_visual_id, fallback_w=3, fallback_h=7)

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
    saucer_w, saucer_h = _sprite_wh_scalar(sprite_dims, saucer_visual_id, fallback_w=8, fallback_h=7)
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
    ufo_w, ufo_h = _sprite_wh_scalar(sprite_dims, ufo_visual_id, fallback_w=7, fallback_h=6)
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
    bullet_w, bullet_h = _sprite_wh_vector(sprite_dims, p_visual, fallback_w=1, fallback_h=2)

    # --- Solar system objects (planets/reactor/obstacle/spawn marker) ---
    planets_active = (state.planets_pi >= 0).astype(jnp.int32)
    planet_x = jnp.clip(state.planets_px, 0.0, float(WINDOW_WIDTH)).astype(jnp.int16)
    planet_y = jnp.clip(state.planets_py, 0.0, float(WINDOW_HEIGHT)).astype(jnp.int16)
    planet_visual = jnp.where(state.planets_pi >= 0, state.planets_pi, jnp.int32(0)).astype(jnp.int16)
    planet_w, planet_h = _sprite_wh_vector(sprite_dims, planet_visual, fallback_w=0, fallback_h=0)

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
    terrain_w, terrain_h = _sprite_wh_scalar(
        sprite_dims, terrain_visual, fallback_w=WINDOW_WIDTH, fallback_h=WINDOW_HEIGHT
    )

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
    reactor_dest_w, reactor_dest_h = _sprite_wh_scalar(
        sprite_dims, reactor_dest_visual, fallback_w=5, fallback_h=5
    )

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
def _get_observation_from_ship_state(ship: ShipState, sprite_dims: jnp.ndarray) -> GravitarObservation:
    sx, sy = _clip_xy_to_screen(ship.x, ship.y)
    ship_visual_id = SHIP_SPRITE_INDICES[ship.angle_idx].astype(jnp.int16)
    ship_w, ship_h = _sprite_wh_scalar(sprite_dims, ship_visual_id, fallback_w=3, fallback_h=7)

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
    saucer_w, saucer_h = _sprite_wh_scalar(sprite_dims, saucer_visual_id, fallback_w=8, fallback_h=7)
    ufo_visual_id = jnp.array(int(SpriteIdx.ENEMY_UFO), dtype=jnp.int16)
    ufo_w, ufo_h = _sprite_wh_scalar(sprite_dims, ufo_visual_id, fallback_w=7, fallback_h=6)
    bullet_visual_id = jnp.full((MAX_ENEMIES,), int(SpriteIdx.ENEMY_BULLET), dtype=jnp.int16)
    bullet_w, bullet_h = _sprite_wh_vector(sprite_dims, bullet_visual_id, fallback_w=1, fallback_h=2)

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


@struct.dataclass
class StepModeContext:
    is_map_mode: jnp.ndarray
    is_level_mode: jnp.ndarray
    is_arena_mode: jnp.ndarray
    allow_exit_top: jnp.ndarray
    allow_ufo: jnp.ndarray
    allow_turrets: jnp.ndarray
    allow_saucer: jnp.ndarray
    max_active_player_bullets: jnp.ndarray


@jax.jit
def build_mode_context(env_state: EnvState) -> StepModeContext:
    mode = env_state.mode
    is_map_mode = mode == jnp.int32(0)
    is_level_mode = mode == jnp.int32(1)
    is_arena_mode = mode == jnp.int32(2)
    return StepModeContext(
        is_map_mode=is_map_mode,
        is_level_mode=is_level_mode,
        is_arena_mode=is_arena_mode,
        allow_exit_top=jnp.where(is_level_mode, env_state.exit_allowed, jnp.bool_(False)),
        allow_ufo=is_level_mode & (env_state.terrain_bank_idx != jnp.int32(5)) & (env_state.terrain_bank_idx != jnp.int32(2)),
        allow_turrets=is_level_mode,
        allow_saucer=is_map_mode | is_arena_mode,
        max_active_player_bullets=jnp.where(
            is_map_mode,
            env_state.max_active_player_bullets_map,
            jnp.where(is_level_mode, env_state.max_active_player_bullets_level, env_state.max_active_player_bullets_arena),
        ),
    )

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
    sprite_dir = os.path.join(render_utils.get_base_sprite_dir(), "gravitar")
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


def _load_and_convert_sprites(numpy_sprites: tuple) -> dict:
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


_DEFAULT_OBS_SPRITES: tuple = load_sprites_tuple()


def _resolve_obs_sprites(consts: GravitarConstants) -> tuple:
    """Sprite tuple used by env and renderer; constants override defaults when set."""
    if consts.SPRITES_TUPLE is not None:
        return consts.SPRITES_TUPLE
    return _DEFAULT_OBS_SPRITES


def _build_default_planets(obs_sprites: tuple) -> tuple:
    MAP_SCALE = 3
    HITBOX_SCALE = 0.90
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
        spr = obs_sprites[int(idx)]
        if spr is not None:
            if idx == SpriteIdx.SPAWN_LOC:
                r = 0.0
            elif idx == SpriteIdx.OBSTACLE:
                r = 0.15 * max(spr.shape[1], spr.shape[0]) * MAP_SCALE * HITBOX_SCALE
            else:
                r = 0.3 * max(spr.shape[1], spr.shape[0]) * MAP_SCALE * HITBOX_SCALE
        else:
            r = 4.0
        px.append(cx); py.append(cy); pr.append(r); pi.append(int(idx))
    return (np.array(px, dtype=np.float32), np.array(py, dtype=np.float32),
            np.array(pr, dtype=np.float32), np.array(pi, dtype=np.int32))


def _build_default_level_data(obs_sprites: tuple) -> tuple:
    """Build static level/layout JAX arrays from a sprite tuple (defaults at import, or per-env)."""
    num_levels = max(LEVEL_LAYOUTS.keys()) + 1
    max_objects = max(len(v) for v in LEVEL_LAYOUTS.values()) if LEVEL_LAYOUTS else 1
    layout_types = np.full((num_levels, max_objects), -1, dtype=np.int32)
    layout_coords_x = np.zeros((num_levels, max_objects), dtype=np.float32)
    layout_coords_y = np.zeros((num_levels, max_objects), dtype=np.float32)
    for level_id, layout_data in LEVEL_LAYOUTS.items():
        for i, obj in enumerate(layout_data):
            if isinstance(obj, dict):
                obj_type = obj["type"]
                coord_x, coord_y = obj["coords"]
            else:
                obj_type, coord_x, coord_y = obj
            layout_types[level_id, i] = int(obj_type)
            layout_coords_x[level_id, i] = coord_x
            layout_coords_y[level_id, i] = coord_y
    jax_layout = {
        "types": jnp.array(layout_types),
        "coords_x": jnp.array(layout_coords_x),
        "coords_y": jnp.array(layout_coords_y),
    }

    max_sprite_id = max(int(e) for e in SpriteIdx)
    dims_array = np.zeros((max_sprite_id + 1, 2), dtype=np.float32)
    for sprite_idx_enum in SpriteIdx:
        surf = obs_sprites[int(sprite_idx_enum)]
        if surf is not None:
            dims_array[int(sprite_idx_enum)] = (surf.shape[1], surf.shape[0])
    jax_sprite_dims = jnp.array(dims_array)

    level_ids_sorted = sorted(LEVEL_ID_TO_TERRAIN_SPRITE.keys())
    jax_level_to_terrain = jnp.array([LEVEL_ID_TO_TERRAIN_SPRITE[k] for k in level_ids_sorted])
    jax_level_to_bank = jnp.array([LEVEL_ID_TO_BANK_IDX[k] for k in level_ids_sorted])
    jax_level_offsets = jnp.array([LEVEL_OFFSETS[k] for k in level_ids_sorted])

    level_transforms = np.zeros((num_levels, 3), dtype=np.float32)
    for level_id in level_ids_sorted:
        terrain_sprite_enum = LEVEL_ID_TO_TERRAIN_SPRITE[level_id]
        terr_surf = obs_sprites[int(terrain_sprite_enum)]
        th, tw = terr_surf.shape[0], terr_surf.shape[1]
        scale = min(WINDOW_WIDTH / tw, WINDOW_HEIGHT / th)
        extra = TERRAIN_SCALE_OVERRIDES.get(terrain_sprite_enum, 1.0)
        scale *= float(extra)
        sw, sh = int(tw * scale), int(th * scale)
        level_offset = LEVEL_OFFSETS.get(level_id, (0, 0))
        ox = (WINDOW_WIDTH - sw) // 2 + level_offset[0]
        oy = (WINDOW_HEIGHT - sh) // 2 + level_offset[1]
        level_transforms[level_id] = [scale, ox, oy]
    jax_level_transforms = jnp.array(level_transforms)

    return jax_layout, jax_sprite_dims, jax_level_to_terrain, jax_level_to_bank, jax_level_offsets, jax_level_transforms


_DEFAULT_PLANETS = _build_default_planets(_DEFAULT_OBS_SPRITES)
(_DEFAULT_JAX_LAYOUT, _DEFAULT_JAX_SPRITE_DIMS, _DEFAULT_JAX_LEVEL_TO_TERRAIN,
 _DEFAULT_JAX_LEVEL_TO_BANK, _DEFAULT_JAX_LEVEL_OFFSETS,
 _DEFAULT_JAX_LEVEL_TRANSFORMS) = _build_default_level_data(_DEFAULT_OBS_SPRITES)

# Terrain bank keyed by sprite tuple object identity (default tuple shared across vanilla envs).
_TERRAIN_BANK_CACHE: dict = {}


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
            angle_idx=jnp.int32(0),
            is_thrusting=jnp.array(False),
            rotation_cooldown=jnp.int32(0)
        ),
        bullets=create_empty_bullets_64(),
        player_bullet_idx=jnp.int32(0),
        cooldown=jnp.array(0, dtype=jnp.int32),
        enemies=create_empty_enemies(),
        fuel_tanks=FuelTanks(
            x=jnp.zeros(MAX_ENEMIES), y=jnp.zeros(MAX_ENEMIES), w=jnp.zeros(MAX_ENEMIES),
            h=jnp.zeros(MAX_ENEMIES), sprite_idx=jnp.full(MAX_ENEMIES, -1),
            active=jnp.zeros(MAX_ENEMIES, dtype=bool)
        ),
        shield_active=jnp.array(False),
        enemy_bullets=create_empty_bullets_16(),
        enemy_bullet_idx=jnp.int32(0),
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
        terrain_scale=jnp.array(1.0),
        terrain_offset=jnp.array([0.0, 0.0]),
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
            map_return_angle_idx=jnp.int32(0),
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
    angle_idx_up = jnp.int32(0)
    angle_idx_down = jnp.int32(8)

    is_reactor = (jnp.asarray(level_id, dtype=jnp.int32) == 4)
    x = jnp.where(is_reactor, x - 55.0, x)
    y = jnp.where(is_reactor, REACTOR_START_Y, y)
    angle = jnp.where(is_reactor, angle_down, angle)
    angle_idx = jnp.where(is_reactor, angle_idx_down, angle_idx_up)

    return ShipState(x=x, y=y, vx=jnp.float32(0.0), vy=jnp.float32(0.0), angle=angle, is_thrusting=jnp.array(False), rotation_cooldown=jnp.int32(0), angle_idx=angle_idx)


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
@jax.jit
def fire_bullet_fast(bullets: Bullets, ship_x, ship_y, ship_angle, bullet_speed, bullet_idx):
    """O(1) player bullet spawn via ring-buffer index."""
    new_vx = jnp.cos(ship_angle) * bullet_speed
    new_vy = jnp.sin(ship_angle) * bullet_speed

    new_bullets = Bullets(
        x=bullets.x.at[bullet_idx].set(ship_x),
        y=bullets.y.at[bullet_idx].set(ship_y),
        vx=bullets.vx.at[bullet_idx].set(new_vx),
        vy=bullets.vy.at[bullet_idx].set(new_vy),
        alive=bullets.alive.at[bullet_idx].set(True),
        sprite_idx=bullets.sprite_idx.at[bullet_idx].set(int(SpriteIdx.SHIP_BULLET)),
    )
    next_idx = (bullet_idx + 1) % MAX_BULLETS
    return new_bullets, next_idx


@jax.jit
def fire_enemy_bullets_fast(
    bullets: Bullets,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    vxs: jnp.ndarray,
    vys: jnp.ndarray,
    alives: jnp.ndarray,
    sprites: jnp.ndarray,
    start_idx: jnp.ndarray,
):
    """O(1) batched enemy bullet spawning via ring-buffer index."""
    num_fired = jnp.sum(alives.astype(jnp.int32))
    pool_size = bullets.x.shape[0]
    offsets = jnp.cumsum(alives.astype(jnp.int32)) - 1
    target_indices = (start_idx + offsets) % pool_size

    def update_attr(old_attr, new_vals):
        safe_targets = jnp.where(alives, target_indices, 0)
        return old_attr.at[safe_targets].set(jnp.where(alives, new_vals, old_attr[safe_targets]))

    new_bullets = Bullets(
        x=update_attr(bullets.x, xs),
        y=update_attr(bullets.y, ys),
        vx=update_attr(bullets.vx, vxs),
        vy=update_attr(bullets.vy, vys),
        alive=update_attr(bullets.alive, jnp.ones_like(alives, dtype=bool)),
        sprite_idx=update_attr(bullets.sprite_idx, sprites),
    )
    next_idx = (start_idx + num_fired) % pool_size
    return new_bullets, next_idx


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
def _check_single_ship_enemy_collision(sx, sy, ship_radius, ex, ey, ew, eh):
    enemy_half_w = ew / 2
    enemy_half_h = eh / 2
    delta_x = sx - ex
    delta_y = sy - ey
    clamped_x = jnp.clip(delta_x, -enemy_half_w, enemy_half_w)
    clamped_y = jnp.clip(delta_y, -enemy_half_h, enemy_half_h)
    closest_point_dx = delta_x - clamped_x
    closest_point_dy = delta_y - clamped_y
    distance_sq = closest_point_dx ** 2 + closest_point_dy ** 2
    return (distance_sq < ship_radius ** 2) & (ew > 0.0)

_check_ship_enemy_collisions_batch = jax.vmap(_check_single_ship_enemy_collision, in_axes=(None, None, None, 0, 0, 0, 0))

@jax.jit
def check_ship_enemy_collisions(ship: ShipState, enemies: Enemies, ship_radius: float) -> jnp.ndarray:
    return _check_ship_enemy_collisions_batch(ship.x, ship.y, ship_radius, enemies.x, enemies.y, enemies.w, enemies.h)


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
def _check_single_enemy_hit(bx, by, b_alive, ex, ey, ew, eh):
    padding = 0.2
    ex1 = ex - ew / 2 - padding
    ex2 = ex + ew / 2 + padding
    ey1 = ey - eh / 2 - padding
    ey2 = ey + eh / 2
    cond_x = (bx >= ex1) & (bx <= ex2)
    cond_y = (by >= ey1) & (by <= ey2)
    return cond_x & cond_y & b_alive & (ew > 0)

_check_enemy_hit_batch = jax.vmap(
    jax.vmap(_check_single_enemy_hit, in_axes=(None, None, None, 0, 0, 0, 0)),
    in_axes=(0, 0, 0, None, None, None, None)
)

@jax.jit
def check_enemy_hit(bullets: Bullets, enemies: Enemies) -> Tuple[Bullets, Enemies]:
    hit_matrix = _check_enemy_hit_batch(bullets.x, bullets.y, bullets.alive, enemies.x, enemies.y, enemies.w, enemies.h)
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
def terrain_hit(env_state: EnvState, terrain_bank: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray,
                radius=jnp.float32(0.3)) -> jnp.ndarray:
    adjusted_x, adjusted_y = x, y
    H, W = terrain_bank.shape[1] - 2 * _TERRAIN_HIT_RMAX, terrain_bank.shape[2] - 2 * _TERRAIN_HIT_RMAX

    xi = jnp.clip(jnp.round(adjusted_x).astype(jnp.int32), 0, W - 1)
    yi = jnp.clip(jnp.round(adjusted_y).astype(jnp.int32), 0, H - 1)

    bi = jnp.clip(env_state.terrain_bank_idx, 0, terrain_bank.shape[0] - 1)
    page = terrain_bank[bi]
    bg_val = terrain_bank[0, 0, 0]

    patch_size = 2 * _TERRAIN_HIT_RMAX + 1
    patch = jax.lax.dynamic_slice(page, (yi, xi), (patch_size, patch_size))

    r_eff = jnp.minimum(jnp.float32(radius), jnp.float32(_TERRAIN_HIT_RMAX))
    mask = _TERRAIN_HIT_DIST2 <= (r_eff ** 2)
    is_not_black = patch != bg_val

    return jnp.any(is_not_black & mask)


@jax.jit
def terrain_hit_ship(env_state: 'EnvState', terrain_bank: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Ship terrain collision using a precomputed 5x5 circular mask."""
    H = terrain_bank.shape[1] - 2 * _TERRAIN_HIT_RMAX
    W = terrain_bank.shape[2] - 2 * _TERRAIN_HIT_RMAX
    xi = jnp.clip(jnp.round(x).astype(jnp.int32), 0, W - 1)
    yi = jnp.clip(jnp.round(y).astype(jnp.int32), 0, H - 1)
    bi = jnp.clip(env_state.terrain_bank_idx, 0, terrain_bank.shape[0] - 1)
    page = terrain_bank[bi]
    bg_val = terrain_bank[0, 0, 0]
    patch = jax.lax.dynamic_slice(
        page,
        (yi + _TERRAIN_HIT_RMAX - _TERRAIN_HIT_SHIP_R, xi + _TERRAIN_HIT_RMAX - _TERRAIN_HIT_SHIP_R),
        (_TERRAIN_HIT_SHIP_SIZE, _TERRAIN_HIT_SHIP_SIZE),
    )
    return jnp.any((patch != bg_val) & _TERRAIN_HIT_MASK_SHIP)


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
def _aggregate_ship_death_flags(
    crashed_on_enemy: jnp.ndarray,
    hit_by_enemy_bullet: jnp.ndarray,
    shield_active: jnp.ndarray,
    hit_terrain_now: jnp.ndarray,
    timer_ran_out: jnp.ndarray,
    hit_by_ufo: jnp.ndarray,
    hit_reactor_dest: jnp.ndarray,
) -> jnp.ndarray:
    bullet_kill = hit_by_enemy_bullet & (~shield_active)
    return crashed_on_enemy | bullet_kill | hit_terrain_now | timer_ran_out | hit_by_ufo | hit_reactor_dest

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
    # --- Action classification via precomputed lookup tables (O(1) vs O(n) isin) ---
    is_thrusting_now = _ACTION_IS_THRUST[action] & (fuel > 0.0)
    right = _ACTION_IS_ROTATE_RIGHT[action]
    left  = _ACTION_IS_ROTATE_LEFT[action]
    thrust_pressed = _ACTION_IS_THRUST[action]

    # --- Physics Parameters ---
    THRUST_POWER = jnp.asarray(thrust_power, dtype=jnp.float32) / WORLD_SCALE
    scaled_solar_gravity = solar_gravity / WORLD_SCALE
    scaled_planetary_gravity = planetary_gravity / WORLD_SCALE
    scaled_reactor_gravity = reactor_gravity / WORLD_SCALE
    MAX_SPEED = jnp.asarray(max_speed, dtype=jnp.float32) / WORLD_SCALE

    bounce_damping = 0.75

    # --- 1. Initialize velocity variables for this frame ---
    vx = state.vx
    vy = state.vy

    # --- 2. Rotation Logic (Discrete 16-angle system) ---
    # Use cached discrete index from state to avoid per-step trig + argmin.
    current_idx = state.angle_idx

    can_rotate = state.rotation_cooldown <= 0
    next_idx = jnp.where(can_rotate & right, (current_idx + 1) % 16, current_idx)
    next_idx = jnp.where(can_rotate & left,  (current_idx - 1) % 16, next_idx)

    angle = SHIP_ANGLES[next_idx]
    did_rotate = (next_idx != current_idx)
    new_rotation_cooldown = jnp.where(
        did_rotate,
        jnp.int32(ROTATION_COOLDOWN_FRAMES),
        jnp.maximum(jnp.int32(0), state.rotation_cooldown - 1)
    )

    # --- 3. Thrust Calculation ---
    # DOWN actions are shield/tractor — no thrust applied.
    can_thrust = fuel > 0.0
    thrust_ax = jnp.cos(angle) * THRUST_POWER
    thrust_ay = jnp.sin(angle) * THRUST_POWER
    apply_thrust = thrust_pressed & can_thrust
    vx = jnp.where(apply_thrust, vx + thrust_ax, vx)
    vy = jnp.where(apply_thrust, vy + thrust_ay, vy)

    # --- Gravity ---
    bank_idx = jnp.clip(terrain_bank_idx.astype(jnp.int32), 0, 5)
    gravity_by_bank = jnp.array(
        [
            scaled_solar_gravity,      # map
            scaled_planetary_gravity,  # planet 1
            scaled_planetary_gravity,  # planet 2
            scaled_planetary_gravity,  # planet 3
            scaled_planetary_gravity,  # planet 4
            scaled_reactor_gravity,    # reactor
        ],
        dtype=jnp.float32,
    )
    gravity = gravity_by_bank[bank_idx]
    is_map_mode = bank_idx == 0
    is_central_gravity = (bank_idx == 2) | (bank_idx == 5)

    # Sun position (OBSTACLE sprite center): (82, 86)
    dx_to_sun = jnp.float32(82.0) - state.x
    dy_to_sun = jnp.float32(86.0) - state.y
    dist2_sun = dx_to_sun * dx_to_sun + dy_to_sun * dy_to_sun
    inv_dist_sun = jax.lax.rsqrt(jnp.maximum(dist2_sun, 1.0))
    gravity_strength = jnp.clip(gravity * (3.2 * inv_dist_sun), 0.0, gravity * 5.0)

    level_center_x = jnp.float32(window_size[0] / 2.0 + 5.0)
    level_center_y = jnp.float32(window_size[1] / 2.0)
    dx_to_center = level_center_x - state.x
    dy_to_center = level_center_y - state.y
    dist2_center = dx_to_center * dx_to_center + dy_to_center * dy_to_center
    inv_dist_center = jax.lax.rsqrt(jnp.maximum(dist2_center, 1.0))
    radial_gravity_strength = jnp.clip(gravity * (50.0 * inv_dist_center), 0.0, gravity * 2.0)

    vx = jnp.where(is_map_mode,
                   vx + dx_to_sun * inv_dist_sun * gravity_strength,
                   jnp.where(is_central_gravity,
                             vx + dx_to_center * inv_dist_center * radial_gravity_strength,
                             vx))
    vy = jnp.where(is_map_mode,
                   vy + dy_to_sun * inv_dist_sun * gravity_strength,
                   jnp.where(is_central_gravity,
                             vy + dy_to_center * inv_dist_center * radial_gravity_strength,
                             vy + gravity))

    # --- 4. Speed cap via jnp.where (avoids lax.cond branch overhead) ---
    speed_sq = vx * vx + vy * vy
    over_limit = speed_sq > MAX_SPEED * MAX_SPEED
    inv_speed = jax.lax.rsqrt(jnp.maximum(speed_sq, 1e-12))
    scale = jnp.where(over_limit, MAX_SPEED * inv_speed, jnp.float32(1.0))
    vx = vx * scale
    vy = vy * scale

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
    return ShipState(x=final_x, y=final_y, vx=final_vx, vy=final_vy, angle=normalized_angle, is_thrusting=is_thrusting_now, rotation_cooldown=new_rotation_cooldown, angle_idx=next_idx)


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

    return s.replace(x=s.x + vx, y=s.y + vy, vx=vx, vy=vy)


@jax.jit
def _update_saucer_horizontal(s: SaucerState, target_x, reactor_y, speed) -> SaucerState:
    """Update saucer to move only horizontally at reactor height, following the ship's x position"""
    # Move horizontally towards target_x
    dx = target_x - s.x
    # Determine direction: move left or right
    vx = jnp.where(dx > 0, speed, -speed)
    vx = jnp.where(jnp.abs(dx) < speed, jnp.float32(0.0), vx)  # Stop if close enough
    
    # Keep y fixed at reactor height
    return s.replace(x=s.x + vx, y=jnp.float32(reactor_y), vx=vx, vy=jnp.float32(0.0))


@jax.jit
def _update_saucer_mode_linear(
    saucer: SaucerState,
    ship: ShipState,
    reactor_y: jnp.ndarray,
    mode_ctx: StepModeContext,
) -> SaucerState:
    # Compute both movement styles and select via mode masks.
    saucer_map = _update_saucer_horizontal(saucer, ship.x, reactor_y, SAUCER_SPEED_MAP)
    saucer_arena = _update_saucer_seek(saucer, ship.x, ship.y, SAUCER_SPEED_ARENA)
    mode_updated = jax.tree_util.tree_map(
        lambda m_v, a_v: jnp.where(mode_ctx.is_map_mode, m_v, a_v),
        saucer_map,
        saucer_arena,
    )
    should_apply = mode_ctx.allow_saucer & saucer.alive
    return jax.tree_util.tree_map(
        lambda new_v, old_v: jnp.where(should_apply, new_v, old_v),
        mode_updated,
        saucer,
    )


@jax.jit
def _saucer_fire_one(sauc: SaucerState,
                     ship_x: jnp.ndarray,
                     ship_y: jnp.ndarray,
                     prev_enemy_bullets: Bullets,
                     enemy_bullet_idx: jnp.ndarray,
                     mode_timer: jnp.ndarray,
               max_active_bullets: jnp.ndarray = jnp.int32(1),
                     ) -> tuple[Bullets, jnp.ndarray]:
    can_fire = sauc.alive & ((mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0) \
           & (_bullets_alive_count(prev_enemy_bullets) < max_active_bullets)

    dx = ship_x - sauc.x
    dy = ship_y - sauc.y
    d = jnp.maximum(jnp.sqrt(dx * dx + dy * dy), 1e-3)
    vx = SAUCER_BULLET_SPEED * dx / d
    vy = SAUCER_BULLET_SPEED * dy / d
    alives = jnp.array([can_fire], dtype=bool)
    sprites = jnp.array([int(SpriteIdx.ENEMY_BULLET)], dtype=jnp.int32)
    new_bullets, next_idx = fire_enemy_bullets_fast(
        prev_enemy_bullets,
        jnp.array([sauc.x], dtype=jnp.float32),
        jnp.array([sauc.y], dtype=jnp.float32),
        jnp.array([vx], dtype=jnp.float32),
        jnp.array([vy], dtype=jnp.float32),
        alives,
        sprites,
        enemy_bullet_idx,
    )
    capped = _enforce_cap_keep_old(new_bullets, cap=max_active_bullets)
    final_idx = jnp.where(can_fire, next_idx, enemy_bullet_idx)
    return capped, final_idx


@jax.jit
def _saucer_fire_random(sauc: SaucerState,
                        prev_enemy_bullets: Bullets,
                        enemy_bullet_idx: jnp.ndarray,
                        mode_timer: jnp.ndarray,
                        key: jnp.ndarray,
                        max_active_bullets: jnp.ndarray = jnp.int32(2),
                        ) -> tuple[Bullets, jnp.ndarray]:
    """Saucer fires in random directions with max 2 bullets"""
    can_fire = sauc.alive & ((mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0) \
               & (_bullets_alive_count(prev_enemy_bullets) < max_active_bullets)

    angle = jax.random.uniform(key, minval=0.0, maxval=2.0 * jnp.pi)
    vx = SAUCER_BULLET_SPEED * jnp.cos(angle)
    vy = SAUCER_BULLET_SPEED * jnp.sin(angle)
    alives = jnp.array([can_fire], dtype=bool)
    sprites = jnp.array([int(SpriteIdx.ENEMY_BULLET)], dtype=jnp.int32)
    new_bullets, next_idx = fire_enemy_bullets_fast(
        prev_enemy_bullets,
        jnp.array([sauc.x], dtype=jnp.float32),
        jnp.array([sauc.y], dtype=jnp.float32),
        jnp.array([vx], dtype=jnp.float32),
        jnp.array([vy], dtype=jnp.float32),
        alives,
        sprites,
        enemy_bullet_idx,
    )
    capped = _enforce_cap_keep_old(new_bullets, cap=max_active_bullets)
    final_idx = jnp.where(can_fire, next_idx, enemy_bullet_idx)
    return capped, final_idx


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

@jax.jit
def _bullets_hit_terrain(env_state: EnvState, terrain_bank: jnp.ndarray, bullets: Bullets) -> Bullets:
    H, W = terrain_bank.shape[1] - 2 * _TERRAIN_HIT_RMAX, terrain_bank.shape[2] - 2 * _TERRAIN_HIT_RMAX

    bank_idx = jnp.clip(env_state.terrain_bank_idx, 0, terrain_bank.shape[0] - 1)
    terrain_map = terrain_bank[bank_idx]

    xi = jnp.clip(jnp.round(bullets.x).astype(jnp.int32) + _TERRAIN_HIT_RMAX, 0, W + 2 * _TERRAIN_HIT_RMAX - 1)
    yi = jnp.clip(jnp.round(bullets.y).astype(jnp.int32) + _TERRAIN_HIT_RMAX, 0, H + 2 * _TERRAIN_HIT_RMAX - 1)
    safe_xi = jnp.where(bullets.alive, xi, jnp.int32(0))
    safe_yi = jnp.where(bullets.alive, yi, jnp.int32(0))
    pixel_colors = terrain_map[safe_yi, safe_xi]
    bg_val = terrain_bank[0, 0, 0]
    hit_terrain_mask = (pixel_colors != bg_val) & bullets.alive

    final_hit_mask = hit_terrain_mask

    return bullets.replace(alive=bullets.alive & ~final_hit_mask)


@jax.jit
def _ufo_ground_safe_y_at(heightmaps: jnp.ndarray, terrain_bank_idx: jnp.ndarray, xf: jnp.ndarray):
    W = WINDOW_WIDTH
    bank_idx = jnp.clip(terrain_bank_idx, 0, heightmaps.shape[0] - 1)
    col_x = jnp.clip(xf.astype(jnp.int32), 0, W - 1)
    ground_y = heightmaps[bank_idx, col_x]
    return jnp.float32(ground_y) - 20.0  # CLEARANCE


def _update_ufo(
    ufo: UFOState,
    ufo_spawn_timer: jnp.ndarray,
    terrain_bank_idx: jnp.ndarray,
    ship: ShipState,
    bullets: Bullets,
    terrain_heightmaps: jnp.ndarray,
) -> tuple:
    """Returns (ufo, ufo_spawn_timer, bullets) — never touches full EnvState."""
    u = ufo
    LEFT_BOUNDARY = 8.0
    RIGHT_BOUNDARY = jnp.float32(WINDOW_WIDTH - 8.0)
    MIN_ALTITUDE = jnp.float32(HUD_HEIGHT + 20.0)
    VERTICAL_ADJUST_SPEED = jnp.float32(0.5 / WORLD_SCALE)

    # ── alive path ──
    hit_horizontal = (u.x + u.vx <= LEFT_BOUNDARY) | (u.x + u.vx >= RIGHT_BOUNDARY)
    alive_vx = jnp.where(hit_horizontal, -u.vx, u.vx)
    alive_x = jnp.clip(u.x + alive_vx, LEFT_BOUNDARY, RIGHT_BOUNDARY)
    safe_y_here = _ufo_ground_safe_y_at(terrain_heightmaps, terrain_bank_idx, alive_x)
    target_y = jnp.maximum(safe_y_here - 20.0, MIN_ALTITUDE)
    alive_vy = jnp.clip(target_y - u.y, -VERTICAL_ADJUST_SPEED, VERTICAL_ADJUST_SPEED)
    alive_y = jnp.clip(u.y + alive_vy, MIN_ALTITUDE, jnp.float32(WINDOW_HEIGHT - 20.0))
    bullets_after_hit, hit_by_bullet = _bullets_hit_ufo(bullets, u.replace(x=alive_x, y=alive_y))
    alive_hp = u.hp - jnp.where(hit_by_bullet, 1, 0)
    alive_just_died = u.alive & (alive_hp <= 0)
    alive_ufo = u.replace(
        x=alive_x, y=alive_y, vx=alive_vx, vy=alive_vy,
        hp=alive_hp,
        alive=u.alive & (alive_hp > 0),
        death_timer=jnp.where(alive_just_died, jnp.int32(SAUCER_EXPLOSION_FRAMES), u.death_timer),
    )
    alive_spawn_timer = ufo_spawn_timer

    # ── dead path ──
    dead_ufo = u.replace(death_timer=jnp.maximum(u.death_timer - 1, 0))
    dead_spawn_timer = jnp.where(
        u.death_timer == 0,
        jnp.maximum(ufo_spawn_timer - 1, 0),
        ufo_spawn_timer,
    )
    dead_bullets = bullets

    # ── merge: select alive vs dead by ufo.alive ──
    final_ufo = jax.tree_util.tree_map(
        lambda av, dv: jnp.where(u.alive, av, dv), alive_ufo, dead_ufo
    )
    final_spawn_timer = jnp.where(u.alive, alive_spawn_timer, dead_spawn_timer)
    final_bullets = jax.tree_util.tree_map(
        lambda ab, db: jnp.where(u.alive, ab, db), bullets_after_hit, dead_bullets
    )
    return final_ufo, final_spawn_timer, final_bullets


# ========== Step Core ==========
@jax.jit
def step_core_linear(
    env_state: EnvState,
    action: int,
    terrain_bank: jnp.ndarray,
    terrain_heightmaps: jnp.ndarray,
    obs_sprite_dims: jnp.ndarray,
):
    def _game_is_over(state, _):
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
        obs = _get_observation_from_state(state, obs_sprite_dims)
        return obs, state, 0.0, jnp.array(True), info, jnp.array(False), jnp.int32(-1)

    def _unified_game_loop(state, act):
        mode_ctx = build_mode_context(state)
        is_map = mode_ctx.is_map_mode
        is_level = mode_ctx.is_level_mode
        is_arena = mode_ctx.is_arena_mode

        # ── 1. UFO spawn (level-mode only) — unconditional math + jnp.where, no lax.cond ──
        ship_low_enough_for_ufo = state.state.y >= jnp.float32(UFO_SPAWN_Y_THRESHOLD)
        can_spawn_ufo = (
            mode_ctx.allow_ufo
            & (state.ufo_spawn_timer == 0)
            & (~state.ufo.alive)
            & ship_low_enough_for_ufo
        )
        _ufo_b = state.terrain_bank_idx
        _ufo_born_left = (_ufo_b == 2) | (_ufo_b == 4)
        _ufo_x0 = jnp.where(_ufo_born_left, jnp.float32(-30.0), jnp.float32(WINDOW_WIDTH + 30.0))
        _ufo_vx = jnp.where(_ufo_born_left, jnp.float32(0.6 / WORLD_SCALE), jnp.float32(-0.6 / WORLD_SCALE))
        _ufo_bank = jnp.clip(state.terrain_bank_idx, 0, terrain_heightmaps.shape[0] - 1)
        _ufo_highest = jnp.min(terrain_heightmaps[_ufo_bank])
        _ufo_safe_y = jnp.where(
            _ufo_highest < WINDOW_HEIGHT,
            jnp.float32(_ufo_highest) - 10.0,
            jnp.float32(WINDOW_HEIGHT) - 10.0,
        )
        _ufo_y0 = jnp.clip(_ufo_safe_y, jnp.float32(HUD_HEIGHT + 20.0), jnp.float32(WINDOW_HEIGHT - 20.0))
        _spawned_ufo = UFOState(
            x=_ufo_x0, y=_ufo_y0, vx=_ufo_vx, vy=jnp.float32(0.0),
            hp=jnp.int32(1), alive=jnp.array(True), death_timer=jnp.int32(0),
        )
        state_after_spawn = state.replace(
            ufo=jax.tree_util.tree_map(
                lambda nv, ov: jnp.where(can_spawn_ufo, nv, ov), _spawned_ufo, state.ufo
            ),
            ufo_spawn_timer=jnp.where(can_spawn_ufo, jnp.int32(UFO_RESPAWN_DELAY_FRAMES), state.ufo_spawn_timer),
            ufo_home_x=jnp.where(can_spawn_ufo, _ufo_x0, state.ufo_home_x),
            ufo_home_y=jnp.where(can_spawn_ufo, _ufo_y0, state.ufo_home_y),
            ufo_bullets=jax.tree_util.tree_map(
                lambda ev, cv: jnp.where(can_spawn_ufo, ev, cv), create_empty_bullets_16(), state.ufo_bullets
            ),
        )

        # ── 2. Reactor timer (level-mode, reactor level only) ──
        is_in_reactor = (state.current_level == 4)
        timer_after_tick = state_after_spawn.reactor_timer - 1
        new_reactor_timer = jnp.where(is_in_reactor, timer_after_tick, state_after_spawn.reactor_timer)
        timer_ran_out = is_in_reactor & (new_reactor_timer <= 0)

        # ── 3. Ship movement ──
        was_crashing = state_after_spawn.crash_timer > 0
        actual_action = jnp.where(was_crashing, NOOP, act)
        ship_before = state_after_spawn.state.replace(
            vx=jnp.where(was_crashing, 0.0, state_after_spawn.state.vx),
            vy=jnp.where(was_crashing, 0.0, state_after_spawn.state.vy),
        )
        ship_after = ship_step(
            ship_before, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT,
            state_after_spawn.fuel, state_after_spawn.terrain_bank_idx, mode_ctx.allow_exit_top,
            thrust_power=state_after_spawn.thrust_power, max_speed=state_after_spawn.max_speed,
            solar_gravity=state_after_spawn.solar_gravity,
            planetary_gravity=state_after_spawn.planetary_gravity,
            reactor_gravity=state_after_spawn.reactor_gravity,
        )

        # ── 4. Fuel consumption ──
        is_thrusting = _ACTION_IS_THRUST[actual_action]
        is_using_shield_tractor = _ACTION_IS_SHIELD[actual_action]
        # Arena: no shield fuel cost (arena step didn't charge for it)
        fuel_consumed = (
            jnp.where(is_thrusting, state_after_spawn.fuel_consume_thrust, 0.0)
            + jnp.where(is_using_shield_tractor & (~is_arena), state_after_spawn.fuel_consume_shield_tractor, 0.0)
        )

        # ── 5. Player bullets ──
        is_fire_pressed = _ACTION_IS_FIRE[act]
        was_fire_pressed = _ACTION_IS_FIRE[state_after_spawn.prev_action]
        fire_just_pressed = is_fire_pressed & (~was_fire_pressed)
        can_fire_player = (
            fire_just_pressed
            & (state_after_spawn.cooldown == 0)
            & (_bullets_alive_count(state_after_spawn.bullets) < mode_ctx.max_active_player_bullets)
        )
        fired_bullets, next_player_bullet_idx = fire_bullet_fast(
            state_after_spawn.bullets, ship_after.x, ship_after.y, ship_after.angle,
            PLAYER_BULLET_SPEED, state_after_spawn.player_bullet_idx,
        )
        bullets = jax.tree_util.tree_map(
            lambda new, old: jnp.where(can_fire_player, new, old), fired_bullets, state_after_spawn.bullets
        )
        bullets = update_bullets(bullets)
        cooldown = jnp.where(can_fire_player, PLAYER_FIRE_COOLDOWN_FRAMES, jnp.maximum(state_after_spawn.cooldown - 1, 0))
        player_bullet_idx = jnp.where(can_fire_player, next_player_bullet_idx, state_after_spawn.player_bullet_idx)

        # ── 6. Fuel tank collection (level-mode only) ──
        tanks = state_after_spawn.fuel_tanks
        ship_left = ship_after.x - SHIP_RADIUS
        ship_right = ship_after.x + SHIP_RADIUS
        ship_top = ship_after.y - SHIP_RADIUS
        ship_bottom = ship_after.y + SHIP_RADIUS
        tank_half_w = tanks.w / 2
        tank_half_h = tanks.h / 2
        tank_left = tanks.x - tank_half_w
        tank_right = tanks.x + tank_half_w
        tank_top = tanks.y - tank_half_h
        tank_bottom = tanks.y + tank_half_h
        overlap_x = (ship_right > tank_left) & (ship_left < tank_right)
        overlap_y = (ship_bottom > tank_top) & (ship_top < tank_bottom)
        direct_collision = tanks.active & overlap_x & overlap_y
        is_reactor_terrain = state_after_spawn.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)
        can_use_tractor = is_level & ((~is_reactor_terrain) | state_after_spawn.allow_tractor_in_reactor)
        dx_tank = tanks.x - ship_after.x
        dy_tank = tanks.y - ship_after.y
        distance_sq_tank = dx_tank * dx_tank + dy_tank * dy_tank
        in_tractor_range = distance_sq_tank <= (TRACTOR_BEAM_RANGE ** 2)
        can_collect_tanks = ~was_crashing
        tractor_pickup = can_collect_tanks & can_use_tractor & is_using_shield_tractor & in_tractor_range & tanks.active
        collision_mask = can_collect_tanks & (direct_collision | tractor_pickup)
        new_fuel_tanks = tanks.replace(active=tanks.active & ~collision_mask)
        num_tanks_collected = jnp.sum(collision_mask)
        fuel_gained = jnp.where(is_level, num_tanks_collected * 5000.0, 0.0)

        # ── 7. Saucer (map/arena) ──
        rx, ry, has_reactor = _get_reactor_center(state_after_spawn.planets_px, state_after_spawn.planets_py, state_after_spawn.planets_pi)
        saucer_spawn_timer = state_after_spawn.saucer_spawn_timer
        should_tick_timer = mode_ctx.allow_saucer & (~state_after_spawn.saucer.alive) & (state_after_spawn.saucer.death_timer == 0)
        saucer_spawn_timer = jnp.where(should_tick_timer, jnp.maximum(saucer_spawn_timer - 1, 0), saucer_spawn_timer)
        should_spawn_saucer = is_map & (saucer_spawn_timer == 0) & (~state_after_spawn.saucer.alive) & has_reactor
        saucer_spawned = jax.tree_util.tree_map(
            lambda spawn_v, old_v: jnp.where(should_spawn_saucer, spawn_v, old_v),
            _spawn_saucer_at(rx, ry, ship_after.x, ship_after.y, SAUCER_SPEED_MAP),
            state_after_spawn.saucer,
        )
        saucer_spawn_timer = jnp.where(should_spawn_saucer, jnp.int32(99999), saucer_spawn_timer)
        # Arena uses saucer.y for ry; map uses reactor ry
        saucer_ry = jnp.where(is_arena, state_after_spawn.saucer.y, ry)
        saucer_after_move = _update_saucer_mode_linear(saucer_spawned, ship_after, saucer_ry, mode_ctx)

        # ── 8. Saucer fire (map/arena) + turret fire (level) ──
        # Key usage must exactly match per-mode originals:
        #   map:   fire_key, new_main_key = split(key)        → saucer uses fire_key(key[0]); stores new_main_key(key[1])
        #   arena: fire_key, new_main_key = split(key)        → saucer uses fire_key(key[0]); stores new_main_key(key[1])
        #   level: current_key, rank_key = split(current_key) → turret uses rank_key(key[1]); stores current_key(key[0])
        fire_key, key_after_fire = jax.random.split(state_after_spawn.key)
        level_rank_key = key_after_fire  # level uses rank_key = key[1]
        level_key_out = fire_key         # level stores current_key = key[0]

        # Map/arena: mode_timer is from state_after_spawn; level uses turrets instead
        map_mode_timer = jnp.where(is_map, state_after_spawn.mode_timer + 1, 0)
        saucer_fire_timer = jnp.where(is_arena, state_after_spawn.mode_timer, map_mode_timer)
        saucer_enemy_bullets, saucer_enemy_bullet_idx = _saucer_fire_random(
            saucer_after_move, state_after_spawn.enemy_bullets, state_after_spawn.enemy_bullet_idx,
            saucer_fire_timer, fire_key, state_after_spawn.max_active_saucer_bullets,
        )
        saucer_enemy_bullets = update_bullets(saucer_enemy_bullets)

        # Turrets (always computed; masked to level-mode)
        enemies = enemy_step(state_after_spawn.enemies, WINDOW_WIDTH)
        is_exploding = enemies.death_timer > 0
        enemies = enemies.replace(
            death_timer=jnp.maximum(enemies.death_timer - 1, 0),
            w=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.w),
            h=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.h),
        )
        is_turret = (
            (enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE))
            | (enemies.sprite_idx == int(SpriteIdx.ENEMY_GREEN))
            | (enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE_FLIPPED))
        )
        turrets_ready_mask = (
            (enemies.w > 0) & (state_after_spawn.fire_cooldown == 0)
            & (enemies.death_timer == 0) & is_turret & mode_ctx.allow_turrets
        )
        enemy_alive_ct = _bullets_alive_count(state_after_spawn.enemy_bullets)
        space_left = jnp.maximum(state_after_spawn.max_active_enemy_bullets - enemy_alive_ct, 0)
        rand_vals = jax.random.uniform(level_rank_key, shape=turrets_ready_mask.shape)
        scores_turret = jnp.where(turrets_ready_mask, rand_vals, -1.0)
        ranks_turret = jnp.sum(scores_turret[:, None] > scores_turret[None, :], axis=0)
        should_fire_mask = turrets_ready_mask & (ranks_turret < space_left)
        base_interval = state_after_spawn.enemy_fire_cooldown_frames
        variance_max = jnp.maximum(jnp.int32(1), base_interval)
        varied_interval = base_interval + jnp.int32((enemies.x * 0.5) % variance_max)
        fire_cooldown = jnp.maximum(state_after_spawn.fire_cooldown - 1, 0)
        fire_cooldown = jnp.where(should_fire_mask, varied_interval, fire_cooldown)
        level_mode_timer = state_after_spawn.mode_timer
        angle_seed = (enemies.x + enemies.y + jnp.float32(level_mode_timer)) * 0.1
        fire_arc = jnp.pi + jnp.deg2rad(20.0)
        half_arc = fire_arc * 0.5
        random_offset = (angle_seed % 1.0) * fire_arc
        is_flipped = enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE_FLIPPED)
        normal_start = -jnp.pi * 0.5 - half_arc
        flipped_start = jnp.pi * 0.5 - half_arc
        random_angle = jnp.where(is_flipped, flipped_start + random_offset, normal_start + random_offset)
        vx_out = jnp.where(should_fire_mask, jnp.cos(random_angle) * ENEMY_BULLET_SPEED, 0.0)
        vy_out = jnp.where(should_fire_mask, jnp.sin(random_angle) * ENEMY_BULLET_SPEED, 0.0)
        x_out = jnp.where(should_fire_mask, enemies.x, 0.0)
        y_out = jnp.where(should_fire_mask, enemies.y, 0.0)
        bullet_sprite = jnp.where(
            enemies.sprite_idx == int(SpriteIdx.ENEMY_GREEN),
            int(SpriteIdx.ENEMY_GREEN_BULLET), int(SpriteIdx.ENEMY_BULLET),
        )
        turret_enemy_bullets, turret_enemy_bullet_idx = fire_enemy_bullets_fast(
            state_after_spawn.enemy_bullets, x_out, y_out, vx_out, vy_out,
            should_fire_mask, bullet_sprite, state_after_spawn.enemy_bullet_idx,
        )
        turret_enemy_bullets = _enforce_cap_keep_old(turret_enemy_bullets, cap=state_after_spawn.max_active_enemy_bullets)
        turret_enemy_bullets = update_bullets(turret_enemy_bullets)

        # Select correct enemy bullets by mode
        enemy_bullets = jax.tree_util.tree_map(
            lambda s_val, t_val: jnp.where(is_level, t_val, s_val),
            saucer_enemy_bullets, turret_enemy_bullets,
        )
        enemy_bullet_idx = jnp.where(is_level, turret_enemy_bullet_idx, saucer_enemy_bullet_idx)

        # ── 9. UFO update — no lax.cond, only touches UFO fields ──
        final_key = jnp.where(is_level, level_key_out, key_after_fire)
        _ufo_out, _ufo_spawn_timer_out, _ufo_bullets_out = _update_ufo(
            state_after_spawn.ufo, state_after_spawn.ufo_spawn_timer,
            state_after_spawn.terrain_bank_idx,
            ship_after, bullets, terrain_heightmaps,
        )
        # UFO only matters in level mode; freeze in map/arena
        ufo = jax.tree_util.tree_map(
            lambda new_v, old_v: jnp.where(mode_ctx.allow_ufo, new_v, old_v),
            _ufo_out, state_after_spawn.ufo,
        )
        ufo_spawn_timer_after_ufo = jnp.where(
            mode_ctx.allow_ufo, _ufo_spawn_timer_out, state_after_spawn.ufo_spawn_timer
        )
        bullets = jax.tree_util.tree_map(
            lambda nb, ob: jnp.where(mode_ctx.allow_ufo, nb, ob), _ufo_bullets_out, bullets
        )

        # ── 10. Bullet-terrain collisions ──
        bullets = _bullets_hit_terrain(state_after_spawn, terrain_bank, bullets)
        enemy_bullets = _bullets_hit_terrain(state_after_spawn, terrain_bank, enemy_bullets)

        # ── 11. Enemy collisions ──
        enemies_before_hits = enemies
        bullets, enemies = check_enemy_hit(bullets, enemies)
        hit_enemy_mask = check_ship_enemy_collisions(ship_after, enemies, SHIP_RADIUS)
        enemies = enemies.replace(
            death_timer=jnp.where(hit_enemy_mask, ENEMY_EXPLOSION_FRAMES, enemies.death_timer)
        )

        # ── 12. Green enemy reveals hidden tank ──
        green_sprite_idx = int(SpriteIdx.ENEMY_GREEN)
        was_green_alive = (state_after_spawn.enemies.sprite_idx == green_sprite_idx) & (state_after_spawn.enemies.hp > 0)
        is_green_dead_now = (enemies.sprite_idx == green_sprite_idx) & (enemies.hp <= 0)
        green_killed_by_hp = was_green_alive & is_green_dead_now
        green_killed_by_contact = (enemies.sprite_idx == green_sprite_idx) & hit_enemy_mask
        green_just_killed = green_killed_by_hp | green_killed_by_contact
        same_x = jnp.abs(new_fuel_tanks.x[:, None] - enemies.x[None, :]) < 0.51
        same_y = jnp.abs(new_fuel_tanks.y[:, None] - enemies.y[None, :]) < 0.51
        same_position = same_x & same_y
        tank_matches_killed_green = jnp.any(same_position & green_just_killed[None, :], axis=1)
        hidden_tank_to_reveal = (~new_fuel_tanks.active) & (new_fuel_tanks.sprite_idx == int(SpriteIdx.FUEL_TANK)) & tank_matches_killed_green
        new_fuel_tanks = new_fuel_tanks.replace(active=new_fuel_tanks.active | hidden_tank_to_reveal)

        # ── 13. Saucer and UFO bullet/contact hits ──
        bullets, hit_saucer_by_bullet = _bullets_hit_saucer(bullets, saucer_after_move)
        bullets, _ = _bullets_hit_ufo(bullets, ufo)
        enemy_bullets, hit_ship_by_enemy_bullet = consume_ship_hits(ship_after, enemy_bullets, SHIP_RADIUS)

        hit_terr = terrain_hit_ship(state_after_spawn, terrain_bank, ship_after.x, ship_after.y)
        hit_saucer_by_contact = (
            _circle_hit(ship_after.x, ship_after.y, SHIP_RADIUS, saucer_after_move.x, saucer_after_move.y, SAUCER_RADIUS)
            & saucer_after_move.alive
        )
        # UFO collision: use pre-update position (same as level step)
        ufo_before_update = state_after_spawn.ufo
        hit_by_ufo = (
            _circle_hit(ship_after.x, ship_after.y, SHIP_RADIUS, ufo_before_update.x, ufo_before_update.y, UFO_HIT_RADIUS)
            & ufo_before_update.alive & is_level
        )

        # ── 14. Obstacle check (map-mode only) ──
        px, py, pr, pi, pid = (
            state_after_spawn.planets_px, state_after_spawn.planets_py,
            state_after_spawn.planets_pr, state_after_spawn.planets_pi, state_after_spawn.planets_id,
        )
        dx, dy = px - ship_after.x, py - ship_after.y
        dist2 = dx * dx + dy * dy
        hit_obstacle = jnp.any((pi == SpriteIdx.OBSTACLE) & (dist2 <= (pr + SHIP_RADIUS) ** 2))

        # ── 15. Reactor core hit (level reactor only) ──
        dx_reactor = ship_after.x - state_after_spawn.reactor_dest_x
        dy_reactor = ship_after.y - state_after_spawn.reactor_dest_y
        dist_sq_reactor = dx_reactor ** 2 + dy_reactor ** 2
        hit_reactor_dest = (
            is_in_reactor & state_after_spawn.reactor_dest_active
            & (dist_sq_reactor < (state_after_spawn.reactor_dest_radius + SHIP_RADIUS) ** 2)
        )
        can_activate_reactor = is_in_reactor & ~state_after_spawn.reactor_activated
        _rdx = bullets.x - state_after_spawn.reactor_dest_x
        _rdy = bullets.y - state_after_spawn.reactor_dest_y
        _reactor_hit_mask = (
            bullets.alive
            & can_activate_reactor
            & ((_rdx ** 2 + _rdy ** 2) < (state_after_spawn.reactor_dest_radius + 5.0) ** 2)
        )
        hit_reactor_core = jnp.any(_reactor_hit_mask)
        bullets = bullets.replace(alive=bullets.alive & ~_reactor_hit_mask)
        new_reactor_activated = state_after_spawn.reactor_activated | hit_reactor_core

        # ── 16. Death flags aggregation ──
        hit_enemy_types = jnp.where(hit_enemy_mask, enemies.sprite_idx, -1)
        crashed_on_turret = jnp.any(
            (hit_enemy_types == int(SpriteIdx.ENEMY_ORANGE))
            | (hit_enemy_types == int(SpriteIdx.ENEMY_ORANGE_FLIPPED))
            | (hit_enemy_types == int(SpriteIdx.ENEMY_GREEN))
        ) & is_level
        dead = _aggregate_ship_death_flags(
            crashed_on_turret | (hit_saucer_by_contact & is_arena) | (hit_obstacle & is_map),
            hit_ship_by_enemy_bullet,
            is_using_shield_tractor,
            hit_terr & is_level,
            timer_ran_out,
            hit_by_ufo,
            hit_reactor_dest,
        )

        # ── 17. Crash timer ──
        start_crash = dead & (~was_crashing)
        crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(state_after_spawn.crash_timer - 1, 0))
        is_crashing_now = crash_timer_next > 0
        crash_animation_finished = (state_after_spawn.crash_timer == 1)

        # ── 18. Level-mode respawn (lax.cond: different shapes avoided via jnp.where) ──
        respawn_now = crash_animation_finished & (~is_in_reactor) & is_level
        reset_from_reactor_crash = crash_animation_finished & is_in_reactor & is_level
        death_event_level = respawn_now | reset_from_reactor_crash

        # ── 19. Level win checks ──
        exited_top = ship_after.y < (HUD_HEIGHT + SHIP_RADIUS)
        all_enemies_gone = jnp.all(enemies.w == 0)
        reset_level_win = all_enemies_gone & (~is_in_reactor) & exited_top & is_level
        win_reactor = is_in_reactor & new_reactor_activated & exited_top & is_level
        level_cleared_now = reset_level_win | win_reactor

        # ── 20. Level scoring ──
        just_started_exploding = (
            (enemies_before_hits.w > 0)
            & (enemies_before_hits.death_timer == 0)
            & (enemies.death_timer > 0)
        )
        is_orange_e = (enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_ORANGE))) | \
                      (enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_ORANGE_FLIPPED)))
        is_green_e = enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_GREEN))
        k_orange = jnp.sum(just_started_exploding & is_orange_e).astype(jnp.float32)
        k_green = jnp.sum(just_started_exploding & is_green_e).astype(jnp.float32)
        score_from_enemies = state_after_spawn.enemy_kill_score * (k_orange + k_green)
        score_from_level_clear = jnp.where(level_cleared_now, state_after_spawn.level_clear_score, 0.0)
        ufo_just_died = (~ufo.alive) & state_after_spawn.ufo.alive & (ufo.death_timer > 0)
        score_from_ufo = jnp.where(ufo_just_died, state_after_spawn.ufo_kill_score, 0.0)
        level_score_delta = score_from_enemies + score_from_level_clear + score_from_ufo

        # ── 21. Saucer HP and death (map/arena) ──
        saucer_hp_after = saucer_after_move.hp - jnp.where(hit_saucer_by_bullet, 1, 0)
        saucer_just_died = saucer_after_move.alive & (saucer_hp_after <= 0)
        # Map: set respawn timer on death; Arena: no respawn
        saucer_respawn_timer_on_death = jnp.where(is_map, SAUCER_RESPAWN_DELAY_FRAMES, saucer_spawn_timer)
        saucer_spawn_timer = jnp.where(saucer_just_died & is_map, saucer_respawn_timer_on_death, saucer_spawn_timer)
        saucer_final = saucer_after_move.replace(
            hp=saucer_hp_after,
            alive=saucer_after_move.alive & (saucer_hp_after > 0),
            death_timer=jnp.where(
                saucer_just_died, SAUCER_EXPLOSION_FRAMES,
                jnp.maximum(saucer_after_move.death_timer - 1, 0),
            ),
        )
        reward_saucer = jnp.where(saucer_just_died & mode_ctx.allow_saucer, state_after_spawn.saucer_kill_score, jnp.float32(0.0))

        # ── 22. Lives / score / fuel ──
        # Level: lives decrease on death_event; map/arena: handled externally
        lives_after_death = state_after_spawn.lives - jnp.where(death_event_level, 1, 0)
        score_before = state_after_spawn.score
        level_score_after = score_before + level_score_delta
        bonus_life_crossed = (level_score_after // 10000) > (score_before // 10000)
        lives_gained_from_score = jnp.where(bonus_life_crossed & is_level, 1, 0)
        final_lives = lives_after_death + lives_gained_from_score
        game_over_level = death_event_level & (final_lives <= 0)

        # Map/arena: saucer reward; level: turret/clear/UFO rewards
        map_score_delta = reward_saucer
        score_delta = jnp.where(is_level, level_score_delta, map_score_delta)
        fuel_next = jnp.maximum(0.0, state_after_spawn.fuel - fuel_consumed + fuel_gained)

        # mode_timer: map increments from map_mode_timer computed above; arena also increments; level increments
        mode_timer_next = state_after_spawn.mode_timer + 1

        # ── 23. Level respawn (jnp.where to avoid shape mismatch in lax.cond) ──
        ship_respawn = make_level_start_state(state_after_spawn.current_level)
        ship_respawn = ship_respawn.replace(x=ship_respawn.x + state_after_spawn.respawn_shift_x)
        empty_pb = create_empty_bullets_64()
        empty_eb = create_empty_bullets_16()
        empty_fc = jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32)
        do_respawn = respawn_now & ~game_over_level
        ship_state_final = jax.tree_util.tree_map(
            lambda r, k: jnp.where(do_respawn, r, k), ship_respawn, ship_after
        )
        bullets = jax.tree_util.tree_map(
            lambda r, k: jnp.where(do_respawn, r, k), empty_pb, bullets
        )
        enemy_bullets = jax.tree_util.tree_map(
            lambda r, k: jnp.where(do_respawn, r, k), empty_eb, enemy_bullets
        )
        fire_cooldown = jnp.where(do_respawn, empty_fc, fire_cooldown)
        cooldown = jnp.where(do_respawn, jnp.int32(0), cooldown)
        ufo = jax.tree_util.tree_map(
            lambda r, k: jnp.where(do_respawn, r, k), make_empty_ufo(), ufo
        )
        ufo_spawn_timer = jnp.where(do_respawn, jnp.int32(UFO_RESPAWN_DELAY_FRAMES), ufo_spawn_timer_after_ufo)

        # ── 24. Map-mode: save map_return_* when entering planet ──
        allowed_planets = jnp.any(jnp.stack(
            [pi == SpriteIdx.PLANET1, pi == SpriteIdx.PLANET2,
             pi == SpriteIdx.PLANET3, pi == SpriteIdx.PLANET4, pi == SpriteIdx.REACTOR], 0), axis=0)
        is_reactor_and_destroyed = (pi == int(SpriteIdx.REACTOR)) & state_after_spawn.reactor_destroyed
        allowed_planets = allowed_planets & (~state_after_spawn.planets_cleared_mask) & (~is_reactor_and_destroyed)
        hit_planet = allowed_planets & (dist2 <= (pr * 0.85 + SHIP_RADIUS) ** 2)
        can_enter_planet = is_map & jnp.any(hit_planet) & (~is_crashing_now)
        hit_idx = jnp.argmax(hit_planet.astype(jnp.int32))
        level_id = jax.lax.cond(can_enter_planet, lambda: pid[hit_idx], lambda: jnp.int32(-1))
        map_return_x = jnp.where(can_enter_planet, ship_after.x, state_after_spawn.map_return_x)
        map_return_y = jnp.where(can_enter_planet, ship_after.y, state_after_spawn.map_return_y)
        map_return_vx = jnp.where(can_enter_planet, ship_after.vx, state_after_spawn.map_return_vx)
        map_return_vy = jnp.where(can_enter_planet, ship_after.vy, state_after_spawn.map_return_vy)
        map_return_angle = jnp.where(can_enter_planet, ship_after.angle, state_after_spawn.map_return_angle)
        map_return_angle_idx = jnp.where(can_enter_planet, ship_after.angle_idx, state_after_spawn.map_return_angle_idx)

        # ── 25. Map-mode: arena trigger ──
        ARENA_TRIGGER_RADIUS = SAUCER_RADIUS * 3.0
        hit_to_arena = (
            saucer_final.alive
            & _circle_hit(ship_after.x, ship_after.y, SHIP_RADIUS, saucer_final.x, saucer_final.y, ARENA_TRIGGER_RADIUS)
            & (~is_crashing_now) & is_map
        )
        W_f, H_f = jnp.float32(WINDOW_WIDTH), jnp.float32(WINDOW_HEIGHT)
        ship_approached_from_above = ship_after.y < saucer_final.y
        ship_spawn_y = jnp.where(ship_approached_from_above, H_f * 0.20, H_f * 0.80)
        saucer_spawn_y = jnp.where(ship_approached_from_above, H_f * 0.80, H_f * 0.20)
        arena_ship = ship_after.replace(x=W_f * 0.80, y=ship_spawn_y)
        arena_saucer = saucer_final.replace(
            x=W_f * 0.20, y=saucer_spawn_y,
            vx=jnp.float32(SAUCER_SPEED_ARENA), vy=jnp.float32(0.0),
            hp=jnp.int32(SAUCER_INIT_HP), alive=jnp.array(True), death_timer=jnp.int32(0),
        )
        empty_pb_arena = create_empty_bullets_64()
        empty_eb_arena = create_empty_bullets_16()
        ship_state_final = jax.tree_util.tree_map(
            lambda a, c: jnp.where(hit_to_arena, a, c), arena_ship, ship_state_final
        )
        saucer_final = jax.tree_util.tree_map(
            lambda a, c: jnp.where(hit_to_arena, a, c), arena_saucer, saucer_final
        )
        bullets = jax.tree_util.tree_map(
            lambda e, c: jnp.where(hit_to_arena, e, c), empty_pb_arena, bullets
        )
        enemy_bullets = jax.tree_util.tree_map(
            lambda e, c: jnp.where(hit_to_arena, e, c), empty_eb_arena, enemy_bullets
        )
        map_return_x = jnp.where(hit_to_arena, ship_after.x, map_return_x)
        map_return_y = jnp.where(hit_to_arena, ship_after.y, map_return_y)
        map_return_vx = jnp.where(hit_to_arena, ship_after.vx, map_return_vx)
        map_return_vy = jnp.where(hit_to_arena, ship_after.vy, map_return_vy)
        map_return_angle = jnp.where(hit_to_arena, ship_after.angle, map_return_angle)
        map_return_angle_idx = jnp.where(hit_to_arena, ship_after.angle_idx, map_return_angle_idx)

        # ── 26. Arena: back-to-map restore ──
        reset_signal_arena = (state_after_spawn.crash_timer == 1) & is_arena
        back_to_map = (
            (~(hit_saucer_by_contact | hit_ship_by_enemy_bullet))
            & (~saucer_final.alive) & (saucer_final.death_timer == 0) & is_arena
        )
        restored_ship = ShipState(
            x=state_after_spawn.map_return_x, y=state_after_spawn.map_return_y,
            vx=state_after_spawn.map_return_vx, vy=state_after_spawn.map_return_vy,
            angle=state_after_spawn.map_return_angle, is_thrusting=jnp.array(False),
            rotation_cooldown=jnp.int32(0), angle_idx=state_after_spawn.map_return_angle_idx,
        )
        ship_state_final = jax.tree_util.tree_map(
            lambda r, c: jnp.where(back_to_map, r, c), restored_ship, ship_state_final
        )
        saucer_final = jax.tree_util.tree_map(
            lambda r, c: jnp.where(back_to_map, r, c), make_default_saucer(), saucer_final
        )
        saucer_spawn_timer = jnp.where(
            back_to_map, jnp.int32(SAUCER_RESPAWN_DELAY_FRAMES), saucer_spawn_timer
        )
        mode_after = jnp.where(hit_to_arena, jnp.int32(2), state_after_spawn.mode)

        # ── 27. Level exit_allowed and cleared mask ──
        new_exit_allowed = is_in_reactor | new_reactor_activated | (all_enemies_gone & (~is_in_reactor))
        current_planet_idx = jnp.argmax((state_after_spawn.planets_id == state_after_spawn.current_level).astype(jnp.int32))
        cleared_mask_next = jnp.where(
            reset_level_win,
            state_after_spawn.planets_cleared_mask.at[current_planet_idx].set(True),
            state_after_spawn.planets_cleared_mask,
        )

        # ── 28. Reset signals ──
        # Map: enter planet or crash animation done
        reset_signal_from_map_crash = (state_after_spawn.crash_timer > 0) & (crash_timer_next == 0) & is_map
        final_level_id_map = jnp.where(reset_signal_from_map_crash, jnp.int32(-2), level_id)
        should_reset_map = (can_enter_planet | reset_signal_from_map_crash) & is_map

        # Level: win or respawn-to-map or reactor crash
        reset_reactor_early_exit = is_in_reactor & (~new_reactor_activated) & exited_top & is_level
        should_reset_level = (reset_level_win | win_reactor | reset_from_reactor_crash | reset_reactor_early_exit) & is_level

        # Arena
        should_reset_arena = (reset_signal_arena | back_to_map) & is_arena

        should_reset = should_reset_map | should_reset_level | should_reset_arena
        final_level_id = jnp.where(is_map, final_level_id_map, jnp.int32(-1))

        # ── 29. Assemble final state ──
        final_env_state = state_after_spawn.replace(
            state=ship_state_final,
            bullets=bullets,
            player_bullet_idx=player_bullet_idx,
            cooldown=cooldown,
            enemies=enemies,
            enemy_bullets=enemy_bullets,
            enemy_bullet_idx=enemy_bullet_idx,
            fire_cooldown=fire_cooldown,
            key=final_key,
            score=score_before + score_delta,
            fuel=fuel_next,
            saucer=saucer_final,
            saucer_spawn_timer=saucer_spawn_timer,
            ufo=ufo,
            ufo_spawn_timer=ufo_spawn_timer,
            ufo_bullets=create_empty_bullets_16(),
            fuel_tanks=new_fuel_tanks,
            crash_timer=crash_timer_next,
            shield_active=is_using_shield_tractor,
            mode=jnp.where(back_to_map, jnp.int32(0), mode_after),
            mode_timer=mode_timer_next,
            prev_action=act,
            reactor_timer=new_reactor_timer,
            reactor_activated=new_reactor_activated,
            lives=jnp.where(is_level, final_lives, state_after_spawn.lives),
            done=jnp.where(is_level, game_over_level, jnp.array(False)),
            reactor_destroyed=state_after_spawn.reactor_destroyed | win_reactor,
            planets_cleared_mask=cleared_mask_next,
            exit_allowed=jnp.where(is_level, new_exit_allowed, state_after_spawn.exit_allowed),
            map_return_x=map_return_x,
            map_return_y=map_return_y,
            map_return_vx=map_return_vx,
            map_return_vy=map_return_vy,
            map_return_angle=map_return_angle,
            map_return_angle_idx=map_return_angle_idx,
        )

        obs = _get_observation_from_state(final_env_state, obs_sprite_dims)

        # Info: map/arena returns saucer reward; level returns turret/level/ufo rewards
        all_rewards = jnp.where(
            is_level,
            jnp.array([score_from_enemies, score_from_level_clear, score_from_ufo, jnp.float32(0.0), jnp.float32(0.0)], dtype=jnp.float32),
            jnp.array([jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), reward_saucer, jnp.float32(0.0)], dtype=jnp.float32),
        )
        info = GravitarInfo(
            lives=final_env_state.lives,
            score=final_env_state.score,
            fuel=final_env_state.fuel,
            mode=final_env_state.mode,
            crash_timer=final_env_state.crash_timer,
            done=final_env_state.done,
            current_level=final_env_state.current_level,
            crash=start_crash,
            hit_by_bullet=hit_ship_by_enemy_bullet,
            reactor_crash_exit=reset_from_reactor_crash,
            all_rewards=all_rewards,
        )
        reward = jnp.where(is_level, level_score_delta, map_score_delta)
        return obs, final_env_state, reward, final_env_state.done, info, should_reset, final_level_id

    def _game_is_running(state, act):
        return _unified_game_loop(state, act)

    return jax.lax.cond(
        env_state.done,
        _game_is_over,
        _game_is_running,
        env_state,
        action
    )


@jax.jit
def step_core(
    env_state: EnvState,
    action: int,
    terrain_bank: jnp.ndarray,
    terrain_heightmaps: jnp.ndarray,
    obs_sprite_dims: jnp.ndarray,
):
    return step_core_linear(env_state, action, terrain_bank, terrain_heightmaps, obs_sprite_dims)


@partial(jax.jit, static_argnums=(2,))
def step_full(env_state: EnvState, action: int, env_instance: 'JaxGravitar'):
    """
    Executes a full step of game logic with flat transition routing.
    """
    # 1) core physics and per-mode stepping
    obs, state_after_step, reward, done_core, info, reset_signal, level_target = step_core(
        env_state,
        action,
        env_instance.terrain_bank,
        env_instance.terrain_heightmaps,
        env_instance.obs_sprite_dims,
    )

    # 2) event detection (branch-free boolean math)
    life_lost = state_after_step.lives < env_state.lives
    fuel_depleted = state_after_step.fuel <= 0.0
    is_level_mode = env_state.mode == jnp.int32(1)
    is_reactor_level = env_state.current_level == jnp.int32(4)
    is_arena_mode = env_state.mode == jnp.int32(2)

    is_arena_death = is_arena_mode & (level_target == -1) & (
        state_after_step.saucer.alive | (state_after_step.saucer.death_timer > 0)
    )
    is_death_event = (
        (level_target == -2)
        | info.get("crash", False)
        | info.get("hit_by_bullet", False)
        | info.get("reactor_crash_exit", False)
        | is_arena_death
        | fuel_depleted
        | done_core
    )

    forced_death_reset = life_lost & (~reset_signal) & (~is_level_mode) & (~done_core)
    forced_fuel_reset = fuel_depleted & (~reset_signal) & (~done_core)
    effective_reset = reset_signal | forced_death_reset | forced_fuel_reset | done_core

    force_full_reset = (is_level_mode & (~is_reactor_level)) | fuel_depleted
    lives_after_death = jnp.where(
        force_full_reset | is_level_mode,
        state_after_step.lives,
        state_after_step.lives - 1,
    )
    is_game_over = force_full_reset | (lives_after_death <= 0)
    stay_in_level = is_level_mode & (~is_game_over) & (~is_reactor_level)
    effective_level = jnp.where(forced_death_reset | forced_fuel_reset, jnp.int32(-2), level_target)

    # 3) transition routing as data table (last matching rule wins)
    is_win = effective_reset & (~is_death_event) & (effective_level < 0)
    is_death_to_map = effective_reset & is_death_event & (~stay_in_level)
    is_death_stay = effective_reset & is_death_event & stay_in_level
    is_entering = effective_reset & (effective_level >= 0)
    transition_conditions = jnp.stack([is_win, is_death_to_map, is_death_stay, is_entering], axis=0)
    # Position 0 -> continue, 1..4 -> table-driven transitions in increasing precedence.
    transition_pos = jnp.max(jnp.where(transition_conditions, jnp.arange(1, 5, dtype=jnp.int32), jnp.int32(0)))
    t_idx = jnp.array([0, 4, 3, 2, 1], dtype=jnp.int32)[transition_pos]

    args = (
        obs,
        state_after_step,
        reward,
        info,
        env_state.key,
        effective_level,
        lives_after_death,
        is_game_over,
        effective_reset,
    )

    # 4) flat branch table
    def branch_continue(branch_args):
        obs_b, st_b, rew_b, inf_b, _, lvl_b, _, _, rst_b = branch_args
        st_b = st_b.replace(done=jnp.array(False, dtype=bool))
        return obs_b, st_b, rew_b, jnp.array(False), inf_b.replace(level_cleared=jnp.array(False)), rst_b, lvl_b

    def branch_enter_level(branch_args):
        _, st_b, rew_b, inf_b, key_b, lvl_b, _, _, rst_b = branch_args
        new_key, subkey = jax.random.split(key_b)
        new_obs, new_state = env_instance.reset_level(subkey, lvl_b, st_b)
        new_state = new_state.replace(key=new_key)
        return new_obs, new_state, rew_b, jnp.array(False), inf_b.replace(level_cleared=jnp.array(False)), rst_b, lvl_b

    def branch_death_stay(branch_args):
        _, st_b, rew_b, inf_b, key_b, _, _, _, rst_b = branch_args
        new_key, subkey = jax.random.split(key_b)
        new_obs, new_state = env_instance.reset_level(subkey, st_b.current_level, st_b)
        new_state = new_state.replace(key=new_key, done=jnp.array(False))
        return (
            new_obs,
            new_state,
            rew_b,
            jnp.array(False),
            inf_b.replace(level_cleared=jnp.array(False)),
            rst_b,
            st_b.current_level,
        )

    def branch_death_map(branch_args):
        _, st_b, rew_b, inf_b, key_b, lvl_b, lives_b, game_over_b, rst_b = branch_args
        new_key, subkey = jax.random.split(key_b)

        new_obs, map_state = jax.lax.cond(
            game_over_b,
            lambda: env_instance.reset_map(subkey),
            lambda: env_instance.reset_map(
                subkey,
                lives=lives_b,
                score=st_b.score,
                fuel=st_b.fuel,
                reactor_destroyed=st_b.reactor_destroyed,
                planets_cleared_mask=st_b.planets_cleared_mask,
            ),
        )
        map_state = map_state.replace(key=new_key, done=jnp.array(False))
        return new_obs, map_state, rew_b, game_over_b, inf_b.replace(level_cleared=jnp.array(False)), rst_b, lvl_b

    def branch_win_map(branch_args):
        _, st_b, rew_b, inf_b, key_b, lvl_b, _, _, rst_b = branch_args
        new_key, subkey = jax.random.split(key_b)

        all_planets_cleared = jnp.all(st_b.planets_cleared_mask)
        solar_system_complete = st_b.reactor_destroyed | all_planets_cleared

        final_fuel = st_b.fuel + jnp.where(solar_system_complete, SOLAR_SYSTEM_BONUS_FUEL, 0.0)
        final_lives = st_b.lives + jnp.where(solar_system_complete, SOLAR_SYSTEM_BONUS_LIVES, 0)
        final_score = st_b.score + jnp.where(solar_system_complete, SOLAR_SYSTEM_BONUS_SCORE, 0.0)
        final_reactor = jnp.where(solar_system_complete, jnp.array(False), st_b.reactor_destroyed)
        final_planets = jnp.where(
            solar_system_complete,
            jnp.zeros_like(st_b.planets_cleared_mask),
            st_b.planets_cleared_mask,
        )

        def setup_arena_return():
            return st_b.replace(
                mode=jnp.int32(0),
                key=new_key,
                done=jnp.array(False),
                fuel=final_fuel,
                lives=final_lives,
                score=final_score,
                reactor_destroyed=final_reactor,
                planets_cleared_mask=final_planets,
            )

        def setup_level_return():
            return_x = jnp.where(st_b.current_level == 4, jnp.float32(77.0), st_b.map_return_x)
            return_y = jnp.where(st_b.current_level == 4, jnp.float32(131.0), st_b.map_return_y)
            restored_ship = ShipState(
                x=return_x,
                y=return_y,
                vx=jnp.float32(0.0),
                vy=jnp.float32(0.0),
                angle=jnp.float32(-jnp.pi / 2),
                is_thrusting=jnp.array(False),
                rotation_cooldown=jnp.int32(0),
                angle_idx=jnp.int32(0),
            )
            _, map_state = env_instance.reset_map(
                subkey,
                lives=final_lives,
                score=final_score,
                fuel=final_fuel,
                reactor_destroyed=final_reactor,
                planets_cleared_mask=final_planets,
            )
            return map_state.replace(
                key=new_key,
                state=restored_ship,
                map_return_x=st_b.map_return_x,
                map_return_y=st_b.map_return_y,
            )

        final_state = jax.lax.cond(st_b.mode == 2, setup_arena_return, setup_level_return)
        new_obs = env_instance._get_observation(final_state)
        return new_obs, final_state, rew_b, jnp.array(False), inf_b.replace(level_cleared=jnp.array(True)), rst_b, lvl_b

    return jax.lax.switch(
        t_idx,
        [branch_continue, branch_enter_level, branch_death_stay, branch_death_map, branch_win_map],
        args,
    )

class JaxGravitar(JaxEnvironment):
    def __init__(self, consts: GravitarConstants = None):
        consts = consts or GravitarConstants()
        super().__init__(consts)
        self.obs_shape = (5,)
        self.num_actions = 18

        # ---- Sprites and derived layout from constants (defaults use module tuple) ----
        self.obs_sprites = _resolve_obs_sprites(self.consts)
        self.sprites = self.obs_sprites
        self.obs_sprite_dims = _build_obs_sprite_dims(self.obs_sprites)

        self.planets = _build_default_planets(self.obs_sprites)
        (
            self._jax_layout_default,
            self.jax_sprite_dims,
            self.jax_level_to_terrain,
            self.jax_level_to_bank,
            self.jax_level_offsets,
            self.jax_level_transforms,
        ) = _build_default_level_data(self.obs_sprites)

        self.renderer = GravitarRenderer(
            width=self.consts.WINDOW_WIDTH, height=self.consts.WINDOW_HEIGHT, consts=self.consts
        )

        _tb_key = id(self.obs_sprites)
        if _tb_key not in _TERRAIN_BANK_CACHE:
            _TERRAIN_BANK_CACHE[_tb_key] = self._build_terrain_bank()
        self.terrain_bank = _TERRAIN_BANK_CACHE[_tb_key]
        self.terrain_heightmaps = self._build_heightmaps(self.terrain_bank)

        reactor_override = tuple(self.consts.REACTOR_LEVEL_LAYOUT)
        if reactor_override:
            # Reactor layout override: rebuild jax_layout with the custom level-4 entries.
            num_levels = max(LEVEL_LAYOUTS.keys()) + 1
            max_default_objects = max(len(v) for v in LEVEL_LAYOUTS.values()) if LEVEL_LAYOUTS else 0
            max_objects = max(max_default_objects, len(reactor_override))
            layout_types = np.full((num_levels, max_objects), -1, dtype=np.int32)
            layout_coords_x = np.zeros((num_levels, max_objects), dtype=np.float32)
            layout_coords_y = np.zeros((num_levels, max_objects), dtype=np.float32)
            for level_id, layout_data in LEVEL_LAYOUTS.items():
                level_layout_data = reactor_override if level_id == 4 else layout_data
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
        else:
            self.jax_layout = self._jax_layout_default

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

    def __hash__(self):
        # Instances that share terrain_bank, layout, and sprite table are JIT-equivalent:
        # step_full(static_argnums=2) uses this to avoid recompiling for every new instance.
        return hash(
            (id(self.terrain_bank), id(self.jax_layout["types"]), id(self.obs_sprites))
        )

    def __eq__(self, other):
        if not isinstance(other, JaxGravitar):
            return NotImplemented
        return (
            self.terrain_bank is other.terrain_bank
            and self.jax_layout["types"] is other.jax_layout["types"]
            and self.obs_sprites is other.obs_sprites
        )

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
        return _get_observation_from_state(state, self.obs_sprite_dims)
    

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

    def render(self, env_state: EnvState) -> jnp.ndarray:
        """Renders the state using the pure JAX renderer."""
        return self.renderer.render(env_state)

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
            rotation_cooldown=jnp.int32(0),
            angle_idx=jnp.int32(0)
        )
        px_np, py_np, pr_np, pi_np = self.planets
        ids_np = [SPRITE_TO_LEVEL_ID.get(sprite_idx, -1) for sprite_idx in pi_np]
        final_reactor_destroyed = reactor_destroyed if reactor_destroyed is not None else jnp.array(False)
        final_cleared_mask = planets_cleared_mask if planets_cleared_mask is not None else jnp.zeros_like(
            self.planets[0], dtype=bool)

        env_state = EnvState(
            mode=jnp.int32(0), state=ship_state, bullets=create_empty_bullets_64(),
            player_bullet_idx=jnp.int32(0),
            cooldown=jnp.array(0, dtype=jnp.int32), enemies=create_empty_enemies(),
            fuel_tanks=FuelTanks(x=jnp.zeros(self.consts.MAX_ENEMIES), y=jnp.zeros(self.consts.MAX_ENEMIES), w=jnp.zeros(self.consts.MAX_ENEMIES),
                                 h=jnp.zeros(self.consts.MAX_ENEMIES), sprite_idx=jnp.full(self.consts.MAX_ENEMIES, -1),
                                 active=jnp.zeros(self.consts.MAX_ENEMIES, dtype=bool)),
            enemy_bullets=create_empty_bullets_16(), fire_cooldown=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.int32),
            enemy_bullet_idx=jnp.int32(0),
            key=key, key_alt=key, score=jnp.array(score if score is not None else 0.0, dtype=jnp.float32),
            done=jnp.array(False), lives=jnp.array(lives if lives is not None else self.consts.MAX_LIVES, dtype=jnp.int32),
            fuel=jnp.array(fuel if fuel is not None else self.consts.STARTING_FUEL, dtype=jnp.float32),
            shield_active=jnp.array(False),
            reactor_timer=jnp.int32(0),
            reactor_activated=jnp.array(False),
            crash_timer=jnp.int32(0), planets_px=jnp.array(px_np), planets_py=jnp.array(py_np),
            planets_pr=jnp.array(pr_np), planets_pi=jnp.array(pi_np), planets_id=jnp.array(ids_np),
            current_level=jnp.int32(-1), terrain_sprite_idx=jnp.int32(-1),
            terrain_scale=jnp.array(1.0), 
            terrain_offset=jnp.array([0.0, 0.0]),
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
            map_return_angle_idx=jnp.int32(0),
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

        init_enemies = create_empty_enemies()
        init_tanks = FuelTanks(x=jnp.full((MAX_ENEMIES,), -1.0), y=jnp.full((MAX_ENEMIES,), -1.0),
                               w=jnp.zeros((MAX_ENEMIES,)), h=jnp.zeros((MAX_ENEMIES,)),
                               sprite_idx=jnp.full((MAX_ENEMIES,), -1), active=jnp.zeros((MAX_ENEMIES,), dtype=bool))
        enemies, fuel_tanks, e_idx, t_idx = init_enemies, init_tanks, 0, 0
        terrain_sprite = self.jax_level_to_terrain[level_id]
        tank_w, tank_h = self.jax_sprite_dims[int(SpriteIdx.FUEL_TANK)]
        for i in range(self.jax_layout["types"].shape[1]):
            obj_type = self.jax_layout["types"][level_id, i]
            should_place = obj_type != -1
            orig_idx = jnp.where(obj_type == SpriteIdx.ENEMY_ORANGE_FLIPPED, SpriteIdx.ENEMY_ORANGE, obj_type)
            w, h = self.jax_sprite_dims[orig_idx]
            x = self.jax_layout["coords_x"][level_id, i]
            y = self.jax_layout["coords_y"][level_id, i]
            is_tank = should_place & (obj_type == SpriteIdx.FUEL_TANK)
            is_green_enemy = should_place & (obj_type == SpriteIdx.ENEMY_GREEN)
            spawn_tank = is_tank | is_green_enemy
            spawn_enemy = should_place & (~is_tank)
            spawned_tank_active = is_tank

            enemies = enemies.replace(
                x=enemies.x.at[e_idx].set(jnp.where(spawn_enemy, x, enemies.x[e_idx])),
                y=enemies.y.at[e_idx].set(jnp.where(spawn_enemy, y, enemies.y[e_idx])),
                w=enemies.w.at[e_idx].set(jnp.where(spawn_enemy, w, enemies.w[e_idx])),
                h=enemies.h.at[e_idx].set(jnp.where(spawn_enemy, h, enemies.h[e_idx])),
                sprite_idx=enemies.sprite_idx.at[e_idx].set(jnp.where(spawn_enemy, obj_type, enemies.sprite_idx[e_idx])),
                hp=enemies.hp.at[e_idx].set(jnp.where(spawn_enemy, 1, enemies.hp[e_idx])),
            )
            fuel_tanks = fuel_tanks.replace(
                x=fuel_tanks.x.at[t_idx].set(jnp.where(spawn_tank, x, fuel_tanks.x[t_idx])),
                y=fuel_tanks.y.at[t_idx].set(jnp.where(spawn_tank, y, fuel_tanks.y[t_idx])),
                w=fuel_tanks.w.at[t_idx].set(jnp.where(spawn_tank, tank_w, fuel_tanks.w[t_idx])),
                h=fuel_tanks.h.at[t_idx].set(jnp.where(spawn_tank, tank_h, fuel_tanks.h[t_idx])),
                sprite_idx=fuel_tanks.sprite_idx.at[t_idx].set(
                    jnp.where(spawn_tank, int(SpriteIdx.FUEL_TANK), fuel_tanks.sprite_idx[t_idx])
                ),
                active=fuel_tanks.active.at[t_idx].set(
                    jnp.where(spawn_tank, spawned_tank_active.astype(bool), fuel_tanks.active[t_idx])
                ),
            )
            e_idx = e_idx + spawn_enemy.astype(jnp.int32)
            t_idx = t_idx + spawn_tank.astype(jnp.int32)

        ship_state = make_level_start_state(level_id)
        initial_timer = jnp.where(level_id == 4, 60 * 60, 0)

        env_state = prev_env_state.replace(
            mode=jnp.int32(1), state=ship_state,
            bullets=create_empty_bullets_64(), 
            player_bullet_idx=jnp.int32(0),
            cooldown=jnp.int32(0),
            enemies=enemies,
            fuel_tanks=fuel_tanks,
            shield_active=jnp.array(False),
            enemy_bullets=create_empty_bullets_16(),
            enemy_bullet_idx=jnp.int32(0),
            fire_cooldown=jnp.full((MAX_ENEMIES,), 60, dtype=jnp.int32),
            key=key, crash_timer=jnp.int32(0), current_level=level_id,
            done=jnp.array(False),
            terrain_sprite_idx=terrain_sprite_idx,
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
    def _build_heightmaps(self, terrain_bank: jnp.ndarray) -> jnp.ndarray:
        W, H = WINDOW_WIDTH, WINDOW_HEIGHT
        bg_val = terrain_bank[0, 0, 0]
        page_content = terrain_bank[
            :,
            _TERRAIN_HIT_RMAX:_TERRAIN_HIT_RMAX + H,
            _TERRAIN_HIT_RMAX:_TERRAIN_HIT_RMAX + W,
        ]
        is_ground = page_content != bg_val
        ground_y = jnp.argmax(is_ground, axis=1)
        has_ground = jnp.any(is_ground, axis=1)
        return jnp.where(has_ground, ground_y, H)

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
        bank_array = jnp.array(np.stack(bank, axis=0), dtype=jnp.int32)
        return jnp.pad(bank_array, ((0,0), (_TERRAIN_HIT_RMAX, _TERRAIN_HIT_RMAX), (_TERRAIN_HIT_RMAX, _TERRAIN_HIT_RMAX)), mode='constant', constant_values=self.renderer.jr.TRANSPARENT_ID)


class GravitarRenderer(JAXGameRenderer):
    def __init__(self, width: int = None, height: int = None, consts: GravitarConstants = None,
                 config: render_utils.RendererConfig = None):
        super().__init__()
        self.consts = consts or GravitarConstants()

        if config is None:
            game_h = height if height is not None else self.consts.WINDOW_HEIGHT
            game_w = width if width is not None else self.consts.WINDOW_WIDTH
            self.config = render_utils.RendererConfig(
                game_dimensions=(game_h, game_w),
                channels=3,
                downscale=None,
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        obs_sprites = _resolve_obs_sprites(self.consts)
        sprite_dir = os.path.join(render_utils.get_base_sprite_dir(), "gravitar")

        # Build the flipped orange enemy as procedural data so load_and_setup_assets
        # includes it in the palette scan and downscaling pass.
        orange_raw = obs_sprites[int(SpriteIdx.ENEMY_ORANGE)]
        orange_flipped_data = (
            jnp.array(np.flip(orange_raw, axis=0), dtype=jnp.uint8)
            if orange_raw is not None
            else jnp.zeros((1, 1, 4), dtype=jnp.uint8)
        )

        # Procedural 1×1 black background keeps load_and_setup_assets happy while
        # rendering uses terrain rasters (built below) as the actual background.
        black_bg = jnp.zeros((1, 1, 4), dtype=jnp.uint8)
        black_bg = black_bg.at[0, 0, 3].set(255)

        asset_config = [
            {'name': 'background', 'type': 'background', 'data': black_bg},
            # Ship sprites
            {'name': 'ship_idle',   'type': 'single', 'file': 'spaceship.npy'},
            {'name': 'ship_thrust', 'type': 'single', 'file': 'ship_thrust.npy'},
            {'name': 'ship_crash',  'type': 'single', 'file': 'ship_crash.npy'},
            {'name': 'ship_thrust_back', 'type': 'single', 'file': 'ship_thrust_back.npy'},
            {'name': 'shield',      'type': 'single', 'file': 'shield.npy'},
            {'name': 'ship_bullet', 'type': 'single', 'file': 'ship_bullet.npy'},
            # 16 ship orientations as a group so they get padded to equal size
            {'name': 'ship_orientations', 'type': 'group', 'files': [
                'spaceship.npy',     # 0: N  (= SHIP_IDLE)
                'spaceship_nne.npy', # 1: NNE
                'spaceship_ne.npy',  # 2: NE
                'spaceship_nee.npy', # 3: NEE
                'spaceship_e.npy',   # 4: E
                'spaceship_see.npy', # 5: SEE
                'spaceship_se.npy',  # 6: SE
                'spaceship_sse.npy', # 7: SSE
                'spaceship_s.npy',   # 8: S
                'spaceship_ssw.npy', # 9: SSW
                'spaceship_sw.npy',  # 10: SW
                'spaceship_sww.npy', # 11: SWW
                'spaceship_w.npy',   # 12: W
                'spaceship_nww.npy', # 13: NWW
                'spaceship_nw.npy',  # 14: NW
                'spaceship_nnw.npy', # 15: NNW
            ]},
            # Enemies
            {'name': 'enemy_orange',   'type': 'single', 'file': 'enemy_orange.npy'},
            {'name': 'enemy_green',    'type': 'single', 'file': 'enemy_green.npy'},
            {'name': 'enemy_saucer',   'type': 'single', 'file': 'saucer.npy'},
            {'name': 'enemy_ufo',      'type': 'single', 'file': 'UFO.npy'},
            {'name': 'enemy_crash',    'type': 'single', 'file': 'enemy_crash.npy'},
            {'name': 'saucer_crash',   'type': 'single', 'file': 'saucer_crash.npy'},
            {'name': 'enemy_orange_flipped', 'type': 'procedural', 'data': orange_flipped_data},
            # Bullets
            {'name': 'enemy_bullet',       'type': 'single', 'file': 'enemy_bullet.npy'},
            {'name': 'enemy_green_bullet', 'type': 'single', 'file': 'enemy_green_bullet.npy'},
            # Level objects
            {'name': 'fuel_tank',    'type': 'single', 'file': 'fuel_tank.npy'},
            {'name': 'obstacle',     'type': 'single', 'file': 'obstacle.npy'},
            {'name': 'spawn_loc',    'type': 'single', 'file': 'spawn_location.npy'},
            {'name': 'reactor',      'type': 'single', 'file': 'reactor.npy'},
            {'name': 'reactor_dest',     'type': 'single', 'file': 'reactor_destination.npy'},
            {'name': 'reactor_dest_hit', 'type': 'single', 'file': 'reactor_destination_hit.npy'},
            # Map screen planets / reactor icon
            {'name': 'planet1', 'type': 'single', 'file': 'planet1.npy'},
            {'name': 'planet2', 'type': 'single', 'file': 'planet2.npy'},
            {'name': 'planet3', 'type': 'single', 'file': 'planet3.npy'},
            {'name': 'planet4', 'type': 'single', 'file': 'planet4.npy'},
            # Terrain sprites (included so their colors enter the palette)
            {'name': 'terrain1',      'type': 'single', 'file': 'terrain1.npy'},
            {'name': 'terrain2',      'type': 'single', 'file': 'terrain2.npy'},
            {'name': 'terrain3',      'type': 'single', 'file': 'terrain3.npy'},
            {'name': 'terrain4',      'type': 'single', 'file': 'terrain4.npy'},
            {'name': 'reactor_terr',  'type': 'single', 'file': 'reactor_terrain.npy'},
            # HUD
            {'name': 'hp_ui', 'type': 'single', 'file': 'HP.npy'},
            # Digits 0-9 as a group
            {'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        ]

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_dir)

        SM = self.SHAPE_MASKS
        T = self.jr.TRANSPARENT_ID

        self.digit_masks = SM['digits']  # shape (10, H, W)

        # All ship variants must be padded to the same spatial size so they can be
        # selected via jax.lax.select (requires equal shapes).
        ship_variant_names = ['ship_idle', 'ship_crash', 'ship_thrust']
        all_ship_masks = [SM['ship_orientations'][i] for i in range(16)] + \
                         [SM[n] for n in ship_variant_names]
        oh = max(m.shape[0] for m in all_ship_masks)
        ow = max(m.shape[1] for m in all_ship_masks)

        def _pad_to(mask, h, w):
            ph = h - mask.shape[0]
            pw = w - mask.shape[1]
            return jnp.pad(mask,
                           ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                           constant_values=T)

        # Re-pad orientation group to the unified size
        self.ship_orientations_array = jnp.stack(
            [_pad_to(SM['ship_orientations'][i], oh, ow) for i in range(16)]
        )

        self.padded_ship_idle   = _pad_to(SM['ship_idle'],   oh, ow)
        self.padded_ship_crash  = _pad_to(SM['ship_crash'],  oh, ow)
        self.padded_ship_thrust = _pad_to(SM['ship_thrust'], oh, ow)

        # Sprite stacks indexed by SpriteIdx integer value.
        # Each stack is (max_idx+1, H, W) with transparent fill for unused slots.
        def _build_named_stack(name_map: dict):
            """name_map: {SpriteIdx_value: mask_name_in_SHAPE_MASKS}"""
            masks = {idx: SM[name] for idx, name in name_map.items() if name in SM}
            if not masks:
                return jnp.full((1, 1, 1), T, dtype=jnp.int32)
            max_idx = max(masks)
            mh = max(m.shape[0] for m in masks.values())
            mw = max(m.shape[1] for m in masks.values())
            rows = []
            for i in range(max_idx + 1):
                if i in masks:
                    rows.append(_pad_to(masks[i], mh, mw))
                else:
                    rows.append(jnp.full((mh, mw), T, dtype=jnp.int32))
            return jnp.stack(rows)

        self.map_elements_stack = _build_named_stack({
            int(SpriteIdx.PLANET1):  'planet1',
            int(SpriteIdx.PLANET2):  'planet2',
            int(SpriteIdx.PLANET3):  'planet3',
            int(SpriteIdx.PLANET4):  'planet4',
            int(SpriteIdx.REACTOR):  'reactor',
            int(SpriteIdx.OBSTACLE): 'obstacle',
            int(SpriteIdx.SPAWN_LOC): 'spawn_loc',
        })

        self.enemy_stack = _build_named_stack({
            int(SpriteIdx.ENEMY_ORANGE):         'enemy_orange',
            int(SpriteIdx.ENEMY_GREEN):          'enemy_green',
            int(SpriteIdx.ENEMY_ORANGE_FLIPPED): 'enemy_orange_flipped',
            int(SpriteIdx.ENEMY_CRASH):          'enemy_crash',
        })

        self.enemy_bullet_stack = _build_named_stack({
            int(SpriteIdx.ENEMY_BULLET):       'enemy_bullet',
            int(SpriteIdx.ENEMY_GREEN_BULLET): 'enemy_green_bullet',
        })

        # Build downscaled terrain rasters for rendering.
        # These are (num_banks, render_H, render_W) ID-rasters.
        # Bank 0 = empty (transparent), banks 1-5 = terrain levels.
        self.terrain_rasters = self._build_terrain_rasters(obs_sprites)

    def _build_terrain_rasters(self, obs_sprites: tuple) -> jnp.ndarray:
        """Build terrain ID-rasters at render resolution (respects downscale)."""
        if self.config.downscale:
            rH, rW = self.config.downscale
        else:
            rH, rW = self.config.game_dimensions
        GH, GW = self.config.game_dimensions

        T = self.jr.TRANSPARENT_ID
        empty_page = np.full((rH, rW), T, dtype=np.int32)
        bank = [empty_page]

        BANK_IDX_TO_LEVEL_ID = {v: k for k, v in LEVEL_ID_TO_BANK_IDX.items()}

        terrain_entries = [
            (SpriteIdx.TERRAIN1,     1),
            (SpriteIdx.TERRAIN2,     2),
            (SpriteIdx.TERRAIN3,     3),
            (SpriteIdx.TERRAIN4,     4),
            (SpriteIdx.REACTOR_TERR, 5),
        ]

        for sprite_idx_enum, bank_idx in terrain_entries:
            mask_name = {
                SpriteIdx.TERRAIN1:     'terrain1',
                SpriteIdx.TERRAIN2:     'terrain2',
                SpriteIdx.TERRAIN3:     'terrain3',
                SpriteIdx.TERRAIN4:     'terrain4',
                SpriteIdx.REACTOR_TERR: 'reactor_terr',
            }[sprite_idx_enum]

            # SHAPE_MASKS already contains downscaled ID-masks from the pipeline.
            # Re-derive from the raw sprite so we can position it correctly on the canvas.
            surf = obs_sprites[int(sprite_idx_enum)]
            if surf is None:
                bank.append(empty_page.copy())
                continue

            th, tw = surf.shape[0], surf.shape[1]
            scale = min(GW / tw, GH / th)
            extra = TERRAIN_SCALE_OVERRIDES.get(sprite_idx_enum, 1.0)
            scale *= float(extra)
            sw, sh = int(tw * scale), int(th * scale)

            ox, oy = (GW - sw) // 2, (GH - sh) // 2
            level_id = BANK_IDX_TO_LEVEL_ID.get(bank_idx)
            if level_id is not None:
                lox, loy = LEVEL_OFFSETS.get(level_id, (0, 0))
                ox += lox
                oy += loy

            # Scale sprite to full-game-resolution canvas, then apply color IDs, then
            # downscale to render resolution.
            if surf.shape[0] != sh or surf.shape[1] != sw:
                scale_h = max(1, int(round(sh / surf.shape[0])))
                scale_w = max(1, int(round(sw / surf.shape[1])))
                rgba_scaled = np.repeat(np.repeat(surf, scale_h, axis=0), scale_w, axis=1)[:sh, :sw]
            else:
                rgba_scaled = surf

            id_mask = np.array(self.jr._create_id_mask(
                jnp.array(rgba_scaled, dtype=jnp.uint8), self.COLOR_TO_ID
            ))

            full_canvas = np.full((GH, GW), T, dtype=np.int32)
            dst_x, dst_y = max(ox, 0), max(oy, 0)
            src_x = abs(min(ox, 0))
            src_y = abs(min(oy, 0))
            copy_w = min(GW - dst_x, id_mask.shape[1] - src_x)
            copy_h = min(GH - dst_y, id_mask.shape[0] - src_y)
            if copy_w > 0 and copy_h > 0:
                full_canvas[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
                    id_mask[src_y:src_y + copy_h, src_x:src_x + copy_w]

            if rH != GH or rW != GW:
                render_page = np.array(
                    jax.image.resize(
                        jnp.array(full_canvas, dtype=jnp.float32),
                        (rH, rW), method='nearest'
                    ).astype(jnp.int32)
                )
            else:
                render_page = full_canvas

            bank.append(render_page)

        return jnp.array(np.stack(bank, axis=0), dtype=jnp.int32)

    @partial(jax.jit, static_argnames=('self',))
    def render(self, state: EnvState) -> jnp.ndarray:
        rH, rW = self.terrain_rasters.shape[1], self.terrain_rasters.shape[2]
        empty_bg = jnp.full((rH, rW), self.jr.TRANSPARENT_ID, dtype=jnp.int32)
        bank_idx = jnp.clip(state.terrain_bank_idx, 0, self.terrain_rasters.shape[0] - 1)
        level_bg = self.terrain_rasters[bank_idx]

        frame = jax.lax.select(state.mode == 1, level_bg, empty_bg)

        def render_centered(f_in, x, y, sprite_arr):
            # Sprite dimensions are in render-space; convert to game-space for coordinate math
            # so that render_at_clipped's internal scaling produces correct placement.
            w_game = sprite_arr.shape[1] / self.config.width_scaling
            h_game = sprite_arr.shape[0] / self.config.height_scaling
            cx = jnp.round(x - w_game / 2).astype(jnp.int32)
            cy = jnp.round(y - h_game / 2).astype(jnp.int32)
            return self.jr.render_at_clipped(f_in, cx, cy, sprite_arr)

        # === 1. Draw Map Elements ===
        def draw_map_elements(f):
            empty_bg_local = jnp.full_like(f, self.jr.TRANSPARENT_ID)

            @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
            def render_planet_layer(empty, x, y, pid, is_cleared, reactor_destroyed):
                is_reactor_destroyed = (pid == int(SpriteIdx.REACTOR)) & reactor_destroyed
                should_draw = (pid >= 0) & ~(is_cleared | is_reactor_destroyed)
                safe_pid = jnp.clip(pid, 0, self.map_elements_stack.shape[0] - 1)
                sprite_to_draw = self.map_elements_stack[safe_pid]
                return jax.lax.cond(
                    should_draw,
                    lambda: render_centered(empty, x, y, sprite_to_draw),
                    lambda: empty,
                )

            n = state.planets_pi.shape[0]
            mask_len = state.planets_cleared_mask.shape[0]
            idx = jnp.arange(n, dtype=jnp.int32)
            safe_mask_idx = jnp.clip(idx, 0, jnp.maximum(mask_len - 1, 0))
            is_cleared = (idx < mask_len) & state.planets_cleared_mask[safe_mask_idx]
            layers = render_planet_layer(
                empty_bg_local,
                state.planets_px,
                state.planets_py,
                state.planets_pi,
                is_cleared,
                jnp.full((n,), state.reactor_destroyed),
            )
            return jnp.minimum(f, jnp.min(layers, axis=0))

        frame = jax.lax.cond(state.mode == 0, draw_map_elements, lambda f: f, frame)

        # === 3. Draw Level Actors (Enemies & Tanks) ===
        def draw_level_actors(f):
            empty_bg_local = jnp.full_like(f, self.jr.TRANSPARENT_ID)

            # Fuel Tanks
            sprite_arr_tank = self.SHAPE_MASKS['fuel_tank']
            @partial(jax.vmap, in_axes=(None, 0, 0, 0))
            def render_tanks(empty, x, y, active):
                return jax.lax.cond(
                    active,
                    lambda: render_centered(empty, x, y, sprite_arr_tank),
                    lambda: empty
                )

            tank_layers = render_tanks(
                empty_bg_local,
                state.fuel_tanks.x,
                state.fuel_tanks.y,
                state.fuel_tanks.active,
            )
            f_tanks = jnp.minimum(f, jnp.min(tank_layers, axis=0))

            @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
            def render_enemies(empty, x, y, w, death_timer, sprite_idx):
                is_alive = w > 0
                is_exploding = death_timer > 0
                is_active = is_alive & ~is_exploding
                should_draw = is_active | is_exploding
                final_sprite_id = jnp.where(is_exploding, int(SpriteIdx.ENEMY_CRASH), sprite_idx)

                def _draw():
                    safe_id = jnp.clip(final_sprite_id, 0, self.enemy_stack.shape[0] - 1)
                    sprite_to_draw = self.enemy_stack[safe_id]
                    return render_centered(empty, x, y, sprite_to_draw)

                return jax.lax.cond(should_draw, _draw, lambda: empty)

            enemy_layers = render_enemies(
                empty_bg_local,
                state.enemies.x,
                state.enemies.y,
                state.enemies.w,
                state.enemies.death_timer,
                state.enemies.sprite_idx,
            )
            f_enemies = jnp.minimum(f_tanks, jnp.min(enemy_layers, axis=0))

            # UFO
            ufo = state.ufo
            f_ufo = jax.lax.cond(ufo.alive,
                                 lambda r: render_centered(r, ufo.x, ufo.y, self.SHAPE_MASKS['enemy_ufo']),
                                 lambda r: r, f_enemies)
            f_ufo = jax.lax.cond(ufo.death_timer > 0,
                                 lambda r: render_centered(r, ufo.x, ufo.y, self.SHAPE_MASKS['enemy_crash']),
                                 lambda r: r, f_ufo)
            return f_ufo

        frame = jax.lax.cond(state.mode == 1, draw_level_actors, lambda f: f, frame)

        # === 3.5. Draw Saucer and Reactor Destination ===
        def draw_saucer(f):
            saucer = state.saucer
            f = jax.lax.cond(saucer.alive,
                             lambda r: render_centered(r, saucer.x, saucer.y,
                                                       self.SHAPE_MASKS['enemy_saucer']),
                             lambda r: r, f)
            f = jax.lax.cond(saucer.death_timer > 0,
                             lambda r: render_centered(r, saucer.x, saucer.y,
                                                       self.SHAPE_MASKS['saucer_crash']),
                             lambda r: r, f)
            return f

        frame = jax.lax.cond((state.mode == 0) | (state.mode == 2), draw_saucer, lambda f: f, frame)

        def draw_reactor_destination(f):
            sprite_arr = jax.lax.select(
                state.reactor_activated,
                self.SHAPE_MASKS['reactor_dest_hit'],
                self.SHAPE_MASKS['reactor_dest']
            )
            return render_centered(f, state.reactor_dest_x, state.reactor_dest_y, sprite_arr)

        should_draw_destination = (state.mode == 1) & (
                state.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)) & state.reactor_dest_active
        frame = jax.lax.cond(should_draw_destination, draw_reactor_destination, lambda f: f, frame)

        # === 4. Bullets ===
        def draw_player_bullets(f):
            sprite_arr = self.SHAPE_MASKS['ship_bullet']
            empty_bg_local = jnp.full_like(f, self.jr.TRANSPARENT_ID)

            @partial(jax.vmap, in_axes=(None, 0, 0, 0))
            def render_batch(empty, x, y, alive):
                return jax.lax.cond(
                    alive,
                    lambda: render_centered(empty, x, y, sprite_arr),
                    lambda: empty
                )

            layers = render_batch(empty_bg_local, state.bullets.x, state.bullets.y, state.bullets.alive)
            bullets_layer = jnp.min(layers, axis=0)
            return jnp.minimum(f, bullets_layer)

        def draw_enemy_bullets(f):
            empty_bg_local = jnp.full_like(f, self.jr.TRANSPARENT_ID)

            @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
            def render_batch(empty, x, y, sprite_idx, alive):
                def _draw():
                    safe_id = jnp.clip(sprite_idx, 0, self.enemy_bullet_stack.shape[0] - 1)
                    sprite_to_draw = self.enemy_bullet_stack[safe_id]
                    return render_centered(empty, x, y, sprite_to_draw)
                return jax.lax.cond(alive, _draw, lambda: empty)

            layers = render_batch(
                empty_bg_local,
                state.enemy_bullets.x,
                state.enemy_bullets.y,
                state.enemy_bullets.sprite_idx,
                state.enemy_bullets.alive,
            )
            eb_layer = jnp.min(layers, axis=0)
            return jnp.minimum(f, eb_layer)

        frame = draw_player_bullets(frame)
        frame = draw_enemy_bullets(frame)

        # === 5. Draw the ship ===
        ship_state = state.state
        is_crashing = state.crash_timer > 0
        is_thrusting = ship_state.is_thrusting

        def draw_shield_and_tractor(f):
            f_with_shield = render_centered(f, ship_state.x, ship_state.y, self.SHAPE_MASKS['shield'])

            is_planet_level = state.mode == 1
            is_reactor = state.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)
            can_show_tractor = is_planet_level & ((~is_reactor) | state.allow_tractor_in_reactor)

            def draw_tractor(frame_in):
                TRACTOR_OFFSET = 8.0
                tractor_x = ship_state.x
                tractor_y = ship_state.y + TRACTOR_OFFSET
                return render_centered(frame_in, tractor_x, tractor_y, self.SHAPE_MASKS['ship_thrust_back'])

            return jax.lax.cond(can_show_tractor, draw_tractor, lambda frame_in: frame_in, f_with_shield)

        frame = jax.lax.cond(state.shield_active, draw_shield_and_tractor, lambda f: f, frame)

        def draw_thrust_flame(f):
            THRUST_OFFSET = 5.0
            thrust_x = ship_state.x - jnp.cos(ship_state.angle) * THRUST_OFFSET
            thrust_y = ship_state.y - jnp.sin(ship_state.angle) * THRUST_OFFSET
            return render_centered(f, thrust_x, thrust_y, self.padded_ship_thrust)

        frame = jax.lax.cond(is_thrusting & (~is_crashing), draw_thrust_flame, lambda f: f, frame)

        oriented_ship_mask = self.ship_orientations_array[ship_state.angle_idx]

        ship_sprite = jax.lax.select(is_crashing, self.padded_ship_crash, oriented_ship_mask)
        frame = render_centered(frame, ship_state.x, ship_state.y, ship_sprite)

        # === 6. Draw the HUD ===
        def draw_hud(f):
            rW = self.terrain_rasters.shape[2]
            rH = self.terrain_rasters.shape[1]

            # Fuel
            f = self.jr.render_label(
                f, x=10, y=5,
                digits=self.jr.int_to_digits(state.fuel.astype(jnp.int32), max_digits=5),
                digit_masks=self.digit_masks, spacing=8, max_digits=5
            )

            # Score
            f = self.jr.render_label(
                f, x=rW - 55, y=5,
                digits=self.jr.int_to_digits(state.score.astype(jnp.int32), max_digits=6),
                digit_masks=self.digit_masks, spacing=8, max_digits=6
            )

            # Lives
            f = self.jr.render_indicator(
                f, x=rW - 50, y=17,
                value=state.lives, shape_mask=self.SHAPE_MASKS['hp_ui'],
                spacing=8, max_value=self.consts.MAX_LIVES
            )

            # Reactor Timer
            def draw_reactor_timer(frame_carry):
                seconds_left = state.reactor_timer // 60
                return self.jr.render_label(
                    frame_carry, x=rW // 2 - 8, y=5,
                    digits=self.jr.int_to_digits(seconds_left, max_digits=2),
                    digit_masks=self.digit_masks, spacing=8, max_digits=2
                )

            is_in_reactor = (state.mode == 1) & (state.current_level == 4)
            return jax.lax.cond(is_in_reactor, draw_reactor_timer, lambda fc: fc, f)

        frame = draw_hud(frame)

        return self.jr.render_from_palette(frame, self.PALETTE)

__all__ = ["JaxGravitar", "get_env_and_renderer"]


def get_env_and_renderer():
    env = JaxGravitar()
    # Just instantiate it, or pass in your game resolution as parameters
    renderer = GravitarRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    return env, renderer
