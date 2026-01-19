import os
import jax
import jax.numpy as jnp
from functools import partial

from jax import Array
from jax.scipy import ndimage
import numpy as np
import jaxatari.spaces as spaces
from jaxatari.core import JaxEnvironment
from typing import NamedTuple, Tuple, Dict, Any, Optional
from enum import IntEnum
from jaxatari.renderers import JAXGameRenderer
import jax.debug

"""
    Group members of the Gravitar base: Xusong Yin, Elizaveta Kuznetsova, Li Dai

    Gravitar Rework by Tiago Soares
"""
FORCE_SPRITES = True
WORLD_SCALE = 3.0
# ========== Constants ==========
SPRITE_DIR = os.path.join(os.path.dirname(__file__), "sprites", "gravitar")
SCALE = 1
MAX_BULLETS = 16  # Reduced from 64 for faster compilation
MAX_ENEMIES = 4   # Reduced from 16 for faster compilation
# 18 discrete spaceship action constants
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17

# HUD settings
HUD_HEIGHT = 24
MAX_LIVES = 3
HUD_PADDING = 5
HUD_SHIP_WIDTH = 10
HUD_SHIP_HEIGHT = 12
HUD_SHIP_SPACING = 12
# Pygame window dimensions
WINDOW_WIDTH = 160
WINDOW_HEIGHT = 210

SAUCER_SPAWN_DELAY_FRAMES = 60 * 3
SAUCER_RESPAWN_DELAY_FRAMES = 180 * 3
UFO_RESPAWN_DELAY_FRAMES = 180 * 2
SAUCER_SPEED_MAP = jnp.float32(0.4) / WORLD_SCALE
SAUCER_SPEED_ARENA = jnp.float32(0.4) / WORLD_SCALE
SAUCER_RADIUS = jnp.float32(3.0)
SHIP_RADIUS = jnp.float32(2.0)
TRACTOR_BEAM_RANGE = jnp.float32(15.0)  # Range for tractor beam to pick up fuel tanks
SAUCER_INIT_HP = jnp.int32(1)

# Solar system completion bonus rewards (for destroying reactor OR clearing all planets)
SOLAR_SYSTEM_BONUS_FUEL = 7000.0
SOLAR_SYSTEM_BONUS_LIVES = 2
SOLAR_SYSTEM_BONUS_SCORE = 4000.0

# SAUCER_SCALE              = 2.2
# ENEMY_ORANGE_SCALE = 2.5
# ENEMY_GREEN_SCALE = 2.5
# FUEL_TANK_SCALE = 2.5
# UFO_SCALE = 2.5

PLAYER_BULLET_SPEED = jnp.float32(4) / WORLD_SCALE
SAUCER_EXPLOSION_FRAMES = jnp.int32(60)
SAUCER_FIRE_INTERVAL_FRAMES = jnp.int32(24)
SAUCER_BULLET_SPEED = jnp.float32(2) / WORLD_SCALE
ENEMY_EXPLOSION_FRAMES = jnp.int32(60)
UFO_HIT_RADIUS = jnp.float32(3.0)

SHIP_ANCHOR_X = None
SHIP_ANCHOR_Y = None
DEBUG_DRAW_SHIP_ORIGIN = True
PLAYER_FIRE_COOLDOWN_FRAMES = 30


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


class SpriteIdx(IntEnum):
    # Ship & bullets
    SHIP_IDLE = 0  # spaceship.npy
    SHIP_THRUST = 1  # ship_thrust.npy
    SHIP_BULLET = 2  # ship_bullet.npy

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
    REACTOR_TERR = 16  # reactor_terrant.npy
    TERRANT1 = 17  # terrant1.npy
    TERRANT2 = 18  # terrant2.npy
    TERRANT3 = 19  # terrant_3.npy
    TERRANT4 = 20  # terrant_4.npy

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


TERRANT_SCALE_OVERRIDES = {
    SpriteIdx.TERRANT2: 1,
}

LEVEL_LAYOUTS = {
    # Level 0 (Planet 1)
    0: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (37, 44)},  # 158
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (82, 32)},  # 146     114
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (152, -3)},  # 112
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (22, 71)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (104, 60)},  # 174 114
    ],
    # Level 1 (Planet 2)
    1: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (92 + 32, 16)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (51 + 32, 78)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (8 + 32, 39)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (12 + 32, 58)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (29 + 32, -1)},
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
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (88, 93 - 114 + 48)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (116, 73 - 114 + 51)},
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (122, 180 - 114 + 47)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (76, 126 - 114 + 47)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (19, 162 - 114 + 47)},
    ],
    # Level 4 (Reactor)
    4: [],
}

LEVEL_OFFSETS = {
    0: (0, 30),
    1: (0, 0),
    2: (0, 30),
    3: (0, 30),
    4: (0, 0),
}

SPRITE_TO_LEVEL_ID = {
    int(SpriteIdx.PLANET1): 0,
    int(SpriteIdx.PLANET2): 1,
    int(SpriteIdx.PLANET3): 2,
    int(SpriteIdx.PLANET4): 3,
    int(SpriteIdx.REACTOR): 4,
}

LEVEL_ID_TO_TERRAIN_SPRITE = {
    0: SpriteIdx.TERRANT1,
    1: SpriteIdx.TERRANT2,
    2: SpriteIdx.TERRANT3,
    3: SpriteIdx.TERRANT4,
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
class Bullets(NamedTuple):
    x: jnp.ndarray  # shape(MAX_BULLETS, )
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    alive: jnp.ndarray  # boolean array
    sprite_idx: jnp.ndarray  # sprite index for each bullet (for different bullet types)


# ========== Enemies States ==========
# Initializes the state of enemies
class Enemies(NamedTuple):
    x: jnp.ndarray  # shape (MAX_ENEMIES,)
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    vx: jnp.ndarray
    sprite_idx: jnp.ndarray
    death_timer: jnp.ndarray
    hp: jnp.ndarray


# ========== Ship State ==========
class ShipState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    angle: jnp.ndarray
    is_thrusting: jnp.ndarray  # Boolean flag to track if ship is actively thrusting


# ========== Saucer State ==========
class SaucerState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    hp: jnp.ndarray
    alive: jnp.ndarray
    death_timer: jnp.ndarray


# ========== UFO State ==========
class UFOState(NamedTuple):
    x: jnp.ndarray  # f32
    y: jnp.ndarray  # f32
    vx: jnp.ndarray  # f32
    vy: jnp.ndarray  # f32
    hp: jnp.ndarray  # i32
    alive: jnp.ndarray  # bool
    death_timer: jnp.ndarray


# ========== FuelTanks State ==========
class FuelTanks(NamedTuple):
    x: jnp.ndarray  # (MAX_ENEMIES,)
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    sprite_idx: jnp.ndarray
    active: jnp.ndarray  # A boolean array to indicate if it's still active


# ========== Env State ==========
class EnvState(NamedTuple):
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
    terrain_sprite_idx: jnp.ndarray  # int32, terrain sprite for the current level (TERRANT* / REACTOR_TERR)
    terrain_mask: jnp.ndarray  # (Hmask, Wmask) bool/uint8
    terrain_scale: jnp.ndarray  # float32, rendering scale factor
    terrain_offset: jnp.ndarray  # (2,) float32, screen-top-left offset [ox, oy]

    terrain_bank: jnp.ndarray  # uint8，shape (B, H, W)
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
    prev_action: jnp.ndarray  # int32, previous action taken

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


# Maps planet sprite indices to terrain bank indices (0=empty, 1..4 correspond to TERRANT1..4)
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
def map_planet_to_terrant(planet_sprite_idx: jnp.ndarray) -> jnp.ndarray:
    P1 = jnp.int32(int(SpriteIdx.PLANET1))
    P2 = jnp.int32(int(SpriteIdx.PLANET2))
    P3 = jnp.int32(int(SpriteIdx.PLANET3))
    P4 = jnp.int32(int(SpriteIdx.PLANET4))
    PR = jnp.int32(int(SpriteIdx.REACTOR))

    T1 = jnp.int32(int(SpriteIdx.TERRANT1))
    T2 = jnp.int32(int(SpriteIdx.TERRANT2))
    T3 = jnp.int32(int(SpriteIdx.TERRANT3))
    T4 = jnp.int32(int(SpriteIdx.TERRANT4))
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
    path = os.path.join(SPRITE_DIR, f"{name_wo_ext}.npy")
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
        SpriteIdx.SHIP_IDLE: "spaceship",
        SpriteIdx.SHIP_THRUST: "ship_thrust",
        SpriteIdx.SHIP_BULLET: "ship_bullet",
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
        SpriteIdx.REACTOR_TERR: "reactor_terrant",
        SpriteIdx.TERRANT1: "terrant1",
        SpriteIdx.TERRANT2: "terrant2",
        SpriteIdx.TERRANT3: "terrant_3",
        SpriteIdx.TERRANT4: "terrant_4",
        SpriteIdx.PLANET1: "planet1",
        SpriteIdx.PLANET2: "planet2",
        SpriteIdx.PLANET3: "planet3",
        SpriteIdx.PLANET4: "planet4",
        SpriteIdx.REACTOR_DEST: "reactor_destination",
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
    spawn_x = jnp.array(WINDOW_WIDTH * 0.50, dtype=jnp.float32)
    spawn_y = jnp.array(WINDOW_HEIGHT * 0.56, dtype=jnp.float32)

    return EnvState(
        mode=jnp.int32(1),
        state=ShipState(
            x=spawn_x,
            y=spawn_y,
            vx=jnp.array(0.0),
            vy=jnp.array(0.0),
            angle=jnp.array(-jnp.pi / 2),
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
        fuel=jnp.array(10000.0, dtype=jnp.float32),
        current_level=jnp.int32(-1),
        terrain_sprite_idx=jnp.int32(-1),
        terrain_mask=jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=jnp.uint8),
        terrain_scale=jnp.array(1.0),
        terrain_offset=jnp.array([0.0, 0.0]),
        terrain_bank=jnp.zeros((6, WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=jnp.uint8), 
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
        prev_action=jnp.int32(0),
    )


@jax.jit
def make_level_start_state(level_id: int) -> ShipState:
    START_Y = jnp.float32(30.0)
    REACTOR_START_Y = jnp.float32(60.0)  # Lower spawn point for reactor

    x = jnp.array(WINDOW_WIDTH / 2, dtype=jnp.float32)
    y = jnp.array(START_Y, dtype=jnp.float32)

    angle = jnp.array(-jnp.pi / 2, dtype=jnp.float32)  # Pointing up for normal levels
    angle_down = jnp.array(jnp.pi / 2, dtype=jnp.float32)  # Pointing down for reactor

    is_reactor = (jnp.asarray(level_id, dtype=jnp.int32) == 4)
    x = jnp.where(is_reactor, x - 60.0, x)
    y = jnp.where(is_reactor, REACTOR_START_Y, y)
    angle = jnp.where(is_reactor, angle_down, angle)

    return ShipState(x=x, y=y, vx=jnp.float32(0.0), vy=jnp.float32(0.0), angle=angle, is_thrusting=jnp.array(False))


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

    RMAX = 16
    dx = jnp.arange(-RMAX, RMAX + 1, dtype=jnp.int32)
    dy = jnp.arange(-RMAX, RMAX + 1, dtype=jnp.int32)
    xs = jnp.clip(xi + dx, 0, W - 1)
    ys = jnp.clip(yi + dy, 0, H - 1)

    bi = jnp.clip(env_state.terrain_bank_idx, 0, env_state.terrain_bank.shape[0] - 1)
    page = env_state.terrain_bank[bi]

    patch = page[ys[:, None], xs[None, :]]

    dxf, dyf = dx.astype(jnp.float32), dy.astype(jnp.float32)
    dist2 = dyf[:, None] ** 2 + dxf[None, :] ** 2

    r_eff = jnp.minimum(jnp.float32(radius), jnp.float32(RMAX))
    mask = dist2 <= (r_eff ** 2)
    is_not_black = jnp.sum(patch, axis=-1) > 0

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
              allow_exit_top: jnp.ndarray = jnp.bool_(False)) -> ShipState:
    # --- Track thrusting state for rendering ---
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])  # UP, UPRIGHT, UPLEFT, UPFIRE, UPRIGHTFIRE, UPLEFTFIRE
    is_thrusting_now = jnp.isin(action, thrust_actions) & (fuel > 0.0)
    
    # --- Physics Parameters ---
    rotation_speed = 0.2 / WORLD_SCALE
    thrust_power = 0.05 / WORLD_SCALE  # Increased from 0.03 to better overcome gravity
    gravity = 0.008 / WORLD_SCALE
    max_speed = 1.0 / WORLD_SCALE

    # 0.0 = full stop on collision (inelastic)
    # 1.0 = perfect bounce (elastic)
    # We start with a softer value, e.g., 0.5 (retains 50% energy)
    bounce_damping = 0.5

    # --- 1. Initialize velocity variables for this frame ---
    #     we get the initial velocity from the state
    vx = state.vx
    vy = state.vy

    # --- 2. Rotation Logic ---
    rotate_right_actions = jnp.array([3, 6, 8, 11, 14, 16])
    rotate_left_actions = jnp.array([4, 7, 9, 12, 15, 17])
    right = jnp.isin(action, rotate_right_actions)
    left = jnp.isin(action, rotate_left_actions)

    # In a Y-down coordinate system:
    # Turn left -> decrease angle value
    angle = jnp.where(left, state.angle - rotation_speed, state.angle)
    # Turn right -> increase angle value
    angle = jnp.where(right, angle + rotation_speed, angle)

    # --- 3. Thrust Calculation ---
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])
    down_thrust_actions = jnp.array([5, 8, 9, 13, 16, 17])

    thrust_pressed = jnp.isin(action, thrust_actions)
    down_pressed = jnp.isin(action, down_thrust_actions)

    can_thrust = fuel > 0.0

    # Forward thrust (vector addition), controlled by the UP key
    vx = jnp.where(thrust_pressed & can_thrust, vx + jnp.cos(angle) * thrust_power, vx)
    vy = jnp.where(thrust_pressed & can_thrust, vy + jnp.sin(angle) * thrust_power, vy)

    # Reverse thrust (vector subtraction), controlled by the DOWN key
    vx = jnp.where(down_pressed & can_thrust, vx - jnp.cos(angle) * thrust_power, vx)
    vy = jnp.where(down_pressed & can_thrust, vy - jnp.sin(angle) * thrust_power, vy)

    # Apply gravity based on mode and terrain
    # Map mode (terrain_bank_idx == 0): pull toward sun
    # Terrant2 (bank_idx == 2): pull toward center (radial gravity)
    # Other planets: pull downward
    is_map_mode = (terrain_bank_idx == 0)
    is_terrant2 = (terrain_bank_idx == 2)
    
    # Sun position (the OBSTACLE sprite: 57% width, 38% height)
    sun_x = window_size[0] * 0.57
    sun_y = window_size[1] * 0.38
    
    # Calculate direction to sun (for map mode)
    dx_to_sun = sun_x - state.x
    dy_to_sun = sun_y - state.y
    dist_to_sun = jnp.sqrt(dx_to_sun**2 + dy_to_sun**2)
    dist_to_sun = jnp.maximum(dist_to_sun, 1.0)  # Avoid division by zero
    
    # Gravity magnitude (stronger when closer to sun)
    gravity_strength = gravity * (100.0 / dist_to_sun)  # Inverse distance law
    gravity_strength = jnp.clip(gravity_strength, 0.0, gravity * 3.0)  # Cap maximum gravity
    
    # Level center for terrant2's radial gravity
    level_center_x = window_size[0] / 2.0
    level_center_y = window_size[1] / 2.0
    
    # Calculate direction to level center (for terrant2)
    dx_to_center = level_center_x - state.x
    dy_to_center = level_center_y - state.y
    dist_to_center = jnp.sqrt(dx_to_center**2 + dy_to_center**2)
    dist_to_center = jnp.maximum(dist_to_center, 1.0)
    
    # Radial gravity strength for terrant2
    radial_gravity_strength = gravity * (50.0 / dist_to_center)
    radial_gravity_strength = jnp.clip(radial_gravity_strength, 0.0, gravity * 2.0)
    
    # Apply gravitational pull based on terrain type
    # Map mode: toward sun
    # Terrant2: toward center (radial)
    # Other planets: downward only
    vx = jnp.where(is_map_mode, 
                   vx + (dx_to_sun / dist_to_sun) * gravity_strength,
                   jnp.where(is_terrant2,
                            vx + (dx_to_center / dist_to_center) * radial_gravity_strength,
                            vx))
    vy = jnp.where(is_map_mode,
                   vy + (dy_to_sun / dist_to_sun) * gravity_strength,
                   jnp.where(is_terrant2,
                            vy + (dy_to_center / dist_to_center) * radial_gravity_strength,
                            vy + gravity))

    # --- 4. Apply maximum speed limit ---
    speed_sq = vx ** 2 + vy ** 2

    def cap_velocity(v_tuple):
        v_x, v_y, spd_sq = v_tuple
        speed = jnp.sqrt(spd_sq)
        scale = max_speed / speed

        return v_x * scale, v_y * scale

    def no_op(v_tuple):
        return v_tuple[0], v_tuple[1]

    vx, vy = jax.lax.cond(
        speed_sq > max_speed ** 2,
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
    return ShipState(x=final_x, y=final_y, vx=final_vx, vy=final_vy, angle=normalized_angle, is_thrusting=is_thrusting_now)


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
        hp=SAUCER_INIT_HP, alive=jnp.array(True),
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
    vx = jnp.where(jnp.abs(dx) < speed, 0.0, vx)  # Stop if close enough
    
    # Keep y fixed at reactor height
    return s._replace(x=s.x + vx, y=reactor_y, vx=vx, vy=0.0)


@jax.jit
def _saucer_fire_one(sauc: SaucerState,
                     ship_x: jnp.ndarray,
                     ship_y: jnp.ndarray,
                     prev_enemy_bullets: Bullets,
                     mode_timer: jnp.ndarray,
                     ) -> Bullets:
    can_fire = sauc.alive & ((mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0) \
               & (_bullets_alive_count(prev_enemy_bullets) < jnp.int32(1))

    def do_fire(_):
        merged = _fire_single_from_to(
            prev_enemy_bullets,
            sauc.x, sauc.y,
            ship_x, ship_y,
            SAUCER_BULLET_SPEED
        )

        return _enforce_cap_keep_old(merged, cap=1)

    return jax.lax.cond(can_fire, do_fire, lambda _: prev_enemy_bullets, operand=None)


@jax.jit
def _saucer_fire_random(sauc: SaucerState,
                        prev_enemy_bullets: Bullets,
                        mode_timer: jnp.ndarray,
                        key: jnp.ndarray,
                        ) -> Bullets:
    """Saucer fires in random directions with max 2 bullets"""
    can_fire = sauc.alive & ((mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0) \
               & (_bullets_alive_count(prev_enemy_bullets) < jnp.int32(2))

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
        return _enforce_cap_keep_old(merged, cap=2)

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


# ========== Collision Detection ==========
@jax.jit
def check_collision(bullets: Bullets, enemies: Enemies):
    def bullet_hits_enemy(i, carry):  # `carry` is the cumulative result, a boolean array of shape (MAX_BULLETS,)
        x = bullets.x[i]  # x, y are the current bullet coordinates
        y = bullets.y[i]
        alive = bullets.alive[i]

        def check_each_enemy(j, hit):
            # Rectangle bounding box, Enemy's bounding box: (x, x+w), (y, y+h)
            within_x = (x > enemies.x[j]) & (x < enemies.x[j] + enemies.w[j])
            within_y = (y > enemies.y[j]) & (y < enemies.y[j] + enemies.h[j])

            return hit | (within_x & within_y)

        hit_any = jax.lax.fori_loop(0, MAX_ENEMIES, check_each_enemy, False)

        return carry.at[i].set(hit_any & alive)

    hits = jnp.zeros((MAX_BULLETS,), dtype=bool)
    hits = jax.lax.fori_loop(0, MAX_BULLETS, bullet_hits_enemy, hits)

    return hits


# ========== Step Core Map ==========
@jax.jit
def step_core_map(state: ShipState,
                  action: int,
                  window_size: Tuple[int, int],
                  hud_height: int
                  ) -> Tuple[jnp.ndarray, ShipState, float, bool, dict, bool, int]:
    new_state = ship_step(state, action, window_size, hud_height)

    obs = jnp.array([
        new_state.x,
        new_state.y,
        new_state.vx,
        new_state.vy,
        new_state.angle
    ])

    reward = 0.0
    done = False
    info = {}
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
    R_MAX = 8
    r = jnp.int32(jnp.clip(radius, 1.0, float(R_MAX)))

    dx_full = jnp.arange(-R_MAX, R_MAX + 1, dtype=jnp.int32)
    dy_full = jnp.arange(-R_MAX, R_MAX + 1, dtype=jnp.int32)

    DX, DY = jnp.meshgrid(dx_full, dy_full, indexing='xy')
    valid = (jnp.abs(DX) <= r) & (jnp.abs(DY) <= r) & ((DX * DX + DY * DY) <= (r * r))

    mx = jnp.floor(x).astype(jnp.int32)
    my = jnp.floor(y).astype(jnp.int32)

    sx = jnp.clip(mx + DX, 0, W - 1)
    sy = jnp.clip(my + DY, 0, H - 1)

    samples = mask[sy, sx].astype(jnp.uint8)
    samples = jnp.where(valid, samples, 0)

    return jnp.any(samples > 0)


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
    ship_after_move = ship_step(ship_state_before_move, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT, env_state.fuel, env_state.terrain_bank_idx)
    
    # Calculate fuel consumption in map mode
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])
    shield_tractor_actions = jnp.array([5, 8, 9, 13, 16, 17])
    is_thrusting = jnp.isin(actual_action, thrust_actions)
    is_using_shield_tractor = jnp.isin(actual_action, shield_tractor_actions)
    
    FUEL_CONSUME_THRUST = 1.0
    FUEL_CONSUME_SHIELD_TRACTOR = 1.5
    
    fuel_consumed = jnp.where(is_thrusting, FUEL_CONSUME_THRUST, 0.0)
    fuel_consumed += jnp.where(is_using_shield_tractor, FUEL_CONSUME_SHIELD_TRACTOR, 0.0)
    new_fuel = jnp.maximum(0.0, env_state.fuel - fuel_consumed)
    
    # Detect fire button press (not hold) - only fire on transition from not-pressed to pressed
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_fire_pressed = jnp.isin(action, fire_actions)
    was_fire_pressed = jnp.isin(env_state.prev_action, fire_actions)
    fire_just_pressed = is_fire_pressed & (~was_fire_pressed)
    
    can_fire = fire_just_pressed & (env_state.cooldown == 0) & (
                _bullets_alive_count(env_state.bullets) < 2)

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
    enemy_bullets = _saucer_fire_random(sauc_final, new_env.enemy_bullets, mode_timer, fire_key)
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

    # Check if ship is close enough to saucer to trigger arena battle
    # Use a larger trigger radius (3x the collision radius) to match "fly near" behavior
    ARENA_TRIGGER_RADIUS = SAUCER_RADIUS * 3.0
    hit_to_arena = sauc_final.alive & _circle_hit(new_env.state.x, new_env.state.y, SHIP_RADIUS, sauc_final.x,
                                                  sauc_final.y, ARENA_TRIGGER_RADIUS) & (~is_crashing_now)

    def _enter_arena(env):
        W, H = jnp.float32(WINDOW_WIDTH), jnp.float32(WINDOW_HEIGHT)
        # Save current position BEFORE modifying state
        return_x, return_y = env.state.x, env.state.y
        return env._replace(
            mode=jnp.int32(2), mode_timer=jnp.int32(0),
            state=env.state._replace(x=W * 0.20, y=H * 0.50, vx=0.0, vy=0.0),
            saucer=sauc_final._replace(
                x=W * 0.80, y=H * 0.50, vx=-SAUCER_SPEED_ARENA, vy=0.0,
                hp=SAUCER_INIT_HP, alive=True, death_timer=0
            ),
            # Clear all bullets when entering arena to prevent rogue bullets
            bullets=create_empty_bullets_64(),
            enemy_bullets=create_empty_bullets_16(),
            # Save the position for restoration after arena
            map_return_x=return_x,
            map_return_y=return_y
        )

    new_env = jax.lax.cond(hit_to_arena, _enter_arena, lambda e: e, new_env)

    # f) Only signal a Reset when the animation is finished
    reset_signal_from_crash = (env_state.crash_timer > 0) & (crash_timer_next == 0)
    # --- 5. Final Return Values ---
    hit_idx = jnp.argmax(hit_planet.astype(jnp.int32))
    level_id = jax.lax.cond(can_enter_planet, lambda: pid[hit_idx], lambda: -1)
    should_reset = can_enter_planet | reset_signal_from_crash
    final_level_id = jnp.where(reset_signal_from_crash, -2, level_id)

    obs_vector = jnp.array([new_env.state.x, new_env.state.y, new_env.state.vx, new_env.state.vy, new_env.state.angle])
    obs = {'vector': obs_vector}  

    reward_saucer = jnp.where(just_died, jnp.float32(100.0), jnp.float32(0.0))
    reward = reward_saucer
    info = {
        "crash": start_crash,
        "hit_by_bullet": hit_ship_by_bullet,
        "reactor_crash_exit": jnp.array(False),
        "all_rewards": jnp.array([
            jnp.float32(0.0),  # enemies
            jnp.float32(0.0),  # reactor
            jnp.float32(0.0),  # ufo
            reward_saucer,  # saucer_kill
            jnp.float32(0.0),  # no penalty
        ], dtype=jnp.float32),
    }

    new_env = new_env._replace(score=new_env.score + reward, shield_active=is_using_shield_tractor)

    return obs, new_env, reward, jnp.array(False), info, should_reset, final_level_id


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
            ufo=UFOState(x=x0, y=final_y0, vx=vx, vy=0.0, hp=1, alive=True, death_timer=0),
            ufo_spawn_timer=UFO_RESPAWN_DELAY_FRAMES, ufo_home_x=x0, ufo_home_y=final_y0,
            ufo_bullets=create_empty_bullets_16(),
        )

    # UFO spawns in planet levels when timer is 0, but not in reactor (5) or Planet 2 (2)
    can_spawn_ufo = (env_state.mode == 1) & (env_state.ufo_spawn_timer == 0) & (~env_state.ufo.alive) & (env_state.terrain_bank_idx != 5) & (env_state.terrain_bank_idx != 2)
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

    ship_after_move = ship_step(ship_state_before_move, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT, state_after_spawn.fuel, state_after_spawn.terrain_bank_idx, allow_exit_top)
    
    # Detect fire button press (not hold) - only fire on transition from not-pressed to pressed
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_fire_pressed = jnp.isin(action, fire_actions)
    was_fire_pressed = jnp.isin(state_after_spawn.prev_action, fire_actions)
    fire_just_pressed = is_fire_pressed & (~was_fire_pressed)
    
    can_fire_player = fire_just_pressed & (
                state_after_spawn.cooldown == 0) & (_bullets_alive_count(state_after_spawn.bullets) < 2)

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

    FUEL_CONSUME_THRUST = 1.0 
    FUEL_CONSUME_SHIELD_TRACTOR = 1.5
    
    fuel_consumed = jnp.where(is_thrusting, FUEL_CONSUME_THRUST, 0.0)
    fuel_consumed += jnp.where(is_using_shield_tractor, FUEL_CONSUME_SHIELD_TRACTOR, 0.0)

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
    can_use_tractor = is_planet_level & ~is_reactor
    
    # Calculate distance from ship to each tank
    dx = tanks.x - ship.x
    dy = tanks.y - ship.y
    distance_sq = dx * dx + dy * dy
    in_tractor_range = distance_sq <= (TRACTOR_BEAM_RANGE ** 2)
    
    # Pickup happens on direct collision OR when using tractor beam and in range (planet levels only)
    tractor_pickup = can_use_tractor & is_using_shield_tractor & in_tractor_range & tanks.active
    collision_mask = direct_collision | tractor_pickup
    
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
    should_fire_mask = turrets_ready_mask
    any_turret_firing = jnp.any(should_fire_mask)

    # 3. Calculate the cooldown for the "next frame"
    # First, decrement the cooldown for all turrets
    next_frame_cooldown = jnp.maximum(current_fire_cooldown - 1, 0)
    # Then, for turrets that "just" fired, reset their cooldown to random interval (60-120 frames)
    # Use deterministic approach: vary based on position
    base_interval = 60
    varied_interval = base_interval + jnp.int32((enemies.x * 0.5) % 60)  # Varies by position
    next_frame_cooldown = jnp.where(should_fire_mask, varied_interval, next_frame_cooldown)

    # 4. If any turrets are firing, generate new bullets
    def _generate_bullets(_):
        # Spawn bullets at the CENTER of the enemy sprite (not offset)
        ex_center = enemies.x
        ey_center = enemies.y

        # Generate random angles for bullets in 180-degree arc
        # Normal bunkers: shoot upward (away from ground below)
        # Flipped bunkers: shoot downward (away from ground above)
        # Use position + frame counter to create pseudo-random angles
        angle_seed = (enemies.x + enemies.y + jnp.float32(state_after_ufo.mode_timer)) * 0.1
        random_offset = (angle_seed % 1.0) * jnp.pi  # Random value between 0 and π
        
        # Check if enemy is flipped
        is_flipped = (enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE_FLIPPED))
        
        # Normal: angle between -π and 0 (upward hemisphere)
        # Flipped: angle between 0 and π (downward hemisphere)
        random_angle = jnp.where(is_flipped, random_offset, -jnp.pi + random_offset)

        # Bullet speed
        bullet_speed = 1.5 / WORLD_SCALE
        vx = jnp.cos(random_angle) * bullet_speed
        vy = jnp.sin(random_angle) * bullet_speed

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
    fire_cooldown = next_frame_cooldown
    key = current_key
    # === Enemy LOGIC Over===

    # --- 5. Advance All Bullets ---
    bullets = update_bullets(bullets)
    enemy_bullets = update_bullets(enemy_bullets)

    # --- 6. Collision Detection ---
    bullets = _bullets_hit_terrain(state_after_ufo, bullets)
    enemy_bullets = _bullets_hit_terrain(state_after_ufo, enemy_bullets)
    bullets, enemies = check_enemy_hit(bullets, enemies)

    hit_enemy_mask = check_ship_enemy_collisions(ship_after_move, enemies, SHIP_RADIUS)
    enemies = enemies._replace(death_timer=jnp.where(hit_enemy_mask, ENEMY_EXPLOSION_FRAMES, enemies.death_timer))

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
    crashed_on_turret = jnp.any(
        (hit_enemy_types == int(SpriteIdx.ENEMY_ORANGE)) | (hit_enemy_types == int(SpriteIdx.ENEMY_GREEN)))
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

    # c) Score calculation
    w_before_hit = state_after_ufo.enemies.w
    just_killed_mask = (w_before_hit > 0) & (enemies.w == 0)
    is_orange = enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_ORANGE))
    is_green = enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_GREEN))

    k_orange = jnp.sum(just_killed_mask & is_orange).astype(jnp.float32)
    k_green = jnp.sum(just_killed_mask & is_green).astype(jnp.float32)

    score_from_enemies = 250.0 * k_orange + 350.0 * k_green
    score_from_reactor = jnp.where(win_reactor, 500.0, 0.0) 
    ufo_just_died = (state_after_ufo.ufo.alive == False) & (env_state.ufo.alive == True) & (
                state_after_ufo.ufo.death_timer > 0)
    score_from_ufo = jnp.where(ufo_just_died, 100.0, 0.0)
    score_delta = score_from_enemies + score_from_reactor + score_from_ufo

    all_rewards = jnp.array([
        score_from_enemies,
        score_from_reactor,
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
    
    # e) The final Reset signal
    # UFO (rammer) doesn't count as an enemy that must be killed to win
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
        return (ship_respawn, create_empty_bullets_64(), create_empty_bullets_16(), jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32), 0)

    def _keep_state_no_respawn(operands):
        return (ship_after_move, bullets, enemy_bullets, fire_cooldown, cooldown)

    state, bullets, enemy_bullets, fire_cooldown, cooldown = jax.lax.cond(
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

    obs_vector = jnp.array([state.x, state.y, state.vx, state.vy, state.angle])
    obs = {'vector': obs_vector}

    reward = score_delta
    info = {
        "crash": start_crash,
        "hit_by_bullet": hit_by_enemy_bullet,
        "reactor_crash_exit": reset_from_reactor_crash,
        "all_rewards": all_rewards,
    }

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
    ship_after_move = ship_step(ship, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT, env_state.fuel, env_state.terrain_bank_idx)

    # Detect fire button press (not hold) - only fire on transition
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_fire_pressed = jnp.isin(action, fire_actions)
    was_fire_pressed = jnp.isin(env_state.prev_action, fire_actions)
    fire_just_pressed = is_fire_pressed & (~was_fire_pressed)
    
    can_fire = fire_just_pressed & (env_state.cooldown == 0) & (
                _bullets_alive_count(env_state.bullets) < 2)

    bullets = jax.lax.cond(
        can_fire,
        lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, PLAYER_BULLET_SPEED),
        lambda b: b,
        env_state.bullets
    )

    bullets = update_bullets(bullets)
    cooldown = jnp.where(can_fire, PLAYER_FIRE_COOLDOWN_FRAMES, jnp.maximum(env_state.cooldown - 1, 0))

    # --- 3. Saucer Movement and Firing ---
    saucer_after_move = jax.lax.cond(saucer.alive,
                                     lambda s: _update_saucer_seek(s, ship_after_move.x, ship_after_move.y,
                                                                   SAUCER_SPEED_ARENA), lambda s: s, operand=saucer)
    can_shoot_saucer = saucer_after_move.alive & (_bullets_alive_count(env_state.enemy_bullets) < 1) & (
                (env_state.mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0)

    enemy_bullets = jax.lax.cond(can_shoot_saucer,
                                 lambda eb: _fire_single_from_to(eb, saucer_after_move.x, saucer_after_move.y,
                                                                 ship_after_move.x, ship_after_move.y,
                                                                 SAUCER_BULLET_SPEED), lambda eb: eb,
                                 operand=env_state.enemy_bullets)
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
    obs_vector = jnp.array(
        [ship_after_move.x, ship_after_move.y, ship_after_move.vx, ship_after_move.vy, ship_after_move.angle])
    obs = {'vector': obs_vector}
    reward = jnp.where(just_died, 100.0, 0.0)
    info = {
        "crash": start_crash,
        "hit_by_bullet": hit_ship_by_bullet,
        "reactor_crash_exit": jnp.array(False),
        "all_rewards": jnp.array([
            jnp.float32(0.0),  # enemies
            jnp.float32(0.0),  # reactor
            jnp.float32(0.0),  # ufo
            reward,  # saucer_kill
            jnp.float32(0.0),  # penalty
        ], dtype=jnp.float32),
    }

    final_env_state = env_state._replace(
        state=ship_after_move,
        bullets=bullets,
        cooldown=cooldown,
        saucer=saucer_final,
        enemy_bullets=enemy_bullets,
        crash_timer=crash_timer_next,
        mode_timer=env_state.mode_timer + 1,
        score=env_state.score + reward,
        prev_action=action,
    )

    # If it's a "win" exit, return directly to the map and restore ship position
    def _go_to_map_win(env):
        # Restore the ship to its position before entering the arena
        # Keep the ship's original angle and thrusting state
        restored_ship = ShipState(
            x=env.map_return_x,
            y=env.map_return_y,
            vx=jnp.float32(0.0),
            vy=jnp.float32(0.0),
            angle=env.state.angle,
            is_thrusting=jnp.array(False)
        )
        return env._replace(
            mode=jnp.int32(0), 
            state=restored_ship,
            saucer=make_default_saucer(),
            saucer_spawn_timer=jnp.int32(SAUCER_RESPAWN_DELAY_FRAMES)
        )

    final_env_state = jax.lax.cond(back_to_map_signal, _go_to_map_win, lambda e: e, final_env_state)

    # The final `reset` signal is either "crash finished" or "win exit"
    return obs, final_env_state, reward, jnp.array(False), info, reset_signal | back_to_map_signal, jnp.int32(-1)


@jax.jit
def _bullets_hit_terrain(env_state: EnvState, bullets: Bullets) -> Bullets:
    H, W = env_state.terrain_bank.shape[1], env_state.terrain_bank.shape[2]

    bank_idx = jnp.clip(env_state.terrain_bank_idx, 0, env_state.terrain_bank.shape[0] - 1)
    terrain_map = env_state.terrain_bank[bank_idx]

    xi = jnp.clip(jnp.round(bullets.x).astype(jnp.int32), 0, W - 1)
    yi = jnp.clip(jnp.round(bullets.y).astype(jnp.int32), 0, H - 1)

    pixel_colors = terrain_map[yi, xi]

    hit_terrain_mask = jnp.sum(pixel_colors, axis=-1) > 0

    final_hit_mask = bullets.alive & hit_terrain_mask

    return bullets._replace(alive=bullets.alive & ~final_hit_mask)


@jax.jit
def _ufo_ground_safe_y_at(terrain_bank, terrain_bank_idx, xf):
    W, H = WINDOW_WIDTH, WINDOW_HEIGHT
    bank_idx = jnp.clip(terrain_bank_idx, 0, terrain_bank.shape[0] - 1)

    terrain_page = terrain_bank[bank_idx]
    col_x = jnp.clip(xf.astype(jnp.int32), 0, W - 1)

    is_ground_in_col = jnp.sum(terrain_page[:, col_x], axis=-1) > 0
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

    is_dead_now = hit_by_ship | (hp_after_hit <= 0)
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
        info = {
            "crash": jnp.array(False),
            "hit_by_bullet": jnp.array(False),
            "reactor_crash_exit": jnp.array(False),

            "all_rewards": jnp.array([
                jnp.float32(0.0),  # enemies
                jnp.float32(0.0),  # reactor
                jnp.float32(0.0),  # ufo
                jnp.float32(0.0),  # saucer_kill
                jnp.float32(0.0),  # penalty
            ], dtype=jnp.float32),
        }

        # 2. Return a tuple with the same pytree structure as the other branch.
        obs_vector = jnp.array([state.state.x, state.state.y, state.state.vx, state.state.vy, state.state.angle])
        obs = {'vector': obs_vector}

        return obs, state, 0.0, True, info, False, -1

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

            enter_info = {**info, "level_cleared": jnp.array(False)}
            return obs_reset, next_state, reward, jnp.array(False), enter_info, jnp.array(True), level

        # === BRANCH 2: RETURN TO THE MAP ===
        def _return_to_map(_):
            """Handles the transition from a level back to the map (due to win, loss, or crash)."""
            is_a_death_event = (level == -2) | info.get("crash", False) | info.get("hit_by_bullet", False) | info.get(
                "reactor_crash_exit", False)

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
                
                obs_reset, map_state = env_instance.reset_map(
                    subkey_for_reset,
                    lives=final_lives,
                    score=final_score,
                    fuel=final_fuel,
                    reactor_destroyed=final_reactor_destroyed,
                    planets_cleared_mask=final_planets_cleared
                )
                map_state = map_state._replace(key=new_main_key)
                win_info = {**info, "level_cleared": jnp.array(True)}
                return obs_reset, map_state, reward, jnp.array(False), win_info, jnp.array(True), level

            def _on_death(_):
                lives_after_death = current_state.lives - 1
                death_info = {**info, "level_cleared": jnp.array(False)}
                is_game_over = (lives_after_death <= 0)
                new_main_key, subkey_for_reset = jax.random.split(current_state.key)
                obs_reset, map_state = env_instance.reset_map(
                    subkey_for_reset,
                    lives=lives_after_death,
                    score=current_state.score,
                    fuel=current_state.fuel,
                    reactor_destroyed=current_state.reactor_destroyed,
                    planets_cleared_mask=current_state.planets_cleared_mask
                )
                final_map_state = map_state._replace(
                    key=new_main_key,
                    done=is_game_over
                )
                return obs_reset, final_map_state, reward, is_game_over, death_info, jnp.array(True), level

            return jax.lax.cond(is_a_death_event, _on_death, _on_win, operand=None)

        return jax.lax.cond(level >= 0, _enter_level, _return_to_map, operand=None)

    def _no_reset(operands):
        obs, new_env_state, reward, done, info, reset, level = operands
        no_reset_info = {**info, "level_cleared": jnp.array(False)}
        return obs, new_env_state, reward, done, no_reset_info, reset, level

    obs, new_env_state, reward, done, info, reset, level = step_core(env_state, action)
    operands = (obs, new_env_state, reward, done, info, reset, level)
    return jax.lax.cond(reset, _handle_reset, _no_reset, operands)


def get_action_from_key():
    """Placeholder function for key input - returns NOOP since we don't use Pygame input in benchmarks"""
    return NOOP

class JaxGravitar(JaxEnvironment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (5,)
        self.num_actions = 18

        # ---- Resource Loading and JAX Renderer Initialization ----
        self.sprites = load_sprites_tuple()
        self.renderer = GravitarRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

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
        layout = [
            (SpriteIdx.PLANET1, 0.82, 0.18), (SpriteIdx.PLANET2, 0.22, 0.24),
            (SpriteIdx.REACTOR, 0.18, 0.43), (SpriteIdx.SPAWN_LOC, 0.50, 0.56),
            (SpriteIdx.OBSTACLE, 0.57, 0.38), (SpriteIdx.PLANET3, 0.76, 0.76),
            (SpriteIdx.PLANET4, 0.14, 0.88),
        ]
        px, py, pr, pi = [], [], [], []
        for idx, xp, yp in layout:
            cx, cy = xp * WINDOW_WIDTH, yp * WINDOW_HEIGHT
            spr = self.sprites[idx]
            if spr is not None:
                r = 8.0 / WORLD_SCALE if idx == SpriteIdx.OBSTACLE else 0.3 * max(spr.shape[1],
                                                                                  spr.shape[0]) * MAP_SCALE * HITBOX_SCALE
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
        num_levels = max(LEVEL_LAYOUTS.keys()) + 1
        max_objects = max(len(v) for v in LEVEL_LAYOUTS.values()) if LEVEL_LAYOUTS else 0
        layout_types = np.full((num_levels, max_objects), -1, dtype=np.int32)
        layout_coords_x = np.zeros((num_levels, max_objects), dtype=np.float32)
        layout_coords_y = np.zeros((num_levels, max_objects), dtype=np.float32)
        for level_id, layout_data in LEVEL_LAYOUTS.items():
            for i, obj in enumerate(layout_data):
                layout_types[level_id, i] = obj['type']
                layout_coords_x[level_id, i] = obj['coords'][0]
                layout_coords_y[level_id, i] = obj['coords'][1]
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
            extra = TERRANT_SCALE_OVERRIDES.get(terrain_sprite_enum, 1.0)
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

    def _get_observation(self, state: EnvState) -> Dict[str, jnp.ndarray]:
        """
        Extracts the structured observation from the environment state.
        Args:
            state: The current environment state.

        Returns: A dictionary containing the vector observation.
        """
        ship = state.state
        obs_vector = jnp.array([
            ship.x, ship.y, ship.vx, ship.vy, ship.angle
        ], dtype=jnp.float32)

        return {'vector': obs_vector}

    def _get_info(self, state: EnvState, all_rewards: Optional[jnp.ndarray] = None) -> Dict[str, Any]:
        """
        Extracts debugging information from the environment state.
        Args:
            state: The current environment state.
            all_rewards: Optional array of rewards from the last step, if available.

        Returns: A dictionary of information.
        """
        info = {
            "lives": state.lives,
            "score": state.score,
            "fuel": state.fuel,
            "mode": state.mode,
            "crash_timer": state.crash_timer,
            "done": state.done,
            "current_level": state.current_level,
        }

        if all_rewards is not None:
            reward_names = [
                "enemies", "reactor", "ufo",
                "saucer_kill", "penalty"
            ]
            for i, name in enumerate(reward_names):
                if i < len(all_rewards):
                    info[f"reward_{name}"] = all_rewards[i]

        return info

    # === Implement all required abstract methods ===
    def reset(self, key: jnp.ndarray) -> tuple[dict[str, Array], EnvState]:
        """Implements the main reset entry point of the environment."""
        return self.reset_map(key)

    def step(self, env_state: EnvState, action: int):
        """Implements the main step entry point of the environment."""
        obs, ns, reward, done, info, _reset, _level = step_full(env_state, action, self)
        # Convert JAX types to standard Python types for compatibility
        try:
            reward = float(reward.item() if hasattr(reward, "item") else reward)
        except Exception:
            pass
        try:
            done = bool(done.item() if hasattr(done, "item") else done)
        except Exception:
            pass
        jax.debug.print("JaxGravitar.step is returning reward: {x}", x=reward)

        return obs, ns, reward, done, info

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self) -> spaces.Dict: 
        low = jnp.array([0.0, 0.0, -10.0, -10.0, -jnp.pi], dtype=jnp.float32)
        high = jnp.array([float(WINDOW_WIDTH), float(WINDOW_HEIGHT), 10.0, 10.0, jnp.pi], dtype=jnp.float32)

        vector_space = spaces.Box(low=low, high=high, shape=self.obs_shape, dtype=jnp.float32)

        return spaces.Dict({'vector': vector_space})

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=jnp.uint8)

    def obs_to_flat_array(self, obs: Any) -> jnp.ndarray:

        if isinstance(obs, dict):
            leaves, _ = jax.tree_util.tree_flatten(obs)
            return jnp.concatenate([leaf.flatten() for leaf in leaves])
        elif isinstance(obs, tuple):
            leaves, _ = jax.tree_util.tree_flatten(obs)
            return jnp.concatenate([leaf.flatten() for leaf in leaves])
        else:
            return obs.flatten()

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
                  ) -> tuple[dict[str, Array], EnvState]:
        spawn_x = jnp.array(WINDOW_WIDTH * 0.50, dtype=jnp.float32)
        spawn_y = jnp.array(WINDOW_HEIGHT * 0.56, dtype=jnp.float32)

        ship_state = ShipState(
            x=spawn_x,
            y=spawn_y,
            vx=jnp.array(jnp.cos(-jnp.pi / 4) * 0.3, dtype=jnp.float32),
            vy=jnp.array(jnp.sin(-jnp.pi / 4) * 0.3, dtype=jnp.float32),
            angle=jnp.array(-jnp.pi / 2, dtype=jnp.float32),
            is_thrusting=jnp.array(False)
        )
        px_np, py_np, pr_np, pi_np = self.planets
        ids_np = [SPRITE_TO_LEVEL_ID.get(sprite_idx, -1) for sprite_idx in pi_np]
        final_reactor_destroyed = reactor_destroyed if reactor_destroyed is not None else jnp.array(False)
        final_cleared_mask = planets_cleared_mask if planets_cleared_mask is not None else jnp.zeros_like(
            self.planets[0], dtype=bool)

        env_state = EnvState(
            mode=jnp.int32(0), state=ship_state, bullets=create_empty_bullets_64(),
            cooldown=jnp.array(0, dtype=jnp.int32), enemies=create_empty_enemies(),
            fuel_tanks=FuelTanks(x=jnp.zeros(MAX_ENEMIES), y=jnp.zeros(MAX_ENEMIES), w=jnp.zeros(MAX_ENEMIES),
                                 h=jnp.zeros(MAX_ENEMIES), sprite_idx=jnp.full(MAX_ENEMIES, -1),
                                 active=jnp.zeros(MAX_ENEMIES, dtype=bool)),
            enemy_bullets=create_empty_bullets_16(), fire_cooldown=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
            key=key, key_alt=key, score=jnp.array(score if score is not None else 0.0, dtype=jnp.float32),
            done=jnp.array(False), lives=jnp.array(lives if lives is not None else MAX_LIVES, dtype=jnp.int32),
            fuel=jnp.array(fuel if fuel is not None else 10000.0, dtype=jnp.float32),
            shield_active=jnp.array(False),
            reactor_timer=jnp.int32(0),
            reactor_activated=jnp.array(False),
            crash_timer=jnp.int32(0), planets_px=jnp.array(px_np), planets_py=jnp.array(py_np),
            planets_pr=jnp.array(pr_np), planets_pi=jnp.array(pi_np), planets_id=jnp.array(ids_np),
            current_level=jnp.int32(-1), terrain_sprite_idx=jnp.int32(-1),
            terrain_mask=jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=jnp.uint8),
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
            ufo=make_empty_ufo(), ufo_spawn_timer=jnp.int32(0), 
            ufo_home_x=jnp.float32(0.0), 
            ufo_home_y=jnp.float32(0.0),
            ufo_bullets=create_empty_bullets_16(), 
            level_offset=jnp.array([0, 0], dtype=jnp.float32),
            reactor_destroyed=final_reactor_destroyed, 
            planets_cleared_mask=final_cleared_mask,
            exit_allowed=jnp.array(False),
            prev_action=jnp.int32(0),
        )

        # Ensure obs is a JAX array
        obs_vector = jnp.array([
            ship_state.x, ship_state.y, ship_state.vx, ship_state.vy, ship_state.angle
        ], dtype=jnp.float32)

        # Return it inside a dictionary to match the new observation_space
        return {'vector': obs_vector}, env_state

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
                
                # For terrant2 (level 1), coordinates are designed for 160-width but sprite is 96-width
                # Scale coordinates to account for this: multiply by (96/160) for x coords
                terrain_sprite = self.jax_level_to_terrain[level_id]
                is_terrant2 = (terrain_sprite == int(SpriteIdx.TERRANT2))
                coord_x = self.jax_layout["coords_x"][level_id, i]
                coord_y = self.jax_layout["coords_y"][level_id, i]
                # Adjust x coordinate for terrant2's narrower width
                adjusted_coord_x = jnp.where(is_terrant2, coord_x * 0.6, coord_x)  # 96/160 = 0.6
                
                x = ox + coord_x * scale
                y = oy + coord_y * scale
                is_tank = (obj_type == SpriteIdx.FUEL_TANK).astype(jnp.int32)
                new_enemies = enemies_in._replace(x=enemies_in.x.at[e_idx_in].set(jnp.where(is_tank, -1.0, x)),
                                                  y=enemies_in.y.at[e_idx_in].set(jnp.where(is_tank, -1.0, y)),
                                                  w=enemies_in.w.at[e_idx_in].set(jnp.where(is_tank, 0.0, w)),
                                                  h=enemies_in.h.at[e_idx_in].set(jnp.where(is_tank, 0.0, h)),
                                                  sprite_idx=enemies_in.sprite_idx.at[e_idx_in].set(
                                                      jnp.where(is_tank, -1, obj_type)),
                                                  hp=enemies_in.hp.at[e_idx_in].set(jnp.where(is_tank, 0, 1)), )
                new_tanks = tanks_in._replace(x=tanks_in.x.at[t_idx_in].set(jnp.where(is_tank, x, -1.0)),
                                              y=tanks_in.y.at[t_idx_in].set(jnp.where(is_tank, y, -1.0)),
                                              w=tanks_in.w.at[t_idx_in].set(jnp.where(is_tank, w, 0.0)),
                                              h=tanks_in.h.at[t_idx_in].set(jnp.where(is_tank, h, 0.0)),
                                              sprite_idx=tanks_in.sprite_idx.at[t_idx_in].set(
                                                  jnp.where(is_tank, obj_type, -1)),
                                              active=tanks_in.active.at[t_idx_in].set(
                                                  jnp.where(is_tank, True, False)), )
                return new_enemies, new_tanks, e_idx_in + (1 - is_tank), t_idx_in + is_tank

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
            terrain_sprite_idx=terrain_sprite_idx,
            terrain_mask=jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=jnp.uint8),
            terrain_scale=scale,
            terrain_offset=jnp.array([ox, oy]),
            terrain_bank_idx=bank_idx,
            reactor_dest_active=(level_id == 4),
            reactor_dest_x=jnp.float32(95),
            reactor_dest_y=jnp.float32(114),
            mode_timer=jnp.int32(0), ufo=make_empty_ufo(),
            ufo_spawn_timer=jnp.int32(0),
            level_offset=jnp.array(level_offset, dtype=jnp.float32),
            reactor_timer=initial_timer.astype(jnp.int32),
            reactor_activated=jnp.array(False),
            exit_allowed=(level_id == 4),  # Allow exit from start in reactor level
        )

        # Ensure obs is a JAX array
        obs_vector = jnp.array([
            ship_state.x, ship_state.y, ship_state.vx, ship_state.vy, ship_state.angle
        ], dtype=jnp.float32)

        # Return it inside a dictionary
        return {'vector': obs_vector}, env_state

    # --- Helper Methods ---
    def _build_terrain_bank(self) -> jnp.ndarray:
        W, H = WINDOW_WIDTH, WINDOW_HEIGHT
        bank = [np.zeros((H, W, 3), dtype=np.uint8)]
        BANK_IDX_TO_LEVEL_ID = {v: k for k, v in LEVEL_ID_TO_BANK_IDX.items()}

        def sprite_to_mask(idx: int, bank_idx: int) -> np.ndarray:
            surf = self.sprites[SpriteIdx(idx)]
            th, tw = surf.shape[0], surf.shape[1]
            
            # Calculate scale with overrides (terrant2 uses 0.80)
            scale = min(W / tw, H / th)
            extra = TERRANT_SCALE_OVERRIDES.get(SpriteIdx(idx), 1.0)
            scale *= float(extra)
            
            # Calculate scaled dimensions
            sw, sh = int(tw * scale), int(th * scale)
            
            # For terrant2 (narrow sprite), we need to center it within the full window width
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
                rgb_array_hwc = np.repeat(np.repeat(surf[:, :, :3], scale_h, axis=0), scale_w, axis=1)[:sh, :sw]
            else:
                rgb_array_hwc = surf[:, :, :3]

            color_map = np.zeros((H, W, 3), dtype=np.uint8)
            src_w, src_h = rgb_array_hwc.shape[1], rgb_array_hwc.shape[0]
            dst_x, dst_y = max(ox, 0), max(oy, 0)
            src_x = abs(min(ox, 0))
            src_y = abs(min(oy, 0))
            copy_w = min(W - dst_x, src_w - src_x)
            copy_h = min(H - dst_y, src_h - src_y)
            if copy_w > 0 and copy_h > 0:
                color_map[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = rgb_array_hwc[src_y:src_y + copy_h, src_x:src_x + copy_w]
            return color_map

        terrains_to_build = [
            (SpriteIdx.TERRANT1, 1), (SpriteIdx.TERRANT2, 2), (SpriteIdx.TERRANT3, 3),
            (SpriteIdx.TERRANT4, 4), (SpriteIdx.REACTOR_TERR, 5),
        ]
        for sprite_idx, bank_idx in terrains_to_build:
            bank.append(sprite_to_mask(int(sprite_idx), bank_idx))
        return jnp.array(np.stack(bank, axis=0), dtype=np.uint8)


class GravitarRenderer(JAXGameRenderer):
    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT):
        super().__init__()
        self.width = width
        self.height = height

        jax_sprites = _load_and_convert_sprites()

        def _no_op_blit(frame, x, y):
            return frame

        blit_functions = {}
        for sprite_idx, sprite_array_rgba in jax_sprites.items():
            if sprite_array_rgba is None: continue

            def make_blit_func(sprite_data):
                sprite_h, sprite_w, _ = sprite_data.shape
                sprite_rgb = sprite_data[..., :3]
                sprite_alpha = (sprite_data[..., 3] / 255.0)[..., None]

                def _blit_sprite(frame, x, y):
                    start_x = jnp.round(x - sprite_w / 2).astype(jnp.int32)
                    start_y = jnp.round(y - sprite_h / 2).astype(jnp.int32)
                    target_patch = jax.lax.dynamic_slice(frame, (start_y, start_x, 0), (sprite_h, sprite_w, 3))
                    blended_patch = sprite_rgb * sprite_alpha + target_patch * (1 - sprite_alpha)
                    return jax.lax.dynamic_update_slice(frame, blended_patch.astype(jnp.uint8), (start_y, start_x, 0))

                return _blit_sprite

            blit_functions[sprite_idx] = make_blit_func(sprite_array_rgba)

        max_idx = max(jax_sprites.keys()) if jax_sprites else -1
        self.blit_branches = tuple(blit_functions.get(i, _no_op_blit) for i in range(max_idx + 1))

        idle_sprite = jax_sprites.get(int(SpriteIdx.SHIP_IDLE))
        crash_sprite = jax_sprites.get(int(SpriteIdx.SHIP_CRASH))
        thrust_sprite = jax_sprites.get(int(SpriteIdx.SHIP_THRUST))

        # Unify the dimensions of all ship state sprites for use with lax.cond
        if all(s is not None for s in [idle_sprite, crash_sprite, thrust_sprite]):
            max_h = max(s.shape[0] for s in [idle_sprite, crash_sprite, thrust_sprite])
            max_w = max(s.shape[1] for s in [idle_sprite, crash_sprite, thrust_sprite])

            def pad_sprite(sprite, h, w):
                pad_h = (h - sprite.shape[0]) // 2
                pad_w = (w - sprite.shape[1]) // 2
                return jnp.pad(sprite,
                               ((pad_h, h - sprite.shape[0] - pad_h), (pad_w, w - sprite.shape[1] - pad_w), (0, 0)))

            self.padded_ship_idle = pad_sprite(idle_sprite, max_h, max_w)
            self.padded_ship_crash = pad_sprite(crash_sprite, max_h, max_w)
            self.padded_ship_thrust = pad_sprite(thrust_sprite, max_h, max_w)
        else:
            self.padded_ship_idle = self.padded_ship_crash = self.padded_ship_thrust = jnp.zeros((1, 1, 4),
                                                                                                 dtype=jnp.uint8)

    @partial(jax.jit, static_argnames=('self',))
    def render(self, state: EnvState) -> jnp.ndarray:
        H, W = self.height, self.width
        # Start with a black frame for each render cycle.
        frame = jnp.zeros((H, W, 3), dtype=jnp.uint8)

        # 1. Draw map elements
        def draw_map_elements(f):
            def draw_one_planet(i, frame_carry):
                sprite_idx = state.planets_pi[i]
                x, y = state.planets_px[i], state.planets_py[i]

                # Conditions to check before drawing a map object.
                is_planet_cleared = (i < state.planets_cleared_mask.shape[0]) & state.planets_cleared_mask[i]
                is_reactor_and_destroyed = (sprite_idx == int(SpriteIdx.REACTOR)) & state.reactor_destroyed
                should_draw = ~(is_planet_cleared | is_reactor_and_destroyed)

                def perform_blit(fc):
                    # Safely select the correct blit function and draw the sprite.
                    safe_idx = jnp.clip(sprite_idx, 0, len(self.blit_branches) - 1)
                    branches = tuple(lambda op, b=b: b(op[0], op[1], op[2]) for b in self.blit_branches)

                    return jax.lax.switch(safe_idx, branches, (fc, x, y))

                return jax.lax.cond(should_draw, perform_blit, lambda fc: fc, frame_carry)

            return jax.lax.fori_loop(0, state.planets_pi.shape[0], draw_one_planet, f)

        frame = jax.lax.cond(state.mode == 0, draw_map_elements, lambda f: f, frame)

        # === 2. Draw Terrain (only in level mode) ===
        def draw_level_terrain(f):
            # Select the correct pre-rendered terrain map from the bank.
            bank_idx = jnp.clip(state.terrain_bank_idx, 0, state.terrain_bank.shape[0] - 1)
            terrain_map = state.terrain_bank[bank_idx]
            # Draw the terrain map wherever its pixels are not black.
            is_terrain_pixel = jnp.sum(terrain_map, axis=-1) > 0

            return jnp.where(is_terrain_pixel[..., None], terrain_map, f)

        frame = jax.lax.cond(state.mode == 1, draw_level_terrain, lambda f: f, frame)

        # === 3. Draw Level Actors (Enemies, Fuel Tanks, UFO) ===
        def draw_enemies(f):
            def draw_enemy_func(i, current_frame):
                is_alive = state.enemies.w[i] > 0
                is_exploding = state.enemies.death_timer[i] > 0

                sprite_idx = state.enemies.sprite_idx[i]
                x, y = state.enemies.x[i], state.enemies.y[i]

                def perform_blit(sprite_id, frame_in):
                    safe_idx = jnp.clip(sprite_id, 0, len(self.blit_branches) - 1)
                    branches = tuple(lambda op, b=b: b(op[0], op[1], op[2]) for b in self.blit_branches)

                    return jax.lax.switch(safe_idx, branches, (frame_in, x, y))

                # Draw either the alive sprite or nothing.
                frame_alive = jax.lax.cond(is_alive & ~is_exploding, lambda f_in: perform_blit(sprite_idx, f_in),
                                           lambda f_in: f_in, current_frame)
                # On top of that, draw the explosion if the enemy is exploding.
                frame_exploding = jax.lax.cond(is_exploding,
                                               lambda f_in: perform_blit(int(SpriteIdx.ENEMY_CRASH), f_in),
                                               lambda f_in: f_in, frame_alive)

                return frame_exploding

            return jax.lax.fori_loop(0, MAX_ENEMIES, draw_enemy_func, f)

        def draw_fuel_tanks(f):
            def draw_tank_func(i, current_frame):
                # 1. Read data from the state.fuel_tanks field
                is_active = state.fuel_tanks.active[i]
                sprite_idx = state.fuel_tanks.sprite_idx[i]
                x, y = state.fuel_tanks.x[i], state.fuel_tanks.y[i]

                # 2. Define the blit function
                def perform_blit(frame_in):
                    safe_idx = jnp.clip(sprite_idx, 0, len(self.blit_branches) - 1)
                    branches = tuple(lambda op, b=b: b(op[0], op[1], op[2]) for b in self.blit_branches)
                    return jax.lax.switch(safe_idx, branches, (frame_in, x, y))

                # 3. Only draw the tank if it is active
                return jax.lax.cond(is_active, perform_blit, lambda f_in: f_in, current_frame)

            # Iterate through all possible tank slots
            return jax.lax.fori_loop(0, MAX_ENEMIES, draw_tank_func, f)

        def draw_ufo(f):
            ufo = state.ufo

            def draw_alive_ufo(frame_in):
                blit_func = self.blit_branches[int(SpriteIdx.ENEMY_UFO)]
                return blit_func(frame_in, ufo.x, ufo.y)

            def draw_ufo_explosion(frame_in):
                blit_func = self.blit_branches[int(SpriteIdx.ENEMY_CRASH)]
                return blit_func(frame_in, ufo.x, ufo.y)

            # First, attempt to draw the alive UFO
            frame_after_alive = jax.lax.cond(ufo.alive, draw_alive_ufo, lambda f_in: f_in, f)
            # Then, on top of that, attempt to draw the explosion if its timer is active
            final_frame = jax.lax.cond(ufo.death_timer > 0, draw_ufo_explosion, lambda f_in: f_in, frame_after_alive)

            return final_frame

        # Combine all level actor drawing functions into one group.
        def draw_level_actors(f):
            frame_with_enemies = draw_enemies(f)
            frame_with_tanks = draw_fuel_tanks(frame_with_enemies)
            frame_with_ufo = draw_ufo(frame_with_tanks)

            return frame_with_ufo

        # This group is only executed in level mode.
        frame = jax.lax.cond(state.mode == 1, draw_level_actors, lambda f: f, frame)

        # === 3.5. Draw Saucer and Reactor Destination ===
        def draw_saucer(f):
            saucer = state.saucer

            def draw_alive_saucer(frame_in):
                blit_func = self.blit_branches[int(SpriteIdx.ENEMY_SAUCER)]
                return blit_func(frame_in, saucer.x, saucer.y)

            def draw_saucer_explosion(frame_in):
                blit_func = self.blit_branches[int(SpriteIdx.SAUCER_CRASH)]
                return blit_func(frame_in, saucer.x, saucer.y)

            # Similar to UFO, use a chained conditional draw
            frame_after_alive = jax.lax.cond(saucer.alive, draw_alive_saucer, lambda f_in: f_in, f)
            final_frame = jax.lax.cond(saucer.death_timer > 0, draw_saucer_explosion, lambda f_in: f_in,
                                       frame_after_alive)

            return final_frame

        # Saucer is only drawn in map or arena mode.
        frame = jax.lax.cond((state.mode == 0) | (state.mode == 2), draw_saucer, lambda f: f, frame)

        def draw_reactor_destination(f):
            blit_func = self.blit_branches[int(SpriteIdx.REACTOR_DEST)]

            return blit_func(f, state.reactor_dest_x, state.reactor_dest_y)

        # The destination is only drawn under specific conditions.
        should_draw_destination = (state.mode == 1) & (
                    state.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)) & state.reactor_dest_active
        frame = jax.lax.cond(should_draw_destination, draw_reactor_destination, lambda f: f, frame)

        # === 4. Draw Bullets ===
        def draw_bullets_with_sprite_idx(bullets, current_frame):
            """Draw bullets using their individual sprite_idx field."""
            # Pre-fetch the two possible blit functions
            orange_blit = self.blit_branches[int(SpriteIdx.ENEMY_BULLET)]
            green_blit = self.blit_branches[int(SpriteIdx.ENEMY_GREEN_BULLET)]
            
            def draw_one_bullet(i, f):
                x, y = bullets.x[i], bullets.y[i]
                is_alive = bullets.alive[i]
                bullet_sprite = bullets.sprite_idx[i]
                
                # Select blit function based on sprite_idx (green bullet uses different sprite)
                is_green = bullet_sprite == int(SpriteIdx.ENEMY_GREEN_BULLET)
                
                def blit_bullet(frame_in):
                    return jax.lax.cond(
                        is_green,
                        lambda f_in: green_blit(f_in, x, y),
                        lambda f_in: orange_blit(f_in, x, y),
                        frame_in
                    )
                
                return jax.lax.cond(is_alive, blit_bullet, lambda frame_in: frame_in, f)

            return jax.lax.fori_loop(0, bullets.x.shape[0], draw_one_bullet, current_frame)
        
        def draw_bullets_fixed_sprite(bullets, sprite_idx, current_frame):
            """Draw bullets using a fixed sprite (for ship bullets)."""
            blit_func = self.blit_branches[sprite_idx]

            def draw_one_bullet(i, f):
                x, y = bullets.x[i], bullets.y[i]
                is_alive = bullets.alive[i]

                return jax.lax.cond(is_alive, lambda frame_in: blit_func(frame_in, x, y), lambda frame_in: frame_in, f)

            return jax.lax.fori_loop(0, bullets.x.shape[0], draw_one_bullet, current_frame)

        # Draw all types of bullets.
        frame = draw_bullets_fixed_sprite(state.bullets, int(SpriteIdx.SHIP_BULLET), frame)
        frame = draw_bullets_with_sprite_idx(state.enemy_bullets, frame)
        # UFOs don't shoot, no ufo_bullets to render

        # --- 5. Draw the ship ---
        ship_state = state.state
        is_crashing = state.crash_timer > 0
        is_thrusting = ship_state.is_thrusting

        # Select the base ship sprite (crash or idle)
        ship_sprite_data = jax.lax.cond(
            is_crashing,
            lambda _: self.padded_ship_crash,
            lambda _: self.padded_ship_idle,
            operand=None
        )

        # Rotate the ship sprite according to its physical angle.
        # Using order=1 (bilinear) for smoother rotation, reshape=False to maintain size
        angle_deg = jnp.degrees(ship_state.angle) + 90.0
        rotated_ship_rgba = _jax_rotate(ship_sprite_data, angle_deg, reshape=False, order=1)

        # Blit the rotated ship onto the frame
        ship_h, ship_w, _ = rotated_ship_rgba.shape
        ship_rgb = rotated_ship_rgba[..., :3]
        ship_alpha = (rotated_ship_rgba[..., 3] / 255.0)[..., None]

        start_x = jnp.round(ship_state.x - ship_w / 2).astype(jnp.int32)
        start_y = jnp.round(ship_state.y - ship_h / 2).astype(jnp.int32)

        # Safe blitting logic to handle cases where the sprite is partially off-screen
        # (This is a simplified version; for full safety, more complex slicing is needed)
        target_patch = jax.lax.dynamic_slice(frame, (start_y, start_x, 0), (ship_h, ship_w, 3))
        blended_patch = ship_rgb * ship_alpha + target_patch * (1 - ship_alpha)
        frame = jax.lax.dynamic_update_slice(frame, blended_patch.astype(jnp.uint8), (start_y, start_x, 0))
        
        # Layer thrust flame sprite at the back of the ship if thrusting (and not crashing)
        def draw_thrust_flame(f):
            # Calculate position at the back of the ship
            # Back is opposite to forward direction (angle points forward)
            THRUST_OFFSET = 5.0  # Distance from ship center to thrust position
            thrust_x = ship_state.x - jnp.cos(ship_state.angle) * THRUST_OFFSET
            thrust_y = ship_state.y - jnp.sin(ship_state.angle) * THRUST_OFFSET
            
            # Rotate the thrust sprite to match ship angle
            rotated_thrust_rgba = _jax_rotate(self.padded_ship_thrust, angle_deg, reshape=False, order=1)
            
            # Calculate blit position for thrust
            thrust_start_x = jnp.round(thrust_x - ship_w / 2).astype(jnp.int32)
            thrust_start_y = jnp.round(thrust_y - ship_h / 2).astype(jnp.int32)
            
            # Blit the thrust flame at the back position
            thrust_rgb = rotated_thrust_rgba[..., :3]
            thrust_alpha = (rotated_thrust_rgba[..., 3] / 255.0)[..., None]
            
            target_patch_thrust = jax.lax.dynamic_slice(f, (thrust_start_y, thrust_start_x, 0), (ship_h, ship_w, 3))
            blended_thrust = thrust_rgb * thrust_alpha + target_patch_thrust * (1 - thrust_alpha)
            return jax.lax.dynamic_update_slice(f, blended_thrust.astype(jnp.uint8), (thrust_start_y, thrust_start_x, 0))
        
        frame = jax.lax.cond(is_thrusting & (~is_crashing), draw_thrust_flame, lambda f: f, frame)

        # Draw shield (works in all modes) and tractor beam (only in planet surface levels)
        def draw_shield_and_tractor(f):
            # Always draw shield when shield_active is true (works in solar system, planet levels, and reactor)
            shield_blit_func = self.blit_branches[int(SpriteIdx.SHIELD)]
            f_with_shield = shield_blit_func(f, ship_state.x, ship_state.y)
            
            # Check if we should also show tractor beam (only in planet surface levels, not reactor)
            is_planet_level = state.mode == 1
            is_reactor = state.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)
            can_show_tractor = is_planet_level & ~is_reactor
            
            # Draw tractor beam (thrust back sprite) at the back of the ship in planet surface levels only
            def draw_tractor(frame_in):
                # Position tractor beam at the back of the ship, opposite to forward direction
                TRACTOR_OFFSET = 8.0  # Distance from ship center to tractor beam position
                tractor_x = ship_state.x - jnp.cos(ship_state.angle) * TRACTOR_OFFSET
                tractor_y = ship_state.y - jnp.sin(ship_state.angle) * TRACTOR_OFFSET
                
                tractor_blit_func = self.blit_branches[int(SpriteIdx.SHIP_THRUST_BACK)]
                return tractor_blit_func(frame_in, tractor_x, tractor_y)
            
            return jax.lax.cond(can_show_tractor, draw_tractor, lambda frame_in: frame_in, f_with_shield)

        frame = jax.lax.cond(state.shield_active, draw_shield_and_tractor, lambda f: f, frame)
        
        # --- 6. Draw the HUD ---
        def draw_hud(f):
            # --- Common parameters ---
            W, H = self.width, self.height
            RIGHT_MARGIN = 15
            LEFT_MARGIN = 10
            DIGIT_WIDTH = 8
            HP_WIDTH = 8
            Y_TOP_ROW = 5   
            Y_LIVES_ROW = 17 

            def draw_fuel_display(frame_carry):
                fuel_val = state.fuel.astype(jnp.int32)
                digits = jnp.array([(fuel_val // 10 ** (4 - i)) % 10 for i in range(5)])
                def draw_one_fuel_digit(i, f_carry):
                    sprite_idx = digits[i] + int(SpriteIdx.DIGIT_0)
                    x_pos = LEFT_MARGIN + i * DIGIT_WIDTH
                    y_pos = Y_TOP_ROW
                    operand = (f_carry, x_pos, y_pos)
                    branches = tuple(lambda op, branch=b: branch(op[0], op[1], op[2]) for b in self.blit_branches)
                    safe_idx = jnp.clip(sprite_idx, 0, len(branches) - 1)
                    return jax.lax.switch(safe_idx, branches, operand)
                return jax.lax.fori_loop(0, 5, draw_one_fuel_digit, frame_carry)
            
            frame_after_fuel = draw_fuel_display(f)

            def draw_score_display(frame_carry):
                score_val = state.score.astype(jnp.int32)
                digits = jnp.array([(score_val // 10 ** (5 - i)) % 10 for i in range(6)])
                def draw_one_digit(i, f_carry):
                    sprite_idx = digits[i] + int(SpriteIdx.DIGIT_0)
                    x_pos = W - RIGHT_MARGIN - (6 - i) * DIGIT_WIDTH
                    y_pos = Y_TOP_ROW
                    operand = (f_carry, x_pos, y_pos)
                    branches = tuple(lambda op, branch=b: branch(op[0], op[1], op[2]) for b in self.blit_branches)
                    safe_idx = jnp.clip(sprite_idx, 0, len(branches) - 1)
                    return jax.lax.switch(safe_idx, branches, operand)
                return jax.lax.fori_loop(0, 6, draw_one_digit, frame_carry)

            frame_after_score = draw_score_display(frame_after_fuel)

            def draw_lives_display(frame_carry):
                hp_blit_func = self.blit_branches[int(SpriteIdx.HP_UI)]
                def draw_one_life(i, f_carry):
                    is_active = i < state.lives
                    x_pos = W - RIGHT_MARGIN - (MAX_LIVES - i) * HP_WIDTH
                    y_pos = Y_LIVES_ROW
                    return jax.lax.cond(is_active, lambda fc: hp_blit_func(fc, x_pos, y_pos), lambda fc: fc, f_carry)
                return jax.lax.fori_loop(0, MAX_LIVES, draw_one_life, frame_carry)
            
            frame_after_lives = draw_lives_display(frame_after_score)

            def draw_reactor_timer(frame_carry):
                seconds_left = state.reactor_timer // 60
                digits = jnp.array([(seconds_left // 10) % 10, seconds_left % 10])
                center_x = W // 2
                start_x = center_x - DIGIT_WIDTH
                def draw_one_timer_digit(i, f_carry):
                    sprite_idx = digits[i] + int(SpriteIdx.DIGIT_0)
                    x_pos = start_x + i * DIGIT_WIDTH
                    y_pos = Y_TOP_ROW
                    operand = (f_carry, x_pos, y_pos)
                    branches = tuple(lambda op, branch=b: branch(op[0], op[1], op[2]) for b in self.blit_branches)
                    safe_idx = jnp.clip(sprite_idx, 0, len(branches) - 1)
                    return jax.lax.switch(safe_idx, branches, operand)
                return jax.lax.fori_loop(0, 2, draw_one_timer_digit, frame_carry)

            is_in_reactor = (state.mode == 1) & (state.current_level == 4)
            final_frame = jax.lax.cond(is_in_reactor, draw_reactor_timer, lambda fc: fc, frame_after_lives)
            
            return final_frame

        frame = draw_hud(frame)

        return frame


__all__ = ["JaxGravitar", "get_env_and_renderer"]


def get_env_and_renderer():
    env = JaxGravitar()
    # Just instantiate it, or pass in your game resolution as parameters
    renderer = GravitarRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    return env, renderer
