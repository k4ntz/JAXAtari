import os
from functools import partial
from typing import Dict, Any, Tuple

import jax
import jax.image
import jax.numpy as jnp
import jax.tree_util
import chex
from jax import Array
from numpy import ndarray, dtype
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants


FIREACTIONS = jnp.array([
    Action.FIRE, Action.UPFIRE, Action.RIGHTFIRE,
    Action.LEFTFIRE, Action.DOWNFIRE, Action.UPRIGHTFIRE,
    Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
], dtype=jnp.int32)
UPACTIONS = jnp.array([Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE], dtype=jnp.int32)
DOWNACTIONS = jnp.array([Action.DOWN, Action.DOWNRIGHT, Action.DOWNLEFT, Action.DOWNFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE], dtype=jnp.int32)
LEFTACTIONS = jnp.array([Action.LEFT, Action.UPLEFT, Action.DOWNLEFT, Action.LEFTFIRE, Action.UPLEFTFIRE, Action.DOWNLEFTFIRE], dtype=jnp.int32)
RIGHTACTIONS = jnp.array([Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT, Action.RIGHTFIRE, Action.UPRIGHTFIRE, Action.DOWNRIGHTFIRE], dtype=jnp.int32)


def _create_wall_map_from_sprite(sprite_path: str) -> chex.Array:
    """Loads a sprite and creates a binary wall map from it.

    Assumes any non-black pixel (R, G, or B > 0) is a wall (1),
    and only pure black pixels (R=0, G=0, B=0) are walkable space (0).
    """
    sprite_rgba = jnp.load(sprite_path)
    # Ensure RGBA
    if sprite_rgba.ndim == 3 and sprite_rgba.shape[2] == 3:
        alpha = jnp.full(sprite_rgba.shape[:2] + (1,), 255, dtype=jnp.uint8)
        sprite_rgba = jnp.concatenate([sprite_rgba, alpha], axis=2)
    
    target_shape = (210, 160, 4)
    if sprite_rgba.shape != target_shape:
        sprite_rgba = jax.image.resize(sprite_rgba, target_shape, method='nearest').astype(jnp.uint8)

    rgb_channels = sprite_rgba[:, :, :3]
    color_sum = jnp.sum(rgb_channels.astype(jnp.int32), axis=-1)
    wall_map = jnp.where(color_sum > 0, jnp.uint8(1), jnp.uint8(0))
    return wall_map


# Define paths to sprite-based maps for different levels.
SPRITE_MAP_PATHS = {
    0: 'map.npy',  # Main Map
    1: 'room1.npy',
    2: 'room2.npy',
    3: 'room3.npy',
    4: 'room4.npy',
}

# Dynamically generate all wall maps for each world and level.
all_worlds_wall_maps_list = []
base_sprite_path = os.path.join(render_utils.get_base_sprite_dir(), 'venture')

# Iterate through defined worlds (2 worlds).
for world_num in [1, 2]:
    world_suffix = "" if world_num == 1 else str(world_num)
    current_world_level_maps = []
    for level_id in range(len(SPRITE_MAP_PATHS)):
        if level_id == 0:  # Main map sprite naming convention
            sprite_filename = f'map{world_suffix}.npy'
        else:  # Room sprite naming convention (e.g., room1.npy for World 1, room21.npy for World 2 Room 1)
            sprite_filename = f'room{world_suffix}{level_id}.npy'

        full_path = os.path.join(base_sprite_path, sprite_filename)
        current_world_level_maps.append(_create_wall_map_from_sprite(full_path))
    all_worlds_wall_maps_list.append(jnp.stack(current_world_level_maps, axis=0))

# Stack all world wall maps into a single JAX array: (num_worlds, num_levels, height, width).
ALL_WALL_MAPS_PER_WORLD = jnp.stack(all_worlds_wall_maps_list, axis=0)

SPAWN_BOUNDARY_OFFSET_ENTER = 6.0  # Offset player when ENTERING a room.
SPAWN_BOUNDARY_OFFSET_EXIT = 1   # Offset player when EXITING a room (a smaller push).


def _calculate_spawn(rect, push_vector, target_level):
    """Calculates a spawn point offset from a given rectangle and push vector."""
    x, y, w, h = rect
    center_x = x + w / 2
    center_y = y + h / 2

    is_entering_room = (target_level > 0)
    offset = jnp.where(is_entering_room, SPAWN_BOUNDARY_OFFSET_ENTER, SPAWN_BOUNDARY_OFFSET_EXIT)

    spawn_x = center_x + push_vector[0] * (w / 2 + offset)
    spawn_y = center_y + push_vector[1] * (h / 2 + offset)
    return spawn_x, spawn_y


# Define all portal transitions for World 1.
# Each entry specifies a source rectangle, target level, and spawn coordinates.
WORLD1_PORTAL_DEFINITIONS = {
    0: [  # Portals on the Main Map (Level 0)
        {"rect": [20, 60, 4, 4], "to": (1, *_calculate_spawn([28, 100, 4, 8], (1, 0), target_level=1))},
        {"rect": [48, 52, 4, 4], "to": (1, *_calculate_spawn([128, 100, 4, 8], (-1, 0), target_level=1))},
        {"rect": [88, 44, 4, 4], "to": (2, *_calculate_spawn([16, 52, 4, 8], (1, 0), target_level=2))},
        {"rect": [136, 44, 4, 4], "to": (2, *_calculate_spawn([140, 52, 4, 8], (-1, 0), target_level=2))},
        {"rect": [32, 96, 5, 4], "to": (3, *_calculate_spawn([56, 24, 4, 4], (0, 1), target_level=3))},
        {"rect": [60, 148, 4, 4], "to": (3, *_calculate_spawn([140, 148, 4, 8], (-1, 0), target_level=3))},
        {"rect": [108, 120, 5, 4], "to": (4, *_calculate_spawn([60, 76, 4, 4], (0, -1), target_level=4))},
        {"rect": [140, 120, 4, 4], "to": (4, *_calculate_spawn([140, 48, 4, 8], (-1, 0), target_level=4))},
    ],
    1: [  # Portals in Room 1 -> Main Map
        {"rect": [28, 100, 4, 8], "to": (0, *_calculate_spawn([20, 60, 4, 4], (-1, 0), target_level=0))},
        {"rect": [128, 100, 4, 8], "to": (0, *_calculate_spawn([48, 52, 4, 4], (1, 0), target_level=0))},
    ],
    2: [  # Portals in Room 2 -> Main Map
        {"rect": [16, 52, 4, 8], "to": (0, *_calculate_spawn([88, 44, 4, 4], (-1, 0), target_level=0))},
        {"rect": [140, 52, 4, 8], "to": (0, *_calculate_spawn([136, 44, 4, 4], (1, 0), target_level=0))},
    ],
    3: [  # Portals in Room 3 -> Main Map
        {"rect": [56, 24, 4, 4], "to": (0, *_calculate_spawn([32, 96, 4, 4], (0, -1), target_level=0))},
        {"rect": [140, 148, 4, 8], "to": (0, *_calculate_spawn([60, 148, 4, 4], (1, 0), target_level=0))},
    ],
    4: [  # Portals in Room 4 -> Main Map
        {"rect": [60, 76, 4, 4], "to": (0, *_calculate_spawn([108, 120, 4, 4], (0, 1), target_level=0))},
        {"rect": [140, 48, 4, 8], "to": (0, *_calculate_spawn([140, 120, 4, 4], (1, 0), target_level=0))},
    ]
}

# Define all portal transitions for World 2.
WORLD2_PORTAL_DEFINITIONS = {
    0: [  # Portals on the Main Map (Level 0) for World 2
        {"rect": [16, 40, 4, 4], "to": (1, *_calculate_spawn([16, 44, 4, 8], (1, 0), target_level=1))},
        {"rect": [60, 40, 4, 4], "to": (1, *_calculate_spawn([140, 44, 4, 8], (-1, 0), target_level=1))},
        {"rect": [112, 72, 5, 4], "to": (2, *_calculate_spawn([76, 180, 8, 4], (0, -1), target_level=2))},
        {"rect": [80, 108, 5, 4], "to": (3, *_calculate_spawn([76, 180, 8, 4], (0, -1), target_level=3))},
        {"rect": [16, 144, 4, 4], "to": (4, *_calculate_spawn([16, 100, 4, 8], (1, 0), target_level=4))},
        {"rect": [140, 144, 4, 4], "to": (4, *_calculate_spawn([140, 100, 4, 8], (-1, 0), target_level=4))},
    ],
    1: [  # Portals in Room 1 (World 2) -> Main Map (World 2)
        {"rect": [16, 44, 4, 8], "to": (0, *_calculate_spawn([16, 40, 4, 4], (-1, 0), target_level=0))},
        {"rect": [140, 44, 4, 8], "to": (0, *_calculate_spawn([60, 40, 4, 4], (1, 0), target_level=0))},
    ],
    2: [  # Portals in Room 2 (World 2) -> Main Map (World 2)
        {"rect": [76, 180, 8, 4], "to": (0, *_calculate_spawn([112, 72, 4, 4], (0, 1), target_level=0))},
    ],
    3: [  # Portals in Room 3 (World 2) -> Main Map (World 2)
        {"rect": [76, 180, 8, 4], "to": (0, *_calculate_spawn([80, 108, 4, 4], (0, 1), target_level=0))},
    ],
    4: [  # Portals in Room 4 (World 2) -> Main Map (World 2)
        {"rect": [16, 100, 4, 8], "to": (0, *_calculate_spawn([16, 144, 4, 4], (-1, 0), target_level=0))},
        {"rect": [140, 100, 4, 8], "to": (0, *_calculate_spawn([140, 144, 4, 4], (1, 0), target_level=0))},
    ]
}

# Aggregate all portal definitions by world.
ALL_WORLD_PORTAL_DEFINITIONS = {
    1: WORLD1_PORTAL_DEFINITIONS,
    2: WORLD2_PORTAL_DEFINITIONS,
}

# Process portal definitions into a JAX-compatible array for efficient lookup.
# The structure will be (world_idx, level_idx, portal_idx, portal_data).
_JAX_TRANSITIONS_PER_WORLD_LIST = []
ROOM_PORTAL_PADDING = 4.0  # Extra padding for room portals for easier interaction.

for world_id in sorted(ALL_WORLD_PORTAL_DEFINITIONS.keys()):
    world_portal_data = ALL_WORLD_PORTAL_DEFINITIONS[world_id]
    current_world_transitions_list = []
    num_levels = max(world_portal_data.keys()) + 1 if world_portal_data else 5

    for source_level_id in range(num_levels):
        level_portals_data = world_portal_data.get(source_level_id, [])
        level_portals_list = []

        for portal in level_portals_data:
            rect = portal["rect"]
            x, y, w, h = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])

            # Apply padding to room portals.
            is_in_room = (source_level_id > 0)
            if is_in_room:
                x -= ROOM_PORTAL_PADDING
                y -= ROOM_PORTAL_PADDING
                w += 2 * ROOM_PORTAL_PADDING
                h += 2 * ROOM_PORTAL_PADDING

            target_level, spawn_x, spawn_y = portal["to"]
            level_portals_list.append([
                x, y, w, h,  # Portal bounding box (possibly padded)
                float(target_level), float(spawn_x), float(spawn_y)  # Target and spawn
            ])
        current_world_transitions_list.append(level_portals_list)
    _JAX_TRANSITIONS_PER_WORLD_LIST.append(current_world_transitions_list)

# Find the maximum number of portals in any level across all worlds for padding.
max_portals_per_level = 0
for world_list in _JAX_TRANSITIONS_PER_WORLD_LIST:
    if any(world_list):
        max_portals_per_level = max(max_portals_per_level, max(len(p) for p in world_list if p))

# Pad all level portal lists to ensure uniform size.
final_jax_transitions_list = []
for world_list in _JAX_TRANSITIONS_PER_WORLD_LIST:
    padded_world_list = []
    for level_list in world_list:
        padding_needed = max_portals_per_level - len(level_list)
        if padding_needed > 0:
            level_list.extend([[0.0] * 7] * padding_needed)  # Pad with dummy zero data
        padded_world_list.append(level_list)
    final_jax_transitions_list.append(padded_world_list)

# The final JAX-compatible array for all transitions, structured as (world_idx, level_idx, portal_idx, portal_data).
_JAX_TRANSITIONS = jnp.array(final_jax_transitions_list, dtype=jnp.float32)


# Monster configurations for each level across both worlds.
# Global level indices: 0-4 for World 1 (Main + 4 rooms), 5-9 for World 2 (Main + 4 rooms).
_LEVEL_MONSTER_CONFIGS = (
    # World 1
    {"num": 6, "spawns": jnp.array([[10, 36], [60, 77], [54, 127], [110, 74], [10, 126], [150, 127]], dtype=jnp.float32), "is_immortal": jnp.array([False]*6)},
    {"num": 0, "spawns": jnp.empty((0, 2), dtype=jnp.float32), "is_immortal": jnp.empty((0,), dtype=jnp.bool_)},
    {"num": 3, "spawns": jnp.array([[70.0, 50.0], [120.0, 120.0], [130.0, 130.0]], dtype=jnp.float32), "is_immortal": jnp.array([False]*3)},
    {"num": 3, "spawns": jnp.array([[40.0, 80.0], [50.0, 140.0], [100.0, 150.0]], dtype=jnp.float32), "is_immortal": jnp.array([False]*3)},
    {"num": 3, "spawns": jnp.array([[90.0, 40.0], [50.0, 150.0], [120.0, 90.0]], dtype=jnp.float32), "is_immortal": jnp.array([False]*3)},
    # World 2
    {"num": 6, "spawns": jnp.array([[70, 47], [10, 76], [7, 117], [130, 67], [120, 116], [124, 167]], dtype=jnp.float32), "is_immortal": jnp.array([False]*6)},
    {"num": 3, "spawns": jnp.array([[72, 85], [115, 37], [41, 35]], dtype=jnp.float32), "is_immortal": jnp.array([False]*3)},
    {"num": 3, "spawns": jnp.array([[73, 38], [123.0, 59.0], [93.0, 109.0]], dtype=jnp.float32), "is_immortal": jnp.array([False]*3)},
    {"num": 3, "spawns": jnp.array([[61.0, 109.0], [74, 65], [101, 118]], dtype=jnp.float32), "is_immortal": jnp.array([False]*3)},
    {"num": 3, "spawns": jnp.array([[42, 82], [112, 83], [72, 103]], dtype=jnp.float32), "is_immortal": jnp.array([False]*3)},
)

# Pre-calculate total monsters and offsets for efficient indexing.
_TOTAL_MONSTERS = sum(config["num"] for config in _LEVEL_MONSTER_CONFIGS)
_LEVEL_OFFSETS = jnp.cumsum(jnp.array([0] + [c["num"] for c in _LEVEL_MONSTER_CONFIGS], dtype=jnp.int32))
_ALL_MONSTER_SPAWNS = jnp.concatenate([c["spawns"] for c in _LEVEL_MONSTER_CONFIGS]).astype(jnp.float32)
_ALL_MONSTER_IMMORTAL_FLAGS = jnp.concatenate([c["is_immortal"] for c in _LEVEL_MONSTER_CONFIGS])


class MonsterState(struct.PyTreeNode):
    """Holds the dynamic state of all monsters."""
    x: chex.Array
    y: chex.Array
    dx: chex.Array
    dy: chex.Array
    active: chex.Array  # Boolean array indicating active monsters.
    is_immortal: chex.Array  # Boolean array indicating immortal monsters.


class DeadMonsterState(struct.PyTreeNode):
    """Holds the state of recently killed monsters (corpses)."""
    x: chex.Array
    y: chex.Array
    active: chex.Array  # Boolean array indicating active dead monsters.
    lifetime: chex.Array  # Remaining frames until a dead monster despawns.


class ProjectileState(struct.PyTreeNode):
    """Holds the state of the player's projectile."""
    x: chex.Array
    y: chex.Array
    dx: chex.Array
    dy: chex.Array
    active: chex.Array  # Whether the projectile is currently in flight.
    lifetime: chex.Array  # Remaining frames until projectile despawns.


class ChaserState(struct.PyTreeNode):
    """Holds the state of the chaser monster, which appears in rooms after a delay."""
    x: chex.Array
    y: chex.Array
    active: chex.Array  # Boolean, whether the chaser has spawned in this level.


class PlayerState(struct.PyTreeNode):
    """Holds the dynamic state of the player."""
    x: chex.Array
    y: chex.Array
    last_valid_x: chex.Array
    last_valid_y: chex.Array
    last_dx: chex.Array  # Records the last horizontal movement direction.
    last_dy: chex.Array  # Records the last vertical movement direction.


class VentureConstants(AutoDerivedConstants):
    """Defines all static constants for the Venture game."""
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=210)
    PLAYER_SPEED: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1, dtype=jnp.float32))
    PLAYER_DOT_RENDER_WIDTH: int = struct.field(pytree_node=False, default=1)
    PLAYER_DOT_RENDER_HEIGHT: int = struct.field(pytree_node=False, default=2)
    PLAYER_DETAILED_RENDER_WIDTH: int = struct.field(pytree_node=False, default=6)  # Player sprite dimensions when in rooms.
    PLAYER_DETAILED_RENDER_HEIGHT: int = struct.field(pytree_node=False, default=6)
    PLAYER_ROOM_RADIUS: int = struct.field(pytree_node=False, default=3)  # Collision radius for the player when in rooms.

    ALL_WALL_MAPS_PER_WORLD: chex.Array = struct.field(pytree_node=False, default_factory=lambda: ALL_WALL_MAPS_PER_WORLD)  # Multi-world wall maps.
    PLAY_AREA_Y_START: int = struct.field(pytree_node=False, default=20)  # Top boundary of the playable area.
    PLAY_AREA_Y_END: int = struct.field(pytree_node=False, default=180)  # Bottom boundary of the playable area.
    MONSTER_SPEED: float = struct.field(pytree_node=False, default=1.0)
    MONSTER_RENDER_WIDTH: int = struct.field(pytree_node=False, default=7)
    MONSTER_RENDER_HEIGHT: int = struct.field(pytree_node=False, default=10)
    MONSTER_CHANGE_DIR_PROB: float = struct.field(pytree_node=False, default=0.01)  # Probability of a monster changing direction each frame.
    MAX_DEAD_MONSTERS: int = struct.field(pytree_node=False, default=10)  # Maximum number of dead monsters to track.
    DEAD_MONSTER_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=90)  # Frames a dead monster remains on screen (1.5 sec.).

    LIVES: int = struct.field(pytree_node=False, default=4)
    PLAYER_INITIAL_X: float = struct.field(pytree_node=False, default=45.0)
    PLAYER_INITIAL_Y: float = struct.field(pytree_node=False, default=185.0)
    FINAL_GAME_OVER_DELAY_FRAMES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(60, dtype=jnp.int32))  # Delay before fully ending the game.
    LIFE_LOST_DELAY_FRAMES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(45, dtype=jnp.int32))  # Delay after losing a life (0.75 sec. @ 60fps).

    WORLD_TRANSITION_DELAY_FRAMES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(90, dtype=jnp.int32))  # Pause duration when transitioning between worlds.

    PROJECTILE_SPEED: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(2.0, dtype=jnp.float32))
    PROJECTILE_RADIUS: int = struct.field(pytree_node=False, default=2)
    PROJECTILE_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=30)
    AIMING_DOT_OFFSET: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(5.0, dtype=jnp.float32))   # Distance of aiming dot from player.

    CHEST_WIDTH: int = struct.field(pytree_node=False, default=7)
    CHEST_HEIGHT: int = struct.field(pytree_node=False, default=11)
    CHEST_SCORE: int = struct.field(pytree_node=False, default=200)
    # Chest positions for each level (global index: 0-4 for World 1, 5-9 for World 2).
    CHEST_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [0.0, 0.0],  # Level 0 (World 1 Main map, no chest)
        [80.0, 105.0],  # Level 1 (World 1 Room 1)
        [115.0, 170.0],  # Level 2 (World 1 Room 2)
        [30.0, 170.0],  # Level 3 (World 1 Room 3)
        [30.0, 170.0],  # Level 4 (World 1 Room 4)
        [0.0, 0.0],  # Level 5 (World 2 Main map, no chest)
        [80, 163],  # Level 6 (World 2 Room 1)
        [120, 35],  # Level 7 (World 2 Room 2)
        [75, 35],  # Level 8 (World 2 Room 3)
        [80, 67]  # Level 9 (World 2 Room 4)
    ], dtype=jnp.float32))

    CHASER_SPAWN_FRAMES: int = struct.field(pytree_node=False, default=1080)  # Frames until the chaser monster spawns in a room.
    CHASER_SPEED: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(0.4, dtype=jnp.float32))
    CHASER_RENDER_WIDTH: int = struct.field(pytree_node=False, default=5)
    CHASER_RENDER_HEIGHT: int = struct.field(pytree_node=False, default=15)
    CHASER_SPAWN_POS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([10.0, 30.0], dtype=jnp.float32))  # Top-left corner spawn.

    LASER_SPEED: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(0.3, dtype=jnp.float32))
    LASER_THICKNESS: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(4.0, dtype=jnp.float32))
    # Movement boundaries for 4 lasers ([min_coord, max_coord]).
    LASER_BOUNDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [45.0, 65.0],  # Vertical Laser 1 (moves Left -> Right)
        [115.0, 95.0],  # Vertical Laser 2 (moves Right -> Left)
        [50.0, 80.0],  # Horizontal Laser 1 (moves Top -> Bottom)
        [130.0, 160.0],  # Horizontal Laser 2 (moves Bottom -> Top)
    ], dtype=jnp.float32))
    LASER_INITIAL_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        70.0,  # Laser 0 starts at the left
        90.0,  # Laser 1 starts at the right
        95.0,  # Laser 2 starts at the top
        115.0,  # Laser 3 starts at the bottom
    ], dtype=jnp.float32))
    LASER_INITIAL_DIRECTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float32))
    LASER_ROOM_SPAN: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([70.0, 90.0, 95.0, 115.0], dtype=jnp.float32))  # Full span of lasers across the room.

    # Assign pre-calculated monster and portal data.
    LEVEL_MONSTER_CONFIGS: Tuple[Dict, ...] = struct.field(pytree_node=False, default=_LEVEL_MONSTER_CONFIGS)
    TOTAL_MONSTERS: int = struct.field(pytree_node=False, default=_TOTAL_MONSTERS)
    LEVEL_OFFSETS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: _LEVEL_OFFSETS)
    ALL_MONSTER_SPAWNS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: _ALL_MONSTER_SPAWNS)
    ALL_MONSTER_IMMORTAL_FLAGS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: _ALL_MONSTER_IMMORTAL_FLAGS)
    JAX_TRANSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: _JAX_TRANSITIONS)


class LaserState(struct.PyTreeNode):
    """Holds the state of the moving laser walls."""
    positions: chex.Array  # Current x or y coordinate of the 4 lasers.
    directions: chex.Array  # Current movement direction (-1 or 1).


class GameState(struct.PyTreeNode):
    """Full state of the Venture game at a given time step."""
    player: PlayerState
    monsters: MonsterState
    dead_monsters: DeadMonsterState
    projectile: ProjectileState
    chaser: ChaserState
    lasers: LaserState
    chests_active: chex.Array  # Boolean array (per room) indicating if chests are yet to be collected.
    kill_bonus_active: chex.Array  # Boolean array (per room) indicating if monster kill bonus is active.
    key: jax.random.PRNGKey
    game_over_timer: chex.Array
    life_lost_timer: chex.Array
    level_timer: chex.Array
    step_counter: chex.Array
    score: chex.Array
    lives: chex.Array
    is_in_collision: chex.Array
    current_level: chex.Array  # Current level (0 for main map, 1-4 for rooms).
    world_level: chex.Array  # Current world (1 or 2).
    world_transition_timer: chex.Array  # Countdown timer for world transition.
    last_level: chex.Array  # Tracks the previous level to detect transitions.
    collected_chest_in_current_visit: chex.Array # Records if a chest was collected in the current room visit.


class EntityPosition(struct.PyTreeNode):
    """Simplified position and dimension info for rendering and observation."""
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class VentureObservation(struct.PyTreeNode):
    """Observation provided to the agent."""
    player: ObjectObservation
    monsters: ObjectObservation
    portals: ObjectObservation
    chest: ObjectObservation
    lasers: ObjectObservation
    chaser: ObjectObservation

class VentureInfo(struct.PyTreeNode):
    """Auxiliary information about the game state."""
    time: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray


class JaxVenture(JaxEnvironment[GameState, VentureObservation, VentureInfo, VentureConstants]):
    """JAX-based implementation of the Venture Atari game."""

    def __init__(self, consts: VentureConstants = None):
        super().__init__(consts or VentureConstants())

        self.renderer = VentureRenderer(self.consts)


    def reset(self, key: jax.random.PRNGKey = None) -> tuple[VentureObservation, GameState]:
        """Resets the environment to its initial state for a new episode."""
        key, monster_key = jax.random.split(key, 2)

        player_state = PlayerState(
            x=jnp.array(self.consts.PLAYER_INITIAL_X, dtype=jnp.float32),
            y=jnp.array(self.consts.PLAYER_INITIAL_Y, dtype=jnp.float32),
            last_valid_x=jnp.array(self.consts.PLAYER_INITIAL_X, dtype=jnp.float32),
            last_valid_y=jnp.array(self.consts.PLAYER_INITIAL_Y, dtype=jnp.float32),
            last_dx=jnp.array(1.0, dtype=jnp.float32),
            last_dy=jnp.array(0.0, dtype=jnp.float32)
        )

        projectile_state = ProjectileState(
            x=jnp.array(0.0, dtype=jnp.float32), y=jnp.array(0.0, dtype=jnp.float32),
            dx=jnp.array(0.0, dtype=jnp.float32), dy=jnp.array(0.0, dtype=jnp.float32),
            active=jnp.array(False, dtype=jnp.bool_),
            lifetime=jnp.array(0, dtype=jnp.int32)
        )

        # Initialize monsters for World 1 Main Map (level 0).
        angles = jax.random.uniform(monster_key, shape=(self.consts.TOTAL_MONSTERS,), minval=0, maxval=2 * jnp.pi, dtype=jnp.float32)
        monster_dx, monster_dy = jnp.cos(angles), jnp.sin(angles)
        indices = jnp.arange(self.consts.TOTAL_MONSTERS)
        num_main_map_monsters_w1 = self.consts.LEVEL_OFFSETS[1]  # Monsters for World 1, Level 0.
        active_monsters = indices < num_main_map_monsters_w1

        monster_state = MonsterState(
            x=self.consts.ALL_MONSTER_SPAWNS[:, 0].astype(jnp.float32),
            y=self.consts.ALL_MONSTER_SPAWNS[:, 1].astype(jnp.float32),
            dx=monster_dx, dy=monster_dy, active=active_monsters,
            is_immortal=self.consts.ALL_MONSTER_IMMORTAL_FLAGS
        )

        # Initialize dead monsters as all inactive.
        dead_monster_state = DeadMonsterState(
            x=jnp.zeros(self.consts.MAX_DEAD_MONSTERS, dtype=jnp.float32),
            y=jnp.zeros(self.consts.MAX_DEAD_MONSTERS, dtype=jnp.float32),
            active=jnp.zeros(self.consts.MAX_DEAD_MONSTERS, dtype=jnp.bool_),
            lifetime=jnp.zeros(self.consts.MAX_DEAD_MONSTERS, dtype=jnp.int32)
        )

        chaser_state = ChaserState(
            x=jnp.array(0.0, dtype=jnp.float32), y=jnp.array(0.0, dtype=jnp.float32), active=jnp.array(False)
        )

        laser_state = LaserState(
            positions=self.consts.LASER_INITIAL_POSITIONS,
            directions=self.consts.LASER_INITIAL_DIRECTIONS
        )

        num_rooms = 4
        state = GameState(
            player=player_state,
            monsters=monster_state,
            dead_monsters=dead_monster_state,
            projectile=projectile_state,
            chaser=chaser_state,
            chests_active=jnp.ones(num_rooms, dtype=jnp.bool_),
            lasers=laser_state,
            kill_bonus_active=jnp.zeros(num_rooms, dtype=jnp.bool_),
            key=key,
            game_over_timer=jnp.array(0, dtype=jnp.int32),
            life_lost_timer=jnp.array(0, dtype=jnp.int32),
            level_timer=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.LIVES, dtype=jnp.int32),
            is_in_collision=jnp.array(False, dtype=jnp.bool_),
            current_level=jnp.array(0, dtype=jnp.int32),
            world_level=jnp.array(1, dtype=jnp.int32),
            world_transition_timer=jnp.array(0, dtype=jnp.int32),
            last_level=jnp.array(0, dtype=jnp.int32),
            collected_chest_in_current_visit=jnp.array(-1, dtype=jnp.int32),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int) -> tuple[
        VentureObservation, Any, Array | ndarray[Any, dtype[Any]], bool, VentureInfo]:
        """Performs one step of the environment given the agent's actions."""

        # Store the current level before any transitions occur.
        state = state.replace(last_level=state.current_level)

        def handle_world_transition_delay(current_state: GameState) -> GameState:
            """Manages the countdown and execution of world transitions."""
            final_state = jax.lax.cond(
                current_state.world_transition_timer == 1,
                self._perform_world_switch,
                lambda s: s,
                current_state
            )
            return final_state.replace(world_transition_timer=current_state.world_transition_timer - 1)

        def handle_final_game_over(current_state: GameState) -> GameState:
            """Manages the countdown for the final game over sequence."""
            return current_state.replace(game_over_timer=current_state.game_over_timer - 1)

        def handle_life_lost_delay(current_state: GameState) -> GameState:
            """Manages the delay and respawn logic after the player loses a life."""
            final_state = jax.lax.cond(
                current_state.life_lost_timer == 1,
                self._respawn_entities,
                lambda s: s,
                current_state
            )
            return final_state.replace(life_lost_timer=current_state.life_lost_timer - 1)

        def handle_normal_gameplay(current_state: GameState) -> GameState:
            """Executes the core game logic for a normal step."""
            fire_action = jnp.isin(action, FIREACTIONS)

            # get move direction from the action map
            def get_move_from_action(act):
                dx = 0.0
                dy = 0.0

                dy = jnp.where(jnp.isin(act, UPACTIONS), -self.consts.PLAYER_SPEED, dy)
                dy = jnp.where(jnp.isin(act, DOWNACTIONS), self.consts.PLAYER_SPEED, dy)
                dx = jnp.where(jnp.isin(act, LEFTACTIONS), -self.consts.PLAYER_SPEED, dx)
                dx = jnp.where(jnp.isin(act, RIGHTACTIONS), self.consts.PLAYER_SPEED, dx)

                return dx, dy

            dx, dy = get_move_from_action(action)

            key, monster_update_key = jax.random.split(current_state.key, 2)

            proposed_x = current_state.player.x + dx
            proposed_y = current_state.player.y + dy
            hypothetical_player_state = current_state.player.replace(x=proposed_x, y=proposed_y)
            hypothetical_state = current_state.replace(player=hypothetical_player_state)

            # Handle level transitions (e.g., entering/exiting rooms).
            post_transition_state = self._handle_level_transitions(hypothetical_state)
            transition_occurred = (post_transition_state.current_level != current_state.current_level)

            def perform_normal_move(_):
                world_idx = current_state.world_level - 1
                original_wall_map = self.consts.ALL_WALL_MAPS_PER_WORLD[world_idx, current_state.current_level]

                # Dynamically "fill" portals on the main map if their corresponding chest is not collected.
                def _maybe_add_air_walls(s: GameState, base_map: chex.Array) -> chex.Array:
                    world_idx_inner = s.world_level - 1

                    def _fill_one_portal(portal_idx_local, current_map):
                        portal_data = self.consts.JAX_TRANSITIONS[world_idx_inner, 0, portal_idx_local]
                        to_level = portal_data[4].astype(jnp.int32)
                        is_valid_portal = portal_data[2] > 0
                        is_portal_locked = jax.lax.cond(
                            to_level > 0,
                            lambda: ~s.chests_active[to_level - 1],
                            lambda: jnp.array(False, dtype=jnp.bool_)
                        )
                        should_fill = is_valid_portal & is_portal_locked

                        def _fill_it(m):
                            x, y, w, h = portal_data[0:4]
                            grid_y, grid_x = jnp.mgrid[0:self.consts.SCREEN_HEIGHT, 0:self.consts.SCREEN_WIDTH]
                            portal_mask = (grid_x >= x) & (grid_x < x + w) & (grid_y >= y) & (grid_y < y + h)
                            return jnp.where(portal_mask, jnp.uint8(1), m)

                        return jax.lax.cond(should_fill, _fill_it, lambda m: m, current_map)

                    num_portals = self.consts.JAX_TRANSITIONS[world_idx_inner, 0].shape[0]
                    return jax.lax.fori_loop(0, num_portals, _fill_one_portal, base_map)

                effective_wall_map = jax.lax.cond(
                    current_state.current_level == 0,
                    lambda: _maybe_add_air_walls(current_state, original_wall_map),
                    lambda: original_wall_map
                )

                is_in_room_flag = current_state.current_level != 0
                new_player_state, new_is_in_collision = self._update_player(
                    current_state.player, action, current_state.is_in_collision,
                    effective_wall_map, is_in_room_flag
                )
                new_monster_state = self._update_monsters(current_state.monsters, monster_update_key, effective_wall_map)
                new_dead_monsters_state = self._update_dead_monsters(current_state.dead_monsters)
                return current_state.replace(
                    player=new_player_state, monsters=new_monster_state,
                    dead_monsters=new_dead_monsters_state, is_in_collision=new_is_in_collision
                )

            # Choose between performing a transition or normal movement.
            state_after_move = jax.lax.cond(transition_occurred, lambda _: post_transition_state, perform_normal_move, operand=None)

            # Check if all chests in World 1 are collected and trigger a world transition.
            def check_and_trigger_world_transition(s: GameState) -> GameState:
                is_world_1 = s.world_level == 1
                all_rewards_collected = jnp.all(~s.chests_active)
                returned_to_main_map = (s.current_level == 0) & (s.last_level > 0)
                should_trigger = is_world_1 & all_rewards_collected & returned_to_main_map

                return jax.lax.cond(
                    should_trigger,
                    lambda state_to_update: state_to_update.replace(
                        world_transition_timer=self.consts.WORLD_TRANSITION_DELAY_FRAMES
                    ),
                    lambda state_to_update: state_to_update,
                    s
                )

            state_after_move = check_and_trigger_world_transition(state_after_move)

            # Projectile Firing Logic.
            should_fire = (state_after_move.current_level != 0) & jnp.logical_not(state_after_move.projectile.active) & fire_action
            hypothetical_fired_projectile = ProjectileState(
                x=state_after_move.player.x, y=state_after_move.player.y, dx=state_after_move.player.last_dx,
                dy=state_after_move.player.last_dy, active=jnp.array(True),
                lifetime=jnp.array(self.consts.PROJECTILE_LIFETIME_FRAMES)
            )
            new_projectile_state = jax.tree_util.tree_map(
                lambda if_fired, if_not_fired: jnp.where(should_fire, if_fired, if_not_fired),
                hypothetical_fired_projectile, state_after_move.projectile
            )
            state_after_firing = state_after_move.replace(projectile=new_projectile_state)

            # Projectile Update Logic.
            def update_active_projectile(s: GameState) -> GameState:
                proj = s.projectile
                new_x = proj.x + proj.dx * self.consts.PROJECTILE_SPEED
                new_y = proj.y + proj.dy * self.consts.PROJECTILE_SPEED
                world_idx = s.world_level - 1
                wall_map = self.consts.ALL_WALL_MAPS_PER_WORLD[world_idx, s.current_level]
                map_val = wall_map[jnp.clip(new_y.astype(jnp.int32), 0, 209), jnp.clip(new_x.astype(jnp.int32), 0, 159)]
                hit_wall = (map_val == 1)
                proj_radius = self.consts.PROJECTILE_RADIUS
                mon_x, mon_y = s.monsters.x, s.monsters.y
                mon_hw, mon_hh = self.consts.MONSTER_RENDER_WIDTH / 2.0, self.consts.MONSTER_RENDER_HEIGHT / 2.0
                closest_x = jnp.clip(new_x, mon_x - mon_hw, mon_x + mon_hw)
                closest_y = jnp.clip(new_y, mon_y - mon_hh, mon_y + mon_hh)
                dist_sq = (new_x - closest_x) ** 2 + (new_y - closest_y) ** 2
                monsters_hit_mask = (dist_sq < proj_radius ** 2) & s.monsters.active & ~s.monsters.is_immortal
                hit_monster = jnp.any(monsters_hit_mask)
                level_idx = s.current_level
                chest_idx = level_idx - 1
                is_bonus_active = jax.lax.cond(level_idx > 0, lambda: s.kill_bonus_active[chest_idx], lambda: jnp.array(False))
                num_killed_monsters = jnp.sum(monsters_hit_mask)
                score_to_add = jax.lax.cond(is_bonus_active, lambda: num_killed_monsters * 100, lambda: 0)
                new_score = s.score + score_to_add
                available_dead_slots = jnp.where(~s.dead_monsters.active, size=self.consts.MAX_DEAD_MONSTERS, fill_value=-1)[0]
                num_available_slots = jnp.sum(available_dead_slots != -1)
                monsters_to_kill_indices = jnp.where(monsters_hit_mask, size=self.consts.TOTAL_MONSTERS, fill_value=-1)[0]
                num_actually_killed = jnp.sum(monsters_to_kill_indices != -1)
                num_corpses_to_add = jnp.minimum(num_actually_killed, num_available_slots)
                new_mon_active = s.monsters.active
                new_dead_mon = s.dead_monsters

                def _add_corpse_loop_body(k, carry_loop):
                    current_new_mon_active, current_new_dead_mon = carry_loop
                    monster_idx_to_kill = monsters_to_kill_indices[k]
                    is_valid_kill_slot = (monster_idx_to_kill != -1) & (k < num_corpses_to_add)

                    def _process_one_kill(c):
                        active_monsters, dead_mon_state = c
                        active_monsters = active_monsters.at[monster_idx_to_kill].set(False)
                        dead_slot_idx = available_dead_slots[k]
                        dead_mon_state = dead_mon_state.replace(
                            x=dead_mon_state.x.at[dead_slot_idx].set(s.monsters.x[monster_idx_to_kill]),
                            y=dead_mon_state.y.at[dead_slot_idx].set(s.monsters.y[monster_idx_to_kill]),
                            active=dead_mon_state.active.at[dead_slot_idx].set(True),
                            lifetime=dead_mon_state.lifetime.at[dead_slot_idx].set(self.consts.DEAD_MONSTER_LIFETIME_FRAMES)
                        )
                        return active_monsters, dead_mon_state
                    return jax.lax.cond(is_valid_kill_slot, _process_one_kill, lambda c: c, (current_new_mon_active, current_new_dead_mon))

                final_mon_active, final_dead_monsters = jax.lax.fori_loop(0, self.consts.TOTAL_MONSTERS, _add_corpse_loop_body, (new_mon_active, new_dead_mon))
                new_monsters = s.monsters.replace(active=final_mon_active)
                new_lifetime = proj.lifetime - 1
                lifetime_over = new_lifetime <= 0
                should_deactivate = hit_wall | hit_monster | lifetime_over
                new_proj_state = proj.replace(
                    x=new_x, y=new_y, lifetime=new_lifetime, active=jnp.where(should_deactivate, False, True)
                )
                return s.replace(projectile=new_proj_state, monsters=new_monsters, dead_monsters=final_dead_monsters, score=new_score)

            hypothetical_updated_state = update_active_projectile(state_after_firing)
            state_with_updated_projectile = jax.tree_util.tree_map(
                lambda if_active, if_inactive: jnp.where(
                    state_after_firing.projectile.active,
                    if_active,
                    if_inactive
                ),
                hypothetical_updated_state,
                state_after_firing
            )

            # Check for chest collection.
            def _check_and_collect_chest(s: GameState) -> GameState:
                def _do_collection_logic(current_s: GameState) -> GameState:
                    level_idx = current_s.current_level
                    chest_idx = level_idx - 1

                    world_offset = (current_s.world_level - 1) * 5
                    chest_lookup_idx = world_offset + level_idx

                    # Check for collection only if chest is permanently available and not collected in current visit.
                    is_permanently_available = current_s.chests_active[chest_idx]
                    is_not_collected_this_visit = (current_s.collected_chest_in_current_visit != chest_idx)
                    should_check_for_collection = is_permanently_available & is_not_collected_this_visit

                    def _do_collection_check(cs: GameState) -> GameState:
                        chest_pos = self.consts.CHEST_POSITIONS[chest_lookup_idx]
                        chest_hw, chest_hh = self.consts.CHEST_WIDTH / 2, self.consts.CHEST_HEIGHT / 2
                        px, py = cs.player.x, cs.player.y
                        player_radius = self.consts.PLAYER_ROOM_RADIUS
                        closest_x = jnp.clip(px, chest_pos[0] - chest_hw, chest_pos[0] + chest_hw)
                        closest_y = jnp.clip(py, chest_pos[1] - chest_hh, chest_pos[1] + chest_hh)
                        dist_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2
                        collided = dist_sq < (player_radius ** 2)

                        def _collect(c_s: GameState) -> GameState:
                            return c_s.replace(
                                score=c_s.score + self.consts.CHEST_SCORE,
                                collected_chest_in_current_visit=chest_idx,
                                kill_bonus_active=c_s.kill_bonus_active.at[chest_idx].set(True)
                            )
                        return jax.lax.cond(collided, _collect, lambda c_s: c_s, cs)

                    return jax.lax.cond(should_check_for_collection, _do_collection_check, lambda s_in: s_in, current_s)

                return jax.lax.cond(s.current_level > 0, _do_collection_logic, lambda s_in: s_in, s)

            state_after_chest_collection = _check_and_collect_chest(state_with_updated_projectile)

            # Update level timer and spawn chaser if conditions met.
            new_level_timer = jax.lax.cond(state_after_chest_collection.current_level > 0, lambda: state_after_chest_collection.level_timer + 1, lambda: jnp.array(0, dtype=jnp.int32))
            def spawn_chaser_if_needed(s: GameState) -> GameState:
                should_spawn = (s.current_level > 0) & (new_level_timer == self.consts.CHASER_SPAWN_FRAMES) & ~s.chaser.active
                def _spawn(current_s: GameState) -> GameState:
                    new_chaser = current_s.chaser.replace(x=self.consts.CHASER_SPAWN_POS[0], y=self.consts.CHASER_SPAWN_POS[1], active=jnp.array(True))
                    return current_s.replace(chaser=new_chaser)
                return jax.lax.cond(should_spawn, _spawn, lambda s_in: s_in, s)
            state_after_spawn = spawn_chaser_if_needed(state_after_chest_collection)

            # Move chaser if active.
            def _move_chaser(s: GameState) -> GameState:
                return s.replace(chaser=self._update_chaser(s.chaser, s.player))
            state_after_chaser_move = jax.lax.cond(state_after_spawn.chaser.active, _move_chaser, lambda s_in: s_in, state_after_spawn)

            # Move lasers if in World 1 Room 1.
            def _move_lasers(s: GameState) -> GameState:
                return s.replace(lasers=self._update_lasers(s.lasers))
            should_move_lasers = (state_after_chaser_move.current_level == 1) & (state_after_chaser_move.world_level == 1)
            state_after_laser_move = jax.lax.cond(
                should_move_lasers,
                _move_lasers,
                lambda s_in: s_in,
                state_after_chaser_move
            )

            # Check for player collision with hazards.
            player_hazard_collision = self._check_player_hazard_collision(
                state_after_laser_move.player, state_after_laser_move.monsters,
                state_after_laser_move.dead_monsters, state_after_laser_move.chaser,
                state_after_laser_move.lasers, state_after_laser_move.current_level,
                state_after_laser_move.world_level
            )
            def on_collision(s: GameState) -> GameState:
                is_final_life = (s.lives == 1)
                new_game_over_timer = jnp.where(is_final_life, self.consts.FINAL_GAME_OVER_DELAY_FRAMES, s.game_over_timer)
                return s.replace(lives=s.lives - 1, life_lost_timer=self.consts.LIFE_LOST_DELAY_FRAMES, game_over_timer=new_game_over_timer)
            state_after_collision = jax.lax.cond(player_hazard_collision, on_collision, lambda s_in: s_in, state_after_laser_move)

            # Check for progression (e.g., finishing a world or the game).
            def check_for_progression(s: GameState) -> GameState:
                all_rewards_collected = jnp.all(~s.chests_active)
                returned_to_main_map = (s.current_level == 0) & (s.last_level > 0)
                should_progress = all_rewards_collected & returned_to_main_map

                def _handle_progression(state_to_update: GameState) -> GameState:
                    def _complete_world_1(s1: GameState) -> GameState:
                        return s1.replace(world_transition_timer=self.consts.WORLD_TRANSITION_DELAY_FRAMES)
                    def _complete_world_2(s2: GameState) -> GameState:
                        return s2.replace(game_over_timer=self.consts.FINAL_GAME_OVER_DELAY_FRAMES)

                    return jax.lax.cond(
                        state_to_update.world_level == 1,
                        _complete_world_1,
                        _complete_world_2,
                        state_to_update
                    )
                return jax.lax.cond(
                    should_progress,
                    _handle_progression,
                    lambda state_to_update: state_to_update,
                    s
                )

            # Apply progression check and update final frame state.
            final_state_for_frame = state_after_collision.replace(
                key=key,
                step_counter=current_state.step_counter + 1,
                level_timer=new_level_timer
            )
            return check_for_progression(final_state_for_frame)

        # previous_score = state.score

        # Main logic branch: prioritize timers (world transition, game over, life lost) over normal gameplay.
        new_state = jax.lax.cond(
            state.world_transition_timer > 0,
            handle_world_transition_delay,
            lambda s: jax.lax.cond(
                s.game_over_timer > 0,
                handle_final_game_over,
                lambda s2: jax.lax.cond(
                    s2.life_lost_timer > 0,
                    handle_life_lost_delay,
                    handle_normal_gameplay,
                    s2
                ),
                s
            ),
            state
        )

        env_reward = self._get_reward(state, new_state)

        done = self._get_done(new_state)
        info = self._get_info(new_state)
        obs = self._get_observation(new_state)

        return obs, new_state, env_reward, done, info

    def _perform_world_switch(self, state: GameState) -> GameState:
        """Transitions the game state to the next world, resetting level-specific entities."""
        key, monster_key = jax.random.split(state.key, 2)

        # Player returns to initial spawn point.
        player_state = state.player.replace(
            x=jnp.array(self.consts.PLAYER_INITIAL_X),
            y=jnp.array(self.consts.PLAYER_INITIAL_Y),
            last_valid_x=jnp.array(self.consts.PLAYER_INITIAL_X),
            last_valid_y=jnp.array(self.consts.PLAYER_INITIAL_Y),
        )

        # Reset monsters for the new world's main map.
        angles = jax.random.uniform(monster_key, shape=(self.consts.TOTAL_MONSTERS,), minval=0, maxval=2 * jnp.pi)
        monster_dx, monster_dy = jnp.cos(angles), jnp.sin(angles)
        indices = jnp.arange(self.consts.TOTAL_MONSTERS)

        total_levels_per_world = 5
        next_world_level_id = state.world_level + 1
        next_world_main_map_global_idx = (next_world_level_id - 1) * total_levels_per_world + 0

        offset_start = self.consts.LEVEL_OFFSETS[next_world_main_map_global_idx]
        offset_end = self.consts.LEVEL_OFFSETS[next_world_main_map_global_idx + 1]

        active_monsters = (indices >= offset_start) & (indices < offset_end)

        new_monster_state = MonsterState(
            x=self.consts.ALL_MONSTER_SPAWNS[:, 0],
            y=self.consts.ALL_MONSTER_SPAWNS[:, 1],
            dx=monster_dx, dy=monster_dy, active=active_monsters,
            is_immortal=self.consts.ALL_MONSTER_IMMORTAL_FLAGS
        )

        # Reset all other level-specific entities and timers.
        return state.replace(
            player=player_state,
            monsters=new_monster_state,
            dead_monsters=state.dead_monsters.replace(active=jnp.zeros_like(state.dead_monsters.active)),
            chaser=state.chaser.replace(active=jnp.array(False)),
            lasers=state.lasers.replace(
                positions=self.consts.LASER_INITIAL_POSITIONS,
                directions=self.consts.LASER_INITIAL_DIRECTIONS
            ),
            chests_active=jnp.ones_like(state.chests_active),  # All chests become collectible again.
            kill_bonus_active=jnp.zeros_like(state.kill_bonus_active),
            current_level=jnp.array(0, dtype=jnp.int32),  # Return to the main map of the new world.
            last_level=jnp.array(0, dtype=jnp.int32),
            level_timer=jnp.array(0, dtype=jnp.int32),
            key=key,
            world_level=next_world_level_id,  # Increment world level.
            world_transition_timer=jnp.array(0, dtype=jnp.int32),
            collected_chest_in_current_visit=jnp.array(-1, dtype=jnp.int32)
        )

    def _respawn_entities(self, state: GameState) -> GameState:
        """Resets player and monsters to their initial state after a life is lost."""
        key, monster_key = jax.random.split(state.key, 2)

        # Player returns to initial spawn point.
        new_player_state = PlayerState(
            x=jnp.array(self.consts.PLAYER_INITIAL_X),
            y=jnp.array(self.consts.PLAYER_INITIAL_Y),
            last_valid_x=jnp.array(self.consts.PLAYER_INITIAL_X),
            last_valid_y=jnp.array(self.consts.PLAYER_INITIAL_Y),
            last_dx=jnp.array(1.0),
            last_dy=jnp.array(0.0)
        )

        # Reset monsters to the current world's main map configuration.
        angles = jax.random.uniform(monster_key, shape=(self.consts.TOTAL_MONSTERS,), minval=0, maxval=2 * jnp.pi)
        monster_dx, monster_dy = jnp.cos(angles), jnp.sin(angles)
        indices = jnp.arange(self.consts.TOTAL_MONSTERS)

        total_levels_per_world = 5
        current_world_main_map_global_idx = (state.world_level - 1) * total_levels_per_world + 0

        offset_start = self.consts.LEVEL_OFFSETS[current_world_main_map_global_idx]
        offset_end = self.consts.LEVEL_OFFSETS[current_world_main_map_global_idx + 1]
        active_monsters = (indices >= offset_start) & (indices < offset_end)

        new_monster_state = MonsterState(
            x=self.consts.ALL_MONSTER_SPAWNS[:, 0],
            y=self.consts.ALL_MONSTER_SPAWNS[:, 1],
            dx=monster_dx, dy=monster_dy, active=active_monsters,
            is_immortal=self.consts.ALL_MONSTER_IMMORTAL_FLAGS
        )

        # Reset dead monsters, chaser, and lasers.
        new_dead_monster_state = DeadMonsterState(
            x=jnp.zeros(self.consts.MAX_DEAD_MONSTERS, dtype=jnp.float32),
            y=jnp.zeros(self.consts.MAX_DEAD_MONSTERS, dtype=jnp.float32),
            active=jnp.zeros(self.consts.MAX_DEAD_MONSTERS, dtype=jnp.bool_),
            lifetime=jnp.zeros(self.consts.MAX_DEAD_MONSTERS, dtype=jnp.int32)
        )

        inactive_chaser_state = ChaserState(
            x=jnp.array(0.0), y=jnp.array(0.0), active=jnp.array(False)
        )

        initial_laser_state = LaserState(
            positions=self.consts.LASER_INITIAL_POSITIONS,
            directions=self.consts.LASER_INITIAL_DIRECTIONS
        )

        # Return to main map, keep existing chest collection status.
        return state.replace(
            player=new_player_state,
            monsters=new_monster_state,
            dead_monsters=new_dead_monster_state,
            chaser=inactive_chaser_state,
            lasers=initial_laser_state,
            key=key,
            current_level=jnp.array(0, dtype=jnp.int32),
            is_in_collision=jnp.array(False),
            level_timer=jnp.array(0, dtype=jnp.int32),
            kill_bonus_active=jnp.zeros_like(state.kill_bonus_active),
            collected_chest_in_current_visit=jnp.array(-1, dtype=jnp.int32)
        )

    def _handle_level_transitions(self, state: GameState) -> GameState:
        """Manages player transitions between levels (main map and rooms) via portals."""
        px, py = state.player.x, state.player.y

        def check_and_transition(current_state, flat_params):
            rect, to_level_float, spawn_pos = flat_params[0:4], flat_params[4], flat_params[5:7]
            to_level = to_level_float.astype(jnp.int32)
            rx, ry, rw, rh = rect[0], rect[1], rect[2], rect[3]
            in_portal = (px > rx) & (px < rx + rw) & (py > ry) & (py < ry + rh)

            # Portal lock: rooms' portals are locked if their chest is not collected.
            is_portal_locked = (to_level > 0) & ~current_state.chests_active[to_level - 1]
            is_portal_active = ~is_portal_locked
            can_transition = in_portal & is_portal_active

            def perform_transition(s):
                pending_chest_idx = s.collected_chest_in_current_visit
                is_exiting_with_collection = (s.current_level > 0) & (to_level == 0) & (pending_chest_idx != -1)

                def _commit_collection(current_s: GameState) -> GameState:
                    return current_s.replace(
                        chests_active=current_s.chests_active.at[pending_chest_idx].set(False)
                    )
                state_after_commit = jax.lax.cond(
                    is_exiting_with_collection,
                    _commit_collection,
                    lambda cs: cs,
                    s
                )

                new_player = s.player.replace(
                    x=spawn_pos[0],
                    y=spawn_pos[1],
                    last_valid_x=spawn_pos[0],
                    last_valid_y=spawn_pos[1]
                )

                # Activate monsters specific to the target level and current world.
                total_levels_per_world = 5
                current_world_config_base_idx = (s.world_level - 1) * total_levels_per_world
                target_level_global_idx = current_world_config_base_idx + to_level

                offset_start = self.consts.LEVEL_OFFSETS[target_level_global_idx]
                offset_end = self.consts.LEVEL_OFFSETS[target_level_global_idx + 1]

                indices = jnp.arange(self.consts.TOTAL_MONSTERS)
                new_active_monsters_mask = (indices >= offset_start) & (indices < offset_end)

                new_monsters = s.monsters.replace(
                    active=new_active_monsters_mask,
                    x=self.consts.ALL_MONSTER_SPAWNS[:, 0],
                    y=self.consts.ALL_MONSTER_SPAWNS[:, 1]
                )

                # Reset dead monsters, chaser, and lasers upon level transition.
                new_dead_monsters = s.dead_monsters.replace(
                    active=jnp.zeros_like(s.dead_monsters.active),
                    lifetime=jnp.zeros_like(s.dead_monsters.lifetime)
                )

                inactive_chaser = s.chaser.replace(active=jnp.array(False))

                initial_lasers = s.lasers.replace(
                    positions=self.consts.LASER_INITIAL_POSITIONS,
                    directions=self.consts.LASER_INITIAL_DIRECTIONS
                )

                return state_after_commit.replace(
                    current_level=to_level,
                    player=new_player,
                    monsters=new_monsters,
                    dead_monsters=new_dead_monsters,
                    level_timer=jnp.array(0, dtype=jnp.int32),
                    chaser=inactive_chaser,
                    lasers=initial_lasers,
                    collected_chest_in_current_visit=jnp.array(-1, dtype=jnp.int32),
                    is_in_collision=jnp.array(False)
                )


            return jax.lax.cond(can_transition, perform_transition, lambda s_in: s_in, current_state)

        world_idx = state.world_level - 1
        level_transitions = self.consts.JAX_TRANSITIONS[world_idx, state.current_level]
        max_portals = level_transitions.shape[0]

        def body_fn(i, current_s):
            flat_params = level_transitions[i]
            is_valid_portal = flat_params[2] > 0 # Check for non-padded portal data.
            return jax.lax.cond(is_valid_portal, lambda: check_and_transition(current_s, flat_params),
                                lambda: current_s)

        return jax.lax.fori_loop(0, max_portals, body_fn, state)

    def _update_monsters(self, monster_state: MonsterState, key: jax.random.PRNGKey,
                         wall_map: chex.Array) -> MonsterState:
        """Updates the positions and directions of active monsters, handling wall collisions."""
        key, dir_key, move_key = jax.random.split(key, 3)
        change_dir = jax.random.uniform(dir_key, (self.consts.TOTAL_MONSTERS,)) < self.consts.MONSTER_CHANGE_DIR_PROB

        angles = jax.random.uniform(move_key, shape=(self.consts.TOTAL_MONSTERS,), minval=0, maxval=2 * jnp.pi)

        new_dx, new_dy = jnp.cos(angles), jnp.sin(angles)
        dx = jnp.where(change_dir, new_dx, monster_state.dx)
        dy = jnp.where(change_dir, new_dy, monster_state.dy)
        px, py = monster_state.x + dx * self.consts.MONSTER_SPEED, monster_state.y + dy * self.consts.MONSTER_SPEED

        def check_collision(x, y):
            hw, hh = self.consts.MONSTER_RENDER_WIDTH / 2, self.consts.MONSTER_RENDER_HEIGHT / 2
            corners_x = jnp.array([x - hw, x + hw - 1, x - hw, x + hw - 1])
            corners_y = jnp.array([y - hh, y - hh, y + hh - 1, y + hh - 1])

            clipped_corners_x = jnp.clip(corners_x.astype(jnp.int32), 0, self.consts.SCREEN_WIDTH - 1)
            clipped_corners_y = jnp.clip(corners_y.astype(jnp.int32), 0, self.consts.SCREEN_HEIGHT - 1)

            map_vals = wall_map[clipped_corners_y, clipped_corners_x]

            # Check for out-of-bounds collision with playable area.
            oob = (corners_x[0] < 0) | (corners_x[1] >= self.consts.SCREEN_WIDTH) | \
                  (corners_y[0] < self.consts.PLAY_AREA_Y_START) | (corners_y[3] >= self.consts.PLAY_AREA_Y_END)

            return jnp.any(map_vals == 1) | oob

        is_colliding = jax.vmap(check_collision)(px, py)

        # Reverse direction and revert position for colliding monsters.
        final_dx, final_dy = jnp.where(is_colliding, -dx, dx), jnp.where(is_colliding, -dy, dy)
        final_x, final_y = jnp.where(is_colliding, monster_state.x, px), jnp.where(is_colliding, monster_state.y, py)

        # Update only active monsters.
        active = monster_state.active
        return MonsterState(
            x=jnp.where(active, final_x, monster_state.x),
            y=jnp.where(active, final_y, monster_state.y),
            dx=jnp.where(active, final_dx, monster_state.dx),
            dy=jnp.where(active, final_dy, monster_state.dy),
            active=monster_state.active,
            is_immortal=monster_state.is_immortal
        )

    def _update_dead_monsters(self, dead_monster_state: DeadMonsterState) -> DeadMonsterState:
        """Decrements the lifetime of active dead monsters and deactivates them when it expires."""
        new_lifetime = dead_monster_state.lifetime - 1
        should_deactivate = (new_lifetime <= 0)

        new_active = jnp.where(dead_monster_state.active & should_deactivate, False, dead_monster_state.active)
        new_lifetime = jnp.where(dead_monster_state.active, new_lifetime, dead_monster_state.lifetime)

        return dead_monster_state.replace(active=new_active, lifetime=new_lifetime)

    def _update_chaser(self, chaser_state: ChaserState, player_state: PlayerState) -> ChaserState:
        """Moves the chaser monster directly towards the player's position."""
        dx = player_state.x - chaser_state.x
        dy = player_state.y - chaser_state.y

        norm = jnp.sqrt(dx ** 2 + dy ** 2)
        safe_norm = jnp.where(norm == 0, 1.0, norm)  # Avoid division by zero.

        dir_x = dx / safe_norm
        dir_y = dy / safe_norm

        new_x = chaser_state.x + dir_x * self.consts.CHASER_SPEED
        new_y = chaser_state.y + dir_y * self.consts.CHASER_SPEED

        return chaser_state.replace(x=new_x, y=new_y)

    def _update_lasers(self, laser_state: LaserState) -> LaserState:
        """Moves the laser walls back and forth within their defined bounds."""
        new_positions = laser_state.positions + laser_state.directions * self.consts.LASER_SPEED

        min_bounds = jnp.minimum(self.consts.LASER_BOUNDS[:, 0], self.consts.LASER_BOUNDS[:, 1])
        max_bounds = jnp.maximum(self.consts.LASER_BOUNDS[:, 0], self.consts.LASER_BOUNDS[:, 1])

        hit_min = new_positions < min_bounds
        hit_max = new_positions > max_bounds
        should_flip_direction = hit_min | hit_max

        new_directions = jnp.where(should_flip_direction, -laser_state.directions, laser_state.directions)
        final_positions = jnp.clip(new_positions, min_bounds, max_bounds)

        return LaserState(positions=final_positions, directions=new_directions)

    def _check_player_hazard_collision(self, player_state: PlayerState, monster_state: MonsterState,
                                       dead_monster_state: DeadMonsterState,
                                       chaser_state: ChaserState, laser_state: LaserState,
                                       current_level: int, world_level: int) -> chex.Array:
        """Checks for player collisions with various hazards."""

        is_in_room = current_level != 0
        player_half_width = jax.lax.cond(is_in_room,
                                          lambda: self.consts.PLAYER_DETAILED_RENDER_WIDTH / 2.0,
                                          lambda: self.consts.PLAYER_DOT_RENDER_WIDTH / 2.0)
        player_half_height = jax.lax.cond(is_in_room,
                                           lambda: self.consts.PLAYER_DETAILED_RENDER_HEIGHT / 2.0,
                                           lambda: self.consts.PLAYER_DOT_RENDER_HEIGHT / 2.0)

        def monster_collision_logic(entity_x, entity_y, entity_active, render_width, render_height):
            entity_hw = render_width / 2.0
            entity_hh = render_height / 2.0

            def circle_collision_branch():  # Player is a circle in rooms.
                px, py = player_state.x, player_state.y
                player_radius = self.consts.PLAYER_ROOM_RADIUS
                closest_x = jnp.clip(px, entity_x - entity_hw, entity_x + entity_hw)
                closest_y = jnp.clip(py, entity_y - entity_hh, entity_y + entity_hh)
                dist_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2
                return dist_sq < (player_radius ** 2)

            def rect_collision_branch():  # Player is a rectangle on main map.
                px_hw, py_hh = player_half_width, player_half_height
                coll_x = (jnp.abs(player_state.x - entity_x) < (px_hw + entity_hw))
                coll_y = (jnp.abs(player_state.y - entity_y) < (py_hh + entity_hh))
                return coll_x & coll_y

            collision_mask = jax.lax.cond(is_in_room, circle_collision_branch, rect_collision_branch)
            return jnp.any(collision_mask & entity_active)

        # Check collision with live monsters.
        any_monster_collision = monster_collision_logic(
            monster_state.x, monster_state.y, monster_state.active,
            self.consts.MONSTER_RENDER_WIDTH, self.consts.MONSTER_RENDER_HEIGHT
        )

        # Check collision with dead monsters (corpses).
        any_dead_monster_collision = monster_collision_logic(
            dead_monster_state.x, dead_monster_state.y, dead_monster_state.active,
            self.consts.MONSTER_RENDER_WIDTH, self.consts.MONSTER_RENDER_HEIGHT
        )

        # Check collision with the chaser.
        def chaser_collision_logic():
            chaser_hw = self.consts.CHASER_RENDER_WIDTH / 2.0
            chaser_hh = self.consts.CHASER_RENDER_HEIGHT / 2.0

            def circle_chaser_collision_branch():
                px, py = player_state.x, player_state.y
                player_radius = self.consts.PLAYER_ROOM_RADIUS
                closest_x = jnp.clip(px, chaser_state.x - chaser_hw, chaser_state.x + chaser_hw)
                closest_y = jnp.clip(py, chaser_state.y - chaser_hh, chaser_state.y + chaser_hh)
                dist_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2
                return dist_sq < (player_radius ** 2)

            def rect_chaser_collision_branch():
                px_hw, py_hh = player_half_width, player_half_height
                coll_x = (jnp.abs(player_state.x - chaser_state.x) < (px_hw + chaser_hw))
                coll_y = (jnp.abs(player_state.y - chaser_state.y) < (py_hh + chaser_hh))
                return coll_x & coll_y

            return jax.lax.cond(is_in_room, circle_chaser_collision_branch, rect_chaser_collision_branch)

        any_chaser_collision = jax.lax.cond(chaser_state.active, chaser_collision_logic, lambda: jnp.array(False))

        # Check collision with lasers.
        def check_laser_collision():
            px, py = player_state.x, player_state.y

            def check_one_laser(laser_rect):
                rect_x, rect_y, rect_w, rect_h = laser_rect

                def circle_laser_collision_branch():
                    player_radius = self.consts.PLAYER_ROOM_RADIUS
                    closest_x = jnp.clip(px, rect_x, rect_x + rect_w)
                    closest_y = jnp.clip(py, rect_y, rect_y + rect_h)
                    dist_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2
                    return dist_sq < (player_radius ** 2)

                def rect_laser_collision_branch():
                    px_hw, py_hh = player_half_width, player_half_height
                    coll_x = (jnp.abs(px - (rect_x + rect_w / 2.0)) < (px_hw + rect_w / 2.0))
                    coll_y = (jnp.abs(py - (rect_y + rect_h / 2.0)) < (py_hh + rect_h / 2.0))
                    return coll_x & coll_y

                return jax.lax.cond(is_in_room, circle_laser_collision_branch, rect_laser_collision_branch)

            x_span_start, x_span_end, y_span_start, y_span_end = self.consts.LASER_ROOM_SPAN
            room_w = x_span_end - x_span_start
            room_h = y_span_end - y_span_start

            v_laser1_rect = jnp.array(
                [laser_state.positions[0] - self.consts.LASER_THICKNESS / 2, y_span_start, self.consts.LASER_THICKNESS,
                 room_h])
            v_laser2_rect = jnp.array(
                [laser_state.positions[1] - self.consts.LASER_THICKNESS / 2, y_span_start, self.consts.LASER_THICKNESS,
                 room_h])
            h_laser1_rect = jnp.array([x_span_start, laser_state.positions[2] - self.consts.LASER_THICKNESS / 2, room_w,
                                       self.consts.LASER_THICKNESS])
            h_laser2_rect = jnp.array([x_span_start, laser_state.positions[3] - self.consts.LASER_THICKNESS / 2, room_w,
                                       self.consts.LASER_THICKNESS])

            all_laser_rects = jnp.stack([v_laser1_rect, v_laser2_rect, h_laser1_rect, h_laser2_rect])
            collisions = jax.vmap(check_one_laser)(all_laser_rects)
            return jnp.any(collisions)

        should_check_lasers = (current_level == 1) & (world_level == 1) # Lasers only active in World 1 Room 1.
        any_laser_collision = jax.lax.cond(
            should_check_lasers,
            check_laser_collision,
            lambda: jnp.array(False)
        )
        return any_monster_collision | any_dead_monster_collision | any_chaser_collision | any_laser_collision

    def _update_player(self, player_state: PlayerState, action: int, is_in_collision: bool, wall_map: chex.Array,
                       is_in_room: bool) -> \
            tuple[PlayerState, chex.Array]:
        """Updates player position based on action, handling wall collisions and boundary checks."""

        player_hw = jax.lax.cond(is_in_room,
                                 lambda: self.consts.PLAYER_DETAILED_RENDER_WIDTH / 2.0,
                                 lambda: self.consts.PLAYER_DOT_RENDER_WIDTH / 2.0)
        player_hh = jax.lax.cond(is_in_room,
                                 lambda: self.consts.PLAYER_DETAILED_RENDER_HEIGHT / 2.0,
                                 lambda: self.consts.PLAYER_DOT_RENDER_HEIGHT / 2.0)
        player_radius = float(self.consts.PLAYER_ROOM_RADIUS)

        def check_collision_circle(pos_x, pos_y):
            angles = jnp.linspace(0, 2 * jnp.pi, 8, endpoint=False)
            dx_samples = jnp.cos(angles) * player_radius
            dy_samples = jnp.sin(angles) * player_radius

            sample_points_x = pos_x + dx_samples
            sample_points_y = pos_y + dy_samples

            clipped_ix = jnp.clip(jnp.round(sample_points_x).astype(jnp.int32), 0, self.consts.SCREEN_WIDTH - 1)
            clipped_iy = jnp.clip(jnp.round(sample_points_y).astype(jnp.int32), 0, self.consts.SCREEN_HEIGHT - 1)

            map_vals = wall_map[clipped_iy, clipped_ix]
            return jnp.any(map_vals == 1)

        def check_collision_rect(pos_x, pos_y):
            corners_x = jnp.array([pos_x - player_hw, pos_x + player_hw - 1, pos_x - player_hw, pos_x + player_hw - 1])
            corners_y = jnp.array([pos_y - player_hh, pos_y - player_hh, pos_y + player_hh - 1, pos_y + player_hh - 1])

            clipped_corners_x = jnp.clip(corners_x.astype(jnp.int32), 0, self.consts.SCREEN_WIDTH - 1)
            clipped_corners_y = jnp.clip(corners_y.astype(jnp.int32), 0, self.consts.SCREEN_HEIGHT - 1)

            map_vals = wall_map[clipped_corners_y, clipped_corners_x]
            return jnp.any(map_vals == 1)

        def bounce_back(_):
            """Player reverts to last valid position if previously in collision."""
            new_player = PlayerState(
                x=player_state.last_valid_x,
                y=player_state.last_valid_y,
                last_valid_x=player_state.last_valid_x,
                last_valid_y=player_state.last_valid_y,
                last_dx=player_state.last_dx,
                last_dy=player_state.last_dy
            )
            new_collision_flag = jnp.array(False)
            return new_player, new_collision_flag

        def normal_move(_):
            """Calculates new player position and checks for collisions."""
            dx = 0.0
            dy = 0.0

            dy = jnp.where(jnp.isin(action, UPACTIONS), -self.consts.PLAYER_SPEED, dy)
            dy = jnp.where(jnp.isin(action, DOWNACTIONS), self.consts.PLAYER_SPEED, dy)
            dx = jnp.where(jnp.isin(action, LEFTACTIONS), -self.consts.PLAYER_SPEED, dx)
            dx = jnp.where(jnp.isin(action, RIGHTACTIONS), self.consts.PLAYER_SPEED, dx)

            # Update player orientation based on movement.
            is_moving = (dx != 0.0) | (dy != 0.0)
            norm = jnp.sqrt(dx ** 2 + dy ** 2)
            safe_norm = jnp.where(norm == 0, 1.0, norm)
            normalized_dx = dx / safe_norm
            normalized_dy = dy / safe_norm

            new_last_dx = jnp.where(is_moving, normalized_dx, player_state.last_dx)
            new_last_dy = jnp.where(is_moving, normalized_dy, player_state.last_dy)

            proposed_x = player_state.x + dx
            proposed_y = player_state.y + dy

            # Clip proposed position to playable area boundaries.
            min_x_clip = jax.lax.cond(is_in_room, lambda: player_radius, lambda: player_hw)
            max_x_clip = self.consts.SCREEN_WIDTH - min_x_clip
            min_y_clip = self.consts.PLAY_AREA_Y_START + jax.lax.cond(is_in_room, lambda: player_radius,
                                                                      lambda: player_hh)
            max_y_clip = self.consts.PLAY_AREA_Y_END - jax.lax.cond(is_in_room, lambda: player_radius,
                                                                    lambda: player_hh)

            proposed_x = jnp.clip(proposed_x, min_x_clip, max_x_clip)
            proposed_y = jnp.clip(proposed_y, min_y_clip, max_y_clip)

            # Check for collision with walls at the new position.
            is_colliding_now = jax.lax.cond(
                is_in_room,
                lambda: check_collision_circle(proposed_x, proposed_y),
                lambda: check_collision_rect(proposed_x, proposed_y)
            )

            new_last_valid_x = jnp.where(is_colliding_now, player_state.x, proposed_x)
            new_last_valid_y = jnp.where(is_colliding_now, player_state.y, proposed_y)

            new_player = PlayerState(
                x=proposed_x,
                y=proposed_y,
                last_valid_x=new_last_valid_x,
                last_valid_y=new_last_valid_y,
                last_dx=new_last_dx,
                last_dy=new_last_dy
            )
            return new_player, is_colliding_now

        return jax.lax.cond(is_in_collision, bounce_back, normal_move, operand=None)

    def render(self, state: GameState) -> chex.Array:
        """Renders the current game state into an image array."""
        return self.renderer.render(state)

    def _get_observation(self, state: GameState) -> VentureObservation:
        """Constructs an observation from the current game state."""
        is_in_room = state.current_level != 0

        player_width = jax.lax.cond(is_in_room,
                                    lambda: jnp.array(self.consts.PLAYER_DETAILED_RENDER_WIDTH, dtype=jnp.float32),
                                    lambda: jnp.array(self.consts.PLAYER_DOT_RENDER_WIDTH, dtype=jnp.float32))

        player_height = jax.lax.cond(is_in_room,
                                     lambda: jnp.array(self.consts.PLAYER_DETAILED_RENDER_HEIGHT, dtype=jnp.float32),
                                     lambda: jnp.array(self.consts.PLAYER_DOT_RENDER_HEIGHT, dtype=jnp.float32))

        player = ObjectObservation.create(
            x=state.player.x,
            y=state.player.y,
            width=player_width,
            height=player_height,
            active=True,
        )

        monsters_x = jnp.where(state.monsters.active, state.monsters.x, -1)  # Move inactive monsters off-screen
        monsters_y = jnp.where(state.monsters.active, state.monsters.y, -1)

        monsters = ObjectObservation.create(
            x=monsters_x,
            y=monsters_y,
            width = jnp.full((self.consts.TOTAL_MONSTERS,), self.consts.MONSTER_RENDER_WIDTH, dtype=jnp.float32),
            height = jnp.full((self.consts.TOTAL_MONSTERS,), self.consts.MONSTER_RENDER_HEIGHT, dtype=jnp.float32),
            active=state.monsters.active
        )

        # Portals (Walls)
        world_idx = state.world_level - 1
        level_idx = state.current_level
        portals_array = self.consts.JAX_TRANSITIONS[world_idx, level_idx]

        # Determine effective active portals
        portal_active = portals_array[..., 2] > 0 # active if exists
        # Could also be more sophisticated, but this is also not visible in the image!

        portals = ObjectObservation.create(
            x=jnp.where(portal_active, portals_array[:, 0] + portals_array[:, 2] / 2, -1.0),
            y=jnp.where(portal_active, portals_array[:, 1] + portals_array[:, 3] / 2, -1.0),
            width=portals_array[:, 2],
            height=portals_array[:, 3],
            active=portal_active
        )

        # Chests
        chest_global_idx = world_idx * 5 + level_idx
        chest_pos = self.consts.CHEST_POSITIONS[chest_global_idx]
        room_idx = level_idx - 1
        chest_active = (level_idx > 0) & state.chests_active[room_idx] & (state.collected_chest_in_current_visit != room_idx)

        chest = ObjectObservation.create(
            x=jnp.where(chest_active, chest_pos[0], -1.0),
            y=jnp.where(chest_active, chest_pos[1], -1.0),
            width=jnp.array(self.consts.CHEST_WIDTH, dtype=jnp.float32),
            height=jnp.array(self.consts.CHEST_HEIGHT, dtype=jnp.float32),
            active=chest_active
        )

        # Lasers
        is_laser_level = (level_idx == 1) & (state.world_level == 1)
        x_span_start, x_span_end, y_span_start, y_span_end = self.consts.LASER_ROOM_SPAN
        room_w = x_span_end - x_span_start
        room_h = y_span_end - y_span_start
        thickness = self.consts.LASER_THICKNESS

        # Four lasers: 2 vertical, 2 horizontal
        lasers_x = jnp.array([
            state.lasers.positions[0],
            state.lasers.positions[1],
            x_span_start + room_w / 2,
            x_span_start + room_w / 2
        ])
        lasers_y = jnp.array([
            y_span_start + room_h / 2,
            y_span_start + room_h / 2,
            state.lasers.positions[2],
            state.lasers.positions[3]
        ])
        lasers_w = jnp.array([thickness, thickness, room_w, room_w])
        lasers_h = jnp.array([room_h, room_h, thickness, thickness])

        lasers = ObjectObservation.create(
            x=jnp.where(is_laser_level, lasers_x, -1.0),
            y=jnp.where(is_laser_level, lasers_y, -1.0),
            width=lasers_w,
            height=lasers_h,
            active=jnp.full(4, is_laser_level, dtype=jnp.bool_)
        )

        # Chaser
        chaser = ObjectObservation.create(
            x=jnp.where(state.chaser.active, state.chaser.x, -1.0),
            y=jnp.where(state.chaser.active, state.chaser.y, -1.0),
            width=jnp.array(self.consts.CHASER_RENDER_WIDTH, dtype=jnp.float32),
            height=jnp.array(self.consts.CHASER_RENDER_HEIGHT, dtype=jnp.float32),
            active=state.chaser.active
        )
        obs = VentureObservation(
            player=player,
            monsters=monsters,
            portals=portals,
            chest=chest,
            lasers=lasers,
            chaser=chaser
        )
        return obs

    def _get_reward(self, previous_state: GameState, state: GameState) -> Array | ndarray[Any, dtype[Any]]:
        """
        Calculates the reward for the current step.
        The reward is the score gained in the step.
        """
        reward = (state.score - previous_state.score).astype(jnp.float32)

        return reward

    def _get_done(self, state: GameState) -> bool:
        """Determines if the episode has ended."""
        return state.game_over_timer == 1

    def _get_info(self, state: GameState) -> VentureInfo:
        """Provides auxiliary information about the game state."""
        return VentureInfo(time=state.step_counter, score=state.score, lives=state.lives)

    def action_space(self) -> spaces.Discrete:
        """Returns the action space of the environment."""
        return spaces.Discrete(len(Action.get_all_values()))

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space of the environment."""
        h = int(self.consts.SCREEN_HEIGHT)
        w = int(self.consts.SCREEN_WIDTH)
        screen_size = (h, w)
        single_obj = spaces.get_object_space(n=None, screen_size=screen_size)
        return spaces.Dict({
            "player": single_obj,
            "monsters": spaces.get_object_space(n=self.consts.TOTAL_MONSTERS, screen_size=screen_size),
            "portals": spaces.get_object_space(n=self.consts.JAX_TRANSITIONS.shape[2], screen_size=screen_size),  # Max portals per level based on manifest
            "chest": single_obj, 
            "lasers": spaces.get_object_space(n=4, screen_size=screen_size),  # 4 lasers max in the current game design
            "chaser": single_obj
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )


def _get_venture_asset_config() -> list[dict]:
    """Returns the declarative asset manifest for Venture."""
    return [
        # Background - procedural black
        {
            'name': 'background',
            'type': 'background',
            'data': jnp.zeros((210, 160, 4), dtype=jnp.uint8)
        },
        # Walls/Maps - load as group
        {'name': 'map_w1', 'type': 'single', 'file': 'map.npy'},
        {'name': 'room1_w1', 'type': 'single', 'file': 'room1.npy'},
        {'name': 'room2_w1', 'type': 'single', 'file': 'room2.npy'},
        {'name': 'room3_w1', 'type': 'single', 'file': 'room3.npy'},
        {'name': 'room4_w1', 'type': 'single', 'file': 'room4.npy'},
        
        {'name': 'map_w2', 'type': 'single', 'file': 'map2.npy'},
        {'name': 'room1_w2', 'type': 'single', 'file': 'room21.npy'},
        {'name': 'room2_w2', 'type': 'single', 'file': 'room22.npy'},
        {'name': 'room3_w2', 'type': 'single', 'file': 'room23.npy'},
        {'name': 'room4_w2', 'type': 'single', 'file': 'room24.npy'},

        # Player
        {'name': 'player_dot_w1', 'type': 'single', 'file': 'player_dot.npy'},
        {'name': 'player_dot_w2', 'type': 'single', 'file': 'player_dot2.npy'},
        {'name': 'player_detailed', 'type': 'single', 'file': 'player_detailed.npy'},
        
        # Monsters W1
        {'name': 'monster_map_w1', 'type': 'single', 'file': 'main_map_monster.npy'},
        {'name': 'monster_r2_w1', 'type': 'single', 'file': 'monster2.npy'},
        {'name': 'monster_r3_w1', 'type': 'single', 'file': 'monster3.npy'},
        {'name': 'monster_r4_w1', 'type': 'single', 'file': 'monster4.npy'},
        
        # Monsters W2
        {'name': 'monster_map_w2', 'type': 'single', 'file': 'main_map_monster2.npy'},
        {'name': 'monster_r1_w2', 'type': 'single', 'file': 'monster21.npy'},
        {'name': 'monster_r2_w2', 'type': 'single', 'file': 'monster22.npy'},
        {'name': 'monster_r3_w2', 'type': 'single', 'file': 'monster23.npy'},
        {'name': 'monster_r4_w2', 'type': 'single', 'file': 'monster24.npy'},

        # Dead Monsters W1
        {'name': 'monster_dead_map_w1', 'type': 'single', 'file': 'monster2_dead.npy'}, 
        {'name': 'monster_dead_r2_w1', 'type': 'single', 'file': 'monster2_dead.npy'},
        {'name': 'monster_dead_r3_w1', 'type': 'single', 'file': 'monster3_dead.npy'},
        {'name': 'monster_dead_r4_w1', 'type': 'single', 'file': 'monster4_dead.npy'},
        
        # Dead Monsters W2
        {'name': 'monster_dead_map_w2', 'type': 'single', 'file': 'monster21_dead.npy'},
        {'name': 'monster_dead_r1_w2', 'type': 'single', 'file': 'monster21_dead.npy'},
        {'name': 'monster_dead_r2_w2', 'type': 'single', 'file': 'monster22_dead.npy'},
        {'name': 'monster_dead_r3_w2', 'type': 'single', 'file': 'monster23_dead.npy'},
        {'name': 'monster_dead_r4_w2', 'type': 'single', 'file': 'monster24_dead.npy'},

        # Rewards (Chests)
        {'name': 'reward1_w1', 'type': 'single', 'file': 'reward1.npy'},
        {'name': 'reward2_w1', 'type': 'single', 'file': 'reward2.npy'},
        {'name': 'reward3_w1', 'type': 'single', 'file': 'reward3.npy'},
        {'name': 'reward4_w1', 'type': 'single', 'file': 'reward4.npy'},
        
        {'name': 'reward1_w2', 'type': 'single', 'file': 'reward21.npy'},
        {'name': 'reward2_w2', 'type': 'single', 'file': 'reward22.npy'},
        {'name': 'reward3_w2', 'type': 'single', 'file': 'reward23.npy'},
        {'name': 'reward4_w2', 'type': 'single', 'file': 'reward24.npy'},

        # Shared/UI
        {'name': 'health_w1', 'type': 'single', 'file': 'health.npy'},
        {'name': 'health_w2', 'type': 'single', 'file': 'health2.npy'},
        {'name': 'chaser', 'type': 'single', 'file': 'chaser.npy'},
        {'name': 'laser_ho', 'type': 'single', 'file': 'laser_wall_ho.npy'},
        {'name': 'laser_ve', 'type': 'single', 'file': 'laser_wall_ve.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': '{}.npy'},
    ]

class VentureRenderer(JAXGameRenderer):
    """Renders the Venture game state using JAX and sprite assets."""
    
    def __init__(self, consts: VentureConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or VentureConstants()
        
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "venture")
        asset_config = _get_venture_asset_config()
        
        # Add procedural sprites (projectiles, lasers)
        procedural_assets = self._create_procedural_assets(sprite_path)
        asset_config.extend(procedural_assets)
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
    
    def _create_procedural_assets(self, sprite_path: str) -> list[dict]:
        """Creates resized/procedural sprites for lasers and projectile."""
        assets = []
        
        # Helper to load and resize
        def load_resize(filename, target_shape, name):
            # Shape is (H, W, 4)
            path = os.path.join(sprite_path, filename)
            frame = self.jr.loadFrame(path)
            resized = jax.image.resize(frame, target_shape, method='nearest').astype(jnp.uint8)
            # Ensure 4 channels
            if resized.shape[-1] == 3:
                resized = jnp.concatenate([resized, jnp.full(resized.shape[:2] + (1,), 255, dtype=jnp.uint8)], axis=-1)
            # Add wrapper dims for 'procedural' type (it expects (H, W, 4) or usually (1, H, W, 4) for single frame?)
            # jr.load_and_setup_assets expects 'data' for procedural to be the array.
            # If it's a single sprite, it usually expects (H, W, 4) or maybe (1, H, W, 4).
            # Looking at source/seaquest:
            # "rgba = jnp.array(list(color) + [255], dtype=jnp.uint8).reshape(1, 1, 4)"
            # It seems to accept (H, W, 4) directly if it's treated as a single sprite frame? 
            # Or does it need stacking?
            # load_and_setup_assets logic:
            # if type == 'procedural':
            #    raw_data = item['data']
            #    if raw_data.ndim == 3: raw_data = raw_data[None, ...]
            # So (H,W,4) is fine, it will be expanded to (1,H,W,4).
            return {'name': name, 'type': 'procedural', 'data': resized}

        # 1. Projectile
        # Size: Radius * 2. 
        # But Radius=2, so Dia=4.
        # Check projectile target shape in original code:
        # proj_size = self.consts.PROJECTILE_RADIUS * 2
        # proj_target_shape = (proj_size, proj_size, 4)
        proj_size = int(self.consts.PROJECTILE_RADIUS)
        assets.append(load_resize('player_dot.npy', (proj_size, proj_size, 4), 'projectile_resized'))

        # 2. Lasers
        # Span: [70, 90, 95, 115] -> x_start, x_end, y_start, y_end
        x_span_start, x_span_end, y_span_start, y_span_end = self.consts.LASER_ROOM_SPAN
        room_w = int(x_span_end - x_span_start) # 90-70 = 20
        room_h = int(y_span_end - y_span_start) # 115-95= 20
        thickness = int(self.consts.LASER_THICKNESS) # 4
        
        # Vertical laser: thickness x room_h
        assets.append(load_resize('laser_wall_ve.npy', (room_h, thickness, 4), 'laser_ve_stretched'))
        
        # Horizontal laser: room_w x thickness
        assets.append(load_resize('laser_wall_ho.npy', (thickness, room_w, 4), 'laser_ho_stretched'))
        
        return assets

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """Renders the game state to an RGBA image array."""
        # Start with black background (ID 0)
        canvas = self.jr.create_object_raster(self.BACKGROUND)
        
        # --- Draw Map/Room Walls ---
        # Select correct wall mask based on world and level
        def get_wall_mask():
            # Level 0 is map. Levels 1-4 are rooms.
            # World 1: 0=map_w1, 1=room1_w1, ...
            # World 2: 0=map_w2, 1=room1_w2, ...
            
            # Since we didn't group them perfectly in a single stack (to avoid loading complexity), 
            # we can switch. Or we could have laid them out in a robust structure.
            # Given strictly 2 worlds and 5 levels, switch is fine.
            
            def _get_w1():
                return jax.lax.switch(state.current_level, [
                    lambda: self.SHAPE_MASKS['map_w1'],
                    lambda: self.SHAPE_MASKS['room1_w1'],
                    lambda: self.SHAPE_MASKS['room2_w1'],
                    lambda: self.SHAPE_MASKS['room3_w1'],
                    lambda: self.SHAPE_MASKS['room4_w1'],
                ])
                
            def _get_w2():
                return jax.lax.switch(state.current_level, [
                    lambda: self.SHAPE_MASKS['map_w2'],
                    lambda: self.SHAPE_MASKS['room1_w2'],
                    lambda: self.SHAPE_MASKS['room2_w2'],
                    lambda: self.SHAPE_MASKS['room3_w2'],
                    lambda: self.SHAPE_MASKS['room4_w2'],
                ])
                
            return jax.lax.cond(state.world_level == 1, _get_w1, _get_w2)

        wall_mask = get_wall_mask()
        canvas = self.jr.render_at(canvas, 0, 0, wall_mask)

        # --- Draw Score (Digits) ---
        score_digits = self.jr.int_to_digits(state.score, max_digits=6)
        canvas = self.jr.render_label(canvas, 8, 10, score_digits, self.SHAPE_MASKS['digits'], spacing=6, max_digits=6)

        # --- Draw Lives ---
        life_mask = jax.lax.cond(
            state.world_level == 1,
            lambda: self.SHAPE_MASKS['health_w1'],
            lambda: self.SHAPE_MASKS['health_w2']
        )
        canvas = self.jr.render_indicator(canvas, 120, 10, state.lives - 1, life_mask, spacing=10, max_value=3)

        # --- Draw Chests ---
        # Only in rooms (current_level > 0). Specific chest for specific room.
        def draw_chests(c):
            # Calculate index: 0-3 for room 1-4.
            # But wait: state.current_level is 1-4. chest_idx for chests_active is 0-3.
            chest_idx = state.current_level - 1
            
            # Lookup position
            # global_idx uses full 0-9 range (0=map, 1-4=rooms w1, 5=map2, 6-9=rooms w2)
            # The CHEST_POSITIONS array includes dummy 0 and 5 positions for maps?
            # Or is it packed?
            # Assuming CHEST_POSITIONS is (10, 2).
            global_idx = (state.world_level - 1) * 5 + state.current_level
            pos = self.consts.CHEST_POSITIONS[global_idx]
            
            # is_active logic:
            # state.chests_active is (4,) bool array for current world?
            is_active = state.chests_active[chest_idx] & (state.collected_chest_in_current_visit != chest_idx)
            
            # Center it: pos is center, need top-left for render_at
            top_left_x = (pos[0] - self.consts.CHEST_WIDTH / 2).astype(jnp.int32)
            top_left_y = (pos[1] - self.consts.CHEST_HEIGHT / 2).astype(jnp.int32)
            
            def do_render(_c):
                 # w1: reward1..4, w2: reward1..4
                def _w1():
                    return jax.lax.switch(chest_idx, [
                        lambda: self.jr.render_at(_c, top_left_x, top_left_y, self.SHAPE_MASKS['reward1_w1']),
                        lambda: self.jr.render_at(_c, top_left_x, top_left_y, self.SHAPE_MASKS['reward2_w1']),
                        lambda: self.jr.render_at(_c, top_left_x, top_left_y, self.SHAPE_MASKS['reward3_w1']),
                        lambda: self.jr.render_at(_c, top_left_x, top_left_y, self.SHAPE_MASKS['reward4_w1']),
                    ])
                def _w2():
                    return jax.lax.switch(chest_idx, [
                        lambda: self.jr.render_at(_c, top_left_x, top_left_y, self.SHAPE_MASKS['reward1_w2']),
                        lambda: self.jr.render_at(_c, top_left_x, top_left_y, self.SHAPE_MASKS['reward2_w2']),
                        lambda: self.jr.render_at(_c, top_left_x, top_left_y, self.SHAPE_MASKS['reward3_w2']),
                        lambda: self.jr.render_at(_c, top_left_x, top_left_y, self.SHAPE_MASKS['reward4_w2']),
                    ])
                return jax.lax.cond(state.world_level == 1, _w1, _w2)

            return jax.lax.cond(
                is_active,
                do_render,
                lambda _c: _c,
                c
            )

        canvas = jax.lax.cond(state.current_level > 0, draw_chests, lambda c: c, canvas)

        # --- Draw Active Monsters ---
        def draw_monsters(c):
            total_levels = 5
            global_level = (state.world_level - 1) * total_levels + state.current_level
            start_idx = self.consts.LEVEL_OFFSETS[global_level]
            end_idx = self.consts.LEVEL_OFFSETS[global_level + 1]
            
            def draw_single_monster(i, _c):
                should_draw = state.monsters.active[i]
                mx = (state.monsters.x[i] - self.consts.MONSTER_RENDER_WIDTH / 2).astype(jnp.int32)
                my = (state.monsters.y[i] - self.consts.MONSTER_RENDER_HEIGHT / 2).astype(jnp.int32)

                def do_render(__c):
                    # W1 switch logic
                    def _w1():
                         return jax.lax.switch(state.current_level, [
                             lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_map_w1']),
                             lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_map_w1']), # Room 1 uses map monster in W1
                             lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_r2_w1']),
                             lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_r3_w1']),
                             lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_r4_w1']),
                         ])
                    # W2 switch logic
                    def _w2():
                        return jax.lax.switch(state.current_level, [
                            lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_map_w2']),
                            lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_r1_w2']),
                            lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_r2_w2']),
                            lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_r3_w2']),
                            lambda: self.jr.render_at(__c, mx, my, self.SHAPE_MASKS['monster_r4_w2']),
                        ])
                    return jax.lax.cond(state.world_level == 1, _w1, _w2)

                return jax.lax.cond(
                    should_draw,
                    do_render,
                    lambda __c: __c,
                    _c
                )
            
            return jax.lax.fori_loop(start_idx, end_idx, draw_single_monster, c)
            
        canvas = draw_monsters(canvas)

        # --- Draw Dead Monsters ---
        def draw_dead_monsters(c):
             # Logic is slightly different in original code for dead monsters mapping
             # "main_map_dead_monster_filename = f'monster{w_suffix}2_dead.npy' if world_num == 1 else f'monster21_dead.npy'"
             # So W1 Map uses monster2_dead. W2 Map uses monster21_dead.
             # W1 Rooms:
             # dead_monster_sprite_branches W1: [dead2, dead2 (room1?), dead2, dead3, dead4] ??
             
             # Old code for W1 (w_suffix=""):
             # map -> monster2_dead (main_map_dead_monster_filename)
             # room1 -> monster2_dead
             # room2 -> monster2_dead
             # room3 -> monster3_dead
             # room4 -> monster4_dead
             
             # Old code for W2 (w_suffix="2"):
             # map -> monster21_dead
             # room1 (21) -> monster21_dead
             # room2 (22) -> monster22_dead
             # room3 (23) -> monster23_dead
             # room4 (24) -> monster24_dead
            
            def get_dead_monster_mask():
                def _w1():
                    return jax.lax.switch(state.current_level, [
                        lambda: self.SHAPE_MASKS['monster_dead_map_w1'], # map -> 2_dead
                        # My asset config mapped monster_dead_map_w1 to 'monster2_dead.npy'. Correct.
                        lambda: self.SHAPE_MASKS['monster_dead_map_w1'], # room1 -> 2_dead
                        lambda: self.SHAPE_MASKS['monster_dead_r2_w1'], # room2 -> 2_dead (Wait, 2_dead again?)
                        # Yes, existing code used monster2_dead for room2.
                        lambda: self.SHAPE_MASKS['monster_dead_r3_w1'],
                        lambda: self.SHAPE_MASKS['monster_dead_r4_w1'],
                    ])
                
                def _w2():
                    return jax.lax.switch(state.current_level, [
                        lambda: self.SHAPE_MASKS['monster_dead_map_w2'], # map -> 21_dead
                        lambda: self.SHAPE_MASKS['monster_dead_r1_w2'], # room1 -> 21_dead
                        lambda: self.SHAPE_MASKS['monster_dead_r2_w2'],
                        lambda: self.SHAPE_MASKS['monster_dead_r3_w2'],
                        lambda: self.SHAPE_MASKS['monster_dead_r4_w2'],
                    ])
                return jax.lax.cond(state.world_level == 1, _w1, _w2)
            
            mask = get_dead_monster_mask()

            def _loop_body(i, _c):
                should_draw = state.dead_monsters.active[i]
                mx = (state.dead_monsters.x[i] - self.consts.MONSTER_RENDER_WIDTH / 2).astype(jnp.int32)
                my = (state.dead_monsters.y[i] - self.consts.MONSTER_RENDER_HEIGHT / 2).astype(jnp.int32)
                return jax.lax.cond(
                    should_draw,
                    lambda __c: self.jr.render_at(__c, mx, my, mask),
                    lambda __c: __c,
                    _c
                )
            # Reusing same indices logic? Old code iterated 0..MAX_DEAD_MONSTERS
            return jax.lax.fori_loop(0, self.consts.MAX_DEAD_MONSTERS, _loop_body, c)

        canvas = draw_dead_monsters(canvas)

        # --- Draw Chaser ---
        canvas = jax.lax.cond(
            state.chaser.active,
            lambda c: self.jr.render_at(
                c, 
                (state.chaser.x - self.consts.CHASER_RENDER_WIDTH / 2).astype(jnp.int32), 
                (state.chaser.y - self.consts.CHASER_RENDER_HEIGHT / 2).astype(jnp.int32), 
                self.SHAPE_MASKS['chaser']
            ),
            lambda c: c,
            canvas
        )

        # --- Draw Lasers (World 1, Room 1 only) ---
        def draw_lasers(c):
            x_span_start, x_span_end, y_span_start, y_span_end = self.consts.LASER_ROOM_SPAN
            
            v_laser_width = self.consts.LASER_THICKNESS
            h_laser_height = self.consts.LASER_THICKNESS
            
            v_laser1_x = state.lasers.positions[0] - v_laser_width / 2
            c = self.jr.render_at(c, v_laser1_x.astype(jnp.int32), y_span_start.astype(jnp.int32), self.SHAPE_MASKS['laser_ve_stretched'])
            
            v_laser2_x = state.lasers.positions[1] - v_laser_width / 2
            c = self.jr.render_at(c, v_laser2_x.astype(jnp.int32), y_span_start.astype(jnp.int32), self.SHAPE_MASKS['laser_ve_stretched'])
            
            h_laser1_y = state.lasers.positions[2] - h_laser_height / 2
            c = self.jr.render_at(c, x_span_start.astype(jnp.int32), h_laser1_y.astype(jnp.int32), self.SHAPE_MASKS['laser_ho_stretched'])
            
            h_laser2_y = state.lasers.positions[3] - h_laser_height / 2
            c = self.jr.render_at(c, x_span_start.astype(jnp.int32), h_laser2_y.astype(jnp.int32), self.SHAPE_MASKS['laser_ho_stretched'])
            
            return c

        canvas = jax.lax.cond(
            (state.current_level == 1) & (state.world_level == 1),
            draw_lasers,
            lambda c: c,
            canvas
        )

        # --- Draw Player ---
        is_in_room = state.current_level != 0
        def draw_player_map(c):
             # W1: player_dot_w1, W2: player_dot_w2
             mask = jax.lax.cond(state.world_level == 1, lambda: self.SHAPE_MASKS['player_dot_w1'], lambda: self.SHAPE_MASKS['player_dot_w2'])
             px = (state.player.x - self.consts.PLAYER_DOT_RENDER_WIDTH / 2).astype(jnp.int32)
             py = (state.player.y - self.consts.PLAYER_DOT_RENDER_HEIGHT / 2).astype(jnp.int32)
             return self.jr.render_at(c, px, py, mask)

        def draw_player_room(c):
             mask = self.SHAPE_MASKS['player_detailed']
             px = (state.player.x - self.consts.PLAYER_DETAILED_RENDER_WIDTH / 2).astype(jnp.int32)
             py = (state.player.y - self.consts.PLAYER_DETAILED_RENDER_HEIGHT / 2).astype(jnp.int32)
             return self.jr.render_at(c, px, py, mask)

        canvas = jax.lax.cond(is_in_room, draw_player_room, draw_player_map, canvas)

        # --- Draw Aiming Dot / Projectile ---
        # Uses 'projectile' (which maps to player_dot files in my config? No, I defined 'projectile' group.)
        # Actually simplest to just reuse player dot mask logic
        
        def draw_aiming_dot(c):
            # Same sprite as player dot (map mode)
            mask = jax.lax.cond(state.world_level == 1, lambda: self.SHAPE_MASKS['player_dot_w1'], lambda: self.SHAPE_MASKS['player_dot_w2'])
            # Dot offset
            dot_x = state.player.x + state.player.last_dx * self.consts.AIMING_DOT_OFFSET
            dot_y = state.player.y + state.player.last_dy * self.consts.AIMING_DOT_OFFSET
            
            # Dimensions of dot? Old code: 
            # proj_sprite = jr.get_sprite_frame(self.sprites['player_dot'][world_idx], 0)
            # px = ... - proj_size_x / 2
            
            # Using constant dims for safety (1x2)
            px = (dot_x - self.consts.PLAYER_DOT_RENDER_WIDTH / 2).astype(jnp.int32)
            py = (dot_y - self.consts.PLAYER_DOT_RENDER_HEIGHT / 2).astype(jnp.int32)
            return self.jr.render_at(c, px, py, mask)

        def draw_projectile_active(c):
             mask = jax.lax.cond(state.world_level == 1, lambda: self.SHAPE_MASKS['player_dot_w1'], lambda: self.SHAPE_MASKS['player_dot_w2'])
             # In old code, projectile uses player_dot sprite but resized to PROJECTILE_RADIUS*2 ?
             # "proj_size = self.consts.PROJECTILE_RADIUS * 2 ... jax.image.resize(player_dot_frame...)"
             # So the projectile is BIGGER than the player dot?
             # PLAYER_DOT_RENDER_WIDTH=1, HEIGHT=2.
             # PROJECTILE_RADIUS=2 -> dia=4.
             # So I need a separate sprite for the projectile!
             # My asset config reused `player_dot_files`. That's wrong if the size is different.
             # I need to create a procedural resized version of player_dot for the projectile.
             
             mask = self.SHAPE_MASKS['projectile_resized'] # We will create this
             px = (state.projectile.x - self.consts.PROJECTILE_RADIUS).astype(jnp.int32)
             py = (state.projectile.y - self.consts.PROJECTILE_RADIUS).astype(jnp.int32)
             return self.jr.render_at(c, px, py, mask)

        canvas = jax.lax.cond(
            is_in_room & ~state.projectile.active,
            draw_aiming_dot,
            lambda c: c,
            canvas
        )
        canvas = jax.lax.cond(
            state.projectile.active,
            draw_projectile_active,
            lambda c: c,
            canvas
        )

        return self.jr.render_from_palette(canvas, self.PALETTE)