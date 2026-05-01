import os
from typing import Dict, Any, Tuple
from functools import lru_cache
import jax
import jax.numpy as jnp
import chex
from flax import struct
from jaxatari.environment import JAXAtariAction as Action, ObjectObservation
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
    sprite_rgba = jnp.load(sprite_path)
    if sprite_rgba.ndim == 3 and sprite_rgba.shape[2] == 3:
        alpha = jnp.full(sprite_rgba.shape[:2] + (1,), 255, dtype=jnp.uint8)
        sprite_rgba = jnp.concatenate([sprite_rgba, alpha], axis=2)
    target_shape = (210, 160, 4)
    assert sprite_rgba.shape == target_shape, f"Expected sprite shape {target_shape}, got {sprite_rgba.shape} for {sprite_path}"
    rgb_channels = sprite_rgba[:, :, :3]
    color_sum = jnp.sum(rgb_channels.astype(jnp.int32), axis=-1)
    wall_map = jnp.where(color_sum > 0, jnp.uint8(1), jnp.uint8(0))
    return wall_map

SPRITE_MAP_PATHS = {
    0: 'map.npy',
    1: 'room1.npy',
    2: 'room2.npy',
    3: 'room3.npy',
    4: 'room4.npy',
}
SPAWN_BOUNDARY_OFFSET_ENTER = 6.0
SPAWN_BOUNDARY_OFFSET_EXIT = 1.0
ROOM_PORTAL_PADDING = 4.0

def _calculate_spawn(rect, push_vector, target_level):
    x, y, w, h = rect
    center_x = x + w / 2
    center_y = y + h / 2
    is_entering_room = (target_level > 0)
    offset = jnp.where(is_entering_room, SPAWN_BOUNDARY_OFFSET_ENTER, SPAWN_BOUNDARY_OFFSET_EXIT)
    spawn_x = center_x + push_vector[0] * (w / 2 + offset)
    spawn_y = center_y + push_vector[1] * (h / 2 + offset)
    return spawn_x, spawn_y

def _build_wall_maps_per_world() -> chex.Array:
    all_worlds_wall_maps_list = []
    base_sprite_path = os.path.join(render_utils.get_base_sprite_dir(), 'venture')
    for world_num in [1, 2]:
        world_suffix = "" if world_num == 1 else str(world_num)
        current_world_level_maps = []
        for level_id in range(len(SPRITE_MAP_PATHS)):
            if level_id == 0:
                sprite_filename = f'map{world_suffix}.npy'
            else:
                sprite_filename = f'room{world_suffix}{level_id}.npy'
            full_path = os.path.join(base_sprite_path, sprite_filename)
            current_world_level_maps.append(_create_wall_map_from_sprite(full_path))
        all_worlds_wall_maps_list.append(jnp.stack(current_world_level_maps, axis=0))
    return jnp.stack(all_worlds_wall_maps_list, axis=0)

def _build_all_world_portal_definitions() -> Dict[int, Dict[int, list[Dict[str, Any]]]]:
    world1_portal_definitions = {
        0: [
            {"rect": [20, 60, 4, 4], "to": (1, *_calculate_spawn([28, 100, 4, 8], (1, 0), target_level=1))},
            {"rect": [48, 52, 4, 4], "to": (1, *_calculate_spawn([128, 100, 4, 8], (-1, 0), target_level=1))},
            {"rect": [88, 44, 4, 4], "to": (2, *_calculate_spawn([16, 52, 4, 8], (1, 0), target_level=2))},
            {"rect": [136, 44, 4, 4], "to": (2, *_calculate_spawn([140, 52, 4, 8], (-1, 0), target_level=2))},
            {"rect": [32, 96, 5, 4], "to": (3, *_calculate_spawn([56, 24, 4, 4], (0, 1), target_level=3))},
            {"rect": [60, 148, 4, 4], "to": (3, *_calculate_spawn([140, 148, 4, 8], (-1, 0), target_level=3))},
            {"rect": [108, 120, 5, 4], "to": (4, *_calculate_spawn([60, 76, 4, 4], (0, -1), target_level=4))},
            {"rect": [140, 120, 4, 4], "to": (4, *_calculate_spawn([140, 48, 4, 8], (-1, 0), target_level=4))},
        ],
        1: [
            {"rect": [28, 100, 4, 8], "to": (0, *_calculate_spawn([20, 60, 4, 4], (-1, 0), target_level=0))},
            {"rect": [128, 100, 4, 8], "to": (0, *_calculate_spawn([48, 52, 4, 4], (1, 0), target_level=0))},
        ],
        2: [
            {"rect": [16, 52, 4, 8], "to": (0, *_calculate_spawn([88, 44, 4, 4], (-1, 0), target_level=0))},
            {"rect": [140, 52, 4, 8], "to": (0, *_calculate_spawn([136, 44, 4, 4], (1, 0), target_level=0))},
        ],
        3: [
            {"rect": [56, 24, 4, 4], "to": (0, *_calculate_spawn([32, 96, 4, 4], (0, -1), target_level=0))},
            {"rect": [140, 148, 4, 8], "to": (0, *_calculate_spawn([60, 148, 4, 4], (1, 0), target_level=0))},
        ],
        4: [
            {"rect": [60, 76, 4, 4], "to": (0, *_calculate_spawn([108, 120, 4, 4], (0, 1), target_level=0))},
            {"rect": [140, 48, 4, 8], "to": (0, *_calculate_spawn([140, 120, 4, 4], (1, 0), target_level=0))},
        ]
    }
    world2_portal_definitions = {
        0: [
            {"rect": [16, 40, 4, 4], "to": (1, *_calculate_spawn([16, 44, 4, 8], (1, 0), target_level=1))},
            {"rect": [60, 40, 4, 4], "to": (1, *_calculate_spawn([140, 44, 4, 8], (-1, 0), target_level=1))},
            {"rect": [112, 72, 5, 4], "to": (2, *_calculate_spawn([76, 180, 8, 4], (0, -1), target_level=2))},
            {"rect": [80, 108, 5, 4], "to": (3, *_calculate_spawn([76, 180, 8, 4], (0, -1), target_level=3))},
            {"rect": [16, 144, 4, 4], "to": (4, *_calculate_spawn([16, 100, 4, 8], (1, 0), target_level=4))},
            {"rect": [140, 144, 4, 4], "to": (4, *_calculate_spawn([140, 100, 4, 8], (-1, 0), target_level=4))},
        ],
        1: [
            {"rect": [16, 44, 4, 8], "to": (0, *_calculate_spawn([16, 40, 4, 4], (-1, 0), target_level=0))},
            {"rect": [140, 44, 4, 8], "to": (0, *_calculate_spawn([60, 40, 4, 4], (1, 0), target_level=0))},
        ],
        2: [
            {"rect": [76, 180, 8, 4], "to": (0, *_calculate_spawn([112, 72, 4, 4], (0, 1), target_level=0))},
        ],
        3: [
            {"rect": [76, 180, 8, 4], "to": (0, *_calculate_spawn([80, 108, 4, 4], (0, 1), target_level=0))},
        ],
        4: [
            {"rect": [16, 100, 4, 8], "to": (0, *_calculate_spawn([16, 144, 4, 4], (-1, 0), target_level=0))},
            {"rect": [140, 100, 4, 8], "to": (0, *_calculate_spawn([140, 144, 4, 4], (1, 0), target_level=0))},
        ]
    }
    return {1: world1_portal_definitions, 2: world2_portal_definitions}

def _build_jax_transitions(all_world_portal_definitions: Dict[int, Dict[int, list[Dict[str, Any]]]]) -> chex.Array:
    transitions_per_world_list = []
    for world_id in sorted(all_world_portal_definitions.keys()):
        world_portal_data = all_world_portal_definitions[world_id]
        current_world_transitions_list = []
        num_levels = max(world_portal_data.keys()) + 1 if world_portal_data else 5
        for source_level_id in range(num_levels):
            level_portals_data = world_portal_data.get(source_level_id, [])
            level_portals_list = []
            for portal in level_portals_data:
                rect = portal["rect"]
                x, y, w, h = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
                is_in_room = (source_level_id > 0)
                if is_in_room:
                    x -= ROOM_PORTAL_PADDING
                    y -= ROOM_PORTAL_PADDING
                    w += 2 * ROOM_PORTAL_PADDING
                    h += 2 * ROOM_PORTAL_PADDING
                target_level, spawn_x, spawn_y = portal["to"]
                level_portals_list.append([x, y, w, h, float(target_level), float(spawn_x), float(spawn_y)])
            current_world_transitions_list.append(level_portals_list)
        transitions_per_world_list.append(current_world_transitions_list)
    max_portals_per_level = 0
    for world_list in transitions_per_world_list:
        if any(world_list):
            max_portals_per_level = max(max_portals_per_level, max(len(p) for p in world_list if p))
    final_jax_transitions_list = []
    for world_list in transitions_per_world_list:
        padded_world_list = []
        for level_list in world_list:
            padding_needed = max_portals_per_level - len(level_list)
            if padding_needed > 0:
                level_list.extend([[0.0] * 7] * padding_needed)
            padded_world_list.append(level_list)
        final_jax_transitions_list.append(padded_world_list)
    return jnp.array(final_jax_transitions_list, dtype=jnp.float32)

def _build_main_map_portal_masks(all_world_portal_definitions, screen_height, screen_width) -> Tuple[chex.Array, chex.Array]:
    max_portals = max(len(world_defs.get(0, [])) for world_defs in all_world_portal_definitions.values())
    all_world_masks, all_world_to_levels = [], []
    for world_id in sorted(all_world_portal_definitions.keys()):
        level0_portals = all_world_portal_definitions[world_id].get(0, [])
        world_masks, world_to_levels = [], []
        for portal in level0_portals:
            x, y, w, h = portal["rect"]
            to_level = int(portal["to"][0])
            mask = jnp.zeros((screen_height, screen_width), dtype=jnp.bool_)
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + int(w), screen_width), min(y0 + int(h), screen_height)
            mask = mask.at[y0:y1, x0:x1].set(True)
            world_masks.append(mask)
            world_to_levels.append(to_level)
        padding_needed = max_portals - len(world_masks)
        if padding_needed > 0:
            world_masks.extend([jnp.zeros((screen_height, screen_width), dtype=jnp.bool_)] * padding_needed)
            world_to_levels.extend([0] * padding_needed)
        all_world_masks.append(jnp.stack(world_masks, axis=0))
        all_world_to_levels.append(jnp.array(world_to_levels, dtype=jnp.int32))
    return jnp.stack(all_world_masks, axis=0), jnp.stack(all_world_to_levels, axis=0)

def _build_level_monster_configs() -> Tuple[Dict[str, Any], ...]:
    return (
        {"num": 6, "spawns": jnp.array([[10, 36], [60, 77], [54, 127], [110, 74], [10, 126], [150, 127]], dtype=jnp.float32), "is_immortal": jnp.array([False] * 6)},
        {"num": 0, "spawns": jnp.empty((0, 2), dtype=jnp.float32), "is_immortal": jnp.empty((0,), dtype=jnp.bool_)},
        {"num": 3, "spawns": jnp.array([[70.0, 50.0], [120.0, 120.0], [130.0, 130.0]], dtype=jnp.float32), "is_immortal": jnp.array([False] * 3)},
        {"num": 3, "spawns": jnp.array([[40.0, 80.0], [50.0, 140.0], [100.0, 150.0]], dtype=jnp.float32), "is_immortal": jnp.array([False] * 3)},
        {"num": 3, "spawns": jnp.array([[90.0, 40.0], [50.0, 150.0], [120.0, 90.0]], dtype=jnp.float32), "is_immortal": jnp.array([False] * 3)},
        {"num": 6, "spawns": jnp.array([[70, 47], [10, 76], [7, 117], [130, 67], [120, 116], [124, 167]], dtype=jnp.float32), "is_immortal": jnp.array([False] * 6)},
        {"num": 3, "spawns": jnp.array([[72, 85], [115, 37], [41, 35]], dtype=jnp.float32), "is_immortal": jnp.array([False] * 3)},
        {"num": 3, "spawns": jnp.array([[73, 38], [123.0, 59.0], [93.0, 109.0]], dtype=jnp.float32), "is_immortal": jnp.array([False] * 3)},
        {"num": 3, "spawns": jnp.array([[61.0, 109.0], [74, 65], [101, 118]], dtype=jnp.float32), "is_immortal": jnp.array([False] * 3)},
        {"num": 3, "spawns": jnp.array([[42, 82], [112, 83], [72, 103]], dtype=jnp.float32), "is_immortal": jnp.array([False] * 3)},
    )

@lru_cache(maxsize=1)
def _build_venture_static_data() -> Dict[str, Any]:
    level_monster_configs = _build_level_monster_configs()
    all_world_portal_definitions = _build_all_world_portal_definitions()
    total_monsters = sum(config["num"] for config in level_monster_configs)
    level_offsets = jnp.cumsum(jnp.array([0] + [c["num"] for c in level_monster_configs], dtype=jnp.int32))
    all_monster_spawns = jnp.concatenate([c["spawns"] for c in level_monster_configs]).astype(jnp.float32)
    all_monster_immortal_flags = jnp.concatenate([c["is_immortal"] for c in level_monster_configs])
    main_map_portal_masks, main_map_portal_to_levels = _build_main_map_portal_masks(all_world_portal_definitions, 210, 160)
    return {
        "all_wall_maps_per_world": _build_wall_maps_per_world(),
        "level_monster_configs": level_monster_configs,
        "total_monsters": total_monsters,
        "level_offsets": level_offsets,
        "all_monster_spawns": all_monster_spawns,
        "all_monster_immortal_flags": all_monster_immortal_flags,
        "jax_transitions": _build_jax_transitions(all_world_portal_definitions),
        "main_map_portal_masks": main_map_portal_masks,
        "main_map_portal_to_levels": main_map_portal_to_levels,
    }

class MonsterState(struct.PyTreeNode):
    x: chex.Array; y: chex.Array; dx: chex.Array; dy: chex.Array; active: chex.Array; is_immortal: chex.Array; dead_for: chex.Array
class ProjectileState(struct.PyTreeNode):
    x: chex.Array; y: chex.Array; dx: chex.Array; dy: chex.Array; active: chex.Array; lifetime: chex.Array
class ChaserState(struct.PyTreeNode):
    x: chex.Array; y: chex.Array; active: chex.Array
class PlayerState(struct.PyTreeNode):
    x: chex.Array; y: chex.Array; last_valid_x: chex.Array; last_valid_y: chex.Array; last_dx: chex.Array; last_dy: chex.Array
class LaserState(struct.PyTreeNode):
    positions: chex.Array; directions: chex.Array

class GameState(struct.PyTreeNode):
    player: PlayerState; monsters: MonsterState; projectile: ProjectileState; chaser: ChaserState; lasers: LaserState
    chests_active: chex.Array; kill_bonus_active: chex.Array; key: jax.random.PRNGKey; game_over_timer: chex.Array
    life_lost_timer: chex.Array; level_timer: chex.Array; step_counter: chex.Array; score: chex.Array; lives: chex.Array
    is_in_collision: chex.Array; current_level: chex.Array; world_level: chex.Array; monster_speed_index: chex.Array
    world_transition_timer: chex.Array; last_level: chex.Array; collected_chest_in_current_visit: chex.Array

class VentureObservation(struct.PyTreeNode):
    player: ObjectObservation; monsters: ObjectObservation; portals: ObjectObservation; chest: ObjectObservation
    lasers: ObjectObservation; chaser: ObjectObservation
class VentureInfo(struct.PyTreeNode):
    time: jnp.ndarray; score: jnp.ndarray; lives: jnp.ndarray

class VentureConstants(AutoDerivedConstants):
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=210)
    PLAYER_SPEED: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(1, dtype=jnp.float32))
    PLAYER_DOT_RENDER_WIDTH: int = struct.field(pytree_node=False, default=1); PLAYER_DOT_RENDER_HEIGHT: int = struct.field(pytree_node=False, default=2)
    PLAYER_DETAILED_RENDER_WIDTH: int = struct.field(pytree_node=False, default=6); PLAYER_DETAILED_RENDER_HEIGHT: int = struct.field(pytree_node=False, default=6)
    PLAYER_ROOM_RADIUS: int = struct.field(pytree_node=False, default=3)
    ALL_WALL_MAPS_PER_WORLD: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.zeros((2, 5, 210, 160), dtype=jnp.uint8))
    PLAY_AREA_Y_START: int = struct.field(pytree_node=False, default=20); PLAY_AREA_Y_END: int = struct.field(pytree_node=False, default=180)
    MONSTER_SPEEDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.5, 2.0, 2.5], dtype=jnp.float32))
    MAX_MONSTER_SPEED_INDEX: int = struct.field(pytree_node=False, default=3)
    MONSTER_RENDER_WIDTH: int = struct.field(pytree_node=False, default=7); MONSTER_RENDER_HEIGHT: int = struct.field(pytree_node=False, default=10)
    MONSTER_CHANGE_DIR_PROB: float = struct.field(pytree_node=False, default=0.01)
    DEAD_MONSTER_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=90)
    LIVES: int = struct.field(pytree_node=False, default=4)
    PLAYER_INITIAL_X: float = struct.field(pytree_node=False, default=67.0); PLAYER_INITIAL_Y: float = struct.field(pytree_node=False, default=185.0)
    FINAL_GAME_OVER_DELAY_FRAMES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(60, dtype=jnp.int32))
    LIFE_LOST_DELAY_FRAMES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(45, dtype=jnp.int32))
    WORLD_TRANSITION_DELAY_FRAMES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(90, dtype=jnp.int32))
    PROJECTILE_SPEED: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(2.0, dtype=jnp.float32))
    PROJECTILE_RADIUS: int = struct.field(pytree_node=False, default=2); PROJECTILE_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=30)
    AIMING_DOT_OFFSET: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(6.0, dtype=jnp.float32))
    CHEST_WIDTH: int = struct.field(pytree_node=False, default=7); CHEST_HEIGHT: int = struct.field(pytree_node=False, default=11); CHEST_SCORE: int = struct.field(pytree_node=False, default=200)
    CHEST_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[0.0, 0.0], [80.0, 105.0], [115.0, 170.0], [30.0, 170.0], [30.0, 170.0], [0.0, 0.0], [80, 163], [120, 35], [75, 35], [80, 67]], dtype=jnp.float32))
    CHASER_SPAWN_FRAMES: int = struct.field(pytree_node=False, default=1080); CHASER_SPEED: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(0.4, dtype=jnp.float32))
    CHASER_RENDER_WIDTH: int = struct.field(pytree_node=False, default=5); CHASER_RENDER_HEIGHT: int = struct.field(pytree_node=False, default=15); CHASER_SPAWN_POS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([10.0, 30.0], dtype=jnp.float32))
    LASER_SPEED: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(0.3, dtype=jnp.float32)); LASER_THICKNESS: float = struct.field(pytree_node=False, default_factory=lambda: jnp.array(4.0, dtype=jnp.float32))
    LASER_BOUNDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[40.0, 65.0], [120.0, 95.0], [45.0, 85.0], [130.0, 170.0]], dtype=jnp.float32))
    LASER_INITIAL_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([70.0, 90.0, 95.0, 115.0], dtype=jnp.float32)); LASER_INITIAL_DIRECTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float32))
    LASER_ROOM_SPAN: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([70.0, 90.0, 95.0, 115.0], dtype=jnp.float32))
    LEVEL_MONSTER_CONFIGS: Tuple[Dict, ...] = struct.field(pytree_node=False, default_factory=tuple); TOTAL_MONSTERS: int = struct.field(pytree_node=False, default=0); LEVEL_OFFSETS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.zeros((1,), dtype=jnp.int32)); ALL_MONSTER_SPAWNS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.empty((0, 2), dtype=jnp.float32)); ALL_MONSTER_IMMORTAL_FLAGS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.empty((0,), dtype=jnp.bool_)); JAX_TRANSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.zeros((2, 5, 1, 7), dtype=jnp.float32)); MAIN_MAP_PORTAL_MASKS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.zeros((2, 1, 210, 160), dtype=jnp.bool_)); MAIN_MAP_PORTAL_TO_LEVELS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
