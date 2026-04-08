import jax.numpy as jnp
import jax.random as jrandom
from flax import struct
import os

from jaxatari.environment import ObjectObservation

class Montezuma2Constants(struct.PyTreeNode):
    # Homogeneous Padding Limits
    MAX_ENEMIES_PER_ROOM: int = struct.field(pytree_node=False, default=3)
    MAX_LADDERS_PER_ROOM: int = struct.field(pytree_node=False, default=4)
    MAX_ROPES_PER_ROOM: int = struct.field(pytree_node=False, default=2)
    MAX_DOORS_PER_ROOM: int = struct.field(pytree_node=False, default=2)
    MAX_ITEMS_PER_ROOM: int = struct.field(pytree_node=False, default=3)
    MAX_CONVEYORS_PER_ROOM: int = struct.field(pytree_node=False, default=1)
    MAX_LASERS_PER_ROOM: int = struct.field(pytree_node=False, default=8)
    MAX_PLATFORMS_PER_ROOM: int = struct.field(pytree_node=False, default=12)
    MAX_ROOMS: int = struct.field(pytree_node=False, default=33)
    
    # Gameplay Constants
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=7)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=20)
    INITIAL_PLAYER_X: int = struct.field(pytree_node=False, default=77)
    INITIAL_PLAYER_Y: int = struct.field(pytree_node=False, default=26)
    INITIAL_ROOM_ID: int = struct.field(pytree_node=False, default=11)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=1)
    
    # Room Transition Coordinates
    ROOM_ENTRY_X_LEFT: int = struct.field(pytree_node=False, default=4)
    ROOM_ENTRY_X_RIGHT: int = struct.field(pytree_node=False, default=148)
    ROOM_ENTRY_Y_TOP: int = struct.field(pytree_node=False, default=6)
    ROOM_ENTRY_Y_BOTTOM: int = struct.field(pytree_node=False, default=129)
    ROOM_EXIT_Y_TOP: int = struct.field(pytree_node=False, default=2)
    ROOM_EXIT_Y_BOTTOM: int = struct.field(pytree_node=False, default=130)
    
    JUMP_Y_OFFSETS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0, -1, -1, -2, -2, -2, -3, -3, -3], dtype=jnp.int32))
    GRAVITY: int = struct.field(pytree_node=False, default=2)
    MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # HUD Constants
    SCORE_X: int = struct.field(pytree_node=False, default=56)
    SCORE_Y: int = struct.field(pytree_node=False, default=6)
    LIFES_STARTING_Y: int = struct.field(pytree_node=False, default=15)
    ITEMBAR_STARTING_Y: int = struct.field(pytree_node=False, default=28)
    ITEMBAR_LIFES_STARTING_X: int = struct.field(pytree_node=False, default=56)
    DIGIT_WIDTH: int = struct.field(pytree_node=False, default=7)
    DIGIT_OFFSET: int = struct.field(pytree_node=False, default=1)
    DIGIT_HEIGHT: int = struct.field(pytree_node=8, default=8)
    
    # Gameplay Rules
    OUT_OF_LADDER_DELAY: int = struct.field(pytree_node=False, default=5)
    MAX_FALL_DISTANCE: int = struct.field(pytree_node=False, default=33) # ladder_height (39) - 6
    BOUNCE_OFFSETS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 27, 27, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0], dtype=jnp.int32))
    DEATH_TIMER_FRAMES: int = struct.field(pytree_node=False, default=70)
    PLATFORM_ACTIVE_DURATION: int = struct.field(pytree_node=False, default=90) # For spawning and disappearing platform
    PLATFORM_CYCLE_LENGTH: int = struct.field(pytree_node=False, default=128)
    AMULET_DURATION: int = struct.field(pytree_node=False, default=660)

@struct.dataclass
class Montezuma2State:
    # Game State
    room_id: jnp.ndarray
    lives: jnp.ndarray
    score: jnp.ndarray
    frame_count: jnp.ndarray
    
    # Player State
    player_x: jnp.ndarray
    player_y: jnp.ndarray
    player_vx: jnp.ndarray
    player_vy: jnp.ndarray
    player_dir: jnp.ndarray
    
    entry_x: jnp.ndarray
    entry_y: jnp.ndarray
    entry_is_climbing: jnp.ndarray
    entry_last_ladder: jnp.ndarray
    
    is_jumping: jnp.ndarray
    is_falling: jnp.ndarray
    fall_start_y: jnp.ndarray
    jump_counter: jnp.ndarray
    is_climbing: jnp.ndarray
    out_of_ladder_delay: jnp.ndarray
    last_rope: jnp.ndarray
    last_ladder: jnp.ndarray
    
    # Homogeneous Entities for the CURRENT room
    enemies_x: jnp.ndarray
    enemies_y: jnp.ndarray
    enemies_active: jnp.ndarray
    enemies_direction: jnp.ndarray
    enemies_type: jnp.ndarray
    enemies_min_x: jnp.ndarray
    enemies_max_x: jnp.ndarray
    enemies_bouncing: jnp.ndarray
    
    ladders_x: jnp.ndarray
    ladders_top: jnp.ndarray
    ladders_bottom: jnp.ndarray
    ladders_active: jnp.ndarray
    
    ropes_x: jnp.ndarray
    ropes_top: jnp.ndarray
    ropes_bottom: jnp.ndarray
    ropes_active: jnp.ndarray
    
    items_x: jnp.ndarray
    items_y: jnp.ndarray
    items_active: jnp.ndarray
    items_type: jnp.ndarray
    
    doors_x: jnp.ndarray
    doors_y: jnp.ndarray
    doors_active: jnp.ndarray
    global_doors_active: jnp.ndarray
    global_items_active: jnp.ndarray
    global_items_type: jnp.ndarray
    global_enemies_active: jnp.ndarray
    global_enemies_type: jnp.ndarray
    
    conveyors_x: jnp.ndarray
    conveyors_y: jnp.ndarray
    conveyors_active: jnp.ndarray
    conveyors_direction: jnp.ndarray
    
    lasers_x: jnp.ndarray
    lasers_active: jnp.ndarray
    laser_cycle: jnp.ndarray
    
    platforms_x: jnp.ndarray
    platforms_y: jnp.ndarray
    platforms_width: jnp.ndarray
    platforms_active: jnp.ndarray
    platform_cycle: jnp.ndarray
    
    death_timer: jnp.ndarray
    death_type: jnp.ndarray
    
    inventory: jnp.ndarray # keys, sword, torch, amulet
    amulet_time: jnp.ndarray
    bonus_room_timer: jnp.ndarray
    first_gem_pickup: jnp.ndarray
    key: jrandom.PRNGKey

@struct.dataclass
class Montezuma2Observation:
    player: ObjectObservation
    enemies: ObjectObservation
    items: ObjectObservation
    conveyors: ObjectObservation
    doors: ObjectObservation
    ropes: ObjectObservation
    platforms: ObjectObservation

@struct.dataclass
class Montezuma2Info:
    lives: jnp.ndarray
    room_id: jnp.ndarray

def get_room_idx(room_id):
    return jnp.where(room_id == 3, 0,
           jnp.where(room_id == 4, 1,
           jnp.where(room_id == 5, 2,
           jnp.where(room_id == 11, 3,
           jnp.where(room_id == 10, 4,
           jnp.where(room_id == 12, 5,
           jnp.where(room_id == 13, 6,
           jnp.where(room_id == 14, 7,
           jnp.where(room_id == 18, 8,
           jnp.where(room_id == 17, 9,
           jnp.where(room_id == 19, 10,
           jnp.where(room_id == 20, 11,
           jnp.where(room_id == 21, 12,
           jnp.where(room_id == 22, 13,
           jnp.where(room_id == 23, 14,
           jnp.where(room_id == 31, 15, 
           jnp.where(room_id == 32, 16, 
           jnp.where(room_id == 30, 17, 
           jnp.where(room_id == 28, 18, 
           jnp.where(room_id == 27, 19,
           jnp.where(room_id == 29, 20,
           jnp.where(room_id == 25, 21,
           jnp.where(room_id == 26, 22,
           jnp.where(room_id == 24, 23, 0))))))))))))))))))))))))

def check_platform(col_map, y, x, width):
    x_m2 = jnp.clip(x - 2, 0, width - 1)
    x_m1 = jnp.clip(x - 1, 0, width - 1)
    x_p1 = jnp.clip(x + 1, 0, width - 1)
    x_p2 = jnp.clip(x + 2, 0, width - 1)
    return jnp.logical_or(
        col_map[y, x_m2] == 1,
        jnp.logical_or(
            col_map[y, x_m1] == 1,
            jnp.logical_or(
                col_map[y, x, ...] == 1,
                jnp.logical_or(
                    col_map[y, x_p1] == 1,
                    col_map[y, x_p2] == 1
                )
            )
        )
    )
