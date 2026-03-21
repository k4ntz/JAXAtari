from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.spaces import Discrete, Box
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
import chex
import jax.lax
import os
from functools import partial
from PIL import Image
import numpy as np
import jaxatari.spaces as spaces
import time

@partial(jax.jit, static_argnums=(0,))
def compute_binary_matrix(level: int, data: jnp.ndarray) -> jnp.ndarray:
    """
    Converts an image array containing white and purple
    pixels into a binary matrix (purple=1, white=0).
    """
    # Purple in the provided floor_one.npy file is represented as [80, 0, 132, 255]
    purple_color = jnp.array([80, 0, 132, 255], dtype=data.dtype)
    red_color = jnp.array([148, 0, 0, 255], dtype=data.dtype)
    black_color = jnp.array([0, 0, 0, 255], dtype=data.dtype)
    brown_color = jnp.array([72, 44, 0, 255], dtype=data.dtype)

    map_colors = [purple_color, red_color, black_color, brown_color]

    # Check if we have an RGBA image (shape ends with 4)
    # This branch is evaluated at trace time based on the static shape of the input
    if data.shape[-1] == 4:
        is_map_color = jnp.all(data == map_colors[level - 1], axis=-1)
    else:
        is_map_color = jnp.all(data[..., :3] == map_colors[level - 1][:3], axis=-1)
        
    # Convert boolean mask to binary matrix (0s and 1s)
    binary_matrix = is_map_color.astype(jnp.int8)
    # Ensure sides are not walkable (0)
    binary_matrix = binary_matrix.at[:, 0].set(0)
    binary_matrix = binary_matrix.at[:, -1].set(0)
    return binary_matrix

    
def create_binary_matrix(level: int, npy_path: str) -> jnp.ndarray:
    """
    Reads a .npy image file containing white (or transparent black) and purple pixels,
    and returns a binary matrix where white/transparent pixels are 0 and purple pixels are 1.
        
    Args:
        npy_path: The path to the .npy file (e.g., floor_one.npy)
        
    Returns:
        A 2D JAX array containing 0s and 1s.
    """
    # File I/O must be done outside of JAX jit, so we load with numpy
    data_np = np.load(npy_path)
    
    # Process the loaded array with our jitted JAX function
    binary_matrix = compute_binary_matrix(level, jnp.array(data_np))
    
    return binary_matrix

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for BankHeist.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    # Define file lists for groups
    # tombs = [f"tomb_{i + 1}.npy" for i in range(4)]

    # Define the sprites
    config = (
        # Backgrounds (loaded as a group)
        # Note: The 'background' type is not used here, as the city map is the primary background.
        # We will treat 'tombs' as our base background sprites.

        # Roomparts
        {'name': 'floor', 'type': 'group', 'files': ['floor_map1.npy', 'floor_map2.npy', 'floor_map3.npy', 'floor_map4.npy']},
        {'name': 'flash_floor', 'type': 'group', 'files': ['flash_floor_1.npy', 'flash_floor_2.npy', 'flash_floor_3.npy', 'flash_floor_4.npy']},

        # Player (loaded as groups for automatic padding to largest sprite)
        {'name': 'player', 'type': 'group', 'files': ['player_idle.npy', 'player_key_idle.npy']},
        {'name': 'player_move', 'type': 'group', 'files': ['player_move_00.npy', 'player_move_01.npy', 'player_key_move_00.npy', 'player_key_move_01.npy']},
        {'name': 'player_death ', 'type': 'single', 'file': 'player_death.npy'},
        {'name': 'bullet', 'type': 'group', 'files': ['bullet_00.npy', 'bullet_01.npy', 'bullet_02.npy', 'bullet_03.npy']},

        # Creatures (loaded as single sprites for manual padding)
        {'name': 'creature_00', 'type': 'group', 'files': ['creature_snake_00.npy', 'creature_scorpion_00.npy', 'creature_bat_00.npy', 'creature_turtle_00.npy', 'creature_jackel_00.npy', 'creature_condor_00.npy', 'creature_lion_00.npy', 'creature_moth_00.npy', 'creature_virus_00.npy', 'creature_monkey_00.npy', 'creature_mysteryweapon_00.npy']},
        {'name': 'creature_01', 'type': 'group', 'files': ['creature_snake_01.npy', 'creature_scorpion_01.npy', 'creature_bat_01.npy', 'creature_turtle_01.npy', 'creature_jackel_01.npy', 'creature_condor_01.npy', 'creature_lion_01.npy', 'creature_moth_01.npy', 'creature_virus_01.npy', 'creature_monkey_01.npy', 'creature_mysteryweapon_01.npy']},
        {'name': 'kill_sprites', 'type': 'group', 'files': ['kill_snake_00.npy', 'kill_scorpion_00.npy', 'kill_bat_00.npy', 'kill_turtle_00.npy', 'kill_jackel_00.npy', 'kill_condor_00.npy', 'kill_lion_00.npy', 'kill_moth_00.npy', 'kill_virus_00.npy', 'kill_monkey_00.npy', 'kill_mysteryweapon_00.npy']},

        # Treasures
        {'name': 'treasure', 'type': 'group', 'files': ['map1_treasure_key.npy', 'map1_treasure_crown_01.npy', 'map1_treasure_ring.npy', 'map1_treasure_ruby.npy', 'map1_treasure_chalice.npy', 'map1_treasure_crown_02.npy', 'map2_treasure_key.npy', 'map2_treasure_ring.npy', 'map2_treasure_crown.npy', 'map2_treasure_emerald.npy', 'map2_treasure_goblet.npy', 'map2_treasure_bust.npy', 'map3_treasure_key.npy', 'map3_treasure_trident.npy', 'map3_treasure_ring.npy', 'map3_treasure_herb.npy', 'map3_treasure_diamond.npy', 'map3_treasure_candelabra.npy', 'map4_treasure_key.npy', 'map4_treasure_ring.npy', 'map4_treasure_amulet.npy', 'map4_treasure_fan.npy', 'map4_treasure_crystal.npy', 'map4_treasure_zircon.npy', 'map4_treasure_dagger.npy']},

        # UI
        {'name': 'ammo_timer', 'type': 'single', 'file': 'ui_ammo_timer.npy'},
        {'name': 'ammo_map', 'type': 'group', 'files': ['ui_map1_ammo_timer.npy', 'ui_map2_ammo_timer.npy', 'ui_map3_ammo_timer.npy', 'ui_map4_ammo_timer.npy']},
        {'name': 'stats', 'type': 'group', 'files': ['ui_stats_1.npy', 'ui_stats_2.npy', 'ui_stats_3.npy']},
        {'name': 'digits', 'type': 'group', 'files': ['digit_0.npy', 'digit_1.npy', 'digit_2.npy', 'digit_3.npy', 'digit_4.npy', 'digit_5.npy', 'digit_6.npy', 'digit_7.npy', 'digit_8.npy', 'digit_9.npy']},
        {'name': 'goal', 'type': 'group', 'files': ['map1_exitdoor.npy', 'map2_exitdoor.npy', 'map3_exitdoor.npy', 'map4_exitdoor.npy']},
        {'name': 'ui_footer_header', 'type': 'group', 'files': ['ui_map1_footer_header.npy', 'ui_map2_footer_header.npy', 'ui_map3_footer_header.npy', 'ui_map4_footer_header.npy']},
        {'name': 'background', 'type': 'background', 'file': 'background_full.npy'}
    )
    return config


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
class TutankhamConstants(struct.PyTreeNode):
    # Game Window
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    # Player constants
    PLAYER_SPEED: float = struct.field(pytree_node=False, default=1)
    PLAYER_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([5, 8], dtype=jnp.int32))
    PLAYER_LIVES: int = struct.field(pytree_node=False, default=3)

    # Missile constants
    BULLET_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([1, 2], dtype=jnp.int32))
    BULLET_SPEED: float = struct.field(pytree_node=False, default=3.5)
    BULLET_ANIM_SPEED: int = struct.field(pytree_node=False, default=4)
    #AMMO_SUPPLY: int = 7500  # frames until ammo runs out

    MAX_LASER_FLASHES: int = struct.field(pytree_node=False, default=3)
    LASER_FLASH_COOLDOWN: int = struct.field(pytree_node=False, default=60)  # frames

    # Creature constants -------------------------------------

    # Creature Types
    SNAKE: int = struct.field(pytree_node=False, default=0)
    SCORPION: int = struct.field(pytree_node=False, default=1)
    BAT: int = struct.field(pytree_node=False, default=2)
    TURTLE: int = struct.field(pytree_node=False, default=3)
    JACKEL: int = struct.field(pytree_node=False, default=4)
    CONDOR: int = struct.field(pytree_node=False, default=5)
    LION: int = struct.field(pytree_node=False, default=6)
    MOTH: int = struct.field(pytree_node=False, default=7)
    VIRUS: int = struct.field(pytree_node=False, default=8)
    MONKEY: int = struct.field(pytree_node=False, default=9)
    MYSTERY_WEAPON: int = struct.field(pytree_node=False, default=10)

    CREATURE_SIZES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [8, 8],   # SNAKE (00: [6, 8], 01: [8, 7])
        [8, 8],   # SCORPION (00 & 01: [8, 8])
        [8, 8],   # BAT (00: [8, 7], 01: [8, 8])
        [8, 8],   # TURTLE (00 & 01: [8, 8])
        [8, 8],   # JACKEL (00 & 01: [8, 8])
        [8, 7],   # CONDOR (00 & 01: [8, 7])
        [8, 8],   # LION (00: [8, 7], 01: [7, 8])
        [8, 8],   # MOTH (00 & 01 padded to: [8, 8])
        [8, 6],   # VIRUS (00 & 01: [8, 6])
        [8, 8],   # MONKEY (00 & 01: [8, 8])
        [8, 8]    # MYSTERY_WEAPON (00 & 01: [8, 8])
    ], dtype=jnp.int32))

    CREATURE_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([10, 10], dtype=jnp.int32))

    INACTIVE: int = struct.field(pytree_node=False, default=0)
    ACTIVE: int = struct.field(pytree_node=False, default=1)

    CREATURE_SPEED: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(
        [0.65, 0.75, 0.95, 0.5, 0.75, 0.95, 0.85, 0.95, 0.75, 0.85, 0.95],
        dtype=jnp.float32))  # speed for each creature type
    CREATURE_POINTS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(
        [1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3],
        dtype=jnp.int32))  # points for defeating each creature type

    MAX_CREATURES: int = struct.field(pytree_node=False, default=2)  # max number of creatures on screen at once
    CREATURE_DETECTION_RANGE_X: int = struct.field(pytree_node=False, default=35)  # horizontal detection range
    CREATURE_DETECTION_RANGE_Y: int = struct.field(pytree_node=False, default=25)  # vertical detection range

    # Item constants ----------------------------------------------------

    # Item Types
    KEY_MAP1: int = struct.field(pytree_node=False, default=0)
    CROWN_01_MAP1: int = struct.field(pytree_node=False, default=1)
    RING_MAP1: int = struct.field(pytree_node=False, default=2)
    RUBY_MAP1: int = struct.field(pytree_node=False, default=3)
    CHALICE_MAP1: int = struct.field(pytree_node=False, default=4)
    CROWN_02_MAP1: int = struct.field(pytree_node=False, default=5)

    KEY_MAP2: int = struct.field(pytree_node=False, default=6)
    RING_MAP2: int = struct.field(pytree_node=False, default=7)
    CROWN_MAP2: int = struct.field(pytree_node=False, default=8)
    EMERALD_MAP2: int = struct.field(pytree_node=False, default=9)
    GOBLET_MAP2: int = struct.field(pytree_node=False, default=10)
    BUST_MAP2: int = struct.field(pytree_node=False, default=11)

    KEY_MAP3: int = struct.field(pytree_node=False, default=12)
    TRIDENT_MAP3: int = struct.field(pytree_node=False, default=13)
    RING_MAP3: int = struct.field(pytree_node=False, default=14)
    HERB_MAP3: int = struct.field(pytree_node=False, default=15)
    DIAMOND_MAP3: int = struct.field(pytree_node=False, default=16)
    CANDELABRA_MAP3: int = struct.field(pytree_node=False, default=17)

    KEY_MAP4: int = struct.field(pytree_node=False, default=18)
    RING_MAP4: int = struct.field(pytree_node=False, default=19)
    AMULET_MAP4: int = struct.field(pytree_node=False, default=20)
    FAN_MAP4: int = struct.field(pytree_node=False, default=21)
    CRYSTAL_MAP4: int = struct.field(pytree_node=False, default=22)
    ZIRCON_MAP4: int = struct.field(pytree_node=False, default=23)
    DAGGER_MAP4: int = struct.field(pytree_node=False, default=24)

    ITEM_SIZES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [8, 8],   # KEY_MAP1
        [7, 7],   # CROWN_01_MAP1
        [6, 8],   # RING_MAP1
        [8, 7],   # RUBY_MAP1
        [7, 7],   # CHALICE_MAP1
        [7, 8],   # CROWN_02_MAP1
        [8, 8],   # KEY_MAP2
        [6, 8],   # RING_MAP2
        [7, 5],   # CROWN_MAP2
        [8, 8],   # EMERALD_MAP2
        [6, 7],   # GOBLET_MAP2
        [7, 8],   # BUST_MAP2
        [8, 8],   # KEY_MAP3
        [5, 8],   # TRIDENT_MAP3
        [6, 8],   # RING_MAP3
        [5, 8],   # HERB_MAP3
        [7, 7],   # DIAMOND_MAP3
        [7, 7],   # CANDELABRA_MAP3
        [8, 8],   # KEY_MAP4
        [6, 8],   # RING_MAP4
        [5, 8],   # AMULET_MAP4
        [8, 8],   # FAN_MAP4
        [8, 8],   # CRYSTAL_MAP4
        [8, 7],   # ZIRCON_MAP4
        [8, 6]    # DAGGER_MAP4
    ], dtype=jnp.int32))

    ITEM_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([5, 5], dtype=jnp.int32))

    ITEM_POINTS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(
        [20, 15, 25, 25, 20, 45, 40, 30, 25, 40, 20, 20, 60, 35, 30, 25, 30, 5, 55, 40, 25, 80, 20, 40, 35],
        dtype=jnp.int32))  # points for collecting each item type

    # Asset config baked into constants
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)

    VALID_POS_MAPS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        create_binary_matrix(1, f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tutankham/floor_map1.npy"),
        create_binary_matrix(2, f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tutankham/floor_map2.npy"),
        create_binary_matrix(3, f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tutankham/floor_map3.npy"),
        create_binary_matrix(4, f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tutankham/floor_map4.npy"),
    ]))

    # Levels -----------------------------------------
    MAP_ITEMS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        # Level 1 (MAP 1)
        [
            [51, 87, 0, 1],   # KEY_MAP1=0       [x, y, item_type, active]
            [99, 183, 5, 1],  # CROWN_02_MAP1=5
            [68, 262, 2, 1],  # RING_MAP1=2
            [7, 311, 3, 1],   # RUBY_MAP1=3
            [93, 382, 4, 1],  # CHALICE_MAP1=4
            [18, 494, 1, 1],  # CROWN_01_MAP1=1
            [0, 0, 0, 0]      # Padding for levels with fewer items -> this item is always inactive
        ],
        # Level 2 (MAP 2)
        [
            [21, 272, 6, 1],  # KEY_MAP2=6       [x, y, item_type, active]
            [44, 155, 8, 1],  # CROWN_MAP2=8
            [128, 98, 7, 1],  # RING_MAP2=7
            [37, 406, 9, 1],  # EMERALD_MAP2=9
            [91, 482, 10, 1], # GOBLET_MAP2=10
            [23, 547, 11, 1], # BUST_MAP2=11
            [0, 0, 0, 0]      # Padding for levels with fewer items -> this item is always inactive
        ],
        # Level 3 (MAP 3)
        [
            [22, 411, 12, 1], # KEY_MAP3=12      [x, y, item_type, active]
            [15, 173, 14, 1], # RING_MAP3=14
            [128, 98, 13, 1], # TRIDENT_MAP3=13
            [17, 278, 15, 1], # HERB_MAP3=15
            [108, 323, 16, 1],# DIAMOND_MAP3=16
            [27, 656, 17, 1], # CANDELABRA_MAP3=17
            [0, 0, 0, 0]      # Padding for levels with fewer items -> this item is always inactive
        ],
        # Level 4 (MAP 4)
        [
            [144, 110, 18, 1], # KEY_MAP4=18     [x, y, item_type, active]
            [125, 221, 19, 1], # RING_MAP4=19
            [117, 269, 20, 1], # AMULET_MAP4=20
            [19, 326, 21, 1],  # FAN_MAP4=21
            [55, 510, 23, 1],  # ZIRCON_MAP4=23
            [110, 401, 22, 1], # CRYSTAL_MAP4=22
            [66, 607, 24, 1]   # DAGGER_MAP4=24  MAP 4 has 7 items (no padding)
        ]
    ], dtype=jnp.int32))  # Repeat for 16 levels (4 maps x 4 difficulty levels)

    # Number of valid item types per level (non-padded entries), shape (16,)
    MAP_N_ITEMS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array(
        [6, 6, 6, 7], dtype=jnp.int32
    ))

    # creature types per level, shape (16, 3)
    MAP_CREATURES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [0, 1, 2],  # MAP 1: SNAKE=0, SCORPION=1, BAT=2
        [3, 4, 5],  # MAP 2: TURTLE=3, JACKEL=4, CONDOR=5
        [0, 6, 7],  # MAP 3: SNAKE=0, LION=6, MOTH=7
        [8, 9, 10]  # MAP 4: VIRUS=8, MONKEY=9, MYSTERY_WEAPON=10
    ], dtype=jnp.int32))

    # Level checkpoints
    MAP_CHECKPOINTS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        # MAP 1
        [
            [0  , 198, 134, 61], # [checkpoint zone top y, checkpoint zone bottom y, checkpoint_x, checkpoint_y]
            [199, 402, 78, 199],
            [403, 567, 12, 403],
            [568, 800, 80, 568]
        ],
        # MAP 2
        [
            [0  , 258, 136, 60],
            [259, 425, 78, 259],
            [426, 571, 78, 426],
            [572, 800, 24, 572]
        ],
        # MAP 3
        [
            [0, 244, 16, 93],
            [245, 395, 39, 248],
            [396, 549, 78, 396],
            [550, 800, 98, 550]
        ],
        # MAP 4
        [
            [0, 202, 82, 95],
            [203, 390, 30, 203],
            [391, 530, 18, 391],
            [531, 800, 119, 531]
        ]
    ], dtype=jnp.int32))

    # Positions of creature spawners on the map, shape (N_SPAWNERS, 2)
    MAP_SPAWNER_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        # MAP 1
        [
            [77  ,107],
            [28, 235],
            [107, 235],
            [39, 345],
            [119, 345],
            [77, 479],
            [77, 643],
            [0, 0], # Padding for maps with fewer spawners
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ],
        # MAP 2
        [
            [10, 119],
            [143, 119],
            [27, 195],
            [107, 195],
            [17, 294],
            [136, 294],
            [80, 454],
            [78, 535],
            [32, 603],
            [112, 603],
            [91, 675],
            [64, 675]
        ],
        # MAP 3
        [
            [19, 141],
            [100, 140],
            [19, 247],
            [100, 247],
            [19, 363],
            [136, 363],
            [29, 433],
            [103, 671],
            [51, 671],
            [0, 0], # Padding for maps with fewer spawners
            [0, 0],
            [0, 0]
        ],
        # MAP 4
        [
            [31, 127],
            [124, 127],
            [125, 203],
            [31, 303],
            [112, 303],
            [133, 347],
            [21, 346],
            [78, 479],
            [61, 593],
            [141, 594],
            [0, 0], # Padding for maps with fewer spawners
            [0, 0]
        ]
    ], dtype=jnp.int32))

    MAP_TELEPORTER_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        # MAP 1
        [
            [128, 152, Action.LEFT, 27, 152], #[x_in, y_in, trigger_on (left or right action input), x_out, y_out]
            [27, 152, Action.RIGHT, 128, 152],
            [144, 604, Action.LEFT, 11, 604], #[x_in, y_in, trigger_on (left or right action input), x_out, y_out]
            [11, 604, Action.RIGHT, 144, 604]
        ],
        # MAP 2
        [
            [59, 340, Action.RIGHT, 96, 340], #[x_in, y_in, trigger_on (left or right action input), x_out, y_out]
            [96, 340, Action.LEFT, 59, 340],
            [0, 0, 0, 0, 0], # Padding for maps with fewer teleporters
            [0, 0, 0, 0, 0]
        ],
        # MAP 3
        [
            [136, 292, Action.LEFT, 19, 292], #[x_in, y_in, trigger_on (left or right action input), x_out, y_out]
            [19, 292, Action.RIGHT, 136, 292],
            [0, 0, 0, 0, 0], # Padding for maps with fewer teleporters
            [0, 0, 0, 0, 0]
        ],
        # MAP 4
        [
            [55, 148, Action.RIGHT, 100, 148], #[x_in, y_in, trigger_on (left or right action input), x_out, y_out]
            [100, 148, Action.LEFT, 55, 148],
            [132, 372, Action.LEFT, 23, 372], #[x_in, y_in, trigger_on (left or right action input), x_out, y_out]
            [23, 372, Action.RIGHT, 132, 372]
        ]
    ], dtype=jnp.int32))

    # define goal zones for each map
    GOAL_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([5, 5], dtype=jnp.int32))  #TODO: adjust based on actual goal sprite size
    MAP_GOAL_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [[18, 684]],  # MAP 1
        [[19, 634]],  # MAP 2
        [[107, 715]], # MAP 3
        [[77, 719]]   # MAP 4
    ], dtype=jnp.int32))

    # Spawn-rate per level, shape (16,)
    LEVEL_SPAWN_RATES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        0.00005, 0.00005, 0.00005, 0.00005,
        0.00006, 0.00006, 0.00006, 0.00006,
        0.00007, 0.00007, 0.00007, 0.00007,
        0.00008, 0.00008, 0.00008, 0.00008,
    ], dtype=jnp.float32))

    # Speed multiplier per level, shape (16,)
    LEVEL_CREATURE_SPEED_MULTIPLIERS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        1.0, 1.0, 1.0, 1.0,
        1.04, 1.04, 1.04, 1.04,
        1.08, 1.08, 1.08, 1.08,
        1.1, 1.1, 1.1, 1.1,
    ], dtype=jnp.float32))

    LEVEL_AMMO_SUPPLY: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([  #TODO CHANE AMMO SUPPLY
        7500, 7350, 7100, 7000,
        6300, 6150, 6000, 5850,
        5150, 5000, 4850, 4700,
        4000, 4000, 4000, 4000
    ], dtype=jnp.int32))


class TutankhamState(struct.PyTreeNode):
    # key state for creature spawn randomness
    rng_key: int

    level: int  # current level (up to 16 Levels)

    camera_offset: int  # vertical offset for camera scrolling

    player_x: chex.Array
    player_y: chex.Array
    player_lives: int
    tutankham_score: int  # current score


    bullet_state: chex.Array  # (, 4) array with (x, y, bullet_rotation, bullet_active)
    laser_flash_count: int  # number of laser flashes that can be fired
    laser_flash_cooldown: int  # cooldown timer for next laser flash
    amonition_timer: int  # if timer runs out, player can not fire again

    creature_states: chex.Array  # (2, 6) array with (x, y, creature_type, active, direction, death_timer) for each creature
    last_creature_spawn: int  # time since last creature spawn

    item_states: chex.Array # (N, 4) array with (x, y, item_type, active) for each item

    player_direction: int   # Last horizontal direction: 3=RIGHT, 4=LEFT
    is_moving: bool         # True if the player moved this frame
    step_counter: int       # Increments every frame, drives animation clock
    player_subpixel: float  # sub pixel position accumulator for smooth movement
    creature_subpixels: chex.Array  # (MAX_CREATURES,) sub pixel position accumulators
    bullet_subpixel: float  # sub pixel position accumulator for bullet

    has_key: bool  # whether the player has collected the key or not
    goal_reached: bool  # whether the player has reached the goal with the key to complete the level

    last_directional_action: int  # last action that had a directional component



class TutankhamObservation(struct.PyTreeNode):
    # Dummy observation — pixel training uses render() via PixelObsWrapper instead.
    dummy: jnp.ndarray



class TutankhamInfo(struct.PyTreeNode):
    step: jnp.ndarray
    score: jnp.ndarray




@jax.jit
def is_onscreen(y: jax.Array, height: jax.Array, camera_offset: jax.Array) -> jnp.ndarray:
    """
    Returns True if a sprite's top edge is above the bottom footer (y=175) and its bottom edge is below the top header (y=35).
    """
    sprite_top_edge = y - camera_offset
    sprite_bottom_edge = sprite_top_edge + height    
    return jnp.logical_and(sprite_bottom_edge > 35, sprite_top_edge < 175)


@jax.jit
def can_walk_to(entity_size: jax.Array, new_x: jax.Array, new_y: jax.Array, old_x: jax.Array, old_y: jax.Array, valid_pos_mat: jax.Array) -> jnp.ndarray:
    entity_width = entity_size[0]
    entity_height = entity_size[1]
    
    mid_width = entity_width // 2
    mid_height = entity_height // 2
    end_width = entity_width - 1
    end_height = entity_height - 1

    # Check 9 anchor points of the hitbox (Corners, Edge midpoints, Center)
    p1 = valid_pos_mat[new_y, new_x]                 # Top-Left
    p2 = valid_pos_mat[new_y, new_x + mid_width]     # Top-Mid
    p3 = valid_pos_mat[new_y, new_x + end_width]     # Top-Right
    p4 = valid_pos_mat[new_y + mid_height, new_x]    # Mid-Left
    p5 = valid_pos_mat[new_y + mid_height, new_x + end_width] # Mid-Right
    p6 = valid_pos_mat[new_y + end_height, new_x]    # Bottom-Left
    p7 = valid_pos_mat[new_y + end_height, new_x + mid_width] # Bottom-Mid
    p8 = valid_pos_mat[new_y + end_height, new_x + end_width] # Bottom-Right

    is_walkable = p1 & p2 & p3 & p4 & p5 & p6 & p7 & p8
    
    player_x = jnp.where(is_walkable, new_x, old_x)
    player_y = jnp.where(is_walkable, new_y, old_y)
    player_x = jnp.clip(player_x, 0, 160 - 1)
    player_y = jnp.clip(player_y, 0, valid_pos_mat.shape[0] - 1)
    return player_x, player_y, is_walkable
   

# ---------------------------------------------------------------------
# Renderer (No JAX)
# ---------------------------------------------------------------------
class TutankhamRenderer(JAXGameRenderer):
    def __init__(self, consts: TutankhamConstants = None):
        super().__init__()
        self.consts = consts or TutankhamConstants()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tutankham"

        # 1. Configure the rendering utility
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 3. Make one call to load and process all assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

    # ---------------------------------------------------------
    # Main render() method
    # ---------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TutankhamState):
        ZERO_FLIP = jnp.array([0, 0], dtype=jnp.int32)
        indices_to_update = 0
        new_color_ids = 0
        level_index = (state.level%4)

        # Calculate camera offset to keep player roughly centered vertically
        camera_offset = jnp.where(state.player_y < self.consts.HEIGHT // 2, 0, state.player_y - self.consts.HEIGHT // 2)
        
        # Ensure the camera doesn't scroll past the bottom of the map
        max_offset = self.consts.VALID_POS_MAPS[level_index].shape[0] - self.consts.HEIGHT
        camera_offset = jnp.clip(camera_offset, 0, max_offset)

        # 1. Start with the static blue background
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self.jr.render_at_clipped(
            raster,
            0,  # x
            -camera_offset,  # y
            self.SHAPE_MASKS["floor"][level_index],
            flip_offset=ZERO_FLIP
        )
        # Render Flash Floors
        raster = jax.lax.cond(
                state.laser_flash_cooldown > 0,
                lambda _: self.jr.render_at_clipped(
                    raster,
                    0,
                    - camera_offset,
                    self.SHAPE_MASKS["flash_floor"][level_index],
                    flip_offset=ZERO_FLIP
                ),
                lambda raster: raster,
                raster
            )

        # 2. Render Player
        ANIM_SPEED = 8
        frame_idx = (state.step_counter // ANIM_SPEED) % 2
        
        # Calculate index offset if player has a key
        # Idle: [0] = no key, [1] = with key
        # Move: [0,1] = no key, [2,3] = with key
        key_offset_idle = jnp.where(state.has_key, 1, 0)
        key_offset_move = jnp.where(state.has_key, 2, 0)

        player_mask = jax.lax.cond(
            state.is_moving,
            lambda _: self.SHAPE_MASKS['player_move'][frame_idx + key_offset_move],
            lambda _: self.SHAPE_MASKS['player'][key_offset_idle],
            operand=None
        )
        
        player_flip_offset = jax.lax.cond(
            state.is_moving,
            lambda _: self.FLIP_OFFSETS['player_move'],
            lambda _: self.FLIP_OFFSETS['player'],
            operand=None
        )
        
        flip = jnp.where(state.player_direction == 3, True, False)
        raster = self.jr.render_at_clipped(
            raster,
            state.player_x,
            state.player_y - camera_offset,
            player_mask,
            flip_offset=player_flip_offset,
            flip_horizontal=flip,
        )

        # creatures
        creature_one_state = state.creature_states[0]
        creature_one = creature_one_state[2]
        dir_one = creature_one_state[4]
        death_timer_one = creature_one_state[5]
        
        # Render normal creature
        raster = jax.lax.cond(
            death_timer_one == -1,
            lambda r: self.jr.render_at_clipped(
                r,
                creature_one_state[0],
                creature_one_state[1] - camera_offset,
                jax.lax.cond(
                    frame_idx == 0,
                    lambda _: self.SHAPE_MASKS['creature_00'][creature_one],
                    lambda _: self.SHAPE_MASKS['creature_01'][creature_one],
                    operand=None
                ),
                flip_offset=ZERO_FLIP,
                flip_horizontal=(dir_one == -1)
            ),
            lambda r: r,
            raster
        )
        
        # Render kill sprite
        raster = jax.lax.cond(
            death_timer_one > 0,
            lambda r: self.jr.render_at_clipped(
                r,
                creature_one_state[0],
                creature_one_state[1] - camera_offset,
                self.SHAPE_MASKS['kill_sprites'][creature_one],
                flip_offset=ZERO_FLIP,
                flip_horizontal=False
            ),
            lambda r: r,
            raster
        )
        
        creature_two_state = state.creature_states[1]
        creature_two = creature_two_state[2]
        dir_two = creature_two_state[4]
        death_timer_two = creature_two_state[5]
        
        # Render normal creature
        raster = jax.lax.cond(
            death_timer_two == -1,
            lambda r: self.jr.render_at_clipped(
                r,
                creature_two_state[0],
                creature_two_state[1] - camera_offset,
                jax.lax.cond(
                    frame_idx == 0,
                    lambda _: self.SHAPE_MASKS['creature_00'][creature_two],
                    lambda _: self.SHAPE_MASKS['creature_01'][creature_two],
                    operand=None
                ),
                flip_offset=ZERO_FLIP,
                flip_horizontal=(dir_two == -1)
            ),
            lambda r: r,
            raster
        )
        
        # Render kill sprite
        raster = jax.lax.cond(
            death_timer_two > 0,
            lambda r: self.jr.render_at_clipped(
                r,
                creature_two_state[0],
                creature_two_state[1] - camera_offset,
                self.SHAPE_MASKS['kill_sprites'][creature_two],
                flip_offset=ZERO_FLIP,
                flip_horizontal=False
            ),
            lambda r: r,
            raster
        )
    
        # 2.5 Animations
        # 5. Render Treasures
        def render_all_treasures(i: int, raster: jnp.ndarray):
            treasure_x = state.item_states[i][0]
            treasure_y = state.item_states[i][1]
            treasure_type = state.item_states[i][2]
            is_active = state.item_states[i][3] == 1
            treasure_mask = self.SHAPE_MASKS["treasure"][treasure_type]
            return jax.lax.cond(
                is_active & is_onscreen(treasure_y, 8, camera_offset),
                lambda r: self.jr.render_at_clipped(
                    r,
                    treasure_x,
                    treasure_y - camera_offset,
                    treasure_mask,
                    flip_offset=ZERO_FLIP
                ),
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, 7, render_all_treasures, raster)

        # Render Goal                     
        raster = jax.lax.cond(is_onscreen(self.consts.MAP_GOAL_POSITIONS[state.level%4, 0, 1], 8, camera_offset),
                              lambda r: self.jr.render_at_clipped(
                                  raster,
                                  self.consts.MAP_GOAL_POSITIONS[state.level%4, 0, 0],
                                  self.consts.MAP_GOAL_POSITIONS[state.level%4, 0, 1] - camera_offset,
                                  self.SHAPE_MASKS["goal"][level_index],
                                  flip_offset=ZERO_FLIP
                                  # self.FLIP_OFFSETS['player_group'],
                              ),
                              lambda r: r,
                              raster)
        # 6. Render Bullets
        bullet_frame_idx = jnp.clip(state.bullet_state[4] // self.consts.BULLET_ANIM_SPEED, 0, 3)
        raster = jax.lax.cond(state.bullet_state[3] == 1,
                              lambda r: self.jr.render_at_clipped(
                                  r,
                                  state.bullet_state[0],
                                  state.bullet_state[1] - camera_offset,
                                  self.SHAPE_MASKS["bullet"][bullet_frame_idx.astype(jnp.int32)],
                                  flip_offset=ZERO_FLIP
                                  # self.FLIP_OFFSETS['player_group'],
                              ),
                              lambda r: r,
                              raster)


        # 8. Render UI Footer and Header
        raster = self.jr.render_at_clipped(
            raster,
            0,  # x
            0,  # y
            self.SHAPE_MASKS["ui_footer_header"][level_index],
            flip_offset=ZERO_FLIP
        )
        # Render stats (lives)
        raster = self.jr.render_at_clipped(
            raster,
            12,
            24,
            self.SHAPE_MASKS["stats"][state.player_lives-1],
            flip_offset=ZERO_FLIP
        )
        # Render stats (flashes)
        raster = jax.lax.cond(
            state.laser_flash_count > 0,
            lambda r: self.jr.render_at_clipped(
                r,
                108,
                24,
                self.SHAPE_MASKS["stats"][state.laser_flash_count-1],
                flip_offset=ZERO_FLIP
            ),
            lambda r: r,
            raster
        )
        raster = self.jr.render_at_clipped(
            raster,
            114,
            197,
            self.SHAPE_MASKS["ammo_timer"],
            flip_offset=ZERO_FLIP
        )
        # Calculate ammo timer bar position
        # Scales with AMMO_SUPPLY. Range is 30 pixels (from 84 to 114).
        ammo_ratio = jnp.maximum(0, state.amonition_timer) / self.consts.LEVEL_AMMO_SUPPLY[state.level]
        ammo_offset = jnp.ceil(ammo_ratio * 30)
        ammo_x = 114 - ammo_offset.astype(jnp.int32)
        raster = self.jr.render_at_clipped(
            raster,
            ammo_x,
            197,
            self.SHAPE_MASKS["ammo_map"][level_index],
            flip_offset=ZERO_FLIP
        )
        
        # Render Score
        def render_score_digit(i: int, raster: jnp.ndarray):
            # calculate value of 10^(5-i)
            # score is maximum 999999 so 6 digits
            # extract digit i (where i=0 is most significant digit, i=5 is least)
            divisor = 10 ** (5 - i)
            digit_val = (state.tutankham_score // divisor) % 10
            digit_mask = self.SHAPE_MASKS["digits"][digit_val]
            # score is located at bottom left, approximately at x=24, y=190
            # each digit is 6 wide, with 2 pixel spacing
            digit_x = 24 + (i * 8)
            digit_y = 190
            
            # Only draw if it's the last digit (index 5) or if the score is large enough
            should_draw = (i == 5) | (state.tutankham_score >= divisor)
            
            return jax.lax.cond(
                should_draw,
                lambda r: self.jr.render_at_clipped(
                    raster,
                    digit_x,
                    digit_y,
                    digit_mask,
                    flip_offset=ZERO_FLIP
                ),
                lambda r: raster,
                operand=raster
            )
            
        raster = jax.lax.fori_loop(0, 6, render_score_digit, raster)
        # 9. Final Palette Lookup
        return self.jr.render_from_palette(
            raster,
            self.PALETTE,
            indices_to_update=indices_to_update,
            new_color_ids=new_color_ids
        )



# Environment
class JaxTutankham(JaxEnvironment[TutankhamState, TutankhamObservation, TutankhamInfo, TutankhamConstants]):
    def __init__(self):
        consts = TutankhamConstants()
        super().__init__(consts)
        self.renderer = TutankhamRenderer()
        self.consts = consts

        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE
        ]

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, key=None):
        # Generate random seed based on current time for creature spawn randomness
        seed = int(time.time())
        key = jax.random.PRNGKey(seed)
        level = 0
        start_x = 134
        start_y = 61
        tutankham_score = 0
        goal_reached = False
        player_lives = self.consts.PLAYER_LIVES
        amonition_timer = self.consts.LEVEL_AMMO_SUPPLY[level]
        bullet_state = jnp.array([0, 0, 0, 0, 0], dtype=jnp.int32)  # (x, y, bullet_rotation, bullet_active, anim_counter)
        creature_states = jnp.zeros((self.consts.MAX_CREATURES, 6), dtype=jnp.int32)  # (x, y, creature_type, active, direction, death_timer)
        creature_states = creature_states.at[:, 5].set(-1)
        item_states = self.consts.MAP_ITEMS[level%4]  # (N, 4) array with (x, y, item_type, active)
        last_creature_spawn = 0
        laser_flash_count = self.consts.MAX_LASER_FLASHES
        laser_flash_cooldown = 0
        has_key = False
        player_direction = 3  # Start facing RIGHT
        is_moving = False
        last_directional_action = 0
        step_counter = 0

        # Sub Pixel accumulators for smooth movement
        player_subpixel=0.0
        creature_subpixels=jnp.zeros(self.consts.MAX_CREATURES, dtype=jnp.float32)
        bullet_subpixel=0.0


        #TODO: only for testing
        # level = 12
        # start_x = 136
        # start_y = 60
        # item_states = self.consts.MAP_ITEMS[level%4]
        # amonition_timer = self.consts.LEVEL_AMMO_SUPPLY[level]
        #---------------------

        camera_offset = jnp.where(start_x < self.consts.HEIGHT // 2, 0, start_y - self.consts.HEIGHT // 2)


        state = TutankhamState(level=level,
                               player_x=start_x,
                               player_y=start_y,
                               tutankham_score=tutankham_score,
                               player_lives=player_lives,
                               bullet_state=bullet_state,
                               amonition_timer=amonition_timer,
                               creature_states=creature_states,
                               item_states=item_states,
                               last_creature_spawn=last_creature_spawn,
                               laser_flash_count=laser_flash_count,
                               laser_flash_cooldown=laser_flash_cooldown,
                               player_direction=player_direction,
                               is_moving=is_moving,
                               step_counter=step_counter,
                               player_subpixel=player_subpixel,
                               creature_subpixels=creature_subpixels,
                               bullet_subpixel=bullet_subpixel,
                               has_key=has_key,
                               last_directional_action=last_directional_action,
                               rng_key=key,
                               camera_offset=camera_offset,
                               goal_reached=goal_reached
                               )
        
        obs = self._get_observation(state)
        return obs, state

    
    @partial(jax.jit, static_argnums=(0,))
    def teleporter_check(self, player_x, player_y, action, level):
        # Check if player is on a teleporter and has the correct action input to trigger it
        teleporters = self.consts.MAP_TELEPORTER_POSITIONS[level%4]  # (N, 5) N teleporters for current map with (x_in, y_in, trigger_action, x_out, y_out)
        teleporter_height = 4 # Define a vertical hitbox for the teleporter

        teleport_trigger_action = (teleporters[:, 2] == action) # check if trigger action matches the current action input
        player_on_teleporter_x = (teleporters[:, 0] == player_x)
        player_on_teleporter_y_range = (player_y >= teleporters[:, 1]) & (player_y < teleporters[:, 1] + teleporter_height)


        teleporter_active_mask = player_on_teleporter_x & player_on_teleporter_y_range & teleport_trigger_action
        should_teleport = jnp.any(teleporter_active_mask)

        teleporter_out_x = jnp.sum(teleporters[:, 3] * teleporter_active_mask)
        teleporter_out_y = jnp.sum(teleporters[:, 4] * teleporter_active_mask)

        return teleporter_out_x, teleporter_out_y, should_teleport
    

    # sub pixel accumulator for smooth movement
    @partial(jax.jit, static_argnums=(0,))
    def subpixel_accumulator(self, speed, subpixel):
        '''
        Adds the fractional part of the speed to the subpixel accumulator.
        If subpixel accumulator > 1, then one extra pixel is moved this step, subpixel accumulator is reduced by 1.
        Returns the integer speed and updated subpixel accumulator.
        '''
        base_speed = jnp.floor(speed).astype(jnp.int32)
        sub_speed = speed % 1.0
        new_subpixel = subpixel + sub_speed
        extra = (new_subpixel >= 1.0).astype(jnp.int32)
        new_subpixel = jnp.where(new_subpixel >= 1.0, new_subpixel - 1.0, new_subpixel)
        actual_speed = base_speed + extra

        return actual_speed, new_subpixel
    

    # Player Step
    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, player_x, player_y, action, last_directional_action, player_direction, step_counter, player_subpixel, level):
        
        speed = self.consts.PLAYER_SPEED
        actual_speed, new_subpixel = self.subpixel_accumulator(speed, player_subpixel)

        dx = jnp.array([
        0,            # 0  NOOP
        0,            # 1  FIRE
        0,            # 2  UP
        actual_speed, # 3  RIGHT
        -actual_speed,# 4  LEFT
        0,            # 5  DOWN
        0,            # 6  UPRIGHT
        0,            # 7  UPLEFT
        0,            # 8  DOWNRIGHT
        0,            # 9  DOWNLEFT
        0,            # 10 UPFIRE
        0,            # 11 RIGHTFIRE
        0,            # 12 LEFTFIRE
        0,            # 13 DOWNFIRE
        0,            # 14 UPRIGHTFIRE
        0,            # 15 UPLEFTFIRE
        0,            # 16 DOWNRIGHTFIRE
        0,            # 17 DOWNLEFTFIRE
        ])

        dy = jnp.array([
            0,            # 0  NOOP
            0,            # 1  FIRE
            -actual_speed,# 2  UP
            0,            # 3  RIGHT
            0,            # 4  LEFT
            actual_speed, # 5  DOWN
            0,            # 6  UPRIGHT
            0,            # 7  UPLEFT
            0,            # 8  DOWNRIGHT
            0,            # 9  DOWNLEFT
            0,            # 10 UPFIRE
            0,            # 11 RIGHTFIRE
            0,            # 12 LEFTFIRE
            0,            # 13 DOWNFIRE
            0,            # 14 UPRIGHTFIRE
            0,            # 15 UPLEFTFIRE
            0,            # 16 DOWNRIGHTFIRE
            0,            # 17 DOWNLEFTFIRE
        ])

        # For wall collision
        w = self.consts.PLAYER_SIZE[0]
        h = self.consts.PLAYER_SIZE[1]

        # If the current action has no directional component, fall back to the last directional action
        has_movement = (dx[action] != 0) | (dy[action] != 0)
        effective_action = jnp.where(has_movement, action, last_directional_action)
        new_last_directional_action = jnp.where(has_movement, action, last_directional_action)

        # If player hits teleporter and right action input is triggered then teleport player to teleporter out coordinates
        # Is always computed, but only effects player poisition if should_teleport is True
        teleporter_out_x, teleporter_out_y, should_teleport = self.teleporter_check(player_x, player_y, effective_action, level)

        new_x = player_x + dx[effective_action]
        new_y = player_y + dy[effective_action]
        player_x, player_y, is_walkable = can_walk_to(self.consts.PLAYER_SIZE, new_x, new_y, player_x, player_y, self.consts.VALID_POS_MAPS[level%4])        
        
        #is_walkable = True # TODO: only for testing---------------------------
        # player_x = new_x
        # player_y = new_y
        # new_last_directional_action = 0
        #--------------------------------------------------------------------

        # If teleporter is triggered, the player position is set to teleporter out coordinates 
        player_x = jnp.where(should_teleport, teleporter_out_x, player_x)
        player_y = jnp.where(should_teleport, teleporter_out_y, player_y)

        # Animation / orientation state
        is_moving_now = jnp.logical_and(is_walkable, jnp.logical_or(dx[effective_action] != 0, dy[effective_action] != 0))
        new_direction = jnp.where(dx[effective_action] > 0, 3,
                        jnp.where(dx[effective_action] < 0, 4, player_direction))
        new_step_counter = step_counter + 1


        # update camera offset based on player y position
        camera_offset = jnp.where(player_y < self.consts.HEIGHT // 2, 0, player_y - self.consts.HEIGHT // 2)

        return player_x, player_y, new_last_directional_action, new_direction, is_moving_now, new_step_counter, new_subpixel, camera_offset

    #Bullet Step
    @partial(jax.jit, static_argnums=(0,))
    def bullet_step(self, bullet_state, bullet_subpixel, player_x, player_y, amonition_timer, action, level):
        
        
        def get_rotation(action):
            return jax.lax.cond(
                action == Action.RIGHTFIRE,
                lambda _: 1, # bullet travels right
                lambda _: jax.lax.cond(
                    action == Action.LEFTFIRE,
                    lambda _: -1, # bullet travels left
                    lambda _: 0, # default if firing up/down/etc
                    operand=None
                ),
                operand=None
            )

        space = jnp.logical_or(action == Action.LEFTFIRE, action == Action.RIGHTFIRE)

        new_bullet = bullet_state #array with (x, y, bullet_rotation, bullet_active, anim_counter)

        # Update animation counter
        anim_counter = jnp.where(bullet_state[3] == 1, bullet_state[4] + 1, 0)
        new_bullet = new_bullet.at[4].set(anim_counter)


        # --- update bullet x position if active (bullet only travels horizontal so no vertical movement) ---
        actual_speed, new_subpixel = self.subpixel_accumulator(self.consts.BULLET_SPEED, bullet_subpixel)
        # Only advance subpixel accumulator while bullet is active
        new_subpixel = jnp.where(bullet_state[3] == 1, new_subpixel, bullet_subpixel)

        bullet_x = jax.lax.cond(
            bullet_state[3] == 1,  # if bullet is active
            lambda x: x + actual_speed * bullet_state[2],
            lambda x: x,
            bullet_state[0],
        )
        new_bullet = new_bullet.at[0].set(bullet_x)
        
        # Deactivate if out of bounds or hits wall
        bullet_y = bullet_state[1]
        old_bullet_x = bullet_state[0]
        
        _, _, is_walkable = can_walk_to(self.consts.BULLET_SIZE, bullet_x, bullet_y, old_bullet_x, bullet_y, self.consts.VALID_POS_MAPS[level%4])        

        should_deactivate = jnp.logical_not(is_walkable)

        new_bullet = jax.lax.cond(
            should_deactivate,
            lambda _: jnp.zeros((5,), dtype=new_bullet.dtype),
            lambda bullet: bullet,
            operand=new_bullet
        )
        # Reset subpixel accumulator if bullet is deactivated
        new_subpixel = jnp.where(should_deactivate, jnp.float32(0), new_subpixel)



        # --- firing logic ---
        bullet_rdy = 1 - bullet_state[3]  # 1 if bullet inactive, 0 if active
        fired = (space & bullet_rdy & (amonition_timer > 0)) == 1

        new_bullet = jax.lax.cond(
            fired, # if firing action & bullet is inactive & amonition available
            lambda _: jnp.array([player_x+2, player_y+3, get_rotation(action), 1, 0], dtype=jnp.int32), # shoot bullet at player face position,
            lambda bullet: bullet, # don't shoot bullet
            operand=new_bullet
        )
        # Reset subpixel accumulator when a new bullet is fired
        new_subpixel = jnp.where(fired, jnp.float32(0), new_subpixel)

        amonition_timer -= 1 # TODO: adjust amonition timer

        return new_bullet, new_subpixel, amonition_timer

    @partial(jax.jit, static_argnums=(0,))
    def laser_flash_step(self, creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn, action):

        # decrease cooldown timer in every step
        laser_flash_cooldown = jnp.maximum(laser_flash_cooldown - 1, 0)

        # check if laser flash is being fired this step
        is_firing = (action == Action.UPFIRE) & (laser_flash_count > 0) & (laser_flash_cooldown == 0)

        # if firing, set all creatures to inactive, decrease flash count by 1, reset cooldown and reset creature spawn timer
        new_laser_flash_count = jnp.where(is_firing, laser_flash_count - 1, laser_flash_count)
        new_laser_flash_cooldown = jnp.where(is_firing, self.consts.LASER_FLASH_COOLDOWN, laser_flash_cooldown)
        new_last_creature_spawn = jnp.where(is_firing, 0, last_creature_spawn)

        # active mask contains 1 for active creatures and 0 for inactive creatures, if firing set all to 0
        active_mask = creature_states[:, 3]
        # if firing, set all active_mask values to 0, otherwise keep them as they are
        new_active_mask = jnp.where(is_firing, 0, active_mask)
        new_death_timer = jnp.where(is_firing, -1, creature_states[:, 5])

        new_creature_states = creature_states.at[:, 3].set(new_active_mask)
        new_creature_states = new_creature_states.at[:, 5].set(new_death_timer)

        return new_creature_states, new_laser_flash_cooldown, new_laser_flash_count, new_last_creature_spawn


    @partial(jax.jit, static_argnums=(0,))
    def process_death_timer(self, active, death_timer, creature_x, creature_y):
        # Handle death timer logic
        # If death_timer > 0, decrement. If it reaches 0, set active=0 and death_timer=-1
        new_death_timer = jnp.where(death_timer > 0, death_timer - 1, death_timer)
        new_active = jnp.where((death_timer == 1) & (new_death_timer == 0), self.consts.INACTIVE, active)
        new_death_timer = jnp.where((death_timer == 1) & (new_death_timer == 0), -1, new_death_timer)
        
        # If creature becomes inactive due to screen bounds or death, cancel death timer and zero position
        new_death_timer = jnp.where(new_active == self.consts.INACTIVE, -1, new_death_timer)
        new_creature_x = creature_x * new_active
        new_creature_y = creature_y * new_active
        
        return new_active, new_death_timer, new_creature_x, new_creature_y

    # creature step
    @partial(jax.jit, static_argnums=(0,))
    def creature_step(self, creature_states, creature_subpixels, camera_offset, step_counter, level, player_x, player_y, rng_key):


        @jax.jit
        def creature_pathing(creature_x, creature_y, direction, creature_type, player_x, player_y, rng_key):
            """
            Determines the next direction for a creature.
            down=1, up=2, right=-1, left=-2
            """

            # Natural patrol: randomly change direction --------------------------------
            change_probability = 0.08
            possible_directions = jnp.array([-1, -2, 1, 2])

            rng_key, subkey_01, subkey_02 = jax.random.split(rng_key, 3)
            random_dir = jax.random.choice(subkey_01, possible_directions)
            should_change = jax.random.bernoulli(subkey_02, p=change_probability)

            new_direction = jnp.where(should_change, random_dir, direction)
            new_direction = jnp.where(direction == 0, random_dir, new_direction)
            #--------------------------------------------------------------------------


            # Chase player if nearby ---------------------------------------------------
            dx = player_x - creature_x
            dy = player_y - creature_y
            player_near = (jnp.abs(dx) < self.consts.CREATURE_DETECTION_RANGE_X) & (jnp.abs(dy) < self.consts.CREATURE_DETECTION_RANGE_Y)
            horizontal_direction = jnp.where(dx >= 0, jnp.int32(-1), jnp.int32(-2))
            vertical_direction = jnp.where(dy >= 0, jnp.int32(1),  jnp.int32(2))

            # Try primary direction (larger gap); fall back to other axis if wall
            prefer_h = jnp.abs(dx) >= jnp.abs(dy)
            primary_dir   = jnp.where(prefer_h, horizontal_direction, vertical_direction)
            secondary_dir = jnp.where(prefer_h, vertical_direction, horizontal_direction)
            
            # Indices: 0, 1(Down), 2(Up), -1(Right), -2(Left)
            # mapping array where the index matches the direction value
            lookup_x = jnp.array([0, 0, 0, -1, 1])
            lookup_y = jnp.array([0, 1, -1, 0, 0])
            next_x = creature_x + lookup_x[primary_dir]
            next_y = creature_y + lookup_y[primary_dir]
            _, _, primary_walkable = can_walk_to(self.consts.CREATURE_SIZE, next_x, next_y, creature_x, creature_y, self.consts.VALID_POS_MAPS[level%4])

            toward_player = jnp.where(primary_walkable, primary_dir, secondary_dir)
            new_direction = jnp.where(player_near, toward_player, new_direction)

            return new_direction, rng_key
        
        @jax.jit
        def move_creature(creature_state, creature_subpixel, key):
            creature_x, creature_y, creature_type, active, direction, death_timer = creature_state

            # get creature speed apply subpixel accumulator for smooth movement
            speed = self.consts.CREATURE_SPEED[creature_type.astype(jnp.int32)]
            actual_speed, new_subpixel = self.subpixel_accumulator(speed, creature_subpixel)

            # Only move if active and NOT dying
            is_alive = (active == self.consts.ACTIVE) & (death_timer == -1)

            # calculate the next movement direction for the creature
            direction, rng_key = creature_pathing(creature_x, creature_y, direction, creature_type, player_x, player_y, key)

            # Indices: 0, 1(Down), 2(Up), -1(Right), -2(Left)
            # mapping array where the index matches the direction value
            lookup_x = jnp.array([0, 0,  0, -1, 1])
            lookup_y = jnp.array([0, 1, -1, 0,  0])
            x_direction = lookup_x[direction]
            y_direction = lookup_y[direction]

            # x_direction = jnp.where(direction == -1, 1, jnp.where(direction == -2, -1, 0))
            # y_direction = jnp.where(direction == 1, 1, jnp.where(direction == 2, -1, 0))

            # move creature
            new_x = creature_x + actual_speed * x_direction * is_alive
            new_y = creature_y + actual_speed * y_direction * is_alive           
            creature_x, creature_y, is_walkable = can_walk_to(self.consts.CREATURE_SIZE, new_x, new_y, creature_x, creature_y, self.consts.VALID_POS_MAPS[level%4])
            

            # Deactivate creature of it is offscreen
            creature_on_screen = is_onscreen(creature_y, self.consts.CREATURE_SIZE[1], camera_offset)
            active = jnp.where(creature_on_screen, active, self.consts.INACTIVE)

            # Process death timer in separated function
            active, new_death_timer, creature_x, creature_y = self.process_death_timer(active, death_timer, creature_x, creature_y)

            return jnp.array([creature_x, creature_y, creature_type, active, direction, new_death_timer], dtype=jnp.int32), new_subpixel




        creature_keys = jax.random.split(rng_key, self.consts.MAX_CREATURES)
        move_creature_vmapped = jax.vmap(move_creature)
        new_creature_states, new_creature_subpixels = move_creature_vmapped(creature_states, creature_subpixels, creature_keys)

        return new_creature_states, new_creature_subpixels

    # creature spawner step
    @partial(jax.jit, static_argnums=(0,))
    def spawner_step(self, creature_states, last_creature_spawn, level, rng_key, camera_offset, laser_flash_cooldown):
        spawners = self.consts.MAP_SPAWNER_POSITIONS[level%4] # (n, 2) array with (x, y) positions of the n spawners for current level
        growth = self.consts.LEVEL_SPAWN_RATES[level]
        MAX_PROB = 0.8

        # Only increment timer if less than MAX_CREATURES are currently active and laser flash cooldown is 0
        active_count = jnp.sum(creature_states[:, 3] == self.consts.ACTIVE)
        new_last_creature_spawn = jnp.where((active_count < 2) & (laser_flash_cooldown == 0), last_creature_spawn + 1, last_creature_spawn)

        # deactivate spawners at at (0, 0) which are only used as padding
        valid_spawner_mask = (spawners[:, 0] != 0) | (spawners[:, 1] != 0)
        # check which spawners are on screen
        on_screen_mask = jax.vmap(is_onscreen, in_axes=(0, None, None))(spawners[:, 1], 1, camera_offset)

        # combine masks to get active spawners
        active_spawner_mask = on_screen_mask & valid_spawner_mask
        any_on_screen = jnp.any(active_spawner_mask)

        # Spawn chance grows linearly with time, capped at MAX_PROB
        spawn_chance = jnp.clip(new_last_creature_spawn * growth, 0.0, MAX_PROB)

        # Split key: one for the probability roll, one for the creature type
        rng_key, key_roll, key_type, key_spawner = jax.random.split(rng_key, 4)
        roll = jax.random.uniform(key_roll)
        should_spawn = roll < spawn_chance

        p_weights = active_spawner_mask.astype(jnp.float32)
        selected_spawner_idx = jax.random.choice(key_spawner, spawners.shape[0], p=p_weights)
        chosen_pos = spawners[selected_spawner_idx]


        # Only spawn if there is a free slot in creature_states
        inactive_mask = creature_states[:, 3] == self.consts.INACTIVE
        has_free_slot = jnp.any(inactive_mask)
        first_free_slot = jnp.argmax(inactive_mask)  # index of first inactive creature

        # Select creature type based on current map
        type_idx = jax.random.randint(key_type, shape=(), minval=0, maxval=self.consts.MAP_CREATURES.shape[1])
        new_type = self.consts.MAP_CREATURES[level%4, type_idx]

        do_spawn = should_spawn & has_free_slot & any_on_screen & (laser_flash_cooldown == 0)

        # Construct the new creature with chosen spawner coordinates
        # direction: -1 for RIGHT, -2 for LEFT
        # Spawners on the right (x > 80) should move LEFT, spawners on the left move RIGHT
        direction = jnp.where(chosen_pos[0] > 80, -2, -1)
        
        new_creature = jnp.array([
            chosen_pos[0],      # X from selected spawner
            chosen_pos[1],      # Y from selected spawner
            new_type, 
            self.consts.ACTIVE,
            direction,
            -1                  # death_timer initialized to -1
        ], dtype=jnp.int32)

        # if do spawn is true, insert new creature at first free slot, otherwise keep creature states unchanged
        new_row = jnp.where(do_spawn, new_creature, creature_states[first_free_slot])
        new_creature_states = creature_states.at[first_free_slot].set(new_row)

        # Reset timer on spawn
        final_last_creature_spawn = jnp.where(do_spawn, jnp.int32(0), new_last_creature_spawn)

        return new_creature_states, final_last_creature_spawn, rng_key
    

    @partial(jax.jit, static_argnums=(0,))
    def check_entity_collision(self, x1, y1, size1, x2, y2, size2):
        '''Check AABB collision between two entities.'''
        return (
            (x1 < x2 + size2[0]) & (x1 + size1[0] > x2) &
            (y1 < y2 + size2[1]) & (y1 + size1[1] > y2)
        )


    @partial(jax.jit, static_argnums=(0,))
    def respawn_player(self, player_x, player_y, player_lives, level, last_directional_action):
        '''
        Resets player position to last checkpoint for each checkpoint zone and decreases lives by 1
        Sets bullet state and creature states to default values
        Resets creature spawn timer to 0
        '''
        
        
        checkpoints = self.consts.MAP_CHECKPOINTS[level%4] # (5, 4) array with (y_min, y_max, respawn_x, respawn_y) for each checkpoint zone in current map
        
        # player_y is between y_min (col 0) and y_max (col 1)
        is_in_zone = (player_y >= checkpoints[:, 0]) & (player_y <= checkpoints[:, 1])

        # extract the respawn coordinates from the checkpoints
        # if multiple zones overlap, this takes the first one found
        checkpoint_idx = jnp.argmax(is_in_zone)
        
        # get respawn_x and respawn_y from the checkpoint
        respawn_x = checkpoints[checkpoint_idx, 2]
        respawn_y = checkpoints[checkpoint_idx, 3]


        creature_states = jnp.zeros((self.consts.MAX_CREATURES, 6), dtype=jnp.int32)
        creature_states = creature_states.at[:, 5].set(-1)
        bullet_state = jnp.zeros(5, dtype=jnp.int32)
        last_creature_spawn = jnp.int32(0)

        #set last_directional_action to 0, to avoid player moving immediately on respawn based on previous action
        return respawn_x, respawn_y, bullet_state, creature_states, player_lives - 1, last_creature_spawn, 0 



    @partial(jax.jit, static_argnums=(0,))
    def resolve_bullet_collisions(self, creature_states, bullet_state):
        active_mask = (creature_states[:, 3] == self.consts.ACTIVE) & (creature_states[:, 5] == -1)

        # check bullet-creature collisions (vectorized over all creatures)
        def bullet_hits_creature(creature):
            creature_type = creature[2]
            return self.check_entity_collision(
                bullet_state[0], bullet_state[1], self.consts.BULLET_SIZE,
                creature[0], creature[1], self.consts.CREATURE_SIZES[creature_type],
            )

        bullet_hits = jax.vmap(bullet_hits_creature)(creature_states)
        bullet_hits = bullet_hits & active_mask & (bullet_state[3] == 1)

        any_bullet_hit = jnp.any(bullet_hits)
        # Deactivate only the first hit creature (if multiple collisions happen in the same step, only the first one counts)
        first_bullet_hit = (jnp.cumsum(bullet_hits) == 1) & bullet_hits

        new_death_timer = jnp.where(first_bullet_hit, 15, creature_states[:, 5])
        new_creature_states = creature_states.at[:, 5].set(new_death_timer)
        new_bullet_state = jnp.where(any_bullet_hit, jnp.zeros(5, dtype=bullet_state.dtype), bullet_state)

        return new_creature_states, new_bullet_state


    @partial(jax.jit, static_argnums=(0,))
    def resolve_player_creature_collisions(self, player_x, player_y, creature_states, creature_subpixels, bullet_state, player_lives, last_creature_spawn, level, last_directional_action):

        # check player-creature collisions (vectorized over all creatures)
        def player_hits_creature(creature):
            creature_type = creature[2]
            return self.check_entity_collision(
                player_x, player_y, self.consts.PLAYER_SIZE,
                creature[0], creature[1], self.consts.CREATURE_SIZES[creature_type],
            )

        player_hits = jax.vmap(player_hits_creature)(creature_states)
        player_hits = player_hits & (creature_states[:, 3] == self.consts.ACTIVE) & (creature_states[:, 5] == -1)

        player_hit = jnp.any(player_hits) # is true if player collides with any active creature

        # Compute respawn state unconditionally, then select with jnp.where
        (respawn_x, respawn_y,
         respawn_bullet, respawn_creatures,
         respawn_lives, respawn_spawn, respawn_directional_action) = self.respawn_player(player_x, player_y, player_lives, level, last_directional_action)

        final_player_x = jnp.where(player_hit, respawn_x, player_x)
        final_player_y = jnp.where(player_hit, respawn_y, player_y)
        final_bullet_state = jnp.where(player_hit, respawn_bullet, bullet_state)
        final_creature_states = jnp.where(player_hit, respawn_creatures, creature_states)
        final_creature_subpixels = jnp.where(player_hit, jnp.zeros_like(creature_subpixels), creature_subpixels)
        final_player_lives = jnp.where(player_hit, respawn_lives, player_lives)
        final_last_creature_spawn = jnp.where(player_hit, respawn_spawn, last_creature_spawn)
        final_last_directional_action = jnp.where(player_hit, respawn_directional_action, last_directional_action)

        return (final_player_x, final_player_y,
                final_bullet_state, final_creature_states, final_creature_subpixels,
                final_player_lives, final_last_creature_spawn, final_last_directional_action)

    @partial(jax.jit, static_argnums=(0,))
    def resolve_player_item_collisions(self, player_x, player_y, item_states):

        # check player-item collisions (vectorized over all items)
        def player_collects_item(item):
            '''
                Returns True if player collides with item
            '''
            return self.check_entity_collision(
                player_x, player_y, self.consts.PLAYER_SIZE,
                item[0], item[1], self.consts.ITEM_SIZE,
            )
        
        item_collected = jax.vmap(player_collects_item)(item_states)
        item_collected = item_collected & (item_states[:, 3] == self.consts.ACTIVE)

        # deactivate collected item
        new_item_active = jnp.where(item_collected, self.consts.INACTIVE, item_states[:, 3])
        new_item_states = item_states.at[:, 3].set(new_item_active)

        return new_item_states
    

    @partial(jax.jit, static_argnums=(0,))
    def check_key(self, item_states, has_key):
        # check if player has collected the key item
        has_collected_key = item_states[0, 3] == self.consts.INACTIVE # key item is always at index 0 in item_states
        new_has_key = has_key | has_collected_key

        return new_has_key
    
    @partial(jax.jit, static_argnums=(0,))
    def check_goal(self, player_x, player_y, has_key, level):
        # check if collides with goal and has the key to complete the level
        goal_position = self.consts.MAP_GOAL_POSITIONS[level%4, 0] # (x, y) coordinates of the goal tile for current level
        on_goal = self.check_entity_collision(
            player_x, player_y, self.consts.PLAYER_SIZE,
            goal_position[0], goal_position[1], self.consts.GOAL_SIZE
        )
        complete_level = on_goal & has_key

        return complete_level
    
    @partial(jax.jit, static_argnums=(0,))
    def map_transition(self, goal_reached, level,
                       player_x, player_y, bullet_state,
                       creature_states, item_states,
                       last_creature_spawn, amonition_timer,
                       laser_flash_cooldown, has_key, laser_flash_count):
        '''
        If goal_reached is True:
        increment level,
        reset player position to start coordinates of the next level, 
        reset bullet state, 
        reset creature states, 
        initialize item states for the new level, 
        reset last creature spawn timer to zero, 
        reset amonition timer, 
        set laser flash cooldown to zero, 
        reset has_key to False
        '''

        level = jnp.where(goal_reached, level + 1, level)
        player_x = jnp.where(goal_reached, self.consts.MAP_CHECKPOINTS[level%4, 0, 2], player_x) # respawn_x of first checkpoint is the start coordinates for each map
        player_y = jnp.where(goal_reached, self.consts.MAP_CHECKPOINTS[level%4, 0, 3], player_y) # respawn_y of first checkpoint is the start coordinates for each map
        bullet_state = jnp.where(goal_reached, jnp.zeros((5,), dtype=bullet_state.dtype), bullet_state)
        creature_states = jnp.where(goal_reached, jnp.zeros_like(creature_states).at[:, 5].set(-1), creature_states)
        item_states = jnp.where(goal_reached, self.consts.MAP_ITEMS[level%4], item_states)
        last_creature_spawn = jnp.where(goal_reached, 0, last_creature_spawn)
        amonition_timer = jnp.where(goal_reached, self.consts.LEVEL_AMMO_SUPPLY[level], amonition_timer)
        laser_flash_cooldown = jnp.where(goal_reached, 0, laser_flash_cooldown)
        has_key = jnp.where(goal_reached, False, has_key)

        # On completing map 4, the player is awarded with an extra laser flash, up to the maximum of 3 flashes
        laser_flash_count = jnp.where(
            goal_reached & ((level % 4) == 3) & (laser_flash_count < 3),
            laser_flash_count + 1,
            laser_flash_count
        )

        # TODO: add rendering stuff for level transition
        return level, player_x, player_y, bullet_state, creature_states, item_states, last_creature_spawn, amonition_timer, laser_flash_cooldown, has_key, laser_flash_count



    # score update based on creature deaths & item collections
    @partial(jax.jit, static_argnums=(0,))
    def update_score(self, score, prev_creature_states, new_creature_states, 
                     prev_item_states, new_item_states, 
                     prev_lives, new_lives,
                     goal_reached, amonition_timer, level):
        
        # check if player has died
        has_died = (prev_lives > new_lives)

        # compare previous and new death_timer to detect kills
        # Kills happen exactly when death_timer goes from -1 to 15
        killed_creatures_mask = (prev_creature_states[:, 5] == -1) & (new_creature_states[:, 5] == 15)
        creature_points = killed_creatures_mask * self.consts.CREATURE_POINTS[prev_creature_states[:, 2]]
        # if player has died, don't give points for defeated creatures in this step, otherwise sum up points for defeated creatures
        total_score_for_defeating_creatures = jnp.where(has_died, 0, jnp.sum(creature_points))
        
        # compare previous and new item_states to detect collections
        collected_items_mask = (prev_item_states[:, 3] == self.consts.ACTIVE) & (new_item_states[:, 3] == self.consts.INACTIVE)
        item_points = collected_items_mask * self.consts.ITEM_POINTS[prev_item_states[:, 2]]
        total_score_for_collected_items = jnp.sum(item_points)

        # if goal is reached, give a score bonus [0, 100] in steps of 10 based on remaining ammunition
        ammonition_percentage = jnp.clip(amonition_timer / self.consts.LEVEL_AMMO_SUPPLY[level], 0.0, 1.0)
        goal_bonus = jnp.where(goal_reached, (jnp.floor(ammonition_percentage * 10) * 10).astype(jnp.int32), 0)


        new_score = score + total_score_for_defeating_creatures + total_score_for_collected_items + goal_bonus

        return new_score


    # Step logic
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TutankhamState, action: int):
        level=state.level
        player_x = state.player_x
        player_y = state.player_y
        tutankham_score = state.tutankham_score
        bullet_state = state.bullet_state
        creature_states = state.creature_states
        item_states = state.item_states
        laser_flash_count = state.laser_flash_count
        last_creature_spawn = state.last_creature_spawn
        laser_flash_cooldown = state.laser_flash_cooldown
        amonition_timer = state.amonition_timer
        player_lives = state.player_lives
        has_key = state.has_key
        last_directional_action = state.last_directional_action
        player_direction= state.player_direction
        step_counter = state.step_counter
        player_subpixel = state.player_subpixel
        creature_subpixels = state.creature_subpixels
        bullet_subpixel = state.bullet_subpixel
        rng_key = state.rng_key
        camera_offset = state.camera_offset
        goal_reached = state.goal_reached

        # move player based on action input and check for teleporter trigger, also update camera offset
        (player_x, player_y,
         last_directional_action, player_direction,
         is_moving, step_counter, player_subpixel, camera_offset)= self.player_step(
            player_x, player_y,
            action, last_directional_action,
            player_direction, step_counter, player_subpixel, level
        )


        bullet_state, bullet_subpixel, amonition_timer = self.bullet_step(bullet_state, bullet_subpixel, player_x, player_y, amonition_timer, action, level)

        creature_states, creature_subpixels = self.creature_step(creature_states, creature_subpixels, camera_offset, step_counter, level, player_x, player_y, rng_key)

        creature_states, last_creature_spawn, rng_key = self.spawner_step(creature_states, last_creature_spawn, level, rng_key, camera_offset, laser_flash_cooldown)

        # laser flash step should go after creature step to immediately remove creatures
        creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn = self.laser_flash_step(creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn, action)

        # store creature_states and item_states before resolving collisions (for score update)
        prev_creature_states = creature_states.copy()
        prev_item_states = item_states.copy()
        prev_lives = player_lives

        # resolve bullet-creature collisions
        creature_states, bullet_state = self.resolve_bullet_collisions(creature_states, bullet_state)

        (player_x, player_y, 
         bullet_state, creature_states, creature_subpixels,
         player_lives, last_creature_spawn,
         last_directional_action
         ) = self.resolve_player_creature_collisions(
             player_x, player_y,
             creature_states, creature_subpixels, bullet_state,
             player_lives, last_creature_spawn,
             level, last_directional_action
             )
        
        # resolve player-item collisions
        item_states = self.resolve_player_item_collisions(player_x, player_y, item_states)

        # check if player has collected the key
        has_key = self.check_key(item_states, has_key)

        # check if player has reached the goal with the key to complete the level
        goal_reached = self.check_goal(player_x, player_y, has_key, level)

        # Update score based on creature deaths & items collected
        tutankham_score = self.update_score(tutankham_score, 
                                            prev_creature_states, creature_states, 
                                            prev_item_states, item_states,
                                            prev_lives, player_lives,
                                            goal_reached, amonition_timer, level
                                            )
        
        # Prepares state for the next level if goal is reached for current level
        (level, player_x, player_y,
         bullet_state, creature_states, item_states,
         last_creature_spawn, amonition_timer, laser_flash_cooldown,
         has_key, laser_flash_count) = self.map_transition(
             goal_reached, level,
             player_x, player_y, bullet_state,
             creature_states, item_states,
             last_creature_spawn, amonition_timer,
             laser_flash_cooldown, has_key, laser_flash_count)



        new_state = TutankhamState(level=level,
                                player_x=player_x,
                                player_y=player_y,
                                tutankham_score=tutankham_score,
                                player_lives=player_lives,
                                bullet_state=bullet_state,
                                amonition_timer=amonition_timer,
                                creature_states=creature_states,
                                item_states=item_states,
                                last_creature_spawn=last_creature_spawn,
                                laser_flash_count=laser_flash_count,
                                laser_flash_cooldown=laser_flash_cooldown,
                                player_direction=player_direction,
                                is_moving=is_moving,
                                step_counter=step_counter,
                                player_subpixel=player_subpixel,
                                creature_subpixels=creature_subpixels,
                                bullet_subpixel=bullet_subpixel,
                                has_key=has_key,
                                last_directional_action=last_directional_action,
                                rng_key=rng_key,
                                camera_offset=camera_offset,
                                goal_reached=goal_reached
                                )

        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        # return observation, new_state, env_reward, done, info
        return obs, new_state, reward, done, info



    # -----------------------------
    # Action & Observation Space
    # -----------------------------
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))


    def observation_space(self) -> spaces.Box:
        # Pixel-only training: PixelObsWrapper ignores this and builds its own
        # space from image_space(). This dummy satisfies AtariWrapper.__init__.
        return spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32)
    

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8
        )
    

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: TutankhamState) -> TutankhamObservation:
        # Pixel-only training: this obs is discarded by PixelObsWrapper.
        return TutankhamObservation(dummy=jnp.array(0.0))
    
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TutankhamState) -> jnp.ndarray:
        return self.renderer.render(state)


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TutankhamState) -> TutankhamInfo:
        return TutankhamInfo(step=state.step_counter, score=state.tutankham_score)


    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TutankhamState, current_state: TutankhamState):
        return current_state.tutankham_score - previous_state.tutankham_score
    

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TutankhamState) -> bool:
        game_over = state.player_lives <= 0
        beat_game = state.level >= 16
        return jnp.logical_or(game_over, beat_game)