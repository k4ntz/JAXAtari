import os
from enum import IntEnum
from functools import partial
from typing import NamedTuple, Tuple, Optional
import jax.lax
import jax.numpy as jnp
import chex
from PIL.ImageChops import logical_and
from tensorstore import int32
from flax import struct

from jaxatari.modification import AutoDerivedConstants
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation


def _load_collision_masks():
    """Helper to load WALL and LIGHT masks for collision detection."""
    base_dir = render_utils.get_base_sprite_dir()
    jr_temp = render_utils.JaxRenderingUtils(render_utils.RendererConfig())
    wall = jr_temp.loadFrame(os.path.join(base_dir, "hauntedhouse/Wall.npy"))
    light = jr_temp.loadFrame(os.path.join(base_dir, "hauntedhouse/SpriteLight1.npy"))
    WALL_COLOR_BLUE = jnp.array([24, 26, 167, 255], dtype=jnp.uint8)
    LIGHT_COLOR = jnp.array([198, 108, 58, 255], dtype=jnp.uint8)
    wall_mask = jnp.any(wall == WALL_COLOR_BLUE, axis=-1)
    light_mask = jnp.any(light == LIGHT_COLOR, axis=-1)
    return wall_mask, light_mask


_COLLISION_WALL, _COLLISION_LIGHT = _load_collision_masks()

def _create_static_procedural_sprites() -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # Procedural colors for wall palette swapping
    wall_colors = jnp.array([
        [24, 26, 167, 255],   # Blue (original)
        [163, 57, 21, 255],   # Red
        [24, 98, 78, 255],    # Green
        [162, 134, 56, 255],  # Yellow
        [255, 255, 255, 255], # White
        [198, 108, 58, 255],  # Light Color
    ], dtype=jnp.uint8).reshape(-1, 1, 1, 4)
    
    return {
        'wall_colors': wall_colors,
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for HauntedHouse.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    static_procedural = _create_static_procedural_sprites()
    
    # Define item files for grouping
    ground_item_files = [
        'SpriteItemScepter.npy',
        'SpriteItemUrnLeft.npy',
        'SpriteItemUrnMiddle.npy',
        'SpriteItemUrnRight.npy',
        'SpriteItemUrnLeftMiddle.npy',
        'SpriteItemUrnMiddleRight.npy',
        'SpriteItemUrnLeftRight.npy',
        'SpriteItemUrn.npy'
    ]
    
    held_item_files = [
        'SpriteHeldItemScepter.npy',
        'SpriteHeldItemUrnLeft.npy',
        'SpriteHeldItemUrnMiddle.npy',
        'SpriteHeldItemUrnRight.npy',
        'SpriteHeldItemUrnLeftMiddle.npy',
        'SpriteHeldItemUrnMiddleRight.npy',
        'SpriteHeldItemUrnLeftRight.npy',
        'SpriteHeldItemUrn.npy'
    ]

    return (
        # Background
        {'name': 'background', 'type': 'background', 'file': 'SpriteBackground.npy'},
        
        # Procedural Colors
        {'name': 'wall_colors', 'type': 'procedural', 'data': static_procedural['wall_colors']},
        
        # Sprites (as groups for padding)
        {'name': 'player_group', 'type': 'group', 'files': [
            'SpriteEyesMiddle.npy', 'SpriteEyesUp.npy', 'SpriteEyesUpLeft.npy',
            'SpriteEyesLeft.npy', 'SpriteEyesDownLeft.npy', 'SpriteEyesDown.npy',
            'SpriteEyesDownRight.npy', 'SpriteEyesRight.npy', 'SpriteEyesUpRight.npy'
        ]},
        {'name': 'ghost_group', 'type': 'group', 'files': ['SpriteGhost1.npy', 'SpriteGhost2.npy']},
        {'name': 'spider_group', 'type': 'group', 'files': ['SpriteSpider1.npy', 'SpriteSpider2.npy']},
        {'name': 'bat_group', 'type': 'group', 'files': ['SpriteBat1.npy', 'SpriteBat2.npy']},
        {'name': 'light_group', 'type': 'group', 'files': ['SpriteLight1.npy', 'SpriteLight2.npy']},
        # Wall & UI
        {'name': 'wall', 'type': 'single', 'file': 'Wall.npy'},
        {'name': 'scoreboard', 'type': 'single', 'file': 'SpriteScoreboard.npy'},
        {'name': 'blackbar', 'type': 'single', 'file': 'SpriteBlackBar.npy'},
        {'name': 'stairsthintop', 'type': 'single', 'file': 'SpriteStairsThin.npy'},
        {'name': 'stairswide', 'type': 'single', 'file': 'SpriteStairsWide.npy'},
        # Items (Ground) - Load as groups for uniform padding
        {'name': 'ground_item_group', 'type': 'group', 'files': ground_item_files},
        # Items (Held) - Load as groups for uniform padding
        {'name': 'held_item_group', 'type': 'group', 'files': held_item_files},
        # Digits
        {'name': 'digits', 'type': 'digits', 'pattern': 'SpriteNumber{}.npy'},
    )

@struct.dataclass
class EntityPosition:
    x: jnp.ndarray
    y: jnp.ndarray
    floor: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    
    

class HauntedHouseConstants(AutoDerivedConstants):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    Y_WALKING_DISTANCE: int = struct.field(pytree_node=False, default=2)  # This is compensation for the stretched pixels in the ALE version. However, the game
                                 # feels much better when this is set to 1. Possible values: [1, 2]


    WALL_Y_OFFSET: int = 4
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (0, 0, 0))
    WALL_COLOR_BLUE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([24, 26, 167, 255], dtype=jnp.uint8))
    WALL_COLOR_RED: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([163, 57, 21, 255], dtype=jnp.uint8))
    WALL_COLOR_GREEN: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([24, 98, 78, 255], dtype=jnp.uint8))
    WALL_COLOR_YELLOW: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([162, 134, 56, 255], dtype=jnp.uint8))
    WALL_COLOR_WHITE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([255, 255, 255, 255], dtype=jnp.uint8))
    LIGHT_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([198, 108, 58, 255], dtype=jnp.uint8))
    ITEM_OFFSETS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([(0, 0), (0, 2), (2, 0), (5, 2),
                                          (0, 0), (2, 0), (0, 2), (0, 0)], dtype=jnp.uint8))


    STAIRS_TOP_LEFT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([True, True, True, True], dtype=jnp.bool))
    STAIRS_TOP_RIGHT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([False, True, True, False], dtype=jnp.bool))
    STAIRS_BOTTOM_LEFT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([False, True, True, False], dtype=jnp.bool))
    STAIRS_BOTTOM_RIGHT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([True, True, True, True], dtype=jnp.bool))
    STAIRS_LEFT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([False, True, True, False], dtype=jnp.bool))
    STAIRS_RIGHT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([False, False, True, True], dtype=jnp.bool))

    DIRECTION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
    0,  # Action.NOOP (0)
    0,  # Action.FIRE (1)
    1,  # Action.UP (2)
    7,  # Action.RIGHT (3)
    3,  # Action.LEFT (4)
    5,  # Action.DOWN (5)
    8,  # Action.UPRIGHT (6)
    2,  # Action.UPLEFT (7)
    6,  # Action.DOWNRIGHT (8)
    4,  # Action.DOWNLEFT (9)
    1,  # Action.UPFIRE (10)
    7,  # Action.RIGHTFIRE (11)
    3,  # Action.LEFTFIRE (12)
    5,  # Action.DOWNFIRE (13)
    8,  # Action.UPRIGHTFIRE (14)
    2,  # Action.UPLEFTFIRE (15)
    6,  # Action.DOWNRIGHTFIRE (16)
    4,  # Action.DOWNLEFTFIRE (17)
], dtype=jnp.int32))

    ADJACENCY_MATRIX: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([(0,0,0,0),                                         #0
                                              (2,3,7,0),    (1,4,0,0),     (1,4,5,0),     (2,3,6,0),    #1-4
                                              (3,6,0,0),    (4,5,8,0),     (1,0,0,0),     (6,0,0,0),    #5-8
                                              (10,11,15,0), (9,12,16,0),   (9,12,13,19),  (10,11,14,0), #9-12
                                              (11,14,17,0), (12,13,16,0),  (9,0,0,0),     (10,0,0,0),   #13-16
                                              (13,0,0,0),   (14,0,0,0),    (11,0,0,0),    (21,22,26,0), #17-20
                                              (20,23,27,0), (20,23,24,30), (21,22,25,31), (22,25,28,0), #21-24
                                              (23,24,29,0), (20,0,0,0),    (21,0,0,0),    (24,0,0,0),   #25-28
                                              (25,0,0,0),   (22,0,0,0),    (23,0,0,0),    (33,34,38,0), #29-32
                                              (32,35,0,0),  (32,35,36,0),  (33,34,37,40), (34,37,0,0),  #33-36
                                              (35,36,39,0), (32,0,0,0),    (37,0,0,0),    (35,0,0,0)    #37-40
                                              ], dtype=jnp.int32))
    LOCATIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([(0, 0, 1),                                         #0
                                       (36, 106, 1),  (116, 106, 1), (36, 234, 1),  (116, 234, 1), #1-4
                                       (36, 428, 1),  (116, 428, 1), (36, 6, 1),    (116, 496, 1), #5-8
                                       (36, 106, 2),  (116, 106, 2), (36, 234, 2),  (116, 234, 2), #9-12
                                       (36, 428, 2),  (116, 428, 2), (36, 6, 2),    (116, 6, 2),   #13-16
                                       (36, 496, 2),  (116, 496, 2), (4, 234, 2),   (36, 106, 3),  #17-20
                                       (116, 106, 3), (36, 234, 3),  (116, 234, 3), (36, 428, 3),  #21-24
                                       (116, 428, 3), (36, 6, 3),    (116, 6, 3),   (36, 496, 3),  #25-28
                                       (116, 496, 3), (4, 234, 3),   (148, 234, 3), (36, 106, 4),  #29-32
                                       (116, 106, 4), (36, 234, 4),  (116, 234, 4), (36, 428, 4),  #33-36
                                       (116, 428, 4), (36, 6, 4),    (116, 496, 4), (148, 234, 4)  #37-40
                                       ], dtype=jnp.int32))

    UP_STAIRS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([7, 8, 16, 17, 19, 26, 29, 31]))
    DOWN_STAIRS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([15, 18, 27, 28, 30, 38, 39, 40]))
    ROOMS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([1,2,3,5,6,9,10,11,12,13,14,20,21,22,23,24,25,32,33,34,35,36,37]))
    STAIR_TRANSITIONS: chex. Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0,0,0,0,0,0,0,15,18,0,0, #0-10
                                                0,0,0,0,7,27,28,8,30,0,  #11-20
                                                0,0,0,0,0,38,16,17,39,19, #21-30
                                                40,0,0,0,0,0,0,26,29,31])) #31-40

    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 6))
    GHOST_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 16))
    SPIDER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 16))
    BAT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 20))
    SCEPTER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 16))
    URN_LEFT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 8))
    URN_MIDDLE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 16))
    URN_RIGHT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 8))
    URN_LEFT_MIDDLE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (6, 16))
    URN_MIDDLE_RIGHT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (6, 16))
    URN_LEFT_RIGHT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 8))
    URN_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 16))

    # Derived constants (computed from other constants)
    ITEM_SIZES: Optional[chex.Array] = struct.field(pytree_node=False, default=None)
    ROOM_POSITIONS: Optional[chex.Array] = struct.field(pytree_node=False, default=None)
    ROOM_SIZES: Optional[chex.Array] = struct.field(pytree_node=False, default=None)
    WALL: Optional[chex.Array] = struct.field(pytree_node=False, default=None)
    LIGHT: Optional[chex.Array] = struct.field(pytree_node=False, default=None)

    INVISIBLE_ENTITY: EntityPosition = struct.field(pytree_node=False, default_factory=lambda: EntityPosition(
        x=jnp.array(-1), 
        y=jnp.array(-1), 
        floor=jnp.array(-1), 
        width=jnp.array(-1), 
        height=jnp.array(-1)
    ))

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=lambda: _get_default_asset_config())

    def compute_derived(self):
        """Compute derived constants based on static fields."""
        return {
            'ITEM_SIZES': jnp.array([
                self.SCEPTER_SIZE, 
                self.URN_LEFT_SIZE, 
                self.URN_MIDDLE_SIZE, 
                self.URN_RIGHT_SIZE,
                self.URN_LEFT_MIDDLE_SIZE, 
                self.URN_MIDDLE_RIGHT_SIZE, 
                self.URN_LEFT_RIGHT_SIZE, 
                self.URN_SIZE
            ], dtype=jnp.int32),
            # The -2 is a jank hack that allows it to always pass same_floor checks
            'ROOM_POSITIONS': jnp.array([
                (0, 0, -2), (80, 0, -2), (0, 178, -2), (80, 178, -2), (0, 338, -2), (80, 338, -2)
            ], dtype=jnp.int32),
            'ROOM_SIZES': jnp.array([
                (80, 177), (80, 177), (80, 161), (80, 161), (80, 177), (80, 177)
            ], dtype=jnp.int32),
            # Used for collision detection
            'WALL': _COLLISION_WALL,
            'LIGHT': _COLLISION_LIGHT,
        }


# immutable state container
@struct.dataclass
class HauntedHouseState:
    step_counter: chex.Array
    player: chex.Array
    player_direction: chex.Array
    ghost: chex.Array
    bat: chex.Array
    spider: chex.Array
    current_nodes: chex.Array
    previous_nodes: chex.Array
    chasing: chex.Array
    stun_duration: chex.Array
    match_duration: chex.Array
    matches_used: chex.Array
    lives: chex.Array
    item_held: chex.Array
    scepter: chex.Array
    urn_left: chex.Array
    urn_middle: chex.Array
    urn_right: chex.Array
    urn_left_middle: chex.Array
    urn_middle_right: chex.Array
    urn_left_right: chex.Array
    urn: chex.Array
    stairs_active: chex.Array
    fire_button_active: chex.Array


@struct.dataclass
class HauntedHouseObservation:
    player: ObjectObservation
    enemies: ObjectObservation  # Merged Ghost, Spider, Bat
    items: ObjectObservation    # Merged Scepter, Urn Parts, Full Urn
    item_held: jnp.ndarray
    match_duration: jnp.ndarray
    matches_used: jnp.ndarray
    chasing: jnp.ndarray
    lives: jnp.ndarray

@struct.dataclass
class HauntedHouseInfo:
    time: jnp.ndarray


class JaxHauntedHouse(JaxEnvironment[HauntedHouseState, HauntedHouseObservation, HauntedHouseInfo, HauntedHouseConstants]):
    # Minimal ALE action set for Haunted House
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
    
    def __init__(self, consts: HauntedHouseConstants = None):
        consts = consts or HauntedHouseConstants()
        super().__init__(consts)
        self.renderer = HauntedHouseRenderer(self.consts)
        self.obs_size = 3*4+1+1



    @partial(jax.jit, static_argnums=(0,))
    def enemy_step(self, state: HauntedHouseState):

        random_key = jax.random.PRNGKey(state.step_counter + state.player[1])
        should_move = state.step_counter % 4 == 0
        p = state.player[2]
        same_floor = jnp.array([p == state.ghost[2], p == state.spider[2], p == state.bat[2]])
        could_chase = jnp.logical_and(state.item_held != 1, same_floor)
        should_chase = jnp.logical_and(state.stun_duration == 0, could_chase)

        ghost_chase = jax.lax.cond(should_chase[0],
                                   lambda x: self.check_same_room(state.player, self.consts.PLAYER_SIZE, state.ghost, self.consts.GHOST_SIZE),
                                   lambda x: False,
                                   None)

        spider_chase = jax.lax.cond(should_chase[1],
                                   lambda x: self.check_same_room(state.player, self.consts.PLAYER_SIZE, state.spider, self.consts.SPIDER_SIZE),
                                   lambda x: False,
                                   None)

        bat_chase = jax.lax.cond(should_chase[2],
                                   lambda x: self.check_same_room(state.player, self.consts.PLAYER_SIZE, state.bat, self.consts.BAT_SIZE),
                                   lambda x: False,
                                   None)


        ghost, ghost_c_node, ghost_p_node = jax.lax.cond(jnp.logical_and(should_move, jnp.logical_not(ghost_chase)),
                                                    lambda g, cn, pn: self.enemy_roaming(g, cn, pn, random_key),
                                                    lambda g, cn, pn: (g, cn, pn),
                                                    state.ghost, state.current_nodes[0], state.previous_nodes[0])

        spider, spider_c_node, spider_p_node = jax.lax.cond(jnp.logical_and(should_move, jnp.logical_not(spider_chase)),
                                                         lambda s, cn, pn: self.enemy_roaming(s, cn, pn, random_key),
                                                         lambda s, cn, pn: (s, cn, pn),
                                                         state.spider, state.current_nodes[1], state.previous_nodes[1])

        bat, bat_c_node, bat_p_node = jax.lax.cond(jnp.logical_and(should_move, jnp.logical_not(bat_chase)),
                                                         lambda b, cn, pn: self.enemy_roaming(b, cn, pn, random_key),
                                                         lambda b, cn, pn: (b, cn, pn),
                                                         state.bat, state.current_nodes[2], state.previous_nodes[2])


        ghost = jax.lax.cond(jnp.logical_and(should_move, ghost_chase),
                             lambda x: self.enemy_chasing(state.player, ghost),
                             lambda x: ghost,
                             None)

        spider = jax.lax.cond(jnp.logical_and(should_move, spider_chase),
                             lambda x: self.enemy_chasing(state.player, spider),
                             lambda x: spider,
                             None)

        bat = jax.lax.cond(jnp.logical_and(should_move, bat_chase),
                             lambda x: self.enemy_chasing(state.player, bat),
                             lambda x: bat,
                             None)


        current_nodes = jnp.array([ghost_c_node, spider_c_node, bat_c_node])
        previous_nodes = jnp.array([ghost_p_node, spider_p_node, bat_p_node])

        ghost, spider, bat, current_nodes, previous_nodes = jax.lax.cond(state.stun_duration == 1,
                                                                         lambda x: self.random_spawns(random_key),
                                                                         lambda x: (ghost, spider, bat, current_nodes, previous_nodes),
                                                                         None)

        chasing = jax.lax.cond(
            jnp.any(jnp.array([ghost_chase, spider_chase, bat_chase])),
            lambda s: s + 1,
            lambda s: jnp.array(0),
            operand=state.chasing,
        )


        return ghost, spider, bat, current_nodes, previous_nodes, chasing


    @partial(jax.jit, static_argnums=(0,))
    def enemy_roaming(self, enemy, c_node, p_node, random_key):
        direction_x = jnp.sign(self.consts.LOCATIONS[c_node][0] - enemy[0])
        direction_y = jnp.sign(self.consts.LOCATIONS[c_node][1] - enemy[1])
        direction_y = jnp.where(jnp.logical_and(self.consts.Y_WALKING_DISTANCE == 2, self.consts.LOCATIONS[c_node][1] - enemy[1] == 1), 0, direction_y)

        new_x = enemy[0] + direction_x.astype(jnp.int32)
        new_y = enemy[1] + (direction_y.astype(jnp.int32) * self.consts.Y_WALKING_DISTANCE) # Compensation for the long stretched pixels
        arrived = jnp.where(self.consts.Y_WALKING_DISTANCE == 2,
                            jnp.logical_and(new_x == self.consts.LOCATIONS[c_node][0], jnp.logical_or(new_y == self.consts.LOCATIONS[c_node][1], new_y == self.consts.LOCATIONS[c_node][1] - 1)),
                            jnp.logical_and(new_x == self.consts.LOCATIONS[c_node][0], new_y == self.consts.LOCATIONS[c_node][1]))
        up = jnp.any((c_node == self.consts.UP_STAIRS))
        down = jnp.any((c_node == self.consts.DOWN_STAIRS))


        same_floor = jnp.logical_and(arrived, jnp.logical_not(jnp.logical_or(up, down)))
        up_floor = jnp.logical_and(arrived, up)
        down_floor = jnp.logical_and(arrived, down)

        def get_next_node(current_node, previous_node, key):
            # Create boolean masks to filter out placeholders and previous nodes
            potential_nodes = self.consts.ADJACENCY_MATRIX[current_node]

            valid_nodes_mask = potential_nodes != 0
            not_previous_node_mask = potential_nodes != previous_node

            combined_mask = jnp.logical_and(valid_nodes_mask, not_previous_node_mask)

            probs = jnp.where(combined_mask, 1.0, 0.0)
            probs = probs / jnp.sum(probs)  # Normalize probabilities

            next_node = jax.random.choice(key, potential_nodes, p=probs)
            return next_node


        enemy_floor, c_node, p_node = jax.lax.cond(same_floor,
                                                     lambda x: (enemy[2], get_next_node(c_node, p_node, random_key), c_node),
                                                     lambda x: (enemy[2], c_node, p_node),
                                                     None)

        enemy_floor, c_node, p_node = jax.lax.cond(up_floor,
                                                   lambda x: (enemy_floor + 1, get_next_node(self.consts.STAIR_TRANSITIONS[c_node], p_node, random_key), self.consts.STAIR_TRANSITIONS[c_node]),
                                                   lambda x: (enemy_floor, c_node, p_node),
                                                   None)

        enemy_floor, c_node, p_node = jax.lax.cond(down_floor,
                                                   lambda x: (enemy_floor - 1, get_next_node(self.consts.STAIR_TRANSITIONS[c_node], p_node, random_key), self.consts.STAIR_TRANSITIONS[c_node]),
                                                   lambda x: (enemy_floor, c_node, p_node),
                                                   None)



        enemy = jnp.array([new_x, new_y, enemy_floor])
        return enemy, c_node, p_node

    @partial(jax.jit, static_argnums=(0,))
    def enemy_chasing(self, player, enemy):
        direction_x = jnp.sign(player[0] - enemy[0])
        direction_y = jnp.sign(player[1] - enemy[1])
        new_x = enemy[0] + direction_x.astype(jnp.int32)
        new_y = enemy[1] + direction_y.astype(jnp.int32)
        return jnp.array([new_x, new_y, enemy[2]])


    @partial(jax.jit, static_argnums=(0,))
    def check_same_room(self, pos1, size1, pos2, size2):
        vmapped_check_collision = jax.vmap(
            self.check_entity_collision,
            in_axes=(None, None, 0, 0))

        entity1_room = vmapped_check_collision(pos1, size1, self.consts.ROOM_POSITIONS, self.consts.ROOM_SIZES)
        entity2_room = vmapped_check_collision(pos2, size2, self.consts.ROOM_POSITIONS, self.consts.ROOM_SIZES)

        return jnp.any(jnp.logical_and(entity1_room, entity2_room))

    @partial(jax.jit, static_argnums=(0,))
    def random_spawns(self, key):
        # The 'replace=False' argument guarantees the indices are not repeated.
        random_rooms = jax.random.choice(key, self.consts.ROOMS, shape=(3,), replace=False)

        ghost = self.consts.LOCATIONS[random_rooms[0]]
        spider = self.consts.LOCATIONS[random_rooms[1]]
        bat = self.consts.LOCATIONS[random_rooms[2]]

        current_nodes = random_rooms
        previous_nodes = jnp.zeros(3, dtype=jnp.int32)

        return ghost, spider, bat, current_nodes, previous_nodes

    @partial(jax.jit, static_argnums=(0,))
    def random_item_spawns(self, key):
        # The 'replace=False' argument prevents the items from spawning on top of each other.
        random_rooms = jax.random.choice(key, self.consts.ROOMS, shape=(4,), replace=False)

        scepter = self.consts.LOCATIONS[random_rooms[0]]
        urn_left = self.consts.LOCATIONS[random_rooms[1]]
        urn_middle = self.consts.LOCATIONS[random_rooms[2]]
        urn_right = self.consts.LOCATIONS[random_rooms[3]]

        return scepter, urn_left, urn_middle, urn_right



    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: HauntedHouseState, action: chex.Array):
        stun_duration = state.stun_duration
        match_duration = state.match_duration

        player = state.player

        fire = jnp.logical_or(action == Action.FIRE, jnp.logical_and(action >= 10, action <= 17))

        player_direction = jnp.array([state.player_direction[1], self.consts.DIRECTION[action]])
        try_move_up = jnp.logical_or(player_direction[0] == 1,
                                 jnp.logical_or(player_direction[0] == 2, player_direction[0] == 8))
        try_move_down = jnp.logical_or(player_direction[0] == 5,
                                 jnp.logical_or(player_direction[0] == 4, player_direction[0] == 6))
        try_move_left = jnp.logical_or(player_direction[0] == 3,
                                 jnp.logical_or(player_direction[0] == 2, player_direction[0] == 4))
        try_move_right = jnp.logical_or(player_direction[0] == 7,
                                 jnp.logical_or(player_direction[0] == 6, player_direction[0] == 8))

        move_up = jax.lax.cond(jnp.logical_and(try_move_up, stun_duration == 0),
                               lambda p: jnp.logical_not(self.check_wall_collision(jnp.array([p[0], p[1] - self.consts.Y_WALKING_DISTANCE, p[2]]),
                                                                                   self.consts.PLAYER_SIZE)),
                               lambda p: False,
                               operand=player)
        move_down = jax.lax.cond(jnp.logical_and(try_move_down, stun_duration == 0),
                                 lambda p: jnp.logical_not(self.check_wall_collision(jnp.array([p[0], p[1] + self.consts.Y_WALKING_DISTANCE, p[2]]),
                                                                                     self.consts.PLAYER_SIZE)),
                                 lambda p: False,
                                 operand=player)
        move_up_half = jax.lax.cond(jnp.logical_and(jnp.logical_not(move_up), jnp.logical_and(try_move_up, stun_duration == 0)),
                               lambda p: jnp.logical_not(self.check_wall_collision(jnp.array([p[0], p[1] - 1, p[2]]), self.consts.PLAYER_SIZE)),
                               lambda p: False,
                               operand=player)
        move_down_half = jax.lax.cond(jnp.logical_and(jnp.logical_not(move_down), jnp.logical_and(try_move_down, stun_duration == 0)),
                               lambda p: jnp.logical_not(self.check_wall_collision(jnp.array([p[0], p[1] + 1, p[2]]), self.consts.PLAYER_SIZE)),
                               lambda p: False,
                               operand=player)
        move_left = jax.lax.cond(jnp.logical_and(try_move_left, stun_duration == 0),
                               lambda p: jnp.logical_not(self.check_wall_collision(jnp.array([p[0] - 1, p[1], p[2]]), self.consts.PLAYER_SIZE)),
                               lambda p: False,
                               operand=player)
        move_right = jax.lax.cond(jnp.logical_and(try_move_right, stun_duration == 0),
                               lambda p: jnp.logical_not(self.check_wall_collision(jnp.array([p[0] + 1, p[1], p[2]]), self.consts.PLAYER_SIZE)),
                               lambda p: False,
                               operand=player)


        player = player.at[1].set(jnp.where(move_up, jnp.clip(player[1] - self.consts.Y_WALKING_DISTANCE, min=6, max=506), player[1]))
        player = player.at[1].set(jnp.where(move_down, jnp.clip(player[1] + self.consts.Y_WALKING_DISTANCE, min=6, max=506), player[1]))
        player = player.at[1].set(jnp.where(move_up_half, jnp.clip(player[1] - 1, min=6, max=506), player[1]))
        player = player.at[1].set(jnp.where(move_down_half, jnp.clip(player[1] + 1, min=6, max=506), player[1]))
        player = player.at[0].set(jnp.where(move_left, jnp.clip(player[0] - 1, min=4, max=156), player[0]))
        player = player.at[0].set(jnp.where(move_right, jnp.clip(player[0] + 1, min=4, max=156), player[0]))

        up, down, escape, stairs_active = self.check_stairs_collision(player, self.consts.PLAYER_SIZE, state.stairs_active)

        player = player.at[2].set(jnp.where(up,   player[2] + 1, player[2]))
        player = player.at[2].set(jnp.where(down, player[2] - 1, player[2]))


        # Monsters
        ghost = self.check_entity_collision(player, self.consts.PLAYER_SIZE, state.ghost, self.consts.GHOST_SIZE)
        spider = self.check_entity_collision(player, self.consts.PLAYER_SIZE, state.spider, self.consts.SPIDER_SIZE)
        bat = self.check_entity_collision(player, self.consts.PLAYER_SIZE, state.bat, self.consts.BAT_SIZE)
        monster_hit = jnp.logical_and(state.item_held != 1, jnp.logical_or(jnp.logical_or(ghost, spider), bat))

        lives = jnp.where(jnp.logical_and(monster_hit, stun_duration == 0), state.lives - 1, state.lives)
        stun_duration = jnp.where(jnp.logical_and(monster_hit, stun_duration == 0), 256, stun_duration)
        stun_duration = jnp.where(stun_duration > 0, stun_duration - 1, stun_duration)


        # Items
        item_dropped = jnp.all(jnp.array([fire, state.fire_button_active, stun_duration == 0, match_duration > 0]))

        match_duration = jax.lax.cond(match_duration > 0,
                                      lambda s: s - 1,
                                      lambda s: s,
                                      operand=match_duration)

        match_lit = jnp.all(jnp.array([fire, state.fire_button_active, stun_duration == 0, match_duration == 0, state.chasing == 0]))
        match_duration = jnp.where(match_lit, 1192, match_duration)
        match_duration = jnp.where(state.chasing > 0, jnp.array(0), match_duration)
        matches_used = jnp.where(match_lit, state.matches_used + 1, state.matches_used)

        fire_button_active = jnp.where(fire, False, True)
        game_ends = jnp.logical_or(lives == 0, jnp.logical_and(escape, state.item_held == 8))
        return player, player_direction, stun_duration, match_duration, matches_used, item_dropped, stairs_active, fire_button_active, lives, game_ends





    @partial(jax.jit, static_argnums=(0,))
    def item_step(self, state: HauntedHouseState, item_dropped):

        item_held = state.item_held

        item_held, scepter, urn_left, urn_middle, urn_right, urn_left_middle, urn_middle_right, urn_left_right, urn = (
            jax.lax.cond(item_dropped,
                         lambda x: self.drop_item(state),
                         lambda x: (x, state.scepter, state.urn_left, state.urn_middle, state.urn_right,
                                    state.urn_left_middle, state.urn_middle_right, state.urn_left_right, state.urn),
                         item_held))

        light = state.match_duration > 0

        scepter_collision = jnp.logical_and(light, self.check_entity_collision(state.player, self.consts.PLAYER_SIZE, scepter, self.consts.SCEPTER_SIZE))
        urn_left_collision = jnp.logical_and(light, self.check_entity_collision(state.player, self.consts.PLAYER_SIZE, urn_left, self.consts.URN_LEFT_SIZE))
        urn_middle_collision = jnp.logical_and(light, self.check_entity_collision(state.player, self.consts.PLAYER_SIZE, urn_middle, self.consts.URN_MIDDLE_SIZE))
        urn_right_collision = jnp.logical_and(light, self.check_entity_collision(state.player, self.consts.PLAYER_SIZE, urn_right, self.consts.URN_RIGHT_SIZE))
        urn_left_middle_collision = jnp.logical_and(light, self.check_entity_collision(state.player, self.consts.PLAYER_SIZE, urn_left_middle, self.consts.URN_LEFT_MIDDLE_SIZE))
        urn_middle_right_collision = jnp.logical_and(light, self.check_entity_collision(state.player, self.consts.PLAYER_SIZE, urn_middle_right, self.consts.URN_MIDDLE_RIGHT_SIZE))
        urn_left_right_collision = jnp.logical_and(light, self.check_entity_collision(state.player, self.consts.PLAYER_SIZE, urn_left_right, self.consts.URN_LEFT_RIGHT_SIZE))
        urn_collision = jnp.logical_and(light, self.check_entity_collision(state.player, self.consts.PLAYER_SIZE, urn, self.consts.URN_SIZE))

        urn_pieces = jnp.array([urn_left_collision, urn_middle_collision, urn_right_collision,
                                urn_left_middle_collision, urn_middle_right_collision, urn_left_right_collision])

        urn_piece_picked_up = jnp.any(urn_pieces)

        whole_item_picked_up = jnp.logical_or(scepter_collision, urn_collision)

        item_held, scepter, urn_left, urn_middle, urn_right, urn_left_middle, urn_middle_right, urn_left_right, urn = (
            jax.lax.cond(jnp.logical_or(whole_item_picked_up, jnp.logical_and(item_held == 1, urn_piece_picked_up)),
                         lambda x: self.drop_item(state),
                         lambda x: (x, scepter, urn_left, urn_middle, urn_right,
                                    urn_left_middle, urn_middle_right, urn_left_right, urn),
                         item_held))


        item_held = jnp.where(scepter_collision, jnp.array(1), item_held)
        scepter = jnp.where(scepter_collision, jnp.array([-1, -1, -1]), scepter)
        item_held = jnp.where(urn_collision, jnp.array(8), item_held)
        urn = jnp.where(urn_collision, jnp.array([-1, -1, -1]), urn)

        item_held = jax.lax.cond(urn_piece_picked_up,
                                lambda x: self.combine_urn(urn_pieces, item_held),
                                lambda x: item_held, None)

        urn_left = jnp.where(urn_left_collision, jnp.array([-1, -1, -1]), urn_left)
        urn_middle = jnp.where(urn_middle_collision, jnp.array([-1, -1, -1]), urn_middle)
        urn_right = jnp.where(urn_right_collision, jnp.array([-1, -1, -1]), urn_right)
        urn_left_middle = jnp.where(urn_left_middle_collision, jnp.array([-1, -1, -1]), urn_left_middle)
        urn_middle_right = jnp.where(urn_middle_right_collision, jnp.array([-1, -1, -1]), urn_middle_right)
        urn_left_right = jnp.where(urn_left_right_collision, jnp.array([-1, -1, -1]), urn_left_right)

        return item_held, scepter, urn_left, urn_middle, urn_right, urn_left_middle, urn_middle_right, urn_left_right, urn




    @partial(jax.jit, static_argnums=(0,))
    def drop_item(self, state: HauntedHouseState):
        scepter = state.scepter
        urn_left = state.urn_left
        urn_middle = state.urn_middle
        urn_right = state.urn_right
        urn_left_middle = state.urn_left_middle
        urn_middle_right = state.urn_middle_right
        urn_left_right = state.urn_left_right
        urn = state.urn

        i = (scepter, urn_left, urn_middle, urn_right, urn_left_middle, urn_middle_right, urn_left_right, urn)

        def no_item():
            return jnp.array(0), i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]
        def drop_scepter():
            pos = self.find_drop_position(state.player, state.player_direction, self.consts.SCEPTER_SIZE)
            return jnp.array(0), pos, i[1], i[2], i[3], i[4], i[5], i[6], i[7]
        def drop_urn_left():
            pos = self.find_drop_position(state.player, state.player_direction, self.consts.URN_LEFT_SIZE)
            return jnp.array(0), i[0], pos, i[2], i[3], i[4], i[5], i[6], i[7]
        def drop_urn_middle():
            pos = self.find_drop_position(state.player, state.player_direction, self.consts.URN_MIDDLE_SIZE)
            return jnp.array(0), i[0], i[1], pos, i[3], i[4], i[5], i[6], i[7]
        def drop_urn_right():
            pos = self.find_drop_position(state.player, state.player_direction, self.consts.URN_RIGHT_SIZE)
            return jnp.array(0), i[0], i[1], i[2], pos, i[4], i[5], i[6], i[7]
        def drop_urn_left_middle():
            pos = self.find_drop_position(state.player, state.player_direction, self.consts.URN_LEFT_MIDDLE_SIZE)
            return jnp.array(0), i[0], i[1], i[2], i[3], pos, i[5], i[6], i[7]
        def drop_urn_middle_right():
            pos = self.find_drop_position(state.player, state.player_direction, self.consts.URN_MIDDLE_RIGHT_SIZE)
            return jnp.array(0), i[0], i[1], i[2], i[3], i[4], pos, i[6], i[7]
        def drop_urn_left_right():
            pos = self.find_drop_position(state.player, state.player_direction, self.consts.URN_LEFT_RIGHT_SIZE)
            return jnp.array(0), i[0], i[1], i[2], i[3], i[4], i[5], pos, i[7]
        def drop_urn():
            pos = self.find_drop_position(state.player, state.player_direction, self.consts.URN_SIZE)
            return jnp.array(0), i[0], i[1], i[2], i[3], i[4], i[5], i[6], pos

        item_functions = [no_item, drop_scepter, drop_urn_left, drop_urn_middle, drop_urn_right,
                          drop_urn_left_middle, drop_urn_middle_right, drop_urn_left_right, drop_urn]

        return jax.lax.switch(state.item_held, item_functions)





    @partial(jax.jit, static_argnums=(0,))
    def find_drop_position(self, player, player_direction, size):
        position1 = jnp.array([player[0] - 10, player[1] - 10, player[2]]).astype(jnp.int32)
        position2 = jnp.array([player[0], player[1] + 30, player[2]]).astype(jnp.int32)
        position3 = jnp.array([player[0] + 10, player[1] - 10, player[2]]).astype(jnp.int32)

        pos1_collision = self.check_wall_collision(position1, size)
        pos2_collision = self.check_wall_collision(position2, size)

        standing_position = jnp.where(pos1_collision,
                         jnp.where(pos2_collision, position3, position2),
                         position1)

        position = jnp.where(player_direction[1] == 1,
                             jnp.array([player[0] + ((8 - size[0]) / 2), player[1] + 10, player[2]]).astype(jnp.int32),
                             standing_position)
        position = jnp.where(player_direction[1] == 2,
                             jnp.array([player[0] + 12 + ((8 - size[0]) / 2), player[1] + 10 - (size[1] / 4), player[2]]).astype(jnp.int32),
                             position)
        position = jnp.where(player_direction[1] == 3,
                             jnp.array([player[0] + 12 + ((8 - size[0]) / 2), player[1] - (size[1] / 4), player[2]]).astype(jnp.int32),
                             position)
        position = jnp.where(player_direction[1] == 4,
                             jnp.array([player[0] + 12 + ((8 - size[0]) / 2), player[1] - 20 - (size[1] / 4), player[2]]).astype(jnp.int32),
                             position)
        position = jnp.where(player_direction[1] == 5,
                             jnp.array([player[0] + ((8 - size[0]) / 2), player[1] - 20, player[2]]).astype(jnp.int32),
                             position)
        position = jnp.where(player_direction[1] == 6,
                             jnp.array([player[0] - 12 + ((8 - size[0]) / 2), player[1] - 20 - (size[1] / 4), player[2]]).astype(jnp.int32),
                             position)
        position = jnp.where(player_direction[1] == 7,
                             jnp.array([player[0] - 12 + ((8 - size[0]) / 2), player[1] - (size[1] / 4), player[2]]).astype(jnp.int32),
                             position)
        position = jnp.where(player_direction[1] == 8,
                             jnp.array([player[0] - 12 + ((8 - size[0]) / 2), player[1] + 10 - (size[1] / 4), player[2]]).astype(jnp.int32),
                             position)

        position = jnp.where(self.check_wall_collision(position, size), standing_position, position)

        return position

    @partial(jax.jit, static_argnums=(0,))
    def combine_urn(self, urn_piece, item_held):
        item = item_held

        item = jnp.where(jnp.logical_and(urn_piece[0], item_held == 0), 2, item)
        item = jnp.where(jnp.logical_and(urn_piece[0], item_held == 3), 5, item)
        item = jnp.where(jnp.logical_and(urn_piece[0], item_held == 4), 7, item)
        item = jnp.where(jnp.logical_and(urn_piece[0], item_held == 6), 8, item)

        item = jnp.where(jnp.logical_and(urn_piece[1], item_held == 0), 3, item)
        item = jnp.where(jnp.logical_and(urn_piece[1], item_held == 2), 5, item)
        item = jnp.where(jnp.logical_and(urn_piece[1], item_held == 4), 6, item)
        item = jnp.where(jnp.logical_and(urn_piece[1], item_held == 7), 8, item)

        item = jnp.where(jnp.logical_and(urn_piece[2], item_held == 0), 4, item)
        item = jnp.where(jnp.logical_and(urn_piece[2], item_held == 2), 7, item)
        item = jnp.where(jnp.logical_and(urn_piece[2], item_held == 3), 6, item)
        item = jnp.where(jnp.logical_and(urn_piece[2], item_held == 5), 8, item)

        item = jnp.where(jnp.logical_and(urn_piece[3], item_held == 0), 5, item)
        item = jnp.where(jnp.logical_and(urn_piece[3], item_held == 4), 8, item)

        item = jnp.where(jnp.logical_and(urn_piece[4], item_held == 0), 6, item)
        item = jnp.where(jnp.logical_and(urn_piece[4], item_held == 2), 8, item)

        item = jnp.where(jnp.logical_and(urn_piece[5], item_held == 0), 7, item)
        item = jnp.where(jnp.logical_and(urn_piece[5], item_held == 3), 8, item)

        return jnp.array(item).astype(jnp.int32)







    @partial(jax.jit, static_argnums=(0,))
    def check_entity_collision(self, pos1, size1, pos2, size2):
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

        same_floor = jnp.logical_or(pos1[2] == pos2[2], pos2[2] == -2) # The -2 is for the room checks

        return jnp.logical_and(same_floor, jnp.logical_and(horizontal_overlap, vertical_overlap))

    @partial(jax.jit, static_argnums=(0,))
    def check_wall_collision(self, pos, size):
        """Check collision between an entity and the wall"""

        # Because the wall sprite is not at (0,0)
        pos = jnp.array([pos[0], pos[1] - self.consts.WALL_Y_OFFSET])

        collision_top_left = self.consts.WALL[pos[1]][pos[0]]
        collision_top_right = self.consts.WALL[pos[1]][pos[0] + size[0] - 1]
        collision_bottom_left = self.consts.WALL[pos[1] + size[1] - 1][pos[0]]
        collision_bottom_right = self.consts.WALL[pos[1] + size[1] - 1][pos[0] + size[0] - 1]

        return jnp.any(jnp.array([collision_top_left, collision_top_right, collision_bottom_right, collision_bottom_left]))
        # return False

    @partial(jax.jit, static_argnums=(0,))
    def check_stairs_collision(self, entity, size, stairs_active):
        """Check collision between an entity and the stairs"""
        in_range = jnp.logical_and(entity[0] < 70, entity[1] < 7)
        same_floor = self.consts.STAIRS_TOP_LEFT[entity[2] - 1]
        top_left = jnp.logical_and(same_floor, in_range)
        in_range = jnp.logical_and(entity[0] > 70, entity[1] < 7)
        same_floor = self.consts.STAIRS_TOP_RIGHT[entity[2] - 1]
        top_right = jnp.logical_and(same_floor, in_range)

        in_range = jnp.logical_and(entity[0] < 70, entity[1] + size[1] > 510)
        same_floor = self.consts.STAIRS_BOTTOM_LEFT[entity[2] - 1]
        bottom_left = jnp.all(jnp.array([same_floor, in_range]))
        in_range = jnp.logical_and(entity[0] > 70, entity[1] + size[1] > 510)
        same_floor = self.consts.STAIRS_BOTTOM_RIGHT[entity[2] - 1]
        bottom_right = jnp.all(jnp.array([same_floor, in_range]))

        in_range = entity[0] < 5
        same_floor = self.consts.STAIRS_LEFT[entity[2] - 1]
        left = jnp.all(jnp.array([same_floor, in_range]))

        in_range = entity[0] + size[0] > 155
        same_floor = self.consts.STAIRS_RIGHT[entity[2] - 1]
        right = jnp.all(jnp.array([same_floor, in_range]))


        even_floor = entity[2] % 2 == 0
        up_even = jnp.logical_and(jnp.any(jnp.array([top_right, bottom_left, left])), even_floor)
        up_odd = jnp.logical_and(jnp.any(jnp.array([top_left, bottom_right, right])), jnp.logical_not(even_floor))
        down_even = jnp.logical_and(jnp.any(jnp.array([top_left, bottom_right, right])), even_floor)
        down_odd = jnp.logical_and(jnp.any(jnp.array([top_right, bottom_left, left])), jnp.logical_not(even_floor))
        up_trigger = jnp.logical_or(up_even, up_odd)
        down_trigger = jnp.logical_or(down_even, down_odd)
        escape = jnp.logical_and(in_range, entity[2] == 1)

        up = jnp.where(stairs_active, up_trigger, False)
        down = jnp.where(stairs_active, down_trigger, False)
        stairs_active = jnp.where(jnp.logical_or(up_trigger, down_trigger), False, True)

        return up, down, escape, stairs_active

    @partial(jax.jit, static_argnums=(0,))
    def check_visibility(self, player, entity: EntityPosition):

        camera_offset = jnp.where(player[1] < 82, 0, player[1] - 82)
        camera_offset = jnp.clip(camera_offset, min=0, max=356)

        # Calculate the on-screen Y-coordinate
        on_screen_y = entity.y - camera_offset

        # Check if the entity's bounding box is within the on-screen view
        is_horizontally_visible = jnp.logical_and(entity.x + entity.width > 0, entity.x < self.consts.WIDTH)
        is_vertically_visible = jnp.logical_and(on_screen_y + entity.height > 0,
                                                on_screen_y < self.consts.HEIGHT)
        return jnp.logical_and(is_horizontally_visible, is_vertically_visible)

    @partial(jax.jit, static_argnums=(0,))
    def check_illuminated_single(self, player, pos, size):
        pos = jnp.array([pos[0] - player.x + 12, pos[1] - player.y + 24])
        height, width = self.consts.LIGHT.shape

        oob = jnp.logical_or(jnp.logical_or(pos[1] < 0, pos[1] >= height),
                             jnp.logical_or(pos[0] < 0, pos[0] >= width))
        collision_top_left = jax.lax.cond(oob,
                                          lambda x: False,
                                          lambda x: self.consts.LIGHT[pos[1]][pos[0]],
                                          operand=None)

        oob = jnp.logical_or(jnp.logical_or(pos[1] < 0, pos[1] >= height),
                             jnp.logical_or(pos[0] + size[0] - 1 < 0, pos[0] + size[0] - 1 >= width))
        collision_top_right = jax.lax.cond(oob,
                                           lambda x: False,
                                           lambda x: self.consts.LIGHT[pos[1]][pos[0] + size[0] - 1],
                                           operand=None)

        oob = jnp.logical_or(jnp.logical_or(pos[1] < 0 + size[1] - 1, pos[1] + size[1] - 1 >= height),
                             jnp.logical_or(pos[0] < 0, pos[0] >= width))
        collision_bottom_left = jax.lax.cond(oob,
                                             lambda x: False,
                                             lambda x: self.consts.LIGHT[pos[1] + size[1] - 1][pos[0]],
                                             operand=None)

        oob = jnp.logical_or(jnp.logical_or(pos[1] + size[1] - 1 < 0, pos[1] + size[1] - 1 >= height),
                             jnp.logical_or(pos[0] + size[0] - 1 < 0, pos[0] + size[0] - 1 >= width))
        collision_bottom_right = jax.lax.cond(oob,
                                              lambda x: False,
                                              lambda x: self.consts.LIGHT[pos[1] + size[1] - 1][pos[0] + size[0] - 1],
                                              operand=None)

        return jnp.any(
            jnp.array([collision_top_left, collision_top_right, collision_bottom_right, collision_bottom_left]))




    def reset(self, key=None) -> Tuple[HauntedHouseObservation, HauntedHouseState]:

        ghost, spider, bat, current_nodes, previous_nodes = self.random_spawns(key)
        key, _ = jax.random.split(key, 2)
        scepter, urn_left, urn_middle, urn_right = self.random_item_spawns(key)

        state = HauntedHouseState(
            step_counter=jnp.array(0).astype(jnp.int32),
            player=jnp.array([128, 240, 1]).astype(jnp.int32),
            player_direction=jnp.array([0, 0]), # 0: no direction, 1: up, then clockwise
            ghost=ghost,
            spider=spider,
            bat=bat,
            current_nodes=current_nodes,
            previous_nodes = previous_nodes,
            chasing = jnp.array(0),
            stun_duration=jnp.array(0).astype(jnp.int32),
            match_duration=jnp.array(0).astype(jnp.int32),
            matches_used=jnp.array(0).astype(jnp.int32),
            lives=jnp.array(9).astype(jnp.int32),
            item_held=jnp.array(0).astype(jnp.int32),
            scepter=scepter,
            urn_left=urn_left,
            urn_middle=urn_middle,
            urn_right=urn_right,
            urn_left_middle=jnp.array([-1, -1, -1]).astype(jnp.int32),
            urn_middle_right=jnp.array([-1, -1, -1]).astype(jnp.int32),
            urn_left_right=jnp.array([-1, -1, -1]).astype(jnp.int32),
            urn=jnp.array([-1, -1, -1]).astype(jnp.int32),
            stairs_active=jnp.array(True).astype(jnp.bool),
            fire_button_active=jnp.array(True).astype(jnp.bool)
        )

        initial_obs = self._get_observation(state)

        return initial_obs, state



    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HauntedHouseState, action: chex.Array) -> Tuple[HauntedHouseObservation, HauntedHouseState, float, bool, HauntedHouseInfo]:
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

        # Step 1: Player Mechanics
        (player, player_direction, stun_duration, match_duration, matches_used,
         item_dropped, stairs_active, fire_button_active, lives, game_ends) = self.player_step(state, atari_action)

        # Step 2: Item Mechanics
        (item_held, scepter, urn_left, urn_middle, urn_right,
         urn_left_middle, urn_middle_right, urn_left_right, urn) = self.item_step(state, item_dropped)

        # Step 3: Enemy Mechanics
        ghost, spider, bat, current_nodes, previous_nodes, chasing = self.enemy_step(state)


        # Step 4: Increase Step Counter
        step_counter_reset_condition = False
        step_counter = jax.lax.cond(
            step_counter_reset_condition,
            lambda s: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )


        new_state = HauntedHouseState(
            step_counter=step_counter,
            player=player,
            player_direction=player_direction,
            ghost=ghost,
            spider=spider,
            bat=bat,
            current_nodes=current_nodes,
            previous_nodes=previous_nodes,
            chasing=chasing,
            stun_duration=stun_duration,
            match_duration=match_duration,
            matches_used=matches_used,
            lives=lives,
            item_held=item_held,
            scepter=scepter,
            urn_left=urn_left,
            urn_middle=urn_middle,
            urn_right=urn_right,
            urn_left_middle=urn_left_middle,
            urn_middle_right=urn_middle_right,
            urn_left_right=urn_left_right,
            urn=urn,
            stairs_active=stairs_active,
            fire_button_active=fire_button_active
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info




    def render(self, state: HauntedHouseState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: HauntedHouseState) -> HauntedHouseObservation:
        # --- Player ---
        p_dir = state.player_direction[1]
        p_ori = jnp.select(
            [p_dir == 1, p_dir == 3, p_dir == 5, p_dir == 7],
            [0.0, 90.0, 180.0, 270.0],
            default=0.0
        )

        player = ObjectObservation.create(
            x=jnp.clip(state.player[0], 0, self.consts.WIDTH),
            y=jnp.clip(state.player[1], 0, self.consts.HEIGHT),
            width=jnp.array(self.consts.PLAYER_SIZE[0], dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_SIZE[1], dtype=jnp.int32),
            orientation=p_ori.astype(jnp.float32),
            state=state.player[2].astype(jnp.int32), # Floor number as state
            active=jnp.array(1, dtype=jnp.int32)
        )

        # --- Enemies (Merged) ---
        # 0: Ghost, 1: Spider, 2: Bat
        en_pos = jnp.stack([state.ghost, state.spider, state.bat])
        en_sizes = jnp.array([
            self.consts.GHOST_SIZE, 
            self.consts.SPIDER_SIZE, 
            self.consts.BAT_SIZE
        ])
        
        # Visibility Check
        def check_enemy_vis(pos, size):
            visible_floor = (state.player[2] == pos[2])
            # Reconstruct EntityPosition just for the check function
            e_ent = EntityPosition(pos[0], pos[1], pos[2], size[0], size[1])
            visible_screen = self.check_visibility(state.player, e_ent)
            return visible_floor & visible_screen

        en_active = jax.vmap(check_enemy_vis)(en_pos, en_sizes)
        
        enemies = ObjectObservation.create(
            x=jnp.clip(en_pos[:, 0], 0, self.consts.WIDTH),
            y=jnp.clip(en_pos[:, 1], 0, self.consts.HEIGHT),
            width=en_sizes[:, 0].astype(jnp.int32),
            height=en_sizes[:, 1].astype(jnp.int32),
            visual_id=jnp.arange(3, dtype=jnp.int32), # ID distinguishes enemy type
            active=en_active.astype(jnp.int32)
        )

        # --- Items (Merged) ---
        # 0: Scepter
        # 1-6: Urn Pieces
        # 7: Full Urn
        item_pos_stack = jnp.stack([
            state.scepter, 
            state.urn_left, state.urn_middle, state.urn_right,
            state.urn_left_middle, state.urn_middle_right, state.urn_left_right, 
            state.urn
        ])
        
        item_sizes = jnp.array([
            self.consts.SCEPTER_SIZE,
            self.consts.URN_LEFT_SIZE, self.consts.URN_MIDDLE_SIZE, self.consts.URN_RIGHT_SIZE,
            self.consts.URN_LEFT_MIDDLE_SIZE, self.consts.URN_MIDDLE_RIGHT_SIZE, self.consts.URN_LEFT_RIGHT_SIZE,
            self.consts.URN_SIZE
        ])
        
        is_lit = state.match_duration > 0
        
        def check_item_vis(pos, size):
            visible_floor = (state.player[2] == pos[2])
            p_ent = EntityPosition(state.player[0], state.player[1], state.player[2], self.consts.PLAYER_SIZE[0], self.consts.PLAYER_SIZE[1])
            visible_light = self.check_illuminated_single(p_ent, pos, size)
            return visible_floor & is_lit & visible_light

        item_active = jax.vmap(check_item_vis)(item_pos_stack, item_sizes)

        items = ObjectObservation.create(
            x=jnp.clip(item_pos_stack[:, 0], 0, self.consts.WIDTH),
            y=jnp.clip(item_pos_stack[:, 1], 0, self.consts.HEIGHT),
            width=item_sizes[:, 0].astype(jnp.int32),
            height=item_sizes[:, 1].astype(jnp.int32),
            visual_id=jnp.arange(8, dtype=jnp.int32), # Matches the item_held ID logic
            active=item_active.astype(jnp.int32)
        )

        return HauntedHouseObservation(
            player=player,
            enemies=enemies,
            items=items,
            item_held=state.item_held,
            match_duration=jnp.int32(state.match_duration > 0),
            matches_used=state.matches_used,
            chasing=jnp.int32(state.chasing > 0),
            lives=state.lives
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            
            # Enemies: Ghost, Spider, Bat (n=3)
            "enemies": spaces.get_object_space(n=3, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            
            # Items: Scepter, 6 Urn pieces, Full Urn (n=8)
            "items": spaces.get_object_space(n=8, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            
            "item_held": spaces.Discrete(9),
            "match_duration": spaces.Discrete(2),
            "matches_used": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
            "chasing": spaces.Discrete(2),
            "lives": spaces.Box(low=0, high=9, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: HauntedHouseState) -> HauntedHouseInfo:
        return HauntedHouseInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: HauntedHouseState, state: HauntedHouseState):
        penality_chasing = jnp.where(state.chasing > 0, 1, 0)
        return state.item_held + state.lives - state.stun_duration / 10 - state.matches_used - penality_chasing

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: HauntedHouseState) -> bool:
        win = jnp.logical_and(jnp.logical_and(state.item_held == 8, state.player[2] == 1),
                               state.player[0] + self.consts.PLAYER_SIZE[0] > 155)
        loss = state.lives == 0
        result = jax.lax.cond(jnp.logical_or(win, loss),
                              lambda x: True,
                              lambda x: False,
                              None)
        return result


class HauntedHouseRenderer(JAXGameRenderer):
    def __init__(self, consts: HauntedHouseConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or HauntedHouseConstants()
        super().__init__(self.consts)
        self.sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "hauntedhouse")
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
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
        
        # 4. Store key color IDs for wall palette swapping
        self.BLUE_WALL_ID = self.COLOR_TO_ID.get((24, 26, 167), 0)
        self.WALL_COLOR_IDS = jnp.array([
            self.COLOR_TO_ID.get((255, 255, 255), 0),  # 0: White
            self.BLUE_WALL_ID,                         # 1: Blue
            self.COLOR_TO_ID.get((163, 57, 21), 0),    # 2: Red
            self.COLOR_TO_ID.get((24, 98, 78), 0),    # 3: Green
            self.COLOR_TO_ID.get((162, 134, 56), 0)   # 4: Yellow
        ])
        
        # 5. Store animation lengths
        self.anim_len = {
            'light': self.SHAPE_MASKS['light_group'].shape[0],
        }
        
        # 6. Get the pre-built stacks from SHAPE_MASKS
        # These are now guaranteed to have the same shape.
        self.HELD_ITEM_STACKS = self.SHAPE_MASKS['held_item_group']
        self.ITEM_MASKS_STACK = self.SHAPE_MASKS['ground_item_group']
        
        # 7. Get the single, uniform offset for each group
        self.HELD_ITEM_OFFSET = self.FLIP_OFFSETS['held_item_group']
        self.ITEM_OFFSET = self.FLIP_OFFSETS['ground_item_group']

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state:HauntedHouseState):
        # 1. Start with the static blue background
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # 2. Calculate camera offset
        camera_offset = jnp.where(state.player[1] < 82, 0, state.player[1] - 82)
        camera_offset = jnp.clip(camera_offset, min=0, max=356)
        
        # 3. Check which items are lit
        illuminated = jnp.where(
            state.match_duration > 0,
            self._check_illuminated(state),
            jnp.array([False, False, False, False, False, False, False, False])
        )
        
        # 4. Render Player
        player_frame = jnp.where(state.stun_duration > 0, state.stun_duration % 8 + 1, state.player_direction[1])
        player_mask = self.SHAPE_MASKS['player_group'][player_frame]
        raster = self.jr.render_at(
            raster, state.player[0], state.player[1] - camera_offset,
            player_mask, flip_offset=self.FLIP_OFFSETS['player_group']
        )
        
        # 5. Render Monsters
        k = jax.random.key(state.step_counter)
        k_ghost, k_spider, k_bat = jax.random.split(k, 3)
        ghost_frame = jax.random.randint(k_ghost, minval=0, maxval=2, shape=())
        spider_frame = jax.random.randint(k_spider, minval=0, maxval=2, shape=())
        bat_frame = jax.random.randint(k_bat, minval=0, maxval=2, shape=())
        
        vanish_with_scepter = (state.item_held == 1) & (state.step_counter % 4 != 0)
        not_vanishing = ~vanish_with_scepter
        
        floor_checks = (state.player[2] == state.ghost[2],
                        state.player[2] == state.spider[2],
                        state.player[2] == state.bat[2])
        
        raster = jax.lax.cond(
            floor_checks[0] & not_vanishing,
            lambda r: self.jr.render_at_clipped(
                r, state.ghost[0], state.ghost[1] - camera_offset,
                self.SHAPE_MASKS['ghost_group'][ghost_frame],
                flip_offset=self.FLIP_OFFSETS['ghost_group']
            ),
            lambda r: r,
            raster
        )
        
        raster = jax.lax.cond(
            floor_checks[1] & not_vanishing,
            lambda r: self.jr.render_at_clipped(
                r, state.spider[0], state.spider[1] - camera_offset,
                self.SHAPE_MASKS['spider_group'][spider_frame],
                flip_offset=self.FLIP_OFFSETS['spider_group']
            ),
            lambda r: r,
            raster
        )
        
        raster = jax.lax.cond(
            floor_checks[2] & not_vanishing,
            lambda r: self.jr.render_at_clipped(
                r, state.bat[0], state.bat[1] - camera_offset,
                self.SHAPE_MASKS['bat_group'][bat_frame],
                flip_offset=self.FLIP_OFFSETS['bat_group']
            ),
            lambda r: r,
            raster
        )
        
        # 6. Render Items
        item_positions = jnp.stack([
            state.scepter, state.urn_left, state.urn_middle, state.urn_right,
            state.urn_left_middle, state.urn_middle_right, state.urn_left_right, state.urn
        ])
        
        def render_item(i, r):
            item_pos = item_positions[i]
            on_same_floor = state.player[2] == item_pos[2]
            is_lit = illuminated[i]
            
            # Use the single uniform offset for the group
            return jax.lax.cond(
                on_same_floor & is_lit,
                lambda r_in: self.jr.render_at_clipped(
                    r_in, item_pos[0], item_pos[1] - camera_offset,
                    self.ITEM_MASKS_STACK[i],
                    flip_offset=self.ITEM_OFFSET  # Use the single group offset
                ),
                lambda r_in: r_in,
                r
            )
        
        raster = jax.lax.fori_loop(0, 8, render_item, raster)
        
        # 7. Render Light
        light_frame = jnp.where((state.step_counter % 8) < 2, 1, 0)
        light_mask = self.SHAPE_MASKS['light_group'][light_frame]
        raster = jax.lax.cond(
            (state.step_counter % 2 == 0) & (state.match_duration > 0),
            # Use render_at_clipped for negative y positions
            lambda r: self.jr.render_at_clipped(
                r, state.player[0] - 12, state.player[1] - 24 - camera_offset,
                light_mask, flip_offset=self.FLIP_OFFSETS['light_group']
            ),
            lambda r: r,
            raster
        )
        
        # 8. Render Stairs
        stairsthintop_mask = self.SHAPE_MASKS['stairsthintop']
        stairsthintop_offset = self.FLIP_OFFSETS['stairsthintop']
        # Procedurally create the flipped bottom stair mask
        stairsthinbottom_mask = jnp.flip(stairsthintop_mask, axis=0)
        
        stairswide_mask = self.SHAPE_MASKS['stairswide']
        stairswide_offset = self.FLIP_OFFSETS['stairswide']
        floor_idx = state.player[2] - 1  # Convert 1-based floor to 0-based index
        
        raster = jax.lax.cond(
            self.consts.STAIRS_TOP_LEFT[floor_idx] & (state.match_duration > 0),
            lambda r: self.jr.render_at_clipped(r, 32, 7 - camera_offset, stairsthintop_mask, flip_offset=stairsthintop_offset),
            lambda r: r, raster
        )
        
        raster = jax.lax.cond(
            self.consts.STAIRS_TOP_RIGHT[floor_idx] & (state.match_duration > 0),
            lambda r: self.jr.render_at_clipped(r, 112, 7 - camera_offset, stairsthinbottom_mask, flip_offset=stairsthintop_offset),
            lambda r: r, raster
        )
        
        raster = jax.lax.cond(
            self.consts.STAIRS_BOTTOM_LEFT[floor_idx] & (state.match_duration > 0),
            lambda r: self.jr.render_at_clipped(r, 32, 478 - camera_offset, stairsthintop_mask, flip_offset=stairsthintop_offset),
            lambda r: r, raster
        )
        
        raster = jax.lax.cond(
            self.consts.STAIRS_BOTTOM_RIGHT[floor_idx] & (state.match_duration > 0),
            lambda r: self.jr.render_at_clipped(r, 112, 478 - camera_offset, stairsthinbottom_mask, flip_offset=stairsthintop_offset),
            lambda r: r, raster
        )
        
        raster = jax.lax.cond(
            self.consts.STAIRS_LEFT[floor_idx] & (state.match_duration > 0),
            lambda r: self.jr.render_at_clipped(r, 5, 226 - camera_offset, stairswide_mask, flip_offset=stairswide_offset),
            lambda r: r, raster
        )
        
        raster = jax.lax.cond(
            self.consts.STAIRS_RIGHT[floor_idx] & (state.match_duration > 0),
            lambda r: self.jr.render_at_clipped(r, 140, 225 - camera_offset, stairswide_mask, flip_offset=stairswide_offset),
            lambda r: r, raster
        )
        
        # 9. Render Wall (with palette swap)
        k_flash = jax.random.key(state.step_counter)
        k_flash, _ = jax.random.split(k_flash)
        chase_flash = (state.chasing % 64 == 1) | (state.chasing % 64 == 9)
        stun_flash = (state.stun_duration % 4 == 1) & jax.random.bernoulli(k_flash)
        wall_flashing = chase_flash | stun_flash
        wall_color_selector = jnp.where(wall_flashing, 0, state.player[2])  # 0 is White
        wall_color_id = self.WALL_COLOR_IDS[wall_color_selector]
        
        indices_to_update = jnp.array([self.BLUE_WALL_ID])
        new_color_ids = jnp.array([wall_color_id])
        
        raster = self.jr.render_at_clipped(
            raster, 0, self.consts.WALL_Y_OFFSET - camera_offset,
            self.SHAPE_MASKS['wall'], flip_offset=self.FLIP_OFFSETS['wall']
        )
        
        # 10. Render UI
        # Use render_at_clipped for the black bars (they use camera offset)
        raster = self.jr.render_at_clipped(
            raster, 32, self.consts.WALL_Y_OFFSET - camera_offset,
            self.SHAPE_MASKS['blackbar'], flip_offset=self.FLIP_OFFSETS['blackbar']
        )
        raster = self.jr.render_at_clipped(
            raster, 112, self.consts.WALL_Y_OFFSET - camera_offset,
            self.SHAPE_MASKS['blackbar'], flip_offset=self.FLIP_OFFSETS['blackbar']
        )
        # The scoreboard is at a fixed positive Y, so 'render_at' is fine.
        raster = self.jr.render_at(
            raster, 0, 161,
            self.SHAPE_MASKS['scoreboard'], flip_offset=self.FLIP_OFFSETS['scoreboard']
        )
        
        # Render Held Item
        def render_held_item(r):
            item_idx = state.item_held - 1
            mask = self.HELD_ITEM_STACKS[item_idx]
            offset = self.consts.ITEM_OFFSETS[item_idx].astype(jnp.int32)
            # Use the single uniform offset for the group
            return self.jr.render_at(
                r, 91 + offset[0], 166 + offset[1],
                mask, flip_offset=self.HELD_ITEM_OFFSET
            )
        
        raster = jax.lax.cond(state.item_held > 0, render_held_item, lambda r: r, raster)
        
        # Render Numbers
        matches_used_digits = self.jr.int_to_digits(state.matches_used, max_digits=2)
        lives_digits = self.jr.int_to_digits(state.lives, max_digits=1)
        player_floor_digits = self.jr.int_to_digits(state.player[2], max_digits=1)
        
        raster = self.jr.render_label_selective(
            raster, 37, 176, matches_used_digits, self.SHAPE_MASKS['digits'],
            0, 2, spacing=1, max_digits_to_render=2
        )
        
        raster = self.jr.render_label_selective(
            raster, 110, 176, lives_digits, self.SHAPE_MASKS['digits'],
            0, 1, spacing=0, max_digits_to_render=1
        )
        
        raster = self.jr.render_label_selective(
            raster, 61, 166, player_floor_digits, self.SHAPE_MASKS['digits'],
            0, 1, spacing=0, max_digits_to_render=1
        )
        
        # 11. Final Palette Lookup
        return self.jr.render_from_palette(
            raster,
            self.PALETTE,
            indices_to_update=indices_to_update,
            new_color_ids=new_color_ids
        )


    @partial(jax.jit, static_argnums=(0,))
    def _check_illuminated(self, state: HauntedHouseState) -> chex.Array:
        """JIT-compatible check to see which items are illuminated by the match."""
        
        # We use the 'light_group' mask for collision, not consts.LIGHT
        # Assuming frame 0 is the main light mask
        light_mask = self.SHAPE_MASKS['light_group'][0]
        light_h, light_w = light_mask.shape
        
        # Player's light is centered at (player_x + 12, player_y + 24)
        # and we check from the top-left of the light mask
        light_topleft_x = state.player[0] - 12
        light_topleft_y = state.player[1] - 24
        entities = jnp.array([
            state.scepter, state.urn_left, state.urn_middle, state.urn_right,
            state.urn_left_middle, state.urn_middle_right, state.urn_left_right, state.urn
        ])
        
        def _check_single(item_pos, item_size):
            # Calculate item pos relative to the light mask's top-left corner
            rel_x = item_pos[0] - light_topleft_x
            rel_y = item_pos[1] - light_topleft_y
            
            # Check 4 corners
            points_x = jnp.array([rel_x, rel_x + item_size[0] - 1, rel_x, rel_x + item_size[0] - 1])
            points_y = jnp.array([rel_y, rel_y, rel_y + item_size[1] - 1, rel_y + item_size[1] - 1])
            
            def check_point(i):
                px, py = points_x[i], points_y[i]
                # Check bounds
                oob = (px < 0) | (px >= light_w) | (py < 0) | (py >= light_h)
                # Check if light mask is "on" at this pixel
                # We assume the light mask's non-transparent pixels are not TRANSPARENT_ID
                is_lit = light_mask[py, px] != self.jr.TRANSPARENT_ID
                return jnp.where(oob, False, is_lit)
            
            # Check if any of the 4 corners are lit
            return jnp.any(jax.vmap(check_point)(jnp.arange(4)))
        
        return jax.vmap(_check_single)(entities, self.consts.ITEM_SIZES)