from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Adventure.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        #all rooms in order ToDo put overview map into the readme?
        {'name': 'room_number', 'type': 'group', 'files': ['Room_1.npy', 
                                                           'Room_2.npy', 
                                                           'Room_3.npy', 
                                                           'Room_4.npy', 
                                                           'Room_5.npy', 
                                                           'Room_6.npy', 
                                                           'Room_7.npy', 
                                                           'Room_8.npy', 
                                                           'Room_9.npy', 
                                                           'Room_10.npy', 
                                                           'Room_11.npy', 
                                                           'Room_12.npy', 
                                                           'Room_13.npy', 
                                                           'Room_14.npy']},
        {'name': 'bg', 'type': 'background', 'file': 'Room_1.npy'},
        #Player in all the different colors, which change depending on the background
        {'name': 'player_colors', 'type': 'group', 'files': ["Player_Yellow.npy",
                                                             "Player_Yellow.npy", 
                                                             "Player_Green.npy",
                                                             "Player_Purple.npy",
                                                             "Player_Pink.npy",
                                                             "Player_Green_Yellow.npy",
                                                             "Player_Blue.npy",
                                                             "Player_Blue.npy",
                                                             "Player_Blue.npy",
                                                             "Player_Blue.npy",
                                                             "Player_Blue.npy",
                                                             "Player_Black.npy",
                                                             "Player_Pink.npy",
                                                             "Player_Magenta.npy"]},
        #Dragons and their animations
        {'name': 'dragon_yellow', 'type': 'group', 'files': ['Dragon_yellow_neutral.npy',
                                                             'Dragon_yellow_attack.npy',
                                                             'Dragon_yellow_dead.npy']},
        {'name': 'dragon_green', 'type': 'group', 'files': ['Dragon_green_neutral.npy',
                                                             'Dragon_green_attack.npy',
                                                             'Dragon_green_dead.npy']},
        {'name': 'dragon_red', 'type': 'group', 'files': ['Dragon_green_neutral.npy',
                                                             'Dragon_green_attack.npy',
                                                             'Dragon_green_dead.npy']},                                                     
        #Keys
        {'name': 'key_yellow', 'type': 'single', 'file': 'Key_yellow.npy'},
        {'name': 'key_black', 'type': 'single', 'file': 'Key_black.npy'},
        {'name': 'key_white', 'type': 'single', 'file': 'Key_black.npy'},
        #Gate and its animation
        {'name': 'gate_state', 'type': 'group', 'files': ['Gate_closed.npy',
                                                          'Gate_opening_0.npy',
                                                          'Gate_opening_1.npy',
                                                          'Gate_opening_2.npy',
                                                          'Gate_opening_3.npy',
                                                          'Gate_opening_4.npy',
                                                          'Gate_open.npy']},
        #Items
        {'name': 'sword', 'type': 'single', 'file': 'Sword.npy'},
        {'name': 'bridge', 'type': 'single', 'file': 'Bridge.npy'},
        {'name': 'magnet', 'type': 'single', 'file': 'Magnet.npy'},
        #Chalice and its diffrent colors
        {'name': 'chalice', 'type': 'group', 'files': ['Chalice_Black.npy',
                                                       'Chalice_DarkBlue.npy',
                                                       'Chalice_Gray.npy',
                                                       'Chalice_Green.npy',
                                                       'Chalice_LightBlue.npy',
                                                       'Chalice_Pink.npy',
                                                       'Chalice_Purple.npy',
                                                       'Chalice_Red.npy',
                                                       'Chalice_Turquoise.npy',
                                                       'Chalice_Yellow.npy']},
        {'name': 'bat', 'type': 'group', 'files': ['Dragon_green_neutral.npy',
                                                             'Dragon_green_attack.npy']},
        {'name': 'dot', 'type': 'single', 'file': 'Key_black.npy'}
    )


class AdventureConstants(struct.PyTreeNode):
    #Map Size,  coordinates are (x,y) and the upper left corner is (0,0)
    WIDTH: int = struct.field(pytree_node=False, default = 160)
    HEIGHT: int = struct.field(pytree_node=False, default= 250)
    #Entity Sizes
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default= (4, 8))
    KEY_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default= (8, 6))
    DRAGON_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default= (8, 44))
    GATE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default= (7, 32))
    SWORD_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default= (8, 10))
    BRIDGE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default= (4, 48))
    MAGNET_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default= (8, 16))
    CHALICE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default= (8, 18))
    DOT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default= (1,1))
    #Inventory IDs
    EMPTY_HAND_ID: int = struct.field(pytree_node=False, default= 0)
    KEY_YELLOW_ID: int = struct.field(pytree_node=False, default= 1)
    KEY_BLACK_ID: int = struct.field(pytree_node=False, default= 2)
    SWORD_ID: int = struct.field(pytree_node=False, default= 3)
    BRIDGE_ID: int = struct.field(pytree_node=False, default= 4)
    MAGNET_ID: int = struct.field(pytree_node=False, default= 5)
    CHALICE_ID: int = struct.field(pytree_node=False, default= 6)
    KEY_WHITE_ID: int = struct.field(pytree_node=False, default= 7)
    DOT_ID: int = struct.field(pytree_node=False, default= 8)
    #dragons (X,Y, Room, state, counter, eat, activate)
    DRAGON_YELLOW_SPAWN: Tuple[int, int, int, int ,int, int, int] = struct.field(pytree_node=False, default= (80, 170, 5, 0, 0, 0, 0))
    DRAGON_GREEN_SPAWN: Tuple[int, int, int, int, int, int, int] = struct.field(pytree_node=False, default= (80, 130, 4, 0, 0, 0, 0))
    DRAGON_RED_SPAWN: Tuple[int, int, int, int, int, int, int] = struct.field(pytree_node=False, default= (80, 130, 19, 0, 0, 0, 0))
    #Spawn Locations of all Entities: (X, Y, Room/Tile)
    YELLOW_GATE_POS: Tuple[int, int, int] = struct.field(pytree_node=False, default= (76, 140, 0))
    BLACK_GATE_POS: Tuple[int, int, int] = struct.field(pytree_node=False, default= (76, 140, 11))
    WHITE_GATE_POS: Tuple[int, int, int] = struct.field(pytree_node=False, default= (76, 140, 24))
    PLAYER_SPAWN: Tuple[int, int, int] = struct.field(pytree_node=False, default= (78, 174, 0)) #Changed from (78, 174, 0)
    KEY_YELLOW_SPAWN: Tuple[int, int, int] = struct.field(pytree_node=False, default= (31, 110, 0)) #Changed from (31, 110, 0) for Testing
    KEY_BLACK_SPAWN: Tuple[int, int, int] = struct.field(pytree_node=False, default= (31, 100, 4))
    KEY_WHITE_SPAWN: Tuple[int, int, int] = struct.field(pytree_node=False, default= (31, 110, 19))
    SWORD_SPAWN: Tuple[int, int, int] = struct.field(pytree_node=False, default= (31,180,1))
    BRIDGE_SPAWN: Tuple[int, int, int] = struct.field(pytree_node=False, default= (40,130,10))
    MAGNET_SPAWN: Tuple[int, int, int] = struct.field(pytree_node=False, default= (120,180,12))
    CHALICE_SPAWN: Tuple[int, int, int, int] = struct.field(pytree_node=False, default= (35,180,13, 7))
    BAT_SPAWN: Tuple[int, int, int, int] = struct.field(pytree_node=False, default= (76, 140, 19, 0))
    DOT_SPAWN: Tuple[int, int, int] = struct.field(pytree_node=False, default= (76, 140, 29))
    GATE_SPAWN: Tuple[int, int] = struct.field(pytree_node=False, default=(0, 0))
    
    #Constants that are used for restricting player movement, for easy of fine tuning
    # Wall coordinates the player cannot pass through
    LEFT_WALL_X: int = struct.field(pytree_node=False, default= 8)
    RIGHT_WALL_X: int = struct.field(pytree_node=False, default= 148)
    UPPER_WALL_Y: int = struct.field(pytree_node=False, default= 43)
    LOWER_WALL_Y: int = struct.field(pytree_node=False, default= 199)
    #special black borders to the left and right
    SPECIAL_WALL_LEFT: int = struct.field(pytree_node=False, default= 12)
    SPECIAL_WALL_RIGHT: int = struct.field(pytree_node=False, default= 145)
    # Path South and North to another Room, X-Coordinates that offer hole in the wall
    PATH_VERTICAL_LEFT: int = struct.field(pytree_node=False, default= 64)
    PATH_VERTICAL_RIGHT: int = struct.field(pytree_node=False, default= 95)
    # Path East and West, Y-Coordinates that offer hole in the wall
    PATH_HORIZONTAL_UP: int = struct.field(pytree_node=False, default= 40)
    PATH_HORIZONTAL_DOWN: int = struct.field(pytree_node=False, default= 200)
    # Castle Edges
    CASTLE_TOWER_LEFT_X: int = struct.field(pytree_node=False, default= 35)
    CASTLE_TOWER_RIGHT_X: int = struct.field(pytree_node=False, default= 120)
    CASTLE_BASE_LEFT_X: int = struct.field(pytree_node=False, default= 45)
    CASTLE_BASE_RIGHT_X: int = struct.field(pytree_node=False, default= 113)
    CASTLE_TOWER_CORNER_Y: int = struct.field(pytree_node=False, default= 105)
    CASTLE_BASE_CORNER_Y: int = struct.field(pytree_node=False, default= 170)

    # sset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory= _get_default_asset_config)

    #Dragon constants
    DRAGON_SPEED: int = struct.field(pytree_node=False, default= 2)
    DRAGON_BITE_TIMER: int = struct.field(pytree_node=False, default= 15)


# immutable state container
class AdventureState(struct.PyTreeNode):
    #step conter for performance indicator?
    step_counter: chex.Array
    #position player: x ,y ,tile, inventory
    player: chex.Array
    #positions dragons: x, y ,tile ,state, counter, eat, activate
    dragon_yellow: chex.Array
    dragon_green: chex.Array
    dragon_red: chex.Array
    #positions keys: x, y, tile
    key_yellow: chex.Array
    key_black: chex.Array
    key_white: chex.Array
    #gates: state, counter
    gate_yellow: chex.Array
    gate_black: chex.Array
    gate_white: chex.Array
    #position sword: x, y, tile
    sword: chex.Array
    #position bridge: x, y, tile
    bridge: chex.Array
    #position magnet: x, y, tile
    magnet: chex.Array
    #position chalice: x, y, tile, color
    chalice: chex.Array
    #random key
    rndKey: chex.PRNGKey
    #bat: x, y, tile, state
    bat: chex.Array
    #dot: x, y, tile
    dot: chex.Array


class AdventureObservation(struct.PyTreeNode):
    player: ObjectObservation
    dragon_yellow: ObjectObservation
    dragon_green: ObjectObservation
    key_yellow: ObjectObservation
    key_black: ObjectObservation
    gate_yellow: ObjectObservation
    gate_black: ObjectObservation
    sword: ObjectObservation
    bridge: ObjectObservation
    magnet: ObjectObservation
    chalice: ObjectObservation
    dragon_red: ObjectObservation
    key_white: ObjectObservation
    gate_white: ObjectObservation
    bat: ObjectObservation
    dot: ObjectObservation


class AdventureInfo(struct.PyTreeNode):
    time: jnp.ndarray


def _load_background_map(path: str) -> jnp.ndarray:
    background_map = jnp.load(path)
    return background_map

class JaxAdventure(JaxEnvironment[AdventureState, AdventureObservation, AdventureInfo, AdventureConstants]):
    def __init__(self, consts: AdventureConstants = None):
        consts = consts or AdventureConstants()
        super().__init__(consts)
        self.renderer = AdventureRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
        ]

        #jax.debug.print("base dir:{a}", a=render_utils.get_base_sprite_dir())
        #jax.debug.print("path:{a}", a=os.path.join(render_utils.get_base_sprite_dir(), "adventure", "Room_2.npy"))
        #jax.debug.print("sprite path: {a}", a=f"{os.path.dirname(os.path.abspath(__file__))}/sprites/adventure")
        #background_assets_names = _get_default_asset_config()[0]["files"]
        
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/adventure"
#
        self.BackgroundRoom1 = _load_background_map(os.path.join(sprite_path, "Room_1.npy"))
        self.BackgroundRoom2 = _load_background_map(os.path.join(sprite_path, "Room_2.npy"))
        self.BackgroundRoom3 = _load_background_map(os.path.join(sprite_path, "Room_3.npy"))
        self.BackgroundRoom4 = _load_background_map(os.path.join(sprite_path, "Room_4.npy"))
        self.BackgroundRoom5 = _load_background_map(os.path.join(sprite_path, "Room_5.npy"))
        self.BackgroundRoom6 = _load_background_map(os.path.join(sprite_path, "Room_6.npy"))
        self.BackgroundRoom7 = _load_background_map(os.path.join(sprite_path, "Room_7.npy"))
        self.BackgroundRoom8 = _load_background_map(os.path.join(sprite_path, "Room_8.npy"))
        self.BackgroundRoom9 = _load_background_map(os.path.join(sprite_path, "Room_9.npy"))
        self.BackgroundRoom10 = _load_background_map(os.path.join(sprite_path, "Room_10.npy"))
        self.BackgroundRoom11 = _load_background_map(os.path.join(sprite_path, "Room_11.npy"))
        self.BackgroundRoom12 = _load_background_map(os.path.join(sprite_path, "Room_12.npy"))
        self.BackgroundRoom13 = _load_background_map(os.path.join(sprite_path, "Room_13.npy"))
        self.BackgroundRoom14 = _load_background_map(os.path.join(sprite_path, "Room_14.npy"))

    def _check_for_wall(self, state: AdventureState, direction: int) -> bool:
        room = state.player[2]

        # direction 0: left, 1: right, 2: up, 3: down
        player_x = state.player[0]
        player_x = jax.lax.cond(
            direction == 0,
            lambda x: x-4,
            lambda x: x,
            operand = player_x,
        )
        player_x = jax.lax.cond(
            direction == 1,
            lambda x: x+4,
            lambda x: x,
            operand = player_x,
        )

        player_y = state.player[1]
        player_y = jax.lax.cond(
            direction == 2,
            lambda y: y-8,
            lambda y: y,
            operand = player_y,
        )
        player_y = jax.lax.cond(
            direction == 3,
            lambda y: y+8,
            lambda y: y,
            operand = player_y,
        )

        #jax.debug.print("step")
        #test load background of Rooms
        def is_tile_walkable(tileset: jnp.ndarray, Pos_x: int, Pos_y: int) -> bool:
            #determin if we should be allowed to walk, based on the background only
            #tileset data at a given x and y position is [r, g, b, 255] 
            #[151, 151, 151, 255] = Grey (allowed player movement) 
            #[0, 0, 0, 255] are top or bottom border allow movement for tilechange
            #anything else are walls (inversed in certain maze tileset) .
            is_walkable_1 = (tileset[Pos_y+2,Pos_x][0] == jnp.uint8(151))
            is_walkable_2 = (tileset[Pos_y+2,Pos_x][1] == jnp.uint8(151))
            is_walkable_3 = (tileset[Pos_y+2,Pos_x][2] == jnp.uint8(151))
            is_walkable = jnp.logical_and(is_walkable_1, jnp.logical_and(is_walkable_2,is_walkable_3))
            is_border_1 = (tileset[Pos_y+2,Pos_x][0] == jnp.uint8(0))
            is_border_2 = (tileset[Pos_y+2,Pos_x][1] == jnp.uint8(0))
            is_border_3 = (tileset[Pos_y+2,Pos_x][2] == jnp.uint8(0))
            is_border = jnp.logical_and(is_border_1, jnp.logical_and(is_border_2,is_border_3))
            #jax.debug.print("Tile: {a} is walkable {b}",a=tileset[Pos_y,Pos_x][0:3], b=is_walkable)
            return jnp.logical_or(is_walkable,is_border)            
        
        #jax.debug.print("Room: {a} is equal to 0 {b}, is walkable {c}",a=room, b=(room == 0),c=is_tile_walkable(self.BackgroundRoom1, player_x, player_y))
        in_Room_1_and_walkable = jnp.logical_and(jnp.equal(room, 0), is_tile_walkable(self.BackgroundRoom1, player_x, player_y))
        in_Room_2_and_walkable = jnp.logical_and(jnp.equal(room, 1), is_tile_walkable(self.BackgroundRoom2, player_x, player_y))
        in_Room_3_and_walkable = jnp.logical_and(jnp.equal(room, 2), is_tile_walkable(self.BackgroundRoom3, player_x, player_y))
        in_Room_4_and_walkable = jnp.logical_and(jnp.equal(room, 3), is_tile_walkable(self.BackgroundRoom4, player_x, player_y))
        in_Room_5_and_walkable = jnp.logical_and(jnp.equal(room, 4), is_tile_walkable(self.BackgroundRoom5, player_x, player_y))
        in_Room_6_and_walkable = jnp.logical_and(jnp.equal(room, 5), is_tile_walkable(self.BackgroundRoom6, player_x, player_y))
        in_Room_7_and_walkable = jnp.logical_and(jnp.equal(room, 6), is_tile_walkable(self.BackgroundRoom7, player_x, player_y))
        in_Room_8_and_walkable = jnp.logical_and(jnp.equal(room, 7), is_tile_walkable(self.BackgroundRoom8, player_x, player_y))
        in_Room_9_and_walkable = jnp.logical_and(jnp.equal(room, 8), is_tile_walkable(self.BackgroundRoom9, player_x, player_y))
        in_Room_10_and_walkable = jnp.logical_and(jnp.equal(room, 9), is_tile_walkable(self.BackgroundRoom10, player_x, player_y))
        in_Room_11_and_walkable = jnp.logical_and(jnp.equal(room, 10), is_tile_walkable(self.BackgroundRoom11, player_x, player_y))
        in_Room_12_and_walkable = jnp.logical_and(jnp.equal(room, 11), is_tile_walkable(self.BackgroundRoom12, player_x, player_y))
        in_Room_13_and_walkable = jnp.logical_and(jnp.equal(room, 12), is_tile_walkable(self.BackgroundRoom13, player_x, player_y))
        in_Room_14_and_walkable = jnp.logical_and(jnp.equal(room, 13), is_tile_walkable(self.BackgroundRoom14, player_x, player_y))

        Room_1_or_2_and_walkable = jnp.logical_or(in_Room_1_and_walkable, in_Room_2_and_walkable)
        Room_3_or_4_and_walkable = jnp.logical_or(in_Room_3_and_walkable, in_Room_4_and_walkable)
        Room_5_or_6_and_walkable = jnp.logical_or(in_Room_5_and_walkable, in_Room_6_and_walkable)
        Room_7_or_8_and_walkable = jnp.logical_or(in_Room_7_and_walkable, in_Room_8_and_walkable)
        Room_9_or_10_and_walkable = jnp.logical_or(in_Room_9_and_walkable, in_Room_10_and_walkable)
        Room_11_or_12_and_walkable = jnp.logical_or(in_Room_11_and_walkable, in_Room_12_and_walkable)
        Room_13_or_14_and_walkable = jnp.logical_or(in_Room_13_and_walkable, in_Room_14_and_walkable)

        Room_1_or_2_or_3_or_4_and_walkable = jnp.logical_or(Room_1_or_2_and_walkable, Room_3_or_4_and_walkable)
        Room_5_or_6_or_7_or_8_and_walkable = jnp.logical_or(Room_5_or_6_and_walkable, Room_7_or_8_and_walkable)
        Room_9_or_10_or_11_or_12_and_walkable = jnp.logical_or(Room_9_or_10_and_walkable, Room_11_or_12_and_walkable)

        Room_1_or_2_or_3_or_4_or_5_or_6_or_7_or_8_and_walkable = jnp.logical_or(Room_1_or_2_or_3_or_4_and_walkable, Room_5_or_6_or_7_or_8_and_walkable)
        Room_9_or_10_or_11_or_12_or_13_or_14_and_walkable = jnp.logical_or(Room_9_or_10_or_11_or_12_and_walkable, Room_13_or_14_and_walkable)

        current_Room_is_walkable = jnp.logical_or(Room_1_or_2_or_3_or_4_or_5_or_6_or_7_or_8_and_walkable,Room_9_or_10_or_11_or_12_or_13_or_14_and_walkable)
        #jax.debug.print("is walkable {a}", a= current_Room_is_walkable)


        edge_left = self.consts.PATH_VERTICAL_LEFT
        edge_right = self.consts.PATH_VERTICAL_RIGHT

        edge_left = self.consts.PATH_VERTICAL_LEFT
        edge_right = self.consts.PATH_VERTICAL_RIGHT

        #Castle Collisions
        castle_tower_left = self.consts.CASTLE_TOWER_LEFT_X
        castle_tower_right = self.consts.CASTLE_TOWER_RIGHT_X
        castle_tower_height = self.consts.CASTLE_TOWER_CORNER_Y
        castle_base_left = self.consts.CASTLE_BASE_LEFT_X
        castle_base_right = self.consts.CASTLE_BASE_RIGHT_X
        castle_base_height = self.consts.CASTLE_BASE_CORNER_Y

        castle_towers_out = jnp.logical_or(player_x<=castle_tower_left, player_x>=castle_tower_right)
        castle_towers_in = jnp.logical_and(player_x>=edge_left, player_x<=edge_right)
        castle_towers = jnp.logical_or(player_y >= castle_tower_height, jnp.logical_or(castle_towers_in, castle_towers_out))

        castle_base_out = jnp.logical_or(player_x<=castle_base_left, player_x>=castle_base_right)
        castle_base_in_1 = jnp.logical_and(jnp.logical_and(player_y>= castle_tower_height, player_y <= castle_base_height),jnp.logical_and(player_x>=edge_left+8, player_x<=edge_right-10))
        castle_base_in_2 = jnp.logical_and(player_y <= castle_tower_height, jnp.logical_and(player_x>=edge_left, player_x<=edge_right))
        castle_base_in = jnp.logical_or(castle_base_in_1, castle_base_in_2)
        castle_base = jnp.logical_or(player_y >= castle_base_height, jnp.logical_or(castle_base_in, castle_base_out))

        castle_walls = jnp.logical_and(castle_towers, castle_base)

        ##logic implementation gate border

        gate_yellow_open = state.gate_yellow[0]

        gate_white_open = state.gate_white[0]

        gate_black_open = state.gate_black[0]

        gate_yellow_not_block = jnp.logical_or(
            jnp.logical_not(room == 0),
            gate_yellow_open > 4
        )

        gate_white_not_block = jnp.logical_or(
            jnp.logical_not(room == 24),
            gate_white_open > 4
        )

        gate_black_not_block = jnp.logical_or(
            jnp.logical_not(room == 11),
            gate_black_open > 4
        )

        gates_not_blocking = jnp.logical_and(jnp.logical_and(gate_yellow_not_block, gate_black_not_block),gate_white_not_block)

        castle_gate = jnp.logical_or(
            gates_not_blocking,
            jnp.logical_or(
                jnp.logical_or(
                player_x >= edge_right,
                player_x <= edge_left
            ),
            player_y >= castle_base_height
            )
        )
        
        castle_collision = jnp.logical_or(
            jnp.logical_not(jnp.logical_or(jnp.logical_or(room==0, room==11),room==24)), #either it is not a castle tile, or
            jnp.logical_and(castle_walls, castle_gate)
        )


        walls_detected = jnp.logical_and(current_Room_is_walkable, castle_collision )

        #Check for Bridge negating wall
        
        bridgeX = state.bridge[0]
        bridgeY = state.bridge[1]
        bridgeTile =state.bridge[2]

        bridgeOnSameTile = bridgeTile == room
        bridgeInRange = jnp.logical_and(
            jnp.logical_and(player_x >= bridgeX + 8, player_x <= bridgeX + 24),
            jnp.logical_and(player_y >= bridgeY, player_y <= bridgeY + 48)
        )

        bridge_detected = jnp.logical_and(bridgeOnSameTile, bridgeInRange)


        return_bool = jnp.logical_or(walls_detected, bridge_detected)
        
        return return_bool

    def _player_step(self, state: AdventureState, action: chex.Array) -> AdventureState:
        left = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.LEFT, action == Action.LEFTFIRE),action== Action.UPLEFT),action == Action.UPLEFTFIRE), action==Action.DOWNLEFT), action==Action.DOWNLEFTFIRE)
        right = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.RIGHT, action == Action.RIGHTFIRE),action== Action.UPRIGHT),action == Action.UPRIGHTFIRE), action==Action.DOWNRIGHT), action==Action.DOWNRIGHTFIRE)
        up = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.UP, action == Action.UPFIRE),action== Action.UPRIGHT),action == Action.UPRIGHTFIRE), action==Action.UPLEFT), action==Action.UPLEFTFIRE)
        down = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.DOWN, action == Action.DOWNFIRE),action== Action.DOWNRIGHT),action == Action.DOWNRIGHTFIRE), action==Action.DOWNLEFT), action==Action.DOWNLEFTFIRE)

        #check for no wall before walking
        left_no_wall = jnp.logical_and(left,self._check_for_wall(state, 0))
        right_no_wall = jnp.logical_and(right,self._check_for_wall(state, 1))
        up_no_wall =  jnp.logical_and(up,self._check_for_wall(state, 2))
        down_no_wall =  jnp.logical_and(down,self._check_for_wall(state, 3))
        
        new_step_counter = state.step_counter

        #get x cord of the item beeing held
        new_item_x = jax.lax.switch(
            state.player[3],
            [lambda:0,
             lambda:state.key_yellow[0],
             lambda:state.key_black[0],
             lambda:state.sword[0],
             lambda:state.bridge[0],
             lambda:state.magnet[0],
             lambda:state.chalice[0],
             lambda:state.key_white[0],
             lambda:state.dot[0]
             ]
        )

        new_player_x = state.player[0]
        new_player_x, new_item_x, new_step_counter = jax.lax.cond(
            left_no_wall,
            lambda y: (y[0]-4,y[1]-4,y[2]+1),
            lambda y: y,
            operand = (new_player_x,new_item_x,new_step_counter),
        )
        new_player_x, new_item_x, new_step_counter = jax.lax.cond(
            right_no_wall,
            lambda y: (y[0]+4,y[1]+4,y[2]+1),
            lambda y: y,
            operand = (new_player_x,new_item_x,new_step_counter),
        )

        #get y cord of the item beeing held
        new_item_y = jax.lax.switch(
            state.player[3],
            [lambda:0,
             lambda:state.key_yellow[1],
             lambda:state.key_black[1],
             lambda:state.sword[1],
             lambda:state.bridge[1],
             lambda:state.magnet[1],
             lambda:state.chalice[1],
             lambda:state.key_white[1],
             lambda:state.dot[1]
             ]
        )

        new_player_y = state.player[1]
        new_player_y, new_item_y, new_step_counter = jax.lax.cond(
            down_no_wall,
            lambda y: (y[0]+8,y[1]+8,y[2]+1),
            lambda y: y,
            operand = (new_player_y,new_item_y,new_step_counter)
        )
        new_player_y, new_item_y, new_step_counter = jax.lax.cond(
            up_no_wall,
            lambda y: (y[0]-8,y[1]-8,y[2]+1),
            lambda y: y,
            operand = (new_player_y,new_item_y,new_step_counter)
        )
        new_player_tile = state.player[2]
        new_item_tile = jax.lax.switch(
            state.player[3],
            [lambda:0,
             lambda:state.key_yellow[2],
             lambda:state.key_black[2],
             lambda:state.sword[2],
             lambda:state.bridge[2],
             lambda:state.magnet[2],
             lambda:state.chalice[2],
             lambda:state.key_white[2],
             lambda:state.dot[2]
             ]
        )
        
        #enter yellow castle
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            jnp.logical_and(new_player_tile == 0, jnp.logical_and(new_player_y <148,jnp.logical_and(new_player_x<110, new_player_x>50))),
            lambda: (212, 1, 1,new_item_y+(212-new_player_y)),
            lambda: (new_player_y, new_player_tile, new_item_tile, new_item_y)
        )

        #leave yellow castle
        new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x = jax.lax.cond(
            jnp.logical_and(new_player_tile == 1, new_player_y >212),
            lambda: (77, 148, 0, 0, new_item_y-(new_player_y-148),new_item_x+(77-new_player_x)),
            lambda: (new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x)
        )

        #enter black castle
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            jnp.logical_and(new_player_tile == 11, jnp.logical_and(new_player_y <148,jnp.logical_and(new_player_x<110, new_player_x>50))),
            lambda: (212, 12, 12,new_item_y+(212-new_player_y)),
            lambda: (new_player_y, new_player_tile, new_item_tile, new_item_y)
        )

        #leave black castle
        new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x = jax.lax.cond(
            jnp.logical_and(new_player_tile == 12, new_player_y >212),
            lambda: (77, 148, 11, 11, new_item_y-(new_player_y-148),new_item_x+(77-new_player_x)),
            lambda: (new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x)
        )

        #change of rooms
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            new_player_y > 212,
            lambda _: (27, jax.lax.switch( new_player_tile, [lambda:2,lambda:0,lambda:0, 
                                                             lambda:4, lambda:0, lambda:0, 
                                                             lambda:5, lambda:8, lambda:0, 
                                                             lambda: 6, lambda:7, lambda:10, 
                                                             lambda:11, lambda:12]), 
                                                             jax.lax.switch( new_item_tile, [lambda:2,lambda:0,lambda:0, 
                                                             lambda:4, lambda:0, lambda:0, 
                                                             lambda:5, lambda:8, lambda:0, 
                                                             lambda: 6, lambda:7, lambda:10, 
                                                             lambda:11, lambda:12]), new_item_y-(new_player_y-27)),
            lambda _: (new_player_y, new_player_tile, new_item_tile, new_item_y),
            operand = None,
        )
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            new_player_y < 27,
            lambda _: (212, jax.lax.switch( new_player_tile, [lambda:1,lambda:0,lambda:0, 
                                                              lambda:0, lambda:3, lambda:6, 
                                                              lambda:9, lambda:10, lambda:7, 
                                                              lambda: 0, lambda:11, lambda:12, 
                                                              lambda:13, lambda:0]),
                                                              jax.lax.switch( new_player_tile, [lambda:1,lambda:0,lambda:0, 
                                                              lambda:0, lambda:3, lambda:6, 
                                                              lambda:9, lambda:10, lambda:7, 
                                                              lambda: 0, lambda:11, lambda:12, 
                                                              lambda:13, lambda:0]), new_item_y+(212-new_player_y)),
            lambda _: (new_player_y, new_player_tile, new_item_tile, new_item_y),
            operand = None,
        )
        new_player_x, new_player_tile, new_item_tile, new_item_x = jax.lax.cond(
            new_player_x > 160,
            lambda _: (0, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:3, 
                                                            lambda:0, lambda:0, lambda:2, 
                                                            lambda:7, lambda:6, lambda:10, 
                                                            lambda: 8, lambda:9, lambda:0, 
                                                            lambda:0, lambda:0]),
                                                            jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:3, 
                                                            lambda:0, lambda:0, lambda:2, 
                                                            lambda:7, lambda:6, lambda:10, 
                                                            lambda: 8, lambda:9, lambda:0, 
                                                            lambda:0, lambda:0]), new_item_x-new_player_x),
            lambda _: (new_player_x, new_player_tile, new_item_tile, new_item_x),
            operand = None,
        )
        new_player_x, new_player_tile, new_item_tile, new_item_x = jax.lax.cond(
            new_player_x < 0,
            lambda _: (160, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:5, 
                                                              lambda:2, lambda:0, lambda:0, 
                                                              lambda:7, lambda:6, lambda:9, 
                                                              lambda: 10, lambda:8, lambda:0, 
                                                              lambda:0, lambda:0]),
                                                              jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:5, 
                                                              lambda:2, lambda:0, lambda:0, 
                                                              lambda:7, lambda:6, lambda:9, 
                                                              lambda: 10, lambda:8, lambda:0, 
                                                              lambda:0, lambda:0]), new_item_x+(160-new_player_x)),
            lambda _: (new_player_x, new_player_tile, new_item_tile, new_item_x),
            operand = None,
        )

        return state.replace(
            step_counter = jnp.array(new_step_counter),
            player = jnp.array([new_player_x,new_player_y,new_player_tile,state.player[3]]).astype(jnp.int32), #SEEMS NOT GOOD
            key_yellow = jax.lax.cond(state.player[3]==self.consts.KEY_YELLOW_ID,
                                      lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                      lambda op: op[3],
                                      operand=(new_item_x,new_item_y,new_item_tile,state.key_yellow),
                                      ),
            key_black= jax.lax.cond(state.player[3]==self.consts.KEY_BLACK_ID,
                                    lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                    lambda op: op[3],
                                    operand=(new_item_x,new_item_y,new_item_tile,state.key_black)
                                    ),
            key_white= jax.lax.cond(state.player[3]==self.consts.KEY_WHITE_ID,
                                    lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                    lambda op: op[3],
                                    operand=(new_item_x,new_item_y,new_item_tile,state.key_white)
                                    ),
            sword= jax.lax.cond(state.player[3]==self.consts.SWORD_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.sword)
                                ),
            bridge= jax.lax.cond(state.player[3]==self.consts.BRIDGE_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.bridge)
                                ),
            magnet= jax.lax.cond(state.player[3]==self.consts.MAGNET_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.magnet)
                                ),
            chalice= jax.lax.cond(state.player[3]==self.consts.CHALICE_ID,
                                  lambda op: jnp.array([op[0],op[1],op[2],op[3]]).astype(jnp.int32),
                                  lambda op: op[4],
                                  operand=(new_item_x,new_item_y,new_item_tile,state.chalice[3],state.chalice)
                                  ),
            dot= jax.lax.cond(state.player[3]==self.consts.DOT_ID,
                                    lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                    lambda op: op[3],
                                    operand=(new_item_x,new_item_y,new_item_tile,state.dot)
                                    ),
        )
    
    def _gate_interaction(self, state: AdventureState) -> AdventureState:
        gate_yellow_state = state.gate_yellow[0]
        gate_yellow_close = False
        gate_white_state = state.gate_white[0]
        gate_white_close = False
        gate_black_state = state.gate_black[0]
        gate_black_close = False
        gate_yellow_counter = state.gate_yellow[1]
        gate_white_counter = state.gate_white[1]
        gate_black_counter = state.gate_black[1]

        room = state.player[2]
        player_x = state.player[0]
        player_y = state.player[1]

        yellow_key_in_inventory = (state.player[3] == 1)
        black_key_in_inventory = (state.player[3] == 2)
        white_key_in_inventory = (state.player[3] == 7)

        player_infront_yellow_gate = jnp.logical_and(
            room == 0,
            jnp.logical_and(
                jnp.logical_and(
                    player_x >= self.consts.PATH_VERTICAL_LEFT,
                    player_x <= self.consts.PATH_VERTICAL_RIGHT
                ),jnp.logical_and(
                    player_y >= self.consts.CASTLE_BASE_CORNER_Y,
                    player_y <= self.consts.CASTLE_BASE_CORNER_Y + 8
                )
            )
        )

        player_infront_black_gate = jnp.logical_and(
            room == 11,
            jnp.logical_and(
                jnp.logical_and(
                    player_x >= self.consts.PATH_VERTICAL_LEFT,
                    player_x <= self.consts.PATH_VERTICAL_RIGHT
                ),jnp.logical_and(
                    player_y >= self.consts.CASTLE_BASE_CORNER_Y,
                    player_y <= self.consts.CASTLE_BASE_CORNER_Y + 8
                )
            )
        )

        player_infront_white_gate = jnp.logical_and(
            room == 24,
            jnp.logical_and(
                jnp.logical_and(
                    player_x >= self.consts.PATH_VERTICAL_LEFT,
                    player_x <= self.consts.PATH_VERTICAL_RIGHT
                ),jnp.logical_and(
                    player_y >= self.consts.CASTLE_BASE_CORNER_Y,
                    player_y <= self.consts.CASTLE_BASE_CORNER_Y + 8
                )
            )
        )


        yellow_key_in_range = jnp.logical_and(yellow_key_in_inventory, player_infront_yellow_gate)
        black_key_in_range = jnp.logical_and(black_key_in_inventory, player_infront_black_gate)
        white_key_in_range = jnp.logical_and(white_key_in_inventory, player_infront_white_gate)
        
        gate_opening_yellow = jnp.logical_and(jnp.logical_and(jnp.logical_and(gate_yellow_state>=0, gate_yellow_state<6), yellow_key_in_range), gate_yellow_counter == 0)
        gate_opening_black = jnp.logical_and(jnp.logical_and(jnp.logical_and(gate_black_state>=0, gate_black_state<6), black_key_in_range), gate_black_counter == 0)
        gate_opening_white = jnp.logical_and(jnp.logical_and(jnp.logical_and(gate_white_state>=0, gate_white_state<6), white_key_in_range), gate_white_counter == 0)
    
        gate_yellow_close =jnp.logical_and(jnp.logical_and(gate_yellow_state>0, gate_yellow_counter > 20), yellow_key_in_range)
        gate_black_close = jnp.logical_and(jnp.logical_and(gate_black_state>0, gate_black_counter > 20), black_key_in_range)
        gate_white_close =jnp.logical_and(jnp.logical_and(gate_white_state>0, gate_white_counter > 20), white_key_in_range)

        gate_opening_yellow = jnp.logical_and(gate_opening_yellow, jnp.logical_not(gate_yellow_close))
        gate_opening_black = jnp.logical_and(gate_opening_black, jnp.logical_not(gate_black_close))
        gate_opening_white = jnp.logical_and(gate_opening_white, jnp.logical_not(gate_white_close))
        
        gate_yellow_state = jax.lax.cond(
            gate_opening_yellow,
            lambda op: op + 1,
            lambda op: op,
            operand = gate_yellow_state
        )

        gate_yellow_state = jax.lax.cond(
            gate_yellow_close,
            lambda op: op - 1,
            lambda op:op,
            operand = gate_yellow_state
        )

        gate_black_state = jax.lax.cond(
            gate_opening_black,
            lambda op: op + 1,
            lambda op: op,
            operand = gate_black_state
        )

        gate_black_state = jax.lax.cond(
            gate_black_close,
            lambda op: op - 1,
            lambda op:op,
            operand = gate_black_state
        )

        gate_white_state = jax.lax.cond(
            gate_opening_white,
            lambda op: op + 1,
            lambda op: op,
            operand = gate_white_state
        )

        gate_white_state = jax.lax.cond(
            gate_white_close,
            lambda op: op - 1,
            lambda op:op,
            operand = gate_white_state
        )

        gate_yellow_counter = jax.lax.cond(
            jnp.logical_or(gate_yellow_state == 6, jnp.logical_and(gate_yellow_state==0, gate_yellow_counter<30)),
            lambda op:op + 1,
            lambda op:op,
            operand = gate_yellow_counter
        )

        gate_yellow_counter = jax.lax.cond(
            jnp.logical_and(gate_yellow_state == 0, gate_yellow_counter>=30),
            lambda _: 0,
            lambda op:op,
            operand = gate_yellow_counter
        )

        gate_black_counter = jax.lax.cond(
            jnp.logical_or(gate_black_state == 6, jnp.logical_and(gate_black_state==0, gate_black_counter<30)),
            lambda op:op + 1,
            lambda op:op,
            operand = gate_black_counter
        )

        gate_black_counter = jax.lax.cond(
            jnp.logical_and(gate_black_state == 0, gate_black_counter>=30),
            lambda _: 0,
            lambda op:op,
            operand = gate_black_counter
        )

        gate_white_counter = jax.lax.cond(
            jnp.logical_or(gate_white_state == 6, jnp.logical_and(gate_white_state==0, gate_white_counter<30)),
            lambda op:op + 1,
            lambda op:op,
            operand = gate_white_counter
        )

        gate_white_counter = jax.lax.cond(
            jnp.logical_and(gate_white_state == 0, gate_white_counter>=30),
            lambda _: 0,
            lambda op:op,
            operand = gate_white_counter
        )

        return state.replace(
            gate_yellow=jnp.array([gate_yellow_state, gate_yellow_counter]).astype(jnp.int32),
            gate_black=jnp.array([gate_black_state, gate_black_counter]).astype(jnp.int32),
            gate_white=jnp.array([gate_white_state, gate_white_counter]).astype(jnp.int32),
        )
    
    def _item_pickup(self, state: AdventureState, action: chex.Array) -> AdventureState:

        #helper function that chhecks if an item is near the player
        #same tile, in range of 4 pixels
        #it used the corners of the player and calculates if they are in x and y range to the item corners
        #followed by TRYING to integrate that items are only piced p when walked into. NOT woring, tried, gave up
        #VERY questionalble performance due to hardcoded checks
        def check_for_item(self:JaxAdventure, state: AdventureState, item_ID: int) -> bool:
            item_x, item_y, tile, item_width, item_height = jax.lax.switch(
                item_ID,
                [lambda:(0,0,0,0,0), #this should never occour
                lambda:(state.key_yellow[0],state.key_yellow[1],state.key_yellow[2],self.consts.KEY_SIZE[0],self.consts.KEY_SIZE[1]),
                lambda:(state.key_black[0],state.key_black[1],state.key_black[2],self.consts.KEY_SIZE[0],self.consts.KEY_SIZE[1]),
                lambda:(state.sword[0],state.sword[1],state.sword[2],self.consts.SWORD_SIZE[0],self.consts.SWORD_SIZE[1]),
                lambda:(state.bridge[0],state.bridge[1],state.bridge[2],self.consts.BRIDGE_SIZE[0],self.consts.BRIDGE_SIZE[1]),
                lambda:(state.magnet[0],state.magnet[1],state.magnet[2],self.consts.MAGNET_SIZE[0],self.consts.MAGNET_SIZE[1]),
                lambda:(state.chalice[0],state.chalice[1],state.chalice[2],self.consts.CHALICE_SIZE[0],self.consts.CHALICE_SIZE[1]),
                lambda:(state.key_white[0],state.key_white[1],state.key_white[2],self.consts.KEY_SIZE[0],self.consts.KEY_SIZE[1]),
                lambda:(state.dot[0],state.dot[1],state.dot[2],self.consts.DOT_SIZE[0],self.consts.DOT_SIZE[1])
                ])
            #jax.debug.print("Hitbox values item:{a},{b},{c},{d},{e}",a=item_x,b=item_y,c=tile,d=item_width,e=item_height)
            
            on_same_tile = (tile==state.player[2])
            player_hitbox_nw = (state.player[0],state.player[1])
            player_hitbox_ne = (state.player[0]+self.consts.PLAYER_SIZE[0],state.player[1])
            player_hitbox_se = (state.player[0]+self.consts.PLAYER_SIZE[0],state.player[1]+self.consts.PLAYER_SIZE[1])
            player_hitbox_sw = (state.player[0],state.player[1]+self.consts.PLAYER_SIZE[1])

            #jax.debug.print("Hitbox values Player:{a},{b}|{c},{d}|{e},{f}|{g},{h}",
            #                a=player_hitbox_nw[0],b=player_hitbox_nw[1],
            #                c=player_hitbox_ne[0],d=player_hitbox_ne[1],
            #                e=player_hitbox_se[0],f=player_hitbox_se[1],
            #                g=player_hitbox_sw[0],h=player_hitbox_sw[1])
            
            
            #change the waling and FIRE actions to ony walking actions for simplicity 
            walk_direction = jax.lax.switch(
                action,
                [lambda:Action.NOOP, #this should never occour
                lambda:Action.FIRE,
                lambda:Action.UP,
                lambda:Action.RIGHT,
                lambda:Action.LEFT,
                lambda:Action.DOWN,
                lambda:Action.UPRIGHT,
                lambda:Action.UPLEFT,
                lambda:Action.DOWNRIGHT,
                lambda:Action.DOWNLEFT,
                lambda:Action.UP,           #UPFIRE,
                lambda:Action.RIGHT,        #RIGHTFIRE,
                lambda:Action.LEFT,         #LEFTFIRE....etc
                lambda:Action.DOWN,
                lambda:Action.UPRIGHT,
                lambda:Action.UPLEFT,
                lambda:Action.DOWNRIGHT,
                lambda:Action.DOWNLEFT
                ]
            )
            
            def diff_of_4(val1:int, val2:int) -> bool:
                return ((val1 - val2) < 4)

            nw_close_in_x = jnp.logical_and(diff_of_4(item_x,player_hitbox_nw[0]),diff_of_4(player_hitbox_nw[0],(item_x+item_width-1)))
            nw_close_in_y = jnp.logical_and(diff_of_4(item_y,player_hitbox_nw[1]),diff_of_4(player_hitbox_nw[1],(item_y+item_height-1)))
            nw_close = jnp.logical_and(nw_close_in_x,nw_close_in_y)

            ne_close_in_x = jnp.logical_and(diff_of_4(item_x,player_hitbox_ne[0]),diff_of_4(player_hitbox_ne[0],(item_x+item_width-1)))
            ne_close_in_y = jnp.logical_and(diff_of_4(item_y,player_hitbox_ne[1]),diff_of_4(player_hitbox_ne[1],(item_y+item_height-1)))
            ne_close = jnp.logical_and(ne_close_in_x,ne_close_in_y)

            se_close_in_x = jnp.logical_and(diff_of_4(item_x,player_hitbox_se[0]),diff_of_4(player_hitbox_se[0],(item_x+item_width-1)))
            se_close_in_y = jnp.logical_and(diff_of_4(item_y,player_hitbox_se[1]),diff_of_4(player_hitbox_se[1],(item_y+item_height-1)))
            se_close = jnp.logical_and(se_close_in_x,se_close_in_y)

            sw_close_in_x = jnp.logical_and(diff_of_4(item_x,player_hitbox_sw[0]),diff_of_4(player_hitbox_sw[0],(item_x+item_width)))
            sw_close_in_y = jnp.logical_and(diff_of_4(item_y,player_hitbox_sw[1]),diff_of_4(player_hitbox_sw[1],(item_y+item_height)))
            sw_close = jnp.logical_and(sw_close_in_x,sw_close_in_y)   
            
            #player is north to the item
            player_north = jnp.logical_and(jnp.logical_or(sw_close,se_close),
                                           jnp.logical_or(nw_close_in_x,ne_close_in_x))
            player_north_walks_south = jnp.logical_and(player_north, 
                                                       jnp.logical_or(walk_direction==Action.DOWN,
                                                                      jnp.logical_or(walk_direction==Action.DOWNLEFT,
                                                                                     walk_direction==Action.DOWNRIGHT)))
            #player is north-east to the item

            #player is east to the item
            player_east = jnp.logical_and(jnp.logical_or(nw_close,sw_close),
                                           jnp.logical_or(ne_close_in_y,se_close_in_y))
            player_east_walks_west = jnp.logical_and(player_east,
                                                     jnp.logical_or(walk_direction==Action.LEFT,
                                                                    jnp.logical_or(walk_direction==Action.DOWNLEFT,
                                                                                   walk_direction==Action.UPLEFT)))
            #player is south-east to the item

            #player is south to the item
            player_south = jnp.logical_and(jnp.logical_or(nw_close,ne_close),
                                           jnp.logical_or(sw_close_in_x,se_close_in_x))
            player_south_walks_north = jnp.logical_and(player_south,
                                                      jnp.logical_or(walk_direction==Action.UP,
                                                                     jnp.logical_or(walk_direction==Action.UPLEFT,
                                                                                    walk_direction==Action.UPRIGHT)))
            #player is south-west to the item

            #player is west to the item
            player_west = jnp.logical_and(jnp.logical_or(ne_close,se_close),
                                           jnp.logical_or(nw_close_in_y,sw_close_in_y))
            player_west_walks_east = jnp.logical_and(player_west,
                                                     jnp.logical_or(walk_direction==Action.RIGHT,
                                                                    jnp.logical_or(walk_direction==Action.DOWNRIGHT,
                                                                                   walk_direction==Action.UPRIGHT)))
            #player is north-west to the item

            #jax.debug.print("Walking Direction: {a},{b},{c},{d},{e}",
            #                a=walk_direction,
            #                b=player_north_walks_south,
            #                c=player_east_walks_west,
            #                d=player_south_walks_north,
            #                e=player_west_walks_east)
            #item is on the same tile and is being approached from the correct side
            item_touches = jnp.logical_and(on_same_tile,
                                           jnp.logical_or(jnp.logical_or(player_north_walks_south,
                                                                         player_east_walks_west),
                                                            jnp.logical_or(player_south_walks_north,
                                                                           player_west_walks_east)))
            #jax.debug.print("Logical values: nw:{a},{b},ne:{c},{d},se:{e},{f},sw:{g},{h}",
            #                a=nw_close_in_x,
            #                b=nw_close_in_y,
            #                c=ne_close_in_x,
            #                d=ne_close_in_y,
            #                e=se_close_in_x,
            #                f=se_close_in_y,
            #                g=sw_close_in_x,
            #                h=sw_close_in_y)
            #jax.debug.print("Logical values: {a},{b},{c},{d},{e}",a=on_same_tile,b=nw_touches_item,c=sw_touches_item,d=ne_touches_item,e=se_touches_item)
            return item_touches

        #HOLY ASS, this is a sin
        #check if the player is moving (not action NOOP)
        #it checks for every item if it is near the player
        #if that is the case it puts it in the Player inventory
        new_player_inventory = jax.lax.cond(
            action == Action.NOOP,
            lambda op: op,
            lambda _: jax.lax.cond(
                check_for_item(self=self, state=state, item_ID=self.consts.KEY_YELLOW_ID),
                lambda _: self.consts.KEY_YELLOW_ID, 
                lambda _: jax.lax.cond(
                    check_for_item(self=self, state=state, item_ID=self.consts.KEY_BLACK_ID),
                    lambda _: self.consts.KEY_BLACK_ID, 
                    lambda _: jax.lax.cond(
                        check_for_item(self=self, state=state, item_ID=self.consts.SWORD_ID),
                        lambda _: self.consts.SWORD_ID, 
                        lambda _: jax.lax.cond(
                            check_for_item(self=self, state=state, item_ID=self.consts.BRIDGE_ID),
                            lambda _: self.consts.BRIDGE_ID, 
                            lambda _: jax.lax.cond(
                                check_for_item(self=self, state=state, item_ID=self.consts.MAGNET_ID),
                                lambda _: self.consts.MAGNET_ID, 
                                lambda _: jax.lax.cond(
                                    check_for_item(self=self, state=state, item_ID=self.consts.CHALICE_ID),
                                    lambda _: self.consts.CHALICE_ID, 
                                    lambda _: jax.lax.cond(
                                        check_for_item(self=self, state=state, item_ID=self.consts.KEY_WHITE_ID),
                                        lambda _: self.consts.KEY_WHITE_ID, 
                                        lambda _: jax.lax.cond(
                                            check_for_item(self=self, state=state, item_ID=self.consts.DOT_ID),
                                            lambda _: self.consts.DOT_ID, 
                                            lambda op: op,
                                            operand=state.player[3]
                                        ),
                                        operand=state.player[3]
                                    ),
                                    operand=state.player[3]
                                ),
                                operand=state.player[3]
                            ),
                            operand=state.player[3]
                        ),
                        operand=state.player[3]
                    ),
                    operand=state.player[3]
                ),
                operand=state.player[3]
            ),
            operand=state.player[3]
        )
        

        return state.replace(
            player = jnp.array([state.player[0],state.player[1],state.player[2],new_player_inventory]).astype(jnp.int32)
        )
    
    def _item_drop(self, state: AdventureState, action: chex.Array) -> AdventureState:
        #check if the Action Fire is used (ToDo, for all Fire actions?)
        #set the player inventory to EMPTY_HAND_ID
        new_player_inventory = jax.lax.cond(
            action == Action.FIRE,
            lambda _: self.consts.EMPTY_HAND_ID,
            lambda op: op,
            operand=state.player[3]
        )

        return state.replace(
            player = jnp.array([state.player[0],state.player[1],state.player[2],new_player_inventory]).astype(jnp.int32)
        )
        

    def _dragon_step(self, state: AdventureState) -> AdventureState:
        speed = self.consts.DRAGON_SPEED

        #get sword position to kill dragons
        sword_x = state.sword[0]
        sword_y = state.sword[1]
        sword_room = state.sword[2]

        #yellow dragon
        direction_x = jnp.sign(state.player[0] - state.dragon_yellow[0])
        direction_y = jnp.sign(state.player[1]- state.dragon_yellow[1])
        dragon_yellow_x = state.dragon_yellow[0]
        dragon_yellow_y = state.dragon_yellow[1]
        dragon_yellow_tile = state.dragon_yellow[2]
        dragon_yellow_animation = state.dragon_yellow[3]
        dragon_yellow_counter = state.dragon_yellow[4]
        dragon_yellow_activate = state.dragon_yellow[6]

        # wait after attack
        dragon_yellow_counter = jax.lax.cond(
            dragon_yellow_animation == 1,
            lambda f: f+1,
            lambda f:f,
            operand = dragon_yellow_counter
        )
        dragon_yellow_freeze = dragon_yellow_counter % self.consts.DRAGON_BITE_TIMER != 0
    
        #dragon starts looking for plyer room after first encounter
        dragon_yellow_activate = jax.lax.cond(state.player[2] == dragon_yellow_tile, lambda:1, lambda: dragon_yellow_activate)
        rndKey, subkey = jax.random.split(state.rndKey)
        dragon_yellow_x, dragon_yellow_y, dragon_yellow_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_yellow_tile != state.player[2], jnp.logical_not(dragon_yellow_freeze)),dragon_yellow_activate==1),
            lambda: (jax.lax.cond(dragon_yellow_x>156, lambda:4, lambda:dragon_yellow_x +2), 
                     jax.lax.cond(dragon_yellow_y>208, lambda:4, lambda:dragon_yellow_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_yellow_x>156,dragon_yellow_y>208), lambda:jax.random.randint(subkey, (), 0, 13) , lambda:dragon_yellow_tile)),
            lambda:(dragon_yellow_x, dragon_yellow_y, dragon_yellow_tile)
        )

        #dragon eats player
        dragon_yellow_eat = jax.lax.cond(
            jnp.logical_and(jnp.logical_and((state.player[0]-dragon_yellow_x)*direction_x<4,(state.player[1]-dragon_yellow_y)*direction_y<4),jnp.logical_and(dragon_yellow_animation==1,jnp.logical_not(dragon_yellow_freeze))),
            lambda:1,
            lambda:0
        )

        #move towards player and attack
        dragon_yellow_x, dragon_yellow_y, dragon_yellow_animation, dragon_yellow_counter= jax.lax.cond(
            jnp.logical_and(state.player[2]==dragon_yellow_tile,jnp.logical_not(dragon_yellow_freeze)),
            lambda _: (dragon_yellow_x + direction_x*speed, dragon_yellow_y + direction_y*speed, jax.lax.cond(
                jnp.logical_and((state.player[0]-dragon_yellow_x)*direction_x<4,(state.player[1]-dragon_yellow_y)*direction_y<4),
                lambda _:jax.lax.cond(dragon_yellow_animation==2, lambda _:2, lambda _:1, operand = None),
                lambda _:jax.lax.cond(dragon_yellow_animation==2, lambda _:2, lambda _:0, operand = None),
                operand = None
            ),0),
            lambda _: (dragon_yellow_x, dragon_yellow_y, jax.lax.cond(jnp.logical_not(dragon_yellow_freeze), lambda _: jax.lax.cond(dragon_yellow_animation==2, lambda _:2, lambda _:0, operand = None), lambda _: jax.lax.cond(dragon_yellow_animation==2, lambda _:2, lambda _:1, operand = None), operand = None), dragon_yellow_counter),
            operand  = None
        )

        #kill dragon
        direction_x = jnp.sign(sword_x - state.dragon_yellow[0])
        direction_y = jnp.sign(sword_y- state.dragon_yellow[1])
        dragon_yellow_animation = jax.lax.cond(
            jnp.logical_and(dragon_yellow_tile==sword_room, jnp.logical_and((sword_x-dragon_yellow_x)*direction_x<4, (sword_y-dragon_yellow_y)*direction_y<22)),
            lambda _:2,
            lambda a:a,
            operand= dragon_yellow_animation
        )

        # dont ever move again when dead
        dragon_yellow_counter = jax.lax.cond(
            dragon_yellow_animation == 2,
            lambda _: 1,
            lambda f:f,
            operand=dragon_yellow_counter
        )


        #green dragon
        direction_x = jnp.sign(state.player[0] - state.dragon_green[0])
        direction_y = jnp.sign(state.player[1]- state.dragon_green[1])
        dragon_green_x = state.dragon_green[0]
        dragon_green_y = state.dragon_green[1]
        dragon_green_tile = state.dragon_green[2]
        dragon_green_animation = state.dragon_green[3]
        dragon_green_counter = state.dragon_green[4]
        dragon_green_activate = state.dragon_green[6]

        # wait after attack
        dragon_green_counter = jax.lax.cond(
            dragon_green_animation == 1,
            lambda f: f+1,
            lambda f:f,
            operand = dragon_green_counter
        )
        dragon_green_freeze = dragon_green_counter % self.consts.DRAGON_BITE_TIMER != 0

        #dragon starts looking for plyer room after first encounter
        dragon_green_activate = jax.lax.cond(state.player[2] == dragon_green_tile, lambda:1, lambda: dragon_green_activate)
        rndKey, subkey = jax.random.split(rndKey)
        dragon_green_x, dragon_green_y, dragon_green_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_green_tile != state.player[2], jnp.logical_not(dragon_green_freeze)),dragon_green_activate==1),
            lambda: (jax.lax.cond(dragon_green_x>156, lambda:4, lambda:dragon_green_x +2), 
                     jax.lax.cond(dragon_green_y>208, lambda:4, lambda:dragon_green_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_green_x>156,dragon_green_y>208), lambda:jax.random.randint(subkey, (), 0, 13) , lambda:dragon_green_tile)),
            lambda:(dragon_green_x, dragon_green_y, dragon_green_tile)
        )

        #dragon eats player
        dragon_green_eat = jax.lax.cond(
            jnp.logical_and(jnp.logical_and((state.player[0]-dragon_green_x)*direction_x<4,(state.player[1]-dragon_green_y)*direction_y<4),jnp.logical_and(dragon_green_animation==1,jnp.logical_not(dragon_green_freeze))),
            lambda:1,
            lambda:0
        )

        #move towards player and attack
        dragon_green_x, dragon_green_y, dragon_green_animation, dragon_green_counter= jax.lax.cond(
            jnp.logical_and(state.player[2]==dragon_green_tile,jnp.logical_not(dragon_green_freeze)),
            lambda _: (dragon_green_x + direction_x*speed, dragon_green_y + direction_y*speed, jax.lax.cond(
                jnp.logical_and(jnp.logical_and((state.player[0]-dragon_green_x)*direction_x<4,(state.player[1]-dragon_green_y)*direction_y<4),dragon_green_animation!=2),
                lambda _: jax.lax.cond(dragon_green_animation==2, lambda _:2, lambda _:1, operand = None),
                lambda _: jax.lax.cond(dragon_green_animation==2, lambda _:2, lambda _:0, operand = None),
                operand = None
            ),0),
            lambda _: (dragon_green_x, dragon_green_y, jax.lax.cond(jnp.logical_not(dragon_green_freeze), lambda _: jax.lax.cond(dragon_green_animation==2, lambda _:2, lambda _:0, operand = None), lambda _: jax.lax.cond(dragon_green_animation==2, lambda _:2, lambda _:1, operand = None), operand = None), dragon_green_counter),
            operand  = None
        )

        #kill dragon
        direction_x = jnp.sign(sword_x - state.dragon_green[0])
        direction_y = jnp.sign(sword_y- state.dragon_green[1])
        dragon_green_animation = jax.lax.cond(
            jnp.logical_and(dragon_green_tile==sword_room, jnp.logical_and((sword_x-dragon_green_x)*direction_x<4, (sword_y-dragon_green_y)*direction_y<22)),
            lambda _:2,
            lambda a:a,
            operand = dragon_green_animation
        )

        # dont ever move again when dead
        dragon_green_counter = jax.lax.cond(
            dragon_green_animation == 2,
            lambda _: 1,
            lambda f:f,
            operand=dragon_green_counter
        )


        return state.replace(
            dragon_yellow = jnp.array([dragon_yellow_x,dragon_yellow_y,dragon_yellow_tile,dragon_yellow_animation,dragon_yellow_counter,dragon_yellow_eat, dragon_yellow_activate]).astype(jnp.int32),
            dragon_green = jnp.array([dragon_green_x,dragon_green_y,dragon_green_tile,dragon_green_animation,dragon_green_counter,dragon_green_eat, dragon_green_activate]).astype(jnp.int32),
            rndKey=rndKey
        )

    def _magnet_step(self, state: AdventureState) -> AdventureState:
        magnet_x = state.magnet[0]
        magnet_y = state.magnet[1]

        #try to pull sword
        sword_x = state.sword[0]
        sword_y = state.sword[1]
        direction_x = jnp.sign(magnet_x - sword_x)
        direction_y = jnp.sign(magnet_y - sword_y)
        sword_x, sword_y = jax.lax.cond(
            jnp.logical_and(state.sword[2]==state.magnet[2], state.player[3]!=3),
            lambda _: (sword_x+direction_x,sword_y+direction_y),
            lambda _: (sword_x,sword_y), 
            operand = None
        )

        #try to pull yellow key
        key_yellow_x = state.key_yellow[0]
        key_yellow_y = state.key_yellow[1]
        direction_x = jnp.sign(magnet_x - key_yellow_x)
        direction_y = jnp.sign(magnet_y - key_yellow_y)
        key_yellow_x, key_yellow_y = jax.lax.cond(
            jnp.logical_and(state.key_yellow[2]==state.magnet[2], state.player[3]!=1),
            lambda _: (key_yellow_x+direction_x,key_yellow_y+direction_y),
            lambda _: (key_yellow_x,key_yellow_y), 
            operand = None
        )

        #try to pull black key
        key_black_x = state.key_black[0]
        key_black_y = state.key_black[1]
        direction_x = jnp.sign(magnet_x - key_black_x)
        direction_y = jnp.sign(magnet_y - key_black_y)
        key_black_x, key_black_y = jax.lax.cond(
            jnp.logical_and(state.key_black[2]==state.magnet[2], state.player[3]!=2),
            lambda _: (key_black_x+direction_x,key_black_y+direction_y),
            lambda _: (key_black_x,key_black_y), 
            operand = None
        )

        #try to pull white key
        key_white_x = state.key_white[0]
        key_white_y = state.key_white[1]
        direction_x = jnp.sign(magnet_x - key_white_x)
        direction_y = jnp.sign(magnet_y - key_white_y)
        key_white_x, key_white_y = jax.lax.cond(
            jnp.logical_and(state.key_white[2]==state.magnet[2], state.player[3]!=7),
            lambda _: (key_white_x+direction_x,key_white_y+direction_y),
            lambda _: (key_white_x,key_white_y), 
            operand = None
        )

        #try to pull bridge
        bridge_x = state.bridge[0]
        bridge_y = state.bridge[1]
        direction_x = jnp.sign(magnet_x - bridge_x)
        direction_y = jnp.sign(magnet_y - bridge_y)
        bridge_x, bridge_y = jax.lax.cond(
            jnp.logical_and(state.bridge[2]==state.magnet[2], state.player[3]!=4),
            lambda _: (bridge_x+direction_x,bridge_y+direction_y),
            lambda _: (bridge_x,bridge_y), 
            operand = None
        )

        #try to pull chalice
        chalice_x = state.chalice[0]
        chalice_y = state.chalice[1]
        direction_x = jnp.sign(magnet_x - chalice_x)
        direction_y = jnp.sign(magnet_y - chalice_y)
        chalice_x, chalice_y = jax.lax.cond(
            jnp.logical_and(state.chalice[2]==state.magnet[2], state.player[3]!=6),
            lambda _: (chalice_x+direction_x,chalice_y+direction_y),
            lambda _: (chalice_x,chalice_y), 
            operand = None
        )

        return state.replace(
            key_yellow=jnp.array([key_yellow_x,key_yellow_y,state.key_yellow[2]]).astype(jnp.int32),
            key_black=jnp.array([key_black_x,key_black_y,state.key_black[2]]).astype(jnp.int32),
            key_white=jnp.array([key_white_x,key_white_y,state.key_white[2]]).astype(jnp.int32),
            sword=jnp.array([sword_x,sword_y,state.sword[2]]).astype(jnp.int32),
            bridge=jnp.array([bridge_x,bridge_y,state.bridge[2]]).astype(jnp.int32),
            chalice=jnp.array([chalice_x,chalice_y,state.chalice[2],state.chalice[3]]).astype(jnp.int32)
        )
    
    
    def _chalice_step(self, state:AdventureState) -> AdventureState:
        
        chalice_color=state.chalice[3]
        chalice_color = (chalice_color +1) % 10

        return state.replace(
            chalice=jnp.array([state.chalice[0],state.chalice[1],state.chalice[2],chalice_color]).astype(jnp.int32)
        )
    
    """This function is called when the game starts and when it is reseted
    It initializes the Adventure state, for the most part these Values are pulled from the consts"""
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[AdventureObservation, AdventureState]:

        state_key, _step_key = jax.random.split(key)
        state = AdventureState(
            step_counter = jnp.array(0).astype(jnp.int32),
            #Player Spawn: x, y, tile, inventory
            player = jnp.array([self.consts.PLAYER_SPAWN[0],
                                self.consts.PLAYER_SPAWN[1],
                                self.consts.PLAYER_SPAWN[2],
                                self.consts.EMPTY_HAND_ID]).astype(jnp.int32),
            #Dragons: x, y ,tile ,state(neutral,dead,atacking), counter( ToDo for?? )
            dragon_yellow = jnp.array([self.consts.DRAGON_YELLOW_SPAWN[0],
                                       self.consts.DRAGON_YELLOW_SPAWN[1],
                                       self.consts.DRAGON_YELLOW_SPAWN[2],
                                       self.consts.DRAGON_YELLOW_SPAWN[3],
                                       self.consts.DRAGON_YELLOW_SPAWN[4],
                                       self.consts.DRAGON_YELLOW_SPAWN[5],
                                       self.consts.DRAGON_YELLOW_SPAWN[6]]).astype(jnp.int32), #ToDo
            dragon_green = jnp.array([self.consts.DRAGON_GREEN_SPAWN[0],
                                      self.consts.DRAGON_GREEN_SPAWN[1],
                                      self.consts.DRAGON_GREEN_SPAWN[2],
                                      self.consts.DRAGON_GREEN_SPAWN[3],
                                      self.consts.DRAGON_GREEN_SPAWN[4],
                                      self.consts.DRAGON_GREEN_SPAWN[5],
                                      self.consts.DRAGON_GREEN_SPAWN[6]]).astype(jnp.int32),
            dragon_red = jnp.array([self.consts.DRAGON_RED_SPAWN[0],
                                      self.consts.DRAGON_RED_SPAWN[1],
                                      self.consts.DRAGON_RED_SPAWN[2],
                                      self.consts.DRAGON_RED_SPAWN[3],
                                      self.consts.DRAGON_RED_SPAWN[4],
                                      self.consts.DRAGON_RED_SPAWN[5],
                                      self.consts.DRAGON_RED_SPAWN[6]]).astype(jnp.int32),
            #Keys: x ,y, tile
            key_yellow = jnp.array([self.consts.KEY_YELLOW_SPAWN[0],
                                    self.consts.KEY_YELLOW_SPAWN[1],
                                    self.consts.KEY_YELLOW_SPAWN[2]]).astype(jnp.int32),
            key_black = jnp.array([self.consts.KEY_BLACK_SPAWN[0],
                                    self.consts.KEY_BLACK_SPAWN[1],
                                    self.consts.KEY_BLACK_SPAWN[2]]).astype(jnp.int32),
            key_white = jnp.array([self.consts.KEY_WHITE_SPAWN[0],
                                    self.consts.KEY_WHITE_SPAWN[1],
                                    self.consts.KEY_WHITE_SPAWN[2]]).astype(jnp.int32),
            #Gate: state, counter (ToDo for animation?)
            gate_yellow=jnp.array([self.consts.GATE_SPAWN[0],
                                  self.consts.GATE_SPAWN[1]]).astype(jnp.int32),
            gate_black=jnp.array([self.consts.GATE_SPAWN[0],
                                  self.consts.GATE_SPAWN[1]]).astype(jnp.int32),
            gate_white=jnp.array([self.consts.GATE_SPAWN[0],
                                  self.consts.GATE_SPAWN[1]]).astype(jnp.int32),
            #Items: x, y, tile
            sword = jnp.array([self.consts.SWORD_SPAWN[0],
                               self.consts.SWORD_SPAWN[1],
                               self.consts.SWORD_SPAWN[2]]).astype(jnp.int32), #ToDo
            bridge = jnp.array([self.consts.BRIDGE_SPAWN[0],
                               self.consts.BRIDGE_SPAWN[1],
                               self.consts.BRIDGE_SPAWN[2]]).astype(jnp.int32), #ToDo
            magnet= jnp.array([self.consts.MAGNET_SPAWN[0],
                               self.consts.MAGNET_SPAWN[1],
                               self.consts.MAGNET_SPAWN[2]]).astype(jnp.int32), #ToDo
            #Chalice: x, y, tile, color (ToDo move color to constants)
            chalice = jnp.array([self.consts.CHALICE_SPAWN[0],
                                 self.consts.CHALICE_SPAWN[1],
                                 self.consts.CHALICE_SPAWN[2],
                                 self.consts.CHALICE_SPAWN[3]]).astype(jnp.int32), #ToDo
            #random key
            rndKey = state_key,
            bat = jnp.array([self.consts.BAT_SPAWN[0],
                                      self.consts.BAT_SPAWN[1],
                                      self.consts.BAT_SPAWN[2],
                                      self.consts.BAT_SPAWN[3]]).astype(jnp.int32),
            dot = jnp.array([self.consts.DOT_SPAWN[0],
                                    self.consts.DOT_SPAWN[1],
                                    self.consts.DOT_SPAWN[2]]).astype(jnp.int32)
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    #this is the ??main loop??, it will go throught all called steps, that change the Adventure state
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AdventureState, action: chex.Array) -> Tuple[AdventureObservation, AdventureState, float, bool, AdventureInfo]:
        # Split step key from state and keep a new key for the next state
        previous_state = state
        # Make per-step key available to helpers that may read state.key
        state = AdventureState(
            step_counter=state.step_counter,
            player=state.player,
            dragon_yellow=state.dragon_yellow,
            dragon_green=state.dragon_green,
            dragon_red=state.dragon_red,
            key_yellow=state.key_yellow,
            key_black=state.key_black,
            key_white=state.key_white,
            gate_yellow=state.gate_yellow,
            gate_black=state.gate_black,
            gate_white=state.gate_white,
            sword=state.sword,
            bridge=state.bridge,
            magnet=state.magnet,
            chalice=state.chalice,
            rndKey=state.rndKey,
            bat=state.bat,
            dot=state.dot
        )
        state = self._player_step(state, action)
        state = self._item_pickup(state, action)
        state = self._item_drop(state, action)
        state = self._dragon_step(state)
        state = self._gate_interaction(state)
        state = self._magnet_step(state)
        state = self._chalice_step(state)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: AdventureState) -> jnp.ndarray:
        return self.renderer.render(state)

    #ToDo, done for all movable entities, why, no clue
    def _get_observation(self, state: AdventureState) -> AdventureObservation:
        player = ObjectObservation.create(
            x=state.player[0],
            y=state.player[1],
            width=self.consts.PLAYER_SIZE[0], 
            height=self.consts.PLAYER_SIZE[1], 
            state=state.player[4]
        )
        dragon_yellow = ObjectObservation.create(
            x=state.dragon_yellow[0],
            y=state.dragon_yellow[1],
            active=state.dragon_yellow[2]==state.player[2],
            width=self.consts.DRAGON_SIZE[0], 
            height=self.consts.DRAGON_SIZE[1], 
            state=state.dragon_yellow[3]
        )
        dragon_green = ObjectObservation.create(
            x=state.dragon_green[0],
            y=state.dragon_green[1],
            active=state.dragon_green[2]==state.player[2],
            width=self.consts.DRAGON_SIZE[0], 
            height=self.consts.DRAGON_SIZE[1], 
            state=state.dragon_green[3]
        )
        key_yellow = ObjectObservation.create(
            x=state.key_yellow[0],
            y=state.key_yellow[1],
            active=state.key_yellow[2]==state.player[2],
            width=self.consts.KEY_SIZE[0], 
            height=self.consts.KEY_SIZE[1]
        )
        key_black = ObjectObservation.create(
            x=state.key_black[0],
            y=state.key_black[1],
            active=state.key_black[2]==state.player[2],
            width=self.consts.KEY_SIZE[0], 
            height=self.consts.KEY_SIZE[1],
        )
        gate_yellow = ObjectObservation.create(
            x=self.consts.YELLOW_GATE_POS[0],
            y=self.consts.YELLOW_GATE_POS[1],
            active=self.consts.YELLOW_GATE_POS[2]==state.player[2],
            width=self.consts.GATE_SIZE[0], 
            height=self.consts.GATE_SIZE[1], 
            state=state.gate_yellow[0]
        )
        gate_black = ObjectObservation.create(
            x=self.consts.BLACK_GATE_POS[0],
            y=self.consts.BLACK_GATE_POS[1],
            active=self.consts.BLACK_GATE_POS[2]==state.player[2],
            width=self.consts.GATE_SIZE[0], 
            height=self.consts.GATE_SIZE[1], 
            state=state.gate_black[0]
        )
        sword = ObjectObservation.create(
            x=state.sword[0],
            y=state.sword[1],
            active=state.sword[2]==state.player[2],
            width=self.consts.SWORD_SIZE[0], 
            height=self.consts.SWORD_SIZE[1]
        )
        bridge = ObjectObservation.create(
            x=state.bridge[0],
            y=state.bridge[1],
            active=state.bridge[2]==state.player[2],
            width=self.consts.BRIDGE_SIZE[0], 
            height=self.consts.BRIDGE_SIZE[1]
        )
        magnet = ObjectObservation.create(
            x=state.magnet[0],
            y=state.magnet[1],
            active=state.magnet[2]==state.player[2],
            width=self.consts.MAGNET_SIZE[0], 
            height=self.consts.MAGNET_SIZE[1]
        )
        chalice = ObjectObservation.create(
            x=state.chalice[0],
            y=state.chalice[1],
            active=state.chalice[2]==state.player[2],
            width=self.consts.CHALICE_SIZE[0], 
            height=self.consts.CHALICE_SIZE[1]
        )
        dragon_red = ObjectObservation.create(
            x=state.dragon_red[0],
            y=state.dragon_red[1],
            active=state.dragon_red[2]==state.player[2],
            width=self.consts.DRAGON_SIZE[0], 
            height=self.consts.DRAGON_SIZE[1], 
            state=state.dragon_red[3]
        )
        key_white = ObjectObservation.create(
            x=state.key_white[0],
            y=state.key_white[1],
            active=state.key_white[2]==state.player[2],
            width=self.consts.KEY_SIZE[0], 
            height=self.consts.KEY_SIZE[1]
        )
        gate_white = ObjectObservation.create(
            x=self.consts.WHITE_GATE_POS[0],
            y=self.consts.WHITE_GATE_POS[1],
            active=self.consts.WHITE_GATE_POS[2]==state.player[2],
            width=self.consts.GATE_SIZE[0], 
            height=self.consts.GATE_SIZE[1], 
            state=state.gate_white[0]
        )
        bat = ObjectObservation.create(
            x=state.bat[0],
            y=state.bat[1],
            active=state.bat[2]==state.player[2],
            width=self.consts.DOT_SIZE[0],
            height=self.consts.DOT_SIZE[1]
        )
        dot = ObjectObservation.create(
            x=state.dot[0],
            y=state.dot[1],
            active=state.dot[2]==state.player[2],
            width=self.consts.DOT_SIZE[0], 
            height=self.consts.DOT_SIZE[1]
        )

        return AdventureObservation(
            player=player,
            dragon_yellow=dragon_yellow,
            dragon_green=dragon_green,
            key_yellow=key_yellow,
            key_black=key_black,
            gate_yellow=gate_yellow,
            gate_black=gate_black,
            sword=sword,
            bridge=bridge,
            magnet=magnet,
            chalice=chalice,
            dragon_red=dragon_red,
            key_white=key_white,
            gate_white=gate_white,
            bat=bat,
            dot=dot
        )

    #ToDo, no clue what this is used for
    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: AdventureObservation) -> jnp.ndarray:
           return jnp.concatenate([
               obs.player.x.flatten(),
               obs.player.y.flatten(),
               obs.player.tile.flatten(),
               obs.player.height.flatten(),
               obs.player.width.flatten(),
               obs.player.state.flatten(),
               obs.dragon_yellow.x.flatten(),
               obs.dragon_yellow.y.flatten(),
               obs.dragon_yellow.tile.flatten(),
               obs.dragon_yellow.height.flatten(),
               obs.dragon_yellow.width.flatten(),
               obs.dragon_yellow.state.flatten(),
               obs.dragon_green.x.flatten(),
               obs.dragon_green.y.flatten(),
               obs.dragon_green.tile.flatten(),
               obs.dragon_green.height.flatten(),
               obs.dragon_green.width.flatten(),
               obs.dragon_green.state.flatten(),
               obs.key_yellow.x.flatten(),
               obs.key_yellow.y.flatten(),
               obs.key_yellow.tile.flatten(),
               obs.key_yellow.height.flatten(),
               obs.key_yellow.width.flatten(),
               obs.key_yellow.state.flatten(),
               obs.key_black.x.flatten(),
               obs.key_black.y.flatten(),
               obs.key_black.tile.flatten(),
               obs.key_black.height.flatten(),
               obs.key_black.width.flatten(),
               obs.key_black.state.flatten(),
               obs.gate_yellow.x.flatten(),
               obs.gate_yellow.y.flatten(),
               obs.gate_yellow.tile.flatten(),
               obs.gate_yellow.height.flatten(),
               obs.gate_yellow.width.flatten(),
               obs.gate_yellow.state.flatten(),
               obs.gate_black.x.flatten(),
               obs.gate_black.y.flatten(),
               obs.gate_black.tile.flatten(),
               obs.gate_black.height.flatten(),
               obs.gate_black.width.flatten(),
               obs.gate_black.state.flatten(),
               obs.sword.x.flatten(),
               obs.sword.y.flatten(),
               obs.sword.tile.flatten(),
               obs.sword.height.flatten(),
               obs.sword.width.flatten(),
               obs.sword.state.flatten(),
               obs.bridge.x.flatten(),
               obs.bridge.y.flatten(),
               obs.bridge.tile.flatten(),
               obs.bridge.height.flatten(),
               obs.bridge.width.flatten(),
               obs.bridge.state.flatten(),
               obs.magnet.x.flatten(),
               obs.magnet.y.flatten(),
               obs.magnet.tile.flatten(),
               obs.magnet.height.flatten(),
               obs.magnet.width.flatten(),
               obs.magnet.state.flatten(),
               obs.chalice.x.flatten(),
               obs.chalice.y.flatten(),
               obs.chalice.tile.flatten(),
               obs.chalice.height.flatten(),
               obs.chalice.width.flatten(),
               obs.chalice.state.flatten(),
               obs.dragon_red.x.flatten(),
               obs.dragon_red.y.flatten(),
               obs.dragon_red.tile.flatten(),
               obs.dragon_red.height.flatten(),
               obs.dragon_red.width.flatten(),
               obs.dragon_red.state.flatten(),
               obs.key_white.x.flatten(),
               obs.key_white.y.flatten(),
               obs.key_white.tile.flatten(),
               obs.key_white.height.flatten(),
               obs.key_white.width.flatten(),
               obs.key_white.state.flatten(),
               obs.gate_white.x.flatten(),
               obs.gate_white.y.flatten(),
               obs.gate_white.tile.flatten(),
               obs.gate_white.height.flatten(),
               obs.gate_white.width.flatten(),
               obs.gate_white.state.flatten(),
               obs.bat.x.flatten(),
               obs.bat.y.flatten(),
               obs.bat.tile.flatten(),
               obs.bat.height.flatten(),
               obs.bat.width.flatten(),
               obs.bat.state.flatten(),
               obs.dot.x.flatten(),
               obs.dot.y.flatten(),
               obs.dot.tile.flatten(),
               obs.dot.height.flatten(),
               obs.dot.width.flatten(),
               obs.dot.state.flatten(),
            ]
           )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    #ToDo, used for the RL?
    def observation_space(self) -> spaces.Dict:

        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "dragon_yellow": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "dragon_green": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "key_yellow": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "key_black": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "gate_yellow": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "gate_black":spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "sword": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "bridge":spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "magnet": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "chalice": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "dragon_red": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "key_white": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "gate_white": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "bat": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "dot":spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(250, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AdventureState, ) -> AdventureInfo:
        return AdventureInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AdventureState, state: AdventureState):
        reward = jax.lax.cond(
            jnp.logical_or(jnp.logical_or(state.dragon_yellow[5]==1,state.dragon_green[5]==1),state.dragon_red[5]==1), #lose when eaten by dragon
            lambda :-1,
            lambda : jax.lax.cond(
                state.chalice[2]==1, #win when chalice in yellow castle
                lambda :state.step_counter,
                lambda :0 #this should happen on reset?
            )
        )
        #reward = jax.lax.cond(
        #    state.chalice[2]==1,
        #    lambda :1,
        #    lambda :0
        #)
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AdventureState) -> bool:
        return jnp.logical_or(jnp.logical_or(jnp.logical_or(state.dragon_yellow[5]==1,state.dragon_green[5]==1),state.dragon_red[5]==1), state.chalice[2]==1)


class AdventureRenderer(JAXGameRenderer):
    def __init__(self, consts: AdventureConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or AdventureConstants()
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 4. Bake assets once
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/adventure"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        #in general, assets are loaded in based on the Adventure state
        #where the state.player[2] is the current room of the player
        #dragon_xxx[3] are their state (neutral,attacking, dead)
        #gate[0] is the state (open, closing, closed)
        #chalice[3] is or the blinking
        
        #set bg here
        raster = self.jr.create_object_raster(self.BACKGROUND)
        room_mask =self.SHAPE_MASKS["room_number"][state.player[2]]
        raster = self.jr.render_at(raster, 0, 0, room_mask)

        #set player color here
        player_mask = self.SHAPE_MASKS["player_colors"][state.player[2]]
        raster = self.jr.render_at(raster, state.player[0], state.player[1], player_mask)

        #dragons
        dragon_yellow_mask = self.SHAPE_MASKS["dragon_yellow"][state.dragon_yellow[3]]
        raster = jax.lax.cond(
            state.dragon_yellow[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.dragon_yellow[0], state.dragon_yellow[1], dragon_yellow_mask),
            lambda r : r,
            operand = raster,
        )
        dragon_green_mask = self.SHAPE_MASKS["dragon_green"][state.dragon_green[3]]
        raster = jax.lax.cond(
            state.dragon_green[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.dragon_green[0], state.dragon_green[1], dragon_green_mask),
            lambda r : r,
            operand = raster,
        )
        dragon_red_mask = self.SHAPE_MASKS["dragon_red"][state.dragon_green[3]]
        raster = jax.lax.cond(
            state.dragon_red[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.dragon_red[0], state.dragon_red[1], dragon_red_mask),
            lambda r : r,
            operand = raster,
        )
        
        #keys
        key_yellow_mask = self.SHAPE_MASKS["key_yellow"]
        raster = jax.lax.cond(
            state.key_yellow[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.key_yellow[0], state.key_yellow[1], key_yellow_mask),
            lambda r : r,
            operand = raster,
        )
        key_black_mask = self.SHAPE_MASKS["key_black"]
        raster = jax.lax.cond(
            state.key_black[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.key_black[0], state.key_black[1], key_black_mask),
            lambda r : r,
            operand = raster,
        )
        key_white_mask = self.SHAPE_MASKS["key_white"]
        raster = jax.lax.cond(
            state.key_white[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.key_white[0], state.key_white[1], key_white_mask),
            lambda r : r,
            operand = raster,
        )

        #Gates
        gate_yellow_mask = self.SHAPE_MASKS["gate_state"][state.gate_yellow[0]]
        
        raster = jax.lax.cond(
            self.consts.YELLOW_GATE_POS[2]==state.player[2],
            lambda r : self.jr.render_at(raster, self.consts.YELLOW_GATE_POS[0], self.consts.YELLOW_GATE_POS[1], gate_yellow_mask),
            lambda r : r,
            operand = raster,
        )
        gate_black_mask = self.SHAPE_MASKS["gate_state"][state.gate_black[0]]
        raster = jax.lax.cond(
            self.consts.BLACK_GATE_POS[2]==state.player[2],
            lambda r : self.jr.render_at(raster, self.consts.BLACK_GATE_POS[0], self.consts.BLACK_GATE_POS[1], gate_black_mask),#ToDO
            lambda r : r,
            operand = raster,
        )
        gate_white_mask = self.SHAPE_MASKS["gate_state"][state.gate_white[0]]
        raster = jax.lax.cond(
            self.consts.WHITE_GATE_POS[2]==state.player[2],
            lambda r : self.jr.render_at(raster, self.consts.WHITE_GATE_POS[0], self.consts.WHITE_GATE_POS[1], gate_white_mask),#ToDO
            lambda r : r,
            operand = raster,
        )
        

        #items
        sword_mask = self.SHAPE_MASKS["sword"]
        raster = jax.lax.cond(
            state.sword[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.sword[0], state.sword[1], sword_mask),
            lambda r : r,
            operand = raster,
        )
        bridge_mask = self.SHAPE_MASKS["bridge"]
        raster = jax.lax.cond(
            state.bridge[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.bridge[0], state.bridge[1], bridge_mask),
            lambda r : r,
            operand = raster,
        )
        magnet_mask = self.SHAPE_MASKS["magnet"]
        raster = jax.lax.cond(
            state.magnet[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.magnet[0], state.magnet[1], magnet_mask),
            lambda r : r,
            operand = raster,
        )

        #chalice
        chalice_mask = self.SHAPE_MASKS["chalice"][state.chalice[3]]
        raster = jax.lax.cond(
            state.chalice[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.chalice[0], state.chalice[1], chalice_mask),
            lambda r : r,
            operand = raster,
        )

        bat_mask = self.SHAPE_MASKS["bat"][state.bat[3]]
        raster = jax.lax.cond(
            state.bat[2]==state.player[2],
            lambda r : self.jr.render_at(raster, self.consts.BAT_SPAWN[0], self.consts.BAT_SPAWN[1], bat_mask),#ToDO
            lambda r : r,
            operand = raster,
        )
        dot_mask = self.SHAPE_MASKS["dot"]
        raster = jax.lax.cond(
            state.dot[2]==state.player[2],
            lambda r : self.jr.render_at(raster, state.dot[0], state.dot[1], dot_mask),
            lambda r : r,
            operand = raster,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)
