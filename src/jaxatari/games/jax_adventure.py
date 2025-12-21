from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Adventure.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        #all rooms in order ToDo pt overview map into the readme?
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
        #Player in all the different colors
        {'name': 'player_colors', 'type': 'group', 'files': ["Player_Yellow.npy",
                                                             "Player_Yellow.npy", 
                                                             "Player_Green.npy",
                                                             "Player_Purple.npy",
                                                             "Player_Pink.npy",
                                                             "Player_Green_yellow.npy",
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
        #Keys
        {'name': 'key_yellow', 'type': 'single', 'file': 'Key_yellow.npy'},
        {'name': 'key_black', 'type': 'single', 'file': 'Key_black.npy'},
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
        #Chalice
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
    )


class AdventureConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 250
    # Wall coordinates the player cannot pass through
    LEFT_WALL_X: int = 8
    RIGHT_WALL_X: int = 148
    UPPER_WALL_Y: int = 43
    LOWER_WALL_Y: int = 199
    #special black borders to the left and right
    SPECIAL_WALL_LEFT: int = 12
    SPECIAL_WALL_RIGHT: int = 145
    # Path South and North to another Room, X-Coordinates that offer hole in the wall
    PATH_VERTICAL_LEFT: int = 64
    PATH_VERTICAL_RIGHT: int = 95
    # Path East and West, Y-Coordinates that offer hole in the wall
    PATH_HORIZONTAL_UP: int = 40
    PATH_HORIZONTAL_DOWN: int = 200
    # Castle Edges
    CASTLE_TOWER_LEFT_X: int = 35
    CASTLE_TOWER_RIGHT_X: int = 120
    CASTLE_BASE_LEFT_X: int = 45
    CASTLE_BASE_RIGHT_X: int = 113
    CASTLE_TOWER_CORNER_Y: int = 105
    CASTLE_BASE_CORNER_Y: int = 170

    #upper left corner is 0, 0
    PLAYER_SIZE: Tuple[int, int] = (4, 8)
    KEY_SIZE: Tuple[int, int] = (8, 6)
    DRAGON_SIZE: Tuple[int, int] = (8, 44)
    GATE_SIZE: Tuple[int, int] = (7, 32)
    SWORD_SIZE: Tuple[int, int] = (8, 10)
    BRIDGE_SIZE: Tuple[int, int] = (32, 48)
    MAGNET_SIZE: Tuple[int, int] = (8, 16)
    CHALICE_SIZE: Tuple[int, int] = (8, 18)
    EMPTY_HAND_ID: int = 0
    KEY_YELLOW_ID: int = 1
    KEY_BLACK_ID: int = 2
    SWORD_ID: int = 3
    BRIDGE_ID: int = 4
    MAGNET_ID: int = 5
    CHALICE_ID: int = 6
    #Spawn Locations of all Entities: (X, Y, Room/Tile)
    YELLOW_GATE_POS: Tuple[int, int, int] = (76, 140, 0)
    BLACK_GATE_POS: Tuple[int, int, int] = (76, 140, 11)
    PLAYER_SPAWN: Tuple[int, int, int] = (78, 174, 0) #Changed from (78, 174, 0)
    DRAGON_YELLOW_SPAWN: Tuple[int, int, int] = (80, 170, 5, 0)
    DRAGON_GREEN_SPAWN: Tuple[int, int, int] = (80, 130, 4, 0)
    KEY_YELLOW_SPAWN: Tuple[int, int, int] = (31, 110, 0) #Changed from (31, 110, 0) for Testing
    KEY_BLACK_SPAWN: Tuple[int, int, int] = (31, 100, 4)
    SWORD_SPAWN: Tuple[int, int, int] = (31,180,1)
    BRIDGE_SPAWN: Tuple[int, int, int] = (40,130,10)
    MAGNET_SPAWN: Tuple[int, int, int] = (120,180,12)
    CHALICE_SPAWN: Tuple[int, int, int] = (35,180,13)
    # sset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = _get_default_asset_config()


# immutable state container

class AdventureState(NamedTuple):
    #step conter for performance indicator?
    step_counter: chex.Array
    #position player: x ,y ,tile, inventory, inventory cooldown
    player: chex.Array
    #positions dragons: x, y ,tile ,state
    dragon_yellow: chex.Array
    dragon_green: chex.Array
    #positions keys: x, y, tile
    key_yellow: chex.Array
    key_black: chex.Array
    #gates: state
    gate_yellow: chex.Array
    gate_black: chex.Array
    #position sword: x, y, tile
    sword: chex.Array
    #position bridge: x, y, tile
    bridge: chex.Array
    #position magnet: x, y, tile
    magnet: chex.Array
    #position chalice: x, y, tile, color
    chalice: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    tile: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    state: jnp.ndarray


class AdventureObservation(NamedTuple):
    player: EntityPosition
    dragon_yellow: EntityPosition
    dragon_green: EntityPosition
    key_yellow: EntityPosition
    key_black: EntityPosition
    gate_yellow: EntityPosition
    gate_black: EntityPosition
    sword: EntityPosition
    bridge: EntityPosition
    magnet: EntityPosition
    chalice: EntityPosition


class AdventureInfo(NamedTuple):
    time: jnp.ndarray


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

        ### Left Wall with and without Path
        left_wall = self.consts.LEFT_WALL_X
        upper_edge = self.consts.PATH_HORIZONTAL_UP
        lower_edge = self.consts.PATH_HORIZONTAL_DOWN
        collision_left_wall = player_x >= left_wall
        collision_left_wall_path = jnp.logical_or(collision_left_wall, jnp.logical_and(player_y>=upper_edge, player_y<=lower_edge))
        collision_left_special = player_x >= self.consts.SPECIAL_WALL_LEFT

        ### Right Wall with and without Path
        right_wall = self.consts.RIGHT_WALL_X
        upper_edge = self.consts.PATH_HORIZONTAL_UP
        lower_edge = self.consts.PATH_HORIZONTAL_DOWN
        collision_right_wall = player_x <= right_wall
        collision_right_wall_path = jnp.logical_or(collision_right_wall, jnp.logical_and(player_y>=upper_edge, player_y<=lower_edge))
        collision_right_special = player_x <= self.consts.SPECIAL_WALL_RIGHT

        ### Upper Wall with and without Path
        upper_wall = self.consts.UPPER_WALL_Y
        edge_left = self.consts.PATH_VERTICAL_LEFT
        edge_right = self.consts.PATH_VERTICAL_RIGHT
        collision_upper_wall = player_y >= upper_wall
        collision_upper_wall_path = jnp.logical_or(collision_upper_wall, jnp.logical_and(player_x>=edge_left, player_x<=edge_right))

        ### Lower Wall with and without Path
        lower_wall = self.consts.LOWER_WALL_Y
        edge_left = self.consts.PATH_VERTICAL_LEFT
        edge_right = self.consts.PATH_VERTICAL_RIGHT
        collision_lower_wall = player_y <= lower_wall
        collision_lower_wall_path = jnp.logical_or(collision_lower_wall, jnp.logical_and(player_x>=edge_left, player_x<=edge_right))


        ## Rooms:
        room_1_clear = jnp.logical_or(
                jnp.logical_not(room == 0), #either it is not room 1 or
                jnp.logical_and(            #walls of the room are not being crossed
                    jnp.logical_and(
                    collision_left_wall,        
                    collision_right_wall        
                    ),
                    jnp.logical_and(
                    collision_upper_wall_path,  
                    collision_lower_wall_path   
                    )
                )
        )

        room_2_clear = jnp.logical_or(
                jnp.logical_not(room == 1), #either it is not room 1 or
                jnp.logical_and(            #walls of the room are not being crossed
                    jnp.logical_and(
                    collision_left_wall,        
                    collision_right_wall        
                    ),
                    jnp.logical_and(
                    collision_upper_wall,  
                    collision_lower_wall_path   
                    )
                )
        )

        room_3_clear = jnp.logical_or(
                jnp.logical_not(room == 2), #either it is not room 1 or
                jnp.logical_and(            #walls of the room are not being crossed
                    jnp.logical_and(
                    collision_left_wall_path,        
                    collision_right_wall_path        
                    ),
                    jnp.logical_and(
                    collision_upper_wall_path,  
                    collision_lower_wall   
                    )
                )
        )

        room_4_clear = jnp.logical_or(
                jnp.logical_not(room == 3), #either it is not room 1 or
                jnp.logical_and(            #walls of the room are not being crossed
                    jnp.logical_and(
                    collision_left_wall_path,        
                    collision_right_special        
                    ),
                    jnp.logical_and(
                    collision_upper_wall,  
                    collision_lower_wall_path   
                    )
                )
        )

        room_5_clear = jnp.logical_or(
                jnp.logical_not(room == 4), #either it is not room 1 or
                jnp.logical_and(            #walls of the room are not being crossed
                    jnp.logical_and(
                    collision_left_wall,        
                    collision_right_wall        
                    ),
                    jnp.logical_and(
                    collision_upper_wall_path,  
                    collision_lower_wall   
                    )
                )
        )

        room_6_clear = jnp.logical_or(
                jnp.logical_not(room == 5), #either it is not room 1 or
                jnp.logical_and(            #walls of the room are not being crossed
                    jnp.logical_and(
                    collision_left_special,        
                    collision_right_wall_path        
                    ),
                    jnp.logical_and(
                    collision_upper_wall_path,  
                    collision_lower_wall  
                    )
                )
        )

        ### extra maze walls

        room_7_walls = jnp.logical_or(
            player_y >= 170,        # either below lowest thick wall or
            jnp.logical_or(
                jnp.logical_or(     # the two corridors up
                    jnp.logical_and(player_x>=30, player_x <= 38),  #left corridor
                    jnp.logical_and(player_x>=120, player_x <= 126) #right corridor
                ),
                jnp.logical_or(
                    jnp.logical_or(
                        jnp.logical_and(
                            jnp.logical_and(player_y <= 135, player_y >= 105),
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >=45, player_x <= 110),
                                    player_x <= 20
                                ),
                                player_x >=135
                            )
                        ),
                        jnp.logical_and(
                            jnp.logical_and(player_y <= 105, player_y >= 75),
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 45, player_x <=53),
                                    jnp.logical_and(player_x >= 15, player_x <= 20)
                                ),
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 102, player_x <= 110),
                                    jnp.logical_and(player_x >= 135, player_x <= 143)
                                )
                            )
                        )
                    ),
                    jnp.logical_or(
                        jnp.logical_and(
                            jnp.logical_and(player_y <= 70, player_y >= 40),
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 62, player_x <= 94),
                                    jnp.logical_or(player_x <= 20, player_x >=135)
                                ),
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 102, player_x <= 110),
                                    jnp.logical_and(player_x >= 45, player_x <= 53)
                                )
                            )
                        ),
                        jnp.logical_and(
                            player_y <= 40,
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 62, player_x <= 70), 
                                    jnp.logical_and(player_x >= 86, player_x <= 94) 
                                ),
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 102, player_x <= 110), 
                                    jnp.logical_and(player_x >= 45, player_x <= 53) 
                                )
                            )
                        )
                    )
                )
            )
        )

        room_8_walls = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_or(
                    jnp.logical_and(
                        player_y >= 199,
                        jnp.logical_or(
                            jnp.logical_or(
                                jnp.logical_and(player_x >= 30, player_x <= 38), 
                                jnp.logical_and(player_x >= 120, player_x <= 126)  
                            ),
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 102, player_x <= 110), 
                                    jnp.logical_and(player_x >= 45, player_x <= 53) 
                                ),
                                jnp.logical_and(player_x>=edge_left, player_x<=edge_right)
                            )
                        )
                    ),
                    jnp.logical_and(
                        jnp.logical_and(player_y >= 170, player_y <= 199),
                        jnp.logical_or(
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 30, player_x <= 38), 
                                    jnp.logical_and(player_x >= 120, player_x <= 126)  
                                ),
                                jnp.logical_or(
                                    jnp.logical_or(
                                        jnp.logical_and(player_x >= 102, player_x <= 110), 
                                        jnp.logical_and(player_x >= 45, player_x <= 53) 
                                    ),
                                    jnp.logical_and(player_x>=edge_left, player_x<=edge_right)
                                )
                                ),
                                jnp.logical_or(
                                    player_x <= 20,
                                    player_x >= 135
                                )
                            )
                        )
                    ),
                
                    jnp.logical_or(
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 135, player_y <= 170),
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_or(
                                        jnp.logical_and(player_x >= 30, player_x <= 38), 
                                        jnp.logical_and(player_x >= 120, player_x <= 126)  
                                    ),
                                    jnp.logical_or(
                                        jnp.logical_or(
                                            jnp.logical_and(player_x >= 102, player_x <= 110), 
                                            jnp.logical_and(player_x >= 45, player_x <= 53) 
                                        ),
                                        jnp.logical_and(player_x>=72, player_x<=84)
                                    )
                                    ),
                                    jnp.logical_or(
                                        jnp.logical_and(player_x <= 20, player_x >= 14),
                                        jnp.logical_and(player_x >= 135, player_x <= 142)
                                    )
                                )
                        ),
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 105, player_y <= 135),
                        jnp.logical_or(
                                jnp.logical_or(
                                    player_x <= 38, 
                                    player_x >= 120
                                ),
                                jnp.logical_or(
                                    jnp.logical_or(
                                        jnp.logical_and(player_x >= 102, player_x <= 110), 
                                        jnp.logical_and(player_x >= 45, player_x <= 53) 
                                    ),
                                    jnp.logical_and(player_x>=72, player_x<=84)
                                )
                            )
                        )
                    ),
                ),
                jnp.logical_or(
                    jnp.logical_or(
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 75, player_y <= 105),
                        jnp.logical_or(
                            jnp.logical_or(
                                jnp.logical_and(player_x >= 102, player_x <= 110), 
                                jnp.logical_and(player_x >= 45, player_x <= 53) 
                            ),
                            jnp.logical_and(player_x>=72, player_x<=84)
                        ),
                        ),
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 40, player_y <= 70),
                         jnp.logical_or(
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 30, player_x <= 53), 
                                    jnp.logical_and(player_x >= 102, player_x <= 126)  
                                ),
                                jnp.logical_and(player_x>=72, player_x<=84)
                                ),
                                jnp.logical_or(
                                    player_x <= 20,
                                    player_x >= 135
                                )
                            )
                        )
                    ),
                    jnp.logical_and(
                        player_y <= 40,
                        jnp.logical_or(
                            jnp.logical_or(
                                jnp.logical_and(player_x >= 30, player_x <= 38), 
                                jnp.logical_and(player_x >= 120, player_x <= 126)  
                            ),
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x <= 20, player_x >= 14),
                                    jnp.logical_and(player_x >= 135, player_x <= 142)
                                ),
                                jnp.logical_and(player_x>=72, player_x<=84)
                            )
                        )
                    )                    
                )
            )
        
        room_9_walls = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_or(
                    jnp.logical_and(
                        player_y >= 199,
                        False
                    ),
                    jnp.logical_and(
                        jnp.logical_and(player_y >= 170, player_y <= 199),
                        jnp.logical_or(
                            
                                jnp.logical_and(
                                    player_x >= 30,
                                    player_x <= 126
                                ),
                                
                                jnp.logical_or(
                                    player_x <= 20,
                                    player_x >= 135
                                )
                            )
                        )
                    ),
                
                    jnp.logical_or(
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 135, player_y <= 170),
                        jnp.logical_and(
                                player_x >= 30,
                                player_x <= 126
                            )
                        ),
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 105, player_y <= 135),
                        jnp.logical_and(
                            player_x >= 14,
                            player_x <= 142
                        )
                        )
                    ),
                ),
                jnp.logical_or(
                    jnp.logical_or(
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 75, player_y <= 105),
                        jnp.logical_or(
                            jnp.logical_or(
                                jnp.logical_and(player_x <= 20, player_x >= 14),
                                jnp.logical_and(player_x >= 135, player_x <= 142)
                            ),
                            jnp.logical_and(player_x>=edge_left, player_x<=edge_right)
                        ),
                        ),
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 40, player_y <= 70),
                         jnp.logical_or(
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 30, player_x <= 53), 
                                    jnp.logical_and(player_x >= 102, player_x <= 126)  
                                ),
                                jnp.logical_and(player_x>=edge_left, player_x<=edge_right)
                                ),
                                jnp.logical_or(
                                    player_x <= 20,
                                    player_x >= 135
                                )
                            )
                        )
                    ),
                    jnp.logical_and(
                        player_y <= 40,
                        jnp.logical_or(
                            jnp.logical_or(
                                jnp.logical_and(player_x >= 30, player_x <= 38), 
                                jnp.logical_and(player_x >= 120, player_x <= 126)  
                            ),
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 102, player_x <= 110), 
                                        jnp.logical_and(player_x >= 45, player_x <= 53) 
                                ),
                                jnp.logical_and(player_x>=edge_left, player_x<=edge_right)
                            )
                        )
                    )                    
                )
            )
        
        room_10_walls = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_or(
                    jnp.logical_and(
                        player_y >= 199,
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_or(
                                        jnp.logical_and(player_x >= 62, player_x <= 70), 
                                        jnp.logical_and(player_x >= 86, player_x <= 94) 
                                    ),
                                    jnp.logical_or(
                                        jnp.logical_and(player_x >= 102, player_x <= 110), 
                                        jnp.logical_and(player_x >= 45, player_x <= 53) 
                                    )
                                ),
                                jnp.logical_or(     # the two corridors up
                                    jnp.logical_and(player_x>=30, player_x <= 38),  #left corridor
                                    jnp.logical_and(player_x>=120, player_x <= 126) #right corridor
                                )
                            )
                    ),
                    jnp.logical_and(
                        jnp.logical_and(player_y >= 170, player_y <= 199),
                        jnp.logical_or(
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 30, player_x <= 53), 
                                    jnp.logical_and(player_x >= 102, player_x <= 126)  
                                ),
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 62, player_x <= 70), 
                                    jnp.logical_and(player_x >= 86, player_x <= 94) 
                                )
                                ),
                                jnp.logical_or(
                                    player_x <= 20,
                                    player_x >= 135
                                )
                            )
                        )
                    ),
                
                    jnp.logical_or(
                        jnp.logical_and(
                            jnp.logical_and(player_y >= 135, player_y <= 170), 
                            jnp.logical_or(
                                jnp.logical_or(
                                    jnp.logical_and(player_x >= 62, player_x <= 70), 
                                    jnp.logical_and(player_x >= 86, player_x <= 94)  
                                ),
                                jnp.logical_or(
                                    jnp.logical_and(player_x <= 20, player_x >= 14),
                                    jnp.logical_and(player_x >= 135, player_x <= 142)
                                )
                            )
                        ),
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 105, player_y <= 135),
                        jnp.logical_or(
                                jnp.logical_and(player_x >=14, player_x <= 70),
                                jnp.logical_and(player_x >= 86, player_x <= 142)
                            )
                        )
                    ),
                ),
                jnp.logical_or(
                    jnp.logical_or(
                        jnp.logical_and(
                        jnp.logical_and(player_y >= 75, player_y <= 105),
                        jnp.logical_or(
                            jnp.logical_and(player_x >= 38, player_x <= 44),
                            jnp.logical_and(player_x >= 112, player_x <= 118)
                        ),
                        ),
                        jnp.logical_and(
                            player_y >= 40,
                            player_y <= 70
                        )
                    ),
                    jnp.logical_and(
                        player_y <= 40,
                        False
                    )                    
                )
            )
        


        ### end extra maze walls

        room_7_clear = jnp.logical_or(
            jnp.logical_not(room == 6),
            jnp.logical_and(
                collision_lower_wall_path,
                room_7_walls
            )
        )

        room_8_clear = jnp.logical_or(
            jnp.logical_not(room == 7),
            room_8_walls
            )

        room_9_clear = jnp.logical_or(
            jnp.logical_not(room == 8),
            room_9_walls
            )
        
        room_10_clear = jnp.logical_or(
            jnp.logical_not(room == 9),
            room_10_walls
        )

        room_11_clear = True

        room_12_clear = jnp.logical_or(
                jnp.logical_not(room == 11), #either it is not room 1 or
                jnp.logical_and(            #walls of the room are not being crossed
                    jnp.logical_and(
                    collision_left_wall,        
                    collision_right_wall        
                    ),
                    jnp.logical_and(
                    collision_upper_wall_path,  
                    collision_lower_wall_path 
                    )
                )
        )

        room_13_clear = jnp.logical_or(
                jnp.logical_not(room == 12), #either it is not room 1 or
                jnp.logical_and(            #walls of the room are not being crossed
                    jnp.logical_and(
                    collision_left_wall,        
                    collision_right_wall        
                    ),
                    jnp.logical_and(
                    collision_upper_wall_path,  
                    collision_lower_wall_path  
                    )
                )
        )

        room_14_clear = jnp.logical_or(
                jnp.logical_not(room == 13), #either it is not room 1 or
                jnp.logical_and(            #walls of the room are not being crossed
                    jnp.logical_and(
                    collision_left_wall,        
                    collision_right_wall        
                    ),
                    jnp.logical_and(
                    collision_upper_wall,  
                    collision_lower_wall_path  
                    )
                )
        )

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
        castle_base_in = jnp.logical_and(player_x>=edge_left, player_x<=edge_right)
        castle_base = jnp.logical_or(player_y >= castle_base_height, jnp.logical_or(castle_base_in, castle_base_out))

        castle_collision = jnp.logical_or(
            jnp.logical_not(jnp.logical_or(room==0, room==11)), #either it is not a castle tile, or
            jnp.logical_and(castle_towers, castle_base)
        )

        room_1_and_2 = jnp.logical_and(room_1_clear, room_2_clear)
        room_3_and_4 = jnp.logical_and(room_3_clear, room_4_clear)
        room_1_to_4 = jnp.logical_and(room_1_and_2, room_3_and_4)
        room_5_and_6 = jnp.logical_and(room_5_clear, room_6_clear)
        room_7_and_8 = jnp.logical_and(room_7_clear, room_8_clear)
        room_5_to_8 = jnp.logical_and(room_5_and_6, room_7_and_8)
        room_9_and_10 = jnp.logical_and(room_9_clear, room_10_clear)
        room_11_and_12= jnp.logical_and(room_11_clear, room_12_clear)
        room_9_to_12 = jnp.logical_and(room_9_and_10, room_11_and_12)
        room_13_and_14 = jnp.logical_and(room_13_clear, room_14_clear)

        room_1_to_8 = jnp.logical_and(room_1_to_4,room_5_to_8)
        room_9_to_14 = jnp.logical_and(room_9_to_12, room_13_and_14)

        base_rooms = jnp.logical_and(room_1_to_8, room_9_to_14)

        return_bool = jnp.logical_and(base_rooms, castle_collision )# todo
        
        return return_bool

    def _player_step(self, state: AdventureState, action: chex.Array) -> AdventureState:
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        up = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)

      

        #check for no wall before walking
        left_no_wall = jnp.logical_and(left,self._check_for_wall(state, 0))
        right_no_wall = jnp.logical_and(right,self._check_for_wall(state, 1))
        up_no_wall =  jnp.logical_and(up,self._check_for_wall(state, 2))
        down_no_wall =  jnp.logical_and(down,self._check_for_wall(state, 3))

        #chek for item
        #has_item = jax.lax.cond(
        #    pred=state.player[3]==self.consts.EMPTY_HAND_ID,
        #    true_fun=False,
        #    false_fun=True
        #)
        
        new_item_x = jax.lax.switch(
            state.player[3],
            [lambda:0,
             lambda:state.key_yellow[0],
             lambda:state.key_black[0],
             lambda:state.sword[0],
             lambda:state.bridge[0],
             lambda:state.magnet[0],
             lambda:state.chalice[0]
             ]
        )

        new_player_x = state.player[0]
        new_player_x, new_item_x = jax.lax.cond(
            left_no_wall,
            lambda y: (y[0]-4,y[1]-4),
            lambda y: y,
            operand = (new_player_x,new_item_x),
        )
        new_player_x, new_item_x = jax.lax.cond(
            right_no_wall,
            lambda y: (y[0]+4,y[1]+4),
            lambda y: y,
            operand = (new_player_x,new_item_x),
        )


        new_item_y = jax.lax.switch(
            state.player[3],
            [lambda:0,
             lambda:state.key_yellow[1],
             lambda:state.key_black[1],
             lambda:state.sword[1],
             lambda:state.bridge[1],
             lambda:state.magnet[1],
             lambda:state.chalice[1]
             ]
        )

        new_player_y = state.player[1]
        new_player_y, new_item_y = jax.lax.cond(
            down_no_wall,
            lambda y: (y[0]+8,y[1]+8),
            lambda y: y,
            operand = (new_player_y,new_item_y)
        )
        new_player_y, new_item_y = jax.lax.cond(
            up_no_wall,
            lambda y: (y[0]-8,y[1]-8),
            lambda y: y,
            operand = (new_player_y,new_item_y)
        )
        new_player_tile = state.player[2]
        new_player_y, new_player_tile = jax.lax.cond(
            new_player_y > 212,
            lambda _: (27, jax.lax.switch( new_player_tile, [lambda:2,lambda:0,lambda:0, 
                                                             lambda:4, lambda:0, lambda:0, 
                                                             lambda:5, lambda:8, lambda:0, 
                                                             lambda: 6, lambda:7, lambda:10, 
                                                             lambda:11, lambda:12])),
            lambda _: (new_player_y, new_player_tile),
            operand = None,
        )
        new_player_y, new_player_tile = jax.lax.cond(
            new_player_y < 27,
            lambda _: (212, jax.lax.switch( new_player_tile, [lambda:1,lambda:0,lambda:0, 
                                                              lambda:0, lambda:3, lambda:6, 
                                                              lambda:9, lambda:10, lambda:7, 
                                                              lambda: 0, lambda:11, lambda:12, 
                                                              lambda:13, lambda:0])),
            lambda _: (new_player_y, new_player_tile),
            operand = None,
        )
        new_player_x, new_player_tile = jax.lax.cond(
            new_player_x > 160,
            lambda _: (0, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:3, 
                                                            lambda:0, lambda:0, lambda:2, 
                                                            lambda:7, lambda:6, lambda:10, 
                                                            lambda: 8, lambda:9, lambda:0, 
                                                            lambda:0, lambda:0])),
            lambda _: (new_player_x, new_player_tile),
            operand = None,
        )
        new_player_x, new_player_tile = jax.lax.cond(
            new_player_x < 0,
            lambda _: (160, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:5, 
                                                              lambda:2, lambda:0, lambda:0, 
                                                              lambda:7, lambda:6, lambda:9, 
                                                              lambda: 10, lambda:8, lambda:0, 
                                                              lambda:0, lambda:0])),
            lambda _: (new_player_x, new_player_tile),
            operand = None,
        )

        return AdventureState(
            step_counter = state.step_counter,
            player = jnp.array([new_player_x,new_player_y,new_player_tile,state.player[3]]).astype(jnp.int32), #SEEMS NOT GOOD
            dragon_yellow = state.dragon_yellow,
            dragon_green = state.dragon_green,
            key_yellow = jax.lax.cond(state.player[3]==self.consts.KEY_YELLOW_ID,
                                      lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                      lambda op: op[3],
                                      operand=(new_item_x,new_item_y,state.key_yellow[2],state.key_yellow),
                                      ),
            key_black= jax.lax.cond(state.player[3]==self.consts.KEY_BLACK_ID,
                                    lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                    lambda op: op[3],
                                    operand=(new_item_x,new_item_y,state.key_black[2],state.key_black)
                                    ),
            gate_yellow=state.gate_yellow,
            gate_black=state.gate_black,
            sword= jax.lax.cond(state.player[3]==self.consts.SWORD_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,state.sword[2],state.sword)
                                ),
            bridge= jax.lax.cond(state.player[3]==self.consts.BRIDGE_ID,
                                 lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,state.bridge[2],state.bridge)
                                ),
            magnet= jax.lax.cond(state.player[3]==self.consts.MAGNET_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,state.magnet[2],state.magnet)
                                ),
            chalice= jax.lax.cond(state.player[3]==self.consts.CHALICE_ID,
                                  lambda op: jnp.array([op[0],op[1],op[2],op[3]]).astype(jnp.int32),
                                  lambda op: op[4],
                                  operand=(new_item_x,new_item_y,state.chalice[2],state.chalice[3],state.chalice)
                                  )
        )
    
    def _item_pickup(self, state: AdventureState, action: chex.Array) -> AdventureState:
        
        def check_for_item(self:JaxAdventure, state: AdventureState, item_ID: int) -> bool:
            item_x, item_y, tile, item_width, item_height = jax.lax.switch(
                item_ID,
                [lambda:(0,0,0,0,0), #this should never occour
                lambda:(state.key_yellow[0],state.key_yellow[1],state.key_yellow[2],self.consts.KEY_SIZE[0],self.consts.KEY_SIZE[1]),
                lambda:(state.key_black[0],state.key_black[1],state.key_black[2],self.consts.KEY_SIZE[0],self.consts.KEY_SIZE[1]),
                lambda:(state.sword[0],state.sword[1],state.sword[2],self.consts.SWORD_SIZE[0],self.consts.SWORD_SIZE[1]),
                lambda:(state.bridge[0],state.bridge[1],state.bridge[2],self.consts.BRIDGE_SIZE[0],self.consts.BRIDGE_SIZE[1]),
                lambda:(state.magnet[0],state.magnet[1],state.magnet[2],self.consts.MAGNET_SIZE[0],self.consts.MAGNET_SIZE[1]),
                lambda:(state.chalice[0],state.chalice[1],state.chalice[2],self.consts.CHALICE_SIZE[0],self.consts.CHALICE_SIZE[1])
                ])
            #jax.debug.print("Hitbox values item:{a},{b},{c},{d},{e}",a=item_x,b=item_y,c=tile,d=item_width,e=item_height)
            #HARDCODED BAAAAAD, but i dont care right now (performance?)(items smaler then 4 pixels would be buggy)
            on_same_tile = (tile==state.player[2])
            player_hitbox_nw = (state.player[0],state.player[1])
            player_hitbox_ne = (state.player[0]+self.consts.PLAYER_SIZE[0]-1,state.player[1])
            player_hitbox_se = (state.player[0]+self.consts.PLAYER_SIZE[0]-1,state.player[1]+self.consts.PLAYER_SIZE[1]-1)
            player_hitbox_sw = (state.player[0],state.player[1]+self.consts.PLAYER_SIZE[1]-1)

            #jax.debug.print("Hitbox values Player:{a},{b}|{c},{d}|{e},{f}|{g},{h}",
            #                a=player_hitbox_nw[0],b=player_hitbox_nw[1],
            #                c=player_hitbox_ne[0],d=player_hitbox_ne[1],
            #                e=player_hitbox_se[0],f=player_hitbox_se[1],
            #                g=player_hitbox_sw[0],h=player_hitbox_sw[1])
            
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
        )
        

        return AdventureState(
            step_counter=state.step_counter,
            player = jnp.array([state.player[0],state.player[1],state.player[2],new_player_inventory]).astype(jnp.int32),
            dragon_yellow=state.dragon_yellow,
            dragon_green=state.dragon_green,
            key_yellow=state.key_yellow,
            key_black=state.key_black,
            gate_yellow=state.gate_yellow,
            gate_black=state.gate_black,
            sword=state.sword,
            bridge=state.bridge,
            magnet=state.magnet,
            chalice=state.chalice
        )
    
    def _item_drop(self, state: AdventureState, action: chex.Array) -> AdventureState:

        new_player_inventory = jax.lax.cond(
            action == Action.FIRE,
            lambda _: self.consts.EMPTY_HAND_ID,
            lambda op: op,
            operand=state.player[3]
        )

        return AdventureState(
            step_counter=state.step_counter,
            player = jnp.array([state.player[0],state.player[1],state.player[2],new_player_inventory]).astype(jnp.int32),
            dragon_yellow=state.dragon_yellow,
            dragon_green=state.dragon_green,
            key_yellow=state.key_yellow,
            key_black=state.key_black,
            gate_yellow=state.gate_yellow,
            gate_black=state.gate_black,
            sword=state.sword,
            bridge=state.bridge,
            magnet=state.magnet,
            chalice=state.chalice
        )
        

    def _dragon_step(self, state: AdventureState) -> AdventureState:
        direction_x = jnp.sign(state.player[0] - state.dragon_yellow[0])
        direction_y = jnp.sign(state.player[1]- state.dragon_yellow[1])
        dragon_yellow_x = state.dragon_yellow[0]
        dragon_yellow_y = state.dragon_yellow[1]
        dragon_yellow_x, dragon_yellow_y = jax.lax.cond(
            state.player[2]==state.dragon_yellow[2],
            lambda _: ((dragon_yellow_x + direction_x*2, dragon_yellow_y + direction_y*2)),
            lambda _: (dragon_yellow_x, dragon_yellow_y),
            operand  = None
        )
        direction_x = jnp.sign(state.player[0] - state.dragon_green[0])
        direction_y = jnp.sign(state.player[1]- state.dragon_green[1])
        dragon_green_x = state.dragon_green[0]
        dragon_green_y = state.dragon_green[1]
        dragon_green_x, dragon_green_y = jax.lax.cond(
            state.player[2]==state.dragon_green[2],
            lambda _: ((dragon_green_x + direction_x*2, dragon_green_y + direction_y*2)),
            lambda _: (dragon_green_x, dragon_green_y),
            operand  = None
        )


        return AdventureState(
            step_counter = state.step_counter,
            player = state.player,
            dragon_yellow = jnp.array([dragon_yellow_x,dragon_yellow_y,state.dragon_yellow[2]]).astype(jnp.int32),
            dragon_green = jnp.array([dragon_green_x,dragon_green_y,state.dragon_green[2]]).astype(jnp.int32),
            key_yellow=state.key_yellow,
            key_black=state.key_black,
            gate_yellow=state.gate_yellow,
            gate_black=state.gate_black,
            sword=state.sword,
            bridge=state.bridge,
            magnet=state.magnet,
            chalice=state.chalice
        )
    
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[AdventureObservation, AdventureState]:

        state = AdventureState(
            step_counter = jnp.array(0).astype(jnp.int32),
            #Player Spawn: x, y, tile, inventory, inventory cooldown
            player = jnp.array([self.consts.PLAYER_SPAWN[0],
                                self.consts.PLAYER_SPAWN[1],
                                self.consts.PLAYER_SPAWN[2],
                                self.consts.EMPTY_HAND_ID,
                                0]).astype(jnp.int32),
            #Dragons: x, y ,tile ,state
            dragon_yellow = jnp.array([self.consts.DRAGON_YELLOW_SPAWN[0],
                                       self.consts.DRAGON_YELLOW_SPAWN[1],
                                       self.consts.DRAGON_YELLOW_SPAWN[2],
                                       self.consts.DRAGON_YELLOW_SPAWN[3]]).astype(jnp.int32), #ToDo
            dragon_green = jnp.array([self.consts.DRAGON_GREEN_SPAWN[0],
                                      self.consts.DRAGON_GREEN_SPAWN[1],
                                      self.consts.DRAGON_GREEN_SPAWN[2],
                                      self.consts.DRAGON_GREEN_SPAWN[3]]).astype(jnp.int32), #ToDo
            #Keys: x ,y, tile
            key_yellow = jnp.array([self.consts.KEY_YELLOW_SPAWN[0],
                                    self.consts.KEY_YELLOW_SPAWN[1],
                                    self.consts.KEY_YELLOW_SPAWN[2]]).astype(jnp.int32),
            key_black = jnp.array([self.consts.KEY_BLACK_SPAWN[0],
                                    self.consts.KEY_BLACK_SPAWN[1],
                                    self.consts.KEY_BLACK_SPAWN[2]]).astype(jnp.int32),
            #Gate: state
            gate_yellow=jnp.array([0]).astype(jnp.int32),
            gate_black=jnp.array([0]).astype(jnp.int32),
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
            #Chalice: x, y, tile, color
            chalice = jnp.array([self.consts.CHALICE_SPAWN[0],
                                 self.consts.CHALICE_SPAWN[1],
                                 self.consts.CHALICE_SPAWN[2],7]).astype(jnp.int32), #ToDo
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

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
            key_yellow=state.key_yellow,
            key_black=state.key_black,
            gate_yellow=state.gate_yellow,
            gate_black=state.gate_black,
            sword=state.sword,
            bridge=state.bridge,
            magnet=state.magnet,
            chalice=state.chalice
        )
        state = self._player_step(state, action)
        state = self._item_pickup(state, action)
        state = self._item_drop(state, action)
        state = self._dragon_step(state)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: AdventureState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: AdventureState):
        player = EntityPosition(
            x=state.player[0],
            y=state.player[1],
            tile=state.player[2],
            width=self.consts.PLAYER_SIZE[0], 
            height=self.consts.PLAYER_SIZE[1], 
            state=state.player[4]
        )
        dragon_yellow = EntityPosition(
            x=state.dragon_yellow[0],
            y=state.dragon_yellow[1],
            tile=state.dragon_yellow[2],
            width=self.consts.DRAGON_SIZE[0], 
            height=self.consts.DRAGON_SIZE[1], 
            state=state.dragon_yellow[3]
        )
        dragon_green = EntityPosition(
            x=state.dragon_green[0],
            y=state.dragon_green[1],
            tile=state.dragon_green[2],
            width=self.consts.DRAGON_SIZE[0], 
            height=self.consts.DRAGON_SIZE[1], 
            state=state.dragon_green[3]
        )
        key_yellow = EntityPosition(
            x=state.key_yellow[0],
            y=state.key_yellow[1],
            tile=state.key_yellow[2],
            width=self.consts.KEY_SIZE[0], 
            height=self.consts.KEY_SIZE[1],
            state=0 #Key has no relevant state
        )
        key_black = EntityPosition(
            x=state.key_black[0],
            y=state.key_black[1],
            tile=state.key_black[2],
            width=self.consts.KEY_SIZE[0], 
            height=self.consts.KEY_SIZE[1],
            state=0 #Key has no relevant state
        )
        gate_yellow = EntityPosition(
            x=self.consts.YELLOW_GATE_POS[0],
            y=self.consts.YELLOW_GATE_POS[1],
            tile=self.consts.YELLOW_GATE_POS[2],
            width=self.consts.GATE_SIZE[0], 
            height=self.consts.GATE_SIZE[1], 
            state=state.gate_yellow[0]
        )
        gate_black = EntityPosition(
            x=self.consts.BLACK_GATE_POS[0],
            y=self.consts.BLACK_GATE_POS[1],
            tile=self.consts.BLACK_GATE_POS[2],
            width=self.consts.GATE_SIZE[0], 
            height=self.consts.GATE_SIZE[1], 
            state=state.gate_black[0]
        )
        sword = EntityPosition(
            x=state.sword[0],
            y=state.sword[1],
            tile=state.sword[2],
            width=self.consts.SWORD_SIZE[0], 
            height=self.consts.SWORD_SIZE[1], 
            state=0 #Sword has no relevant state
        )
        bridge = EntityPosition(
            x=state.bridge[0],
            y=state.bridge[1],
            tile=state.bridge[2],
            width=self.consts.BRIDGE_SIZE[0], 
            height=self.consts.BRIDGE_SIZE[1], 
            state=0 #Bridge has no relevant state
        )
        magnet = EntityPosition(
            x=state.magnet[0],
            y=state.magnet[1],
            tile=state.magnet[2],
            width=self.consts.MAGNET_SIZE[0], 
            height=self.consts.MAGNET_SIZE[1],
            state=0 #Magnet has no relevant state
        )
        chalice = EntityPosition(
            x=state.chalice[0],
            y=state.chalice[1],
            tile=state.chalice[2],
            width=self.consts.CHALICE_SIZE[0], 
            height=self.consts.CHALICE_SIZE[1],
            state=0 #Chalice has no relevant state
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
            chalice=chalice
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: AdventureObservation) -> jnp.ndarray:
           return jnp.concatenate([
               obs.player.x.flatten(),
               obs.player.y.flatten(),
               obs.player.height.flatten(),
               obs.player.width.flatten()
            ]
           )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
            }),
            "key_yellow": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
            }),

            #ToDo for the rest of the dragons, items etc..... ?
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
        return 1

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AdventureState) -> bool:
        return 0

class AdventureRenderer(JAXGameRenderer):
    def __init__(self, consts: AdventureConstants = None):
        super().__init__(consts)
        self.consts = consts or AdventureConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(250, 160),
            channels=3,
            #ownscale=(200, 128)
        )
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
        #set bg color here
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

        return self.jr.render_from_palette(raster, self.PALETTE)
