import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple
from jaxatari.games.jax_adventure import AdventureState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import chex
import os
from jaxatari.environment import JAXAtariAction as Action
import random

# --- 1. Individual Mod Plugins ---
class FasterDragonsMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "DRAGON_SPEED": 4,
    }

class FasterBiteMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "DRAGON_BITE_TIMER": 0,
    }

class FleaingDragonMod(JaxAtariInternalModPlugin):

    conflicts_with = ["dragon_revive"]

    @partial(jax.jit, static_argnums=(0,))
    def _dragon_step(self, state: AdventureState) -> AdventureState:
        speed = self._env.consts.DRAGON_SPEED

        #get sword position to kill dragons
        sword_x = state.sword[0]
        sword_y = state.sword[1]
        sword_room = state.sword[2]

        #yellow dragon
        direction_x = jnp.sign(state.player[0] - state.dragon_yellow[0])
        direction_y = jnp.sign(state.player[1]- state.dragon_yellow[1])
        move_direction_x, move_direction_y = jax.lax.cond(
            state.player[3]==self._env.consts.SWORD_ID,
            lambda: (-direction_x, -direction_y),
            lambda: (direction_x, direction_y)
        )
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
        dragon_yellow_freeze = dragon_yellow_counter % self._env.consts.DRAGON_BITE_TIMER != 0
    
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
            lambda _: (dragon_yellow_x + move_direction_x*speed, dragon_yellow_y + move_direction_y*speed, jax.lax.cond(
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
        move_direction_x, move_direction_y = jax.lax.cond(
            state.player[3]==self._env.consts.SWORD_ID,
            lambda: (-direction_x, -direction_y),
            lambda: (direction_x, direction_y)
        )
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
        dragon_green_freeze = dragon_green_counter % self._env.consts.DRAGON_BITE_TIMER != 0

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
            lambda _: (dragon_green_x + move_direction_x*speed, dragon_green_y + move_direction_y*speed, jax.lax.cond(
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
    
class DragonReviveMod(JaxAtariInternalModPlugin):

    conflicts_with = ["fleaing_dragon"]

    @partial(jax.jit, static_argnums=(0,))
    def _dragon_step(self, state: AdventureState) -> AdventureState:
        speed = self._env.consts.DRAGON_SPEED

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
            jnp.logical_and(dragon_yellow_animation == 1, dragon_yellow_counter >=0),
            lambda f: f+1,
            lambda f:f,
            operand = dragon_yellow_counter
        )
        dragon_yellow_freeze = dragon_yellow_counter % self._env.consts.DRAGON_BITE_TIMER != 0

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
            jnp.logical_and(state.player[2]==dragon_yellow_tile,jnp.logical_and(jnp.logical_not(dragon_yellow_freeze),dragon_yellow_counter>=0)),
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
            jnp.logical_and(dragon_yellow_animation == 2, dragon_yellow_counter >=0),
            lambda _: -1,
            lambda f:f,
            operand=dragon_yellow_counter
        )

        #revive after delay
        dragon_yellow_counter, dragon_yellow_animation = jax.lax.cond(
            dragon_yellow_counter <0 ,
            lambda: jax.lax.cond(
                dragon_yellow_counter == -150,
                lambda: (0, 0),
                lambda: (dragon_yellow_counter - 1, dragon_yellow_animation)
            ),
            lambda: (dragon_yellow_counter, dragon_yellow_animation)
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
        dragon_green_freeze = dragon_green_counter % self._env.consts.DRAGON_BITE_TIMER != 0

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
    
class RandomPlayerSpawnMod(JaxAtariInternalModPlugin):
    spawnlocations = [(78,174,0),(78,174,2),(78,174,3),(78,70,4),(78,100,5),(78,174,6),(78,174,7),(78,174,8),(78,60,9),(78,174,10),(78,174,11)]

    rnd = random.randint(0, 10)
    
    constants_overrides = {
        "PLAYER_SPAWN": spawnlocations[rnd],
    }

class LevelTwoMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        #all rooms in order
        'room_number': {'name': 'room_number', 'type': 'group', 'files': ['Room_1.npy', 
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
                                                           'Room_14.npy',
                                                           'Room_15.npy', 
                                                           'Room_16.npy', 
                                                           'Room_17.npy', 
                                                           'Room_18.npy', 
                                                           'Room_20.npy', 
                                                           'Room_21.npy', 
                                                           'Room_22.npy', 
                                                           'Room_23.npy', 
                                                           'Room_24.npy', 
                                                           'Room_25.npy', 
                                                           'Room_26.npy', 
                                                           'Room_27.npy', 
                                                           'Room_28.npy',
                                                           'Room_29.npy', 
                                                           'Room_30.npy'
                                                           ]},
        #Player in all the different colors, which change depending on the background
        "player_colors": {'name': 'player_colors', 'type': 'group', 'files': ["Player_Yellow.npy",
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
                                                             "Player_Magenta.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_BabyBlue.npy",
                                                             "Player_Turquoise.npy",
                                                             "Player_LightBlue.npy",
                                                             "Player_White.npy",
                                                             "Player_Orange.npy",
                                                             "Player_Orange.npy",
                                                             "Player_Orange.npy",
                                                             "Player_Orange.npy"]},
        #Red Dragon
        "dragon_red": {'name': 'dragon_red', 'type': 'group', 'files': ['Dragon_red_neutral.npy',
                                                             'Dragon_red_attack.npy',
                                                             'Dragon_red_dead.npy']},
        #white Key
        "key_white": {'name': 'key_white', 'type': 'single', 'file': 'Key_white.npy'},

        #bat
        "bat": {'name': 'bat', 'type': 'group', 'files': ['bat_1.npy',
                                                             'bat_2.npy']}
    }

    constants_overrides ={
        "DRAGON_YELLOW_SPAWN": (80, 170, 25, 0, 0, 0, 0),
        "DRAGON_GREEN_SPAWN": (80, 130, 5, 0, 0, 0, 0),
        "DRAGON_RED_SPAWN":  (80, 130, 15, 0, 0, 0, 0),
        "KEY_YELLOW_SPAWN": (31, 170, 19), 
        "KEY_BLACK_SPAWN": (31, 100, 28),
        "KEY_WHITE_SPAWN": (31, 110, 8), 
        "SWORD_SPAWN": (31,180,0),
        "BRIDGE_SPAWN": (65,130,20),
        "MAGNET_SPAWN": (120,180,23),
        "CHALICE_SPAWN": (50,170,15),
        "BAT_SPAWN": (76, 140, 11, 0)
    }

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: AdventureState, action: chex.Array) -> AdventureState:
        def _check_walls_new_rooms(state: AdventureState, direction: int) -> bool:
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

            def is_inverted_walkable(tileset: jnp.ndarray, Pos_x: int, Pos_y: int) -> bool:
                #determin if we should be allowed to walk, based on the background only
                #tileset data at a given x and y position is [r, g, b, 255] 
                #[151, 151, 151, 255] = Grey (allowed player movement) 
                #[0, 0, 0, 255] are top or bottom border allow movement for tilechange
                #anything else are walls (inversed in certain maze tileset) .
                is_walkable_1 = (tileset[Pos_y+2,Pos_x][0] == jnp.uint8(151))
                is_walkable_2 = (tileset[Pos_y+2,Pos_x][1] == jnp.uint8(151))
                is_walkable_3 = (tileset[Pos_y+2,Pos_x][2] == jnp.uint8(151))
                is_walkable = jnp.logical_and(is_walkable_1, jnp.logical_and(is_walkable_2,is_walkable_3))
                #jax.debug.print("Tile: {a} is walkable {b}",a=tileset[Pos_y,Pos_x][0:3], b=is_walkable)
                return is_walkable

            def _load_background_map(path: str) -> jnp.ndarray:
                background_map = jnp.load(path)
                return background_map


            sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites"     

            #jax.debug.print("Room: {a} is equal to 0 {b}, is walkable {c}",a=room, b=(room == 0),c=is_tile_walkable(self.BackgroundRoom1, player_x, player_y))
            in_Room_15_and_walkable = jnp.logical_and(jnp.equal(room, 14), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_15.npy")), player_x, player_y)))
            in_Room_16_and_walkable = jnp.logical_and(jnp.equal(room, 15), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_16.npy")), player_x, player_y)))
            in_Room_17_and_walkable = jnp.logical_and(jnp.equal(room, 16), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_17.npy")), player_x, player_y)))
            in_Room_18_and_walkable = jnp.logical_and(jnp.equal(room, 17), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_18.npy")), player_x, player_y)))
            in_Room_20_and_walkable = jnp.logical_and(jnp.equal(room, 18), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_20.npy")), player_x, player_y)))
            in_Room_21_and_walkable = jnp.logical_and(jnp.equal(room, 19), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_21.npy")), player_x, player_y)))
            in_Room_22_and_walkable = jnp.logical_and(jnp.equal(room, 20), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_22.npy")), player_x, player_y)))
            in_Room_23_and_walkable = jnp.logical_and(jnp.equal(room, 21), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_23.npy")), player_x, player_y))
            in_Room_24_and_walkable = jnp.logical_and(jnp.equal(room, 22), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_24.npy")), player_x, player_y))
            in_Room_25_and_walkable = jnp.logical_and(jnp.equal(room, 23), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_25.npy")), player_x, player_y))
            in_Room_26_and_walkable = jnp.logical_and(jnp.equal(room, 24), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_26.npy")), player_x, player_y))
            in_Room_27_and_walkable = jnp.logical_and(jnp.equal(room, 25), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_27.npy")), player_x, player_y))
            in_Room_28_and_walkable = jnp.logical_and(jnp.equal(room, 26), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_28.npy")), player_x, player_y))
            in_Room_29_and_walkable = jnp.logical_and(jnp.equal(room, 27), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_29.npy")), player_x, player_y))
            in_Room_30_and_walkable = jnp.logical_and(jnp.equal(room, 28), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_30.npy")), player_x, player_y))

            Room_15_or_16_and_walkable = jnp.logical_or(in_Room_15_and_walkable, in_Room_16_and_walkable)
            Room_17_or_18_and_walkable = jnp.logical_or(in_Room_17_and_walkable, in_Room_18_and_walkable)
            Room_20_or_21_and_walkable = jnp.logical_or(in_Room_20_and_walkable, in_Room_21_and_walkable)
            Room_22_or_23_and_walkable = jnp.logical_or(in_Room_22_and_walkable, in_Room_23_and_walkable)
            Room_24_or_25_and_walkable = jnp.logical_or(in_Room_24_and_walkable, in_Room_25_and_walkable)
            Room_26_or_27_and_walkable = jnp.logical_or(in_Room_26_and_walkable, in_Room_27_and_walkable)
            Room_28_or_29_and_walkable = jnp.logical_or(in_Room_28_and_walkable, in_Room_29_and_walkable)

            Room_15_or_16_or_17_or_18_and_walkable = jnp.logical_or(Room_15_or_16_and_walkable, Room_17_or_18_and_walkable)
            Room_20_or_21_or_22_or_23_and_walkable = jnp.logical_or(Room_20_or_21_and_walkable, Room_22_or_23_and_walkable)
            Room_24_or_25_or_26_or_27_and_walkable = jnp.logical_or(Room_24_or_25_and_walkable, Room_26_or_27_and_walkable)

            Room_15_or_16_or_17_or_18_or_20_or_21_or_22_or_23_and_walkable = jnp.logical_or(Room_15_or_16_or_17_or_18_and_walkable, Room_20_or_21_or_22_or_23_and_walkable)
            Room_24_or_25_or_26_or_27_or_28_or_29_or_30_and_walkable = jnp.logical_or(jnp.logical_or(Room_24_or_25_or_26_or_27_and_walkable, Room_28_or_29_and_walkable),in_Room_30_and_walkable)

            current_Room_is_walkable = jnp.logical_or(jnp.logical_or(Room_15_or_16_or_17_or_18_or_20_or_21_or_22_or_23_and_walkable, Room_24_or_25_or_26_or_27_or_28_or_29_or_30_and_walkable), self._env._check_for_wall(state, direction))
            #jax.debug.print("is walkable {a}", a= current_Room_is_walkable)


            edge_left = self._env.consts.PATH_VERTICAL_LEFT
            edge_right = self._env.consts.PATH_VERTICAL_RIGHT

            edge_left = self._env.consts.PATH_VERTICAL_LEFT
            edge_right = self._env.consts.PATH_VERTICAL_RIGHT

            #Castle Collisions
            castle_tower_left = self._env.consts.CASTLE_TOWER_LEFT_X
            castle_tower_right = self._env.consts.CASTLE_TOWER_RIGHT_X
            castle_tower_height = self._env.consts.CASTLE_TOWER_CORNER_Y
            castle_base_left = self._env.consts.CASTLE_BASE_LEFT_X
            castle_base_right = self._env.consts.CASTLE_BASE_RIGHT_X
            castle_base_height = self._env.consts.CASTLE_BASE_CORNER_Y

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

            gate_black_open = state.gate_black[0]

            gate_white_open =state.gate_white[0]

            gate_yellow_not_block = jnp.logical_or(
                jnp.logical_not(room == 0),
                gate_yellow_open > 4
            )

            gate_black_not_block = jnp.logical_or(
                jnp.logical_not(room == 11),
                gate_black_open > 4
            )

            gate_white_not_block = jnp.logical_or(
                jnp.logical_not(room == 24),
                gate_white_open > 4
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
                jnp.logical_not(jnp.logical_or(jnp.logical_or(room==0, room==11), room==24)), #either it is not a castle tile, or
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

        left = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.LEFT, action == Action.LEFTFIRE),action== Action.UPLEFT),action == Action.UPLEFTFIRE), action==Action.DOWNLEFT), action==Action.DOWNLEFTFIRE)
        right = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.RIGHT, action == Action.RIGHTFIRE),action== Action.UPRIGHT),action == Action.UPRIGHTFIRE), action==Action.DOWNRIGHT), action==Action.DOWNRIGHTFIRE)
        up = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.UP, action == Action.UPFIRE),action== Action.UPRIGHT),action == Action.UPRIGHTFIRE), action==Action.UPLEFT), action==Action.UPLEFTFIRE)
        down = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.DOWN, action == Action.DOWNFIRE),action== Action.DOWNRIGHT),action == Action.DOWNRIGHTFIRE), action==Action.DOWNLEFT), action==Action.DOWNLEFTFIRE)

        #check for no wall before walking
        left_no_wall = jnp.logical_and(left,_check_walls_new_rooms(state, 0))
        right_no_wall = jnp.logical_and(right,_check_walls_new_rooms(state, 1))
        up_no_wall =  jnp.logical_and(up,_check_walls_new_rooms(state, 2))
        down_no_wall =  jnp.logical_and(down,_check_walls_new_rooms(state, 3))
        
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
             lambda:state.key_white[0]
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
             lambda:state.key_white[1]
             ]
        )

        new_player_y = state.player[1]
        new_player_y, new_item_y, new_step_counter = jax.lax.cond(
            down_no_wall,
            lambda y: (y[0]+8,y[1]+8,y[2]),
            lambda y: y,
            operand = (new_player_y,new_item_y,new_step_counter)
        )
        new_player_y, new_item_y, new_step_counter = jax.lax.cond(
            up_no_wall,
            lambda y: (y[0]-8,y[1]-8,y[2]),
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
             lambda:state.key_white[2]
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
            lambda: (77, 148, 0, 0, new_item_y-(new_player_y-145),new_item_x+(77-new_player_x)),
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

        #enter white castle
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            jnp.logical_and(new_player_tile == 24, jnp.logical_and(new_player_y <148,jnp.logical_and(new_player_x<110, new_player_x>50))),
            lambda: (212, 25, 25,new_item_y+(212-new_player_y)),
            lambda: (new_player_y, new_player_tile, new_item_tile, new_item_y)
        )

        #leave white castle
        new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x = jax.lax.cond(
            jnp.logical_and(new_player_tile == 25, new_player_y >212),
            lambda: (77, 148, 24, 24, new_item_y-(new_player_y-148),new_item_x+(77-new_player_x)),
            lambda: (new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x)
        )

        #change of rooms
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            new_player_y > 212,
            lambda _: (27, jax.lax.switch( new_player_tile, [lambda:2,lambda:0,lambda:0, 
                                                             lambda:18, lambda:0, lambda:0, 
                                                             lambda:5, lambda:8, lambda:0, 
                                                             lambda: 6, lambda:7, lambda:10, 
                                                             lambda:11, lambda:21, lambda:12,
                                                             lambda:14, lambda:17, lambda:16,
                                                             lambda:19, lambda:20, lambda:0,
                                                             lambda:4, lambda:23, lambda:0,
                                                             lambda:22, lambda:24, lambda:25,
                                                             lambda:28, lambda:24]),
                                                             jax.lax.switch( new_item_tile, [lambda:2,lambda:0,lambda:0, 
                                                             lambda:18, lambda:0, lambda:0, 
                                                             lambda:5, lambda:8, lambda:0, 
                                                             lambda: 6, lambda:7, lambda:10, 
                                                             lambda:11, lambda:21, lambda:12,
                                                             lambda:14, lambda:17, lambda:16,
                                                             lambda:19, lambda:20, lambda:0,
                                                             lambda:4, lambda:23, lambda:0,
                                                             lambda:22, lambda:24, lambda:25,
                                                             lambda:28, lambda:24]), new_item_y-(new_player_y-27)),
            lambda _: (new_player_y, new_player_tile, new_item_tile, new_item_y),
            operand = None,
        )
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            new_player_y < 27,
            lambda _: (212, jax.lax.switch( new_player_tile, [lambda:1,lambda:0,lambda:0, 
                                                              lambda:0, lambda:21, lambda:6, 
                                                              lambda:9, lambda:10, lambda:7, 
                                                              lambda: 0, lambda:11, lambda:12, 
                                                              lambda:14, lambda:0, lambda:15,
                                                              lambda:0, lambda:17, lambda:16,
                                                              lambda:3, lambda:18, lambda:19,
                                                              lambda:13, lambda:24, lambda:22,
                                                              lambda:25, lambda:26, lambda:0,
                                                              lambda:0, lambda:27]),
                                                              jax.lax.switch( new_player_tile, [lambda:1,lambda:0,lambda:0, 
                                                              lambda:0, lambda:21, lambda:6, 
                                                              lambda:9, lambda:10, lambda:7, 
                                                              lambda: 0, lambda:11, lambda:12, 
                                                              lambda:14, lambda:0, lambda:15,
                                                              lambda:0, lambda:17, lambda:16,
                                                              lambda:3, lambda:18, lambda:19,
                                                              lambda:13, lambda:24, lambda:22,
                                                              lambda:25, lambda:26, lambda:0,
                                                              lambda:0, lambda:27]), new_item_y+(212-new_player_y)),
            lambda _: (new_player_y, new_player_tile, new_item_tile, new_item_y),
            operand = None,
        )
        new_player_x, new_player_tile, new_item_tile, new_item_x = jax.lax.cond(
            new_player_x > 160,
            lambda _: (0, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:3, 
                                                            lambda:0, lambda:0, lambda:2, 
                                                            lambda:7, lambda:6, lambda:10, 
                                                            lambda: 8, lambda:9, lambda:0, 
                                                            lambda:0, lambda:0, lambda:16,
                                                            lambda:17, lambda:15, lambda:14,
                                                            lambda:19, lambda:18, lambda:21,
                                                            lambda:0, lambda:20, lambda:0,
                                                            lambda:0, lambda:28, lambda:27,
                                                            lambda:26, lambda:25]),
                                                            jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:3, 
                                                            lambda:0, lambda:0, lambda:2, 
                                                            lambda:7, lambda:6, lambda:10, 
                                                            lambda: 8, lambda:9, lambda:0, 
                                                            lambda:0, lambda:0, lambda:16,
                                                            lambda:17, lambda:15, lambda:14,
                                                            lambda:19, lambda:18, lambda:21,
                                                            lambda:0, lambda:20, lambda:0,
                                                            lambda:0, lambda:28, lambda:27,
                                                            lambda:26, lambda:25]), new_item_x-new_player_x),
            lambda _: (new_player_x, new_player_tile, new_item_tile, new_item_x),
            operand = None,
        )
        new_player_x, new_player_tile, new_item_tile, new_item_x = jax.lax.cond(
            new_player_x < 0,
            lambda _: (160, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:5, 
                                                              lambda:2, lambda:0, lambda:0, 
                                                              lambda:7, lambda:6, lambda:9, 
                                                              lambda: 10, lambda:8, lambda:0, 
                                                              lambda:0, lambda:0, lambda:17,
                                                              lambda:16, lambda:14, lambda:15,
                                                              lambda:19, lambda:18, lambda:22,
                                                              lambda:20, lambda:0, lambda:20,
                                                              lambda:0, lambda:28, lambda:27,
                                                              lambda:26, lambda:25]),
                                                              jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:5, 
                                                              lambda:2, lambda:0, lambda:0, 
                                                              lambda:7, lambda:6, lambda:9, 
                                                              lambda: 10, lambda:8, lambda:0, 
                                                              lambda:0, lambda:0, lambda:17,
                                                              lambda:16, lambda:14, lambda:15,
                                                              lambda:19, lambda:18, lambda:22,
                                                              lambda:20, lambda:0, lambda:20,
                                                              lambda:0, lambda:28, lambda:27,
                                                              lambda:26, lambda:25]), new_item_x+(160-new_player_x)),
            lambda _: (new_player_x, new_player_tile, new_item_tile, new_item_x),
            operand = None,
        )

        return state.replace(
            step_counter = jnp.array(new_step_counter).astype(jnp.int32),
            player = jnp.array([new_player_x,new_player_y,new_player_tile,state.player[3]]).astype(jnp.int32), #SEEMS NOT GOOD
            key_yellow = jax.lax.cond(state.player[3]==self._env.consts.KEY_YELLOW_ID,
                                      lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                      lambda op: op[3],
                                      operand=(new_item_x,new_item_y,new_item_tile,state.key_yellow),
                                      ),
            key_black= jax.lax.cond(state.player[3]==self._env.consts.KEY_BLACK_ID,
                                    lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                    lambda op: op[3],
                                    operand=(new_item_x,new_item_y,new_item_tile,state.key_black)
                                    ),
            key_white = jax.lax.cond(state.player[3]==self._env.consts.KEY_WHITE_ID,
                                      lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                      lambda op: op[3],
                                      operand=(new_item_x,new_item_y,new_item_tile,state.key_white),
                                      ),                        
            sword= jax.lax.cond(state.player[3]==self._env.consts.SWORD_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.sword)
                                ),
            bridge= jax.lax.cond(state.player[3]==self._env.consts.BRIDGE_ID,
                                 lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.bridge)
                                ),
            magnet= jax.lax.cond(state.player[3]==self._env.consts.MAGNET_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.magnet)
                                ),
            chalice= jax.lax.cond(state.player[3]==self._env.consts.CHALICE_ID,
                                  lambda op: jnp.array([op[0],op[1],op[2],op[3]]).astype(jnp.int32),
                                  lambda op: op[4],
                                  operand=(new_item_x,new_item_y,new_item_tile,state.chalice[3],state.chalice)
                                  )
        )
    
    #dragons with bat
    @partial(jax.jit, static_argnums=(0,))
    def _dragon_step(self, state: AdventureState) -> AdventureState:
        speed = self._env.consts.DRAGON_SPEED

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
        dragon_yellow_freeze = dragon_yellow_counter % self._env.consts.DRAGON_BITE_TIMER != 0
    
        #dragon starts looking for plyer room after first encounter
        dragon_yellow_activate = jax.lax.cond(state.player[2] == dragon_yellow_tile, lambda:1, lambda: dragon_yellow_activate)
        rndKey, subkey = jax.random.split(state.rndKey)
        dragon_yellow_x, dragon_yellow_y, dragon_yellow_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_yellow_tile != state.player[2], jnp.logical_not(dragon_yellow_freeze)),dragon_yellow_activate==1),
            lambda: (jax.lax.cond(dragon_yellow_x>156, lambda:4, lambda:dragon_yellow_x +2), 
                     jax.lax.cond(dragon_yellow_y>208, lambda:4, lambda:dragon_yellow_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_yellow_x>156,dragon_yellow_y>208), lambda:jax.random.randint(subkey, (), 0, 28) , lambda:dragon_yellow_tile)),
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
        dragon_green_freeze = dragon_green_counter % self._env.consts.DRAGON_BITE_TIMER != 0

        #dragon starts looking for plyer room after first encounter
        dragon_green_activate = jax.lax.cond(state.player[2] == dragon_green_tile, lambda:1, lambda: dragon_green_activate)
        rndKey, subkey = jax.random.split(rndKey)
        dragon_green_x, dragon_green_y, dragon_green_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_green_tile != state.player[2], jnp.logical_not(dragon_green_freeze)),dragon_green_activate==1),
            lambda: (jax.lax.cond(dragon_green_x>156, lambda:4, lambda:dragon_green_x +2), 
                     jax.lax.cond(dragon_green_y>208, lambda:4, lambda:dragon_green_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_green_x>156,dragon_green_y>208), lambda:jax.random.randint(subkey, (), 0, 28) , lambda:dragon_green_tile)),
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

        #red dragon
        direction_x = jnp.sign(state.player[0] - state.dragon_red[0])
        direction_y = jnp.sign(state.player[1]- state.dragon_red[1])
        dragon_red_x = state.dragon_red[0]
        dragon_red_y = state.dragon_red[1]
        dragon_red_tile = state.dragon_red[2]
        dragon_red_animation = state.dragon_red[3]
        dragon_red_counter = state.dragon_red[4]
        dragon_red_activate = state.dragon_red[6]

        # wait after attack
        dragon_red_counter = jax.lax.cond(
            dragon_red_animation == 1,
            lambda f: f+1,
            lambda f:f,
            operand = dragon_red_counter
        )
        dragon_red_freeze = dragon_red_counter % self._env.consts.DRAGON_BITE_TIMER != 0
    
        #dragon starts looking for plyer room after first encounter
        dragon_red_activate = jax.lax.cond(state.player[2] == dragon_red_tile, lambda:1, lambda: dragon_red_activate)
        rndKey, subkey = jax.random.split(state.rndKey)
        dragon_red_x, dragon_red_y, dragon_red_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_red_tile != state.player[2], jnp.logical_not(dragon_red_freeze)),dragon_red_activate==1),
            lambda: (jax.lax.cond(dragon_red_x>156, lambda:4, lambda:dragon_red_x +2), 
                     jax.lax.cond(dragon_red_y>208, lambda:4, lambda:dragon_red_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_red_x>156,dragon_red_y>208), lambda:jax.random.randint(subkey, (), 0, 28) , lambda:dragon_red_tile)),
            lambda:(dragon_red_x, dragon_red_y, dragon_red_tile)
        )

        #dragon eats player
        dragon_red_eat = jax.lax.cond(
            jnp.logical_and(jnp.logical_and((state.player[0]-dragon_red_x)*direction_x<4,(state.player[1]-dragon_red_y)*direction_y<4),jnp.logical_and(dragon_red_animation==1,jnp.logical_not(dragon_red_freeze))),
            lambda:1,
            lambda:0
        )

        #move towards player and attack
        dragon_red_x, dragon_red_y, dragon_red_animation, dragon_red_counter= jax.lax.cond(
            jnp.logical_and(state.player[2]==dragon_red_tile,jnp.logical_not(dragon_red_freeze)),
            lambda _: (dragon_red_x + direction_x*speed, dragon_red_y + direction_y*speed, jax.lax.cond(
                jnp.logical_and((state.player[0]-dragon_red_x)*direction_x<4,(state.player[1]-dragon_red_y)*direction_y<4),
                lambda _:jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:1, operand = None),
                lambda _:jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:0, operand = None),
                operand = None
            ),0),
            lambda _: (dragon_red_x, dragon_red_y, jax.lax.cond(jnp.logical_not(dragon_red_freeze), lambda _: jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:0, operand = None), lambda _: jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:1, operand = None), operand = None), dragon_red_counter),
            operand  = None
        )

        #kill dragon
        direction_x = jnp.sign(sword_x - state.dragon_red[0])
        direction_y = jnp.sign(sword_y- state.dragon_red[1])
        dragon_red_animation = jax.lax.cond(
            jnp.logical_and(dragon_red_tile==sword_room, jnp.logical_and((sword_x-dragon_red_x)*direction_x<4, (sword_y-dragon_red_y)*direction_y<22)),
            lambda _:2,
            lambda a:a,
            operand= dragon_red_animation
        )

        # dont ever move again when dead
        dragon_red_counter = jax.lax.cond(
            dragon_red_animation == 2,
            lambda _: 1,
            lambda f:f,
            operand=dragon_red_counter
        )


        return state.replace(
            dragon_yellow = jnp.array([dragon_yellow_x,dragon_yellow_y,dragon_yellow_tile,dragon_yellow_animation,dragon_yellow_counter,dragon_yellow_eat, dragon_yellow_activate]).astype(jnp.int32),
            dragon_green = jnp.array([dragon_green_x,dragon_green_y,dragon_green_tile,dragon_green_animation,dragon_green_counter,dragon_green_eat, dragon_green_activate]).astype(jnp.int32),
            dragon_red = jnp.array([dragon_red_x,dragon_red_y,dragon_red_tile,dragon_red_animation,dragon_red_counter,dragon_red_eat, dragon_red_activate]).astype(jnp.int32),
            rndKey=rndKey
        )
    
class LevelThreeMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        #all rooms in order
        'room_number': {'name': 'room_number', 'type': 'group', 'files': ['Room_1.npy', 
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
                                                           'Room_14.npy',
                                                           'Room_15.npy', 
                                                           'Room_16.npy', 
                                                           'Room_17.npy', 
                                                           'Room_18.npy', 
                                                           'Room_20.npy', 
                                                           'Room_21.npy', 
                                                           'Room_22.npy', 
                                                           'Room_23.npy', 
                                                           'Room_24.npy', 
                                                           'Room_25.npy', 
                                                           'Room_26.npy', 
                                                           'Room_27.npy', 
                                                           'Room_28.npy',
                                                           'Room_29.npy', 
                                                           'Room_30.npy'
                                                           ]},
        #Player in all the different colors, which change depending on the background
        "player_colors": {'name': 'player_colors', 'type': 'group', 'files': ["Player_Yellow.npy",
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
                                                             "Player_Magenta.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_BabyBlue.npy",
                                                             "Player_Turquoise.npy",
                                                             "Player_LightBlue.npy",
                                                             "Player_White.npy",
                                                             "Player_Orange.npy",
                                                             "Player_Orange.npy",
                                                             "Player_Orange.npy",
                                                             "Player_Orange.npy"]},
        #Red Dragon
        "dragon_red": {'name': 'dragon_red', 'type': 'group', 'files': ['Dragon_red_neutral.npy',
                                                             'Dragon_red_attack.npy',
                                                             'Dragon_red_dead.npy']},
        #white Key
        "key_white": {'name': 'key_white', 'type': 'single', 'file': 'Key_white.npy'},

        #bat
        "bat": {'name': 'bat', 'type': 'group', 'files': ['bat_1.npy',
                                                             'bat_2.npy']}
    }

    rndSpawnLocations = [(31,170,19),(31,100,28),(31,110,8),(31,180,0),(65,130,20),(120,180,23),(50,170,15)]
    random.shuffle(rndSpawnLocations)
    constants_overrides ={
        

        "DRAGON_YELLOW_SPAWN": (80, 170, 25, 0, 0, 0, 0),       
        "DRAGON_GREEN_SPAWN": (80, 130, 5, 0, 0, 0, 0),     
        "DRAGON_RED_SPAWN":  (80, 130, 15, 0, 0, 0, 0),     
        "KEY_YELLOW_SPAWN": rndSpawnLocations[0],      
        "KEY_BLACK_SPAWN": rndSpawnLocations[1],       
        "KEY_WHITE_SPAWN": rndSpawnLocations[2],        
        "SWORD_SPAWN": rndSpawnLocations[3],      
        "BRIDGE_SPAWN": rndSpawnLocations[4],        
        "MAGNET_SPAWN": rndSpawnLocations[5],       
        "CHALICE_SPAWN": rndSpawnLocations[6],       
        "BAT_SPAWN": (76, 140, 11, 0)       
    }

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: AdventureState, action: chex.Array) -> AdventureState:
        def _check_walls_new_rooms(state: AdventureState, direction: int) -> bool:
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

            def is_inverted_walkable(tileset: jnp.ndarray, Pos_x: int, Pos_y: int) -> bool:
                #determin if we should be allowed to walk, based on the background only
                #tileset data at a given x and y position is [r, g, b, 255] 
                #[151, 151, 151, 255] = Grey (allowed player movement) 
                #[0, 0, 0, 255] are top or bottom border allow movement for tilechange
                #anything else are walls (inversed in certain maze tileset) .
                is_walkable_1 = (tileset[Pos_y+2,Pos_x][0] == jnp.uint8(151))
                is_walkable_2 = (tileset[Pos_y+2,Pos_x][1] == jnp.uint8(151))
                is_walkable_3 = (tileset[Pos_y+2,Pos_x][2] == jnp.uint8(151))
                is_walkable = jnp.logical_and(is_walkable_1, jnp.logical_and(is_walkable_2,is_walkable_3))
                #jax.debug.print("Tile: {a} is walkable {b}",a=tileset[Pos_y,Pos_x][0:3], b=is_walkable)
                return is_walkable

            def _load_background_map(path: str) -> jnp.ndarray:
                background_map = jnp.load(path)
                return background_map


            sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites"     

            #jax.debug.print("Room: {a} is equal to 0 {b}, is walkable {c}",a=room, b=(room == 0),c=is_tile_walkable(self.BackgroundRoom1, player_x, player_y))
            in_Room_15_and_walkable = jnp.logical_and(jnp.equal(room, 14), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_15.npy")), player_x, player_y)))
            in_Room_16_and_walkable = jnp.logical_and(jnp.equal(room, 15), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_16.npy")), player_x, player_y)))
            in_Room_17_and_walkable = jnp.logical_and(jnp.equal(room, 16), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_17.npy")), player_x, player_y)))
            in_Room_18_and_walkable = jnp.logical_and(jnp.equal(room, 17), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_18.npy")), player_x, player_y)))
            in_Room_20_and_walkable = jnp.logical_and(jnp.equal(room, 18), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_20.npy")), player_x, player_y)))
            in_Room_21_and_walkable = jnp.logical_and(jnp.equal(room, 19), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_21.npy")), player_x, player_y)))
            in_Room_22_and_walkable = jnp.logical_and(jnp.equal(room, 20), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_22.npy")), player_x, player_y)))
            in_Room_23_and_walkable = jnp.logical_and(jnp.equal(room, 21), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_23.npy")), player_x, player_y))
            in_Room_24_and_walkable = jnp.logical_and(jnp.equal(room, 22), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_24.npy")), player_x, player_y))
            in_Room_25_and_walkable = jnp.logical_and(jnp.equal(room, 23), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_25.npy")), player_x, player_y))
            in_Room_26_and_walkable = jnp.logical_and(jnp.equal(room, 24), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_26.npy")), player_x, player_y))
            in_Room_27_and_walkable = jnp.logical_and(jnp.equal(room, 25), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_27.npy")), player_x, player_y))
            in_Room_28_and_walkable = jnp.logical_and(jnp.equal(room, 26), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_28.npy")), player_x, player_y))
            in_Room_29_and_walkable = jnp.logical_and(jnp.equal(room, 27), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_29.npy")), player_x, player_y))
            in_Room_30_and_walkable = jnp.logical_and(jnp.equal(room, 28), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_30.npy")), player_x, player_y))

            Room_15_or_16_and_walkable = jnp.logical_or(in_Room_15_and_walkable, in_Room_16_and_walkable)
            Room_17_or_18_and_walkable = jnp.logical_or(in_Room_17_and_walkable, in_Room_18_and_walkable)
            Room_20_or_21_and_walkable = jnp.logical_or(in_Room_20_and_walkable, in_Room_21_and_walkable)
            Room_22_or_23_and_walkable = jnp.logical_or(in_Room_22_and_walkable, in_Room_23_and_walkable)
            Room_24_or_25_and_walkable = jnp.logical_or(in_Room_24_and_walkable, in_Room_25_and_walkable)
            Room_26_or_27_and_walkable = jnp.logical_or(in_Room_26_and_walkable, in_Room_27_and_walkable)
            Room_28_or_29_and_walkable = jnp.logical_or(in_Room_28_and_walkable, in_Room_29_and_walkable)

            Room_15_or_16_or_17_or_18_and_walkable = jnp.logical_or(Room_15_or_16_and_walkable, Room_17_or_18_and_walkable)
            Room_20_or_21_or_22_or_23_and_walkable = jnp.logical_or(Room_20_or_21_and_walkable, Room_22_or_23_and_walkable)
            Room_24_or_25_or_26_or_27_and_walkable = jnp.logical_or(Room_24_or_25_and_walkable, Room_26_or_27_and_walkable)

            Room_15_or_16_or_17_or_18_or_20_or_21_or_22_or_23_and_walkable = jnp.logical_or(Room_15_or_16_or_17_or_18_and_walkable, Room_20_or_21_or_22_or_23_and_walkable)
            Room_24_or_25_or_26_or_27_or_28_or_29_or_30_and_walkable = jnp.logical_or(jnp.logical_or(Room_24_or_25_or_26_or_27_and_walkable, Room_28_or_29_and_walkable),in_Room_30_and_walkable)

            current_Room_is_walkable = jnp.logical_or(jnp.logical_or(Room_15_or_16_or_17_or_18_or_20_or_21_or_22_or_23_and_walkable, Room_24_or_25_or_26_or_27_or_28_or_29_or_30_and_walkable), self._env._check_for_wall(state, direction))
            #jax.debug.print("is walkable {a}", a= current_Room_is_walkable)


            edge_left = self._env.consts.PATH_VERTICAL_LEFT
            edge_right = self._env.consts.PATH_VERTICAL_RIGHT

            edge_left = self._env.consts.PATH_VERTICAL_LEFT
            edge_right = self._env.consts.PATH_VERTICAL_RIGHT

            #Castle Collisions
            castle_tower_left = self._env.consts.CASTLE_TOWER_LEFT_X
            castle_tower_right = self._env.consts.CASTLE_TOWER_RIGHT_X
            castle_tower_height = self._env.consts.CASTLE_TOWER_CORNER_Y
            castle_base_left = self._env.consts.CASTLE_BASE_LEFT_X
            castle_base_right = self._env.consts.CASTLE_BASE_RIGHT_X
            castle_base_height = self._env.consts.CASTLE_BASE_CORNER_Y

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

            gate_black_open = state.gate_black[0]

            gate_white_open =state.gate_white[0]

            gate_yellow_not_block = jnp.logical_or(
                jnp.logical_not(room == 0),
                gate_yellow_open > 4
            )

            gate_black_not_block = jnp.logical_or(
                jnp.logical_not(room == 11),
                gate_black_open > 4
            )

            gate_white_not_block = jnp.logical_or(
                jnp.logical_not(room == 24),
                gate_white_open > 4
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
                jnp.logical_not(jnp.logical_or(jnp.logical_or(room==0, room==11), room==24)), #either it is not a castle tile, or
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

        left = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.LEFT, action == Action.LEFTFIRE),action== Action.UPLEFT),action == Action.UPLEFTFIRE), action==Action.DOWNLEFT), action==Action.DOWNLEFTFIRE)
        right = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.RIGHT, action == Action.RIGHTFIRE),action== Action.UPRIGHT),action == Action.UPRIGHTFIRE), action==Action.DOWNRIGHT), action==Action.DOWNRIGHTFIRE)
        up = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.UP, action == Action.UPFIRE),action== Action.UPRIGHT),action == Action.UPRIGHTFIRE), action==Action.UPLEFT), action==Action.UPLEFTFIRE)
        down = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.DOWN, action == Action.DOWNFIRE),action== Action.DOWNRIGHT),action == Action.DOWNRIGHTFIRE), action==Action.DOWNLEFT), action==Action.DOWNLEFTFIRE)

        #check for no wall before walking
        left_no_wall = jnp.logical_and(left,_check_walls_new_rooms(state, 0))
        right_no_wall = jnp.logical_and(right,_check_walls_new_rooms(state, 1))
        up_no_wall =  jnp.logical_and(up,_check_walls_new_rooms(state, 2))
        down_no_wall =  jnp.logical_and(down,_check_walls_new_rooms(state, 3))
        
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
             lambda:state.key_white[0]
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
             lambda:state.key_white[1]
             ]
        )

        new_player_y = state.player[1]
        new_player_y, new_item_y, new_step_counter = jax.lax.cond(
            down_no_wall,
            lambda y: (y[0]+8,y[1]+8,y[2]),
            lambda y: y,
            operand = (new_player_y,new_item_y,new_step_counter)
        )
        new_player_y, new_item_y, new_step_counter = jax.lax.cond(
            up_no_wall,
            lambda y: (y[0]-8,y[1]-8,y[2]),
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
             lambda:state.key_white[2]
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
            lambda: (77, 148, 0, 0, new_item_y-(new_player_y-145),new_item_x+(77-new_player_x)),
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

        #enter white castle
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            jnp.logical_and(new_player_tile == 24, jnp.logical_and(new_player_y <148,jnp.logical_and(new_player_x<110, new_player_x>50))),
            lambda: (212, 25, 25,new_item_y+(212-new_player_y)),
            lambda: (new_player_y, new_player_tile, new_item_tile, new_item_y)
        )

        #leave white castle
        new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x = jax.lax.cond(
            jnp.logical_and(new_player_tile == 25, new_player_y >212),
            lambda: (77, 148, 24, 24, new_item_y-(new_player_y-148),new_item_x+(77-new_player_x)),
            lambda: (new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x)
        )

        #change of rooms
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            new_player_y > 212,
            lambda _: (27, jax.lax.switch( new_player_tile, [lambda:2,lambda:0,lambda:0, 
                                                             lambda:18, lambda:0, lambda:0, 
                                                             lambda:5, lambda:8, lambda:0, 
                                                             lambda: 6, lambda:7, lambda:10, 
                                                             lambda:11, lambda:21, lambda:12,
                                                             lambda:14, lambda:17, lambda:16,
                                                             lambda:19, lambda:20, lambda:0,
                                                             lambda:4, lambda:23, lambda:0,
                                                             lambda:22, lambda:24, lambda:25,
                                                             lambda:28, lambda:24]),
                                                             jax.lax.switch( new_item_tile, [lambda:2,lambda:0,lambda:0, 
                                                             lambda:18, lambda:0, lambda:0, 
                                                             lambda:5, lambda:8, lambda:0, 
                                                             lambda: 6, lambda:7, lambda:10, 
                                                             lambda:11, lambda:21, lambda:12,
                                                             lambda:14, lambda:17, lambda:16,
                                                             lambda:19, lambda:20, lambda:0,
                                                             lambda:4, lambda:23, lambda:0,
                                                             lambda:22, lambda:24, lambda:25,
                                                             lambda:28, lambda:24]), new_item_y-(new_player_y-27)),
            lambda _: (new_player_y, new_player_tile, new_item_tile, new_item_y),
            operand = None,
        )
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            new_player_y < 27,
            lambda _: (212, jax.lax.switch( new_player_tile, [lambda:1,lambda:0,lambda:0, 
                                                              lambda:0, lambda:21, lambda:6, 
                                                              lambda:9, lambda:10, lambda:7, 
                                                              lambda: 0, lambda:11, lambda:12, 
                                                              lambda:14, lambda:0, lambda:15,
                                                              lambda:0, lambda:17, lambda:16,
                                                              lambda:3, lambda:18, lambda:19,
                                                              lambda:13, lambda:24, lambda:22,
                                                              lambda:25, lambda:26, lambda:0,
                                                              lambda:0, lambda:27]),
                                                              jax.lax.switch( new_player_tile, [lambda:1,lambda:0,lambda:0, 
                                                              lambda:0, lambda:21, lambda:6, 
                                                              lambda:9, lambda:10, lambda:7, 
                                                              lambda: 0, lambda:11, lambda:12, 
                                                              lambda:14, lambda:0, lambda:15,
                                                              lambda:0, lambda:17, lambda:16,
                                                              lambda:3, lambda:18, lambda:19,
                                                              lambda:13, lambda:24, lambda:22,
                                                              lambda:25, lambda:26, lambda:0,
                                                              lambda:0, lambda:27]), new_item_y+(212-new_player_y)),
            lambda _: (new_player_y, new_player_tile, new_item_tile, new_item_y),
            operand = None,
        )
        new_player_x, new_player_tile, new_item_tile, new_item_x = jax.lax.cond(
            new_player_x > 160,
            lambda _: (0, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:3, 
                                                            lambda:0, lambda:0, lambda:2, 
                                                            lambda:7, lambda:6, lambda:10, 
                                                            lambda: 8, lambda:9, lambda:0, 
                                                            lambda:0, lambda:0, lambda:16,
                                                            lambda:17, lambda:15, lambda:14,
                                                            lambda:19, lambda:18, lambda:21,
                                                            lambda:0, lambda:20, lambda:0,
                                                            lambda:0, lambda:28, lambda:27,
                                                            lambda:26, lambda:25]),
                                                            jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:3, 
                                                            lambda:0, lambda:0, lambda:2, 
                                                            lambda:7, lambda:6, lambda:10, 
                                                            lambda: 8, lambda:9, lambda:0, 
                                                            lambda:0, lambda:0, lambda:16,
                                                            lambda:17, lambda:15, lambda:14,
                                                            lambda:19, lambda:18, lambda:21,
                                                            lambda:0, lambda:20, lambda:0,
                                                            lambda:0, lambda:28, lambda:27,
                                                            lambda:26, lambda:25]), new_item_x-new_player_x),
            lambda _: (new_player_x, new_player_tile, new_item_tile, new_item_x),
            operand = None,
        )
        new_player_x, new_player_tile, new_item_tile, new_item_x = jax.lax.cond(
            new_player_x < 0,
            lambda _: (160, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:5, 
                                                              lambda:2, lambda:0, lambda:0, 
                                                              lambda:7, lambda:6, lambda:9, 
                                                              lambda: 10, lambda:8, lambda:0, 
                                                              lambda:0, lambda:0, lambda:17,
                                                              lambda:16, lambda:14, lambda:15,
                                                              lambda:19, lambda:18, lambda:22,
                                                              lambda:20, lambda:0, lambda:20,
                                                              lambda:0, lambda:28, lambda:27,
                                                              lambda:26, lambda:25]),
                                                              jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:5, 
                                                              lambda:2, lambda:0, lambda:0, 
                                                              lambda:7, lambda:6, lambda:9, 
                                                              lambda: 10, lambda:8, lambda:0, 
                                                              lambda:0, lambda:0, lambda:17,
                                                              lambda:16, lambda:14, lambda:15,
                                                              lambda:19, lambda:18, lambda:22,
                                                              lambda:20, lambda:0, lambda:20,
                                                              lambda:0, lambda:28, lambda:27,
                                                              lambda:26, lambda:25]), new_item_x+(160-new_player_x)),
            lambda _: (new_player_x, new_player_tile, new_item_tile, new_item_x),
            operand = None,
        )

        return state.replace(
            step_counter = jnp.array(new_step_counter).astype(jnp.int32),
            player = jnp.array([new_player_x,new_player_y,new_player_tile,state.player[3]]).astype(jnp.int32), #SEEMS NOT GOOD
            key_yellow = jax.lax.cond(state.player[3]==self._env.consts.KEY_YELLOW_ID,
                                      lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                      lambda op: op[3],
                                      operand=(new_item_x,new_item_y,new_item_tile,state.key_yellow),
                                      ),
            key_black= jax.lax.cond(state.player[3]==self._env.consts.KEY_BLACK_ID,
                                    lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                    lambda op: op[3],
                                    operand=(new_item_x,new_item_y,new_item_tile,state.key_black)
                                    ),
            key_white = jax.lax.cond(state.player[3]==self._env.consts.KEY_WHITE_ID,
                                      lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                      lambda op: op[3],
                                      operand=(new_item_x,new_item_y,new_item_tile,state.key_white),
                                      ),                        
            sword= jax.lax.cond(state.player[3]==self._env.consts.SWORD_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.sword)
                                ),
            bridge= jax.lax.cond(state.player[3]==self._env.consts.BRIDGE_ID,
                                 lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.bridge)
                                ),
            magnet= jax.lax.cond(state.player[3]==self._env.consts.MAGNET_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.magnet)
                                ),
            chalice= jax.lax.cond(state.player[3]==self._env.consts.CHALICE_ID,
                                  lambda op: jnp.array([op[0],op[1],op[2],op[3]]).astype(jnp.int32),
                                  lambda op: op[4],
                                  operand=(new_item_x,new_item_y,new_item_tile,state.chalice[3],state.chalice)
                                  )
        )
    
    #dragons with bat
    @partial(jax.jit, static_argnums=(0,))
    def _dragon_step(self, state: AdventureState) -> AdventureState:
        speed = self._env.consts.DRAGON_SPEED

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
        dragon_yellow_freeze = dragon_yellow_counter % self._env.consts.DRAGON_BITE_TIMER != 0
    
        #dragon starts looking for plyer room after first encounter
        dragon_yellow_activate = jax.lax.cond(state.player[2] == dragon_yellow_tile, lambda:1, lambda: dragon_yellow_activate)
        rndKey, subkey = jax.random.split(state.rndKey)
        dragon_yellow_x, dragon_yellow_y, dragon_yellow_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_yellow_tile != state.player[2], jnp.logical_not(dragon_yellow_freeze)),dragon_yellow_activate==1),
            lambda: (jax.lax.cond(dragon_yellow_x>156, lambda:4, lambda:dragon_yellow_x +2), 
                     jax.lax.cond(dragon_yellow_y>208, lambda:4, lambda:dragon_yellow_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_yellow_x>156,dragon_yellow_y>208), lambda:jax.random.randint(subkey, (), 0, 28) , lambda:dragon_yellow_tile)),
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
        dragon_green_freeze = dragon_green_counter % self._env.consts.DRAGON_BITE_TIMER != 0

        #dragon starts looking for plyer room after first encounter
        dragon_green_activate = jax.lax.cond(state.player[2] == dragon_green_tile, lambda:1, lambda: dragon_green_activate)
        rndKey, subkey = jax.random.split(rndKey)
        dragon_green_x, dragon_green_y, dragon_green_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_green_tile != state.player[2], jnp.logical_not(dragon_green_freeze)),dragon_green_activate==1),
            lambda: (jax.lax.cond(dragon_green_x>156, lambda:4, lambda:dragon_green_x +2), 
                     jax.lax.cond(dragon_green_y>208, lambda:4, lambda:dragon_green_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_green_x>156,dragon_green_y>208), lambda:jax.random.randint(subkey, (), 0, 28) , lambda:dragon_green_tile)),
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

        #red dragon
        direction_x = jnp.sign(state.player[0] - state.dragon_red[0])
        direction_y = jnp.sign(state.player[1]- state.dragon_red[1])
        dragon_red_x = state.dragon_red[0]
        dragon_red_y = state.dragon_red[1]
        dragon_red_tile = state.dragon_red[2]
        dragon_red_animation = state.dragon_red[3]
        dragon_red_counter = state.dragon_red[4]
        dragon_red_activate = state.dragon_red[6]

        # wait after attack
        dragon_red_counter = jax.lax.cond(
            dragon_red_animation == 1,
            lambda f: f+1,
            lambda f:f,
            operand = dragon_red_counter
        )
        dragon_red_freeze = dragon_red_counter % self._env.consts.DRAGON_BITE_TIMER != 0
    
        #dragon starts looking for plyer room after first encounter
        dragon_red_activate = jax.lax.cond(state.player[2] == dragon_red_tile, lambda:1, lambda: dragon_red_activate)
        rndKey, subkey = jax.random.split(state.rndKey)
        dragon_red_x, dragon_red_y, dragon_red_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_red_tile != state.player[2], jnp.logical_not(dragon_red_freeze)),dragon_red_activate==1),
            lambda: (jax.lax.cond(dragon_red_x>156, lambda:4, lambda:dragon_red_x +2), 
                     jax.lax.cond(dragon_red_y>208, lambda:4, lambda:dragon_red_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_red_x>156,dragon_red_y>208), lambda:jax.random.randint(subkey, (), 0, 28) , lambda:dragon_red_tile)),
            lambda:(dragon_red_x, dragon_red_y, dragon_red_tile)
        )

        #dragon eats player
        dragon_red_eat = jax.lax.cond(
            jnp.logical_and(jnp.logical_and((state.player[0]-dragon_red_x)*direction_x<4,(state.player[1]-dragon_red_y)*direction_y<4),jnp.logical_and(dragon_red_animation==1,jnp.logical_not(dragon_red_freeze))),
            lambda:1,
            lambda:0
        )

        #move towards player and attack
        dragon_red_x, dragon_red_y, dragon_red_animation, dragon_red_counter= jax.lax.cond(
            jnp.logical_and(state.player[2]==dragon_red_tile,jnp.logical_not(dragon_red_freeze)),
            lambda _: (dragon_red_x + direction_x*speed, dragon_red_y + direction_y*speed, jax.lax.cond(
                jnp.logical_and((state.player[0]-dragon_red_x)*direction_x<4,(state.player[1]-dragon_red_y)*direction_y<4),
                lambda _:jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:1, operand = None),
                lambda _:jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:0, operand = None),
                operand = None
            ),0),
            lambda _: (dragon_red_x, dragon_red_y, jax.lax.cond(jnp.logical_not(dragon_red_freeze), lambda _: jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:0, operand = None), lambda _: jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:1, operand = None), operand = None), dragon_red_counter),
            operand  = None
        )

        #kill dragon
        direction_x = jnp.sign(sword_x - state.dragon_red[0])
        direction_y = jnp.sign(sword_y- state.dragon_red[1])
        dragon_red_animation = jax.lax.cond(
            jnp.logical_and(dragon_red_tile==sword_room, jnp.logical_and((sword_x-dragon_red_x)*direction_x<4, (sword_y-dragon_red_y)*direction_y<22)),
            lambda _:2,
            lambda a:a,
            operand= dragon_red_animation
        )

        # dont ever move again when dead
        dragon_red_counter = jax.lax.cond(
            dragon_red_animation == 2,
            lambda _: 1,
            lambda f:f,
            operand=dragon_red_counter
        )


        return state.replace(
            dragon_yellow = jnp.array([dragon_yellow_x,dragon_yellow_y,dragon_yellow_tile,dragon_yellow_animation,dragon_yellow_counter,dragon_yellow_eat, dragon_yellow_activate]).astype(jnp.int32),
            dragon_green = jnp.array([dragon_green_x,dragon_green_y,dragon_green_tile,dragon_green_animation,dragon_green_counter,dragon_green_eat, dragon_green_activate]).astype(jnp.int32),
            dragon_red = jnp.array([dragon_red_x,dragon_red_y,dragon_red_tile,dragon_red_animation,dragon_red_counter,dragon_red_eat, dragon_red_activate]).astype(jnp.int32),
            rndKey=rndKey
        )
    
class EasterEggMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        #all rooms in order
        'room_number': {'name': 'room_number', 'type': 'group', 'files': ['Room_1.npy', 
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
                                                           'Room_14.npy',
                                                           'Room_15.npy', 
                                                           'Room_16.npy', 
                                                           'Room_17.npy', 
                                                           'Room_18.npy', 
                                                           'Room_20.npy', 
                                                           'Room_21.npy', 
                                                           'Room_22.npy', 
                                                           'Room_23.npy', 
                                                           'Room_24.npy', 
                                                           'Room_25.npy', 
                                                           'Room_26.npy', 
                                                           'Room_27.npy', 
                                                           'Room_28.npy',
                                                           'Room_29.npy', 
                                                           'Room_30.npy',
                                                           'Room_31.npy'
                                                           ]},
        #Player in all the different colors, which change depending on the background
        "player_colors": {'name': 'player_colors', 'type': 'group', 'files': ["Player_Yellow.npy",
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
                                                             "Player_Magenta.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_inverted.npy",
                                                             "Player_BabyBlue.npy",
                                                             "Player_Turquoise.npy",
                                                             "Player_LightBlue.npy",
                                                             "Player_White.npy",
                                                             "Player_Orange.npy",
                                                             "Player_Orange.npy",
                                                             "Player_Orange.npy",
                                                             "Player_Orange.npy"]},
        #Red Dragon
        "dragon_red": {'name': 'dragon_red', 'type': 'group', 'files': ['Dragon_red_neutral.npy',
                                                             'Dragon_red_attack.npy',
                                                             'Dragon_red_dead.npy']},
        #white Key
        "key_white": {'name': 'key_white', 'type': 'single', 'file': 'Key_white.npy'},

        #bat
        "bat": {'name': 'bat', 'type': 'group', 'files': ['bat_1.npy',
                                                             'bat_2.npy']},
        #dot
        "dot": {'name': 'dot', 'type': 'single', 'file': 'Player_BabyBlue.npy'} #TODO change File
    }

    rndSpawnLocations = [(31,170,19),(31,100,28),(31,110,8),(31,180,0),(65,130,20),(120,180,23),(50,170,15)]
    random.shuffle(rndSpawnLocations)
    constants_overrides ={
        

        "DRAGON_YELLOW_SPAWN": (80, 170, 25, 0, 0, 0, 0),       
        "DRAGON_GREEN_SPAWN": (80, 130, 5, 0, 0, 0, 0),     
        "DRAGON_RED_SPAWN":  (80, 130, 15, 0, 0, 0, 0),     
        "KEY_YELLOW_SPAWN": rndSpawnLocations[0],      
        "KEY_BLACK_SPAWN": rndSpawnLocations[1],       
        "KEY_WHITE_SPAWN": rndSpawnLocations[2],        
        "SWORD_SPAWN": rndSpawnLocations[3],      
        "BRIDGE_SPAWN": rndSpawnLocations[4],        
        "MAGNET_SPAWN": rndSpawnLocations[5],       
        "CHALICE_SPAWN": rndSpawnLocations[6],       
        "BAT_SPAWN": (76, 140, 11, 0),
        "DOT_SPAWN": (50, 130, 15)
    }

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: AdventureState, action: chex.Array) -> AdventureState:
        def _check_walls_new_rooms(state: AdventureState, direction: int) -> bool:
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

            def is_inverted_walkable(tileset: jnp.ndarray, Pos_x: int, Pos_y: int) -> bool:
                #determin if we should be allowed to walk, based on the background only
                #tileset data at a given x and y position is [r, g, b, 255] 
                #[151, 151, 151, 255] = Grey (allowed player movement) 
                #[0, 0, 0, 255] are top or bottom border allow movement for tilechange
                #anything else are walls (inversed in certain maze tileset) .
                is_walkable_1 = (tileset[Pos_y+2,Pos_x][0] == jnp.uint8(151))
                is_walkable_2 = (tileset[Pos_y+2,Pos_x][1] == jnp.uint8(151))
                is_walkable_3 = (tileset[Pos_y+2,Pos_x][2] == jnp.uint8(151))
                is_walkable = jnp.logical_and(is_walkable_1, jnp.logical_and(is_walkable_2,is_walkable_3))
                #jax.debug.print("Tile: {a} is walkable {b}",a=tileset[Pos_y,Pos_x][0:3], b=is_walkable)
                return is_walkable

            def _load_background_map(path: str) -> jnp.ndarray:
                background_map = jnp.load(path)
                return background_map


            sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites"     

            #jax.debug.print("Room: {a} is equal to 0 {b}, is walkable {c}",a=room, b=(room == 0),c=is_tile_walkable(self.BackgroundRoom1, player_x, player_y))
            in_Room_15_and_walkable = jnp.logical_and(jnp.equal(room, 14), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_15.npy")), player_x, player_y)))
            in_Room_16_and_walkable = jnp.logical_and(jnp.equal(room, 15), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_16.npy")), player_x, player_y)))
            in_Room_17_and_walkable = jnp.logical_and(jnp.equal(room, 16), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_17.npy")), player_x, player_y)))
            in_Room_18_and_walkable = jnp.logical_and(jnp.equal(room, 17), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_18.npy")), player_x, player_y)))
            in_Room_20_and_walkable = jnp.logical_and(jnp.equal(room, 18), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_20.npy")), player_x, player_y)))
            in_Room_21_and_walkable = jnp.logical_and(jnp.equal(room, 19), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_21.npy")), player_x, player_y)))
            in_Room_22_and_walkable = jnp.logical_and(jnp.equal(room, 20), jnp.logical_not(is_inverted_walkable(_load_background_map(os.path.join(sprite_path, "Room_22.npy")), player_x, player_y)))
            in_Room_23_and_walkable = jnp.logical_and(jnp.equal(room, 21), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_23.npy")), player_x, player_y))
            in_Room_24_and_walkable = jnp.logical_and(jnp.equal(room, 22), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_24.npy")), player_x, player_y))
            in_Room_25_and_walkable = jnp.logical_and(jnp.equal(room, 23), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_25.npy")), player_x, player_y))
            in_Room_26_and_walkable = jnp.logical_and(jnp.equal(room, 24), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_26.npy")), player_x, player_y))
            in_Room_27_and_walkable = jnp.logical_and(jnp.equal(room, 25), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_27.npy")), player_x, player_y))
            in_Room_28_and_walkable = jnp.logical_and(jnp.equal(room, 26), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_28.npy")), player_x, player_y))
            in_Room_29_and_walkable = jnp.logical_and(jnp.equal(room, 27), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_29.npy")), player_x, player_y))
            in_Room_30_and_walkable = jnp.logical_and(jnp.equal(room, 28), is_tile_walkable(_load_background_map(os.path.join(sprite_path, "Room_30.npy")), player_x, player_y))

            Room_15_or_16_and_walkable = jnp.logical_or(in_Room_15_and_walkable, in_Room_16_and_walkable)
            Room_17_or_18_and_walkable = jnp.logical_or(in_Room_17_and_walkable, in_Room_18_and_walkable)
            Room_20_or_21_and_walkable = jnp.logical_or(in_Room_20_and_walkable, in_Room_21_and_walkable)
            Room_22_or_23_and_walkable = jnp.logical_or(in_Room_22_and_walkable, in_Room_23_and_walkable)
            Room_24_or_25_and_walkable = jnp.logical_or(in_Room_24_and_walkable, in_Room_25_and_walkable)
            Room_26_or_27_and_walkable = jnp.logical_or(in_Room_26_and_walkable, in_Room_27_and_walkable)
            Room_28_or_29_and_walkable = jnp.logical_or(in_Room_28_and_walkable, in_Room_29_and_walkable)

            Room_15_or_16_or_17_or_18_and_walkable = jnp.logical_or(Room_15_or_16_and_walkable, Room_17_or_18_and_walkable)
            Room_20_or_21_or_22_or_23_and_walkable = jnp.logical_or(Room_20_or_21_and_walkable, Room_22_or_23_and_walkable)
            Room_24_or_25_or_26_or_27_and_walkable = jnp.logical_or(Room_24_or_25_and_walkable, Room_26_or_27_and_walkable)

            Room_15_or_16_or_17_or_18_or_20_or_21_or_22_or_23_and_walkable = jnp.logical_or(Room_15_or_16_or_17_or_18_and_walkable, Room_20_or_21_or_22_or_23_and_walkable)
            Room_24_or_25_or_26_or_27_or_28_or_29_or_30_and_walkable = jnp.logical_or(jnp.logical_or(Room_24_or_25_or_26_or_27_and_walkable, Room_28_or_29_and_walkable),in_Room_30_and_walkable)

            current_Room_is_walkable = jnp.logical_or(jnp.logical_or(Room_15_or_16_or_17_or_18_or_20_or_21_or_22_or_23_and_walkable, Room_24_or_25_or_26_or_27_or_28_or_29_or_30_and_walkable), self._env._check_for_wall(state, direction))
            #jax.debug.print("is walkable {a}", a= current_Room_is_walkable)


            edge_left = self._env.consts.PATH_VERTICAL_LEFT
            edge_right = self._env.consts.PATH_VERTICAL_RIGHT

            edge_left = self._env.consts.PATH_VERTICAL_LEFT
            edge_right = self._env.consts.PATH_VERTICAL_RIGHT

            #Castle Collisions
            castle_tower_left = self._env.consts.CASTLE_TOWER_LEFT_X
            castle_tower_right = self._env.consts.CASTLE_TOWER_RIGHT_X
            castle_tower_height = self._env.consts.CASTLE_TOWER_CORNER_Y
            castle_base_left = self._env.consts.CASTLE_BASE_LEFT_X
            castle_base_right = self._env.consts.CASTLE_BASE_RIGHT_X
            castle_base_height = self._env.consts.CASTLE_BASE_CORNER_Y

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

            gate_black_open = state.gate_black[0]

            gate_white_open =state.gate_white[0]

            gate_yellow_not_block = jnp.logical_or(
                jnp.logical_not(room == 0),
                gate_yellow_open > 4
            )

            gate_black_not_block = jnp.logical_or(
                jnp.logical_not(room == 11),
                gate_black_open > 4
            )

            gate_white_not_block = jnp.logical_or(
                jnp.logical_not(room == 24),
                gate_white_open > 4
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
                jnp.logical_not(jnp.logical_or(jnp.logical_or(room==0, room==11), room==24)), #either it is not a castle tile, or
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

        left = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.LEFT, action == Action.LEFTFIRE),action== Action.UPLEFT),action == Action.UPLEFTFIRE), action==Action.DOWNLEFT), action==Action.DOWNLEFTFIRE)
        right = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.RIGHT, action == Action.RIGHTFIRE),action== Action.UPRIGHT),action == Action.UPRIGHTFIRE), action==Action.DOWNRIGHT), action==Action.DOWNRIGHTFIRE)
        up = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.UP, action == Action.UPFIRE),action== Action.UPRIGHT),action == Action.UPRIGHTFIRE), action==Action.UPLEFT), action==Action.UPLEFTFIRE)
        down = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action== Action.DOWN, action == Action.DOWNFIRE),action== Action.DOWNRIGHT),action == Action.DOWNRIGHTFIRE), action==Action.DOWNLEFT), action==Action.DOWNLEFTFIRE)

        #check for no wall before walking
        left_no_wall = jnp.logical_and(left,_check_walls_new_rooms(state, 0))
        right_no_wall = jnp.logical_and(right,_check_walls_new_rooms(state, 1))
        up_no_wall =  jnp.logical_and(up,_check_walls_new_rooms(state, 2))
        down_no_wall =  jnp.logical_and(down,_check_walls_new_rooms(state, 3))
        
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
            lambda y: (y[0]+8,y[1]+8,y[2]),
            lambda y: y,
            operand = (new_player_y,new_item_y,new_step_counter)
        )
        new_player_y, new_item_y, new_step_counter = jax.lax.cond(
            up_no_wall,
            lambda y: (y[0]-8,y[1]-8,y[2]),
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
            lambda: (77, 148, 0, 0, new_item_y-(new_player_y-145),new_item_x+(77-new_player_x)),
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

        #enter white castle
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            jnp.logical_and(new_player_tile == 24, jnp.logical_and(new_player_y <148,jnp.logical_and(new_player_x<110, new_player_x>50))),
            lambda: (212, 25, 25,new_item_y+(212-new_player_y)),
            lambda: (new_player_y, new_player_tile, new_item_tile, new_item_y)
        )

        #leave white castle
        new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x = jax.lax.cond(
            jnp.logical_and(new_player_tile == 25, new_player_y >212),
            lambda: (77, 148, 24, 24, new_item_y-(new_player_y-148),new_item_x+(77-new_player_x)),
            lambda: (new_player_x, new_player_y, new_player_tile, new_item_tile, new_item_y, new_item_x)
        )

        #change of rooms
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            new_player_y > 212,
            lambda _: (27, jax.lax.switch( new_player_tile, [lambda:2,lambda:0,lambda:0, 
                                                             lambda:18, lambda:0, lambda:0, 
                                                             lambda:5, lambda:8, lambda:0, 
                                                             lambda: 6, lambda:7, lambda:10, 
                                                             lambda:11, lambda:21, lambda:12,
                                                             lambda:14, lambda:17, lambda:16,
                                                             lambda:19, lambda:20, lambda:0,
                                                             lambda:4, lambda:23, lambda:0,
                                                             lambda:22, lambda:24, lambda:25,
                                                             lambda:28, lambda:24]),
                                                             jax.lax.switch( new_item_tile, [lambda:2,lambda:0,lambda:0, 
                                                             lambda:18, lambda:0, lambda:0, 
                                                             lambda:5, lambda:8, lambda:0, 
                                                             lambda: 6, lambda:7, lambda:10, 
                                                             lambda:11, lambda:21, lambda:12,
                                                             lambda:14, lambda:17, lambda:16,
                                                             lambda:19, lambda:20, lambda:0,
                                                             lambda:4, lambda:23, lambda:0,
                                                             lambda:22, lambda:24, lambda:25,
                                                             lambda:28, lambda:24]), new_item_y-(new_player_y-27)),
            lambda _: (new_player_y, new_player_tile, new_item_tile, new_item_y),
            operand = None,
        )
        new_player_y, new_player_tile, new_item_tile, new_item_y = jax.lax.cond(
            new_player_y < 27,
            lambda _: (212, jax.lax.switch( new_player_tile, [lambda:1,lambda:0,lambda:0, 
                                                              lambda:0, lambda:21, lambda:6, 
                                                              lambda:9, lambda:10, lambda:7, 
                                                              lambda: 0, lambda:11, lambda:12, 
                                                              lambda:14, lambda:0, lambda:15,
                                                              lambda:0, lambda:17, lambda:16,
                                                              lambda:3, lambda:18, lambda:19,
                                                              lambda:13, lambda:24, lambda:22,
                                                              lambda:25, lambda:26, lambda:0,
                                                              lambda:0, lambda:27]),
                                                              jax.lax.switch( new_player_tile, [lambda:1,lambda:0,lambda:0, 
                                                              lambda:0, lambda:21, lambda:6, 
                                                              lambda:9, lambda:10, lambda:7, 
                                                              lambda: 0, lambda:11, lambda:12, 
                                                              lambda:14, lambda:0, lambda:15,
                                                              lambda:0, lambda:17, lambda:16,
                                                              lambda:3, lambda:18, lambda:19,
                                                              lambda:13, lambda:24, lambda:22,
                                                              lambda:25, lambda:26, lambda:0,
                                                              lambda:0, lambda:27]), new_item_y+(212-new_player_y)),
            lambda _: (new_player_y, new_player_tile, new_item_tile, new_item_y),
            operand = None,
        )
        new_player_x, new_player_tile, new_item_tile, new_item_x = jax.lax.cond(
            new_player_x > 160,
            lambda _: (0, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:3, 
                                                            lambda:0, lambda:0, lambda:2, 
                                                            lambda:7, lambda:6, lambda:10, 
                                                            lambda: 8, lambda:9, lambda:0, 
                                                            lambda:0, lambda:0, lambda:16,
                                                            lambda:17, lambda:15, lambda:14,
                                                            lambda:19, lambda:18, lambda:21,
                                                            lambda:0, lambda:20, lambda:0,
                                                            lambda:0, lambda:28, lambda:27,
                                                            lambda:26, lambda:25]),
                                                            jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:3, 
                                                            lambda:0, lambda:0, lambda:2, 
                                                            lambda:7, lambda:6, lambda:10, 
                                                            lambda: 8, lambda:9, lambda:0, 
                                                            lambda:0, lambda:0, lambda:16,
                                                            lambda:17, lambda:15, lambda:14,
                                                            lambda:19, lambda:18, lambda:21,
                                                            lambda:0, lambda:20, lambda:0,
                                                            lambda:0, lambda:28, lambda:27,
                                                            lambda:26, lambda:25]), new_item_x-new_player_x),
            lambda _: (new_player_x, new_player_tile, new_item_tile, new_item_x),
            operand = None,
        )
        new_player_x, new_player_tile, new_item_tile, new_item_x = jax.lax.cond(
            new_player_x < 0,
            lambda _: (160, jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:5, 
                                                              lambda:2, lambda:0, lambda:0, 
                                                              lambda:7, lambda:6, lambda:9, 
                                                              lambda: 10, lambda:8, lambda:0, 
                                                              lambda:0, lambda:0, lambda:17,
                                                              lambda:16, lambda:14, lambda:15,
                                                              lambda:19, lambda:18, lambda:22,
                                                              lambda:20, lambda:0, lambda:20,
                                                              lambda:0, lambda:28, lambda:27,
                                                              lambda:26, lambda:25]),
                                                              jax.lax.switch( new_player_tile, [lambda:0,lambda:0,lambda:5, 
                                                              lambda:2, lambda:0, lambda:0, 
                                                              lambda:7, lambda:6, lambda:9, 
                                                              lambda: 10, lambda:8, lambda:0, 
                                                              lambda:0, lambda:0, lambda:17,
                                                              lambda:16, lambda:14, lambda:15,
                                                              lambda:19, lambda:18, lambda:22,
                                                              lambda:20, lambda:0, lambda:20,
                                                              lambda:0, lambda:28, lambda:27,
                                                              lambda:26, lambda:25]), new_item_x+(160-new_player_x)),
            lambda _: (new_player_x, new_player_tile, new_item_tile, new_item_x),
            operand = None,
        )

        new_player_tile, new_player_x = jax.lax.cond(
            jnp.logical_and(state.player[3]==self._env.consts.DOT_ID, jnp.logical_and(state.player[2]==3,state.player[0]>140)),
            lambda: (29, 27),
            lambda: (new_player_tile, new_player_x)
        )

        return state.replace(
            step_counter = jnp.array(new_step_counter).astype(jnp.int32),
            player = jnp.array([new_player_x,new_player_y,new_player_tile,state.player[3]]).astype(jnp.int32), #SEEMS NOT GOOD
            key_yellow = jax.lax.cond(state.player[3]==self._env.consts.KEY_YELLOW_ID,
                                      lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                      lambda op: op[3],
                                      operand=(new_item_x,new_item_y,new_item_tile,state.key_yellow),
                                      ),
            key_black= jax.lax.cond(state.player[3]==self._env.consts.KEY_BLACK_ID,
                                    lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                    lambda op: op[3],
                                    operand=(new_item_x,new_item_y,new_item_tile,state.key_black)
                                    ),
            key_white = jax.lax.cond(state.player[3]==self._env.consts.KEY_WHITE_ID,
                                      lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                      lambda op: op[3],
                                      operand=(new_item_x,new_item_y,new_item_tile,state.key_white),
                                      ),                        
            sword= jax.lax.cond(state.player[3]==self._env.consts.SWORD_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.sword)
                                ),
            bridge= jax.lax.cond(state.player[3]==self._env.consts.BRIDGE_ID,
                                 lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.bridge)
                                ),
            magnet= jax.lax.cond(state.player[3]==self._env.consts.MAGNET_ID,
                                lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                lambda op: op[3],
                                operand=(new_item_x,new_item_y,new_item_tile,state.magnet)
                                ),
            chalice= jax.lax.cond(state.player[3]==self._env.consts.CHALICE_ID,
                                  lambda op: jnp.array([op[0],op[1],op[2],op[3]]).astype(jnp.int32),
                                  lambda op: op[4],
                                  operand=(new_item_x,new_item_y,new_item_tile,state.chalice[3],state.chalice)
                                  ),
            dot= jax.lax.cond(state.player[3]==self._env.consts.DOT_ID,
                                  lambda op: jnp.array([op[0],op[1],op[2]]).astype(jnp.int32),
                                  lambda op: op[3],
                                  operand=(new_item_x,new_item_y,new_item_tile,state.dot)
                                  )
        )
    
    #dragons with bat
    @partial(jax.jit, static_argnums=(0,))
    def _dragon_step(self, state: AdventureState) -> AdventureState:
        speed = self._env.consts.DRAGON_SPEED

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
        dragon_yellow_freeze = dragon_yellow_counter % self._env.consts.DRAGON_BITE_TIMER != 0
    
        #dragon starts looking for plyer room after first encounter
        dragon_yellow_activate = jax.lax.cond(state.player[2] == dragon_yellow_tile, lambda:1, lambda: dragon_yellow_activate)
        rndKey, subkey = jax.random.split(state.rndKey)
        dragon_yellow_x, dragon_yellow_y, dragon_yellow_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_yellow_tile != state.player[2], jnp.logical_not(dragon_yellow_freeze)),dragon_yellow_activate==1),
            lambda: (jax.lax.cond(dragon_yellow_x>156, lambda:4, lambda:dragon_yellow_x +2), 
                     jax.lax.cond(dragon_yellow_y>208, lambda:4, lambda:dragon_yellow_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_yellow_x>156,dragon_yellow_y>208), lambda:jax.random.randint(subkey, (), 0, 28) , lambda:dragon_yellow_tile)),
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
        dragon_green_freeze = dragon_green_counter % self._env.consts.DRAGON_BITE_TIMER != 0

        #dragon starts looking for plyer room after first encounter
        dragon_green_activate = jax.lax.cond(state.player[2] == dragon_green_tile, lambda:1, lambda: dragon_green_activate)
        rndKey, subkey = jax.random.split(rndKey)
        dragon_green_x, dragon_green_y, dragon_green_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_green_tile != state.player[2], jnp.logical_not(dragon_green_freeze)),dragon_green_activate==1),
            lambda: (jax.lax.cond(dragon_green_x>156, lambda:4, lambda:dragon_green_x +2), 
                     jax.lax.cond(dragon_green_y>208, lambda:4, lambda:dragon_green_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_green_x>156,dragon_green_y>208), lambda:jax.random.randint(subkey, (), 0, 28) , lambda:dragon_green_tile)),
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

        #red dragon
        direction_x = jnp.sign(state.player[0] - state.dragon_red[0])
        direction_y = jnp.sign(state.player[1]- state.dragon_red[1])
        dragon_red_x = state.dragon_red[0]
        dragon_red_y = state.dragon_red[1]
        dragon_red_tile = state.dragon_red[2]
        dragon_red_animation = state.dragon_red[3]
        dragon_red_counter = state.dragon_red[4]
        dragon_red_activate = state.dragon_red[6]

        # wait after attack
        dragon_red_counter = jax.lax.cond(
            dragon_red_animation == 1,
            lambda f: f+1,
            lambda f:f,
            operand = dragon_red_counter
        )
        dragon_red_freeze = dragon_red_counter % self._env.consts.DRAGON_BITE_TIMER != 0
    
        #dragon starts looking for plyer room after first encounter
        dragon_red_activate = jax.lax.cond(state.player[2] == dragon_red_tile, lambda:1, lambda: dragon_red_activate)
        rndKey, subkey = jax.random.split(state.rndKey)
        dragon_red_x, dragon_red_y, dragon_red_tile = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(dragon_red_tile != state.player[2], jnp.logical_not(dragon_red_freeze)),dragon_red_activate==1),
            lambda: (jax.lax.cond(dragon_red_x>156, lambda:4, lambda:dragon_red_x +2), 
                     jax.lax.cond(dragon_red_y>208, lambda:4, lambda:dragon_red_y+2), 
                     jax.lax.cond(jnp.logical_or(dragon_red_x>156,dragon_red_y>208), lambda:jax.random.randint(subkey, (), 0, 28) , lambda:dragon_red_tile)),
            lambda:(dragon_red_x, dragon_red_y, dragon_red_tile)
        )

        #dragon eats player
        dragon_red_eat = jax.lax.cond(
            jnp.logical_and(jnp.logical_and((state.player[0]-dragon_red_x)*direction_x<4,(state.player[1]-dragon_red_y)*direction_y<4),jnp.logical_and(dragon_red_animation==1,jnp.logical_not(dragon_red_freeze))),
            lambda:1,
            lambda:0
        )

        #move towards player and attack
        dragon_red_x, dragon_red_y, dragon_red_animation, dragon_red_counter= jax.lax.cond(
            jnp.logical_and(state.player[2]==dragon_red_tile,jnp.logical_not(dragon_red_freeze)),
            lambda _: (dragon_red_x + direction_x*speed, dragon_red_y + direction_y*speed, jax.lax.cond(
                jnp.logical_and((state.player[0]-dragon_red_x)*direction_x<4,(state.player[1]-dragon_red_y)*direction_y<4),
                lambda _:jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:1, operand = None),
                lambda _:jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:0, operand = None),
                operand = None
            ),0),
            lambda _: (dragon_red_x, dragon_red_y, jax.lax.cond(jnp.logical_not(dragon_red_freeze), lambda _: jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:0, operand = None), lambda _: jax.lax.cond(dragon_red_animation==2, lambda _:2, lambda _:1, operand = None), operand = None), dragon_red_counter),
            operand  = None
        )

        #kill dragon
        direction_x = jnp.sign(sword_x - state.dragon_red[0])
        direction_y = jnp.sign(sword_y- state.dragon_red[1])
        dragon_red_animation = jax.lax.cond(
            jnp.logical_and(dragon_red_tile==sword_room, jnp.logical_and((sword_x-dragon_red_x)*direction_x<4, (sword_y-dragon_red_y)*direction_y<22)),
            lambda _:2,
            lambda a:a,
            operand= dragon_red_animation
        )

        # dont ever move again when dead
        dragon_red_counter = jax.lax.cond(
            dragon_red_animation == 2,
            lambda _: 1,
            lambda f:f,
            operand=dragon_red_counter
        )


        return state.replace(
            dragon_yellow = jnp.array([dragon_yellow_x,dragon_yellow_y,dragon_yellow_tile,dragon_yellow_animation,dragon_yellow_counter,dragon_yellow_eat, dragon_yellow_activate]).astype(jnp.int32),
            dragon_green = jnp.array([dragon_green_x,dragon_green_y,dragon_green_tile,dragon_green_animation,dragon_green_counter,dragon_green_eat, dragon_green_activate]).astype(jnp.int32),
            dragon_red = jnp.array([dragon_red_x,dragon_red_y,dragon_red_tile,dragon_red_animation,dragon_red_counter,dragon_red_eat, dragon_red_activate]).astype(jnp.int32),
            rndKey=rndKey
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AdventureState, state: AdventureState):
        reward = jax.lax.cond(
            jnp.logical_or(jnp.logical_or(state.dragon_yellow[5]==1,state.dragon_green[5]==1),state.dragon_red[5]==1), #lose when eaten by dragon
            lambda :-1,
            lambda : jax.lax.cond(
                state.player[2]==29, #win when player is in easteregg room
                lambda :state.step_counter,
                lambda :0 #this should happen on reset?
            )
        )
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AdventureState) -> bool:
        return jnp.logical_or(jnp.logical_or(jnp.logical_or(state.dragon_yellow[5]==1,state.dragon_green[5]==1),state.dragon_red[5]==1), state.player[2]==29)
