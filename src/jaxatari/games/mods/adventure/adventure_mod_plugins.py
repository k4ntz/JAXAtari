import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_adventure import AdventureState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import chex
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
