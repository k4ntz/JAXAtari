import os
from enum import IntEnum
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
from PIL.ImageChops import logical_and
from tensorstore import int32

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils_legacy as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    floor: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    
    

class HauntedHouseConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210

    Y_WALKING_DISTANCE: int = 2  # This is compensation for the stretched pixels in the ALE version. However, the game
                                 # feels much better when this is set to 1. Possible values: [1, 2]


    WALL_Y_OFFSET: int = 4
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    WALL_COLOR_BLUE: chex.Array = jnp.array([24, 26, 167, 255], dtype=jnp.uint8)
    WALL_COLOR_RED: chex.Array = jnp.array([163, 57, 21, 255], dtype=jnp.uint8)
    WALL_COLOR_GREEN: chex.Array = jnp.array([24, 98, 78, 255], dtype=jnp.uint8)
    WALL_COLOR_YELLOW: chex.Array = jnp.array([162, 134, 56, 255], dtype=jnp.uint8)
    WALL_COLOR_WHITE: chex.Array = jnp.array([255, 255, 255, 255], dtype=jnp.uint8)
    LIGHT_COLOR: chex.Array = jnp.array([198, 108, 58, 255], dtype=jnp.uint8)
    ITEM_OFFSETS: chex.Array = jnp.array([(0, 0), (0, 2), (2, 0), (5, 2),
                                          (0, 0), (2, 0), (0, 2), (0, 0)], dtype=jnp.uint8)


    STAIRS_TOP_LEFT: chex.Array = jnp.array([True, True, True, True], dtype=jnp.bool)
    STAIRS_TOP_RIGHT: chex.Array = jnp.array([False, True, True, False], dtype=jnp.bool)
    STAIRS_BOTTOM_LEFT: chex.Array = jnp.array([False, True, True, False], dtype=jnp.bool)
    STAIRS_BOTTOM_RIGHT: chex.Array = jnp.array([True, True, True, True], dtype=jnp.bool)
    STAIRS_LEFT: chex.Array = jnp.array([False, True, True, False], dtype=jnp.bool)
    STAIRS_RIGHT: chex.Array = jnp.array([False, False, True, True], dtype=jnp.bool)

    DIRECTION: chex.Array = jnp.array([
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
], dtype=jnp.int32)

    ADJACENCY_MATRIX: chex.Array = jnp.array([(0,0,0,0),                                         #0
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
                                              ], dtype=jnp.int32)
    LOCATIONS: chex.Array = jnp.array([(0, 0, 1),                                         #0
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
                                       ], dtype=jnp.int32)

    UP_STAIRS: chex.Array = jnp.array([7, 8, 16, 17, 19, 26, 29, 31])
    DOWN_STAIRS: chex.Array = jnp.array([15, 18, 27, 28, 30, 38, 39, 40])
    ROOMS: chex.Array = jnp.array([1,2,3,5,6,9,10,11,12,13,14,20,21,22,23,24,25,32,33,34,35,36,37])
    STAIR_TRANSITIONS: chex. Array = jnp.array([0,0,0,0,0,0,0,15,18,0,0, #0-10
                                                0,0,0,0,7,27,28,8,30,0,  #11-20
                                                0,0,0,0,0,38,16,17,39,19, #21-30
                                                40,0,0,0,0,0,0,26,29,31]) #31-40

    PLAYER_SIZE: Tuple[int, int] = (8, 6)
    GHOST_SIZE: Tuple[int, int] = (8, 16)
    SPIDER_SIZE: Tuple[int, int] = (8, 16)
    BAT_SIZE: Tuple[int, int] = (8, 20)
    SCEPTER_SIZE: Tuple[int, int] = (8, 16)
    URN_LEFT_SIZE: Tuple[int, int] = (4, 8)
    URN_MIDDLE_SIZE: Tuple[int, int] = (4, 16)
    URN_RIGHT_SIZE: Tuple[int, int] = (4, 8)
    URN_LEFT_MIDDLE_SIZE: Tuple[int, int] = (6, 16)
    URN_MIDDLE_RIGHT_SIZE: Tuple[int, int] = (6, 16)
    URN_LEFT_RIGHT_SIZE: Tuple[int, int] = (8, 8)
    URN_SIZE: Tuple[int, int] = (8, 16)
    ITEM_SIZES: chex.Array = jnp.array([SCEPTER_SIZE, URN_LEFT_SIZE, URN_MIDDLE_SIZE, URN_RIGHT_SIZE,
                                        URN_LEFT_MIDDLE_SIZE, URN_MIDDLE_RIGHT_SIZE, URN_LEFT_RIGHT_SIZE, URN_SIZE])

    # The -2 is a jank hack that allows it to always pass same_floor checks
    ROOM_POSITIONS: chex.Array = jnp.array([(0, 0, -2), (80, 0, -2), (0, 178, -2), (80, 178, -2), (0, 338, -2), (80, 338, -2)])
    ROOM_SIZES: chex.Array = jnp.array([(80, 177), (80, 177), (80, 161), (80, 161), (80, 177), (80, 177)])

    # Used for collision detection
    WALL: chex.Array = jnp.any(jr.loadFrame(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites/hauntedhouse/Wall.npy")) == WALL_COLOR_BLUE, axis=-1)
    LIGHT: chex.Array = jnp.any(jr.loadFrame(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites/hauntedhouse/SpriteLight1.npy")) == LIGHT_COLOR, axis=-1)

    INVISIBLE_ENTITY: EntityPosition = EntityPosition(x=jnp.array(-1), y=jnp.array(-1), floor=jnp.array(-1), width=jnp.array(-1), height=jnp.array(-1))


# immutable state container
class HauntedHouseState(NamedTuple):
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






class HauntedHouseObservation(NamedTuple):
    player: EntityPosition
    ghost: EntityPosition
    spider: EntityPosition
    bat: EntityPosition
    item_held: jnp.ndarray
    scepter: EntityPosition
    urnleft: EntityPosition
    urnmiddle: EntityPosition
    urnright: EntityPosition
    urnleftmiddle: EntityPosition
    urnmiddleright: EntityPosition
    urnleftright: EntityPosition
    urn: EntityPosition
    match_duration: jnp.ndarray
    matches_used: jnp.ndarray
    chasing: jnp.ndarray
    lives: jnp.ndarray


class HauntedHouseInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array





class JaxHauntedHouse(JaxEnvironment[HauntedHouseState, HauntedHouseObservation, HauntedHouseInfo, HauntedHouseConstants]):
    def __init__(self, consts: HauntedHouseConstants = None, reward_funcs: list[callable]=None):
        consts = consts or HauntedHouseConstants()
        super().__init__(consts)
        self.renderer = HauntedHouseRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPFIRE,
            Action.DOWNFIRE,
            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE,
            Action.DOWNLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
            Action.UPLEFT,
            Action.UPRIGHT,
            Action.DOWNLEFT,
            Action.DOWNRIGHT
        ]
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
        pos = jnp.array([pos[0] - player[0] + 12, pos[1] - player[1] + 24])
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

        # Step 1: Player Mechanics
        (player, player_direction, stun_duration, match_duration, matches_used,
         item_dropped, stairs_active, fire_button_active, lives, game_ends) = self.player_step(state, action)

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
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info




    def render(self, state: HauntedHouseState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: HauntedHouseState):
        player = EntityPosition(
            x=state.player[0],
            y=state.player[1],
            floor=state.player[2],
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )

        ghost = EntityPosition(
            x=state.ghost[0],
            y=state.ghost[1],
            floor=state.ghost[2],
            width=jnp.array(self.consts.GHOST_SIZE[0]),
            height=jnp.array(self.consts.GHOST_SIZE[1]),
        )

        spider = EntityPosition(
            x=state.spider[0],
            y=state.spider[1],
            floor=state.spider[2],
            width=jnp.array(self.consts.SPIDER_SIZE[0]),
            height=jnp.array(self.consts.SPIDER_SIZE[1]),
        )

        bat = EntityPosition(
            x=state.bat[0],
            y=state.bat[1],
            floor=state.bat[2],
            width=jnp.array(self.consts.BAT_SIZE[0]),
            height=jnp.array(self.consts.BAT_SIZE[1]),
        )

        scepter = EntityPosition(
            x=state.scepter[0],
            y=state.scepter[1],
            floor=state.scepter[2],
            width=jnp.array(self.consts.SCEPTER_SIZE[0]),
            height=jnp.array(self.consts.SCEPTER_SIZE[1]),
        )

        urnleft = EntityPosition(
            x=state.urn_left[0],
            y=state.urn_left[1],
            floor=state.urn_left[2],
            width=jnp.array(self.consts.URN_LEFT_SIZE[0]),
            height=jnp.array(self.consts.URN_LEFT_SIZE[1]),
        )

        urnmiddle = EntityPosition(
            x=state.urn_middle[0],
            y=state.urn_middle[1],
            floor=state.urn_middle[2],
            width=jnp.array(self.consts.URN_MIDDLE_SIZE[0]),
            height=jnp.array(self.consts.URN_MIDDLE_SIZE[1]),
        )

        urnright = EntityPosition(
            x=state.urn_right[0],
            y=state.urn_right[1],
            floor=state.urn_right[2],
            width=jnp.array(self.consts.URN_RIGHT_SIZE[0]),
            height=jnp.array(self.consts.URN_RIGHT_SIZE[1]),
        )

        urnleftmiddle = EntityPosition(
            x=state.urn_left_middle[0],
            y=state.urn_left_middle[1],
            floor=state.urn_left_middle[2],
            width=jnp.array(self.consts.URN_LEFT_MIDDLE_SIZE[0]),
            height=jnp.array(self.consts.URN_LEFT_MIDDLE_SIZE[1]),
        )

        urnmiddleright = EntityPosition(
            x=state.urn_middle_right[0],
            y=state.urn_middle_right[1],
            floor=state.urn_middle_right[2],
            width=jnp.array(self.consts.URN_MIDDLE_RIGHT_SIZE[0]),
            height=jnp.array(self.consts.URN_MIDDLE_RIGHT_SIZE[1]),
        )

        urnleftright = EntityPosition(
            x=state.urn_left_right[0],
            y=state.urn_left_right[1],
            floor=state.urn_left_right[2],
            width=jnp.array(self.consts.URN_LEFT_RIGHT_SIZE[0]),
            height=jnp.array(self.consts.URN_LEFT_RIGHT_SIZE[1]),
        )

        urn = EntityPosition(
            x=state.urn[0],
            y=state.urn[1],
            floor=state.urn[2],
            width=jnp.array(self.consts.URN_SIZE[0]),
            height=jnp.array(self.consts.URN_SIZE[1]),
        )

        visible = jax.lax.cond(state.player[2] == state.ghost[2],
                               lambda x: self.check_visibility(state.player, ghost),
                               lambda x: False, None)
        ghost = jax.lax.cond(visible, lambda x: ghost, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.spider[2],
                               lambda x: self.check_visibility(state.player, spider),
                               lambda x: False, None)
        spider = jax.lax.cond(visible, lambda x: spider, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.bat[2],
                               lambda x: self.check_visibility(state.player, bat),
                               lambda x: False, None)
        bat = jax.lax.cond(visible, lambda x: bat, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.scepter[2],
                               lambda x: self.check_illuminated_single(player, state.scepter, self.consts.SCEPTER_SIZE),
                               lambda x: False, None)
        scepter = jax.lax.cond(visible, lambda x: scepter, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.ghost[2],
                               lambda x: self.check_illuminated_single(player, state.urn_left, self.consts.URN_LEFT_SIZE),
                               lambda x: False, None)
        urnleft = jax.lax.cond(visible, lambda x: urnleft, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.ghost[2],
                               lambda x: self.check_illuminated_single(player, state.urn_middle, self.consts.URN_MIDDLE_SIZE),
                               lambda x: False, None)
        urnmiddle = jax.lax.cond(visible, lambda x: urnmiddle, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.ghost[2],
                               lambda x: self.check_illuminated_single(player, state.urn_right, self.consts.URN_RIGHT_SIZE),
                               lambda x: False, None)
        urnright = jax.lax.cond(visible, lambda x: urnright, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.ghost[2],
                               lambda x: self.check_illuminated_single(player, state.urn_left_middle, self.consts.URN_LEFT_MIDDLE_SIZE),
                               lambda x: False, None)
        urnleftmiddle = jax.lax.cond(visible, lambda x: urnleftmiddle, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.ghost[2],
                               lambda x: self.check_illuminated_single(player, state.urn_middle_right, self.consts.URN_MIDDLE_RIGHT_SIZE),
                               lambda x: False, None)
        urnmiddleright = jax.lax.cond(visible, lambda x: urnmiddleright, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.ghost[2],
                               lambda x: self.check_illuminated_single(player, state.urn_left_right, self.consts.URN_LEFT_RIGHT_SIZE),
                               lambda x: False, None)
        urnleftright = jax.lax.cond(visible, lambda x: urnleftright, lambda x: self.consts.INVISIBLE_ENTITY, None)

        visible = jax.lax.cond(state.player[2] == state.ghost[2],
                               lambda x: self.check_illuminated_single(player, state.urn, self.consts.URN_SIZE),
                               lambda x: False, None)
        urn = jax.lax.cond(visible, lambda x: urn, lambda x: self.consts.INVISIBLE_ENTITY, None)



        return HauntedHouseObservation(
            player=player,
            ghost=ghost,
            spider=spider,
            bat=bat,
            item_held=state.item_held,
            scepter=scepter,
            urnleft=urnleft,
            urnmiddle=urnmiddle,
            urnright=urnright,
            urnleftmiddle=urnleftmiddle,
            urnmiddleright=urnmiddleright,
            urnleftright=urnleftright,
            urn=urn,
            match_duration=state.match_duration > 0,
            matches_used=state.matches_used,
            chasing=state.chasing > 0,
            lives=state.lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: HauntedHouseObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.floor.flatten(),
            obs.player.width.flatten(),
            obs.player.height.flatten(),

            obs.ghost.x.flatten(),
            obs.ghost.y.flatten(),
            obs.ghost.floor.flatten(),
            obs.ghost.width.flatten(),
            obs.ghost.height.flatten(),

            obs.spider.x.flatten(),
            obs.spider.y.flatten(),
            obs.spider.floor.flatten(),
            obs.spider.width.flatten(),
            obs.spider.height.flatten(),

            obs.bat.x.flatten(),
            obs.bat.y.flatten(),
            obs.bat.floor.flatten(),
            obs.bat.width.flatten(),
            obs.bat.height.flatten(),

            obs.item_held.flatten(),

            obs.scepter.x.flatten(),
            obs.scepter.y.flatten(),
            obs.scepter.floor.flatten(),
            obs.scepter.width.flatten(),
            obs.scepter.height.flatten(),

            obs.urnleft.x.flatten(),
            obs.urnleft.y.flatten(),
            obs.urnleft.floor.flatten(),
            obs.urnleft.width.flatten(),
            obs.urnleft.height.flatten(),

            obs.urnmiddle.x.flatten(),
            obs.urnmiddle.y.flatten(),
            obs.urnmiddle.floor.flatten(),
            obs.urnmiddle.width.flatten(),
            obs.urnmiddle.height.flatten(),

            obs.urnright.x.flatten(),
            obs.urnright.y.flatten(),
            obs.urnright.floor.flatten(),
            obs.urnright.width.flatten(),
            obs.urnright.height.flatten(),

            obs.urnleftmiddle.x.flatten(),
            obs.urnleftmiddle.y.flatten(),
            obs.urnleftmiddle.floor.flatten(),
            obs.urnleftmiddle.width.flatten(),
            obs.urnleftmiddle.height.flatten(),

            obs.urnmiddleright.x.flatten(),
            obs.urnmiddleright.y.flatten(),
            obs.urnmiddleright.floor.flatten(),
            obs.urnmiddleright.width.flatten(),
            obs.urnmiddleright.height.flatten(),

            obs.urnleftright.x.flatten(),
            obs.urnleftright.y.flatten(),
            obs.urnleftright.floor.flatten(),
            obs.urnleftright.width.flatten(),
            obs.urnleftright.height.flatten(),

            obs.urn.x.flatten(),
            obs.urn.y.flatten(),
            obs.urn.floor.flatten(),
            obs.urn.width.flatten(),
            obs.urn.height.flatten(),

            obs.match_duration.flatten(),
            obs.matches_used.flatten(),
            obs.chasing.flatten(),
            obs.lives.flatten()])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)

    def observation_space(self) -> spaces:
        # Define a reusable space for entities that can be invisible
        entity_space = spaces.Dict({
            "x": spaces.Box(low=-1, high=160, shape=(), dtype=jnp.int32),
            "y": spaces.Box(low=-1, high=506, shape=(), dtype=jnp.int32),  # Max y is 506
            "floor": spaces.Box(low=-1, high=4, shape=(), dtype=jnp.int32),
            "width": spaces.Box(low=-1, high=16, shape=(), dtype=jnp.int32),
            "height": spaces.Box(low=-1, high=20, shape=(), dtype=jnp.int32),  # Bat height is 20
        })

        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=506, shape=(), dtype=jnp.int32),  # Max y is 506
                "floor": spaces.Box(low=1, high=4, shape=(), dtype=jnp.int32),  # Player is always on floor 1-4
                "width": spaces.Box(low=0, high=16, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=16, shape=(), dtype=jnp.int32),
            }),
            "ghost": entity_space,
            "spider": entity_space,
            "bat": entity_space,
            "item_held": spaces.Discrete(9),
            "scepter": entity_space,
            "urnleft": entity_space,
            "urnmiddle": entity_space,
            "urnright": entity_space,
            "urnleftmiddle": entity_space,
            "urnmiddleright": entity_space,
            "urnleftright": entity_space,
            "urn": entity_space,
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
    def _get_info(self, state: HauntedHouseState, all_rewards: chex.Array = None) -> HauntedHouseInfo:
        return HauntedHouseInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: HauntedHouseState, state: HauntedHouseState):
        penality_chasing = jnp.where(state.chasing > 0, 1, 0)
        return state.item_held + state.lives - state.stun_duration / 10 - state.matches_used - penality_chasing

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: HauntedHouseState, state: HauntedHouseState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

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
    def __init__(self, consts: HauntedHouseConstants = None):
        super().__init__()
        self.consts = consts or HauntedHouseConstants()
        (
            self.bg, self.wall, self.player, self.ghost, self.spider, self.bat, self.light, self.numbers, self.scoreboard,
            self.blackbar, self.stairsthintop, self.stairsthinbottom, self.stairswide,
            self.scepter, self.scepterheld, self.urn, self.urnheld,
            self.urnleft, self.urnleftheld, self.urnmiddle, self.urnmiddleheld, self.urnright, self.urnrightheld,
            self.urnleftmiddle, self.urnleftmiddleheld, self.urnmiddleright,
            self.urnmiddlerightheld, self.urnleftright, self.urnleftrightheld
        ) = self.load_sprites()

    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        player0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteEyesMiddle.npy"))
        player1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteEyesUp.npy"))
        player2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteEyesUpLeft.npy"))
        player3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteEyesLeft.npy"))
        player4 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteEyesDownLeft.npy"))
        player5 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteEyesDown.npy"))
        player6 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteEyesDownRight.npy"))
        player7 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteEyesRight.npy"))
        player8 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteEyesUpRight.npy"))

        ghost0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteGhost1.npy"))
        ghost1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteGhost2.npy"))
        spider0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteSpider1.npy"))
        spider1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteSpider2.npy"))
        bat0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteBat1.npy"))
        bat1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteBat2.npy"))

        scepter = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteItemScepter.npy"))
        scepterheld = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteHeldItemScepter.npy"))
        urn = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteItemUrn.npy"))
        urnheld = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteHeldItemUrn.npy"))
        urnleft = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteItemUrnLeft.npy"))
        urnmiddle = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteItemUrnMiddle.npy"))
        urnright = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteItemUrnRight.npy"))
        urnleftmiddle = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteItemUrnLeftMiddle.npy"))
        urnmiddleright = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteItemUrnMiddleRight.npy"))
        urnleftright = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteItemUrnLeftRight.npy"))
        urnleftheld = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteHeldItemUrnLeft.npy"))
        urnmiddleheld = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteHeldItemUrnMiddle.npy"))
        urnrightheld = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteHeldItemUrnRight.npy"))
        urnleftmiddleheld = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteHeldItemUrnLeftMiddle.npy"))
        urnmiddlerightheld = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteHeldItemUrnMiddleRight.npy"))
        urnleftrightheld = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteHeldItemUrnLeftRight.npy"))

        light0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteLight1.npy"))
        light1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteLight2.npy"))
        stairswide = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteStairsWide.npy"))
        stairsthintop = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteStairsThin.npy"))
        stairsthinbottom = stairsthintop[::-1, :, :]

        bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteBackground.npy"))
        wall = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/Wall.npy"))
        scoreboard = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteScoreboard.npy"))
        blackbar = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteBlackBar.npy"))

        player, _ = jr.pad_to_match([player0, player1, player2, player3, player4, player5, player6, player7, player8])
        ghost, _ = jr.pad_to_match([ghost0, ghost1])
        spider, _ = jr.pad_to_match([spider0, spider1])
        bat, _ = jr.pad_to_match([bat0, bat1])
        light, _ = jr.pad_to_match([light0, light1])

        player = jnp.concatenate(
            [
                jnp.repeat(player[0][None], 1, axis=0),
                jnp.repeat(player[1][None], 1, axis=0),
                jnp.repeat(player[2][None], 1, axis=0),
                jnp.repeat(player[3][None], 1, axis=0),
                jnp.repeat(player[4][None], 1, axis=0),
                jnp.repeat(player[5][None], 1, axis=0),
                jnp.repeat(player[6][None], 1, axis=0),
                jnp.repeat(player[7][None], 1, axis=0),
                jnp.repeat(player[8][None], 1, axis=0)
            ]
        )

        ghost = jnp.concatenate([
                jnp.repeat(ghost[0][None], 1, axis=0),
                jnp.repeat(ghost[1][None], 1, axis=0)])

        spider = jnp.concatenate([
                 jnp.repeat(spider[0][None], 1, axis=0),
                 jnp.repeat(spider[1][None], 1, axis=0)])

        bat = jnp.concatenate([
              jnp.repeat(bat[0][None], 1, axis=0),
              jnp.repeat(bat[1][None], 1, axis=0)])

        light = jnp.concatenate([
                jnp.repeat(light[0][None], 1, axis=0),
                jnp.repeat(light[1][None], 1, axis=0)])


        bg = jnp.expand_dims(bg, axis=0)
        wall = jnp.expand_dims(wall, axis=0)
        scoreboard = jnp.expand_dims(scoreboard, axis=0)

        scepter = jnp.expand_dims(scepter, axis=0)
        urnleft = jnp.expand_dims(urnleft, axis=0)
        urnmiddle = jnp.expand_dims(urnmiddle, axis=0)
        urnright = jnp.expand_dims(urnright, axis=0)
        urnleftmiddle = jnp.expand_dims(urnleftmiddle, axis=0)
        urnmiddleright = jnp.expand_dims(urnmiddleright, axis=0)
        urnleftright = jnp.expand_dims(urnleftright, axis=0)
        urn = jnp.expand_dims(urn, axis=0)

        scepterheld = jnp.expand_dims(scepterheld, axis=0)
        urnleftheld = jnp.expand_dims(urnleftheld, axis=0)
        urnmiddleheld = jnp.expand_dims(urnmiddleheld, axis=0)
        urnrightheld = jnp.expand_dims(urnrightheld, axis=0)
        urnleftmiddleheld = jnp.expand_dims(urnleftmiddleheld, axis=0)
        urnmiddlerightheld = jnp.expand_dims(urnmiddlerightheld, axis=0)
        urnleftrightheld = jnp.expand_dims(urnleftrightheld, axis=0)
        urnheld = jnp.expand_dims(urnheld, axis=0)

        stairsthintop = jnp.expand_dims(stairsthintop, axis=0)
        stairsthinbottom = jnp.expand_dims(stairsthinbottom, axis=0)
        stairswide = jnp.expand_dims(stairswide, axis=0)
        blackbar = jnp.expand_dims(blackbar, axis=0)




        numbers = jr.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/hauntedhouse/SpriteNumber{}.npy"),
            num_chars=10,
        )


        return (
            bg, wall, player, ghost, spider, bat, light, numbers, scoreboard, blackbar,
            stairsthintop, stairsthinbottom, stairswide, scepter, scepterheld, urn, urnheld,
            urnleft, urnleftheld, urnmiddle, urnmiddleheld, urnright, urnrightheld,
            urnleftmiddle, urnleftmiddleheld, urnmiddleright, urnmiddlerightheld, urnleftright, urnleftrightheld
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state:HauntedHouseState):
        raster = jr.create_initial_frame(width=160, height=210)

        camera_offset = jnp.where(state.player[1] < 82,0, state.player[1] - 82)
        camera_offset = jnp.clip(camera_offset, min=0, max=356)
        bg = jr.get_sprite_frame(self.bg, 0)
        raster = jr.render_at(raster, 0, 0, bg)

        floor_checks = self.check_same_floor(state)

        player_frame = jnp.where(state.stun_duration > 0, state.stun_duration % 8 + 1, state.player_direction[1])
        player = jr.get_sprite_frame(self.player, player_frame)
        raster = jr.render_at(raster, state.player[0], state.player[1] - camera_offset, player)


        # Monster
        k = jax.random.key(state.step_counter)
        k, subkey = jax.random.split(k)
        ghostframe = jax.random.randint(k, minval=0, maxval=2, shape=())
        k, subkey = jax.random.split(subkey)
        spiderframe = jax.random.randint(k, minval=0, maxval=2, shape=())
        k, subkey = jax.random.split(subkey)
        batframe = jax.random.randint(k, minval=0, maxval=2, shape=())


        vanish_with_scepter = jnp.logical_and(state.item_held == 1, state.step_counter % 4 != 0)
        not_vanishing = jnp.logical_not(vanish_with_scepter)

        ghost = jr.get_sprite_frame(self.ghost, ghostframe)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[0], not_vanishing),
                              lambda r: jr.render_at(r, state.ghost[0], state.ghost[1] - camera_offset, ghost),
                              lambda r: r,
                              raster)
        spider = jr.get_sprite_frame(self.spider, spiderframe)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[1], not_vanishing),
                              lambda r: jr.render_at(r, state.spider[0], state.spider[1] - camera_offset, spider),
                              lambda r: r,
                              raster)
        bat = jr.get_sprite_frame(self.bat, batframe)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[2], not_vanishing),
                              lambda r: jr.render_at(r, state.bat[0], state.bat[1] - camera_offset, bat),
                              lambda r: r,
                              raster)


        # Items
        illuminated = jnp.where(state.match_duration > 0,
                                self.check_illuminated(state),
                                jnp.array([False, False, False, False, False, False, False, False]))

        scepter = jr.get_sprite_frame(self.scepter, 0)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[3], illuminated[0]),
                              lambda r: jr.render_at(r, state.scepter[0], state.scepter[1] - camera_offset, scepter),
                              lambda r: r,
                              raster)
        urn_left = jr.get_sprite_frame(self.urnleft, 0)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[4], illuminated[1]),
                              lambda r: jr.render_at(r, state.urn_left[0], state.urn_left[1] - camera_offset, urn_left),
                              lambda r: r,
                              raster)
        urn_middle = jr.get_sprite_frame(self.urnmiddle, 0)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[5], illuminated[2]),
                              lambda r: jr.render_at(r, state.urn_middle[0], state.urn_middle[1] - camera_offset, urn_middle),
                              lambda r: r,
                              raster)
        urn_right = jr.get_sprite_frame(self.urnright, 0)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[6], illuminated[3]),
                              lambda r: jr.render_at(r, state.urn_right[0], state.urn_right[1] - camera_offset, urn_right),
                              lambda r: r,
                              raster)
        urn_left_middle = jr.get_sprite_frame(self.urnleftmiddle, 0)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[7], illuminated[4]),
                              lambda r: jr.render_at(r, state.urn_left_middle[0], state.urn_left_middle[1] - camera_offset, urn_left_middle),
                              lambda r: r,
                              raster)
        urn_middle_right = jr.get_sprite_frame(self.urnmiddleright, 0)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[8], illuminated[5]),
                              lambda r: jr.render_at(r, state.urn_middle_right[0], state.urn_middle_right[1] - camera_offset, urn_middle_right),
                              lambda r: r,
                              raster)
        urn_left_right = jr.get_sprite_frame(self.urnleftright, 0)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[9], illuminated[6]),
                              lambda r: jr.render_at(r, state.urn_left_right[0], state.urn_left_right[1] - camera_offset, urn_left_right),
                              lambda r: r,
                              raster)
        urn = jr.get_sprite_frame(self.urn, 0)
        raster = jax.lax.cond(jnp.logical_and(floor_checks[10], illuminated[7]),
                              lambda r: jr.render_at(r, state.urn[0], state.urn[1] - camera_offset, urn),
                              lambda r: r,
                              raster)



        # Light
        lightframe = jnp.where((state.step_counter % 8) < 2, 1, 0)
        light = jr.get_sprite_frame(self.light, lightframe)
        raster = jax.lax.cond(jnp.logical_and(state.step_counter % 2 == 0, state.match_duration > 0),
                              lambda r: jr.render_at(r, state.player[0] - 12, state.player[1] - 24 - camera_offset, light),
                              lambda r: r,
                              raster)


        # Stairs
        stairsthintop = jr.get_sprite_frame(self.stairsthintop, 0)
        stairsthinbottom = jr.get_sprite_frame(self.stairsthinbottom, 0)
        stairswide = jr.get_sprite_frame(self.stairswide, 0)
        raster = jax.lax.cond(jnp.logical_and(self.consts.STAIRS_TOP_LEFT[state.player[2] - 1], state.match_duration > 0),
                              lambda r: jr.render_at(r, 32, 7 - camera_offset, stairsthintop),
                              lambda r: r,
                              raster)
        raster = jax.lax.cond(jnp.logical_and(self.consts.STAIRS_TOP_RIGHT[state.player[2] - 1], state.match_duration > 0),
                              lambda r: jr.render_at(r, 112, 7 - camera_offset, stairsthinbottom),
                              lambda r: r,
                              raster)
        raster = jax.lax.cond(jnp.logical_and(self.consts.STAIRS_BOTTOM_LEFT[state.player[2] - 1], state.match_duration > 0),
                              lambda r: jr.render_at(r,32 , 478 - camera_offset, stairsthintop),
                              lambda r: r,
                              raster)
        raster = jax.lax.cond(jnp.logical_and(self.consts.STAIRS_BOTTOM_RIGHT[state.player[2] - 1], state.match_duration > 0),
                              lambda r: jr.render_at(r, 112, 478 - camera_offset, stairsthinbottom),
                              lambda r: r,
                              raster)
        raster = jax.lax.cond(jnp.logical_and(self.consts.STAIRS_LEFT[state.player[2] - 1], state.match_duration > 0),
                              lambda r: jr.render_at(r, 5, 226 - camera_offset, stairswide),
                              lambda r: r,
                              raster)
        raster = jax.lax.cond(jnp.logical_and(self.consts.STAIRS_RIGHT[state.player[2] - 1], state.match_duration > 0),
                              lambda r: jr.render_at(r, 140, 225 - camera_offset, stairswide),
                              lambda r: r,
                              raster)




        # Walls
        wall = jr.get_sprite_frame(self.wall, 0)

        k, subkey = jax.random.split(subkey)
        chase_flash = jnp.logical_or(state.chasing % 64 == 1, state.chasing % 64 == 9)
        stun_flash = jnp.logical_and(state.stun_duration % 4 == 1, jax.random.choice(k, jnp.array([True, False])))
        wall_flashing = jnp.logical_or(chase_flash, stun_flash)

        wall_color = jnp.where(wall_flashing, self.get_wall_color(0), self.get_wall_color(state.player[2]))

        wall_color_mask = jnp.all(wall == self.consts.WALL_COLOR_BLUE, axis=-1, keepdims=True)

        wall = jnp.where(wall_color_mask, wall_color, wall)

        raster = jr.render_at(raster, 0, self.consts.WALL_Y_OFFSET - camera_offset, wall)

        black_bar = jr.get_sprite_frame(self.blackbar, 0)
        raster = jr.render_at(raster, 32, self.consts.WALL_Y_OFFSET - camera_offset, black_bar)
        raster = jr.render_at(raster, 112, self.consts.WALL_Y_OFFSET - camera_offset, black_bar)

        # Scoreboard
        scoreboard = jr.get_sprite_frame(self.scoreboard, 0)
        raster = jr.render_at(raster, 0, 161, scoreboard)

        scepter_held = jr.get_sprite_frame(self.scepterheld, 0)
        urn_left_held = jr.get_sprite_frame(self.urnleftheld, 0)
        urn_middle_held = jr.get_sprite_frame(self.urnmiddleheld, 0)
        urn_right_held = jr.get_sprite_frame(self.urnrightheld, 0)
        urn_left_middle_held = jr.get_sprite_frame(self.urnleftmiddleheld, 0)
        urn_middle_right_held = jr.get_sprite_frame(self.urnmiddlerightheld, 0)
        urn_left_right_held = jr.get_sprite_frame(self.urnleftrightheld, 0)
        urn_held = jr.get_sprite_frame(self.urnheld, 0)

        held_item_sprites = [scepter_held, urn_left_held, urn_middle_held, urn_right_held,
                             urn_left_middle_held, urn_middle_right_held, urn_left_right_held, urn_held]

        raster = jax.lax.cond(state.item_held == 1,
                              lambda r: jr.render_at(r, 91 + self.consts.ITEM_OFFSETS[state.item_held][0],
                                                     166 + self.consts.ITEM_OFFSETS[state.item_held][1],
                                                     held_item_sprites[0]),
                              lambda r: r,
                              raster)

        raster = jax.lax.cond(state.item_held == 2,
                              lambda r: jr.render_at(r, 91 + self.consts.ITEM_OFFSETS[state.item_held][0],
                                                     166 + self.consts.ITEM_OFFSETS[state.item_held][1],
                                                     held_item_sprites[1]),
                              lambda r: r,
                              raster)

        raster = jax.lax.cond(state.item_held == 3,
                              lambda r: jr.render_at(r, 91 + self.consts.ITEM_OFFSETS[state.item_held][0],
                                                     166 + self.consts.ITEM_OFFSETS[state.item_held][1],
                                                     held_item_sprites[2]),
                              lambda r: r,
                              raster)

        raster = jax.lax.cond(state.item_held == 4,
                              lambda r: jr.render_at(r, 91 + self.consts.ITEM_OFFSETS[state.item_held][0],
                                                     166 + self.consts.ITEM_OFFSETS[state.item_held][1],
                                                     held_item_sprites[3]),
                              lambda r: r,
                              raster)

        raster = jax.lax.cond(state.item_held == 5,
                              lambda r: jr.render_at(r, 91 + self.consts.ITEM_OFFSETS[state.item_held][0],
                                                     166 + self.consts.ITEM_OFFSETS[state.item_held][1],
                                                     held_item_sprites[4]),
                              lambda r: r,
                              raster)

        raster = jax.lax.cond(state.item_held == 6,
                              lambda r: jr.render_at(r, 91 + self.consts.ITEM_OFFSETS[state.item_held][0],
                                                     166 + self.consts.ITEM_OFFSETS[state.item_held][1],
                                                     held_item_sprites[5]),
                              lambda r: r,
                              raster)

        raster = jax.lax.cond(state.item_held == 7,
                              lambda r: jr.render_at(r, 91 + self.consts.ITEM_OFFSETS[state.item_held][0],
                                                     166 + self.consts.ITEM_OFFSETS[state.item_held][1],
                                                     held_item_sprites[6]),
                              lambda r: r,
                              raster)

        raster = jax.lax.cond(state.item_held == 8,
                              lambda r: jr.render_at(r, 91 + self.consts.ITEM_OFFSETS[state.item_held][0],
                                                     166 + self.consts.ITEM_OFFSETS[state.item_held][1],
                                                     held_item_sprites[7]),
                              lambda r: r,
                              raster)


        matches_used_digits = jr.int_to_digits(state.matches_used, max_digits=2)
        lives_digits = jr.int_to_digits(state.lives, max_digits=1)
        player_floor_digits = jr.int_to_digits(state.player[2], max_digits=1)

        raster = jr.render_label_selective(raster, 37, 176,
                                            matches_used_digits, self.numbers,
                                            0, 2,
                                            spacing=1)

        raster = jr.render_label_selective(raster, 110, 176,
                                           lives_digits, self.numbers,
                                           0, 1,
                                           spacing=0)

        raster = jr.render_label_selective(raster, 61, 166,
                                           player_floor_digits, self.numbers,
                                           0, 1,
                                           spacing=0)


        return raster


    @partial(jax.jit, static_argnums=(0,))
    def get_wall_color(self, selector):

        def case_white():
            return self.consts.WALL_COLOR_WHITE
        def case_blue():
            return self.consts.WALL_COLOR_BLUE
        def case_red():
            return self.consts.WALL_COLOR_RED
        def case_green():
            return self.consts.WALL_COLOR_GREEN
        def case_yellow():
            return self.consts.WALL_COLOR_YELLOW

        branches = (case_white, case_blue, case_red, case_green, case_yellow)

        return jax.lax.switch(selector, branches)


    @partial(jax.jit, static_argnums=(0,))
    def check_same_floor(self, state: HauntedHouseState):
        """
        Returns a list of 10 booleans representing whether objects and monsters are on the same floor as the player.
        Returns: [ghost, spider, bat, scepter, urn_left, urn_middle, urn_right,
        urn_left_middle, urn_middle_right, urn_left_right, urn]
        """

        p = state.player[2]
        return jnp.array([p == state.ghost[2], p == state.spider[2], p == state.bat[2], p == state.scepter[2],
                          p == state.urn_left[2], p == state.urn_middle[2], p == state.urn_right[2],
                          p == state.urn_left_middle[2], p == state.urn_middle_right[2], p == state.urn_left_right[2],
                          p == state.urn[2]]).astype(jnp.bool)

    @partial(jax.jit, static_argnums=(0,))
    def check_illuminated(self, state: HauntedHouseState):

        vmapped_func = jax.vmap(
            self.check_illuminated_single,
            in_axes=(None, 0, 0)
        )

        entities = jnp.array([state.scepter, state.urn_left, state.urn_middle, state.urn_right,
                              state.urn_left_middle, state.urn_middle_right, state.urn_left_right, state.urn])

        result = vmapped_func(state.player, entities, self.consts.ITEM_SIZES)

        return result

    @partial(jax.jit, static_argnums=(0,))
    def check_illuminated_single(self, player, pos, size):
        pos = jnp.array([pos[0] - player[0] + 12, pos[1] - player[1] + 24])
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

















