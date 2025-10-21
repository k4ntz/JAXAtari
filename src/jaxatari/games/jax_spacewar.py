"""
authors: Paula Troszt, Ernst Christian Böhringer, Aiman Sammy Rahlf
"""

import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax.lax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils_legacy as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class SpaceWarConstants(NamedTuple):
    # Constants for game environment
    WIDTH: int = 160
    HEIGHT: int = 250

    # Player position can be "in between" pixels.
    # This is not visible on screen but relevant for calculation.
    # Internal representations of the player position must first be converted to screen coordinates.
    # A change of a coordinate value of this internal position by any value below 256
    # might not be visible in game if no pixel-boundary is crossed with said change.

    # Object sizes (width, height)
    STAR_SHIP_SIZE: Tuple[int, int] = (5, 10)
    STAR_BASE_SIZE: Tuple[int, int] = (2, 4)
    MISSILE_SIZE: Tuple[int, int] = (2, 4)

    # Rendering constants
    PLAYER_STATS_COLOR: Tuple[int, int, int, int] = (111, 217, 158, 255)
    ENEMY_STATS_COLOR: Tuple[int, int, int, int] = (104, 186, 220, 255)
    WALL_TOP_HEIGHT: int = 12
    BLACK_BORDER_TOP_HEIGHT: int = 29
    BLACK_BORDER_BOTTOM_HEIGHT: int = 12

    # Positions
    INITIAL_PLAYER_X: int = int(43 * 256/2)
    INITIAL_PLAYER_Y: int = int((36 + BLACK_BORDER_TOP_HEIGHT) * 256/2)
    INITIAL_PLAYER_ROTATION: int = 8
    ENEMY_X: int = 109
    ENEMY_Y: int = 169 + BLACK_BORDER_TOP_HEIGHT
    STAR_BASE_X: int = 80
    STAR_BASE_Y: int = 105 + BLACK_BORDER_TOP_HEIGHT

    # Game constants
    MAX_FUEL: int = 8*256 - 1
    MAX_AMMO: int = 8
    POINTS_TO_WIN: int = 10
    DEATH_DELAY: int = 64
    H_SPACE_CONSUMPTION: int = 256
    MAX_ENTITY_X: int = WIDTH - 1
    MAX_ENTITY_Y: int = HEIGHT - BLACK_BORDER_BOTTOM_HEIGHT - 1
    MIN_ENTITY_X: int = 0
    MIN_ENTITY_Y: int = BLACK_BORDER_TOP_HEIGHT + WALL_TOP_HEIGHT

    # Player constants
    MAX_PLAYER_SPEED: int = 63*256 + 255
    MAX_PLAYER_X: int = int(((WIDTH - STAR_SHIP_SIZE[0])* 256 - 1)/2)
    MAX_PLAYER_Y: int = int((((HEIGHT - BLACK_BORDER_BOTTOM_HEIGHT) - STAR_SHIP_SIZE[1]) * 256 - 1)/2)
    MIN_PLAYER_X: int = 0
    MIN_PLAYER_Y: int = int(((BLACK_BORDER_TOP_HEIGHT + WALL_TOP_HEIGHT) * 256 - 1)/2)
    ACCEL_PER_ROTATION: chex.Array = jnp.array([
        (0, -72),
        (-27, -66),
        (-51, -51),
        (-66, -27),
        (-72, 0),
        (-66, 27),
        (-51, 51),
        (-27, 66),
        (0, 72),
        (27, 66),
        (51, 51),
        (66, 27),
        (72, 0),
        (66, -27),
        (51, -51),
        (27, -66)
    ])

    # Missile constants
    SHOOTING_COOLDOWN: int = 127
    MISSILE_SPEED_PER_ROTATION: chex.Array = jnp.array([
        (0, -578),
        (-208, -528),
        (-400, -416),
        (-528, -224),
        (-578, 0),
        (-528, 208),
        (-400, 400),
        (-208, 528),
        (0, 578),
        (224, 528),
        (416, 400),
        (528, 208),
        (578, 0),
        (528, -224),
        (416, -416),
        (224, -528)
    ])

# immutable state container
class SpaceWarState(NamedTuple):

    player_state: chex.Array # (6, ) array with (x, y, speed_x, speed_y, rotation, is_in_hyperspace)
    player_death_timer: chex.Array # for death animation and tracking if player is active
    enemy_death_timer: chex.Array # for death animation and tracking if enemy is active

    player_missile_state: chex.Array # (5, ) array with (x, y, speed_x, speed_y, active)
    shooting_cooldown_timer: chex.Array # missile can only be shot every 128 steps
    h_space_rdy: chex.Array # making sure that hyperspace behaves correctly when down button is held
    rotation_timer: chex.Array # rotation is updated every eighth step when left/right button is held (but instantly if button is pressed for a short time)

    player_score: chex.Array
    player_fuel: chex.Array
    player_ammo: chex.Array

    enemy_score: chex.Array
    enemy_won_animation_has_started: chex.Array
    enemy_victory: chex.Array # signals whether enemy has won already

    step_counter: chex.Array

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    rotation: jnp.ndarray
    active: jnp.ndarray

class MissilePosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class SpaceWarObservation(NamedTuple):
    player: EntityPosition # (x, y, width, height, rotation, active)
    player_missile: EntityPosition  # (x, y, width, height, active)

    player_score: jnp.ndarray
    player_fuel: jnp.ndarray
    player_ammo: jnp.ndarray

    enemy_score: jnp.ndarray

class SpaceWarInfo(NamedTuple):
    player_score: chex.Array
    enemy_score: chex.Array
    step_counter: chex.Array
    all_rewards: chex.Array

class JaxSpaceWar(JaxEnvironment[SpaceWarState, SpaceWarObservation, SpaceWarInfo, SpaceWarConstants]):
    def __init__(self, consts: SpaceWarConstants = None, reward_funcs: list[callable]=None):
        consts = consts or SpaceWarConstants()
        super().__init__(consts)
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = jnp.array([
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
            Action.UPLEFT,
            Action.UPRIGHT,
            Action.DOWNLEFT,
            Action.DOWNRIGHT,
            Action.UPFIRE,
            Action.DOWNFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ])
        self.obs_size = 6 + 5 + 4
        self.renderer = SpaceWarRenderer(consts)

    @partial(jax.jit, static_argnums=(0,))
    def final_pos(self, min_pos, max_pos, cur_pos):
        """
        Handles bouncing of objects off of walls
        """
        pos = jax.lax.cond(
            cur_pos > max_pos,
            lambda: 2*max_pos - cur_pos,
            lambda: cur_pos
        )

        pos = jax.lax.cond(
            cur_pos < min_pos,
            lambda: 2*min_pos - cur_pos,
            lambda: cur_pos
        )

        return pos

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        player_state,
        player_fuel,
        action,
        h_space_rdy,
        rotation_timer,
        enemy_victory
    ):
        # get pressed buttons
        left = jnp.logical_and(jnp.logical_or(
            jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE),
            jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE)), jnp.logical_not(enemy_victory))
        right = jnp.logical_and(jnp.logical_or(
            jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE),
            jnp.logical_or(action == Action.UPRIGHT, action == Action.UPRIGHTFIRE)), jnp.logical_not(enemy_victory))
        up = jnp.logical_and(jnp.logical_or(
            jnp.logical_or(
                jnp.logical_or(action == Action.UP, action == Action.UPRIGHT),
                jnp.logical_or(action == Action.UPLEFT, action == Action.UPFIRE)),
            jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE)), jnp.logical_not(enemy_victory))
        down = jnp.logical_and(jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE), jnp.logical_not(enemy_victory))

        player_x = player_state[0]
        player_y = player_state[1]
        player_speed_x = player_state[2]
        player_speed_y = player_state[3]
        player_rotation = player_state[4]

        # rotate player left
        player_rotation, rotation_timer = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(left, player_state[5] < 0), rotation_timer <= 0),
            lambda: ((player_rotation+1)%16, 8),
            lambda: (player_rotation, rotation_timer)
        )

        # rotate player right
        player_rotation, rotation_timer = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(right, player_state[5] < 0), rotation_timer <= 0),
            lambda: ((player_rotation-1)%16, 8),
            lambda: (player_rotation, rotation_timer)
        )

        # update rotation timer if player is rotated
        rotation_timer = jax.lax.cond(
            jnp.logical_and(jnp.logical_or(right, left), rotation_timer > 0),
            lambda: rotation_timer - 1,
            lambda: rotation_timer
        )

        # reset rotation timer to 0 if player is not rotated
        rotation_timer = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(jnp.logical_or(right, left)), rotation_timer > 0),
            lambda: 0,
            lambda: rotation_timer
        )

        # get acceleration for movement with thrusters
        accel_x = self.consts.ACCEL_PER_ROTATION[player_state[4]][0]
        accel_y = self.consts.ACCEL_PER_ROTATION[player_state[4]][1]

        # calculate new player speed
        player_speed_x = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(up, player_state[5] < 0), player_fuel > 0),
            lambda: player_speed_x + accel_x,
            lambda: player_speed_x
        )
        player_speed_y = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(up, player_state[5] < 0), player_fuel > 0),
            lambda: player_speed_y + accel_y,
            lambda: player_speed_y
        )

        # calculate the change in position depending on the new speed
        displace_x = player_speed_x//16
        displace_y = player_speed_y//16

        # flip movement direction on wall collision
        player_speed_x = jax.lax.cond(
            jnp.logical_or(player_x + displace_x > self.consts.MAX_PLAYER_X, player_x + displace_x < self.consts.MIN_PLAYER_X),
            lambda: -player_speed_x,
            lambda: player_speed_x
        )
        player_speed_y = jax.lax.cond(
            jnp.logical_or(player_y + displace_y > self.consts.MAX_PLAYER_Y, player_y + displace_y < self.consts.MIN_PLAYER_Y),
            lambda: -player_speed_y,
            lambda: player_speed_y
        )

        # calculate new player position
        player_x = self.final_pos(self.consts.MIN_PLAYER_X, self.consts.MAX_PLAYER_X, player_x + displace_x)
        player_y = self.final_pos(self.consts.MIN_PLAYER_Y, self.consts.MAX_PLAYER_Y, player_y + displace_y)

        # activate hyperspace
        h_space, fuel = jax.lax.cond(
                jnp.logical_and(
                    jnp.logical_and(down, player_state[5] < 0),
                    jnp.logical_and(h_space_rdy, player_fuel > 0)),
            lambda: (1, player_fuel-256),
            lambda: (player_state[5], player_fuel)
        )
        # deactivate hyperspace
        h_space = jax.lax.cond(
            jnp.logical_or(jnp.logical_and(jnp.logical_and(down, player_state[5] >= 0),
                                           h_space_rdy),
                           player_fuel <= 0),
            lambda: -1,
            lambda: h_space
        )

        # update fuel
        fuel = jax.lax.cond(
            jnp.logical_or(jnp.logical_and(up, h_space < 0), player_state[5] > 0),
            lambda: fuel - 1,
            lambda: fuel
        )
        # limit fuel at 0
        fuel = jax.lax.cond(
            fuel < 0,
            lambda: 0,
            lambda: fuel
        )

        # enemy receives point if player tries to enter hyperspace without fuel
        enemy_point =jnp.logical_or(
            jnp.logical_and(jnp.logical_and(jnp.logical_and(down, h_space < 0),
                                            jnp.logical_and(h_space_rdy, player_fuel <= 0)),
                            player_state[5] < 0),
            jnp.logical_and(player_fuel > 0, fuel <= 0)
        )

        ret_state = jnp.array([player_x, player_y, player_speed_x, player_speed_y, player_rotation, h_space])

        return ret_state, fuel, enemy_point, rotation_timer

    @partial(jax.jit, static_argnums=(0,))
    def to_screen_pos(self, player_pos):
        """
        Converts a subpixel position to a screen position.
        """
        return jnp.sign(player_pos)*(jnp.abs(player_pos)//256)*2

    @partial(jax.jit, static_argnums=(0,))
    def missile_step(
        self,
        player_missile_state,
        cur_player_state,
        action,
        player_ammo,
        shooting_cool_down_timer,
        enemy_victory
    ):  
        # get pressed buttons
        space = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(action == Action.FIRE, action == Action.DOWNRIGHTFIRE),
                                                             action == Action.LEFTFIRE),
                                              jnp.logical_or(action == Action.UPFIRE, action == Action.UPLEFTFIRE)),
                               jnp.logical_or(jnp.logical_or(action == Action.DOWNFIRE, action == Action.UPRIGHTFIRE),
                                              jnp.logical_or(action == Action.RIGHTFIRE, action == Action.DOWNLEFTFIRE)))
        
        # check whether new missile is shot
        init_missile = jnp.logical_and(jnp.logical_and(space, jnp.logical_and(jnp.logical_and(shooting_cool_down_timer <= 0, cur_player_state[5] == -1),
                                                              player_ammo > 0)), jnp.logical_not(enemy_victory))

        # initialize missile on shot
        player_missile_state, shooting_cool_down_timer, player_ammo = jax.lax.cond(
            init_missile,
            lambda: (jnp.array([cur_player_state[0],
                                cur_player_state[1],
                                cur_player_state[2]//16 + self.consts.MISSILE_SPEED_PER_ROTATION[cur_player_state[4]][0],
                                cur_player_state[3]//16 + self.consts.MISSILE_SPEED_PER_ROTATION[cur_player_state[4]][1], 1]),
                     self.consts.SHOOTING_COOLDOWN, player_ammo-1),
            lambda: (player_missile_state, shooting_cool_down_timer, player_ammo)
        )

        # set missile to inactive after missile lifespan
        player_missile_state = jax.lax.cond(
            shooting_cool_down_timer <= 0,
            lambda: player_missile_state.at[4].set(0),
            lambda: player_missile_state
        )

        # update missile speeds
        new_missile_speed_x = jax.lax.cond(
            jnp.logical_and(
                player_missile_state[4] > 0,
                jnp.logical_or(player_missile_state[0] + player_missile_state[2] < self.consts.MIN_PLAYER_X,
                               player_missile_state[0] + player_missile_state[2] > self.consts.MAX_PLAYER_X - self.consts.MISSILE_SIZE[0])
            ),
            lambda: -player_missile_state[2],
            lambda: player_missile_state[2]
        )
        new_missile_speed_y = jax.lax.cond(
            jnp.logical_and(
                player_missile_state[4] > 0,
                jnp.logical_or(player_missile_state[1] + player_missile_state[3] < self.consts.MIN_PLAYER_Y,
                               player_missile_state[1] + player_missile_state[3] > self.consts.MAX_PLAYER_Y - self.consts.MISSILE_SIZE[1])
            ),
            lambda: -player_missile_state[3],
            lambda: player_missile_state[3]
        )

        # update missile if it is active
        player_missile_state = jax.lax.cond(
            player_missile_state[4] > 0,
            lambda: jnp.array([self.final_pos(self.consts.MIN_PLAYER_X, self.consts.MAX_PLAYER_X - self.consts.MISSILE_SIZE[0], player_missile_state[0] + player_missile_state[2]),
                                self.final_pos(self.consts.MIN_PLAYER_Y, self.consts.MAX_PLAYER_Y - self.consts.MISSILE_SIZE[1], player_missile_state[1] + player_missile_state[3]),
                                new_missile_speed_x, new_missile_speed_y, player_missile_state[4]]),
            lambda: player_missile_state
        )

        # update timer
        shooting_cool_down_timer = jax.lax.cond(
            shooting_cool_down_timer > 0,
            lambda: shooting_cool_down_timer - 1,
            lambda: shooting_cool_down_timer
        )

        return player_missile_state, player_ammo, shooting_cool_down_timer

    @partial(jax.jit, static_argnums=(0,))
    def is_hit(
        self,
        e_x,
        e_y,
        e_active,
        e_size,
        missile_state
    ):
        """
        Checks whether entity is hit by missile
        """
        return jax.lax.cond(
            jnp.logical_and(e_active, missile_state[4] > 0),
            lambda: self.entities_collide(e_x, e_y, e_size[0], e_size[1],
                                          self.to_screen_pos(missile_state[0]), self.to_screen_pos(missile_state[1]),
                                          self.consts.MISSILE_SIZE[0], self.consts.MISSILE_SIZE[1]),
            lambda: False
        )

    @partial(jax.jit, static_argnums=(0,))
    def entities_collide(
        self,
        e1_x: chex.Array,
        e1_y: chex.Array,
        e1_w: chex.Array,
        e1_h: chex.Array,
        e2_x: chex.Array,
        e2_y: chex.Array,
        e2_w: chex.Array,
        e2_h: chex.Array,
    ) -> chex.Array:
        """
        Checks whether two entities are colliding
        """
        overlap_start_x = jnp.maximum(e1_x, e2_x)
        overlap_end_x = jnp.minimum(e1_x + e1_w, e2_x + e2_w)
        overlap_start_y = jnp.maximum(e1_y, e2_y)
        overlap_end_y = jnp.minimum(e1_y + e1_h, e2_y + e2_h)

        return jnp.logical_and(overlap_start_x < overlap_end_x, overlap_start_y < overlap_end_y)

    def reset(self, key) -> Tuple[SpaceWarObservation, SpaceWarState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and respective observation.
        """
        state = SpaceWarState(

            player_state=jnp.array([self.consts.INITIAL_PLAYER_X, self.consts.INITIAL_PLAYER_Y, 
                                    0, 0, self.consts.INITIAL_PLAYER_ROTATION, -1]).astype(jnp.int32),
            player_death_timer=jnp.array(0).astype(jnp.int32),
            enemy_death_timer=jnp.array(0).astype(jnp.int32),

            player_missile_state=jnp.zeros((5, )).astype(jnp.int32),
            shooting_cooldown_timer=jnp.array(0).astype(jnp.int32),
            h_space_rdy=jnp.array(True).astype(jnp.bool),
            rotation_timer=jnp.array(0).astype(jnp.int32),

            player_score=jnp.array(0).astype(jnp.int32),
            player_fuel=jnp.array(self.consts.MAX_FUEL).astype(jnp.int32),
            player_ammo=jnp.array(self.consts.MAX_AMMO).astype(jnp.int32),

            enemy_score=jnp.array(0).astype(jnp.int32),
            enemy_won_animation_has_started=jnp.array(0).astype(jnp.int32),
            enemy_victory=jnp.array(False).astype(jnp.bool),

            step_counter=jnp.array(0).astype(jnp.int32)
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SpaceWarState, action: chex.Array
    ) -> Tuple[SpaceWarObservation, SpaceWarState, float, bool, SpaceWarInfo]:

        # update player state and fuel stat and determine whether enemy gains a point
        player_state, player_fuel, enemy_point, rotation_timer = self.player_step(
            state.player_state, state.player_fuel, action, state.h_space_rdy, state.rotation_timer, state.enemy_victory
        )

        # player speed is updated in every 4th step only
        player_state = jax.lax.cond(
            state.step_counter % 4 == 0,
            lambda: player_state,
            lambda: player_state.at[2].set(state.player_state[2]).at[3].set(state.player_state[3])
        )

        # player fuel is updated in every 4th step only (or if hyperspace is activated)
        player_fuel = jax.lax.cond(
            jnp.logical_or(state.step_counter % 4 == 0, jnp.logical_and(state.player_state[5] == -1, player_state[5] >= 0)),
            lambda: player_fuel,
            lambda: state.player_fuel
        )

        # making sure that hyperspace behaves correctly when down button is held
        h_space_rdy = jnp.logical_not(jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE))

        # rotate player after death and making sure player cannot die during death animation
        player_state, enemy_point = jax.lax.cond(
            state.player_death_timer > 0,
            lambda: (player_state.at[4].set((player_state[4] + 1)%16), False),
            lambda: (player_state, enemy_point)
        )

         # update death timers if they are not zero already
        player_death_timer = jax.lax.cond(
            state.player_death_timer >= 0,
            lambda: state.player_death_timer - 1,
            lambda: state.player_death_timer
        )

        # update enemy score and trigger player death if enemy gains a point
        enemy_score, player_death_timer = jax.lax.cond(
            enemy_point,
            lambda: (state.enemy_score + 1, self.consts.DEATH_DELAY),
            lambda: (state.enemy_score, player_death_timer)
        )

        # update player missile state, ammo stat and shooting cooldown timer
        player_missile_state, player_ammo, shooting_cooldown_timer = self.missile_step(
            state.player_missile_state, player_state, action, state.player_ammo, state.shooting_cooldown_timer, state.enemy_victory
        )

        # check which objects are hit by missile
        enemy_hit = self.is_hit(
            self.consts.ENEMY_X, self.consts.ENEMY_Y, True, self.consts.STAR_SHIP_SIZE, player_missile_state
        )
        player_hit = self.is_hit(
            self.to_screen_pos(player_state[0]), 
            self.to_screen_pos(player_state[1]), player_state[5] < 0, self.consts.STAR_SHIP_SIZE, player_missile_state
        )
        star_base_hit = self.is_hit(
            self.consts.STAR_BASE_X, self.consts.STAR_BASE_Y, True, self.consts.STAR_BASE_SIZE, player_missile_state
        )

        # set missile to inactive if it hits something
        player_missile_state = jax.lax.cond(
            jnp.logical_or(
                # do not deactivate missile on player hit if it has not left player hit box yet
                jnp.logical_and(player_hit, shooting_cooldown_timer < self.consts.SHOOTING_COOLDOWN - 3), 
                jnp.logical_or(enemy_hit, star_base_hit)),
            lambda: jnp.array([0, 0, 0, 0, 0]),
            lambda: player_missile_state
        )

        # update player score and trigger enemy death if enemy is hit
        player_score, enemy_death_timer = jax.lax.cond(
            jnp.logical_and(enemy_hit, state.enemy_death_timer <= 0),
            lambda: (state.player_score + 1, self.consts.DEATH_DELAY),
            lambda: (state.player_score, state.enemy_death_timer)
        )

        # handle docking with star base
        player_state, player_fuel, player_ammo = jax.lax.cond(
            self.entities_collide(self.to_screen_pos(player_state[0]), 
                                  self.to_screen_pos(player_state[1]), 
                                  self.consts.STAR_SHIP_SIZE[0],
                                  self.consts.STAR_SHIP_SIZE[1],
                                  self.consts.STAR_BASE_X,
                                  self.consts.STAR_BASE_Y,
                                  self.consts.STAR_BASE_SIZE[0],
                                  self.consts.STAR_BASE_SIZE[1]
                                  ),
            lambda: (jnp.array([self.consts.INITIAL_PLAYER_X, self.consts.INITIAL_PLAYER_Y, 0, 0, self.consts.INITIAL_PLAYER_ROTATION, -1]),
                     jnp.array(self.consts.MAX_FUEL), jnp.array(self.consts.MAX_AMMO)),
            lambda: (player_state, player_fuel, player_ammo)
        )

        # update timer
        enemy_death_timer = jax.lax.cond(
            enemy_death_timer > 0,
            lambda: enemy_death_timer - 1,
            lambda: enemy_death_timer
        )

        # update step counter (reset if player wins)
        step_counter = jax.lax.cond(
            player_score >= 10,
            lambda: 0,
            lambda: state.step_counter + 1
        )

        # go completely crazy when enemy reaches 10 points!
        enemy_won_animation_has_started = jax.lax.cond(
            jnp.logical_and((step_counter - 116)%256 == 0, enemy_score >= 10),
            lambda: 1,
            lambda: state.enemy_won_animation_has_started
        )

        # check whether enemy has won
        enemy_victory = enemy_score >= 10

        # update game state
        new_state = SpaceWarState(

            player_state=player_state,
            player_death_timer=player_death_timer,
            enemy_death_timer=enemy_death_timer,

            player_missile_state=player_missile_state,
            shooting_cooldown_timer=shooting_cooldown_timer,
            h_space_rdy=h_space_rdy,
            rotation_timer=rotation_timer,

            player_score=player_score,
            player_fuel=player_fuel,
            player_ammo=player_ammo,

            enemy_score=enemy_score,
            enemy_won_animation_has_started=enemy_won_animation_has_started,
            enemy_victory=enemy_victory,

            step_counter=step_counter
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_rewards(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def render(self, state: SpaceWarState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SpaceWarState):

        # player
        player = EntityPosition(
            x=self.to_screen_pos(state.player_state[0]),
            y=self.to_screen_pos(state.player_state[1]),
            width=jnp.array(self.consts.STAR_SHIP_SIZE[0]),
            height=jnp.array(self.consts.STAR_SHIP_SIZE[1]),
            rotation=state.player_state[4],
            active=state.player_state[5] < 0
        )

        # missile
        player_missile = MissilePosition(
            x=self.to_screen_pos(state.player_missile_state[0]),
            y=self.to_screen_pos(state.player_missile_state[1]),
            width=jnp.array(self.consts.MISSILE_SIZE[0]),
            height=jnp.array(self.consts.MISSILE_SIZE[1]),
            active=state.player_missile_state[4] > 0
        )

        return SpaceWarObservation(
            player=player,
            player_missile=player_missile,

            player_score=state.player_score,
            player_fuel=(state.player_fuel + 255)//256,
            player_ammo=state.player_ammo,

            enemy_score=state.enemy_score
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: SpaceWarObservation) -> jnp.ndarray:
        """Converts the observation to a flat array."""

        def entity_pos_to_flat_array(obj: EntityPosition):
            return jnp.concatenate([
                jnp.atleast_1d(obj.x),
                jnp.atleast_1d(obj.y),
                jnp.atleast_1d(obj.width),
                jnp.atleast_1d(obj.height),
                jnp.atleast_1d(obj.rotation),
                jnp.atleast_1d(obj.active)
            ])
        
        def missile_pos_to_flat_array(obj: MissilePosition):
            return jnp.concatenate([
                jnp.atleast_1d(obj.x),
                jnp.atleast_1d(obj.y),
                jnp.atleast_1d(obj.width),
                jnp.atleast_1d(obj.height),
                jnp.atleast_1d(obj.active)
            ])

        return jnp.concatenate([
            entity_pos_to_flat_array(obs.player),
            missile_pos_to_flat_array(obs.player_missile),

            obs.player_score.flatten(),
            obs.player_fuel.flatten(),
            obs.player_ammo.flatten(),

            obs.enemy_score.flatten()
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Box:
        """Returns the observation space for SpaceWar.
        The observation contains:
        - player: EntityPosition (x, y, width, height, rotation, active)
        - player_missile: MissilePosition (x, y, width, height, active)
        - player_score: int (0-10)
        - player_fuel: int (0-8)
        - player_ammo: int (0-8)
        - enemy_score: int (0-10)
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160*256, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=250*256, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
                "rotation": spaces.Box(low=0, high=16, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "player_missile": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "player_score": spaces.Box(low=0, high=self.consts.POINTS_TO_WIN, shape=(), dtype=jnp.int32),
            "player_fuel": spaces.Box(low=0, high=self.consts.MAX_FUEL, shape=(), dtype=jnp.int32),
            "player_ammo": spaces.Box(low=0, high=self.consts.MAX_AMMO, shape=(), dtype=jnp.int32),
            "enemy_score": spaces.Box(low=0, high=self.consts.POINTS_TO_WIN, shape=(), dtype=jnp.int32)
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for SpaceWar.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(250, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SpaceWarState, all_rewards: chex.Array = None) -> SpaceWarInfo:
        return SpaceWarInfo(player_score=state.player_score, enemy_score=state.enemy_score, step_counter=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: SpaceWarState, state: SpaceWarState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: SpaceWarState, state: SpaceWarState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SpaceWarState) -> bool:
        return state.player_score >= 10

class SpaceWarRenderer(JAXGameRenderer):
    """JAX-based SpaceWar game renderer, optimized with JIT compilation."""

    def __init__(self, consts: SpaceWarConstants = None):
        """
        Initializes the renderer by loading sprites, including background.
        """
        super().__init__()
        self.consts = consts or SpaceWarConstants()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/spacewar"
        self.sprites = self._load_sprites()
        # store background sprite directly for use in render function
        self.background = self.sprites.get('background')

    def _load_sprites(self) -> dict[str, Any]:
        """Loads all necessary sprites from .npy files."""
        sprites: Dict[str, Any] = {}

        # helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = jr.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)

        # background
        sprites['background'] = _load_sprite_frame('background')

        # living player
        player_sprites = []
        for i in range(16):
            player_sprites.append(_load_sprite_frame(f'player_pos{i}'))
        
        # player death animation
        for i in list(range(2, 6)) + list(range(10, 14)):
            player_sprites.append(_load_sprite_frame(f'playerdeath_pos{i}'))

        # pad the player and death animation sprites since they have to be used interchangeably 
        # (and jax enforces same sizes)
        player_sprites, _ = jr.pad_to_match(player_sprites)

        sprites['player_pos'] = player_sprites[:16]
        sprites['playerdeath_pos'] = player_sprites[:2] + player_sprites[16:20] + player_sprites[6:10] + player_sprites[20:24] + player_sprites[14:16]

        # enemy (including death animation)
        enemy_sprites = []
        for i in range(16):
            enemy_sprites.append(_load_sprite_frame(f'enemy_pos{i}'))

        # pad the enemy sprites since they have to be used interchangeably 
        # (and jax enforces same sizes)
        enemy_sprites, _ = jr.pad_to_match(enemy_sprites)

        sprites['enemy_pos'] = [enemy_sprites[i] for i in [0]+list(range(15, 0, -1))]

        # missile
        sprites['missile'] = _load_sprite_frame('missile')

        # star base
        sprites['star_base'] = _load_sprite_frame('star_base')
        
        # digits for player score
        digit_path = os.path.join(self.sprite_path, 'digit_{}.npy')
        digit_sprites = jr.load_and_pad_digits(digit_path, num_chars=10)
        sprites['digits'] = digit_sprites

        # digits for enemy score
        digit_path = os.path.join(self.sprite_path, 'enemyDigit_{}.npy')
        digit_sprites = jr.load_and_pad_digits(digit_path, num_chars=10)
        sprites['enemy_digits'] = digit_sprites

        # expand all sprites
        for key, value in sprites.items():
            if isinstance(value, (list, tuple)):
                sprites[key] = jnp.array([jnp.expand_dims(sprite, axis=0) for sprite in value])
            else:
                sprites[key] = jnp.expand_dims(value, axis=0)

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SpaceWarState) -> chex.Array:
        """
        Renders the current game state using JAX operations.

        Args:
            state: A SpaceWarState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # empty raster
        raster = jr.create_initial_frame(width=self.consts.WIDTH, height=self.consts.HEIGHT)

        # background
        frame_bg = jr.get_sprite_frame(self.background, 0)
        # for enemy victory animation
        alternative_bg_color = jax.lax.cond(
            (state.step_counter-116)%512 >= 256,
            lambda: jnp.asarray((212,252,144), dtype=jnp.uint8),
            lambda: jnp.asarray((198,128,236), dtype=jnp.uint8)
        )
        
        # alternate backgrounds if enemy has won
        raster = jax.lax.cond(
            state.enemy_won_animation_has_started == 1,
            lambda: raster.at[self.consts.BLACK_BORDER_TOP_HEIGHT:self.consts.HEIGHT - self.consts.BLACK_BORDER_BOTTOM_HEIGHT, :, :].set(alternative_bg_color),
            lambda: jr.render_at(raster, 0, self.consts.BLACK_BORDER_TOP_HEIGHT, frame_bg)
        )

        # player
        # set player sprite (normal sprite if player is alive, otherwise sprite of death animation)
        player_sprite = jax.lax.cond(
            state.player_death_timer <= 0,
            lambda: self.sprites['player_pos'][state.player_state[4]],
            lambda: self.sprites['playerdeath_pos'][state.player_state[4]]
        )

        frame_player = jr.get_sprite_frame(player_sprite, state.step_counter)
        # do not render player when in hyperspace
        raster = jax.lax.cond(
            state.player_state[5] < 0,
            lambda: jr.render_at( # convert x and y position to screen coordinates again
                raster, 
                jnp.sign(state.player_state[0])*(jnp.abs(state.player_state[0]//256))*2,
                jnp.sign(state.player_state[1])*(jnp.abs(state.player_state[1])//256)*2, 
                frame_player),
            lambda: raster
        )

        # enemy
        # set enemy sprite
        enemy_sprite = self.sprites['enemy_pos'][state.enemy_death_timer%16]
        frame_enemy = jr.get_sprite_frame(enemy_sprite, state.step_counter)
        raster = jr.render_at(raster, self.consts.ENEMY_X, self.consts.ENEMY_Y, frame_enemy)

        # missile
        # set missile sprite
        frame_missile = jr.get_sprite_frame(self.sprites['missile'], state.step_counter)
        # only render missile if it is active
        raster = jax.lax.cond(
            state.player_missile_state[4] > 0,
            lambda: jr.render_at( # convert x and y position to screen coordinates again
                raster, 
                jnp.sign(state.player_missile_state[0])*(jnp.abs(state.player_missile_state[0]//256))*2,
                jnp.sign(state.player_missile_state[1])*(jnp.abs(state.player_missile_state[1]//256))*2,
                frame_missile),
            lambda: raster
        )

        # star base
        # set star base sprite
        frame_star_base = jr.get_sprite_frame(self.sprites['star_base'], state.step_counter)
        raster = jr.render_at(raster, self.consts.STAR_BASE_X, self.consts.STAR_BASE_Y, frame_star_base)

        # numbers
        def _get_number_of_digits(val):
            return jax.lax.cond(val < 10, lambda: 1, lambda: 2)

        # player score
        player_digit_sprites = self.sprites.get('digits', None)
        player_score_digits = jr.int_to_digits(state.player_score, max_digits = 2)
        number_of_additional_score_digits = _get_number_of_digits(state.player_score) - 1
        raster = jr.render_label_selective(raster,
                                           32 - 12 * number_of_additional_score_digits,
                                           1 + self.consts.BLACK_BORDER_TOP_HEIGHT,
                                           player_score_digits,
                                           player_digit_sprites[0],
                                           1 - number_of_additional_score_digits,
                                           number_of_additional_score_digits + 1,
                                           spacing = 12)
        
        # player fuel (mirrored because bar is growing to the left)
        raster = jr.render_bar(raster, 48, 1 + self.consts.BLACK_BORDER_TOP_HEIGHT, 
                               (self.consts.MAX_FUEL+1)//256 - (state.player_fuel + 255)//256, 
                               (self.consts.MAX_FUEL+1)//256, 32, 4, (0, 0, 0, 0), self.consts.PLAYER_STATS_COLOR)
        # player ammo (mirrored because bar is growing to the left)
        raster = jr.render_bar(raster, 48, 7 + self.consts.BLACK_BORDER_TOP_HEIGHT, 
                               self.consts.MAX_AMMO - state.player_ammo, 
                               self.consts.MAX_AMMO, 32, 4, (0, 0, 0, 0), self.consts.PLAYER_STATS_COLOR)

        # enemy score
        enemy_digit_sprites = self.sprites.get('enemy_digits', None)
        enemy_score_digits = jr.int_to_digits(state.enemy_score, max_digits = 2)
        number_of_additional_score_digits = _get_number_of_digits(state.enemy_score) - 1
        raster = jr.render_label_selective(raster,
                                           112 - 12 * number_of_additional_score_digits,
                                           1 + self.consts.BLACK_BORDER_TOP_HEIGHT,
                                           enemy_score_digits,
                                           enemy_digit_sprites[0],
                                           1 - number_of_additional_score_digits,
                                           number_of_additional_score_digits + 1,
                                           spacing = 12)
        
        # enemy fuel (mirrored because bar is growing to the left)
        raster = jr.render_bar(raster, 128, 1 + self.consts.BLACK_BORDER_TOP_HEIGHT, 8, 
                               self.consts.MAX_FUEL//256, 32, 4, self.consts.ENEMY_STATS_COLOR, (0, 0, 0, 0))
        # enemy ammo (mirrored because bar is growing to the left)
        raster = jr.render_bar(raster, 128, 7 + self.consts.BLACK_BORDER_TOP_HEIGHT, 8, 
                               self.consts.MAX_AMMO, 32, 4, self.consts.ENEMY_STATS_COLOR, (0, 0, 0, 0))

        return raster