"""
authors: Paula Troszt, Ernst Christian Böhringer, Aiman Sammy Rahlf
"""

import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional, List
import jax
import jax.lax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces

from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

def _create_static_procedural_sprites() -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # Use default constants for procedural colors
    PLAYER_STATS_COLOR = (111, 217, 158, 255)
    ENEMY_STATS_COLOR = (104, 186, 220, 255)
    
    return {
        'player_stats_color': jnp.array([[list(PLAYER_STATS_COLOR)]], dtype=jnp.uint8),
        'enemy_stats_color': jnp.array([[list(ENEMY_STATS_COLOR)]], dtype=jnp.uint8),
        'victory_color_1': jnp.array([[[212,252,144,255]]], dtype=jnp.uint8),
        'victory_color_2': jnp.array([[[198,128,236,255]]], dtype=jnp.uint8),
        'black': jnp.array([[[0,0,0,255]]], dtype=jnp.uint8),
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for SpaceWar.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    static_procedural = _create_static_procedural_sprites()
    
    # Player sprites (16 move + 8 death)
    player_files = [f'player_pos{i}.npy' for i in range(16)]
    player_death_files = [f'playerdeath_pos{i}.npy' for i in list(range(2, 6)) + list(range(10, 14))]
    
    # Enemy sprites (16 frames)
    enemy_files = [f'enemy_pos{i}.npy' for i in range(16)]
    
    return (
        # Background
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        
        # Player (group all frames to pad them together)
        {'name': 'player_all', 'type': 'group', 'files': player_files + player_death_files},
        
        # Enemy (group all frames to pad them together)
        {'name': 'enemy_all', 'type': 'group', 'files': enemy_files},
        
        # Other single sprites
        {'name': 'missile', 'type': 'single', 'file': 'missile.npy'},
        {'name': 'star_base', 'type': 'single', 'file': 'star_base.npy'},
        
        # Digits
        {'name': 'digits', 'type': 'digits', 'pattern': 'digit_{}.npy'},
        {'name': 'enemy_digits', 'type': 'digits', 'pattern': 'enemyDigit_{}.npy'},
        
        # Procedural sprites to ensure colors are in the palette
        {'name': 'player_stats_color', 'type': 'procedural', 'data': static_procedural['player_stats_color']},
        {'name': 'enemy_stats_color', 'type': 'procedural', 'data': static_procedural['enemy_stats_color']},
        {'name': 'victory_color_1', 'type': 'procedural', 'data': static_procedural['victory_color_1']},
        {'name': 'victory_color_2', 'type': 'procedural', 'data': static_procedural['victory_color_2']},
        {'name': 'black', 'type': 'procedural', 'data': static_procedural['black']},
    )

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
    BLACK_BORDER_BOTTOM_HEIGHT: int = 21

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
    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = _get_default_asset_config()

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

class JaxSpaceWar(JaxEnvironment[SpaceWarState, SpaceWarObservation, SpaceWarInfo, SpaceWarConstants]):
    def __init__(self, consts: SpaceWarConstants = None):
        consts = consts or SpaceWarConstants()
        super().__init__(consts)
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
        info = self._get_info(new_state)
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
    def _get_info(self, state: SpaceWarState) -> SpaceWarInfo:
        return SpaceWarInfo(player_score=state.player_score, enemy_score=state.enemy_score, step_counter=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: SpaceWarState, state: SpaceWarState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SpaceWarState) -> bool:
        return state.player_score >= 10

class SpaceWarRenderer(JAXGameRenderer):
    """JAX-based SpaceWar game renderer, optimized with JIT compilation."""

    def __init__(self, consts: SpaceWarConstants = None):
        """
        Initializes the renderer by loading and pre-processing all assets.
        """
        super().__init__()
        self.consts = consts or SpaceWarConstants()
        
        # 1. Configure the renderer
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        # 2. Define sprite path
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/spacewar"
        
        # 3. Use asset config from constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # 4. Load all assets, create palette, and generate ID masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        # 5. Pad the background to match full game dimensions
        # The background sprite is smaller and needs padding on top and bottom
        self.BACKGROUND = self._pad_background(self.BACKGROUND)
        
        # 6. Pre-compute/cache values for rendering
        self._cache_sprite_stacks()


    def _pad_background(self, background):
        """Pads the background with black borders on top and bottom to match target dimensions."""
        bg_height, bg_width = background.shape
        
        # Determine target dimensions (downscaled if configured, else game dimensions)
        if self.config.downscale:
            target_h, target_w = self.config.downscale[0], self.config.downscale[1]
        else:
            target_h, target_w = self.config.game_dimensions[0], self.config.game_dimensions[1]
        
        # Scale border heights if downscaling is enabled, otherwise use constants directly
        if self.config.downscale:
            height_scale = self.config.height_scaling
            top_padding = int(round(self.consts.BLACK_BORDER_TOP_HEIGHT * height_scale))
            bottom_padding = int(round(self.consts.BLACK_BORDER_BOTTOM_HEIGHT * height_scale))
        else:
            top_padding = self.consts.BLACK_BORDER_TOP_HEIGHT
            bottom_padding = self.consts.BLACK_BORDER_BOTTOM_HEIGHT
        
        # Get black color ID from the palette
        black_id = self.COLOR_TO_ID[(0, 0, 0)]
        
        # Pad top
        top_pad = jnp.full((top_padding, bg_width), black_id, dtype=background.dtype)
        # Pad bottom  
        bottom_pad = jnp.full((bottom_padding, bg_width), black_id, dtype=background.dtype)
        
        # Concatenate: top padding + background + bottom padding
        padded_bg = jnp.concatenate([top_pad, background, bottom_pad], axis=0)
        
        # Ensure we match target dimensions exactly (resize if needed due to rounding)
        if padded_bg.shape[0] != target_h:
            padded_bg = jax.image.resize(padded_bg[None, :, :], (1, target_h, target_w), method='nearest')[0]
        
        return padded_bg

    def _cache_sprite_stacks(self):
        """Caches the correctly ordered sprite stacks from the loaded assets."""
        all_player = self.SHAPE_MASKS['player_all']
        
        # Player move sprites (first 16)
        self.player_sprites = all_player[:16]
        
        # Player death sprites (last 8, re-ordered)
        death_sprites = all_player[16:]
        self.player_death_sprites = jnp.stack([
            all_player[2], all_player[3], # Original pos2, pos3
            death_sprites[0], death_sprites[1], death_sprites[2], death_sprites[3], # death 4-5
            all_player[6], all_player[7], all_player[8], all_player[9], # Original pos6-9
            death_sprites[4], death_sprites[5], death_sprites[6], death_sprites[7], # death 10-13
            all_player[14], all_player[15] # Original pos14-15
        ])
        
        # Enemy sprites (re-ordered as per original logic)
        all_enemy = self.SHAPE_MASKS['enemy_all']
        enemy_order = jnp.array([0] + list(range(15, 0, -1)))
        self.enemy_sprites = all_enemy[enemy_order]


    @partial(jax.jit, static_argnums=(0,))
    def _render_background(self, state):
        """Selects and renders the correct background (static or victory animation)."""
        
        # Get the actual target dimensions (downscaled if configured, else game dimensions)
        # This matches what the static BACKGROUND uses
        target_h, target_w = self.BACKGROUND.shape
        
        # Get color IDs from procedural sprite shape masks (single pixel sprites)
        # Extract color ID from the center pixel of each procedural sprite
        victory_color_1_id = self.SHAPE_MASKS['victory_color_1'][0, 0]
        victory_color_2_id = self.SHAPE_MASKS['victory_color_2'][0, 0]
        black_id = self.SHAPE_MASKS['black'][0, 0]
        
        # Determine the correct palette ID for the background
        alternative_bg_id = jax.lax.cond(
            (state.step_counter - 116) % 512 >= 256,
            lambda: victory_color_1_id,
            lambda: victory_color_2_id
        )
        
        # Scale border heights if downscaling is enabled
        height_scale = target_h / self.consts.HEIGHT
        scaled_top_border = jnp.round(self.consts.BLACK_BORDER_TOP_HEIGHT * height_scale).astype(jnp.int32)
        scaled_bottom_border = jnp.round(self.consts.BLACK_BORDER_BOTTOM_HEIGHT * height_scale).astype(jnp.int32)
        
        # Create coordinate grids for masking
        yy = jnp.arange(target_h)[:, None]  # (H, 1)
        
        # Create masks for border areas
        top_border_mask = yy < scaled_top_border  # (H, 1)
        bottom_border_mask = yy >= (target_h - scaled_bottom_border)  # (H, 1)
        border_mask = (top_border_mask | bottom_border_mask)  # (H, 1)
        
        # Create the dynamic background raster with correct target dimensions
        victory_bg_raster = jnp.full((target_h, target_w), alternative_bg_id, dtype=self.BACKGROUND.dtype)
        
        # Black out the border areas using where (JAX-compatible)
        # Broadcast mask from (H, 1) to (H, W) to match raster shape
        border_mask_broadcast = jnp.broadcast_to(border_mask, (target_h, target_w))
        victory_bg_raster = jnp.where(border_mask_broadcast, black_id, victory_bg_raster)
        
        # Choose between the pre-rendered static BG or the dynamic victory BG
        # Both should now have the same shape (target_h, target_w)
        return jax.lax.cond(
            state.enemy_won_animation_has_started == 1,
            lambda: victory_bg_raster,
            lambda: self.BACKGROUND
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_stats(self, state, raster):
        """Renders all player and enemy stats (score and bars)."""
        
        # Get color IDs from procedural sprite shape masks (single pixel sprites)
        black_id = self.SHAPE_MASKS['black'][0, 0]
        player_stats_id = self.SHAPE_MASKS['player_stats_color'][0, 0]
        enemy_stats_id = self.SHAPE_MASKS['enemy_stats_color'][0, 0]
        
        # --- Player Score ---
        player_digit_sprites = self.SHAPE_MASKS['digits']
        player_score_digits = self.jr.int_to_digits(state.player_score, max_digits=2)
        
        is_single_digit = state.player_score < 10
        start_idx = jax.lax.select(is_single_digit, 1, 0)
        num_to_render = jax.lax.select(is_single_digit, 1, 2)
        render_x = jax.lax.select(is_single_digit, 32, 32 - 12) # 32 or (32-12)
        
        raster = self.jr.render_label_selective(raster,
                                           render_x,
                                           1 + self.consts.BLACK_BORDER_TOP_HEIGHT,
                                           player_score_digits,
                                           player_digit_sprites,
                                           start_idx,
                                           num_to_render,
                                           spacing = 12,
                                           max_digits_to_render=2)
        
        # --- Player Fuel (inverted fill) ---
        player_fuel_val = (self.consts.MAX_FUEL+1)//256 - (state.player_fuel + 255)//256
        raster = self.jr.render_bar(raster, 48, 1 + self.consts.BLACK_BORDER_TOP_HEIGHT, 
                               player_fuel_val, 
                               (self.consts.MAX_FUEL+1)//256, # max_val
                               32, 4, # w, h
                               black_id, # "fill" color (black)
                               player_stats_id) # "background" color (player)
        
        # --- Player Ammo (inverted fill) ---
        player_ammo_val = self.consts.MAX_AMMO - state.player_ammo
        raster = self.jr.render_bar(raster, 48, 7 + self.consts.BLACK_BORDER_TOP_HEIGHT, 
                               player_ammo_val,
                               self.consts.MAX_AMMO, # max_val
                               32, 4, # w, h
                               black_id, # "fill" color
                               player_stats_id) # "background" color

        # --- Enemy Score ---
        enemy_digit_sprites = self.SHAPE_MASKS['enemy_digits']
        enemy_score_digits = self.jr.int_to_digits(state.enemy_score, max_digits=2)
        
        is_single_digit_enemy = state.enemy_score < 10
        start_idx_enemy = jax.lax.select(is_single_digit_enemy, 1, 0)
        num_to_render_enemy = jax.lax.select(is_single_digit_enemy, 1, 2)
        render_x_enemy = jax.lax.select(is_single_digit_enemy, 112, 112 - 12) # 112 or (112-12)
        raster = self.jr.render_label_selective(raster,
                                           render_x_enemy,
                                           1 + self.consts.BLACK_BORDER_TOP_HEIGHT,
                                           enemy_score_digits,
                                           enemy_digit_sprites,
                                           start_idx_enemy,
                                           num_to_render_enemy,
                                           spacing = 12,
                                           max_digits_to_render=2)
        
        # --- Enemy Fuel ---
        raster = self.jr.render_bar(raster, 128, 1 + self.consts.BLACK_BORDER_TOP_HEIGHT, 
                               8, (self.consts.MAX_FUEL+1)//256, # val, max_val
                               32, 4, # w, h
                               enemy_stats_id, # "fill" color
                               black_id) # "background" color
        
        # --- Enemy Ammo ---
        raster = self.jr.render_bar(raster, 128, 7 + self.consts.BLACK_BORDER_TOP_HEIGHT, 
                               self.consts.MAX_AMMO, self.consts.MAX_AMMO, # val, max_val
                               32, 4, # w, h
                               enemy_stats_id, # "fill" color
                               black_id) # "background" color
        
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SpaceWarState) -> chex.Array:
        """
        Renders the current game state using JAX operations.
        """
        # 1. Start with the correct background (static or victory animation)
        raster = self._render_background(state)

        # 2. Render Star Base (static object)
        raster = self.jr.render_at_clipped(raster, self.consts.STAR_BASE_X, self.consts.STAR_BASE_Y, self.SHAPE_MASKS['star_base'])

        # 3. Render Player
        player_sprite_stack = jax.lax.cond(
            state.player_death_timer <= 0,
            lambda: self.player_sprites,
            lambda: self.player_death_sprites
        )
        player_sprite = player_sprite_stack[state.player_state[4]]
        
        # Convert game coordinates to screen coordinates
        player_x = jnp.sign(state.player_state[0])*(jnp.abs(state.player_state[0]//256))*2
        player_y = jnp.sign(state.player_state[1])*(jnp.abs(state.player_state[1]//256))*2
        
        # Render player if not in hyperspace
        raster = jax.lax.cond(
            state.player_state[5] < 0,
            lambda r: self.jr.render_at_clipped(r, player_x, player_y, player_sprite),
            lambda r: r,
            raster
        )

        # 4. Render Enemy
        enemy_sprite = self.enemy_sprites[state.enemy_death_timer % 16]
        raster = self.jr.render_at_clipped(raster, self.consts.ENEMY_X, self.consts.ENEMY_Y, enemy_sprite)

        # 5. Render Missile
        missile_x = jnp.sign(state.player_missile_state[0])*(jnp.abs(state.player_missile_state[0]//256))*2
        missile_y = jnp.sign(state.player_missile_state[1])*(jnp.abs(state.player_missile_state[1]//256))*2
        
        raster = jax.lax.cond(
            state.player_missile_state[4] > 0,
            lambda r: self.jr.render_at_clipped(r, missile_x, missile_y, self.SHAPE_MASKS['missile']),
            lambda r: r,
            raster
        )

        # 6. Render Stats (Score and Bars)
        raster = self._render_stats(state, raster)

        # 7. Final conversion from palette IDs to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)