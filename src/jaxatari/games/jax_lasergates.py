"""

Lukas Bergholz, Linus Orlob, Vincent Jahn

"""
import os
import time
from enum import IntEnum
from functools import partial
from typing import Tuple, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jaxatari.rendering.atraJaxis as aj
import pygame
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer

# -------- Game constants --------
WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 5

SCROLL_SPEED = 1 # Normal scroll speed
SCROLL_MULTIPLIER = 1.5 # When at the right player bound, multiply scroll speed by this constant

# -------- Mountains constants --------
PLAYING_FILED_BG_COLLISION_COLOR = (255, 255, 255, 255)
PLAYING_FILED_BG_COLOR_FADE_SPEED = 0.2 # Higher = faster fade out, exponential

# -------- Mountains constants --------
MOUNTAIN_SIZE = (60, 12) # Width, Height

LOWER_MOUNTAINS_Y = 80 # Y Spawn position of lower mountains. This does not change
UPPER_MOUNTAINS_Y = 19 # Y Spawn position of upper mountains. This does not change

LOWER_MOUNTAINS_START_X = -44 # X Spawn position of lower mountains.
UPPER_MOUNTAINS_START_X = -4 # X Spawn position of upper mountains.

MOUNTAINS_DISTANCE = 20 # Distance between two given mountains

UPDATE_EVERY = 4 # The mountain position is updated every UPDATE_EVERY-th frame.

# -------- Player constants --------
PLAYER_SIZE = (8, 6) # Width, Height
PLAYER_NORMAL_COLOR = (85, 92, 197, 255) # Normal color of the player
PLAYER_COLLISION_COLOR = (137, 81, 26, 255) # Players color for PLAYER_COLOR_CHANGE_DURATION frames after a collision
PLAYER_COLOR_CHANGE_DURATION = 10 # How long (in frames) the player changes its color to PLAYER_COLLISION_COLOR if a collision occurs

PLAYER_BOUNDS = (20, WIDTH - 20 - PLAYER_SIZE[0]), (19, 80 + PLAYER_SIZE[1]) # left x, right x, upper y and lower y bound of player

PLAYER_START_X = 20 # X Spawn position of player
PLAYER_START_Y = 52 # Y Spawn position of player

PLAYER_VELOCITY_Y = 1.5 # Y Velocity of player
PLAYER_VELOCITY_X = 1.5 # X Velocity of player

MAX_ENERGY = 5100
MAX_SHIELDS = 24
MAX_DTIME = 10200

# -------- Player missile constants --------
PLAYER_MISSILE_SIZE = (16, 1)
PLAYER_MISSILE_BASE_COLOR = (140, 79, 24, 255) # Initial color of player missile. Every value except for transparency is incremented by the missiles velocity * PLAYER_MISSILE_COLOR_CHANGE_SPEED
PLAYER_MISSILE_COLOR_CHANGE_SPEED = 10 # Defines how fast the player missile changes its color towards white.

PLAYER_MISSILE_INITIAL_VELOCITY = 2.5 # Starting speed of player missile
PLAYER_MISSILE_VELOCITY_MULTIPLIER = 1.1 # Multiply the current speed at a given moment of the player missile by this number

# -------- Instrument panel constants --------
SHIELD_LOSS_COL_SMALL = 1
SHIELD_LOSS_COL_BIG = 6

# -------- Entity constants (constants that apply to all entity types --------
NUM_ENTITY_TYPES = 8 # How many different (!) entity types there are

# -------- Radar mortar constants --------
RADAR_MORTAR_SIZE = (8, 26) # Width, Height

RADAR_MORTAR_SPRITE_ROTATION_SPEED = 15 # Change sprite frame (left, middle, right) of radar mortar every RADAR_MORTAR_SPRITE_ROTATION_SPEED frames
RADAR_MORTAR_SPAWN_X = WIDTH            # Spawm barely outside of bounds TODO: Either change this to a lot more or use cooldown before next entitiy init
RADAR_MORTAR_SPAWN_Y_BOTTOM = 66        # Since the radar mortar can spawn at the top or at the bottom of the screen, we define two y positions.
RADAR_MORTAR_SPAWN_Y_TOP = 19

# -------- Enemy Missile constants --------
ENEMY_MISSILE_COLOR = (85, 92, 197, 255)

# -------- GUI constants --------
GUI_COLORED_BACKGROUND_SIZE = (128, 12)
GUI_BLACK_BACKGROUND_SIZE = (56, 10)
GUI_TEXT_SCORE_SIZE = (21, 7)
GUI_TEXT_ENERGY_SIZE = (23, 5)
GUI_TEXT_SHIELDS_SIZE = (23, 5)
GUI_TEXT_DTIME_SIZE = (23, 5)

GUI_COLORED_BACKGROUND_COLOR_BLUE = (47, 90, 160, 255)
GUI_COLORED_BACKGROUND_COLOR_GREEN = (50, 152, 82, 255)
GUI_COLORED_BACKGROUND_COLOR_BEIGE = (160, 107, 50, 255)
GUI_COLORED_BACKGROUND_COLOR_GRAY = (182, 182, 182, 255)
GUI_TEXT_COLOR_GRAY = (118, 118, 118, 255)
GUI_TEXT_COLOR_BEIGE = (160, 107, 50, 255)

GUI_BLACK_BACKGROUND_X_OFFSET = 36
GUI_Y_SPACE_BETWEEN_PLAYING_FIELD = 10
GUI_Y_SPACE_BETWEEN_BACKGROUNDS = 10

# -------- States --------
class EntityType(IntEnum):
    NONE = 0
    RADAR_MORTAR = 1    # Radar mortars appear along the top and bottom of the Computer passage. Avoid Mortar fire. Demolish Radar Mortars with laser fire.
    BYTE_BAT = 2        # Green bat looking entity flying at you without warning.
    ROCK_MUNCHER = 3    #
    HOMING_MISSILE = 4  # Bomb looking entity flying and tracking you
    FORCEFIELD = 5      # Flashing, flexing or fixed "wall". Time your approach to cross.
    DENSEPACK = 6       # Grey densepack columns of varying width appear along the dark Computer passage. Blast your way through.
    DETONATOR = 7       # "Failsafe detonators" are large and grey and have the numbers "6507" etched on the side. Laser fire must strike one of the pins on the side of a detonator to destroy it.
    ENERGY_POD = 8      # To replenish energy reserves, touch Energy Pods as they appear along the Computer passageway. Do not fire at Energy Pods! You may not survive until another appears!

class RadarMortarState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array

class ByteBatState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array

class RockMuncherState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array

class HomingMissileState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array

class ForceFieldState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array

class DensepackState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array

class DetonatorState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array

class EnergyPodState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array

class Entities(NamedTuple):
    radar_mortar_state: RadarMortarState
    byte_bat_state: ByteBatState
    rock_muncher_state: RockMuncherState
    homing_missile_state: HomingMissileState
    forcefield_state: ForceFieldState
    dense_pack_state: DensepackState
    detonator_state: DetonatorState
    energy_pod_state: EnergyPodState

class MountainState(NamedTuple):
    x1: chex.Array
    x2: chex.Array
    x3: chex.Array
    y: chex.Array

class PlayerMissileState(NamedTuple):
    x: chex.Array
    y: chex.Array
    direction: chex.Array
    velocity: chex.Array

class LaserGatesState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_facing_direction: chex.Array
    player_missile: PlayerMissileState
    player_collision: chex.Array
    entities: Entities
    animation_timer: chex.Array
    lower_mountains: MountainState
    upper_mountains: MountainState
    score: chex.Array
    energy: chex.Array
    shields: chex.Array
    dtime: chex.Array
    scroll_speed: chex.Array
    rng_key:  chex.PRNGKey
    step_counter: chex.Array
    # TODO: fill

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class LaserGatesObservation(NamedTuple):
    player: EntityPosition
    # enemy1
    # enemy2: EntityPosition
    # ...: EntityPosition
    # ...: EntityPosition
    # TODO: fill

class LaserGatesInfo(NamedTuple):
    # difficulty: jnp.ndarray # add if necessary
    step_counter: jnp.ndarray
    all_rewards: jnp.ndarray

# -------- Render Constants --------
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Background parts
    upper_brown_bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/upper_brown_bg.npy"))
    lower_brown_bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/lower_brown_bg.npy"))
    playing_field_bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/playing_field_bg.npy"))
    gray_gui_bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/gray_gui_bg.npy"))
    lower_mountain = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/mountains/lower_mountain.npy"))
    upper_mountain = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/mountains/upper_mountain.npy"))
    black_stripe = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/black_stripe.npy"))

    # Player and player missile
    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/player/player.npy"))
    player_missile = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/missiles/player_missile.npy"))

    # Instrument panel parts
    gui_colored_background = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/colored_background.npy"))
    gui_black_background = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/black_background.npy"))
    gui_text_score = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/text/score.npy"))
    gui_text_energy = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/text/energy.npy"))
    gui_text_shields = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/text/shields.npy"))
    gui_text_dtime = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/text/dtime.npy"))
    gui_score_digits = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/lasergates/gui/score_numbers/{}.npy"))
    gui_score_comma = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/score_numbers/comma.npy"))

    # Entities
    # Radar mortar
    radar_mortar_frame_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/radar_mortar/1.npy"))
    radar_mortar_frame_middle = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/radar_mortar/2.npy"))
    radar_mortar_frame_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/radar_mortar/3.npy"))

    rms = aj.pad_to_match([radar_mortar_frame_left, radar_mortar_frame_middle, radar_mortar_frame_right])
    radar_mortar_sprites = jnp.concatenate([
        jnp.repeat(rms[0][None], RADAR_MORTAR_SPRITE_ROTATION_SPEED, axis=0),
        jnp.repeat(rms[1][None], RADAR_MORTAR_SPRITE_ROTATION_SPEED, axis=0),
        jnp.repeat(rms[2][None], RADAR_MORTAR_SPRITE_ROTATION_SPEED, axis=0),
        jnp.repeat(rms[1][None], RADAR_MORTAR_SPRITE_ROTATION_SPEED, axis=0),
    ]) # Radar mortar rotation animation

    return (
        # Player sprites
        player,
        player_missile,

        # Entity sprites
        radar_mortar_sprites,

        # Background sprites
        upper_brown_bg,
        lower_brown_bg,
        playing_field_bg,
        gray_gui_bg,
        lower_mountain,
        upper_mountain,
        black_stripe,

        # Instrument panel sprites
        gui_colored_background,
        gui_black_background,
        gui_text_score,
        gui_text_energy,
        gui_text_shields,
        gui_text_dtime,
        gui_score_digits,
        gui_score_comma,
    )

(
    # Player sprites
    SPRITE_PLAYER,
    SPRITE_PLAYER_MISSILE,

    # Entity sprites
    SPRITE_RADAR_MORTAR,

    # Background sprites
    SPRITE_UPPER_BROWN_BG,
    SPRITE_LOWER_BROWN_BG,
    SPRITE_PLAYING_FIELD_BG,
    SPRITE_GRAY_GUI_BG,
    SPRITE_LOWER_MOUNTAIN,
    SPRITE_UPPER_MOUNTAIN,
    SPRITE_BLACK_STRIPE,

    # Instrument panel sprites
    SPRITE_GUI_COLORED_BACKGROUND,
    SPRITE_GUI_BLACK_BACKGROUND,
    SPRITE_GUI_TEXT_SCORE,
    SPRITE_GUI_TEXT_ENERGY,
    SPRITE_GUI_TEXT_SHIELDS,
    SPRITE_GUI_TEXT_DTIME,
    SPRITE_GUI_SCORE_DIGITS,
    SPRITE_GUI_SCORE_COMMA,
) = load_sprites()

# -------- Game Logic --------

@jax.jit
def maybe_initialize_random_entity(entities, state):
    """
    Spawns an entity with a random type if no other entities are present in the current state.
    """
    key_pick_type, key_intern = jax.random.split(state.rng_key) # rng for picking the type and rng for type-specific need for randomness

    all_is_in_current_event_flags = jnp.stack([
        entities.radar_mortar_state.is_in_current_event,
        entities.byte_bat_state.is_in_current_event,
        entities.rock_muncher_state.is_in_current_event,
        entities.homing_missile_state.is_in_current_event,
        entities.forcefield_state.is_in_current_event,
        entities.dense_pack_state.is_in_current_event,
        entities.detonator_state.is_in_current_event,
        entities.energy_pod_state.is_in_current_event,
    ])
    active_event = jnp.any(all_is_in_current_event_flags) # If there is an entity that is in the current event

    def initialize_radar_mortar(entities, state):

        top_or_bot = jax.random.bernoulli(key_intern)

        new_radar_mortar_state = RadarMortarState(
                is_in_current_event = jnp.bool(True),
                is_alive= jnp.bool(True),
                x=jnp.array(RADAR_MORTAR_SPAWN_X).astype(entities.radar_mortar_state.x.dtype),
                y=jnp.where(top_or_bot, RADAR_MORTAR_SPAWN_Y_BOTTOM, RADAR_MORTAR_SPAWN_Y_TOP),
            )
        return entities._replace(radar_mortar_state=new_radar_mortar_state)

    def initialize_byte_bat(entities, state):
        new_state = entities.byte_bat_state._replace(is_in_current_event=jnp.bool(True))
        return entities._replace(byte_bat_state=new_state)

    def initialize_rock_muncher(entities, state):
        new_state = entities.rock_muncher_state._replace(is_in_current_event=jnp.bool(True))
        return entities._replace(rock_muncher_state=new_state)

    def initialize_homing_missile(entities, state):
        new_state = entities.homing_missile_state._replace(is_in_current_event=jnp.bool(True))
        return entities._replace(homing_missile_state=new_state)

    def initialize_forcefield(entities, state):
        new_state = entities.forcefield_state._replace(is_in_current_event=jnp.bool(True))
        return entities._replace(forcefield_state=new_state)

    def initialize_dense_pack(entities, state):
        new_state = entities.dense_pack_state._replace(is_in_current_event=jnp.bool(True))
        return entities._replace(dense_pack_state=new_state)

    def initialize_detonator(entities, state):
        new_state = entities.detonator_state._replace(is_in_current_event=jnp.bool(True))
        return entities._replace(detonator_state=new_state)

    def initialize_energy_pod(entities, state):
        new_state = entities.energy_pod_state._replace(is_in_current_event=jnp.bool(True))
        return entities._replace(energy_pod_state=new_state)

    init_fns = [
        initialize_radar_mortar,
        initialize_byte_bat,
        initialize_rock_muncher,
        initialize_homing_missile,
        initialize_forcefield,
        initialize_dense_pack,
        initialize_detonator,
        initialize_energy_pod,
    ] # All initialize functions of all entity types

    def initialize_random_entity(_):
        picked_index = jax.random.randint(key_pick_type, shape=(), minval=0, maxval=1) # TODO: Change maxval to len(init_fns) when all init functions are implemented
        chosen_fn = lambda i: jax.lax.switch(i, init_fns, entities, state)
        return chosen_fn(picked_index) # Initialize function of randomly picked entity

    return jax.lax.cond(
        active_event,
        lambda _: entities,         # Return the current entities state if there still is an active entity present
        initialize_random_entity,   # Else spawn a new entity with random type (see initialize_random_entity)
        operand=None,
    )


@jax.jit
def mountains_step(
        mountain_state: MountainState, state: LaserGatesState
) -> MountainState:

    # If this is true, update the position
    update_tick = state.step_counter % UPDATE_EVERY == 0

    # Update x positions
    new_x1 = jnp.where(update_tick, mountain_state.x1 - UPDATE_EVERY * state.scroll_speed, mountain_state.x1)
    new_x2 = jnp.where(update_tick, mountain_state.x2 - UPDATE_EVERY * state.scroll_speed, mountain_state.x2)
    new_x3 = jnp.where(update_tick, mountain_state.x3 - UPDATE_EVERY * state.scroll_speed, mountain_state.x3)

    # If completely behind left border, set x position to the right again
    new_x1 = jnp.where(new_x1 < 0 - MOUNTAIN_SIZE[0], new_x3 + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE, new_x1)
    new_x2 = jnp.where(new_x2 < 0 - MOUNTAIN_SIZE[0], new_x1 + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE, new_x2)
    new_x3 = jnp.where(new_x3 < 0 - MOUNTAIN_SIZE[0], new_x2 + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE, new_x3)

    return MountainState(x1=new_x1, x2=new_x2, x3=new_x3, y=mountain_state.y)

@jax.jit
def all_entities_step(game_state: LaserGatesState) -> Entities:
    """
    steps the entity (actually entities, but we only have one entity per event) that is currently in game (if is_in_current_event of said entity is True).
    """

    def radar_mortar_step(state: LaserGatesState) -> RadarMortarState:
        radar_mortar_state = state.entities.radar_mortar_state
        new_x = radar_mortar_state.x - state.scroll_speed

        kill_done = jnp.bool(False) # TODO: Collision detection + animation with timer in state
        no_longer_in_event = jnp.logical_or(radar_mortar_state.x < 0 - RADAR_MORTAR_SIZE[0], kill_done)

        return radar_mortar_state._replace(
            x=new_x.astype(radar_mortar_state.x.dtype),
            is_in_current_event=jnp.logical_not(no_longer_in_event),
        )

    def byte_bat_step(state: LaserGatesState) -> ByteBatState:
        return state.entities.byte_bat_state

    def rock_muncher_step(state: LaserGatesState) -> RockMuncherState:
        return state.entities.rock_muncher_state

    def homing_missile_step(state: LaserGatesState) -> HomingMissileState:
        return state.entities.homing_missile_state

    def forcefield_step(state: LaserGatesState) -> ForceFieldState:
        return state.entities.forcefield_state

    def densepack_step(state: LaserGatesState) -> DensepackState:
        return state.entities.dense_pack_state

    def detonator_step(state: LaserGatesState) -> DetonatorState:
        return state.entities.detonator_state

    def energy_pod_step(state: LaserGatesState) -> EnergyPodState:
        return state.entities.energy_pod_state


    def entity_maybe_step(step_fn, entity_state):
        return jax.lax.cond(
            entity_state.is_in_current_event,
            lambda _: step_fn(game_state),  # Only update the entity if in current event
            lambda _: entity_state,         # Use old state if not in current event
            operand=None
        )

    s_entities = game_state.entities
    return Entities(
        radar_mortar_state = entity_maybe_step(radar_mortar_step, s_entities.radar_mortar_state),
        byte_bat_state = entity_maybe_step(byte_bat_step, s_entities.byte_bat_state),
        rock_muncher_state = entity_maybe_step(rock_muncher_step, s_entities.rock_muncher_state),
        homing_missile_state = entity_maybe_step(homing_missile_step, s_entities.homing_missile_state),
        forcefield_state = entity_maybe_step(forcefield_step, s_entities.forcefield_state),
        dense_pack_state = entity_maybe_step(densepack_step, s_entities.dense_pack_state),
        detonator_state = entity_maybe_step(detonator_step, s_entities.detonator_state),
        energy_pod_state = entity_maybe_step(energy_pod_step, s_entities.energy_pod_state),
    ) # Return the new step state for every entity. Only the currently active entity is updated. Since we use lax.cond (which is lazy), only the active branch is executed.


@jax.jit
def player_step(
        state: LaserGatesState, action: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array]:
    up = jnp.isin(action, jnp.array([
        Action.UP,
        Action.UPRIGHT,
        Action.UPLEFT,
        Action.UPFIRE,
        Action.UPRIGHTFIRE,
        Action.UPLEFTFIRE
    ]))
    down = jnp.isin(action, jnp.array([
        Action.DOWN,
        Action.DOWNRIGHT,
        Action.DOWNLEFT,
        Action.DOWNFIRE,
        Action.DOWNRIGHTFIRE,
        Action.DOWNLEFTFIRE
    ]))
    left = jnp.isin(action, jnp.array([
        Action.LEFT,
        Action.UPLEFT,
        Action.DOWNLEFT,
        Action.LEFTFIRE,
        Action.UPLEFTFIRE,
        Action.DOWNLEFTFIRE
    ]))
    right = jnp.isin(action, jnp.array([
        Action.RIGHT,
        Action.UPRIGHT,
        Action.DOWNRIGHT,
        Action.RIGHTFIRE,
        Action.UPRIGHTFIRE,
        Action.DOWNRIGHTFIRE
    ]))

    # Move x
    delta_x = jnp.where(left, -PLAYER_VELOCITY_X, jnp.where(right, PLAYER_VELOCITY_X, 0))
    player_x = jnp.clip(state.player_x + delta_x, PLAYER_BOUNDS[0][0], PLAYER_BOUNDS[0][1])

    # Move y
    delta_y = jnp.where(up, -PLAYER_VELOCITY_Y, jnp.where(down, PLAYER_VELOCITY_Y, 0))
    player_y = jnp.clip(state.player_y + delta_y, PLAYER_BOUNDS[1][0], PLAYER_BOUNDS[1][1])

    # Player facing direction
    new_player_facing_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_facing_direction))

    no_x_input = jnp.logical_and(
        jnp.logical_not(left), jnp.logical_not(right)
        )

    # SCROLL LEFT
    player_x = jnp.where(no_x_input, player_x - SCROLL_SPEED, player_x)

    return player_x, player_y, new_player_facing_direction

@jax.jit
def player_missile_step(
        state: LaserGatesState, action: chex.Array
) -> PlayerMissileState:

    fire = jnp.isin(action, jnp.array([
        Action.FIRE,
        Action.UPFIRE,
        Action.RIGHTFIRE,
        Action.LEFTFIRE,
        Action.DOWNFIRE,
        Action.UPRIGHTFIRE,
        Action.UPLEFTFIRE,
        Action.DOWNRIGHTFIRE,
        Action.DOWNLEFTFIRE
    ]))


    is_alive = state.player_missile.direction != 0
    out_of_bounds = jnp.logical_or(
        state.player_missile.x < 0 - PLAYER_MISSILE_SIZE[0],
        state.player_missile.x > WIDTH
    )
    kill = jnp.logical_and(is_alive, out_of_bounds)

    # Kill missile
    new_x = jnp.where(kill, 0, state.player_missile.x)
    new_y = jnp.where(kill, 0, state.player_missile.y)
    new_direction = jnp.where(kill, 0, state.player_missile.direction)
    new_velocity = jnp.where(kill, 0, state.player_missile.velocity)

    # Move missile
    new_x = jnp.where(
        is_alive,
        new_x + jnp.where(new_direction > 0, state.player_missile.velocity, -state.player_missile.velocity),
        new_x
    ) # Move by the velocity in state
    new_velocity = jnp.where(
        is_alive,
        new_velocity * PLAYER_MISSILE_VELOCITY_MULTIPLIER,
        new_velocity
    ) # Multiply velocity by given constant

    # Spawn missile
    spawn = jnp.logical_and(jnp.logical_not(is_alive), fire)
    new_x = jnp.where(spawn, jnp.where(
        state.player_facing_direction > 0,
        state.player_x + PLAYER_SIZE[0],
        state.player_x - 2 * PLAYER_SIZE[0] - 1
    ), new_x)
    new_y = jnp.where(spawn, state.player_y + 4, new_y)
    new_direction = jnp.where(spawn, state.player_facing_direction, new_direction)
    new_velocity = jnp.where(spawn, PLAYER_MISSILE_INITIAL_VELOCITY, new_velocity)

    return PlayerMissileState(x=new_x, y=new_y, direction=new_direction, velocity=new_velocity)

@jax.jit
def check_collision_single(pos1, size1, pos2, size2):
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

    return jnp.logical_and(horizontal_overlap, vertical_overlap)


@jax.jit
def any_collision_for_group(player_pos: jnp.ndarray,
                            player_size: jnp.ndarray,
                            group_xs: jnp.ndarray,
                            group_ys: jnp.ndarray,
                            segment_offsets: jnp.ndarray,
                            segment_sizes: jnp.ndarray) -> jnp.ndarray:
    """
    Checks collision with a group of objects (e.g., multiple mountain chains),
    each composed of identical segments (offsets + sizes).

    - player_pos: (2,)
    - player_size: (2,)
    - group_xs:   (n_groups,)
    - group_ys:   (n_groups,)
    - segment_offsets: (n_segments, 2)
    - segment_sizes:   (n_segments, 2)

    Returns a Boolean-(n_groups,) array, one per group.
    """

    # vectorize single-segment collision check
    collision_per_segment = jax.vmap(
        check_collision_single,
        in_axes=(None, None, 0, 0),
        out_axes=0)

    def collision_for_one(x, y):
        # compute absolute positions of all segments in this group (n_segments, 2)
        block_positions = jnp.stack([
            x + segment_offsets[:, 0],
            y + segment_offsets[:, 1],
        ], axis=-1)

        # check collisions for each segment
        seg_hits = collision_per_segment(
            player_pos, player_size,
            block_positions, segment_sizes
        )
        # return True if any segment collides
        return jnp.any(seg_hits)

    # map over all group positions
    return jax.vmap(collision_for_one, in_axes=(0, 0))(group_xs, group_ys)

@jax.jit
def check_player_collision(
        state: LaserGatesState
) -> tuple[chex.Array, chex.Array]:

    # -------- Bounds and mountains --------

    # Segment definitions for Upper Mountains
    upper_offsets = jnp.array([
        ( 0, 0),
        ( 8, 3),
        (12, 6),
        (20, 9),
    ], dtype=jnp.int32)
    upper_sizes = jnp.array([
        (60, 3),
        (44, 3),
        (36, 3),
        (20, 3),
    ], dtype=jnp.int32)

    # Segment definitions for Lower Mountains
    lower_offsets = jnp.array([
        (20, 0),
        (12, 3),
        ( 8, 6),
        ( 0, 9),
    ], dtype=jnp.int32)
    lower_sizes = jnp.array([
        (20, 3),
        (36, 3),
        (44, 3),
        (60, 3)
    ], dtype=jnp.int32)

    # Extract group coordinates from state
    upper_xs = jnp.array([
        state.upper_mountains.x1,
        state.upper_mountains.x2,
        state.upper_mountains.x3,
    ], dtype=jnp.int32)
    upper_ys = jnp.array([
        state.upper_mountains.y,
        state.upper_mountains.y,
        state.upper_mountains.y,
    ], dtype=jnp.int32)

    lower_xs = jnp.array([
        state.lower_mountains.x1,
        state.lower_mountains.x2,
        state.lower_mountains.x3,
    ], dtype=jnp.int32)
    lower_ys = jnp.array([
        state.lower_mountains.y,
        state.lower_mountains.y,
        state.lower_mountains.y,
    ], dtype=jnp.int32)

    # Player parameters
    player_pos  = jnp.array((state.player_x, state.player_y), dtype=jnp.int32)
    player_size = jnp.array(PLAYER_SIZE, dtype=jnp.int32)

    # Check collisions for both groups
    upper_collisions = any_collision_for_group(
        player_pos, player_size, upper_xs, upper_ys,
        segment_offsets=upper_offsets,
        segment_sizes=upper_sizes
    )
    lower_collisions = any_collision_for_group(
        player_pos, player_size, lower_xs, lower_ys,
        segment_offsets=lower_offsets,
        segment_sizes=lower_sizes
    )

    # Include normal bound
    upper_collision = jnp.logical_or(jnp.any(upper_collisions), state.player_y <= PLAYER_BOUNDS[1][0])
    lower_collision = jnp.logical_or(jnp.any(lower_collisions), state.player_y >= PLAYER_BOUNDS[1][1])

    return upper_collision, lower_collision


class JaxLaserGates(JaxEnvironment[LaserGatesState, LaserGatesObservation, LaserGatesInfo]):
    def __init__(self, reward_funcs: list[callable] =None):
        super().__init__()
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
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
            Action.DOWNLEFTFIRE
        ]
        # self.frame_stack_size = 4 # ???
        # self.obs_size = 1024 # ???

    # TODO: add other funtions if needed

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: LaserGatesState) -> LaserGatesObservation:
        # TODO: fill
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(0),
            height=jnp.array(0), # TODO: Import sizes
            active=jnp.array(1),
        )

        return LaserGatesObservation(
            player=player,
        )

    @partial(jax.jit, static_argnums=(0, ))
    def _get_info(self, state: LaserGatesState, all_rewards: jnp.ndarray) -> LaserGatesInfo:
        # TODO: fill
        return LaserGatesInfo(
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @jax.jit
    def _get_env_reward(self, previous_state: LaserGatesState, state: LaserGatesState) -> jnp.ndarray:
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: LaserGatesState, state: LaserGatesState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @jax.jit
    def _get_done(self, state: LaserGatesState) -> bool:
        return state.shields <= 0

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self) -> Tuple[LaserGatesObservation, LaserGatesState]:
        """Initialize game state"""

        initial_lower_mountains = MountainState(
            x1=jnp.array(LOWER_MOUNTAINS_START_X),
            x2=jnp.array(LOWER_MOUNTAINS_START_X + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE),
            x3=jnp.array(LOWER_MOUNTAINS_START_X + 2 * MOUNTAIN_SIZE[0] + 2 * MOUNTAINS_DISTANCE),
            y=jnp.array(LOWER_MOUNTAINS_Y)
        )

        initial_upper_mountains = MountainState(
            x1=jnp.array(UPPER_MOUNTAINS_START_X),
            x2=jnp.array(UPPER_MOUNTAINS_START_X + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE),
            x3=jnp.array(UPPER_MOUNTAINS_START_X + 2 * MOUNTAIN_SIZE[0] + 2 * MOUNTAINS_DISTANCE),
            y=jnp.array(UPPER_MOUNTAINS_Y)
        )

        initial_player_missile = PlayerMissileState(
            x=jnp.array(0),
            y=jnp.array(0),
            direction=jnp.array(0),
            velocity=jnp.array(0),
        )

        initial_entities = Entities(
            radar_mortar_state=RadarMortarState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
            ),
            byte_bat_state=ByteBatState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
            ),
            rock_muncher_state=RockMuncherState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
            ),
            homing_missile_state=HomingMissileState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
            ),
            forcefield_state=ForceFieldState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
            ),
            dense_pack_state=DensepackState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
            ),
            detonator_state=DetonatorState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
            ),
            energy_pod_state=EnergyPodState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
            ),
        )

        key = jax.random.PRNGKey(time.time_ns() % (2**32)) # Pseudo random number generator seed key, based on current system time.
        new_key0, key0 = jax.random.split(key, 2)

        reset_state = LaserGatesState( # TODO: fill
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_facing_direction=jnp.array(1),
            player_missile=initial_player_missile,
            player_collision=jnp.bool(False),
            entities=initial_entities,
            animation_timer=jnp.array(0).astype(jnp.uint8),
            lower_mountains=initial_lower_mountains,
            upper_mountains=initial_upper_mountains,
            score=jnp.array(0), # Start with no initial score
            energy=jnp.array(MAX_ENERGY), # As the manual says, energy is consumed at a regular pace. We use 5100 for the initial value and subtract one for every frame to match the timing of the real game. (It takes 85 seconds for the energy to run out. 85 * 60 (fps) = 5100
            shields=jnp.array(MAX_SHIELDS), # As the manual says, the Dante Dart starts with 24 shield units
            dtime=jnp.array(MAX_DTIME), # Same idea as energy.
            scroll_speed=jnp.array(SCROLL_SPEED),
            rng_key=new_key0,  # Pseudo random number generator seed key, based on current time and initial key used.
            step_counter=jnp.array(0),
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0, ))
    def step(
            self, state: LaserGatesState, action: Action
    ) -> Tuple[LaserGatesObservation, LaserGatesState, float, bool, LaserGatesInfo]:
        # TODO: fill

        # -------- Move player --------
        new_player_x, new_player_y, new_player_facing_direction = player_step(state, action)
        player_animation_timer = state.animation_timer
        new_player_animation_timer = jnp.where(player_animation_timer != 0, player_animation_timer - 1, player_animation_timer)

        # -------- Move player missile --------
        new_player_missile_state = player_missile_step(state, action)

        # -------- Move entities --------

        new_entities = all_entities_step(state)
        # new_entities = check_player_missile_collision_entities # TODO: Implement
        # new_entities = check_player_collision_entities # TODO: Implement
        new_entities = maybe_initialize_random_entity(new_entities, state)

        # -------- Move mountains --------
        new_lower_mountains_state = mountains_step(state.lower_mountains, state)
        new_upper_mountains_state = mountains_step(state.upper_mountains, state)

        # -------- Update scroll speed --------
        new_scroll_speed = jnp.where(state.player_x != PLAYER_BOUNDS[0][1], SCROLL_SPEED, SCROLL_SPEED * SCROLL_MULTIPLIER)

        # -------- Check player collision --------
        upper_col, lower_col = check_player_collision(state)
        any_player_collision = jnp.logical_or(upper_col, lower_col)

        # -------- Update things that have to be updated at collision --------
        new_player_animation_timer = jnp.where(any_player_collision, 255, new_player_animation_timer)

        new_player_y = jnp.where(upper_col, new_player_y + 4, new_player_y)
        new_player_y = jnp.where(lower_col, new_player_y - 4, new_player_y)

        # -------- Update energy, score, shields and d-time --------
        new_energy = state.energy - 1
        new_score = state.score + 1
        new_shields = jnp.where(jnp.logical_or(upper_col, lower_col), state.shields - 1, state.shields)

        # -------- New rng key --------
        new_rng_key, new_key = jax.random.split(state.rng_key)

        return_state = state._replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_facing_direction=new_player_facing_direction,
            animation_timer=new_player_animation_timer,
            player_missile=new_player_missile_state,
            player_collision=any_player_collision,
            entities=new_entities,
            lower_mountains=new_lower_mountains_state,
            upper_mountains=new_upper_mountains_state,
            scroll_speed=new_scroll_speed,
            score=new_score,
            energy=new_energy,
            shields=new_shields,
            rng_key=new_rng_key,
            step_counter=state.step_counter + 1
        )

        obs = self._get_observation(return_state)
        all_rewards = self._get_all_rewards(state, return_state)
        info = self._get_info(return_state, all_rewards)

        return obs, return_state, 0.0, False, info

class LaserGatesRenderer(AtraJaxisRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        def recolor_sprite(sprite: jnp.ndarray, color: jnp.ndarray) -> jnp.ndarray:
            assert sprite.ndim == 3 and sprite.shape[2] in (3, 4), "Sprite must be HxWx3 or HxWx4"
            assert color.shape[0] == sprite.shape[2], "Color channels must match sprite channels"

            # Define a visibility mask: pixel is visible if any of its channels > 0
            visible_mask = jnp.any(sprite != 0, axis=-1)  # (H, W)
            visible_mask = visible_mask[:, :, None]  # (H, W, 1) for broadcasting

            # Broadcast color to the same shape as sprite
            color_broadcasted = jnp.broadcast_to(color, sprite.shape)

            # Where visible, use the new color; otherwise keep black (zeros)
            return jnp.where(visible_mask, color_broadcasted, 0)

        any_player_collision = state.player_collision

        # -------- Render background parts --------

        # Upper brown background above upper mountains
        raster = aj.render_at(
            raster,
            0,
            0,
            SPRITE_UPPER_BROWN_BG,
        )

        # Playing field background, color adjusts if player collision
        pfb_t = jnp.clip(255 * jnp.exp(-PLAYING_FILED_BG_COLOR_FADE_SPEED * (255 - state.animation_timer)), 0, 255)
        pfb_t = pfb_t.astype(jnp.uint8)
        raster = aj.render_at(
            raster,
            0,
            19,
            recolor_sprite(SPRITE_PLAYING_FIELD_BG, jnp.array((PLAYING_FILED_BG_COLLISION_COLOR[0], PLAYING_FILED_BG_COLLISION_COLOR[1], PLAYING_FILED_BG_COLLISION_COLOR[2], pfb_t))),
        )

        # Lower brown background below upper mountains
        raster = aj.render_at(
            raster,
            0,
            92,
            SPRITE_LOWER_BROWN_BG,
        )

        # Background of instrument panel
        raster = aj.render_at(
            raster,
            0,
            109,
            SPRITE_GRAY_GUI_BG,
        )

        # -------- Render gui --------

        sprite_gui_colored_background_blue_bg = recolor_sprite(SPRITE_GUI_COLORED_BACKGROUND, jnp.array(GUI_COLORED_BACKGROUND_COLOR_BLUE))
        sprite_gui_colored_background_green_bg = recolor_sprite(SPRITE_GUI_COLORED_BACKGROUND, jnp.array(GUI_COLORED_BACKGROUND_COLOR_GREEN))
        sprite_gui_colored_background_beige_bg = recolor_sprite(SPRITE_GUI_COLORED_BACKGROUND, jnp.array(GUI_COLORED_BACKGROUND_COLOR_BEIGE))
        sprite_gui_colored_background_gray_bg = recolor_sprite(SPRITE_GUI_COLORED_BACKGROUND, jnp.array(GUI_COLORED_BACKGROUND_COLOR_GRAY))

        # Colored backgrounds ---------------

        # Colored background for score
        score_col_bg_y = 111 + GUI_Y_SPACE_BETWEEN_PLAYING_FIELD
        raster = aj.render_at(
            raster,
            16,
            score_col_bg_y,
            sprite_gui_colored_background_blue_bg,
        )

        # Colored background for energy
        energy_col_bg_y = 111 + GUI_Y_SPACE_BETWEEN_PLAYING_FIELD + GUI_COLORED_BACKGROUND_SIZE[1] + GUI_Y_SPACE_BETWEEN_BACKGROUNDS
        raster = aj.render_at(
            raster,
            16,
            energy_col_bg_y,
            sprite_gui_colored_background_green_bg,
        )

        # Colored background for shields
        shields_col_bg_y = 111 + GUI_Y_SPACE_BETWEEN_PLAYING_FIELD + 2 * GUI_COLORED_BACKGROUND_SIZE[1] + 2 * GUI_Y_SPACE_BETWEEN_BACKGROUNDS
        raster = aj.render_at(
            raster,
            16,
            shields_col_bg_y,
            sprite_gui_colored_background_green_bg,
        )

        # Colored background for d-time # TODO: Implement color rendering logic
        dtime_col_bg_y = 111 + GUI_Y_SPACE_BETWEEN_PLAYING_FIELD + 3 * GUI_COLORED_BACKGROUND_SIZE[1] + 3 * GUI_Y_SPACE_BETWEEN_BACKGROUNDS
        raster = aj.render_at(
            raster,
            16,
            dtime_col_bg_y,
            sprite_gui_colored_background_green_bg,
        )

        # Black backgrounds ---------------

        # Black background for score
        raster = aj.render_at(
            raster,
            16 + GUI_BLACK_BACKGROUND_X_OFFSET,
            score_col_bg_y + 1,
            SPRITE_GUI_BLACK_BACKGROUND,
        )

        # Black background for energy
        raster = aj.render_at(
            raster,
            16 + GUI_BLACK_BACKGROUND_X_OFFSET,
            energy_col_bg_y + 1,
            SPRITE_GUI_BLACK_BACKGROUND,
        )

        # Black background for shields
        raster = aj.render_at(
            raster,
            16 + GUI_BLACK_BACKGROUND_X_OFFSET,
            shields_col_bg_y + 1,
            SPRITE_GUI_BLACK_BACKGROUND,
        )

        # Black background for d-time
        raster = aj.render_at(
            raster,
            16 + GUI_BLACK_BACKGROUND_X_OFFSET,
            dtime_col_bg_y + 1,
            SPRITE_GUI_BLACK_BACKGROUND,
        )

        # Text ---------------

        # score text
        required_text_and_bar_color = jnp.where(jnp.array(True), jnp.array(GUI_TEXT_COLOR_GRAY), jnp.array(GUI_TEXT_COLOR_BEIGE))
        raster = aj.render_at(
            raster,
            16 + GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            score_col_bg_y + 2,
            recolor_sprite(SPRITE_GUI_TEXT_SCORE, required_text_and_bar_color),
        )

        # energy text
        raster = aj.render_at(
            raster,
            16 + GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            energy_col_bg_y + 2,
            recolor_sprite(SPRITE_GUI_TEXT_ENERGY, required_text_and_bar_color),
        )

        # shields text
        raster = aj.render_at(
            raster,
            16 + GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            shields_col_bg_y + 2,
            recolor_sprite(SPRITE_GUI_TEXT_SHIELDS, required_text_and_bar_color),
        )

        # d-time text
        raster = aj.render_at(
            raster,
            16 + GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            dtime_col_bg_y + 2,
            recolor_sprite(SPRITE_GUI_TEXT_DTIME, required_text_and_bar_color),
        )

        # Bars ---------------

        # energy bar
        raster = aj.render_bar(
            raster, # raster
            16 + GUI_BLACK_BACKGROUND_X_OFFSET + 4, # x pos
            energy_col_bg_y + 8, # y pos
            state.energy, # current value
            MAX_ENERGY, # maximum value
            40, # width
            2, # height
            required_text_and_bar_color, # color of filled part
            jnp.array((0, 0, 0, 0)) # color of unfilled part
        )

        # shields bar
        raster = aj.render_bar(
            raster, # raster
            16 + GUI_BLACK_BACKGROUND_X_OFFSET + 4, # x pos
            shields_col_bg_y + 8, # y pos
            state.shields, # current value
            MAX_SHIELDS, # maximum value
            31, # width
            2, # height
            required_text_and_bar_color, # color of filled part
            jnp.array((0, 0, 0, 0)) # color of unfilled part
        )

        # d-time bar TODO: Implement correct color picking
        raster = aj.render_bar(
            raster, # raster
            16 + GUI_BLACK_BACKGROUND_X_OFFSET + 4, # x pos
            dtime_col_bg_y + 8, # y pos
            state.dtime, # current value
            MAX_DTIME, # maximum value
            40, # width
            2, # height
            required_text_and_bar_color, # color of filled part
            jnp.array((0, 0, 0, 0)) # color of unfilled part
        )

        # Score ---------------

        # digits of score
        score_array = aj.int_to_digits(state.score, 6) # Convert integer to array with its digits

        recolor_single = lambda sprite_idx: recolor_sprite(sprite_idx, required_text_and_bar_color)
        recolored_sprites = jax.vmap(recolor_single)(SPRITE_GUI_SCORE_DIGITS) # Vmap over all digit sprites and recolor to desired color

        first_non_zero = jnp.argmax(score_array != 0) # Index of first element in score_array that is not zero
        num_to_render = score_array.shape[0] - first_non_zero # number of digits we have to render
        base_x = 16 + GUI_BLACK_BACKGROUND_X_OFFSET + 52 # base x position
        number_spacing = 4 # Spacing of digits (including digit itself)
        score_numbers_x = base_x - number_spacing * num_to_render # Subtrating offset of x position, since we want the score to be right-aligned

        raster = jnp.where(state.score > 0, # Render only if score is more than 0
                           aj.render_label_selective(
                               raster,
                               score_numbers_x,
                               score_col_bg_y + 3,
                               score_array,
                               recolored_sprites,
                               first_non_zero,
                               num_to_render,
                               number_spacing
                           ),
                           raster
                           )

        # Comma, render only if score > 999
        raster = jnp.where(state.score > 999,
                           aj.render_at(
                               raster,
                               base_x - 14,
                               score_col_bg_y + 8,
                               recolor_sprite(SPRITE_GUI_SCORE_COMMA, required_text_and_bar_color),
                           ),
                           raster
                           )


        # -------- Render player --------
        timer = state.animation_timer.astype(jnp.int32) - (255 - PLAYER_COLOR_CHANGE_DURATION)
        raster = aj.render_at(
            raster,
            state.player_x,
            state.player_y,
            recolor_sprite(SPRITE_PLAYER, jnp.where(timer <= 0, jnp.array(PLAYER_NORMAL_COLOR), jnp.array(PLAYER_COLLISION_COLOR))),
            flip_horizontal=state.player_facing_direction < 0,
        )

        # -------- Render player missile --------

        base_r, base_g, base_b, base_t = PLAYER_MISSILE_BASE_COLOR
        color_change = state.player_missile.velocity * PLAYER_MISSILE_COLOR_CHANGE_SPEED

        r = jnp.clip(base_r + color_change, 0, 255)
        g = jnp.clip(base_g + color_change, 0, 255)
        b = jnp.clip(base_b + color_change, 0, 255)
        t = jnp.clip(base_t + color_change, 0, 255)

        colored_player_missile = recolor_sprite(SPRITE_PLAYER_MISSILE, jnp.array((r, g, b, t)))
        raster = jnp.where(state.player_missile.direction != 0,
                      aj.render_at(
                      raster,
                      state.player_missile.x,
                      state.player_missile.y,
                      colored_player_missile,
                      flip_horizontal=state.player_missile.direction < 0,
                      ),
                  raster
                  )

        # -------- Render entities --------

        # Render radar mortar
        radar_mortar_frame = aj.get_sprite_frame(SPRITE_RADAR_MORTAR, state.step_counter)
        raster = jnp.where(state.entities.radar_mortar_state.is_alive,
                           aj.render_at(
                               raster,
                               state.entities.radar_mortar_state.x,
                               state.entities.radar_mortar_state.y,
                               radar_mortar_frame,
                               flip_vertical=state.entities.radar_mortar_state.y == RADAR_MORTAR_SPAWN_Y_TOP,
                           ),
                           raster
                           )

        # -------- Render mountains --------

        # Lower mountains
        raster = aj.render_at(
            raster,
            state.lower_mountains.x1,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
        )

        raster = aj.render_at(
            raster,
            state.lower_mountains.x2,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
        )

        raster = aj.render_at(
            raster,
            state.lower_mountains.x3,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
        )

        # Upper mountains
        raster = aj.render_at(
            raster,
            state.upper_mountains.x1,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
        )

        raster = aj.render_at(
            raster,
            state.upper_mountains.x2,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
        )

        raster = aj.render_at(
            raster,
            state.upper_mountains.x3,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
        )

        # Weird black stripe 1
        raster = aj.render_at(
            raster,
            0,
            18,
            SPRITE_BLACK_STRIPE,
        )

        # Weird black stripe 2
        raster = aj.render_at(
            raster,
            0,
            109,
            SPRITE_BLACK_STRIPE,
        )

        return raster

def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(Action.UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(Action.UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(Action.DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(Action.DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(Action.UPFIRE)
    if down and fire:
        return jnp.array(Action.DOWNFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if right and fire:
        return jnp.array(Action.RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(Action.UPRIGHT)
    if up and left:
        return jnp.array(Action.UPLEFT)
    if down and right:
        return jnp.array(Action.DOWNRIGHT)
    if down and left:
        return jnp.array(Action.DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(Action.UP)
    if down:
        return jnp.array(Action.DOWN)
    if left:
        return jnp.array(Action.LEFT)
    if right:
        return jnp.array(Action.RIGHT)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)

if __name__ == "__main__":
    # Initialize game and renderer
    game = JaxLaserGates()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    clock = pygame.time.Clock()

    renderer_AtraJaxis = LaserGatesRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_obs, curr_state = jitted_reset()

    # Game loop with rendering
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        curr_obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )
                        print(f"Observations: {curr_obs}")
                        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_obs, curr_state, reward, done, info = jitted_step(
                    curr_state, action
                )

        # render and update pygame
        raster = renderer_AtraJaxis.render(curr_state)
        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        counter += 1
        clock.tick(60)

    pygame.quit()