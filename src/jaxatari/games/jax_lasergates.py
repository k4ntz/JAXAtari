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
PLAYING_FIELD_BG_COLLISION_COLOR = (255, 255, 255, 255)
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
PLAYER_MISSILE_SIZE = (16, 1) # Width, Height
PLAYER_MISSILE_BASE_COLOR = (140, 79, 24, 255) # Initial color of player missile. Every value except for transparency is incremented by the missiles velocity * PLAYER_MISSILE_COLOR_CHANGE_SPEED
PLAYER_MISSILE_COLOR_CHANGE_SPEED = 10 # Defines how fast the player missile changes its color towards white.

PLAYER_MISSILE_INITIAL_VELOCITY = 2.5 # Starting speed of player missile
PLAYER_MISSILE_VELOCITY_MULTIPLIER = 1.1 # Multiply the current speed at a given moment of the player missile by this number

# -------- Instrument panel constants --------
SHIELD_LOSS_COL_SMALL = 1 # see game manual, different collision lose a different amount of shield points
SHIELD_LOSS_COL_BIG = 6

# -------- Entity constants (constants that apply to all entity types --------
NUM_ENTITY_TYPES = 8 # How many different (!) entity types there are
ENTITY_DEATH_SPRITES_SIZE = (8, 45) # Width, Height
ENTITY_DEATH_SPRITE_Y_OFFSET = 10 # Y offset to add to the death sprite (the constant is being added to the y coordinate)
ENTITY_MISSILE_SIZE = (4, 1) # Width, Height
ENTITY_DEATH_ANIMATION_TIMER = 100 # Duration of death sprite animation in frames

# -------- Radar mortar constants --------
RADAR_MORTAR_SIZE = (8, 26) # Width, Height
RADAR_MORTAR_COLOR_BLUE = (96, 162, 228, 255)
RADAR_MORTAR_COLOR_GRAY = (155, 155, 155, 255)

RADAR_MORTAR_SPRITE_ANIMATION_SPEED = 15 # Change sprite frame (left, middle, right) of radar mortar every RADAR_MORTAR_SPRITE_ROTATION_SPEED frames
RADAR_MORTAR_SPAWN_X = WIDTH # Spawn barely outside of bounds
RADAR_MORTAR_SPAWN_BOTTOM_Y = 66
RADAR_MORTAR_SPAWN_UPPER_Y = 19 # Since the radar mortar can spawn at the top or at the bottom of the screen, we define two y positions.

RADAR_MORTAR_MISSILE_COLOR = (85, 92, 197, 255)
RADAR_MORTAR_MISSILE_SPAWN_EVERY = 100 # A missile is spawned every RADAR_MORTAR_MISSILE_SPAWN_EVERY-th frame.
RADAR_MORTAR_MISSILE_SPEED = 3 # Speed of radar mortar missile
RADAR_MORTAR_MISSILE_SHOOT_NUMBER = 3 # How often missile gets teleported back before final shot (exept when shooting up or down)
RADAR_MORTAR_MISSILE_SMALL_OUT_OF_BOUNDS_THRESHOLD = 50 # How far the missile needs to be away from the radar mortar (vertically or/and horizontally) for the missile to be teleported back to the mortar (to be shot again)
RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD = 10 # This defines how far the player needs to be away from the radar mortar (vertically or/and horizontally) for the missile to be shot diagonally

# -------- Byte bat constants --------
BYTE_BAT_SIZE = (7, 8) # Width, Height
BYTE_BAT_COLOR = (90, 169, 99, 255)

BYTE_BAT_ANIMATION_SPEED = 16 # Flap speed of byte bat, higher is slower

BYTE_BAT_UPPER_BORDER_Y = UPPER_MOUNTAINS_Y + MOUNTAIN_SIZE[1] + 2 # Upper border where byte bat inverts direction
BYTE_BAT_BOTTOM_BORDER_Y = LOWER_MOUNTAINS_Y - MOUNTAIN_SIZE[1] # Lower border where byte bat inverts direction

BYTE_BAT_SPAWN_X = WIDTH # Spawn barely outside of screen
BYTE_BAT_SPAWN_Y = BYTE_BAT_UPPER_BORDER_Y + 1

BYTE_BAT_X_SPEED = 0.7 # Speed of byte bat in x direction
BYTE_BAT_Y_SPEED = 1 # Speed of byte bat in y direction

# -------- Rock muncher constants --------
ROCK_MUNCHER_SIZE = (8, 11) # Width, Height

ROCK_MUNCHER_ANIMATION_SPEED = 10 # Animation speed of rock muncher

ROCK_MUNCHER_UPPER_BORDER_Y = UPPER_MOUNTAINS_Y + MOUNTAIN_SIZE[1] + 5 + 10 # Upper border where rock muncher inverts direction
ROCK_MUNCHER_BOTTOM_BORDER_Y = LOWER_MOUNTAINS_Y - MOUNTAIN_SIZE[1] - 3 # Lower border where rock muncher inverts direction

ROCK_MUNCHER_SPAWN_X = WIDTH # Spawn barely outside of screen
ROCK_MUNCHER_SPAWN_Y = ROCK_MUNCHER_UPPER_BORDER_Y + 1

ROCK_MUNCHER_X_SPEED = 0.7 # Speed of rock muncher in x direction
ROCK_MUNCHER_Y_SPEED = 1 # Speed of rock muncher in y direction

ROCK_MUNCHER_MISSILE_COLOR = (85, 92, 197, 255)
ROCK_MUNCHER_MISSILE_SPAWN_EVERY = 50 # Rock muncher shoots a new missile every ROCK_MUNCHER_MISSILE_SPAWN_EVERY frames
ROCK_MUNCHER_MISSILE_SPEED = 4 # Speed of rock muncher missile

# -------- Homing Missile constants --------
HOMING_MISSILE_SIZE = (8, 5) # Width, Height

HOMING_MISSILE_Y_BOUNDS = (32, 74)
HOMING_MISSILE_PLAYER_TRACKING_RANGE = 15 # The minimum y position difference between player and homing missile needed for the homing missile to start tracking the player
HOMING_MISSILE_Y_PLAYER_OFFSET = 2 # Sets the y position this many pixels above the player (for positive numbers) or below the player (for negative numbers)
HOMING_MISSILE_X_SPEED = 2.5
HOMING_MISSILE_Y_SPEED = 1

# -------- Forcefield constants --------
FORCEFIELD_SIZE = (8, 73) # Width, Height of a single normal forcefield column
FORCEFIELD_WIDE_SIZE = (16, 73) # Width, Height of a single wide forcefield column

FORCEFIELD_IS_WIDE_PROBABILITY = 0.2 # Probability that a forcefield is wide

FORCEFIELD_FLASHING_SPACING = 40 # x spacing between the forcefields when in flashing mode
FORCEFIELD_FLASHING_SPEED = 35 # Forcefield changes state from on to off or from off to on every FORCEFIELD_FLASHING_SPEED frames

FORCEFIELD_FLEXING_SPACING = 60 # x spacing between the forcefields when in flexing mode
FORCEFIELD_FLEXING_SPEED = 0.6 # Flexing (Crushing motion) speed
FORCEFIELD_FLEXING_MINIMUM_DISTANCE = 2 # Minimum y distance between the upper and lower forcefields when flexing
FORCEFIELD_FLEXING_MAXIMUM_DISTANCE = 20 # Maximum y distance between the upper and lower forcefields when flexing

FORCEFIELD_FIXED_SPACING = 50 # x spacing between the forcefields when in fixed mode
FORCEFIELD_FIXED_SPEED = 0.5 # Fixed (up and down movement) speed
FORCEFIELD_FIXED_UPPER_BOUND = -FORCEFIELD_SIZE[1] + 33 # Highest allowed y position for forcefields while fixed
FORCEFIELD_FIXED_LOWER_BOUND = -FORCEFIELD_SIZE[1] + 68 # Lowest allowed y position for forcefields while fixed

# -------- Densepack constants --------
DENSEPACK_NORMAL_PART_SIZE = (8, 4)
DENSEPACK_WIDE_PART_SIZE  = (16, 4)
DENSEPACK_COLOR = (142, 142, 142, 255)

DENSEPACK_NUMBER_OF_PARTS = 19 # number of segments in the densepack
DENSEPACK_IS_WIDE_PROBABILITY = 0.4

# -------- Detonator constants --------
DETONATOR_SIZE = (8, 73)
DETONATOR_COLOR = (142, 142, 142, 255)

# -------- Energy pod constants --------

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

# -------- Debug constants --------
DEBUG_ACTIVATE_MOUNTAINS_SCROLL = jnp.bool(True)

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
    missile_x: chex.Array
    missile_y: chex.Array
    missile_direction: chex.Array
    shoot_again_timer: chex.Array

class ByteBatState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    direction_is_up: jnp.bool
    direction_is_left: jnp.bool

class RockMuncherState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    direction_is_up: jnp.bool
    direction_is_left: jnp.bool
    missile_x: chex.Array
    missile_y: chex.Array

class HomingMissileState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    is_tracking_player: jnp.bool

class ForceFieldState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x0: chex.Array
    y0: chex.Array
    x1: chex.Array
    y1: chex.Array
    x2: chex.Array
    y2: chex.Array
    x3: chex.Array
    y3: chex.Array
    x4: chex.Array
    y4: chex.Array
    x5: chex.Array
    y5: chex.Array
    num_of_forcefields: chex.Array
    is_wide: jnp.bool
    is_flexing: jnp.bool
    is_fixed: jnp.bool
    flash_on: jnp.bool
    flex_upper_direction_is_up: jnp.bool
    fixed_upper_direction_is_up: jnp.bool

class DensepackState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    upmost_y: chex.Array
    is_wide: jnp.bool
    number_of_parts: chex.Array
    broken_states: chex.Array

class DetonatorState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    collision_is_pin: jnp.bool

class EnergyPodState(NamedTuple):
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array

class CollisionPropertiesState(NamedTuple):
    collision_with_player: jnp.bool
    collision_with_player_missile: jnp.bool
    is_big_collision: jnp.bool
    is_energy_pod: jnp.bool
    is_detonator: jnp.bool
    is_ff_or_dp: jnp.bool
    score_to_add: chex.Array
    death_timer: chex.Array

class EntitiesState(NamedTuple):
    radar_mortar_state: RadarMortarState
    byte_bat_state: ByteBatState
    rock_muncher_state: RockMuncherState
    homing_missile_state: HomingMissileState
    forcefield_state: ForceFieldState
    dense_pack_state: DensepackState
    detonator_state: DetonatorState
    energy_pod_state: EnergyPodState

    collision_properties_state: CollisionPropertiesState

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
    animation_timer: chex.Array
    entities: EntitiesState
    lower_mountains: MountainState
    upper_mountains: MountainState
    score: chex.Array
    energy: chex.Array
    shields: chex.Array
    dtime: chex.Array
    scroll_speed: chex.Array
    rng_key:  chex.PRNGKey
    step_counter: chex.Array

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
    playing_field_small_bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/mountains/playing_field_small_bg.npy"))
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
    # Entity missile
    entity_missile = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/missiles/enemy_missile.npy"))

    # Death sprites
    upper_death_sprites_temp = []
    for i in range(1, 13):
        temp = aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/lasergates/enemies/enemy_death/top/{i}.npy"))
        upper_death_sprites_temp.append(temp)
        upper_death_sprites_temp[i - 1] = jnp.expand_dims(upper_death_sprites_temp[i - 1], axis=0)

    upper_death_sprites = jnp.concatenate(upper_death_sprites_temp, axis=0)

    lower_death_sprites_temp = []
    for i in range(1, 13):
        temp = aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/lasergates/enemies/enemy_death/bottom/{i}.npy"))
        lower_death_sprites_temp.append(temp)
        lower_death_sprites_temp[i - 1] = jnp.expand_dims(lower_death_sprites_temp[i - 1], axis=0)

    lower_death_sprites = jnp.concatenate(lower_death_sprites_temp, axis=0)

    # Radar mortar
    radar_mortar_frame_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/radar_mortar/1.npy"))
    radar_mortar_frame_middle = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/radar_mortar/2.npy"))
    radar_mortar_frame_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/radar_mortar/3.npy"))

    rms = aj.pad_to_match([radar_mortar_frame_left, radar_mortar_frame_middle, radar_mortar_frame_right])
    radar_mortar_sprites = jnp.concatenate([
        jnp.repeat(rms[0][None], RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
        jnp.repeat(rms[1][None], RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
        jnp.repeat(rms[2][None], RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
        jnp.repeat(rms[1][None], RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
    ]) # Radar mortar rotation animation

    # Byte bat
    byte_bat_frame_up = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/byte_bat/1.npy"))
    byte_bat_frame_mid = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/byte_bat/2.npy"))
    byte_bat_frame_down = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/byte_bat/3.npy"))

    bbs = aj.pad_to_match([byte_bat_frame_up, byte_bat_frame_mid, byte_bat_frame_down, byte_bat_frame_mid])
    byte_bat_sprites = jnp.concatenate([
        jnp.repeat(bbs[0][None], BYTE_BAT_ANIMATION_SPEED, axis=0),
        jnp.repeat(bbs[1][None], BYTE_BAT_ANIMATION_SPEED, axis=0),
        jnp.repeat(bbs[2][None], BYTE_BAT_ANIMATION_SPEED, axis=0),
        jnp.repeat(bbs[1][None], BYTE_BAT_ANIMATION_SPEED, axis=0),
    ]) # Byte bat flap animation

    # Rock muncher
    rock_muncher_frame_small = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/rock_muncher/1.npy"))
    rock_muncher_frame_mid = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/rock_muncher/2.npy"))
    rock_muncher_frame_big = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/rock_muncher/3.npy"))

    rmus = aj.pad_to_match([rock_muncher_frame_small, rock_muncher_frame_mid, rock_muncher_frame_big, rock_muncher_frame_mid])
    rock_muncher_sprites = jnp.concatenate([
        jnp.repeat(rmus[0][None], ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
        jnp.repeat(rmus[1][None], ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
        jnp.repeat(rmus[2][None], ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
        jnp.repeat(rmus[1][None], ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
    ]) # Rock muncher animation

    # Homing missile
    homing_missile_sprite = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/homing_missile/homing_missile.npy"))

    # Forcefield
    forcefield_sprite = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/forcefield/forcefield.npy"))

    # Densepack
    densepack_frame_0 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/5.npy"))
    densepack_frame_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/4.npy"))
    densepack_frame_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/3.npy"))
    densepack_frame_3 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/2.npy"))
    densepack_frame_4 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/1.npy"))

    densepack_sprites = jnp.array([
        densepack_frame_0, densepack_frame_1, densepack_frame_2, densepack_frame_3, densepack_frame_4
    ])

    # Detonator
    detonator_sprite = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/detonator/detonator.npy"))
    detonator_6507 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/detonator/6507.npy"))

    return (
        # Player sprites
        player,
        player_missile,

        # Entity sprites
        entity_missile,
        upper_death_sprites,
        lower_death_sprites,
        radar_mortar_sprites,
        byte_bat_sprites,
        rock_muncher_sprites,
        homing_missile_sprite,
        forcefield_sprite,
        densepack_sprites,
        detonator_sprite,
        detonator_6507,

        # Background sprites
        upper_brown_bg,
        lower_brown_bg,
        playing_field_bg,
        playing_field_small_bg,
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
    SPRITE_ENTITY_MISSILE,
    SPRITE_UPPER_DEATH_SPRITES,
    SPRITE_LOWER_DEATH_SPRITES,
    SPRITE_RADAR_MORTAR,
    SPRITE_BYTE_BAT,
    SPRITE_ROCK_MUNCHER,
    SPRITE_HOMING_MISSILE,
    SPRITE_FORCEFIELD,
    SPRITE_DENSEPACK,
    SPRITE_DETONATOR,
    SPRITE_6507,

    # Background sprites
    SPRITE_UPPER_BROWN_BG,
    SPRITE_LOWER_BROWN_BG,
    SPRITE_PLAYING_FIELD_BG,
    SPRITE_PLAYING_FIELD_SMALL_BG,
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
            is_alive=jnp.bool(True),
            x=jnp.array(RADAR_MORTAR_SPAWN_X).astype(entities.radar_mortar_state.x.dtype),
            y=jnp.where(top_or_bot, RADAR_MORTAR_SPAWN_BOTTOM_Y, RADAR_MORTAR_SPAWN_UPPER_Y),
            missile_x = jnp.array(0),
            missile_y = jnp.array(0),
            missile_direction = jnp.array((0, 0)),
            shoot_again_timer = jnp.array(0),
        )
        return entities._replace(radar_mortar_state=new_radar_mortar_state)

    def initialize_byte_bat(entities, state):

        initial_direction_is_up = jnp.bool(BYTE_BAT_SPAWN_Y < BYTE_BAT_UPPER_BORDER_Y)
        new_byte_bat_state = ByteBatState(
            is_in_current_event=jnp.bool(True),
            is_alive=jnp.bool(True),
            x=jnp.array(BYTE_BAT_SPAWN_X).astype(entities.byte_bat_state.x.dtype),
            y=jnp.array(BYTE_BAT_SPAWN_Y).astype(entities.byte_bat_state.y.dtype),
            direction_is_up=initial_direction_is_up,
            direction_is_left=jnp.bool(True)
        )
        return entities._replace(byte_bat_state=new_byte_bat_state)

    def initialize_rock_muncher(entities, state):

        initial_direction_is_up = jnp.bool(ROCK_MUNCHER_SPAWN_Y < ROCK_MUNCHER_UPPER_BORDER_Y)
        new_rock_muncher_state = RockMuncherState(
            is_in_current_event=jnp.bool(True),
            is_alive=jnp.bool(True),
            x=jnp.array(ROCK_MUNCHER_SPAWN_X).astype(entities.byte_bat_state.x.dtype),
            y=jnp.array(ROCK_MUNCHER_SPAWN_Y).astype(entities.byte_bat_state.y.dtype),
            direction_is_up=initial_direction_is_up,
            direction_is_left=jnp.bool(True),
            missile_x=jnp.array(0),
            missile_y=jnp.array(0),
        )
        return entities._replace(rock_muncher_state=new_rock_muncher_state)

    def initialize_homing_missile(entities, state):

        initial_y_position = jax.random.randint(key_intern, (), HOMING_MISSILE_Y_BOUNDS[0], HOMING_MISSILE_Y_BOUNDS[1])
        new_homing_missile_state = HomingMissileState(
            is_in_current_event=jnp.bool(True),
            is_alive=jnp.bool(True),
            x=jnp.array(WIDTH).astype(entities.homing_missile_state.x.dtype),
            y=initial_y_position,
            is_tracking_player=jnp.bool(False),
        )
        return entities._replace(homing_missile_state=new_homing_missile_state)

    def initialize_forcefield(entities, state):

        key_num_of_ff, key_type_of_ff, key_is_wide = jax.random.split(key_intern, 3)
        number_of_forcefields = jax.random.randint(key_num_of_ff, (), minval=1, maxval=5) # Spawn 1 to 4 forcefields at a time.

        type_of_forcefield = jax.random.randint(key_type_of_ff, (), minval=0, maxval=3)
        init_is_flexing = type_of_forcefield == 0
        init_is_fixed = type_of_forcefield == 1

        init_is_wide = jax.random.bernoulli(key_is_wide, p=FORCEFIELD_IS_WIDE_PROBABILITY)

        number_of_forcefields = jnp.where(init_is_wide, 1, number_of_forcefields)

        new_forcefield_state = entities.forcefield_state._replace(
            is_in_current_event=jnp.bool(True),
            is_alive=jnp.bool(True),
            x0=jnp.array(WIDTH, dtype=jnp.float32),
            y0=jnp.where(init_is_flexing, -10, jnp.where(init_is_fixed, -20, -17)).astype(jnp.float32),
            x1=jnp.array(WIDTH, dtype=jnp.float32),
            y1=jnp.where(init_is_flexing, 65, jnp.where(init_is_fixed, 65, 56)).astype(jnp.float32),
            x2=jnp.array(WIDTH, dtype=jnp.float32),
            y2=jnp.where(init_is_flexing, -10, jnp.where(init_is_fixed, -20, -17)).astype(jnp.float32),
            x3=jnp.array(WIDTH, dtype=jnp.float32),
            y3=jnp.where(init_is_flexing, 65, jnp.where(init_is_fixed, 65, 56)).astype(jnp.float32),
            x4=jnp.array(WIDTH, dtype=jnp.float32),
            y4=jnp.where(init_is_flexing, -10, jnp.where(init_is_fixed, -20, -17)).astype(jnp.float32),
            x5=jnp.array(WIDTH, dtype=jnp.float32),
            y5=jnp.where(init_is_flexing, 65, jnp.where(init_is_fixed, 65, 56)).astype(jnp.float32),
            num_of_forcefields=jnp.array(number_of_forcefields),
            is_wide=init_is_wide,
            is_flexing=init_is_flexing,
            is_fixed=init_is_fixed,
            flash_on=jnp.array(True),
            flex_upper_direction_is_up=jnp.array(True),
            fixed_upper_direction_is_up=jnp.array(True),
        )
        return entities._replace(forcefield_state=new_forcefield_state)

    def initialize_densepack(entities, state):

        initial_is_wide = jax.random.bernoulli(key_intern, p=DENSEPACK_IS_WIDE_PROBABILITY)

        new_densepack_state = entities.dense_pack_state._replace(
            is_in_current_event=jnp.bool(True),
            is_alive=jnp.bool(True),
            x=jnp.array(WIDTH).astype(jnp.float32),
            upmost_y=jnp.array(19).astype(jnp.float32),
            is_wide=initial_is_wide,
            number_of_parts=jnp.array(DENSEPACK_NUMBER_OF_PARTS).astype(jnp.int32),
            broken_states=jnp.full(DENSEPACK_NUMBER_OF_PARTS, 4, jnp.int32),
        )
        return entities._replace(dense_pack_state=new_densepack_state)

    def initialize_detonator(entities, state):
        new_detonator_state = entities.detonator_state._replace(
            is_in_current_event=jnp.bool(True),
            is_alive=jnp.bool(True),
            x=jnp.array(WIDTH).astype(jnp.float32),
            y=jnp.array(19).astype(jnp.float32),
            collision_is_pin=jnp.bool(False),
        )
        return entities._replace(detonator_state=new_detonator_state)

    def initialize_energy_pod(entities, state):
        new_state = entities.energy_pod_state._replace(is_in_current_event=jnp.bool(True))
        return entities._replace(energy_pod_state=new_state)

    init_fns = [
        initialize_radar_mortar,
        initialize_byte_bat,
        initialize_rock_muncher,
        initialize_homing_missile,
        initialize_forcefield,
        initialize_densepack,
        initialize_detonator,
        initialize_energy_pod,
    ] # All initialize functions of all entity types

    def initialize_random_entity(_):
        picked_index = jax.random.randint(key_pick_type, shape=(), minval=6, maxval=7) # TODO: Change maxval to len(init_fns) when all init functions are implemented
        # If you want only one specific entity to spawn, change minval, maxval to:
        # Radar Mortar:     minval=0, maxval=1
        # Byte Bat:         minval=1, maxval=2
        # Rock Muncher:     minval=2, maxval=3
        # Homing Missile:   minval=3, maxval=4
        # Forcefields:      minval=4, maxval=5
        # Densepack:        minval=5, maxval=6
        # Detonator:        minval=6, maxval=7
        # Energy pod:       minval=7, maxval=8

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
    update_tick = jnp.logical_and(update_tick, DEBUG_ACTIVATE_MOUNTAINS_SCROLL)

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
def all_entities_step(game_state: LaserGatesState) -> EntitiesState:
    """
    steps the entity (actually entities, but we only have one entity per event) that is currently in game (if is_in_current_event of said entity is True).
    """

    def radar_mortar_step(state: LaserGatesState) -> tuple[RadarMortarState, CollisionPropertiesState]:
        rm = state.entities.radar_mortar_state
        new_x = jnp.where(rm.is_alive, rm.x - state.scroll_speed, rm.x)

        # Compute spawn position & 45 degree - direction
        is_at_bottom = rm.y == RADAR_MORTAR_SPAWN_BOTTOM_Y
        offset_y = jnp.where(is_at_bottom, 0, RADAR_MORTAR_SIZE[1])
        spawn_x = rm.x
        spawn_y = rm.y + offset_y

        is_left = state.player_x < (spawn_x - RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD)
        is_right = state.player_x > (spawn_x + RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD)
        is_above = state.player_y < (spawn_y - RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD)
        is_below = state.player_y > (spawn_y + RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD)
        dx = jnp.where(is_left, -2, jnp.where(is_right, 2, 0))
        dy = jnp.where(is_above, -1, jnp.where(is_below, 1, 0))
        dir_to_player = jnp.array([dx, dy])

        # Out-of-bounds check for final kill
        out_of_bounds = jnp.logical_or(
            jnp.logical_or(rm.missile_x < 0, rm.missile_x > WIDTH),
            jnp.logical_or(rm.missile_y < PLAYER_BOUNDS[1][0],
                           rm.missile_y > PLAYER_BOUNDS[1][1])
        )

        # Fresh spawn condition
        missile_dead = jnp.all(rm.missile_direction == 0)
        spawn_trigger = jnp.logical_and(rm.is_alive, (state.step_counter % RADAR_MORTAR_MISSILE_SPAWN_EVERY) == 0)

        # small_out_of_bounds: moved beyond 5px from spawn?
        small_oob = jnp.logical_or(
            jnp.abs(rm.missile_x - spawn_x) > RADAR_MORTAR_MISSILE_SMALL_OUT_OF_BOUNDS_THRESHOLD,
            jnp.abs(rm.missile_y - spawn_y) > RADAR_MORTAR_MISSILE_SMALL_OUT_OF_BOUNDS_THRESHOLD
        )

        # Check if the direction is up or down (0, 1)
        slow_direction = jnp.logical_or(
            jnp.all(dir_to_player == jnp.array([0, 1])),
            jnp.all(dir_to_player == jnp.array([0, -1]))
        )

        # Only start repeat fire when a fresh spawn occurred and is alive
        fresh_spawn = jnp.logical_and(jnp.logical_and(missile_dead, spawn_trigger), rm.is_alive)

        # Decide new timer value
        should_decrement = jnp.logical_and(rm.shoot_again_timer > 0, small_oob)

        # Apply conditional timer set:
        # - RADAR_MORTAR_MISSILE_SHOOT_NUMBER if fresh spawn and direction valid
        # - 1 if fresh spawn and direction is a slow direction
        new_timer = jnp.where(fresh_spawn,
                              jnp.where(slow_direction, 1, RADAR_MORTAR_MISSILE_SHOOT_NUMBER),
                              jnp.where(should_decrement, rm.shoot_again_timer - 1, rm.shoot_again_timer))

        # in_spawn_phase: teleport back if fresh_spawn or (timer > 0 and small_oob)
        in_spawn_phase = jnp.logical_or(
            fresh_spawn,
            jnp.logical_and(new_timer > 0, small_oob)
        )

        # Base position & direction: either spawn or keep old
        base_x = jnp.where(in_spawn_phase, spawn_x, rm.missile_x)
        base_y = jnp.where(in_spawn_phase, spawn_y, rm.missile_y)
        # Keep original direction until timer runs out
        base_dir = jnp.where(fresh_spawn, dir_to_player, rm.missile_direction)

        # Kill only if timer == 0 and fully out_of_bounds
        kill = jnp.logical_and(new_timer == 0, out_of_bounds)
        missile_x = jnp.where(kill, 0, base_x)
        missile_y = jnp.where(kill, 0, base_y)
        missile_dir = jnp.where(kill, jnp.array([0, 0], dtype=jnp.int32), base_dir)

        # Move if alive and not in spawn phase
        alive = jnp.any(missile_dir != 0)
        speed = jnp.where(slow_direction, 1, RADAR_MORTAR_MISSILE_SPEED)
        move_cond = jnp.logical_and(alive, jnp.logical_not(in_spawn_phase))
        missile_x = jnp.where(move_cond,
                              missile_x + missile_dir[0] * speed,
                              missile_x)
        missile_y = jnp.where(move_cond,
                              missile_y + missile_dir[1] * speed,
                              missile_y)

        # ----- Collision detection -----

        # If collision with player occurred. Only valid if death timer is still in alive state
        collision_with_player = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_x, state.player_y), PLAYER_SIZE, (new_x, rm.y), RADAR_MORTAR_SIZE),
            jnp.bool(False)
        )

        # If collision with player missile occurred. Only valid if death timer is still in alive state
        collision_with_player_missile = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_missile.x, state.player_missile.y), PLAYER_MISSILE_SIZE, (new_x, rm.y), RADAR_MORTAR_SIZE),
            jnp.bool(False))

        # If collision with entity missile occurred. Only valid if death timer is still in alive state
        rm_missile_collision_with_player = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_x, state.player_y), PLAYER_SIZE, (rm.missile_x, rm.missile_y), ENTITY_MISSILE_SIZE),
            jnp.bool(False)
        )

        # Is still alive if was already alive and no collision occurred
        new_is_alive = jnp.logical_and(rm.is_alive, jnp.logical_and(jnp.logical_not(collision_with_player_missile), jnp.logical_not(collision_with_player)))

        # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
        new_death_timer = jnp.where(new_is_alive, ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
        new_death_timer = jnp.where(jnp.logical_not(new_is_alive), jnp.maximum(new_death_timer - 1, 0), new_death_timer)
        new_death_timer = jnp.where(collision_with_player, -1, new_death_timer)

        # Update is_in_current_event for player missile collision
        new_is_in_current_event = jnp.where(collision_with_player_missile, rm.is_alive, rm.is_in_current_event)
        new_is_in_current_event = jnp.where(new_death_timer == 0, jnp.bool(False), new_is_in_current_event)

        # Update is_in_current_event for player collision
        new_is_in_current_event = jnp.where(collision_with_player, jnp.bool(True), new_is_in_current_event)

        collision_with_player = jnp.logical_or(collision_with_player, rm_missile_collision_with_player)

        return rm._replace(
            is_in_current_event=jnp.logical_and(new_is_in_current_event, rm.x > 0),
            is_alive=new_is_alive,
            x=new_x,
            missile_x=(missile_x - state.scroll_speed).astype(rm.missile_x.dtype),
            missile_y=missile_y,
            missile_direction=missile_dir,
            shoot_again_timer=new_timer
        ), state.entities.collision_properties_state._replace(
            collision_with_player=collision_with_player,
            collision_with_player_missile=collision_with_player_missile,
            is_big_collision=jnp.logical_not(rm_missile_collision_with_player),
            is_energy_pod=jnp.bool(False),
            is_detonator=jnp.bool(False),
            is_ff_or_dp=jnp.bool(False),
            score_to_add=jnp.array(115),
            death_timer=new_death_timer,
        )

    def byte_bat_step(state: LaserGatesState) -> tuple[ByteBatState, CollisionPropertiesState]:
        bb = state.entities.byte_bat_state

        # If one of the y borders are hit, only register if alive
        y_border_hit = jnp.logical_and(bb.is_alive,
                                       jnp.logical_or(bb.y <= BYTE_BAT_UPPER_BORDER_Y, bb.y >= BYTE_BAT_BOTTOM_BORDER_Y)
                                       )

        # If player is left of the byte bat, update only if hitting border
        new_direction_is_left = jnp.where(
            y_border_hit,
            state.player_x + PLAYER_SIZE[0] < bb.x,
            bb.direction_is_left
        )
        # Invert y direction if one of the two y borders is hit
        new_direction_is_up = jnp.where(
            y_border_hit,
            jnp.logical_not(bb.direction_is_up),
            bb.direction_is_up
        )

        # Update positions
        moved_x = jnp.where(new_direction_is_left, bb.x - BYTE_BAT_X_SPEED, bb.x + BYTE_BAT_X_SPEED)
        moved_x = jnp.where(state.player_x == PLAYER_BOUNDS[0][1], bb.x, moved_x)
        moved_y = jnp.where(new_direction_is_up, bb.y - BYTE_BAT_Y_SPEED, bb.y + BYTE_BAT_Y_SPEED)

        # Only apply position when alive
        new_x = jnp.where(bb.is_alive, moved_x, bb.x)
        new_y = jnp.where(bb.is_alive, moved_y, bb.y)

        # ----- Collision detection -----

        # If collision with player occurred. Only valid if death timer is still in alive state
        collision_with_player = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_x, state.player_y), PLAYER_SIZE, (new_x, new_y), BYTE_BAT_SIZE),
            jnp.bool(False)
        )

        # If collision with player missile occurred. Only valid if death timer is still in alive state
        collision_with_player_missile = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_missile.x, state.player_missile.y), PLAYER_MISSILE_SIZE, (new_x, new_y), BYTE_BAT_SIZE),
            jnp.bool(False))

        # Is still alive if was already alive and no collision occurred
        new_is_alive = jnp.logical_and(bb.is_alive, jnp.logical_and(jnp.logical_not(collision_with_player_missile), jnp.logical_not(collision_with_player)))

        # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
        new_death_timer = jnp.where(new_is_alive, ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
        new_death_timer = jnp.where(jnp.logical_not(new_is_alive), jnp.maximum(new_death_timer - 1, 0), new_death_timer)
        new_death_timer = jnp.where(collision_with_player, -1, new_death_timer)

        # Update is_in_current_event for player missile collision
        new_is_in_current_event = jnp.where(collision_with_player_missile, bb.is_alive, bb.is_in_current_event)
        new_is_in_current_event = jnp.where(new_death_timer == 0, jnp.bool(False), new_is_in_current_event)

        # Update is_in_current_event for player collision
        new_is_in_current_event = jnp.where(collision_with_player, jnp.bool(True), new_is_in_current_event)

        return bb._replace(
            is_in_current_event=new_is_in_current_event,
            is_alive=new_is_alive,
            x=new_x,
            y=new_y,
            direction_is_up=new_direction_is_up,
            direction_is_left=new_direction_is_left,
        ), state.entities.collision_properties_state._replace(
            collision_with_player=collision_with_player,
            collision_with_player_missile=collision_with_player_missile,
            is_big_collision=jnp.bool(True),
            is_energy_pod=jnp.bool(False),
            is_detonator=jnp.bool(False),
            is_ff_or_dp=jnp.bool(False),
            score_to_add=jnp.array(330),
            death_timer=new_death_timer,
        )

    def rock_muncher_step(state: LaserGatesState) -> tuple[RockMuncherState, CollisionPropertiesState]:
        rm = state.entities.rock_muncher_state

        # If one of the y borders are hit
        y_border_hit = jnp.logical_or(rm.y <= ROCK_MUNCHER_UPPER_BORDER_Y, rm.y >= ROCK_MUNCHER_BOTTOM_BORDER_Y)
        # If player is left of the byte bat, update only if hitting border
        new_direction_is_left = jnp.where(y_border_hit, state.player_x + PLAYER_SIZE[0] < rm.x, rm.direction_is_left)
        # Invert y direction if one of the two y borders is hit
        new_direction_is_up = jnp.where(y_border_hit, jnp.logical_not(rm.direction_is_up), rm.direction_is_up)

        # Update positions
        new_x = jnp.where(new_direction_is_left, rm.x - BYTE_BAT_X_SPEED, rm.x + BYTE_BAT_X_SPEED) # Move left or right
        new_x = jnp.where(jnp.logical_or(state.player_x == PLAYER_BOUNDS[0][1], jnp.logical_not(rm.is_alive)), rm.x, new_x) # Do not move in x direction if player speeds up scroll speed (is at right player bound) or is not alive (death sprite active)
        new_y = jnp.where(new_direction_is_up, rm.y - BYTE_BAT_Y_SPEED, rm.y + BYTE_BAT_Y_SPEED) # Move up or down
        new_y = jnp.where(jnp.logical_not(rm.is_alive), rm.y, new_y) # Do not move if not alive

        # Missile
        spawn_trigger = jnp.logical_and(rm.is_alive, (state.step_counter % ROCK_MUNCHER_MISSILE_SPAWN_EVERY) == 0)

        # Spawn
        new_missile_x = jnp.where(jnp.logical_and(rm.is_alive, spawn_trigger), rm.x, rm.missile_x).astype(rm.missile_x.dtype)
        new_missile_y = jnp.where(jnp.logical_and(rm.is_alive, spawn_trigger), rm.y + 6, rm.missile_y).astype(rm.missile_y.dtype)

        # Move
        new_missile_x = new_missile_x - ROCK_MUNCHER_MISSILE_SPEED

        # Kill
        kill = jnp.logical_or(new_missile_x < 0 - ENTITY_MISSILE_SIZE[0], jnp.bool(False))
        new_missile_x = jnp.where(kill, 0, new_missile_x)
        new_missile_y = jnp.where(kill, 0, new_missile_y)

        # ----- Collision detection -----

        # If collision with player occurred. Only valid if death timer is still in alive state
        collision_with_player = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_x, state.player_y), PLAYER_SIZE, (new_x, new_y), ROCK_MUNCHER_SIZE),
            jnp.bool(False)
        )

        # If collision with player missile occurred. Only valid if death timer is still in alive state
        collision_with_player_missile = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_missile.x, state.player_missile.y), PLAYER_MISSILE_SIZE, (new_x, new_y), ROCK_MUNCHER_SIZE),
            jnp.bool(False))

        # If collision with entity missile occurred. Only valid if death timer is still in alive state
        rm_missile_collision_with_player = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_x, state.player_y), PLAYER_SIZE, (rm.missile_x, rm.missile_y), ENTITY_MISSILE_SIZE),
            jnp.bool(False) #TODO: Currently, the missile is still updated after death of rock muncher. You can not be hit by it.  Find out if you can be hit by the missile in the real game after the rock muncher is already dead, or if it is even there.
        )

        # Is still alive if was already alive and no collision occurred
        new_is_alive = jnp.logical_and(rm.is_alive, jnp.logical_and(jnp.logical_not(collision_with_player_missile), jnp.logical_not(collision_with_player)))

        # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
        new_death_timer = jnp.where(new_is_alive, ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
        new_death_timer = jnp.where(jnp.logical_not(new_is_alive), jnp.maximum(new_death_timer - 1, 0), new_death_timer)
        new_death_timer = jnp.where(collision_with_player, -1, new_death_timer)

        # Update is_in_current_event for player missile collision
        new_is_in_current_event = jnp.where(collision_with_player_missile, rm.is_alive, rm.is_in_current_event)
        new_is_in_current_event = jnp.where(new_death_timer == 0, jnp.bool(False), new_is_in_current_event)

        # Update is_in_current_event for player collision
        new_is_in_current_event = jnp.where(collision_with_player, jnp.bool(True), new_is_in_current_event)

        collision_with_player = jnp.logical_or(collision_with_player, rm_missile_collision_with_player)

        return rm._replace(
            is_in_current_event=new_is_in_current_event,
            is_alive=new_is_alive,
            x=new_x,
            y=new_y,
            direction_is_up=new_direction_is_up,
            direction_is_left=new_direction_is_left,
            missile_x=new_missile_x.astype(rm.missile_x.dtype),
            missile_y=new_missile_y.astype(rm.missile_y.dtype),
        ), state.entities.collision_properties_state._replace(
            collision_with_player=collision_with_player,
            collision_with_player_missile=collision_with_player_missile,
            is_big_collision=jnp.logical_not(rm_missile_collision_with_player),
            is_energy_pod=jnp.bool(False),
            is_detonator=jnp.bool(False),
            is_ff_or_dp=jnp.bool(False),
            score_to_add=jnp.array(325),
            death_timer=new_death_timer,
        )

    def homing_missile_step(state: LaserGatesState) -> tuple[HomingMissileState, CollisionPropertiesState]:
        hm = state.entities.homing_missile_state

        # Track player if in range or already tracking
        player_is_in_y_range = jnp.abs(state.player_y - hm.y) < HOMING_MISSILE_PLAYER_TRACKING_RANGE
        new_is_tracking_player = jnp.logical_or(hm.is_tracking_player, player_is_in_y_range)

        player_is_below_missile = state.player_y - HOMING_MISSILE_Y_PLAYER_OFFSET > hm.y

        # Update position
        new_x = jnp.where(hm.is_alive, hm.x - HOMING_MISSILE_X_SPEED, hm.x)
        new_y = jnp.where(jnp.logical_and(hm.is_alive, jnp.logical_and(new_is_tracking_player, jnp.logical_not(jnp.abs(state.player_y - HOMING_MISSILE_Y_PLAYER_OFFSET - hm.y) <= HOMING_MISSILE_Y_SPEED))), jnp.where(
            player_is_below_missile,
            hm.y + HOMING_MISSILE_Y_SPEED,
            hm.y - HOMING_MISSILE_Y_SPEED
        ), hm.y)
        # Clip y position to bounds
        new_y = jnp.clip(new_y, HOMING_MISSILE_Y_BOUNDS[0], HOMING_MISSILE_Y_BOUNDS[1])

        # ----- Collision detection -----

        # If collision with player occurred. Only valid if death timer is still in alive state
        collision_with_player = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_x, state.player_y), PLAYER_SIZE, (new_x, new_y), HOMING_MISSILE_SIZE),
            jnp.bool(False)
        )

        # If collision with player missile occurred. Only valid if death timer is still in alive state
        collision_with_player_missile = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_missile.x, state.player_missile.y), PLAYER_MISSILE_SIZE, (new_x, new_y), HOMING_MISSILE_SIZE),
            jnp.bool(False))

        # Is still alive if was already alive and no collision occurred
        new_is_alive = jnp.logical_and(hm.is_alive, jnp.logical_and(jnp.logical_not(collision_with_player_missile), jnp.logical_not(collision_with_player)))

        # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
        new_death_timer = jnp.where(new_is_alive, ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
        new_death_timer = jnp.where(jnp.logical_not(new_is_alive), jnp.maximum(new_death_timer - 1, 0), new_death_timer)
        new_death_timer = jnp.where(collision_with_player, -1, new_death_timer)

        # Update is_in_current_event for player missile collision
        new_is_in_current_event = jnp.where(collision_with_player_missile, hm.is_alive, hm.is_in_current_event)
        new_is_in_current_event = jnp.where(new_death_timer == 0, jnp.bool(False), new_is_in_current_event)

        # Update is_in_current_event for player collision
        new_is_in_current_event = jnp.where(collision_with_player, jnp.bool(True), new_is_in_current_event)

        return hm._replace(
            is_in_current_event=jnp.logical_and(new_is_in_current_event, hm.x > 0),
            is_alive=new_is_alive,
            x=new_x,
            y=new_y,
            is_tracking_player=new_is_tracking_player,
        ), state.entities.collision_properties_state._replace(
            collision_with_player=collision_with_player,
            collision_with_player_missile=collision_with_player_missile,
            is_big_collision=jnp.bool(True),
            is_energy_pod=jnp.bool(False),
            is_detonator=jnp.bool(False),
            is_ff_or_dp=jnp.bool(False),
            score_to_add=jnp.array(525),
            death_timer=new_death_timer,
        )

    def forcefield_step(state: LaserGatesState) -> tuple[ForceFieldState, CollisionPropertiesState]:
        ff = state.entities.forcefield_state

        is_flexing, is_fixed = ff.is_flexing, ff.is_fixed
        is_flashing = jnp.logical_not(jnp.logical_or(is_flexing, is_fixed))
        number_of_forcefields = ff.num_of_forcefields
        new_x0, new_x1, new_x2, new_x3, new_x4, new_x5 = ff.x0, ff.x1, ff.x2, ff.x3, ff.x4, ff.x5
        new_y0, new_y1, new_y2, new_y3, new_y4, new_y5 = ff.y0, ff.y1, ff.y2, ff.y3, ff.y4, ff.y5

        # Flashing --------------
        new_flash_on = jnp.where(jnp.logical_and(state.step_counter % FORCEFIELD_FLASHING_SPEED == 0, is_flashing), jnp.logical_not(ff.flash_on), ff.flash_on)
        is_flashing_and_alive = jnp.logical_and(is_flashing, ff.is_alive)

        new_x0 = jnp.where(is_flashing_and_alive, new_x0 - state.scroll_speed, new_x0) # First forcefield upper
        new_x1 = jnp.where(is_flashing_and_alive, new_x0, new_x1) # First forcefield lower

        new_x2 = jnp.where(jnp.logical_and(is_flashing_and_alive, number_of_forcefields > 1), new_x0 + FORCEFIELD_FLASHING_SPACING, new_x2)
        new_x3 = jnp.where(jnp.logical_and(is_flashing_and_alive, number_of_forcefields > 1), new_x2, new_x3)

        new_x4 = jnp.where(jnp.logical_and(is_flashing_and_alive, number_of_forcefields > 2), new_x0 + 2 * FORCEFIELD_FLASHING_SPACING, new_x4)
        new_x5 = jnp.where(jnp.logical_and(is_flashing_and_alive, number_of_forcefields > 2), new_x4, new_x5)
        # There is no need for setting the y position, since it remains unchanged. We use the default y positions set in initialize_forcefield

        # Flexing --------------
        distance = new_y1 - (new_y0 + FORCEFIELD_SIZE[1])
        new_flex_upper_direction_is_up = jnp.where(distance <= FORCEFIELD_FLEXING_MINIMUM_DISTANCE, jnp.bool(True), jnp.where(distance >= FORCEFIELD_FLEXING_MAXIMUM_DISTANCE, jnp.bool(False), ff.flex_upper_direction_is_up))
        is_flexing_and_alive = jnp.logical_and(is_flexing, ff.is_alive)

        new_x0 = jnp.where(is_flexing_and_alive, new_x0 - state.scroll_speed, new_x0)
        new_y0 = jnp.where(is_flexing_and_alive, jnp.where(new_flex_upper_direction_is_up, new_y0 - FORCEFIELD_FLEXING_SPEED, new_y0 + FORCEFIELD_FLEXING_SPEED), new_y0) # First forcefield upper
        new_x1 = jnp.where(is_flexing_and_alive, new_x0, new_x1)
        new_y1 = jnp.where(is_flexing_and_alive, jnp.where(new_flex_upper_direction_is_up, new_y1 + FORCEFIELD_FLEXING_SPEED, new_y1 - FORCEFIELD_FLEXING_SPEED), new_y1) # First forcefield lower

        new_x2 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 1), new_x0 + FORCEFIELD_FLEXING_SPACING, new_x2)
        new_y2 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 1), jnp.where(new_flex_upper_direction_is_up, new_y2 - FORCEFIELD_FLEXING_SPEED, new_y2 + FORCEFIELD_FLEXING_SPEED), new_y2) # Second forcefield upper
        new_x3 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 1), new_x0 + FORCEFIELD_FLEXING_SPACING, new_x3)
        new_y3 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 1), jnp.where(new_flex_upper_direction_is_up, new_y3 + FORCEFIELD_FLEXING_SPEED, new_y3 - FORCEFIELD_FLEXING_SPEED), new_y3) # Second forcefield lower

        new_x4 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 2), new_x0 + 2 * FORCEFIELD_FLEXING_SPACING, new_x4)
        new_y4 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 2), jnp.where(new_flex_upper_direction_is_up, new_y4 - FORCEFIELD_FLEXING_SPEED, new_y4 + FORCEFIELD_FLEXING_SPEED), new_y4) # Third forcefield upper
        new_x5 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 2), new_x0 + 2 * FORCEFIELD_FLEXING_SPACING, new_x5)
        new_y5 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 2), jnp.where(new_flex_upper_direction_is_up, new_y5 + FORCEFIELD_FLEXING_SPEED, new_y5 - FORCEFIELD_FLEXING_SPEED), new_y5) # Third forcefield lower

        # Fixed --------------
        new_fixed_upper_direction_is_up = jnp.where(new_y0 < FORCEFIELD_FIXED_UPPER_BOUND, jnp.bool(False), jnp.where(new_y0 > FORCEFIELD_FIXED_LOWER_BOUND, jnp.bool(True), ff.fixed_upper_direction_is_up))
        is_fixed_and_alive = jnp.logical_and(is_fixed, ff.is_alive)

        new_x0 = jnp.where(is_fixed_and_alive, new_x0 - state.scroll_speed, new_x0)
        new_y0 = jnp.where(is_fixed_and_alive, jnp.where(new_fixed_upper_direction_is_up, new_y0 - FORCEFIELD_FIXED_SPEED, new_y0 + FORCEFIELD_FIXED_SPEED), new_y0) # First forcefield upper
        new_x1 = jnp.where(is_fixed_and_alive, new_x1 - state.scroll_speed, new_x1)
        new_y1 = jnp.where(is_fixed_and_alive, jnp.where(new_fixed_upper_direction_is_up, new_y1 - FORCEFIELD_FIXED_SPEED, new_y1 + FORCEFIELD_FIXED_SPEED), new_y1) # First forcefield lower

        new_x2 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 1), new_x0 + FORCEFIELD_FIXED_SPACING, new_x2)
        new_y2 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 1), new_y0, new_y2) # Second forcefield upper
        new_x3 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 1), new_x0 + FORCEFIELD_FIXED_SPACING, new_x3)
        new_y3 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 1), new_y1, new_y3) # Second forcefield lower

        new_x4 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 2), new_x0 + 2 * FORCEFIELD_FIXED_SPACING, new_x4)
        new_y4 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 2), new_y0, new_y4) # Third forcefield upper
        new_x5 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 2), new_x0 + 2 * FORCEFIELD_FIXED_SPACING, new_x5)
        new_y5 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 2), new_y1, new_y5) # Third forcefield lower

        # Find rightmost x
        all_x_values = jnp.array([new_x0, new_x1, new_x2, new_x3, new_x4, new_x5])
        rightmost_x = jnp.max(jnp.where(all_x_values != WIDTH, all_x_values, -jnp.inf)) # Ignore x coordinates that are at the spawn/dead point

        # ----- Collision detection -----

        allow_check_collision_flashing = jnp.logical_or(jnp.logical_not(is_flashing), jnp.logical_and(is_flashing, new_flash_on))

        x_positions = jnp.array([new_x0, new_x1, new_x2, new_x3, new_x4, new_x5])
        y_positions = jnp.array([new_y0, new_y1, new_y2, new_y3, new_y4, new_y5])
        no_offsets = jnp.array([(0, 0)])
        normal_size = jnp.array([FORCEFIELD_SIZE])
        wide_size = jnp.array([FORCEFIELD_WIDE_SIZE])
        size = jnp.where(ff.is_wide, wide_size, normal_size)

        # If collision with player occurred. Only valid if death timer is still in alive state
        collision_with_player = jnp.where(
            jnp.logical_and(state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER, allow_check_collision_flashing),
            jnp.any(any_collision_for_group((state.player_x, state.player_y), PLAYER_SIZE, x_positions, y_positions, no_offsets, size)),
            jnp.bool(False)
        )

        collision_with_player_missile = jnp.where(
            jnp.logical_and(state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER, allow_check_collision_flashing),
            jnp.any(any_collision_for_group((state.player_missile.x, state.player_missile.y), PLAYER_MISSILE_SIZE, x_positions, y_positions, no_offsets, size)),
            jnp.bool(False)
        )

        # Is still alive if was already alive and no collision occurred
        new_is_alive = jnp.logical_and(ff.is_alive, jnp.logical_not(collision_with_player))

        # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
        new_death_timer = jnp.where(new_is_alive, ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
        new_death_timer = jnp.where(jnp.logical_not(new_is_alive), jnp.maximum(new_death_timer - 1, 0), new_death_timer)
        new_death_timer = jnp.where(collision_with_player, -1, new_death_timer)

        # Update is_in_current_event for player missile collision
        new_is_in_current_event = jnp.where(collision_with_player_missile, ff.is_alive, ff.is_in_current_event)
        new_is_in_current_event = jnp.where(new_death_timer == 0, jnp.bool(False), new_is_in_current_event)

        # Update is_in_current_event for player collision
        new_is_in_current_event = jnp.where(collision_with_player, jnp.bool(True), new_is_in_current_event)

        return ff._replace(
            is_in_current_event=jnp.logical_and(new_is_in_current_event, rightmost_x > 0),
            is_alive=new_is_alive,
            x0=new_x0.astype(ff.x0.dtype),
            y0=new_y0.astype(ff.y0.dtype),
            x1=new_x1.astype(ff.x1.dtype),
            y1=new_y1.astype(ff.y1.dtype),
            x2=new_x2.astype(ff.x2.dtype),
            y2=new_y2.astype(ff.y2.dtype),
            x3=new_x3.astype(ff.x3.dtype),
            y3=new_y3.astype(ff.y3.dtype),
            x4=new_x4.astype(ff.x4.dtype),
            y4=new_y4.astype(ff.y4.dtype),
            x5=new_x5.astype(ff.x5.dtype),
            y5=new_y5.astype(ff.y5.dtype),
            flash_on=new_flash_on,
            flex_upper_direction_is_up=new_flex_upper_direction_is_up,
            fixed_upper_direction_is_up=new_fixed_upper_direction_is_up,
        ), state.entities.collision_properties_state._replace(
            collision_with_player=collision_with_player,
            collision_with_player_missile=collision_with_player_missile,
            is_big_collision=jnp.bool(True),
            is_energy_pod=jnp.bool(False),
            is_detonator=jnp.bool(False),
            is_ff_or_dp=jnp.bool(True),
            score_to_add=jnp.array(525),
            death_timer=new_death_timer,
        )

    @jax.jit
    def densepack_step(state: LaserGatesState) -> tuple[DensepackState, CollisionPropertiesState]:
        dp = state.entities.dense_pack_state

        # base X coord for all segments (with scrolling)
        base_x = dp.x - state.scroll_speed
        # starting Y + vertical spacing
        y = dp.upmost_y
        height = DENSEPACK_NORMAL_PART_SIZE[1]

        # world positions array (shape (n_parts,))
        group_xs = jnp.full((DENSEPACK_NUMBER_OF_PARTS,), base_x, dtype=jnp.float32)
        group_ys = y + jnp.arange(DENSEPACK_NUMBER_OF_PARTS, dtype=jnp.float32) * height

        # offsets lookup as before
        offset_lookup_normal = jnp.array([
            (WIDTH, 0), (6, 0), (4, 0), (2, 0), (0, 0)
        ], dtype=jnp.float32)
        offset_lookup_wide = jnp.array([
            (WIDTH, 0), (12, 0), (8, 0), (4, 0), (0, 0)
        ], dtype=jnp.float32)
        # size lookup, one size per broken_state
        size_lookup_normal = jnp.array([
            (0, 0),  # 0 → fully gone
            (2, 4),  # 1 → small
            (4, 4),  # 2
            (6, 4),  # 3
            (8, 4),  # 4 → intact
        ], dtype=jnp.float32)
        size_lookup_wide = jnp.array([
            (0, 0),  # 0 → fully gone
            (4, 4),  # 1 → small
            (8, 4),  # 2
            (12, 4),  # 3
            (16, 4),  # 4 → intact
        ], dtype=jnp.float32)

        # pick per‐segment offset and size
        segment_offsets = jnp.where(
            dp.is_wide,
            offset_lookup_wide[dp.broken_states],
            offset_lookup_normal[dp.broken_states]
        )  # shape (n_parts,2)
        segment_sizes = jnp.where(
            dp.is_wide,
            size_lookup_wide[dp.broken_states],
            size_lookup_normal[dp.broken_states]
        )  # shape (n_parts,2)

        # --- collision vs. player ---
        def hit_by_player(gx, gy, offs, sz):
            seg_x, seg_y = gx + offs[0], gy + offs[1]
            return check_collision_single(
                jnp.array((state.player_x, state.player_y), dtype=jnp.float32),
                jnp.array(PLAYER_SIZE, dtype=jnp.float32),
                jnp.array((seg_x, seg_y), dtype=jnp.float32),
                sz
            )

        player_hits_mask = jax.vmap(hit_by_player)(
            group_xs, group_ys, segment_offsets, segment_sizes
        )
        collision_with_player = jnp.any(player_hits_mask)

        # --- collision vs. missile ---
        def hit_by_missile(gx, gy, offs, sz):
            seg_x, seg_y = gx + offs[0], gy + offs[1]
            px = state.player_missile.x.astype(jnp.float32)
            py = state.player_missile.y.astype(jnp.float32)
            return check_collision_single(
                jnp.array((px, py), dtype=jnp.float32),
                jnp.array(PLAYER_MISSILE_SIZE, dtype=jnp.float32),
                jnp.array((seg_x, seg_y), dtype=jnp.float32),
                sz
            )

        missile_hits_mask = jax.vmap(hit_by_missile)(
            group_xs, group_ys, segment_offsets, segment_sizes
        )
        collision_with_player_missile = jnp.any(missile_hits_mask)

        # decrement broken_states only where missile hit
        new_broken_states = jnp.where(missile_hits_mask,
                                      jnp.maximum(dp.broken_states - 1, 0),
                                      dp.broken_states)

        # --- life & death logic unchanged ---
        new_is_alive = jnp.logical_and(dp.is_alive, jnp.logical_not(collision_with_player))
        new_death_timer = jnp.where(new_is_alive, ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
        new_death_timer = jnp.where(jnp.logical_not(new_is_alive), jnp.maximum(new_death_timer - 1, 0), new_death_timer)
        new_death_timer = jnp.where(collision_with_player, -1, new_death_timer)

        new_is_in_current_event = dp.is_in_current_event
        new_is_in_current_event = jnp.where(collision_with_player_missile, dp.is_alive, new_is_in_current_event)
        new_is_in_current_event = jnp.where(new_death_timer == 0, jnp.bool(False), new_is_in_current_event)
        new_is_in_current_event = jnp.where(collision_with_player, jnp.bool(True), new_is_in_current_event)
        new_is_in_current_event = jnp.where(base_x > 0, new_is_in_current_event, jnp.bool(False))

        return dp._replace(
            is_in_current_event=new_is_in_current_event,
            is_alive=new_is_alive,
            x=base_x,
            broken_states=new_broken_states,
        ), state.entities.collision_properties_state._replace(
            collision_with_player=collision_with_player,
            collision_with_player_missile=collision_with_player_missile,
            is_big_collision=jnp.bool(True),
            is_ff_or_dp=jnp.bool(True),
            score_to_add=jnp.array(525), # TODO score
            death_timer=new_death_timer,
        )

    def detonator_step(state: LaserGatesState) -> tuple[DetonatorState, CollisionPropertiesState]:
        dn = state.entities.detonator_state

        base_x = dn.x - state.scroll_speed
        y = dn.y

        # ----- Collision detection -----

        x_positions = jnp.array([base_x, base_x, base_x, base_x])
        y_positions = jnp.array([y + 17, y + 29, y + 41, y + 53])
        no_offsets = jnp.array([(0, 0)])
        sizes = jnp.array([(1, 4)])

        # If collision with player occurred. Only valid if death timer is still in alive state
        collision_with_player = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            jnp.any(any_collision_for_group((state.player_x, state.player_y), PLAYER_SIZE, x_positions, y_positions, no_offsets, sizes)),
            jnp.bool(False)
        )
        collision_player_detonator = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_x, state.player_y), PLAYER_SIZE, (base_x, y), DETONATOR_SIZE),
            jnp.bool(False)
        )
        collision_with_player = jnp.logical_or(collision_with_player, collision_player_detonator)

        collision_player_missile_pin = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            jnp.any(any_collision_for_group((state.player_missile.x, state.player_missile.y), PLAYER_MISSILE_SIZE, x_positions, y_positions, no_offsets, sizes)),
            jnp.bool(False)
        )
        collision_player_missile_detonator = jnp.where(
            state.entities.collision_properties_state.death_timer == ENTITY_DEATH_ANIMATION_TIMER,
            check_collision_single((state.player_missile.x, state.player_missile.y), PLAYER_MISSILE_SIZE, (base_x, y), DETONATOR_SIZE),
            jnp.bool(False)
        )

        # If is collision with pin or not. We need this to kill the missile but not the detonator at non-pin collision
        new_collision_is_pin = jnp.bool(False)
        new_collision_is_pin = jnp.where(collision_player_missile_pin, jnp.bool(True), new_collision_is_pin)

        # Is still alive if was already alive and no collision occurred
        new_is_alive = jnp.logical_and(dn.is_alive, jnp.logical_and(jnp.logical_not(collision_player_missile_pin), jnp.logical_not(collision_with_player)))

        # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
        new_death_timer = jnp.where(new_is_alive, ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
        new_death_timer = jnp.where(jnp.logical_not(new_is_alive), jnp.maximum(new_death_timer - 1, 0), new_death_timer)
        new_death_timer = jnp.where(collision_with_player, -1, new_death_timer)

        # Update is_in_current_event for player missile collision
        new_is_in_current_event = jnp.where(collision_player_missile_pin, dn.is_alive, dn.is_in_current_event)
        new_is_in_current_event = jnp.where(new_death_timer == 0, jnp.bool(False), new_is_in_current_event)

        # Update is_in_current_event for player collision
        new_is_in_current_event = jnp.where(collision_with_player, jnp.bool(True), new_is_in_current_event)

        collision_player_missile_pin = jnp.logical_or(collision_player_missile_pin, collision_player_missile_detonator)

        return state.entities.detonator_state._replace(
            is_in_current_event=jnp.logical_and(new_is_in_current_event, base_x > 0), # Second condition should never happen, since you can only collide or destroy the detonator
            is_alive=new_is_alive,
            x=base_x.astype(jnp.float32),
            collision_is_pin=new_collision_is_pin,
        ), state.entities.collision_properties_state._replace(
            collision_with_player=collision_with_player,
            collision_with_player_missile=collision_player_missile_pin,
            is_big_collision=jnp.bool(True),
            is_detonator=jnp.bool(True),
            is_ff_or_dp=jnp.bool(False),
            score_to_add=jnp.array(6507),
            death_timer=new_death_timer,
        )

    def energy_pod_step(state: LaserGatesState) -> tuple[EnergyPodState, CollisionPropertiesState]:
        return state.entities.energy_pod_state, state.entities.collision_properties_state


    def entity_maybe_step(step_fn, entity_state):
        def run_step(_):
            stepped_entity, updates = step_fn(game_state)
            return stepped_entity, updates

        def no_step(_):
            return entity_state, game_state.entities.collision_properties_state

        return jax.lax.cond(
            entity_state.is_in_current_event,
            run_step,
            no_step,
            operand=None
        )

    s_entities = game_state.entities

    rm_state, rm_coll = entity_maybe_step(radar_mortar_step, s_entities.radar_mortar_state)
    bb_state, bb_coll = entity_maybe_step(byte_bat_step, s_entities.byte_bat_state)
    rmu_state, rmu_coll = entity_maybe_step(rock_muncher_step, s_entities.rock_muncher_state)
    hm_state, hm_coll = entity_maybe_step(homing_missile_step, s_entities.homing_missile_state)
    ff_state, ff_coll = entity_maybe_step(forcefield_step, s_entities.forcefield_state)
    dp_state, dp_coll = entity_maybe_step(densepack_step, s_entities.dense_pack_state)
    dt_state, dt_coll = entity_maybe_step(detonator_step, s_entities.detonator_state)
    ep_state, ep_coll = entity_maybe_step(energy_pod_step, s_entities.energy_pod_state)

    return EntitiesState(
        radar_mortar_state = rm_state,
        byte_bat_state = bb_state,
        rock_muncher_state = rmu_state,
        homing_missile_state = hm_state,
        forcefield_state = ff_state,
        dense_pack_state = dp_state,
        detonator_state = dt_state,
        energy_pod_state = ep_state,  # Return the new step state for every entity. Only the currently active entity is updated. Since we use lax.cond (which is lazy), only the active branch is executed.

        collision_properties_state=jax.lax.cond( # Return the new collision state for the active entity. Since we use lax.cond (which is lazy), only the active branch is executed.
            rm_state.is_in_current_event,
            lambda _: rm_coll,
            lambda _: jax.lax.cond(
                bb_state.is_in_current_event,
                lambda _: bb_coll,
                lambda _: jax.lax.cond(
                    rmu_state.is_in_current_event,
                    lambda _: rmu_coll,
                    lambda _: jax.lax.cond(
                        hm_state.is_in_current_event,
                        lambda _: hm_coll,
                        lambda _: jax.lax.cond(
                            ff_state.is_in_current_event,
                            lambda _: ff_coll,
                            lambda _: jax.lax.cond(
                                dp_state.is_in_current_event,
                                lambda _: dp_coll,
                                lambda _: jax.lax.cond(
                                    dt_state.is_in_current_event,
                                    lambda _: dt_coll,
                                    lambda _: ep_coll,
                                    operand=None
                                ),
                                operand=None
                            ),
                            operand=None
                        ),
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            ),
            operand=None
        )

    )


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
def check_player_and_player_missile_collision_bounds(
        state: LaserGatesState
) -> tuple[chex.Array, chex.Array, chex.Array]:

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

    player_missile_pos = jnp.array((state.player_missile.x, state.player_missile.y), dtype=jnp.int32)
    player_missile_size = jnp.array(PLAYER_MISSILE_SIZE, dtype=jnp.int32)

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
    upper_missile_collisions = any_collision_for_group(
        player_missile_pos, player_missile_size, upper_xs, upper_ys,
        segment_offsets=lower_offsets,
        segment_sizes=lower_sizes
    )
    lower_missile_collisions = any_collision_for_group(
        player_missile_pos, player_missile_size, upper_xs, upper_ys,
        segment_offsets=lower_offsets,
        segment_sizes=lower_sizes
    )

    # Include normal bound player
    upper_player_collision = jnp.logical_or(jnp.any(upper_collisions), state.player_y <= PLAYER_BOUNDS[1][0])
    lower_player_collision = jnp.logical_or(jnp.any(lower_collisions), state.player_y >= PLAYER_BOUNDS[1][1])

    # Include normal bound player missile
    player_missile_collision = jnp.logical_or(jnp.any(upper_missile_collisions), jnp.any(lower_missile_collisions))

    return upper_player_collision, lower_player_collision, player_missile_collision


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

    # TODO: add other functions if needed

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
    def reset(self, key = 42) -> Tuple[LaserGatesObservation, LaserGatesState]:
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

        initial_entities = EntitiesState(
            radar_mortar_state=RadarMortarState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
                missile_x = jnp.array(0),
                missile_y = jnp.array(0),
                missile_direction = jnp.array((0, 0)),
                shoot_again_timer = jnp.array(0),
            ),
            byte_bat_state=ByteBatState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0.0).astype(jnp.float32),
                y=jnp.array(0.0).astype(jnp.float32),
                direction_is_up=jnp.bool(False),
                direction_is_left=jnp.bool(False)
            ),
            rock_muncher_state=RockMuncherState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0.0).astype(jnp.float32),
                y=jnp.array(0.0).astype(jnp.float32),
                direction_is_up=jnp.bool(False),
                direction_is_left=jnp.bool(False),
                missile_x=jnp.array(0),
                missile_y=jnp.array(0),
            ),
            homing_missile_state=HomingMissileState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
                is_tracking_player=jnp.bool(False),
            ),
            forcefield_state=ForceFieldState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x0=jnp.array(0, dtype=jnp.float32),
                y0=jnp.array(0, dtype=jnp.float32),
                x1=jnp.array(0, dtype=jnp.float32),
                y1=jnp.array(0, dtype=jnp.float32),
                x2=jnp.array(0, dtype=jnp.float32),
                y2=jnp.array(0, dtype=jnp.float32),
                x3=jnp.array(0, dtype=jnp.float32),
                y3=jnp.array(0, dtype=jnp.float32),
                x4=jnp.array(0, dtype=jnp.float32),
                y4=jnp.array(0, dtype=jnp.float32),
                x5=jnp.array(0, dtype=jnp.float32),
                y5=jnp.array(0, dtype=jnp.float32),
                num_of_forcefields=jnp.array(0),
                is_wide=jnp.bool(False),
                is_flexing=jnp.bool(False),
                is_fixed=jnp.bool(False),
                flash_on=jnp.bool(False),
                flex_upper_direction_is_up=jnp.bool(False),
                fixed_upper_direction_is_up=jnp.bool(False),
            ),
            dense_pack_state=DensepackState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                upmost_y=jnp.array(0).astype(jnp.float32),
                is_wide=jnp.bool(False),
                number_of_parts=jnp.array(0).astype(jnp.int32),
                broken_states=jnp.full(DENSEPACK_NUMBER_OF_PARTS, 3, jnp.int32),
            ),
            detonator_state=DetonatorState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0).astype(jnp.float32),
                collision_is_pin=jnp.bool(False),
            ),
            energy_pod_state=EnergyPodState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0),
            ),
            collision_properties_state=CollisionPropertiesState(
                collision_with_player=jnp.bool(False),
                collision_with_player_missile=jnp.bool(False),
                is_big_collision=jnp.bool(False),
                is_energy_pod=jnp.bool(False),
                is_detonator=jnp.bool(False),
                is_ff_or_dp=jnp.bool(False),
                score_to_add=jnp.array(0),
                death_timer=jnp.array(ENTITY_DEATH_ANIMATION_TIMER),
            )
        )

        key = jax.random.PRNGKey(time.time_ns() % (2**32)) # Pseudo random number generator seed key, based on current system time.
        new_key0, key0 = jax.random.split(key, 2)

        reset_state = LaserGatesState(
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_facing_direction=jnp.array(1),
            player_missile=initial_player_missile,
            animation_timer=jnp.array(0).astype(jnp.uint8),
            entities=initial_entities,
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

        # -------- Move player --------
        new_player_x, new_player_y, new_player_facing_direction = player_step(state, action)
        player_animation_timer = state.animation_timer
        new_player_animation_timer = jnp.where(player_animation_timer != 0, player_animation_timer - 1, player_animation_timer)

        # -------- Move player missile --------
        new_player_missile_state = player_missile_step(state, action)

        # -------- Move entities --------
        new_entities = all_entities_step(state)
        new_entities = maybe_initialize_random_entity(new_entities, state)

        # -------- Move mountains --------
        new_lower_mountains_state = mountains_step(state.lower_mountains, state)
        new_upper_mountains_state = mountains_step(state.upper_mountains, state)

        # -------- Update scroll speed --------
        new_scroll_speed = jnp.where(state.player_x != PLAYER_BOUNDS[0][1], SCROLL_SPEED, SCROLL_SPEED * SCROLL_MULTIPLIER)

        # -------- Check bound and entity collisions --------
        upper_player_collision, lower_player_collision, player_missile_collision = check_player_and_player_missile_collision_bounds(state)
        collision_with_player = jnp.logical_and(jnp.logical_not(state.entities.collision_properties_state.collision_with_player), new_entities.collision_properties_state.collision_with_player) # Only allow flag to be True once per collision
        any_player_collision = jnp.logical_or(collision_with_player, jnp.logical_or(upper_player_collision, lower_player_collision))

        # -------- Update things that have to be updated at collision --------
        new_player_animation_timer = jnp.where(any_player_collision, 255, new_player_animation_timer)

        new_player_y = jnp.where(upper_player_collision, new_player_y + 4, new_player_y)
        new_player_y = jnp.where(lower_player_collision, new_player_y - 4, new_player_y)

        # -------- Kill missile --------
        kill_missile = jnp.logical_or(player_missile_collision, new_entities.collision_properties_state.collision_with_player_missile)
        new_player_missile_state = PlayerMissileState(
                                             x=jnp.where(kill_missile, jnp.array(0).astype(new_player_missile_state.x.dtype), new_player_missile_state.x),
                                             y=jnp.where(kill_missile, jnp.array(0), new_player_missile_state.y),
                                             direction=jnp.where(kill_missile, jnp.array(0).astype(new_player_missile_state.direction.dtype), new_player_missile_state.direction),
                                             velocity=jnp.where(kill_missile, jnp.array(0).astype(new_player_missile_state.velocity.dtype), new_player_missile_state.velocity)
                                             )


        # -------- Update energy, score, shields and d-time --------
        new_energy = state.energy - 1
        # Dorbid score change if densepack or forcefield is hit with missile
        allow_score_change = jnp.logical_and(new_entities.collision_properties_state.collision_with_player_missile, jnp.logical_not(state.entities.collision_properties_state.is_ff_or_dp))
        # Forbid score change if detonator is not hit with a missile at a pin
        allow_score_change = jnp.where(new_entities.collision_properties_state.is_detonator, new_entities.detonator_state.collision_is_pin, allow_score_change)
        new_score = jnp.where(allow_score_change, state.score + new_entities.collision_properties_state.score_to_add, state.score)
        new_shields = jnp.where(jnp.logical_or(upper_player_collision, lower_player_collision), state.shields - 1, state.shields)
        new_shields = jnp.where(collision_with_player,
                                new_shields - jnp.where(new_entities.collision_properties_state.is_big_collision, 6, 1),
                                new_shields)
        # TODO: Gain 6 shield points for every 10000 points scored

        # -------- New rng key --------
        new_rng_key, new_key = jax.random.split(state.rng_key)

        return_state = state._replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_facing_direction=new_player_facing_direction,
            animation_timer=new_player_animation_timer,
            player_missile=new_player_missile_state,
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

        def recolor_sprite(
                sprite: jnp.ndarray,
                color: jnp.ndarray,  # RGB, up to 4 dimensions
                bounds: tuple[int, int, int, int] = None  # (top, left, bottom, right)
        ) -> jnp.ndarray:
            # Ensure color is the same dtype as sprite
            dtype = sprite.dtype
            color = color.astype(dtype)

            assert sprite.ndim == 3 and sprite.shape[2] in (3, 4), "Sprite must be HxWx3 or HxWx4"

            if color.shape[0] < sprite.shape[2]:
                missing = sprite.shape[2] - color.shape[0]
                pad = jnp.full((missing,), 255, dtype=dtype)
                color = jnp.concatenate([color, pad], axis=0)

            assert color.shape[0] == sprite.shape[2], "Color channels must match sprite channels"

            H, W, _ = sprite.shape

            if bounds is None:
                region = sprite
            else:
                top, left, bottom, right = bounds
                assert 0 <= left < right <= H and 0 <= top < bottom <= W, "Invalid bounds"
                region = sprite[left:right, top:bottom]

            visible_mask = jnp.any(region != 0, axis=-1, keepdims=True)  # (h, w, 1)

            color_broadcasted = jnp.broadcast_to(color, region.shape).astype(dtype)
            recolored_region = jnp.where(visible_mask, color_broadcasted, jnp.zeros_like(color_broadcasted))

            if bounds is None:
                return recolored_region
            else:
                top, left, bottom, right = bounds
                recolored_sprite = sprite.at[left:right, top:bottom].set(recolored_region)
                return recolored_sprite


        def get_death_sprite_index(death_timer: jnp.ndarray, total_duration: int) -> jnp.ndarray:
            sprite_length = total_duration // 12
            clamped_timer = jnp.clip(death_timer, 1, total_duration)
            index = 12 - (clamped_timer - 1) // sprite_length
            return index

        # -------- Render playing field background --------

        # Playing field background, color adjusts if player collision
        pfb_t = jnp.clip(255 * jnp.exp(-PLAYING_FILED_BG_COLOR_FADE_SPEED * (255 - state.animation_timer)), 0, 255)
        pfb_t = pfb_t.astype(jnp.uint8)
        PLAYING_FIELD_COLOR = jnp.array((PLAYING_FIELD_BG_COLLISION_COLOR[0], PLAYING_FIELD_BG_COLLISION_COLOR[1], PLAYING_FIELD_BG_COLLISION_COLOR[2], pfb_t))
        raster = aj.render_at(
            raster,
            0,
            19,
            recolor_sprite(SPRITE_PLAYING_FIELD_BG, PLAYING_FIELD_COLOR),
        )

        # -------- Render Entity Death Sprites --------

        # Death sprites
        death_sprite_index = get_death_sprite_index(state.entities.collision_properties_state.death_timer, ENTITY_DEATH_ANIMATION_TIMER)
        death_sprite_upper_frame = SPRITE_LOWER_DEATH_SPRITES[death_sprite_index]
        death_sprite_lower_frame = SPRITE_UPPER_DEATH_SPRITES[death_sprite_index]

        # Radar mortar
        rm_state = state.entities.radar_mortar_state
        raster = jnp.where(jnp.logical_and(rm_state.is_in_current_event, jnp.logical_and(jnp.logical_not(rm_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                          # Case: in event but dead -> render death sprites
                          jnp.where(rm_state.y == RADAR_MORTAR_SPAWN_UPPER_Y,
                                    aj.render_at( #upper
                                        raster,
                                        rm_state.x,
                                        rm_state.y + 15,
                                        death_sprite_upper_frame,
                                    ),
                                    aj.render_at( #lower
                                        raster,
                                        rm_state.x,
                                        rm_state.y - 25,
                                        death_sprite_lower_frame,
                                    )
                                    ),
                          raster
                          )

        # Byte bat
        bb_state = state.entities.byte_bat_state
        raster = jnp.where(jnp.logical_and(bb_state.is_in_current_event, jnp.logical_and(jnp.logical_not(bb_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                           aj.render_at(
                               aj.render_at(
                                   raster,
                                   bb_state.x,
                                   bb_state.y + ENTITY_DEATH_SPRITE_Y_OFFSET,
                                   death_sprite_upper_frame,
                               ),
                               bb_state.x,
                               bb_state.y - ENTITY_DEATH_SPRITES_SIZE[1] + ENTITY_DEATH_SPRITE_Y_OFFSET,
                               death_sprite_lower_frame,
                           ),
                           # Case: in event but dead and death animation over -> do not render
                           raster
                           )

        # Rock muncher
        rmu_state = state.entities.rock_muncher_state
        raster = jnp.where(jnp.logical_and(rmu_state.is_in_current_event, jnp.logical_and(jnp.logical_not(rmu_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                           aj.render_at(
                               aj.render_at(
                                   raster,
                                   rmu_state.x,
                                   rmu_state.y + ENTITY_DEATH_SPRITE_Y_OFFSET,
                                   death_sprite_upper_frame,
                               ),
                               rmu_state.x,
                               rmu_state.y - ENTITY_DEATH_SPRITES_SIZE[1] + ENTITY_DEATH_SPRITE_Y_OFFSET,
                               death_sprite_lower_frame,
                           ),
                           # Case: in event but dead and death animation over -> do not render
                           raster
                           )

        # Homing missile
        hm_state = state.entities.homing_missile_state
        raster = jnp.where(jnp.logical_and(hm_state.is_in_current_event, jnp.logical_and(jnp.logical_not(hm_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                           aj.render_at(
                               aj.render_at(
                                   raster,
                                   hm_state.x,
                                   hm_state.y + ENTITY_DEATH_SPRITE_Y_OFFSET,
                                   death_sprite_upper_frame,
                               ),
                               hm_state.x,
                               hm_state.y - ENTITY_DEATH_SPRITES_SIZE[1] + ENTITY_DEATH_SPRITE_Y_OFFSET,
                               death_sprite_lower_frame,
                           ),
                           # Case: in event but dead and death animation over -> do not render
                           raster
                           )

        # Detonator
        dn_state = state.entities.detonator_state
        raster = jnp.where(jnp.logical_and(dn_state.is_in_current_event, jnp.logical_and(jnp.logical_not(dn_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                          # Case: in event but dead -> render death sprites
                                aj.render_at( #lower
                                    raster,
                                    dn_state.x,
                                    dn_state.y + 12,
                                    death_sprite_upper_frame,
                                ),
                          raster
                          )

        # -------- Render Mountain Playing Field Background --------

        colored_playing_field_small_bg = recolor_sprite(SPRITE_PLAYING_FIELD_SMALL_BG, PLAYING_FIELD_COLOR)

        raster = aj.render_at(
                    aj.render_at(
                        aj.render_at(
                            aj.render_at(
                                raster,
                                0,
                                19,
                                SPRITE_PLAYING_FIELD_SMALL_BG  # upper background of background
                            ),
                            0,
                            19,
                            colored_playing_field_small_bg, # upper playing field background
                        ),
                        0,
                        80,
                        SPRITE_PLAYING_FIELD_SMALL_BG # lower background of background
                    ),
                    0,
                    80,
                    colored_playing_field_small_bg, # lower playing field background
                )

        # -------- Render Radar Mortar --------

        # Normal radar mortar
        radar_mortar_frame = aj.get_sprite_frame(SPRITE_RADAR_MORTAR, state.step_counter)
        radar_mortar_frame = jnp.where(state.entities.radar_mortar_state.y == RADAR_MORTAR_SPAWN_BOTTOM_Y, radar_mortar_frame, recolor_sprite(radar_mortar_frame, jnp.array(RADAR_MORTAR_COLOR_GRAY)))

        raster = jnp.where(
            jnp.logical_and(rm_state.is_in_current_event, rm_state.is_alive),
                # Case: alive -> render normally
                aj.render_at(
                    raster,
                    rm_state.x,
                    rm_state.y,
                    radar_mortar_frame,
                    flip_vertical=rm_state.y == RADAR_MORTAR_SPAWN_UPPER_Y,
                ),
            # Case: not in event -> do not render
            raster
        )

        # Render radar mortar missile
        should_render_rock_muncher_missile = jnp.logical_and(state.entities.radar_mortar_state.missile_x != 0, state.entities.radar_mortar_state.missile_y != 0)
        rock_muncher_missile_sprite = recolor_sprite(SPRITE_ENTITY_MISSILE, jnp.array(RADAR_MORTAR_COLOR_BLUE))

        raster = jnp.where(
            jnp.logical_and(should_render_rock_muncher_missile, state.entities.radar_mortar_state.is_in_current_event),
               aj.render_at(
                   raster,
                   state.entities.radar_mortar_state.missile_x,
                   state.entities.radar_mortar_state.missile_y,
                   rock_muncher_missile_sprite,
                   flip_horizontal=state.entities.radar_mortar_state.missile_direction[0] < 0,
               ),
               # Case: not in event -> do not render
           raster
           )


        # -------- Render Byte Bat --------

        # Normal Byte Bat
        byte_bat_frame = aj.get_sprite_frame(SPRITE_BYTE_BAT, state.step_counter)
        byte_bat_frame = recolor_sprite(byte_bat_frame, jnp.array(BYTE_BAT_COLOR))

        raster = jnp.where(
            jnp.logical_and(bb_state.is_in_current_event, bb_state.is_alive),
                # Case: alive -> render normally
                aj.render_at(
                    raster,
                    bb_state.x,
                    bb_state.y,
                    byte_bat_frame,
                ),
            # Case: not in event -> do not render
            raster
            )

        # -------- Render Rock Muncher --------

        # Normal rock_muncher
        rock_muncher_frame = aj.get_sprite_frame(SPRITE_ROCK_MUNCHER, state.step_counter)

        raster = jnp.where(
            jnp.logical_and(rmu_state.is_in_current_event, rmu_state.is_alive),
                # Case: alive -> render normally
                aj.render_at(
                    raster,
                    rmu_state.x,
                    rmu_state.y,
                    rock_muncher_frame,
                ),
            # Case: not in event -> do not render
            raster
            )


        # Render rock muncher missile
        rock_muncher_missile_sprite = recolor_sprite(SPRITE_ENTITY_MISSILE, jnp.array(ROCK_MUNCHER_MISSILE_COLOR))

        raster = jnp.where(jnp.logical_and(jnp.logical_and(rmu_state.missile_x != 0, rmu_state.missile_y != 0), rmu_state.is_in_current_event),
                           aj.render_at(
                               raster,
                               rmu_state.missile_x,
                               rmu_state.missile_y,
                               rock_muncher_missile_sprite,
                               flip_horizontal=True,
                           ),
                           raster
                           )

        # -------- Render Homing Missile --------

        raster = jnp.where(jnp.logical_and(hm_state.is_in_current_event, hm_state.is_alive),
            # Case: alive -> render normally
            aj.render_at(
                raster,
                hm_state.x,
                hm_state.y,
                SPRITE_HOMING_MISSILE,
            ),
            # Case: not in event -> do not render
            raster
        )

        # -------- Render Forcefield --------

        ff_state = state.entities.forcefield_state

        @jax.jit
        def recolor_forcefield(
                sprite: jnp.ndarray,
                x_position: jnp.ndarray,
                y_position: jnp.ndarray,
                flipped: jnp.ndarray
        ) -> jnp.ndarray:
            H, W, C = sprite.shape
            seed = jnp.asarray(x_position, dtype=jnp.int32)

            # Indices over columns (x-axis)
            xs = jnp.arange(W, dtype=jnp.int32)
            # Generate ys starting from y_position incrementing by 1 per column
            ys = xs + y_position.astype(jnp.int32)  # (W,) → y_position .. y_position+W-1
            # If flipped, reverse ys array, otherwise keep it as is
            ys = jnp.where(flipped, ys[::-1], ys)

            def sample_color(x):
                # Create PRNG key based on seed + column index for deterministic color
                key = jax.random.PRNGKey(seed + x)
                # Sample random RGB color values between 0 and 255
                rgb = jax.random.randint(key, (3,), 0, 256, dtype=jnp.int32)
                alpha = jnp.array([255], dtype=jnp.int32)  # Fully opaque alpha channel
                # Concatenate RGB with alpha channel (shape: (4,))
                return jnp.concatenate([rgb, alpha], axis=0)

            # Generate colors for each column using vmap for vectorization
            col_colors = jax.vmap(sample_color)(xs)  # shape: (W, C)

            # Freeze color boundaries (y positions) where color should not change
            freeze_lower_border = 32
            freeze_upper_border = 80  # Colors outside this range are frozen

            # Choose freeze color depending on whether ys >= freeze_lower_border
            frozen_color = col_colors[jnp.where(ys >= freeze_lower_border, freeze_upper_border, freeze_lower_border)]

            # Create mask for columns where y is within freeze bounds (inclusive)
            mask = jnp.logical_and(ys >= freeze_lower_border, ys <= freeze_upper_border)  # shape: (W,)
            # Use dynamic color where mask is True, else use frozen_color
            final_cols = jnp.where(
                mask[:, None],  # broadcast mask to (W, 1)
                col_colors,  # dynamic colors
                frozen_color  # frozen color for columns outside mask
            )

            # Broadcast final colors over height dimension to get full (H, W, C) image
            color_grid = jnp.broadcast_to(final_cols[None, :, :], (H, W, C))

            return color_grid

        # Despawn earlier
        move_left = FORCEFIELD_SIZE[0]
        render_x0 = jnp.where(ff_state.x0 <= 0, ff_state.x0 - move_left, ff_state.x0)
        render_x1 = jnp.where(ff_state.x1 <= 0, ff_state.x1 - move_left, ff_state.x1)
        render_x2 = jnp.where(ff_state.x2 <= 0, ff_state.x2 - move_left, ff_state.x2)
        render_x3 = jnp.where(ff_state.x3 <= 0, ff_state.x3 - move_left, ff_state.x3)
        render_x4 = jnp.where(ff_state.x4 <= 0, ff_state.x4 - move_left, ff_state.x4)
        render_x5 = jnp.where(ff_state.x5 <= 0, ff_state.x5 - move_left, ff_state.x5)

        x_positions = jnp.array([render_x0, render_x1 + 1, render_x2 + 2, render_x3 + 3, render_x4 + 4, render_x5 + 5], dtype=jnp.int32)
        y_positions = jnp.array([ff_state.y0, ff_state.y1, ff_state.y2, ff_state.y3, ff_state.y4, ff_state.y5], dtype=jnp.int32)
        flipped = jnp.array([False, True, False, True, False, True], dtype=jnp.bool)

        batched_recolor = jax.vmap(
            recolor_forcefield,
            in_axes=(None, 0, 0, 0),  # sprite bleibt gleich, x_position variiert
            out_axes=0  # erste Achse im Output wird die Batch‑Achse
        )
        all_sprites = batched_recolor(SPRITE_FORCEFIELD, x_positions, y_positions, flipped)

        def resize_sprite_width_ff(sprite: jnp.ndarray, new_width: int) -> jnp.ndarray:
            H, W, C = sprite.shape
            return jax.image.resize(sprite, (new_width, W, C), method='nearest')

        sprites_normal = all_sprites  # (6, 8, 73, 4)
        sprites_wide = jax.vmap(lambda sprite: resize_sprite_width_ff(sprite, FORCEFIELD_WIDE_SIZE[0]))(
            all_sprites)  # (6, 16, 73, 4)

        max_width = max(sprites_normal.shape[1], sprites_wide.shape[1])

        def pad_to_width_ff(sprites, width):
            pad = width - sprites.shape[1]
            return jnp.pad(sprites, ((0, 0), (0, pad), (0, 0), (0, 0)))  # pad in Width

        sprites_normal_padded = pad_to_width_ff(sprites_normal, max_width)  # (6, 16, 73, 4)
        sprites_wide_padded = pad_to_width_ff(sprites_wide, max_width)  # (6, 16, 73, 4)

        # Choose sprite if forcefield is wide
        all_sprites = jax.lax.cond(
            ff_state.is_wide,
            lambda _: sprites_wide_padded,
            lambda _: sprites_normal_padded,
            operand=None
        )

        raster = jnp.where(jnp.logical_and(ff_state.is_in_current_event, jnp.logical_and(ff_state.is_alive, ff_state.flash_on)),
            # Case: alive -> render normally
            aj.render_at(
                aj.render_at(
                    aj.render_at(
                        aj.render_at(
                            aj.render_at(
                                aj.render_at(
                                    raster,
                                    render_x0,
                                    ff_state.y0,
                                    all_sprites[0],
                                ),
                                render_x1,
                                ff_state.y1,
                                all_sprites[1],
                                flip_vertical=True
                            ),
                            render_x2,
                            ff_state.y2,
                            all_sprites[2],
                        ),
                        render_x3,
                        ff_state.y3,
                        all_sprites[3],
                        flip_vertical=True
                    ),
                    render_x4,
                    ff_state.y4,
                    all_sprites[4],
                ),
                render_x5,
                ff_state.y5,
                all_sprites[5],
                flip_vertical=True
            ),
            # Case: not in event -> do not render
            raster
        )

        # -------- Render Densepack --------

        dp_state = state.entities.dense_pack_state
        dp_x, dp_upmost_y, dp_height = dp_state.x, dp_state.upmost_y, DENSEPACK_NORMAL_PART_SIZE[1]

        # select correct sprites based on broken_states
        densepack_correct_sprites = SPRITE_DENSEPACK[dp_state.broken_states]
        # first recolor each part sprite
        recolored_sprites = jax.vmap(
            lambda sp: recolor_sprite(sp, jnp.array(DENSEPACK_COLOR)))(densepack_correct_sprites)

        def resize_sprite_width_dp(sprite: jnp.ndarray, new_width: int) -> jnp.ndarray:
            H, W, C = sprite.shape
            return jax.image.resize(sprite, (new_width, W, C), method='nearest')

        sprites_normal = recolored_sprites  # (6, 8, 73, 4)
        sprites_wide = jax.vmap(lambda sprite: resize_sprite_width_dp(sprite, FORCEFIELD_WIDE_SIZE[0]))(recolored_sprites)  # (6, 16, 73, 4)

        max_width = max(sprites_normal.shape[1], sprites_wide.shape[1])

        def pad_to_width_dp(sprites, width):
            pad = width - sprites.shape[1]
            return jnp.pad(sprites, ((0, 0), (0, pad), (0, 0), (0, 0)))  # pad in Width

        sprites_normal_padded = pad_to_width_dp(sprites_normal, max_width)  # (6, 16, 73, 4)
        sprites_wide_padded = pad_to_width_dp(sprites_wide, max_width)  # (6, 16, 73, 4)

        # Choose sprite if densepack is wide
        all_sprites = jax.lax.cond(
            dp_state.is_wide,
            lambda _: sprites_wide_padded,
            lambda _: sprites_normal_padded,
            operand=None
        )

        def render_densepack_parts(raster):
            def body(i, r):
                # compute y position of part i
                part_y = dp_upmost_y + i * dp_height
                # render the i‑th recolored sprite
                return aj.render_at(r, dp_x, part_y, all_sprites[i])

            # loop i from 0 to DENSEPACK_NUMBER_OF_PARTS
            return jax.lax.fori_loop(0, DENSEPACK_NUMBER_OF_PARTS, body, raster)

        # conditional: only render the fleet if in event & alive
        raster = jnp.where(
            jnp.logical_and(dp_state.is_in_current_event, dp_state.is_alive),
            render_densepack_parts(raster),  # true_fn
            raster  # false_fn
        )

        # -------- Render Densepack --------

        raster = jnp.where(
            jnp.logical_and(dn_state.is_in_current_event, dn_state.is_alive),
            aj.render_at(
                aj.render_at(
                    raster,
                    dn_state.x,
                    dn_state.y,
                    recolor_sprite(SPRITE_DETONATOR, jnp.array(DETONATOR_COLOR)),
                ),
                dn_state.x + 2,
                dn_state.y + 25,
                SPRITE_6507,
            ),
            raster
        )

        # -------- Render background parts --------

        # Upper brown background above upper mountains
        raster = aj.render_at(
            raster,
            0,
            0,
            SPRITE_UPPER_BROWN_BG,
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
        score_numbers_x = base_x - number_spacing * num_to_render # Subtracting offset of x position, since we want the score to be right-aligned

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


        def update_pygame(pygame_screen, raster, SCALING_FACTOR=3, WIDTH=400, HEIGHT=300): #TODO Delete when import of scripts/utils is working
            """Updates the Pygame display with the rendered raster.

            Args:
                pygame_screen: The Pygame screen surface.
                raster: JAX array of shape (Width, Height, 3/4) containing the image data.
                SCALING_FACTOR: Factor to scale the raster for display.
                WIDTH: Expected width of the input raster (used for scaling calculation).
                HEIGHT: Expected height of the input raster (used for scaling calculation).
            """
            pygame_screen.fill((0, 0, 0))

            # Convert JAX array (W, H, C) to NumPy (W, H, C)
            raster_np = jnp.array(raster)
            raster_np = raster_np.astype(jnp.uint8)

            # Pygame surface needs (W, H). make_surface expects (W, H, C) correctly.
            frame_surface = pygame.surfarray.make_surface(raster_np)

            # Pygame scale expects target (width, height)
            target_width_px = int(WIDTH * SCALING_FACTOR)
            target_height_px = int(HEIGHT * SCALING_FACTOR)
            # Optional: Adjust scaling if raster size differs from constants
            if raster_np.shape[0] != WIDTH or raster_np.shape[1] != HEIGHT:
                target_width_px = int(raster_np.shape[0] * SCALING_FACTOR)
                target_height_px = int(raster_np.shape[1] * SCALING_FACTOR)

            frame_surface_scaled = pygame.transform.scale(
                frame_surface, (target_width_px, target_height_px)
            )

            pygame_screen.blit(frame_surface_scaled, (0, 0))
            pygame.display.flip()


        update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        counter += 1
        clock.tick(60)

    pygame.quit()