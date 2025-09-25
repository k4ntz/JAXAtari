"""

Lukas Bergholz, Linus Orlob, Vincent Jahn

"""
import os
from functools import partial
from typing import Tuple, NamedTuple, Callable
import chex
import jax
import jax.numpy as jnp
import jaxatari.rendering.jax_rendering_utils as jru
from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, JAXAtariAction
from jaxatari.renderers import JAXGameRenderer

# -------- Game constants --------
class LaserGatesConstants:
    WIDTH = 160
    HEIGHT = 250
    SCALING_FACTOR = 5

    SCROLL_SPEED = 0.8 # Normal scroll speed
    SCROLL_MULTIPLIER = 2 # When at the right player bound, multiply scroll speed by this constant

    # -------- Mountains constants --------
    PLAYING_FIELD_BG_COLLISION_COLOR = (255, 255, 255, 255)
    PLAYING_FILED_BG_COLOR_FADE_SPEED = 0.2  # Higher = faster fade out, exponential

    # -------- Mountains constants --------
    MOUNTAIN_SIZE = (60, 12)  # Width, Height

    LOWER_MOUNTAINS_Y = 80  # Y Spawn position of lower mountains. This does not change
    UPPER_MOUNTAINS_Y = 19  # Y Spawn position of upper mountains. This does not change

    LOWER_MOUNTAINS_START_X = -44  # X Spawn position of lower mountains.
    UPPER_MOUNTAINS_START_X = -4  # X Spawn position of upper mountains.

    MOUNTAINS_DISTANCE = 20  # Distance between two given mountains

    UPDATE_EVERY = 4  # The mountain position is updated every UPDATE_EVERY-th frame.

    # -------- Player constants --------
    PLAYER_SIZE = (8, 6)  # Width, Height
    PLAYER_NORMAL_COLOR = (99, 138, 196, 255)  # Normal color of the player
    PLAYER_COLLISION_COLOR = (137, 81, 26,255)  # Players color for PLAYER_COLOR_CHANGE_DURATION frames after a collision
    PLAYER_COLOR_CHANGE_DURATION = 10  # How long (in frames) the player changes its color to PLAYER_COLLISION_COLOR if a collision occurs

    PLAYER_BOUNDS = (20, WIDTH - 20 - PLAYER_SIZE[0]), (19, 80 + PLAYER_SIZE[1])  # left x, right x, upper y and lower y bound of player

    PLAYER_START_X = 20  # X Spawn position of player
    PLAYER_START_Y = 52  # Y Spawn position of player

    PLAYER_VELOCITY_Y = 2  # Y Velocity of player
    PLAYER_VELOCITY_X = 2  # X Velocity of player

    # -------- Player missile constants --------
    PLAYER_MISSILE_SIZE = (16, 1)  # Width, Height
    PLAYER_MISSILE_BASE_COLOR = (140, 79, 24, 255)  # Initial color of player missile. Every value except for transparency is incremented by the missiles velocity * PLAYER_MISSILE_COLOR_CHANGE_SPEED
    PLAYER_MISSILE_COLOR_CHANGE_SPEED = 10  # Defines how fast the player missile changes its color towards white.

    PLAYER_MISSILE_INITIAL_VELOCITY = 9  # Starting speed of player missile
    PLAYER_MISSILE_VELOCITY_MULTIPLIER = 1.1  # Multiply the current speed at a given moment of the player missile by this number

    # -------- Entity constants (constants that apply to all entity types --------
    ENTITY_DEATH_SPRITES_SIZE = (8, 45)  # Width, Height
    ENTITY_MISSILE_SIZE = (4, 1)  # Width, Height

    NUM_ENTITY_TYPES = 8  # How many different (!) entity types there are
    ENTITY_DEATH_SPRITE_Y_OFFSET = 7  # Y offset to add to the death sprite (the constant is being added to the y coordinate)
    ENTITY_DEATH_ANIMATION_TIMER = 100  # Duration of death sprite animation in frames
    ENTITY_DEATH_SPRITES_NUMBER_COLOR = (117, 117, 213, 255)

    # -------- Radar mortar constants --------
    RADAR_MORTAR_SIZE = (8, 26)  # Width, Height
    RADAR_MORTAR_COLOR_BLUE = (96, 162, 228, 255)
    RADAR_MORTAR_COLOR_GRAY = (155, 155, 155, 255)

    RADAR_MORTAR_SPAWN_X = WIDTH  # Spawn barely outside of bounds
    RADAR_MORTAR_SPAWN_BOTTOM_Y = 66
    RADAR_MORTAR_SPAWN_UPPER_Y = 19  # Since the radar mortar can spawn at the top or at the bottom of the screen, we define two y positions.

    RADAR_MORTAR_MISSILE_COLOR = (85, 92, 197, 255)
    RADAR_MORTAR_MISSILE_SPAWN_EVERY = 100  # A missile is spawned every RADAR_MORTAR_MISSILE_SPAWN_EVERY-th frame.
    RADAR_MORTAR_MISSILE_SPEED = 3  # Speed of radar mortar missile
    RADAR_MORTAR_MISSILE_SHOOT_NUMBER = 3  # How often the missile gets teleported back before final shot (exept when shooting up or down)
    RADAR_MORTAR_MISSILE_SMALL_OUT_OF_BOUNDS_THRESHOLD = 50  # How far the missile needs to be away from the radar mortar (vertically or/and horizontally) for the missile to be teleported back to the mortar (to be shot again)
    RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD = 10  # This defines how far the player needs to be away from the radar mortar (vertically or/and horizontally) for the missile to be shot diagonally

    # -------- Byte bat constants --------
    BYTE_BAT_SIZE = (7, 8)  # Width, Height
    BYTE_BAT_COLOR = (90, 169, 99, 255)

    BYTE_BAT_UPPER_BORDER_Y = UPPER_MOUNTAINS_Y + MOUNTAIN_SIZE[1] + 2  # Upper border where byte bat inverts the direction
    BYTE_BAT_BOTTOM_BORDER_Y = LOWER_MOUNTAINS_Y - MOUNTAIN_SIZE[1]  # Lower border where byte bat inverts the direction
    BYTE_BAT_SUBTRACT_FROM_BORDER = 20
    BYTE_BAT_PLAYER_DIST_TRIGGER = 50

    BYTE_BAT_SPAWN_X = WIDTH  # Spawn barely outside screen
    BYTE_BAT_SPAWN_Y = BYTE_BAT_UPPER_BORDER_Y + 1

    BYTE_BAT_X_SPEED = 1  # Speed of byte bat in x direction
    BYTE_BAT_Y_SPEED = 1.8  # Speed of byte bat in y direction

    # -------- Rock muncher constants --------
    ROCK_MUNCHER_SIZE = (8, 11)  # Width, Height

    ROCK_MUNCHER_UPPER_BORDER_Y = UPPER_MOUNTAINS_Y + MOUNTAIN_SIZE[1] + 5 + 10  # Upper border where rock muncher inverts the direction
    ROCK_MUNCHER_BOTTOM_BORDER_Y = LOWER_MOUNTAINS_Y - MOUNTAIN_SIZE[1] - 3  # Lower border where rock muncher inverts the direction

    ROCK_MUNCHER_SPAWN_X = WIDTH  # Spawn barely outside of screen
    ROCK_MUNCHER_SPAWN_Y = ROCK_MUNCHER_UPPER_BORDER_Y + 1

    ROCK_MUNCHER_X_SPEED = 0.7  # Speed of rock muncher in x direction
    ROCK_MUNCHER_Y_SPEED = 1  # Speed of rock muncher in y direction

    ROCK_MUNCHER_MISSILE_COLOR = (85, 92, 197, 255)
    ROCK_MUNCHER_MISSILE_SPAWN_EVERY = 50  # Rock muncher shoots a new missile every ROCK_MUNCHER_MISSILE_SPAWN_EVERY frames
    ROCK_MUNCHER_MISSILE_SPEED = 4  # Speed of rock muncher missile

    # -------- Homing Missile constants --------
    HOMING_MISSILE_SIZE = (8, 5)  # Width, Height

    HOMING_MISSILE_Y_BOUNDS = (32, 74)
    HOMING_MISSILE_PLAYER_TRACKING_RANGE = 15  # The minimum y position difference between player and homing missile needed for the homing missile to start tracking the player
    HOMING_MISSILE_Y_PLAYER_OFFSET = 2  # Sets the y position this many pixels above the player (for positive numbers) or below the player (for negative numbers)
    HOMING_MISSILE_X_SPEED = 2.5
    HOMING_MISSILE_Y_SPEED = 1

    # -------- Forcefield constants --------
    FORCEFIELD_SIZE = (8, 73)  # Width, Height of a single normal forcefield column
    FORCEFIELD_WIDE_SIZE = (16, 73)  # Width, Height of a single wide forcefield column

    FORCEFIELD_IS_WIDE_PROBABILITY = 0.2  # Probability that a forcefield is wide

    FORCEFIELD_FLASHING_SPACING = 32  # x spacing between the forcefields when in flashing mode
    FORCEFIELD_FLASHING_SPEED = 35  # Forcefield changes state from on to off or from off to on every FORCEFIELD_FLASHING_SPEED frames

    FORCEFIELD_FLEXING_SPACING = 64  # x spacing between the forcefields when in flexing mode
    FORCEFIELD_FLEXING_SPEED = 0.6  # Flexing (Crushing motion) speed
    FORCEFIELD_FLEXING_MINIMUM_DISTANCE = 2  # Minimum y distance between the upper and lower forcefields when flexing
    FORCEFIELD_FLEXING_MAXIMUM_DISTANCE = 32  # Maximum y distance between the upper and lower forcefields when flexing

    FORCEFIELD_FIXED_SPACING = 64  # x spacing between the forcefields when in fixed mode
    FORCEFIELD_FIXED_SPEED = 0.3  # Fixed (up and down movement) speed
    FORCEFIELD_FIXED_UPPER_BOUND = -FORCEFIELD_SIZE[1] + 33  # Highest allowed y position for forcefields while fixed
    FORCEFIELD_FIXED_LOWER_BOUND = -FORCEFIELD_SIZE[1] + 68  # Lowest allowed y position for forcefields while fixed

    # -------- Densepack constants --------
    DENSEPACK_NORMAL_PART_SIZE = (8, 4)  # Width, Height of a single, normal densepack part in a normal densepack column
    DENSEPACK_WIDE_PART_SIZE = (16, 4)  # Width, Height of a single, wide densepack part in a wide densepack column
    DENSEPACK_COLOR = (142, 142, 142, 255)

    DENSEPACK_NUMBER_OF_PARTS = 19  # number of segments in the densepack
    DENSEPACK_IS_WIDE_PROBABILITY = 0.4  # Probability that a spawned densepack is wide

    # -------- Detonator constants --------
    DETONATOR_SIZE = (8, 73)
    DETONATOR_COLOR = (142, 142, 142, 255)

    # -------- Energy pod constants --------
    ENERGY_POD_SIZE = (6, 4)  # Width, Height
    ENERGY_POD_COLOR_GREEN = (84, 171, 96, 255)  # Energy pod color in green frame
    ENERGY_POD_COLOR_GRAY = (142, 142, 142, 255)  # Energy pod color in gray frame

    ENERGY_POD_ANIMATION_SPEED = 16  # Higher is slower

    # -------- Probability constants --------

    ALLOW_ENERGY_POD_PERCENTAGE = 0.3  # The energy pod is allowed to spawn (one in 7 to 8 chance) when current energy is smaller than ALLOW_ENERGY_POD_PERCENTAGE * MAX_ENERGY
    ALLOW_DETONATOR_PERCENTAGE = 0.3  # The detonator is allowed to spawn (one in 7 to 8 chance) when current energy is smaller than ALLOW_DETONATOR_PERCENTAGE * MAX_DTIME

    ENERGY_POD_SPAWN_PROBABILITY = 0.4  # If energy pod spawning is allowed, this is the probability for the energy pod to be the next entity spawned.
    DETONATOR_SPAWN_PROBABILITY = 0.4  # If detonator spawning is allowed, this is the probability for the detonator to be the next entity spawned.

    ENERGY_START_BLINKING_PERCENTAGE = 0.2 # see below
    SHIELDS_START_BLINKING_PERCENTAGE = 0.2 # see below
    DTIME_START_BLINKING_PERCENTAGE = 0.2 # The Field in the instrument panel starts blinking when the current value is smaller than VALUE_START_BLINKING_PERCENTAGE * MAX_VALUE.

    # -------- Instrument panel constants --------

    INSTRUMENT_PANEL_ANIMATION_SPEED = 14 # Blinking speed when energy, shields or dtime is low. Lower is faster. Should ideally be an even number.

    MAX_ENERGY = 5100  # As the manual says, energy is consumed at a regular pace. We use 5100 for the initial value and subtract one for every frame to match the timing of the real game. (It takes 85 seconds for the energy to run out. 85 * 60 (fps) = 5100)
    MAX_SHIELDS = 24  # As the manual says, the Dante Dart starts with 24 shield units
    MAX_DTIME = 10200  # Same idea as energy.

    SHIELD_LOSS_COL_SMALL = 1  # See is_big_collision entry in CollisionPropertiesState for extensive explanation. This constant defines the shield points to lose
    SHIELD_LOSS_COL_BIG = 6

    # -------- GUI constants --------
    GUI_COLORED_BACKGROUND_SIZE = (128, 12)  # Width, Height of colored background of black rectangle background
    GUI_BLACK_BACKGROUND_SIZE = (56, 10)  # Width, Height of black background of the text
    GUI_TEXT_SCORE_SIZE = (21, 7)  # Width, Height of "Score" text
    GUI_TEXT_ENERGY_SIZE = (23, 5)  # Width, Height of "Energy" text
    GUI_TEXT_SHIELDS_SIZE = (23, 5)  # Width, Height of "Shields" text
    GUI_TEXT_DTIME_SIZE = (23, 5)  # Width, Height of "Dtime" text

    GUI_COLORED_BACKGROUND_COLOR_BLUE = (47, 90, 160, 255)
    GUI_COLORED_BACKGROUND_COLOR_GREEN = (50, 152, 82, 255)
    GUI_COLORED_BACKGROUND_COLOR_BEIGE = (160, 107, 50, 255)
    GUI_COLORED_BACKGROUND_COLOR_GRAY = (182, 182, 182, 255)
    GUI_TEXT_COLOR_GRAY = (118, 118, 118, 255)
    GUI_TEXT_COLOR_BEIGE = (160, 107, 50, 255)

    GUI_BLACK_BACKGROUND_X_OFFSET = 36
    GUI_Y_BASE = 117
    GUI_X_BASE = 16
    GUI_Y_SPACE_BETWEEN_PANELS = 21

    # -------- Debug constants --------
    DEBUG_ACTIVATE_MOUNTAINS_SCROLL = jnp.bool(True)


# -------- States --------
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
    rightmost_x: chex.Array
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
    animation_timer: chex.Array

class CollisionPropertiesState(NamedTuple):
    collision_with_player: jnp.bool             # Player collision with entity
    collision_with_player_missile: jnp.bool     # Player missile collision with entity
    is_big_collision: jnp.bool                  # If 1 or 6 shield points should be subtracted at collision.
                                                # Candidates for small collision: Computer wall (bounds, not handled here, see check_player_and_player_missile_collision_bounds), Rock Muncher missile, Radar Mortar Missile
                                                # Candidates for big collision: Byte Bat, Rock Muncher, Homing Missile, Radar Cannon, Densepack Column, any Forcefield or Detonator
    is_energy_pod: jnp.bool                     # If entity is energy pod
    is_detonator: jnp.bool                      # If entity is detonator
    is_ff_or_dp: jnp.bool                       # If entity is forcefield or densepack
    score_to_add: chex.Array                    # Score to add to the current score at collision. Radar Mortar: 115, Rock Muncher: 325, Byte Bat: 330, Pass Forcefield: 400, Homing Missile: 525, Detonator: 6507
    death_timer: chex.Array                     # Animation timer used for entity death animations. Change the speed with ENTITY_DEATH_ANIMATION_TIMER

class EntitiesState(NamedTuple):
    radar_mortar_state: RadarMortarState        # Radar mortars appear along the top and bottom of the Computer passage. Avoid Mortar fire. Demolish Radar Mortars with laser fire.
    byte_bat_state: ByteBatState                # Green bat looking entity flying at you without warning.
    rock_muncher_state: RockMuncherState        # Pink brown green entity flying at you without warning. Shoots missiles.
    homing_missile_state: HomingMissileState    # Bomb looking entity flying and tracking you
    forcefield_state: ForceFieldState           # Flashing, flexing or fixed "wall". Time your approach to cross.
    dense_pack_state: DensepackState            # Gray densepack columns of varying width appear along the dark Computer passage. Blast your way through.
    detonator_state: DetonatorState             # "Failsafe detonators" are large and grey and have the numbers "6507" etched on the side. Laser fire must strike one of the pins on the side of a detonator to destroy it.
    energy_pod_state: EnergyPodState            # To replenish energy reserves, touch Energy Pods as they appear along the Computer passageway. Do not fire at Energy Pods! You may not survive until another appears!

    collision_properties_state: CollisionPropertiesState # Holds attributes relevant for collision logic

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
    # Player + Player Missile
    player: EntityPosition
    player_missile: EntityPosition

    # Single-entity enemies (+ missiles)
    radar_mortar: EntityPosition
    radar_mortar_missile: EntityPosition
    byte_bat: EntityPosition
    rock_muncher: EntityPosition
    rock_muncher_missile: EntityPosition
    homing_missile: EntityPosition

    # Forcefields: up to 3 Pairs (upper/lower)
    forcefield_0_upper: EntityPosition
    forcefield_0_lower: EntityPosition
    forcefield_1_upper: EntityPosition
    forcefield_1_lower: EntityPosition
    forcefield_2_upper: EntityPosition
    forcefield_2_lower: EntityPosition

    # Densepack, Detonator, Energy Pod
    densepack: EntityPosition
    detonator: EntityPosition
    energy_pod: EntityPosition

    # Mountains (3 top, 3 bottom)
    upper_mountain_0: EntityPosition
    upper_mountain_1: EntityPosition
    upper_mountain_2: EntityPosition
    lower_mountain_0: EntityPosition
    lower_mountain_1: EntityPosition
    lower_mountain_2: EntityPosition


class LaserGatesInfo(NamedTuple):
    # difficulty: jnp.ndarray # add if necessary
    step_counter: jnp.ndarray
    all_rewards: jnp.ndarray

# -------- Render Constants --------
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Background parts
    upper_brown_bg = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/upper_brown_bg.npy"))
    lower_brown_bg = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/lower_brown_bg.npy"))
    playing_field_bg = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/playing_field_bg.npy"))
    playing_field_small_bg = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/mountains/playing_field_small_bg.npy"))
    gray_gui_bg = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/gray_gui_bg.npy"))
    lower_mountain = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/mountains/lower_mountain.npy"))
    upper_mountain = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/mountains/upper_mountain.npy"))
    black_stripe = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/black_stripe.npy"))

    # Player and player missile
    player = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/player/player.npy"))
    player_missile = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/missiles/player_missile.npy"))

    # Instrument panel parts
    gui_colored_background = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/colored_background.npy"))
    gui_black_background = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/black_background.npy"))
    gui_text_score = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/text/score.npy"))
    gui_text_energy = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/text/energy.npy"))
    gui_text_shields = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/text/shields.npy"))
    gui_text_dtime = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/text/dtime.npy"))
    gui_score_digits = jru.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/lasergates/gui/score_numbers/{}.npy"))
    gui_score_comma = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/gui/score_numbers/comma.npy"))

    # Entities
    # Entity missile
    entity_missile = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/missiles/enemy_missile.npy"))

    # Death sprites
    upper_death_sprites_temp = []
    for i in range(1, 13):
        temp = jru.loadFrame(os.path.join(MODULE_DIR, f"sprites/lasergates/enemies/enemy_death/top/{i}.npy"))
        upper_death_sprites_temp.append(temp)
        upper_death_sprites_temp[i - 1] = jnp.expand_dims(upper_death_sprites_temp[i - 1], axis=0)

    upper_death_sprites = jnp.concatenate(upper_death_sprites_temp, axis=0)

    lower_death_sprites_temp = []
    for i in range(1, 13):
        temp = jru.loadFrame(os.path.join(MODULE_DIR, f"sprites/lasergates/enemies/enemy_death/bottom/{i}.npy"))
        lower_death_sprites_temp.append(temp)
        lower_death_sprites_temp[i - 1] = jnp.expand_dims(lower_death_sprites_temp[i - 1], axis=0)

    lower_death_sprites = jnp.concatenate(lower_death_sprites_temp, axis=0)

    death_sprite_number_325 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/enemy_death/numbers/325.npy"))
    death_sprite_number_525 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/enemy_death/numbers/525.npy"))
    death_sprite_number_bg = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/enemy_death/numbers/background.npy"))


    # Radar mortar
    radar_mortar_frame_left = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/radar_mortar/1.npy"))
    radar_mortar_frame_middle = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/radar_mortar/2.npy"))
    radar_mortar_frame_right = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/radar_mortar/3.npy"))

    rms, _ = jru.pad_to_match([radar_mortar_frame_left, radar_mortar_frame_middle, radar_mortar_frame_right])
    RADAR_MORTAR_SPRITE_ANIMATION_SPEED = 15  # Change sprite frame (left, middle, right) of radar mortar every RADAR_MORTAR_SPRITE_ROTATION_SPEED frames
    radar_mortar_sprites = jnp.concatenate([
        jnp.repeat(rms[0][None], RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
        jnp.repeat(rms[1][None], RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
        jnp.repeat(rms[2][None], RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
        jnp.repeat(rms[1][None], RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
    ]) # Radar mortar rotation animation

    # Byte bat
    byte_bat_frame_up = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/byte_bat/1.npy"))
    byte_bat_frame_mid = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/byte_bat/2.npy"))
    byte_bat_frame_down = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/byte_bat/3.npy"))

    bbs, _ = jru.pad_to_match([byte_bat_frame_up, byte_bat_frame_mid, byte_bat_frame_down, byte_bat_frame_mid])
    BYTE_BAT_ANIMATION_SPEED = 16  # Flap speed of byte bat, higher is slower
    byte_bat_sprites = jnp.concatenate([
        jnp.repeat(bbs[0][None], BYTE_BAT_ANIMATION_SPEED, axis=0),
        jnp.repeat(bbs[1][None], BYTE_BAT_ANIMATION_SPEED, axis=0),
        jnp.repeat(bbs[2][None], BYTE_BAT_ANIMATION_SPEED, axis=0),
        jnp.repeat(bbs[1][None], BYTE_BAT_ANIMATION_SPEED, axis=0),
    ]) # Byte bat flap animation

    # Rock muncher
    rock_muncher_frame_small = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/rock_muncher/1.npy"))
    rock_muncher_frame_mid = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/rock_muncher/2.npy"))
    rock_muncher_frame_big = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/rock_muncher/3.npy"))

    rmus, _ = jru.pad_to_match([rock_muncher_frame_small, rock_muncher_frame_mid, rock_muncher_frame_big, rock_muncher_frame_mid])
    ROCK_MUNCHER_ANIMATION_SPEED = 10  # Animation speed of rock muncher
    rock_muncher_sprites = jnp.concatenate([
        jnp.repeat(rmus[0][None], ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
        jnp.repeat(rmus[1][None], ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
        jnp.repeat(rmus[2][None], ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
        jnp.repeat(rmus[1][None], ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
    ]) # Rock muncher animation

    # Homing missile
    homing_missile_sprite = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/homing_missile/homing_missile.npy"))

    # Forcefield
    forcefield_sprite = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/forcefield/forcefield.npy"))

    # Densepack
    densepack_frame_0 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/5.npy"))
    densepack_frame_1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/4.npy"))
    densepack_frame_2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/3.npy"))
    densepack_frame_3 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/2.npy"))
    densepack_frame_4 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/densepack/1.npy"))

    densepack_sprites = jnp.array([
        densepack_frame_0, densepack_frame_1, densepack_frame_2, densepack_frame_3, densepack_frame_4
    ])

    # Detonator
    detonator_sprite = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/detonator/detonator.npy"))
    detonator_6507 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/detonator/6507.npy"))

    # Energy pods
    energy_pod_sprite = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/enemies/energy_pod/energy_pod.npy"))

    return (
        # Player sprites
        player,
        player_missile,

        # Entity sprites
        entity_missile,
        upper_death_sprites,
        lower_death_sprites,
        death_sprite_number_325,
        death_sprite_number_525,
        death_sprite_number_bg,
        radar_mortar_sprites,
        byte_bat_sprites,
        rock_muncher_sprites,
        homing_missile_sprite,
        forcefield_sprite,
        densepack_sprites,
        detonator_sprite,
        detonator_6507,
        energy_pod_sprite,

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
    SPRITE_DEATH_NUMBER_325,
    SPRITE_DEATH_NUMBER_525,
    SPRITE_DEATH_NUMBER_BG,
    SPRITE_RADAR_MORTAR,
    SPRITE_BYTE_BAT,
    SPRITE_ROCK_MUNCHER,
    SPRITE_HOMING_MISSILE,
    SPRITE_FORCEFIELD,
    SPRITE_DENSEPACK,
    SPRITE_DETONATOR,
    SPRITE_6507,
    SPRITE_ENERGY_POD,

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

class JaxLaserGates(JaxEnvironment[LaserGatesState, LaserGatesObservation, LaserGatesInfo, LaserGatesConstants]):

    def __init__(self, consts: LaserGatesConstants = None, frameskip: int = 1, reward_funcs: list[Callable] =None):
        consts = consts or LaserGatesConstants()
        super().__init__(consts)
        self.frameskip = frameskip
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
        self.frame_stack_size = 4
        self.num_obs_slots = 23
        self.features_per_slot = 5  # x, y, w, h, active
        self.obs_size = self.num_obs_slots * self.features_per_slot
        self.renderer = LaserGatesRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: LaserGatesState) -> jnp.ndarray:
        img = self.renderer.render(state)
        img = jnp.clip(img, 0, 255)
        return img.astype(jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def maybe_initialize_random_entity(self, entities, state):
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

        def initialize_radar_mortar(entities):
            top_or_bot = jax.random.bernoulli(key_intern)

            new_radar_mortar_state = RadarMortarState(
                is_in_current_event = jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.RADAR_MORTAR_SPAWN_X).astype(entities.radar_mortar_state.x.dtype),
                y=jnp.where(top_or_bot, self.consts.RADAR_MORTAR_SPAWN_BOTTOM_Y, self.consts.RADAR_MORTAR_SPAWN_UPPER_Y),
                missile_x = jnp.array(0),
                missile_y = jnp.array(0),
                missile_direction = jnp.array((0, 0)),
                shoot_again_timer = jnp.array(0),
            )
            return entities._replace(radar_mortar_state=new_radar_mortar_state)

        def initialize_byte_bat(entities):
            initial_direction_is_up = jnp.bool(self.consts.BYTE_BAT_SPAWN_Y < self.consts.BYTE_BAT_UPPER_BORDER_Y)
            new_byte_bat_state = ByteBatState(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.BYTE_BAT_SPAWN_X).astype(entities.byte_bat_state.x.dtype),
                y=jnp.array(self.consts.BYTE_BAT_SPAWN_Y).astype(entities.byte_bat_state.y.dtype),
                direction_is_up=initial_direction_is_up,
                direction_is_left=jnp.bool(True)
            )
            return entities._replace(byte_bat_state=new_byte_bat_state)

        def initialize_rock_muncher(entities):
            initial_direction_is_up = jnp.bool(self.consts.ROCK_MUNCHER_SPAWN_Y < self.consts.ROCK_MUNCHER_UPPER_BORDER_Y)
            new_rock_muncher_state = RockMuncherState(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.ROCK_MUNCHER_SPAWN_X).astype(entities.byte_bat_state.x.dtype),
                y=jnp.array(self.consts.ROCK_MUNCHER_SPAWN_Y).astype(entities.byte_bat_state.y.dtype),
                direction_is_up=initial_direction_is_up,
                direction_is_left=jnp.bool(True),
                missile_x=jnp.array(0),
                missile_y=jnp.array(0),
            )
            return entities._replace(rock_muncher_state=new_rock_muncher_state)

        def initialize_homing_missile(entities):
            initial_y_position = jax.random.randint(key_intern, (), self.consts.HOMING_MISSILE_Y_BOUNDS[0], self.consts.HOMING_MISSILE_Y_BOUNDS[1])
            new_homing_missile_state = HomingMissileState(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.WIDTH).astype(entities.homing_missile_state.x.dtype),
                y=initial_y_position,
                is_tracking_player=jnp.bool(False),
            )
            return entities._replace(homing_missile_state=new_homing_missile_state)

        def initialize_forcefield(entities):
            key_num_of_ff, key_type_of_ff, key_is_wide = jax.random.split(key_intern, 3)
            number_of_forcefields = jax.random.randint(key_num_of_ff, (), minval=1, maxval=5) # Spawn 1 to 4 forcefields at a time.

            type_of_forcefield = jax.random.randint(key_type_of_ff, (), minval=0, maxval=3)
            init_is_flexing = type_of_forcefield == 0
            init_is_fixed = type_of_forcefield == 1

            init_is_wide = jax.random.bernoulli(key_is_wide, p=self.consts.FORCEFIELD_IS_WIDE_PROBABILITY)

            number_of_forcefields = jnp.where(init_is_wide, 1, number_of_forcefields)

            new_forcefield_state = entities.forcefield_state._replace(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x0=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y0=jnp.where(init_is_flexing, -10, jnp.where(init_is_fixed, -20, -17)).astype(jnp.float32),
                x1=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y1=jnp.where(init_is_flexing, 65, jnp.where(init_is_fixed, 65, 56)).astype(jnp.float32),
                x2=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y2=jnp.where(init_is_flexing, -10, jnp.where(init_is_fixed, -20, -17)).astype(jnp.float32),
                x3=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y3=jnp.where(init_is_flexing, 65, jnp.where(init_is_fixed, 65, 56)).astype(jnp.float32),
                x4=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y4=jnp.where(init_is_flexing, -10, jnp.where(init_is_fixed, -20, -17)).astype(jnp.float32),
                x5=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y5=jnp.where(init_is_flexing, 65, jnp.where(init_is_fixed, 65, 56)).astype(jnp.float32),
                rightmost_x=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                num_of_forcefields=jnp.array(number_of_forcefields),
                is_wide=init_is_wide,
                is_flexing=init_is_flexing,
                is_fixed=init_is_fixed,
                flash_on=jnp.array(True),
                flex_upper_direction_is_up=jnp.array(True),
                fixed_upper_direction_is_up=jnp.array(True),
            )
            return entities._replace(forcefield_state=new_forcefield_state)

        def initialize_densepack(entities):
            initial_is_wide = jax.random.bernoulli(key_intern, p=self.consts.DENSEPACK_IS_WIDE_PROBABILITY)

            new_densepack_state = entities.dense_pack_state._replace(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.WIDTH).astype(jnp.float32),
                upmost_y=jnp.array(19).astype(jnp.float32),
                is_wide=initial_is_wide,
                number_of_parts=jnp.array(self.consts.DENSEPACK_NUMBER_OF_PARTS).astype(jnp.int32),
                broken_states=jnp.full(self.consts.DENSEPACK_NUMBER_OF_PARTS, 4, jnp.int32),
            )
            return entities._replace(dense_pack_state=new_densepack_state)

        def initialize_detonator(entities):
            new_detonator_state = entities.detonator_state._replace(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.WIDTH).astype(jnp.float32),
                y=jnp.array(19).astype(jnp.float32),
                collision_is_pin=jnp.bool(False),
            )
            return entities._replace(detonator_state=new_detonator_state)

        def initialize_energy_pod(entities):
            new_energy_pod_state = entities.energy_pod_state._replace(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.WIDTH).astype(jnp.float32),
                y=jnp.array(73).astype(jnp.float32),
                animation_timer=jnp.array(0),
            )
            return entities._replace(energy_pod_state=new_energy_pod_state)

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
            key_normal_index, key_energy_pod, key_detonator, key_edge_case = jax.random.split(key_pick_type, 4)

            # Randomly choose one of the entities, except for the energy pod and detonator (see below)
            picked_index = jax.random.randint(key_normal_index, shape=(), minval=0, maxval=6) # Default: minval=0, maxval=6
            # If you want only one specific entity to spawn, change minval, maxval to:
            # Radar Mortar:     minval=0, maxval=1
            # Byte Bat:         minval=1, maxval=2
            # Rock Muncher:     minval=2, maxval=3
            # Homing Missile:   minval=3, maxval=4
            # Forcefields:      minval=4, maxval=5
            # Densepack:        minval=5, maxval=6
            # Detonator:        minval=6, maxval=7
            # Energy pod:       minval=7, maxval=8

            # Allow spawning of detonator or energy pod if values in the state are low enough
            allow_spawn_detonator = state.dtime < (self.consts.ALLOW_DETONATOR_PERCENTAGE * self.consts.MAX_DTIME)
            allow_spawn_energy_pod = state.energy < (self.consts.ALLOW_ENERGY_POD_PERCENTAGE * self.consts.MAX_ENERGY)
            # Spawn detonator or energy pod if is allowed and probability is hit
            spawn_detonator = jnp.where(allow_spawn_detonator, jax.random.bernoulli(key_detonator, p=self.consts.DETONATOR_SPAWN_PROBABILITY), jnp.bool(False))
            spawn_energy_pod = jnp.where(allow_spawn_energy_pod, jax.random.bernoulli(key_energy_pod, p=self.consts.ENERGY_POD_SPAWN_PROBABILITY), jnp.bool(False))
            # Spawn detonator or energy pod
            picked_index = jnp.where(spawn_detonator, 6, picked_index)
            picked_index = jnp.where(spawn_energy_pod, 7, picked_index)
            # In the rare case that both detonator and energy pod are spawned, reroll
            picked_index = jnp.where(jnp.logical_and(spawn_detonator, spawn_energy_pod),
            jnp.where(jax.random.bernoulli(key_edge_case), 6, 7),
            picked_index)

            # Call initialize function of picked entity
            return jax.lax.switch(picked_index, init_fns, entities) # Initialize function of randomly picked entity

        return jax.lax.cond(
            active_event,
            lambda _: entities,         # Return the current entities state if there still is an active entity present
            initialize_random_entity,   # Else spawn a new entity with random type (see initialize_random_entity)
            operand=None,
        )


    @partial(jax.jit, static_argnums=(0,))
    def mountains_step(
            self, mountain_state: MountainState, state: LaserGatesState
    ) -> MountainState:

        # If this is true, update the position
        update_tick = state.step_counter % self.consts.UPDATE_EVERY == 0
        update_tick = jnp.logical_and(update_tick, self.consts.DEBUG_ACTIVATE_MOUNTAINS_SCROLL)

        # Update x positions
        new_x1 = jnp.where(update_tick, mountain_state.x1 - self.consts.UPDATE_EVERY * state.scroll_speed, mountain_state.x1)
        new_x2 = jnp.where(update_tick, mountain_state.x2 - self.consts.UPDATE_EVERY * state.scroll_speed, mountain_state.x2)
        new_x3 = jnp.where(update_tick, mountain_state.x3 - self.consts.UPDATE_EVERY * state.scroll_speed, mountain_state.x3)

        # If completely behind the left border, set x position to the right again
        new_x1 = jnp.where(new_x1 < 0 - self.consts.MOUNTAIN_SIZE[0], new_x3 + self.consts.MOUNTAIN_SIZE[0] + self.consts.MOUNTAINS_DISTANCE, new_x1)
        new_x2 = jnp.where(new_x2 < 0 - self.consts.MOUNTAIN_SIZE[0], new_x1 + self.consts.MOUNTAIN_SIZE[0] + self.consts.MOUNTAINS_DISTANCE, new_x2)
        new_x3 = jnp.where(new_x3 < 0 - self.consts.MOUNTAIN_SIZE[0], new_x2 + self.consts.MOUNTAIN_SIZE[0] + self.consts.MOUNTAINS_DISTANCE, new_x3)

        return MountainState(x1=new_x1.astype(jnp.int32), x2=new_x2.astype(jnp.int32), x3=new_x3.astype(jnp.int32), y=mountain_state.y.astype(jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def all_entities_step(self, game_state: LaserGatesState) -> EntitiesState:
        """
        steps the entity (actually entities, but we only have one entity per event) that is currently in game (if is_in_current_event of said entity is True).
        """

        @jax.jit
        def radar_mortar_step(state: LaserGatesState) -> tuple[RadarMortarState, CollisionPropertiesState]:
            rm = state.entities.radar_mortar_state
            new_x = jnp.where(rm.is_alive, rm.x - state.scroll_speed, rm.x)

            # Compute spawn position & 45 degree - direction
            is_at_bottom = rm.y == self.consts.RADAR_MORTAR_SPAWN_BOTTOM_Y
            offset_y = jnp.where(is_at_bottom, 0, self.consts.RADAR_MORTAR_SIZE[1])
            spawn_x = rm.x
            spawn_y = rm.y + offset_y

            is_left = state.player_x < (spawn_x - self.consts.RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD)
            is_right = state.player_x > (spawn_x + self.consts.RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD)
            is_above = state.player_y < (spawn_y - self.consts.RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD)
            is_below = state.player_y > (spawn_y + self.consts.RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD)
            dx = jnp.where(is_left, -2, jnp.where(is_right, 2, 0))
            dy = jnp.where(is_above, -1, jnp.where(is_below, 1, 0))
            dir_to_player = jnp.array([dx, dy])

            # Out-of-bounds check for final kill
            out_of_bounds = jnp.logical_or(
                jnp.logical_or(rm.missile_x < 0, rm.missile_x > self.consts.WIDTH),
                jnp.logical_or(rm.missile_y < self.consts.PLAYER_BOUNDS[1][0],
                               rm.missile_y > self.consts.PLAYER_BOUNDS[1][1])
            )

            # Fresh spawn condition
            missile_dead = jnp.all(rm.missile_direction == 0)
            spawn_trigger = jnp.logical_and(rm.is_alive, (state.step_counter % self.consts.RADAR_MORTAR_MISSILE_SPAWN_EVERY) == 0)

            # small_out_of_bounds: moved beyond 5px from spawn?
            small_oob = jnp.logical_or(
                jnp.abs(rm.missile_x - spawn_x) > self.consts.RADAR_MORTAR_MISSILE_SMALL_OUT_OF_BOUNDS_THRESHOLD,
                jnp.abs(rm.missile_y - spawn_y) > self.consts.RADAR_MORTAR_MISSILE_SMALL_OUT_OF_BOUNDS_THRESHOLD
            )

            # Check if the direction is up or down (0, 1)
            slow_direction = jnp.logical_or(
                jnp.all(dir_to_player == jnp.array([0, 1])),
                jnp.all(dir_to_player == jnp.array([0, -1]))
            )

            # Only start repeat fire when fresh spawn occurred and is alive
            fresh_spawn = jnp.logical_and(jnp.logical_and(missile_dead, spawn_trigger), rm.is_alive)

            # Decide new timer value
            should_decrement = jnp.logical_and(rm.shoot_again_timer > 0, small_oob)

            # Apply conditional timer set:
            # - RADAR_MORTAR_MISSILE_SHOOT_NUMBER if fresh spawn and direction valid
            # - 1 if fresh spawn and direction is a slow direction
            new_timer = jnp.where(fresh_spawn,
                                  jnp.where(slow_direction, 1, self.consts.RADAR_MORTAR_MISSILE_SHOOT_NUMBER),
                                  jnp.where(should_decrement, rm.shoot_again_timer - 1, rm.shoot_again_timer))

            # in_spawn_phase: teleport back if fresh_spawn or (timer > 0 and small_oob)
            in_spawn_phase = jnp.logical_or(
                fresh_spawn,
                jnp.logical_and(new_timer > 0, small_oob)
            )

            # Base position and direction: either spawn or keep old
            base_x = jnp.where(in_spawn_phase, spawn_x, rm.missile_x)
            base_y = jnp.where(in_spawn_phase, spawn_y, rm.missile_y)
            # Keep the original direction until the timer runs out
            base_dir = jnp.where(fresh_spawn, dir_to_player, rm.missile_direction)

            # Kill only if timer == 0 and fully out_of_bounds
            kill = jnp.logical_and(new_timer == 0, out_of_bounds)
            missile_x = jnp.where(kill, 0, base_x)
            missile_y = jnp.where(kill, 0, base_y)
            missile_dir = jnp.where(kill, jnp.array([0, 0], dtype=jnp.int32), base_dir)

            # Move if alive and not in spawn phase
            alive = jnp.any(missile_dir != 0)
            speed = jnp.where(slow_direction, 1, self.consts.RADAR_MORTAR_MISSILE_SPEED)
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
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_x, state.player_y), self.consts.PLAYER_SIZE, (new_x, rm.y), self.consts.RADAR_MORTAR_SIZE),
                jnp.bool(False)
            )

            # If collision with player missile occurred. Only valid if death timer is still in alive state
            collision_with_player_missile = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_missile.x, state.player_missile.y), self.consts.PLAYER_MISSILE_SIZE, (new_x, rm.y), self.consts.RADAR_MORTAR_SIZE),
                jnp.bool(False))

            # If collision with entity missile occurred. Only valid if death timer is still in alive state
            rm_missile_collision_with_player = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_x, state.player_y), self.consts.PLAYER_SIZE, (rm.missile_x, rm.missile_y), self.consts.ENTITY_MISSILE_SIZE),
                jnp.bool(False)
            )

            # Is still alive if was already alive and no collision occurred
            new_is_alive = jnp.logical_and(rm.is_alive, jnp.logical_and(jnp.logical_not(collision_with_player_missile), jnp.logical_not(collision_with_player)))

            # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
            new_death_timer = jnp.where(new_is_alive, self.consts.ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
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

        @jax.jit
        def byte_bat_step(state: LaserGatesState) -> tuple[ByteBatState, CollisionPropertiesState]:
            bb = state.entities.byte_bat_state

            # x distance to player and flags
            distance = jnp.abs(bb.x - state.player_x)
            near_player = distance < self.consts.BYTE_BAT_PLAYER_DIST_TRIGGER
            player_is_in_bottom_half = state.player_y > (
                        self.consts.PLAYER_BOUNDS[1][0] + self.consts.PLAYER_BOUNDS[1][1]) // 2

            # Calculate borders and corridor
            upper_base = self.consts.BYTE_BAT_UPPER_BORDER_Y
            bottom_base = self.consts.BYTE_BAT_BOTTOM_BORDER_Y
            corridor_h = bottom_base - upper_base

            # maximum allowed shrinking on one side
            max_single_side_shrink = jnp.maximum(0, corridor_h - 1)
            shrink_amount = jnp.minimum(self.consts.BYTE_BAT_SUBTRACT_FROM_BORDER, max_single_side_shrink)

            shrink_top = jnp.where(jnp.logical_and(near_player, player_is_in_bottom_half), shrink_amount, 0)
            shrink_bottom = jnp.where(jnp.logical_and(near_player, jnp.logical_not(player_is_in_bottom_half)),
                                      shrink_amount, 0)

            upper_border = upper_base + shrink_top
            bottom_border = bottom_base - shrink_bottom

            # Border hit - only if moving in the respective direction
            hit_top = jnp.logical_and(bb.direction_is_up, bb.y <= upper_border)
            hit_bottom = jnp.logical_and(jnp.logical_not(bb.direction_is_up), bb.y >= bottom_border)

            y_border_hit = jnp.logical_and(bb.is_alive, jnp.logical_or(hit_top, hit_bottom))

            # Only update flag if border is hit
            new_direction_is_left = jnp.where(
                y_border_hit,
                state.player_x + self.consts.PLAYER_SIZE[0] < bb.x,
                bb.direction_is_left
            )

            new_direction_is_up = jnp.where(
                y_border_hit,
                jnp.logical_not(bb.direction_is_up),
                bb.direction_is_up
            )

            # Update positions
            moved_x = jnp.where(new_direction_is_left, bb.x - self.consts.BYTE_BAT_X_SPEED,
                                bb.x + self.consts.BYTE_BAT_X_SPEED)
            # Freeze x if player is at right border
            moved_x = jnp.where(state.player_x == self.consts.PLAYER_BOUNDS[0][1], bb.x, moved_x)

            moved_y = jnp.where(new_direction_is_up, bb.y - self.consts.BYTE_BAT_Y_SPEED,
                                bb.y + self.consts.BYTE_BAT_Y_SPEED)

            new_x = jnp.where(bb.is_alive, moved_x, bb.x)
            new_y = jnp.where(bb.is_alive, moved_y, bb.y)

            # ----- Collision detection -----

            # If collision with player occurred. Only valid if death timer is still in alive state
            collision_with_player = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_x, state.player_y), self.consts.PLAYER_SIZE, (new_x, new_y), self.consts.BYTE_BAT_SIZE),
                jnp.bool(False)
            )

            # If collision with player missile occurred. Only valid if death timer is still in alive state
            collision_with_player_missile = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_missile.x, state.player_missile.y), self.consts.PLAYER_MISSILE_SIZE, (new_x, new_y), self.consts.BYTE_BAT_SIZE),
                jnp.bool(False))

            # Is still alive if was already alive and no collision occurred
            new_is_alive = jnp.logical_and(bb.is_alive, jnp.logical_and(jnp.logical_not(collision_with_player_missile), jnp.logical_not(collision_with_player)))

            # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
            new_death_timer = jnp.where(new_is_alive, self.consts.ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
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

        @jax.jit
        def rock_muncher_step(state: LaserGatesState) -> tuple[RockMuncherState, CollisionPropertiesState]:
            rm = state.entities.rock_muncher_state

            # If one of the y borders are hit
            y_border_hit = jnp.logical_or(rm.y <= self.consts.ROCK_MUNCHER_UPPER_BORDER_Y, rm.y >= self.consts.ROCK_MUNCHER_BOTTOM_BORDER_Y)
            # If player is left of the byte bat, update only if hitting border
            new_direction_is_left = jnp.where(y_border_hit, state.player_x + self.consts.PLAYER_SIZE[0] < rm.x, rm.direction_is_left)
            # Invert y direction if one of the two y borders is hit
            new_direction_is_up = jnp.where(y_border_hit, jnp.logical_not(rm.direction_is_up), rm.direction_is_up)

            # Update positions
            new_x = jnp.where(new_direction_is_left, rm.x - self.consts.BYTE_BAT_X_SPEED, rm.x + self.consts.BYTE_BAT_X_SPEED) # Move left or right
            new_x = jnp.where(jnp.logical_or(state.player_x == self.consts.PLAYER_BOUNDS[0][1], jnp.logical_not(rm.is_alive)), rm.x, new_x) # Do not move in x direction if player speeds up scroll speed (is at right player bound) or is not alive (death sprite active)
            new_y = jnp.where(new_direction_is_up, rm.y - self.consts.BYTE_BAT_Y_SPEED, rm.y + self.consts.BYTE_BAT_Y_SPEED) # Move up or down
            new_y = jnp.where(jnp.logical_not(rm.is_alive), rm.y, new_y) # Do not move if not alive

            # Missile
            spawn_trigger = jnp.logical_and(rm.is_alive, (state.step_counter % self.consts.ROCK_MUNCHER_MISSILE_SPAWN_EVERY) == 0)

            # Spawn
            new_missile_x = jnp.where(jnp.logical_and(rm.is_alive, spawn_trigger), rm.x, rm.missile_x).astype(rm.missile_x.dtype)
            new_missile_y = jnp.where(jnp.logical_and(rm.is_alive, spawn_trigger), rm.y + 6, rm.missile_y).astype(rm.missile_y.dtype)

            # Move
            new_missile_x = new_missile_x - self.consts.ROCK_MUNCHER_MISSILE_SPEED

            # Kill
            kill = jnp.logical_or(new_missile_x < 0 - self.consts.ENTITY_MISSILE_SIZE[0], jnp.bool(False))
            new_missile_x = jnp.where(kill, 0, new_missile_x)
            new_missile_y = jnp.where(kill, 0, new_missile_y)

            # ----- Collision detection -----

            # If collision with player occurred. Only valid if death timer is still in alive state
            collision_with_player = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_x, state.player_y), self.consts.PLAYER_SIZE, (new_x, new_y), self.consts.ROCK_MUNCHER_SIZE),
                jnp.bool(False)
            )

            # If collision with player missile occurred. Only valid if death timer is still in alive state
            collision_with_player_missile = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_missile.x, state.player_missile.y), self.consts.PLAYER_MISSILE_SIZE, (new_x, new_y), self.consts.ROCK_MUNCHER_SIZE),
                jnp.bool(False))

            # If collision with entity missile occurred. Only valid if death timer is still in alive state
            rm_missile_collision_with_player = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_x, state.player_y), self.consts.PLAYER_SIZE, (rm.missile_x, rm.missile_y), self.consts.ENTITY_MISSILE_SIZE),
                jnp.bool(False)
            )

            # Is still alive if was already alive and no collision occurred
            new_is_alive = jnp.logical_and(rm.is_alive, jnp.logical_and(jnp.logical_not(collision_with_player_missile), jnp.logical_not(collision_with_player)))

            # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
            new_death_timer = jnp.where(new_is_alive, self.consts.ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
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

        @jax.jit
        def homing_missile_step(state: LaserGatesState) -> tuple[HomingMissileState, CollisionPropertiesState]:
            hm = state.entities.homing_missile_state

            # Track player if in range or already tracking
            player_is_in_y_range = jnp.abs(state.player_y - hm.y) < self.consts.HOMING_MISSILE_PLAYER_TRACKING_RANGE
            new_is_tracking_player = jnp.logical_or(hm.is_tracking_player, player_is_in_y_range)

            player_is_below_missile = state.player_y - self.consts.HOMING_MISSILE_Y_PLAYER_OFFSET > hm.y

            # Update position
            new_x = jnp.where(hm.is_alive, hm.x - self.consts.HOMING_MISSILE_X_SPEED, hm.x)
            new_y = jnp.where(jnp.logical_and(hm.is_alive, jnp.logical_and(new_is_tracking_player, jnp.logical_not(jnp.abs(state.player_y - self.consts.HOMING_MISSILE_Y_PLAYER_OFFSET - hm.y) <= self.consts.HOMING_MISSILE_Y_SPEED))), jnp.where(
                player_is_below_missile,
                hm.y + self.consts.HOMING_MISSILE_Y_SPEED,
                hm.y - self.consts.HOMING_MISSILE_Y_SPEED
            ), hm.y)
            # Clip y position to bounds
            new_y = jnp.clip(new_y, self.consts.HOMING_MISSILE_Y_BOUNDS[0], self.consts.HOMING_MISSILE_Y_BOUNDS[1])

            # ----- Collision detection -----

            # If collision with player occurred. Only valid if death timer is still in alive state
            collision_with_player = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_x, state.player_y), self.consts.PLAYER_SIZE, (new_x, new_y), self.consts.HOMING_MISSILE_SIZE),
                jnp.bool(False)
            )

            # If collision with player missile occurred. Only valid if death timer is still in alive state
            collision_with_player_missile = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_missile.x, state.player_missile.y), self.consts.PLAYER_MISSILE_SIZE, (new_x, new_y), self.consts.HOMING_MISSILE_SIZE),
                jnp.bool(False))

            # Is still alive if was already alive and no collision occurred
            new_is_alive = jnp.logical_and(hm.is_alive, jnp.logical_and(jnp.logical_not(collision_with_player_missile), jnp.logical_not(collision_with_player)))

            # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
            new_death_timer = jnp.where(new_is_alive, self.consts.ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
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

        @jax.jit
        def forcefield_step(state: LaserGatesState) -> tuple[ForceFieldState, CollisionPropertiesState]:
            ff = state.entities.forcefield_state

            is_flexing, is_fixed = ff.is_flexing, ff.is_fixed
            is_flashing = jnp.logical_not(jnp.logical_or(is_flexing, is_fixed))
            number_of_forcefields = ff.num_of_forcefields
            new_x0, new_x1, new_x2, new_x3, new_x4, new_x5 = ff.x0, ff.x1, ff.x2, ff.x3, ff.x4, ff.x5
            new_y0, new_y1, new_y2, new_y3, new_y4, new_y5 = ff.y0, ff.y1, ff.y2, ff.y3, ff.y4, ff.y5
            scroll_speed = state.scroll_speed

            # Flashing --------------
            new_flash_on = jnp.where(jnp.logical_and(state.step_counter % self.consts.FORCEFIELD_FLASHING_SPEED == 0, is_flashing), jnp.logical_not(ff.flash_on), ff.flash_on)
            is_flashing_and_alive = jnp.logical_and(is_flashing, ff.is_alive)

            new_x0 = jnp.where(is_flashing_and_alive, new_x0 - scroll_speed, new_x0) # First forcefield upper
            new_x1 = jnp.where(is_flashing_and_alive, new_x0, new_x1) # First forcefield lower

            new_x2 = jnp.where(jnp.logical_and(is_flashing_and_alive, number_of_forcefields > 1), new_x0 + self.consts.FORCEFIELD_FLASHING_SPACING, new_x2)
            new_x3 = jnp.where(jnp.logical_and(is_flashing_and_alive, number_of_forcefields > 1), new_x2, new_x3)

            new_x4 = jnp.where(jnp.logical_and(is_flashing_and_alive, number_of_forcefields > 2), new_x0 + 2 * self.consts.FORCEFIELD_FLASHING_SPACING, new_x4)
            new_x5 = jnp.where(jnp.logical_and(is_flashing_and_alive, number_of_forcefields > 2), new_x4, new_x5)
            # There is no need for setting the y position, since it remains unchanged. We use the default y positions set in initialize_forcefield

            # Flexing --------------
            distance = new_y1 - (new_y0 + self.consts.FORCEFIELD_SIZE[1])
            new_flex_upper_direction_is_up = jnp.where(distance <= self.consts.FORCEFIELD_FLEXING_MINIMUM_DISTANCE, jnp.bool(True), jnp.where(distance >= self.consts.FORCEFIELD_FLEXING_MAXIMUM_DISTANCE, jnp.bool(False), ff.flex_upper_direction_is_up))
            is_flexing_and_alive = jnp.logical_and(is_flexing, ff.is_alive)

            new_x0 = jnp.where(is_flexing_and_alive, new_x0 - scroll_speed, new_x0)
            new_y0 = jnp.where(is_flexing_and_alive, jnp.where(new_flex_upper_direction_is_up, new_y0 - self.consts.FORCEFIELD_FLEXING_SPEED, new_y0 + self.consts.FORCEFIELD_FLEXING_SPEED), new_y0) # First forcefield upper
            new_x1 = jnp.where(is_flexing_and_alive, new_x0, new_x1)
            new_y1 = jnp.where(is_flexing_and_alive, jnp.where(new_flex_upper_direction_is_up, new_y1 + self.consts.FORCEFIELD_FLEXING_SPEED, new_y1 - self.consts.FORCEFIELD_FLEXING_SPEED), new_y1) # First forcefield lower

            new_x2 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 1), new_x0 + self.consts.FORCEFIELD_FLEXING_SPACING, new_x2)
            new_y2 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 1), jnp.where(new_flex_upper_direction_is_up, new_y2 - self.consts.FORCEFIELD_FLEXING_SPEED, new_y2 + self.consts.FORCEFIELD_FLEXING_SPEED), new_y2) # Second forcefield upper
            new_x3 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 1), new_x0 + self.consts.FORCEFIELD_FLEXING_SPACING, new_x3)
            new_y3 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 1), jnp.where(new_flex_upper_direction_is_up, new_y3 + self.consts.FORCEFIELD_FLEXING_SPEED, new_y3 - self.consts.FORCEFIELD_FLEXING_SPEED), new_y3) # Second forcefield lower

            new_x4 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 2), new_x0 + 2 * self.consts.FORCEFIELD_FLEXING_SPACING, new_x4)
            new_y4 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 2), jnp.where(new_flex_upper_direction_is_up, new_y4 - self.consts.FORCEFIELD_FLEXING_SPEED, new_y4 + self.consts.FORCEFIELD_FLEXING_SPEED), new_y4) # Third forcefield upper
            new_x5 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 2), new_x0 + 2 * self.consts.FORCEFIELD_FLEXING_SPACING, new_x5)
            new_y5 = jnp.where(jnp.logical_and(is_flexing_and_alive, number_of_forcefields > 2), jnp.where(new_flex_upper_direction_is_up, new_y5 + self.consts.FORCEFIELD_FLEXING_SPEED, new_y5 - self.consts.FORCEFIELD_FLEXING_SPEED), new_y5) # Third forcefield lower

            # Fixed --------------
            new_fixed_upper_direction_is_up = jnp.where(new_y0 < self.consts.FORCEFIELD_FIXED_UPPER_BOUND, jnp.bool(False), jnp.where(new_y0 > self.consts.FORCEFIELD_FIXED_LOWER_BOUND, jnp.bool(True), ff.fixed_upper_direction_is_up))
            is_fixed_and_alive = jnp.logical_and(is_fixed, ff.is_alive)

            new_x0 = jnp.where(is_fixed_and_alive, new_x0 - scroll_speed, new_x0)
            new_y0 = jnp.where(is_fixed_and_alive, jnp.where(new_fixed_upper_direction_is_up, new_y0 - self.consts.FORCEFIELD_FIXED_SPEED, new_y0 + self.consts.FORCEFIELD_FIXED_SPEED), new_y0) # First forcefield upper
            new_x1 = jnp.where(is_fixed_and_alive, new_x1 - scroll_speed, new_x1)
            new_y1 = jnp.where(is_fixed_and_alive, jnp.where(new_fixed_upper_direction_is_up, new_y1 - self.consts.FORCEFIELD_FIXED_SPEED, new_y1 + self.consts.FORCEFIELD_FIXED_SPEED), new_y1) # First forcefield lower

            new_x2 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 1), new_x0 + self.consts.FORCEFIELD_FIXED_SPACING, new_x2)
            new_y2 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 1), new_y0, new_y2) # Second forcefield upper
            new_x3 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 1), new_x0 + self.consts.FORCEFIELD_FIXED_SPACING, new_x3)
            new_y3 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 1), new_y1, new_y3) # Second forcefield lower

            new_x4 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 2), new_x0 + 2 * self.consts.FORCEFIELD_FIXED_SPACING, new_x4)
            new_y4 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 2), new_y0, new_y4) # Third forcefield upper
            new_x5 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 2), new_x0 + 2 * self.consts.FORCEFIELD_FIXED_SPACING, new_x5)
            new_y5 = jnp.where(jnp.logical_and(is_fixed_and_alive, number_of_forcefields > 2), new_y1, new_y5) # Third forcefield lower

            # Find rightmost x
            all_x_values = jnp.array([new_x0, new_x1, new_x2, new_x3, new_x4, new_x5])
            rightmost_x = jnp.max(jnp.where(all_x_values != self.consts.WIDTH, all_x_values, -jnp.inf)) # Ignore x coordinates that are at the spawn/dead point

            # ----- Collision detection -----

            allow_check_collision_flashing = jnp.logical_or(jnp.logical_not(is_flashing), jnp.logical_and(is_flashing, new_flash_on))

            x_positions = jnp.array([new_x0, new_x1, new_x2, new_x3, new_x4, new_x5])
            y_positions = jnp.array([new_y0, new_y1, new_y2, new_y3, new_y4, new_y5])
            no_offsets = jnp.array([(0, 0)])
            normal_size = jnp.array([self.consts.FORCEFIELD_SIZE])
            wide_size = jnp.array([self.consts.FORCEFIELD_WIDE_SIZE])
            size = jnp.where(ff.is_wide, wide_size, normal_size)

            # If collision with player occurred. Only valid if death timer is still in alive state
            collision_with_player = jnp.where(
                jnp.logical_and(state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER, allow_check_collision_flashing),
                jnp.any(self.any_collision_for_group(jnp.array((state.player_x, state.player_y)), jnp.array(self.consts.PLAYER_SIZE), x_positions, y_positions, no_offsets, size)),
                jnp.bool(False)
            )

            collision_with_player_missile = jnp.where(
                jnp.logical_and(state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER, allow_check_collision_flashing),
                jnp.any(self.any_collision_for_group(jnp.array((state.player_missile.x, state.player_missile.y)), jnp.array(self.consts.PLAYER_MISSILE_SIZE), x_positions, y_positions, no_offsets, size)),
                jnp.bool(False)
            )

            # Is still alive if was already alive and no collision occurred
            new_is_alive = jnp.logical_and(ff.is_alive, jnp.logical_not(collision_with_player))

            # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
            new_death_timer = jnp.where(new_is_alive, self.consts.ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
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
                rightmost_x=rightmost_x.astype(ff.rightmost_x.dtype),
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
                score_to_add=jnp.array(400),
                death_timer=new_death_timer,
            )

        @jax.jit
        def densepack_step(state: LaserGatesState) -> tuple[DensepackState, CollisionPropertiesState]:
            dp = state.entities.dense_pack_state

            # base X coord for all segments (with scrolling)
            base_x = dp.x - state.scroll_speed
            # starting Y + vertical spacing
            y = dp.upmost_y
            height = self.consts.DENSEPACK_NORMAL_PART_SIZE[1]

            # world positions array (shape (n_parts,))
            group_xs = jnp.full((self.consts.DENSEPACK_NUMBER_OF_PARTS,), base_x, dtype=jnp.float32)
            group_ys = y + jnp.arange(self.consts.DENSEPACK_NUMBER_OF_PARTS, dtype=jnp.float32) * height

            # offsets lookup as before
            offset_lookup_normal = jnp.array([
                (self.consts.WIDTH, 0), (6, 0), (4, 0), (2, 0), (0, 0)
            ], dtype=jnp.float32)
            offset_lookup_wide = jnp.array([
                (self.consts.WIDTH, 0), (12, 0), (8, 0), (4, 0), (0, 0)
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
            @jax.jit
            def hit_by_player(gx, gy, offs, sz):
                seg_x, seg_y = gx + offs[0], gy + offs[1]
                return self.check_collision_single(
                    jnp.array((state.player_x, state.player_y), dtype=jnp.float32),
                    jnp.array(self.consts.PLAYER_SIZE, dtype=jnp.float32),
                    jnp.array((seg_x, seg_y), dtype=jnp.float32),
                    sz
                )

            player_hits_mask = jax.vmap(hit_by_player)(
                group_xs, group_ys, segment_offsets, segment_sizes
            )
            collision_with_player = jnp.any(player_hits_mask)

            # --- collision vs. missile ---
            @jax.jit
            def hit_by_missile(gx, gy, offs, sz):
                seg_x, seg_y = gx + offs[0], gy + offs[1]
                px = state.player_missile.x.astype(jnp.float32)
                py = state.player_missile.y.astype(jnp.float32)
                return self.check_collision_single(
                    jnp.array((px, py), dtype=jnp.float32),
                    jnp.array(self.consts.PLAYER_MISSILE_SIZE, dtype=jnp.float32),
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
            new_death_timer = jnp.where(new_is_alive, self.consts.ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
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
                score_to_add=jnp.array(400),
                death_timer=new_death_timer,
            )

        @jax.jit
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
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                jnp.any(self.any_collision_for_group(jnp.array((state.player_x, state.player_y)), jnp.array(self.consts.PLAYER_SIZE), x_positions, y_positions, no_offsets, sizes)),
                jnp.bool(False)
            )
            collision_player_detonator = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single(jnp.array((state.player_x, state.player_y)), jnp.array(self.consts.PLAYER_SIZE), (base_x, y), self.consts.DETONATOR_SIZE),
                jnp.bool(False)
            )
            collision_with_player = jnp.logical_or(collision_with_player, collision_player_detonator)

            collision_player_missile_pin = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                jnp.any(self.any_collision_for_group(jnp.array((state.player_missile.x, state.player_missile.y)), jnp.array(self.consts.PLAYER_MISSILE_SIZE), x_positions, y_positions, no_offsets, sizes)),
                jnp.bool(False)
            )
            collision_player_missile_detonator = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_missile.x, state.player_missile.y), self.consts.PLAYER_MISSILE_SIZE, (base_x - 2, y), self.consts.DETONATOR_SIZE), # Subtract two to account for missile hitbox
                jnp.bool(False)
            )

            # If is collision with pin or not. We need this to kill the missile but not the detonator at non-pin collision
            new_collision_is_pin = jnp.bool(False)
            new_collision_is_pin = jnp.where(collision_player_missile_pin, jnp.bool(True), new_collision_is_pin)

            # Is still alive if was already alive and no collision occurred
            new_is_alive = jnp.logical_and(dn.is_alive, jnp.logical_and(jnp.logical_not(collision_player_missile_pin), jnp.logical_not(collision_with_player)))

            # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
            new_death_timer = jnp.where(new_is_alive, self.consts.ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
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

        @jax.jit
        def energy_pod_step(state: LaserGatesState) -> tuple[EnergyPodState, CollisionPropertiesState]:
            ep = state.entities.energy_pod_state

            new_x = ep.x - state.scroll_speed
            y = ep.y
            animation_timer = ep.animation_timer
            new_animation_timer = jnp.where(animation_timer > self.consts.ENERGY_POD_ANIMATION_SPEED, 0, animation_timer + 1)

            # ----- Collision detection -----

            # If collision with player occurred. Only valid if death timer is still in alive state
            collision_with_player = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_x, state.player_y), self.consts.PLAYER_SIZE, (new_x, y), self.consts.BYTE_BAT_SIZE),
                jnp.bool(False)
            )

            # If collision with player missile occurred. Only valid if death timer is still in alive state
            collision_with_player_missile = jnp.where(
                state.entities.collision_properties_state.death_timer == self.consts.ENTITY_DEATH_ANIMATION_TIMER,
                self.check_collision_single((state.player_missile.x, state.player_missile.y), self.consts.PLAYER_MISSILE_SIZE, (new_x, y), self.consts.BYTE_BAT_SIZE),
                jnp.bool(False))

            # Is still alive if was already alive and no collision occurred
            new_is_alive = jnp.logical_and(ep.is_alive, jnp.logical_and(jnp.logical_not(collision_with_player_missile), jnp.logical_not(collision_with_player)))

            # Death timer updates - set alive if is alive, decrement if death animation, deactivate completely if player collision (no animation)
            new_death_timer = jnp.where(new_is_alive, self.consts.ENTITY_DEATH_ANIMATION_TIMER, state.entities.collision_properties_state.death_timer)
            new_death_timer = jnp.where(jnp.logical_not(new_is_alive), jnp.maximum(new_death_timer - 1, 0), new_death_timer)
            new_death_timer = jnp.where(collision_with_player, -1, new_death_timer)

            # Update is_in_current_event for player missile collision
            new_is_in_current_event = jnp.where(collision_with_player_missile, ep.is_alive, ep.is_in_current_event)
            new_is_in_current_event = jnp.where(new_death_timer == 0, jnp.bool(False), new_is_in_current_event)

            # Update is_in_current_event for player collision
            new_is_in_current_event = jnp.where(collision_with_player, jnp.bool(True), new_is_in_current_event)

            return state.entities.energy_pod_state._replace(
                is_in_current_event=jnp.logical_and(new_is_in_current_event, new_x > 0),
                is_alive=new_is_alive,
                x=new_x.astype(jnp.float32),
                animation_timer=new_animation_timer.astype(jnp.int32),
            ), state.entities.collision_properties_state._replace(
                collision_with_player=collision_with_player,
                collision_with_player_missile=collision_with_player_missile,
                is_big_collision=jnp.bool(False),
                is_energy_pod=jnp.bool(True),
                is_detonator=jnp.bool(False),
                is_ff_or_dp=jnp.bool(False),
                score_to_add=jnp.array(0),
                death_timer=new_death_timer,
            )

        @partial(jax.jit, static_argnums=(0,))
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


    @partial(jax.jit, static_argnums=(0,))
    def player_step(
            self, state: LaserGatesState, action: JAXAtariAction
    ) -> tuple[chex.Array, chex.Array, chex.Array]:

        action = jnp.array(action)

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
        delta_x = jnp.where(left, -self.consts.PLAYER_VELOCITY_X, jnp.where(right, self.consts.PLAYER_VELOCITY_X, 0))
        player_x = jnp.clip(state.player_x + delta_x, self.consts.PLAYER_BOUNDS[0][0], self.consts.PLAYER_BOUNDS[0][1])

        # Move y
        delta_y = jnp.where(up, -self.consts.PLAYER_VELOCITY_Y, jnp.where(down, self.consts.PLAYER_VELOCITY_Y, 0))
        player_y = jnp.clip(state.player_y + delta_y, self.consts.PLAYER_BOUNDS[1][0], self.consts.PLAYER_BOUNDS[1][1])

        # Player facing direction
        new_player_facing_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_facing_direction))

        no_x_input = jnp.logical_and(
            jnp.logical_not(left), jnp.logical_not(right)
            )

        # SCROLL LEFT
        player_x = jnp.where(no_x_input, player_x - self.consts.SCROLL_SPEED, player_x)

        return player_x, player_y, new_player_facing_direction

    @partial(jax.jit, static_argnums=(0,))
    def player_missile_step(
            self, state: LaserGatesState, action: JAXAtariAction
    ) -> PlayerMissileState:

        action = jnp.array(action)

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
            state.player_missile.x < 0 - self.consts.PLAYER_MISSILE_SIZE[0],
            state.player_missile.x > self.consts.WIDTH
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
            new_velocity * self.consts.PLAYER_MISSILE_VELOCITY_MULTIPLIER,
            new_velocity
        ) # Multiply velocity by given constant

        # Spawn missile
        spawn = jnp.logical_and(jnp.logical_not(is_alive), fire)
        new_x = jnp.where(spawn, jnp.where(
            state.player_facing_direction > 0,
            state.player_x + self.consts.PLAYER_SIZE[0],
            state.player_x - 2 * self.consts.PLAYER_SIZE[0] - 1
        ), new_x)
        new_y = jnp.where(spawn, state.player_y + 4, new_y)
        new_direction = jnp.where(spawn, state.player_facing_direction, new_direction)
        new_velocity = jnp.where(spawn, self.consts.PLAYER_MISSILE_INITIAL_VELOCITY, new_velocity)

        return PlayerMissileState(x=new_x, y=new_y, direction=new_direction, velocity=new_velocity)

    @partial(jax.jit, static_argnums=(0,))
    def check_collision_single(self, pos1, size1, pos2, size2):
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


    @partial(jax.jit, static_argnums=(0,))
    def any_collision_for_group(self,
                                player_pos: jnp.ndarray,
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
            self.check_collision_single,
            in_axes=(None, None, 0, 0),
            out_axes=0)

        @jax.jit
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

    @partial(jax.jit, static_argnums=(0,))
    def check_player_and_player_missile_collision_bounds(
            self,
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
        player_size = jnp.array(self.consts.PLAYER_SIZE, dtype=jnp.int32)

        player_missile_pos = jnp.array((state.player_missile.x, state.player_missile.y), dtype=jnp.int32)
        player_missile_size = jnp.array(self.consts.PLAYER_MISSILE_SIZE, dtype=jnp.int32)

        # Check collisions for both groups
        upper_collisions = self.any_collision_for_group(
            player_pos, player_size, upper_xs, upper_ys,
            segment_offsets=upper_offsets,
            segment_sizes=upper_sizes
        )
        lower_collisions = self.any_collision_for_group(
            player_pos, player_size, lower_xs, lower_ys,
            segment_offsets=lower_offsets,
            segment_sizes=lower_sizes
        )
        upper_missile_collisions = self.any_collision_for_group(
            player_missile_pos, player_missile_size, upper_xs, upper_ys,
            segment_offsets=upper_offsets,
            segment_sizes=upper_sizes
        )
        lower_missile_collisions = self.any_collision_for_group(
            player_missile_pos, player_missile_size, lower_xs, lower_ys,
            segment_offsets=lower_offsets,
            segment_sizes=lower_sizes
        )

        # Include normal bound player
        upper_player_collision = jnp.logical_or(jnp.any(upper_collisions), state.player_y <= self.consts.PLAYER_BOUNDS[1][0])
        lower_player_collision = jnp.logical_or(jnp.any(lower_collisions), state.player_y >= self.consts.PLAYER_BOUNDS[1][1])

        # Include normal bound player missile
        player_missile_collision = jnp.logical_or(jnp.any(upper_missile_collisions), jnp.any(lower_missile_collisions))

        return upper_player_collision, lower_player_collision, player_missile_collision


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    @staticmethod
    @jax.jit
    def _f32(x):
        return jnp.asarray(x, jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _make_ep(self, x, y, w, h, active_bool):
        """EntityPosition with float32 und active ∈ {0.0, 1.0}."""
        active = jnp.where(active_bool, jnp.float32(1.0), jnp.float32(0.0))
        return EntityPosition(
            x=self._f32(x),
            y=self._f32(y),
            width=self._f32(w),
            height=self._f32(h),
            active=active,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _masked_ep(self, x, y, w, h, active_bool):
        """Like _make_ep, but null values if inactive."""
        ep = self._make_ep(x, y, w, h, active_bool)
        ax = ep.active
        return EntityPosition(
            x=ep.x * ax, y=ep.y * ax, width=ep.width * ax, height=ep.height * ax, active=ax
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: LaserGatesState) -> LaserGatesObservation:
        c = self.consts

        # --- Player ---
        player = self._masked_ep(
            state.player_x, state.player_y,
            c.PLAYER_SIZE[0], c.PLAYER_SIZE[1],
            jnp.bool_(True)
        )

        # --- Player missile ---
        pm_alive = state.player_missile.direction != 0
        player_missile = self._masked_ep(
            state.player_missile.x, state.player_missile.y,
            c.PLAYER_MISSILE_SIZE[0], c.PLAYER_MISSILE_SIZE[1],
            pm_alive
        )

        # --- Radar mortar (+ missile) ---
        rm = state.entities.radar_mortar_state
        rm_active = jnp.logical_and(rm.is_in_current_event, rm.is_alive)
        radar_mortar = self._masked_ep(
            rm.x, rm.y, c.RADAR_MORTAR_SIZE[0], c.RADAR_MORTAR_SIZE[1], rm_active
        )
        rm_missile_active = jnp.logical_and(
            rm.is_in_current_event,
            jnp.logical_and(rm.missile_x != 0, rm.missile_y != 0)
        )
        radar_mortar_missile = self._masked_ep(
            rm.missile_x, rm.missile_y, c.ENTITY_MISSILE_SIZE[0], c.ENTITY_MISSILE_SIZE[1], rm_missile_active
        )

        # --- Byte bat ---
        bb = state.entities.byte_bat_state
        bb_active = jnp.logical_and(bb.is_in_current_event, bb.is_alive)
        byte_bat = self._masked_ep(
            bb.x, bb.y, c.BYTE_BAT_SIZE[0], c.BYTE_BAT_SIZE[1], bb_active
        )

        # --- Rock muncher (+ missile) ---
        rmu = state.entities.rock_muncher_state
        rmu_active = jnp.logical_and(rmu.is_in_current_event, rmu.is_alive)
        rock_muncher = self._masked_ep(
            rmu.x, rmu.y, c.ROCK_MUNCHER_SIZE[0], c.ROCK_MUNCHER_SIZE[1], rmu_active
        )
        rmu_missile_active = jnp.logical_and(
            rmu.is_in_current_event,
            jnp.logical_and(rmu.missile_x != 0, rmu.missile_y != 0)
        )
        rock_muncher_missile = self._masked_ep(
            rmu.missile_x, rmu.missile_y, c.ENTITY_MISSILE_SIZE[0], c.ENTITY_MISSILE_SIZE[1], rmu_missile_active
        )

        # --- Homing missile ---
        hm = state.entities.homing_missile_state
        hm_active = jnp.logical_and(hm.is_in_current_event, hm.is_alive)
        homing_missile = self._masked_ep(
            hm.x, hm.y, c.HOMING_MISSILE_SIZE[0], c.HOMING_MISSILE_SIZE[1], hm_active
        )

        # --- Forcefields (up to 3 pairs (upper and lower, so 6 in total)) ---
        ff = state.entities.forcefield_state
        ff_base_active = jnp.logical_and(ff.is_in_current_event, ff.is_alive)
        is_flashing = jnp.logical_and(jnp.logical_not(ff.is_flexing), jnp.logical_not(ff.is_fixed))
        ff_visible = jnp.logical_or(jnp.logical_not(is_flashing), ff.flash_on)
        ff_col_active = lambda idx_lt_num: jnp.logical_and(ff_base_active, jnp.logical_and(ff_visible, idx_lt_num))
        ff_w = jnp.where(ff.is_wide, c.FORCEFIELD_WIDE_SIZE[0], c.FORCEFIELD_SIZE[0])
        ff_h = c.FORCEFIELD_SIZE[1]

        # column 0 (x0/y0 upper, x1/y1 lower)
        ff0_u = self._masked_ep(ff.x0, ff.y0, ff_w, ff_h, ff_col_active(ff.num_of_forcefields > 0))
        ff0_l = self._masked_ep(ff.x1, ff.y1, ff_w, ff_h, ff_col_active(ff.num_of_forcefields > 0))

        # column 1 (x2/y2 upper, x3/y3 lower)
        ff1_u = self._masked_ep(ff.x2, ff.y2, ff_w, ff_h, ff_col_active(ff.num_of_forcefields > 1))
        ff1_l = self._masked_ep(ff.x3, ff.y3, ff_w, ff_h, ff_col_active(ff.num_of_forcefields > 1))

        # column 2 (x4/y4 upper, x5/y5 lower)
        ff2_u = self._masked_ep(ff.x4, ff.y4, ff_w, ff_h, ff_col_active(ff.num_of_forcefields > 2))
        ff2_l = self._masked_ep(ff.x5, ff.y5, ff_w, ff_h, ff_col_active(ff.num_of_forcefields > 2))

        # --- Densepack (not every single part of the densepack is returned here, since there are a lot) ---
        dp = state.entities.dense_pack_state
        dp_active = jnp.logical_and(dp.is_in_current_event, dp.is_alive)
        dp_w = jnp.where(dp.is_wide, c.DENSEPACK_WIDE_PART_SIZE[0], c.DENSEPACK_NORMAL_PART_SIZE[0])
        dp_h = dp.number_of_parts * c.DENSEPACK_NORMAL_PART_SIZE[1]
        densepack = self._masked_ep(dp.x, dp.upmost_y, dp_w, dp_h, dp_active)

        # --- Detonator ---
        dn = state.entities.detonator_state
        dn_active = jnp.logical_and(dn.is_in_current_event, dn.is_alive)
        detonator = self._masked_ep(dn.x, dn.y, c.DETONATOR_SIZE[0], c.DETONATOR_SIZE[1], dn_active)

        # --- Energy pod ---
        ep = state.entities.energy_pod_state
        ep_active = jnp.logical_and(ep.is_in_current_event, ep.is_alive)
        energy_pod = self._masked_ep(ep.x, ep.y, c.ENERGY_POD_SIZE[0], c.ENERGY_POD_SIZE[1], ep_active)

        # --- Mountains ---
        mw, mh = c.MOUNTAIN_SIZE

        def mountain_visible(x):
            # only active if bounding box cuts screen
            return jnp.logical_and(x + mw > 0, x < c.WIDTH)

        um0 = self._masked_ep(state.upper_mountains.x1, state.upper_mountains.y, mw, mh, mountain_visible(state.upper_mountains.x1))
        um1 = self._masked_ep(state.upper_mountains.x2, state.upper_mountains.y, mw, mh, mountain_visible(state.upper_mountains.x2))
        um2 = self._masked_ep(state.upper_mountains.x3, state.upper_mountains.y, mw, mh, mountain_visible(state.upper_mountains.x3))

        lm0 = self._masked_ep(state.lower_mountains.x1, state.lower_mountains.y, mw, mh, mountain_visible(state.lower_mountains.x1))
        lm1 = self._masked_ep(state.lower_mountains.x2, state.lower_mountains.y, mw, mh, mountain_visible(state.lower_mountains.x2))
        lm2 = self._masked_ep(state.lower_mountains.x3, state.lower_mountains.y, mw, mh, mountain_visible(state.lower_mountains.x3))

        return LaserGatesObservation(
            player=player,
            player_missile=player_missile,
            radar_mortar=radar_mortar,
            radar_mortar_missile=radar_mortar_missile,
            byte_bat=byte_bat,
            rock_muncher=rock_muncher,
            rock_muncher_missile=rock_muncher_missile,
            homing_missile=homing_missile,
            forcefield_0_upper=ff0_u,
            forcefield_0_lower=ff0_l,
            forcefield_1_upper=ff1_u,
            forcefield_1_lower=ff1_l,
            forcefield_2_upper=ff2_u,
            forcefield_2_lower=ff2_l,
            densepack=densepack,
            detonator=detonator,
            energy_pod=energy_pod,
            upper_mountain_0=um0,
            upper_mountain_1=um1,
            upper_mountain_2=um2,
            lower_mountain_0=lm0,
            lower_mountain_1=lm1,
            lower_mountain_2=lm2,
        )

    def _get_info(self, state: LaserGatesState, all_rewards: jnp.ndarray | None = None) -> LaserGatesInfo:
        return LaserGatesInfo(
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: LaserGatesState, state: LaserGatesState) -> jnp.ndarray:
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: LaserGatesState, state: LaserGatesState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: LaserGatesState) -> chex.Array:
        return (
                (state.shields <= 0)
                | (state.dtime <= 0)
                | (state.energy <= 0)
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8
        )

    def _scalar_box(self, low, high):
        # Scalar-Box (shape=()) for dict leaves, float32
        return spaces.Box(
            low=jnp.array(low, dtype=jnp.float32),
            high=jnp.array(high, dtype=jnp.float32),
            shape=(),
            dtype=jnp.float32,
        )

    def _entity_space(self) -> spaces.Dict:
        # Choose bounds generously:
        x_box = self._scalar_box(-self.consts.WIDTH, 4 * self.consts.WIDTH)
        y_box = self._scalar_box(-self.consts.HEIGHT, 2 * self.consts.HEIGHT)
        w_box = self._scalar_box(0.0, float(self.consts.WIDTH))
        h_box = self._scalar_box(0.0, float(self.consts.HEIGHT))
        a_box = self._scalar_box(0.0, 1.0)
        return spaces.Dict({
            "x": x_box, "y": y_box, "width": w_box, "height": h_box, "active": a_box
        })

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player": self._entity_space(),
            "player_missile": self._entity_space(),
            "radar_mortar": self._entity_space(),
            "radar_mortar_missile": self._entity_space(),
            "byte_bat": self._entity_space(),
            "rock_muncher": self._entity_space(),
            "rock_muncher_missile": self._entity_space(),
            "homing_missile": self._entity_space(),
            "forcefield_0_upper": self._entity_space(),
            "forcefield_0_lower": self._entity_space(),
            "forcefield_1_upper": self._entity_space(),
            "forcefield_1_lower": self._entity_space(),
            "forcefield_2_upper": self._entity_space(),
            "forcefield_2_lower": self._entity_space(),
            "densepack": self._entity_space(),
            "detonator": self._entity_space(),
            "energy_pod": self._entity_space(),
            "upper_mountain_0": self._entity_space(),
            "upper_mountain_1": self._entity_space(),
            "upper_mountain_2": self._entity_space(),
            "lower_mountain_0": self._entity_space(),
            "lower_mountain_1": self._entity_space(),
            "lower_mountain_2": self._entity_space(),
        })

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_array(self, obs: LaserGatesObservation) -> jnp.ndarray:
        def row(ep: EntityPosition):
            return jnp.array([ep.x, ep.y, ep.width, ep.height, ep.active], dtype=jnp.float32)

        rows = jnp.stack([
            row(obs.player),
            row(obs.player_missile),
            row(obs.radar_mortar),
            row(obs.radar_mortar_missile),
            row(obs.byte_bat),
            row(obs.rock_muncher),
            row(obs.rock_muncher_missile),
            row(obs.homing_missile),
            row(obs.forcefield_0_upper),
            row(obs.forcefield_0_lower),
            row(obs.forcefield_1_upper),
            row(obs.forcefield_1_lower),
            row(obs.forcefield_2_upper),
            row(obs.forcefield_2_lower),
            row(obs.densepack),
            row(obs.detonator),
            row(obs.energy_pod),
            row(obs.upper_mountain_0),
            row(obs.upper_mountain_1),
            row(obs.upper_mountain_2),
            row(obs.lower_mountain_0),
            row(obs.lower_mountain_1),
            row(obs.lower_mountain_2),
        ], axis=0)  # (num_slots, 5)
        return rows

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: LaserGatesObservation) -> jnp.ndarray:
        mat = self.obs_to_array(obs)  # (num_slots, 5)
        return jnp.reshape(mat, (self.obs_size,))  # float32-Vector

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key = jax.random.PRNGKey(42)) -> Tuple[LaserGatesObservation, LaserGatesState]:
        """Initialize game state"""

        initial_lower_mountains = MountainState(
            x1=jnp.array(self.consts.LOWER_MOUNTAINS_START_X).astype(jnp.int32),
            x2=jnp.array(self.consts.LOWER_MOUNTAINS_START_X + self.consts.MOUNTAIN_SIZE[0] + self.consts.MOUNTAINS_DISTANCE).astype(jnp.int32),
            x3=jnp.array(self.consts.LOWER_MOUNTAINS_START_X + 2 * self.consts.MOUNTAIN_SIZE[0] + 2 * self.consts.MOUNTAINS_DISTANCE).astype(jnp.int32),
            y=jnp.array(self.consts.LOWER_MOUNTAINS_Y).astype(jnp.int32)
        )

        initial_upper_mountains = MountainState(
            x1=jnp.array(self.consts.UPPER_MOUNTAINS_START_X).astype(jnp.int32),
            x2=jnp.array(self.consts.UPPER_MOUNTAINS_START_X + self.consts.MOUNTAIN_SIZE[0] + self.consts.MOUNTAINS_DISTANCE).astype(jnp.int32),
            x3=jnp.array(self.consts.UPPER_MOUNTAINS_START_X + 2 * self.consts.MOUNTAIN_SIZE[0] + 2 * self.consts.MOUNTAINS_DISTANCE).astype(jnp.int32),
            y=jnp.array(self.consts.UPPER_MOUNTAINS_Y).astype(jnp.int32)
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
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x0=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y0=jnp.array(-17).astype(jnp.float32),
                x1=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y1=jnp.array(56).astype(jnp.float32),
                x2=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y2=jnp.array(-17).astype(jnp.float32),
                x3=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y3=jnp.array(56).astype(jnp.float32),
                x4=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y4=jnp.array(-17).astype(jnp.float32),
                x5=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                y5=jnp.array(56).astype(jnp.float32),
                rightmost_x=jnp.array(self.consts.WIDTH, dtype=jnp.float32),
                num_of_forcefields=jnp.array(2),
                is_wide=False,
                is_flexing=False,
                is_fixed=False,
                flash_on=jnp.array(True),
                flex_upper_direction_is_up=jnp.array(True),
                fixed_upper_direction_is_up=jnp.array(True),
            ),
            dense_pack_state=DensepackState(
                is_in_current_event=jnp.bool(False),
                is_alive=jnp.bool(False),
                x=jnp.array(0).astype(jnp.float32),
                upmost_y=jnp.array(0).astype(jnp.float32),
                is_wide=jnp.bool(False),
                number_of_parts=jnp.array(0).astype(jnp.int32),
                broken_states=jnp.full(self.consts.DENSEPACK_NUMBER_OF_PARTS, 3, jnp.int32),
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
                y=jnp.array(0).astype(jnp.float32),
                animation_timer=jnp.array(0),
            ),
            collision_properties_state=CollisionPropertiesState(
                collision_with_player=jnp.bool(False),
                collision_with_player_missile=jnp.bool(False),
                is_big_collision=jnp.bool(False),
                is_energy_pod=jnp.bool(False),
                is_detonator=jnp.bool(False),
                is_ff_or_dp=jnp.bool(False),
                score_to_add=jnp.array(0),
                death_timer=jnp.array(self.consts.ENTITY_DEATH_ANIMATION_TIMER),
            )
        )

        """
        EXPLANATION BELOW
        """

        """
        initial_entities = initial_entities._replace(
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
            rightmost_x=jnp.array(0, dtype=jnp.float32),
            num_of_forcefields=jnp.array(0),
            is_wide=jnp.bool(False),
            is_flexing=jnp.bool(False),
            is_fixed=jnp.bool(False),
            flash_on=jnp.bool(False),
            flex_upper_direction_is_up=jnp.bool(False),
            fixed_upper_direction_is_up=jnp.bool(False),
            )
        )
        import time
        key = jax.random.PRNGKey(time.time_ns() % (2**31)) # Pseudo random number generator seed key, based on current system time.
        """


        """
        EXPLANATION:
        
        In the original game (and in our implementation), the first entity is always a
        flashing two-column forcefield. All subsequent spawns are deterministic as well, however only the first spawn is equivalent to the original.
        
        To ensure reproducibility, we use a constant RNG key by default. If you run two
        instances and don’t move, the spawns will match.
        
        If the function is called with a different RNG key, the spawn sequence after the first (forcefields) entity changes. 
        For non-reproducible (random) spawns, uncomment the snippet above to generate a fresh key and fresh first spawn each run.
        """

        reset_state = LaserGatesState(
            player_x=jnp.array(self.consts.PLAYER_START_X).astype(jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y).astype(jnp.int32),
            player_facing_direction=jnp.array(1, dtype=jnp.int32),
            player_missile=initial_player_missile,
            animation_timer=jnp.array(0).astype(jnp.int32),
            entities=initial_entities,
            lower_mountains=initial_lower_mountains,
            upper_mountains=initial_upper_mountains,
            scroll_speed=jnp.array(self.consts.SCROLL_SPEED).astype(jnp.float32),
            score=jnp.array(0).astype(jnp.int32), # Start with no initial score
            energy=jnp.array(self.consts.MAX_ENERGY).astype(jnp.int32), # As the manual says, energy is consumed at a regular pace. We use 5100 for the initial value and subtract one for every frame to match the timing of the real game. (It takes 85 seconds for the energy to run out. 85 * 60 (fps) = 5100
            shields=jnp.array(self.consts.MAX_SHIELDS).astype(jnp.int32), # As the manual says, the Dante Dart starts with 24 shield units
            dtime=jnp.array(self.consts.MAX_DTIME).astype(jnp.int32), # Same idea as energy.
            rng_key=key, # Pseudo random number generator seed key, based on current time and initial key used.
            step_counter=jnp.array(0),
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0, ))
    def step(
            self, state: LaserGatesState, action: Action
    ) -> Tuple[LaserGatesObservation, LaserGatesState, float, bool, LaserGatesInfo]:

        # -------- Move player --------
        new_player_x, new_player_y, new_player_facing_direction = self.player_step(state, action)
        player_animation_timer = state.animation_timer
        new_player_animation_timer = jnp.where(player_animation_timer != 0, player_animation_timer - 1, player_animation_timer)

        # -------- Move player missile --------
        new_player_missile_state = self.player_missile_step(state, action)

        # -------- Move entities --------
        new_entities = self.all_entities_step(state)
        new_entities = self.maybe_initialize_random_entity(new_entities, state)

        # -------- Move mountains --------
        new_lower_mountains_state = self.mountains_step(state.lower_mountains, state)
        new_upper_mountains_state = self.mountains_step(state.upper_mountains, state)

        # -------- Update scroll speed --------
        new_scroll_speed = jnp.where(state.player_x != self.consts.PLAYER_BOUNDS[0][1], self.consts.SCROLL_SPEED, self.consts.SCROLL_SPEED * self.consts.SCROLL_MULTIPLIER)

        # -------- Check bound and entity collisions --------
        upper_player_collision, lower_player_collision, player_missile_collision = self.check_player_and_player_missile_collision_bounds(state)
        collision_with_player = jnp.logical_and(jnp.logical_not(state.entities.collision_properties_state.collision_with_player), new_entities.collision_properties_state.collision_with_player) # Only allow flag to be True once per collision
        # Do not register as "bad" collision if collision with energy pod. Instead, save as own variable
        collision_with_energy_pod = jnp.where(new_entities.collision_properties_state.is_energy_pod, collision_with_player, jnp.bool(False))
        collision_with_player = jnp.where(new_entities.collision_properties_state.is_energy_pod, jnp.bool(False), collision_with_player)

        any_player_collision = jnp.logical_or(collision_with_player, jnp.logical_or(upper_player_collision, lower_player_collision))

        # -------- Update things that have to be updated at collision --------
        new_player_animation_timer = jnp.where(any_player_collision, 255, new_player_animation_timer)

        new_player_y = jnp.where(upper_player_collision, new_player_y + 4, new_player_y)
        new_player_y = jnp.where(lower_player_collision, new_player_y - 4, new_player_y)

        # -------- Kill missile --------
        kill_missile = jnp.logical_or(player_missile_collision, new_entities.collision_properties_state.collision_with_player_missile)
        new_player_missile_state = PlayerMissileState(
                                             x=jnp.where(kill_missile, jnp.array(0).astype(new_player_missile_state.x.dtype), new_player_missile_state.x).astype(jnp.int32),
                                             y=jnp.where(kill_missile, jnp.array(0), new_player_missile_state.y).astype(jnp.int32),
                                             direction=jnp.where(kill_missile, jnp.array(0).astype(new_player_missile_state.direction.dtype), new_player_missile_state.direction).astype(jnp.int32),
                                             velocity=jnp.where(kill_missile, jnp.array(0).astype(new_player_missile_state.velocity.dtype), new_player_missile_state.velocity).astype(jnp.int32),
                                             )
        # Register player missile collision with detonator pin for restoring dtime
        player_missile_collision_detonator_pin = jnp.logical_and(new_entities.collision_properties_state.collision_with_player_missile, new_entities.detonator_state.collision_is_pin)

        # -------- Update energy, score, shields and d-time --------

        # Drain energy every frame
        new_energy = state.energy - 1
        # Restore energy if player collided with energy pod
        new_energy = jnp.where(collision_with_energy_pod, self.consts.MAX_ENERGY, new_energy)


        # Forbid score change if densepack or forcefield is hit with missile
        allow_score_change = jnp.logical_and(new_entities.collision_properties_state.collision_with_player_missile, jnp.logical_not(state.entities.collision_properties_state.is_ff_or_dp))
        # Forbid score change if detonator is not hit with a missile at a pin
        allow_score_change = jnp.where(new_entities.collision_properties_state.is_detonator, new_entities.detonator_state.collision_is_pin, allow_score_change)
        # Allow change if forcefield or densepack is passed (rightmost x position (ff) or x position (dp) crossed the x=1 line)
        allow_score_change_dp_ff = jnp.logical_or(
            jnp.logical_and(new_entities.forcefield_state.rightmost_x <= 1, state.entities.forcefield_state.rightmost_x > 1),
            jnp.logical_and(new_entities.dense_pack_state.x <= 1, state.entities.dense_pack_state.x > 1))
        allow_score_change = jnp.where(state.entities.collision_properties_state.is_ff_or_dp, allow_score_change_dp_ff, allow_score_change)

        # Add score based on entity if allowed
        new_score = jnp.where(allow_score_change, state.score + new_entities.collision_properties_state.score_to_add, state.score)


        # Player loses 1 shield, if collision with bounds or mountains
        new_shields = jnp.where(jnp.logical_or(upper_player_collision, lower_player_collision), state.shields - 1, state.shields)
        # Player loses 6 or 1 shield, if collision with entity. Different entity types cause a big or small decrease. (See is_big_collision)
        new_shields = jnp.where(collision_with_player,
                                new_shields - jnp.where(new_entities.collision_properties_state.is_big_collision, 6, 1),
                                new_shields)
        # Clip to minimum and maximum value
        new_shields = jnp.clip(new_shields, 0, self.consts.MAX_SHIELDS)
        # Add 6 shield points every 10000 score points
        crossed_10k = jnp.array(state.score/10000).astype(jnp.int32) < jnp.array(new_score/10000).astype(jnp.int32)
        new_shields += jnp.where(crossed_10k,6,0)

        # Drain dtime every frame
        new_dtime = state.dtime - 1
        # Restore dtime if player shot detonator pin
        new_dtime = jnp.where(player_missile_collision_detonator_pin, self.consts.MAX_DTIME, new_dtime)

        # -------- New rng key --------
        new_rng_key, new_key = jax.random.split(state.rng_key)

        return_state = state._replace(
            player_x=new_player_x.astype(jnp.int32),
            player_y=new_player_y.astype(jnp.int32),
            player_facing_direction=new_player_facing_direction.astype(jnp.int32),
            player_missile=new_player_missile_state,
            animation_timer=new_player_animation_timer.astype(jnp.int32),
            entities=new_entities,
            lower_mountains=new_lower_mountains_state,
            upper_mountains=new_upper_mountains_state,
            scroll_speed=new_scroll_speed.astype(jnp.float32),
            score=new_score.astype(jnp.int32),
            energy=new_energy.astype(jnp.int32),
            shields=new_shields.astype(jnp.int32),
            dtime=new_dtime.astype(jnp.int32),
            rng_key=new_rng_key,
            step_counter=state.step_counter + 1
        )

        def get_reset_state() -> LaserGatesState:
            _, reset_state = self.reset()
            return reset_state

        # Reset if no shields, dtime or energy
        return_state = jax.lax.cond(
            jnp.logical_or(new_shields <= 0, jnp.logical_or(new_dtime <= 0, new_energy <= 0)),
            lambda _: get_reset_state(),
            lambda _: return_state,
            operand=None
        )

        obs = self._get_observation(return_state)
        all_rewards = self._get_all_rewards(state, return_state)
        info = self._get_info(return_state, all_rewards)

        return obs, return_state, 0.0, False, info

class LaserGatesRenderer(JAXGameRenderer):
    def __init__(self, consts: LaserGatesConstants = None):
        super().__init__()
        self.consts = consts or LaserGatesConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: LaserGatesState):
        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3))

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
        pfb_t = jnp.clip(255 * jnp.exp(-self.consts.PLAYING_FILED_BG_COLOR_FADE_SPEED * (255 - state.animation_timer)), 0, 255)
        pfb_t = pfb_t.astype(jnp.uint8)
        PLAYING_FIELD_COLOR = jnp.array((self.consts.PLAYING_FIELD_BG_COLLISION_COLOR[0], self.consts.PLAYING_FIELD_BG_COLLISION_COLOR[1], self.consts.PLAYING_FIELD_BG_COLLISION_COLOR[2], pfb_t))
        raster = jru.render_at(
            raster,
            0,
            19,
            recolor_sprite(SPRITE_PLAYING_FIELD_BG, PLAYING_FIELD_COLOR),
        )

        # -------- Render Entity Death Sprites --------

        # Death sprites
        death_sprite_index = get_death_sprite_index(state.entities.collision_properties_state.death_timer, self.consts.ENTITY_DEATH_ANIMATION_TIMER)
        death_sprite_upper_frame = SPRITE_LOWER_DEATH_SPRITES[death_sprite_index]
        death_sprite_lower_frame = SPRITE_UPPER_DEATH_SPRITES[death_sprite_index]

        # Radar mortar
        rm_state = state.entities.radar_mortar_state
        raster = jnp.where(jnp.logical_and(rm_state.is_in_current_event, jnp.logical_and(jnp.logical_not(rm_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                          # Case: in event but dead -> render death sprites
                          jnp.where(rm_state.y == self.consts.RADAR_MORTAR_SPAWN_UPPER_Y,
                                    jru.render_at( #upper
                                        raster,
                                        rm_state.x,
                                        rm_state.y + 15,
                                        death_sprite_upper_frame,
                                    ),
                                    jru.render_at( #lower
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
                           jru.render_at(
                               jru.render_at(
                                   raster,
                                   bb_state.x,
                                   bb_state.y + self.consts.ENTITY_DEATH_SPRITE_Y_OFFSET,
                                   death_sprite_upper_frame,
                               ),
                               bb_state.x,
                               bb_state.y - self.consts.ENTITY_DEATH_SPRITES_SIZE[1] + self.consts.ENTITY_DEATH_SPRITE_Y_OFFSET,
                               death_sprite_lower_frame,
                           ),
                           # Case: in event but dead and death animation over -> do not render
                           raster
                           )

        # Rock muncher
        rmu_state = state.entities.rock_muncher_state
        rmu_x = rmu_state.x
        rmu_y = rmu_state.y
        raster = jnp.where(jnp.logical_and(rmu_state.is_in_current_event, jnp.logical_and(jnp.logical_not(rmu_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                           jru.render_at(
                                jru.render_at(
                                    jru.render_at(
                                        jru.render_at(
                                            jru.render_at(
                                                raster,
                                                rmu_x,
                                                rmu_y + self.consts.ENTITY_DEATH_SPRITE_Y_OFFSET,
                                                death_sprite_upper_frame, # Upper death animation
                                            ),
                                            rmu_x,
                                            rmu_y - self.consts.ENTITY_DEATH_SPRITES_SIZE[1] + self.consts.ENTITY_DEATH_SPRITE_Y_OFFSET,
                                            death_sprite_lower_frame, # Lower death animation
                                        ),
                                        rmu_x,
                                        rmu_y + 2,
                                        SPRITE_DEATH_NUMBER_BG # Black background two layers behind the number
                                    ),
                                    rmu_x,
                                    rmu_y + 2,
                                    recolor_sprite(SPRITE_DEATH_NUMBER_BG, PLAYING_FIELD_COLOR), # Black background mimicing playing field bg for number
                                ),
                               rmu_x,
                               rmu_y + 2,
                               recolor_sprite(SPRITE_DEATH_NUMBER_325, jnp.array(self.consts.ENTITY_DEATH_SPRITES_NUMBER_COLOR)), # Number showing the score
                           ),
                           # Case: in event but dead and death animation over -> do not render
                           raster
                           )

        # Homing missile
        hm_state = state.entities.homing_missile_state
        hm_x = hm_state.x
        hm_y = hm_state.y
        raster = jnp.where(jnp.logical_and(hm_state.is_in_current_event, jnp.logical_and(jnp.logical_not(hm_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                           jru.render_at(
                                jru.render_at(
                                    jru.render_at(
                                        jru.render_at(
                                            jru.render_at(
                                                raster,
                                                hm_x,
                                                hm_y + self.consts.ENTITY_DEATH_SPRITE_Y_OFFSET,
                                                death_sprite_upper_frame, # Upper death animation
                                            ),
                                            hm_x,
                                            hm_y - self.consts.ENTITY_DEATH_SPRITES_SIZE[1] + self.consts.ENTITY_DEATH_SPRITE_Y_OFFSET,
                                            death_sprite_lower_frame, # Lower death animation
                                        ),
                                        hm_x,
                                        hm_y + 1,
                                        SPRITE_DEATH_NUMBER_BG # Black background two layers behind the number
                                    ),
                                    hm_x,
                                    hm_y + 1,
                                    recolor_sprite(SPRITE_DEATH_NUMBER_BG, PLAYING_FIELD_COLOR), # Black background mimicing playing field bg for number
                                ),
                                hm_x,
                                hm_y + 1,
                                recolor_sprite(SPRITE_DEATH_NUMBER_525, jnp.array(self.consts.ENTITY_DEATH_SPRITES_NUMBER_COLOR)), # Number showing the score
                            ),
                           # Case: in event but dead and death animation over -> do not render
                           raster
                           )

        # Detonator
        dn_state = state.entities.detonator_state
        raster = jnp.where(jnp.logical_and(dn_state.is_in_current_event, jnp.logical_and(jnp.logical_not(dn_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                           # Case: in event but dead -> render death sprites
                           jru.render_at( #lower
                                    raster,
                                    dn_state.x,
                                    dn_state.y + 12,
                                    death_sprite_upper_frame,
                                ),
                           raster
                           )

        # Energy pod
        ep_state = state.entities.energy_pod_state
        raster = jnp.where(jnp.logical_and(ep_state.is_in_current_event, jnp.logical_and(jnp.logical_not(ep_state.is_alive), state.entities.collision_properties_state.death_timer > 0)),
                           # Case: in event but dead -> render death sprites
                           jru.render_at( #lower
                                    raster,
                                    ep_state.x,
                                    ep_state.y - 40,
                                    death_sprite_lower_frame,
                                ),
                           raster
                           )


        # -------- Render Mountain Playing Field Background --------

        colored_playing_field_small_bg = recolor_sprite(SPRITE_PLAYING_FIELD_SMALL_BG, PLAYING_FIELD_COLOR)

        raster = jru.render_at(
                    jru.render_at(
                        jru.render_at(
                            jru.render_at(
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
        radar_mortar_frame = jru.get_sprite_frame(SPRITE_RADAR_MORTAR, state.step_counter)
        radar_mortar_frame = jnp.where(state.entities.radar_mortar_state.y == self.consts.RADAR_MORTAR_SPAWN_BOTTOM_Y, radar_mortar_frame, recolor_sprite(radar_mortar_frame, jnp.array(self.consts.RADAR_MORTAR_COLOR_GRAY)))

        raster = jnp.where(
            jnp.logical_and(rm_state.is_in_current_event, rm_state.is_alive),
                # Case: alive -> render normally
                jru.render_at(
                    raster,
                    rm_state.x,
                    rm_state.y,
                    radar_mortar_frame,
                    flip_vertical=rm_state.y == self.consts.RADAR_MORTAR_SPAWN_UPPER_Y,
                ),
            # Case: not in event -> do not render
            raster
        )

        # Render radar mortar missile
        should_render_rock_muncher_missile = jnp.logical_and(state.entities.radar_mortar_state.missile_x != 0, state.entities.radar_mortar_state.missile_y != 0)
        rock_muncher_missile_sprite = recolor_sprite(SPRITE_ENTITY_MISSILE, jnp.array(self.consts.RADAR_MORTAR_COLOR_BLUE))

        raster = jnp.where(
            jnp.logical_and(should_render_rock_muncher_missile, state.entities.radar_mortar_state.is_in_current_event),
               jru.render_at(
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
        byte_bat_frame = jru.get_sprite_frame(SPRITE_BYTE_BAT, state.step_counter)
        byte_bat_frame = recolor_sprite(byte_bat_frame, jnp.array(self.consts.BYTE_BAT_COLOR))

        raster = jnp.where(
            jnp.logical_and(bb_state.is_in_current_event, bb_state.is_alive),
                # Case: alive -> render normally
                jru.render_at(
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
        rock_muncher_frame = jru.get_sprite_frame(SPRITE_ROCK_MUNCHER, state.step_counter)

        raster = jnp.where(
            jnp.logical_and(rmu_state.is_in_current_event, rmu_state.is_alive),
                # Case: alive -> render normally
                jru.render_at(
                    raster,
                    rmu_state.x,
                    rmu_state.y,
                    rock_muncher_frame,
                ),
            # Case: not in event -> do not render
            raster
            )


        # Render rock muncher missile
        rock_muncher_missile_sprite = recolor_sprite(SPRITE_ENTITY_MISSILE, jnp.array(self.consts.ROCK_MUNCHER_MISSILE_COLOR))

        raster = jnp.where(jnp.logical_and(jnp.logical_and(rmu_state.missile_x != 0, rmu_state.missile_y != 0), rmu_state.is_in_current_event),
                           jru.render_at(
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
                           jru.render_at(
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
            W, H, C = sprite.shape

            # Column indices (x) and their corresponding y-positions
            xs = jnp.arange(W, dtype=jnp.int32)
            ys = xs + jnp.asarray(y_position, jnp.int32)
            ys = jnp.where(flipped, ys[::-1], ys)

            # Using hashes instead of PRNG key for better performance
            def _splitmix32(u32):
                x = u32.astype(jnp.uint32)
                x = (x + jnp.uint32(0x9E3779B9)) & jnp.uint32(0xFFFFFFFF)
                x = (x ^ (x >> 16)) * jnp.uint32(0x85EBCA6B) & jnp.uint32(0xFFFFFFFF)
                x = (x ^ (x >> 13)) * jnp.uint32(0xC2B2AE35) & jnp.uint32(0xFFFFFFFF)
                x = x ^ (x >> 16)
                return x

            seed = jnp.asarray(x_position, jnp.uint32)
            ux = (seed + xs.astype(jnp.uint32)) & jnp.uint32(0xFFFFFFFF)

            r = (_splitmix32(ux) >> 24).astype(jnp.int32)
            g = (_splitmix32(ux + jnp.uint32(0x9E37)) >> 24).astype(jnp.int32)
            b = (_splitmix32(ux + jnp.uint32(0x2C1B)) >> 24).astype(jnp.int32)

            rgb = jnp.stack([r, g, b], axis=1)  # (W, 3)
            # rgb: (W, 3)
            alpha = jnp.full((W, 1), 255, jnp.int32)  # always build
            col_colors4 = jnp.concatenate([rgb, alpha], axis=1)  # (W, 4)
            col_colors = col_colors4[:, :C]  # (W, C) with C∈{3,4}

            # Freeze logic (y positions where colors remain constant)
            lower = jnp.int32(32)
            upper = jnp.int32(80)

            # Anchor colors from column 32 and 80 (no clamping/mods)
            anchor_low = col_colors[lower]  # (C,)
            anchor_up = col_colors[upper]  # (C,)

            above_lower = ys >= lower
            within_band = jnp.logical_and(ys >= lower, ys <= upper)

            # For columns outside the band: if y >= 32 -> anchor 80, else anchor 32
            frozen_color = jnp.where(above_lower[:, None], anchor_up[None, :], anchor_low[None, :])  # (W, C)
            # Inside the band: dynamic column color
            final_cols = jnp.where(within_band[:, None], col_colors, frozen_color)  # (W, C)

            # Broadcast correctly to (H, W, C) and cast to sprite.dtype
            color_grid = jnp.broadcast_to(final_cols[:, None, :], (W, H, C)).astype(sprite.dtype)
            return color_grid

        # Despawn earlier
        move_left = self.consts.FORCEFIELD_SIZE[0]
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

        # now make a wide version of each, swapping H/W
        def resize_sprite_width_ff(sprite: jnp.ndarray, new_width: int) -> jnp.ndarray:
            H, W, C = sprite.shape
            # resize to (H, new_width, C), not (new_width, W, C)
            return jax.image.resize(sprite, (H, new_width, C), method='nearest')

        sprites_normal = all_sprites  # (6, H, W, C)
        sprites_wide = jax.vmap(
            lambda sprite: resize_sprite_width_ff(sprite, self.consts.FORCEFIELD_WIDE_SIZE[0])
        )(all_sprites)  # (6, H, wide, C)

        max_width = max(sprites_normal.shape[2], sprites_wide.shape[2])

        def pad_to_width_ff(sprites, width):
            # sprites.shape == (6, H, W0, C)
            H, W0, C = sprites.shape[1:]
            pad = width - W0
            # pad on width axis (axis=2), not height
            return jnp.pad(sprites, ((0, 0), (0, 0), (0, pad), (0, 0)))

        sprites_normal_padded = pad_to_width_ff(sprites_normal, max_width)  # (6, H, max_w, C)
        sprites_wide_padded = pad_to_width_ff(sprites_wide, max_width)

        # Choose sprite if forcefield is wide
        all_sprites = jax.lax.cond(
            ff_state.is_wide,
            lambda _: sprites_wide_padded,
            lambda _: sprites_normal_padded,
            operand=None
        )

        raster = jnp.where(jnp.logical_and(ff_state.is_in_current_event, jnp.logical_and(ff_state.is_alive, ff_state.flash_on)),
            # Case: alive -> render normally
            jru.render_at(
                jru.render_at(
                    jru.render_at(
                        jru.render_at(
                            jru.render_at(
                                jru.render_at(
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
        dp_x, dp_upmost_y, dp_height = dp_state.x, dp_state.upmost_y, self.consts.DENSEPACK_NORMAL_PART_SIZE[1]

        # select correct sprites based on broken_states
        densepack_correct_sprites = SPRITE_DENSEPACK[dp_state.broken_states]
        # first recolor each part sprite
        recolored_sprites = jax.vmap(
            lambda sp: recolor_sprite(sp, jnp.array(self.consts.DENSEPACK_COLOR))
        )(densepack_correct_sprites)

        def resize_sprite_width_dp(sprite: jnp.ndarray, new_width: int) -> jnp.ndarray:
            H, W, C = sprite.shape
            # swap back to (H, new_width, C)
            return jax.image.resize(sprite, (H, new_width, C), method='nearest')

        sprites_normal = recolored_sprites  # (n_parts, H, W, C)
        sprites_wide = jax.vmap(
            lambda s: resize_sprite_width_dp(s, self.consts.FORCEFIELD_WIDE_SIZE[0])
        )(recolored_sprites)  # (n_parts, H, wide, C)

        max_width = max(sprites_normal.shape[2], sprites_wide.shape[2])

        def pad_to_width_dp(sprites, width):
            H, W0, C = sprites.shape[1:]  # sprites.shape == (n_parts, H, W0, C)
            pad = width - W0
            # pad on the width axis (axis=2), not height
            return jnp.pad(sprites, ((0, 0), (0, 0), (0, pad), (0, 0)))

        sprites_normal_padded = pad_to_width_dp(sprites_normal, max_width)  # (n_parts, H, max_w, C)
        sprites_wide_padded = pad_to_width_dp(sprites_wide, max_width)

        # Choose sprite if densepack is wide
        all_sprites = jax.lax.cond(
            dp_state.is_wide,
            lambda _: sprites_wide_padded,
            lambda _: sprites_normal_padded,
            operand=None
        )

        def render_densepack_parts(raster):
            idx = jnp.arange(self.consts.DENSEPACK_NUMBER_OF_PARTS, dtype=jnp.int32)
            ys = dp_upmost_y + idx * dp_height

            def step(r, args):
                i, y = args
                return jru.render_at(r, dp_x, y, all_sprites[i]), None

            raster_out, _ = jax.lax.scan(step, raster, (idx, ys))
            return raster_out

        raster = jnp.where(
            jnp.logical_and(dp_state.is_in_current_event, dp_state.is_alive),
            render_densepack_parts(raster),
            raster
        )

        # -------- Render Detonator --------

        raster = jnp.where(
            jnp.logical_and(dn_state.is_in_current_event, dn_state.is_alive),
            jru.render_at(
                jru.render_at(
                    raster,
                    dn_state.x,
                    dn_state.y,
                    recolor_sprite(SPRITE_DETONATOR, jnp.array(self.consts.DETONATOR_COLOR)),
                ),
                dn_state.x + 2,
                dn_state.y + 25,
                SPRITE_6507,
            ),
            raster
        )

        # -------- Render Energy Pod --------

        energy_pod_colored = jnp.where(ep_state.animation_timer < (self.consts.ENERGY_POD_ANIMATION_SPEED // 2),
                                       recolor_sprite(SPRITE_ENERGY_POD, jnp.array(self.consts.ENERGY_POD_COLOR_GREEN)),
                                       recolor_sprite(SPRITE_ENERGY_POD, jnp.array(self.consts.ENERGY_POD_COLOR_GRAY))
                                       )

        raster = jnp.where(
            jnp.logical_and(ep_state.is_in_current_event, ep_state.is_alive),
            jru.render_at(
                raster,
                ep_state.x,
                ep_state.y,
                energy_pod_colored,
            ),
            raster
        )


        # -------- Render background parts --------

        # Upper brown background above upper mountains
        raster = jru.render_at(
            raster,
            0,
            0,
            SPRITE_UPPER_BROWN_BG,
        )

        # Lower brown background below upper mountains
        raster = jru.render_at(
            raster,
            0,
            92,
            SPRITE_LOWER_BROWN_BG,
        )

        # Background of instrument panel
        raster = jru.render_at(
            raster,
            0,
            109,
            SPRITE_GRAY_GUI_BG,
        )

        # -------- Render gui --------

        sprite_gui_colored_background_blue = recolor_sprite(SPRITE_GUI_COLORED_BACKGROUND, jnp.array(self.consts.GUI_COLORED_BACKGROUND_COLOR_BLUE)) # For Score
        sprite_gui_colored_background_green = recolor_sprite(SPRITE_GUI_COLORED_BACKGROUND, jnp.array(self.consts.GUI_COLORED_BACKGROUND_COLOR_GREEN)) # For Energy, Shields and Dtime
        sprite_gui_colored_background_beige = recolor_sprite(SPRITE_GUI_COLORED_BACKGROUND, jnp.array(self.consts.GUI_COLORED_BACKGROUND_COLOR_BEIGE))
        sprite_gui_colored_background_gray = recolor_sprite(SPRITE_GUI_COLORED_BACKGROUND, jnp.array(self.consts.GUI_COLORED_BACKGROUND_COLOR_GRAY))
        blinking_sprite_gui_colored_background = jnp.where((state.step_counter % self.consts.INSTRUMENT_PANEL_ANIMATION_SPEED) < (self.consts.INSTRUMENT_PANEL_ANIMATION_SPEED // 2), sprite_gui_colored_background_beige, sprite_gui_colored_background_gray)

        # Colored backgrounds ---------------


        # Colored background for score
        score_col_bg_y = self.consts.GUI_Y_BASE
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE,
            score_col_bg_y,
            sprite_gui_colored_background_blue,
        )

        # Colored background for energy
        energy_col_bg_y = self.consts.GUI_Y_BASE + self.consts.GUI_COLORED_BACKGROUND_SIZE[1] + self.consts.GUI_Y_SPACE_BETWEEN_PANELS
        low_energy = state.energy < (0.2 * self.consts.MAX_ENERGY)
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE,
            energy_col_bg_y,
            jnp.where(low_energy, blinking_sprite_gui_colored_background, sprite_gui_colored_background_green),
        )

        # Colored background for shields
        shields_col_bg_y = self.consts.GUI_Y_BASE + 2 * self.consts.GUI_COLORED_BACKGROUND_SIZE[1] + 2 * self.consts.GUI_Y_SPACE_BETWEEN_PANELS
        low_shields = state.shields < (0.2 * self.consts.MAX_SHIELDS)
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE,
            shields_col_bg_y,
            jnp.where(low_shields, blinking_sprite_gui_colored_background, sprite_gui_colored_background_green),
        )

        # Colored background for d-time
        dtime_col_bg_y = self.consts.GUI_Y_BASE + 3 * self.consts.GUI_COLORED_BACKGROUND_SIZE[1] + 3 * self.consts.GUI_Y_SPACE_BETWEEN_PANELS
        low_dtime = state.dtime < (0.2 * self.consts.MAX_DTIME)
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE,
            dtime_col_bg_y,
            jnp.where(low_dtime, blinking_sprite_gui_colored_background, sprite_gui_colored_background_green),
        )

        # Black backgrounds ---------------

        # Black background for score
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET,
            score_col_bg_y + 1,
            SPRITE_GUI_BLACK_BACKGROUND,
        )

        # Black background for energy
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET,
            energy_col_bg_y + 1,
            SPRITE_GUI_BLACK_BACKGROUND,
        )

        # Black background for shields
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET,
            shields_col_bg_y + 1,
            SPRITE_GUI_BLACK_BACKGROUND,
        )

        # Black background for d-time
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET,
            dtime_col_bg_y + 1,
            SPRITE_GUI_BLACK_BACKGROUND,
        )

        # Text ---------------

        # score text
        required_text_and_bar_color = jnp.where(jnp.array(True), jnp.array(self.consts.GUI_TEXT_COLOR_GRAY), jnp.array(self.consts.GUI_TEXT_COLOR_BEIGE))
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            score_col_bg_y + 2,
            recolor_sprite(SPRITE_GUI_TEXT_SCORE, required_text_and_bar_color),
        )

        # energy text
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            energy_col_bg_y + 2,
            recolor_sprite(SPRITE_GUI_TEXT_ENERGY, required_text_and_bar_color),
        )

        # shields text
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            shields_col_bg_y + 2,
            recolor_sprite(SPRITE_GUI_TEXT_SHIELDS, required_text_and_bar_color),
        )

        # d-time text
        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            dtime_col_bg_y + 2,
            recolor_sprite(SPRITE_GUI_TEXT_DTIME, required_text_and_bar_color),
        )

        # Bars ---------------

        # energy bar
        raster = jru.render_bar(
            raster, # raster
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 4, # x pos
            energy_col_bg_y + 8, # y pos
            state.energy, # current value
            self.consts.MAX_ENERGY, # maximum value
            40, # width
            2, # height
            required_text_and_bar_color, # color of filled part
            jnp.array((0, 0, 0, 0)) # color of unfilled part
        )

        # shields bar
        raster = jru.render_bar(
            raster, # raster
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 4, # x pos
            shields_col_bg_y + 8, # y pos
            state.shields, # current value
            self.consts.MAX_SHIELDS, # maximum value
            31, # width
            2, # height
            required_text_and_bar_color, # color of filled part
            jnp.array((0, 0, 0, 0)) # color of unfilled part
        )

        # d-time bar
        raster = jru.render_bar(
            raster, # raster
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 4, # x pos
            dtime_col_bg_y + 8, # y pos
            state.dtime, # current value
            self.consts.MAX_DTIME, # maximum value
            40, # width
            2, # height
            required_text_and_bar_color, # color of filled part
            jnp.array((0, 0, 0, 0)) # color of unfilled part
        )

        # Score ---------------

        # digits of score
        score_array = jru.int_to_digits(state.score, 6) # Convert integer to array with its digits

        recolor_single = lambda sprite_idx: recolor_sprite(sprite_idx, required_text_and_bar_color)
        recolored_sprites = jax.vmap(recolor_single)(SPRITE_GUI_SCORE_DIGITS) # Vmap over all digit sprites and recolor to desired color

        first_non_zero = jnp.argmax(score_array != 0) # Index of first element in score_array that is not zero
        num_to_render = score_array.shape[0] - first_non_zero # number of digits we have to render
        base_x = self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 52 # base x position
        number_spacing = 4 # Spacing of digits (including digit itself)
        score_numbers_x = base_x - number_spacing * num_to_render # Subtracting offset of x position, since we want the score to be right-aligned

        raster = jnp.where(state.score > 0,  # Render only if score is more than 0
                           jru.render_label_selective(
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
                           jru.render_at(
                               raster,
                               base_x - 14,
                               score_col_bg_y + 8,
                               recolor_sprite(SPRITE_GUI_SCORE_COMMA, required_text_and_bar_color),
                           ),
                           raster
                           )


        # -------- Render player --------
        timer = state.animation_timer.astype(jnp.int32) - (255 - self.consts.PLAYER_COLOR_CHANGE_DURATION)
        raster = jru.render_at(
            raster,
            state.player_x,
            state.player_y,
            recolor_sprite(SPRITE_PLAYER, jnp.where(timer <= 0, jnp.array(self.consts.PLAYER_NORMAL_COLOR), jnp.array(self.consts.PLAYER_COLLISION_COLOR))),
            flip_horizontal=state.player_facing_direction < 0,
        )

        # -------- Render player missile --------

        base_r, base_g, base_b, base_t = self.consts.PLAYER_MISSILE_BASE_COLOR
        color_change = state.player_missile.velocity * self.consts.PLAYER_MISSILE_COLOR_CHANGE_SPEED

        r = jnp.clip(base_r + color_change, 0, 255)
        g = jnp.clip(base_g + color_change, 0, 255)
        b = jnp.clip(base_b + color_change, 0, 255)
        t = jnp.clip(base_t + color_change, 0, 255)

        colored_player_missile = recolor_sprite(SPRITE_PLAYER_MISSILE, jnp.array((r, g, b, t)))
        raster = jnp.where(state.player_missile.direction != 0,
                           jru.render_at(
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
        raster = jru.render_at(
            raster,
            state.lower_mountains.x1,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
        )

        raster = jru.render_at(
            raster,
            state.lower_mountains.x2,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
        )

        raster = jru.render_at(
            raster,
            state.lower_mountains.x3,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
        )

        # Upper mountains
        raster = jru.render_at(
            raster,
            state.upper_mountains.x1,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
        )

        raster = jru.render_at(
            raster,
            state.upper_mountains.x2,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
        )

        raster = jru.render_at(
            raster,
            state.upper_mountains.x3,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
        )

        # Weird black stripe 1
        raster = jru.render_at(
            raster,
            0,
            18,
            SPRITE_BLACK_STRIPE,
        )

        # Weird black stripe 2
        raster = jru.render_at(
            raster,
            0,
            109,
            SPRITE_BLACK_STRIPE,
        )

        return raster