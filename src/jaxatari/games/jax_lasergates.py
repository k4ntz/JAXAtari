"""

Lukas Bergholz, Linus Orlob, Vincent Jahn

"""
import os
from functools import partial
from typing import Tuple, NamedTuple, Callable
import chex
from flax import struct
import jax
import jax.numpy as jnp
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, JAXAtariAction
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import AutoDerivedConstants

# -------- Asset Configuration --------
def _get_static_asset_config() -> Tuple[dict, ...]:
    """Only returns the file-based assets. Procedural assets are added by the Renderer."""
    densepack_files = [f"enemies/densepack/{i}.npy" for i in range(5, 0, -1)]
    radar_mortar_files = [f"enemies/radar_mortar/{i}.npy" for i in range(1, 4)]
    byte_bat_files = [f"enemies/byte_bat/{i}.npy" for i in range(1, 4)]
    rock_muncher_files = [f"enemies/rock_muncher/{i}.npy" for i in range(1, 4)]
    upper_death_files = [f"enemies/enemy_death/top/{i}.npy" for i in range(1, 13)]
    lower_death_files = [f"enemies/enemy_death/bottom/{i}.npy" for i in range(1, 13)]

    return (
        {"name": "upper_brown_bg", "type": "single", "file": "background/upper_brown_bg.npy"},
        {"name": "lower_brown_bg", "type": "single", "file": "background/lower_brown_bg.npy"},
        {"name": "playing_field_bg", "type": "single", "file": "background/playing_field_bg.npy"},
        {"name": "playing_field_small_bg", "type": "single", "file": "background/mountains/playing_field_small_bg.npy"},
        {"name": "gray_gui_bg", "type": "single", "file": "background/gray_gui_bg.npy"},
        {"name": "lower_mountain", "type": "single", "file": "background/mountains/lower_mountain.npy"},
        {"name": "upper_mountain", "type": "single", "file": "background/mountains/upper_mountain.npy"},
        {"name": "black_stripe", "type": "single", "file": "background/black_stripe.npy"},
        {"name": "player", "type": "single", "file": "player/player.npy"},
        {"name": "player_missile", "type": "single", "file": "missiles/player_missile.npy"},
        {"name": "gui_colored_background", "type": "single", "file": "gui/colored_background.npy"},
        {"name": "gui_black_background", "type": "single", "file": "gui/black_background.npy"},
        {"name": "gui_text_score", "type": "single", "file": "gui/text/score.npy"},
        {"name": "gui_text_energy", "type": "single", "file": "gui/text/energy.npy"},
        {"name": "gui_text_shields", "type": "single", "file": "gui/text/shields.npy"},
        {"name": "gui_text_dtime", "type": "single", "file": "gui/text/dtime.npy"},
        {"name": "gui_score_digits", "type": "digits", "pattern": "gui/score_numbers/{}.npy"},
        {"name": "gui_score_comma", "type": "single", "file": "gui/score_numbers/comma.npy"},
        {"name": "entity_missile", "type": "single", "file": "missiles/enemy_missile.npy"},
        {"name": "homing_missile", "type": "single", "file": "enemies/homing_missile/homing_missile.npy"},
        {"name": "forcefield", "type": "single", "file": "enemies/forcefield/forcefield.npy"},
        {"name": "detonator", "type": "single", "file": "enemies/detonator/detonator.npy"},
        {"name": "detonator_6507", "type": "single", "file": "enemies/detonator/6507.npy"},
        {"name": "energy_pod", "type": "single", "file": "enemies/energy_pod/energy_pod.npy"},
        {"name": "upper_death_frames", "type": "group", "files": upper_death_files},
        {"name": "lower_death_frames", "type": "group", "files": lower_death_files},
        {"name": "death_number_325", "type": "single", "file": "enemies/enemy_death/numbers/325.npy"},
        {"name": "death_number_525", "type": "single", "file": "enemies/enemy_death/numbers/525.npy"},
        {"name": "death_number_bg", "type": "single", "file": "enemies/enemy_death/numbers/background.npy"},
        {"name": "radar_mortar_frames", "type": "group", "files": radar_mortar_files},
        {"name": "byte_bat_frames", "type": "group", "files": byte_bat_files},
        {"name": "rock_muncher_frames", "type": "group", "files": rock_muncher_files},
        {"name": "densepack_frames", "type": "group", "files": densepack_files},
    )

# -------- Game constants --------
class LaserGatesConstants(AutoDerivedConstants):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=250)
    SCALING_FACTOR: int = struct.field(pytree_node=False, default=5)

    SCROLL_SPEED: float = struct.field(pytree_node=False, default=0.8) # Normal scroll speed
    SCROLL_MULTIPLIER: int = struct.field(pytree_node=False, default=2) # When at the right player bound, multiply scroll speed by this constant

    # -------- Mountains constants --------
    PLAYING_FIELD_BG_COLLISION_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (255, 255, 255, 255))
    PLAYING_FILED_BG_COLOR_FADE_SPEED: float = struct.field(pytree_node=False, default=0.2)  # Higher = faster fade out, exponential

    # -------- Mountains constants --------
    MOUNTAIN_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (60, 12))  # Width, Height

    LOWER_MOUNTAINS_Y: int = struct.field(pytree_node=False, default=80)  # Y Spawn position of lower mountains. This does not change
    UPPER_MOUNTAINS_Y: int = struct.field(pytree_node=False, default=19)  # Y Spawn position of upper mountains. This does not change

    LOWER_MOUNTAINS_START_X: int = struct.field(pytree_node=False, default=-44)  # X Spawn position of lower mountains.
    UPPER_MOUNTAINS_START_X: int = struct.field(pytree_node=False, default=-4)  # X Spawn position of upper mountains.

    MOUNTAINS_DISTANCE: int = struct.field(pytree_node=False, default=20)  # Distance between two given mountains

    UPDATE_EVERY: int = struct.field(pytree_node=False, default=4)  # The mountain position is updated every UPDATE_EVERY-th frame.

    # -------- Player constants --------
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 6))  # Width, Height
    PLAYER_NORMAL_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (99, 138, 196, 255))  # Normal color of the player
    PLAYER_COLLISION_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (137, 81, 26,255))  # Players color for PLAYER_COLOR_CHANGE_DURATION frames after a collision
    PLAYER_COLOR_CHANGE_DURATION: int = struct.field(pytree_node=False, default=10)  # How long (in frames) the player changes its color to PLAYER_COLLISION_COLOR if a collision occurs

    PLAYER_BOUNDS: Tuple[Tuple[int, int], Tuple[int, int]] = struct.field(pytree_node=False, default=None)  # left x, right x, upper y and lower y bound of player

    PLAYER_START_X: int = struct.field(pytree_node=False, default=20)  # X Spawn position of player
    PLAYER_START_Y: int = struct.field(pytree_node=False, default=52)  # Y Spawn position of player

    PLAYER_VELOCITY_Y: float = struct.field(pytree_node=False, default=2)  # Y Velocity of player
    PLAYER_VELOCITY_X: float = struct.field(pytree_node=False, default=2)  # X Velocity of player

    # -------- Player missile constants --------
    PLAYER_MISSILE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (16, 1))  # Width, Height
    PLAYER_MISSILE_BASE_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (140, 79, 24, 255))  # Initial color of player missile. Every value except for transparency is incremented by the missiles velocity * PLAYER_MISSILE_COLOR_CHANGE_SPEED
    PLAYER_MISSILE_COLOR_CHANGE_SPEED: int = struct.field(pytree_node=False, default=10)  # Defines how fast the player missile changes its color towards white.

    PLAYER_MISSILE_INITIAL_VELOCITY: int = struct.field(pytree_node=False, default=9)  # Starting speed of player missile
    PLAYER_MISSILE_VELOCITY_MULTIPLIER: float = struct.field(pytree_node=False, default=1.1)  # Multiply the current speed at a given moment of the player missile by this number

    # -------- Entity constants (constants that apply to all entity types --------
    ENTITY_DEATH_SPRITES_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 45))  # Width, Height
    ENTITY_MISSILE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 1))  # Width, Height

    NUM_ENTITY_TYPES: int = struct.field(pytree_node=False, default=8)  # How many different (!) entity types there are
    ENTITY_DEATH_SPRITE_Y_OFFSET: int = struct.field(pytree_node=False, default=7)  # Y offset to add to the death sprite (the constant is being added to the y coordinate)
    ENTITY_DEATH_ANIMATION_TIMER: int = struct.field(pytree_node=False, default=100)  # Duration of death sprite animation in frames
    ENTITY_DEATH_SPRITES_NUMBER_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (117, 117, 213, 255))

    # -------- Radar mortar constants --------
    RADAR_MORTAR_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 26))  # Width, Height
    RADAR_MORTAR_COLOR_BLUE: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (96, 162, 228, 255))
    RADAR_MORTAR_COLOR_GRAY: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (155, 155, 155, 255))

    RADAR_MORTAR_SPAWN_X: int = struct.field(pytree_node=False, default=None)  # Spawn barely outside screen
    RADAR_MORTAR_SPAWN_BOTTOM_Y: int = struct.field(pytree_node=False, default=66)
    RADAR_MORTAR_SPAWN_UPPER_Y: int = struct.field(pytree_node=False, default=19)  # Since the radar mortar can spawn at the top or at the bottom of the screen, we define two y positions.

    RADAR_MORTAR_MISSILE_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (85, 92, 197, 255))
    RADAR_MORTAR_MISSILE_SPAWN_EVERY: int = struct.field(pytree_node=False, default=100)  # A missile is spawned every RADAR_MORTAR_MISSILE_SPAWN_EVERY-th frame.
    RADAR_MORTAR_MISSILE_SPEED: int = struct.field(pytree_node=False, default=3)  # Speed of radar mortar missile
    RADAR_MORTAR_MISSILE_SHOOT_NUMBER: int = struct.field(pytree_node=False, default=3)  # How often the missile gets teleported back before final shot (exept when shooting up or down)
    RADAR_MORTAR_MISSILE_SMALL_OUT_OF_BOUNDS_THRESHOLD: int = struct.field(pytree_node=False, default=50)  # How far the missile needs to be away from the radar mortar (vertically or/and horizontally) for the missile to be teleported back to the mortar (to be shot again)
    RADAR_MORTAR_SHOOT_STRAIGHT_THRESHOLD: int = struct.field(pytree_node=False, default=10)  # This defines how far the player needs to be away from the radar mortar (vertically or/and horizontally) for the missile to be shot diagonally

    # -------- Byte bat constants --------
    BYTE_BAT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (7, 8))  # Width, Height
    BYTE_BAT_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (90, 169, 99, 255))

    BYTE_BAT_UPPER_BORDER_Y: int = struct.field(pytree_node=False, default=None)  # Upper border where byte bat inverts the direction
    BYTE_BAT_BOTTOM_BORDER_Y: int = struct.field(pytree_node=False, default=None)  # Lower border where byte bat inverts the direction
    BYTE_BAT_SUBTRACT_FROM_BORDER: int = struct.field(pytree_node=False, default=20)
    BYTE_BAT_PLAYER_DIST_TRIGGER: int = struct.field(pytree_node=False, default=50)

    BYTE_BAT_SPAWN_X: int = struct.field(pytree_node=False, default=None)  # Spawn barely outside screen
    BYTE_BAT_SPAWN_Y: int = struct.field(pytree_node=False, default=None)  # Spawn barely outside screen

    BYTE_BAT_X_SPEED: float = struct.field(pytree_node=False, default=1)  # Speed of byte bat in x direction
    BYTE_BAT_Y_SPEED: float = struct.field(pytree_node=False, default=1.8)  # Speed of byte bat in y direction

    # -------- Rock muncher constants --------
    ROCK_MUNCHER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 11))  # Width, Height

    ROCK_MUNCHER_UPPER_BORDER_Y: int = struct.field(pytree_node=False, default=None)  # Upper border where rock muncher inverts the direction
    ROCK_MUNCHER_BOTTOM_BORDER_Y: int = struct.field(pytree_node=False, default=None)  # Lower border where rock muncher inverts the direction

    ROCK_MUNCHER_SPAWN_X: int = struct.field(pytree_node=False, default=None)  # Spawn barely outside of screen
    ROCK_MUNCHER_SPAWN_Y: int = struct.field(pytree_node=False, default=None)  # Spawn barely outside screen

    ROCK_MUNCHER_X_SPEED: float = struct.field(pytree_node=False, default=0.7)  # Speed of rock muncher in x direction
    ROCK_MUNCHER_Y_SPEED: float = struct.field(pytree_node=False, default=1)  # Speed of rock muncher in y direction

    ROCK_MUNCHER_MISSILE_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (85, 92, 197, 255))
    ROCK_MUNCHER_MISSILE_SPAWN_EVERY: int = struct.field(pytree_node=False, default=50)  # Rock muncher shoots a new missile every ROCK_MUNCHER_MISSILE_SPAWN_EVERY frames
    ROCK_MUNCHER_MISSILE_SPEED: int = struct.field(pytree_node=False, default=4)  # Speed of rock muncher missile

    # -------- Homing Missile constants --------
    HOMING_MISSILE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 5))  # Width, Height

    HOMING_MISSILE_Y_BOUNDS: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (32, 74))
    HOMING_MISSILE_PLAYER_TRACKING_RANGE: int = struct.field(pytree_node=False, default=15)  # The minimum y position difference between player and homing missile needed for the homing missile to start tracking the player
    HOMING_MISSILE_Y_PLAYER_OFFSET: int = struct.field(pytree_node=False, default=2)  # Sets the y position this many pixels above the player (for positive numbers) or below the player (for negative numbers)
    HOMING_MISSILE_X_SPEED: float = struct.field(pytree_node=False, default=2.5)
    HOMING_MISSILE_Y_SPEED: float = struct.field(pytree_node=False, default=1)

    # -------- Forcefield constants --------
    FORCEFIELD_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 73))  # Width, Height of a single normal forcefield column
    FORCEFIELD_WIDE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (16, 73))  # Width, Height of a single wide forcefield column

    FORCEFIELD_IS_WIDE_PROBABILITY: float = struct.field(pytree_node=False, default=0.2)  # Probability that a forcefield is wide

    FORCEFIELD_FLASHING_SPACING: int = struct.field(pytree_node=False, default=32)  # x spacing between the forcefields when in flashing mode
    FORCEFIELD_FLASHING_SPEED: int = struct.field(pytree_node=False, default=35)  # Forcefield changes state from on to off or from off to on every FORCEFIELD_FLASHING_SPEED frames

    FORCEFIELD_FLEXING_SPACING: int = struct.field(pytree_node=False, default=64)  # x spacing between the forcefields when in flexing mode
    FORCEFIELD_FLEXING_SPEED: float = struct.field(pytree_node=False, default=0.6)  # Flexing (Crushing motion) speed
    FORCEFIELD_FLEXING_MINIMUM_DISTANCE: int = struct.field(pytree_node=False, default=2)  # Minimum y distance between the upper and lower forcefields when flexing
    FORCEFIELD_FLEXING_MAXIMUM_DISTANCE: int = struct.field(pytree_node=False, default=32)  # Maximum y distance between the upper and lower forcefields when flexing

    FORCEFIELD_FIXED_SPACING: int = struct.field(pytree_node=False, default=64)  # x spacing between the forcefields when in fixed mode
    FORCEFIELD_FIXED_SPEED: float = struct.field(pytree_node=False, default=0.3)  # Fixed (up and down movement) speed
    FORCEFIELD_FIXED_UPPER_BOUND: int = struct.field(pytree_node=False, default=None)  # Highest allowed y position for forcefields while fixed
    FORCEFIELD_FIXED_LOWER_BOUND: int = struct.field(pytree_node=False, default=None)  # Lowest allowed y position for forcefields while fixed

    # -------- Densepack constants --------
    DENSEPACK_NORMAL_PART_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 4))  # Width, Height of a single, normal densepack part in a normal densepack column
    DENSEPACK_WIDE_PART_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (16, 4))  # Width, Height of a single, wide densepack part in a wide densepack column
    DENSEPACK_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (142, 142, 142, 255))

    DENSEPACK_NUMBER_OF_PARTS: int = struct.field(pytree_node=False, default=19)  # number of segments in the densepack
    DENSEPACK_IS_WIDE_PROBABILITY: float = struct.field(pytree_node=False, default=0.4)  # Probability that a spawned densepack is wide

    # -------- Detonator constants --------
    DETONATOR_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 73))
    DETONATOR_COLOR: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (142, 142, 142, 255))

    # -------- Energy pod constants --------
    ENERGY_POD_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (6, 4))  # Width, Height
    ENERGY_POD_COLOR_GREEN: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (84, 171, 96, 255))  # Energy pod color in green frame
    ENERGY_POD_COLOR_GRAY: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (142, 142, 142, 255))  # Energy pod color in gray frame

    ENERGY_POD_ANIMATION_SPEED: int = struct.field(pytree_node=False, default=16)  # Higher is slower

    # -------- Probability constants --------

    ALLOW_ENERGY_POD_PERCENTAGE: float = struct.field(pytree_node=False, default=0.3)  # The energy pod is allowed to spawn (one in 7 to 8 chance) when current energy is smaller than ALLOW_ENERGY_POD_PERCENTAGE * MAX_ENERGY
    ALLOW_DETONATOR_PERCENTAGE: float = struct.field(pytree_node=False, default=0.3)  # The detonator is allowed to spawn (one in 7 to 8 chance) when current energy is smaller than ALLOW_DETONATOR_PERCENTAGE * MAX_DTIME

    ENERGY_POD_SPAWN_PROBABILITY: float = struct.field(pytree_node=False, default=0.4)  # If energy pod spawning is allowed, this is the probability for the energy pod to be the next entity spawned.
    DETONATOR_SPAWN_PROBABILITY: float = struct.field(pytree_node=False, default=0.4)  # If detonator spawning is allowed, this is the probability for the detonator to be the next entity spawned.

    ENERGY_START_BLINKING_PERCENTAGE: float = struct.field(pytree_node=False, default=0.2) # see below
    SHIELDS_START_BLINKING_PERCENTAGE: float = struct.field(pytree_node=False, default=0.2) # see below
    DTIME_START_BLINKING_PERCENTAGE: float = struct.field(pytree_node=False, default=0.2) # The Field in the instrument panel starts blinking when the current value is smaller than VALUE_START_BLINKING_PERCENTAGE * MAX_VALUE.

    # -------- Instrument panel constants --------

    INSTRUMENT_PANEL_ANIMATION_SPEED: int = struct.field(pytree_node=False, default=14) # Blinking speed when energy, shields or dtime is low. Lower is faster. Should ideally be an even number.

    MAX_ENERGY: int = struct.field(pytree_node=False, default=5100)  # As the manual says, energy is consumed at a regular pace. We use 5100 for the initial value and subtract one for every frame to match the timing of the real game. (It takes 85 seconds for the energy to run out. 85 * 60 (fps) = 5100)
    MAX_SHIELDS: int = struct.field(pytree_node=False, default=24)  # As the manual says, the Dante Dart starts with 24 shield units
    MAX_DTIME: int = struct.field(pytree_node=False, default=10200)  # Same idea as energy.

    SHIELD_LOSS_COL_SMALL: int = struct.field(pytree_node=False, default=1)  # See is_big_collision entry in CollisionPropertiesState for extensive explanation. This constant defines the shield points to lose
    SHIELD_LOSS_COL_BIG: int = struct.field(pytree_node=False, default=6)

    # -------- GUI constants --------
    GUI_COLORED_BACKGROUND_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (128, 12))  # Width, Height of colored background of black rectangle background
    GUI_BLACK_BACKGROUND_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (56, 10))  # Width, Height of black background of the text
    GUI_TEXT_SCORE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (21, 7))  # Width, Height of "Score" text
    GUI_TEXT_ENERGY_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (23, 5))  # Width, Height of "Energy" text
    GUI_TEXT_SHIELDS_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (23, 5))  # Width, Height of "Shields" text
    GUI_TEXT_DTIME_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (23, 5))  # Width, Height of "Dtime" text

    GUI_COLORED_BACKGROUND_COLOR_BLUE: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (47, 90, 160, 255))
    GUI_COLORED_BACKGROUND_COLOR_GREEN: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (50, 152, 82, 255))
    GUI_COLORED_BACKGROUND_COLOR_BEIGE: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (160, 107, 50, 255))
    GUI_COLORED_BACKGROUND_COLOR_GRAY: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (182, 182, 182, 255))
    GUI_TEXT_COLOR_GRAY: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (118, 118, 118, 255))
    GUI_TEXT_COLOR_BEIGE: Tuple[int, int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (160, 107, 50, 255))

    GUI_BLACK_BACKGROUND_X_OFFSET: int = struct.field(pytree_node=False, default=36)
    GUI_Y_BASE: int = struct.field(pytree_node=False, default=117)
    GUI_X_BASE: int = struct.field(pytree_node=False, default=16)
    GUI_Y_SPACE_BETWEEN_PANELS: int = struct.field(pytree_node=False, default=21)

    # -------- Render Asset Configuration constants --------
    RADAR_MORTAR_SPRITE_ANIMATION_SPEED: int = struct.field(pytree_node=False, default=15)
    BYTE_BAT_ANIMATION_SPEED: int = struct.field(pytree_node=False, default=16)
    ROCK_MUNCHER_ANIMATION_SPEED: int = struct.field(pytree_node=False, default=10)

    PROCEDURAL_COLORS: dict = struct.field(
        pytree_node=False,
        default_factory=lambda: {
            "PFB_BASE_COLOR": (0, 0, 0, 255),
            "PFB_COLLISION_COLOR": (104, 104, 0, 255),
            "DEATH_NUMBER_COLOR": (208, 208, 0, 255),
            "RADAR_MORTAR_COLOR_GRAY": (104, 104, 104, 255),
            "RADAR_MORTAR_COLOR_BLUE": (0, 0, 172, 255),
            "RADAR_MORTAR_MISSILE_COLOR": (85, 92, 197, 255),
            "BYTE_BAT_COLOR": (0, 88, 0, 255),
            "ROCK_MUNCHER_MISSILE_COLOR": (208, 208, 0, 255),
            "DENSEPACK_COLOR": (0, 88, 0, 255),
            "DETONATOR_COLOR": (104, 104, 104, 255),
            "ENERGY_POD_COLOR_GREEN": (0, 172, 0, 255),
            "ENERGY_POD_COLOR_GRAY": (104, 104, 104, 255),
            "GUI_COLORED_BACKGROUND_COLOR_BLUE": (0, 0, 172, 255),
            "GUI_COLORED_BACKGROUND_COLOR_GREEN": (0, 172, 0, 255),
            "GUI_COLORED_BACKGROUND_COLOR_BEIGE": (208, 208, 0, 255),
            "GUI_COLORED_BACKGROUND_COLOR_GRAY": (104, 104, 104, 255),
            "GUI_TEXT_COLOR_GRAY": (104, 104, 104, 255),
            "GUI_TEXT_COLOR_BEIGE": (208, 208, 0, 255),
            "GUI_BAR_EMPTY_COLOR": (0, 0, 0, 0),
            "PLAYER_NORMAL_COLOR": (208, 208, 0, 255),
            "PLAYER_COLLISION_COLOR": (172, 0, 0, 255),
            "PLAYER_MISSILE_BASE_COLOR": (208, 208, 0, 255),
            "RASTER_BACKGROUND": (0, 0, 0, 255),
            "FORCEFIELD_COLOR_0": (255, 0, 0, 255),
            "FORCEFIELD_COLOR_1": (255, 128, 0, 255),
            "FORCEFIELD_COLOR_2": (255, 255, 0, 255),
            "FORCEFIELD_COLOR_3": (128, 255, 0, 255),
            "FORCEFIELD_COLOR_4": (0, 255, 0, 255),
            "FORCEFIELD_COLOR_5": (0, 255, 128, 255),
            "FORCEFIELD_COLOR_6": (0, 255, 255, 255),
            "FORCEFIELD_COLOR_7": (0, 128, 255, 255),
            "FORCEFIELD_COLOR_8": (0, 0, 255, 255),
            "FORCEFIELD_COLOR_9": (128, 0, 255, 255),
            "FORCEFIELD_COLOR_10": (255, 0, 255, 255),
            "FORCEFIELD_COLOR_11": (255, 0, 128, 255),
            "FORCEFIELD_COLOR_12": (255, 255, 255, 255),
            "FORCEFIELD_COLOR_13": (192, 192, 192, 255),
            "FORCEFIELD_COLOR_14": (128, 128, 128, 255),
            "FORCEFIELD_COLOR_15": (255, 255, 255, 255),
        }
    )

    # -------- Debug constants --------
    DEBUG_ACTIVATE_MOUNTAINS_SCROLL: bool = struct.field(pytree_node=False, default=True)

    # -------- Asset config --------
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_static_asset_config)

    def compute_derived(self):
        byte_bat_upper = self.UPPER_MOUNTAINS_Y + self.MOUNTAIN_SIZE[1] + 2
        rock_muncher_upper = self.UPPER_MOUNTAINS_Y + self.MOUNTAIN_SIZE[1] + 15  # (5 + 10)
        return {
            "PLAYER_BOUNDS": ((20, self.WIDTH - 20 - self.PLAYER_SIZE[0]), (19, 80 + self.PLAYER_SIZE[1])),
            "RADAR_MORTAR_SPAWN_X": self.WIDTH,
            "BYTE_BAT_UPPER_BORDER_Y": self.UPPER_MOUNTAINS_Y + self.MOUNTAIN_SIZE[1] + 2,
            "BYTE_BAT_BOTTOM_BORDER_Y": self.LOWER_MOUNTAINS_Y - self.MOUNTAIN_SIZE[1],
            "BYTE_BAT_SPAWN_X": self.WIDTH,
            "BYTE_BAT_SPAWN_Y": byte_bat_upper + 1,
            "ROCK_MUNCHER_UPPER_BORDER_Y": self.UPPER_MOUNTAINS_Y + self.MOUNTAIN_SIZE[1] + 5 + 10,
            "ROCK_MUNCHER_BOTTOM_BORDER_Y": self.LOWER_MOUNTAINS_Y - self.MOUNTAIN_SIZE[1] - 3,
            "ROCK_MUNCHER_SPAWN_X": self.WIDTH,
            "ROCK_MUNCHER_SPAWN_Y": rock_muncher_upper + 1,
            "FORCEFIELD_FIXED_UPPER_BOUND": -self.FORCEFIELD_SIZE[1] + 33,
            "FORCEFIELD_FIXED_LOWER_BOUND": -self.FORCEFIELD_SIZE[1] + 68,
        }


# -------- States --------
@struct.dataclass
class RadarMortarState:
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    missile_x: chex.Array
    missile_y: chex.Array
    missile_direction: chex.Array
    shoot_again_timer: chex.Array

@struct.dataclass
class ByteBatState:
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    direction_is_up: jnp.bool
    direction_is_left: jnp.bool

@struct.dataclass
class RockMuncherState:
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    direction_is_up: jnp.bool
    direction_is_left: jnp.bool
    missile_x: chex.Array
    missile_y: chex.Array

@struct.dataclass
class HomingMissileState:
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    is_tracking_player: jnp.bool

@struct.dataclass
class ForceFieldState:
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

@struct.dataclass
class DensepackState:
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    upmost_y: chex.Array
    is_wide: jnp.bool
    number_of_parts: chex.Array
    broken_states: chex.Array

@struct.dataclass
class DetonatorState:
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    collision_is_pin: jnp.bool

@struct.dataclass
class EnergyPodState:
    is_in_current_event: jnp.bool
    is_alive: jnp.bool
    x: chex.Array
    y: chex.Array
    animation_timer: chex.Array

@struct.dataclass
class CollisionPropertiesState:
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

@struct.dataclass
class EntitiesState:
    radar_mortar_state: RadarMortarState        # Radar mortars appear along the top and bottom of the Computer passage. Avoid Mortar fire. Demolish Radar Mortars with laser fire.
    byte_bat_state: ByteBatState                # Green bat looking entity flying at you without warning.
    rock_muncher_state: RockMuncherState        # Pink brown green entity flying at you without warning. Shoots missiles.
    homing_missile_state: HomingMissileState    # Bomb looking entity flying and tracking you
    forcefield_state: ForceFieldState           # Flashing, flexing or fixed "wall". Time your approach to cross.
    dense_pack_state: DensepackState            # Gray densepack columns of varying width appear along the dark Computer passage. Blast your way through.
    detonator_state: DetonatorState             # "Failsafe detonators" are large and grey and have the numbers "6507" etched on the side. Laser fire must strike one of the pins on the side of a detonator to destroy it.
    energy_pod_state: EnergyPodState            # To replenish energy reserves, touch Energy Pods as they appear along the Computer passageway. Do not fire at Energy Pods! You may not survive until another appears!

    collision_properties_state: CollisionPropertiesState # Holds attributes relevant for collision logic

@struct.dataclass
class MountainState:
    x1: chex.Array
    x2: chex.Array
    x3: chex.Array
    y: chex.Array

@struct.dataclass
class PlayerMissileState:
    x: chex.Array
    y: chex.Array
    direction: chex.Array
    velocity: chex.Array

@struct.dataclass
class LaserGatesState:
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

@struct.dataclass
class EntityPosition:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

@struct.dataclass
class LaserGatesObservation:
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


@struct.dataclass
class LaserGatesInfo:
    # difficulty: jnp.ndarray # add if necessary
    step_counter: jnp.ndarray

# -------- Game Logic --------

class JaxLaserGates(JaxEnvironment[LaserGatesState, LaserGatesObservation, LaserGatesInfo, LaserGatesConstants]):
    # Minimal ALE action set for Laser Gates
    ACTION_SET: jnp.ndarray = jnp.array(
        [
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
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: LaserGatesConstants = None):
        consts = consts or LaserGatesConstants()
        super().__init__(consts)
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
            return entities.replace(radar_mortar_state=new_radar_mortar_state)

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
            return entities.replace(byte_bat_state=new_byte_bat_state)

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
            return entities.replace(rock_muncher_state=new_rock_muncher_state)

        def initialize_homing_missile(entities):
            initial_y_position = jax.random.randint(key_intern, (), self.consts.HOMING_MISSILE_Y_BOUNDS[0], self.consts.HOMING_MISSILE_Y_BOUNDS[1])
            new_homing_missile_state = HomingMissileState(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.WIDTH).astype(entities.homing_missile_state.x.dtype),
                y=initial_y_position,
                is_tracking_player=jnp.bool(False),
            )
            return entities.replace(homing_missile_state=new_homing_missile_state)

        def initialize_forcefield(entities):
            key_num_of_ff, key_type_of_ff, key_is_wide = jax.random.split(key_intern, 3)
            number_of_forcefields = jax.random.randint(key_num_of_ff, (), minval=1, maxval=5) # Spawn 1 to 4 forcefields at a time.

            type_of_forcefield = jax.random.randint(key_type_of_ff, (), minval=0, maxval=3)
            init_is_flexing = type_of_forcefield == 0
            init_is_fixed = type_of_forcefield == 1

            init_is_wide = jax.random.bernoulli(key_is_wide, p=self.consts.FORCEFIELD_IS_WIDE_PROBABILITY)

            number_of_forcefields = jnp.where(init_is_wide, 1, number_of_forcefields)

            new_forcefield_state = entities.forcefield_state.replace(
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
            return entities.replace(forcefield_state=new_forcefield_state)

        def initialize_densepack(entities):
            initial_is_wide = jax.random.bernoulli(key_intern, p=self.consts.DENSEPACK_IS_WIDE_PROBABILITY)

            new_densepack_state = entities.dense_pack_state.replace(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.WIDTH).astype(jnp.float32),
                upmost_y=jnp.array(19).astype(jnp.float32),
                is_wide=initial_is_wide,
                number_of_parts=jnp.array(self.consts.DENSEPACK_NUMBER_OF_PARTS).astype(jnp.int32),
                broken_states=jnp.full(self.consts.DENSEPACK_NUMBER_OF_PARTS, 4, jnp.int32),
            )
            return entities.replace(dense_pack_state=new_densepack_state)

        def initialize_detonator(entities):
            new_detonator_state = entities.detonator_state.replace(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.WIDTH).astype(jnp.float32),
                y=jnp.array(19).astype(jnp.float32),
                collision_is_pin=jnp.bool(False),
            )
            return entities.replace(detonator_state=new_detonator_state)

        def initialize_energy_pod(entities):
            new_energy_pod_state = entities.energy_pod_state.replace(
                is_in_current_event=jnp.bool(True),
                is_alive=jnp.bool(True),
                x=jnp.array(self.consts.WIDTH).astype(jnp.float32),
                y=jnp.array(73).astype(jnp.float32),
                animation_timer=jnp.array(0),
            )
            return entities.replace(energy_pod_state=new_energy_pod_state)

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

            return rm.replace(
                is_in_current_event=jnp.logical_and(new_is_in_current_event, rm.x > 0),
                is_alive=new_is_alive,
                x=new_x,
                missile_x=(missile_x - state.scroll_speed).astype(rm.missile_x.dtype),
                missile_y=missile_y,
                missile_direction=missile_dir,
                shoot_again_timer=new_timer
            ), state.entities.collision_properties_state.replace(
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

            return bb.replace(
                is_in_current_event=new_is_in_current_event,
                is_alive=new_is_alive,
                x=new_x,
                y=new_y,
                direction_is_up=new_direction_is_up,
                direction_is_left=new_direction_is_left,
            ), state.entities.collision_properties_state.replace(
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

            return rm.replace(
                is_in_current_event=new_is_in_current_event,
                is_alive=new_is_alive,
                x=new_x,
                y=new_y,
                direction_is_up=new_direction_is_up,
                direction_is_left=new_direction_is_left,
                missile_x=new_missile_x.astype(rm.missile_x.dtype),
                missile_y=new_missile_y.astype(rm.missile_y.dtype),
            ), state.entities.collision_properties_state.replace(
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

            return hm.replace(
                is_in_current_event=jnp.logical_and(new_is_in_current_event, hm.x > 0),
                is_alive=new_is_alive,
                x=new_x,
                y=new_y,
                is_tracking_player=new_is_tracking_player,
            ), state.entities.collision_properties_state.replace(
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

            return ff.replace(
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
            ), state.entities.collision_properties_state.replace(
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

            return dp.replace(
                is_in_current_event=new_is_in_current_event,
                is_alive=new_is_alive,
                x=base_x,
                broken_states=new_broken_states,
            ), state.entities.collision_properties_state.replace(
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

            return state.entities.detonator_state.replace(
                is_in_current_event=jnp.logical_and(new_is_in_current_event, base_x > 0), # Second condition should never happen, since you can only collide or destroy the detonator
                is_alive=new_is_alive,
                x=base_x.astype(jnp.float32),
                collision_is_pin=new_collision_is_pin,
            ), state.entities.collision_properties_state.replace(
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

            return state.entities.energy_pod_state.replace(
                is_in_current_event=jnp.logical_and(new_is_in_current_event, new_x > 0),
                is_alive=new_is_alive,
                x=new_x.astype(jnp.float32),
                animation_timer=new_animation_timer.astype(jnp.int32),
            ), state.entities.collision_properties_state.replace(
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
        return spaces.Discrete(len(self.ACTION_SET))

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

    def _get_info(self, state: LaserGatesState) -> LaserGatesInfo:
        return LaserGatesInfo(
            step_counter=state.step_counter,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: LaserGatesState, state: LaserGatesState) -> jnp.ndarray:
        return state.score - previous_state.score

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
        initial_entities = initial_entities.replace(
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
            self, state: LaserGatesState, action: int
    ) -> Tuple[LaserGatesObservation, LaserGatesState, float, bool, LaserGatesInfo]:
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

        # -------- Move player --------
        new_player_x, new_player_y, new_player_facing_direction = self.player_step(state, atari_action)
        player_animation_timer = state.animation_timer
        new_player_animation_timer = jnp.where(player_animation_timer != 0, player_animation_timer - 1, player_animation_timer)

        # -------- Move player missile --------
        new_player_missile_state = self.player_missile_step(state, atari_action)

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

        return_state = state.replace(
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
        info = self._get_info(return_state)

        return obs, return_state, 0.0, False, info

class LaserGatesRenderer(JAXGameRenderer):
    def __init__(self, consts: LaserGatesConstants = None):
        super().__init__()
        self.consts = consts or LaserGatesConstants()

        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Helper to generate 1x1 pixel for masks
        def _procedural_sprite(color: Tuple[int, int, int, int]) -> jnp.ndarray:
            return jnp.array(color, dtype=jnp.uint8).reshape(1, 1, 4)

        # 2. Helper to generate the large background based on current constants
        def _build_background() -> jnp.ndarray:
            color = jnp.array(self.consts.PROCEDURAL_COLORS["RASTER_BACKGROUND"], dtype=jnp.uint8).reshape(1, 1, 4)
            return jnp.tile(color, (self.consts.HEIGHT, self.consts.WIDTH, 1))

        # 3. Start with static file assets
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 4. Append the dynamic Background
        final_asset_config.append({
            "name": "base_background", 
            "type": "background", 
            "data": _build_background()
        })

        # 5. Append all the 1x1 color sprites from constants
        for name, color in self.consts.PROCEDURAL_COLORS.items():
            final_asset_config.append({
                "name": name, 
                "type": "procedural", 
                "data": _procedural_sprite(color)
            })

        # 6. Load everything
        sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites/lasergates")
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        def get_template_id(name: str) -> jnp.ndarray:
            """More robustly finds the first non-transparent color ID."""
            mask = self.SHAPE_MASKS[name]
            if mask.ndim == 3:  # Handle groups by taking first sprite
                mask = mask[0]
            flat_mask = mask.reshape(-1)
            # Find the ID at the first non-transparent index
            first_color_id = flat_mask[jnp.argmax(flat_mask != self.jr.TRANSPARENT_ID)]
            return first_color_id

        self.TEMPLATE_WHITE_ID = self.COLOR_TO_ID.get((255, 255, 255), 0)
        self.TEMPLATE_BLACK_ID = self.COLOR_TO_ID.get((0, 0, 0), 0)
        self.TEMPLATE_PFB_ID = get_template_id("playing_field_bg")

        self.COLOR_IDS = {
            name: self.COLOR_TO_ID.get(tuple(color[:3]), self.jr.TRANSPARENT_ID)
            for name, color in self.consts.PROCEDURAL_COLORS.items()
        }
        self.FORCEFIELD_COLOR_IDS = jnp.array(
            [self.COLOR_IDS[f"FORCEFIELD_COLOR_{i}"] for i in range(16)],
            dtype=jnp.int32,
        )

        self.PLAYING_FIELD_SMALL_BG_TEMPLATE_ID = get_template_id("playing_field_small_bg")
        self.DEATH_NUMBER_BG_TEMPLATE_ID = get_template_id("death_number_bg")
        self.DEATH_NUMBER_325_TEMPLATE_ID = get_template_id("death_number_325")
        self.DEATH_NUMBER_525_TEMPLATE_ID = get_template_id("death_number_525")
        self.ENTITY_MISSILE_TEMPLATE_ID = get_template_id("entity_missile")
        self.GUI_TEMPLATE_ID = get_template_id("gui_colored_background")
        self.GUI_TEXT_TEMPLATE_ID = get_template_id("gui_text_score")
        self.GUI_COMMA_TEMPLATE_ID = get_template_id("gui_score_comma")
        self.GUI_DIGITS_TEMPLATE_ID = get_template_id("gui_score_digits")
        self.PLAYER_TEMPLATE_ID = get_template_id("player")
        self.PLAYER_MISSILE_TEMPLATE_ID = get_template_id("player_missile")
        self.DENSEPACK_TEMPLATE_ID = get_template_id("densepack_frames")
        self.FORCEFIELD_TEMPLATE_ID = get_template_id("forcefield")
        self.DETONATOR_TEMPLATE_ID = get_template_id("detonator")
        self.ENERGY_POD_TEMPLATE_ID = get_template_id("energy_pod")

        rms = self.SHAPE_MASKS["radar_mortar_frames"]
        self.SPRITE_RADAR_MORTAR = jnp.concatenate(
            [
                jnp.repeat(rms[0][None], self.consts.RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
                jnp.repeat(rms[1][None], self.consts.RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
                jnp.repeat(rms[2][None], self.consts.RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
                jnp.repeat(rms[1][None], self.consts.RADAR_MORTAR_SPRITE_ANIMATION_SPEED, axis=0),
            ]
        )

        bbs = self.SHAPE_MASKS["byte_bat_frames"]
        self.SPRITE_BYTE_BAT = jnp.concatenate(
            [
                jnp.repeat(bbs[0][None], self.consts.BYTE_BAT_ANIMATION_SPEED, axis=0),
                jnp.repeat(bbs[1][None], self.consts.BYTE_BAT_ANIMATION_SPEED, axis=0),
                jnp.repeat(bbs[2][None], self.consts.BYTE_BAT_ANIMATION_SPEED, axis=0),
                jnp.repeat(bbs[1][None], self.consts.BYTE_BAT_ANIMATION_SPEED, axis=0),
            ]
        )

        rmus = self.SHAPE_MASKS["rock_muncher_frames"]
        self.SPRITE_ROCK_MUNCHER = jnp.concatenate(
            [
                jnp.repeat(rmus[0][None], self.consts.ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
                jnp.repeat(rmus[1][None], self.consts.ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
                jnp.repeat(rmus[2][None], self.consts.ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
                jnp.repeat(rmus[1][None], self.consts.ROCK_MUNCHER_ANIMATION_SPEED, axis=0),
            ]
        )

        # Convenient sprite aliases for palette-indexed masks
        self.SPRITE_PLAYING_FIELD_BG = self.SHAPE_MASKS["playing_field_bg"]
        self.SPRITE_PLAYING_FIELD_SMALL_BG = self.SHAPE_MASKS["playing_field_small_bg"]
        self.SPRITE_LOWER_DEATH_SPRITES = self.SHAPE_MASKS["lower_death_frames"]
        self.SPRITE_UPPER_DEATH_SPRITES = self.SHAPE_MASKS["upper_death_frames"]
        self.SPRITE_DEATH_NUMBER_BG = self.SHAPE_MASKS["death_number_bg"]
        self.SPRITE_DEATH_NUMBER_325 = self.SHAPE_MASKS["death_number_325"]
        self.SPRITE_DEATH_NUMBER_525 = self.SHAPE_MASKS["death_number_525"]
        self.SPRITE_ENTITY_MISSILE = self.SHAPE_MASKS["entity_missile"]
        self.SPRITE_FORCEFIELD = self.SHAPE_MASKS["forcefield"]
        self.SPRITE_DENSEPACK = self.SHAPE_MASKS["densepack_frames"]
        self.SPRITE_DETONATOR = self.SHAPE_MASKS["detonator"]
        self.SPRITE_6507 = self.SHAPE_MASKS["detonator_6507"]
        self.SPRITE_ENERGY_POD = self.SHAPE_MASKS["energy_pod"]
        self.SPRITE_UPPER_BROWN_BG = self.SHAPE_MASKS["upper_brown_bg"]
        self.SPRITE_LOWER_BROWN_BG = self.SHAPE_MASKS["lower_brown_bg"]
        self.SPRITE_GRAY_GUI_BG = self.SHAPE_MASKS["gray_gui_bg"]
        self.SPRITE_GUI_COLORED_BACKGROUND = self.SHAPE_MASKS["gui_colored_background"]
        self.SPRITE_GUI_BLACK_BACKGROUND = self.SHAPE_MASKS["gui_black_background"]
        self.SPRITE_GUI_TEXT_SCORE = self.SHAPE_MASKS["gui_text_score"]
        self.SPRITE_GUI_TEXT_ENERGY = self.SHAPE_MASKS["gui_text_energy"]
        self.SPRITE_GUI_TEXT_SHIELDS = self.SHAPE_MASKS["gui_text_shields"]
        self.SPRITE_GUI_TEXT_DTIME = self.SHAPE_MASKS["gui_text_dtime"]
        self.SPRITE_GUI_SCORE_DIGITS = self.SHAPE_MASKS["gui_score_digits"]
        self.SPRITE_GUI_SCORE_COMMA = self.SHAPE_MASKS["gui_score_comma"]
        self.SPRITE_PLAYER = self.SHAPE_MASKS["player"]
        self.SPRITE_PLAYER_MISSILE = self.SHAPE_MASKS["player_missile"]
        self.SPRITE_LOWER_MOUNTAIN = self.SHAPE_MASKS["lower_mountain"]
        self.SPRITE_UPPER_MOUNTAIN = self.SHAPE_MASKS["upper_mountain"]
        self.SPRITE_BLACK_STRIPE = self.SHAPE_MASKS["black_stripe"]
        self.SPRITE_HOMING_MISSILE = self.SHAPE_MASKS["homing_missile"]

        # --- Add Aliases for Flip Offsets ---
        self.OFFSET_LOWER_DEATH = self.FLIP_OFFSETS["lower_death_frames"]
        self.OFFSET_UPPER_DEATH = self.FLIP_OFFSETS["upper_death_frames"]
        self.OFFSET_DEATH_BG = self.FLIP_OFFSETS["death_number_bg"]
        self.OFFSET_DEATH_325 = self.FLIP_OFFSETS["death_number_325"]
        self.OFFSET_DEATH_525 = self.FLIP_OFFSETS["death_number_525"]
        self.OFFSET_RADAR_MORTAR = self.FLIP_OFFSETS["radar_mortar_frames"]
        self.OFFSET_ENTITY_MISSILE = self.FLIP_OFFSETS["entity_missile"]
        self.OFFSET_BYTE_BAT = self.FLIP_OFFSETS["byte_bat_frames"]
        self.OFFSET_ROCK_MUNCHER = self.FLIP_OFFSETS["rock_muncher_frames"]
        self.OFFSET_HOMING_MISSILE = self.FLIP_OFFSETS["homing_missile"]
        self.OFFSET_FORCEFIELD = self.FLIP_OFFSETS["forcefield"]
        self.OFFSET_DENSEPACK = self.FLIP_OFFSETS["densepack_frames"]
        self.OFFSET_DETONATOR = self.FLIP_OFFSETS["detonator"]
        self.OFFSET_6507 = self.FLIP_OFFSETS["detonator_6507"]
        self.OFFSET_ENERGY_POD = self.FLIP_OFFSETS["energy_pod"]
        self.OFFSET_LOWER_MOUNTAIN = self.FLIP_OFFSETS["lower_mountain"]
        self.OFFSET_UPPER_MOUNTAIN = self.FLIP_OFFSETS["upper_mountain"]
        self.OFFSET_PLAYER = self.FLIP_OFFSETS["player"]
        self.OFFSET_PLAYER_MISSILE = self.FLIP_OFFSETS["player_missile"]

        # Color lookup table (name -> palette ID)
        self.color_ids = {}
        for name, rgba in self.consts.PROCEDURAL_COLORS.items():
            rgb_key = (rgba[0], rgba[1], rgba[2])
            # All procedural colors are part of the palette; fall back to 0 if missing.
            self.color_ids[name] = int(self.COLOR_TO_ID.get(rgb_key, 0))

    def _colorize_mask(self, mask: jnp.ndarray, color_id: jnp.ndarray) -> jnp.ndarray:
        """Return a new mask where all non-transparent pixels become `color_id`."""
        color_id = jnp.asarray(color_id, dtype=mask.dtype)
        transparent = jnp.asarray(self.jr.TRANSPARENT_ID, dtype=mask.dtype)
        return jnp.where(mask != transparent, color_id, mask)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: LaserGatesState):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        jru = self.jr

        def color_id(name: str) -> jnp.ndarray:
            return jnp.int32(self.color_ids[name])

        SPRITE_PLAYING_FIELD_BG = self.SPRITE_PLAYING_FIELD_BG
        SPRITE_PLAYING_FIELD_SMALL_BG = self.SPRITE_PLAYING_FIELD_SMALL_BG
        SPRITE_RADAR_MORTAR = self.SPRITE_RADAR_MORTAR
        SPRITE_BYTE_BAT = self.SPRITE_BYTE_BAT
        SPRITE_ROCK_MUNCHER = self.SPRITE_ROCK_MUNCHER
        SPRITE_LOWER_DEATH_SPRITES = self.SPRITE_LOWER_DEATH_SPRITES
        SPRITE_UPPER_DEATH_SPRITES = self.SPRITE_UPPER_DEATH_SPRITES
        SPRITE_DEATH_NUMBER_BG = self.SPRITE_DEATH_NUMBER_BG
        SPRITE_DEATH_NUMBER_325 = self.SPRITE_DEATH_NUMBER_325
        SPRITE_DEATH_NUMBER_525 = self.SPRITE_DEATH_NUMBER_525
        SPRITE_ENTITY_MISSILE = self.SPRITE_ENTITY_MISSILE
        SPRITE_FORCEFIELD = self.SPRITE_FORCEFIELD
        SPRITE_DENSEPACK = self.SPRITE_DENSEPACK
        SPRITE_DETONATOR = self.SPRITE_DETONATOR
        SPRITE_6507 = self.SPRITE_6507
        SPRITE_ENERGY_POD = self.SPRITE_ENERGY_POD
        SPRITE_UPPER_BROWN_BG = self.SPRITE_UPPER_BROWN_BG
        SPRITE_LOWER_BROWN_BG = self.SPRITE_LOWER_BROWN_BG
        SPRITE_GRAY_GUI_BG = self.SPRITE_GRAY_GUI_BG
        SPRITE_GUI_COLORED_BACKGROUND = self.SPRITE_GUI_COLORED_BACKGROUND
        SPRITE_GUI_BLACK_BACKGROUND = self.SPRITE_GUI_BLACK_BACKGROUND
        SPRITE_GUI_TEXT_SCORE = self.SPRITE_GUI_TEXT_SCORE
        SPRITE_GUI_TEXT_ENERGY = self.SPRITE_GUI_TEXT_ENERGY
        SPRITE_GUI_TEXT_SHIELDS = self.SPRITE_GUI_TEXT_SHIELDS
        SPRITE_GUI_TEXT_DTIME = self.SPRITE_GUI_TEXT_DTIME
        SPRITE_GUI_SCORE_DIGITS = self.SPRITE_GUI_SCORE_DIGITS
        SPRITE_GUI_SCORE_COMMA = self.SPRITE_GUI_SCORE_COMMA
        SPRITE_PLAYER = self.SPRITE_PLAYER
        SPRITE_PLAYER_MISSILE = self.SPRITE_PLAYER_MISSILE
        SPRITE_LOWER_MOUNTAIN = self.SPRITE_LOWER_MOUNTAIN
        SPRITE_UPPER_MOUNTAIN = self.SPRITE_UPPER_MOUNTAIN
        SPRITE_BLACK_STRIPE = self.SPRITE_BLACK_STRIPE
        SPRITE_HOMING_MISSILE = self.SPRITE_HOMING_MISSILE

        def get_death_sprite_index(death_timer: jnp.ndarray, total_duration: int) -> jnp.ndarray:
            sprite_length = total_duration // 12
            clamped_timer = jnp.clip(death_timer, 1, total_duration)
            index = 12 - (clamped_timer - 1) // sprite_length
            return index

        def animation_frame(frames: jnp.ndarray) -> jnp.ndarray:
            frame_count = frames.shape[0]
            frame_idx = jnp.mod(state.step_counter, frame_count)
            return jax.lax.dynamic_index_in_dim(frames, frame_idx, axis=0, keepdims=False)

        # -------- Render playing field background --------

        # Playing field background, color adjusts if player collision
        pfb_t = jnp.clip(255 * jnp.exp(-self.consts.PLAYING_FILED_BG_COLOR_FADE_SPEED * (255 - state.animation_timer)), 0, 255)
        collision_color_active = pfb_t > 128
        playing_field_color_id = jnp.where(
            collision_color_active,
            color_id("PFB_COLLISION_COLOR"),
            color_id("PFB_BASE_COLOR"),
        )
        playing_field_bg = self._colorize_mask(SPRITE_PLAYING_FIELD_BG, playing_field_color_id)
        death_number_bg_colored = self._colorize_mask(SPRITE_DEATH_NUMBER_BG, playing_field_color_id)
        death_number_325_colored = self._colorize_mask(SPRITE_DEATH_NUMBER_325, color_id("DEATH_NUMBER_COLOR"))
        death_number_525_colored = self._colorize_mask(SPRITE_DEATH_NUMBER_525, color_id("DEATH_NUMBER_COLOR"))
        raster = jru.render_at(
            raster,
            0,
            19,
            playing_field_bg,
        )
        playing_field_small_bg = self._colorize_mask(SPRITE_PLAYING_FIELD_SMALL_BG, playing_field_color_id)
        # --- FIX 1: Use render_at_clipped for scrolling elements ---
        raster = jru.render_at_clipped(
            raster,
            0,
            19,
            playing_field_small_bg,
            flip_offset=self.FLIP_OFFSETS["playing_field_small_bg"]
        )
        raster = jru.render_at_clipped(
            raster,
            0,
            80,
            playing_field_small_bg,
            flip_offset=self.FLIP_OFFSETS["playing_field_small_bg"]
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
                                    death_number_bg_colored, # Black background mimicing playing field bg for number
                                ),
                               rmu_x,
                               rmu_y + 2,
                               death_number_325_colored, # Number showing the score
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
                                    death_number_bg_colored, # Black background mimicing playing field bg for number
                                ),
                                hm_x,
                                hm_y + 1,
                                death_number_525_colored, # Number showing the score
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


        # -------- Render Radar Mortar --------

        # Normal radar mortar
        radar_mortar_frame = animation_frame(SPRITE_RADAR_MORTAR)
        radar_mortar_blue = self._colorize_mask(radar_mortar_frame, color_id("RADAR_MORTAR_COLOR_BLUE"))
        radar_mortar_gray = self._colorize_mask(radar_mortar_frame, color_id("RADAR_MORTAR_COLOR_GRAY"))
        radar_mortar_frame = jnp.where(
            state.entities.radar_mortar_state.y == self.consts.RADAR_MORTAR_SPAWN_BOTTOM_Y,
            radar_mortar_blue,
            radar_mortar_gray,
        )

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
        radar_missile_sprite = self._colorize_mask(SPRITE_ENTITY_MISSILE, color_id("RADAR_MORTAR_MISSILE_COLOR"))

        raster = jnp.where(
            jnp.logical_and(should_render_rock_muncher_missile, state.entities.radar_mortar_state.is_in_current_event),
               jru.render_at(
                   raster,
                   state.entities.radar_mortar_state.missile_x,
                   state.entities.radar_mortar_state.missile_y,
                   radar_missile_sprite,
                   flip_horizontal=state.entities.radar_mortar_state.missile_direction[0] < 0,
               ),
               # Case: not in event -> do not render
           raster
           )


        # -------- Render Byte Bat --------

        # Normal Byte Bat
        byte_bat_frame = animation_frame(SPRITE_BYTE_BAT)
        byte_bat_frame = self._colorize_mask(byte_bat_frame, color_id("BYTE_BAT_COLOR"))

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
        rock_muncher_frame = animation_frame(SPRITE_ROCK_MUNCHER)

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
        rock_muncher_missile_sprite = self._colorize_mask(SPRITE_ENTITY_MISSILE, color_id("ROCK_MUNCHER_MISSILE_COLOR"))

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

        # Use the forcefield mask directly (already in correct orientation)
        base_forcefield_mask = SPRITE_FORCEFIELD
        forcefield_flip_offset = self.OFFSET_FORCEFIELD

        ff_state = state.entities.forcefield_state

        # Despawn earlier
        move_left = self.consts.FORCEFIELD_SIZE[0]
        render_x0 = jnp.where(ff_state.x0 <= 0, ff_state.x0 - move_left, ff_state.x0)
        render_x1 = jnp.where(ff_state.x1 <= 0, ff_state.x1 - move_left, ff_state.x1)
        render_x2 = jnp.where(ff_state.x2 <= 0, ff_state.x2 - move_left, ff_state.x2)
        render_x3 = jnp.where(ff_state.x3 <= 0, ff_state.x3 - move_left, ff_state.x3)
        render_x4 = jnp.where(ff_state.x4 <= 0, ff_state.x4 - move_left, ff_state.x4)
        render_x5 = jnp.where(ff_state.x5 <= 0, ff_state.x5 - move_left, ff_state.x5)

        ff_cycle_length = self.FORCEFIELD_COLOR_IDS.shape[0]
        ff_color_idx = jnp.mod(state.step_counter, ff_cycle_length)
        target_ff_color_id = self.FORCEFIELD_COLOR_IDS[ff_color_idx]

        forcefield_mask = self._colorize_mask(base_forcefield_mask, target_ff_color_id)

        raster = jnp.where(
            jnp.logical_and(ff_state.is_in_current_event, jnp.logical_and(ff_state.is_alive, ff_state.flash_on)),
            jru.render_at_clipped(
                jru.render_at_clipped(
                    jru.render_at_clipped(
                        jru.render_at_clipped(
                            jru.render_at_clipped(
                                jru.render_at_clipped(
                                    raster,
                                    render_x0,
                                    ff_state.y0,
                                    forcefield_mask,
                                    flip_offset=forcefield_flip_offset,
                                ),
                                render_x1,
                                ff_state.y1,
                                forcefield_mask,
                                flip_vertical=True,
                                flip_offset=forcefield_flip_offset,
                            ),
                            render_x2,
                            ff_state.y2,
                            forcefield_mask,
                            flip_offset=forcefield_flip_offset,
                        ),
                        render_x3,
                        ff_state.y3,
                        forcefield_mask,
                        flip_vertical=True,
                        flip_offset=forcefield_flip_offset,
                    ),
                    render_x4,
                    ff_state.y4,
                    forcefield_mask,
                    flip_offset=forcefield_flip_offset,
                ),
                render_x5,
                ff_state.y5,
                forcefield_mask,
                flip_vertical=True,
                flip_offset=forcefield_flip_offset,
            ),
            raster,
        )

        # -------- Render Densepack --------

        dp_state = state.entities.dense_pack_state
        dp_x, dp_upmost_y, dp_height = dp_state.x, dp_state.upmost_y, self.consts.DENSEPACK_NORMAL_PART_SIZE[1]

        # select correct sprites based on broken_states
        densepack_correct_sprites = SPRITE_DENSEPACK[dp_state.broken_states]
        densepack_masks = jax.vmap(
            lambda sp: self._colorize_mask(sp, color_id("DENSEPACK_COLOR"))
        )(densepack_correct_sprites)

        def render_densepack_parts(raster):
            idx = jnp.arange(self.consts.DENSEPACK_NUMBER_OF_PARTS, dtype=jnp.int32)
            ys = dp_upmost_y + idx * dp_height

            def step(r, args):
                i, y = args
                mask = densepack_masks[i]
                return jru.render_at(r, dp_x, y, mask), None

            raster_out, _ = jax.lax.scan(step, raster, (idx, ys))
            return raster_out

        raster = jnp.where(
            jnp.logical_and(dp_state.is_in_current_event, dp_state.is_alive),
            render_densepack_parts(raster),
            raster
        )

        # -------- Render Detonator --------

        detonator_colored = self._colorize_mask(SPRITE_DETONATOR, color_id("DETONATOR_COLOR"))

        raster = jnp.where(
            jnp.logical_and(dn_state.is_in_current_event, dn_state.is_alive),
            jru.render_at(
                jru.render_at(
                    raster,
                    dn_state.x,
                    dn_state.y,
                    detonator_colored,
                ),
                dn_state.x + 2,
                dn_state.y + 25,
                SPRITE_6507,
            ),
            raster
        )

        # -------- Render Energy Pod --------

        energy_pod_green = self._colorize_mask(SPRITE_ENERGY_POD, color_id("ENERGY_POD_COLOR_GREEN"))
        energy_pod_gray = self._colorize_mask(SPRITE_ENERGY_POD, color_id("ENERGY_POD_COLOR_GRAY"))
        energy_pod_colored = jnp.where(
            ep_state.animation_timer < (self.consts.ENERGY_POD_ANIMATION_SPEED // 2),
            energy_pod_green,
            energy_pod_gray,
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

        sprite_gui_colored_background_blue = self._colorize_mask(SPRITE_GUI_COLORED_BACKGROUND, color_id("GUI_COLORED_BACKGROUND_COLOR_BLUE")) # For Score
        sprite_gui_colored_background_green = self._colorize_mask(SPRITE_GUI_COLORED_BACKGROUND, color_id("GUI_COLORED_BACKGROUND_COLOR_GREEN")) # For Energy, Shields and Dtime
        sprite_gui_colored_background_beige = self._colorize_mask(SPRITE_GUI_COLORED_BACKGROUND, color_id("GUI_COLORED_BACKGROUND_COLOR_BEIGE"))
        sprite_gui_colored_background_gray = self._colorize_mask(SPRITE_GUI_COLORED_BACKGROUND, color_id("GUI_COLORED_BACKGROUND_COLOR_GRAY"))
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

        required_text_and_bar_color_id = color_id("GUI_TEXT_COLOR_GRAY")
        gui_text_score = self._colorize_mask(SPRITE_GUI_TEXT_SCORE, required_text_and_bar_color_id)
        gui_text_energy = self._colorize_mask(SPRITE_GUI_TEXT_ENERGY, required_text_and_bar_color_id)
        gui_text_shields = self._colorize_mask(SPRITE_GUI_TEXT_SHIELDS, required_text_and_bar_color_id)
        gui_text_dtime = self._colorize_mask(SPRITE_GUI_TEXT_DTIME, required_text_and_bar_color_id)

        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            score_col_bg_y + 2,
            gui_text_score,
        )

        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            energy_col_bg_y + 2,
            gui_text_energy,
        )

        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            shields_col_bg_y + 2,
            gui_text_shields,
        )

        raster = jru.render_at(
            raster,
            self.consts.GUI_X_BASE + self.consts.GUI_BLACK_BACKGROUND_X_OFFSET + 5,
            dtime_col_bg_y + 2,
            gui_text_dtime,
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
            required_text_and_bar_color_id, # color of filled part
            color_id("GUI_BAR_EMPTY_COLOR") # color of unfilled part
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
            required_text_and_bar_color_id, # color of filled part
            color_id("GUI_BAR_EMPTY_COLOR") # color of unfilled part
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
            required_text_and_bar_color_id, # color of filled part
            color_id("GUI_BAR_EMPTY_COLOR") # color of unfilled part
        )

        # Score ---------------

        # digits of score
        score_array = jru.int_to_digits(state.score, 6) # Convert integer to array with its digits

        recolor_single = lambda sprite_idx: self._colorize_mask(sprite_idx, required_text_and_bar_color_id)
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
        gui_score_comma = self._colorize_mask(SPRITE_GUI_SCORE_COMMA, required_text_and_bar_color_id)
        raster = jnp.where(
            state.score > 999,
            jru.render_at(
                raster,
                base_x - 14,
                score_col_bg_y + 8,
                gui_score_comma,
            ),
            raster,
        )


        # -------- Render player --------
        timer = state.animation_timer.astype(jnp.int32) - (255 - self.consts.PLAYER_COLOR_CHANGE_DURATION)
        player_normal = self._colorize_mask(SPRITE_PLAYER, color_id("PLAYER_NORMAL_COLOR"))
        player_collision = self._colorize_mask(SPRITE_PLAYER, color_id("PLAYER_COLLISION_COLOR"))
        player_sprite = jnp.where(timer <= 0, player_normal, player_collision)
        raster = jru.render_at(
            raster,
            state.player_x,
            state.player_y,
            player_sprite,
            flip_horizontal=state.player_facing_direction < 0,
        )

        # -------- Render player missile --------

        colored_player_missile = self._colorize_mask(SPRITE_PLAYER_MISSILE, color_id("PLAYER_MISSILE_BASE_COLOR"))
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
        # --- FIX 1: Use render_at_clipped for scrolling mountains ---

        # Lower mountains
        raster = jru.render_at_clipped(
            raster,
            state.lower_mountains.x1,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
            flip_offset=self.OFFSET_LOWER_MOUNTAIN
        )

        raster = jru.render_at_clipped(
            raster,
            state.lower_mountains.x2,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
            flip_offset=self.OFFSET_LOWER_MOUNTAIN
        )

        raster = jru.render_at_clipped(
            raster,
            state.lower_mountains.x3,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
            flip_offset=self.OFFSET_LOWER_MOUNTAIN
        )

        # Upper mountains
        raster = jru.render_at_clipped(
            raster,
            state.upper_mountains.x1,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
            flip_offset=self.OFFSET_UPPER_MOUNTAIN
        )

        raster = jru.render_at_clipped(
            raster,
            state.upper_mountains.x2,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
            flip_offset=self.OFFSET_UPPER_MOUNTAIN
        )

        raster = jru.render_at_clipped(
            raster,
            state.upper_mountains.x3,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
            flip_offset=self.OFFSET_UPPER_MOUNTAIN
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

        return self.jr.render_from_palette(raster, self.PALETTE)