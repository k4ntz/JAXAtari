# We are Implementing the Game Alien and Montezuma's Revenge
# We are a Group of 4: 
# 
# 
# Dennis Breder	
# Christos Toutoulas	
# David Grguric
# Niklas Ihm
#
#
import numpy as np
import os
import gc
import array

from functools import partial
from typing import NamedTuple, Tuple, Any, Callable
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from jaxatari.renderers import JAXGameRenderer

from jaxatari import spaces

from jaxatari.rendering import  jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment
from jaxatari.environment import JAXAtariAction

from typing import NamedTuple, Tuple, List, Type
import jax.lax as lax
import jax
from typing import Dict, List
from enum import Enum
import numpy as np
from jax import Array as jArray
import jaxatari.spaces as spaces

np.set_printoptions(threshold=np.inf)
from jax import random, Array
from jaxatari.games.jax_montezuma_utils import SANTAH, Room, PyramidLayout, RoomConnectionDirections, NamedTupleFieldType, loadFrameAddAlpha, load_collision_map
from jaxatari.games.jax_montezuma_enums_and_nts import *
from jaxatari.games.jax_montezuma_layouts import *


# Named Tuple defining behavior for all enemies of type "rolling skull"
class RollingSkullOrSpiderBehavior(NamedTuple):
    move_every_nth_frame: int = 3 # How often the enemy moves
    sprite_update_to_movement_offset: int = 0 # How many sprites after the last movement the sprite is updated
    collision_size: int = (4, 9) # What the size of the enemy hitbox is
    sprite_size: int = (8, 13) # Size of the enemy sprite
    sprite_offset: int = (-2, -4) # The offset w.r.t. the actual position with which the sprite is rendered
    num_sprites: int = 16 # How many sprites are in the anymation cycle
    animate_n_sprites_after_movement: int = 7 # The offset off the anymation frame to the movement frame
    movement_distance: int = 1 # How many pixels at a time the enemy moves

ROLLING_SKULL_BEHAVIOR = RollingSkullOrSpiderBehavior()


# Rolling skull & spider share a globel-behavior enum, 
# as they have the same moveset.
SPIDER_BEHAVIOR = RollingSkullOrSpiderBehavior(
    move_every_nth_frame=3, 
    sprite_update_to_movement_offset=0,
    collision_size=(4, 9),
    sprite_size=(8, 11), 
    sprite_offset=(-2, -2),
    num_sprites=2, 
    animate_n_sprites_after_movement=7, 
    movement_distance=1
)

    
    
# Bounceskull has a seperate behavior tuple
class BounceSkullBehavior(NamedTuple):
    #12, 23, 3
    move_every_ntgh_frame: int = 3 # How often the bounce skull moves
    collision_size: jArray = (19, 10) # Size of the bounce skull collision map
    sprite_size: jArray = (23, 12) 
    sprite_offset: jArray = (-2, 0)
    x_movement_distance: int = 1 # How many pixels the bounce skull moves per step.
    bounce_pattern: jArray = jnp.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, -3, -3, -3, -3, -3, -3, -3, -3, -3], jnp.int32)  
    # Bounce pattern of the skull. Can be adjusted to allow the player to pass below the skull more easily.
    bounce_pattern_length: int = 22
    # Stated & actual pattern length need to match. 
    
B_SCULL_BEHAVIOR = BounceSkullBehavior()

# Snake has no movement, so behavior contains only render options.
class SnakeBehavior(NamedTuple):
    frame_length = 20
    collision_size: jArray = jnp.array([1, 9], jnp.int32)
    sprite_size: jArray = jnp.array([7, 13], jnp.int32)
    sprite_offset: jArray = jnp.array([-3, -4], jnp.int32)
SNAKE_BEHAVIOR = SnakeBehavior()

class MontezumaConstants(NamedTuple):
    
    WIDTH: int = 160 # Window width
    HEIGHT: int = 210 # Window height. This is different to the height of the play-area.
    INITIAL_PLAYER_X: int = 77 # Start position of the player
    INITIAL_PLAYER_Y: int = 26
    INITIAL_PLAYER_VELOCITY: int = 1 # Velocity can be changed to make gameplay easier/ harder
    RENDER_SCALE_FACTOR: int = 4
    PLAYER_WIDTH: int = 7 # Dimensions of the player character, this determines the size of the hitbox
    PLAYER_HEIGHT: int = 20 
    FALLING_VELOCITY: int = 2 # Falling velocity is constant, as in the original game
    JUMP_FRAMES:int = 20 # How long a jump lasts
    JUMP_Y_OFFSETS = jnp.array([3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0, -1, -1, -2, -2, -2, -3, -3, -3]) 
    # The jump pattern of the player. Positive numbers mean, that the player goes up.
    SPRITE__F_DICT = { # Deprecated (hopefully)
        "player": "player_sprite.npy", 
        "player_collision": "player_collision_map.npy",
        "gem": "gem.npy",
        "hammer": "hammer.npy",
        "key": "key.npy",
        "sword": "sword.npy",
        "torch_1": "torch_1.npy",
        "torch_2": "torch_2.npy",
        "life_sprite": "life_sprite.npy",
        "0": "0.npy",
        "door":"door.npy",
        "other_dropout_floor":"other_dropout_floor.npy",
        "pitroom_dropout_floor":"pitroom_dropout_floor.npy",
        "conveyor_belt":"conveyor_belt.npy",
        "digit_none":"none.npy",
        "digit_0":"0.npy",
        "digit_1":"1.npy",
        "digit_2":"2.npy",
        "digit_3":"3.npy",
        "digit_4":"4.npy",
        "digit_5":"5.npy",
        "digit_6":"6.npy",
        "digit_7":"7.npy",
        "digit_8":"8.npy",
        "digit_9":"9.npy"
            }
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    LAZER_BARRIER_STRIPE_DISTANCE = 4 # Distance between horizontal stripes in lazer barrier
    LAZER_BARRIER_SCROLL_STEP = 2 # Step width with which the lazer barrier scrolls
    LAZER_BARRIER_ANIMATION_CYCLE = 4 # Animation cycle for the lazer barrier
    LAZER_BARRIER_WIDTH = 4
    DOOR_WIDTH: int = 4 # Doors have standard width & height to allow for more efficient rendering
    DOOR_HEIGHT: int = 38
    ROPE_HORIZONTAL_OFFSET = jnp.array([-2], dtype=jnp.int32) # Horizontal of the player relative to the rope if in the "on_rope" state
    ROPE_FALL_VERTICAL_OFFSET = jnp.array([2], dtype=jnp.int32) # Offset how many pixels above the rope end the player starts to fall.
    ROPE_TOP_ENTRY_SNAP_ON_OFFSET: int = 2 # Y position relative to the rope top to which the player is teleported when getting onto a rope from above
    # Positive value -> Below the rope top
    # Negative value -> Above the rope top
    ROPE_TOP_EXIT_DISTANCE: int = -12 # How far below the rope top you need to be to get off the rope
    # Positive value -> you get off the rope below the top
    # Negative value-> you get off the rope above the top.
    ROPE_COLLISION_X_OFFSET: int = 2# If we detect collision with the rope across the whole width of the player, 
    # This causes issues if we place a rope near the edge of the platform & fall down it
    ROPE_COLLISION_WIDTH: int = 3
    # How wide the sprite is that we use for colision with the rope.
    ROPE_TOP_ENTRANCE_VERTICAL_SNAP_DISTANCE: int = 5
    # Through how many floor pixels the player is allowed to snap to enter a rope from the top.
    
    INPUT_DISABLED_FRAMES_AFTER_JUMP = 1
    PIT_COLOR_PATTERN = jnp.array([[2, 1, 0, 1, 2, 1, 0, 1, 2],
                                   [1, 0, 1, 2, 1, 0, 1, 2, 3],
                                   [0, 1, 2, 1, 0, 1, 2, 3, 2],
                                   [1, 2, 1, 0, 1, 2, 3, 2, 1]],jnp.int32)
    
    # All the ladder attributes
    LADDER_SIDES_LINE_WIDTH: int = 4 #Line width for the sides of the ladder
    LADDER_RUNG_WIDTH: int = 2 # The line width of the individual ladder rungs
    LADDER_SIDE_PADDING: int = 4 # How much cosmetic padding is added to the sides of the ladder
    LADDER_BOTTOM_TOP_PADDING: int = 0 # How much cosmetic padding is added to the top & bottom of the ladder.
    RUNG_START_OFFSET_VERTICAL: int = 4 # Offset of the first rung from the ladder start
    RUNG_SPACING: int = 7 # Spacing between the rungs
    # The fields that dictate how you can enter/ exit a ladder:
    LADDER_TOP_INTERFACE_ZONE_Y_REACH_WHILE_ON: int = 5
    # If you are ON the Ladder: how large your distance to the ladder top y coordinate may be for a leave 
    # input to trigger leaving the ladder. Computet in relation to the player's feet
    LADDER_BOTTOM_INTERFACE_ZONE_Y_REACH_WHILE_ON: int = 5
    # If you are on the ladder: How large your distance to the ladder bottom y coordinate may be for a "leave" 
    # Input to result in you leaving the ladder. Compute in relation to the player's feet.
    LADDER_INTERFACE_ZONE_REACH_OFF_LADDER: int = 5
    # How long THE DISTANCE OF YOUR FEET may be be to either a ladder top or a ladder bottom for a "climb" input to 
    # result in you climbing onto the ladder
    LADDER_ENTRANCE_TELEPORT_DISTANCE: int = 6
    # How high up/ low down you are teleported onto the ladder when you enter/ leave it. 
    LADDER_ROOM_ENTRANCE_GRACE_PIXELS: int = 2
    # If the player enters a room via a ladder, how many pixels ABOVE the room bounds he should spawn. 
    # This is necessary for the player to not immediately leave the room again
    # End of the ladder fields
    ITEMBAR_LIFES_STARTING_X: int = 56 # x coordinate of itembar and lifes
    ITEMBAR_STARTING_Y: int = 28 # y coordinate of itembar
    LIFES_STARTING_Y: int = 15 # y coordinate of lifes
    ANIMATION_CYCLE_DURATION: int = 8 # amount of frames till the next sprite in the cycle for the Torch sprite
    # Options for rendering the score digits
    DIGIT_WIDTH: int = 7
    DIGIT_OFFSET: int = 1
    DIGIT_HEIGHT: int = 8
    SCORE_Y: int = 6
    SCORE_X: int = 56
    HAMMER_DURATION: int = 500 # How long the hammer pickup lasts. After this number of frames the item disapears
        # and you can no longer use it
    MAXIMUM_ITEM_NUMBER: int = 4 # The maximum number of items that can be held by the player at one time.
    # How many points killing an enemy gives
    ENEMY_POINTS: jArray = jnp.ones((4), jnp.int32).at[
        EnemyType.BOUNCE_SKULL.value].set(500).at[
        EnemyType.ROLL_SKULL.value].set(1000).at[
        EnemyType.SNAKE.value].set(400).at[
        EnemyType.SPIDER.value].set(300)
    # Height of the floor in the bonus room.
    BONUS_ROOM_FLOOR_Y: int = 46
    # Width of the collision sprite of the items.
    ITEM_COLLISION_PLAYER_WIDTH: int = 3
    # Which state fields are reset on player death.
    DEATH_RESET_FIELDS: List[str] = [
                    MontezumaStateFields.player_velocity.value, 
                    MontezumaStateFields.horizontal_direction.value, 
                    MontezumaStateFields.vertical_directional_input.value, 
                    MontezumaStateFields.horizontal_direction.value, 
                    MontezumaStateFields.last_horizontal_direction.value, 
                    MontezumaStateFields.last_horizontal_orientation.value, 
                    MontezumaStateFields.current_directional_input.value,### 
                    MontezumaStateFields.horizontal_falling_velocitiy.value, 
                    MontezumaStateFields.is_standing.value, 
                    MontezumaStateFields.is_jumping.value, 
                    MontezumaStateFields.is_falling.value, 
                    MontezumaStateFields.jump_counter.value,#### 
                    MontezumaStateFields.jump_input.value, 
                    MontezumaStateFields.is_climbing.value, 
                    MontezumaStateFields.is_laddering.value, 
                    MontezumaStateFields.is_on_rope.value, 
                    MontezumaStateFields.last_key_press.value, 
                    MontezumaStateFields.force_jump.value, 
                    MontezumaStateFields.disable_directional_input.value, 
                    MontezumaStateFields.queue_enable_directional_input.value, 
                    MontezumaStateFields.velocity_held.value]
    
    # During the game, freezes are triggered at various points.
    # This array contains the freeze length for each of those occasions.
    FREEZE_TYPE_LENGTHS: jArray = jnp.zeros(shape=(len(FreezeType)), dtype=jnp.int32).at[
        FreezeType.FALL_DEATH.value].set(60).at[
        FreezeType.KILLED_A_MONSTER.value].set(5).at[
        FreezeType.KILLED_BY_MONSTER.value].set(70).at[
        FreezeType.FALL_DEATH.value].set(50).at[
        FreezeType.LAZER_BARRIER_DEATH.value].set(70).at[
        FreezeType.SARLACC_PIT_DEATH.value].set(50).at[
        FreezeType.ITEM_PICKUP.value].set(5).at[
        FreezeType.DOOR_UNLOCK.value].set(5)
    
    # Initial RNG key        
    INITIAL_RNG_KEY: int = 42 
    # How wide items are
    ITEM_WIDTH: int = 7
    # X location of first GEM in the bonus room
    # Y location is always fixed.
    DEFAULT_BONUS_ROOM_GEM_X: int = 19
    # Speed of the rope climbing animations
    ROPE_CLIMBING_ANIMATION_FRAME_LENGTH: int = 5
    # Speed of the ladder climbing animation
    LADDER_CLIMBING_ANIMATION_FRAME_LENGTH: int = 5
    # speed of the walking animation
    WALKING_ANIMATION_FRAME_LENGTH: int = 5
    # Maximum height from which the player may fall before death occurs.
    MAXIMUM_ALLOWED_FALL_HEIGHT: int = 25
    # Animation speed of the death-fall animation
    DEATH_FALL_WIGGLE_FRAME_LENGTH: int = 8
    # Speed of the animation on death-by-lazer-barrier
    LAZER_BARRIER_SPLUTTER_FRAME_LENGTH: int = 8
    # Speed of the animation on death-by-enemy
    ENEMY_SPLUTTER_FRAME_LENGTH: int = 8
    # Bouncing skull enemy can be split in two. This parameter determines
    # How wide the area in the middle of the skull sprite is that is deleted when this happend
    DOUBLE_SKULL_MIDDLE_DEMARKATION_LINE_OFFSET: int = 4
    
    #COLOR STUFF
    
    ROOM_COLORS = RoomColors
    
    
    BLACK = jnp.array([0, 0, 0])
    WHITE = jnp.array([255, 255, 255])
    GEM_COLOR = jnp.array([213, 130, 74])
    HAMMER_COLOR = jnp.array([210, 182, 86])
    SWORD_COLOR = jnp.array([214, 214, 214])
    TORCH_COLOR = jnp.array([204, 216, 110])
    ITEMBAR_COLOR = jnp.array([232, 204, 99]) #also for the key item
    KEY_COLOR = ITEMBAR_COLOR
    
    DOOR_COLOR_NORMAL = ITEMBAR_COLOR
    ROPE_COLOR_NORMAL = ITEMBAR_COLOR
    ROPE_COLOR_WHITE = jnp.array([236, 236, 236])
    # Primary colors used for the room layers
    DIF_1_LAYER_1_PRIMARY = ROOM_COLORS.dif_1.l1_p
    DIF_1_LAYER_2_PRIMARY = ROOM_COLORS.dif_1.l2_p
    DIF_1_LAYER_3_PRIMARY = ROOM_COLORS.dif_1.l3_p
    DIF_1_LAYER_4_PRIMARY = ROOM_COLORS.dif_1.l4_p
    DIF_1_LAYER_1_SECONDARY = ROOM_COLORS.dif_1.l1_s
    DIF_1_LAYER_2_SECONDARY = ROOM_COLORS.dif_1.l2_s
    DIF_1_LAYER_3_SECONDARY = ROOM_COLORS.dif_1.l3_s
    DIF_1_LAYER_4_SECONDARY = ROOM_COLORS.dif_1.l4_s
    
    DIF_2_LAYER_1_PRIMARY = ROOM_COLORS.dif_2.l1_p
    DIF_2_LAYER_2_PRIMARY = ROOM_COLORS.dif_2.l2_p
    DIF_2_LAYER_3_PRIMARY = ROOM_COLORS.dif_2.l3_p
    DIF_2_LAYER_4_PRIMARY = ROOM_COLORS.dif_2.l4_p
    DIF_2_LAYER_1_SECONDARY = ROOM_COLORS.dif_2.l1_s
    DIF_2_LAYER_2_SECONDARY = ROOM_COLORS.dif_2.l2_s
    DIF_2_LAYER_3_SECONDARY = ROOM_COLORS.dif_2.l3_s
    DIF_2_LAYER_4_SECONDARY = ROOM_COLORS.dif_2.l4_s
    
    DIF_3_LAYER_1_PRIMARY = ROOM_COLORS.dif_3.l1_p
    DIF_3_LAYER_2_PRIMARY = ROOM_COLORS.dif_3.l2_p
    DIF_3_LAYER_3_PRIMARY = ROOM_COLORS.dif_3.l3_p
    DIF_3_LAYER_4_PRIMARY = ROOM_COLORS.dif_3.l4_p
    DIF_3_LAYER_1_SECONDARY = ROOM_COLORS.dif_3.l1_s
    DIF_3_LAYER_2_SECONDARY = ROOM_COLORS.dif_3.l2_s
    DIF_3_LAYER_3_SECONDARY = ROOM_COLORS.dif_3.l3_s
    DIF_3_LAYER_4_SECONDARY = ROOM_COLORS.dif_3.l4_s
    
    LASER_BARRIER_COLOR_NO_ALPHA = jnp.array([101, 111, 228])
    SARLACC_PIT_COLOR = jnp.array([210, 164, 74])
    BONUS_ROOM_COLOR = jnp.array([0, 28, 136])

    OBSTACLE_COLORS = jnp.array([BLACK,
                                 WHITE,
                                 DOOR_COLOR_NORMAL,
                                 ROPE_COLOR_NORMAL,
                                 ROPE_COLOR_WHITE,
                                 DIF_1_LAYER_1_PRIMARY,
                                 DIF_1_LAYER_2_PRIMARY,
                                 DIF_1_LAYER_3_PRIMARY,
                                 DIF_1_LAYER_4_PRIMARY,
                                 DIF_1_LAYER_1_SECONDARY,
                                 DIF_1_LAYER_2_SECONDARY,
                                 DIF_1_LAYER_3_SECONDARY,
                                 DIF_1_LAYER_4_SECONDARY,
                                 DIF_2_LAYER_1_PRIMARY,
                                 DIF_2_LAYER_2_PRIMARY,
                                 DIF_2_LAYER_3_PRIMARY,
                                 DIF_2_LAYER_4_PRIMARY,
                                 DIF_2_LAYER_1_SECONDARY,
                                 DIF_2_LAYER_2_SECONDARY,
                                 DIF_2_LAYER_3_SECONDARY,
                                 DIF_2_LAYER_4_SECONDARY,
                                 DIF_3_LAYER_1_PRIMARY,
                                 DIF_3_LAYER_2_PRIMARY,
                                 DIF_3_LAYER_3_PRIMARY,
                                 DIF_3_LAYER_4_PRIMARY,
                                 DIF_3_LAYER_1_SECONDARY,
                                 DIF_3_LAYER_2_SECONDARY,
                                 DIF_3_LAYER_3_SECONDARY,
                                 DIF_3_LAYER_4_SECONDARY
                                 ], jnp.int32)
    
    LASER_BARRIER_COLOR = jnp.array([101, 111, 228, 255], dtype=jnp.int16)
    LAYOUT = Layouts.test_layout.value

@jax.jit
def JAXATARI_ACTION_TO_MOVEMENT_DIRECTION(action: jnp.ndarray) -> jnp.ndarray:
    ret = jax.lax.switch(action, [
        lambda x: jnp.array([MovementDirection.NO_DIR.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.NO_DIR.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.UP.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.RIGHT.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.LEFT.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.DOWN.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.RIGHT.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.LEFT.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.RIGHT.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.LEFT.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.NO_DIR.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.RIGHT_UP.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.LEFT_UP.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.NO_DIR.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.NO_DIR.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.NO_DIR.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.NO_DIR.value], dtype=jnp.uint8), 
        lambda x: jnp.array([MovementDirection.NO_DIR.value], dtype=jnp.uint8)
    ], jnp.array([0]))
    return jnp.reshape(ret, (1, ))



@jax.jit
def JAXATARI_GET_JUMP_ACTION(action: jnp.ndarray) -> jnp.ndarray:
    ret = jax.lax.switch(action, [
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.YES.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.YES.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.YES.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.YES.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.NO.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.YES.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.YES.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.YES.value], dtype=jnp.uint8), 
        lambda x: jnp.array([JUMP_INPUT.YES.value], dtype=jnp.uint8)
    ], jnp.array([0]))
    return jnp.reshape(ret, (1, ))

@jax.jit
def get_horizontal_movement_direction(movement_direction: jnp.ndarray) -> jnp.ndarray:
    #
    # Takes in a movement direction and returns the matching horizontal direction.
    #
    left_matches = jnp.array([MovementDirection.LEFT.value, MovementDirection.LEFT_DOWN.value, 
                              MovementDirection.LEFT_UP.value])
    right_matches = jnp.array([MovementDirection.RIGHT.value, MovementDirection.RIGHT_DOWN.value, 
                               MovementDirection.RIGHT_UP.value])
    no_movement_directions = jnp.array([MovementDirection.NO_DIR.value, 
                                        MovementDirection.DOWN.value, 
                                        MovementDirection.UP.value])
    # Check which horizontal direction matches to the actual direction
    l_m = jnp.max(jnp.equal(left_matches, movement_direction))*1
    r_m = jnp.max(jnp.equal(right_matches, movement_direction))*2
    n_m = jnp.max(jnp.equal(no_movement_directions, movement_direction))*3
    dir_selector = l_m + r_m + n_m
    # Select the appropriate horizontal direction value
    horizontal_dir: jnp.ndarray = jax.lax.switch(dir_selector - 1, [lambda x: jnp.array([Horizontal_Direction.LEFT.value], jnp.uint8), 
                                  lambda x: jnp.array([Horizontal_Direction.RIGHT.value], jnp.uint8), 
                                  lambda x: jnp.array([Horizontal_Direction.NO_DIR.value], jnp.uint8)], 0)

    return jnp.reshape(horizontal_dir, (1, ))
    
@jax.jit
def get_vertical_movement_direction(movement_direction: jnp.ndarray) -> jnp.ndarray:
    #
    # Exactly the same as above, but only for vertical directions.
    #
    
    up_matches = jnp.array([MovementDirection.UP.value])
    down_matches = jnp.array([MovementDirection.DOWN.value])
    no_movement_directions = jnp.array([MovementDirection.NO_DIR.value, 
                                        MovementDirection.LEFT.value, 
                                        MovementDirection.RIGHT.value, 
                                        MovementDirection.LEFT_UP.value, 
                                        MovementDirection.RIGHT_UP.value, 
                                        MovementDirection.RIGHT_DOWN.value, 
                                        MovementDirection.LEFT_DOWN.value])
    l_m = jnp.max(jnp.equal(up_matches, movement_direction))*1
    r_m = jnp.max(jnp.equal(down_matches, movement_direction))*2
    n_m = jnp.max(jnp.equal(no_movement_directions, movement_direction))*3
    dir_selector = l_m + r_m + n_m
    horizontal_dir: jnp.ndarray = jax.lax.switch(dir_selector - 1, [lambda x: jnp.array([VERTICAL_Direction.UP.value], jnp.uint8), 
                                  lambda x: jnp.array([VERTICAL_Direction.DOWN.value], jnp.uint8), 
                                  lambda x: jnp.array([VERTICAL_Direction.NO_DIR.value], jnp.uint8)], 0)
    ret = jnp.reshape(horizontal_dir, (1, ))
    ret = jnp.astype(ret, jnp.int32)
    return ret




def colorize_sprite(sprite: jnp.ndarray, color: jnp.ndarray) -> jnp.ndarray:
    """
    Sets all non-black (RGB != [0,0,0]) pixels of the sprite to the given color, preserving alpha.
    sprite: (H, W, 4) jnp.ndarray
    color: (3,) jnp.ndarray, values in 0-255
    Returns: (H, W, 4) jnp.ndarray
    """
    rgb = sprite[..., :3]
    alpha = sprite[..., 3:]
    # Mask for non-black pixels (any channel > 0)
    nonblack = jnp.any(rgb != 0, axis=-1, keepdims=True) # Is this still needed?
    # Broadcast color to sprite shape
    color_broadcast = jnp.broadcast_to(color, rgb.shape)
    # Where nonblack, set to color; else keep original
    new_rgb = jnp.where(nonblack, color_broadcast, rgb)
    
    return jnp.concatenate([new_rgb, alpha], axis=-1).astype(jnp.uint8)


class JaxMontezuma(JaxEnvironment[MontezumaState, MontezumaObservation, MontezumaInfo, MontezumaConstants]):
    def __init__(self, consts: MontezumaConstants = None,
                        frameskip: int = 1, 
                        reward_funcs: list[callable]=None):
        consts = consts or MontezumaConstants()
        super().__init__(consts)
        
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs) 
        self.reward_funcs = reward_funcs

        self.action_set = [
            JAXAtariAction.NOOP,
            JAXAtariAction.FIRE,
            JAXAtariAction.UP,
            JAXAtariAction.RIGHT,
            JAXAtariAction.LEFT,
            JAXAtariAction.DOWN,
            JAXAtariAction.UPRIGHT,
            JAXAtariAction.UPLEFT,
            JAXAtariAction.DOWNRIGHT,
            JAXAtariAction.DOWNLEFT,
            JAXAtariAction.UPFIRE,
            JAXAtariAction.RIGHTFIRE,
            JAXAtariAction.LEFTFIRE,
            JAXAtariAction.DOWNFIRE,
            JAXAtariAction.UPRIGHTFIRE,
            JAXAtariAction.UPLEFTFIRE,
            JAXAtariAction.DOWNRIGHTFIRE,
            JAXAtariAction.DOWNLEFTFIRE 
        ]
        
        # 
        # All functions called from the game state that need to interact with rooms at the individual room level.
        #
        #
        # Renders the room-specific colision map onto a canvas of static size
        self.WRAPPED_AUGMENT_COLLISION_MAP: Callable[[MontezumaState], MontezumaState] = None
        
        # Renders the room sprite onto a canvas of static size
        self.WRAPPED_RENDER_ROOM_SPRITE_ONTO_CANVAS: Callable[[MontezumaState], MontezumaState] = None
        # Handles custom logic for each room that is called when the room is entered
        # This includes teleporting onto ladders, 
        # Resetting of room specific counters for animation cycles, ...
        self.WRAPPED_HANDLE_ROOM_ENTRANCE: Callable[[MontezumaState], MontezumaState] = None
        # Handles updates for various room-specific counters. Used for enemy, lazer gate and sarlacc pit logic.
        self.WRAPPED_HANDLE_COUNTER_UPDATES: Callable[[MontezumaState], MontezumaState] = None
        # Function that initiates the climbing state for both ropes & ladders
        self.WRAPPED_HANDLE_START_CLIMBING: Callable[[MontezumaState], MontezumaState] = None
        # Function that handles getting off ropes & ladders
        self.WRAPPED_HANDLE_STOP_CLIMBING: Callable[[MontezumaState], MontezumaState] = None
        # Function that handles all the enemy collision logic.
        self.WRAPPED_HANDLE_ENEMIES: Callable[[MontezumaState], MontezumaState] = None   
        
        
        self.PROTO_ROOM_LOADER: Callable[[jArray, jArray], Room] = None
        # A loader that loads a Proto Room object from the persistence storage. 
        # This loader is save to be used in functions that are non-static with the ROOM ID
        # Args: (singleton_room_index, persistence_storage) 
            
        self.WRITE_PROTO_ROOM_TO_PERSISTENCE: Callable[[jArray, Room, jArray], jnp.ndarray] = None
        # A writer that writes the fields from the proto room to persistence. 
        # Args: (singleton_index, proto-room, persistence storage)
        
        
       
        
        # Loading of miscaleneaous sprites.
        #
        #
        self.item_scores: jnp.ndarray = jnp.zeros(shape=6, dtype=jnp.uint32)
        self.item_scores = self.item_scores.at[Item_Sprites.GEM.value].set(1000)
        self.item_scores = self.item_scores.at[Item_Sprites.KEY.value].set(100)
        self.item_scores = self.item_scores.at[Item_Sprites.TORCH.value].set(3000)
        self.item_scores = self.item_scores.at[Item_Sprites.HAMMER.value].set(200)
        self.item_scores = self.item_scores.at[Item_Sprites.SWORD.value].set(100)
        
        self.sprite_path: str = os.path.join(self.consts.MODULE_DIR, "sprites", "montezuma")
        
        
        self.room_connection_map: Callable[[jax.Array, jax.Array], jax.Array] = None
         
        self.LAYOUT: PyramidLayout = None
        
        self.PROTO_ROOM_LOADER = None
        self.WRITE_PROTO_ROOM_TO_PERSISTENCE = None
        
        
        self.player_col_map: jnp.ndarray = load_collision_map(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["player_collision"]), 
                                                                       transpose=True)
        self.initial_persistence_state: jnp.ndarray = None
        self.__make_room_infra_ready()
        self.__init_room()
        self.renderer = MontezumaRenderer(consts=self.consts)
        # Vmap functions for enemy movement & animation to increase efficiency
        self.vmapped_single_enemy_movement = jax.vmap(self.handle_single_enemy_movement, in_axes=0, out_axes=0)
        self.vmapped_single_enemy_animation = jax.vmap(self._handle_single_enemy_animation, in_axes=0, out_axes=0)
        
    def _make_barrier_activation_map(self, room: VanillaRoom, barrier_tag: RoomTags.LAZER_BARRIER.value):
        """Only called during pre-jitted phase:
           Precomputes the colision map for all lazer barriers in a room.

        Args:
            room (VerticalBarrierRoom): Room for which the barrier object is generated.
        """
        barrier_canvas: jnp.ndarray = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.uint8)
        # retreive the barrier_tag from the room
        barrier_glob: GlobalLazerBarrierInfo = SANTAH.full_deserializations[GlobalLazerBarrierInfo](barrier_tag.global_barrier_info[0])
        for i in range(barrier_tag.barriers.shape[0]):
            # cast the barrier arrays to named-tuples for easier handling.
            barrier: LAZER_BARRIER = SANTAH.full_deserializations[LAZER_BARRIER](barrier_tag.barriers[i])
            # Paint in collision map for each barrier
            barrier_canvas = barrier_canvas.at[barrier.X[0]:barrier.X[0]+self.consts.LAZER_BARRIER_WIDTH, (room.vertical_offset[0] + barrier.upper_point[0]):(room.vertical_offset[0] + barrier.lower_point[0])].set(1)
        
        return barrier_canvas
    
    def _make_door_collision_map(self, room: VanillaRoom, door_tag: RoomTags.DOORS.value):
        #
        # Precompute door collision map. Only called on startup
        #
        door_collision: jnp.ndarray = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.uint8)
        door_information: RoomTags.DOORS.value = door_tag
        for d_ind in range(door_information.doors.shape[0]):
            door_arr: jArray = door_information.doors[d_ind, ...]
            # retreive single door named tuple
            door: Door = SANTAH.full_deserializations[Door](door_arr)
            # Paint in door, using array_position + 1 as value. 
            # This is necessary so that we can differentiate which doors have already been opened.
            single_door_collison_map = jnp.ones(shape=(self.consts.DOOR_WIDTH, self.consts.DOOR_HEIGHT+2), dtype=jnp.uint8) * (d_ind+1)
            door_collision = jax.lax.dynamic_update_slice(operand=door_collision,
                                                                  update=single_door_collison_map,
                                                                  start_indices=(door.x[0],door.y[0]+ room.vertical_offset[0]))
        return door_collision
    
    def _make_ladder_render_map(self, room: VanillaRoom, ladder_tag: RoomTags.LADDERS.value) -> jArray:
        #
        # Precompute the sprite for the ladders.
        # This is fairly complicated, as the ladder visuals are fully determined by the constants.
        #
        ladder_canvas: jArray = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT, 4), dtype=jnp.uint8)
        ladders: jArray = ladder_tag.ladders
        for l_ind in range(ladders.shape[0]):
            # Retreive ladder which is to be painted in.
            current_ladder: Ladder = SANTAH.full_deserializations[Ladder](ladders[l_ind, ...])
            # Background color for ladder
            b_c_val = self.consts.OBSTACLE_COLORS[current_ladder.background_color[0]]
            background_color = jnp.array([b_c_val[0],b_c_val[1],b_c_val[2],255], jnp.uint8)
            # Ladders can be rendered with transparnet background
            if current_ladder.transparent_background[0] == 1:
                background_color = jnp.array([0, 0, 0, 0], jnp.uint8)    
            background_color = jnp.reshape(background_color, (1, 1, 4))
            # Retreive foreground color. Foreground can also be transparent.
            f_c_val = self.consts.OBSTACLE_COLORS[current_ladder.foreground_color[0]]
            foreground_color = jnp.array([f_c_val[0],f_c_val[1],f_c_val[2],255], jnp.uint8)
            if current_ladder.transparent_foreground[0] == 1:
                foreground_color = jnp.array([0, 0, 0, 0], jnp.uint8) 
            foreground_color = jnp.reshape(foreground_color, (1, 1, 4))            
            # Consider padding & compute which area to color with the background color
            bg_up_left_x = max(current_ladder.left_upper_x[0] - self.consts.LADDER_SIDE_PADDING, 0)
            bg_up_left_y = max(current_ladder.left_upper_y[0] - self.consts.LADDER_BOTTOM_TOP_PADDING, 0)
            
            bg_down_right_x = min(current_ladder.right_lower_x[0] + self.consts.LADDER_SIDE_PADDING, self.consts.WIDTH)
            bg_down_right_y = min(current_ladder.right_lower_y[0] + self.consts.LADDER_BOTTOM_TOP_PADDING, room.room_collision_map.shape[1] - 1)
            
            background_area = jnp.ones(shape=(bg_down_right_x - bg_up_left_x, bg_down_right_y - bg_up_left_y, 4), dtype=jnp.uint8)
            background_area = jnp.multiply(background_area, background_color)
            # Color in the background
            ladder_canvas = ladder_canvas.at[bg_up_left_x:bg_down_right_x, room.vertical_offset[0] + bg_up_left_y : 
                        room.vertical_offset[0] + bg_down_right_y, :].set(background_area)
            
            # Now paint the sides of the ladder using the foreground color.
            left_side_rung_x_range = (current_ladder.left_upper_x[0], current_ladder.left_upper_x[0] + self.consts.LADDER_SIDES_LINE_WIDTH)
            right_side_rung_x_range = (current_ladder.right_lower_x[0] - self.consts.LADDER_SIDES_LINE_WIDTH, current_ladder.right_lower_x[0])
            side_rungs_y_range = (current_ladder.left_upper_y[0], current_ladder.right_lower_y[0])
            side_rung = jnp.ones((left_side_rung_x_range[1] - left_side_rung_x_range[0], side_rungs_y_range[1] - side_rungs_y_range[0], 4), jnp.uint8)
            side_rung = jnp.multiply(side_rung, foreground_color)
            
            ladder_canvas = ladder_canvas.at[left_side_rung_x_range[0]:left_side_rung_x_range[1], 
                                                side_rungs_y_range[0] + room.vertical_offset[0]:side_rungs_y_range[1] + room.vertical_offset[0], :].set(side_rung)
            ladder_canvas = ladder_canvas.at[right_side_rung_x_range[0]:right_side_rung_x_range[1], 
                                             side_rungs_y_range[0] + room.vertical_offset[0] :side_rungs_y_range[1] + room.vertical_offset[0]].set(side_rung)
            
            # Paint in the rungs of the ladder
            # Generate one rung sprite & place along the ladder.
            rung_sprite = jnp.ones((current_ladder.right_lower_x[0] - current_ladder.left_upper_x[0], self.consts.LADDER_RUNG_WIDTH, 4))
            rung_sprite = jnp.multiply(rung_sprite, foreground_color)
            rung_sprite = jnp.astype(rung_sprite,jnp.uint8)
            start_y: int = current_ladder.left_upper_y + self.consts.RUNG_START_OFFSET_VERTICAL
            stop_y: int = current_ladder.right_lower_y - self.consts.LADDER_RUNG_WIDTH
            c = start_y[0]
            while c < stop_y:
                ladder_canvas = ladder_canvas.at[current_ladder.left_upper_x[0]:current_ladder.right_lower_x[0], 
                                    c + room.vertical_offset[0] :c + room.vertical_offset[0] + self.consts.LADDER_RUNG_WIDTH, :].set(rung_sprite)
                c += self.consts.RUNG_SPACING
        return ladder_canvas
    
    
    def _make_ladder_tops_map(self, room: VanillaRoom, ladder_tag: RoomTags.LADDERS.value) -> jArray:
        """
        A function that paints a hitmap of all the zones in which you can enter a ladder from the top.
        Only called during startup. 
        """
        ladder_canvas: jArray = jnp.ones(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.int32)
        # Background color is -1,
        # ladder tops are painted in with their respective ID in the ladder array.
        ladder_canvas = ladder_canvas*(-1)
        ladders: jArray = ladder_tag.ladders
        for l_ind in range(ladders.shape[0]):
            current_ladder: Ladder = SANTAH.full_deserializations[Ladder](ladders[l_ind, ...])
            x_range = (current_ladder.left_upper_x[0], current_ladder.right_lower_x[0])
            self.consts.LADDER_INTERFACE_ZONE_REACH_OFF_LADDER
            # Consider the ladder interface zones, so that we can already leave the ladder
            # before we have fully reached the top.
            y_range = (max(current_ladder.left_upper_y[0] + room.vertical_offset[0] - self.consts.LADDER_INTERFACE_ZONE_REACH_OFF_LADDER, 0), 
                       min(current_ladder.left_upper_y[0] + room.vertical_offset[0] + self.consts.LADDER_INTERFACE_ZONE_REACH_OFF_LADDER, self.consts.HEIGHT))
            ladder_canvas = ladder_canvas.at[x_range[0]:x_range[1], y_range[0]:y_range[1]].set(l_ind)
        return ladder_canvas
        
        
    
    def _make_ladder_bottom_map(self, room: VanillaRoom, ladder_tag: RoomTags.LADDERS.value) -> jArray:
        """
        A function that paints a hitmap of all the zones in which you enter a ladder from the bottom.
        Exactly the same as the above function.
        Only called on startup.
        """
        ladder_canvas: jArray = jnp.ones(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.int32)
        ladder_canvas = ladder_canvas*(-1)
        ladders: jArray = ladder_tag.ladders
        for l_ind in range(ladders.shape[0]):
            current_ladder: Ladder = SANTAH.full_deserializations[Ladder](ladders[l_ind, ...])
            x_range = (current_ladder.left_upper_x[0], current_ladder.right_lower_x[0])
            self.consts.LADDER_INTERFACE_ZONE_REACH_OFF_LADDER
            y_range = (max(current_ladder.right_lower_y[0] + room.vertical_offset[0] - self.consts.LADDER_INTERFACE_ZONE_REACH_OFF_LADDER, 0), 
                       min(current_ladder.right_lower_y[0] + room.vertical_offset[0] + self.consts.LADDER_INTERFACE_ZONE_REACH_OFF_LADDER, self.consts.HEIGHT))
            ladder_canvas = ladder_canvas.at[x_range[0]:x_range[1], y_range[0]:y_range[1]].set(l_ind)
        return ladder_canvas
               
    def _make_rope_render_map(self, room: VanillaRoom, rope_tag: RoomTags.ROPES.value) -> jArray:
        """Makes a RGBA render map containing all the climbeable ropes
        Only called during startup.
        """
        rope_information: RoomTags.ROPES.value = rope_tag
        rope_canvas: jnp.ndarray = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT, 4), dtype=jnp.uint8)
        for r_ind in range(rope_information.ropes.shape[0]):
            rope_arr: jArray = rope_information.ropes[r_ind, ...]
            rope: Rope = SANTAH.full_deserializations[Rope](rope_arr)
            rope_arr = jnp.ones((1, 1 + rope.bottom[0] - rope.top[0], 3), dtype=jnp.uint8)
            color = self.consts.OBSTACLE_COLORS[rope.color_index[0], ...]
            color = jnp.reshape(color, shape=(1, 1, 3))
            rope_arr = jnp.multiply(rope_arr, color).astype(jnp.uint8)
            # Ropes are only a single pixel wide, 
            # Wider ropes need to be generated by placing multiple (deactivated)
            # ropes next to each other.
            rope_canvas = rope_canvas.at[rope.x_pos[0]:rope.x_pos[0] + 1, rope.top[0] + room.vertical_offset[0] : rope.bottom[0] + 1 + room.vertical_offset[0], 0:3].set(rope_arr)
            rope_canvas = rope_canvas.at[rope.x_pos[0]:rope.x_pos[0] + 1, rope.top[0] + room.vertical_offset[0] : rope.bottom[0] + 1 + room.vertical_offset[0], 3].set(255)
        return rope_canvas
    
    
    def _make_rope_collision_map(self, room: VanillaRoom, rope_tag: RoomTags.ROPES.value) -> jArray:
        # Generate a global collision map of all ropes on startup.
        # Again, ropes are only a single pixel wide.
        #
        rope_information: RoomTags.ROPES.value = rope_tag
        rope_canvas: jnp.ndarray = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.uint8)
        for r_ind in range(rope_information.ropes.shape[0]):
            rope_arr: jArray = rope_information.ropes[r_ind, ...]
            rope: Rope = SANTAH.full_deserializations[Rope](rope_arr)
            rope_arr = jnp.ones((1, 1 + rope.bottom[0] - rope.top[0]), dtype=jnp.uint8)
            rope_arr = jnp.multiply(rope_arr, r_ind + 1)
            rope_arr = jnp.multiply(rope_arr, rope.is_climbable[0]).astype(jnp.uint8)
            rope_canvas = rope_canvas.at[rope.x_pos[0]:rope.x_pos[0] + 1, rope.top[0] + room.vertical_offset[0] : rope.bottom[0] + 1 + room.vertical_offset[0]].set(rope_arr)
            
        return rope_canvas
    
    
    def _make_rope_top_map(self, room: VanillaRoom, rope_tag: RoomTags.ROPES.value) -> jArray:
        #
        # same as for the ladders, we need a global map of the rope-tops, 
        # to be able to efficiently handle teleportation onto the ropes from above
        #
        rope_information: RoomTags.ROPES.value = rope_tag
        rope_top_map: jnp.ndarray = jnp.ones(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.int32)
        # Default value is -1, 
        # Ropes are painted using their index in the rope_array.
        rope_top_map = rope_top_map*(-1)
        for r_ind in range(rope_information.ropes.shape[0]):
            rope_arr: jArray = rope_information.ropes[r_ind, ...]
            rope: Rope = SANTAH.full_deserializations[Rope](rope_arr)
            x_pos: jArray = rope.x_pos[0]
            y_pos: jArray = rope.top[0] + room.vertical_offset[0]
            rope_top_map = rope_top_map.at[x_pos, y_pos].set(r_ind*rope.accessible_from_top[0])
            
            
            
        return rope_top_map
    
    def _make_room_surface_map(self, room: VanillaRoom, DO_NOT_TOUCH_THIS: Any) -> jArray:
        #
        # Generate a map of all surfaces in the room
        # For this only surfaces present in the static collision map of the room
        # are considered.
        # This is used to efficiently get off ropes & ladders.
        # When leaving a rope or ladder, the player is automatically teleported onto the 
        # nearest floor surface.
        
        surface_canvas: jnp.ndarray = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.int32)
        casted_collision_map: jnp.ndarray = jnp.ones(shape=(self.consts.WIDTH, self.consts.HEIGHT + 1), dtype=jnp.int32)
        
        aliased_shape = (room.room_collision_map.shape[0], room.room_collision_map.shape[1] + 1)
        aliased_colision_map: jArray = jnp.zeros(shape=aliased_shape, dtype=jnp.int32)
        # Overlay two room collision maps, with one multiplied by (-1) and offset 
        # one pixel along the y axis.
        # Now, top surfaces have a value of 1
        # bottom surfaces a value of -1
        # and "inner-surface" pixels a value of 0
        aliased_colision_map = aliased_colision_map.at[:, :-1].set(room.room_collision_map)
        aliased_colision_map = aliased_colision_map.at[:, 1:].add(room.room_collision_map*(-1))
        aliased_colision_map = jnp.equal(aliased_colision_map, 1)
        aliased_colision_map = aliased_colision_map[:, :-1]
        casted_collision_map: jArray = jax.lax.dynamic_update_slice(casted_collision_map, update=jnp.astype(aliased_colision_map, jnp.int32), 
                                                                    start_indices=(0, room.vertical_offset[0]))
        
        return casted_collision_map
    
    
    def _make_dropout_floor_render_maps(self, room: VanillaRoom, dropout_tag: RoomTags.DROPOUTFLOORS.value) -> jArray:
        #
        # Precomputes render maps for dropout floor. 
        # All dropout floors in a room are rendered on the same cycle, so we can precompute sprites.
        #
        dropout_floor_information: RoomTags.DROPOUTFLOORS.value = dropout_tag
        dropout_canvas: jnp.ndarray = jnp.zeros(shape=(2,self.consts.WIDTH, self.consts.HEIGHT,4), dtype=jnp.uint8)
        for index in range(2):
            for df_ind in range(dropout_floor_information.dropout_floors.shape[0]):
                dropout_floor_arr: jArray = dropout_floor_information.dropout_floors[df_ind, ...]
                dropout_floor: DropoutFloor = SANTAH.full_deserializations[DropoutFloor](dropout_floor_arr)
                sprite_index = dropout_floor.sprite_index
                # Use different sprites for ladder vs pit-room room dropout floors.
                if sprite_index == Dropout_Floor_Sprites.LADDER_FLOOR.value:
                    sprite_name = "other_dropout_floor"
                else:
                    sprite_name = "pitroom_dropout_floor"
                dropout_floor_sprite = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT[sprite_name]),
                                                         add_alpha=True, add_black_as_transparent=True, transpose=True).astype(jnp.uint8)
                compete_dropout_floor = jnp.zeros(shape=(dropout_floor.sprite_width_amount[0]*dropout_floor_sprite.shape[0],
                                                         dropout_floor.sprite_height_amount[0]*dropout_floor_sprite.shape[1],
                                                         4), dtype=jnp.uint8)
                # Dimensions of the dropout floors are given in "sprite_width"
                # Actual dimensions depend on the shape of the sprites used.
                for x in range(dropout_floor.sprite_width_amount[0]):
                    for y in range(dropout_floor.sprite_height_amount[0]):
                        # Orientation of the dropout floor sprite is flipped along the x axis each step along the x&y axis.
                        compete_dropout_floor = jax.lax.dynamic_update_slice(operand=compete_dropout_floor,
                                                                             update=jax.lax.cond(jnp.mod(x+y+index,2), 
                                                                                                 lambda x: x,
                                                                                                 lambda x: jnp.flip(x, axis=1),
                                                                                                 operand=dropout_floor_sprite),
                                                                             start_indices=(x*dropout_floor_sprite.shape[0], y*dropout_floor_sprite.shape[1], 0))
                # Sprites are colorized, to enable different dropout floor colors.
                compete_dropout_floor = colorize_sprite(compete_dropout_floor,self.consts.OBSTACLE_COLORS[dropout_floor.color[1]])
                dropout_canvas =  dropout_canvas.at[index].set(jax.lax.dynamic_update_slice(operand=dropout_canvas[index],
                                                               update=compete_dropout_floor,
                                                               start_indices=(dropout_floor.x[0], dropout_floor.y[0] + room.vertical_offset[0],0)))
        return dropout_canvas
    
    def _make_dropout_floor_collision_map(self, room: VanillaRoom, dropout_tag: RoomTags.DROPOUTFLOORS.value) -> jArray:
        # Generate dropout collision map for dropout floors.
        # Collision with this map is used to decide whether the player needs to be moved along the X-axis.
        dropout_floor_information: RoomTags.DROPOUTFLOORS.value = dropout_tag
        dropout_canvas: jnp.ndarray = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.uint8)
        for df_ind in range(dropout_floor_information.dropout_floors.shape[0]):
                dropout_floor_arr: jArray = dropout_floor_information.dropout_floors[df_ind, ...]
                dropout_floor: DropoutFloor = SANTAH.full_deserializations[DropoutFloor](dropout_floor_arr)
                if dropout_floor.sprite_index == Dropout_Floor_Sprites.LADDER_FLOOR.value:
                    sprite_name = "other_dropout_floor"
                else:
                    sprite_name = "pitroom_dropout_floor"
                dropout_floor_sprite = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT[sprite_name]),
                                                         add_alpha=True, add_black_as_transparent=True, transpose=True).astype(jnp.uint8)
                compete_dropout_floor_collision = jnp.ones(shape=(dropout_floor.sprite_width_amount[0]*dropout_floor_sprite.shape[0],
                                                         dropout_floor.sprite_height_amount[0]*dropout_floor_sprite.shape[1]+dropout_floor.collision_padding_top[0]), dtype=jnp.uint8)
                dropout_canvas =  jax.lax.dynamic_update_slice(operand=dropout_canvas,
                                                               update=compete_dropout_floor_collision,
                                                               start_indices=(dropout_floor.x[0], (dropout_floor.y[0] +  room.vertical_offset[0]) -dropout_floor.collision_padding_top[0]))
        return dropout_canvas
    
    def _make_pit_render_maps(self, room: VanillaRoom, pit_tag: RoomTags.PIT.value) -> jArray:
        # Generates the render maps for the fire pits.
        # 
        col_map = room.room_collision_map
        # Animation for the fire pit consists of 4 sprites.
        pit_canvas: jnp.ndarray = jnp.zeros(shape=(4,col_map.shape[0], col_map.shape[1],4), dtype=jnp.uint8)
        for frame in range(4):
            pit_sprite = jnp.zeros(shape=(self.consts.WIDTH, room.height[0] - pit_tag.starting_pos_y[0],4),dtype=jnp.uint8)
            # Colorize the pit in horizontal bands of height 2
            for x in range(int((room.height[0] - pit_tag.starting_pos_y[0])/2)):
                pit_sprite = pit_sprite.at[:,2*x:2*x+2].set(self.color_for_pit(frame, x, pit_tag.pit_color[0]))
            pit_canvas = pit_canvas.at[frame].set(jax.lax.dynamic_update_slice(pit_canvas[frame],pit_sprite,(0,pit_tag.starting_pos_y[0],0)))
            col_map = jnp.reshape(col_map, (col_map.shape[0],col_map.shape[1],1))
            # Multiply 1-room_colision map onto pit canvas, so that pit is only rendered
            # where there are no walls.
            pit_canvas = pit_canvas.at[frame].set(jnp.multiply(pit_canvas[frame], 1-col_map))
        return pit_canvas
    
    def color_for_pit(self, frame, index, rgb):
        #  Logic for generating the colors for the individual bands in the pit. 
        # This is a pretty close approximation to the look of the original game-pits.
        pattern = self.consts.PIT_COLOR_PATTERN
        color_index = int(index/9) + pattern[frame][jnp.mod(index,9)]
        r = int(-(0.65*color_index**2)-(14*color_index)+rgb[0])
        r = jnp.maximum(r,0)
        g = int(-(color_index**2)-(19*color_index)+rgb[1])
        g = jnp.maximum(g,0)
        b = int(-(0.88*color_index**2)-(11.25*color_index)+rgb[2])
        b = jnp.maximum(b,0)
        output = jnp.zeros(shape=(4))
        output = output.at[0].set(r)
        output = output.at[1].set(g)
        output = output.at[2].set(b)
        output = output.at[3].set(255)
        return jnp.astype(output,jnp.uint8)
    
    def _make_side_walls_render_map(self, room: VanillaRoom, side_wall_tag: RoomTags.SIDEWALLS.value) -> jArray:
        # Side walls are walls that can be dynamically inserted into rooms to block off ways.
        col_map = room.room_collision_map
        side_wall_canvas: jnp.ndarray = jnp.zeros(shape=(col_map.shape[0], col_map.shape[1],4), dtype=jnp.uint8)
        
        # Side walls are always rendered at the same hight, in the original game 
        # a vertical offset of 6 is used.
        single_wall_sprite = jnp.ones(shape=(4,room.height[0]-6,4),dtype=jnp.uint8)
        s_w_col = side_wall_tag.side_wall_color[0]
        wall_color = jnp.array([s_w_col[0],s_w_col[1],s_w_col[2],255], jnp.uint8)
        wall_color = jnp.reshape(wall_color,shape=(1,1,4))
        
        single_wall_sprite = jnp.multiply(single_wall_sprite,wall_color).astype(jnp.uint8)
        # Side walls can only be on left or right edge of the room.
        if side_wall_tag.is_left:
            side_wall_canvas = jax.lax.dynamic_update_slice(operand=side_wall_canvas,
                                                            update=single_wall_sprite,
                                                            start_indices=(0,6,0))
        if side_wall_tag.is_right:
            side_wall_canvas = jax.lax.dynamic_update_slice(operand=side_wall_canvas,
                                                            update=single_wall_sprite,
                                                            start_indices=(self.consts.WIDTH-4,6,0))
        col_map = jnp.reshape(col_map, (col_map.shape[0],col_map.shape[1],1))
        # Erase the sidewalls where they overlap with the room-collision, 
        # So that they are not rendered over the checker-board pattern on the floor.
        side_wall_canvas = jnp.multiply(side_wall_canvas, 1-col_map)
        return side_wall_canvas

    def _make_side_walls_collision_map(self, room: VanillaRoom, side_wall_tag: RoomTags.SIDEWALLS.value) -> jArray:
        #
        # Same logic as for the render map, 
        # only that we do not need to erase the sidewalls where they overlap with the room collision.
        #
        side_wall_canvas: jnp.ndarray = jnp.zeros(shape=(self.consts.WIDTH,self.consts.HEIGHT), dtype=jnp.uint8)
        single_wall_collision_map = jnp.ones(shape=(4,room.height[0]-6), dtype=jnp.uint8)
        if side_wall_tag.is_left:
            side_wall_canvas = jax.lax.dynamic_update_slice(operand=side_wall_canvas,
                                                            update=single_wall_collision_map,
                                                            start_indices=(0,room.vertical_offset[0]+6))
        if side_wall_tag.is_right:
            side_wall_canvas = jax.lax.dynamic_update_slice(operand=side_wall_canvas,
                                                            update=single_wall_collision_map,
                                                            start_indices=(self.consts.WIDTH-4,room.vertical_offset[0]+6))
        
        return side_wall_canvas
    
    def _make_conveyor_belts_movement_collision_map(self, room: VanillaRoom, conveyor_belt_tag: RoomTags.CONVEYORBELTS.value) -> jArray:
        #
        # Conveyor belts have 2 different collision maps:
        # - one that highlights the pixels above the surface, this one is used to detect when the player needs to be moved
        # - and one that is painted onto the overal colision map to prevent the player from falling through the floor.
        #
        conveyor_belt_information: RoomTags.CONVEYORBELTS.value = conveyor_belt_tag
        # Conveyor belts each can have an individual movement direction
        # The hitmap is initialized with the value for "NO_DIR". This is the maximum value
        # used in the enum, so it is important to take the min value of the slice occupied by the player
        # to determine in which direction he should move
        conveyor_belt_canvas: jnp.ndarray = jnp.ones(shape=(self.consts.WIDTH,self.consts.HEIGHT), dtype=jnp.uint8) * MovementDirection.NO_DIR.value
        for cb_ind in range(conveyor_belt_information.conveyor_belts.shape[0]):
            conveyor_belt_arr: jArray = conveyor_belt_information.conveyor_belts[cb_ind, ...]
            conveyor_belt: ConveyorBelt = SANTAH.full_deserializations[ConveyorBelt](conveyor_belt_arr)
            # Multiply the collision map with the movement direction specified for the individual conveyor belt.
            conveyor_belt_collision_map = jnp.ones(shape=(40, 1),dtype=jnp.uint8) * conveyor_belt.movement_dir[0]
            
            conveyor_belt_canvas = jax.lax.dynamic_update_slice(operand=conveyor_belt_canvas,
                                                                update=conveyor_belt_collision_map.astype(jnp.uint8),
                                                                start_indices=(conveyor_belt.x[0],conveyor_belt.y[0]-1+room.vertical_offset[0])
                                                                )
            
        return conveyor_belt_canvas
    
    def _make_conveyor_belts_collision_map(self, room: VanillaRoom, conveyor_belt_tag: RoomTags.CONVEYORBELTS.value) -> jArray:
        #
        # Make collision map for all the conveyor belts in the room.
        # This works exactly the same way as with all the other collision maps.
        #
        conveyor_belt_information: RoomTags.CONVEYORBELTS.value = conveyor_belt_tag
        conveyor_belt_canvas: jnp.ndarray = jnp.zeros(shape=(self.consts.WIDTH,self.consts.HEIGHT), dtype=jnp.uint8)
        for cb_ind in range(conveyor_belt_information.conveyor_belts.shape[0]):
            conveyor_belt_arr: jArray = conveyor_belt_information.conveyor_belts[cb_ind, ...]
            conveyor_belt: ConveyorBelt = SANTAH.full_deserializations[ConveyorBelt](conveyor_belt_arr)
            
            # Use the sprites to determine the size of the collision map for the conveyor belts.
            conveyor_belt_sprite = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["conveyor_belt"]),
                                                         add_alpha=True, add_black_as_transparent=True, transpose=True).astype(jnp.uint8)
            conveyor_belt_collision_map = jnp.ones(shape=(conveyor_belt_sprite.shape[0], conveyor_belt_sprite.shape[1]),dtype=jnp.uint8)
            
            conveyor_belt_canvas = jax.lax.dynamic_update_slice(operand=conveyor_belt_canvas,
                                                                update=conveyor_belt_collision_map,
                                                                start_indices=(conveyor_belt.x[0],conveyor_belt.y[0]+room.vertical_offset[0])
                                                                )
            
        return conveyor_belt_canvas
    
    def _make_conveyor_belts_render_map(self, room: VanillaRoom, conveyor_belt_tag: RoomTags.CONVEYORBELTS.value) -> jArray:
        conveyor_belt_information: RoomTags.CONVEYORBELTS.value = conveyor_belt_tag
        # Conveyor belts have 2 animation frames, so precompute 2 sprites.
        conveyor_belt_canvas: jnp.ndarray = jnp.zeros(shape=(2,self.consts.WIDTH, self.consts.HEIGHT,4), dtype=jnp.uint8)
        for index in range(2):
            for cb_ind in range(conveyor_belt_information.conveyor_belts.shape[0]):
                # Paint each individual conveyor belt onto the canvas
                conveyor_belt_arr: jArray = conveyor_belt_information.conveyor_belts[cb_ind, ...]
                conveyor_belt: ConveyorBelt = SANTAH.full_deserializations[ConveyorBelt](conveyor_belt_arr)
                # Conveyor belts right now have fixed size, but it would be easy to adapt them to a flexible shape.
                conveyor_belt_sprite = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["conveyor_belt"]),
                                                         add_alpha=True, add_black_as_transparent=True, transpose=True).astype(jnp.uint8)
                conveyor_belt_sprite = colorize_sprite(conveyor_belt_sprite,self.consts.OBSTACLE_COLORS[conveyor_belt.color[0]])
                
                # The secod animation sprite is just a flipped version of the original sprite.
                conveyor_belt_sprite = jax.lax.cond(index,
                                                    lambda x: x,
                                                    lambda x: jnp.flip(x, axis=1),
                                                    operand=conveyor_belt_sprite)
                conveyor_belt_canvas =  conveyor_belt_canvas.at[index].set(jax.lax.dynamic_update_slice(operand=conveyor_belt_canvas[index],
                                                                                            update=conveyor_belt_sprite,
                                                                                            start_indices=(conveyor_belt.x[0], conveyor_belt.y[0] + room.vertical_offset[0],0)))
        return conveyor_belt_canvas
    
    def _make_bonus_room_floor_collision_map(self, room: VanillaRoom, bonus_room_tag: RoomTags.BONUSROOM.value) -> jArray:
        #
        # Make collision map for bonus rooms.
        # Bonus rooms are just a solid color & a flat floor, 
        # so collision maps for the room layout can be computed automatically.
        #
        bonus_room_canvas: jnp.ndarray = jnp.zeros(shape=(self.consts.WIDTH,self.consts.HEIGHT), dtype=jnp.uint8)
        floor_collision_map = jnp.ones(shape=(self.consts.WIDTH,room.height[0]-self.consts.BONUS_ROOM_FLOOR_Y),dtype=jnp.uint8)
        bonus_room_canvas = jax.lax.dynamic_update_slice(operand=bonus_room_canvas,
                                                                update=floor_collision_map,
                                                                start_indices=(0,self.consts.BONUS_ROOM_FLOOR_Y+room.vertical_offset[0])
                                                                )
        return bonus_room_canvas
        
    def __make_room_infra_ready(self):
        #
        # This function prepares the room infrastructure. 
        # We register the proto & vanilla room, as well as constructors for all fields that are precomputed on startup.
        #
        #
        SANTAH.register_proto_room(room_field_enum=ConstantShapeRoomFields, proto_room_nt=Room, 
                fields_that_are_shared_but_have_different_shape=FieldsThatAreSharedByAllRoomsButHaveDifferentShape, 
                vanilla_room_type=VanillaRoom, 
                vanilla_room_enum=VanillaRoomFields)
        #
        # Default values are registered per-room tag, so that no further work is necessary
        # when adding more features to a room. 
        # all precomputed fields will be automatically generated.
        #
        SANTAH.register_room_tags(room_tags=RoomTags, room_tags_descriptor=RoomTagsNames, 
                                                                  constructed_fields={
                                                                      RoomTags.LAZER_BARRIER: {
                                                                          RoomTagsNames.LAZER_BARRIER.value.global_barrier_map.value: self._make_barrier_activation_map
                                                                      },
                                                                      RoomTags.DOORS: {
                                                                          RoomTagsNames.DOORS.value.global_collision_map.value: self._make_door_collision_map
                                                                      },
                                                                      RoomTags.ROPES: {
                                                                          RoomTagsNames.ROPES.value.rope_render_map.value: self._make_rope_render_map,
                                                                          RoomTagsNames.ROPES.value.rope_colision_map.value: self._make_rope_collision_map, 
                                                                          RoomTagsNames.ROPES.value.room_surfaces.value: self._make_room_surface_map, 
                                                                          RoomTagsNames.ROPES.value.room_rope_top_pixels.value: self._make_rope_top_map
                                                                      },
                                                                      RoomTags.DROPOUTFLOORS: {
                                                                         RoomTagsNames.DROPOUTFLOORS.value.dropout_floor_colision_map.value: self._make_dropout_floor_collision_map,
                                                                         RoomTagsNames.DROPOUTFLOORS.value.dropout_floor_render_maps.value: self._make_dropout_floor_render_maps
                                                                      },
                                                                      RoomTags.PIT: {
                                                                          RoomTagsNames.PIT.value.pit_render_maps.value: self._make_pit_render_maps
                                                                      },
                                                                      RoomTags.SIDEWALLS: {
                                                                          RoomTagsNames.SIDEWALLS.value.side_walls_collision_map.value: self._make_side_walls_collision_map,
                                                                          RoomTagsNames.SIDEWALLS.value.side_walls_render_map.value: self._make_side_walls_render_map
                                                                          }, 
                                                                      RoomTags.LADDERS: {
                                                                          RoomTagsNames.LADDERS.value.ladders_sprite.value: self._make_ladder_render_map, 
                                                                          RoomTagsNames.LADDERS.value.ladder_tops.value: self._make_ladder_tops_map,
                                                                          RoomTagsNames.LADDERS.value.ladder_bottoms.value: self._make_ladder_bottom_map,
                                                                          RoomTagsNames.LADDERS.value.ladder_room_surface_pixels.value: self._make_room_surface_map
                                                                      },
                                                                      RoomTags.CONVEYORBELTS: {
                                                                          RoomTagsNames.CONVEYORBELTS.value.global_conveyor_movement_collision_map.value: self._make_conveyor_belts_movement_collision_map,
                                                                          RoomTagsNames.CONVEYORBELTS.value.global_conveyor_collision_map.value: self._make_conveyor_belts_collision_map,
                                                                          RoomTagsNames.CONVEYORBELTS.value.global_conveyor_render_map.value: self._make_conveyor_belts_render_map
                                                                      },
                                                                      RoomTags.BONUSROOM: {
                                                                          RoomTagsNames.BONUSROOM.value.bonus_room_floor_collison_map.value: self._make_bonus_room_floor_collision_map
                                                                      }
                                                                  })
        SANTAH.register_room(VanillaRoom, field_enum=VanillaRoomFields)
    
    def __init_room(self):
        
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        SPRITE_PATH: str = os.path.join(MODULE_DIR, "sprites", "montezuma")
        SPRITE_PATH_BACKGROUND: str = os.path.join(SPRITE_PATH, "backgrounds")
        
        global LAYOUT 
        LAYOUT = PyramidLayout()
        
        # Decide which layout to build
        if self.consts.LAYOUT == Layouts.demo_layout.value:
            LAYOUT = make_demo_layout(LAYOUT, self.consts)
        elif self.consts.LAYOUT == Layouts.difficulty_1.value:
            LAYOUT = make_difficulty_1(LAYOUT, self.consts)
        elif self.consts.LAYOUT == Layouts.difficulty_2.value:
            LAYOUT = make_difficulty_2(LAYOUT, self.consts)
        else:
            LAYOUT = make_layout_test(LAYOUT, self.consts)
        
    

        self.room_connection_map = LAYOUT.get_jitted_room_connection_map()
        self.initial_persistence_state = LAYOUT._create_initial_persistence_state()
        
        self.PROTO_ROOM_LOADER, self.WRITE_PROTO_ROOM_TO_PERSISTENCE = LAYOUT.create_proto_room_specific_persistence_infrastructure()
        self.WRAPPED_AUGMENT_COLLISION_MAP = LAYOUT._wrap_lowered_function(lowered_function=self.augment_collision_map, 
                                                       montezuma_state_type=MontezumaState)
        
        
        # Wrap all functions that require access to rooms at the tag-level.
        # This is necessary to deal with the fact, that we have multiple different rooms with multiple different features.
        # All the branching that is necessary to handle the cases for different rooms
        # is generated automatically.
        #
        self.WRAPPED_HANDLE_COUNTER_UPDATES = LAYOUT._wrap_lowered_function(lowered_function=self.handle_counters, 
                                                                            montezuma_state_type=MontezumaState)
        self.WRAPPED_HANDLE_ROOM_ENTRANCE = LAYOUT._wrap_lowered_function(lowered_function=self.handle_enter_room, 
                                                                          montezuma_state_type=MontezumaState)
        
        self.WRAPPED_HANDLE_LAZER_BARRIER_COL = LAYOUT._wrap_lowered_function(lowered_function=self.handle_lazer_barrier_collision, 
                                                                              montezuma_state_type=MontezumaState)
        self.WRAPPED_HANDLE_ITEM_COLLISION_CHECK = LAYOUT._wrap_lowered_function(lowered_function=self.handle_item_collision_check,
                                                                                 montezuma_state_type=MontezumaState)
        self.WRAPPED_HANDLE_DOOR_COLLISION = LAYOUT._wrap_lowered_function(lowered_function=self.handle_door_collision,
                                                                           montezuma_state_type=MontezumaState)
        
        self.WRAPPED_ADD_DOOR_COLLISION_TO_ROOM_COLLISION_MAP = LAYOUT._wrap_lowered_function(lowered_function=self.add_door_collision_to_room_collision_map,
                                                                                              montezuma_state_type=MontezumaState)
        
        self.WRAPPED_ADD_DROPOUT_FLOOR_COLLISION_TO_ROOM_COLLISION_MAP = LAYOUT._wrap_lowered_function(lowered_function=self.add_dropout_floor_collision_to_room_collision_map,
                                                                                              montezuma_state_type=MontezumaState)
        
        self.WRAPPED_HANDLE_START_CLIMBING = LAYOUT._wrap_lowered_function(lowered_function=self._handle_start_climb_wrappable, 
                                                                     montezuma_state_type=MontezumaState)
    
        self.WRAPPED_HANDLE_STOP_CLIMBING = LAYOUT._wrap_lowered_function(lowered_function=self._handle_stop_climbing_wrappable, 
                                                                          montezuma_state_type=MontezumaState)

        self.WRAPPED_HANDLE_SARLACC_PIT_COL = LAYOUT._wrap_lowered_function(lowered_function=self._handle_sarlacc_pit_collision, 
                                                                          montezuma_state_type=MontezumaState)
        self.WRAPPED_ADD_SIDE_WALL_COLLISION_TO_ROOM_COLLISION_MAP = LAYOUT._wrap_lowered_function(lowered_function=self.add_side_wall_collision_to_room_collision_map,
                                                                                              montezuma_state_type=MontezumaState)
        self.WRAPPED_ON_ROOM_CHANGE = LAYOUT._wrap_lowered_function(lowered_function=self._handle_on_room_change, 
                                                                    montezuma_state_type=MontezumaState)
        
        self.WRAPPED_HANDLE_ENEMIES = LAYOUT._wrap_lowered_function(lowered_function=self.handle_enemies, 
                                                                    montezuma_state_type=MontezumaState)
        self.WRAPPED_ADD_CONVEYOR_BELT_COLLISION_TO_ROOM_COLLISION_MAP = LAYOUT._wrap_lowered_function(lowered_function=self.add_conveyor_belt_collision_to_room_collision_map,
                                                                                              montezuma_state_type=MontezumaState)
        self.WRAPPED_HANDLE_CONVEYOR_MOVEMENT = LAYOUT._wrap_lowered_function(lowered_function=self.handle_conveyor_movement,
                                                                              montezuma_state_type=MontezumaState)
        self.WRAPPED_HANDLE_DARKNESS = LAYOUT._wrap_lowered_function(lowered_function=self.handle_darkness,
                                                                              montezuma_state_type=MontezumaState) 
        self.WRAPPED_HANDLE_ENEMY_COLLISION = LAYOUT._wrap_lowered_function(lowered_function=self.handle_enemy_collision_check, 
                                                                            montezuma_state_type=MontezumaState)
        self.WRAPPED_ADD_BONUS_ROOM_FLOOR_COLLISION_TO_ROOM_COLLISION_MAP = LAYOUT._wrap_lowered_function(lowered_function=self.add_bonus_room_floor_collision_to_room_collision_map, 
                                                                            montezuma_state_type=MontezumaState)
        self.WRAPPED_HANDLE_ON_DEATH_LADDER_SEEKING = LAYOUT._wrap_lowered_function(lowered_function=self._handle_on_death_ladder_seeking_behavior, 
                                                                                  montezuma_state_type=MontezumaState)
        self.WRAPPED_HANDLE_COMPUTE_OBSERVATION = LAYOUT._wrap_lowered_function(lowered_function=self._wrappable_get_montezuma_observation, 
                                                                                montezuma_state_type=MontezumaState)
    
    

    def handle_enter_room(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # Handles logic when a room is newly entered.
        #
        # Track the frame in which we have entered a room.
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.room_enter_frame.value](montezuma_state, montezuma_state.frame_count)
        # This field is mainly used to determine the start of the countdown in the bonus room
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.first_item_pickup_frame.value](montezuma_state, jnp.array([0], jnp.int32))
        
        if RoomTags.LAZER_BARRIER in tags:
            # If we enter a vertical barrier room freshly, we need to reset the barrier cycle counter.
            lazer_b_attrs: RoomTags.LAZER_BARRIER.value = SANTAH.extract_tag_from_rooms[RoomTags.LAZER_BARRIER](room_state)
            # Cycle index is shared by all barriers, so we only need to update one value.
            barrier_info: GlobalLazerBarrierInfo = SANTAH.full_deserializations[GlobalLazerBarrierInfo](lazer_b_attrs.global_barrier_info[0, ...])
            barrier_info = SANTAH.attribute_setters[GlobalLazerBarrierInfo][GlobalLazerBarrierInfoEnum.cycle_index.value](barrier_info, jnp.array([0], jnp.int32))
            info_arr: jArray = SANTAH.full_serialisations[GlobalLazerBarrierInfo](barrier_info)
            info_arr = jnp.reshape(info_arr, shape=(1, *info_arr.shape))
            lazer_b_attrs = SANTAH.attribute_setters[
                                RoomTags.LAZER_BARRIER.value
                        ][RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value](lazer_b_attrs, info_arr)
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.LAZER_BARRIER](room_state, lazer_b_attrs)
            return montezuma_state, room_state
            
        elif RoomTags.BONUSROOM in tags:
            bonus_room_attrs: RoomTags.BONUSROOM.value = SANTAH.extract_tag_from_rooms[RoomTags.BONUSROOM](room_state)
            new_cycle_index = jnp.array([0], jnp.int32)
            # reset the bonus room frame counter. This is compared to the first_item_pickup_frame to determine when the player needs to leave the bonus room.
            bonus_room_attrs = SANTAH.attribute_setters[RoomTags.BONUSROOM.value][RoomTagsNames.BONUSROOM.value.bouns_cycle_index.value](bonus_room_attrs, new_cycle_index)
            # Set in the MontezumaState whether all room-states need to be reset when leaving this room. 
            # this is necessary if we have reached the last bonus room, from then on we are only looping in the 
            # last level, and enemies, items & doors need to be reset.
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.reset_room_state_on_room_change.value](montezuma_state, bonus_room_attrs.reset_state_on_leave)
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.BONUSROOM](room_state, bonus_room_attrs)
            return montezuma_state, room_state   
        else:
            return montezuma_state, room_state
        
  
    def handle_counters(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple],  tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # Handles updates for room specific counters
        
        
        # Handle the hammer counter here: Hammer only lasts for a specific amount of frames. 
        # If the time expires, hammer is removed from inventory
        new_hammer_time: jArray = jax.lax.max(montezuma_state.hammer_time - 1, 0)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.hammer_time.value](montezuma_state, new_hammer_time)
        is_hammering: jArray = jnp.greater(new_hammer_time, 0)
        new_hammerbar_val = jnp.multiply(is_hammering, 1) + jnp.multiply(1 - is_hammering, 0)
        new_itembar = montezuma_state.itembar_items.at[ItemBar_Sprites.HAMMER.value].set(new_hammerbar_val[0])
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.itembar_items.value](montezuma_state, new_itembar)
        
        # Handle the counter logic for the ladder climbing animation frame.
        is_on_ladder: jArray = montezuma_state.is_laddering
        # Only update the ladder frame counter if the player is actually moving upwards.
        is_actually_climbing: jArray = jnp.not_equal(montezuma_state.current_velocity[1], 0)
        # Check how long the velocity has actually been held, so that we don't immediately 
        # update the animation after the climbing starts.
        laddering_time: jArray = jnp.multiply(montezuma_state.velocity_held, is_actually_climbing)
        # Change the animation frame only if we have climbed for long enough.
        frame_change: jArray = jnp.logical_and(jnp.equal(jnp.mod(laddering_time, self.consts.LADDER_CLIMBING_ANIMATION_FRAME_LENGTH), 0), 
                                                    jnp.not_equal(montezuma_state.velocity_held, 0))
        frame_change = jnp.multiply(is_actually_climbing, jnp.multiply(is_on_ladder, frame_change))
        new_frame_index: jArray = jnp.mod(montezuma_state.ladder_climb_frame + frame_change, 2)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.ladder_climb_frame.value](montezuma_state, new_frame_index)
        
        # Handle the counter logic for the rope climbing animation frame
        # This works exactly the same as for the ladders.
        is_on_rope: jArray = montezuma_state.is_on_rope
        is_actually_climbing: jArray = jnp.not_equal(montezuma_state.current_velocity[1], 0)
        roping_time: jArray = jnp.multiply(montezuma_state.velocity_held, is_actually_climbing)
        frame_change: jArray = jnp.logical_and(jnp.equal(jnp.mod(roping_time, self.consts.ROPE_CLIMBING_ANIMATION_FRAME_LENGTH), 0), 
                                                    jnp.not_equal(montezuma_state.velocity_held, 0))
        frame_change = jnp.multiply(is_actually_climbing, jnp.multiply(is_on_rope, frame_change))
        new_frame_index: jArray = jnp.mod(montezuma_state.rope_climb_frame + frame_change, 2)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.rope_climb_frame.value](montezuma_state, new_frame_index)
        
        # Handle the counter logic for the walking animation logic.
        # The walking animation depends on the horizontal input, not the movement. 
        # This is done because the conveyer belts interfere with the absolute movement
        is_on_floor: jArray = jnp.logical_and(jnp.logical_not(montezuma_state.is_climbing), jnp.logical_and(jnp.logical_not(montezuma_state.is_jumping), 
                                                                                    jnp.logical_not(montezuma_state.is_falling)))
        
        is_actually_walking: jArray = jnp.not_equal(montezuma_state.horizontal_direction, Horizontal_Direction.NO_DIR.value)
        # Only update the walking animation if the player has been walking for long enough
        walking_time: jArray = jnp.multiply(montezuma_state.horizontal_direction_held, is_actually_walking)
        frame_change: jArray = jnp.logical_and(jnp.equal(jnp.mod(walking_time, self.consts.WALKING_ANIMATION_FRAME_LENGTH), 0), 
                                                    jnp.not_equal(montezuma_state.horizontal_direction_held, 0))
        frame_change = jnp.multiply(is_actually_walking, jnp.multiply(is_on_floor, frame_change))
        # Apply mod to loop animation frames.
        new_frame_index: jArray = jnp.mod(montezuma_state.walk_frame + frame_change, 2)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.walk_frame.value](montezuma_state, new_frame_index)
        
        
        
        if RoomTags.LAZER_BARRIER in tags: 
            # Update the cycle index for the lazer barriers at this point.
            #
            #
            lazer_barrier_attr: RoomTags.LAZER_BARRIER.value = SANTAH.extract_tag_from_rooms[RoomTags.LAZER_BARRIER](room_state)
            barrier_info: GlobalLazerBarrierInfo = SANTAH.full_deserializations[GlobalLazerBarrierInfo](lazer_barrier_attr.global_barrier_info[0, ...])
            new_cycle_index: jArray = jnp.mod(barrier_info.cycle_index+1, barrier_info.cycle_length)
            new_cycle_index = jnp.reshape(new_cycle_index, (1))
            # Write new cycle index back to lazer-barrier tag
            barrier_info = SANTAH.attribute_setters[GlobalLazerBarrierInfo][GlobalLazerBarrierInfoEnum.cycle_index.value](barrier_info, new_cycle_index)
            info_arr: jArray = SANTAH.full_serialisations[GlobalLazerBarrierInfo](barrier_info)
            info_arr = jnp.reshape(info_arr, shape=(1, *info_arr.shape))
            lazer_barrier_attr = SANTAH.attribute_setters[
                                RoomTags.LAZER_BARRIER.value
                        ][RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value](lazer_barrier_attr, info_arr)
            # Write back the lazer barrier tag to the room
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.LAZER_BARRIER](room_state, lazer_barrier_attr)
            
        if RoomTags.BONUSROOM in tags: 
            # Update the bonus_room cycle index. This is used to spawn new gems & eventually kick the player out of the bonus room
            bonus_room_attrs: RoomTags.BONUSROOM.value = SANTAH.extract_tag_from_rooms[RoomTags.BONUSROOM](room_state)
            new_cycle_index = bonus_room_attrs.bouns_cycle_index+1
            bonus_room_attrs = SANTAH.attribute_setters[RoomTags.BONUSROOM.value][RoomTagsNames.BONUSROOM.value.bouns_cycle_index.value](bonus_room_attrs, new_cycle_index)
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.BONUSROOM](room_state, bonus_room_attrs)
        
        return montezuma_state, room_state
            
    

    def augment_collision_map(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # Paints the collision map of the current room onto the a collision map of static size.
        # This is necessary, as rooms have varying shapes & sizes.
        
        # Use 1 as default value, so the player can't get outside the room-area
        canvas: jnp.ndarray = jnp.ones(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.uint8)
         
        room_specific_collision_map: jnp.ndarray = getattr(room_state, FieldsThatAreSharedByAllRoomsButHaveDifferentShape.room_collision_map.value)
        room_specific_collision_map = jnp.astype(room_specific_collision_map, jnp.uint8)
        canvas = jax.lax.dynamic_update_slice(operand=canvas, 
                                     update=room_specific_collision_map, 
                                     start_indices=(0, room_state.vertical_offset[0]))
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.augmented_collision_map.value](montezuma_state, canvas)
        return montezuma_state, room_state 
    

        
      
    

    def add_door_collision_to_room_collision_map(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        if RoomTags.DOORS in tags:  
            door_attrs: RoomTags.DOORS.value = SANTAH.extract_tag_from_rooms[RoomTags.DOORS](room_state)
            # Doors need to be added individually to the collision map, 
            # as they can be deactivated, so we can't compute them beforehand
            #
            # Doors cannot be handled via VMAP, as they need to alter the shared collision_map canvas.
            # Also, per-door logic is very simple.
            
            def single_door_collision_addition(i,doors_and_room : Tuple[RoomTags.DOORS.value, MontezumaState]):
                doors: RoomTags.DOORS.value
                state: MontezumaState
                doors, state = doors_and_room
                
                # Paint individual door on the global collision map
                des_door: Item = SANTAH.full_deserializations[Door](doors[0][i])
                old_collision_map = state.augmented_collision_map
                door_collision_map: jnp.ndarray = jnp.ones(shape=(self.consts.DOOR_WIDTH-1, self.consts.DOOR_HEIGHT+1), dtype=jnp.uint8)
                collision_map_with_door = jax.lax.dynamic_update_slice(operand=old_collision_map,
                                                                 update=door_collision_map,
                                                                 start_indices=(des_door.x[0],des_door.y[0]+ room_state.vertical_offset[0]))
                
                # Only update the collision map with the new door, if the door is still on field.
                new_collision_map = jax.lax.cond(des_door.on_field[0],
                                                   lambda _: collision_map_with_door,
                                                   lambda _: old_collision_map,
                                                   operand=None)
                
                state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.augmented_collision_map.value](state, new_collision_map)
                return doors, state
            
            _, state = jax.lax.fori_loop(lower=0,upper=jnp.shape(door_attrs[0])[0],body_fun=single_door_collision_addition,init_val=(door_attrs, montezuma_state))
            return state, room_state
        else:
            return montezuma_state, room_state
    
    def add_bonus_room_floor_collision_to_room_collision_map(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        if RoomTags.BONUSROOM in tags:
            # If this is a bonus room, just paint the precomputed collision map onto the global
            # collision map.
            bonus_room_attrs: RoomTags.BONUSROOM.value = SANTAH.extract_tag_from_rooms[RoomTags.BONUSROOM](room_state)
            state = montezuma_state
            old_collision_map = state.augmented_collision_map
            new_collision_map = jnp.logical_or(bonus_room_attrs.bonus_room_floor_collison_map,old_collision_map).astype(jnp.uint8)
            floor_drop_cond= jnp.logical_and(jnp.greater(state.frame_count[0],state.first_item_pickup_frame[0]+bonus_room_attrs.bonus_cycle_lenght[0]),state.first_item_pickup_frame[0])
            new_collision_map = jax.lax.cond(floor_drop_cond,
                                         lambda _:old_collision_map,
                                         lambda _:new_collision_map,
                                         operand=None)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.augmented_collision_map.value](state, new_collision_map)
            return state, room_state
        else:
            return montezuma_state, room_state
        
    def add_conveyor_belt_collision_to_room_collision_map(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # Simple function that paints the precomputed collision map for the conveyor belts onto the global collision map.
        #
        if RoomTags.CONVEYORBELTS in tags:
            conveyor_belts_attrs: RoomTags.CONVEYORBELTS.value = SANTAH.extract_tag_from_rooms[RoomTags.CONVEYORBELTS](room_state)
            state = montezuma_state
            old_collision_map = state.augmented_collision_map
            new_collision_map = jnp.logical_or(conveyor_belts_attrs.global_conveyor_collision_map,old_collision_map).astype(jnp.uint8)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.augmented_collision_map.value](state, new_collision_map)
            return state, room_state
        else:
            return montezuma_state, room_state
        
    def handle_conveyor_movement(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # This function handles the movement imparted by the conveyor belt onto the player.
        #
        #
        if RoomTags.CONVEYORBELTS in tags:
            conveyor_belts_attrs: RoomTags.CONVEYORBELTS.value = SANTAH.extract_tag_from_rooms[RoomTags.CONVEYORBELTS](room_state)
            state = montezuma_state
            conveyor_movement_collision_map = conveyor_belts_attrs.global_conveyor_movement_collision_map
            
            # Mimic the "raycast-downwards" implementation to prevent weird glitch when falling down from a conveyor
            # Only check if the middle "feet" pixel are still on the conveyor belt to determine if movement is necessary.
            middle_offset: int = self.consts.PLAYER_WIDTH//2 + state.player_position[0] # X position
            height_offset: int = self.consts.PLAYER_HEIGHT + state.player_position[1] + room_state.vertical_offset[0] - 1 # Y offset
            h_1 = jnp.array([height_offset-1], jnp.uint16)
            player_slice: jArray = jax.lax.dynamic_slice(conveyor_movement_collision_map, start_indices=(middle_offset, height_offset),
                                                         slice_sizes=(self.consts.PLAYER_WIDTH,self.consts.PLAYER_HEIGHT))
            # Conveyor belt map is populated with the movement directions of the individual conveyor belts.
            # Take min, because "NO_DIR" jas max value.
            movement_index = jnp.min(player_slice) 
            player_pos_initial = state.player_position
            conveyor_velocity = jnp.mod(state.frame_count,2)
            # Modify the current player position using the obtained movement direction
            player_pos_new = jax.lax.cond(jnp.equal(movement_index,MovementDirection.LEFT.value),
                                    lambda x: x.at[0].set(x[0]-conveyor_velocity[0]),
                                    lambda x: jax.lax.cond(jnp.equal(movement_index,MovementDirection.RIGHT.value),
                                                 lambda x: x.at[0].set(x[0]+conveyor_velocity[0]),
                                                 lambda x: x,
                                                 operand=player_pos_initial),
                                    operand=player_pos_initial)
            # Player may not be moved if he is climbing, falling or jumping.
            may_not_be_conveyed: jArray = jnp.logical_or(state.is_climbing, jnp.logical_or(state.is_falling, state.is_jumping))
            player_pos = jax.lax.cond(may_not_be_conveyed[0],
                                      lambda _: player_pos_initial,
                                      lambda _: player_pos_new,
                                      operand=None)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, player_pos)
            return state, room_state
        else:
            return montezuma_state, room_state
        
    def handle_darkness(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # If a room implements the dark-room tag, it is not rendered if the player does not possess a torch.
        state = montezuma_state
        darkness = jnp.array([jnp.logical_and(RoomTags.DARKROOM in tags,jnp.logical_not(state.itembar_items[Item_Sprites.TORCH.value].astype(jnp.bool)))],dtype=jnp.bool)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.darkness.value](state, darkness)
        return state, room_state
    
    def add_side_wall_collision_to_room_collision_map(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # Paint the precomputed side wall collision map onto the global collision map.
        #
        #
        if RoomTags.SIDEWALLS in tags:
            sidewalls_attrs: RoomTags.SIDEWALLS.value = SANTAH.extract_tag_from_rooms[RoomTags.SIDEWALLS](room_state)
            state = montezuma_state
            old_collision_map = state.augmented_collision_map
            new_collision_map = jnp.logical_or(sidewalls_attrs.side_walls_collision_map, old_collision_map).astype(jnp.uint8)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.augmented_collision_map.value](state, new_collision_map)
            return state, room_state
        else:
            return montezuma_state, room_state 

    def add_dropout_floor_collision_to_room_collision_map(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # Add collision map for dropout floors onto the global collision map.
        if RoomTags.DROPOUTFLOORS in tags:
            dropout_attrs: RoomTags.DROPOUTFLOORS.value = SANTAH.extract_tag_from_rooms[RoomTags.DROPOUTFLOORS](room_state)
            state = montezuma_state
            old_collision_map = state.augmented_collision_map
            dropout_collision_map_true = dropout_attrs.dropout_floor_colision_map
            dropout_collision_map_false = jnp.zeros(shape=(dropout_collision_map_true.shape),dtype=jnp.uint8)
            # For the first half of the cycle, the dropout map is active.
            # Total cycle length consists of on_time + off_time
            cond = jnp.less_equal(jnp.mod(montezuma_state.frame_count,dropout_attrs.on_time_dropoutfloor[0]+dropout_attrs.off_time_dropoutfloor[0]),dropout_attrs.on_time_dropoutfloor[0])
            dropout_collision_map = jax.lax.cond(cond[0],
                         lambda _: dropout_collision_map_true,
                         lambda _: dropout_collision_map_false,
                         operand=None)
            
            new_collision_map = jnp.logical_or(dropout_collision_map, old_collision_map).astype(jnp.uint8)
            
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.augmented_collision_map.value](state, new_collision_map)
            return state, room_state
        else:
            return montezuma_state, room_state    

    def check_collision_between_two_sprites(self, sprite1, pos1_x, pos1_y, sprite2, pos2_x, pos2_y):
        #
        # This subroutine checks for collisions between two sprites. It is used to check for 
        # collision between player vs enemies & items.
        #
        r1x = pos1_x[0]
        r1w = jnp.shape(sprite1)[0]
        r2x = pos2_x
        r2w = jnp.shape(sprite2)[0]
        
        r1y = pos1_y[0]
        r1h = jnp.shape(sprite1)[1]
        r2y = pos2_y
        r2h = jnp.shape(sprite2)[1]
        return (
            (r1x + r1w >= r2x) &
            (r1x <= r2x + r2w) &
            (r1y + r1h >= r2y) &
            (r1y <= r2y + r2h)
        )
        
    def check_collision_sprite_shapes(self, sprite1_shape1, pos1_x, pos1_y, sprite_shape2, pos2_x, pos2_y):
        r1x = pos1_x
        r1w = sprite1_shape1[0]
        r2x = pos2_x
        r2w = sprite_shape2[0]
        
        r1y = pos1_y
        r1h = sprite1_shape1[1]
        r2y = pos2_y
        r2h = sprite_shape2[1]
        return (
            (r1x + r1w >= r2x) &
            (r1x <= r2x + r2w) &
            (r1y + r1h >= r2y) &
            (r1y <= r2y + r2h)
        ) 
    

    def handle_item_pickup(self, items: RoomTags.ITEMS.value, state: MontezumaState, index, in_bonus_room):
        """This function is called if the player is actually picking up an item

        Args:
            items (RoomTags.ITEMS.value): The Item Tag of the current room
            state (MontezumaState): Current GameState
            index (_type_): Index of the item which should be picked up
            in_bonus_room (_type_): Whether the player is currently in a bonus room.

        Returns:
            _type_: _description_
        """
        # Trigger a game-freeze on item pickup
        state = self.queue_freeze(state, FreezeType.ITEM_PICKUP)
        picked_up_item: jArray = jax.lax.dynamic_index_in_dim(items.items, index=index, axis=0, keepdims=False)
        picked_up_item: Item = SANTAH.full_deserializations[Item](picked_up_item)
        # check if the item is a HAMMER. If hammer: 
            # If no hammer in itembar: Add one to it!
            # If hammer in itembar: Don't add one, but reset counter
        is_hammer = jnp.equal(picked_up_item.sprite, ItemBar_Sprites.HAMMER.value)
        hammer_on_itembar: jArray = jnp.greater_equal(jax.lax.dynamic_index_in_dim(state.itembar_items, index=ItemBar_Sprites.HAMMER.value, axis=0), 1)
        
        # Render item if it is not a hammer, or if we don't already have a hammer on the itembar.
        # We can only have one hammer at a time, since it's a temporary item.
        render_the_item: jArray = jnp.logical_or((1 - is_hammer), (1 - hammer_on_itembar))
        state = jax.lax.cond(render_the_item[0], self.add_item_to_itembar, lambda x, y, z : x, state, items, index)
        
        
        # Reset the lifetime of the current hammer if we have picked up one.
        new_hammer_counter: jArray = jnp.multiply(is_hammer, self.consts.HAMMER_DURATION) + jnp.multiply((1 - is_hammer), state.hammer_time)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.hammer_time.value](state, new_hammer_counter)
        
        # If this is the first item we have picked up, set the first_item_pickup_frame field. 
        # This is done mainly for the bonus room, so then we can start the countdown to the reset.
        new_first_item_pickup = state.first_item_pickup_frame + (state.frame_count - state.first_item_pickup_frame) * jnp.logical_not(state.first_item_pickup_frame)# if item pickup = 0 set to current frame count else hold value of state
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.first_item_pickup_frame.value](state, new_first_item_pickup)

        new_rng_key, subkey = jax.random.split(state.rng_key)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.rng_key.value](state,new_rng_key)
        
        state = self.increase_score_from_item_pickup(state, items, index)
        
        # If we are in the bonus room, we need to update the position of the item we have picked up
        # so that the next game spawns at a different location
        items = jax.lax.cond(in_bonus_room,
                             lambda _: self.set_new_item_position(items, index, subkey, state),
                             lambda _: self.set_item_on_field_false(items, index),
                             operand=None)
        return items, state
    
    def set_new_item_position(self, items: RoomTags.ITEMS.value, index, rng_key, state: MontezumaState):
        # Randomly updates the position of a given item.
        # This is only used for the gems in the bonus room.
        
        des_item: Item = SANTAH.full_deserializations[Item](items[0][index])
        self.consts.DEFAULT_BONUS_ROOM_GEM_X
        right_offset = self.consts.DEFAULT_BONUS_ROOM_GEM_X+self.consts.ITEM_WIDTH
        # Determine random new X-coordinate, the Y coordinate of the gems in the bonus room stays constant.
        random_x = random.uniform(rng_key, dtype=jnp.float32,minval=self.consts.DEFAULT_BONUS_ROOM_GEM_X,maxval=self.consts.WIDTH - (right_offset))
        random_x = jnp.astype(random_x, jnp.int32)
        random_x = jnp.reshape(random_x, (1))
        
        # Update the item with the new position & return the items tag.
        des_item = SANTAH.attribute_setters[Item][ItemEnum.x.value](des_item,random_x)
        item = SANTAH.full_serialisations[Item](des_item)
        new_items = items.items.at[index].set(item)
        items = SANTAH.attribute_setters[MandatoryItemFields][MandatoryItemFieldsEnum.items.value](items, new_items)
        return items
        
    def set_item_on_field_false(self, items: RoomTags.ITEMS.value, index):
        #
        # Simple method to remove an item from the field.
        #
        des_item: Item = SANTAH.full_deserializations[Item](items[0][index])
        des_item = SANTAH.attribute_setters[Item][ItemEnum.on_field.value](des_item,jnp.reshape(False, (1)).astype(jnp.int32))
        item = SANTAH.full_serialisations[Item](des_item)
        new_items = items.items.at[index].set(item)
        items = SANTAH.attribute_setters[MandatoryItemFields][MandatoryItemFieldsEnum.items.value](items, new_items)
        return items
    
    def increase_score_from_item_pickup(self, state: MontezumaState, items: RoomTags.ITEMS.value, index):
        #
        # Increase the game score with the appropriate reward for collecting the given item.
        #
        des_item: Item = SANTAH.full_deserializations[Item](items[0][index])
        item_value = self.item_scores[des_item.sprite]
        old_score = state.score
        new_score = item_value + old_score
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.score.value](state, new_score)
        return state
    
    def add_item_to_itembar(self, state: MontezumaState, items: RoomTags.ITEMS.value, index):
        #
        # Update the itembar array in the state with the new collected item.
        #
        des_item: Item = SANTAH.full_deserializations[Item](items[0][index])
        new_itembar = state.itembar_items.at[des_item.sprite[0]].set(state.itembar_items[des_item.sprite[0]]+1)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.itembar_items.value](state, new_itembar)
        return state
    
    def handle_item_collision_check(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # This function handles the logic for picking up items.
        #
        if RoomTags.ITEMS in tags:
            item_attrs: RoomTags.ITEMS.value = SANTAH.extract_tag_from_rooms[RoomTags.ITEMS](room_state)
            # TODO: Refactor with VMAP.
            def single_item_check(i, items_and_room : Tuple[RoomTags.ITEMS.value, MontezumaState]):
                #
                # Handle pickup logic for a single item.
                #
                items: RoomTags.ITEMS.value
                state: MontezumaState
                items, state = items_and_room
                
                des_item: Item = SANTAH.full_deserializations[Item](items[0][i])
                item_sprite = jnp.ones(shape=(8,7),dtype=jnp.bool)
                # TODO: use the shape based method here
                # Check if the player collides with the item.
                player_sprite: jArray = jnp.ones(shape=(self.consts.ITEM_COLLISION_PLAYER_WIDTH, self.consts.PLAYER_HEIGHT), dtype=jnp.int32)
                altered_x_pos: jArray = montezuma_state.player_position[0] + (self.consts.PLAYER_WIDTH - self.consts.ITEM_COLLISION_PLAYER_WIDTH)//2
                col = self.check_collision_between_two_sprites(item_sprite,des_item.x,des_item.y,player_sprite,altered_x_pos,state.player_position[1])
                # Collision is only detected if the item is still on the field.
                col = jnp.minimum(col,des_item.on_field)
                # Check if player has already reached the maximum number of collected items, 
                # if this is the case, the player can't pick up any more items.
                
                # Gems aren't counted for this purpose.
                check_itembar_items: jArray = state.itembar_items.at[Item_Sprites.GEM.value].set(0)
                can_pickup_more_items: jArray = jnp.less(jnp.sum(check_itembar_items), self.consts.MAXIMUM_ITEM_NUMBER)
                space_in_inventory = jnp.multiply(col, can_pickup_more_items)
                # Handle item pickup if the item may be picked up, otherwise do nothing.
                items, state = jax.lax.cond(jnp.logical_or(jnp.logical_and(col[0],jnp.equal(des_item.sprite[0],Item_Sprites.GEM.value)),space_in_inventory[0]),
                                            lambda x: self.handle_item_pickup(items, state, i, RoomTags.BONUSROOM in tags),
                                            lambda x: x,
                                            operand=(items, state))
                                
                return items, state
            
            items, state = jax.lax.fori_loop(lower=0,upper=jnp.shape(item_attrs[0])[0],body_fun=single_item_check,init_val=(item_attrs,montezuma_state))
            room = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.ITEMS](room_state,items)
        
            return state, room
           
        return montezuma_state, room_state
    
    def _get_enemy_collision_size(self, enemy: Enemy) -> jArray:
        #
        # Return the collision sizes for the various enemies.
        #
        def collision_size_spider() -> jArray:
            ret = jnp.array([SPIDER_BEHAVIOR.collision_size[0], SPIDER_BEHAVIOR.collision_size[1]], jnp.int32)
            return ret
        def collision_size_snake() -> jArray:
            ret = jnp.array([SNAKE_BEHAVIOR.collision_size[0], SNAKE_BEHAVIOR.collision_size[1]], jnp.int32)
            return ret
        def collision_size_bounce_skull() -> jArray:
            ret = jnp.array([B_SCULL_BEHAVIOR.collision_size[0], B_SCULL_BEHAVIOR.collision_size[1]], jnp.int32)
            return ret
        def collision_size_rolling_skull() -> jArray:
            ret = jnp.array([ROLLING_SKULL_BEHAVIOR.collision_size[0], ROLLING_SKULL_BEHAVIOR.collision_size[1]], jnp.int32)
            return ret
        collision_size: jArray = jax.lax.switch(enemy.enemy_type[0], [collision_size_snake, collision_size_rolling_skull, 
                collision_size_bounce_skull, collision_size_spider])
        return collision_size
        
    def _get_new_alive_counter_value(self, bounce_skull_alive_state: jArray, is_left_colliding: jArray, 
                                            is_right_colliding: jArray) -> jArray:
        """The BounceSkull enemy consists of a left & a right half. The left & right half 
            can be killed independantly, this function computes the new alive-state of the 
            bounce skull, depending on which side the player currently collides with.

        Args:
            bounce_skull_alive_state (jArray): The old alive state of the bounce_skull
            is_left_colliding (jArray): Whether we have a collision with the left skull. 
                We assume that we have checked whether there are actual collisions.
            is_right_colliding (jArray): Whether we have a collision with the right skull. 
                Again, assume we have checked whether the collision actually occurs.

        Returns:
            jArray: New Alive state for the bounce skull.
        """
        # Compute which side of the bounce skull is still alive after the detected collision.
        needs_to_be_right_is_dead: jArray = jnp.logical_and(jnp.equal(bounce_skull_alive_state, BounceSkullAliveState.FULLY_ALIVE.value), is_right_colliding)
        needs_to_be_left_is_dead: jArray = jnp.logical_and(jnp.equal(bounce_skull_alive_state, BounceSkullAliveState.FULLY_ALIVE.value), is_left_colliding)
        needs_to_be_fully_dead: jArray = jnp.logical_or(
            jnp.logical_and(jnp.equal(bounce_skull_alive_state, BounceSkullAliveState.LEFT_DEAD.value), is_right_colliding), 
            jnp.logical_and(jnp.equal(bounce_skull_alive_state, BounceSkullAliveState.RIGHT_DEAD.value), is_left_colliding)
        )
        # Alive state of the bounce skull does not change if we collide with neither side.
        does_not_change: jArray = jnp.logical_and(jnp.logical_not(is_left_colliding), jnp.logical_not(is_right_colliding))
        return_alive: jArray = jnp.multiply(needs_to_be_right_is_dead, BounceSkullAliveState.RIGHT_DEAD.value) + jnp.multiply(needs_to_be_left_is_dead, BounceSkullAliveState.LEFT_DEAD.value) + jnp.multiply(needs_to_be_fully_dead, BounceSkullAliveState.ALL_DEAD.value) + jnp.multiply(does_not_change, bounce_skull_alive_state)
        return return_alive
    
    
    def _check_single_enemy_colision_bounce_skull(self, bounce_skull: Enemy, state: MontezumaState) -> Tuple[Enemy, MontezumaState]:
        # Seperate function to handle collision check with the bounce skull
        # This is needed, because the bounce skull consists of two halves which can be killed independantly.
        
        # Split the collision sprite of the bounce skull in 2 and check collision with the player seperately for both sprites
        enemy_collision_size: jArray = self._get_enemy_collision_size(enemy=bounce_skull)
        # DOUBLE_SKULL_MIDDLE_DEMARKATION_LINE_OFFSET determines how much we cut out of the middle of the sprite.
        left_collision_size: jArray = jnp.array([enemy_collision_size[0]//2 - self.consts.DOUBLE_SKULL_MIDDLE_DEMARKATION_LINE_OFFSET, enemy_collision_size[1]])
        right_collision_size: jArray = jnp.array([enemy_collision_size[0]//2 - self.consts.DOUBLE_SKULL_MIDDLE_DEMARKATION_LINE_OFFSET, enemy_collision_size[1]])
        left_collision_offset: jArray = jnp.array([0, 0], jnp.int32)
        right_side_offset: jArray = jnp.array([enemy_collision_size[0]//2 + self.consts.DOUBLE_SKULL_MIDDLE_DEMARKATION_LINE_OFFSET + 1, 0])
        
        left_is_colliding: jArray = self.check_collision_sprite_shapes(sprite1_shape1=(self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT), 
                                                                  pos1_x=state.player_position[0], 
                                                                  pos1_y=state.player_position[1], 
                                                                  sprite_shape2=left_collision_size, 
                                                                  pos2_x=bounce_skull.pos_x + left_collision_offset[0], pos2_y=bounce_skull.pos_y + left_collision_offset[1])
        right_is_colliding: jArray = self.check_collision_sprite_shapes(sprite1_shape1=(self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT), 
                                                                  pos1_x=state.player_position[0], 
                                                                  pos1_y=state.player_position[1], 
                                                                  sprite_shape2=right_collision_size, 
                                                                  pos2_x=bounce_skull.pos_x + right_side_offset[0], pos2_y=bounce_skull.pos_y + right_side_offset[1])
        # We need to check if the appropriate skull halves are actually still alive.
        is_left_side_still_alive: jArray = jnp.logical_or(jnp.equal(bounce_skull.optional_utility_field, BounceSkullAliveState.FULLY_ALIVE.value), 
                                                          jnp.equal(bounce_skull.optional_utility_field, BounceSkullAliveState.RIGHT_DEAD.value))
        is_right_side_still_alive: jArray = jnp.logical_or(jnp.equal(bounce_skull.optional_utility_field, BounceSkullAliveState.FULLY_ALIVE.value), 
                                                          jnp.equal(bounce_skull.optional_utility_field, BounceSkullAliveState.LEFT_DEAD.value))
        # Collision with a side is only detected if the side is still alive.
        left_is_colliding = jnp.multiply(left_is_colliding, is_left_side_still_alive)
        right_is_colliding = jnp.multiply(right_is_colliding, is_right_side_still_alive)
        
        # If both are colliding: left takes precedence over right, so we don't kill both enemies in one collision
        both_colliding: jArray = jnp.logical_and(left_is_colliding, right_is_colliding)
        right_is_colliding = jnp.multiply(both_colliding, 0) + jnp.multiply(1-both_colliding, right_is_colliding)
        
        # If the player is either holding the hammer or falling, it can't collide with either.
        is_hammering: jArray = jnp.greater(state.itembar_items[ItemBar_Sprites.HAMMER.value], 0)
        may_not_collide: jArray = jnp.logical_or(is_hammering, jnp.logical_or(state.is_falling, state.is_dying))
        
        left_is_colliding = jnp.multiply(left_is_colliding, 1 - may_not_collide)
        right_is_colliding = jnp.multiply(right_is_colliding, 1 - may_not_collide)
        
        # Get the new value for the optional_utility_field to determine the alive state of the enemy.
        new_optional_utility_field: jArray = self._get_new_alive_counter_value(
            bounce_skull_alive_state=bounce_skull.optional_utility_field, 
            is_left_colliding=left_is_colliding, 
            is_right_colliding=right_is_colliding
        )
        is_colliding: jArray = jnp.logical_or(left_is_colliding, right_is_colliding)
        # Player is only killed if he doesn't have a sword.
        has_sword: jArray = jnp.greater(jax.lax.dynamic_index_in_dim(state.itembar_items, index=ItemBar_Sprites.SWORD.value, 
                                                         axis=0), 0)
        
        
        # Do the score increase here: Score is increased if Enemy & player are colliding & if the player has a sword
        score_increase: jArray = jnp.logical_and(is_colliding, has_sword)
        state = jax.lax.cond(score_increase[0], partial(self.queue_freeze, freeze_type=FreezeType.KILLED_A_MONSTER), lambda x : x, state)
        enemy_score: jArray = jax.lax.dynamic_index_in_dim(self.consts.ENEMY_POINTS,  index=bounce_skull.enemy_type[0], 
                                                           axis=0)
        new_score: jArray = state.score + jnp.multiply(score_increase, enemy_score)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.score.value](state, new_score)
        
        
        player_death: jArray = jnp.logical_and(is_colliding, jnp.logical_not(has_sword))
        
        # Handle trigger for player monster death animation:
        
        
        state = jax.lax.cond(player_death[0], partial(self.queue_freeze, freeze_type=FreezeType.KILLED_BY_MONSTER), lambda x : x, state)
        new_is_dying = jnp.logical_or(player_death,state.is_dying[0])
        new_is_dying = jnp.astype(new_is_dying, jnp.int32)
        new_is_dying = jnp.reshape(new_is_dying, (1))
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_dying.value](state, new_is_dying)
        
        # If the player holds a sword, it is removed on collision with the enemy.
        sword_count: jArray = jax.lax.dynamic_index_in_dim(state.itembar_items, index=ItemBar_Sprites.SWORD.value, 
                                                         axis=0)
        sword_count = sword_count - is_colliding
        sword_count = jax.lax.max(sword_count, 0)
        itembar: jArray = state.itembar_items
        itembar = jax.lax.dynamic_update_index_in_dim(operand=itembar, 
                                                      update=sword_count, 
                                                      index=ItemBar_Sprites.SWORD.value, 
                                                      axis=0)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.itembar_items.value](state, itembar)
        
        # Now set the enemy to "dead" if it has collided with the player & both sides are dead.
        new_enemy_alive_state: jArray = jnp.not_equal(new_optional_utility_field, BounceSkullAliveState.ALL_DEAD.value)
        new_enemy_alive_state = jnp.astype(jnp.reshape(new_enemy_alive_state, (1)), jnp.int32)
        bounce_skull = SANTAH.attribute_setters[Enemy][EnemyFields.alive.value](bounce_skull, new_enemy_alive_state)
        bounce_skull = SANTAH.attribute_setters[Enemy][EnemyFields.optional_utility_field.value](bounce_skull, new_optional_utility_field)
        
        return bounce_skull, state
    
    def _check_single_enemy_colision_non_bounce_skull(self, enemy: Enemy, state: MontezumaState) -> Tuple[Enemy, MontezumaState]:
        #
        # This function handles player & enemy collision for all other enemy types
        #
        
        enemy_collision_size: jArray = self._get_enemy_collision_size(enemy=enemy)
        is_colliding: jArray = self.check_collision_sprite_shapes(sprite1_shape1=(self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT), 
                                                                  pos1_x=state.player_position[0], 
                                                                  pos1_y=state.player_position[1], 
                                                                  sprite_shape2=enemy_collision_size, 
                                                                  pos2_x=enemy.pos_x, pos2_y=enemy.pos_y)
        # Collision is only detected if the enemy is still alive.
        is_colliding = jnp.logical_and(is_colliding, enemy.alive)
        # Collision is also only detected if the player is not holding a hammer
        is_hammering: jArray = jnp.greater(state.itembar_items[ItemBar_Sprites.HAMMER.value], 0)
        is_colliding = jnp.logical_and(is_colliding, (1 - is_hammering))
        is_colliding = jnp.logical_and(is_colliding, 1 - state.is_falling)
        is_colliding = jnp.logical_and(is_colliding, (1 - state.is_dying))
        
        # Player is only killed if he doesn't have a sword.
        has_sword: jArray = jnp.greater(jax.lax.dynamic_index_in_dim(state.itembar_items, index=ItemBar_Sprites.SWORD.value, 
                                                         axis=0), 0)
        
        
        # Do the score increase here: Score is increased if Enemy & player are colliding & if the player has a sword
        score_increase: jArray = jnp.logical_and(is_colliding, has_sword)
        state = jax.lax.cond(score_increase[0], partial(self.queue_freeze, freeze_type=FreezeType.KILLED_A_MONSTER), lambda x : x, state)
        enemy_score: jArray = jax.lax.dynamic_index_in_dim(self.consts.ENEMY_POINTS,  index=enemy.enemy_type[0], 
                                                           axis=0)
        new_score: jArray = state.score + jnp.multiply(score_increase, enemy_score)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.score.value](state, new_score)
        
        
        # Player only dies if he doesn't have a sword
        player_death: jArray = jnp.logical_and(is_colliding, jnp.logical_not(has_sword))
        
        # Handle trigger for death animation.
        state = jax.lax.cond(player_death[0], partial(self.queue_freeze, freeze_type=FreezeType.KILLED_BY_MONSTER), lambda x : x, state)
        new_is_dying = jnp.logical_or(player_death,state.is_dying[0])
        new_is_dying = jnp.astype(new_is_dying, jnp.int32)
        new_is_dying = jnp.reshape(new_is_dying, (1))
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_dying.value](state, new_is_dying)
        
        # Update the sword count of the player
        sword_count: jArray = jax.lax.dynamic_index_in_dim(state.itembar_items, index=ItemBar_Sprites.SWORD.value, 
                                                         axis=0)
        sword_count = sword_count - is_colliding
        sword_count = jax.lax.max(sword_count, 0)
        itembar: jArray = state.itembar_items
        itembar = jax.lax.dynamic_update_index_in_dim(operand=itembar, 
                                                      update=sword_count, 
                                                      index=ItemBar_Sprites.SWORD.value, 
                                                      axis=0)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.itembar_items.value](state, itembar)
        
        # Now set the enemy to "dead" if it has collided with the player
        new_enemy_alive_state: jArray = jnp.multiply(is_colliding, jnp.array([0], jnp.int32)) + jnp.multiply((1 - is_colliding), enemy.alive)
        enemy = SANTAH.attribute_setters[Enemy][EnemyFields.alive.value](enemy, new_enemy_alive_state)
        
        
        
        return enemy, state
            
    def handle_enemy_collision_check(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # This function handles player-enemy collision.
        #
        if RoomTags.ENEMIES in tags:
            enemies_attr: RoomTags.ENEMIES.value = SANTAH.extract_tag_from_rooms[RoomTags.ENEMIES](room_state)
            
            def single_enemy_check(i, enemies_and_state : Tuple[jArray, MontezumaState]):
                # Handle collision for a single enemy
                #
                
                # Retreive the current enemy from the state
                enemies: jArray = None
                state: MontezumaState = None
                enemies, state = enemies_and_state
                single_enemy_arr: jArray = jax.lax.dynamic_index_in_dim(enemies, index=i, axis=0, keepdims=False)
                enemy: Enemy = SANTAH.full_deserializations[Enemy](single_enemy_arr)
                is_bounce_skull: jArray = jnp.equal(enemy.enemy_type, EnemyType.BOUNCE_SKULL.value)
                # Apply seperate collision routines for Bounce Skull & all other enemies.
                enemy, state = jax.lax.cond(is_bounce_skull[0], self._check_single_enemy_colision_bounce_skull, self._check_single_enemy_colision_non_bounce_skull, enemy, state)
                
                # Write the enemy back to the state.
                single_enemy_arr: jArray = SANTAH.full_serialisations[Enemy](enemy)
                single_enemy_arr = jnp.expand_dims(single_enemy_arr, axis=0)
                enemies = jax.lax.dynamic_update_slice_in_dim(operand=enemies, 
                                                                      update=single_enemy_arr, 
                                                                      start_index=i, 
                                                                      axis=0)
                
                
                return enemies, state
            
            
            enemies: jArray = None
            enemies, montezuma_state = jax.lax.fori_loop(lower=0,upper=jnp.shape(enemies_attr.enemies)[0],body_fun=single_enemy_check,
                                               init_val=(enemies_attr.enemies, montezuma_state))
            enemies_attr = SANTAH.attribute_setters[RoomTags.ENEMIES.value][RoomTagsNames.ENEMIES.value.enemies.value](enemies_attr, enemies)
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.ENEMIES](room_state, enemies_attr)
        
            
        return montezuma_state, room_state
    
    @partial(jax.jit, static_argnames=["self", "behavior"])
    def _handle_rolling_skulls_movement(self, enemy: Enemy, behavior: RollingSkullOrSpiderBehavior) -> Enemy:
        """
        Handles the movement behavior of the rolling skull enemies.
        Assumes the enemy is still alive, this needs to be checked externally.
        """
        
        def do_movement(enemy: Enemy) -> Enemy:
            # This function is called if the enemy needs to actually move on the current frame
            
            # Clip the enemy position to the allowed bounding box.
            _x: jArray = jax.lax.max(enemy.pos_x, enemy.bbox_left_upper_x)
            _x: jArray = jax.lax.min(_x, enemy.bbox_right_lower_x - behavior.collision_size[0] - 1)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_x.value](enemy, _x)
            _y: jArray = jax.lax.max(enemy.pos_y, enemy.bbox_left_upper_y)
            _y: jArray = jax.lax.min(_y , enemy.bbox_right_lower_y - behavior.collision_size[1] - 1)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_y.value](enemy, _y)
            y_position = enemy.bbox_right_lower_y - behavior.collision_size[1]
            
            # Determine proposed new X-coordinate, by either addint or subtracting the movement distance depending on 
            # the current enemy direction.
            is_left = jnp.equal(enemy.horizontal_direction, Horizontal_Direction.LEFT.value)
            is_right = jnp.equal(enemy.horizontal_direction, Horizontal_Direction.RIGHT.value)
            x_offset = jnp.multiply(behavior.movement_distance*(-1), is_left) + jnp.multiply(behavior.movement_distance*(1), is_right)
            proposed_new_x: jArray = enemy.pos_x + x_offset
            
            # Now check if our new X-coordinate would lay outside the bounding box.
            is_illegal_x_coordinate: jArray = jnp.logical_or(jnp.less(proposed_new_x, enemy.bbox_left_upper_x), 
                                            jnp.greater_equal(proposed_new_x + behavior.collision_size[0], enemy.bbox_right_lower_x))
            
            # Only update the X coordinate if the coordinate is legal.
            new_x_coord = jnp.multiply(is_illegal_x_coordinate, enemy.pos_x) + jnp.multiply(1 - is_illegal_x_coordinate, proposed_new_x)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_x.value](enemy, new_x_coord)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_y.value](enemy, y_position)
            
            # If the coordinate lays outside the bounding box, reverse the movement direction (the enemy has reached the edge of the bounding box)
            new_horizontal_direction: jArray = jnp.multiply(jnp.equal(enemy.horizontal_direction, Horizontal_Direction.LEFT.value), 
                jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32)) + jnp.multiply(jnp.equal(enemy.horizontal_direction, Horizontal_Direction.RIGHT.value), 
                jnp.array([Horizontal_Direction.LEFT.value], jnp.int32))
            new_horizontal_direction = jnp.multiply(is_illegal_x_coordinate, new_horizontal_direction) + jnp.multiply(1 - is_illegal_x_coordinate, enemy.horizontal_direction)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.horizontal_direction.value](enemy, new_horizontal_direction)
            
            # Indicate that we have moved at this frame, necessary if we do not execute movement in every frame.
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.last_movement.value](enemy, jnp.array([0], jnp.int32))
            
            # reverse the render order if we have reached the edge of the bounding box.
            new_render_direction = jnp.multiply(is_illegal_x_coordinate, (1 - enemy.render_in_reverse)) + jnp.multiply((1-is_illegal_x_coordinate), enemy.render_in_reverse)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.render_in_reverse.value](enemy, new_render_direction)
            return enemy         
        
        def do_no_movement(enemy: Enemy) -> Enemy:
            
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.last_movement.value](enemy, enemy.last_movement + 1)   
            return enemy
        
        
        # Only move every "move_every_nth_frame"
        
        needs_to_move = jnp.greater_equal(enemy.last_movement, behavior.move_every_nth_frame)
        enemy = jax.lax.cond(needs_to_move[0], do_movement, do_no_movement, enemy)
        
        return enemy
    
    
    @partial(jax.jit, static_argnames=["self"])
    def _handle_snake_movement(self, enemy: Enemy) -> Enemy:
        #
        # Snakes do not move, they stay at their initial position.
        #
        x_pos = enemy.initial_x_pos
        y_pos = enemy.initial_y_pos
        enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_x.value](enemy, x_pos)
        enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_y.value](enemy, y_pos)
        
        return enemy
    
    @partial(jax.jit, static_argnames=["self"])
    def _handle_bounce_skull_movement(self, enemy: Enemy) -> Enemy:
        """
        Handles the movement behavior of the bouncing skull enemies.
        Assumes the enemy is still alive, needs to be checked externally.
        """
        
        
        def do_movement(enemy: Enemy) -> Enemy:
            
            # At the beginning of the movement routine, move the enemy back onto a legal position. This might
            # be neccesary if the init position is badly chosen
            _x: jArray = jax.lax.max(enemy.pos_x, enemy.bbox_left_upper_x)
            _x: jArray = jax.lax.min(_x, enemy.bbox_right_lower_x - B_SCULL_BEHAVIOR.collision_size[0])
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_x.value](enemy, _x)
            _y: jArray = jax.lax.max(enemy.pos_y, enemy.bbox_left_upper_y)
            _y: jArray = jax.lax.min(_y , enemy.bbox_right_lower_y - B_SCULL_BEHAVIOR.collision_size[1])
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_y.value](enemy, _y)
            _is_illegal_x_coordinate: jArray = jnp.logical_or(jnp.less(enemy.pos_x, enemy.bbox_left_upper_x), 
                                            jnp.greater(enemy.pos_x + B_SCULL_BEHAVIOR.collision_size[0], enemy.bbox_right_lower_x))
            
            # Determine new proposed X coordinate
            is_left = jnp.equal(enemy.horizontal_direction, Horizontal_Direction.LEFT.value)
            is_right = jnp.equal(enemy.horizontal_direction, Horizontal_Direction.RIGHT.value)
            x_offset = jnp.multiply(B_SCULL_BEHAVIOR.x_movement_distance*(-1), is_left) + jnp.multiply(B_SCULL_BEHAVIOR.x_movement_distance*(1), is_right)
            
            
            
            proposed_new_x: jArray = enemy.pos_x + x_offset
            
            # Now check if our new X-coordinate would crash with the bounding box. If so, reverse directions
            is_illegal_x_coordinate: jArray = jnp.logical_or(jnp.less(proposed_new_x, enemy.bbox_left_upper_x), 
                                            jnp.greater(proposed_new_x + B_SCULL_BEHAVIOR.collision_size[0], enemy.bbox_right_lower_x))
            # Set the new position, if new X-coord lies outside the bounding box, don't change position
            new_x_coord = jnp.multiply(is_illegal_x_coordinate, enemy.pos_x) + jnp.multiply(1 - is_illegal_x_coordinate, proposed_new_x)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_x.value](enemy, new_x_coord)
            
            
            # Now compute the new y-position:
            current_y_offset: jArray = jax.lax.dynamic_index_in_dim(B_SCULL_BEHAVIOR.bounce_pattern, index=enemy.optional_movement_counter[0], 
                                                                    axis=0, keepdims=False)
            
            proposed_new_y_position: jArray = enemy.pos_y - current_y_offset
            
            # If the enemy has arrived at the beginning of it's bounce cycle, reset y pos to init y_pos to prevent compounding effects of bounce 
            # cycles that do not have the same start & stop position.
            # This is done before clipping, to prevent the enemy from going out of bounds
            y_reset_pos = enemy.bbox_right_lower_y - B_SCULL_BEHAVIOR.collision_size[1]
            do_y_reset: jArray = jnp.equal(enemy.optional_movement_counter, 0)
            y_pos = jnp.multiply((1 - do_y_reset), proposed_new_y_position) + jnp.multiply(do_y_reset, y_reset_pos)
            
            
            # Check if the y position is in bounds & rectify
            y_pos: jArray = jax.lax.max(y_pos, enemy.bbox_left_upper_y)
            y_pos: jArray = jax.lax.min(y_pos, enemy.bbox_right_lower_y - B_SCULL_BEHAVIOR.sprite_size[1])
            
            
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.pos_y.value](enemy, y_pos)
            # If it is an illegal x coordinate, stay in the same place but reverse the movement direction
            
            # Reverse the horizontal direction
            new_horizontal_direction: jArray = jnp.multiply(jnp.equal(enemy.horizontal_direction, Horizontal_Direction.LEFT.value), 
                jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32)) + jnp.multiply(jnp.equal(enemy.horizontal_direction, Horizontal_Direction.RIGHT.value), 
                jnp.array([Horizontal_Direction.LEFT.value], jnp.int32))
            
            new_horizontal_direction = jnp.multiply(is_illegal_x_coordinate, new_horizontal_direction) + jnp.multiply(1 - is_illegal_x_coordinate, enemy.horizontal_direction)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.horizontal_direction.value](enemy, new_horizontal_direction)
            
            # Now update the bounce index. The enemy.optional_movement_counter is used for this
            new_render_direction = jnp.multiply(is_illegal_x_coordinate, (1 - enemy.render_in_reverse)) + jnp.multiply((1-is_illegal_x_coordinate), enemy.render_in_reverse)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.render_in_reverse.value](enemy, new_render_direction)

            proposed_bounce_index: jArray = enemy.optional_movement_counter + 1
            actual_new_bounce_index: jArray = jnp.mod(proposed_bounce_index, B_SCULL_BEHAVIOR.bounce_pattern_length)
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.optional_movement_counter.value](enemy, actual_new_bounce_index)        
            
            
            # Indicate that we have now moved
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.last_movement.value](enemy, jnp.array([0], jnp.int32))
            return enemy         
            
        
        def do_no_movement(enemy: Enemy) -> Enemy:
            
            enemy = SANTAH.attribute_setters[Enemy][EnemyFields.last_movement.value](enemy, enemy.last_movement + 1)   
            return enemy
        
        
        needs_to_move = jnp.greater_equal(enemy.last_movement, B_SCULL_BEHAVIOR.move_every_ntgh_frame)
        enemy = jax.lax.cond(needs_to_move[0], do_movement, do_no_movement, enemy)
        
        return enemy
    
    
    @partial(jax.jit, static_argnames=["self"])
    def handle_single_enemy_movement(self, single_enemy: jArray) -> jArray:
        single_enemy: Enemy = SANTAH.full_deserializations[Enemy](single_enemy)
        
        # Spider & rolling skull share the same movement routine, they differ only in animation.
        single_enemy = jax.lax.switch(single_enemy.enemy_type[0], [self._handle_snake_movement, partial(self._handle_rolling_skulls_movement, behavior=ROLLING_SKULL_BEHAVIOR), 
                self._handle_bounce_skull_movement, partial(self._handle_rolling_skulls_movement, behavior=SPIDER_BEHAVIOR)], single_enemy)
        single_enemy = SANTAH.full_serialisations[Enemy](single_enemy)
        return single_enemy
    
    
    
    def handle_enemy_movement(self, enemy_tag: RoomTags.ENEMIES.value):
        """Handles movement for all enemies in a room.
        """
        enemies: jArray = enemy_tag.enemies
        
        enemies = self.vmapped_single_enemy_movement(enemies)
       
        enemy_tag = SANTAH.attribute_setters[RoomTags.ENEMIES.value][RoomTagsNames.ENEMIES.value.enemies.value](enemy_tag, enemies)
        return enemy_tag
    
    
    # Logic for advancing the animation counters for all enemies
    
    
    
    def _handle_snake_anmiation(self, enemy: Enemy) -> Enemy:
        # The Snake just flips between two sprites
        counter = jnp.mod(enemy.optional_movement_counter + 1, 2*SNAKE_BEHAVIOR.frame_length)
        enemy = SANTAH.attribute_setters[Enemy][EnemyFields.optional_movement_counter.value](enemy, counter)
        return enemy
    
    def _handle_bounce_skull_animation(self, enemy: Enemy) -> Enemy:
        # The Bounce skull is not animated.
        return enemy
    
    
    @partial(jax.jit, static_argnames=["self", "behavior"])
    def _handle_rolling_skull_animation(self, enemy: Enemy, behavior: RollingSkullOrSpiderBehavior) -> Enemy:
        #
        # This function handles both the animation for rolling skull & for the spider. They only differ in the number of animation frames.
        #
        
        
        # Enemies change sprite 'animate_n_sprites_after_movement' after the last movement.
        animate_this_frame: jArray = jnp.equal(enemy.last_animation, behavior.animate_n_sprites_after_movement)
        # Consider if we need to render in reverse direction
        proposed_sprite_index: jArray = jnp.add(enemy.sprite_index, enemy.render_in_reverse*(-1) + (1 - enemy.render_in_reverse)*(1))
        actual_proposed_sprite_index: jArray = jnp.mod(proposed_sprite_index, behavior.num_sprites)
        actual_sprite_index = jnp.multiply(animate_this_frame, actual_proposed_sprite_index) + jnp.multiply((1-animate_this_frame), enemy.sprite_index)
        enemy = SANTAH.attribute_setters[Enemy][EnemyFields.sprite_index.value](enemy, actual_sprite_index)
        # If we have animated this step, write this to the enemy state.
        new_last_animation: jArray = jnp.multiply(animate_this_frame, 0) + jnp.multiply(1 - animate_this_frame, enemy.last_animation + 1)
        enemy = SANTAH.attribute_setters[Enemy][EnemyFields.last_animation.value](enemy, new_last_animation)
        return enemy
    
    
    
    def _handle_single_enemy_animation(self, single_enemy: jArray) -> jArray:
        single_enemy: Enemy = SANTAH.full_deserializations[Enemy](single_enemy)
        
        single_enemy = jax.lax.switch(single_enemy.enemy_type[0], [self._handle_snake_anmiation, partial(self._handle_rolling_skull_animation, behavior=ROLLING_SKULL_BEHAVIOR), 
                self._handle_bounce_skull_animation, partial(self._handle_rolling_skull_animation, behavior=SPIDER_BEHAVIOR)], single_enemy)
        single_enemy = SANTAH.full_serialisations[Enemy](single_enemy)
        return single_enemy
    
    def handle_enemy_animation(self, enemy_tag: RoomTags.ENEMIES.value):
        """Handles movement for all enemies in a room.
        """
        enemies: jArray = enemy_tag.enemies
        
        # Use VMAP to increase performance.
        enemies = self.vmapped_single_enemy_animation(enemies)
       
        enemy_tag = SANTAH.attribute_setters[RoomTags.ENEMIES.value][RoomTagsNames.ENEMIES.value.enemies.value](enemy_tag, enemies)
        return enemy_tag
    
    
    def handle_enemies(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # Hanlde animation & movement for all enemies in the current room.
        #
        if RoomTags.ENEMIES in tags:
            enemy_tag: RoomTags.ENEMIES.value = SANTAH.extract_tag_from_rooms[RoomTags.ENEMIES](room_state)
            enemy_tag = self.handle_enemy_animation(enemy_tag=enemy_tag)
            
            enemy_tag = self.handle_enemy_movement(enemy_tag)
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.ENEMIES](room_state, enemy_tag)
            return montezuma_state, room_state
        else:
            return montezuma_state, room_state

    
    
    # 
    # Here is all the logic for the observations.
    #
    def _find_enemy_distance_vmappable(self, enemy_arr: jArray, player_pos: jArray):
        enemy: Enemy = SANTAH.full_deserializations[Enemy](enemy_arr)
        dist: jArray = jnp.abs(enemy.pos_x - player_pos[0]) + jnp.abs(enemy.pos_y - player_pos[1])
        return dist
    
    def get_nearest_enemy_observation(self, montezuma_state: MontezumaState, enemy_tag: RoomTags.ENEMIES.value) -> EnemyObservation:
        enemy_distances: jArray = jax.vmap(fun=self._find_enemy_distance_vmappable, 
                                           in_axes=(0, None), out_axes=0)(enemy_tag.enemies, montezuma_state.player_position)
        nearest_enemy_index: jArray = jnp.argmin(enemy_distances, keepdims=False)
        nearest_enemy: jArray = jax.lax.dynamic_index_in_dim(enemy_tag.enemies, index=nearest_enemy_index, 
                                                             axis=0, keepdims=False)
        nearest_enemy: Enemy = SANTAH.full_deserializations[Enemy](nearest_enemy)
        enemy_obs: EnemyObservation = EnemyObservation(
            position=jnp.array([nearest_enemy.pos_x[0], nearest_enemy.pos_y[0]], jnp.int32), 
            type=nearest_enemy.enemy_type[0], 
            alive=nearest_enemy.alive[0], 
            dummy=jnp.array(0, jnp.int32)
        )
        return enemy_obs
    
    def _find_item_distance_vmappable(self, item_arr: jArray, player_pos: jArray):
        item: Item = SANTAH.full_deserializations[Item](item_arr)
        dist: jArray = jnp.abs(item.x - player_pos[0]) + jnp.abs(item.y - player_pos[1])
        return dist
    
    def get_nearest_item_observation(self, montezuma_state: MontezumaState, item_tag: RoomTags.ITEMS.value) -> ItemObservation:
        item_distances: jArray = jax.vmap(fun=self._find_item_distance_vmappable, 
                                           in_axes=(0, None), out_axes=0)(item_tag.items, montezuma_state.player_position)
        nearest_item_index: jArray = jnp.argmin(item_distances, keepdims=False)
        nearest_item: jArray = jax.lax.dynamic_index_in_dim(item_tag.items, index=nearest_item_index, 
                                                             axis=0, keepdims=False)
        nearest_item: Item = SANTAH.full_deserializations[Item](nearest_item)
        item_obs: ItemObservation = ItemObservation(
            position=jnp.array([nearest_item.x[0], nearest_item.y[0]], jnp.int32), 
            type=nearest_item.sprite[0], 
            collected=nearest_item.on_field[0], 
            dummy=jnp.array(0, jnp.int32)
        )
        return item_obs
    
    
        
    
    def _wrappable_get_montezuma_observation(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        initial_painted_canvas: jArray = jnp.ones(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.int32)
        
        # First paint the general collision map onto the canvas
        room_specific_collision_map: jnp.ndarray = getattr(room_state, FieldsThatAreSharedByAllRoomsButHaveDifferentShape.room_collision_map.value)
        room_specific_collision_map = jnp.astype(room_specific_collision_map, jnp.uint8)
        initial_painted_canvas = jax.lax.dynamic_update_slice(operand=initial_painted_canvas, 
                                     update=jnp.astype(room_specific_collision_map, jnp.int32)*ObservationCollisionMapAnnotationColors.normal_wall.value, 
                                     start_indices=(0, room_state.vertical_offset[0]))
        # Now we generate an additional collision_map, which is annotated with all special objects that can be interacted 
        # with by the player, but are not permanent (in constrast to the general room collision map)
        additional_object_collision_map: jArray = jnp.ones(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.int32)
        
        if RoomTags.CONVEYORBELTS in tags: 
            # global_conveyor_collision map only contains values (0, 1)
            conveyor_belt_tag: RoomTags.CONVEYORBELTS.value = SANTAH.extract_tag_from_rooms[RoomTags.CONVEYORBELTS](room_state)
            additional_object_collision_map = additional_object_collision_map + conveyor_belt_tag.global_conveyor_collision_map*ObservationCollisionMapAnnotationColors.conveyor.value
        if RoomTags.DOORS in tags:
            # global_collision_map has default value 0, and then 1+door_index
            door_tag: RoomTags.DOORS.value = SANTAH.extract_tag_from_rooms[RoomTags.DOORS](room_state)
            additional_object_collision_map = additional_object_collision_map + jnp.greater_equal(door_tag.global_collision_map, 1)*ObservationCollisionMapAnnotationColors.door.value
        if RoomTags.DROPOUTFLOORS in tags:
            # Dropout floor has value range [0, 1]
            d_o_tag: RoomTags.DROPOUTFLOORS.value = SANTAH.extract_tag_from_rooms[RoomTags.DROPOUTFLOORS](room_state)
            additional_object_collision_map = additional_object_collision_map + d_o_tag.dropout_floor_colision_map*ObservationCollisionMapAnnotationColors.dropout_floor.value
            has_dropout_floor: jArray = jnp.array([1], jnp.int32)
        else:
            has_dropout_floor: jArray = jnp.array([1], jnp.int32)
        if RoomTags.PIT in tags:
            has_pit: jArray = jnp.array([1], jnp.int32)
        else:
            has_pit: jArray = jnp.array([0], jnp.int32)
        if RoomTags.LADDERS in tags:
            # Default value is -1, then the ladder indexes
            ladder_tag: RoomTags.LADDERS.value = SANTAH.extract_tag_from_rooms[RoomTags.LADDERS](room_state)
            additional_object_collision_map = additional_object_collision_map + jnp.greater_equal(ladder_tag.ladder_tops + 1, 1)*ObservationCollisionMapAnnotationColors.ladder.value
        if RoomTags.LAZER_BARRIER in tags:
            # Default value is 0, 1 if theres a lazer barrier.
            lazer_tag: RoomTags.LAZER_BARRIER.value = SANTAH.extract_tag_from_rooms[RoomTags.LAZER_BARRIER](room_state)
            additional_object_collision_map = additional_object_collision_map + lazer_tag.global_barrier_map*ObservationCollisionMapAnnotationColors.lazer_barrier.value
        if RoomTags.ROPES in tags:
            # Again, default 0, then rope_id + 1
            rope_tag: RoomTags.ROPES.value = SANTAH.extract_tag_from_rooms[RoomTags.ROPES](room_state)
            additional_object_collision_map = additional_object_collision_map + rope_tag.rope_colision_map*ObservationCollisionMapAnnotationColors.rope.value
        if RoomTags.ITEMS in tags:
            item_tag: RoomTags.ITEMS.value = SANTAH.extract_tag_from_rooms[RoomTags.ITEMS](room_state)
            item_observation: ItemObservation = self.get_nearest_item_observation(montezuma_state=montezuma_state, 
                                                                                  item_tag=item_tag)
        else:
            item_observation = ItemObservation(
                    position=jnp.array([0, 0], jnp.int32), 
                    type=jnp.array(0, jnp.int32), 
                    collected=jnp.array(0, jnp.int32), 
                    dummy=jnp.array(1, jnp.int32)
            )
        if RoomTags.ENEMIES in tags:
            enemy_tag: RoomTags.ENEMIES.value = SANTAH.extract_tag_from_rooms[RoomTags.ENEMIES](room_state)
            enemy_observation: EnemyObservation = self.get_nearest_enemy_observation(montezuma_state=montezuma_state, 
                                                                                  enemy_tag=enemy_tag)
        else:
            enemy_observation: EnemyObservation = EnemyObservation(
                position=jnp.array([0, 0], jnp.int32), 
                type=jnp.array(0, jnp.int32), 
                alive=jnp.array(0, jnp.int32), 
                dummy=jnp.array(1, jnp.int32)
            )
            
        fully_augmented_collision_map: jArray = jnp.stack([initial_painted_canvas, additional_object_collision_map], axis=-1)
        full_observation: jArray = MontezumaObservation(
            player_position=montezuma_state.player_position, 
            annotated_collision_map=fully_augmented_collision_map, 
            nearest_enemy=enemy_observation, 
            nearest_item=item_observation,
            number_of_lives=montezuma_state.lifes[0], 
            current_items=montezuma_state.itembar_items, 
            is_jumping=jnp.astype(montezuma_state.is_jumping[0], jnp.int32), 
            is_falling=jnp.astype(montezuma_state.is_falling[0], jnp.int32), 
            is_on_ladder=montezuma_state.is_laddering[0], 
            is_on_rope=montezuma_state.is_on_rope[0], 
            has_pit=has_pit[0], 
            has_dropout_floors=has_dropout_floor[0]
        )
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.observation.value](montezuma_state, full_observation)
        return montezuma_state, room_state
    
    
    
    
    
    def handle_door_unlock(self, doors: RoomTags.DOORS.value, state: MontezumaState, index):
        #
        # Handles the unlocking of an individual door.
        #
        
        # Queue freeze
        state = self.queue_freeze(state, FreezeType.DOOR_UNLOCK)
        new_itembar = state.itembar_items.at[Item_Sprites.KEY.value].set(state.itembar_items[Item_Sprites.KEY.value]-1),
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.itembar_items.value](state, new_itembar[0])
        
        # Set the door to unlocked.
        des_door: Item = SANTAH.full_deserializations[Door](doors[0][index])
        des_door = SANTAH.attribute_setters[Door][DoorEnum.on_field.value](des_door,jnp.reshape(False, (1)).astype(jnp.int32))
        door = SANTAH.full_serialisations[Door](des_door)
        new_doors = doors.doors.at[index].set(door)
        doors = SANTAH.attribute_setters[MandatoryDoorFields][MandatoryDoorFieldsEnum.doors.value](doors, new_doors)
        return doors, state
    
    def handle_door_collision(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # Handle collision with all doors in the room
        #
        if RoomTags.DOORS in tags:
            doors: RoomTags.DOORS.value = SANTAH.extract_tag_from_rooms[RoomTags.DOORS](room_state)
            doors_collision_map = doors.global_collision_map
            # Check whether we are colliding with any door
            # the global_collision_map has a default value of 0, 
            # and door_index + 1 for each door collision zone.
            player_slice: jArray = jax.lax.dynamic_slice(doors_collision_map, start_indices=(montezuma_state.player_position[0],
                                                                                             montezuma_state.player_position[1]+room_state.vertical_offset[0]),
                                                         slice_sizes=(self.consts.PLAYER_WIDTH,self.consts.PLAYER_HEIGHT))
            # -1 because default value is 0
            collision_index = jnp.max(player_slice) -1
            specific_door_with_collision: Door = SANTAH.full_deserializations[Door](doors[0][collision_index])
            # collision_index + 1 == 0 iff the player is currently not in contact with any doors.
            doors, state = jax.lax.cond(jnp.logical_and(jnp.logical_and(collision_index+1,
                                                                       specific_door_with_collision.on_field[0]),
                                                       montezuma_state.itembar_items[Item_Sprites.KEY.value]),
                                       lambda x: self.handle_door_unlock(doors, montezuma_state,collision_index),
                                       lambda x: x,
                                       operand=(doors, montezuma_state))
            
            room = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.DOORS](room_state,doors)
            return state, room
        else:
            return montezuma_state, room_state
    
    def handle_lazer_barrier_collision(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # Handle collision with the lazer barriers in the room
        if RoomTags.LAZER_BARRIER in tags:
            lazer_barrier_attributes: RoomTags.LAZER_BARRIER.value = SANTAH.extract_tag_from_rooms[RoomTags.LAZER_BARRIER](room_state)
            barrier_collision_map: jArray = lazer_barrier_attributes.global_barrier_map
            
            barrier_info_glob: GlobalLazerBarrierInfo = SANTAH.full_deserializations[GlobalLazerBarrierInfo](lazer_barrier_attributes.global_barrier_info[0, ...])
            
            # Determine wether lazer barriers are currently active using the cycle index.
            active_start: jnp.ndarray = barrier_info_glob.cycle_offset
            active_stop: jnp.ndarray = barrier_info_glob.cycle_offset + barrier_info_glob.cycle_active_frames
            is_barrier_active: jnp.ndarray = jnp.logical_and(jnp.greater_equal(barrier_info_glob.cycle_index, active_start), 
                                                      jnp.less(barrier_info_glob.cycle_index, active_stop))
            
            # Multiple barrier collision map with is_barrier_active so no collision is detected 
            # if barriers are currently inactive.
            barrier_collision_map = jnp.astype(jnp.multiply(barrier_collision_map, is_barrier_active), jnp.int32)
            
            # Check whether the player is currently colliding with an active barrier.
            player_slice: jArray = jax.lax.dynamic_slice(barrier_collision_map, start_indices=(montezuma_state.player_position[0], 
                     montezuma_state.player_position[1] + room_state.vertical_offset[0]), slice_sizes=(self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT))
            is_dying_from_barrier: jArray = jnp.greater_equal(jnp.sum(player_slice, keepdims=False), 1)
            is_dying_from_barrier = jnp.astype(is_dying_from_barrier, jnp.int32)
            is_dying_from_barrier = jnp.reshape(is_dying_from_barrier, (1))
            is_dying = jax.lax.max(is_dying_from_barrier, montezuma_state.is_dying)
            # Queue a freeze if the player is dying from a lazer barrier:
            lazer_freeze = partial(self.queue_freeze, freeze_type=FreezeType.LAZER_BARRIER_DEATH)
            do_lazer_freeze: jArray = jnp.logical_and(is_dying_from_barrier, jnp.logical_not(montezuma_state.is_dying))
            montezuma_state = jax.lax.cond(do_lazer_freeze[0], lazer_freeze, lambda x : x, montezuma_state)
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_dying.value](montezuma_state, is_dying)
            
            return montezuma_state, room_state
        else:
            return montezuma_state, room_state
         
    def _handle_sarlacc_pit_collision(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        if RoomTags.PIT in tags:
            # Sarlacc pit collision is trigger if the players feet go below the barrier starting position.
            pit_attributes: RoomTags.PIT.value = SANTAH.extract_tag_from_rooms[RoomTags.PIT](room_state)
            state = montezuma_state
            is_dying: jArray = jnp.greater(state.player_position[1]+self.consts.PLAYER_HEIGHT,pit_attributes.starting_pos_y[0])
            do_sarlacc_freeze: jArray = jnp.logical_and(is_dying, jnp.logical_not(montezuma_state.is_dying))
            # Queue a game freeze if the player is killed by the pit.
            state = jax.lax.cond(do_sarlacc_freeze[0], partial(self.queue_freeze, freeze_type=FreezeType.SARLACC_PIT_DEATH), lambda x : x, state)

            is_dying = jnp.astype(is_dying, jnp.int32)
            is_dying = jnp.reshape(is_dying, (1))
            is_dying = jax.lax.max(is_dying, montezuma_state.is_dying)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_dying.value](state, is_dying)
            return state, room_state
        else:
            return montezuma_state, room_state
    
    @partial(jax.jit, static_argnames=["self"])      
    def render(self, state: MontezumaState) -> jnp.ndarray:
        return self.renderer.render(state)
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)
    @partial(jax.jit, static_argnums=(0))   
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[MontezumaObservation, MontezumaState]:
        """
        Resets the game state to the initial state.
        """
        init_room: Room = self.PROTO_ROOM_LOADER(jnp.array([0]), self.initial_persistence_state)
        game_state = MontezumaState(room_state=init_room, 
                               persistence_state=self.initial_persistence_state, 
                               player_position=jnp.array([self.consts.INITIAL_PLAYER_X, self.consts.INITIAL_PLAYER_Y], jnp.int32), 
                               player_collision_map=self.player_col_map, 
                               player_velocity=jnp.array([self.consts.INITIAL_PLAYER_VELOCITY], jnp.uint8), 
                               horizontal_direction=jnp.array([Horizontal_Direction.NO_DIR.value], jnp.uint8),
                               vertical_directional_input= jnp.array([Horizontal_Direction.NO_DIR.value], jnp.int32),
                               last_horizontal_direction=jnp.array([Horizontal_Direction.NO_DIR.value], jnp.uint8),
                               last_horizontal_orientation=jnp.array([Horizontal_Direction.LEFT.value], jnp.uint8),
                               current_directional_input=jnp.array([MovementDirection.NO_DIR.value], jnp.int32),
                               horizontal_falling_velocitiy=jnp.array([0], jnp.int16),
                               is_standing=jnp.array([True], jnp.bool),
                               is_jumping=jnp.array([False], jnp.bool),  
                               is_falling=jnp.array([False], jnp.bool), 
                               jump_counter=jnp.array([0], jnp.uint16), 
                               augmented_collision_map = jnp.ones(shape=(self.consts.WIDTH, self.consts.HEIGHT), dtype=jnp.uint8), 
                               canvas = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT, 4), dtype=jnp.uint8), 
                               jump_input=jnp.array([JUMP_INPUT.NO.value], jnp.uint8),
                               frame_count=jnp.array([0], jnp.int32),
                               is_dying = jnp.array([0], jnp.int32),
                               last_entrance_direction = jnp.array([RoomConnectionDirections.UP.value], jnp.int32), 
                               is_climbing=jnp.array([0], jnp.int32),
                               is_laddering=jnp.array([0], jnp.int32),
                               score=jnp.array([0], jnp.int32),
                               lifes=jnp.array([5],jnp.int32),
                               itembar_items=jnp.zeros(shape=(5),dtype=jnp.int32), 
                               is_on_rope=jnp.array([0],jnp.int32), 
                               last_key_press=jnp.array([-200], jnp.int32), # Doesnt really matter
                               is_key_hold=jnp.array([False], jnp.bool), 
                               force_jump=jnp.array([False], jnp.bool), 
                               disable_directional_input=jnp.array([0], jnp.int32), 
                               queue_enable_directional_input=jnp.array([0], jnp.int32), 
                               get_on_ladder_bottom=jnp.array([0], jnp.int32), 
                               get_on_ladder_top = jnp.array([0], jnp.int32),
                               darkness= jnp.array([False], jnp.bool),
                               hammer_time=jnp.array([0], jnp.int32),
                               room_enter_frame= jnp.array([0], jnp.int32),
                               first_item_pickup_frame= jnp.array([0], jnp.int32),
                               rng_key= key,
                               queue_freeze=jnp.array([0], jnp.int32), 
                               frozen=jnp.array([0], jnp.int32), 
                               freeze_type=jnp.array([0], jnp.int32), 
                               freeze_remaining=jnp.array([0], jnp.int32), 
                               frozen_state=None, 
                               current_velocity=jnp.array([0, 0], jnp.int32), 
                               last_velocity=jnp.array([0, 0], jnp.int32), 
                               velocity_held=jnp.array([0], jnp.int32), 
                               rope_climb_frame=jnp.array([0], jnp.int32), 
                               ladder_climb_frame=jnp.array([0], jnp.int32), 
                               walk_frame=jnp.array([0], jnp.int32), 
                               horizontal_direction_held=jnp.array([0], jnp.int32),
                               fall_position=jnp.array([900, 900], jnp.int32),
                               render_offset_for_fall_dmg=jnp.array([0], jnp.int32), 
                               top_seeking_on_entrance=jnp.array([0], jnp.int32), 
                               bottom_seeking_at_entrance=jnp.array([0], jnp.int32), 
                               is_jump_hold=jnp.array([0], jnp.int32), 
                               reset_room_state_on_room_change=jnp.array([0], jnp.int32), 
                               observation=None
                               )
        
        # Set a dummy frozen state. 
        # The frozen state is necessary because during a freeze, the state of the game before the step in which the freeze occured is rendered, 
        # but the game is resumed from the state after.
        game_state = self.WRAPPED_HANDLE_COMPUTE_OBSERVATION(game_state)
        dummy_frozen_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frozen_state.value](game_state, None)
        game_state: MontezumaState = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frozen_state.value](game_state, dummy_frozen_state)
        
        return game_state.observation, game_state
    
    
    def _may_move_horizontally(self, state: MontezumaState):
        """Checks whether the player is currently allowed to move horizontally, or whether this is forbidden 
        """
        movement_gates: jArray = jnp.array([state.is_on_rope[0], state.is_laddering[0]], dtype=jnp.int32)
        may_move = 1 - jnp.max(movement_gates, keepdims=True)
        return may_move

    def _player_horizontal_movement(self, state: MontezumaState):
        """Handles horizontal movement for the player. Sets new horizontal position based on the
           current movement direction of the player.

        Args:
            state (MontezumaState): Full game state
            
        Returns:
            MontezumaState: New GameState with updated horizontal position.
        """
        # No new movement inputs are accepted while the player is falling
        new_pos: jnp.ndarray =jax.lax.cond(jnp.logical_not(state.is_falling[0]), 
                                           lambda _: jax.lax.switch(index=state.last_horizontal_direction[0],
                                                            branches=[
                                                               lambda x: x.at[0].set(x[0] - state.player_velocity[0]),
                                                               lambda x: x,
                                                               lambda x: x.at[0].set(x[0] + state.player_velocity[0])
                                                            ], 
                       operand=state.player_position),
                                           # If the player is falling, the last horizontal movement direction before the fall is used.
                                           lambda _: state.player_position.at[0].set(state.player_position[0] + ((state.horizontal_falling_velocitiy[0]*(state.last_horizontal_orientation[0].astype(jnp.int16)-1)))*jnp.mod(state.frame_count[0],2)),
                                           operand=None)
        new_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, new_pos)
        
        return new_state
 
    def ray_cast_downwards(self, player_position: jnp.ndarray, level_collision_map: jnp.ndarray, vert_offset: jnp.ndarray) -> int:
        """Raycasts downwards and returns the amount of free space below the player

        Args:
            player_position (jnp.ndarray): Current position of the player as [X, Y]
            level_collision_map (jnp.ndarray): Padded level collision map: Level collision map is 
                padded to have dimensions (WIDTH; HEIGHT)
            vert_offset (jnp.ndarray): Amount of padding applied to the top

        Returns:
            int: Amount of free space below the player.
        """
        middle_offset: int = self.consts.PLAYER_WIDTH//2 + player_position[0] # X position
        height_offset: int = self.consts.PLAYER_HEIGHT + player_position[1] + vert_offset[0] # Y offset
        h_1 = jnp.array([height_offset-1], jnp.uint16)
        ray_slice: jnp.ndarray = jax.lax.dynamic_slice(operand=level_collision_map, 
                                                       start_indices=jnp.array([middle_offset, 0]), 
                                                       slice_sizes=(1, self.consts.HEIGHT))
        ray_slice = ray_slice.astype(jnp.uint16)
        indices = jnp.arange(start=0, stop=self.consts.HEIGHT, dtype=jnp.uint16)
        indices = jax.lax.max(indices, h_1)
        # Height offset represents first index which should be looked at for raycasting
        # Now all indices below height_offset are set to height_offser - 1
        indices = indices - (height_offset - 1)
        indices = indices*2 # Just so integer unterlauf doesn't break things
        indices = indices.astype(jnp.uint16)
        # Now all indices we don't care about are equal to zero
        ray_slice = jax.lax.mul(indices, ray_slice[0, :])
        # We multiple that onto the ray slice to only look at indices below our desired hight offset
        ray_slice = jnp.astype(ray_slice, jnp.uint16)

        ray_slice = ray_slice - 1 # cause integer unterlauf so that 0 (no wall now have max_value)
        min_dex = jnp.argmin(ray_slice, axis=0, keepdims=False) # Find minimum distance to platform below the player
            # All void fields are set to maxvall, as is everything below the height offset
        return jnp.reshape(min_dex - height_offset, (1, ))
        
        
    def handle_user_input(self, state: MontezumaState, action: chex.Array) -> MontezumaState:
        """Handles user input. Sets the current_input and the horizontal_direction fields. 
           If the player is either currently falling or is currently jumping, no input actions will be 
           processed. 
           The jumping attribute is not set here, even though it directly corresponds to an input action. 
           The jumping attribute needs to be set in the jump function

        """
        ack = state.last_key_press[0]
        action = jax.lax.cond(state.disable_directional_input[0], lambda x, y: x, lambda x, y: y, ack, action)
        # Detect whether this is a fresh input
        # and whether the last input was a jump input. This is used to prevent bunny hopping.
        is_hold: jArray = jnp.equal(state.last_key_press, action)
        is_jump_hold: jArray = jnp.logical_and(JAXATARI_GET_JUMP_ACTION(state.last_key_press[0]), JAXATARI_GET_JUMP_ACTION(action))
        is_jump_hold = jnp.astype(is_jump_hold, jnp.int32)
        is_jump_hold = jnp.reshape(is_jump_hold, (1))
        movement_dir: jax.Array = JAXATARI_ACTION_TO_MOVEMENT_DIRECTION(action)
        horizontal_direction: jax.Array = get_horizontal_movement_direction(movement_dir)
        
        # Only allow a change in input if the player is not juming or falling.
        jump_or_fall = jnp.logical_or(state.is_jumping, state.is_falling)
        new_dir_input = jump_or_fall*jnp.array([MovementDirection.NO_DIR.value]) + (1 - jump_or_fall)*movement_dir
        
        new_horizontal_direction = jump_or_fall*state.horizontal_direction + (1 - jump_or_fall)*horizontal_direction
        
        
        horizontal_orientation = jax.lax.cond(state.horizontal_direction[0] == Horizontal_Direction.NO_DIR.value,
                                              lambda _: state.last_horizontal_orientation,
                                              lambda _: state.horizontal_direction,
                                              operand=None)
        
        current_vertical_directional_input: jArray = get_vertical_movement_direction(movement_dir)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.vertical_directional_input.value](state, current_vertical_directional_input)
        
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.last_horizontal_orientation.value](state, horizontal_orientation)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.current_directional_input.value](state, new_dir_input)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.last_horizontal_direction.value](state, state.horizontal_direction)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.horizontal_direction.value](state, new_horizontal_direction)
        is_jump_action = JAXATARI_GET_JUMP_ACTION(action)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.jump_input.value](state, is_jump_action)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_jump_hold.value](state, is_jump_hold)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.last_key_press.value](state, jnp.reshape(action, (1)))
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_key_hold.value](state, is_hold)
        return state
    

    
    
    
    def find_nearest_free_position_2D_conv(self, state: MontezumaState, collision_player_hitmap: jArray):
        # First flip the collision hitmap along both axis, 
        # because we use convolution, which also flips the filter per definition.
        
        #WARNING: The flip along axis 1 is 100% necessary, the flip along axis 0 might not be.
        # if this is the case, you might get weird results for asymetric colision maps.
        
        # Check if the player is actually currently colliding with the level collision map. 
        # If not, no need to find a free position
        augmented_collision_map: jArray = state.augmented_collision_map
        collision_frame: jnp.ndarray = jax.lax.dynamic_slice(augmented_collision_map, 
                              start_indices=(state.player_position[0], state.player_position[1]+ state.room_state.vertical_offset[0]), 
                              slice_sizes=(self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT))
        collision_frame = jnp.multiply(collision_frame, collision_player_hitmap)
        has_collision = jnp.max(collision_frame, keepdims=False)
        
        
        
        collision_player_hitmap = jnp.flip(collision_player_hitmap, axis=0)
        collision_player_hitmap = jnp.flip(collision_player_hitmap, axis=1)
        
        
        
        # Convolve level collision map with player collision map 
        # to find fields that are occupiable by the player
        occupiable_fields: jnp.ndarray = jax.scipy.signal.convolve2d(augmented_collision_map, collision_player_hitmap, 
                                                                     mode='valid')
        occupiable_fields = jnp.astype(jnp.equal(occupiable_fields, 0), jnp.uint16)
        
        
        base_field = jnp.ones_like(occupiable_fields)
        x_range = jnp.arange(start=1, stop=self.consts.WIDTH + 2 - self.consts.PLAYER_WIDTH)
        x_range = jnp.reshape(x_range, (self.consts.WIDTH + 1 - self.consts.PLAYER_WIDTH, 1))
        
        
        # Make a mask consisting of the X-coordinates, multiply with the occupiable fields
        x_mask = jnp.multiply(occupiable_fields, jnp.multiply(base_field, x_range))
        
        x_mask = jnp.astype(x_mask, jnp.uint16)
        
        # Cause integer underflow: 
        # Now all occupiable fields have their X-coordinate as value, and all in-occupiable fields 
        # have the int32 max value.
        x_mask = x_mask - 1
        
        x_mask = jnp.astype(x_mask, jnp.int32)
        
        
        
        # Repeat the same process for the Y coordinate
        y_range = jnp.arange(start=1, stop=self.consts.HEIGHT + 2 - self.consts.PLAYER_HEIGHT)
        y_range = jnp.reshape(y_range, (1, self.consts.HEIGHT + 1 - self.consts.PLAYER_HEIGHT))
        y_mask = jnp.multiply(occupiable_fields, jnp.multiply(base_field, y_range))
        y_mask = jnp.astype(y_mask, jnp.uint16)
        y_mask = y_mask - 1
        y_mask = jnp.astype(y_mask, jnp.int32)
        # Now the y_mask contains a very large value in all unocupiable positions
        # And in all occupiable position, the y_coord + VERTICAL_OFFSET!!!!!
        
        # Now start to compute the field with the minimum distance to the player:
        x_dist = jnp.absolute(x_mask - state.player_position[0])
        
        y_dist = jnp.absolute(y_mask - (state.player_position[1] + state.room_state.vertical_offset[0]))
        total_dist = x_dist + y_dist
        
        # ravel the array so that we can use argmin to determine the nearest free position
        total_dist = jnp.ravel(total_dist, order="C")
        ## "C" means the thing gets unfolded column wise, so columns
        ## are listed one after the other
        best_position_index = jnp.argmin(total_dist, keepdims=False)
        # Recompute X and Y indices from the singleton list index.
        y_index = jnp.remainder(best_position_index, (self.consts.HEIGHT + 1 - self.consts.PLAYER_HEIGHT))
        x_index = jnp.floor_divide(best_position_index, (self.consts.HEIGHT + 1 - self.consts.PLAYER_HEIGHT))
        new_pos = jnp.array([x_index, y_index - state.room_state.vertical_offset[0]], jnp.int32)
        new_pos = has_collision*new_pos + (1 - has_collision)*state.player_position
        new_pos = jnp.astype(new_pos, jnp.int32)
        
        return new_pos
    
    
    def find_nearest_non_default_value_position(self, state: MontezumaState, full_size_querry_map: jArray, default_value: jArray) -> Tuple[jArray, jArray]:
        """A more general utility function. Finds the position on the full_size_querry_map that is not occupied by the default 
            value nearest to the player. Among other places, this function is used while crossing rooms when you are on a ladder.

        Args:
            state (MontezumaState): The Montezuma State
            full_size_querry_map (jArray): Map which is used to find the nearest non-default value. 
                Needs to be of full playfield size, i.e. (self.consts.WIDTH, self.consts.HEIGHT), 
                has to have a datatype of jnp.int32
            default_value (jArray): Default value, all positions with default value are ignored.
            

        Returns:
            Tuple[jArray, jArray]: A tuple consisting of:
                - the nearest non default value found (singleton, int32), this is equivalent to the default value if there 
                    are no non-default value fields
                - the position of the nearest non default value. If no non-default value was found, this return value is to be considered
                    undefined and no guarantees for behavior in this edge case are given.
        """
        default_value = jnp.reshape(default_value, (1))
        non_default_fields = jnp.astype(jnp.not_equal(full_size_querry_map, default_value), jnp.uint16)
        
        
        base_field = jnp.ones_like(non_default_fields)
        x_range = jnp.arange(start=1, stop=self.consts.WIDTH + 1)
        x_range = jnp.reshape(x_range, (self.consts.WIDTH, 1))
        
        
        
        x_mask = jnp.multiply(non_default_fields, jnp.multiply(base_field, x_range))
        
        x_mask = jnp.astype(x_mask, jnp.uint16)
        
        x_mask = x_mask - 1
        
        x_mask = jnp.astype(x_mask, jnp.int32)
        
        
        # Now x_mask is a 2D mask: Occupiable pixels are occupied by their X value, 
        # and un-occupiable cells are 0
        
        y_range = jnp.arange(start=1, stop=self.consts.HEIGHT + 1)
        y_range = jnp.reshape(y_range, (1, self.consts.HEIGHT))
        y_mask = jnp.multiply(non_default_fields, jnp.multiply(base_field, y_range))
        y_mask = jnp.astype(y_mask, jnp.uint16)
        y_mask = y_mask - 1
        y_mask = jnp.astype(y_mask, jnp.int32)
        # Now the y_mask contains a very large value in all unocupiable positions
        # And in all occupiable position, the y_coord + VERTICAL_OFFSET!!!!!
        
        # Now start to compute the field with the minimum distance to the player:
        x_dist = jnp.absolute(x_mask - state.player_position[0])
        
        y_dist = jnp.absolute(y_mask - (state.player_position[1] + state.room_state.vertical_offset[0]))
        total_dist = x_dist + y_dist
        
        total_dist = jnp.ravel(total_dist, order="C")
        ## "C" means the thing gets unfolded column wise, so columns
        ## are listed one after the other
        best_position_index = jnp.argmin(total_dist, keepdims=False)
        y_index = jnp.remainder(best_position_index, (self.consts.HEIGHT))
        x_index = jnp.floor_divide(best_position_index, (self.consts.HEIGHT))
        new_pos = jnp.array([x_index, y_index - state.room_state.vertical_offset[0]], jnp.int32)
        new_pos = jnp.astype(new_pos, jnp.int32)
        non_default_value = jax.lax.dynamic_slice(operand=full_size_querry_map, 
                                                  start_indices=jnp.array([x_index, y_index], jnp.int32), 
                                                  slice_sizes=(1, 1))
        non_default_value = jnp.reshape(non_default_value, (1))
        
        return non_default_value, new_pos
    
    
    def _may_collide_with_environment(self, state: MontezumaState, new_player_position: jArray, old_player_position: jArray):
        """Checks if the player may collide with the environment
           If the player is climbing (rope, ladder), collision checks are not perforemd.
           If the player is jumping straight upwards, collision checks are also not performed. This needs to be done 
           so that the player can jump through small platofrms. As the absolute velocity is used, this prevents the player from clipping into walls.
        """
        diff: jArray = new_player_position - old_player_position
        vertical_jump: jArray = jnp.logical_and(jnp.equal(diff[0], 0), jnp.less(diff[1], 0))
        vertical_jump = jnp.logical_and(vertical_jump, 1 - state.is_on_rope[0])
        inhibitors: jArray = jnp.array([state.is_falling[0], state.is_laddering[0], vertical_jump], dtype=jnp.int32)
        may_climb = 1 - jnp.max(inhibitors, keepdims=True)
        return may_climb
    
    def push_player_back_onto_playfield(self, state: MontezumaState) -> MontezumaState:
        """Pushes the player back into the playing field.
        """
        player_pos: jax.Array = jax.lax.max(state.player_position, jnp.array([0, 0]))
        player_pos = jax.lax.min(player_pos, jnp.array([self.consts.WIDTH - self.consts.PLAYER_WIDTH, state.room_state.height[0] - self.consts.PLAYER_HEIGHT]))
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, player_pos)
        return state        
    
    def set_horizontal_fall_vel(self, state: MontezumaState, last_is_standing: jArray, current_is_standing: jArray):
        # Determines the appropriate falling velocity for the player.
        standing_difference: jArray = jnp.astype(current_is_standing,jnp.int8) - jnp.astype(last_is_standing,jnp.int8)
        standing_difference = standing_difference + 1
        
        # 0 if went from standing to not standing
        # 1 if nothing changed
        # 2 if went from not standing to standing
        horizontal_fall_velocity = jax.lax.switch(standing_difference[0], branches=[
            lambda _: jax.lax.cond(state.is_jumping[0],
                                   lambda _: jnp.array([1], jnp.int16),
                                   lambda _: jnp.array([0], jnp.int16),
                                   operand=None),
            lambda _: state.horizontal_falling_velocitiy,
            lambda _: jnp.array([0], jnp.int16)
        ],operand=None) 
        
        return horizontal_fall_velocity
    
    
    def _actually_get_on_the_rope(self, state: MontezumaState, 
                                  rope_tag: RoomTags.ROPES.value, 
                                  rope_id: jArray, 
                                  from_the_top: bool = False):
        """Code that handles getting on the rope.

        Args:
            state (MontezumaState): Current gamestate
            rope_tag (RoomTags.ROPES.value): All rope specific information
            rope_id (jArray): The integer ID of the rope we are actually supposed to get on.
            from_the_top (bool, optional): Whether we are supposed to get on from the top. 
                If set to false, the getting-on mechanism of jumping on from the side is used. Defaults to False.

        Returns:
            _type_: _description_
        """
        rope_id = jnp.reshape(rope_id, (1))
        rope_id = jnp.astype(rope_id, jnp.int32)
        
        #Retreive the rope we are currently getting on.
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.queue_enable_directional_input.value](state, jnp.array([1], jnp.int32))
        correct_room_row: jArray = jax.lax.dynamic_index_in_dim(rope_tag.ropes, index=rope_id[0], axis=0)
        correct_rope: Rope = SANTAH.full_deserializations[Rope](correct_room_row[0])
        
        # Updates the state to indicate that we are now on a rope.
        is_on_rope = jnp.array([1], dtype=jnp.int32)
        is_climbing = jnp.array([1], dtype=jnp.int32)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_on_rope.value](state, is_on_rope)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_climbing.value](state, is_climbing)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_falling.value](state, jnp.array([False], jnp.bool))
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_jumping.value](state, jnp.array([False], jnp.bool))
        # Update the Rope Tag with the index of the correct rope we are now on.
        rope_tag = SANTAH.attribute_setters[RoomTags.ROPES.value][RoomTagsNames.ROPES.value.rope_index.value](rope_tag, rope_id)
        rope_tag = SANTAH.attribute_setters[RoomTags.ROPES.value][RoomTagsNames.ROPES.value.last_hanged_on_rope.value](rope_tag, rope_id)
        # Teleports the player onto the rope
        # X position is always set according to the rope's X-position
        new_player_x: jnp.ndarray = self.consts.ROPE_HORIZONTAL_OFFSET + correct_rope.x_pos
        if from_the_top:
            # If we enter from the top, teleport the player a set distance below the rope top
            y_position: jnp.ndarray = correct_rope.top[0] + self.consts.ROPE_TOP_ENTRY_SNAP_ON_OFFSET
        else:
            # If we enter from the side, clip the Y position to the range of Y-positions for the rope that are NOT
            # Within the entrance & exit zones
            # Otherwise getting on the rope from below is impossible.
            highest_legal_y_position: jArray = correct_rope.top + self.consts.ROPE_TOP_EXIT_DISTANCE + 1
            lowest_legal_y_position: jArray = correct_rope.bottom - self.consts.ROPE_FALL_VERTICAL_OFFSET - 1
            
            y_position: jnp.ndarray = state.player_position[1]
            y_position = jax.lax.max(y_position, highest_legal_y_position)
            y_position = jax.lax.min(y_position, lowest_legal_y_position)
            y_position = y_position[0]
        new_player_position = state.player_position.at[0].set(new_player_x[0])
        new_player_position = new_player_position.at[1].set(y_position)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, new_player_position)
        # Also terminate any currently ongoing jumps
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_jumping.value](state, jnp.array([False], jnp.bool))
        
        return (state, rope_tag)
    
    
    def _handle_start_climbing_ropes(self, state: MontezumaState, rope_tag: RoomTags.ROPES.value) -> MontezumaState:
        #
        # Handle jumping onto ropes from the side & bottom.
        #
        
        # Check whether the player character is currently overlapping with a rope
        rope_collision_map: jArray =  rope_tag.rope_colision_map
        x_pos: jArray = state.player_position[0] + self.consts.ROPE_COLLISION_X_OFFSET
        collision_area: jArray = jax.lax.dynamic_slice(operand=rope_collision_map, 
                        start_indices=(x_pos, state.player_position[1] + state.room_state.vertical_offset[0]), 
                        slice_sizes=(self.consts.ROPE_COLLISION_WIDTH, self.consts.PLAYER_HEIGHT))
        
        # Retreive the index of the rope we are overlapping with.
        _max_overlapping_rope = jnp.max(collision_area, keepdims=False)
        _max_overlapping_rope = jnp.reshape(_max_overlapping_rope, (1))
        
        # If _max_overlapping_rope == 0: We are not currently overlapping any ropes.
        is_getting_on_rope = 1 - jnp.equal(_max_overlapping_rope, 0)
        
        # We can't get onto a rope if we are already climbing
        is_getting_on_rope = jnp.multiply(is_getting_on_rope, (1 - state.is_climbing))
        # We can't regrab the same rope twice in a row.
        # last_hanged_on_rope gets reset if we stand on the floor.
        is_regrabbing_the_rope = jnp.equal(rope_tag.last_hanged_on_rope, _max_overlapping_rope - 1)
        is_getting_on_rope = jnp.multiply(is_getting_on_rope, 1 - is_regrabbing_the_rope)
        state, rope_tag = jax.lax.cond(is_getting_on_rope[0], self._actually_get_on_the_rope, lambda x, y, z : (x, y), state, rope_tag, _max_overlapping_rope - 1)
        return (state, rope_tag)
    
    
    def _handle_climb_down_onto_rope(self, state: MontezumaState, rope_tag: RoomTags.ROPES.value) -> MontezumaState:
        # Handle getting onto a rope from the top.
        #
        #
        
        # First check whether the player is currently pressing a "down input"
        down_input: jArray = jnp.equal(state.current_directional_input[0], MovementDirection.DOWN.value)
        # Check if the player is currently above a rope, if so he can downclimb
        _downclimbable_rope_tops: jArray = jax.lax.dynamic_slice(rope_tag.room_rope_top_pixels, 
                                start_indices=(state.player_position[0], state.player_position[1] + self.consts.PLAYER_HEIGHT + state.room_state.vertical_offset[0]), 
                                slice_sizes=(self.consts.PLAYER_WIDTH, self.consts.ROPE_TOP_ENTRANCE_VERTICAL_SNAP_DISTANCE))
        max_val = jnp.max(_downclimbable_rope_tops, keepdims=False)
        
        # Now determine if we ACTUALLY need to enter a rope from the top: 
        # We need to be:
        #       1) Pressing down
        #       2) Be sufficiently close above a rope
        #       3) Not be on a rope already
        #       4) the rope we want to get on needs to actually be climbable
        _tmp_rope_ind: jArray = jnp.maximum(max_val, 0)
        correct_rope: jArray = jax.lax.dynamic_index_in_dim(rope_tag.ropes, index=_tmp_rope_ind, axis=0, keepdims=False)
        tmp_rope: Rope = SANTAH.full_deserializations[Rope](correct_rope)
        get_on_rope: jArray = jnp.logical_and(jnp.not_equal(max_val, -1), jnp.logical_and(down_input, 
                                        jnp.logical_and(jnp.logical_not(state.is_climbing[0]), jnp.logical_and(tmp_rope.is_climbable[0], tmp_rope.accessible_from_top[0]))))
        
        new_state, new_rope_tag = jax.lax.cond(get_on_rope, partial(self._actually_get_on_the_rope, from_the_top=True), lambda x, y, z: (x, y),  state, rope_tag, max_val)
        return (new_state, new_rope_tag)
    
    @partial(jax.jit, static_argnames=["self", "at_bottom"])
    def _actually_get_on_the_ladder(self, montezuma_state: MontezumaState, ladder_tag: RoomTags.LADDERS.value, ladder_index: jArray, room_state: Room, at_bottom: bool = True):
        """Actually gets on the ladder. This function is used for both getting on the ladder at the top, and getting on the ladder at the bottom.

        Args:
            montezuma_state (MontezumaState): Current Game State
            ladder_tag (RoomTags.LADDERS.value): Current ladder tag
            ladder_index (jArray): Index of the ladder we are supposed to get on. 
                It is assumed, that this is always a valid ladder index.
            at_bottom (bool, optional): Whether we are supposed to enter the ladder at the bottom. If false, enter ladder 
                from the top. Defaults to True.
        """
        # Set all the climbing related attributes
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_climbing.value](montezuma_state, jnp.array([1], jnp.int32))
        # this is probably not necessary, but just for safety...
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_on_rope.value](montezuma_state, jnp.array([0], jnp.int32))
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_laddering.value](montezuma_state, jnp.array([1], jnp.int32))

        # retreive the ladder we want to get on.
        correct_ladder_arr: jArray = jax.lax.dynamic_index_in_dim(ladder_tag.ladders, index=ladder_index, 
                                                                  axis=0, keepdims=False)
        current_ladder: Ladder = SANTAH.full_deserializations[Ladder](correct_ladder_arr)
        
        # Now set all the necessary attributes in the ladder tag & state
        ladder_tag = SANTAH.attribute_setters[RoomTags.LADDERS.value][RoomTagsNames.LADDERS.value.ladder_index.value](ladder_tag, jnp.reshape(ladder_index, (1)))      
        # Here we also bake in the assumption, that ladder tops always connect to ladder bottoms
        # This limits your flexibility in designing non-euclidean level geometry, 
        # might be worth changing in the future
        # These fields are used if we transition rooms while on a ladder.
        # if we enter a room with "get_on_ladder_bottom/top" set to true, 
        # the game tries to teleport you into the nearest ladder-top/bottom-zone.
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_bottom.value](montezuma_state, current_ladder.rope_seeking_at_top)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_top.value](montezuma_state, current_ladder.rope_seeking_at_bottom)
        # Now set the correct player position:
        # Try to place player horizontally in the middle of the ladder
        # Determine how much wider the ladder is than the player character to place the player in the middle of the ladder.
        wiggle_room_x: jArray = jax.lax.max(current_ladder.right_lower_x - (current_ladder.left_upper_x + self.consts.PLAYER_WIDTH), jnp.array([0], jnp.int32))
        x_offset = jnp.floor_divide(wiggle_room_x, 2)
        correct_x_pos = current_ladder.left_upper_x + x_offset
        
        # Now decide on the correct y-position
        if at_bottom:
            ladder_y_pos: jArray = current_ladder.right_lower_y
            correct_y_pos: jArray = jax.lax.min(ladder_y_pos - (self.consts.LADDER_ENTRANCE_TELEPORT_DISTANCE + self.consts.PLAYER_HEIGHT),room_state.height - self.consts.PLAYER_HEIGHT - self.consts.LADDER_ROOM_ENTRANCE_GRACE_PIXELS)
        else:
            ladder_y_pos: jArray = current_ladder.left_upper_y
            correct_y_pos: jArray = jax.lax.max(ladder_y_pos + self.consts.LADDER_ENTRANCE_TELEPORT_DISTANCE - self.consts.PLAYER_HEIGHT, jnp.array([0], jnp.int32) + self.consts.LADDER_ROOM_ENTRANCE_GRACE_PIXELS)
        
        correct_player_position: jArray = jnp.array([correct_x_pos[0], correct_y_pos[0]], jnp.int32)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](montezuma_state, correct_player_position)
        return (montezuma_state, ladder_tag)
    
    
    
    def _handle_teleport_onto_ladder(self, montezuma_state: MontezumaState, ladder_tag: RoomTags.LADDERS.value, room_state: Room) -> Tuple[MontezumaState, RoomTags.LADDERS.value]:
        # This function is called if we attempt to transition rooms while on a ladder. 
        # In this case, we attempt to teleport the player onto the nearest ladder in the room.
        #
        
        
        # Choices are:
        # - 0: do not teleport onto ANY ladder
        # - 1: teleport to the nearest bottom of a ladder
        # - 2: teleport to the nearest top of a ladder
        tmp = jnp.array([0, montezuma_state.get_on_ladder_bottom[0]*1, montezuma_state.get_on_ladder_top[0]*2], jnp.int32)
        which_ladder_to_get_on: jArray = jnp.max(tmp, keepdims=True)
        
        # Functions to find the nearest ladder top/bottom interface zones.
        #
        
        def _find_nearest_top(state: MontezumaState, ladder_tag: RoomTags.LADDERS.value):
            ladder_index: jArray = None # might be -1 iff no ladder is found
            ladder_index, _ = self.find_nearest_non_default_value_position(state=state, 
                                full_size_querry_map=ladder_tag.ladder_tops, 
                                default_value=jnp.array([-1], jnp.int32) 
                                )
            return ladder_index
        
        def _find_nearest_bottom(state: MontezumaState, ladder_tag: RoomTags.LADDERS.value):
            ladder_index: jArray = None # might be -1 iff no ladder is found
            ladder_index, _ = self.find_nearest_non_default_value_position(state=state, 
                                full_size_querry_map=ladder_tag.ladder_bottoms, 
                                default_value=jnp.array([-1], jnp.int32) 
                                )
            return ladder_index
        # find the nearest ladder index:    
        ladder_index: jArray = jax.lax.switch(which_ladder_to_get_on[0], 
                       [lambda x, y : jnp.array([-1], jnp.int32), _find_nearest_bottom, _find_nearest_top], montezuma_state, ladder_tag)
        #Check if we have found an appropriate ladder
        has_found_ladder = jnp.not_equal(ladder_index, -1)
        
        # Now turn off the flags, so we don't repeatedly teleport onto the ladders
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_bottom.value](montezuma_state, jnp.array([0], jnp.int32))
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_top.value](montezuma_state, jnp.array([0], jnp.int32))
        
        
        # We have found the correct ladder to get on
        # This part handles actually teleporting the player onto the ladder.
        # Values for "how_to_get_on_ladder" mean::
        #   - 0: Don't get on any ladders
        #   - 1: Get on the ladder at the bottom
        #   - 2: get on the ladder at the top
        how_to_get_on_ladder: jArray = jnp.multiply(which_ladder_to_get_on, has_found_ladder)
        get_on_ladder_at_top = partial(self._actually_get_on_the_ladder, at_bottom=False)
        get_on_ladder_at_bottom = partial(self._actually_get_on_the_ladder, at_bottom=True)
        dont_do_anything = lambda monte_state, ld_tag, ld_ind, tmp: (monte_state, ld_tag)
        montezuma_state, ladder_tag = jax.lax.switch(how_to_get_on_ladder[0], 
                                                     [dont_do_anything, get_on_ladder_at_bottom, get_on_ladder_at_top], 
                                                     montezuma_state, ladder_tag, ladder_index[0], room_state)
        
        return montezuma_state, ladder_tag
                
                
                
    #
    #
    # These funcions are called when we get on a ladder the normal way, i.e. by standing above/ belove a ladder and pressing an up/down input
    #
    #
    def _handle_get_on_ladder_at_bottom(self, montezuma_state: MontezumaState, room_state: Room, ladder_tag: RoomTags.LADDERS.value) -> Tuple[MontezumaState, Room]:
        bottom_collision_map: jArray = ladder_tag.ladder_bottoms
        corrected_position = montezuma_state.player_position.at[1].add(room_state.vertical_offset[0])
        collision_index: jArray = jax.lax.dynamic_slice(bottom_collision_map, start_indices=corrected_position, 
                                                        slice_sizes=(self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT))
        collision_index = jnp.max(collision_index, keepdims=False)
        
        # Check whether we should do a hard teleport onto the ladder instead (occurs on room change). In this case, prevent player from getting onto ladder 
        # even if currently overlapping with a ladder to prevent glitches
        no_ladder_teleport_required: jArray = jnp.logical_and(jnp.logical_not(montezuma_state.get_on_ladder_bottom), jnp.logical_not(montezuma_state.get_on_ladder_top))
        
        # We can go onto the ladder if we are colliding with a ladder-entrance zone, 
        # and if our current directional input is "up"
        may_climb_onto_ladder: jArray = jnp.logical_and(jnp.logical_and(jnp.not_equal(collision_index, -1), jnp.logical_not(montezuma_state.is_climbing)), 
                            jnp.logical_and(jnp.equal(montezuma_state.current_directional_input, MovementDirection.UP.value), 
                                            no_ladder_teleport_required))
        montezuma_state, ladder_tag = jax.lax.cond(may_climb_onto_ladder[0], partial(self._actually_get_on_the_ladder, at_bottom=True), lambda x, y, z, u: (x, y), montezuma_state, ladder_tag, collision_index, room_state)       
        
        
        return (montezuma_state, ladder_tag)
    
    def _handle_get_on_ladder_at_top(self, montezuma_state: MontezumaState, room_state: Room, ladder_tag: RoomTags.LADDERS.value) -> Tuple[MontezumaState, Room]:
        top_collision_map: jArray = ladder_tag.ladder_tops
        corrected_position = montezuma_state.player_position.at[1].add(room_state.vertical_offset[0])
        collision_index: jArray = jax.lax.dynamic_slice(top_collision_map, start_indices=corrected_position, 
                                                        slice_sizes=(self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT))
        collision_index = jnp.max(collision_index, keepdims=False)
        # Check whether we should do a hard teleport onto the ladder instead (occurs on room change). In this case, prevent player from getting onto ladder 
        # even if currently overlapping with a ladder to prevent glitches
        no_ladder_teleport_required: jArray = jnp.logical_and(jnp.logical_not(montezuma_state.get_on_ladder_bottom), jnp.logical_not(montezuma_state.get_on_ladder_top))
        
        # We can go onto the ladder if we are colliding with a ladder-entrance zone, 
        # and if our current directional input is "down"
        may_climb_onto_ladder: jArray = jnp.logical_and(jnp.logical_and(jnp.not_equal(collision_index, -1), jnp.logical_not(montezuma_state.is_climbing)), 
                                        jnp.logical_and(jnp.equal(montezuma_state.current_directional_input, MovementDirection.DOWN.value), 
                                                        no_ladder_teleport_required))
        montezuma_state, ladder_tag = jax.lax.cond(may_climb_onto_ladder[0], partial(self._actually_get_on_the_ladder, at_bottom=False), lambda x, y, z, u: (x, y), montezuma_state, ladder_tag, collision_index, room_state)       
        
        
        
        return (montezuma_state, ladder_tag)
    
    
    
    
    def _handle_start_climb_wrappable(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # This function handles transitions to climbing states.
        # Handles both getting on ladders & getting on ropes.
        #
        if RoomTags.ROPES in tags:
            rope_tag: RoomTags.ROPES.value = SANTAH.extract_tag_from_rooms[RoomTags.ROPES](room_state)
            # Reset the last-rope tag if the player is currently standing on the floor.
            # last_hanged_on_rope is used to prevent the player from repeatedly regrabbing the same rope.
            new_last_rope: jArray = jax.lax.cond(jnp.logical_and(montezuma_state.is_standing[0], jnp.logical_not(montezuma_state.is_climbing[0])), lambda rope_tag: jnp.array([-1], jnp.int32), lambda rope_tag: rope_tag.last_hanged_on_rope, rope_tag)
            rope_tag = SANTAH.attribute_setters[RoomTags.ROPES.value][RoomTagsNames.ROPES.value.last_hanged_on_rope.value](rope_tag, new_last_rope)
            montezuma_state, rope_tag = self._handle_climb_down_onto_rope(state=montezuma_state, 
                                                                          rope_tag=rope_tag)
            montezuma_state, rope_tag = self._handle_start_climbing_ropes(state=montezuma_state, 
                                                                   rope_tag=rope_tag)

            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.ROPES](room_state, rope_tag)
        
        if RoomTags.LADDERS in tags:
            ladder_tag: RoomTags.LADDERS.value = SANTAH.extract_tag_from_rooms[RoomTags.LADDERS](room_state)
            
            montezuma_state, ladder_tag = self._handle_get_on_ladder_at_bottom(montezuma_state=montezuma_state, 
                                                                        room_state=room_state, 
                                                                        ladder_tag=ladder_tag)
            montezuma_state, ladder_tag = self._handle_get_on_ladder_at_top(montezuma_state=montezuma_state, 
                                                                        room_state=room_state, 
                                                                        ladder_tag=ladder_tag)
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.LADDERS](room_state, ladder_tag)
        return montezuma_state, room_state
        
        
    def _find_nearest_floor(self, montezuma_state: MontezumaState, floor_map: jArray) -> Tuple[jArray, jArray]:
        """Finds the nearest floor horizontally. A floor is a free pixel with ground below it. 
            Finds the nearest floor along the vertical axis, doesnt look along the horizontal axis.       

        Args:
            montezuma_state (MontezumaState): The current game state.
            floor_map (jArray): An integer (int32) map of shape (consts.WIDTH, consts.HEIGHT) which 
            contains floors: All floor pixels are set to 1, all others are set to 0.

        Returns:
            Tuple[jArray, jArray]: A singleton integer array which tells me whether a free floor pixel could be found, 
                                    A position array for the free pixel. Hitmaps are not considered, 
                                    so the free floor found might not actually be occupiable.
        """
        
        x: jArray = montezuma_state.player_position[0]
        # Padd y position with both vertical offset. NO padding with the player height, as we want to find the floor plane 
        # nearest to the head
        padded_y: jArray = montezuma_state.player_position[1] + montezuma_state.room_state.vertical_offset[0]
        vertical_floor_slice: jArray = jax.lax.dynamic_slice(operand=floor_map, 
                                                                    start_indices=(x, 0), 
                                                                    slice_sizes=(1, self.consts.HEIGHT))
        vertical_floor_slice = vertical_floor_slice[0, ...]
        vertical_floor_slice = jnp.reshape(vertical_floor_slice, (self.consts.HEIGHT))
        y_poss: jArray = jnp.arange(start=0, stop=self.consts.HEIGHT, 
                                    dtype=jnp.int32)
        tmp = jnp.multiply(y_poss, vertical_floor_slice)
        zero_tmp = jnp.astype(jnp.equal(tmp, 0), jnp.int32)
        offset_tmp = jnp.multiply(zero_tmp, 5000)
        signed_pos_diff = (tmp + offset_tmp) - padded_y
        pos_diff: jArray = jnp.absolute(signed_pos_diff)
        nearest_floor_y: jArray = jnp.argmin(pos_diff, axis=0, keepdims=False)
        
        # Now check whether we have actually managed to find a free floor, 
        # and if the head of the player would still be within the playfield bounds
        is_valid_pos: jArray = jnp.greater_equal(nearest_floor_y - self.consts.PLAYER_HEIGHT, montezuma_state.room_state.vertical_offset)
        current_floor_pixel: jArray = jax.lax.dynamic_index_in_dim(vertical_floor_slice, index=nearest_floor_y, axis=0)
        is_actually_floor = jnp.reshape(jnp.logical_and(current_floor_pixel, is_valid_pos), shape=(1))
        new_pos = montezuma_state.player_position.at[1].set(nearest_floor_y - (self.consts.PLAYER_HEIGHT + montezuma_state.room_state.vertical_offset[0]))
        
        return is_actually_floor, new_pos     
    
    def rope_top_exit(self, state: MontezumaState, rope_tag: RoomTags.ROPES.value) -> MontezumaState:
        #
        # Function that handles leaving a rope at the top.
        #
        def actually_leave_the_rope(rope_id: jnp.ndarray, rope_tag:  RoomTags.ROPES.value, state: MontezumaState):
            rope: jArray = jax.lax.dynamic_slice_in_dim(rope_tag.ropes, start_index=rope_id[0], 
                                                        slice_size=1, axis=0)
            rope: Rope = SANTAH.full_deserializations[Rope](rope[0])
            # Check whether we are actually in the upper "leave"-Zone.
            is_at_rope_top: jArray = jnp.logical_or(jnp.logical_and(jnp.less_equal(state.player_position[1] - rope.top, self.consts.ROPE_TOP_EXIT_DISTANCE), jnp.less_equal(0, self.consts.ROPE_TOP_EXIT_DISTANCE)), 
                        jnp.logical_and(jnp.greater_equal(rope.top - state.player_position[1], (-1)*self.consts.ROPE_TOP_EXIT_DISTANCE), jnp.less_equal(self.consts.ROPE_TOP_EXIT_DISTANCE, 0)))
                      
            # Decide whether player should be teleported off the rope: 
            # Player gets of the rope if:
            # - There is a floor available to teleport to
            # - Player is at the top of the rope
            # - And the rope is declared as being accesible from the top
            is_a_floor_available: jArray = None
            new_floor_position: jArray = None
            is_a_floor_available, new_floor_position = self._find_nearest_floor(montezuma_state=state, 
                                                                                floor_map=rope_tag.room_surfaces)
            
            
            needs_to_get_off_rope: jArray = jnp.multiply(jnp.multiply(jnp.multiply(is_at_rope_top, state.is_on_rope), rope.accessible_from_top), is_a_floor_available) 
            needs_to_get_off_rope = jnp.astype(needs_to_get_off_rope, jnp.int32)
            
            # Clip Y coordinate of the player to upper end of the rope to prevent player from
            # Climbing out of non-accessible from top ropes.
            new_y_coordinate: jArray = jax.lax.max(rope.top, state.player_position[1])
            updated_player_position: jArray = state.player_position.at[1].set(new_y_coordinate[0])
            player_position_needs_to_be_clipped: jArray = jnp.logical_and(state.is_on_rope, 1 - rope.accessible_from_top)
            new_player_pos: jArray = jnp.multiply(player_position_needs_to_be_clipped, updated_player_position) + jnp.multiply(1 - player_position_needs_to_be_clipped, state.player_position)             
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, new_player_pos)          
            
            
            # Update the state to reflect whether the player left the rope or is still on the rope.
            new_is_on_rope: jArray = jnp.array([0], jnp.int32) + (1 - needs_to_get_off_rope)*state.is_on_rope
            new_is_climbing: jArray = jnp.array([0], jnp.int32) + (1 - needs_to_get_off_rope)*state.is_climbing
            new_last_h_orientation: jArray = needs_to_get_off_rope*jnp.array([Horizontal_Direction.NO_DIR.value], jnp.uint8)+ (1 - needs_to_get_off_rope)*state.last_horizontal_orientation
            new_last_h_orientation: jArray = jnp.astype(new_last_h_orientation, jnp.uint8)
            new_position = needs_to_get_off_rope*(new_floor_position) + (1 - needs_to_get_off_rope)*state.player_position
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_on_rope.value](state, new_is_on_rope)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_climbing.value](state, new_is_climbing)
            state = SANTAH.attribute_setters[MontezumaState][
                    MontezumaStateFields.last_horizontal_orientation.value](state, new_last_h_orientation)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, new_position)
            return state
        
        
        is_not_on_rope: jArray = jnp.equal(rope_tag.rope_index, -1)
        
        state = jax.lax.cond(is_not_on_rope[0], lambda x, y, z: z, actually_leave_the_rope, rope_tag.rope_index, rope_tag, state)
        return state
        
    def _jump_from_rope(self, state: MontezumaState) -> MontezumaState:
        #
        # Function that handles jumping away from the rope.
        #
        
        # Only pure left/right jumps cause the player to leave the rope.
        jump_allowed_dir_when_on_rope: jnp.ndarray = jnp.logical_or(
            jnp.equal(state.horizontal_direction, jnp.array([Horizontal_Direction.LEFT.value], jnp.int32)), 
            jnp.equal(state.horizontal_direction, jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32)),
        )
        jump_allowed_dir_when_on_rope = jnp.max(jump_allowed_dir_when_on_rope)
        start_jump = jnp.logical_and(jump_allowed_dir_when_on_rope, state.jump_input)
        start_jump = jnp.logical_and(start_jump, 1 - state.is_key_hold) # Only jump away from the rope if fresh key press
        start_jump = jnp.logical_and(start_jump, state.is_on_rope)   
        def start_the_jump(state: MontezumaState) -> MontezumaState:
            # Removes the player form the rope & forces a jump start!
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_climbing.value](state, jnp.array([0], jnp.int32))
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_on_rope.value](state, jnp.array([0], jnp.int32))
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.force_jump.value](state, jnp.array([True], jnp.bool))
            return state
        state = jax.lax.cond(start_jump[0], start_the_jump, lambda x: x, state)
        return state
    
    
    def fall_from_rope(self, state: MontezumaState, rope_tag: RoomTags.ROPES.value) -> MontezumaState:
        #
        # Function that handles the player falling down from the bottom of a rope
        #
        
        def check_fall_from_ropes(rope_id: jnp.ndarray, rope_tag:  RoomTags.ROPES.value, state: MontezumaState):
            # Retreive the rope we are currently on
            rope: jArray = jax.lax.dynamic_slice_in_dim(rope_tag.ropes, start_index=rope_id[0], 
                                                        slice_size=1, axis=0)
            rope: Rope = SANTAH.full_deserializations[Rope](rope[0])
            # Check if we are currently at the bottom of the rope
            is_at_rope_bottom: jArray = jnp.less_equal(jnp.abs(rope.bottom - state.player_position[1]), self.consts.ROPE_FALL_VERTICAL_OFFSET)
            needs_to_get_off_rope: jArray = jnp.multiply(is_at_rope_bottom, state.is_on_rope) 
            # Only need to get off rope if you are currently on a rope.
            needs_to_get_off_rope = jnp.astype(needs_to_get_off_rope, jnp.int32)
            new_is_on_rope: jArray = jnp.array([0], jnp.int32) + (1 - needs_to_get_off_rope)*state.is_on_rope
            new_is_climbing: jArray = jnp.array([0], jnp.int32) + (1 - needs_to_get_off_rope)*state.is_climbing
            new_last_h_orientation: jArray = needs_to_get_off_rope*jnp.array([Horizontal_Direction.NO_DIR.value], jnp.uint8)+ (1 - needs_to_get_off_rope)*state.last_horizontal_orientation
            new_last_h_orientation: jArray = jnp.astype(new_last_h_orientation, jnp.uint8)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_on_rope.value](state, new_is_on_rope)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_climbing.value](state, new_is_climbing)
            state = SANTAH.attribute_setters[MontezumaState][
                    MontezumaStateFields.last_horizontal_orientation.value](state, new_last_h_orientation)
        
            return state

        is_not_on_rope: jArray = jnp.equal(rope_tag.rope_index, -1)
        
        state = jax.lax.cond(is_not_on_rope[0], lambda x, y, z: z, check_fall_from_ropes, rope_tag.rope_index, rope_tag, state)
        return state
        
   
    def _actually_leave_the_ladder(self, montezuma_state: MontezumaState, ladder_tag: RoomTags.LADDERS.value, leave_via_top: bool = True):
        """
            Actually leaves the ladder. Sets all attributes in the state/ roomstate that are necessary to leave the ladder.
            Also updates the player position to actually leave the ladder.
        """
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_climbing.value](montezuma_state, jnp.array([0], jnp.int32))
        # this is probably not necessary, but just for safety...
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_on_rope.value](montezuma_state, jnp.array([0], jnp.int32))
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_laddering.value](montezuma_state, jnp.array([0], jnp.int32))
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_bottom.value](montezuma_state, jnp.array([0], jnp.int32))
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_top.value](montezuma_state, jnp.array([0], jnp.int32))
        
        
        # Now set all the necessary attributes in the ladder tag
        ladder_tag = SANTAH.attribute_setters[RoomTags.LADDERS.value][RoomTagsNames.LADDERS.value.ladder_index.value](ladder_tag, jnp.array([0], jnp.int32))      
        
        # Now return the player to the ground: 
        #   - Fall & automatically push out of the wall if exiting a ladder towards the bottom
        #   - Teleport to next floor if exiting a ladder towards the top.
        if leave_via_top:
            vertical_collision_check_hitmap: jArray = jnp.zeros((self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT))
            vertical_collision_check_hitmap = vertical_collision_check_hitmap.at[:, -1].set(1)
            _, new_position = self._find_nearest_floor(montezuma_state=montezuma_state, 
                                                    floor_map=ladder_tag.ladder_room_surface_pixels
                        )
            
        else:
            new_position = montezuma_state.player_position
        
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](montezuma_state, new_position)
        return tuple([montezuma_state, ladder_tag])    
    def _get_off_the_ladder(self, state: MontezumaState, ladder_tag: RoomTags.LADDERS.value) -> Tuple[MontezumaState, RoomTags.LADDERS.value]:
        """
        Check if player is currently on a ladder and whether he is supposed to get off the ladder.
        """
        is_on_ladder: jArray = state.is_laddering
        

        # Execute code to get off the ladder with dummy 0 ladder index if not currently on a ladder to avoid eccessive branching.
        ladder_index: jArray = jax.lax.max(ladder_tag.ladder_index, jnp.array([0], jnp.int32)) # this breaks if we have a ladder room without ladder.
        
        # Retreive the ladder we are currently on.
        current_ladder_arr: jArray = jax.lax.dynamic_index_in_dim(ladder_tag.ladders, index=ladder_index[0], axis=0, keepdims=False)
        current_ladder: Ladder = SANTAH.full_deserializations[Ladder](current_ladder_arr)
        
        # Check whether the player is in a correct position to leave the ladder either at the top or at the bottom.
        player_feet_position: jArray = state.player_position[1] + self.consts.PLAYER_HEIGHT
        top_ladder_leave_y_threshold: jArray = current_ladder.left_upper_y + self.consts.LADDER_TOP_INTERFACE_ZONE_Y_REACH_WHILE_ON
        bottom_ladder_leave_y_threshold: jArray = current_ladder.right_lower_y - self.consts.LADDER_BOTTOM_INTERFACE_ZONE_Y_REACH_WHILE_ON
        is_in_top_leave_zone: jArray = jnp.less_equal(player_feet_position, top_ladder_leave_y_threshold)
        is_in_bottom_leave_zone: jArray = jnp.greater_equal(player_feet_position, bottom_ladder_leave_y_threshold)
        
        # Check whether the player is hitting the correct inputs for leaving the respective zones:
        bottom_leave_inputs: jArray = jnp.array([MovementDirection.LEFT.value, MovementDirection.RIGHT.value, MovementDirection.DOWN.value, 
                                                 MovementDirection.LEFT_DOWN.value, MovementDirection.RIGHT_DOWN.value], jnp.int32)
        top_leave_inputs: jArray = jnp.array([MovementDirection.UP.value, MovementDirection.LEFT_UP.value, 
                                              MovementDirection.RIGHT_UP.value, MovementDirection.LEFT.value, 
                                              MovementDirection.RIGHT.value], jnp.int32)
        is_top_leave_input: jArray = jnp.equal(top_leave_inputs, state.current_directional_input)
        is_top_leave_input = jnp.max(is_top_leave_input, keepdims=True)
        
        is_bottom_leave_input: jArray = jnp.equal(bottom_leave_inputs, state.current_directional_input)
        is_bottom_leave_input = jnp.max(is_bottom_leave_input, keepdims=True)
        
        # If the ladder is top/ bottom seeking don't actually leave the ladder. Instead transition to the next room and teleport to the 
        # next ladder top/ bottom there. Here we bake in the assumption, that if we leave a room towards the top, 
        # we should teleport onto the bottom-interface zone of the nearest ladder. 
        # This limits your flexibility in designing non-euclidean level geometry, if this becomes an issue
        # feel free to change it.
        not_top_teleporting = jnp.logical_not(current_ladder.rope_seeking_at_top)
        not_bottom_teleporting = jnp.logical_not(current_ladder.rope_seeking_at_bottom)
        
        leave_ladder_at_top: jArray = jnp.logical_and(jnp.logical_and(is_top_leave_input, is_in_top_leave_zone), 
                                                      jnp.logical_and(is_on_ladder, not_top_teleporting))
        leave_ladder_at_bottom: jArray = jnp.logical_and(jnp.logical_and(is_bottom_leave_input, is_in_bottom_leave_zone), 
                                                         jnp.logical_and(is_on_ladder, not_bottom_teleporting))
        
        # We now select the function to execute: 
        # At index:
        #   0: We stay on the ladder
        #   1: We leave the ladder at the bottom
        #   2: We leave the ladder at the top
        func_index: jArray = 0 + 1*leave_ladder_at_bottom + 2*leave_ladder_at_top
        
        leave_at_top_func = partial(self._actually_leave_the_ladder, 
                                    leave_via_top=True)
        leave_at_bottom_func = partial(self._actually_leave_the_ladder, 
                                       leave_via_top=False)
        dont_leave_func = lambda x, y : tuple([x, y])
        
        state, ladder_tag = jax.lax.switch(func_index[0], 
                                           [dont_leave_func, leave_at_bottom_func, leave_at_top_func], 
                                           state, ladder_tag
                                           )
        return (state, ladder_tag)
            
    def _handle_stop_climbing_wrappable(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # This function handles termination of climbing for both ropes & ladders.
        #
        if RoomTags.ROPES in tags:
            rope_tag: RoomTags.ROPES.value = SANTAH.extract_tag_from_rooms[RoomTags.ROPES](room_state)
            montezuma_state = self._jump_from_rope(montezuma_state)
            montezuma_state = self.fall_from_rope(state=montezuma_state, rope_tag=rope_tag)
            montezuma_state = self.rope_top_exit(state=montezuma_state, 
                                                 rope_tag=rope_tag)
            
        if RoomTags.LADDERS in tags:
            ladder_tag: RoomTags.LADDERS.value = SANTAH.extract_tag_from_rooms[RoomTags.LADDERS](room_state)
            montezuma_state, ladder_tag = self._get_off_the_ladder(state=montezuma_state, 
                                                         ladder_tag=ladder_tag)
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.LADDERS](room_state, ladder_tag)
        return montezuma_state, room_state
        
        
        
     
    def fall(self, state: MontezumaState) -> MontezumaState:
        """Handles falling for the player. Jumping takes precedence over falling, so if the player is 
           currently jumping, no falling is done

        Args:
            state (MontezumaState): State of the Player
            augmented_collision_map (jnp.ndarray): Collision map which is padded to the full screen dimensions.

        Returns:
            MontezumaState: Updated state with falling attributes set
        """
        augmented_collision_map: jnp.ndarray = state.augmented_collision_map
        # Check how much space there is below the player
        down_distance: int = self.ray_cast_downwards(state.player_position, augmented_collision_map, state.room_state.vertical_offset)
        
        # Returns 1 if down distance >= 1 and the player is not currently jumping
        do_step: int = jnp.min(jnp.array([down_distance, jnp.array([1], jnp.int16), jnp.logical_not(state.is_jumping)], jnp.uint16), keepdims=False) 
        
        
       
        # Default falling distance is 2 pixels, take minimum with the down distance so we don't fall into the floor.
        step_size = jnp.min(jnp.array([down_distance, jnp.array([2], jnp.int16)]), keepdims=False)
        player_position = state.player_position.at[1].set(state.player_position[1] + step_size*do_step)
        _is_falling = jnp.logical_and(jnp.greater(down_distance, 0), jnp.logical_not(state.is_jumping))
        # Player is curretly falling, IF the down distance is > 0 and if he is not currently jumping.
        _is_standing = jnp.logical_and(jnp.equal(down_distance, 0), jnp.logical_not(state.is_jumping))
        
        # This is used to update the horizontal velocity of the player during falling:
        # During falling, the horizontal velocity is halfed.
        _horizontal_fall_velocity = self.set_horizontal_fall_vel(state,state.is_standing,_is_standing)
    
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.horizontal_falling_velocitiy.value](state, _horizontal_fall_velocity)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_standing.value](state, _is_standing)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_falling.value](state, _is_falling)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, player_position)
        return state
    
    def detect_playfield_bound_collision(self, state: MontezumaState) -> jax.Array:
        """Checks whether the player collides with one of the sides of the playfield.
           This is used for teleporting the player into the next room, when another room
           is connected at that side.
        Args:
            state (MontezumaState): Montezumastate.

        Returns:
            jax.Array: Singleton array giving the collision location in terms of the RoomConnectionDirection, 
                       and whether a collision actually occurs.
        """
        room_checks: jnp.ndarray = jnp.zeros((4, ), dtype=jnp.int32)
        room_checks = room_checks.at[RoomConnectionDirections.LEFT.value].set(state.player_position[0]==0)
        room_checks = room_checks.at[RoomConnectionDirections.RIGHT.value].set(state.player_position[0]+self.consts.PLAYER_WIDTH==self.consts.WIDTH)
        room_checks = room_checks.at[RoomConnectionDirections.UP.value].set(state.player_position[1]==0)
        room_checks = room_checks.at[RoomConnectionDirections.DOWN.value].set(state.player_position[1]+self.consts.PLAYER_HEIGHT == state.room_state.height[0])
        collision_location = jnp.argmax(room_checks, axis=0, keepdims=False)
        has_no_collision = jnp.argmin(room_checks, axis=0, keepdims=False)
        has_collision = collision_location == has_no_collision
        has_collision = jnp.logical_not(has_collision)
        has_collision = jnp.reshape(has_collision, shape=(1, ))
        collision_location = jnp.reshape(collision_location, shape=(1, ))
        return collision_location, has_collision
        
    
    def jump(self, state: MontezumaState) -> MontezumaState:
        #Only start jump if:
        # - the player is not falling
        # - not jumping
        # - if this is a fresh keypress (no bunny hopping)
        # - Or whether a jump is forced externally via the state.force_jump field (mainly used for jumping from ropes.)
        jmp_start = state.jump_input*(1 - state.is_falling)*(1 - state.is_jumping)*(1-state.is_key_hold)
        new_jump_input: jArray = jnp.logical_not(state.is_jump_hold)
        jmp_start = jnp.logical_and(jmp_start, new_jump_input)
        jmp_start = jnp.logical_or(jmp_start, state.force_jump)
        
        # on beginning of jump, disable all directional inputs
        block_directional_inputs: jArray = jax.lax.cond(jmp_start[0], lambda x, y: x, lambda x, y: y, jnp.array([1], jnp.int32), state.disable_directional_input)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.disable_directional_input.value](state, block_directional_inputs)
        
        # Disable force jump in case a jump was forced.
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.force_jump.value](state, jnp.array([False], jnp.bool))
        
        # Checks whether the current directional input represents a jump start
        # and prevents the player from initiating a jump if he is currently falling.
        
        is_jumping_update = jmp_start*jnp.array([1], jnp.uint8) + (1 - jmp_start)*state.is_jumping

        # If player is starting a jump, is_jumping is set to 1, else its set to its old value.
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_jumping.value](state, jnp.reshape(is_jumping_update, (1, )))
        init_jmp_counter = jmp_start*jnp.array([0], jnp.uint8) + (1-jmp_start)*state.jump_counter
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.jump_counter.value](state, jnp.reshape(init_jmp_counter, (1, )))    
        # Initilizes the jump counter. Only sets to 0 if a jump is curretly initialized
        
        jmp_step = ((state.jump_counter + 1)%self.consts.JUMP_Y_OFFSETS.shape[0])*(1 - jmp_start) + jmp_start*state.jump_counter
        # Jump step. Increments the jump counter by one. This is only done if the current step is not the start of the jump.
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.jump_counter.value](state, jnp.reshape(jmp_step, (1, )))

        jmp_amt = jax.lax.dynamic_index_in_dim(self.consts.JUMP_Y_OFFSETS, 
                                               index=state.jump_counter[0], axis=0, keepdims=False)*state.is_jumping
        
        
        # Check if the down distance is Smaller than the amount the player wants to drop, if so disable the jump
        down_jump_distance: jArray = jax.lax.min(jmp_amt, 0)
        down_distance: jArray = self.ray_cast_downwards(player_position=state.player_position, level_collision_map=state.augmented_collision_map, 
                                                        vert_offset=state.room_state.vertical_offset)
        
        jmp_needs_to_be_terminated_early: jArray = jnp.less(down_distance, jnp.abs(down_jump_distance))
        down_jump_distance: jArray = jnp.multiply(jmp_needs_to_be_terminated_early, 0 - down_distance) + jnp.multiply(1 - jmp_needs_to_be_terminated_early, jmp_amt)
        
        # Amount the player is currently supposed to jump 
        new_position: jnp.ndarray = state.player_position.at[1].set(state.player_position[1] - down_jump_distance[0])
        
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, new_position)
        # Updates the player position based on the current jump ammount.
        
        
        #check if all jump frames have been exhausted and the player is supposed to be done jumping
        still_jumping = jnp.logical_and(jnp.logical_not(jnp.equal(self.consts.JUMP_Y_OFFSETS.shape[0] - 1, state.jump_counter)), state.is_jumping)
        still_jumping = jnp.logical_and(still_jumping, jnp.logical_not(jmp_needs_to_be_terminated_early))
        #if the jump is stopped, queue the directional input to be enabled again.
        enable_directional_input_again: jArray = jnp.logical_and(jnp.logical_not(still_jumping), state.is_jumping)
        enable_directional_input_again = jnp.astype(enable_directional_input_again, jnp.int32)
        enable_directional_input_again = jnp.astype(enable_directional_input_again, jnp.int32)*(self.consts.INPUT_DISABLED_FRAMES_AFTER_JUMP + 1) + (1-enable_directional_input_again)*state.queue_enable_directional_input
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.queue_enable_directional_input.value](state, enable_directional_input_again)
        
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_jumping.value](state, jnp.reshape(still_jumping, (1, )))
        
        return state
    
    def _may_fall(self, state: MontezumaState):
        movement_gates: jArray = jnp.array([state.is_on_rope[0], state.is_laddering[0]], dtype=jnp.int32)
        may_move = 1 - jnp.max(movement_gates, keepdims=True)
        return may_move
    
    def _may_climb(self, state: MontezumaState):
        movement_gates: jArray = jnp.array([state.is_on_rope[0], state.is_laddering[0]], dtype=jnp.int32)
        may_climb = jnp.max(movement_gates, keepdims=True)
        return may_climb
    
    

    
    def _may_jump(self, state: MontezumaState):
        movement_gates: jArray = jnp.array([state.is_on_rope[0], state.is_laddering[0]], dtype=jnp.int32)
        may_jump = 1 - jnp.max(movement_gates, keepdims=True)
        may_jump = jnp.logical_or(may_jump, state.force_jump)
        return may_jump
    
    def climb(self, state: MontezumaState) -> MontezumaState:
        """Function is used for climbing movement, both ropes & ladders. 
            
            The "Getting on" and "getting off" logic should be handled in a climbing-modality specific function.

        """
        new_pos: jnp.ndarray =  jax.lax.switch(index=state.vertical_directional_input[0],
                                                            branches=[
                                                               lambda x: x.at[1].set(x[1] - state.player_velocity[0]),
                                                               lambda x: x,
                                                               lambda x: x.at[1].set(x[1] + state.player_velocity[0])
                                                            ], operand=state.player_position)

        new_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, new_pos)
        
        return new_state
        
    
    
    def vertical_movement(self, state: MontezumaState) -> MontezumaState:
        """Handles all Vertical Movement of the player.
           This mainly includes jumping, falling and climbing
        """
                
        may_fall: jArray = self._may_fall(state)
        state = jax.lax.cond(may_fall[0], self.fall, lambda x: x, state)
        may_jump: jArray = self._may_jump(state)
        state = jax.lax.cond(may_jump[0], self.jump, lambda x : x, state)
        may_climb: jArray = self._may_climb(state)
        state = jax.lax.cond(may_climb[0], self.climb, lambda x : x, state)
        return state
        
    def die(self, state: MontezumaState):
        #
        # Handles player death
        #
        def die_die(state: MontezumaState):
            room: Room = state.room_state
            
            # Respawn the player at the position from which the room was entered.
            spawn_points: jArray = jnp.zeros((4, 2), dtype=jnp.int32)
            spawn_points = spawn_points.at[RoomConnectionDirections.DOWN.value].set(room.bottom_start_position)
            spawn_points = spawn_points.at[RoomConnectionDirections.LEFT.value].set(room.left_start_position)
            spawn_points = spawn_points.at[RoomConnectionDirections.RIGHT.value].set(room.right_start_position)
            spawn_points = spawn_points.at[RoomConnectionDirections.UP.value].set(room.top_start_position)

            # choose the correct spawn position
            init_pos: jArray = jax.lax.dynamic_index_in_dim(spawn_points, index=state.last_entrance_direction[0], axis=0)
            init_pos = jnp.reshape(init_pos, (2))
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, init_pos)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_dying.value](state, jnp.array([0], jnp.int32))
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.lifes.value](state, state.lifes - 1)
            
            # Reset all movement specific state attributes to prevent bugs
            _, reset_state = self.reset()
            for f in self.consts.DEATH_RESET_FIELDS:
                content = getattr(reset_state, f)
                state = SANTAH.attribute_setters[MontezumaState][f](state, content)
            
            # If the player entered the rope via a ladder, teleport the player back onto this ladder.
            state = self.WRAPPED_HANDLE_ON_DEATH_LADDER_SEEKING(state)
            return state
        state = jax.lax.cond(state.is_dying[0], die_die, lambda s: s, state)
        return state
        
    
    def handle_room_change(self, state: MontezumaState) -> MontezumaState:
        """
        Handles switching between rooms.
        """
        # Check whether the player is touching the wall. Touching the wall triggers room transitions.
        collision_direction, has_collision = self.detect_playfield_bound_collision(state)
        def check_collision(state: MontezumaState, collision_direction: jax.Array) -> MontezumaState:
            room_id: jArray = None
            other_room_entrance_side: jArray = None
            room_id, other_room_entrance_side = self.room_connection_map(state.room_state.ROOM_ID, collision_direction[0])
            # Room id is either -1 if no room is connected 
            # Or the respective room ID if it is connected
            offset_room_id = room_id + 1
            def stay_in_room(state: MontezumaState, _room_id: jArray, room_entrance_direction: jArray):
                return state
            def switch_rooms(state: MontezumaState, _room_id: jArray, room_entrance_direction: jArray):
                #
                # Handles actually switching between rooms.
                #
                room_spawn_side: jArray = room_entrance_direction
                # Determin the side of the new room at which we spawn. 
                #
                #Write any changes to the current room to storage
                updated_persistence_state = self.WRITE_PROTO_ROOM_TO_PERSISTENCE(state.room_state.ROOM_ID, state.room_state, state.persistence_state)
                state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.persistence_state.value](state, updated_persistence_state)
                
                state = self._handle_reset_persistence(state)
                
                #load the fresh room from storage
                new_room: Room = self.PROTO_ROOM_LOADER(_room_id, state.persistence_state)
                
                # Set the new room as the current room:
                state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.room_state.value](state, new_room)
                
                # Now update the position of the player: use the default spawn positions defined in the new room:
                spawn_points: jArray = jnp.zeros((4, 2), dtype=jnp.int32)
                spawn_points = spawn_points.at[RoomConnectionDirections.DOWN.value].set(new_room.bottom_start_position)
                spawn_points = spawn_points.at[RoomConnectionDirections.LEFT.value].set(new_room.left_start_position)
                spawn_points = spawn_points.at[RoomConnectionDirections.RIGHT.value].set(new_room.right_start_position)
                spawn_points = spawn_points.at[RoomConnectionDirections.UP.value].set(new_room.top_start_position)
                
                # choose the correct spawn position
                init_pos: jArray = jax.lax.dynamic_index_in_dim(spawn_points, index=room_spawn_side[0], axis=0)
                
                # Always store the side from which a room was entered.
                last_entrance: jArray = state.last_entrance_direction
                last_entrance = last_entrance.at[0].set(room_spawn_side[0])
                state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.last_entrance_direction.value](state, last_entrance)
                
                init_pos = jnp.reshape(init_pos, (2))
                state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, init_pos)
                
                return state
            
            # A room id of -1 indicates, that no room is connected at that side.
            stay_or_go = jnp.equal(room_id, -1)
            stay_or_go = stay_or_go[0]

            state = jax.lax.cond(stay_or_go, stay_in_room, switch_rooms, state, room_id, other_room_entrance_side) 
            state = jax.lax.cond(stay_or_go, lambda _state: _state, self.WRAPPED_HANDLE_ROOM_ENTRANCE, state)               
            return state   
          
        def ignore_collision(state: MontezumaState, collision_direction: jax.Array) -> MontezumaState:
            return state
        has_collision = has_collision[0]
        new_state: MontezumaState = jax.lax.cond(has_collision, check_collision, ignore_collision, state, collision_direction)
        return new_state
    
    def fix_player_position(self, state: MontezumaState, new_player_position: jArray, old_player_position: jArray):
        """
        Fixes the player position.
        """
        
        state = self.push_player_back_onto_playfield(state) # Returns the player back into the bounds 
            # of the playfield.
        vertical_collision_check_hitmap: jArray = jnp.zeros((self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT))
        vertical_collision_check_hitmap = vertical_collision_check_hitmap.at[:, -1].set(1)
        # For the vertical collision check, only ever use the feet of the player.
        # This is accurate to the game
        
        # Check if the player is supposed to collide with the environment in the current state
        may_be_pushed_into_free_space: jArray = self._may_collide_with_environment(state, 
                                                                                   new_player_position=new_player_position, 
                                                                                   old_player_position=old_player_position)
        
        # Find the nearest free position that is occupiable by the player
        new_position = self.find_nearest_free_position_2D_conv(state, vertical_collision_check_hitmap)
        
        
        final_new_position = (1 - may_be_pushed_into_free_space[0])*state.player_position + may_be_pushed_into_free_space[0]*new_position
        final_new_position = jnp.astype(final_new_position, jnp.int32)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.player_position.value](state, final_new_position)

        
        
        return state
    
    
    
    def _handle_on_room_change(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        """
            This function handles all actions that need to be performed if we chane a room.
        """
        if RoomTags.BONUSROOM in tags:
            bonus_tag: RoomTags.BONUSROOM.value = SANTAH.extract_tag_from_rooms[RoomTags.BONUSROOM](room_state)
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.reset_room_state_on_room_change.value](montezuma_state, bonus_tag.reset_state_on_leave)
        if RoomTags.LADDERS in tags:
            # If we left a room via a ladder and the room we have entered also implements ladders, 
            # teleport the player onto the nearest ladder.
            ladder_tag: RoomTags.LADDERS.value = SANTAH.extract_tag_from_rooms[RoomTags.LADDERS](room_state)
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.top_seeking_on_entrance.value](montezuma_state, montezuma_state.get_on_ladder_top)
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.bottom_seeking_at_entrance.value](montezuma_state, montezuma_state.get_on_ladder_bottom)
            montezuma_state, ladder_tag = self._handle_teleport_onto_ladder(montezuma_state=montezuma_state, 
                                              ladder_tag=ladder_tag, room_state=room_state)
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.LADDERS](room_state, ladder_tag)
            
        else:
            # Else (should only occur if we have configured the layout badly), disable the ladder-seeking behavior
            # to prevent bugs when next entering a room with ladders.
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_bottom.value](montezuma_state, jnp.array([0], jnp.int32))
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_top.value](montezuma_state, jnp.array([0], jnp.int32))
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.top_seeking_on_entrance.value](montezuma_state, montezuma_state.get_on_ladder_top)
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.bottom_seeking_at_entrance.value](montezuma_state, montezuma_state.get_on_ladder_bottom)
        montezuma_state = self.WRAPPED_HANDLE_DARKNESS(montezuma_state)
        return montezuma_state, room_state
    
    def _handle_on_death_ladder_seeking_behavior(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # This funciton is called when the player dies. 
        # It handles, that if the player entered the room via a ladder, 
        # On death, he is teleported back onto the appropriate ladder
        #
        if RoomTags.LADDERS in tags:
            ladder_tag: RoomTags.LADDERS.value = SANTAH.extract_tag_from_rooms[RoomTags.LADDERS](room_state)
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_top.value](montezuma_state, montezuma_state.top_seeking_on_entrance)
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.get_on_ladder_bottom.value](montezuma_state, montezuma_state.bottom_seeking_at_entrance)
            montezuma_state, ladder_tag = self._handle_teleport_onto_ladder(montezuma_state=montezuma_state, 
                                              ladder_tag=ladder_tag, room_state=room_state)
            room_state = SANTAH.write_back_tag_information_to_room[room_type][RoomTags.LADDERS](room_state, ladder_tag)
            
        
        return montezuma_state, room_state
    
    
    def execute_queued_actions(self, state: MontezumaState) -> MontezumaState:
        # Check whether enabling input again is queued
        
        do_input_unlock, new_queue_value = jax.lax.switch(state.queue_enable_directional_input[0], 
                       [lambda x: (jnp.array([0], jnp.int32), jnp.array([0], jnp.int32)), 
                       lambda x: (jnp.array([1], jnp.int32), jnp.array([0], jnp.int32)),
                       lambda x: (jnp.array([0], jnp.int32), x - 1)],
                       state.queue_enable_directional_input
                       )
        
        
        input_disabled: jArray = jax.lax.cond(do_input_unlock[0], lambda x: jnp.array([0], jnp.int32), lambda x: x, state.disable_directional_input)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.disable_directional_input.value](state, input_disabled)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.queue_enable_directional_input.value](state, new_queue_value)
        return state
    
    @partial(jax.jit, static_argnums=(0))
    def player_step(self, state: MontezumaState, action: chex.Array):
        old_position: jArray = state.player_position
        # Paint Collisions for all featurees onto the global collision map.
        state = self.WRAPPED_AUGMENT_COLLISION_MAP(state)
        state = self.WRAPPED_ADD_BONUS_ROOM_FLOOR_COLLISION_TO_ROOM_COLLISION_MAP(state)
        state = self.WRAPPED_ADD_DOOR_COLLISION_TO_ROOM_COLLISION_MAP(state)
        state = self.WRAPPED_ADD_DROPOUT_FLOOR_COLLISION_TO_ROOM_COLLISION_MAP(state)
        state = self.WRAPPED_ADD_SIDE_WALL_COLLISION_TO_ROOM_COLLISION_MAP(state)
        state = self.WRAPPED_ADD_CONVEYOR_BELT_COLLISION_TO_ROOM_COLLISION_MAP(state)
        state = self.WRAPPED_HANDLE_CONVEYOR_MOVEMENT(state)
        
        # Handles user input. Does not touch the is_jumping or is_falling fields
        # But sets current input and current horizontal direction.
        state = self.handle_user_input(state, action)
         
        # Executes horizontal movement if allowed in the current game state (not jumping, not falling)
        may_move_horizontally: jArray = self._may_move_horizontally(state)
        state = jax.lax.cond(may_move_horizontally[0], self._player_horizontal_movement, lambda x: x, state)
        
        # Handle all vertical movement.
        state = self.WRAPPED_HANDLE_START_CLIMBING(state)
        state = self.WRAPPED_HANDLE_STOP_CLIMBING(state)
        state = self.vertical_movement(state)
        
        
        # Handle all interactions with the room features
        state = self.WRAPPED_HANDLE_SARLACC_PIT_COL(state)
        state = self.WRAPPED_HANDLE_LAZER_BARRIER_COL(state)
        state = self.WRAPPED_HANDLE_ITEM_COLLISION_CHECK(state)
        state = self.WRAPPED_HANDLE_DOOR_COLLISION(state)
        state = self.WRAPPED_HANDLE_ENEMIES(state)
        state = self.WRAPPED_HANDLE_ENEMY_COLLISION(state)
        
        # Correct player position by pushing back onto the playfield.
        new_position: jArray = state.player_position
        state = self.fix_player_position(state, 
                new_player_position=new_position, 
                old_player_position=old_position)
        state = self.die(state)
        state = self.execute_queued_actions(state)
        return state
    
    @partial(jax.jit, static_argnames=["self", "freeze_type"])
    def queue_freeze(self, state: MontezumaState, freeze_type: FreezeType):
        # Function used to schedule freezes for the end of the step.
        # Freezes are triggered at various in game events.
        freeze_type_arr: jArray = jnp.array([freeze_type.value], jnp.int32)
        queue_freeze: jArray = jnp.array([1], jnp.int32)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.freeze_type.value](state, freeze_type_arr)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.queue_freeze.value](state, queue_freeze)
        return state
        
    def _trigger_freeze(self, post_step_state: MontezumaState, pre_step_state: MontezumaState) -> jArray:
        # Assumes that a freeze was triggered somewhere in this step: 
        #
        freeze_time: jArray = jax.lax.dynamic_index_in_dim(self.consts.FREEZE_TYPE_LENGTHS, index=post_step_state.freeze_type[0])
        frozen: jArray = jnp.array([1], jnp.int32)
        freeze_type: jArray = post_step_state.freeze_type
        # now set freeze type and frozen in the old state which is preserved.
        pre_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frozen.value](pre_step_state, frozen)
        pre_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.freeze_remaining.value](pre_step_state, freeze_time)
        pre_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.freeze_type.value](pre_step_state, freeze_type)
        pre_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.queue_freeze.value](pre_step_state, jnp.array([0], jnp.int32))
        # Copy the position from the new step into the old step to make rendering more accurate
        
        # Prepare the post-step state to be cached
        # During a freeze, the pre-step state is rendered, but the game is resumed from the post-step state.
        # This is done, so killed enemies are still rendered during the freeze.
        post_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.queue_freeze.value](post_step_state, jnp.array([0], jnp.int32))
        post_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frozen_state.value](post_step_state, None)
        
        # Set the frozen state
        pre_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frozen_state.value](pre_step_state, post_step_state)
        return pre_step_state
    
    def disable_freeze(self, state: MontezumaState) -> MontezumaState:
        #
        # Disable the freeze!
        #
        # Prepare the restored state to resume gameplay:
        #   - Set the frozen state to the current state (default value to preserve shape)
        #   - Set the freeze attribute to "unfrozen"
        #   - Take the current value of the frame counter
        
        
        restored_state: MontezumaState = state.frozen_state
        
        # Set freeze attribute to unfrozen (should have already been done, but just to be save...)
        restored_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frozen.value](restored_state, jnp.array([0], jnp.int32))
        restored_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.queue_freeze.value](restored_state, jnp.array([0]))
        
        
        # Set dummy value for freeze state to preserve shape
        _dummy_restored: MontezumaState = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frozen_state.value](restored_state, None)
        restored_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frozen_state.value](restored_state, _dummy_restored)
        
        # set the new value for the frame counter
        restored_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frame_count.value](restored_state, state.frame_count)
        restored_state = self.die(restored_state)
        return restored_state
    
    def _continue_freeze(self, state: MontezumaState, action: chex.Array) -> MontezumaState:
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frame_count.value](state, state.frame_count + 1)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.freeze_remaining.value](state, state.freeze_remaining - 1)
        return state
    
    
    def _handle_fall_damage(self, pre_step_state: MontezumaState, post_step_state: MontezumaState) -> MontezumaState:
        """Performs Fall damage handling:
            Does the following things:
                - if the state changed from "non falling" to "falling": Record the pre-fall position
                - if the state changed from "falling" to "non-falling": Check the recorded pre-fall position, 
                    and see if the difference was high enough to incur fall damage. 
                - Fall damage is not incurred if the player: Changes rooms (this is the case in the bonus room)
                - And if the player lands on a rope
        """
        fall_begun: jArray = jnp.logical_and(jnp.logical_not(pre_step_state.is_falling), post_step_state.is_falling)
        new_fall_start_position: jArray = jnp.multiply(fall_begun, pre_step_state.player_position) + jnp.multiply((1-fall_begun), post_step_state.fall_position)
        post_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.fall_position.value](post_step_state, new_fall_start_position)
        
        # Handle the case if the fall has stopped:
        fall_stopped: jArray = jnp.logical_and(pre_step_state.is_falling, jnp.logical_not(post_step_state.is_falling))
        
        down_distance: int = self.ray_cast_downwards(post_step_state.player_position, post_step_state.augmented_collision_map, post_step_state.room_state.vertical_offset)
        
        # Also check if the down_disance is less than 0. This means the player has somehow ended up 
        # above it's initial fall position, and can happen on teleport.
        fall_stopped = jnp.logical_or(fall_stopped, jnp.less_equal(down_distance, 0))
        # If the fall has stopped, set the fall position to a "harmless"  default position
        new_fall_position: jArray = jnp.multiply(fall_stopped, jnp.array([900, 900], jnp.int32)) + jnp.multiply(1-fall_stopped, post_step_state.fall_position)
        post_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.fall_position.value](post_step_state, new_fall_position)
        
        #we had a problem with the fall death and enemy collision interaction and had to trigger the fall dmg death before hitting the ground
        #this is purely for cosmetic reasons, so the dying player gets displayed right
        old_r_o = post_step_state.render_offset_for_fall_dmg[0]
        new_render_offset_for_fall_dmg = 1 - jnp.mod(down_distance + (old_r_o - down_distance) * jnp.logical_not(down_distance),2)
        
        post_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.render_offset_for_fall_dmg.value](post_step_state, new_render_offset_for_fall_dmg)
        
        # Check if the player needs to die:
        # The player doesn't die if he has changed rooms (this is for the bonus room)
        # or if the fall height wasn't great enough.
        needs_to_die: jArray = jnp.logical_and(jnp.logical_and(fall_stopped, 1 - post_step_state.is_on_rope), jnp.logical_and(
            jnp.greater_equal(post_step_state.player_position[1] - pre_step_state.fall_position[1], self.consts.MAXIMUM_ALLOWED_FALL_HEIGHT), 
            jnp.equal(post_step_state.room_state.ROOM_ID, pre_step_state.room_state.ROOM_ID)))
        
        # trigger a freeze if the player has died from fall damage.
        freeze_trigger = partial(self.queue_freeze, freeze_type=FreezeType.FALL_DEATH)
        post_step_state = jax.lax.cond(needs_to_die[0], freeze_trigger, lambda x : x, post_step_state)
        new_death: jArray = jnp.multiply(needs_to_die, jnp.array([1], jnp.int32)) + jnp.multiply(1 - needs_to_die, post_step_state.is_dying)
        post_step_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.is_dying.value](post_step_state, new_death)
        return post_step_state
    
    
    @partial(jax.jit, static_argnums=(0))
    def _non_frozen_step(self, state: MontezumaState, action: chex.Array) -> MontezumaState:
        # Step that is executed if the game is not currently frozen
        old_state: MontezumaState = state
        old_position: jArray = state.player_position
        state = self.player_step(state, action)
        new_position: jArray = state.player_position
        
        
        # Handle the velocity tracking
        # This is important for the ladder & rope animations.
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.last_velocity.value](state, state.current_velocity)
        new_velocity: jArray = new_position - old_position
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.current_velocity.value](state, new_velocity)
        kept_velocity: jArray = jnp.min(jnp.equal(state.current_velocity, state.last_velocity), keepdims=False)
        new_velocity_counter = jnp.multiply(1 - kept_velocity, 0) + jnp.multiply(kept_velocity, state.velocity_held + 1)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.velocity_held.value](state, new_velocity_counter)
        # Same for the horizontal direction
        kept_horizontal_dir: jArray = jnp.equal(state.horizontal_direction, state.last_horizontal_direction)
        new_horizontal_direction_held: jArray = jnp.multiply(1 - kept_horizontal_dir, 0) + jnp.multiply(kept_horizontal_dir, state.horizontal_direction_held + 1)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.horizontal_direction_held.value](state, new_horizontal_direction_held)
        # Handle the room change logic.
        old_room_id: jArray = state.room_state.ROOM_ID
        state = self.handle_room_change(state)
        
        
        new_room_id: jArray = state.room_state.ROOM_ID
        room_change: jArray = jnp.not_equal(old_room_id, new_room_id)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.frame_count.value](state, state.frame_count + 1)
        state = self.WRAPPED_HANDLE_COUNTER_UPDATES(state)
        state = jax.lax.cond(room_change[0], self.WRAPPED_ON_ROOM_CHANGE, lambda x : x, state)
        state = self._handle_fall_damage(old_state, state)
        return state
    
    def _handle_reset_persistence(self, montezuma_state: MontezumaState) -> MontezumaState:
        # This is a seperate function that is called on room change.
        # Handles reset of the rooms on room change.
        # This is used for looping layouts in the bonus room
        needs_to_reset_rooms: jArray = montezuma_state.reset_room_state_on_room_change
        
        new_items: jArray = jnp.zeros_like(montezuma_state.itembar_items)
        new_items = jax.lax.cond(needs_to_reset_rooms[0], lambda x, y: x, lambda x, y: y, new_items, montezuma_state.itembar_items)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.itembar_items.value](montezuma_state, new_items)
        new_rooms_state: jArray = jax.lax.cond(needs_to_reset_rooms[0], lambda x, y: y, lambda x, y: x, montezuma_state.persistence_state, self.initial_persistence_state)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.persistence_state.value](montezuma_state, new_rooms_state)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.reset_room_state_on_room_change.value](montezuma_state, jnp.array([0], jnp.int32))
        my_room: Room = self.PROTO_ROOM_LOADER(montezuma_state.room_state.ROOM_ID, montezuma_state.persistence_state)
        montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.room_state.value](montezuma_state, my_room)
        return montezuma_state
    @partial(jax.jit, static_argnums=(0))
    def step(self, state: MontezumaState, action: chex.Array) -> Tuple[MontezumaObservation, MontezumaState, float, bool, MontezumaInfo]:
        
        # Handle game freeze.
        old_state: MontezumaState = state
        new_state: MontezumaState = jax.lax.cond(state.frozen[0], self._continue_freeze, self._non_frozen_step, state, action)
        
        start_freeze: jArray = new_state.queue_freeze
        new_state = jax.lax.cond(start_freeze[0], self._trigger_freeze, lambda x, y : x, new_state, old_state)
        
        # Check if freeze needs to be terminated
        freeze_needs_to_be_terminated: jArray = jnp.logical_and(new_state.frozen, jnp.less(new_state.freeze_remaining, 0))
        new_state = jax.lax.cond(freeze_needs_to_be_terminated[0], self.disable_freeze, lambda x : x, new_state)
        
        # Observation computation needs to be wrapped, as we need access to the individual room features to compute an observation.
        new_state = self.WRAPPED_HANDLE_COMPUTE_OBSERVATION(new_state)
        observation: MontezumaObservation = new_state.observation
        reward = self._get_reward(old_state, new_state)
        all_rewards = self._get_all_reward(old_state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state, all_rewards)
        return observation, new_state, reward, done, info
    
    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)
    def observation_space(self) -> spaces.Dict:
        
        return spaces.Dict({
            "annotated_collision_map": spaces.Box(low=0, high=200, shape=(160, 210, 2), dtype=jnp.int32),
            "current_items":spaces.Box(low=0, high= max(self.consts.WIDTH, self.consts.HEIGHT), shape=(5, ), dtype=jnp.int32),
            "has_dropout_floors": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "has_pit": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "is_falling": spaces.Box(low=0, high= 1, shape=(), dtype=jnp.int32),
            "is_jumping": spaces.Box(low=0, high= 1, shape=(), dtype=jnp.int32),
            "is_on_ladder": spaces.Box(low=0, high= 1, shape=(), dtype=jnp.int32),
            "is_on_rope": spaces.Box(low=0, high= 1, shape=(), dtype=jnp.int32),
            "nearest_enemy": spaces.Dict({
                "alive":spaces.Box(low=0, high= 1, shape=(), dtype=jnp.int32), 
                "dummy":spaces.Box(low=0, high= 1, shape=(), dtype=jnp.int32),
                "position":spaces.Box(low=0, high= max(self.consts.WIDTH, self.consts.HEIGHT), shape=(2, ), dtype=jnp.int32),
                "type":spaces.Box(low=0, high= len(EnemyType), shape=(), dtype=jnp.int32)
            }),
            "nearest_item": spaces.Dict({
                "dummy":spaces.Box(low=0, high= 1, shape=(), dtype=jnp.int32),
                "position":spaces.Box(low=0, high= max(self.consts.WIDTH, self.consts.HEIGHT), shape=(2, ), dtype=jnp.int32),
                "type":spaces.Box(low=0, high= len(Item_Sprites) - 1, shape=(), dtype=jnp.int32),
                "collected":spaces.Box(low=0, high= 1, shape=(), dtype=jnp.int32)
            }), 
            "number_of_lives":spaces.Box(low=0, high= self.consts.LIFES_STARTING_Y, shape=(), dtype=jnp.int32),
            "player_position":spaces.Box(low=0, high= max(self.consts.WIDTH, self.consts.HEIGHT), shape=(2, ), dtype=jnp.int32) 
                            
            }
        )
    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: MontezumaObservation) -> jnp.ndarray:
        def _flatten_enemy(en: EnemyObservation) -> jnp.ndarray:
            return jnp.concatenate(
                [
                    jnp.array([en.alive, en.dummy], jnp.int32),
                    en.position,
                    jnp.array([en.type], jnp.int32)
                ],
                axis=0,
            )
            
        def _flatten_item(it: ItemObservation) -> jnp.ndarray:
            return jnp.concatenate(
                [   
                    jnp.array([it.dummy], jnp.int32), 
                    it.position,
                    jnp.array([it.type, it.collected], jnp.int32)
                    ],
                axis=0,
            )

        
        flops = jnp.concatenate(
            [
                jnp.ravel(obs.annotated_collision_map),
                obs.current_items,
                jnp.array([obs.has_dropout_floors,
                           obs.has_pit, 
                           obs.is_falling, 
                           obs.is_jumping, 
                           obs.is_on_ladder, 
                           obs.is_on_rope
                           ], jnp.int32),
                _flatten_enemy(obs.nearest_enemy),
                _flatten_item(obs.nearest_item),
                
                jnp.array([ 
                           obs.number_of_lives], jnp.int32),
                
                
                obs.player_position
                
            ],
            axis=0,
        )
        return flops

    def _get_observation(self, state: MontezumaState) -> MontezumaObservation:

        return state.observation
    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: MontezumaState, all_rewards: jnp.ndarray = None) -> MontezumaInfo:
        info = MontezumaInfo(
            step_counter=state.frame_count[0], 
            lives=state.lifes[0], 
            all_rewards=all_rewards, 
            room_id=state.room_state.ROOM_ID[0]
        )
        return info
    

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: MontezumaState, state: MontezumaState):
        return state.score[0] - previous_state.score[0]

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: MontezumaState) -> bool:
        return state.lifes < 0
    
    
    @partial(jax.jit, static_argnums=(0,))    
    def _get_all_reward(
        self, previous_state: MontezumaState, state: MontezumaState
    ):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [
                reward_func(previous_state, state)
                for reward_func in self.reward_funcs
            ]
        )
        return rewards

class MontezumaRenderer(JAXGameRenderer):
    def __init__(self, consts: MontezumaConstants = None):
        # Load all required sprites from disk. 
        # We use our own loading method which interpretes a black background (0, 0, 0) as transparent.
        super().__init__()
        gc.collect()
        self.consts = consts or MontezumaConstants()
        
        self.sprite_path: str = os.path.join(self.consts.MODULE_DIR, "sprites", "montezuma")
        self.sprite_f_dict: Dict[str, str] = self.consts.SPRITE__F_DICT
        self.player_sprite: jnp.ndarray =  loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["player"]), 
                                                            add_alpha=True, add_black_as_transparent=True, transpose=True)
        self.life_sprite: jnp.ndarray = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["life_sprite"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True)
        self.zero: jnp.ndarray = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["0"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True)
        self.door_sprite: jnp.ndarray = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["door"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True)
        
        self.item_sprites: jnp.ndarray = jnp.zeros(shape=(6, 7, 15, 4),dtype=jnp.int32)
        self.item_sprites = self.item_sprites.at[Item_Sprites.GEM.value, ...].set(loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["gem"]),
                                                                      add_alpha=True, add_black_as_transparent=True, transpose=True))
        self.item_sprites = self.item_sprites.at[Item_Sprites.HAMMER.value, ...].set(loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["hammer"]),
                                                                      add_alpha=True, add_black_as_transparent=True, transpose=True))
        self.item_sprites = self.item_sprites.at[Item_Sprites.KEY.value, ...].set(loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["key"]),
                                                                      add_alpha=True, add_black_as_transparent=True, transpose=True))
        self.item_sprites = self.item_sprites.at[Item_Sprites.SWORD.value, ...].set(loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["sword"]),
                                                                      add_alpha=True, add_black_as_transparent=True, transpose=True))
        self.item_sprites = self.item_sprites.at[Item_Sprites.TORCH.value, ...].set(loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["torch_1"]),
                                                                      add_alpha=True, add_black_as_transparent=True, transpose=True))
        self.item_sprites = self.item_sprites.at[Item_Sprites.TORCH_FRAME_2.value, ...].set(loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["torch_2"]),
                                                                      add_alpha=True, add_black_as_transparent=True, transpose=True))
        self.item_sprites_color: jnp.ndarray = jnp.zeros(shape=(6,3),dtype=jnp.int32)
        self.item_sprites_color = self.item_sprites_color.at[Item_Sprites.GEM.value, ...].set(self.consts.GEM_COLOR)
        self.item_sprites_color = self.item_sprites_color.at[Item_Sprites.HAMMER.value, ...].set(self.consts.HAMMER_COLOR)
        self.item_sprites_color = self.item_sprites_color.at[Item_Sprites.KEY.value, ...].set(self.consts.KEY_COLOR)
        self.item_sprites_color = self.item_sprites_color.at[Item_Sprites.SWORD.value, ...].set(self.consts.SWORD_COLOR)
        self.item_sprites_color = self.item_sprites_color.at[Item_Sprites.TORCH.value, ...].set(self.consts.TORCH_COLOR)
        self.item_sprites_color = self.item_sprites_color.at[Item_Sprites.TORCH_FRAME_2.value, ...].set(self.consts.TORCH_COLOR)
        
        digit_none = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_none"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_0 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_0"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_1 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_1"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_2 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_2"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_3 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_3"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_4 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_4"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_5 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_5"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_6 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_6"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_7 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_7"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_8 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_8"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        digit_9 = loadFrameAddAlpha(fileName=os.path.join(self.sprite_path, self.consts.SPRITE__F_DICT["digit_9"])
                                                          ,add_alpha=True, add_black_as_transparent=True, transpose=True) 
        
        self.digit_sprite = jnp.stack([digit_none, digit_0, digit_1, digit_2, digit_3, digit_4, digit_5, digit_6, digit_7, digit_8, digit_9])



        
        
        
        
        
        #
        # All the infrastructure functions that utilize the Proto ROOM class.
        #
        self.PROTO_ROOM_LOADER: Callable[[int, jnp.ndarray], Room] = None
        # A loader that loads a Proto Room object from the persistence storage. 
        # This loader is save to be used in functions that are non-static with the ROOM ID
        
        self.WRITE_PROTO_ROOM_TO_PERSISTENCE: Callable[[int, Room, jnp.ndarray], jnp.ndarray] = None
        # A writer that writes the fields from the proto room to persistence. 
        # Again, save to be used in non-static functions
        
        
        
        
        
        # All wrapped render functions.
        self.RENDER_LAZER_WALLS_WRAPPED: Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_lazer_walls, 
                                                                            montezuma_state_type=MontezumaState)
        self.RENDER_ITEMS_ONTO_CANVAS_WRAPPED:Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_items_onto_canvas,
                                                                            montezuma_state_type=MontezumaState)
        self.RENDER_ROPES_ONTO_CANVAS_WRAPPED: Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_ropez,
                                                                            montezuma_state_type=MontezumaState)
        self.RENDER_DOORS_ONTO_CANVAS_WRAPPED: Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_doors_onto_canvas,
                                                                            montezuma_state_type=MontezumaState)
        self.RENDER_DROPOUT_FLOORS_ONTO_CANVAS_WRAPPED: Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_dropout_floors_onto_canvas,
                                                                            montezuma_state_type=MontezumaState)
        self.RENDER_SARLACC_PIT_ONTO_CANVAS_WRAPPED: Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_sarlacc_pit_onto_canvas,
                                                                            montezuma_state_type=MontezumaState)
        self.RENDER_SIDE_WALLS_ONTO_CANVAS_WRAPPED: Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_side_walls_onto_canvas,
                                                                            montezuma_state_type=MontezumaState)
        self.RENDER_LADDERS_ONTO_CANVAS_WRAPPED: Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_ladders_onto_canvas, 
                                                                                                                                      montezuma_state_type=MontezumaState)
        self.RENDER_ENEMIES_ONTO_CANVAS_WRAPPED: Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_enemies_onto_canva, 
                                                                                                                                      montezuma_state_type=MontezumaState)
        self.RENDER_CONVEYOR_BELTS_ONTO_CANVAS_WRAPPED: Callable[[MontezumaState], MontezumaState] = LAYOUT._wrap_lowered_function(lowered_function=self.render_conveyor_belts_onto_canvas,
                                                                            montezuma_state_type=MontezumaState)
        self.WRAPPED_RENDER_ROOM_SPRITE_ONTO_CANVAS = LAYOUT._wrap_lowered_function(lowered_function=self.render_room_sprite_onto_canvas, 
                                                                        montezuma_state_type=MontezumaState)
        
        self.WRAPPED_RENDER_INITIAL_BLANK_CANVAS = LAYOUT._wrap_lowered_function(lowered_function=self.render_initial_blank_canvas, 
                                                                        montezuma_state_type=MontezumaState)
        self.lazer_wall_background: jArray = self.generate_lazer_wall_background()
        MY_SPRITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites", "montezuma")
        SKULL_SPRITE_PATH = os.path.join(MY_SPRITE_PATH, "skull_cycle")
        # Load the skull animation here.
        skull_1 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_1.npy"), transpose=True, 
                                    add_alpha=False)
        skull_2 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_2.npy"), transpose=True, 
                                    add_alpha=False)
        skull_3 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_3.npy"), transpose=True, 
                                    add_alpha=False)
        skull_4 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_4.npy"), transpose=True, 
                                    add_alpha=False)
        skull_5 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_5.npy"), transpose=True, 
                                    add_alpha=False)
        skull_6 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_6.npy"), transpose=True, 
                                    add_alpha=False)
        skull_7 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_7.npy"), transpose=True, 
                                    add_alpha=False)
        skull_8 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_8.npy"), transpose=True, 
                                    add_alpha=False)
        skull_9 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_9.npy"), transpose=True, 
                                    add_alpha=False)
        skull_10 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_10.npy"), transpose=True, 
                                    add_alpha=False)
        skull_11 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_11.npy"), transpose=True, 
                                    add_alpha=False)
        skull_12 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_12.npy"), transpose=True, 
                                    add_alpha=False)
        skull_13 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_13.npy"), transpose=True, 
                                    add_alpha=False)
        skull_14 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_14.npy"), transpose=True, 
                                    add_alpha=False)
        skull_15 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_15.npy"), transpose=True, 
                                    add_alpha=False)
        skull_16 = loadFrameAddAlpha(os.path.join(SKULL_SPRITE_PATH, "skull_16.npy"), transpose=True, 
                                    add_alpha=False)
        skull_sprites = jnp.stack([skull_1, skull_2, skull_3, skull_4, skull_5, 
                                   skull_6, skull_7, skull_8, skull_9, 
                                   skull_10, skull_11, skull_12, skull_13, skull_14, 
                                   skull_15, skull_16], axis=0)
        skull_sprites = jnp.flip(skull_sprites, axis=0)
        self.skull_sprites: jArray = skull_sprites
        self.bounce_skull_sprite = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "bounce_skull.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        self.spider_sprite: jArray = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "spidder.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        snake_0 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "snake_0.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        snake_1 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "snake_1.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        self.snake_sprites: jArray = jnp.stack([snake_0, snake_1], axis=0)
        ladder_climb_0 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "ladder_climb1.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        ladder_climb_1 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "ladder_climb2.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        self.ladder_climb_sprites: jArray = jnp.stack([ladder_climb_0, ladder_climb_1], 
                                                      axis=0)
        self.ladder_climb_sprite_size: Tuple[int, int] = (7, 19)
        
        rope_climb_0 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "rope_climb_0.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        rope_climb_1 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "rope_climb_1.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        self.rope_climb_sprites: jArray = jnp.stack([rope_climb_0, rope_climb_1], axis=0)
        self.rope_climb_sprite_size: Tuple[int, int] = (7, 20) 
        
        walk_0 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "walking_0.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        walk_1 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "walking_1.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        self.walking_sprites: jArray = jnp.stack([walk_0, walk_1], axis=0)
        
        self.player_jump_frame: jArray = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "player_jump.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        self.player_jump_size: Tuple[int, int] = (8, 20)
        
        player_splosh_on_floor_0 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "player_splosh_0.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        player_splosh_on_floor_1 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "player_splosh_1.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        self.player_fall_damage_sprites: jArray = jnp.stack([player_splosh_on_floor_0, player_splosh_on_floor_1], axis=0)
        
        splutter_0 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "splutter_0.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        splutter_1 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "splutter_1.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True)
        self.enemy_splutter: jArray = jnp.stack([splutter_0, splutter_1], axis=0)
        sarlacc_death_0 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "sarlacc_death_0.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True) 
        sarlacc_death_1 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "sarlacc_death_1.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True) 
        sarlacc_death_2 = loadFrameAddAlpha(os.path.join(MY_SPRITE_PATH, "sarlacc_death_2.npy"), transpose=True, 
                                    add_alpha=True, add_black_as_transparent=True) 
        self.sarlacc_death: jArray = jnp.stack([sarlacc_death_0, sarlacc_death_1, sarlacc_death_2], axis=0)
            
    
    @partial(jax.jit, static_argnums=(0))
    def get_score_sprite(self, score: jnp.ndarray) -> jnp.ndarray:
        """Takes numerical representation of the current score and composes a score sprite from individual digit sprites

        Args:
            score (jnp.ndarray): _description_

        Returns:
            jnp.ndarray: _description_
        """
        # Set dimensions of final sprite
        # Until 10k support
        final_sprite: jnp.ndarray = jnp.zeros((6*self.consts.DIGIT_WIDTH + 5*self.consts.DIGIT_OFFSET, self.consts.DIGIT_HEIGHT, 4), np.uint8)
        k_100_sprite_index: jnp.ndarray = jnp.mod(jnp.floor_divide(score, jnp.array([100000])), 10) + 1
        k_10_sprite_index: jnp.ndarray = jnp.mod(jnp.floor_divide(score, jnp.array([10000])), 10) + 1
        thousands_sprite_index: jnp.ndarray = jnp.mod(jnp.floor_divide(score, jnp.array([1000])), 10) + 1
        hundreds_sprite_index: jnp.ndarray = jnp.mod(jnp.floor_divide(score, jnp.array([100])), 10) + 1
        tens_sprite_index: jnp.ndarray = jnp.mod(jnp.floor_divide(score, jnp.array([10])), 10) + 1
        
        # Ones sprite is always displayed
        ones_sprite_index: jnp.ndarray = jnp.add(jnp.mod(score, jnp.array([10])), 1)
        # Now remove leading zeros:
        k_100_sprite_index = jnp.multiply(k_100_sprite_index, jnp.astype(jnp.greater_equal(score, 100000), jnp.int32))
        k_10_sprite_index = jnp.multiply(k_10_sprite_index, jnp.astype(jnp.greater_equal(score, 10000), jnp.int32))
        thousands_sprite_index = jnp.multiply(thousands_sprite_index, jnp.astype(jnp.greater_equal(score, 1000), jnp.int32))
        hundreds_sprite_index = jnp.multiply(hundreds_sprite_index, jnp.astype(jnp.greater_equal(score, 100), jnp.int32))
        tens_sprite_index = jnp.multiply(tens_sprite_index, jnp.astype(jnp.greater_equal(score, 10), jnp.int32))
        
        final_sprite = final_sprite.at[0:self.consts.DIGIT_WIDTH, ...].set(jnp.squeeze(self.digit_sprite[k_100_sprite_index, ...]))
        final_sprite = final_sprite.at[self.consts.DIGIT_WIDTH + self.consts.DIGIT_OFFSET:2*self.consts.DIGIT_WIDTH + self.consts.DIGIT_OFFSET,...].set(jnp.squeeze(self.digit_sprite[k_10_sprite_index, ...]))
        final_sprite = final_sprite.at[self.consts.DIGIT_WIDTH*2 + self.consts.DIGIT_OFFSET*2:self.consts.DIGIT_WIDTH*3 + self.consts.DIGIT_OFFSET*2,...].set(jnp.squeeze(self.digit_sprite[thousands_sprite_index, ...]))
        final_sprite = final_sprite.at[self.consts.DIGIT_WIDTH*3 + self.consts.DIGIT_OFFSET*3:self.consts.DIGIT_WIDTH*4 + self.consts.DIGIT_OFFSET*3,...].set(jnp.squeeze(self.digit_sprite[hundreds_sprite_index, ...]))
        final_sprite = final_sprite.at[self.consts.DIGIT_WIDTH*4 + self.consts.DIGIT_OFFSET*4:self.consts.DIGIT_WIDTH*5 + self.consts.DIGIT_OFFSET*4,...].set(jnp.squeeze(self.digit_sprite[tens_sprite_index, ...]))
        final_sprite = final_sprite.at[self.consts.DIGIT_WIDTH*5 + self.consts.DIGIT_OFFSET*5:self.consts.DIGIT_WIDTH*6 + self.consts.DIGIT_OFFSET*5,...].set(jnp.squeeze(self.digit_sprite[ones_sprite_index, ...]))
        
        return final_sprite
    
    def render_player_sarlacc_death(self, canvas: jArray, state: MontezumaState):
        # Uses the freeze counter to render the death animation when falling into a pit.
        enemy_death_sprite: jArray = jnp.mod(jnp.floor_divide(state.freeze_remaining, self.consts.ANIMATION_CYCLE_DURATION), 2)
        room_state: NamedTuple = state.room_state
        death_sprite = jax.lax.cond(jnp.equal(state.freeze_remaining[0],50),
                                    lambda x: x[0],
                                    lambda x: x[enemy_death_sprite[0]+1],
                                    operand=self.sarlacc_death)
        player_sprite = jnp.transpose(death_sprite, (1, 0, 2))
        canvas = jr.render_at(canvas, state.player_position[0], state.player_position[1]+room_state.vertical_offset[0] - 6,player_sprite)
        return canvas
 
    def render_initial_blank_canvas(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple],  tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # Used for rendering dark-rooms
        canvas = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT, 3), dtype=jnp.uint8)
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
        return state, room_state   

    def render_items_onto_canvas(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # Renders all items onto the canvas
        if RoomTags.ITEMS in tags:
            items: RoomTags.ITEMS.value = SANTAH.extract_tag_from_rooms[RoomTags.ITEMS](room_state)       
            def single_sprite_onto_canvas(canvas, item:RoomTags.ITEMS.value):
                des_item: Item = SANTAH.full_deserializations[Item](item.items)
                # Torch has 2 different sprites for flickering.
                index = jax.lax.cond(jnp.logical_and(jnp.equal(des_item.sprite[0], 
                                                               Item_Sprites.TORCH.value),
                                                     jnp.less(jnp.mod(montezuma_state.frame_count,self.consts.ANIMATION_CYCLE_DURATION*2),
                                                                 self.consts.ANIMATION_CYCLE_DURATION))[0],
                                 lambda _: Item_Sprites.TORCH_FRAME_2.value,
                                 lambda _: des_item.sprite[0],
                                 operand=None)
                sprite = jax.lax.dynamic_index_in_dim(operand=self.item_sprites,
                                                                    index=index,
                                                                    axis=0,
                                                                    keepdims=False)
                
                # If player is currently in bonus room: 
                # The gems flicker.
                sprite_color =  jax.lax.cond(RoomTags.BONUSROOM in tags,
                                             lambda x: x * jnp.mod(montezuma_state.frame_count[0],2),
                                             lambda x: x,
                                             operand=self.item_sprites_color)
                colorized_sprite = colorize_sprite(sprite,jax.lax.dynamic_index_in_dim(operand=sprite_color,
                                                                    index=index,
                                                                    axis=0,
                                                                    keepdims=False)) 
                
                # Items are only rendered if they are currently on field.
                colorized_sprite = colorized_sprite * des_item.on_field
                canvas = jr.render_at(canvas, des_item.y + room_state.vertical_offset, des_item.x, colorized_sprite)
                return canvas, 0
            canvas, _ = jax.lax.scan(single_sprite_onto_canvas,montezuma_state.canvas, items)
            
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return state, room_state
        else:
            return montezuma_state, room_state
        
    def render_room_sprite_onto_canvas(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple],  tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        #
        # renders the variable-size room sprite onto the canvas.
        #
        room_sprite: jnp.ndarray = getattr(room_state, FieldsThatAreSharedByAllRoomsButHaveDifferentShape.sprite.value)
        vert_offset: jnp.ndarray = room_state.vertical_offset
        canvas = montezuma_state.canvas
        canvas = jax.lax.dynamic_update_slice(operand=canvas, 
                                     update=room_sprite, 
                                     start_indices=(0, vert_offset[0], 0))
        state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
        return state, room_state

        
    # Render functions for the enemies.
    #
    #
    #
    #
    def _render_bounce_skull_onto_canvas(self, canvas: jArray, enemy: Enemy, room_offset: jArray) -> jArray:
        # The Bounce Skull can be half alive, 
        # So we split the sprite in two & render the halves dependant on whether they are still alive.
        b_skull_width: int = B_SCULL_BEHAVIOR.sprite_size[0]
        b_skull_height: int = B_SCULL_BEHAVIOR.sprite_size[1]
        single_b_skull_width: int = b_skull_width//2 - self.consts.DOUBLE_SKULL_MIDDLE_DEMARKATION_LINE_OFFSET
        
        right_b_skull_x_offset: int = b_skull_width//2 + self.consts.DOUBLE_SKULL_MIDDLE_DEMARKATION_LINE_OFFSET + 1
        left_skull: jArray = self.bounce_skull_sprite[0:single_b_skull_width, :]
        right_skull: jArray = self.bounce_skull_sprite[right_b_skull_x_offset:, :]
        
        is_left_skull_alive: jArray = jnp.logical_or(
            jnp.equal(enemy.optional_utility_field, BounceSkullAliveState.RIGHT_DEAD.value), 
            jnp.equal(enemy.optional_utility_field, BounceSkullAliveState.FULLY_ALIVE.value)
        )
        is_right_skull_alive: jArray = jnp.logical_or(
            jnp.equal(enemy.optional_utility_field, BounceSkullAliveState.LEFT_DEAD.value), 
            jnp.equal(enemy.optional_utility_field, BounceSkullAliveState.FULLY_ALIVE.value)
        )
        
        # Multiply Bounce Skulls with their respecitve alive value
        # to disable rendering if they are already dead.
        left_skull = jnp.multiply(left_skull, is_left_skull_alive)
        right_skull = jnp.multiply(right_skull, is_right_skull_alive)
        left_b_skull_x = enemy.pos_x + B_SCULL_BEHAVIOR.sprite_offset[0]
        left_b_skull_y = enemy.pos_y + B_SCULL_BEHAVIOR.sprite_offset[1] + room_offset
        
        canvas = jr.render_at(canvas, left_b_skull_y, left_b_skull_x, left_skull)
        
        right_b_skull_x = enemy.pos_x + B_SCULL_BEHAVIOR.sprite_offset[0] + right_b_skull_x_offset
        right_b_skull_y = enemy.pos_y + B_SCULL_BEHAVIOR.sprite_offset[1] + room_offset
        
        canvas = jr.render_at(canvas, right_b_skull_y, right_b_skull_x, right_skull)
        
        return canvas 
    def _render_rolling_skull_onto_canvas(self, canvas: jArray, enemy: Enemy, room_offset: jArray) -> jArray:
        # For rolling skulls, the sprite at the appropriate index is rendered
        sprite_x = enemy.pos_x + ROLLING_SKULL_BEHAVIOR.sprite_offset[0]
        sprite_y = enemy.pos_y + ROLLING_SKULL_BEHAVIOR.sprite_offset[1] + room_offset
        enemy_sprite: jArray = jax.lax.dynamic_index_in_dim(self.skull_sprites, index=enemy.sprite_index[0], 
                                                            axis=0, keepdims=False)
        canvas = jr.render_at(canvas, sprite_y, sprite_x, enemy_sprite)
        return canvas 
    
    
    def _render_spider_onto_canvas(self, canvas: jArray, enemy: Enemy, room_offset: jArray) -> jArray:
        # For spiders, the sprite at the appropriate index is rendered
        sprite_x = enemy.pos_x + SPIDER_BEHAVIOR.sprite_offset[0]
        sprite_y = enemy.pos_y + SPIDER_BEHAVIOR.sprite_offset[1] + room_offset
        flip = jnp.mod(enemy.sprite_index, 2)
        enemy_sprite = jnp.multiply(flip[0], self.spider_sprite) + jnp.multiply(1 - flip[0], jnp.flip(self.spider_sprite, axis=0))
        
        canvas = jr.render_at(canvas, sprite_y, sprite_x, enemy_sprite)
        return canvas
    
    
    def _render_snake_onto_canvas(self, canvas: jArray, enemy: Enemy, room_offset: jArray) -> jArray:
        # For snakes, the sprite at the appropriate index is rendered.
        sprite_x = enemy.pos_x + SNAKE_BEHAVIOR.sprite_offset[0]
        sprite_y = enemy.pos_y + SNAKE_BEHAVIOR.sprite_offset[1] + room_offset
        sprite_dex = jnp.floor_divide(enemy.optional_movement_counter, SNAKE_BEHAVIOR.frame_length)
        enemy_sprite = jnp.multiply(sprite_dex[0], self.snake_sprites[0, ...]) + jnp.multiply(1 - sprite_dex[0], self.snake_sprites[1, ...])
        
        canvas = jr.render_at(canvas, sprite_y, sprite_x, enemy_sprite)
        return canvas
    
    def _render_single_enemy_onto_canvas(self, canvas_and_offset: Tuple[jArray, jArray], single_enemy: jArray) -> jArray:
        single_enemy: Enemy = SANTAH.full_deserializations[Enemy](single_enemy)
        canvas: jArray = None
        vert_offset: jArray = None
        canvas, vert_offset = canvas_and_offset
        # At this point handle, that dead enemies are not rendered anymore.
        render_index: jArray = jnp.multiply(single_enemy.alive, single_enemy.enemy_type + 1)
        # At this point the indexes of the enemies are hardcoded, this might make adding more enemies in the future tedious.
        canvas = jax.lax.switch(render_index[0], [lambda x, y, z : x, self._render_snake_onto_canvas, self._render_rolling_skull_onto_canvas, 
                self._render_bounce_skull_onto_canvas, self._render_spider_onto_canvas], canvas, single_enemy, vert_offset)
        return (canvas, vert_offset), 0
    
        
    def render_enemies_onto_canva(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        if RoomTags.ENEMIES in tags:
            
            enemy_fields: RoomTags.ENEMIES.value = SANTAH.extract_tag_from_rooms[RoomTags.ENEMIES](room_state)
            tp, _ = jax.lax.scan(self._render_single_enemy_onto_canvas, (montezuma_state.canvas, room_state.vertical_offset), enemy_fields.enemies)
            canvas, _ = tp
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return state, room_state
        else:
            return montezuma_state, room_state
        
    
    def render_side_walls_onto_canvas(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        if RoomTags.SIDEWALLS in tags:
            side_walls: RoomTags.SIDEWALLS.value = SANTAH.extract_tag_from_rooms[RoomTags.SIDEWALLS](room_state)
            canvas = montezuma_state.canvas
            canvas = jr.render_at(canvas,room_state.vertical_offset[0],0,side_walls.side_walls_render_map)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return state, room_state
        else:
            return montezuma_state, room_state
        
    def render_dropout_floors_onto_canvas(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        if RoomTags.DROPOUTFLOORS in tags:
            dropout_floor: RoomTags.DROPOUTFLOORS.value = SANTAH.extract_tag_from_rooms[RoomTags.DROPOUTFLOORS](room_state)
            canvas = montezuma_state.canvas
            # Compute the correct animation frame using the frame_count
            animation_index = jnp.less(jnp.mod(montezuma_state.frame_count,self.consts.ANIMATION_CYCLE_DURATION*2),self.consts.ANIMATION_CYCLE_DURATION).astype(jnp.int32)
            
            # Dropout floors are only rendered if they are in the "alive"-phase of their cycle.
            render_cond = jnp.less_equal(jnp.mod(montezuma_state.frame_count,dropout_floor.on_time_dropoutfloor[0]+dropout_floor.off_time_dropoutfloor[0]),dropout_floor.on_time_dropoutfloor[0])

            canvas = jax.lax.cond(render_cond[0],
                         lambda x: jr.render_at(x,0,0,dropout_floor.dropout_floor_render_maps[animation_index[0]]),
                         lambda x: x,
                         operand= canvas)
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return state, room_state
        else:
            return montezuma_state, room_state
        
    def render_conveyor_belts_onto_canvas(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        if RoomTags.CONVEYORBELTS in tags:
            conveyor_belts: RoomTags.CONVEYORBELTS.value = SANTAH.extract_tag_from_rooms[RoomTags.CONVEYORBELTS](room_state)
            canvas = montezuma_state.canvas
            animation_index = jnp.less(jnp.mod(montezuma_state.frame_count,self.consts.ANIMATION_CYCLE_DURATION*2),self.consts.ANIMATION_CYCLE_DURATION).astype(jnp.int32)
            canvas = jr.render_at(canvas,0,0,conveyor_belts.global_conveyor_render_map[animation_index[0]])
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return state, room_state
        else:
            return montezuma_state, room_state
        
    def render_sarlacc_pit_onto_canvas(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        if RoomTags.PIT in tags:   
            pit: RoomTags.PIT.value = SANTAH.extract_tag_from_rooms[RoomTags.PIT](room_state) 
            canvas = montezuma_state.canvas 
            animation_index = (jnp.mod(montezuma_state.frame_count,self.consts.ANIMATION_CYCLE_DURATION*pit.pit_render_maps.shape[0])/self.consts.ANIMATION_CYCLE_DURATION).astype(jnp.int32)
            
            canvas = jr.render_at(canvas,room_state.vertical_offset, 0, pit.pit_render_maps[animation_index[0]])
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return state, room_state
        else:
            return montezuma_state, room_state
        
        
        
    def render_doors_onto_canvas(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # 
        # Here, each door is rendered individually.
        # This is necessary, as the number of active doors changes throughout the game.
        #
        if RoomTags.DOORS in tags:
            doors: RoomTags.DOORS.value = SANTAH.extract_tag_from_rooms[RoomTags.DOORS](room_state)
            def single_door_onto_canvas(canvas, door:RoomTags.DOORS.value):
                des_door: Door = SANTAH.full_deserializations[Door](door)
                door_color = jax.lax.dynamic_index_in_dim(operand=self.consts.OBSTACLE_COLORS,
                                                          index=des_door.color[0],
                                                          axis=0,
                                                          keepdims=False)
                colorized_door_sprite: jnp.ndarray = colorize_sprite(self.door_sprite,door_color)
                canvas = jax.lax.cond(des_door.on_field[0],
                                      lambda x: jr.render_at(x, des_door.y + room_state.vertical_offset,des_door.x,colorized_door_sprite),
                                      lambda x: x,
                                      operand=canvas)
                return canvas, 0
            canvas, _ = jax.lax.scan(single_door_onto_canvas,montezuma_state.canvas, doors.doors)
            
            state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return state, room_state
        else:
            return montezuma_state, room_state
    
    def generate_lazer_wall_background(self) -> None:
        """Generates a static striped background over which the Lazer wall mask is slid to generate the moving illusion
        """
        # Make the canvas used larger than the maximum possible room size
        # to allow for lazer barriers of arbitrary height to be rendered correctly.
        lazer_wall_background: jArray = jnp.zeros(shape=(self.consts.WIDTH, self.consts.HEIGHT+self.consts.LAZER_BARRIER_SCROLL_STEP*self.consts.LAZER_BARRIER_ANIMATION_CYCLE, 4))
        _height = self.consts.HEIGHT+self.consts.LAZER_BARRIER_SCROLL_STEP*self.consts.LAZER_BARRIER_ANIMATION_CYCLE
        cnt = 0
        while cnt < _height:
            row: jArray = lazer_wall_background[:, cnt, :]
            # Check if at this height a stripe should be rendered.
            if cnt % self.consts.LAZER_BARRIER_STRIPE_DISTANCE == 0:
                row = row + jnp.reshape(self.consts.LASER_BARRIER_COLOR, shape=(1, 4))
                lazer_wall_background = lazer_wall_background.at[:, cnt, :].set(row)
            cnt += 1
        return lazer_wall_background
            
    def render_ropez(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]):
        if RoomTags.ROPES in tags:
            # Render the global rope-render mask onto the canvas.
            rope_attributes: RoomTags.ROPES.value = SANTAH.extract_tag_from_rooms[RoomTags.ROPES](room_state)
            rope_render: jArray = rope_attributes.rope_render_map
            
            rope_render = jnp.transpose(rope_render, axes=(1, 0, 2))
            t_canvas = jnp.transpose(montezuma_state.canvas, axes=(1, 0, 2))
            
            t_canvas = jr.render_at(t_canvas, 0, 0, rope_render)
            canvas = jnp.transpose(t_canvas, axes=(1, 0, 2))
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return montezuma_state, room_state
        else:
            return montezuma_state, room_state
        
        
    def render_ladders_onto_canvas(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]):
        if RoomTags.LADDERS in tags:
            # Render the global ladder mask onto the canvas.
            ladder_attributes: RoomTags.LADDERS.value = SANTAH.extract_tag_from_rooms[RoomTags.LADDERS](room_state)
            render_sprite: jArray = ladder_attributes.ladders_sprite
            canvas: jArray = montezuma_state.canvas
            
            render_sprite = jnp.transpose(render_sprite, axes=(1, 0, 2))
            canvas = jnp.transpose(canvas, axes=(1, 0, 2))
            
            canvas = jr.render_at(canvas, 0, 0, render_sprite)
            canvas = jnp.transpose(canvas, axes=(1, 0, 2))
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return montezuma_state, room_state
        else:
            return montezuma_state, room_state
        
    def render_lazer_walls(self, montezuma_state: MontezumaState, room_state: Room, room_type: Type[NamedTuple], tags: Tuple[RoomTags]) -> Tuple[MontezumaState, Room]:
        # Render the global lazer wall map onto the canvas
        if RoomTags.LAZER_BARRIER in tags:
            
            barrier_attributes: RoomTags.LAZER_BARRIER.value = SANTAH.extract_tag_from_rooms[RoomTags.LAZER_BARRIER](room_state)
                        
            # Extract the global lazer-barrier hitmap.
            barrier_hit_map: jnp.ndarray = barrier_attributes.global_barrier_map
            
            # Now check if the barriers are supposed to be rendered this frame:
            barrier_info_glob: GlobalLazerBarrierInfo = SANTAH.full_deserializations[GlobalLazerBarrierInfo](barrier_attributes.global_barrier_info[0, ...])
            active_start: jnp.ndarray = barrier_info_glob.cycle_offset
            active_stop: jnp.ndarray = barrier_info_glob.cycle_offset + barrier_info_glob.cycle_active_frames
            is_barrier_active: jnp.ndarray = jnp.logical_and(jnp.greater_equal(barrier_info_glob.cycle_index, active_start), 
                                                      jnp.less(barrier_info_glob.cycle_index, active_stop))
            
            barrier_hit_map  = jnp.reshape(barrier_hit_map, (self.consts.WIDTH, self.consts.HEIGHT, 1))
            barrier_hit_map = jnp.repeat(barrier_hit_map, repeats=4, axis=-1)
            barrier_hit_map = jnp.astype(jnp.multiply(barrier_hit_map, is_barrier_active), jnp.int32)
            
            # Now get the appropriate w.r.t. the precomputed stripe background
            # to achieve the scrolling effect.
            sprite_render_offset: jnp.ndarray = jnp.mod(barrier_info_glob.cycle_index, jnp.array([self.consts.LAZER_BARRIER_ANIMATION_CYCLE], jnp.int32))
            
            # Retreive the appropriately sized chunk of the striped background
            striped_slice: jArray = jax.lax.dynamic_slice_in_dim(operand=self.lazer_wall_background, 
                                         start_index=sprite_render_offset[0], slice_size=self.consts.HEIGHT, axis=1)
            # Multiply the lazer-barrier hitmap onto the canvas, so that stripes are only rendered where the lazer barriers are.
            masked_barrier_sprite: jnp.ndarray = jnp.multiply(striped_slice, barrier_hit_map)
            masked_barrier_sprite = jnp.transpose(masked_barrier_sprite, axes=(1, 0, 2))
            t_canvas = jnp.transpose(montezuma_state.canvas, axes=(1, 0, 2))
            
            t_canvas = jr.render_at(t_canvas, 0, 0, masked_barrier_sprite)
            canvas = jnp.transpose(t_canvas, axes=(1, 0, 2))
            montezuma_state = SANTAH.attribute_setters[MontezumaState][MontezumaStateFields.canvas.value](montezuma_state, canvas)
            return montezuma_state, room_state
        else:
            return montezuma_state, room_state
       
    #
    # All the functions used for rendering the player.
    #
    #
    #
    def render_player_standing(self, canvas: jArray, state: MontezumaState):
        # This function assumes that we have already checked that the player is standing.
        room_state: NamedTuple = state.room_state
        player_sprite = jnp.transpose(self.player_sprite, (1, 0, 2))
        flipped_player_sprite = jax.lax.cond(jnp.equal(state.last_horizontal_orientation[0],Horizontal_Direction.LEFT.value),
                    lambda s: jnp.flip(s, axis=1),
                    lambda s: s,
                    player_sprite
        )
        canvas = jr.render_at(canvas, state.player_position[0], state.player_position[1]+room_state.vertical_offset[0],flipped_player_sprite)
        return canvas 
    
    
    def render_player_ladder_climbing(self, canvas: jArray, state: MontezumaState):
        # This function assumes that we have already checked, that the player is currently climbing a ladder.
        room_state: NamedTuple = state.room_state
        ladder_climb_spirte: jArray = jax.lax.dynamic_index_in_dim(self.ladder_climb_sprites, index=state.ladder_climb_frame[0], 
                                                    axis=0, keepdims=False)
        player_sprite = jnp.transpose(ladder_climb_spirte, (1, 0, 2))
        x_pos = state.player_position[0] + (self.consts.PLAYER_WIDTH - self.ladder_climb_sprite_size[0])
        canvas = jr.render_at(canvas, x_pos, state.player_position[1]+room_state.vertical_offset[0],player_sprite)
        return canvas
    
    
    def render_player_rope_climbing(self, canvas: jArray, state: MontezumaState):
        # This function assumes that we have already checked, that the player is currently climbing a rope.
        room_state: NamedTuple = state.room_state
        rope_climb_sprite: jArray = jax.lax.dynamic_index_in_dim(self.rope_climb_sprites, index=state.rope_climb_frame[0], 
                                                    axis=0, keepdims=False)
        player_sprite = jnp.transpose(rope_climb_sprite, (1, 0, 2))
        x_pos = state.player_position[0] + (self.consts.PLAYER_WIDTH - self.rope_climb_sprite_size[0])
        canvas = jr.render_at(canvas, x_pos, state.player_position[1]+room_state.vertical_offset[0],player_sprite)
        return canvas
    
    
    def render_player_walking(self, canvas: jArray, state: MontezumaState):
        # This function assumes that we have already checked, that the player is currently walking.
        room_state: NamedTuple = state.room_state
        walking_sprite: jArray = jax.lax.dynamic_index_in_dim(self.walking_sprites, index=state.walk_frame[0], 
                                                    axis=0, keepdims=False)
        player_sprite = jnp.transpose(walking_sprite, (1, 0, 2))
        flipped_player_sprite = jax.lax.cond(jnp.equal(state.last_horizontal_orientation[0],Horizontal_Direction.RIGHT.value),
                    lambda s: jnp.flip(s, axis=1),
                    lambda s: s,
                    player_sprite
        )
        x_pos = state.player_position[0]
        canvas = jr.render_at(canvas, x_pos, state.player_position[1]+room_state.vertical_offset[0],flipped_player_sprite)
        return canvas
    
    
    def render_player_jumping(self, canvas: jArray, state: MontezumaState):
        # The jump animation consists only of a single frame
        room_state: NamedTuple = state.room_state
        player_sprite = jnp.transpose(self.player_jump_frame, (1, 0, 2))
        flipped_player_sprite = jax.lax.cond(jnp.equal(state.last_horizontal_orientation[0],Horizontal_Direction.RIGHT.value),
                    lambda s: jnp.flip(s, axis=1),
                    lambda s: s,
                    player_sprite
        )
        canvas = jr.render_at(canvas, state.player_position[0], state.player_position[1]+room_state.vertical_offset[0],flipped_player_sprite)
        return canvas 
    
    
    
    #
    #
    #
    # All the death animations:
    # The Freeze type is used to differentiate which animation should be rendered
    # And the freeze_counter is used to index the correct sprites
    #
    #
    #
    
    
    def render_player_dying_from_fall_damage(self, canvas: jArray, state: MontezumaState):
        fall_death_sprite_index: jArray = jnp.mod(jnp.floor_divide(state.freeze_remaining, self.consts.DEATH_FALL_WIGGLE_FRAME_LENGTH), 2)
        room_state: NamedTuple = state.room_state
        death_sprite: jArray = jax.lax.dynamic_index_in_dim(self.player_fall_damage_sprites, index=fall_death_sprite_index[0], 
                                                    axis=0, keepdims=False)
        player_sprite = jnp.transpose(death_sprite, (1, 0, 2))
        canvas = jr.render_at(canvas, state.player_position[0], state.player_position[1]+room_state.vertical_offset[0]+1+state.render_offset_for_fall_dmg[0],player_sprite)
        return canvas
    
    
    def render_player_dying_from_enemy(self, canvas: jArray, state: MontezumaState):
        enemy_death_sprite: jArray = jnp.mod(jnp.floor_divide(state.freeze_remaining, self.consts.ENEMY_SPLUTTER_FRAME_LENGTH), 2)
        room_state: NamedTuple = state.room_state
        death_sprite: jArray = jax.lax.dynamic_index_in_dim(self.enemy_splutter, index=enemy_death_sprite[0], 
                                                    axis=0, keepdims=False)
        player_sprite = jnp.transpose(death_sprite, (1, 0, 2))
        canvas = jr.render_at(canvas, state.player_position[0], state.player_position[1]+room_state.vertical_offset[0] + 1,player_sprite)
        return canvas
    
    def render_player_lazer_barrier_splutter(self, canvas: jArray, state: MontezumaState):
        enemy_death_sprite: jArray = jnp.mod(jnp.floor_divide(state.freeze_remaining, self.consts.LAZER_BARRIER_SPLUTTER_FRAME_LENGTH), 2)
        room_state: NamedTuple = state.room_state
        death_sprite: jArray = jax.lax.dynamic_index_in_dim(self.enemy_splutter, index=enemy_death_sprite[0], 
                                                    axis=0, keepdims=False)
        player_sprite = jnp.transpose(death_sprite, (1, 0, 2))
        canvas = jr.render_at(canvas, state.player_position[0], state.player_position[1]+room_state.vertical_offset[0] + 1,player_sprite)
        return canvas
    
    def render_player(self, canvas: jArray, state: MontezumaState):
        # Select which of the rendering functions to call:
        #   - 0: Render the player standing sprite
        #   - 1: Render the rope sprite for the player
        #   - 2: Render the ladder sprite for the player.
        #   - 3: ...
        
        is_on_floor: jArray = jnp.logical_and(jnp.logical_not(state.is_climbing), jnp.logical_and(jnp.logical_not(state.is_jumping), 
                                                                                    jnp.logical_not(state.is_falling)))
        is_actually_walking: jArray = jnp.not_equal(state.horizontal_direction, Horizontal_Direction.NO_DIR.value)
        is_walking: jArray = jnp.logical_and(is_on_floor, is_actually_walking)
        is_death_wiggling: jArray =  jnp.logical_and(state.frozen, jnp.equal(state.freeze_type, FreezeType.FALL_DEATH.value))
        is_enemy_spluttering: jArray =  jnp.logical_and(state.frozen, jnp.equal(state.freeze_type, FreezeType.KILLED_BY_MONSTER.value))
        is_lazer_spluttering: jArray =  jnp.logical_and(state.frozen, jnp.equal(state.freeze_type, FreezeType.LAZER_BARRIER_DEATH.value))
        is_eaten_by_sarlacc: jArray =  jnp.logical_and(state.frozen, jnp.equal(state.freeze_type, FreezeType.SARLACC_PIT_DEATH.value))
        render_options: jArray = jnp.array([0, 1*state.is_on_rope[0], 2*state.is_laddering[0], 3*is_walking[0], 
                                            4*state.is_jumping[0], 5*is_death_wiggling[0], 6*is_enemy_spluttering[0], 
                                            7*is_lazer_spluttering[0], 8*is_eaten_by_sarlacc[0]], jnp.int32)


        render_choice: jArray = jnp.max(render_options, keepdims=False)
        canvas = jax.lax.switch(render_choice, 
                       [self.render_player_standing, 
                        self.render_player_rope_climbing, 
                        self.render_player_ladder_climbing, 
                        self.render_player_walking, 
                        self.render_player_jumping, 
                        self.render_player_dying_from_fall_damage, 
                        self.render_player_dying_from_enemy, 
                        self.render_player_lazer_barrier_splutter, 
                        self.render_player_sarlacc_death], 
                       canvas, state)
        return canvas
        
    @partial(jax.jit, static_argnums=(0))     
    def render(self, state: MontezumaState) -> Tuple[jnp.ndarray]:
        """Jitted rendering function. receives the alien state, and returns a rendered frame

        Args:
            state (AlienState): _description_

        Returns:
            jnp.ndarray: Returns only the RGB channels, no alpha
            """
        
        


        def render_score(canvas, state: MontezumaState):
            digit_sprite:jnp.ndarray = self.get_score_sprite(score=state.score)
            digit_sprite = jnp.transpose(digit_sprite,(1,0,2))
            canvas = jr.render_at(canvas, self.consts.SCORE_X, self.consts.SCORE_Y, digit_sprite)
            return canvas


        def render_lifes(canvas, state: MontezumaState):
            def render_loop_lifes(i, canvas):
                x = self.consts.ITEMBAR_LIFES_STARTING_X + i * (self.life_sprite.shape[0] + 1)
                y = self.consts.LIFES_STARTING_Y
                sprite = jnp.transpose(self.life_sprite,(1, 0, 2))
                return jr.render_at(canvas, x, y, sprite)
            
            canvas = jax.lax.fori_loop(0, state.lifes[0], render_loop_lifes, canvas)
            return canvas
        
        def render_itembar(canvas, state: MontezumaState):
            def itembar_loop(i,canvas_and_offset):
                canvas, offset = canvas_and_offset
                def render_loop_items(j, canvas_and_offset):
                    canvas, offset = canvas_and_offset                 
                    x = self.consts.ITEMBAR_LIFES_STARTING_X + offset*8
                    y = self.consts.ITEMBAR_STARTING_Y
                    item_sprite = jnp.transpose(jax.lax.dynamic_index_in_dim(operand=self.item_sprites,
                                                                    index=i,
                                                                    axis=0,
                                                                    keepdims=False),axes=(1, 0, 2))
                    colorized_item_sprite = colorize_sprite(item_sprite,self.consts.ITEMBAR_COLOR)
                    return (jr.render_at(canvas, x, y, colorized_item_sprite) , offset + 1)
                return jax.lax.fori_loop(0, state.itembar_items[i], render_loop_items, (canvas, offset))
            canvas, _ = jax.lax.fori_loop(1,state.itembar_items.shape[0], itembar_loop, (canvas,0))
            return canvas
        
        def render_everything_affected_by_darkness(state:MontezumaState):
            state = self.WRAPPED_RENDER_ROOM_SPRITE_ONTO_CANVAS(state)
            state = self.RENDER_LAZER_WALLS_WRAPPED(state)
            state = self.RENDER_ITEMS_ONTO_CANVAS_WRAPPED(state)
            state = self.RENDER_DOORS_ONTO_CANVAS_WRAPPED(state)
            state = self.RENDER_CONVEYOR_BELTS_ONTO_CANVAS_WRAPPED(state)
            state = self.RENDER_SIDE_WALLS_ONTO_CANVAS_WRAPPED(state)
            state = self.RENDER_ROPES_ONTO_CANVAS_WRAPPED(state)
            state = self.RENDER_LADDERS_ONTO_CANVAS_WRAPPED(state)
            return state

        state = self.WRAPPED_RENDER_INITIAL_BLANK_CANVAS(state)
        state = jax.lax.cond(state.darkness[0],
                             lambda x : x,
                             render_everything_affected_by_darkness,
                             state)
        state = self.RENDER_DROPOUT_FLOORS_ONTO_CANVAS_WRAPPED(state)
        state = self.RENDER_SARLACC_PIT_ONTO_CANVAS_WRAPPED(state)
        state = self.RENDER_ENEMIES_ONTO_CANVAS_WRAPPED(state)
        
        canvas: jArray = state.canvas
        canvas = jnp.transpose(canvas, (1, 0, 2))
        
        canvas = render_score(canvas, state)
        canvas = render_lifes(canvas, state)
        canvas = render_itembar(canvas, state)
        canvas = self.render_player(canvas, state)
        
        return canvas[..., 0:3]
        
