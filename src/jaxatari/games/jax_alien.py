# We are Implementing the Game Alien
# We are a Group of 4: 
# 
# 
# Dennis Breder	
# Christos Toutoulas	
# David Grguric
# Niklas Ihm
#
#
import array
import os
from functools import partial
from typing import NamedTuple, Tuple, Any, Callable
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from jaxatari.renderers import JAXGameRenderer

from gymnax.environments import spaces

from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment
from jaxatari.environment import JAXAtariAction

from typing import NamedTuple, Tuple, List
import jax.lax as lax
import jax
from typing import Dict, List

import numpy as np
from jax import random, Array

# Constants for game environment
SEED = 1701
RENDER_SCALE_FACTOR = 5
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
WIDTH = 152 # Width of the playing field in game state
HEIGHT = 188 # height of the playing field in game state
PLAYER_VELOCITY = 1
ENEMY_VELOCITY = 1
PLAYER_WIDTH = 8
PLAYER_HEIGHT = 13
# Action constants
# PLEASE don't change the action constants. At current state, this will break movement horribly.


# player position
PLAYER_X = 67
PLAYER_Y = 56

# Position at which the numbers are rendered
SCORE_X = 30
SCORE_Y = 170
DIGIT_OFFSET = 3 # Offset between numbers
DIGIT_WIDTH = 6
DIGIT_HEIGHT = 7

# Position at which the life counter is rendered
LIFE_X = 13
LIFE_Y = 187
LIFE_OFFSET_X = 2 # Offset between life sprites
LIFE_WIDTH = 5

# Enemy_player_collision_offset
ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW = 4
ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH = 6

# Spawn position of enemies
ENEMY_SPAWN_X = 67
ENEMY_SPAWN_Y = 147 + 5
# Enemy start position between walls
ENEMY_START_Y = 128
FRIGHTENED_DURATION = 100

# Pink enemy constants
ENEMY_COLOR_1 = [236, 140, 224] # Pink
SCATTER_DURATION_1 = 50
CHASE_DURATION_1 = 200
MODECHANGE_PROBABILITY_1 = 0.5
SCATTER_POINT_X_1 = 0
SCATTER_POINT_Y_1 = 0

# Yellow enemy constants
ENEMY_COLOR_2 = [252, 252, 84] # Yellow
SCATTER_DURATION_2 = 50
CHASE_DURATION_2 = 100
MODECHANGE_PROBABILITY_2 = 0.6
SCATTER_POINT_X_2 = 0
SCATTER_POINT_Y_2 = 180

# Green enemy constants
ENEMY_COLOR_3 = [132, 252, 212] # Green
SCATTER_DURATION_3 = 50
CHASE_DURATION_3 = 100
MODECHANGE_PROBABILITY_3 = 0.7
SCATTER_POINT_X_3 = 180
SCATTER_POINT_Y_3 = 180

# Colors
ENEMY_COLORS = jnp.array([ENEMY_COLOR_1, ENEMY_COLOR_2, ENEMY_COLOR_3], dtype=np.int32)
FLAMETHROWER_COLOR = jnp.array([252, 144, 144], dtype=np.int32) # Redish
DEFAULT_PLAYER_COLOR = jnp.array([132, 144, 252], dtype=np.int32)

#Multiplyer for egg score
EGG_SCORE_MULTIPLYER = 10
SPRITE__F_DICT = {
             "BASIC_MAP": "map_sprite.npy",
             "PLAYER_1":"player1.npy",
             "PLAYER_2":"player2.npy",
             "PLAYER_3":"player3.npy",
             "FLAME":"flame_sprite.npy",
             "TELEPORT_1": "teleport1.npy",
             "TELEPORT_2": "teleport2.npy",
             "TELEPORT_3": "teleport3.npy",
             "TELEPORT_4": "teleport4.npy",
             "ENEMY_TELEPORT_1": "enemy_tp1.npy",
             "ENEMY_TELEPORT_2": "enemy_tp2.npy",
             "ENEMY_TELEPORT_3": "enemy_tp3.npy",
             "ENEMY_TELEPORT_4": "enemy_tp4.npy",
             "ENEMY_WALK1": "enemy_walk1.npy",
             "ENEMY_WALK2": "enemy_walk2.npy",
             "ENEMY_WALK3": "enemy_walk3.npy",
             "DEFAULT_ENEMY_SPRITE": "enemy_sprite.npy",
             "YELLOW_EGG": "egg_yellow.npy",
             "ORANGE_EGG": "egg_orange.npy",
             "BLUE_EGG": "egg_blue.npy",
             "PINK_EGG": "egg_pink.npy",
             "TURQUOISE_EGG": "egg_turquoise.npy",
             "ORANGE_BLUE_EGG":"egg_orange_blue.npy",
             "EVIL_ITEM_1": "evil_item_1.npy",
             "EVIL_ITEM_2": "evil_item_2.npy",
             "PULSAR": "pulsar.npy",
             "ROCKET": "rocket.npy",
             "SATURN": "saturn.npy",
             "STARSHIP": "starship.npy",
             "PLAYER_COLLISION_MAP": "player_sprite_collision_map.npy",
             "FLAME_COLLISION_MAP": "flame_sprite_collision_map.npy",
             "LEVEL_COLLISION_MAP": "map_sprite_collision_map.npy",
             "ENEMY_COLLISION_MAP": "player_sprite_collision_map.npy", 
             "DIGIT_NONE": "sprite_none.npy",
             "DIGIT_0": "sprite_0.npy",
             "DIGIT_1": "sprite_1.npy",
             "DIGIT_2": "sprite_2.npy",
             "DIGIT_3": "sprite_3.npy",
             "DIGIT_4": "sprite_4.npy",
             "DIGIT_5": "sprite_5.npy",
             "DIGIT_6": "sprite_6.npy",
             "DIGIT_7": "sprite_7.npy",
             "DIGIT_8": "sprite_8.npy",
             "DIGIT_9": "sprite_9.npy",
             "LIFE": "life_sprite.npy",
             "PLAYER_DEATH_1": "player_death_1_sprite.npy",
             "PLAYER_DEATH_2": "player_death_2_sprite.npy",
             "PLAYER_DEATH_3": "player_death_3_sprite.npy",
             "PLAYER_DEATH_4": "player_death_4_sprite.npy",
        }

# egg discription:
# x-coordinate
# y-coordinate
# status : 1 (on the field),  0 (not on the field)
# color : 0 (yellow), 1 (orange), 2 (blue), 3 (pink), 4 (turquoise), 5 ( orange and blue)

EGG_ARRAY_OLD = jnp.array([
    [18,14,1,4],   [26,16,1,4],   [34,18,1,4],   [48,14,1,1],   [56,16,1,1],   [64,18,1,5],    [81,14,1,4],   [89,16,1,4],   [97,18,1,4],   [110,14,1,1],  [118,16,1,1],  [126,18,1,5],
    [18,26,1,4],                                 [48,26,1,2],                                                                [97,30,1,4],                                 [126,30,1,2],
    [18,38,1,4],   [26,40,1,4],   [34,42,1,4],   [48,38,1,2],   [56,40,1,2],   [64,42,1,2],    [81,38,1,4],   [89,40,1,4],   [97,42,1,4],   [110,38,1,2],  [118,40,1,2],  [126,42,1,2],
    [18,50,1,4],   [26,52,1,4],                                                [64,54,1,2],    [81,50,1,4],                                                [118,52,1,2],  [126,54,1,2],
    [18,62,1,4],   [26,64,1,4],                  [48,62,1,2],   [56,64,1,2],   [64,66,1,2],    [81,62,1,4],   [89,64,1,4],   [97,66,1,4],                  [118,64,1,2],  [126,66,1,2],
                   [26,76,1,4],                  [48,74,1,2],                                                                [97,78,1,4],                  [118,76,1,2],
    [18,86,1,4],   [26,88,1,4],   [34,90,1,4],   [48,86,1,0],   [56,88,1,0],   [64,90,1,0],    [81,86,1,4],   [89,88,1,4],   [97,90,1,4],   [110,86,1,0],  [118,88,1,0],  [126,90,1,0],
                   [26,100,1,4],                                               [64,102,1,0],   [81,98,1,4],                                                [118,100,1,0],
    [18,110,1,4],  [26,112,1,4],  [34,114,1,4],  [48,110,1,0],  [56,112,1,0],  [64,114,1,0],   [81,110,1,4],  [89,112,1,4],  [97,114,1,4],  [110,110,1,0], [118,112,1,0], [126,114,1,0],
    [18,122,1,4],                                [48,122,1,0],                             	                                 [97,126,1,4],                                [126,126,1,0],
    [18,134,1,4],  [26,136,1,4],  [34,138,1,4],  [48,134,1,0],  [56,136,1,0],  [64,138,1,0],   [81,134,1,4],  [89,136,1,4],  [97,138,1,4],  [110,134,1,0], [118,136,1,0], [126,138,1,0],
    [18,146,1,4],                 [34,150,1,4],                           	                                                                [110,146,1,0],                [126,150,1,0],
    [18,158,1,4],  [26,160,1,4],  [34,162,1,4],  [48,158,1,0],  [56,160,1,0],                                 [89,160,1,4],  [97,162,1,4],  [110,158,1,0], [118,160,1,0], [126,162,1,0]
])
EGG_ARRAY = jnp.array([
[
    [18,14,1,4],   [26,16,1,4],   [34,18,1,4],   [48,14,1,1],   [56,16,1,1],   [64,18,1,5],
    [18,26,1,4],                                 [48,26,1,2],
    [18,38,1,4],   [26,40,1,4],   [34,42,1,4],   [48,38,1,2],   [56,40,1,2],   [64,42,1,2],
    [18,50,1,4],   [26,52,1,4],                                                [64,54,1,2],
    [18,62,1,4],   [26,64,1,4],                  [48,62,1,2],   [56,64,1,2],   [64,66,1,2],
                   [26,76,1,4],                  [48,74,1,2],
    [18,86,1,4],   [26,88,1,4],   [34,90,1,4],   [48,86,1,0],   [56,88,1,0],   [64,90,1,0],
                   [26,100,1,4],                                               [64,102,1,0],
    [18,110,1,4],  [26,112,1,4],  [34,114,1,4],  [48,110,1,0],  [56,112,1,0],  [64,114,1,0],
    [18,122,1,4],                                [48,122,1,0],
    [18,134,1,4],  [26,136,1,4],  [34,138,1,4],  [48,134,1,0],  [56,136,1,0],  [64,138,1,0],
    [18,146,1,4],                 [34,150,1,4],
    [18,158,1,4],  [26,160,1,4],  [34,162,1,4],  [48,158,1,0],  [56,160,1,0]
],
[
    [81,14,1,4],   [89,16,1,4],   [97,18,1,4],   [110,14,1,1],  [118,16,1,1],  [126,18,1,5],
                                  [97,30,1,4],                                 [126,30,1,2],
    [81,38,1,4],   [89,40,1,4],   [97,42,1,4],   [110,38,1,2],  [118,40,1,2],  [126,42,1,2],
    [81,50,1,4],                                                [118,52,1,2],  [126,54,1,2],
    [81,62,1,4],   [89,64,1,4],   [97,66,1,4],                  [118,64,1,2],  [126,66,1,2],
                                  [97,78,1,4],                  [118,76,1,2],
    [81,86,1,4],   [89,88,1,4],   [97,90,1,4],   [110,86,1,0],  [118,88,1,0],  [126,90,1,0],
    [81,98,1,4],                                                [118,100,1,0],
    [81,110,1,4],  [89,112,1,4],  [97,114,1,4],  [110,110,1,0], [118,112,1,0], [126,114,1,0],
                                  [97,126,1,4],                                [126,126,1,0],
    [81,134,1,4],  [89,136,1,4],  [97,138,1,4],  [110,134,1,0], [118,136,1,0], [126,138,1,0],
                                                 [110,146,1,0],                [126,150,1,0],
                   [89,160,1,4],  [97,162,1,4],  [110,158,1,0], [118,160,1,0], [126,162,1,0]
]])


ITEM_ARRAY = jnp.array([ # 0: pulsar, 1: rocket, 2: saturn, 3: starship, 4: evil_item at index 3# 1 if the item is active, 0 otherwise
    [68, 9,  1,  4],
    [22,  129,  0,  4],
    [114, 129,  0,  4],
    [68 ,  56,  0,  0] #static score item
])
ITEM_SCORE_MULTIPLYERS = jnp.array([100, 500, 1000, 2000, 0]) # Score for collecting items, last item is not a score item

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class AlienObservation(NamedTuple):
    player: EntityPosition
    enemies: jnp.ndarray # (3, 5) array - 3 enemies, each with x,y,w,h,active
    score: jnp.ndarray

class AlienInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array
    
class AlienConstants(NamedTuple):
    dummy: jnp.ndarray
    
class FlameState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    flame_counter: jnp.ndarray
    flame_flag:jnp.ndarray

class PlayerState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    orientation: jnp.ndarray
    last_horizontal_orientation: jnp.ndarray
    flame: FlameState
    collision_map: jnp.ndarray
    blink: jnp.ndarray

class EnemyMode(NamedTuple):
    scatter_duration: jnp.ndarray
    scatter_point_x: jnp.ndarray
    scatter_point_y: jnp.ndarray
    chase_duration: jnp.ndarray
    mode_change_probability: jnp.ndarray
    frightened_duration: jnp.ndarray
    type: jnp.ndarray # mode_types: 0=chase, 1=scatter, 2=frightend
    duration: jnp.ndarray

class SingleEnemyState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    orientation: jnp.ndarray
    mode: EnemyMode
    last_horizontal_orientation: jnp.ndarray
    enemy_spawn_frame: jnp.ndarray
    enemy_death_frame: jnp.ndarray
    key: jnp.ndarray
    blink: jnp.ndarray

class MultipleEnemiesState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    orientation: jnp.ndarray
    mode: EnemyMode
    last_horizontal_orientation: jnp.ndarray
    enemy_spawn_frame: jnp.ndarray
    enemy_death_frame: jnp.ndarray
    key: jnp.ndarray
    blink: jnp.ndarray

class EnemiesState(NamedTuple):
    multiple_enemies: MultipleEnemiesState
    enemies_amount: jnp.ndarray
    rng_key: jnp.ndarray
    collision_map: jnp.ndarray
    velocity: jnp.ndarray

class LevelState(NamedTuple):
    collision_map: jnp.ndarray
    score: jnp.ndarray
    frame_count: jnp.ndarray
    lifes : jnp.ndarray
    death_frame_counter: jnp.ndarray
    evil_item_frame_counter: jnp.ndarray
    current_active_item_index: jnp.ndarray
    blink_evil_item: jnp.ndarray
    blink_current_active_item: jnp.ndarray


class AlienState(NamedTuple):
    player: PlayerState
    enemies: EnemiesState
    level: LevelState
    eggs: jnp.ndarray
    items: jnp.ndarray

m_fs = set(MultipleEnemiesState._fields)
s_fs = set(SingleEnemyState._fields)
if not (m_fs.issubset(s_fs) and s_fs.issubset(m_fs)):
    raise Exception("Mismatch between fields in SingleEnemyState and MultipleEnemiesState")


def loadFramAddAlpha(fileName, transpose=True, add_alpha: bool = False, add_black_as_transparent: bool = False):
    # Custom loading function which turns black background transparent.
    # This is simply to make editing sprites a bit more convenient.
    frame = jnp.load(fileName)
    if frame.shape[-1] != 4 and add_alpha:
        alphas = jnp.ones((*frame.shape[:-1], 1))
        alphas = alphas*255
        frame = jnp.concatenate([frame, alphas], axis=-1)
        if add_black_as_transparent:
            arr_black = jnp.sum(frame[..., :-1], axis=-1)
            alpha_channel = frame[..., -1]
            alpha_channel = alpha_channel.at[arr_black == 0].set(0)
            frame = frame.at[..., -1].set(alpha_channel)
    # Check if the frame's shape is [[[r, g, b, a], ...], ...]
    if frame.ndim != 3:
        raise ValueError(
            "Invalid frame format. The frame must have a shape of (height, width, 4)."
        )
    return jnp.transpose(frame, (1, 0, 2)) if transpose else frame

def load_collision_map(fileName, transpose=True):
    # Returns a boolean array representing the collision map
    # Load frame (np array) from a .npy file and convert to jnp array
    frame = jnp.load(fileName)
    frame = frame[..., 0].squeeze()
    boolean_frame = jnp.zeros(shape=frame.shape, dtype=jnp.bool)
    boolean_frame = boolean_frame.at[frame==0].set(0)
    boolean_frame = boolean_frame.at[frame > 0].set(1)
    frame = boolean_frame
    return jnp.transpose(frame, (1, 0)) if transpose else frame


screen = pygame.display.set_mode((WIDTH * RENDER_SCALE_FACTOR, HEIGHT * RENDER_SCALE_FACTOR))


# Action space
def get_human_action() -> chex.Array:
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_s] and keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.DOWNLEFTFIRE)
    elif keys[pygame.K_s] and keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.DOWNRIGHTFIRE)
    elif keys[pygame.K_w] and keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.UPLEFTFIRE)
    elif keys[pygame.K_w] and keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.UPRIGHTFIRE)
    
    elif keys[pygame.K_s] and keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.DOWNFIRE)
    elif keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.LEFTFIRE)
    elif keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.RIGHTFIRE)
    elif keys[pygame.K_w] and keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.UPFIRE)
    
    elif keys[pygame.K_s] and keys[pygame.K_a]:
        return jnp.array(JAXAtariAction.DOWNLEFT)
    elif keys[pygame.K_s] and keys[pygame.K_d]:
        return jnp.array(JAXAtariAction.DOWNRIGHT)
    elif keys[pygame.K_w] and keys[pygame.K_a]:
        return jnp.array(JAXAtariAction.UPLEFT)
    elif keys[pygame.K_w] and keys[pygame.K_d]:
        return jnp.array(JAXAtariAction.UPRIGHT)
    
    elif keys[pygame.K_s]:
        return jnp.array(JAXAtariAction.DOWN)
    elif keys[pygame.K_a]:
        return jnp.array(JAXAtariAction.LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(JAXAtariAction.RIGHT)
    elif keys[pygame.K_w]:
        return jnp.array(JAXAtariAction.UP)
    
    elif keys[pygame.K_SPACE]:
        return jnp.array(JAXAtariAction.FIRE)
    
    else:
        return jnp.array(JAXAtariAction.NOOP)

@jax.jit
def check_for_collision(moving_object: jnp.ndarray, background: jnp.ndarray, old_position: jnp.ndarray, new_position: jnp.ndarray) -> jnp.ndarray:
    """Checks position of the moving object against a static background. If new position would collide with the static background,
       returns old position, else new position. Old position is assumed to be valid in any case, and is not rechecked.
       This does not perform bounds checking, which might lead to unexpected crashes if used wrong.
       (Call only after modify wall collision!!! OTHERWISE THINGS BREAK HORIBLY)
    Args:
        moving_object (jnp.ndarray, dtype = bool): Collision Map of the moving object. Has Shape (Width, Height). Entries with 1 indicate that this is a pixel with collision enabled.
        background (jnp.ndarray, dtype = bool): Collision map of the background. Has shape (Width, Height), entries with 1 indicate that a pixel has collision enabled.
        old_position (jnp.ndarray): Old position of the moving object in relation to the background. Position of the upper, left corner as [X-coord, Y-coord]
        new_position (jnp.ndarray): New position of the moving objec in relation to the background.
    Returns:
        jnp.ndarray: Returns collision free position in the form of [X-coord, Y-coord]
    """
    new_position_bg: jnp.ndarray = jax.lax.dynamic_slice(operand=background,
                          start_indices=new_position, slice_sizes=moving_object.shape)
    collisions: jnp.ndarray = jnp.logical_and(moving_object, new_position_bg)
    # Use max to check whether moving_object collision map overlaps with background collision map
    has_collision = jnp.max(a=collisions, axis=None, keepdims=False)
    ret_v = jax.lax.cond(has_collision, lambda x: old_position, lambda x: new_position, 0)
    return ret_v

@jax.jit
def check_for_enemy_player_collision(state: AlienState, new_position: jnp.ndarray) -> jnp.ndarray:
    """Checks for collision with enemies. If player collides with enemy, player is reset to starting position and life is decremented
    Args:
        state (AlienState): Current state of the player
        new_position: (jnp.ndarray): Proposed position of the player after the movement would be executed

     Returns:
        _type_: _description_
    """
    # Determine coord range occupied by the player
    x_higher_player = jnp.add(new_position[0], PLAYER_WIDTH)
    x_lower_player = jnp.add(new_position[0], 0)
    y_higher_player = jnp.add(new_position[1], PLAYER_HEIGHT)
    y_lower_player = jnp.add(new_position[1], 0)

    # Check if player sprite crosses a certian point in enemy sprite
    has_collision_enemy0 = jnp.logical_and(state.enemies.multiple_enemies.x[0] >= x_lower_player,
                           jnp.logical_and(state.enemies.multiple_enemies.x[0]  < x_higher_player,
                           jnp.logical_and(jnp.add(state.enemies.multiple_enemies.y[0], ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW) >= y_lower_player,
                                           jnp.add(state.enemies.multiple_enemies.y[0], ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH) < y_higher_player)))

    has_collision_enemy1 = jnp.logical_and(state.enemies.multiple_enemies.x[1] >= x_lower_player,
                           jnp.logical_and(state.enemies.multiple_enemies.x[1] < x_higher_player,
                           jnp.logical_and(jnp.add(state.enemies.multiple_enemies.y[1], ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW) >= y_lower_player,
                                           jnp.add(state.enemies.multiple_enemies.y[1], ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH) < y_higher_player)))

    has_collision_enemy2 = jnp.logical_and(state.enemies.multiple_enemies.x[2] >= x_lower_player,
                           jnp.logical_and(state.enemies.multiple_enemies.x[2] < x_higher_player,
                           jnp.logical_and(jnp.add(state.enemies.multiple_enemies.y[2], ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW) >= y_lower_player,
                                           jnp.add(state.enemies.multiple_enemies.y[2], ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH) < y_higher_player)))

    return jnp.logical_or(has_collision_enemy0, jnp.logical_or(has_collision_enemy1, has_collision_enemy2))

@staticmethod
@jax.jit
def teleport_object(position,orientation, action):
    return jax.lax.cond(
        jnp.logical_or(jnp.logical_and(position[0] >= 127, orientation == JAXAtariAction.RIGHT), jnp.logical_and(action == JAXAtariAction.RIGHT, position[0] >= 127)),
        lambda x: x.at[0].set(7),
        lambda x: jax.lax.cond(
            jnp.logical_or(jnp.logical_and(position[0] <= 7,orientation == JAXAtariAction.LEFT),jnp.logical_and(action == JAXAtariAction.LEFT, position[0] <= 7)),
            lambda x: x.at[0].set(127),
            lambda x:x,
            position
        ),
        position
    )

@jax.jit
def enemy_step(enemy: SingleEnemyState, state: AlienState)-> SingleEnemyState:
    """Handles step for a single enemy"""

    def normal_enemy_step(enemy: SingleEnemyState, state: AlienState)-> SingleEnemyState:
        """Normal step for a single enemy. When enemy is not in spawning animation"""

        def chase_point(point_x: jnp.ndarray, point_y: jnp.ndarray, allowed_directions: jnp.ndarray, steps_in_all_directions: jnp.ndarray):
            """Chases the player, returns the minimal distance to the player direction"""

            distances = jax.vmap(lambda enemy_position: distance_to_point(enemy_position, point_x, point_y))(steps_in_all_directions)
            distances = jnp.where(allowed_directions == 1, distances, jnp.inf)

            return jnp.add(jnp.argmin(distances), 2)

        def frightend(player_x: jnp.ndarray, player_y: jnp.ndarray, allowed_directions: jnp.ndarray):
            """Frightened mode, enemy moves away from the player"""

            distances = jax.vmap(lambda enemy_position: distance_to_point(enemy_position, player_x, player_y))(steps_in_all_directions)
            distances = jnp.where(allowed_directions == 1, distances, 0)

            return jnp.add(jnp.argmax(distances), 2)

        def distance_to_point(enemy_position, point_x, point_y):
            """Calculates the distance between the player and the enemy"""

            return jnp.sqrt(jnp.square(jnp.subtract(point_x, enemy_position[0])) + jnp.square(jnp.subtract(point_y, enemy_position[1])))

        def new_mode_fun(enemy: SingleEnemyState):
            """Determines the new mode of the enemy based on the current mode and a random value"""

            random_value = jax.random.uniform(enemy.key, shape=(), minval=0, maxval=1)
            return jax.lax.cond(
                jnp.less(random_value, enemy.mode.mode_change_probability),  # Probability
                lambda _: (0, enemy.mode.chase_duration),  # Chase mode
                lambda _: (1, enemy.mode.scatter_duration),  # Scatter mode
                operand=None
                )

        def check_wall_collison(new_pos):
            """Returns Boolean if step in a direction can be made"""

            tmp_pos = check_for_collision(state.enemies.collision_map, state.level.collision_map, position, new_pos)
            return jnp.logical_not(jnp.all(tmp_pos == position))

        opposite_table = jnp.array([
                3,  # UP -> DOWN
                2,  # RIGHT -> LEFT
                1,  # LEFT -> RIGHT
                0,  # DOWN -> UP
                ], dtype=jnp.int32)

        direction_offsets = jnp.array([
                [0, -1],  # UP
                [1, 0],   # RIGHT
                [-1, 0],  # LEFT
                [0, 1],   # DOWN
                ], dtype=jnp.int32)

        position = jnp.array([enemy.x, enemy.y])

        steps_in_all_directions = jnp.add(position, direction_offsets)

        allowed_directions = jax.vmap(check_wall_collison)(steps_in_all_directions).astype(jnp.int32)
        # Remove opposite direction of current directionfrom allowed directions
        allowed_directions = allowed_directions.at[opposite_table[jnp.subtract(enemy.orientation, 2)]].set(0)

        new_mode_type, new_mode_duration = jax.lax.cond(
            jnp.equal(enemy.mode.duration, 0),
            lambda x: new_mode_fun(x),
            lambda x: (x.mode.type, jnp.subtract(x.mode.duration, 1)),
            enemy
        )
        #jax.debug.print("Enemy: {enemy}", "Enemy mode type: {new_mode_type}, duration: {new_mode_duration}", enemy=enemy.mode.frightened_duration , new_mode_type=new_mode_type, new_mode_duration=new_mode_duration )

        #new_mode_type, new_mode_duration = jax.lax.cond(
        #    jnp.logical_and(jnp.greater(state.level.evil_item_frame_counter, 0), jnp.less(enemy.mode[0], 2)),
        #    lambda x: (2, x.mode.frightened_duration),
        #    lambda x: (x.mode[0], jnp.subtract(x.mode[1], 1)),
        #    enemy
        #)

        new_direction = jax.lax.switch(new_mode_type,[
            lambda _: chase_point(state.player.x, state.player.y, allowed_directions, steps_in_all_directions), # Chase mode
            lambda _: chase_point(enemy.mode.scatter_point_x, enemy.mode.scatter_point_y, allowed_directions, steps_in_all_directions), # Scatter mode
            lambda _: frightend(state.player.x, state.player.y, allowed_directions) # Frightened mode
        ], None)

        # if velocity = 1 new_direction=new_direction
        # if velocity = 0 new_direction=current_orientation
        new_direction = (
            jnp.add(
                jnp.multiply(new_direction, state.enemies.velocity),
                jnp.multiply(enemy.orientation, jnp.bitwise_xor(state.enemies.velocity, 1))
            )
        )

        new_last_horizontal_orientation = jax.lax.cond(
            jnp.logical_or(enemy.orientation == JAXAtariAction.LEFT, enemy.orientation == JAXAtariAction.RIGHT),
            lambda _: enemy.orientation,
            lambda _: enemy.last_horizontal_orientation,
            operand=None
        )

        position = teleport_object(position, enemy.orientation, new_direction)

        new_position = jax.lax.switch(jnp.subtract(new_direction, 2), [
            lambda x: x.at[1].subtract(state.enemies.velocity),  # UP
            lambda x: x.at[0].add(state.enemies.velocity),      # RIGHT
            lambda x: x.at[0].subtract(state.enemies.velocity), # LEFT
            lambda x: x.at[1].add(state.enemies.velocity)       # DOWN
        ], position)
        def check_for_enemy_player_collision(state: AlienState, new_position: jnp.ndarray) -> jnp.ndarray:
            """Checks for collision with enemies. If player collides with enemy, player is reset to starting position and life is decremented
            Args:
                state (AlienState): Current state of the player
                new_position: (jnp.ndarray): Proposed position of the player after the movement would be executed

             Returns:
                _type_: _description_
            """
            # Determine coord range occupied by the player
            x_higher_player = jnp.add(new_position[0], PLAYER_WIDTH)
            x_lower_player = jnp.add(new_position[0], 0)
            y_higher_player = jnp.add(new_position[1], PLAYER_HEIGHT)
            y_lower_player = jnp.add(new_position[1], 0)

            # Check if player sprite crosses a certian point in enemy sprite
            has_collision = jnp.logical_and(state.player.x >= x_lower_player,
                                            jnp.logical_and(state.player.x < x_higher_player,
                                                            jnp.logical_and(jnp.add(state.player.y, ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW) >= y_lower_player,
                                                                            jnp.add(state.player.y, ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH) < y_higher_player)))
            return has_collision
        new_enemy_death_frame = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(
                    check_for_enemy_player_collision(state, new_position),
                    jnp.equal(enemy.enemy_death_frame, 0)
                ),
                jnp.greater(state.level.evil_item_frame_counter, 0)  # Evil Item aktiv
            ),
            lambda x: jnp.array(63),  # Setze Death Frame nur wenn Evil Item aktiv
            lambda x: x,
            operand=enemy.enemy_death_frame
        )

        new_enemy_death_frame = jax.lax.cond(jnp.greater(new_enemy_death_frame, 0), lambda x: jnp.subtract(x, 1), lambda x: x, new_enemy_death_frame)

        return SingleEnemyState(
            x=new_position[0],
            y=new_position[1],
            orientation=new_direction,
            mode=EnemyMode(
                scatter_duration=enemy.mode.scatter_duration,
                scatter_point_x=enemy.mode.scatter_point_x,
                scatter_point_y=enemy.mode.scatter_point_y,
                chase_duration=enemy.mode.chase_duration,
                mode_change_probability=enemy.mode.mode_change_probability,
                frightened_duration=enemy.mode.frightened_duration,
                type=new_mode_type,
                duration=new_mode_duration
            ),
            last_horizontal_orientation=new_last_horizontal_orientation,
            enemy_spawn_frame=jnp.array(0, dtype=jnp.int32),
            enemy_death_frame=new_enemy_death_frame,
            key=enemy.key,
            blink=enemy.blink
        )

    def spawn_enemy_step(enemy: SingleEnemyState, state: AlienState)-> SingleEnemyState:
        """Step for a single enemy that is currently in spawning animation"""

        def add_velocity(y, orientation, velocity, death_frame):
            return jnp.add(y, velocity), orientation, death_frame

        def substract_velocity(y, orientation, velocity, death_frame):
            return jnp.subtract(y, velocity), orientation, death_frame

        def change_to_down(y, orientation, velocity, death_frame):
            return jnp.add(y, velocity), JAXAtariAction.LEFT, jnp.subtract(death_frame, 1)

        def change_to_up(y, orientation, velocity, death_frame):
            return jnp.subtract(y, velocity), JAXAtariAction.RIGHT, jnp.subtract(death_frame, 1)


        bound = jax.lax.cond(jnp.equal(jnp.mod(enemy.enemy_spawn_frame , 9), 1), lambda x: ENEMY_START_Y, lambda x: ENEMY_SPAWN_Y - 10, 0)

        new_y, new_orientation, new_spawn_frame = jax.lax.cond(jnp.equal(enemy.orientation, JAXAtariAction.RIGHT),
                             lambda x, y, z: jax.lax.cond(jnp.less_equal(x, bound),
                                                    lambda a, b, c: change_to_down(a, b, state.enemies.velocity, c),
                                                    lambda a, b, c: substract_velocity(a, b, state.enemies.velocity, c),
                                                    x, y, z
                                                    ),
                             lambda x, y, z: jax.lax.cond(jnp.greater_equal(x, ENEMY_SPAWN_Y),
                                                    lambda a, b, c: change_to_up(a, b, state.enemies.velocity, c),
                                                    lambda a, b, c: add_velocity(a, b, state.enemies.velocity, c),
                                                    x, y, z
                                                    ),
                             enemy.y, enemy.orientation, enemy.enemy_spawn_frame
                            )

        return SingleEnemyState(
            x=ENEMY_SPAWN_X,
            y=new_y,
            orientation=new_orientation,
            mode=enemy.mode,
            last_horizontal_orientation=enemy.last_horizontal_orientation,
            enemy_spawn_frame=new_spawn_frame,
            enemy_death_frame=enemy.enemy_death_frame,
            key=enemy.key,
            blink=enemy.blink
        )

    return jax.lax.cond(jnp.greater(enemy.enemy_spawn_frame, 0), lambda x: spawn_enemy_step(x, state), lambda x: normal_enemy_step(x, state), enemy)



class JaxAlien(JaxEnvironment[AlienState, AlienObservation, AlienInfo, AlienConstants]): #[EnvState, EnvObs, EnvInfo, EnvConstants]
    def __init__(self):
        super().__init__()
        self.sprite_path: str = os.path.join(MODULE_DIR, "sprites", "alien")
        self.sprite_f_dict: Dict[str, str] = SPRITE__F_DICT

        self.player_collision_map: jnp.ndarray = load_collision_map(os.path.join(self.sprite_path, self.sprite_f_dict["PLAYER_COLLISION_MAP"]), transpose=True)
        self.level_collision_map: jnp.ndarray = load_collision_map(os.path.join(self.sprite_path, self.sprite_f_dict["LEVEL_COLLISION_MAP"]), transpose=True)
        self.enemy_collision_map: jnp.ndarray = load_collision_map(os.path.join(self.sprite_path, self.sprite_f_dict["ENEMY_COLLISION_MAP"]), transpose=True)
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
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
    def reset(self, *args, **kwargs) -> AlienState:
        """
        Resets the game state to the initial state.
        """
        key = jax.random.PRNGKey(SEED)
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

        player_state = PlayerState(
            x=jnp.array(PLAYER_X).astype(jnp.int32),
            y=jnp.array(PLAYER_Y).astype(jnp.int32),
            orientation=JAXAtariAction.UP,
            last_horizontal_orientation= JAXAtariAction.RIGHT,
            flame= FlameState(
                x=jnp.array(PLAYER_X - 8).astype(jnp.int32),
                y=jnp.array(PLAYER_Y + 5).astype(jnp.int32),
                flame_counter=jnp.array(480).astype(jnp.int32),
                flame_flag=jnp.array(0).astype(jnp.int32),
                ),
            collision_map=self.player_collision_map,
            blink=jnp.array(0).astype(jnp.int32)
            )

        mode = EnemyMode(
            scatter_duration=jnp.array([SCATTER_DURATION_1, SCATTER_DURATION_2, SCATTER_DURATION_3]),
            scatter_point_x=jnp.array([SCATTER_POINT_X_1, SCATTER_POINT_X_2, SCATTER_POINT_X_3]),
            scatter_point_y=jnp.array([SCATTER_POINT_Y_1, SCATTER_POINT_Y_2, SCATTER_POINT_Y_3]),
            chase_duration=jnp.array([CHASE_DURATION_1, CHASE_DURATION_2, CHASE_DURATION_3]),
            mode_change_probability=jnp.array([MODECHANGE_PROBABILITY_1, MODECHANGE_PROBABILITY_2, MODECHANGE_PROBABILITY_3]),
            frightened_duration=jnp.array([FRIGHTENED_DURATION, FRIGHTENED_DURATION, FRIGHTENED_DURATION]),
            type=jnp.array([0, 0, 0]),
            duration=jnp.array([200, 100, 50])
        )

        enemies_state = EnemiesState(
            multiple_enemies = MultipleEnemiesState(
                x=jnp.array([0, 0, 0]),
                y=jnp.array([0, 0, 0]),
                orientation=jnp.array([JAXAtariAction.RIGHT, JAXAtariAction.RIGHT, JAXAtariAction.RIGHT]),
                mode=mode,
                last_horizontal_orientation=jnp.array([JAXAtariAction.RIGHT, JAXAtariAction.RIGHT, JAXAtariAction.RIGHT]),
                enemy_spawn_frame=jnp.array([9, 19, 31]).astype(jnp.int32),
                enemy_death_frame=jnp.array([0, 0, 0]).astype(jnp.int32),
                key=jnp.array([subkey1, subkey2, subkey3]),
                blink=jnp.array([0,0,0]).astype(jnp.int32)
            ),
            enemies_amount=jnp.array(3).astype(jnp.int32),
            rng_key=key,
            collision_map=self.enemy_collision_map,
            velocity=jnp.array(1).astype(jnp.int32)
        )
        
        level_state = LevelState(
            collision_map=self.level_collision_map,
            score=jnp.array([0]).astype(jnp.uint16),
            frame_count=jnp.array(0).astype(jnp.int32),
            lifes=jnp.array(2).astype(jnp.int32),
            death_frame_counter=jnp.array(0).astype(jnp.int32),
            evil_item_frame_counter=jnp.array(0).astype(jnp.int32),
            current_active_item_index=jnp.array(0).astype(jnp.int32),
            blink_evil_item=jnp.array(0).astype(jnp.int32),
            blink_current_active_item=jnp.array(0).astype(jnp.int32)
        )

        state = AlienState(
            player=player_state,
            enemies=enemies_state,
            level= level_state,
            eggs=EGG_ARRAY,
            items= ITEM_ARRAY
        )
        dummy_obs = AlienObservation(player=None,
                                     enemies=None,
                                     score=0)
        return (dummy_obs, state)


    @partial(jax.jit, static_argnums=(0))
    def step(self, state: AlienState, action: chex.Array) -> Tuple[AlienState, AlienObservation, AlienInfo]:

        # Normal game step. This is the function that is called when the game is running normally.
        def normal_game_step(state: AlienState, action: chex.Array) -> Tuple[AlienState, AlienObservation, AlienInfo]:
            """Normal game step. This is the function that is called when the game is running normally.

            Args:
                state (AlienState): Current state of the game
                action (chex.Array): Action to be taken

            Returns:
                Tuple[AlienState, AlienObservation, AlienInfo]: New state, observation and info
            """
            new_player_x, new_player_y, new_player_orientation, new_last_horizontal_orientation, new_lifes, new_death_frame_counter, new_flame =  self.player_step(state, action)

            new_enemies = self.multiple_enemies_step(state)

            new_player_state = PlayerState(
                x=new_player_x,
                y=new_player_y,
                orientation=new_player_orientation,
                last_horizontal_orientation=new_last_horizontal_orientation,
                flame=new_flame,
                collision_map=state.player.collision_map,
                blink=state.player.blink
            )

            new_egg_state, new_score = self.egg_step(
                new_player_x,
                new_player_y,
                state.eggs,
                state.level.score,
                EGG_SCORE_MULTIPLYER
            )
            new_item_state, new_score, new_current_active_item_index, new_evil_item_frame_counter = self.item_step(
                new_player_x,
                new_player_y,
                state.items,
                new_score,
                ITEM_SCORE_MULTIPLYERS,
                state.level.current_active_item_index,
                state.level.evil_item_frame_counter
            )

            new_game_state = LevelState(
                collision_map=state.level.collision_map,
                score=new_score,
                frame_count=state.level.frame_count + 1,
                lifes=new_lifes,
                death_frame_counter=new_death_frame_counter,
                evil_item_frame_counter=new_evil_item_frame_counter,
                current_active_item_index= new_current_active_item_index,
                blink_evil_item=state.level.blink_evil_item,
                blink_current_active_item=state.level.blink_current_active_item
            )
            new_state = AlienState(
                player=new_player_state,
                enemies=new_enemies,
                level= new_game_state,
                eggs=new_egg_state,
                items=new_item_state
            )
            return new_state

        def kill_item_step(state: AlienState) -> Tuple[AlienState, AlienObservation, AlienInfo]:



            freeze_level_state = LevelState(
                    collision_map=state.level.collision_map,
                    score=state.level.score,
                    frame_count=state.level.frame_count + 1,
                    lifes=state.level.lifes,
                    death_frame_counter=state.level.death_frame_counter,
                    evil_item_frame_counter=state.level.evil_item_frame_counter,
                    current_active_item_index= state.level.current_active_item_index,
                    blink_evil_item=state.level.blink_evil_item,
                    blink_current_active_item=state.level.blink_current_active_item
            )
            new_enemies = self.multiple_enemies_step(state)
            freeze_enemies = EnemiesState(
                    multiple_enemies = MultipleEnemiesState(
                        x=state.enemies.multiple_enemies.x,
                        y=state.enemies.multiple_enemies.y,
                        orientation=state.enemies.multiple_enemies.orientation,
                        mode= state.enemies.multiple_enemies.mode,
                        last_horizontal_orientation=state.enemies.multiple_enemies.last_horizontal_orientation,
                        enemy_spawn_frame=state.enemies.multiple_enemies.enemy_spawn_frame,
                        enemy_death_frame=new_enemies.multiple_enemies.enemy_death_frame,
                        key=state.enemies.multiple_enemies.key,
                        blink=jnp.array([0,0,0]).astype(jnp.int32)
                    ),
                    enemies_amount=state.enemies.enemies_amount,
                    rng_key=state.enemies.rng_key,
                    collision_map=state.enemies.collision_map,
                    velocity=state.enemies.velocity
            )



            freeze_state = AlienState(
                    player=state.player,
                    enemies=freeze_enemies,
                    level=freeze_level_state,
                    eggs=state.eggs,
                    items=state.items
                )

            kill_enemies = EnemiesState(
                    multiple_enemies = MultipleEnemiesState(
                        x=new_enemies.multiple_enemies.x,
                        y=new_enemies.multiple_enemies.y,
                        orientation= new_enemies.multiple_enemies.orientation,
                        mode=state.enemies.multiple_enemies.mode,  # 0: chase, 1: scatter, 2: frightend
                        last_horizontal_orientation=new_enemies.multiple_enemies.last_horizontal_orientation,
                        enemy_spawn_frame=new_enemies.multiple_enemies.enemy_spawn_frame,
                        enemy_death_frame=new_enemies.multiple_enemies.enemy_death_frame,
                        key=new_enemies.multiple_enemies.key,
                        blink=jnp.array([0,0,0]).astype(jnp.int32)
                    ),
                    enemies_amount=new_enemies.enemies_amount,
                    rng_key= new_enemies.rng_key,
                    collision_map=new_enemies.collision_map,
                    velocity=new_enemies.velocity
                )

            new_player_x, new_player_y, new_player_orientation, new_last_horizontal_orientation, new_lifes, new_death_frame_counter, new_flame =  self.player_step(state, action)
            new_player_state = PlayerState(
                x=new_player_x,
                y=new_player_y,
                orientation=new_player_orientation,
                last_horizontal_orientation=new_last_horizontal_orientation,
                flame=new_flame,
                collision_map=state.player.collision_map,
                blink=jnp.array(0).astype(jnp.int32)
            )
            new_egg_state, new_score = self.egg_step(
                new_player_x,
                new_player_y,
                state.eggs,
                state.level.score,
                EGG_SCORE_MULTIPLYER
            )
            level_state = LevelState(
                collision_map=state.level.collision_map,
                score=new_score,
                frame_count=state.level.frame_count + 1,
                lifes=state.level.lifes,
                death_frame_counter=state.level.death_frame_counter,
                evil_item_frame_counter=jnp.add(state.level.evil_item_frame_counter, -1),
                current_active_item_index=state.level.current_active_item_index,
                blink_evil_item=state.level.blink_evil_item,
                blink_current_active_item=state.level.blink_current_active_item
            )

            kill_state = AlienState(
                    player=new_player_state,
                    enemies=kill_enemies,
                    level=level_state,
                    eggs=new_egg_state,
                    items=state.items
                )

            new_state = jax.lax.cond(
                    jnp.equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 0),
                    lambda x: kill_state,
                    lambda x: freeze_state,
                    None
                )


            return new_state


        #step for death animation with soft reset
        def death(state: AlienState) -> Tuple[AlienState, AlienObservation, AlienInfo]:

            new_key, subkey1, subkey2, subkey3 = jax.random.split(state.enemies.rng_key, 4)

            new_death_frame_counter = jnp.add(state.level.death_frame_counter, -1)

            new_level_state = LevelState(
                collision_map=state.level.collision_map,
                score=state.level.score,
                frame_count=state.level.frame_count + 1,
                lifes=state.level.lifes,
                death_frame_counter=new_death_frame_counter,
                evil_item_frame_counter=state.level.evil_item_frame_counter,
                current_active_item_index= state.level.current_active_item_index,
                blink_evil_item=jnp.array(0).astype(jnp.int32),
                blink_current_active_item=jnp.array(0).astype(jnp.int32)
            )

            freeze_state = AlienState(
                player=state.player,
                enemies=state.enemies,
                level=new_level_state,
                eggs=state.eggs,
                items=state.items
            )

            new_player = PlayerState(
                x=jnp.array(PLAYER_X).astype(jnp.int32),
                y=jnp.array(PLAYER_Y).astype(jnp.int32),
                orientation=JAXAtariAction.UP,
                last_horizontal_orientation=JAXAtariAction.RIGHT,
                flame= FlameState(
                x=jnp.array(PLAYER_X - 8).astype(jnp.int32),
                y=jnp.array(PLAYER_Y + 5).astype(jnp.int32),
                flame_counter=jnp.array(480).astype(jnp.int32),
                flame_flag=jnp.array(0).astype(jnp.int32),
                ),
                collision_map=state.player.collision_map,
                blink=jnp.array(0).astype(jnp.int32)
            )

            mode = EnemyMode(
                scatter_duration=jnp.array([SCATTER_DURATION_1, SCATTER_DURATION_2, SCATTER_DURATION_3]),
                scatter_point_x=jnp.array([SCATTER_POINT_X_1, SCATTER_POINT_X_2, SCATTER_POINT_X_3]),
                scatter_point_y=jnp.array([SCATTER_POINT_Y_1, SCATTER_POINT_Y_2, SCATTER_POINT_Y_3]),
                chase_duration=jnp.array([CHASE_DURATION_1, CHASE_DURATION_2, CHASE_DURATION_3]),
                mode_change_probability=jnp.array([MODECHANGE_PROBABILITY_1, MODECHANGE_PROBABILITY_2, MODECHANGE_PROBABILITY_3]),
                frightened_duration=jnp.array([FRIGHTENED_DURATION, FRIGHTENED_DURATION, FRIGHTENED_DURATION]),
                type=jnp.array([0, 0, 0]),
                duration=jnp.array([200, 100, 50])
            )

            soft_reset_enemies = EnemiesState(
                multiple_enemies = MultipleEnemiesState(
                x=jnp.array([0, 0, 0]),
                y=jnp.array([ENEMY_SPAWN_Y, ENEMY_SPAWN_Y, ENEMY_SPAWN_Y]),
                orientation=jnp.array([JAXAtariAction.RIGHT, JAXAtariAction.RIGHT, JAXAtariAction.RIGHT]),
                mode=mode,
                last_horizontal_orientation=jnp.array([JAXAtariAction.RIGHT, JAXAtariAction.RIGHT, JAXAtariAction.RIGHT]),
                enemy_spawn_frame=jnp.array([9, 19, 31]).astype(jnp.int32),
                enemy_death_frame=jnp.array([0, 0, 0]).astype(jnp.int32),
                key=jnp.array([subkey1, subkey2, subkey3]),
                blink=jnp.array([0,0,0]).astype(jnp.int32)
            ),
                enemies_amount=state.enemies.enemies_amount,
                rng_key=new_key,
                collision_map=state.enemies.collision_map,
                velocity=jnp.array(1).astype(jnp.int32)
            )
            level_state = LevelState(
                collision_map=state.level.collision_map,
                score=state.level.score,
                frame_count=state.level.frame_count + 1,
                lifes=state.level.lifes,
                death_frame_counter=new_death_frame_counter,
                evil_item_frame_counter=jnp.array(0).astype(jnp.int32),
                current_active_item_index=state.level.current_active_item_index,
                blink_evil_item=jnp.array(0).astype(jnp.int32),
                blink_current_active_item=jnp.array(0).astype(jnp.int32)
            )
            soft_reset_state = AlienState(
                player=new_player,
                enemies=soft_reset_enemies,
                level=level_state,
                eggs=state.eggs,
                items=state.items
            )

            new_state = jax.lax.cond(
                jnp.equal(state.level.death_frame_counter, 1),
                lambda x: soft_reset_state,
                lambda x: freeze_state,
                0
            )

            return new_state

        #step for death animation with hard reset
        def game_over(state: AlienState) -> Tuple[AlienState, AlienObservation, AlienInfo]:
            """Game over state. This is called when the game is over.

            Returns:
                Tuple[AlienState, AlienObservation, AlienInfo]: New state, observation and info
            """

            new_key, subkey1, subkey2, subkey3 = jax.random.split(state.enemies.rng_key, 4)

            new_death_frame_counter = jnp.add(state.level.death_frame_counter, 1)

            new_level_state = LevelState(
                collision_map=state.level.collision_map,
                score=state.level.score,
                frame_count=state.level.frame_count + 1,
                lifes= state.level.lifes,  # Reset lives to 2
                death_frame_counter=new_death_frame_counter,
                evil_item_frame_counter= state.level.evil_item_frame_counter,
                current_active_item_index=jnp.array(0).astype(jnp.int32),
                blink_evil_item=jnp.array(0).astype(jnp.int32),
                blink_current_active_item=jnp.array(0).astype(jnp.int32)
            )
            freeze_state = AlienState(
                player= state.player,
                enemies=state.enemies,
                level=new_level_state,
                eggs=state.eggs,
                items=state.items
            )
            new_player = PlayerState(
                x=jnp.array(PLAYER_X).astype(jnp.int32),
                y=jnp.array(PLAYER_Y).astype(jnp.int32),
                orientation=JAXAtariAction.UP,
                last_horizontal_orientation=JAXAtariAction.RIGHT,
                flame= FlameState(
                x=jnp.array(PLAYER_X - 8).astype(jnp.int32),
                y=jnp.array(PLAYER_Y + 5).astype(jnp.int32),
                flame_counter=jnp.array(480).astype(jnp.int32),
                flame_flag=jnp.array(0).astype(jnp.int32),
                ),
                collision_map=state.player.collision_map,
                blink=jnp.array(0).astype(jnp.int32)
            )

            mode = EnemyMode(
                scatter_duration=jnp.array([SCATTER_DURATION_1, SCATTER_DURATION_2, SCATTER_DURATION_3]),
                scatter_point_x=jnp.array([SCATTER_POINT_X_1, SCATTER_POINT_X_2, SCATTER_POINT_X_3]),
                scatter_point_y=jnp.array([SCATTER_POINT_Y_1, SCATTER_POINT_Y_2, SCATTER_POINT_Y_3]),
                chase_duration=jnp.array([CHASE_DURATION_1, CHASE_DURATION_2, CHASE_DURATION_3]),
                mode_change_probability=jnp.array([MODECHANGE_PROBABILITY_1, MODECHANGE_PROBABILITY_2, MODECHANGE_PROBABILITY_3]),
                frightened_duration=jnp.array([FRIGHTENED_DURATION, FRIGHTENED_DURATION, FRIGHTENED_DURATION]),
                type=jnp.array([0, 0, 0]),
                duration=jnp.array([200, 100, 50])
            )

            new_hard_reset_enemies = EnemiesState(
                multiple_enemies = MultipleEnemiesState(
                x=jnp.array([0, 0, 0]),
                y=jnp.array([0, 0, 0]),
                orientation=jnp.array([JAXAtariAction.RIGHT, JAXAtariAction.RIGHT, JAXAtariAction.RIGHT]),
                mode=mode,
                last_horizontal_orientation=jnp.array([JAXAtariAction.RIGHT, JAXAtariAction.RIGHT, JAXAtariAction.RIGHT]),
                enemy_spawn_frame=jnp.array([9, 19, 31]).astype(jnp.int32),
                enemy_death_frame= jnp.array([0, 0, 0]).astype(jnp.int32),
                key=jnp.array([subkey1, subkey2, subkey3]),
                blink=jnp.array([0,0,0]).astype(jnp.int32)
            ),
                enemies_amount=state.enemies.enemies_amount,
                rng_key=new_key,
                collision_map=state.enemies.collision_map,
                velocity=jnp.array(1).astype(jnp.int32)
            )
            new_hard_reset_level_state = LevelState(
                collision_map=state.level.collision_map,
                score=jnp.array([0]).astype(jnp.uint16),  # Reset score to 0
                frame_count=jnp.array(0).astype(jnp.int32),  # Reset frame count to 0
                lifes=jnp.array(2).astype(jnp.int32),  # Reset lives to 2
                death_frame_counter=new_death_frame_counter,
                evil_item_frame_counter=jnp.array(0).astype(jnp.int32),
                current_active_item_index=jnp.array(0).astype(jnp.int32),
                blink_evil_item=jnp.array(0).astype(jnp.int32),
                blink_current_active_item=jnp.array(0).astype(jnp.int32)
            )
            hard_reset_state = AlienState(
                player=new_player,
                enemies=new_hard_reset_enemies,
                level=new_hard_reset_level_state,
                eggs=EGG_ARRAY,
                items=ITEM_ARRAY
            )

            new_state = jax.lax.cond(jnp.equal(state.level.death_frame_counter, -1), lambda x: hard_reset_state, lambda x: freeze_state, 0)
            return new_state

        #initial state for cond
        new_state = state
        #cond for checking for normal game step or death or game over
        new_state = jax.lax.cond(jnp.greater(state.level.evil_item_frame_counter, 0),
                                    lambda x: kill_item_step(x),
                                    lambda x:  jax.lax.cond(jnp.equal(state.level.death_frame_counter, 0),
                                        lambda y: normal_game_step(y, action),
                                        lambda y: jax.lax.cond(jnp.greater(state.level.death_frame_counter, 0),
                                                        lambda z: death(z),
                                                        lambda z: game_over(z),
                                                        y),

                                 x),
                                 new_state)


        # Only the state_update is currently implemented. We need to handle score & observation still!!
        return None, new_state, 0, None, None

    def render(self, state: AlienState) -> Tuple[jnp.ndarray]:
        raise NotImplementedError("Use provided renderer")
    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)
    def get_observation_space(self) -> Tuple:
        # Not yet implemented
        return None

    def _get_observation(self, state: AlienState) -> AlienObservation:
        # NOt yet implemented
        return AlienObservation(
            player=None,
            enemies=None,
            score=None
        )

    def _get_info(self, state: AlienState) -> AlienObservation:
        # Not yet properly implemented
        return None

    def _get_reward(self, previous_state: AlienState, state: AlienState) -> float:
        #Not yet properly implemented
        return 0.0

    def _get_done(self, state: AlienState) -> bool:
        # Not yet properly implemented
        return False


    @partial(jax.jit, static_argnums=(0))
    def egg_step(self, player_x: jnp.ndarray, player_y: jnp.ndarray, egg_state:jnp.ndarray,
                 score: jnp.ndarray, egg_score_multiplyer: jnp.ndarray):
        """Handles egg collision & score update

        Args:
            player_x (jnp.ndarray): X coord. of player
            player_y (jnp.ndarray): Y coord. of player
            egg_state (jnp.ndarray): 2D array specifying the state of all eggs in the game. 1st Dim is # eggs,
                2nd dim is state of a single egg: [x_pos, y_pos, egg_visible, egg_color]
            score (jnp.ndarray): current score
            egg_score_multiplyer (jnp.ndarray): How many points are given for collecting a single egg

        Returns:
            _type_: _description_
        """
        egg_collision_map_left: jnp.ndarray = jnp.zeros((egg_state.shape[1]))
        egg_collision_map_right: jnp.ndarray = jnp.zeros((egg_state.shape[1]))
        # Determine coord range occupied by the player
        x_lower: jnp.ndarray = player_x
        x_higher: jnp.ndarray = jnp.add(player_x, PLAYER_WIDTH)
        y_lower: jnp.ndarray = player_y
        y_higher: jnp.ndarray = jnp.add(player_y, PLAYER_HEIGHT)

        def check_egg_player_collision(i, l_r, single_egg_col_map):
            # Function that fills out the current egg_collision_map
            # for an index i, checks whether the i-th egg currently collides with the player and sets
            # the i-th index in the collision map to one
            has_collision = jnp.logical_and(egg_state[l_r, i, 0]>= x_lower,
                                           jnp.logical_and(egg_state[l_r, i, 0] < x_higher,
                                           jnp.logical_and(egg_state[l_r, i, 1] >= y_lower,
                                                           egg_state[l_r, i, 1] < y_higher)))
            single_egg_col_map = single_egg_col_map.at[i].set(has_collision.astype(np.uint8))
            return single_egg_col_map
        # Generate full collision map for all eggs
        egg_collision_map_left = jax.lax.fori_loop(0, egg_state.shape[1] - 1, lambda cnt, cm, cnst=0 : check_egg_player_collision(i=cnt, l_r=cnst, single_egg_col_map=cm), egg_collision_map_left)
        egg_collision_map_right = jax.lax.fori_loop(0, egg_state.shape[1] - 1,lambda cnt, cm, cnst=1 :  check_egg_player_collision(i=cnt, l_r=cnst, single_egg_col_map=cm), egg_collision_map_right)
        # Multiply with current active egg-state to prevent the same egg from being collected twice
        score_increas_left: jnp.ndarray = jnp.sum(jnp.multiply(egg_collision_map_left, egg_state[0,:, 2]))
        score_increas_right: jnp.ndarray = jnp.sum(jnp.multiply(egg_collision_map_right, egg_state[1,:, 2]))
        # Multiply collision map onto egg-state to set visible attribute to the appropriate value
        egg_collision_map_left = jnp.subtract(1, egg_collision_map_left)
        egg_collision_map_right = jnp.subtract(1, egg_collision_map_right)

        new_egg_presence_map_left: jnp.ndarray = jnp.multiply(egg_collision_map_left, egg_state[0,:, 2])
        new_egg_presence_map_right: jnp.ndarray = jnp.multiply(egg_collision_map_right, egg_state[1,:, 2])
        egg_state = egg_state.at[0,:, 2].set(new_egg_presence_map_left)
        egg_state = egg_state.at[1,:, 2].set(new_egg_presence_map_right)
        new_score: jnp.ndarray = jnp.add(score, jnp.add(jnp.multiply(score_increas_left, egg_score_multiplyer),jnp.multiply(score_increas_right, egg_score_multiplyer)))
        new_score = new_score.astype(jnp.uint16)

        return egg_state, new_score

    @partial(jax.jit, static_argnums=(0))
    def item_step(self, player_x: jnp.ndarray, player_y: jnp.ndarray, item_state:jnp.ndarray,
                 score: jnp.ndarray, item_mutliplyers: jnp.ndarray, current_active_item_index: jnp.ndarray, evil_item_frame_counter: jnp.ndarray):
        """Handles egg collision & score update

        Args:
            player_x (jnp.ndarray): X coord. of player
            player_y (jnp.ndarray): Y coord. of player
            egg_state (jnp.ndarray): 2D array specifying the state of all eggs in the game. 1st Dim is # eggs,
                2nd dim is state of a single egg: [x_pos, y_pos, egg_visible, egg_color]
            score (jnp.ndarray): current score
            egg_score_multiplyer (jnp.ndarray): How many points are given for collecting a single egg

        Returns:
            _type_: _description_
        """
        item_collision_map: jnp.ndarray = jnp.zeros((item_state.shape[0]))
        # Determine coord range occupied by the player
        x_lower: jnp.ndarray = player_x
        x_higher: jnp.ndarray = jnp.add(player_x, PLAYER_WIDTH)
        y_lower: jnp.ndarray = player_y
        y_higher: jnp.ndarray = jnp.add(player_y, PLAYER_HEIGHT)


        def check_item_player_collision(i, single_item_col_map):
            # Function that fills out the current egg_collision_map
            # for an index i, checks whether the i-th egg currently collides with the player and sets
            # the i-th index in the collision map to one
            has_collision = jnp.logical_and(item_state[i, 0]>= x_lower,
                                            jnp.logical_and(item_state[i, 0] < x_higher,
                                                            jnp.logical_and(item_state[i, 1] >= y_lower,
                                                                            item_state[i, 1] < y_higher)))
            single_item_col_map = single_item_col_map.at[i].set(has_collision.astype(np.uint8))

            return single_item_col_map
        # Generate full collision map for all eggs
        item_collision_map = jax.lax.fori_loop(0, item_state.shape[0], check_item_player_collision, item_collision_map)

        # Multiply with current active egg-state to prevent the same egg from being collected twice
        # Berechne Score-Erhöhung basierend auf Item-Typ
        score_increases = jnp.multiply(
            item_collision_map,  # Kollisionsmaske
            jnp.multiply(
                item_state[:, 2],  # Aktive Items
                item_mutliplyers[item_state[:, 3]]  # Score-Multiplikator für jeden Item-Typ
            )
        )
        new_evil_item_frame_counter= jax.lax.cond(jnp.equal(item_collision_map[current_active_item_index], 1),
                                                  lambda _: jnp.array( 504),
                                                  lambda _: evil_item_frame_counter,
                                                  operand=None
                                                  )

        item_state = jax.lax.cond(
            jnp.logical_and(jnp.equal(item_collision_map[current_active_item_index], 1),jnp.less(current_active_item_index, 2)),
            lambda x: x.at[current_active_item_index+1, 2].set(1),
            lambda x: x,
            operand=item_state
        )
        new_current_active_item_index = jax.lax.cond(
            jnp.equal(item_collision_map[current_active_item_index], 1),
            lambda x: jnp.add(current_active_item_index, 1),  # Wenn Item aktiv ist, gehe zum nächsten Item
            lambda x: current_active_item_index,  # Wenn Item nicht aktiv ist, behalte den aktuellen Index bei
            operand=current_active_item_index
        )

        item_state= jax.lax.cond(
           jnp.equal(score[0],410),
            lambda x: x.at[3,2].set(1),
            lambda x: x,
            operand= item_state# Setze den Status des Items auf aktiv, wenn der Score 410 erreicht
        )
        score_increase = jnp.sum(score_increases)
        # Multiply collision map onto egg-state to set visible attribute to the appropriate value
        item_collision_map = jnp.subtract(1, item_collision_map)
        new_item_presence_map: jnp.ndarray = jnp.multiply(item_collision_map, item_state[:, 2])
        item_state = item_state.at[:, 2].set(new_item_presence_map)
        # Aktualisiere Score
        new_score = jnp.add(score, score_increase)
        new_score = new_score.astype(jnp.uint16)



        return item_state, new_score, new_current_active_item_index, new_evil_item_frame_counter



    @partial(jax.jit, static_argnums=(0))
    def modify_wall_collision(self, new_position: jnp.ndarray):
        """Collision handling for game-field bounds.
           Alters player position to lie within the confines of the playfield

        Args:
            new_position (jnp.ndarray): _description_

        Returns:
            _type_: _description_
        """

        permissible_lower_bounds: jnp.ndarray = jnp.array([0, 0], dtype=np.int32)
        permissible_upper_bound: jnp.ndarray = jnp.array([WIDTH - PLAYER_WIDTH , HEIGHT - PLAYER_HEIGHT ], dtype=np.int32)
        lower_bound_check: jnp.ndarray = jnp.vstack([new_position, permissible_lower_bounds])
        lower_checked = jnp.max(lower_bound_check, axis=0, keepdims=False)

        upper_bound_check: jnp.ndarray = jnp.vstack([lower_checked, permissible_upper_bound])
        checked = jnp.min(upper_bound_check, axis=0, keepdims=False)
        return checked


    @partial(jax.jit, static_argnums=(0))
    def player_step(self, state: AlienState, action: jnp.ndarray):
        """Handles full step for the player

        Args:
            state (AlienState): Current state of the player
            action (jnp.ndarray): Proposed action

        Returns:
            _type_: _description_
        """
        # Prevent change of orientation, if player can't move into that direction.
        # This is done in the original game as well.
        # For each possible change in movement direction, we first check whether the player can currently move in that direction.
        #Could be cleaned up, but currently other_slightly_weirder_check_for_player_collision is necessary, as the other collision check has a different signature.
        # Probably best to merge the two methods at some point...
        velocity_this_frame = jnp.round((117/249)*(state.level.frame_count)).astype(jnp.int32) - jnp.round((117/249)*(state.level.frame_count - 1)).astype(jnp.int32)
        old_orientation = state.player.orientation
        state_player_orientation = old_orientation
        
        # maps the action onto the relevant movement-actions
        moving_action = jax.lax.cond(jnp.greater(action,9),
                                      lambda _: jnp.mod(action,10)+2,
                                      lambda _: action,
                                      operand=None)
        
        # decides orientation based on the imput
        state_player_orientation = jax.lax.switch(moving_action, [
                lambda x: state_player_orientation,
                lambda x: state_player_orientation,
                lambda x: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(state, 0, -velocity_this_frame),
                                       lambda x: JAXAtariAction.UP, 
                                       lambda x: x,
                                       state_player_orientation),   # UP
                
                lambda x: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(state, velocity_this_frame, 0),
                                       lambda x: JAXAtariAction.RIGHT, 
                                       lambda x: x,
                                       state_player_orientation),   # RIGHT
                
                lambda x: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(state, -velocity_this_frame, 0),
                                       lambda x: JAXAtariAction.LEFT, 
                                       lambda x: x,
                                       state_player_orientation),   # LEFT
                
                lambda x: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(state, 0, velocity_this_frame),
                                       lambda x: JAXAtariAction.DOWN, 
                                       lambda x: x,
                                       state_player_orientation),   # DOWN
                
                lambda x: jax.lax.cond(jnp.logical_and(self.other_slightly_weirder_check_for_player_collision(state, velocity_this_frame, 0),jnp.not_equal(state_player_orientation,JAXAtariAction.RIGHT)),
                                       lambda x: JAXAtariAction.RIGHT, 
                                       lambda x: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(state, 0, -velocity_this_frame),
                                                              lambda x: JAXAtariAction.UP,
                                                              lambda x: state_player_orientation,
                                                              state_player_orientation),
                                       state_player_orientation),   # UPRIGHT 
                
                lambda x: jax.lax.cond(jnp.logical_and(self.other_slightly_weirder_check_for_player_collision(state, 0, -velocity_this_frame),jnp.not_equal(state_player_orientation,JAXAtariAction.UP)),
                                       lambda x: JAXAtariAction.UP, 
                                       lambda x: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(state, -velocity_this_frame, 0),
                                                              lambda x: JAXAtariAction.LEFT,
                                                              lambda x: state_player_orientation,
                                                              state_player_orientation),
                                       state_player_orientation),   # UPLEFT 
                
                lambda x: jax.lax.cond(jnp.logical_and(self.other_slightly_weirder_check_for_player_collision(state, 0, velocity_this_frame),jnp.not_equal(state_player_orientation,JAXAtariAction.DOWN)),
                                       lambda x: JAXAtariAction.DOWN, 
                                       lambda x: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(state, velocity_this_frame, 0),
                                                              lambda x: JAXAtariAction.RIGHT,
                                                              lambda x: state_player_orientation,
                                                              state_player_orientation),
                                       state_player_orientation),   #DOWNRIGHT
                
                lambda x: jax.lax.cond(jnp.logical_and(self.other_slightly_weirder_check_for_player_collision(state, -velocity_this_frame, 0),jnp.not_equal(state_player_orientation,JAXAtariAction.LEFT)),
                                       lambda x: JAXAtariAction.LEFT, 
                                       lambda x: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(state, 0, velocity_this_frame),
                                                              lambda x: JAXAtariAction.DOWN,
                                                              lambda x: state_player_orientation,
                                                              state_player_orientation),
                                       state_player_orientation), #DOWNLEFT  
                              
            ], state_player_orientation)
        #Determine last horizontal orientation, this is necessary for correctly displaying the player sprite
        
        last_horizontal_orientation = jax.lax.cond(
            jnp.logical_or(state_player_orientation == JAXAtariAction.LEFT, state_player_orientation == JAXAtariAction.RIGHT),
            lambda _: state_player_orientation,
            lambda _: state.player.last_horizontal_orientation,
            operand=None
        )

        # Handle movement at this point:
        # Choose movement function according to index of proposed action.
        position = jnp.array([state.player.x, state.player.y])
        position = teleport_object(position, state.player.orientation, action)
        up_func = lambda x: x.at[1].subtract(velocity_this_frame)
        down_func = lambda x: x.at[1].add(velocity_this_frame)
        left_func = lambda x: x.at[0].subtract(velocity_this_frame)
        right_func = lambda x: x.at[0].add(velocity_this_frame)
        # This will cause problems later on, we should make it more clean
        func_index = jnp.mod(state_player_orientation, 4).astype(jnp.uint16)

        new_position = jax.lax.switch(func_index, [left_func, down_func, up_func, right_func], position)
        #Checks for collision with the outer wall, and push player back into bounds
        new_position = self.modify_wall_collision(new_position)

        #Checks whether new position collides with the game walls, if it does so, new position is rejected
        new_position = check_for_collision(
            moving_object=state.player.collision_map,
            background=state.level.collision_map,
            old_position=position,
            new_position=new_position
        )

        #initialize new_life and new_death_frame_counter for cond
        new_life = state.level.lifes
        new_death_frame_counter = state.level.death_frame_counter

         # Check for collision with enemies
        new_life = jax.lax.cond(check_for_enemy_player_collision(state, new_position), lambda x: jnp.add(x, -1), lambda x: x, new_life)
        new_death_frame_counter = jax.lax.cond(check_for_enemy_player_collision(state, new_position),
                                               lambda x: jax.lax.cond(jnp.less(new_life, 0),
                                                                      lambda y: jnp.add(y, -40),
                                                                      lambda y: jnp.add(y, 40),
                                                                      new_death_frame_counter
                                                                      ),
                                               lambda x: x,
                                               new_death_frame_counter
                                               )
        
        new_flame_flag = jax.lax.cond(jnp.logical_and(jnp.logical_and(jnp.logical_or(jnp.greater(action,9),jnp.equal(action,1)),jnp.greater(state.player.flame.flame_counter,0)),
                                                      jnp.logical_not(jnp.logical_and(state.player.y == 80,jnp.logical_or(
                jnp.logical_and(state.player.x >= 7, state.player.x <= 11),
                jnp.logical_and(state.player.x >= 123, state.player.x <= 127))
            ))),
                                      lambda _: 1,
                                      lambda _: 0,
                                      operand=None)
        new_flame_counter = state.player.flame.flame_counter - new_flame_flag
        
        new_flame_x = jax.lax.cond(jnp.equal(state.player.last_horizontal_orientation,JAXAtariAction.LEFT),
                                   lambda _: new_position[0] - 6,
                                   lambda _: new_position[0] + 10,
                                   operand=None)

        return (new_position[0],#new_position.at[0]
                new_position[1],#new_position.at[1],
                state_player_orientation,
                last_horizontal_orientation,
                new_life,
                new_death_frame_counter,
                FlameState(
                x=jnp.array(new_flame_x).astype(jnp.int32),
                y=jnp.array(new_position[1] + 6).astype(jnp.int32),
                flame_counter=jnp.array(new_flame_counter).astype(jnp.int32),
                flame_flag=jnp.array(new_flame_flag).astype(jnp.int32)
                )
                )

    @partial(jax.jit, static_argnums=(0))
    def other_slightly_weirder_check_for_player_collision(self, state: AlienState, x_change, y_change) -> jnp.ndarray:
        """Checks for player collision. Basically just a wrapper around check_for_collision for the sake of convenience.

        Args:
            state (AlienState): Current State of the Game
            x_change (_type_): proposed position change of the play in x direction
            y_change (_type_): proposed position change of the player in y direction

        Returns:
            jnp.ndarray: Boolean value indicating whether proposed position change would lead to collision
        """
        m_object: jnp.ndarray = state.player.collision_map
        bg: jnp.ndarray = state.level.collision_map
        current_pos: jnp.ndarray = jnp.array([state.player.x, state.player.y])
        new_pos: jnp.ndarray = current_pos.at[0].add(x_change).at[1].add(y_change)
        new_pos = check_for_collision(
            m_object,
            bg,
            current_pos,
            new_pos
        )
        return jnp.logical_not(jnp.array_equal(new_pos, current_pos))

    @partial(jax.jit, static_argnums=(0))
    def multiple_enemies_step(self, state: AlienState) -> EnemiesState:
        """Handles step for all enemies."""

        def normal_mult_enemies_step(state: AlienState) -> EnemiesState:
            """Normal step for all enemies. When enemies are not frozen"""
            
            new_key, subkey1, subkey2, subkey3 = jax.random.split(state.enemies.rng_key, 4)
            

            
            new_multiple_enemies = MultipleEnemiesState(
                x=state.enemies.multiple_enemies.x,
                y=state.enemies.multiple_enemies.y,
                orientation=state.enemies.multiple_enemies.orientation,
                mode=state.enemies.multiple_enemies.mode,
                last_horizontal_orientation=state.enemies.multiple_enemies.last_horizontal_orientation,
                enemy_spawn_frame=state.enemies.multiple_enemies.enemy_spawn_frame,
                enemy_death_frame= state.enemies.multiple_enemies.enemy_death_frame,
                key=jnp.array([subkey1, subkey2, subkey3]),
                blink=state.enemies.multiple_enemies.blink
            )
            
            new_enemy_velocity_this_frame = jnp.round((117/249)*(state.level.frame_count)).astype(jnp.int32) - jnp.round((117/249)*(state.level.frame_count - 1)).astype(jnp.int32)
            
            return EnemiesState(
                multiple_enemies=jitted_enemy_step(new_multiple_enemies, state),
                enemies_amount=state.enemies.enemies_amount,
                rng_key=new_key,
                collision_map=state.enemies.collision_map,
                velocity=new_enemy_velocity_this_frame
            )

        def freeze_mult_enemies_step(state: AlienState) -> EnemiesState:
            """Step for all enemies when they are frozen. This is the initial state of the game, where enemies are not yet spawned"""
            
            # checks frame if init position of enemies spawning should be set
            new_x = jax.lax.cond(jnp.equal(state.level.frame_count, 99),lambda _: jnp.array([ENEMY_SPAWN_X, ENEMY_SPAWN_X, ENEMY_SPAWN_X]), lambda _: state.enemies.multiple_enemies.x, None)
            new_y = jax.lax.cond(jnp.equal(state.level.frame_count, 99),lambda _: jnp.array([ENEMY_SPAWN_Y, ENEMY_SPAWN_Y, ENEMY_SPAWN_Y]), lambda _: state.enemies.multiple_enemies.y, None)

            frozen_enemies = MultipleEnemiesState(
                x=new_x,
                y=new_y,
                orientation=jnp.array([JAXAtariAction.RIGHT, JAXAtariAction.RIGHT, JAXAtariAction.RIGHT]),
                mode=state.enemies.multiple_enemies.mode,
                last_horizontal_orientation=jnp.array([JAXAtariAction.RIGHT, JAXAtariAction.RIGHT, JAXAtariAction.RIGHT]),
                enemy_spawn_frame=jnp.array([9, 19, 31]).astype(jnp.int32),
                enemy_death_frame=jnp.array([0, 0, 0]).astype(jnp.int32),
                key=state.enemies.multiple_enemies.key,
                blink=jnp.array([0,0,0]).astype(jnp.int32)
            )
            return EnemiesState(
                multiple_enemies=frozen_enemies,
                enemies_amount=state.enemies.enemies_amount,
                rng_key=state.enemies.rng_key,
                collision_map=state.enemies.collision_map,
                velocity=state.enemies.velocity
            )

        return jax.lax.cond(jnp.less_equal(state.level.frame_count, 100), # count-down(100 Frames) before enemies spawn
                            lambda x: freeze_mult_enemies_step(x),
                            lambda x: normal_mult_enemies_step(x),
                            state
                            )

class AlienRenderer(JAXGameRenderer):
    def __init__(self):
        # Load all required sprites from disk. 
        # We use our own loading method which interpretes a black background (0, 0, 0) as transparent.
        self.sprite_path: str = os.path.join(MODULE_DIR, "sprites", "alien")
        self.sprite_f_dict: Dict[str, str] = SPRITE__F_DICT
        
        self.map_sprite: str = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["BASIC_MAP"]), transpose=True, add_alpha=True)

        self.player_1_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["PLAYER_1"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.player_2_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["PLAYER_2"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.player_3_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["PLAYER_3"]), transpose=True, add_alpha=True,   add_black_as_transparent=True)
        self.player_sprite = jnp.stack([self.player_1_sprite, self.player_2_sprite, self.player_3_sprite, self.player_2_sprite])
        
        self.flame_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["FLAME"]), transpose=True, add_alpha=True, add_black_as_transparent=True)

        self.teleport_1_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["TELEPORT_1"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.teleport_2_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["TELEPORT_2"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.teleport_3_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["TELEPORT_3"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.teleport_4_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["TELEPORT_4"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.teleport_sprites = jnp.stack([self.teleport_1_sprite, self.teleport_2_sprite, self.teleport_3_sprite, self.teleport_4_sprite,self.teleport_4_sprite])

        self.player_death_1_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["PLAYER_DEATH_1"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.player_death_2_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["PLAYER_DEATH_2"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.player_death_3_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["PLAYER_DEATH_3"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.player_death_4_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["PLAYER_DEATH_4"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.player_death_sprites = jnp.stack([self.player_death_4_sprite, self.player_death_3_sprite, self.player_death_2_sprite, self.player_death_1_sprite])
        
        self.enemy1_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ENEMY_WALK1"]), transpose=True, add_alpha=True,add_black_as_transparent=True)
        self.enemy2_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ENEMY_WALK2"]), transpose=True, add_alpha=True,add_black_as_transparent=True)
        self.enemy3_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ENEMY_WALK3"]), transpose=True, add_alpha=True,add_black_as_transparent=True)
        self.enemy_sprites: jnp.ndarray = jnp.stack([self.enemy1_sprite, self.enemy2_sprite, self.enemy3_sprite,self.enemy2_sprite])
        
        self.enemy_tp1_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ENEMY_TELEPORT_1"]), transpose=True, add_alpha=True)
        self.enemy_tp2_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ENEMY_TELEPORT_2"]), transpose=True, add_alpha=True)
        self.enemy_tp3_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ENEMY_TELEPORT_3"]), transpose=True, add_alpha=True)
        self.enemy_tp4_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ENEMY_TELEPORT_4"]), transpose=True, add_alpha=True)
        self.enemy_tp_sprites = jnp.stack([self.enemy_tp1_sprite, self.enemy_tp2_sprite, self.enemy_tp3_sprite, self.enemy_tp4_sprite])

        self.yellow_egg_sprite: str = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["YELLOW_EGG"]), transpose=True, add_alpha=True)
        self.orange_egg_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ORANGE_EGG"]), transpose=True, add_alpha=True)
        self.blue_egg_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["BLUE_EGG"]), transpose=True, add_alpha=True)
        self.pink_egg_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["PINK_EGG"]), transpose=True, add_alpha=True)
        self.turquoise_egg_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["TURQUOISE_EGG"]), transpose=True, add_alpha=True)
        self.orange_blue_egg_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ORANGE_BLUE_EGG"]), transpose=True, add_alpha=True)
        self.egg_sprites: jnp.ndarray = jnp.stack([self.yellow_egg_sprite, self.orange_egg_sprite, self.blue_egg_sprite, self.pink_egg_sprite, self.turquoise_egg_sprite, self.orange_blue_egg_sprite])

        self.evil_item1_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["EVIL_ITEM_1"]), transpose=True, add_alpha=True)
        self.evil_item2_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["EVIL_ITEM_2"]), transpose=True, add_alpha=True)
        self.evil_item_sprites: jnp.ndarray = jnp.stack([self.evil_item1_sprite, self.evil_item2_sprite])

        self.pulsar_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["PULSAR"]), transpose=True, add_alpha=True)
        self.rocket_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["ROCKET"]), transpose=True, add_alpha=True)
        self.saturn_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["SATURN"]), transpose=True, add_alpha=True)
        self.starship_sprite = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["STARSHIP"]), transpose=True, add_alpha=True)
        self.items: jnp.ndarray = jnp.stack([self.pulsar_sprite, self.rocket_sprite, self.saturn_sprite, self.starship_sprite])

        self.digit_none = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_NONE"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_0 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_0"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_1 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_1"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_2 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_2"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_3 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_3"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_4 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_4"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_5 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_5"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_6 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_6"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_7 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_7"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_8 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_8"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_9 = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["DIGIT_9"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
        self.digit_sprite: jnp.ndarray = jnp.stack([self.digit_none, self.digit_0, self.digit_1, self.digit_2, self.digit_3, 
                                                    self.digit_4, self.digit_5, self.digit_6, self.digit_7, self.digit_8, self.digit_9])
        
        self.life_sprite: jnp.ndarray = loadFramAddAlpha(os.path.join(self.sprite_path, self.sprite_f_dict["LIFE"]), transpose=True, add_alpha=True, add_black_as_transparent=True)
    
    @partial(jax.jit, static_argnums=(0))
    def _render_at(self, raster, y, x, sprite_frame, flip_horizontal=False, flip_vertical=False):
        """Renders a sprite onto a raster at position (x,y) with optional flipping.
           1-to-1 copy of the old render_at function from the oc_atari branch. 
           The new rendering function produces problems with our code base, we couldn't fix those in time.

        Args:
            raster: JAX array of shape (width, height, 3) for the target image
            y: Integer y coordinate for sprite placement
            x: Integer x coordinate for sprite placement
            sprite_frame: JAX array of shape (height, width, 4) containing RGB + alpha
        """
        ####
        ####
        # THIS IS JUST HE _render_at METHOD FROM THE STATE OF THE OC_ATARI BRANCH WE DEVELOPED ON, 
        # AS THE NEW ONE BREAKS OUR IMPLEMENTATION AND WE COULDN'T FIGURE OUT WHAT WAS GOING ON IN TIME
        # Get dimensions correctly - sprite is in (height, width) format
        ####
        ####
        
        
        sprite_height, sprite_width, _ = sprite_frame.shape
        raster_width, raster_height, _ = raster.shape

        # Create sprite array and handle flipping - axis 0 is height, axis 1 is width
        sprite = sprite_frame
        sprite = jnp.where(
            flip_horizontal, jnp.flip(sprite, axis=0), sprite  # Flip width dimension
        )
        sprite = jnp.where(
            flip_vertical, jnp.flip(sprite, axis=1), sprite  # Flip height dimension
        )

        # Rest remains same but with corrected dimensions
        sprite_rgb = sprite[..., :3]
        alpha = (sprite[..., 3:] / 255.0).astype(jnp.uint8)

        # Use correct dimension ordering in slicing
        raster_region = jax.lax.dynamic_slice(
            raster,
            (x.astype(int), y.astype(int), 0),
            (sprite_height, sprite_width, 3),  # Note width, height order to match raster
        )

        blended = sprite_rgb * alpha + raster_region * (1.0 - alpha)

        blended = blended.astype(jnp.uint8)  # Ensure blended is uint8
        raster = raster.astype(jnp.uint8)  # Ensure raster is uint8

        new_raster = jax.lax.dynamic_update_slice(
            raster, blended, (x.astype(int), y.astype(int), 0)
        )

        return new_raster
    
    
    @partial(jax.jit, static_argnums=(0))# This is ok in this context, as digits never changes throughout the game...
    def get_score_sprite(self, score: jnp.ndarray) -> jnp.ndarray:
        """Takes numerical representation of the current score and composes a score sprite from individual digit sprites

        Args:
            score (jnp.ndarray): _description_

        Returns:
            jnp.ndarray: _description_
        """
        # Set dimensions of final sprite
        final_sprite: jnp.ndarray = jnp.zeros((3*DIGIT_WIDTH + 2*DIGIT_OFFSET, DIGIT_HEIGHT, 4), np.uint8)
        hundreds_sprite_index: jnp.ndarray = jnp.floor_divide(score, jnp.array([100]))
        # Sprite indices are offset by one, as the 0-th position is occupied by a fully transparent sprite. 
        # This is used to hide the 100th position to prevent leading zeros which are not rendered in the original game.
        hundreds_sprite_index = jnp.multiply(jnp.logical_and(hundreds_sprite_index, hundreds_sprite_index>0).astype(jnp.uint8), jnp.add(hundreds_sprite_index, 1))# Make sure, that if score is lower than 100, no leading digits are displayed
        #Same for the 10-position, prevents rendering of leading zeros.
        tens_sprite_index: jnp.ndarray = jnp.floor_divide(jnp.mod(score, jnp.array([100])), jnp.array([10]))
        tens_sprite_index = jnp.multiply(jnp.logical_or((hundreds_sprite_index>0), tens_sprite_index > 0).astype(jnp.uint8), jnp.add(tens_sprite_index, 1))
        # Here, this isn't necessary, as in the game, a score of 0 is displayed as "0", so no need to remove leading zeros at this position.
        ones_sprite_index: jnp.ndarray = jnp.add(jnp.mod(score, jnp.array([10])), 1)
        final_sprite = final_sprite.at[0:DIGIT_WIDTH, ...].set(jnp.squeeze(self.digit_sprite[hundreds_sprite_index, ...]))
        final_sprite = final_sprite.at[DIGIT_WIDTH + DIGIT_OFFSET:2*DIGIT_WIDTH + DIGIT_OFFSET,...].set(jnp.squeeze(self.digit_sprite[tens_sprite_index, ...]))
        final_sprite = final_sprite.at[DIGIT_WIDTH*2 + DIGIT_OFFSET*2:DIGIT_WIDTH*3 + DIGIT_OFFSET*2,...].set(jnp.squeeze(self.digit_sprite[ones_sprite_index, ...]))
        return final_sprite
    
    @partial(jax.jit, static_argnums=(0))
    def render(self, state: AlienState):
        """Jitted rendering function. receives the alien state, and returns a rendered frame

        Args:
            state (AlienState): _description_

        Returns:
            jnp.ndarray: Returns only the RGB channels, no alpha
        """
        sprite_cycle_1 = jnp.array([3, 3, 2, 1, 0])  # Für Positionen 7-11
        sprite_cycle_2 = jnp.array([0, 1, 2, 3, 3])
        sprite_cycle_3 = jnp.array([0,0,1,1,2,2,3,3])
        sprite_cycle_4 = jnp.array([3,3,2,2,1,1,0,0])
        canvas = self.map_sprite
        
         # Load the map sprite and colorize it with the desired color
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
            nonblack = jnp.any(rgb != 0, axis=-1, keepdims=True)
            # Broadcast color to sprite shape
            color_broadcast = jnp.broadcast_to(color, rgb.shape)
            # Where nonblack, set to color; else keep original
            new_rgb = jnp.where(nonblack, color_broadcast, rgb)
            
            return jnp.concatenate([new_rgb, alpha], axis=-1).astype(jnp.uint8)
    
        # Alien rendering done with for_i loop, allows for dynamic # of aliens.
        def render_loop_alien(i, canvas):
            x_positions = state.enemies.multiple_enemies.x
            y_positions = state.enemies.multiple_enemies.y
            last_horizontal_orientations = state.enemies.multiple_enemies.last_horizontal_orientation
            x = x_positions[i]
            y = y_positions[i]
            last_horizontal_orientations = last_horizontal_orientations[i]

            #color = ENEMY_COLORS[state.frame_count % len(ENEMY_COLORS)] #blincking enemys lol
            color = ENEMY_COLORS[i]
            
            sprite_index = jax.lax.cond(
                jnp.logical_and(jnp.logical_and(x >= 7, x <= 14), y == 80),
                lambda _: sprite_cycle_1[(x - 7) % len(sprite_cycle_3)],
                lambda _: jax.lax.cond(
                    jnp.logical_and(jnp.logical_and(x >= 120, x <= 127), y == 80),
                    lambda _: sprite_cycle_2[(x- 120) % len(sprite_cycle_4)],
                    lambda _: (state.level.frame_count // 8) % 4,  # Standardanimation
                    operand=None
                ),
                operand=None
            )
            x_higher_player = jnp.add(state.player.x, PLAYER_WIDTH)
            x_lower_player = jnp.add(state.player.x, 0)
            y_higher_player = jnp.add(state.player.y, PLAYER_HEIGHT)
            y_lower_player = jnp.add(state.player.y, 0)

            # Check if player sprite crosses a certian point in enemy sprite
            has_collision_enemy = jnp.logical_and(x >= x_lower_player,
                                                   jnp.logical_and(x < x_higher_player,
                                                                   jnp.logical_and(jnp.add(y, ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW) >= y_lower_player,
                                                                                   jnp.add(y, ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH) < y_higher_player)))
            enemy_sprite = jax.lax.cond(
                jnp.logical_and(jnp.not_equal(state.level.death_frame_counter, 0), has_collision_enemy),
                lambda _: jnp.zeros(self.enemy_sprites[sprite_index].shape).astype(jnp.uint8),
                lambda _: jax.lax.cond(
                            jnp.logical_or(
                            jnp.logical_and(jnp.logical_and(x >= 7, x <= 14), y == 80),
                            jnp.logical_and(jnp.logical_and(x >= 120, x <= 127), y == 80)
                            ),
                            lambda _: self.enemy_tp_sprites[sprite_index],
                            lambda _: self.enemy_sprites[sprite_index],
                            operand=None
                            ),
                                operand=None
                                )

            enemy_sprite = colorize_sprite(enemy_sprite, color)
            
            flipped_enemy_sprite = jax.lax.cond(
                jnp.logical_or(last_horizontal_orientations == JAXAtariAction.LEFT, last_horizontal_orientations == JAXAtariAction.LEFT),
                lambda s: jnp.flip(s, axis=0),
                lambda s: s,
                enemy_sprite
            )
            
            # Handles blinking of the enemy sprite            
            blinking_enemy_sprite = jax.lax.cond( i == 2,
                                                 lambda _: jax.lax.cond(jnp.logical_and(jnp.mod(state.level.frame_count,2) ==1,state.enemies.multiple_enemies.blink[i]),
                                                                        lambda _: jnp.zeros(flipped_enemy_sprite.shape).astype(jnp.uint8),
                                                                        lambda _: flipped_enemy_sprite,
                                                                        operand=None),
                                                 lambda _: jax.lax.cond(jnp.logical_and(jnp.mod(state.level.frame_count,2) ==0,state.enemies.multiple_enemies.blink[i]),
                                                                        lambda _: jnp.zeros(flipped_enemy_sprite.shape).astype(jnp.uint8),
                                                                        lambda _: flipped_enemy_sprite,
                                                                        operand=None),
                                                 operand=None)
        
            res = self._render_at(canvas, y, x, blinking_enemy_sprite).astype(jnp.float32)
            
            return jax.lax.cond(state.enemies.multiple_enemies.enemy_spawn_frame[i] > 9, lambda x: canvas,lambda x: res, 0)
            
        # Render loop for eggs, handles color & hiding of already collected eggs
        def render_loop_eggs(i, l_r, canvas):
            x = state.eggs[l_r, i , 0]
            y = state.eggs[l_r, i , 1]
            egg_sprite = self.egg_sprites[state.eggs[l_r, i , 3]]
            render_egg = lambda canv : self._render_at(canv, y, x, egg_sprite)
            render_no_egg = lambda canv : self._render_at(canv, y, x, jnp.zeros(self.egg_sprites[0].shape))
            
            rendered_c = jax.lax.cond(state.eggs[l_r, i, 2], render_egg, render_no_egg, canvas)
            return rendered_c.astype(jnp.float32)
        def render_loop_items(i, canvas):
            x = state.items[i , 0]
            y = state.items[i , 1]
            item_sprite_identifier =  state.items[i , 3]
            sprite_index =  (state.level.frame_count // 8) % 4
            def pad_array(array: jnp.ndarray) -> jnp.ndarray:
                if array.shape[0] == 9:
                    # Padding: ((oben,unten), (links,rechts), (channel))
                    paddings = ((0,1), (0,0), (0,0))
                    # Mit Nullen auffüllen
                    padded_array = jnp.pad(array, paddings, mode='constant', constant_values=0)
                    return padded_array
                return array

            item_sprite = jax.lax.cond(
                item_sprite_identifier == 4,
                lambda x: pad_array(self.evil_item_sprites[x[1]]),
                lambda x: self.items[x[0]],
                operand= (item_sprite_identifier, sprite_index)
            )
            
            # Handles blinking of items
            blinking_item_sprite = jax.lax.cond(item_sprite_identifier == 4,
                                                lambda _: jax.lax.cond(jnp.logical_and(jnp.mod(state.level.frame_count,2) == 0,state.level.blink_evil_item),
                                                                       lambda _: jnp.zeros(item_sprite.shape).astype(jnp.uint8),
                                                                       lambda _: item_sprite,
                                                                       operand=None),
                                                lambda _: jax.lax.cond(jnp.logical_and(jnp.mod(state.level.frame_count,2) == 0,state.level.blink_current_active_item),
                                                                       lambda _: jnp.zeros(item_sprite.shape).astype(jnp.uint8),
                                                                       lambda _: item_sprite,
                                                                       operand=None),
                                                operand=None)
            
            render_item = lambda canv : self._render_at(canv, y, x, blinking_item_sprite)
            render_no_item = lambda canv : self._render_at(canv, y, x, jnp.zeros(blinking_item_sprite.shape))

            rendered_c = jax.lax.cond(state.items[i, 2], render_item, render_no_item, canvas)
            return rendered_c.astype(jnp.float32)

        # Render loop for lives, allwos for dynamic # of lives
        def render_loop_lifes(i, canvas):
            x = LIFE_X + i * (LIFE_WIDTH + LIFE_OFFSET_X)
            y = LIFE_Y
            return self._render_at(canvas, y, x, self.life_sprite).astype(jnp.float32)
           
        canvas = jax.lax.cond(jnp.mod(state.level.frame_count,2) ==1,
                              lambda _: jax.lax.fori_loop(0, state.eggs.shape[1] - 1, lambda cnt, rnr, cnst=0 : render_loop_eggs(i=cnt, canvas=rnr, l_r=cnst), canvas),
                              lambda _: jax.lax.fori_loop(0, state.eggs.shape[1] - 1, lambda cnt, rnr, cnst=1 : render_loop_eggs(i=cnt, canvas=rnr, l_r=cnst), canvas),
                              operand=None)

        canvas = jax.lax.fori_loop(0, state.items.shape[0], render_loop_items, canvas)
        canvas = jax.lax.cond(state.level.frame_count < 100, lambda x: x, lambda x: jax.lax.fori_loop(0, 3, render_loop_alien, x), canvas)
        canvas = jax.lax.fori_loop(0, state.level.lifes, render_loop_lifes, canvas)

        def player_death_animation(state: AlienState) -> jnp.ndarray:
            sprite_index = (state.level.death_frame_counter // 4) % 10
            return self.player_death_sprites[sprite_index]
        sprite_index = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(state.player.x >= 7, state.player.x <= 11), state.player.y == 80),
            lambda _: sprite_cycle_1[(state.player.x - 7) % len(sprite_cycle_1)],
            lambda _: jax.lax.cond(
                jnp.logical_and(jnp.logical_and(state.player.x >= 123, state.player.x <= 127), state.player.y == 80),
                lambda _: sprite_cycle_2[(state.player.x - 123) % len(sprite_cycle_2)],
                lambda _: (state.level.frame_count // 4) % 4,  # Standardanimation
                operand=None
            ),
            operand=None
        )
        # Handles player walk or player death animation.
        player_sprite_frame = jax.lax.cond(jnp.not_equal(state.level.death_frame_counter, 0), lambda x: player_death_animation(x), lambda x:jax.lax.cond(
            jnp.logical_or(
                jnp.logical_and(jnp.logical_and(state.player.x >= 7, state.player.x <= 11), state.player.y == 80),
                jnp.logical_and(jnp.logical_and(state.player.x >= 123, state.player.x <= 127), state.player.y == 80)
            ),
            lambda _: self.teleport_sprites[sprite_index].astype(jnp.float32),
            lambda _: colorize_sprite(self.player_sprite[(state.level.frame_count // 4) % 4], jax.lax.cond(state.player.flame.flame_flag,
                                                                                                           lambda _: FLAMETHROWER_COLOR,
                                                                                                           lambda _: DEFAULT_PLAYER_COLOR,
                                                                                                           operand=None)).astype(jnp.float32),
            operand=None
        ), state)
        
        # Handles horizontal flipping of the player sprite depending on the last horizontal direction (preserve sprite orientation during vertical movement)
        flipped_player_sprite = jax.lax.cond(
            jnp.logical_or(state.player.last_horizontal_orientation == JAXAtariAction.LEFT, state.player.orientation == JAXAtariAction.LEFT),
            lambda s: jnp.flip(s, axis=0),
            lambda s: s,
            player_sprite_frame
        )
        
        # Handles blinking of player sprite
        blinking_player_sprite = jax.lax.cond(jnp.logical_or(state.player.flame.flame_flag,state.player.blink),
                                           lambda _: jax.lax.cond(jnp.mod(state.level.frame_count,2) == 0,
                                                                  lambda _ :flipped_player_sprite,
                                                                  lambda _ : jnp.zeros(flipped_player_sprite.shape).astype(np.float32),
                                                                  operand=None),
                                           lambda _: flipped_player_sprite,
                                           operand=None)
        
        # rendering of player sprite
        canvas = self._render_at(canvas, state.player.y, state.player.x, blinking_player_sprite)
        
        # Handles orientation of the flame sprite
        flipped_flame_sprite = jax.lax.cond(
            jnp.logical_or(state.player.last_horizontal_orientation == JAXAtariAction.RIGHT, state.player.orientation == JAXAtariAction.RIGHT),
            lambda s: jnp.flip(s, axis=0),
            lambda s: s,
            self.flame_sprite
        )
        
        # Handles blinking of flame sprite
        blinking_flame_sprite = jax.lax.cond(jnp.logical_and(state.player.flame.flame_flag,jnp.mod(state.level.frame_count,2) == 1),
                                             lambda _: flipped_flame_sprite.astype(np.float32),
                                             lambda _: jnp.zeros(flipped_flame_sprite.shape).astype(np.float32),
                                             operand=None)
        
        # rendering of the flamethrower
        canvas = self._render_at(canvas, state.player.flame.y,state.player.flame.x,blinking_flame_sprite)
        
        digit_sprite:jnp.ndarray = self.get_score_sprite(score=state.level.score)
        canvas = self._render_at(canvas, SCORE_Y, SCORE_X, digit_sprite)
        canvas = jnp.transpose(canvas, (1, 0, 2))
        return canvas[..., 0:3]

#def traverse_multiple_enemy(enemy_state: MultipleEnemiesState, fun: Callable[[SingleEnemyState], Any],
#                            returns_enemies_state: bool = False, number_of_non_traversed_args: int = 0, *args):
#    if returns_enemies_state:
#        def wrapper_function(x: jnp.ndarray, y: jnp, orientation: jnp.ndarray, mode: jnp.ndarray, last_horizontal_orientation: jnp.ndarray, enemy_spawn_frame: jnp.ndarray, key: jnp.ndarray, *args):
#            oof = SingleEnemyState(x, y, orientation, mode, last_horizontal_orientation, enemy_spawn_frame, key)
#            res: SingleEnemyState = fun(oof, *args)
#            return res.x, res.y, res.orientation, res.mode, res.last_horizontal_orientation, res.enemy_spawn_frame, res.key
#
#    else:
#        def wrapper_function(x: jnp.ndarray, y: jnp, orientation: jnp.ndarray, mode: jnp.ndarray, last_horizontal_orientation: jnp.ndarray, enemy_spawn_frame, key, *args):
#            oof = SingleEnemyState(x, y, orientation, last_horizontal_orientation, enemy_spawn_frame, key, *args)
#            res = fun(oof)
#            return res
#    if number_of_non_traversed_args>0:
#        n_s = (None, )*number_of_non_traversed_args
#        in_axes_tuple = (0, 0, 0, 0, 0, 0, 0,*n_s)
#    else:
#        in_axes_tuple = (0, 0, 0, 0, 0, 0, 0)
#
#    if returns_enemies_state:
#        a, b, c, d, e, f, g = jax.vmap(wrapper_function, in_axes=in_axes_tuple, out_axes=(0, 0, 0, 0, 0, 0, 0))(enemy_state.x, enemy_state.y, enemy_state.orientation, enemy_state.modes, enemy_state.last_horizontal_orientation, enemy_state.enemy_spawn_frame_counters, enemy_state.keys, *args)
#        ret = MultipleEnemiesState(a, b, c, d, e, f, g)
#        return ret
#    else:
#        ret = jax.vmap(wrapper_function, in_axes=in_axes_tuple, out_axes=(0))(enemy_state.x, enemy_state.y, enemy_state.orientation, enemy_state.modes, enemy_state.last_horizontal_orientation, enemy_state.enemy_spawn_frame_counters, enemy_state.keys, *args)
#        return ret




def traverse_multiple_enemy(enemy_state: MultipleEnemiesState, fun: Callable[[SingleEnemyState], Any],
                            returns_enemies_state: bool = False, number_of_non_traversed_args: int = 0, *args):
   
    fields = list(SingleEnemyState._fields)
    fields.sort()
    if returns_enemies_state:
        def wrapper_function(*args):
            k_dict: Dict[str, jnp.ndarray] = {}
            for i, f in enumerate(fields):
                k_dict[f] = args[i]
            oof = SingleEnemyState(**k_dict)
            res: SingleEnemyState = fun(oof, *args[len(fields):])
            ret_tup: Tuple[jnp.ndarray] = ()
            for f in fields:
                ret_tup = (*ret_tup, getattr(res, f))
            return ret_tup

    else:
        def wrapper_function(*args):
            k_dict: Dict[str, jnp.ndarray] = {}
            for i, f in enumerate(fields):
                k_dict[f] = args[i]
            oof = SingleEnemyState(**k_dict)
            
            res = fun(oof, *args[len(fields):])
            return res
    state_axes = (0, )*len(fields)
    if number_of_non_traversed_args>0:
        n_s = (None, )*number_of_non_traversed_args
        in_axes_tuple = (*state_axes,*n_s)
    else:
        in_axes_tuple = state_axes
    inputs = ()
    for f in fields:
        inputs = (*inputs, getattr(enemy_state, f))
    inputs = (*inputs, *args)
    if returns_enemies_state:
        r_args = jax.vmap(wrapper_function, in_axes=in_axes_tuple, out_axes=state_axes)(*inputs)
        m_args = {}
        for i, f in enumerate(fields):
            m_args[f] = r_args[i]            
        ret = MultipleEnemiesState(**m_args)
        return ret
    else:
        ret = jax.vmap(wrapper_function, in_axes=in_axes_tuple, out_axes=(0))(inputs)
        return ret


traverse_enemy_step = lambda x, y: traverse_multiple_enemy(x, enemy_step, True, 1, y)
jitted_enemy_step= jax.jit(traverse_enemy_step)


if __name__ == "__main__":
    # Initialize Pygamepython
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * RENDER_SCALE_FACTOR, HEIGHT * RENDER_SCALE_FACTOR))
    pygame.display.set_caption("Alien")
    clock = pygame.time.Clock()
    
    game = JaxAlien()	
    renderer = AlienRenderer()

    _, curr_state = game.reset()
     # Canvas initialisieren




# Game loop
    running = True
    frame_by_frame = False
    counter = 1
    frameskip = 1

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
                        _, curr_state, _, _, _ = game.step()

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                _, curr_state, _, _, _ = game.step(state=curr_state, action=action)

        # Render and display
        raster = renderer.render(state=curr_state)

        jr.update_pygame(screen, raster, RENDER_SCALE_FACTOR , WIDTH, HEIGHT)

        counter += 1
        clock.tick(30)

    pygame.quit()
