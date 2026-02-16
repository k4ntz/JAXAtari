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
from typing import NamedTuple, Tuple, Any, Callable, Dict
import jax.numpy as jnp
import chex
from jaxatari.renderers import JAXGameRenderer
from gymnax.environments import spaces
import jaxatari.spaces as spaces
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction
import jax
import numpy as np
from jax import Array as jArray
from flax import struct

from jaxatari.environment import ObjectObservation

# Palette Constants (RGB Tuples)
COLORS = {
    'BASIC_BLUE': (132, 144, 252),  # Player, Blue Egg, UI
    'ORANGE':     (252, 144, 144),  # Flame, Evil Item, Orange Egg
    'PINK':       (236, 140, 224),  # Pink Enemy
    'GREEN':      (132, 252, 212),  # Green Enemy
    'YELLOW':     (252, 252, 84),   # Yellow Enemy, Items
    'FRIGHTENED': (101, 111, 228),  # "Other Blue" / Killable Enemy
}

def get_alien_asset_config():
    return [
        # --- Backgrounds ---
        {'name': 'map_primary', 'type': 'background', 'file': 'bg/map_sprite.npy'},
        {'name': 'map_bonus',   'type': 'single',     'file': 'bg/bonus_map_sprite.npy'},

        # --- Player ---
        {
            'name': 'player_walk',
            'type': 'group',
            'files': ['player_animation/player1.npy', 
                      'player_animation/player2.npy', 
                      'player_animation/player3.npy',
                      'player_animation/player2.npy'], # Added 4th frame (ping-pong) to match teleport shape
            'recolorings': {
                'normal': COLORS['BASIC_BLUE'],
                'flame':  COLORS['ORANGE']
            }
        },
        {
            'name': 'player_death',
            'type': 'group',
            'files': ['player_death_animation/player_death_1_sprite.npy',
                      'player_death_animation/player_death_2_sprite.npy',
                      'player_death_animation/player_death_3_sprite.npy',
                      'player_death_animation/player_death_4_sprite.npy'],
            'recolorings': {'normal': COLORS['BASIC_BLUE']}
        },
        {
            'name': 'player_teleport',
            'type': 'group',
            'files': ['player_teleport_animation/teleport1.npy',
                      'player_teleport_animation/teleport2.npy',
                      'player_teleport_animation/teleport3.npy',
                      'player_teleport_animation/teleport4.npy'],
            'recolorings': {'normal': COLORS['BASIC_BLUE']}
        },

        # --- Flame ---
        {
            'name': 'flame',
            'type': 'single',
            'file': 'flame/flame_sprite.npy',
            'recolorings': {'normal': COLORS['ORANGE']}
        },

        # --- Enemies ---
        {
            'name': 'enemy_walk',
            'type': 'group',
            'files': ['enemy_animation/enemy_walk1.npy',
                      'enemy_animation/enemy_walk2.npy',
                      'enemy_animation/enemy_walk3.npy',
                      'enemy_animation/enemy_walk2.npy'], # Added 4th frame (ping-pong) to match teleport shape
            'recolorings': {
                'pink':       COLORS['PINK'],
                'yellow':     COLORS['YELLOW'],
                'green':      COLORS['GREEN'],
                'frightened': COLORS['FRIGHTENED']
            }
        },
        {
            'name': 'enemy_teleport',
            'type': 'group',
            'files': ['enemy_teleport_animation/1.npy',
                      'enemy_teleport_animation/2.npy',
                      'enemy_teleport_animation/3.npy',
                      'enemy_teleport_animation/4.npy'],
            'recolorings': {
                'pink':       COLORS['PINK'],
                'yellow':     COLORS['YELLOW'],
                'green':      COLORS['GREEN'],
                'frightened': COLORS['FRIGHTENED']
            }
        },
        {
            'name': 'enemy_death',
            'type': 'group',
            'files': ['alien_death_animation/alien_death1.npy',
                      'alien_death_animation/alien_death2.npy',
                      'alien_death_animation/alien_death3.npy',
                      'alien_death_animation/alien_death4.npy'],
            'recolorings': {'normal': COLORS['FRIGHTENED']}
        },

        # --- Items ---
        {
            'name': 'evil_item',
            'type': 'group',
            'files': ['items/evil_item_1.npy', 'items/evil_item_2.npy'],
            'recolorings': {
                'normal': COLORS['ORANGE'],
                'bonus_green': COLORS['GREEN'] 
            }
        },
        {
            'name': 'score_items', 
            'type': 'group',
            'files': ['items/pulsar.npy', 'items/rocket.npy', 
                      'items/saturn.npy', 'items/starship.npy', 
                      'items/orb.npy',    'items/pi.npy'], 
            'recolorings': {'normal': COLORS['YELLOW']}
        },

        # --- Eggs ---
        {
            'name': 'egg',
            'type': 'single',
            'file': 'egg/egg.npy',
            'recolorings': {
                'yellow': COLORS['YELLOW'],
                'orange': COLORS['ORANGE'],
                'blue':   COLORS['BASIC_BLUE'],
                'pink':   COLORS['PINK'],
                'green':  COLORS['GREEN']
            }
        },
        {
            'name': 'egg_half',
            'type': 'single',
            'file': 'egg/half_egg.npy',
            'recolorings': {
                'yellow': COLORS['YELLOW'],
                'orange': COLORS['ORANGE'],
                'blue':   COLORS['BASIC_BLUE'],
                'pink':   COLORS['PINK'],
                'green':  COLORS['GREEN']
            }
        },

        # --- UI / HUD ---
        {
            'name': 'digits',
            'type': 'digits',
            'pattern': 'digits/{}.npy',
            'recolorings': {'normal': COLORS['BASIC_BLUE']}
        },
        {
            'name': 'life',
            'type': 'single',
            'file': 'life/life_sprite.npy',
            'recolorings': {'normal': COLORS['BASIC_BLUE']}
        }
    ]

#Defines Observation of Alien, where we also need to know if a enemy is killable and where the current items are
class AlienObservation(struct.PyTreeNode):
    player: ObjectObservation
    enemies: ObjectObservation
    enemies_killable: jnp.ndarray
    kill_item_position: jnp.ndarray
    score_item_position: jnp.ndarray
    collision_map: jnp.ndarray

#Defines the Info of Alien, which is score, step counter and all rewards
class AlienInfo(struct.PyTreeNode):
    score: jnp.ndarray
    step_counter: jnp.ndarray
    
class AlienConstants(struct.PyTreeNode):
    SEED: int = struct.field(pytree_node=False, default=42) #seed for randomness
    RENDER_SCALE_FACTOR: int = struct.field(pytree_node=False, default=4)
    MODULE_DIR: str = struct.field(pytree_node=False, default=os.path.dirname(os.path.abspath(__file__)))
    WIDTH: int = struct.field(pytree_node=False, default=160) # Width of the playing field in game state
    HEIGHT: int = struct.field(pytree_node=False, default=210) # height of the playing field in game state
    PLAYER_VELOCITY: int = struct.field(pytree_node=False, default=1)
    ENEMY_VELOCITY: int = struct.field(pytree_node=False, default=1)
    #player sprite dimensions
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=8)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=13)
    # player initial position
    PLAYER_X: int = struct.field(pytree_node=False, default=67)
    PLAYER_Y: int = struct.field(pytree_node=False, default=56)
    # Position at which the numbers are rendered
    SCORE_X: int = struct.field(pytree_node=False, default=19)
    SCORE_Y: int = struct.field(pytree_node=False, default=171)
    #offset for background rendering
    RENDER_OFFSET_Y: int = struct.field(pytree_node=False, default=5)
    RENDER_OFFSET_X: int = struct.field(pytree_node=False, default=8)
    DIGIT_OFFSET: int = 3 # Offset between numbers
    #digit sprite dimensions
    DIGIT_WIDTH: int = struct.field(pytree_node=False, default=6)
    DIGIT_HEIGHT: int = struct.field(pytree_node=False, default=7)

    # Position at which the life counter is rendered
    LIFE_X: int = struct.field(pytree_node=False, default=13)
    LIFE_Y: int = struct.field(pytree_node=False, default=187)
    LIFE_OFFSET_X: int = struct.field(pytree_node=False, default=2) # Offset between life sprites
    LIFE_WIDTH: int = struct.field(pytree_node=False, default=5)

    # Enemy_player_collision_offset
    ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW: int = struct.field(pytree_node=False, default=4)
    ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH: int = struct.field(pytree_node=False, default=6)
    
    # Enemy amount
    ENEMY_AMOUNT_PRIMARY_STAGE: int = struct.field(pytree_node=False, default=3)
    ENEMY_AMOUNT_BONUS_STAGE: int = struct.field(pytree_node=False, default=6)
    # Spawn position of enemies
    ENEMY_SPAWN_X: int = struct.field(pytree_node=False, default=67)
    ENEMY_SPAWN_Y: int = struct.field(pytree_node=False, default=147 + 5)
    # Enemy start position between walls
    ENEMY_START_Y: int = struct.field(pytree_node=False, default=128)
    FRIGHTENED_DURATION: int = struct.field(pytree_node=False, default=100)
    FLAME_FRIGHTENED_DURATION: int = struct.field(pytree_node=False, default=5)

    # Scatter means random movement, chase means the enemy is actively chasing the player
    # Pink enemy constants
    SCATTER_DURATION_1: int =  struct.field(pytree_node=False, default=100)
    CHASE_DURATION_1: int = struct.field(pytree_node=False, default=200)
    MODECHANGE_PROBABILITY_1: float = struct.field(pytree_node=False, default=0.5)
    SCATTER_POINT_X_1: int = struct.field(pytree_node=False, default=0)
    SCATTER_POINT_Y_1: int = struct.field(pytree_node=False, default=0)

    # Yellow enemy constants
    SCATTER_DURATION_2: int = struct.field(pytree_node=False, default=100)
    CHASE_DURATION_2: int = struct.field(pytree_node=False, default=100)
    MODECHANGE_PROBABILITY_2: float = struct.field(pytree_node=False, default=0.6)
    SCATTER_POINT_X_2: int = struct.field(pytree_node=False, default=0)
    SCATTER_POINT_Y_2: int = struct.field(pytree_node=False, default=180)

    # Green enemy constants
    SCATTER_DURATION_3: int = struct.field(pytree_node=False, default=150)
    CHASE_DURATION_3: int = struct.field(pytree_node=False, default=100)
    MODECHANGE_PROBABILITY_3: float = struct.field(pytree_node=False, default=0.7)
    SCATTER_POINT_X_3: int = struct.field(pytree_node=False, default=180)
    SCATTER_POINT_Y_3: int = struct.field(pytree_node=False, default=180)
    
    #Colors
    BASIC_BLUE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([132, 144, 252], dtype=jnp.int32)) # blue egg, lifes, player, digits
    ORANGE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([252, 144, 144], dtype=jnp.int32)) # orange egg, evil item, flamethrower
    PINK: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([236, 140, 224], dtype=jnp.int32)) # pink enemy, pink egg
    GREEN: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([132, 252, 212], dtype=jnp.int32)) # green enemy, green egg
    YELLOW: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([252, 252, 84], dtype=jnp.int32)) # yellow enemy, yellow egg, items
    OTHER_BLUE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([101, 111, 228], dtype=jnp.int32))# fleeing enemys
    
    ENEMY_COLORS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [236, 140, 224],  # PINK
        [252, 252, 84],   # YELLOW
        [132, 252, 212]   # GREEN
    ], dtype=jnp.int32))
    
    # color : 0 (yellow), 1 (orange), 2 (blue), 3 (pink), 4 (turquoise), 5 ( orange and blue)
    EGG_COLORS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [[252, 252, 84], [252, 252, 84]],      # YELLOW, YELLOW
        [[252, 144, 144], [252, 144, 144]],    # ORANGE, ORANGE
        [[132, 144, 252], [132, 144, 252]],    # BASIC_BLUE, BASIC_BLUE
        [[236, 140, 224], [236, 140, 224]],    # PINK, PINK
        [[132, 252, 212], [132, 252, 212]],    # GREEN, GREEN
        [[252, 144, 144], [132, 144, 252]]     # ORANGE, BASIC_BLUE
    ], dtype=jnp.int32))
    
    ITEM_ARRAY: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([ 
    #index: 1-> x_position 2-> y_position 3-> 1 if active 0 if not 4-> type of item -> 0:evil item, 1: pulsar, 2: rocket, 3: saturn, 4: starship
        [68, 9,  1,  0],
        [22,  129,  0,  0],
        [114, 129,  0,  0],
        [68 ,  56,  0,  1] #static score item
    ], dtype=jnp.int32))
    
    ITEM_SCORE_MULTIPLIERS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 100, 500, 1000, 2000, 3000, 5000], dtype=jnp.int32))# Score for collecting items, last item is not a score item
    ENEMY_KILL_SCORE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([500, 1000, 1500], dtype=jnp.int32)) # Score for killing enemies, depending on how many have been killed in succession

    # egg discription:
    # x-coordinate
    # y-coordinate
    # status : 1 (on the field),  0 (not on the field)
    # color : 0 (yellow), 1 (orange), 2 (blue), 3 (pink), 4 (turquoise), 5 ( orange and blue)
    EGG_ARRAY: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
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
    ]], dtype=jnp.int32))
    EGG_SCORE_MULTIPLYER: int = struct.field(pytree_node=False, default=10)

    # assets
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=get_alien_asset_config)
    

#Defines params of FlameThrower
class FlameState(struct.PyTreeNode):
    x: jnp.ndarray # x position of the flame
    y: jnp.ndarray # y position of the flame
    flame_counter: jnp.ndarray # counter for how long the flame is active
    flame_flag:jnp.ndarray  # flag for if the flame is active

# Defines params of the Player
class PlayerState(struct.PyTreeNode):
    x: jnp.ndarray # x position of the player
    y: jnp.ndarray # y position of the player
    orientation: jnp.ndarray # orientation of the player
    last_horizontal_orientation: jnp.ndarray # last horizontal orientation of the player
    flame: FlameState # state of the flamethrower
    collision_map: jnp.ndarray # collision map of the player, defines how the player collides with the environment
    bonus_collision_map: jnp.ndarray # collision map of the player in the bonus stage
    blink: jnp.ndarray # blink state of the player, used for when the player is blinking

# Defines Movement modes of the enemies
class EnemyModeConstants(struct.PyTreeNode):
    scatter_duration: jnp.ndarray # duration of the scatter mode
    scatter_point_x: jnp.ndarray # x position of the scatter point, simulates tendencys in which area aliens move
    scatter_point_y: jnp.ndarray # y position of the scatter point
    chase_duration: jnp.ndarray  # duration of the chase mode
    mode_change_probability: jnp.ndarray # probability of changing mode
    frightened_duration: jnp.ndarray # duration of the frightened mode, frightened mode is when the enemy is killable or is fleeing from flame of the player

# Defines params of a single enemy
class SingleEnemyState(struct.PyTreeNode):
    x: jnp.ndarray # x position of the enemy
    y: jnp.ndarray # y position of the enemy
    orientation: jnp.ndarray   # orientation of the enemy
    mode_type: jnp.ndarray # mode_types: 0=chase, 1=scatter, 2=frightened
    mode_duration: jnp.ndarray # duration of the current mode
    mode_constants: EnemyModeConstants # constants for the enemy modes
    last_horizontal_orientation: jnp.ndarray # last horizontal orientation of the enemy
    enemy_spawn_frame: jnp.ndarray # defines if the enemy is still in spawning sequence, if >0 the enemy is spawning and cannot collide with the player
    enemy_death_frame: jnp.ndarray  # defines if the enemy is in death animation, if >0 the enemy is dying and game is freezed
    key: jnp.ndarray # random key for the enemy to define scatter movement
    blink: jnp.ndarray # blink state of the enemy, used for when the enemy is blinking
    killable: jnp.ndarray # defines if the enemy is killable, 1=killable, 0=not killable and is activated when evil item is collected
    has_been_killed: jnp.ndarray # defines if the enemy has been killed, during evil item duration
    existence: jnp.ndarray # defines if the enemy was on screen when evil item was collected, if 0 the enemy was not on screen and will not be killable when evil item is collected
    position_at_death_x: jnp.ndarray # x position of the enemy when one of the enemies is killed, used to freeze enemies during death animation
    position_at_death_y: jnp.ndarray # y position of the enemy when one of the enemies is killed, used to freeze enemies during death animation
    active_enemy: jnp.ndarray # defines if the enemy is active, used to activate enemies especially from going from primary to bonus stage
    kill_score_flag: jnp.ndarray # defines if first, second or third enemy kill in a row, used to give score for killing enemies in a row (500, 1000, 1500)

#Deines params of all enemies, but vectors of the single enemy params
class MultipleEnemiesState(struct.PyTreeNode):
    x: jnp.ndarray
    y: jnp.ndarray
    orientation: jnp.ndarray
    mode_type: jnp.ndarray # mode_types: 0=chase, 1=scatter, 2=frightened
    mode_duration: jnp.ndarray
    mode_constants: EnemyModeConstants
    last_horizontal_orientation: jnp.ndarray
    enemy_spawn_frame: jnp.ndarray
    enemy_death_frame: jnp.ndarray
    key: jnp.ndarray
    blink: jnp.ndarray
    killable: jnp.ndarray
    has_been_killed: jnp.ndarray
    existence: jnp.ndarray
    position_at_death_x: jnp.ndarray
    position_at_death_y: jnp.ndarray
    active_enemy: jnp.ndarray
    kill_score_flag: jnp.ndarray

# Defines general state of all enemies and its parameters
class EnemiesState(struct.PyTreeNode):
    multiple_enemies: MultipleEnemiesState
    rng_key: jnp.ndarray # random key for the enemies
    collision_map: jnp.ndarray # collision map of the enemies, defines how the enemies collide with the environment
    bonus_collision_map: jnp.ndarray # collision map of the enemies in the bonus stage
    velocity: jnp.ndarray # velocity of the enemies

# Defines general state of the level and its parameters
class LevelState(struct.PyTreeNode):
    collision_map: jnp.ndarray # collision map of the level, defines how the wall collides with the player and enemies
    score: jnp.ndarray # current score of the player
    frame_count: jnp.ndarray # frame counter, used for timing events
    lifes : jnp.ndarray # current lifes of the player
    death_frame_counter: jnp.ndarray # counter for the death animation of the player
    evil_item_frame_counter: jnp.ndarray # counter for the duration of the evil item
    current_active_item_index: jnp.ndarray # index of the current active evil item
    blink_evil_item: jnp.ndarray # blink state of the evil item, used for when the evil item is blinking especially for when flame is active
    blink_current_active_item: jnp.ndarray # blink state of the current active item, used for when the current active item is blinking
    bonus_flag: jnp.ndarray # flag for if the game is in bonus stage or primary stage
    difficulty_stage: jnp.ndarray # current difficulty stage, used to increase difficulty of the game, especially velocity changes and which score item spawns
    evil_item_duration: jnp.ndarray # duration of the evil item gets shorter with increasing difficulty stage
    spawn_score_item_flag_1: jnp.ndarray #first time score item is being spawned when 41 eggs are collected
    spawn_score_item_flag_2: jnp.ndarray # second time score item is being spawned when 82 eggs are collected
    score_item_counter: jnp.ndarray # counter for how long the score item can be active

# Defines the complete state of the game
class AlienState(struct.PyTreeNode):
    player: PlayerState
    enemies: EnemiesState
    level: LevelState
    eggs: jnp.ndarray
    items: jnp.ndarray
    step_counter: jnp.ndarray # counts the number of steps taken in the game, used for statistics and info for training

# to ensure that both enemy states have the same fields
m_fs = set(MultipleEnemiesState.__dataclass_fields__.keys())
s_fs = set(SingleEnemyState.__dataclass_fields__.keys())
if not (m_fs.issubset(s_fs) and s_fs.issubset(m_fs)):
    raise Exception("Mismatch between fields in SingleEnemyState and MultipleEnemiesState")


def load_collision_map(file_path: str, transpose: bool = True, invert: bool = False) -> jnp.ndarray:
    """
    Load collision maps; matches pre-refactor behavior.
    For 3D arrays uses the first channel only (original used frame[..., 0].squeeze()).
    Result is (W, H) when transpose=True for check_for_wall_collision.
    """
    data = jnp.load(file_path)

    # Match original: use first channel only for 3D (not max over channels)
    if data.ndim == 3:
        data = data[..., 0].squeeze()

    # Transpose if requested (needed for coordinate consistency: first axis = x, second = y)
    if transpose:
        data = jnp.transpose(data, (1, 0))

    # 0 = no collision, >0 = solid (match original boolean_frame logic)
    if invert:
        out = (data == 0).astype(jnp.int32)
    else:
        out = (data > 0).astype(jnp.int32)
    return out


def _load_collision_map() -> jnp.ndarray:
    """Loads the player-enemy collision sprite for primary-stage collision checks."""
    base_dir = render_utils.get_base_sprite_dir()
    path = os.path.join(base_dir, "alien", "player_enemy_collision_sprite.npy")
    return load_collision_map(path, transpose=True)


def _load_background_collision_map() -> jnp.ndarray:
    """Loads the level/wall collision map (for observations and wall logic)."""
    base_dir = render_utils.get_base_sprite_dir()
    path = os.path.join(base_dir, "alien", "bg", "map_sprite_collision_map.npy")
    return load_collision_map(path, transpose=True)


# collision maps for enemy collision check with player for primary stage
PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK: jnp.ndarray = _load_collision_map()
# collision map for wall (level) - used in observation; game logic uses env's level_collision_map
BACKGROUND_COLLISION_MAP: jnp.ndarray = _load_background_collision_map()


@jax.jit
def check_for_wall_collision(moving_object: jnp.ndarray, background: jnp.ndarray, old_position: jnp.ndarray, new_position: jnp.ndarray) -> jnp.ndarray:
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
    ret_v = jnp.multiply(has_collision, old_position) + jnp.multiply(1- has_collision, new_position)
    return ret_v

def check_collision_between_two_sprites(sprite1, pos1_x, pos1_y, sprite2, pos2_x, pos2_y):
        # This function is only used to check collision between enemy & player in primary stage
        # We artificially make the sprites a bit smaller, so that enemies & player don't collide at the corners.
        #
        #
        r1x = pos1_x + 4 # 4 pixel offset on x axis so that sprites match position in game logic for sprite1
        r1w = jnp.shape(sprite1)[0] - 8 # 8 pixel reduction on width so that sprites are smaller for collision
        r2x = pos2_x + 4 # 4 pixel offset on x axis so that sprites match position in game logic for sprite2
        r2w = jnp.shape(sprite2)[0] - 8 # the same for sprite2 as for sprite1

        r1y = pos1_y + 6 # 6 pixel offset on y axis so that sprites match position in game logic for sprite1
        r1h = jnp.shape(sprite1)[1] - 12 # 12 pixel reduction on height so that sprites are smaller for collision
        r2y = pos2_y + 6 # 6 pixel offset on y axis so that sprites match position in game logic for sprite2
        r2h = jnp.shape(sprite2)[1] - 12 # the same for sprite2 as for sprite1
        return (
            (r1x + r1w >= r2x) & # evaluates if sprites collide
            (r1x <= r2x + r2w) &
            (r1y + r1h >= r2y) &
            (r1y <= r2y + r2h)
        )
def bonus_check_collision_between_two_sprites(sprite1, pos1_x, pos1_y, sprite2, pos2_x, pos2_y):
    # This function is only used to check collision between enemy & player in bonus stage.
    # We artificially make the sprites a bit smaller, so that enemies & player don't collide at the corners.
    #Logic is the same as in check_collision_between_two_sprites, but the offsets are different
    #
    r1x = pos1_x + 4
    r1w = jnp.shape(sprite1)[0] - 4
    r2x = pos2_x + 4
    r2w = jnp.shape(sprite2)[0] - 4

    r1y = pos1_y + 6
    r1h = jnp.shape(sprite1)[1] - 6
    r2y = pos2_y + 6
    r2h = jnp.shape(sprite2)[1] - 6
    return (
            (r1x + r1w >= r2x) &
            (r1x <= r2x + r2w) &
            (r1y + r1h >= r2y) &
            (r1y <= r2y + r2h)
    )

def collision_check_between_two_objects(obj1_map: jnp.ndarray, obj2_map: jnp.ndarray, obj1_position: jnp.ndarray, obj2_position: jnp.ndarray, background_map: jnp.ndarray) -> jnp.ndarray:
    """checks the collision between two potentialy moving objects

    Args:
        obj1_map (jnp.ndarray): Collision Map of the first object
        obj2_map (jnp.ndarray): Collision Map of the second object
        obj1_position (jnp.ndarray): position tuple of the fist object (x, y)
        obj2_position (jnp.ndarray): position tuple of the second object (x, y)
        background_map (jnp.ndarray): Collision map of the background. Has shape (Width, Height), entries with 1 indicate that a pixel has collision enabled.

    Returns:
        jnp.ndarray, dtype= bool: bolean depending on collision
    """
    canvas = jnp.zeros_like(background_map)
    
    startIndices1 = jnp.array([obj1_position[0]+ 8, obj1_position[1] + 5])#render offset
    startIndices2 = jnp.array([obj2_position[0]+ 8, obj2_position[1] + 5])#render offset

    canvas = jax.lax.dynamic_update_slice(operand=canvas, update=obj2_map, start_indices=startIndices2)
    region = jax.lax.dynamic_slice(operand=canvas, start_indices=startIndices1, slice_sizes=obj1_map.shape)
    has_collision = jnp.max(a=region, axis=None, keepdims=False)

    return has_collision

@jax.jit
def check_for_player_enemy_collision(state: AlienState, new_position: jnp.ndarray) -> jnp.ndarray:
    """Wrapper for collision_check_between_two_objects that checks all enemies
       and checks for player death in primary stage identified by checking collision with non killable enemies
    Args:
        state (AlienState): Current state of Game
        new_position: (jnp.ndarray): Proposed position of the player after the movement would be executed

     Returns:
        jnp.ndarray, dtype= bool: boolean depending on collision with the enemies


    """
    # Determine coord range occupied by the player
    player_enemy_collision: jnp.ndarray = PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK
    # Check if player sprite crosses a certain point in enemy sprite
    enemy_0_collides = check_collision_between_two_sprites(sprite1=player_enemy_collision, pos1_x=state.enemies.multiple_enemies.x[0], 
                                                           pos1_y=state.enemies.multiple_enemies.y[0], 
                                                           sprite2=player_enemy_collision, pos2_x=new_position[0], 
                                                           pos2_y=new_position[1]) # Checks collision between player and enemy 0
    enemy_1_collides = check_collision_between_two_sprites(sprite1=player_enemy_collision, pos1_x=state.enemies.multiple_enemies.x[1], 
                                                           pos1_y=state.enemies.multiple_enemies.y[1],
                                                           sprite2=player_enemy_collision, pos2_x=new_position[0], 
                                                           pos2_y=new_position[1]) # Checks collision between player and enemy 1
    enemy_2_collides = check_collision_between_two_sprites(sprite1=player_enemy_collision, pos1_x=state.enemies.multiple_enemies.x[2], 
                                                           pos1_y=state.enemies.multiple_enemies.y[2], 
                                                           sprite2=player_enemy_collision, pos2_x=new_position[0], 
                                                           pos2_y=new_position[1]) # Checks collision between player and enemy 2

    # Enemy can only collide with player if it is not in spawn animation (enemy_spawn_frame == 0) and if it is not killable (killable == 0)
    has_collision_enemy0 = jnp.logical_and(enemy_0_collides, jnp.logical_and(jnp.equal(state.enemies.multiple_enemies.killable[0], 0), jnp.equal(state.enemies.multiple_enemies.enemy_spawn_frame[0],0)))

    has_collision_enemy1 = jnp.logical_and(enemy_1_collides, jnp.logical_and(jnp.equal(state.enemies.multiple_enemies.killable[1], 0), jnp.equal(state.enemies.multiple_enemies.enemy_spawn_frame[1],0)))

    has_collision_enemy2 =jnp.logical_and(enemy_2_collides, jnp.logical_and(jnp.equal(state.enemies.multiple_enemies.killable[2], 0), jnp.equal(state.enemies.multiple_enemies.enemy_spawn_frame[2],0)))

    return jnp.logical_or(has_collision_enemy0, jnp.logical_or(has_collision_enemy1, has_collision_enemy2)) # True if player collides with any non killable enemy

@staticmethod
@jax.jit
def teleport_object(position: jnp.ndarray, orientation: JAXAtariAction, action: JAXAtariAction):
    """changes the position of the given object on the teleportation logic

    Args:
        position (jnp.ndarray): position tuple of the object (x, y)
        orientation (JAXAtariAction): Orientation of the object
        action (JAXAtariAction): player action

    Returns:
        _type_: new position
    """
    # activates teleportation if the object is at one of the two teleportation points and is moving towards the wall
    x = position[0]
    ac_right = JAXAtariAction.RIGHT
    ac_left = JAXAtariAction.LEFT
    # conditions for teleportation
    cond_1 = jnp.logical_or(jnp.logical_and(x >= 127, orientation == ac_right),jnp.logical_and(action == ac_right, x >= 127)) #right teleport
    cond_2 = jnp.logical_or(jnp.logical_and(x <= 7,orientation == ac_left),jnp.logical_and(action == ac_left, x <= 7)) #left teleport

    new_val_code = cond_1 + jnp.multiply(2 ,cond_2) # 0 = no teleport, 1 = right teleport, 2 = left teleport

    new_position0 = jnp.multiply(x ,(new_val_code == 0)) + jnp.multiply(7 , (new_val_code == 1)) + jnp.multiply(127 ,(new_val_code == 2))
    return position.at[0].set(new_position0)

# Main Environment Class
class JaxAlien(JaxEnvironment[AlienState, AlienObservation, AlienInfo, AlienConstants]):
    # Minimal ALE action set (from scripts/action_space_helper.py)
    ACTION_SET: jnp.ndarray = jnp.array(
        [
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
            JAXAtariAction.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: AlienConstants = None):
        consts = consts or AlienConstants()
        super().__init__(consts)
        
        # 1. Initialize Renderer
        self.renderer = AlienRenderer(consts=self.consts)

        # 2. Initialize Logic Steps
        traverse_enemy_step = traverse_multiple_enemy_(partial(enemy_step, cnsts=consts), 1)
        self.enemy_step = jax.jit(traverse_enemy_step)
        
        traverse_enemy_step_bonus =  traverse_multiple_enemy_(partial(enemy_step_bonus, cnsts=consts), 1)
        self.enemy_step_bonus = jax.jit(traverse_enemy_step_bonus)
        
        # 3. Setup Sprite Paths (Using new utils)
        # This ensures we look in ~/.local/share/jaxatari/sprites/alien/
        self.sprite_path: str = os.path.join(render_utils.get_base_sprite_dir(), "alien")

        # 4. Load Collision Maps
        # We use the generic helper we defined above
        self.player_collision_map: jnp.ndarray = load_collision_map(
            os.path.join(self.sprite_path, "player_animation", "player_sprite_collision_map.npy"), 
            transpose=True
        )
        self.player_bonus_collision_map: jnp.ndarray = load_collision_map(
            os.path.join(self.sprite_path, "player_animation", "player_sprite_bonus_collision_map.npy"), 
            transpose=True
        )
        self.level_collision_map: jnp.ndarray = load_collision_map(
            os.path.join(self.sprite_path, "bg", "map_sprite_collision_map.npy"),
            transpose=True
        )
        self.enemy_collision_map: jnp.ndarray = load_collision_map(
            os.path.join(self.sprite_path, "player_animation", "player_sprite_collision_map.npy"), 
            transpose=True
        )
        self.enemy_bonus_collision_map: jnp.ndarray = load_collision_map(
            os.path.join(self.sprite_path, "enemy_animation", "enemy_sprite_bonus_collision_map.npy"), 
            transpose=True
        )
    
    def render(self, state: AlienState) -> jnp.ndarray:
        return self.renderer.render(state)
    
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))


    def reset(self, rng_key: jArray) -> Tuple[AlienObservation, AlienState]:
        """
        Resets the game state to the initial state.
        """
        key: jax.random.PRNGKey = rng_key # generate random key
        keys = jax.random.split(key, self.consts.ENEMY_AMOUNT_BONUS_STAGE + 1) # split key for each enemy + 1
        main_key = keys[0]  # main key for the game
        subkeys = keys[1:] # subkeys for each enemy

        # Initialize player state
        player_state = PlayerState(
            x=jnp.array(self.consts.PLAYER_X).astype(jnp.int32),
            y=jnp.array(self.consts.PLAYER_Y).astype(jnp.int32),
            orientation=JAXAtariAction.UP,
            last_horizontal_orientation=JAXAtariAction.RIGHT,
            flame= FlameState(
                x=jnp.array(self.consts.PLAYER_X - 8).astype(jnp.int32),
                y=jnp.array(self.consts.PLAYER_Y + 5).astype(jnp.int32),
                flame_counter=jnp.array(90).astype(jnp.int32),
                flame_flag=jnp.array(0).astype(jnp.int32),
            ),
            collision_map=self.player_collision_map,
            bonus_collision_map=self.player_bonus_collision_map,
            blink=jnp.array(0).astype(jnp.int32)
        )

        # Enemy mode constants padded to match the number of enemies in the bonus stage with zeros
        mode_const = EnemyModeConstants(
            scatter_duration=jnp.pad(jnp.array([self.consts.SCATTER_DURATION_1, self.consts.SCATTER_DURATION_2, self.consts.SCATTER_DURATION_3]),
                                     (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0),
            scatter_point_x=jnp.pad(jnp.array([self.consts.SCATTER_POINT_X_1, self.consts.SCATTER_POINT_X_2, self.consts.SCATTER_POINT_X_3]), 
                                    (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0),
            scatter_point_y=jnp.pad(jnp.array([self.consts.SCATTER_POINT_Y_1, self.consts.SCATTER_POINT_Y_2, self.consts.SCATTER_POINT_Y_3]),
                                    (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0),
            chase_duration=jnp.pad(jnp.array([self.consts.CHASE_DURATION_1, self.consts.CHASE_DURATION_2, self.consts.CHASE_DURATION_3]),
                                   (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0),
            mode_change_probability=jnp.pad(jnp.array([self.consts.MODECHANGE_PROBABILITY_1, self.consts.MODECHANGE_PROBABILITY_2, self.consts.MODECHANGE_PROBABILITY_3]),
                                            (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0),
            frightened_duration=jnp.pad(jnp.array([self.consts.FRIGHTENED_DURATION, self.consts.FRIGHTENED_DURATION, self.consts.FRIGHTENED_DURATION]),
                                        (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0)
            )

        # Variables are padded to match the bonus stage
        enemies_state = EnemiesState(
            multiple_enemies = MultipleEnemiesState(
                x=jnp.full((self.consts.ENEMY_AMOUNT_BONUS_STAGE), self.consts.ENEMY_SPAWN_X, dtype=jnp.int32),
                y=jnp.full((self.consts.ENEMY_AMOUNT_BONUS_STAGE), self.consts.ENEMY_SPAWN_Y, dtype=jnp.int32),
                orientation=jnp.full((self.consts.ENEMY_AMOUNT_BONUS_STAGE), JAXAtariAction.RIGHT),
                mode_type=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                mode_duration=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                mode_constants=mode_const,
                last_horizontal_orientation=jnp.full(self.consts.ENEMY_AMOUNT_BONUS_STAGE, JAXAtariAction.RIGHT),
                enemy_spawn_frame=jnp.pad(jnp.array([11, 21, 31]).astype(jnp.int32), # delayed enemy spawns for primary stage enemies
                                          (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0),
                enemy_death_frame=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                key=subkeys,
                blink=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                killable=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                has_been_killed=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                existence = jnp.zeros(self.consts.ENEMY_AMOUNT_BONUS_STAGE, dtype=jnp.int32),
                position_at_death_x= jnp.full((self.consts.ENEMY_AMOUNT_BONUS_STAGE), self.consts.ENEMY_SPAWN_X, dtype=jnp.int32),
                position_at_death_y= jnp.full((self.consts.ENEMY_AMOUNT_BONUS_STAGE), self.consts.ENEMY_SPAWN_Y, dtype=jnp.int32),
                active_enemy=jnp.pad(jnp.array([1, 1, 1]).astype(jnp.int32), # aöö three enemies are active at start
                                          (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0),
                kill_score_flag=jnp.ones((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32)
            ),
            rng_key=main_key,
            collision_map=self.enemy_collision_map,
            bonus_collision_map=self.enemy_bonus_collision_map,
            velocity=jnp.array(1).astype(jnp.int32) # initial velocity of the enemies
        )
        
        level_state = LevelState(
            collision_map=self.level_collision_map,
            score=jnp.array([0]).astype(jnp.uint16),
            frame_count=jnp.array(0).astype(jnp.int32),
            lifes=jnp.array(2).astype(jnp.int32), # player starts with 3 lifes
            death_frame_counter=jnp.array(0).astype(jnp.int32),
            evil_item_frame_counter=jnp.array(0).astype(jnp.int32),
            current_active_item_index=jnp.array(0).astype(jnp.int32),
            blink_evil_item=jnp.array(0).astype(jnp.int32),
            blink_current_active_item=jnp.array(0).astype(jnp.int32),
            bonus_flag=jnp.array(0).astype(jnp.int32),
            difficulty_stage=jnp.array(0).astype(jnp.float32),
            evil_item_duration=jnp.array(505).astype(jnp.int32), # initial duration of the evil item
            spawn_score_item_flag_1=jnp.array(1).astype(jnp.int32),
            spawn_score_item_flag_2=jnp.array(1).astype(jnp.int32),
            score_item_counter=jnp.array(0).astype(jnp.int32)
        )

        reset_state = AlienState(
            player=player_state,
            enemies=enemies_state,
            level= level_state,
            eggs=self.consts.EGG_ARRAY,
            items= self.consts.ITEM_ARRAY,
            step_counter=jnp.array(0).astype(jnp.int32)
        )
        
        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state
    
    
    @partial(jax.jit, static_argnums=(0))
    def start_bonus_stage(self, state: AlienState) -> AlienState:
        """initiates the bonus stage"""
        _, new_state = self.reset(state.enemies.rng_key)
        
        # init new start player position with other collision map for bonus stage
        new_player_state = new_state.player.replace(
            x=jnp.array(self.consts.PLAYER_X).astype(jnp.int32),
            y=jnp.array(self.consts.ENEMY_START_Y + 20).astype(jnp.int32),
            last_horizontal_orientation=JAXAtariAction.LEFT,
            collision_map=self.player_bonus_collision_map
        )
        
        # Choose random enemies start position from 3 presets
        main_key, sub_key = jax.random.split(state.enemies.rng_key)
        random_value = jax.random.randint(sub_key, shape=(), minval=0, maxval=3)
        presets = jnp.array([[70, 20, 80], [10, 80, 140], [40, 100, 110]])
        
        # Position 6 enemies in a diagonal line
        diagonal_spacing = 20
        start_y = 30
        enemy_x_positions = jnp.array([presets[random_value][0], presets[random_value][0]+30+5, presets[random_value][1], presets[random_value][2], presets[random_value][2]-30-5, presets[random_value][1]+30-5], dtype=jnp.int32)
        enemy_y_positions = jnp.arange(self.consts.ENEMY_AMOUNT_BONUS_STAGE, dtype=jnp.int32) * diagonal_spacing + start_y
        
        # init new enemies position with other collision map for bonus stage
        new_multiple_enemies = state.enemies.multiple_enemies.replace(
            x=enemy_x_positions,
            y=enemy_y_positions,
            last_horizontal_orientation=jnp.array([JAXAtariAction.LEFT, JAXAtariAction.RIGHT, JAXAtariAction.LEFT, JAXAtariAction.RIGHT, JAXAtariAction.LEFT, JAXAtariAction.RIGHT], dtype=jnp.int32),
            enemy_spawn_frame=jnp.zeros(self.consts.ENEMY_AMOUNT_BONUS_STAGE, dtype=jnp.int32)
        )
        new_enemies_state = state.enemies.replace(
            multiple_enemies=new_multiple_enemies,
            collision_map=self.enemy_bonus_collision_map,
            rng_key=main_key
        )
        
        new_level_state = new_state.level.replace(
            score=state.level.score + 1,
            lifes=state.level.lifes,
            bonus_flag=jnp.array(1).astype(jnp.int32),
            # FIX: Use int32 to match the rest of the game logic
            collision_map=jnp.zeros(new_state.level.collision_map.shape, dtype=jnp.int32),
            difficulty_stage=state.level.difficulty_stage
        )
        
        new_state = new_state.replace(
            player=new_player_state,
            enemies=new_enemies_state,
            level=new_level_state
        )
        
        return new_state
        
    @partial(jax.jit, static_argnums=(0))
    def start_primary_stage(self, state: AlienState) -> AlienState:
        """initiates the primary stage
        Args:
            state (AlienState): Current state of the game 
        Returns:
            AlienState: the next game frame
        """
        _, reset_state = self.reset(state.enemies.rng_key)
        
        new_level_state = reset_state.level.replace(
            lifes=state.level.lifes,
            score=state.level.score,
            difficulty_stage=state.level.difficulty_stage + 0.5, # increase difficulty stage after each bonus stage
            evil_item_duration=jnp.maximum((-63 * state.level.difficulty_stage + 505), 0).astype(jnp.int32) # decrease evil item duration with increasing difficulty stage
        )
        new_enemies_state = reset_state.enemies.replace(
            rng_key=state.enemies.rng_key
        )
        new_state = reset_state.replace(level=new_level_state, enemies=new_enemies_state)
        return new_state
        
    @partial(jax.jit, static_argnums=(0))
    def normal_game_step(self, state: AlienState, action: chex.Array) -> Tuple[AlienState, AlienObservation, AlienInfo]:
        """Normal game step. This is the function that is called when the game is running normally.
        Args:
            state (AlienState): Current state of the game
            action (chex.Array): player action
        Returns:
            Tuple[AlienState, AlienObservation, AlienInfo]: New state, observation and info
        """
        new_player_state =  self.player_step(state, action) # player step
        new_enemies_state = self.multiple_enemies_step(state) # enemy step
        new_egg_state, new_score = self.egg_step(
            new_player_state.x,
            new_player_state.y,
            state.eggs,
            state.level.score,
            self.consts.EGG_SCORE_MULTIPLYER
        )

        # lose a life if collision with enemy and not already in death animation
        condition_new_life = jnp.logical_and(check_for_player_enemy_collision(state, [new_player_state.x, new_player_state.y]),jnp.equal(state.level.death_frame_counter, 0))
        new_life = jnp.multiply(condition_new_life, jnp.add(state.level.lifes, -1)) + jnp.multiply(1 - condition_new_life, state.level.lifes)

        # start death animation if life lost and checks if game over or soft reset depending on if lives are left
        condition1 = jnp.logical_and(condition_new_life, jnp.less(new_life, 0))
        condition2 = jnp.logical_and(condition_new_life, jnp.greater_equal(new_life,0))
        new_death_frame_counter = state.level.death_frame_counter + jnp.multiply(condition1, -40) + jnp.multiply(condition2, 40)

        # update items
        new_items, new_score, new_current_active_item_index, new_evil_item_frame_counter, new_spawn_score_item_flag_1, new_spawn_score_item_flag_2, new_score_item_counter = self.item_step(
            player_x=new_player_state.x,
            player_y=new_player_state.y,
            score=new_score,
            state=state
        )
        new_level_state = state.level.replace(
            score=new_score,
            lifes=new_life,
            death_frame_counter=new_death_frame_counter,
            evil_item_frame_counter=new_evil_item_frame_counter,
            current_active_item_index=new_current_active_item_index,
            spawn_score_item_flag_1=new_spawn_score_item_flag_1,
            spawn_score_item_flag_2=new_spawn_score_item_flag_2,
            score_item_counter=new_score_item_counter
        )
        
        new_state = state.replace(
            player=new_player_state,
            enemies=new_enemies_state,
            level=new_level_state,
            eggs=new_egg_state,
            items=new_items
        )
        
        return new_state
    
    
    @partial(jax.jit, static_argnums=(0)) 
    def death(self, state: AlienState) -> Tuple[AlienState, AlienObservation, AlienInfo]:
        """ Initiates death animation and handles soft reset if lives are left
        Args:
            state (AlienState): Current state of the game
        Returns:
            Tuple[AlienState, AlienObservation, AlienInfo]: New state, observation and info
        """

        # count down death frame counter during death animation because it is 40 at the start of the death animation
        new_death_frame_counter = state.level.death_frame_counter - 1


        freeze_level_state = state.level.replace(
            death_frame_counter=new_death_frame_counter,
            blink_evil_item=jnp.array(0).astype(jnp.int32),
            blink_current_active_item=jnp.array(0).astype(jnp.int32)
            )

        # state is frozen during death animation except for death frame counter
        freeze_state = state.replace(
            level=freeze_level_state
            )
        
        _, reset_state = self.reset(state.enemies.rng_key)
        
        soft_reset_multiple_enemies = reset_state.enemies.multiple_enemies.replace(
            kill_score_flag=state.enemies.multiple_enemies.kill_score_flag
            )
        
        soft_reset_enemies = reset_state.enemies.replace(
            rng_key=state.enemies.rng_key,
            multiple_enemies=soft_reset_multiple_enemies
            )
        soft_reset_level_state = reset_state.level.replace(
            death_frame_counter=new_death_frame_counter,
            score=state.level.score,
            lifes=state.level.lifes,
            frame_count=state.level.frame_count,
            current_active_item_index=state.level.current_active_item_index,
            evil_item_frame_counter=jnp.array(0).astype(jnp.int32),
            blink_evil_item=jnp.array(0).astype(jnp.int32),
            blink_current_active_item=jnp.array(0).astype(jnp.int32),
            difficulty_stage=state.level.difficulty_stage,
            evil_item_duration=state.level.evil_item_duration,
            spawn_score_item_flag_1=state.level.spawn_score_item_flag_1,
            spawn_score_item_flag_2=state.level.spawn_score_item_flag_2,
            )
        
        items_keep_evil_reset_score = state.items.at[3, 2].set(0)

        # create soft reset state with preserved score, lives, difficulty stage and item state
        soft_reset_state = reset_state.replace(
            enemies=soft_reset_enemies,
            level=soft_reset_level_state,
            eggs=state.eggs,
            items=items_keep_evil_reset_score
            )

        # if death frame counter is 1, switch to soft reset state, else stay in freeze state
        new_state = jax.lax.cond(
            jnp.equal(state.level.death_frame_counter, 1),
            lambda x, y: x,
            lambda x, y: y,
            soft_reset_state, freeze_state
        )
        return new_state

    #step for death animation with hard reset
    @partial(jax.jit, static_argnums=(0))
    def game_over(self, state: AlienState) -> Tuple[AlienState, AlienObservation, AlienInfo]:
        """ Handles hard reset and death animation if no lives are left
        Returns:
            Tuple[AlienState, AlienObservation, AlienInfo]: New state, observation and info
        """

        # count up death frame counter during game over animation because it is -40 at the start of the game over animation
        new_death_frame_counter = jnp.add(state.level.death_frame_counter, 1)

        freeze_level_state = state.level.replace(
            death_frame_counter=new_death_frame_counter,
            current_active_item_index=jnp.array(0).astype(jnp.int32),
            blink_evil_item=jnp.array(0).astype(jnp.int32),
            blink_current_active_item=jnp.array(0).astype(jnp.int32),
            )

        # state is frozen during game over animation except for death frame counter
        freeze_state = state.replace(
            level=freeze_level_state
            )
        
        _, reset_state = self.reset(state.enemies.rng_key)

        hard_reset_enemies = reset_state.enemies.replace(
            rng_key=state.enemies.rng_key
            )
        
        hard_reset_state = reset_state.replace(
            enemies=hard_reset_enemies,
        )

        # if death frame counter is -1, switch to hard reset state, else stay in freeze state
        new_state = jax.lax.cond(jnp.equal(state.level.death_frame_counter, -1),
                                 lambda x, y: x , lambda x, y: y , 
                                 hard_reset_state, freeze_state)
        return new_state
            
    
    @partial(jax.jit, static_argnums=(0))
    def _choose_gamestep_besides_kill_item_step(self, state: AlienState, action: chex.Array):
        """Handels logic on possible life loss (normal_game_step, death, or game_over), when kill item is not active.

        Args:
            state (AlienState): Current state of the game
            action (chex.Array): player action

        Returns:
            _type_: AlienState: the next game frame
        """
        is_alive: jArray = jnp.equal(state.level.death_frame_counter, 0)
        is_normal_death: jArray = jnp.greater(state.level.death_frame_counter, 0)
        is_game_over: jArray = jnp.less(state.level.death_frame_counter, 0)

        # evaluates which step is chosen
        choice_array: jArray = jnp.array([1*is_alive, 2*is_normal_death, 3*is_game_over], jnp.int32)
        actual_choice: jArray = jnp.max(choice_array, keepdims=False)

        # based on evaluation choose step to evaluate step
        new_state: AlienState = jax.lax.switch(actual_choice,
                                               [lambda x, y : x, 
                                                self.normal_game_step, 
                                                lambda x, y: self.death(x), 
                                                lambda x, y: self.game_over(x)], 
                                               state, action)
        
        return new_state
        #lambda x:  jax.lax.cond(jnp.equal(state.level.death_frame_counter, 0),
        #                                    lambda y: self.normal_game_step(y, action),
        #                                    lambda y: jax.lax.cond(jnp.greater(state.level.death_frame_counter, 0),
        #                                                    lambda z: self.death(z),
        #                                                    lambda z: self.game_over(z),
        #                                                    y
        #                                                    ),
        #                                x
        #                                )
    @partial(jax.jit, static_argnums=(0))
    def primary_step(self, state: AlienState, action: chex.Array) -> AlienState:
        """step function for the primary stage
        Args:
            state (AlienState): Current state of the game 
            action (chex.Array): player action
        Returns:
            AlienState: the next game frame
        """

        #initial state for cond
        new_state = state
        
        #cond for checking for normal game step or death or game over
        new_state = jax.lax.cond(jnp.greater(state.level.evil_item_frame_counter, 0),
                                    lambda x, y: self.kill_item_step(x, y),
                                    self._choose_gamestep_besides_kill_item_step,
                                new_state, action
                                )

        # decides to go into bonus stage if all eggs are picked up
        new_state = jax.lax.cond(
                jnp.sum(state.eggs[:, :, 2]) == 0,
                lambda x: self.start_bonus_stage(x),
                lambda x: x,
                new_state
            )


        return new_state
        
        

        
    @partial(jax.jit, static_argnums=(0))
    def kill_item_step(self, state: AlienState, action: jArray) -> Tuple[AlienState, AlienObservation, AlienInfo]:
        """ Handle updating of game step during active kill item
        Args:
            state (AlienState): Current State of the Game
        Returns:
            Tuple[AlienState, AlienObservation, AlienInfo]: New state, observation and info
        """

        new_player_state: PlayerState =  self.player_step(state, action)

        # lose a life if collision with enemy and not already in death animation
        condition_new_life = jnp.logical_and(check_for_player_enemy_collision(state, [new_player_state.x, new_player_state.y]),jnp.equal(state.level.death_frame_counter, 0))
        new_life = jnp.multiply(condition_new_life, state.level.lifes-1) + jnp.multiply(1 - condition_new_life, state.level.lifes)

        # start death animation if life lost and checks if game over or soft reset depending on if lives are left
        condition1 = jnp.logical_and(condition_new_life, jnp.less(new_life, 0))
        condition2 = jnp.logical_and(condition_new_life, jnp.greater_equal(new_life,0))
        new_death_frame_counter = state.level.death_frame_counter + jnp.multiply(condition1, -40) + jnp.multiply(condition2, 40)

        # is to signalise that an enemy was killed to raise score during this kill item period
        new_kill_score_flag = jnp.where(state.enemies.multiple_enemies.enemy_death_frame > 0, 0, state.enemies.multiple_enemies.kill_score_flag)
        score_interation = 2 - jnp.sum(new_kill_score_flag[:3])
        score_increase_kill = jnp.sum((state.enemies.multiple_enemies.kill_score_flag * self.consts.ENEMY_KILL_SCORE[score_interation])*(state.enemies.multiple_enemies.enemy_death_frame > 0)).astype(jnp.uint16)
        new_score_kill = state.level.score + score_increase_kill


        freeze_level_state = state.level.replace(
                lifes=new_life,
                death_frame_counter=new_death_frame_counter,
                score=new_score_kill
        )
        new_enemies: EnemiesState = self.multiple_enemies_step(state)
        freeze_multiple_enemies = state.enemies.multiple_enemies.replace(
                    x=new_enemies.multiple_enemies.x,
                    y=new_enemies.multiple_enemies.y,
                    orientation=new_enemies.multiple_enemies.orientation,
                    enemy_spawn_frame=new_enemies.multiple_enemies.enemy_spawn_frame,
                    enemy_death_frame=new_enemies.multiple_enemies.enemy_death_frame,
                    blink=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                    killable=new_enemies.multiple_enemies.killable,
                    has_been_killed=new_enemies.multiple_enemies.has_been_killed,
                    existence=new_enemies.multiple_enemies.existence,
                    position_at_death_x=new_enemies.multiple_enemies.position_at_death_x,
                    position_at_death_y=new_enemies.multiple_enemies.position_at_death_y,
                    kill_score_flag=new_kill_score_flag
                )
        
        freeze_player_state = new_player_state.replace(
            x=state.player.x,
            y=state.player.y
        )
        
        freeze_enemies = state.enemies.replace(
                multiple_enemies=freeze_multiple_enemies
        )
        freeze_state = state.replace(
            player=freeze_player_state,
            enemies=freeze_enemies,
            level=freeze_level_state,
        )
        
        kill_multiple_enemies = new_enemies.multiple_enemies.replace(
                blink=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                position_at_death_x= new_enemies.multiple_enemies.position_at_death_x,
                position_at_death_y=new_enemies.multiple_enemies.position_at_death_y
            )
        kill_enemies = new_enemies.replace(
                multiple_enemies=kill_multiple_enemies
            )
        new_egg_state, new_score = self.egg_step(
                new_player_state.x,
                new_player_state.y,
                state.eggs,
                state.level.score,
                self.consts.EGG_SCORE_MULTIPLYER
                )
        new_items, new_score, new_current_active_item_index, new_evil_item_frame_counter, new_spawn_score_item_flag_1, new_spawn_score_item_flag_2, new_score_item_counter = self.item_step(
            player_x=new_player_state.x,
            player_y=new_player_state.y,
            score=new_score,
            state=state
        )
        level_state = state.level.replace(
                score=new_score,
                lifes=new_life,
                death_frame_counter=new_death_frame_counter,
                evil_item_frame_counter=jnp.add(new_evil_item_frame_counter, -1),
                current_active_item_index= new_current_active_item_index,
                blink_evil_item=state.level.blink_evil_item,
                blink_current_active_item=state.level.blink_current_active_item,
                spawn_score_item_flag_1=new_spawn_score_item_flag_1,
                spawn_score_item_flag_2=new_spawn_score_item_flag_2,
                score_item_counter=new_score_item_counter
                )
        kill_state = state.replace(
                player=new_player_state,
                enemies=kill_enemies,
                level=level_state,
                eggs=new_egg_state,
                items=new_items
                )
        # conditions for player death or game over
        is_player_dead: jArray = jnp.not_equal(new_death_frame_counter, 0)
        is_game_over: jArray = jnp.less(new_death_frame_counter, 0)
        
        # Handle player death here
        death_index: jArray = 0 + is_player_dead + is_game_over
        # If 0: player is still alive
        # If 1: player is dead but game is not over
        # If 2: Player is dead but game is over!
        new_state = jax.lax.switch(death_index, 
                                   [lambda x: x, 
                                    self.death, 
                                    self.game_over], 
                                   freeze_state)
        # Now Handling the game freeze for when enemies dies
        needs_to_enter_kill_state: jArray = jnp.multiply(jnp.equal(new_death_frame_counter, 0),
                                                         jnp.equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 0)) 
        new_state = jax.lax.cond(needs_to_enter_kill_state, lambda x, y : y, lambda x, y: x, new_state, kill_state)
        
        return new_state
 
    
    @partial(jax.jit, static_argnums=(0))
    def bonus_room_player_step(self, state: AlienState, action: chex.Array) -> PlayerState:
        """player step funtion in the bonus stage
        Args:
            state (AlienState): Current state of the game
            action (chex.Array): player action
        Returns:
            PlayerState: player information
        """

        # defines which action is taken based on inputs which is breaken down into up and down
        moving_action = jnp.multiply(jnp.greater(action,9), jnp.mod(action,10)+2) + jnp.multiply(jnp.less_equal(action,9), action)
        position = jnp.array([state.player.x, state.player.y])

        #limits velocity to move 1 position per frame at max but decides rather how frequent it moves 1 position so either +1/-1 or 0 frequency
        #117/249 is the approximated average of how frequent players move in bonus and 117*2 is due to vertical being twice as fast as horizontal speed
        velocity_vertical = jax.lax.min(jnp.round((((117*2)/249))*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117*2/249))*(state.level.frame_count - 1)).astype(jnp.int32),1)

        # gives back the new vertical position
        new_y_position: jArray = jnp.multiply(jnp.equal(moving_action, JAXAtariAction.UP), position[1]-velocity_vertical) + jnp.multiply(
            jnp.equal(moving_action, JAXAtariAction.DOWN), position[1] + velocity_vertical
        ) + jnp.multiply((1 - jnp.logical_or(jnp.equal(moving_action, JAXAtariAction.UP), jnp.equal(moving_action, JAXAtariAction.DOWN))), 
                         position[1])
        new_position = position.at[1].set(new_y_position)

        # upper limit on vertical position
        max_y = self.consts.ENEMY_START_Y + 20
        bounded_y = jnp.clip(new_position[1], None, max_y)
        
        new_player_state = state.player.replace(y=bounded_y)
        
        return new_player_state

    @partial(jax.jit, static_argnums=(0))
    def multiple_enemies_step_bonus(self, state: AlienState) -> EnemiesState:
        """sets the velocity for the aliens this frame
        Args:
            state (AlienState): Current State of the Game
        Returns:
            EnemiesState: information on the enemies
        """
        #limits velocity to move 1 position per frame at max but decides rather how frequent it moves 1 position so either +1/-1 or 0 frequency
        #56/21 is the approximated average of how frequent enemies move in bonus
        new_enemy_velocity_this_frame = jax.lax.min(jnp.round(((56/121))*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((56/121))*(state.level.frame_count - 1)).astype(jnp.int32),1)
        
        new_enemies = state.enemies.replace(velocity=new_enemy_velocity_this_frame)
        return new_enemies.replace(multiple_enemies=self.enemy_step_bonus(state.enemies.multiple_enemies, state))


    @partial(jax.jit, static_argnums=(0))
    def bonus_step_freeze(self, state: AlienState) -> AlienState:
        """step funtion to handle an game freeze on death in the bonus stage for animation to run and afterwards start the primary stage
        Args:
            state (AlienState): Current State of the Game
        Returns:
            AlienState: the next game frame
        """

        #counts down the death frame counter and freeze counter to 0
        new_death_frame_counter = jnp.clip(state.level.death_frame_counter - 1, 0, None)
        new_freeze_counter = jnp.clip(state.level.evil_item_frame_counter - 1, 0, None)
        new_level = state.level.replace(death_frame_counter=new_death_frame_counter)
        new_level = new_level.replace(evil_item_frame_counter=new_freeze_counter)
        new_state = state.replace(level=new_level)
        
        #freeze until animation is done and then start primary stage
        return jax.lax.cond(
            jnp.logical_or(
                jnp.logical_and(new_level.death_frame_counter == 0,state.level.death_frame_counter == 1),
                jnp.logical_and(new_level.evil_item_frame_counter == 0,state.level.evil_item_frame_counter == 1)
                ),
            self.start_primary_stage,
            lambda x: x,
            new_state
        )

    @partial(jax.jit, static_argnums=(0))
    def normal_bonus_step(self, state: AlienState, action: chex.Array) -> AlienState:
        """Normal bonus room game step.
        Args:
            state (AlienState): Current state of the game
            action (chex.Array): player actio
        Returns:
            AlienState: the next game frame
        """


        new_enemies_state = self.multiple_enemies_step_bonus(state)
        new_player_state = self.bonus_room_player_step(state, action)
        
        # dead_or_alive is 0 if alive, 1 if item is collected, 2 if collision with enemy or framecunt > 500 in bonus
        dead_or_alive = jnp.max(jnp.array([
                (jnp.sum(state.enemies.multiple_enemies.enemy_death_frame) > 0)*2,
                (state.level.frame_count > 500)*2,
                state.player.y < 14 + 5
            ]))
        # set new_death_frame_counter to 40 if dead_or_alive is 2 else to 0
        new_death_frame_counter = (jnp.clip((dead_or_alive - 1), 0, 1) * 40).astype(jnp.int32)
        # set new_freeze_counter to 40 if dead_or_alive is 1 else to 0
        new_freeze_counter = (jnp.mod(dead_or_alive, 2) * 40).astype(jnp.int32)
        # increase score if item(score of item is increased with later stages) is collected (when dead_or_alive is 1)
        new_score = state.level.score.at[0].set((state.level.score[0] + ((jnp.mod(dead_or_alive, 2)) * self.consts.ITEM_SCORE_MULTIPLIERS[jnp.clip(state.level.difficulty_stage.astype(jnp.int32) + 2, 0, len(self.consts.ITEM_SCORE_MULTIPLIERS) - 1)]).astype(jnp.uint16)))
        
        new_level = state.level.replace(death_frame_counter=new_death_frame_counter,
                                         evil_item_frame_counter=new_freeze_counter,
                                         score=new_score)                
        new_state = state.replace(player=new_player_state,
                                   enemies=new_enemies_state,
                                   level=new_level)
        
        return new_state

            
            
            
    @partial(jax.jit, static_argnums=(0))      
    def bonus_step(self, state: AlienState, action: chex.Array) -> AlienState:
        """step funtion for the bonus stage
        Args:
            state (AlienState): Current state of the game 
            action (chex.Array): player action
        Returns:
            AlienState: the next game frame
        """
        
        
        # freeze game if death animation or freeze counter(evil_frame_counter on collected item) is active else do normal bonus step
        return jax.lax.cond(jnp.logical_and(state.level.death_frame_counter == 0, state.level.evil_item_frame_counter == 0),
                                    lambda x, y: self.normal_bonus_step(x, y),
                                    lambda x, y: self.bonus_step_freeze(x),
                                    state, action
                                    )
            



    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AlienState, action: chex.Array) -> Tuple[AlienObservation, AlienState,float, bool,  AlienInfo]:
        """the main step function for transitioning to a new state

        Args:
            state (AlienState): Current state of the game
            action (chex.Array): player action

        Returns:
            Tuple[AlienObservation, AlienState,float, bool,  AlienInfo]: New state, observation and info
        """
        # Translate compact agent action index to ALE console action
        action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        # Choose Step here:
        is_in_bonus: jArray = state.level.bonus_flag
        is_in_regular: jArray = jnp.logical_and(1 - state.level.bonus_flag, jnp.greater(state.level.frame_count, 50))
        is_still_frozen: jArray = jnp.logical_and(1 - state.level.bonus_flag, jnp.less_equal(state.level.frame_count, 50))
        # Compute selected: 
        # 0 - Doesn't happen unless error
        # 1 - in bonus
        # 2 - in regular
        # 3 - still frozen
        choice_index: jArray = jnp.max(jnp.array([is_in_bonus*1, is_in_regular*2, is_still_frozen*3], jnp.int32), keepdims=False)
        new_state: AlienState = jax.lax.switch(choice_index, 
                    [lambda x, y: x, 
                     self.bonus_step, 
                     self.primary_step, 
                     lambda x, y: x], 
                    state, action)

        # signalizes that one frame has passed
        new_level_state = new_state.level.replace(frame_count=new_state.level.frame_count + 1)
        new_state = new_state.replace(level=new_level_state,
                                       step_counter= jnp.add(state.step_counter,1))

        # Here is info and observation secured
        observation = self._get_observation(new_state)
        env_reward = self._get_env_reward(state,new_state)
        done = self._get_done(new_state)
        alieninfo = self._get_info(new_state)
                
        return   (observation, new_state, env_reward, done, alieninfo)


    def get_action_space(self) -> jnp.ndarray:
        # calls the action set
        return self.ACTION_SET

    def image_space(self) -> spaces.Box:
        # Define the observation space for images
        return spaces.Box(
            low=0, # Minimum pixel value (black)
            high=255, # Maximum pixel value (white)
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    def observation_space(self) -> spaces.Dict:
        # Define the observation space as a dictionary of multiple components
        # Use object-centric spaces for player and enemies while keeping boolean fields separate
        enemy_count = int(self.consts.ENEMY_AMOUNT_BONUS_STAGE)
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH), orientation_range=(0.0, 17.0)),
            "enemies": spaces.get_object_space(n=enemy_count, screen_size=(self.consts.HEIGHT, self.consts.WIDTH), orientation_range=(0.0, 17.0)),
            "enemies_killable":
                spaces.Box(low=0, high=1, shape=(enemy_count,), dtype=jnp.int32),
            "kill_item_position":
                spaces.Box(low=0, high=self.consts.HEIGHT, shape=(2,), dtype=jnp.int32),
            "score_item_position":
                spaces.Box(low=0, high=self.consts.HEIGHT, shape=(2,), dtype=jnp.int32),
            "collision_map":
                spaces.Box(low=0, high=1, shape=(152, 188), dtype=jnp.int32),
        })

    def _get_observation(self, state: AlienState) -> AlienObservation:
        """returns the observation to a given state

        Args:
            state (AlienState): the state of the Game

        Returns:
            AlienObservation: observation for the ai agent
        """
        # Get the position of the currently active "kill item" (first 2 coordinates: x, y)
        new_kill_item_position = jnp.take(self.consts.ITEM_ARRAY, state.level.current_active_item_index, axis=0)[:2]

        # Build object-centric observations
        enemy_count = int(self.consts.ENEMY_AMOUNT_BONUS_STAGE)

        player_obj = ObjectObservation.create(
            x=jnp.array(state.player.x, dtype=jnp.int32),
            y=jnp.array(state.player.y, dtype=jnp.int32),
            width=jnp.array(self.consts.PLAYER_WIDTH, dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_HEIGHT, dtype=jnp.int32),
            active=jnp.array(1, dtype=jnp.int32),
            visual_id=jnp.array(0, dtype=jnp.int32),
            orientation=jnp.array(state.player.orientation, dtype=jnp.int32),
        )

        enemies_obj = ObjectObservation.create(
            x=state.enemies.multiple_enemies.x,
            y=state.enemies.multiple_enemies.y,
            width=jnp.full((enemy_count,), self.consts.PLAYER_WIDTH, dtype=jnp.int32),
            height=jnp.full((enemy_count,), self.consts.PLAYER_HEIGHT, dtype=jnp.int32),
            active=state.enemies.multiple_enemies.active_enemy,
            visual_id=jnp.zeros((enemy_count,), dtype=jnp.int32),
            orientation=state.enemies.multiple_enemies.orientation,
        )

        return AlienObservation(
            player=player_obj,
            enemies=enemies_obj,
            enemies_killable=state.enemies.multiple_enemies.killable,
            kill_item_position=new_kill_item_position,
            score_item_position=jnp.array([68, 56], dtype=jnp.int32), # Hardcoded position of the score item
            collision_map=jnp.astype(BACKGROUND_COLLISION_MAP, jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AlienState) -> AlienInfo:
        """Returns auxiliary information about the game state.

        Args:
            state (AlienState): state of the game

        Returns:
            AlienInfo: the information
        """
        return AlienInfo(
            score=state.level.score, # Current game score
            step_counter = state.step_counter, # Number of steps taken in the environment
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: AlienState, state: AlienState) -> float:
        """returns the reward change relative to the last frame

        Args:
            previous_state (AlienState): previous game state
            state (AlienState): current game state

        Returns:
            float: difference of score
        """
        # Compute score difference between current and previous state
        score = state.level.score - previous_state.level.score

        return score[0]
    
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AlienState, state: AlienState) -> float:
        """returns the reward change relative to the last frame

        Args:
            previous_state (AlienState): previous game state
            state (AlienState): current game state

        Returns:
            float: difference of score
        """

        # Compute score difference between current and previous stat
        score = state.level.score - previous_state.level.score
        return score[0]


    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AlienState) -> bool:
        """indicates game over

        Args:
            state (AlienState): state of the game

        Returns:
            bool: done
        """
        # Game ends when the number of lives drops below zero
        return state.level.lifes <= -1

    @partial(jax.jit, static_argnums=(0,))
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
            _type_: augmented egg array and the new score
        """
        # Determine coord range occupied by the player
        x_lower: jnp.ndarray = player_x
        x_higher: jnp.ndarray = jnp.add(player_x, self.consts.PLAYER_WIDTH)
        y_lower: jnp.ndarray = player_y
        y_higher: jnp.ndarray = jnp.add(player_y, self.consts.PLAYER_HEIGHT)

        def check_egg_player_collision(i, l_r):
            """Function that fills out the current egg_collision_map
                for an index i, checks whether the i-th egg currently collides with the player and sets
                the i-th index in the collision map to one

            Args:
                i (_type_): index
                l_r (_type_): indicates the side of the array (left or right)
                egg_col_map (_type_): array of all eggs on this side

            Returns:
                _type_: augmented egg array based on collision
            """
             
            has_collision = jnp.logical_and(egg_state[l_r, i, 0]>= x_lower,
                                           jnp.logical_and(egg_state[l_r, i, 0] < x_higher,
                                           jnp.logical_and(egg_state[l_r, i, 1] >= y_lower,
                                                           egg_state[l_r, i, 1] < y_higher)))
            return has_collision.astype(jnp.int32)
        
        # Generate full collision map for all eggs
        egg_collision_map_left = jax.vmap(lambda i: check_egg_player_collision(i, 0))(jnp.arange(egg_state.shape[1]))
        egg_collision_map_right = jax.vmap(lambda i: check_egg_player_collision(i, 1))(jnp.arange(egg_state.shape[1]))
        # Multiply with current active egg-state to prevent the same egg from being collected twice
        score_increas_left: jnp.ndarray = jnp.sum(jnp.multiply(egg_collision_map_left, egg_state[0,:, 2]))
        score_increas_right: jnp.ndarray = jnp.sum(jnp.multiply(egg_collision_map_right, egg_state[1,:, 2]))
        # Multiply collision map onto egg-state to set visible attribute to the appropriate value
        egg_collision_map_left = jnp.subtract(1, egg_collision_map_left)
        egg_collision_map_right = jnp.subtract(1, egg_collision_map_right)
        # Update egg presence maps based on collisions and current visibility
        new_egg_presence_map_left: jnp.ndarray = jnp.multiply(egg_collision_map_left, egg_state[0,:, 2]).astype(jnp.int32)
        new_egg_presence_map_right: jnp.ndarray = jnp.multiply(egg_collision_map_right, egg_state[1,:, 2]).astype(jnp.int32)
        # Update the egg_state array to reflect collected eggs (0 = collected, 1 = still present)
        egg_state = egg_state.at[0,:, 2].set(new_egg_presence_map_left)
        egg_state = egg_state.at[1,:, 2].set(new_egg_presence_map_right)
        # Calculate new score by adding points for eggs collected on both sides
        new_score: jnp.ndarray = jnp.add(score, jnp.add(jnp.multiply(score_increas_left, egg_score_multiplyer),jnp.multiply(score_increas_right, egg_score_multiplyer)))
        new_score = new_score.astype(jnp.uint16)

        return egg_state, new_score
    @partial(jax.jit, static_argnums=(0, ))
    def check_item_player_collision(self, i, batched_val: Tuple[Any]):
        """checks for collision between an item and the player
        Args:
            i (_type_): index
            single_item_col_map (_type_): collision map for a single item
        Returns:
            _type_: augmented collision map for a single item
        """
        # Unpack batched values for this step
        state, player_x, player_y = batched_val

        # Define the rectangular bounding box of the player
        x_lower: jnp.ndarray = player_x
        x_higher: jnp.ndarray = jnp.add(player_x, self.consts.PLAYER_WIDTH)
        y_lower: jnp.ndarray = player_y
        y_higher: jnp.ndarray = jnp.add(player_y, self.consts.PLAYER_HEIGHT)

        # Check if the current item overlaps with the player's bounding box
        has_collision = jnp.logical_and(state.items[i, 0]>= x_lower,
                                        jnp.logical_and(state.items[i, 0] < x_higher,
                                                        jnp.logical_and(state.items[i, 1] >= y_lower,
                                                                        state.items[i, 1] < y_higher)))

        # Only consider collision if the item is active (state.items[i, 2] == 1)
        has_collision = jnp.logical_and(has_collision, state.items[i, 2] == 1)

        # Disable collision for "evil" items when the player's flamethrower is active
        has_collision = jnp.logical_and(has_collision, 1 - jnp.logical_and(state.items[i, 3] == 0 ,state.player.flame.flame_flag))

        return has_collision.astype(jnp.int32)

    def spawn_score_item(self, items, state: AlienState):
        """Spawns a score item on the field.

           Args:
               items (jnp.ndarray): Current items array in the game.

           Returns:
               tuple:
                   - new_items (jnp.ndarray): Updated items array with the new score item.
           """


        difficulty_stage = jnp.astype(state.level.difficulty_stage, jnp.int32)
        # Update the items array for the score item:
        # Row 3, columns 2 onward (e.g., [active_flag, item_level]) are updated.
        # active_flag = 1 (item is active)
        # item_level = difficulty_stage + 1, clipped to range 1–6
        new_items = items.at[3, 2:].set(jnp.array([1, jnp.clip(difficulty_stage + 1, 1, 6)]))

        # Return the updated items array and placeholders for spawn flag and counter
        return new_items, 0, 0
    
    @partial(jax.jit, static_argnums=(0,))
    def item_step(self, player_x: jnp.ndarray, player_y: jnp.ndarray, score: jnp.ndarray, state: AlienState):
        """Handles all item-related updates in the game.

        Args:
            player_x (jnp.ndarray): Player's X coordinate.
            player_y (jnp.ndarray): Player's Y coordinate.
            score (jnp.ndarray): Current game score.
            state (AlienState): Current state of the game.

        Returns:
            tuple: Updated items array, new score, current active item index,
                   evil item frame counter, spawn flags, and score item counter.
        """

        # Initialize collision map for all items (0 = no collision, 1 = collision)
        item_collision_map: jnp.ndarray = jnp.zeros((state.items.shape[0]))
        # Determine coord range occupied by the player
        curr_active_item_idx = state.level.current_active_item_index


        # Check collisions between player and each item using a JAX vmap
        item_collision_map = jax.vmap(lambda i: self.check_item_player_collision(i, (state, player_x, player_y)))(jnp.arange(state.items.shape[0]))
        
        # Calculate score increases for items that collided with the player
        # Multiply by:
        #   - item_collision_map -> whether player collided with the item
        #   - state.items[:, 2] -> whether the item is active
        #   - ITEM_SCORE_MULTIPLIERS -> multiplier for the item type
        score_increases = jnp.multiply(
            item_collision_map,
            jnp.multiply(
                state.items[:, 2],
                self.consts.ITEM_SCORE_MULTIPLIERS[state.items[:, 3]]
            )
        )

        # Determine if a new evil item should appear
        new_evil_item_cond = jnp.logical_and(jnp.equal(item_collision_map[curr_active_item_idx], 1), jnp.less(curr_active_item_idx,3))
        new_evil_item_frame_counter = jnp.multiply(new_evil_item_cond, jnp.array(state.level.evil_item_duration).astype(jnp.int32)) + jnp.multiply(1 - new_evil_item_cond, state.level.evil_item_frame_counter)

        # Determine if new regular items should appear
        new_items_cond = jnp.logical_and(jnp.equal(item_collision_map[curr_active_item_idx], 1), jnp.less(curr_active_item_idx, 2))
        new_items = jnp.multiply(new_items_cond,state.items.at[curr_active_item_idx+1, 2].set(1)) + jnp.multiply(1 - new_items_cond, state.items)


        # Update the index of the currently active item if needed
        new_current_active_item_index_cond = jnp.equal(jnp.equal(item_collision_map[curr_active_item_idx], 1),jnp.less(curr_active_item_idx, 3))
        new_current_active_item_index = jnp.multiply(new_current_active_item_index_cond, curr_active_item_idx +1) + jnp.multiply(1 - new_current_active_item_index_cond, curr_active_item_idx)

        new_score_item_counter = state.level.score_item_counter + 1

        # Conditionally spawn special score items based on how many eggs have been collected here after 41
        new_items, new_spawn_score_item_flag_1, new_score_item_counter = jax.lax.cond(
            jnp.logical_and(
                (106 - jnp.sum(state.eggs[:, :, 2])) == 41, #41,
                state.level.spawn_score_item_flag_1
                ),
            lambda items, state, _: self.spawn_score_item(items, state),
            lambda items, state, sc: (items, state.level.spawn_score_item_flag_1, sc),
            new_items, state, new_score_item_counter
        )

        # Conditionally spawn special score items based on how many eggs have been collected here after 82
        new_items, new_spawn_score_item_flag_2, new_score_item_counter = jax.lax.cond(
            jnp.logical_and(
                (106 - jnp.sum(state.eggs[:, :, 2])) == 82, #82,
                state.level.spawn_score_item_flag_2
                ),
            lambda items, state, _: self.spawn_score_item(items, state),
            lambda items, state, sc: (items, state.level.spawn_score_item_flag_2, sc),
            new_items, state, new_score_item_counter
        )

        # Disable special score item after counter reaches 600
        new_items = jnp.multiply(new_score_item_counter == 600, new_items.at[3, 2].set(0)) + jnp.multiply(1 - (new_score_item_counter == 600), new_items)

        # Update presence map for items based on collision and active status
        item_collision_map = jnp.subtract(1, item_collision_map)
        new_item_presence_map: jnp.ndarray = jnp.multiply(item_collision_map, new_items[:, 2]).astype(jnp.int32)
        new_items = new_items.at[:, 2].set(new_item_presence_map)

        # Update total score by summing individual item score increases
        score_increase = jnp.sum(score_increases)
        new_score = jnp.add(score, score_increase).astype(jnp.uint16)

        # Return all updated item-related information
        return new_items, new_score, new_current_active_item_index, new_evil_item_frame_counter, new_spawn_score_item_flag_1, new_spawn_score_item_flag_2, new_score_item_counter
    
    @partial(jax.jit, static_argnums=(0,))
    def modify_wall_collision(self, new_position: jnp.ndarray):
        """Ensures an object stays within the bounds of the game field.

        Adjusts a position so that it cannot go beyond the playfield edges.

        Args:
            new_position (jnp.ndarray): Proposed new position of the object [x, y].

        Returns:
            jnp.ndarray: Valid position constrained within the game field.
        """

        # Define minimum permissible coordinates
        permissible_lower_bounds: jnp.ndarray = jnp.array([0, 0], dtype=np.int32)
        # Define maximum permissible coordinates
        permissible_upper_bound: jnp.ndarray = jnp.array([self.consts.WIDTH - self.consts.PLAYER_WIDTH , self.consts.HEIGHT - self.consts.PLAYER_HEIGHT ], dtype=np.int32)
        # First, ensure the new position is not below the minimum bounds
        # Stack new_position and lower bounds vertically and take element-wise max
        lower_bound_check: jnp.ndarray = jnp.vstack([new_position, permissible_lower_bounds])
        lower_checked = jnp.max(lower_bound_check, axis=0, keepdims=False)
        # Then, ensure the position is not above the maximum bounds
        # Stack lower-checked position and upper bounds, take element-wise min
        upper_bound_check: jnp.ndarray = jnp.vstack([lower_checked, permissible_upper_bound])
        checked = jnp.min(upper_bound_check, axis=0, keepdims=False)

        return checked


    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: AlienState, action: jnp.ndarray) -> PlayerState:
        """Handles the full step update for the player, including movement, orientation, and flame updates.

        Args:
            state (AlienState): Current game state.
            action (jnp.ndarray): Player action.

        Returns:
            PlayerState: Updated player information.
        """


        # limits velocity to move 1 position per frame at max but decides rather how frequent it moves 1 position so either +1/-1 or 0 frequency
        # 117/249 is the approximated average of how frequent players move in bonus and 117*2 is due to vertical being twice as fast as horizontal speed
        velocity_horizontal = jax.lax.min(jnp.round(((117/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)
        velocity_vertical = jax.lax.min(jnp.round((((117*(2))/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117*(2)/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)

        # maps the action onto the relevant movement-actions
        moving_action = jnp.multiply(jnp.greater(action,9), jnp.mod(action,10)+2) + jnp.multiply(jnp.less_equal(action,9), action)

        # Functions for setting orientation based on collision and movement
        def set_orientation(self, x, dx, dy, new_orientation):
            """
                Updates player orientation if the proposed move does not result in a collision.

                Args:
                    x: current game state
                    dx: proposed change in X direction
                    dy: proposed change in Y direction
                    new_orientation: orientation to assign if movement is valid

                Returns:
                    int: updated orientation (either new_orientation or current)
                """

            return jnp.multiply(self.other_slightly_weirder_check_for_player_collision(x, dx, dy),new_orientation) + jnp.multiply(1-self.other_slightly_weirder_check_for_player_collision(x, dx, dy),x.player.orientation)

        def set_diagonal(self, x, dx1, dy1, first_orientation, second_orientation, dx2, dy2):
            """
               Updates player orientation for diagonal movement considering two movement checks.

               Args:
                   x: current game state
                   dx1, dy1: first proposed movement delta
                   first_orientation: primary diagonal direction if first check succeeds
                   second_orientation: secondary diagonal direction if second check succeeds
                   dx2, dy2: second proposed movement delta

               Returns:
                   int: updated orientation based on diagonal movement
               """
            # Check first movement delta for collision
            cond1 = jnp.logical_and(
                self.other_slightly_weirder_check_for_player_collision(x, dx1, dy1),
                jnp.not_equal(x.player.orientation, first_orientation)
            )
            # Check second movement delta for collision
            cond2 = self.other_slightly_weirder_check_for_player_collision(x, dx2, dy2)

            # Determine new orientation based on collision results:
            # - cond1 is True → take first_orientation
            # - cond1 is False & cond2 is True → take second_orientation
            # - cond1 is False & cond2 is False → keep current orientation
            return (
                    jnp.multiply(cond1, first_orientation)
                    + jnp.multiply((1 - cond1) * cond2, second_orientation)
                    + jnp.multiply((1 - cond1) * (1 - cond2), x.player.orientation)
            )

        # Update player orientation based on movement action
        state_player_orientation = jax.lax.switch(
            moving_action,
            [
                lambda x, vel_v, vel_h: x.player.orientation,  # noop
                lambda x, vel_v, vel_h: x.player.orientation,  # noop again

                # Cardinal directions
                lambda x, vel_v, vel_h: set_orientation(self, x, 0, -vel_v, JAXAtariAction.UP),
                lambda x, vel_v, vel_h: set_orientation(self, x, vel_h, 0, JAXAtariAction.RIGHT),
                lambda x, vel_v, vel_h: set_orientation(self, x, -vel_h, 0, JAXAtariAction.LEFT),
                lambda x, vel_v, vel_h: set_orientation(self, x, 0, vel_v, JAXAtariAction.DOWN),

                # Diagonal directions
                lambda x, vel_v, vel_h: set_diagonal(self, x, vel_h, 0, JAXAtariAction.RIGHT, JAXAtariAction.UP, 0, -vel_v),   # UPRIGHT
                lambda x, vel_v, vel_h: set_diagonal(self, x, 0, -vel_v, JAXAtariAction.UP, JAXAtariAction.LEFT, -vel_h, 0),  # UPLEFT
                lambda x, vel_v, vel_h: set_diagonal(self, x, 0, vel_v, JAXAtariAction.DOWN, JAXAtariAction.RIGHT, vel_h, 0), # DOWNRIGHT
                lambda x, vel_v, vel_h: set_diagonal(self, x, -vel_h, 0, JAXAtariAction.LEFT, JAXAtariAction.DOWN, 0, vel_v), # DOWNLEFT
            ],
            state,
            velocity_vertical,
            velocity_horizontal
        )

        #Determine last horizontal orientation, this is necessary for correctly displaying the player sprite
        last_horizontal_orientation_cond = jnp.logical_or(state_player_orientation == JAXAtariAction.LEFT, state_player_orientation == JAXAtariAction.RIGHT)
        last_horizontal_orientation = jnp.multiply(last_horizontal_orientation_cond, state_player_orientation) + jnp.multiply(1 - last_horizontal_orientation_cond, state.player.last_horizontal_orientation)

        # Handle movement at this point:
        # Choose movement function according to index of proposed action.
        position = jnp.array([state.player.x, state.player.y])
        position = teleport_object(position, state.player.orientation, action)

        # Movement lambdas
        up_func = lambda x: x.at[1].subtract(velocity_vertical)
        down_func = lambda x: x.at[1].add(velocity_vertical)
        left_func = lambda x: x.at[0].subtract(velocity_horizontal)
        right_func = lambda x: x.at[0].add(velocity_horizontal)

        func_index = jnp.mod(state_player_orientation, 4).astype(jnp.uint16)
        new_position = jax.lax.switch(func_index, [left_func, down_func, up_func, right_func], position)

        #Checks for collision with the outer wall, and push player back into bounds
        new_position = self.modify_wall_collision(new_position)

        #Checks whether new position collides with the game walls, if it does so, new position is rejected
        new_position = check_for_wall_collision(
            moving_object=state.player.collision_map,
            background=state.level.collision_map,
            old_position=position,
            new_position=new_position
        )

        # Update flamethrower status and position
        # Determine whether the flamethrower should be active this step
        # Conditions for new_flame_flag to be 1 (active):
        # 1. Action triggers the flame
        # 2. Player still has flamethrower charges left (flame_counter > 0)
        # 3. Player is not in teleportation zone
        #    - Y position == 80
        #    - X position within forbidden zones (7–11 or 123–127)
        new_flame_flag = jnp.logical_and(jnp.logical_and(jnp.logical_or(jnp.greater(action,9),jnp.equal(action,1)),jnp.greater(state.player.flame.flame_counter,0)),
                                              jnp.logical_not(jnp.logical_and(state.player.y == 80,jnp.logical_or(
                                                  jnp.logical_and(state.player.x >= 7, state.player.x <= 11),
                                                  jnp.logical_and(state.player.x >= 123, state.player.x <= 127))
                                                                              ))).astype(jnp.int32)
        # Subtract the flame usage from the player's flame_counter if flame is active
        new_flame_counter = state.player.flame.flame_counter - new_flame_flag

        # Determine horizontal X-coordinate of the flame visual
        # - If last horizontal orientation was LEFT, flame appears slightly to the left of player (-6 pixels)
        # - Otherwise, flame appears slightly to the right of player (+10 pixels)
        new_flame_x_cond = jnp.equal(state.player.last_horizontal_orientation,JAXAtariAction.LEFT)
        new_flame_x = jnp.multiply(new_flame_x_cond,new_position[0] - 6) + jnp.multiply(1 - new_flame_x_cond,new_position[0] + 10)

        # Update the flamethrower state for the player
        # - x, y: position of the flame visual
        # - flame_counter: remaining flamethrower charges
        # - flame_flag: whether the flame is currently active
        new_flame = state.player.flame.replace(
            x=jnp.array(new_flame_x).astype(jnp.int32),
            y=jnp.array(new_position[1] + 6).astype(jnp.int32),
            flame_counter=jnp.array(new_flame_counter).astype(jnp.int32),
            flame_flag=jnp.array(new_flame_flag).astype(jnp.int32)
        )

        # return updated player
        return state.player.replace(
            x=jnp.array(new_position[0]),
            y=jnp.array(new_position[1]),
            orientation=state_player_orientation,
            last_horizontal_orientation=last_horizontal_orientation,
            flame=new_flame
            )
    
    @partial(jax.jit, static_argnums=(0,))
    def other_slightly_weirder_check_for_player_collision(self, state: AlienState, x_change, y_change) -> jnp.ndarray:
        """Checks for player collision. Basically just a wrapper around check_for_wall_collision for the sake of convenience.

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
        new_pos = check_for_wall_collision(
            m_object,
            bg,
            current_pos,
            new_pos
        )
        return jnp.logical_not(jnp.array_equal(new_pos, current_pos))

    @partial(jax.jit, static_argnums=(0,))
    def multiple_enemies_step(self, state: AlienState) -> EnemiesState:
        """Handles a single update step for all enemies in the game.

       Args:
           state (AlienState): Current game state.

       Returns:
           EnemiesState: Updated enemies state.
       """
        # Split RNG key for each enemy to ensure independent randomness


        main_key, *subkeys = jax.random.split(state.enemies.rng_key, self.consts.ENEMY_AMOUNT_BONUS_STAGE + 1)
        enemy_keys = jnp.array(subkeys)

        # Update the state of all multiple enemies
        new_multiple_enemies = self.enemy_step(state.enemies.multiple_enemies, state)


        new_multiple_enemies = new_multiple_enemies.replace(
            key=enemy_keys,
            )

        new_enemies_state = state.enemies.replace(
            multiple_enemies=new_multiple_enemies,
            rng_key=main_key
        )
        
        return new_enemies_state


def enemy_step(enemy: SingleEnemyState, state: AlienState, cnsts: AlienConstants)-> SingleEnemyState:
        """Handles step for a single enemy by calculating next move or animate spawning on enemy death.

         Args:
            enemy (SingleEnemyState): Current State of the Enemy
            state (AlienState): Current State of the Game
            cnsts (AlienConstants): alien constants

        Returns:
            SingleEnemyState: augmented single enemy
        """
    
        def normal_enemy_step(enemy: SingleEnemyState, state: AlienState)-> SingleEnemyState:
            """Handles the next move action of the Enemy

            Args:
                enemy (SingleEnemyState): Current State of the Enemy
                state (AlienState): Current State of the Game

            Returns:
                SingleEnemyState: augmented single enemy
            """
    
            def chase_point(point_x: jnp.ndarray, point_y: jnp.ndarray, allowed_directions: jnp.ndarray, steps_in_all_directions: jnp.ndarray):
                """Calculates next direction in the maze to a point

                Args:
                    point_x (jnp.ndarray): x coordinate of point
                    point_y (jnp.ndarray): y coordinate of point
                    allowed_directions (jnp.ndarray): directions the enemy can move in (excludes possibly opposite direction of last movement)
                    steps_in_all_directions (jnp.ndarray): positions of enemy if it would move in each direction

                Returns:
                    jnp.ndarray: next direction
                """
    
                distances = jax.vmap(lambda enemy_position: distance_to_point(enemy_position, point_x, point_y))(steps_in_all_directions)
                distances = jnp.where(allowed_directions == 1, distances, jnp.inf)
    
                return jnp.add(jnp.argmin(distances), 2)
    
            def frightened(player_x: jnp.ndarray, player_y: jnp.ndarray, allowed_directions: jnp.ndarray):
                """Frightened mode, enemy moves away from the player"""
    
                distances = jax.vmap(lambda enemy_position: distance_to_point(enemy_position, player_x, player_y))(steps_in_all_directions)
                distances = jnp.where(allowed_directions == 1, distances, 0)
    
                return jnp.add(jnp.argmax(distances), 2)
    
            def distance_to_point(enemy_position, point_x, point_y):
                """Calculates the distance between the player and the enemy"""
    
                return jnp.sqrt(jnp.square(jnp.subtract(point_x, enemy_position[0])) + jnp.square(jnp.subtract(point_y, enemy_position[1])))
    
            def new_mode_fun(enemy: SingleEnemyState):
                """
                   Determines the new mode of an enemy based on its current mode and randomness.

                   Args:
                       enemy (SingleEnemyState): Current state of a single enemy, including its mode constants and PRNG key.

                   Returns:
                       tuple:
                           - mode (int): Encodes the next mode (0 for chase or 1 for scatter)
                           - duration (int): Duration (in frames) the enemy will stay in the new mode
                   """
    
                random_value = jax.random.uniform(enemy.key, shape=(), minval=0, maxval=1)
                fun_cond = jnp.less(random_value, enemy.mode_constants.mode_change_probability)
                mode = (1 - fun_cond).astype(jnp.int32)
                duration = jnp.multiply(fun_cond, enemy.mode_constants.chase_duration) + jnp.multiply(1 - fun_cond, enemy.mode_constants.scatter_duration)
                return mode, duration
    
            def check_wall_collision(new_pos):
                """
                    Checks whether a movement to `new_pos` would result in a collision with a wall.

                    Args:
                        new_pos (jnp.ndarray): Proposed new position for an object or enemy.

                    Returns:
                        jnp.bool_: True if movement is blocked by a wall, False if movement is allowed.
                    """

                tmp_pos = check_for_wall_collision(state.enemies.collision_map, state.level.collision_map, position, new_pos)
                return jnp.logical_not(jnp.all(tmp_pos == position))
            # Tables to help with movement logic
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
            # Compute potential steps in all directions
            steps_in_all_directions = jnp.add(position, direction_offsets)
            # Compute potential steps in all directions
            allowed_directions = jax.vmap(check_wall_collision)(steps_in_all_directions).astype(jnp.int32)

            # Prevent reversing direction unless in chase mode at start of chase
            allowed_directions_no_opp = allowed_directions.at[opposite_table[jnp.subtract(enemy.orientation, 2)]].set(0)
            allowed_directions_cond = jnp.logical_and(jnp.equal(enemy.mode_type, 0), jnp.equal(enemy.mode_duration, enemy.mode_constants.chase_duration))
            allowed_directions_no_opp = jnp.multiply(allowed_directions_cond, allowed_directions) + jnp.multiply(1 - allowed_directions_cond, allowed_directions_no_opp)

            # Mode update: chase, scatter, frightened, flame frightened
            cond = jnp.equal(enemy.mode_duration, 0).astype(jnp.int32)
            mode_type_new, mode_duration_new = new_mode_fun(enemy)
            mode_type_old = enemy.mode_type
            mode_duration_old = enemy.mode_duration - 1
            new_mode_type = cond * mode_type_new + (1 - cond) * mode_type_old
            new_mode_duration = cond * mode_duration_new + (1 - cond) * mode_duration_old

            new_mode_type = 2 * enemy.killable + new_mode_type * (1- enemy.killable)

            # If player flame is nearby, frightened mode triggered
            flame_mode_cond = jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_and(
                        jnp.less(distance_to_point(position, state.player.x, state.player.y), 20),
                        state.player.flame.flame_flag.astype(jnp.bool_)),
                    jnp.logical_or(
                        jnp.equal(enemy.orientation, JAXAtariAction.RIGHT),
                        jnp.equal(enemy.orientation, JAXAtariAction.LEFT))),
                jnp.less(enemy.mode_type, 3))
            new_mode_type = jnp.multiply(flame_mode_cond, 3)  + jnp.multiply(1 - flame_mode_cond, new_mode_type)
            new_mode_duration = jnp.multiply(flame_mode_cond, cnsts.FLAME_FRIGHTENED_DURATION) + jnp.multiply(1 - flame_mode_cond, new_mode_duration)

            # Determine movement direction based on mode
            new_direction = jax.lax.switch(new_mode_type,[
                lambda _state, _enemy, _dir, _dir_noop, _all_dir: chase_point(_state.player.x, _state.player.y, _dir_noop, _all_dir), # Chase mode
                lambda _state, _enemy, _dir, _dir_noop, _all_dir: chase_point(_enemy.mode_constants.scatter_point_x, _enemy.mode_constants.scatter_point_y, _dir_noop, _all_dir), # Scatter mode
                lambda _state, _enemy, _dir, _dir_noop, _all_dir: frightened(_state.player.x, _state.player.y, _dir_noop), # Frightened mode item
                lambda _state, _enemy, _dir, _dir_noop, _all_dir: frightened(_state.player.x, _state.player.y, _dir) # Frightened mode flame
            ], state, enemy, allowed_directions, allowed_directions_no_opp, steps_in_all_directions)

            # if velocity = 1 new_direction=new_direction
            # if velocity = 0 new_direction=current_orientation
            new_direction = (
                jnp.add(
                    jnp.multiply(new_direction, state.enemies.velocity),
                    jnp.multiply(enemy.orientation, jnp.bitwise_xor(state.enemies.velocity, 1))
                )
            )


            # Determine last horizontal orientation for correct sprite display
            nlh_cond = jnp.logical_or(enemy.orientation == JAXAtariAction.LEFT, enemy.orientation == JAXAtariAction.RIGHT)
            new_last_horizontal_orientation = jnp.multiply(nlh_cond, enemy.orientation) + jnp.multiply(1 - nlh_cond, enemy.last_horizontal_orientation)

            #Teleport according to position and orientation (see teleport_object)
            position = teleport_object(position, enemy.orientation, new_direction)
            # Apply slowness if frightened (mode_type 2)
            slowness = jnp.where(jnp.equal(enemy.mode_type,2),0.75,1)
            # limits velocity to move 1 position per frame at max but decides rather how frequent it moves 1 position so either +1/-1 or 0 frequency
            # 117/249 is the approximated average of how frequent players move in bonus and 117*2 is due to vertical being twice as fast as horizontal speed
            velocity_horizontal = jax.lax.min(jnp.round(((117/249)*slowness + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117/249)*slowness + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)
            velocity_vertical = jax.lax.min(jnp.round(((117*2*slowness/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117*2*slowness/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)
            # Move enemy according to orientation
            new_position = jax.lax.switch(jnp.subtract(new_direction, 2), [
                lambda x, vel_v, vel_h: x.at[1].subtract(vel_v),  # UP
                lambda x, vel_v, vel_h: x.at[0].add(vel_h),      # RIGHT
                lambda x, vel_v, vel_h: x.at[0].subtract(vel_h), # LEFT
                lambda x, vel_v, vel_h: x.at[1].add(vel_v)       # DOWN
            ], position, velocity_vertical, velocity_horizontal)

            def check_for_enemy_player_collision(state: AlienState, new_position: jnp.ndarray) -> jnp.ndarray:
                """A wrapper for collision_check_between_two_objects containing the player and enemy

                Args:
                    state (AlienState): state of the game
                    new_position (jnp.ndarray): 

                Returns:
                    jnp.ndarray: boolean value for collision
                """
                has_collision = check_collision_between_two_sprites(sprite1=PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK, 
                                                                    pos1_x=new_position[0], 
                                                                    pos1_y=new_position[1], 
                                                                    sprite2=PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK, 
                                                                    pos2_x=state.player.x, pos2_y=state.player.y)
                # check if out of spawn
                has_collision = jnp.logical_and(
                    has_collision,
                    jnp.equal(enemy.enemy_spawn_frame,0)
                    )
                
                return has_collision


            # Determine enemy existence after spawn, by checking if on screen
            ex_cond1 = jnp.logical_and(jnp.less_equal(enemy.enemy_spawn_frame, 9), jnp.equal(state.level.evil_item_frame_counter, state.level.evil_item_duration-1))
            ex_cond2 = jnp.equal(state.level.evil_item_frame_counter,1)
            new_enemy_existence = ex_cond1 + jnp.multiply((1 - ex_cond1) ,jnp.multiply((1 - ex_cond2),enemy.existence))

            # Determine if enemy becomes killable
            new_enemy_killable = jnp.logical_and(jnp.logical_and(jnp.logical_not(enemy.has_been_killed), jnp.greater(state.level.evil_item_frame_counter, 0)), new_enemy_existence).astype(jnp.int32)

            # Determine if enemy collides with player and should enter death animation
            df_cond1 = jnp.logical_and(
                jnp.logical_and(
                    check_for_enemy_player_collision(state, new_position),
                    new_enemy_killable
                ),
                jnp.equal(enemy.enemy_death_frame, 0)
            )
            # Start death frame countdown if collision occurs, otherwise keep current death frame
            new_enemy_death_frame0 = jnp.multiply(df_cond1, 60) + jnp.multiply(1- df_cond1, enemy.enemy_death_frame)

            # Reduce death frame by 1 each game frame if active, to progress death animation
            df_cond2 = jnp.greater(new_enemy_death_frame0,0)
            new_enemy_death_frame = jnp.multiply(df_cond2, new_enemy_death_frame0- 1) + jnp.multiply(1-df_cond2, new_enemy_death_frame0)

            # Count total spawn frame offsets for all enemies
            spawn_offset_counter = jnp.sum(state.enemies.multiple_enemies.enemy_spawn_frame)

            # Masks for handling spawn logic and death end conditions
            mask_death1 = jnp.equal(new_enemy_death_frame, 1)          # first condition
            mask_evil_end = jnp.equal(state.level.evil_item_frame_counter, state.level.evil_item_duration - 1)  # second condition
            # Compute new spawn frame if enemy just died, to initiate spawn animation
            spawn_frame_if_death1 = jnp.full_like(enemy.enemy_spawn_frame, 11 + spawn_offset_counter + (spawn_offset_counter % 2))
            spawn_frame = enemy.enemy_spawn_frame


            # Update "has been killed" flag based on death animation or evil item end
            killed_if_death1 = jnp.array(1)
            killed_if_evil_end = jnp.array(0)
            killed_else = enemy.has_been_killed

            # here is decision for spawn frame based on its masks
            new_enemy_spawn_frame = (
                    jnp.multiply(mask_death1, spawn_frame_if_death1)
                    + jnp.multiply(1-mask_death1, jnp.multiply(mask_evil_end, spawn_frame) + jnp.multiply(1-mask_evil_end, spawn_frame))
            )
            # here is decision for has_been_killed based on its masks
            new_enemy_has_been_killed = (
                    jnp.multiply(mask_death1, killed_if_death1)
                    + jnp.multiply(1-mask_death1, jnp.multiply(mask_evil_end, killed_if_evil_end) + jnp.multiply(1-mask_evil_end, killed_else))
            )


            # Reset orientation and horizontal orientation for newly spawned enemies
            mask_spawn = jnp.greater_equal(new_enemy_spawn_frame, 11)
            direction_if_spawn = JAXAtariAction.RIGHT
            last_orientation_if_spawn = JAXAtariAction.RIGHT
            new_direction = jnp.multiply(mask_spawn, direction_if_spawn) + jnp.multiply(1 - mask_spawn, new_direction)
            new_last_horizontal_orientation = jnp.multiply(mask_spawn, last_orientation_if_spawn) + jnp.multiply(1 - mask_spawn, new_last_horizontal_orientation)

            # Reset enemy position for newly spawned enemies
            new_x = jnp.multiply(mask_spawn, cnsts.ENEMY_SPAWN_X) + jnp.multiply(1 - mask_spawn, new_position[0])
            new_y = jnp.multiply(mask_spawn, cnsts.ENEMY_SPAWN_Y) + jnp.multiply(1 - mask_spawn, new_position[1])

            # Store position at death for use in freezing during death animations
            position_at_death_cond = jnp.logical_or(jnp.equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 59),jnp.equal(jnp.abs(state.level.death_frame_counter),39))
            new_position_at_death_x = jnp.multiply(position_at_death_cond, enemy.x) + jnp.multiply(1 - position_at_death_cond, enemy.position_at_death_x)
            new_position_at_death_y = jnp.multiply(position_at_death_cond, enemy.y) + jnp.multiply(1 - position_at_death_cond, enemy.position_at_death_y)
            # Create updated enemy state with all new values
            new_enemy0 = enemy.replace(
                x=new_x,
                y=new_y,
                orientation=new_direction,
                mode_type=new_mode_type,
                mode_duration=new_mode_duration,
                last_horizontal_orientation=new_last_horizontal_orientation,
                enemy_spawn_frame=new_enemy_spawn_frame,
                enemy_death_frame=new_enemy_death_frame,
                killable=new_enemy_killable,
                has_been_killed=new_enemy_has_been_killed,
                existence=new_enemy_existence,
                position_at_death_y = new_position_at_death_y,
                position_at_death_x = new_position_at_death_x
            )
            # During global or individual death frames, override position to stay at death coordinates
            new_enemy1 = jax.lax.cond(jnp.logical_or(jnp.not_equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 0),jnp.not_equal(state.level.death_frame_counter,0)),
                                      lambda x: x.replace(
                                          x = new_position_at_death_x,
                                          y = new_position_at_death_y),
                                      lambda x: x,
                                      new_enemy0)
            return new_enemy1
    
        def spawn_enemy_step(enemy: SingleEnemyState, state: AlienState)-> SingleEnemyState:
            """Step update for a single enemy currently in spawning animation.

                 Args:
                     enemy (SingleEnemyState): Current state of the enemy.
                     state (AlienState): Current state of the game.

                 Returns:
                     SingleEnemyState: Updated enemy state after this spawn step.
                 """

            # Helper functions for vertical movement

            def add_velocity(y, orientation, velocity, spawn_frame):
                """Add velocity."""
                return jnp.add(y, velocity), orientation, spawn_frame
    
            def substract_velocity(y, orientation, velocity, spawn_frame):
                """Decrease velocity."""
                return jnp.subtract(y, velocity), orientation, spawn_frame
    
            def change_to_down(y, spawn_frame):
                """Move enemy down by 1 and change orientation to LEFT during spawn."""
                return jnp.add(y, 1), JAXAtariAction.LEFT, jnp.subtract(spawn_frame, 1)
    
            def change_to_up(y, spawn_frame):
                """Move enemy up by 1 and change orientation to RIGHT during spawn."""
                return jnp.subtract(y, 1), JAXAtariAction.RIGHT, jnp.subtract(spawn_frame, 1)

            # limits velocity to move 1 position per frame at max but decides rather how frequent it moves 1 position so either +1/-1 or 0 frequency
            # 117/249 is the approximated average of how frequent players move in bonus and 117*2 is due to vertical being twice as fast as horizontal speed
            velocity_vertical = jax.lax.min(jnp.round((((117*2)/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117*2/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)

            # Determine spawn Y bound based on enemy spawn frame modulo
            bound = jax.lax.cond(jnp.equal(jnp.mod(enemy.enemy_spawn_frame , 9), 1), lambda x: cnsts.ENEMY_START_Y, lambda x: cnsts.ENEMY_SPAWN_Y - 10, 0)

            # Masks for enemy orientation (left vs. right)
            mask_right = jnp.equal(enemy.orientation, JAXAtariAction.RIGHT)
            mask_left  = 1 - mask_right

            # Masks for vertical bounds
            mask_bound_down = jnp.less_equal(enemy.y, bound)
            mask_bound_down_neg = 1 - mask_bound_down
            mask_bound_up = jnp.greater(enemy.y, cnsts.ENEMY_SPAWN_Y)
            mask_bound_up_neg = 1 - mask_bound_up

            # Compute possible new Y, orientation, and spawn frame for each case

            # Right-moving enemy
            y_right_down, orientation_right_down, spawn_right_down = change_to_down(enemy.y, enemy.enemy_spawn_frame)
            y_right_up, orientation_right_up, spawn_right_up = substract_velocity(enemy.y, enemy.orientation, velocity_vertical, enemy.enemy_spawn_frame)

            # Left-moving enemy
            y_left_up, orientation_left_up, spawn_left_up = change_to_up(enemy.y, enemy.enemy_spawn_frame)
            y_left_down, orientation_left_down, spawn_left_down = add_velocity(enemy.y, enemy.orientation, velocity_vertical, enemy.enemy_spawn_frame)

            # Apply masks to select correct movement for right/left orientation
            y_right = jnp.multiply(mask_bound_down, y_right_down) + jnp.multiply(mask_bound_down_neg, y_right_up)
            orientation_right = jnp.multiply(mask_bound_down, orientation_right_down) + jnp.multiply(mask_bound_down_neg, orientation_right_up)
            spawn_right = jnp.multiply(mask_bound_down, spawn_right_down) + jnp.multiply(mask_bound_down_neg, spawn_right_up)

            y_left = jnp.multiply(mask_bound_up, y_left_up) + jnp.multiply(mask_bound_up_neg, y_left_down)
            orientation_left = jnp.multiply(mask_bound_up, orientation_left_up) + jnp.multiply(mask_bound_up_neg, orientation_left_down)
            spawn_left = jnp.multiply(mask_bound_up, spawn_left_up) + jnp.multiply(mask_bound_up_neg, spawn_left_down)

            # Combine left/right results based on orientation
            new_y = jnp.multiply(mask_right, y_right) + jnp.multiply(mask_left, y_left)
            new_orientation = jnp.multiply(mask_right, orientation_right) + jnp.multiply(mask_left, orientation_left)
            new_spawn_frame = jnp.multiply(mask_right, spawn_right) + jnp.multiply(mask_left, spawn_left)

            # Update enemy  has been killed status based on evil item
            mask_hbk = jnp.equal(state.level.evil_item_frame_counter, state.level.evil_item_duration - 1)
            new_enemy_has_been_killed = jnp.multiply(mask_hbk, jnp.array(0)) + jnp.multiply(1 - mask_hbk, enemy.has_been_killed)

            # Update enemy existence flag during spawn, meaning if it was on screen when evil item was picked up
            mask_exc = jnp.logical_and(
                jnp.less_equal(new_spawn_frame, 9),
                jnp.equal(state.level.evil_item_frame_counter, state.level.evil_item_duration - 1)
            )
            new_enemy_existence = jnp.multiply(mask_exc, jnp.array(1)) + jnp.multiply(1 - mask_exc, enemy.existence)


            # Determine if enemy is currently killable
            new_enemy_killable = jnp.logical_and(jnp.logical_and(jnp.logical_not(enemy.has_been_killed), jnp.greater(state.level.evil_item_frame_counter, 0)), new_enemy_existence).astype(jnp.int32)

            # Decrease enemy death frame if needed to 0
            mask_death = jnp.equal(enemy.enemy_death_frame, 1)
            new_enemy_death_frame = jnp.multiply(mask_death, jnp.array(0)) + jnp.multiply(1 - mask_death, enemy.enemy_death_frame)

            # Update positions at death for animation
            mask_death_pos = jnp.logical_or(jnp.equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 59), jnp.equal(jnp.abs(state.level.death_frame_counter), 39))
            new_position_at_death_x = jnp.multiply(mask_death_pos, enemy.x) + jnp.multiply(1 - mask_death_pos, enemy.position_at_death_x)
            new_position_at_death_y = jnp.multiply(mask_death_pos, enemy.y) + jnp.multiply(1 - mask_death_pos, enemy.position_at_death_y)

            # Final Y position considering death animation for freeze during death animation
            mask_y = jnp.logical_or(jnp.not_equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 0), jnp.not_equal(state.level.death_frame_counter, 0))
            y = jnp.multiply(mask_y, new_position_at_death_y) + jnp.multiply(1 - mask_y, new_y)

            # Return updated enemy state
            return enemy.replace(
                x=cnsts.ENEMY_SPAWN_X,
                y=y,
                orientation=new_orientation,
                mode_type=enemy.mode_type,
                mode_duration=enemy.mode_duration,
                mode_constants=enemy.mode_constants,
                last_horizontal_orientation=enemy.last_horizontal_orientation,
                enemy_spawn_frame=new_spawn_frame,
                enemy_death_frame=new_enemy_death_frame,
                key=enemy.key,
                blink=enemy.blink,
                killable=new_enemy_killable,
                has_been_killed=new_enemy_has_been_killed,
                existence=new_enemy_existence,
                active_enemy=enemy.active_enemy,
                position_at_death_x=new_position_at_death_x,
                position_at_death_y=new_position_at_death_y
            )
    
        new_enemy = jax.lax.cond(jnp.greater(enemy.enemy_spawn_frame, 0), lambda x: spawn_enemy_step(x, state), lambda x: normal_enemy_step(x, state), enemy)

        return jax.lax.cond(enemy.active_enemy, lambda _: new_enemy, lambda _: enemy, None)

def enemy_step_bonus(enemy: SingleEnemyState, state: AlienState, cnsts: AlienConstants)-> SingleEnemyState:
    """step function for all enemies in the bonus stage

    Args:
        enemy (SingleEnemyState): Enemy
        state (AlienState): state of the game
        cnsts (AlienConstants): constats because we cant pass self through

    Returns:
        SingleEnemyState: augmented Enemy
    """
    
    # move enemy and its shadow in the direction of the orientation(to left or right) until reach border and teleport back to other side
    new_x = jnp.clip((((enemy.x + (state.enemies.velocity * (2 * (enemy.last_horizontal_orientation % 2) - 1)))) % (cnsts.WIDTH - 35*((enemy.last_horizontal_orientation + 1) % 2))), (35*(enemy.last_horizontal_orientation % 2)), cnsts.WIDTH)
    
    position = jnp.array([enemy.x, enemy.y])
    
    #position of the shadow enemy(enemys move in pairs the second one is the shadow)
    shadow_position = jnp.array([new_x - 35*(2 * (enemy.last_horizontal_orientation % 2) - 1), enemy.y])
    
    player_positon = jnp.array([state.player.x, state.player.y])
    
    #check collision with enemy
    collides_1 = bonus_check_collision_between_two_sprites(sprite1=state.enemies.bonus_collision_map, pos1_x=position[0], pos1_y=position[1],
                                            sprite2=state.player.bonus_collision_map, pos2_x=player_positon[0], pos2_y=player_positon[1])
    
    #check collision with shadow of enemy
    collides_2 = bonus_check_collision_between_two_sprites(sprite1=state.enemies.bonus_collision_map, pos1_x=shadow_position[0], pos1_y=shadow_position[1],
                                            sprite2=state.player.bonus_collision_map, pos2_x=player_positon[0], pos2_y=player_positon[1])
    mask_collision = jnp.logical_or(collides_1, collides_2)
    new_enemy_death_frame = jnp.multiply(mask_collision, jnp.array(1)) + jnp.multiply(1 - mask_collision, enemy.enemy_death_frame)


    new_enemy = enemy.replace(
        x=new_x,
        y=enemy.y,
        orientation=enemy.orientation,
        mode_type=enemy.mode_type,
        mode_duration=enemy.mode_duration,
        mode_constants=enemy.mode_constants,
        last_horizontal_orientation=enemy.last_horizontal_orientation,
        enemy_spawn_frame=enemy.enemy_spawn_frame,
        enemy_death_frame=new_enemy_death_frame,
        key=enemy.key,
        blink=enemy.blink,
        killable=enemy.killable,
        has_been_killed=enemy.has_been_killed,
        existence=enemy.existence,
        active_enemy=enemy.active_enemy,
        position_at_death_x= enemy.position_at_death_x,
        position_at_death_y= enemy.position_at_death_y
    )

    return new_enemy

class AlienRenderer(JAXGameRenderer):

    def __init__(self, consts: AlienConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or AlienConstants()
        super().__init__(self.consts)
        
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "alien")
        
        # Pre-load background
        preprocessed_assets = self._load_and_preprocess_assets(sprite_path)
        
        raw_config = get_alien_asset_config()
        asset_config = []
        
        for asset in raw_config:
            if asset['name'] == 'map_primary':
                asset_copy = asset.copy()
                del asset_copy['file']
                asset_copy['data'] = preprocessed_assets['map_primary']
                asset_config.append(asset_copy)
            else:
                asset_config.append(asset)

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        
        self._cache_sprite_stacks()

    def _load_and_preprocess_assets(self, sprite_path: str) -> dict:
        target_shape = (self.consts.HEIGHT, self.consts.WIDTH, 4)
        
        full_bg = jnp.zeros(target_shape, dtype=jnp.uint8)
        full_bg = full_bg.at[:, :, 3].set(255)
        
        map_path = os.path.join(sprite_path, "bg/map_sprite.npy")
        map_raw = self.jr.loadFrame(map_path) 
        
        # Placement:
        # Offsets are (Y=5, X=8)
        # We write map_raw into full_bg at [5:..., 8:...]
        off_y = self.consts.RENDER_OFFSET_Y
        off_x = self.consts.RENDER_OFFSET_X
        
        h, w, _ = map_raw.shape
        map_primary_full = full_bg.at[off_y:off_y+h, off_x:off_x+w, :].set(map_raw)

        return {
            'map_primary': map_primary_full
        }

    def _cache_sprite_stacks(self):
        self.BONUS_MAP = self.SHAPE_MASKS['map_bonus']
        self.PLAYER_WALK_NORMAL = self.SHAPE_MASKS['player_walk_normal']
        self.PLAYER_WALK_FLAME  = self.SHAPE_MASKS['player_walk_flame']
        self.PLAYER_OFFSETS     = self.FLIP_OFFSETS['player_walk_normal']
        self.PLAYER_DEATH    = self.SHAPE_MASKS['player_death_normal']
        self.PLAYER_TELEPORT = self.SHAPE_MASKS['player_teleport_normal']
        
        self.ENEMY_WALK_STACKS = jnp.stack([
            self.SHAPE_MASKS['enemy_walk_pink'],
            self.SHAPE_MASKS['enemy_walk_yellow'],
            self.SHAPE_MASKS['enemy_walk_green']
        ])
        self.ENEMY_FRIGHTENED = self.SHAPE_MASKS['enemy_walk_frightened']
        self.ENEMY_TP_STACKS = jnp.stack([
            self.SHAPE_MASKS['enemy_teleport_pink'],
            self.SHAPE_MASKS['enemy_teleport_yellow'],
            self.SHAPE_MASKS['enemy_teleport_green']
        ])
        self.ENEMY_DEATH = self.SHAPE_MASKS['enemy_death_normal']
        self.ENEMY_OFFSETS = self.FLIP_OFFSETS['enemy_walk_pink']

        self.EVIL_ITEM_STACKS = jnp.stack([
            self.SHAPE_MASKS['evil_item_normal'],
            self.SHAPE_MASKS['evil_item_bonus_green']
        ])
        self.SCORE_ITEMS = self.SHAPE_MASKS['score_items_normal']
        
        self.EGG_STACK = jnp.stack([
            self.SHAPE_MASKS['egg_yellow'],
            self.SHAPE_MASKS['egg_orange'],
            self.SHAPE_MASKS['egg_blue'],
            self.SHAPE_MASKS['egg_pink'],
            self.SHAPE_MASKS['egg_green'],
            self.SHAPE_MASKS['egg_orange'] 
        ])
        self.EGG_HALF_STACK = jnp.stack([
            self.SHAPE_MASKS['egg_half_yellow'],
            self.SHAPE_MASKS['egg_half_orange'],
            self.SHAPE_MASKS['egg_half_blue'],
            self.SHAPE_MASKS['egg_half_pink'],
            self.SHAPE_MASKS['egg_half_green'],
            self.SHAPE_MASKS['egg_half_orange']
        ])

        digits_raw = self.SHAPE_MASKS['digits_normal']
        if digits_raw.ndim == 4:
             self.DIGITS = digits_raw.squeeze(-1)
        else:
             self.DIGITS = digits_raw
             
        self.LIFE   = self.SHAPE_MASKS['life_normal']
        self.FLAME  = self.SHAPE_MASKS['flame_normal']
        
    @partial(jax.jit, static_argnums=(0,))
    def _render_enemies_primary(self, state: AlienState, raster):
        def render_enemy(i, r):
            enemy = state.enemies.multiple_enemies
            x = enemy.x[i].astype(jnp.int32)
            y = enemy.y[i].astype(jnp.int32)
            killable = enemy.killable[i]
            active = enemy.active_enemy[i]
            last_orient = enemy.last_horizontal_orientation[i]
            death_frame = enemy.enemy_death_frame[i]
            spawn_frame = enemy.enemy_spawn_frame[i]
            blink = enemy.blink[i]

            # Animation Cycles
            cycle_1 = jnp.array([3, 3, 2, 1, 0])
            cycle_2 = jnp.array([0, 1, 2, 3, 3])
            
            is_teleport_left = (x >= 7) & (x <= 14) & (y == 80)
            is_teleport_right = (x >= 120) & (x <= 127) & (y == 80)
            in_teleport = is_teleport_left | is_teleport_right
            
            sprite_idx = jax.lax.cond(
                is_teleport_left,
                lambda: cycle_1[(x - 7) % 5],
                lambda: jax.lax.cond(
                    is_teleport_right,
                    lambda: cycle_2[(x - 120) % 5],
                    lambda: (state.level.frame_count // 8) % 4 
                )
            )

            color_idx = i % 3
            
            # --- FIX: Pure JAX Stack Selection ---
            # We select the correct stack first.
            # All stacks (Walk, TP, Frightened) MUST be shape (4, H, W).
            current_stack = jax.lax.cond(
                killable,
                lambda: self.ENEMY_FRIGHTENED,
                lambda: jax.lax.cond(
                    in_teleport,
                    lambda: self.ENEMY_TP_STACKS[color_idx],
                    lambda: self.ENEMY_WALK_STACKS[color_idx]
                )
            )
            
            normal_sprite = current_stack[sprite_idx]

            # Death Logic
            is_dying = death_frame > 0
            death_idx = jax.lax.select(
                (death_frame >= 47) & (death_frame <= 59), 0,
                jax.lax.select(
                    (death_frame >= 31) & (death_frame <= 46), 1,
                    jax.lax.select(
                        (death_frame >= 15) & (death_frame <= 30), 2,
                        3
                    )
                )
            )
            
            final_sprite = jax.lax.cond(
                is_dying,
                lambda: self.ENEMY_DEATH[death_idx],
                lambda: normal_sprite
            )
            
            flip = last_orient == 2 # RIGHT
            should_blink = (i == 2) & blink & ((state.level.frame_count % 2) == 1)
            # Draw during spawn descent (spawn_frame > 9) OR when unleashed in maze (spawn_frame == 0)
            visible = active & ((spawn_frame > 9) | (spawn_frame == 0)) & (~should_blink)
            
            return jax.lax.cond(
                visible,
                lambda r: self.jr.render_at(
                    r, 
                    x + self.consts.RENDER_OFFSET_X, 
                    y + self.consts.RENDER_OFFSET_Y, 
                    final_sprite, 
                    flip_horizontal=flip,
                    flip_offset=self.ENEMY_OFFSETS
                ),
                lambda r: r,
                r
            )

        return jax.lax.fori_loop(0, self.consts.ENEMY_AMOUNT_PRIMARY_STAGE, render_enemy, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemies_bonus(self, state: AlienState, raster):
        def render_bonus_enemy(i, r):
            enemy = state.enemies.multiple_enemies
            x = enemy.x[i].astype(jnp.int32)
            y = enemy.y[i].astype(jnp.int32)
            last_orient = enemy.last_horizontal_orientation[i]
            
            cycle_1 = jnp.array([3, 3, 2, 1, 0])
            cycle_2 = jnp.array([0, 1, 2, 3, 3])
            
            is_teleport_left = (x >= 7) & (x <= 14) & (y == 80)
            is_teleport_right = (x >= 120) & (x <= 127) & (y == 80)
            in_teleport = is_teleport_left | is_teleport_right
            
            sprite_idx = jax.lax.cond(
                is_teleport_left,
                lambda: cycle_1[(x - 7) % 8],
                lambda: jax.lax.cond(
                    is_teleport_right,
                    lambda: cycle_2[(x - 120) % 8],
                    lambda: (state.level.frame_count // 8) % 4
                )
            )
            
            color_idx = i % 3
            
            # Select Stack (Pure JAX)
            current_stack = jax.lax.cond(
                in_teleport,
                lambda: self.ENEMY_TP_STACKS[color_idx],
                lambda: self.ENEMY_WALK_STACKS[color_idx]
            )
            
            sprite = current_stack[sprite_idx]
            flip = last_orient == 2 
            
            r = self.jr.render_at(
                r, 
                x + self.consts.RENDER_OFFSET_X, 
                y + self.consts.RENDER_OFFSET_Y, 
                sprite, 
                flip_horizontal=flip,
                flip_offset=self.ENEMY_OFFSETS
            )
            
            offset_dir = 2 * (last_orient % 2) - 1
            shadow_x = x - 35 * offset_dir
            
            r = self.jr.render_at(
                r, 
                shadow_x + self.consts.RENDER_OFFSET_X, 
                y + self.consts.RENDER_OFFSET_Y, 
                sprite, 
                flip_horizontal=flip,
                flip_offset=self.ENEMY_OFFSETS
            )
            return r

        return jax.lax.fori_loop(0, self.consts.ENEMY_AMOUNT_BONUS_STAGE, render_bonus_enemy, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_player(self, state: AlienState, raster):
        x = state.player.x.astype(jnp.int32)
        y = state.player.y.astype(jnp.int32)
        orient = state.player.orientation
        last_orient = state.player.last_horizontal_orientation
        death_frame = state.level.death_frame_counter
        flame_active = state.player.flame.flame_flag
        
        cycle_1 = jnp.array([3, 3, 2, 1, 0])
        cycle_2 = jnp.array([0, 1, 2, 3, 3])
        
        is_teleport_left = (x >= 7) & (x <= 11) & (y == 80)
        is_teleport_right = (x >= 123) & (x <= 127) & (y == 80)
        
        sprite_idx = jax.lax.cond(
            is_teleport_left,
            lambda: cycle_1[(x - 7) % 5],
            lambda: jax.lax.cond(
                is_teleport_right,
                lambda: cycle_2[(x - 123) % 5],
                lambda: (state.level.frame_count // 4) % 4
            )
        )
        
        sprite = jax.lax.cond(
            death_frame > 0,
            lambda: self.PLAYER_DEATH[(state.level.frame_count // 4) % 4], 
            lambda: jax.lax.cond(
                (is_teleport_left | is_teleport_right),
                lambda: self.PLAYER_TELEPORT[sprite_idx],
                lambda: jax.lax.cond(
                    flame_active,
                    lambda: self.PLAYER_WALK_FLAME[sprite_idx],
                    lambda: self.PLAYER_WALK_NORMAL[sprite_idx]
                )
            )
        )

        flip = (last_orient == 2) | (orient == 2)
        blink = (flame_active | state.player.blink) & ((state.level.frame_count % 2) == 0)
        visible = ~blink
        
        raster = jax.lax.cond(
            visible,
            lambda r: self.jr.render_at(
                r, 
                x + self.consts.RENDER_OFFSET_X, # FIX: X first
                y + self.consts.RENDER_OFFSET_Y, # FIX: Y second
                sprite,
                flip_horizontal=flip,
                flip_offset=self.PLAYER_OFFSETS
            ),
            lambda r: r,
            raster
        )
        
        def render_flame(r):
            flame_x = state.player.flame.x.astype(jnp.int32)
            flame_y = state.player.flame.y.astype(jnp.int32)
            flame_blink = (state.level.frame_count % 2) == 1
            
            return jax.lax.cond(
                flame_blink,
                lambda _r: self.jr.render_at(
                    _r,
                    flame_x + self.consts.RENDER_OFFSET_X, # FIX: X first
                    flame_y + self.consts.RENDER_OFFSET_Y, # FIX: Y second
                    self.FLAME,
                    flip_horizontal=flip,
                    flip_offset=self.PLAYER_OFFSETS
                ),
                lambda _r: _r,
                r
            )

        return jax.lax.cond(flame_active, render_flame, lambda r: r, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_items(self, state: AlienState, raster):
        def render_item(i, r):
            x = state.items[i, 0].astype(jnp.int32)
            y = state.items[i, 1].astype(jnp.int32)
            active = state.items[i, 2]
            type_id = state.items[i, 3] # 0=Evil, 1-6=Score
            
            is_evil = type_id == 0
            
            sprite = jax.lax.cond(
                is_evil,
                lambda: self.EVIL_ITEM_STACKS[0][0], 
                lambda: self.SCORE_ITEMS[type_id - 1]
            )
            
            blink_evil = state.level.blink_evil_item
            blink_active = state.level.blink_current_active_item
            should_blink = (state.level.frame_count % 2) == 0
            
            visible = active & ~(
                (is_evil & blink_evil & should_blink) | 
                (~is_evil & blink_active & should_blink) |
                (is_evil & state.player.flame.flame_flag)
            )
            
            return jax.lax.cond(
                visible,
                lambda _r: self.jr.render_at(
                    _r,
                    x + self.consts.RENDER_OFFSET_X, # FIX: X first
                    y + self.consts.RENDER_OFFSET_Y, # FIX: Y second
                    sprite
                ),
                lambda _r: _r,
                r
            )
            
        return jax.lax.fori_loop(0, state.items.shape[0], render_item, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_eggs(self, state: AlienState, raster):
        def render_egg_row(i, l_r, r):
            x = state.eggs[l_r, i, 0].astype(jnp.int32)
            y = state.eggs[l_r, i, 1].astype(jnp.int32)
            active = state.eggs[l_r, i, 2]
            color_id = state.eggs[l_r, i, 3].astype(jnp.int32)
            
            def draw_egg(_r):
                # Draw Main Egg
                _r = self.jr.render_at(
                    _r,
                    x + self.consts.RENDER_OFFSET_X, # FIX: X first
                    y + self.consts.RENDER_OFFSET_Y, # FIX: Y second
                    self.EGG_STACK[color_id]
                )
                # Draw Half Egg (offset by 1)
                _r = self.jr.render_at(
                    _r,
                    x + self.consts.RENDER_OFFSET_X, # FIX: X first
                    y + self.consts.RENDER_OFFSET_Y + 1, # FIX: Y second + 1
                    self.EGG_HALF_STACK[color_id]
                )
                return _r

            return jax.lax.cond(active, draw_egg, lambda _r: _r, r)

        row_idx = (state.level.frame_count % 2)
        render_body = lambda i, r: render_egg_row(i, row_idx, r)
        return jax.lax.fori_loop(0, state.eggs.shape[1], render_body, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_hud(self, state: AlienState, raster):
        # 1. Score: right-aligned — starts as 1 digit on the right, expands left as score grows
        score_val = state.level.score.astype(jnp.int32)
        score_digits = self.jr.int_to_digits(score_val, max_digits=6)
        score_flat = score_digits.flatten()
        n = jnp.maximum(score_val, 0)
        num_digits = jnp.where(
            n > 0,
            jnp.ceil(jnp.log10(n.astype(jnp.float32) + 1.0)).astype(jnp.int32),
            jnp.array(1, dtype=jnp.int32)
        )
        num_digits = jnp.squeeze(jnp.minimum(num_digits, 6))
        start_index = 6 - num_digits
        # Right edge of score area (x of rightmost digit when 6 digits); expand left
        score_spacing = 8
        score_right_x = self.consts.SCORE_X + self.consts.RENDER_OFFSET_X + (6 - 1) * score_spacing
        score_x = score_right_x - (num_digits - 1) * score_spacing
        raster = self.jr.render_label_selective(
            raster,
            score_x,
            self.consts.SCORE_Y + self.consts.RENDER_OFFSET_Y,
            score_flat,
            self.DIGITS,
            start_index,
            num_digits,
            spacing=score_spacing,
            max_digits_to_render=6
        )
        
        # 2. Lives (shifted up)
        raster = self.jr.render_indicator(
            raster,
            self.consts.LIFE_X + self.consts.RENDER_OFFSET_X, # X
            self.consts.LIFE_Y + self.consts.RENDER_OFFSET_Y - 8, # Y, shifted up
            jnp.squeeze(state.level.lifes),
            self.LIFE,
            spacing=self.consts.LIFE_WIDTH + self.consts.LIFE_OFFSET_X,
            max_value=3
        )
        
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: AlienState):
        """Main rendering function."""
        
        is_bonus = state.level.bonus_flag
        
        def get_bonus_raster():
            black = jnp.zeros_like(self.BACKGROUND) 
            return self.jr.render_at(black, 0, 0, self.BONUS_MAP)

        raster = jax.lax.cond(
            is_bonus,
            get_bonus_raster,
            lambda: self.BACKGROUND
        )
        
        def render_primary(r):
            r = self._render_eggs(state, r)
            r = self._render_items(state, r)
            r = self._render_enemies_primary(state, r)
            r = self._render_player(state, r)
            return r
            
        def render_bonus(r):
            show_bonus_item = state.level.evil_item_frame_counter <= 0
            bonus_sprite = self.EVIL_ITEM_STACKS[1][0]
            
            r = jax.lax.cond(
                show_bonus_item,
                lambda _r: self.jr.render_at(
                    _r,
                    self.consts.ENEMY_SPAWN_X + self.consts.RENDER_OFFSET_X, # X
                    10 + self.consts.RENDER_OFFSET_Y, # Y
                    bonus_sprite
                ),
                lambda _r: _r,
                r
            )
            
            r = self._render_enemies_bonus(state, r)
            r = self._render_player(state, r)
            return r

        raster = jax.lax.cond(is_bonus, render_bonus, render_primary, raster)
        raster = self._render_hud(state, raster)

        return self.jr.render_from_palette(raster, self.PALETTE)

def traverse_multiple_enemy_(fun: Callable[[SingleEnemyState], Any], number_of_non_traversed_args: int = 0):
    """the broken hopes and dreams of a dynamic enemy count

    Args:
        enemy_state (MultipleEnemiesState): Every Enemy
        fun (Callable[[SingleEnemyState], Any]): enemy step function
        returns_enemies_state (bool, optional):  Defaults to False.
        number_of_non_traversed_args (int, optional): Defaults to 0.

    Returns:
        _type_: output of fun
    """
   
    fields = list(SingleEnemyState.__dataclass_fields__.keys())
    fields.sort()
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

    state_axes = (0, )*len(fields)
    if number_of_non_traversed_args>0:
        n_s = (None, )*number_of_non_traversed_args
        in_axes_tuple = (*state_axes,*n_s)
    else:
        in_axes_tuple = state_axes
    
    def my_return_function(*args, enemy_fields, vmapped_function):
        enemy_state: SingleEnemyState = args[0]
        inputs = ()
        for f in enemy_fields:
            inputs = (*inputs, getattr(enemy_state, f))
        inputs = (*inputs, *args[1:])
        r_args = vmapped_function(*inputs)
        m_args = {}
        for i, f in enumerate(enemy_fields):
            m_args[f] = r_args[i]            
        ret = MultipleEnemiesState(**m_args)
        return ret
    v_func = jax.vmap(wrapper_function, in_axes=in_axes_tuple, out_axes=state_axes)
    return partial(
        my_return_function, 
        enemy_fields=fields, 
        vmapped_function=v_func
    )
    