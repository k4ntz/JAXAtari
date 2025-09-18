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
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction
import jax
import numpy as np
from jax import Array as jArray


class AlienObservation(NamedTuple):
    player_x:jnp.ndarray
    player_y:jnp.ndarray
    player_o:jnp.ndarray
    enemies_position:jnp.ndarray
    enemies_killable: jnp.ndarray
    kill_item_position: jnp.ndarray
    score_item_position: jnp.ndarray
    collision_map: jnp.ndarray


class AlienInfo(NamedTuple):
    score: jnp.ndarray
    step_counter: jnp.ndarray
    all_rewards: jnp.ndarray
    
class AlienConstants(NamedTuple):
    SEED: int = 42 #seed for randomness
    RENDER_SCALE_FACTOR: int = 4
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    WIDTH: int = 160 # Width of the playing field in game state
    HEIGHT: int = 210 # height of the playing field in game state
    PLAYER_VELOCITY: int = 1
    ENEMY_VELOCITY: int = 1
    #player sprite dimensions
    PLAYER_WIDTH: int = 8
    PLAYER_HEIGHT: int = 13
    # player initial position
    PLAYER_X: int = 67
    PLAYER_Y: int = 56
    # Position at which the numbers are rendered
    SCORE_X: int = 19
    SCORE_Y: int = 171
    #offset for background rendering
    RENDER_OFFSET_Y: int = 5
    RENDER_OFFSET_X: int = 8
    DIGIT_OFFSET: int = 3 # Offset between numbers
    #digit sprite dimensions
    DIGIT_WIDTH: int = 6
    DIGIT_HEIGHT: int = 7

    # Position at which the life counter is rendered
    LIFE_X: int = 13
    LIFE_Y: int = 187
    LIFE_OFFSET_X: int = 2 # Offset between life sprites
    LIFE_WIDTH: int = 5

    # Enemy_player_collision_offset
    ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW: int = 4
    ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH: int = 6
    
    # Enemy amount
    ENEMY_AMOUNT_PRIMARY_STAGE: int = 3
    ENEMY_AMOUNT_BONUS_STAGE: int = 6
    # Spawn position of enemies
    ENEMY_SPAWN_X: int = 67
    ENEMY_SPAWN_Y: int = 147 + 5
    # Enemy start position between walls
    ENEMY_START_Y: int = 128
    FRIGHTENED_DURATION: int = 100
    FLAME_FRIGHTENED_DURATION: int = 5

    # Pink enemy constants
    SCATTER_DURATION_1: int = 50
    CHASE_DURATION_1: int = 200
    MODECHANGE_PROBABILITY_1: float = 0.5
    SCATTER_POINT_X_1: int = 0
    SCATTER_POINT_Y_1: int = 0

    # Yellow enemy constants
    SCATTER_DURATION_2: int = 50
    CHASE_DURATION_2: int = 100
    MODECHANGE_PROBABILITY_2: float = 0.6
    SCATTER_POINT_X_2: int = 0
    SCATTER_POINT_Y_2: int = 180

    # Green enemy constants
    SCATTER_DURATION_3: int = 50
    CHASE_DURATION_3: int = 100
    MODECHANGE_PROBABILITY_3: float = 0.7
    SCATTER_POINT_X_3: int = 180
    SCATTER_POINT_Y_3: int = 180
    
    #Colors
    BASIC_BLUE = jnp.array([132, 144, 252]) # blue egg, lifes, player, digits
    ORANGE = jnp.array([252, 144, 144]) # orange egg, evil item, flamethrower
    PINK = jnp.array([236, 140, 224]) # pink enemy, pink egg
    GREEN = jnp.array([132, 252, 212]) # green enemy, green egg
    YELLOW = jnp.array([252, 252, 84]) # yellow enemy, yellow egg, items
    OTHER_BLUE = jnp.array([101, 111, 228])# fleeing enemys
    
    ENEMY_COLORS = jnp.array([PINK, YELLOW, GREEN])
    
    # color : 0 (yellow), 1 (orange), 2 (blue), 3 (pink), 4 (turquoise), 5 ( orange and blue)
    EGG_COLORS = jnp.array([
        [YELLOW,YELLOW],
        [ORANGE,ORANGE],
        [BASIC_BLUE,BASIC_BLUE],
        [PINK,PINK],
        [GREEN,GREEN],
        [ORANGE,BASIC_BLUE]
    ])
    
    ITEM_ARRAY = jnp.array([ 
    # 0: pulsar, 1: rocket, 2: saturn, 3: starship, 4: evil_item at index 3# 1 if the item is active, 0 otherwise
        [68, 9,  1,  0],
        [22,  129,  0,  0],
        [114, 129,  0,  0],
        [68 ,  56,  0,  1] #static score item
    ])
    
    ITEM_SCORE_MULTIPLIERS = jnp.array([0, 100, 500, 1000, 2000, 3000, 5000])# Score for collecting items, last item is not a score item
    ENEMY_KILL_SCORE = jnp.array([500, 1000, 1500]) # Score for killing enemies, depending on how many have been killed in succession

    # egg discription:
    # x-coordinate
    # y-coordinate
    # status : 1 (on the field),  0 (not on the field)
    # color : 0 (yellow), 1 (orange), 2 (blue), 3 (pink), 4 (turquoise), 5 ( orange and blue)
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
    EGG_SCORE_MULTIPLYER = 10
    
    
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

class EnemyModeConstants(NamedTuple):
    scatter_duration: jnp.ndarray
    scatter_point_x: jnp.ndarray
    scatter_point_y: jnp.ndarray
    chase_duration: jnp.ndarray
    mode_change_probability: jnp.ndarray
    frightened_duration: jnp.ndarray

class SingleEnemyState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    orientation: jnp.ndarray
    mode_type: jnp.ndarray # mode_types: 0=chase, 1=scatter, 2=frightend
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

class MultipleEnemiesState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    orientation: jnp.ndarray
    mode_type: jnp.ndarray # mode_types: 0=chase, 1=scatter, 2=frightend
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

class EnemiesState(NamedTuple):
    multiple_enemies: MultipleEnemiesState
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
    bonus_flag: jnp.ndarray
    difficulty_stage: jnp.ndarray
    evil_item_duration: jnp.ndarray
    spawn_score_item_flag_1: jnp.ndarray
    spawn_score_item_flag_2: jnp.ndarray
    score_item_counter: jnp.ndarray

class AlienState(NamedTuple):
    player: PlayerState
    enemies: EnemiesState
    level: LevelState
    eggs: jnp.ndarray
    items: jnp.ndarray
    step_counter: jnp.ndarray


m_fs = set(MultipleEnemiesState._fields)
s_fs = set(SingleEnemyState._fields)
if not (m_fs.issubset(s_fs) and s_fs.issubset(m_fs)):
    raise Exception("Mismatch between fields in SingleEnemyState and MultipleEnemiesState")

def load_collision_map(fileName, transpose=True):
    """converts an npy file to a collision map of shape (height, width)

    Args:
        fileName (os.path): file path plus name
        transpose (bool, optional):  Defaults to True.

    Returns:
        jnp.ndarry: a collison map
    """
    # Returns a boolean array representing the collision map
    # Load frame (np array) from a .npy file and convert to jnp array
    
    frame = jnp.load(fileName)
    frame = frame[..., 0].squeeze()
    boolean_frame = jnp.zeros(shape=frame.shape, dtype=jnp.bool)
    boolean_frame = boolean_frame.at[frame==0].set(False)
    boolean_frame = boolean_frame.at[frame > 0].set(True)
    frame = boolean_frame
    return jnp.transpose(frame, (1, 0)) if transpose else frame

PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK: jnp.ndarray = load_collision_map(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites", "alien", "player_enemy_collision_sprite.npy"), transpose=True)
BACKGROUND_COLLISION_MAP: jnp.ndarray = load_collision_map(os.path.join(os.path.dirname(os.path.abspath(__file__)),"sprites", "alien","bg", "map_sprite_collision_map.npy"), transpose=True)

def loadFramAddAlpha(fileName, transpose=True, add_alpha: bool = False, add_black_as_transparent: bool = False):
    """Custom loading function which turns black background transparent.
        This is simply to make editing sprites a bit more convenient.

    Args:
        fileName (os.path): file path plus name
        transpose (bool, optional):  Defaults to True.
        add_alpha (bool, optional): Defaults to False.
        add_black_as_transparent (bool, optional):  Defaults to False.

    Raises:
        ValueError: when the jnp.ndarry is not of shape (height, width, 4)

    Returns:
        jnp.ndarry: finished sprite frame
    """
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
        
    frame = frame.astype(jnp.uint8)
    return jnp.transpose(frame, (1, 0, 2)) if transpose else frame


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
    ret_v = jax.lax.cond(has_collision, lambda x: old_position, lambda x: new_position, 0)
    return ret_v

def check_collision_between_two_sprites(sprite1, pos1_x, pos1_y, sprite2, pos2_x, pos2_y):
        r1x = pos1_x
        r1w = jnp.shape(sprite1)[0]
        r2x = pos2_x
        r2w = jnp.shape(sprite2)[0]

        r1y = pos1_y
        r1h = jnp.shape(sprite1)[1]
        r2y = pos2_y
        r2h = jnp.shape(sprite2)[1]
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

@partial(jax.jit, static_argnames=['cnsts'])
def check_for_player_enemy_collision(cnsts: AlienConstants, state: AlienState, new_position: jnp.ndarray) -> jnp.ndarray:
    """Wrapper for collision_check_between_two_objects that checks all enemies
    Args:
        state (AlienState): Current state of Game
        new_position: (jnp.ndarray): Proposed position of the player after the movement would be executed

     Returns:
        jnp.ndarray, dtype= bool: bolean depending on collision with the enemies
    """
    # Determine coord range occupied by the player
    player_enemy_collision: jnp.ndarray = PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK
    background_map: jnp.ndarray = BACKGROUND_COLLISION_MAP
    # Check if player sprite crosses a certian point in enemy sprite
    enemy_0_collides = check_collision_between_two_sprites(sprite1=player_enemy_collision, pos1_x=state.enemies.multiple_enemies.x[0], 
                                                           pos1_y=state.enemies.multiple_enemies.y[0], 
                                                           sprite2=player_enemy_collision, pos2_x=new_position[0], 
                                                           pos2_y=new_position[1])
    enemy_1_collides = check_collision_between_two_sprites(sprite1=player_enemy_collision, pos1_x=state.enemies.multiple_enemies.x[1], 
                                                           pos1_y=state.enemies.multiple_enemies.y[1], 
                                                           sprite2=player_enemy_collision, pos2_x=new_position[0], 
                                                           pos2_y=new_position[1])
    enemy_2_collides = check_collision_between_two_sprites(sprite1=player_enemy_collision, pos1_x=state.enemies.multiple_enemies.x[2], 
                                                           pos1_y=state.enemies.multiple_enemies.y[2], 
                                                           sprite2=player_enemy_collision, pos2_x=new_position[0], 
                                                           pos2_y=new_position[1])
    has_collision_enemy0 = jnp.logical_and(enemy_0_collides, jnp.logical_and(jnp.equal(state.enemies.multiple_enemies.killable[0], 0), jnp.equal(state.enemies.multiple_enemies.enemy_spawn_frame[0],0)))

    has_collision_enemy1 = jnp.logical_and(enemy_1_collides, jnp.logical_and(jnp.equal(state.enemies.multiple_enemies.killable[1], 0), jnp.equal(state.enemies.multiple_enemies.enemy_spawn_frame[1],0)))

    has_collision_enemy2 =jnp.logical_and(enemy_2_collides, jnp.logical_and(jnp.equal(state.enemies.multiple_enemies.killable[2], 0), jnp.equal(state.enemies.multiple_enemies.enemy_spawn_frame[2],0)))

    return jnp.logical_or(has_collision_enemy0, jnp.logical_or(has_collision_enemy1, has_collision_enemy2))

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
    # TODO: Change to switch
    return jax.lax.cond(
        jnp.logical_or(jnp.logical_and(position[0] >= 127, orientation == JAXAtariAction.RIGHT), 
                       jnp.logical_and(action == JAXAtariAction.RIGHT, position[0] >= 127)),
        lambda x: x.at[0].set(7),
        lambda x: jax.lax.cond(
            jnp.logical_or(jnp.logical_and(position[0] <= 7,orientation == JAXAtariAction.LEFT),
                           jnp.logical_and(action == JAXAtariAction.LEFT, position[0] <= 7)),
            lambda x: x.at[0].set(127),
            lambda x:x,
            position
        ),
        position
    )

class JaxAlien(JaxEnvironment[AlienState, AlienObservation, AlienInfo, AlienConstants]):
    def __init__(self, consts: AlienConstants = None,frameskip: int = 1, reward_funcs: list[callable]=None):
        consts = consts or AlienConstants()
        super().__init__(consts)
        self.frameskip = frameskip
        self.frame_stack_size = 4
        self.renderer = AlienRenderer(consts=self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs) 
        self.reward_funcs = reward_funcs

        traverse_enemy_step = traverse_multiple_enemy_(partial(enemy_step, cnsts=consts), 1)
        self.enemy_step = jax.jit(traverse_enemy_step)
        
        traverse_enemy_step_bonus =  traverse_multiple_enemy_(partial(enemy_step_bonus, cnsts=consts), 1)
        self.enemy_step_bonus = jax.jit(traverse_enemy_step_bonus)
        
        self.sprite_path: str = os.path.join(self.consts.MODULE_DIR, "sprites", "alien")

        self.player_collision_map: jnp.ndarray = load_collision_map(os.path.join(self.sprite_path, "player_animation" ,"player_sprite_collision_map.npy"), transpose=True)
        self.player_bonus_collision_map: jnp.ndarray = load_collision_map(os.path.join(self.sprite_path, "player_animation", "player_sprite_bonus_collision_map.npy"), transpose=True)
        self.level_collision_map: jnp.ndarray = load_collision_map(os.path.join(self.sprite_path,  "bg", "map_sprite_collision_map.npy"), transpose=True)
        self.enemy_collision_map: jnp.ndarray = load_collision_map(os.path.join(self.sprite_path,  "player_animation","player_sprite_collision_map.npy"), transpose=True)
        self.enemy_bonus_collision_map: jnp.ndarray = load_collision_map(os.path.join(self.sprite_path, "enemy_animation", "enemy_sprite_bonus_collision_map.npy"), transpose=True)

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
    def render(self, state: AlienState) -> jnp.ndarray:
        return self.renderer.render(state)
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
    def reset(self, rng_key: jArray) -> Tuple[AlienObservation, AlienState]:
        """
        Resets the game state to the initial state.
        """
        key: jax.random.PRNGKey = rng_key
        keys = jax.random.split(key, self.consts.ENEMY_AMOUNT_BONUS_STAGE + 1)
        main_key = keys[0]
        subkeys = keys[1:]

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
                enemy_spawn_frame=jnp.pad(jnp.array([11, 21, 31]).astype(jnp.int32),
                                          (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0),
                enemy_death_frame=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                key=subkeys,
                blink=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                killable=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                has_been_killed=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                existence = jnp.zeros(self.consts.ENEMY_AMOUNT_BONUS_STAGE, dtype=jnp.int32),
                position_at_death_x= jnp.full((self.consts.ENEMY_AMOUNT_BONUS_STAGE), self.consts.ENEMY_SPAWN_X, dtype=jnp.int32),
                position_at_death_y= jnp.full((self.consts.ENEMY_AMOUNT_BONUS_STAGE), self.consts.ENEMY_SPAWN_Y, dtype=jnp.int32),
                active_enemy=jnp.pad(jnp.array([1, 1, 1]).astype(jnp.int32),
                                          (0, (self.consts.ENEMY_AMOUNT_BONUS_STAGE - self.consts.ENEMY_AMOUNT_PRIMARY_STAGE)), mode='constant', constant_values=0),
                kill_score_flag=jnp.ones((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32)
            ),
            rng_key=main_key,
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
            blink_current_active_item=jnp.array(0).astype(jnp.int32),
            bonus_flag=jnp.array(0).astype(jnp.int32),
            difficulty_stage=jnp.array(0).astype(jnp.float32),
            evil_item_duration=jnp.array(505).astype(jnp.int32),
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
        """initiates the bonus stage
        Args:
            state (AlienState): Current state of the game
        Returns:
            AlienState: the next game frame
        """
        _, new_state = self.reset(state.enemies.rng_key)
        
        new_player_state = new_state.player._replace(
            x=jnp.array(self.consts.PLAYER_X).astype(jnp.int32),
            y=jnp.array(self.consts.ENEMY_START_Y + 20).astype(jnp.int32),
            last_horizontal_orientation=JAXAtariAction.LEFT,
            collision_map=self.player_bonus_collision_map
        )
        # Choose random enemy start position from 3 presets
        main_key, sub_key = jax.random.split(state.enemies.rng_key)
        random_value = jax.random.randint(sub_key, shape=(), minval=0, maxval=3)
        presets = jnp.array([[70, 20, 80], [10, 80, 140], [40, 100, 110]])
        
        # Position 6 enemies in a diagonal line across the map
        diagonal_spacing = 20
        start_y = 30
        enemy_x_positions = jnp.array([presets[random_value][0], presets[random_value][0]+30+5, presets[random_value][1], presets[random_value][2], presets[random_value][2]-30-5, presets[random_value][1]+30-5], dtype=jnp.int32)
        enemy_y_positions = jnp.arange(self.consts.ENEMY_AMOUNT_BONUS_STAGE, dtype=jnp.int32) * diagonal_spacing + start_y
        
        new_multiple_enemies = state.enemies.multiple_enemies._replace(
            x=enemy_x_positions,
            y=enemy_y_positions,
            last_horizontal_orientation=jnp.array([JAXAtariAction.LEFT, JAXAtariAction.RIGHT, JAXAtariAction.LEFT, JAXAtariAction.RIGHT, JAXAtariAction.LEFT, JAXAtariAction.RIGHT], dtype=jnp.int32),
            enemy_spawn_frame=jnp.zeros(self.consts.ENEMY_AMOUNT_BONUS_STAGE, dtype=jnp.int32)
        )
        new_enemies_state = state.enemies._replace(
            multiple_enemies=new_multiple_enemies,
            collision_map=self.enemy_bonus_collision_map,
            rng_key=main_key
        )
        new_level_state = new_state.level._replace(
            score=state.level.score + 1,
            lifes=state.level.lifes,
            bonus_flag=jnp.array(1).astype(jnp.int32),
            collision_map=jnp.zeros(new_state.level.collision_map.shape, dtype=jnp.bool_),
            difficulty_stage=state.level.difficulty_stage
        )
        new_state = new_state._replace(
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
        
        new_level_state = reset_state.level._replace(
            lifes=state.level.lifes,
            score=state.level.score,
            difficulty_stage=state.level.difficulty_stage + 0.5,
            evil_item_duration=jnp.maximum((-63 * state.level.difficulty_stage + 505), 0).astype(jnp.int32)
        )
        new_enemies_state = reset_state.enemies._replace(
            rng_key=state.enemies.rng_key
        )
        new_state = reset_state._replace(level=new_level_state, enemies=new_enemies_state)
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
        new_player_state =  self.player_step(state, action)
        new_enemies_state = self.multiple_enemies_step(state)
        new_egg_state, new_score = self.egg_step(
            new_player_state.x,
            new_player_state.y,
            state.eggs,
            state.level.score,
            self.consts.EGG_SCORE_MULTIPLYER
        )
        new_life = jax.lax.cond(
            jnp.logical_and(check_for_player_enemy_collision(self.consts, state, [new_player_state.x, new_player_state.y]),
                            jnp.equal(state.level.death_frame_counter, 0)),
            lambda x: jnp.add(x, -1),
            lambda x: x,
            state.level.lifes
        )
        new_death_frame_counter = jax.lax.cond(
            jnp.logical_and(check_for_player_enemy_collision(self.consts, state, [new_player_state.x, new_player_state.y]),
                            jnp.equal(state.level.death_frame_counter, 0)),
            lambda x: jax.lax.cond(
                jnp.less(new_life, 0),
                lambda y: jnp.add(y, -40),
                lambda y: jnp.add(y, 40),
                x
            ),
            lambda x: x,
            state.level.death_frame_counter
        )
        new_items, new_score, new_current_active_item_index, new_evil_item_frame_counter, new_spawn_score_item_flag_1, new_spawn_score_item_flag_2, new_score_item_counter = self.item_step(
            player_x=new_player_state.x,
            player_y=new_player_state.y,
            score=new_score,
            state=state
        )
        new_level_state = state.level._replace(
            score=new_score,
            lifes=new_life,
            death_frame_counter=new_death_frame_counter,
            evil_item_frame_counter=new_evil_item_frame_counter,
            current_active_item_index=new_current_active_item_index,
            spawn_score_item_flag_1=new_spawn_score_item_flag_1,
            spawn_score_item_flag_2=new_spawn_score_item_flag_2,
            score_item_counter=new_score_item_counter
        )
        
        new_state = state._replace(
            player=new_player_state,
            enemies=new_enemies_state,
            level=new_level_state,
            eggs=new_egg_state,
            items=new_items
        )
        
        return new_state
    
    
    @partial(jax.jit, static_argnums=(0)) 
    def death(self, state: AlienState) -> Tuple[AlienState, AlienObservation, AlienInfo]:
        """initaiates the soft reset when loosing a life
        Args:
            state (AlienState): Current state of the game
        Returns:
            Tuple[AlienState, AlienObservation, AlienInfo]: New state, observation and info
        """
        new_death_frame_counter = state.level.death_frame_counter - 1
        freeze_level_state = state.level._replace(
            death_frame_counter=new_death_frame_counter,
            blink_evil_item=jnp.array(0).astype(jnp.int32),
            blink_current_active_item=jnp.array(0).astype(jnp.int32)
            )
        freeze_state = state._replace(
            level=freeze_level_state
            )
        
        _, reset_state = self.reset(state.enemies.rng_key)
        
        soft_reset_multiple_enemies = reset_state.enemies.multiple_enemies._replace(
            kill_score_flag=state.enemies.multiple_enemies.kill_score_flag
            )
        
        soft_reset_enemies = reset_state.enemies._replace(
            rng_key=state.enemies.rng_key,
            multiple_enemies=soft_reset_multiple_enemies
            )
        soft_reset_level_state = reset_state.level._replace(
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
        
        soft_reset_state = reset_state._replace(
            enemies=soft_reset_enemies,
            level=soft_reset_level_state,
            eggs=state.eggs,
            items=items_keep_evil_reset_score
            )
        #new_state = jax.lax.select(jnp.equal(state.level.death_frame_counter, 1), 
        #                           soft_reset_state, freeze_state)
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
        """handles game over
        Returns:
            Tuple[AlienState, AlienObservation, AlienInfo]: New state, observation and info
        """
        new_death_frame_counter = jnp.add(state.level.death_frame_counter, 1)
        freeze_level_state = state.level._replace(  
            death_frame_counter=new_death_frame_counter,
            current_active_item_index=jnp.array(0).astype(jnp.int32),
            blink_evil_item=jnp.array(0).astype(jnp.int32),
            blink_current_active_item=jnp.array(0).astype(jnp.int32),
            )
        
        freeze_state = state._replace(
            level=freeze_level_state
            )
        
        _, reset_state = self.reset(state.enemies.rng_key)
        hard_reset_enemies = reset_state.enemies._replace(
            rng_key=state.enemies.rng_key
            )
        
        hard_reset_state = reset_state._replace(
            enemies=hard_reset_enemies,
        )
        new_state = jax.lax.cond(jnp.equal(state.level.death_frame_counter, -1), 
                                 lambda x, y: x , lambda x, y: y , 
                                 hard_reset_state, freeze_state)
        return new_state
            
    
    @partial(jax.jit, static_argnums=(0))
    def _choose_gamestep_besides_kill_item_step(self, state: AlienState, action: chex.Array):
        is_alive: jArray = jnp.equal(state.level.death_frame_counter, 0)
        is_normal_death: jArray = jnp.greater(state.level.death_frame_counter, 0)
        is_game_over: jArray = jnp.less(state.level.death_frame_counter, 0)
        choice_array: jArray = jnp.array([1*is_alive, 2*is_normal_death, 3*is_game_over], jnp.int32)
        actul_choice: jArray = jnp.max(choice_array, keepdims=False)
        new_state: AlienState = jax.lax.switch(actul_choice, 
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
        """step funtion for the primary stage
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
        new_state = jax.lax.cond(
                jnp.sum(state.eggs[:, :, 2]) == 0,
                lambda x: self.start_bonus_stage(x),
                lambda x: x,
                new_state
            )
        # Only the state_update is currently implemented. We need to handle score & observation still!!
        return new_state
        
        

        
    @partial(jax.jit, static_argnums=(0))
    def kill_item_step(self, state: AlienState, action: jArray) -> Tuple[AlienState, AlienObservation, AlienInfo]:
        """handle behavior related to the kill item
        Args:
            state (AlienState): Current State of the Game
        Returns:
            Tuple[AlienState, AlienObservation, AlienInfo]: New state, observation and info
        """

        new_player_state: PlayerState =  self.player_step(state, action)
        
        new_life = jax.lax.cond(
            jnp.logical_and(check_for_player_enemy_collision(self.consts, state, [new_player_state.x, new_player_state.y]),
                            jnp.logical_not(state.level.death_frame_counter)),
            lambda x: jnp.subtract(x, 1),
            lambda x: x,
            state.level.lifes
        )

        new_death_frame_counter = jax.lax.cond(
            jnp.logical_and(check_for_player_enemy_collision(self.consts, state, [new_player_state.x, new_player_state.y]),
                            jnp.logical_not(state.level.death_frame_counter)),
            lambda x: jax.lax.cond(
                jnp.less(new_life, 0),
                lambda y: jnp.subtract(y, 40),
                lambda y: jnp.add(y, 40),
                x
            ),
            lambda x: x,
            state.level.death_frame_counter
        )
        
        new_kill_score_flag = jnp.where(state.enemies.multiple_enemies.enemy_death_frame > 0, 0, state.enemies.multiple_enemies.kill_score_flag)
        score_interation = 2 - jnp.sum(new_kill_score_flag[:3])
        score_increase_kill = jnp.sum((state.enemies.multiple_enemies.kill_score_flag * self.consts.ENEMY_KILL_SCORE[score_interation])*(state.enemies.multiple_enemies.enemy_death_frame > 0)).astype(jnp.uint16)
        new_score_kill = state.level.score + score_increase_kill
        freeze_level_state = state.level._replace(
                lifes=new_life,
                death_frame_counter=new_death_frame_counter,
                score=new_score_kill
        )
        
        new_enemies: EnemiesState = self.multiple_enemies_step(state)
        freeze_multiple_enemies = state.enemies.multiple_enemies._replace(
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
        
        freeze_player_state = new_player_state._replace(
            x=state.player.x,
            y=state.player.y
        )
        
        freeze_enemies = state.enemies._replace(
                multiple_enemies=freeze_multiple_enemies
        )
        freeze_state = state._replace(
            player=freeze_player_state,
            enemies=freeze_enemies,
            level=freeze_level_state,
        )
        
        kill_multiple_enemies = new_enemies.multiple_enemies._replace(
                blink=jnp.zeros((self.consts.ENEMY_AMOUNT_BONUS_STAGE,), dtype=jnp.int32),
                position_at_death_x= new_enemies.multiple_enemies.position_at_death_x,
                position_at_death_y=new_enemies.multiple_enemies.position_at_death_y
            )
        kill_enemies = new_enemies._replace(
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
        level_state = state.level._replace(
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
        kill_state = state._replace(
                player=new_player_state,
                enemies=kill_enemies,
                level=level_state,
                eggs=new_egg_state,
                items=new_items
                )
        is_player_dead: jArray = jnp.not_equal(new_death_frame_counter, 0)
        is_game_over: jArray = jnp.less(new_death_frame_counter, 0)
        
        # Handle player death here!
        death_index: jArray = 0 + is_player_dead + is_game_over
        # If 0: player is still alive
        # If 1: player is dead but game is not over
        # If 2: Player is dead but game is over!
        new_state = jax.lax.switch(death_index, 
                                   [lambda x: x, 
                                    self.death, 
                                    self.game_over], 
                                   freeze_state)
        # Now Handle the game freeze for when enemies dies
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
        moving_action = jax.lax.cond(jnp.greater(action,9),
                                      lambda ac: jnp.mod(ac,10)+2,
                                      lambda ac: ac,
                                      action
                                      )
        
        position = jnp.array([state.player.x, state.player.y])
        
        velocity_vertical = jax.lax.min(jnp.round((((117*2)/249))*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117*2/249))*(state.level.frame_count - 1)).astype(jnp.int32),1)
        
        new_y_position: jArray = jnp.multiply(jnp.equal(moving_action, JAXAtariAction.UP), position[1]-velocity_vertical) + jnp.multiply(
            jnp.equal(moving_action, JAXAtariAction.DOWN), position[1] + velocity_vertical
        ) + jnp.multiply((1 - jnp.logical_or(jnp.equal(moving_action, JAXAtariAction.UP), jnp.equal(moving_action, JAXAtariAction.DOWN))), 
                         position[1])

        new_position = position.at[1].set(new_y_position)

        
        max_y = self.consts.ENEMY_START_Y + 20
        bounded_y = jnp.clip(new_position[1], None, max_y)
        
        new_player_state = state.player._replace(y=bounded_y)
        
        return new_player_state

    @partial(jax.jit, static_argnums=(0))
    def multiple_enemies_step_bonus(self, state: AlienState) -> EnemiesState:
        """sets the velocity for the aliens this frame
        Args:
            state (AlienState): Current State of the Game
        Returns:
            EnemiesState: information on the enemies
        """
        
        new_enemy_velocity_this_frame = jax.lax.min(jnp.round(((56/121))*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((56/121))*(state.level.frame_count - 1)).astype(jnp.int32),1)
        
        new_enemies = state.enemies._replace(velocity=new_enemy_velocity_this_frame)
        return new_enemies._replace(multiple_enemies=self.enemy_step_bonus(state.enemies.multiple_enemies, state))


    @partial(jax.jit, static_argnums=(0))
    def bonus_step_freeze(self, state: AlienState) -> AlienState:
        """step funtion to handle an game freeze on death in the bonus stage
        Args:
            state (AlienState): Current State of the Game
        Returns:
            AlienState: the next game frame
        """
        new_death_frame_counter = jnp.clip(state.level.death_frame_counter - 1, 0, None)
        new_freeze_counter = jnp.clip(state.level.evil_item_frame_counter - 1, 0, None)
        new_level = state.level._replace(death_frame_counter=new_death_frame_counter)
        new_level = new_level._replace(evil_item_frame_counter=new_freeze_counter)
        new_state = state._replace(level=new_level)
        
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
        #new_enemies_state = state.enemies
        new_player_state = self.bonus_room_player_step(state, action)
        
        dead_or_alive = jnp.max(jnp.array([
                (jnp.sum(state.enemies.multiple_enemies.enemy_death_frame) > 0)*2,
                (state.level.frame_count > 500)*2,
                state.player.y < 14 + 5
            ]))
        new_death_frame_counter = (jnp.clip((dead_or_alive - 1), 0, 1) * 40).astype(jnp.int32)
        new_freeze_counter = (jnp.mod(dead_or_alive, 2) * 40).astype(jnp.int32)
        new_score = state.level.score.at[0].set((state.level.score[0] + ((jnp.mod(dead_or_alive, 2)) * self.consts.ITEM_SCORE_MULTIPLIERS[jnp.clip(state.level.difficulty_stage.astype(jnp.int32) + 2, 0, len(self.consts.ITEM_SCORE_MULTIPLIERS) - 1)]).astype(jnp.uint16)))
        new_level = state.level._replace(death_frame_counter=new_death_frame_counter,
                                         evil_item_frame_counter=new_freeze_counter,
                                         score=new_score)                
        new_state = state._replace(player=new_player_state,
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
        observation = self._get_observation(state)
        env_reward = self._get_env_reward(state,state)
        all_reward = self._get_all_rewards(state,state)
        done = self._get_done(state)
        alieninfo = self._get_info(state,all_reward)
        
        
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
        #new_state = jax.lax.cond(
        #state.level.bonus_flag,
        #lambda x: self.bonus_step(x, action),
        #lambda x: jax.lax.cond(
        #    jnp.less_equal(x.level.frame_count, 50),
        #    lambda y: state,
        #    lambda y: self.primary_step(x, action),
        #    x
        #    ),
        #state
        #)
    
        new_level_state = new_state.level._replace(frame_count=new_state.level.frame_count + 1)
        new_state = new_state._replace(level=new_level_state,
                                       step_counter= jnp.add(state.step_counter,1))

        observation = self._get_observation(new_state)
        env_reward = self._get_env_reward(state,new_state)
        all_reward = self._get_all_rewards(state,new_state)
        done = self._get_done(new_state)
        alieninfo = self._get_info(new_state,all_reward)
                
        return   (observation, new_state, env_reward, done, alieninfo)
        
        
        
                
        
        
        

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)
    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )
    def observation_space(self) -> spaces.Dict:
        
        return spaces.Dict({
            "player_x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
            "player_y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            "player_o": spaces.Box(low=0, high=17, shape=(), dtype=jnp.int32),
            "enemies_position":
                spaces.Box(low=0, high= 210, shape=(2, 6), dtype=jnp.int32),
            "enemies_killable":
                spaces.Box(low=0, high=1, shape=(6, ), dtype=jnp.int32),
            "kill_item_position":
                spaces.Box(low=0, high=210, shape=(2, ), dtype=jnp.int32),
            "score_item_position":
                spaces.Box(low=0, high=210, shape=(2, ), dtype=jnp.int32),
            "collision_map":
                spaces.Box(low=0, high=1, shape=(152, 188), dtype=jnp.int32),
        }
        )

    def _get_observation(self, state: AlienState) -> AlienObservation:
        """returns the observation to a given state

        Args:
            state (AlienState): the state of the Game

        Returns:
            AlienObservation: observation for the ai agent
        """
        new_kill_item_position = jnp.take(AlienConstants.ITEM_ARRAY, state.level.current_active_item_index, axis=0)[:2]
        return AlienObservation(
            player_x= state.player.x,
            player_y= state.player.y,
            player_o= jnp.array(state.player.orientation, jnp.int32),
            enemies_position= jnp.array([state.enemies.multiple_enemies.x,state.enemies.multiple_enemies.y]),
            enemies_killable= state.enemies.multiple_enemies.killable,
            kill_item_position= new_kill_item_position,
            score_item_position= jnp.array([68 ,  56]),
            collision_map = jnp.astype(BACKGROUND_COLLISION_MAP, jnp.int32)
        )
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AlienState, all_rewards: jnp.ndarray=None) -> AlienInfo:
        """returns the informtion

        Args:
            state (AlienState): state of the game
            all_rewards (jnp.ndarray, optional):  Defaults to None.

        Returns:
            AlienInfo: the information
        """
        return AlienInfo(
            score=state.level.score,
            step_counter = state.step_counter,
            all_rewards= all_rewards
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
        score = state.level.score - previous_state.level.score 
        return score[0]
    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: AlienObservation) -> jnp.ndarray:
        """Converts the observation to a flat array."""
        ret = jnp.concatenate(
            [
                obs.player_x.flatten(),
                obs.player_y.flatten(),
                obs.player_o.flatten(),
                obs.enemies_position.flatten(),
                obs.enemies_killable.flatten(),
                obs.kill_item_position.flatten(),
                obs.score_item_position.flatten(),
                obs.collision_map.flatten(),
            ]
        )
        return jnp.squeeze(ret)
    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: AlienState, state: AlienState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards


    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AlienState) -> bool:
        """indicades game over

        Args:
            state (AlienState): state of the game

        Returns:
            bool: done
        """
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
        egg_collision_map_left: jnp.ndarray = jnp.zeros((egg_state.shape[1]))
        egg_collision_map_right: jnp.ndarray = jnp.zeros((egg_state.shape[1]))
        # Determine coord range occupied by the player
        x_lower: jnp.ndarray = player_x
        x_higher: jnp.ndarray = jnp.add(player_x, self.consts.PLAYER_WIDTH)
        y_lower: jnp.ndarray = player_y
        y_higher: jnp.ndarray = jnp.add(player_y, self.consts.PLAYER_HEIGHT)

        def check_egg_player_collision(i, l_r, egg_col_map):
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
            egg_col_map = egg_col_map.at[i].set(has_collision.astype(np.uint8))
            return egg_col_map
        # Generate full collision map for all eggs
        egg_collision_map_left = jax.lax.fori_loop(0, egg_state.shape[1], lambda cnt, cm, cnst=0 : check_egg_player_collision(i=cnt, l_r=cnst, egg_col_map=cm), egg_collision_map_left)
        egg_collision_map_right = jax.lax.fori_loop(0, egg_state.shape[1],lambda cnt, cm, cnst=1 :  check_egg_player_collision(i=cnt, l_r=cnst, egg_col_map=cm), egg_collision_map_right)
        # Multiply with current active egg-state to prevent the same egg from being collected twice
        score_increas_left: jnp.ndarray = jnp.sum(jnp.multiply(egg_collision_map_left, egg_state[0,:, 2]))
        score_increas_right: jnp.ndarray = jnp.sum(jnp.multiply(egg_collision_map_right, egg_state[1,:, 2]))
        # Multiply collision map onto egg-state to set visible attribute to the appropriate value
        egg_collision_map_left = jnp.subtract(1, egg_collision_map_left)
        egg_collision_map_right = jnp.subtract(1, egg_collision_map_right)

        new_egg_presence_map_left: jnp.ndarray = jnp.multiply(egg_collision_map_left, egg_state[0,:, 2]).astype(jnp.int32)
        new_egg_presence_map_right: jnp.ndarray = jnp.multiply(egg_collision_map_right, egg_state[1,:, 2]).astype(jnp.int32)
        egg_state = egg_state.at[0,:, 2].set(new_egg_presence_map_left)
        egg_state = egg_state.at[1,:, 2].set(new_egg_presence_map_right)
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
        single_item_col_map: jArray = None 
        state: AlienState = None 
        player_x: jArray = None 
        player_y: jArray = None
        
        single_item_col_map, state, player_x, player_y = batched_val
        x_lower: jnp.ndarray = player_x
        x_higher: jnp.ndarray = jnp.add(player_x, self.consts.PLAYER_WIDTH)
        y_lower: jnp.ndarray = player_y
        y_higher: jnp.ndarray = jnp.add(player_y, self.consts.PLAYER_HEIGHT)
        has_collision = jnp.logical_and(state.items[i, 0]>= x_lower,
                                        jnp.logical_and(state.items[i, 0] < x_higher,
                                                        jnp.logical_and(state.items[i, 1] >= y_lower,
                                                                        state.items[i, 1] < y_higher)))
        has_collision = jnp.logical_and(has_collision, state.items[i, 2] == 1)#only check for collision if item is active
        has_collision = jnp.logical_and(has_collision, 1 - jnp.logical_and(state.items[i, 3] == 0 ,state.player.flame.flame_flag))#disable collision for evil item when flamethrower is active
        single_item_col_map = single_item_col_map.at[i].set(has_collision.astype(np.uint8))
        return (single_item_col_map, state, player_x, player_y)

    def spawn_score_item(self, items, state: AlienState):
        """sets items on the field
        Args:
            items (_type_):
        Returns:
            _type_: augmented items, new_spawn_score_item_flag, new_score_item_counter
        """
        difficulty_stage = jnp.astype(state.level.difficulty_stage, jnp.int32)
        new_items = items.at[3, 2:].set(jnp.array([1, jnp.clip(difficulty_stage + 1, 1, 6)]))
        
        return new_items, 0, 0
    
    @partial(jax.jit, static_argnums=(0,))
    def item_step(self, player_x: jnp.ndarray, player_y: jnp.ndarray, score: jnp.ndarray, state: AlienState):
        """handles everything item related

        Args:
            player_x (jnp.ndarray): 
            player_y (jnp.ndarray): 
            score (jnp.ndarray): 
            state (AlienState): state of the game

        Returns:
            _type_: item flags and informations
        """
        item_collision_map: jnp.ndarray = jnp.zeros((state.items.shape[0]))
        # Determine coord range occupied by the player
        

        
        
        item_collision_map, _, _, _ = jax.lax.fori_loop(0, state.items.shape[0], self.check_item_player_collision, (item_collision_map, state, player_x, player_y))

        # Berechne Score-Erhöhung basierend auf Item-Typ
        score_increases = jnp.multiply(
            item_collision_map,  # Kollisionsmaske
            jnp.multiply(
                state.items[:, 2],  # Aktive Items
                self.consts.ITEM_SCORE_MULTIPLIERS[state.items[:, 3]]  # Score-Multiplikator für jeden Item-Typ
            )
        )
        
        new_evil_item_frame_counter= jax.lax.cond(jnp.logical_and(jnp.equal(item_collision_map[state.level.current_active_item_index], 1), jnp.less(state.level.current_active_item_index,3)),
                                                  lambda x, y: jnp.array(y).astype(jnp.int32),
                                                  lambda x, y: x,
                                                  state.level.evil_item_frame_counter, state.level.evil_item_duration
                                                  )

        new_items = jax.lax.cond(
            jnp.logical_and(jnp.equal(item_collision_map[state.level.current_active_item_index], 1),
                            jnp.less(state.level.current_active_item_index, 2)),
            lambda x, y: x.at[y, 2].set(1),
            lambda x, y: x,
            state.items, state.level.current_active_item_index+1
        )

        new_current_active_item_index = jax.lax.cond(
            jnp.equal(jnp.equal(item_collision_map[state.level.current_active_item_index], 1),jnp.less(state.level.current_active_item_index, 3)),
            lambda x: jnp.add(x, 1),  # Wenn Item aktiv ist, gehe zum nächsten Item
            lambda x: x,  # Wenn Item nicht aktiv ist, behalte den aktuellen Index bei
            operand=state.level.current_active_item_index
        )

        
        
        new_score_item_counter = state.level.score_item_counter + 1

        new_items, new_spawn_score_item_flag_1, new_score_item_counter = jax.lax.cond(
            jnp.logical_and(
                (106 - jnp.sum(state.eggs[:, :, 2])) == 41, #41,
                state.level.spawn_score_item_flag_1
                ),
            lambda items, state, _: self.spawn_score_item(items, state),
            lambda items, state, sc: (items, state.level.spawn_score_item_flag_1, sc),
            new_items, state, new_score_item_counter
        )
        new_items, new_spawn_score_item_flag_2, new_score_item_counter = jax.lax.cond(
            jnp.logical_and(
                (106 - jnp.sum(state.eggs[:, :, 2])) == 82, #82,
                state.level.spawn_score_item_flag_2
                ),
            lambda items, state, _: self.spawn_score_item(items, state),
            lambda items, state, sc: (items, state.level.spawn_score_item_flag_2, sc),
            new_items, state, new_score_item_counter
        )
        
        new_items = jax.lax.cond(
            new_score_item_counter == 600,
            lambda x: x.at[3, 2].set(0),
            lambda x : x,
            new_items
        )
        
        item_collision_map = jnp.subtract(1, item_collision_map)
        new_item_presence_map: jnp.ndarray = jnp.multiply(item_collision_map, new_items[:, 2]).astype(jnp.int32)
        new_items = new_items.at[:, 2].set(new_item_presence_map)

        # Aktualisiere Score
        score_increase = jnp.sum(score_increases)
        new_score = jnp.add(score, score_increase).astype(jnp.uint16)

        return new_items, new_score, new_current_active_item_index, new_evil_item_frame_counter, new_spawn_score_item_flag_1, new_spawn_score_item_flag_2, new_score_item_counter
    
    @partial(jax.jit, static_argnums=(0,))
    def modify_wall_collision(self, new_position: jnp.ndarray):
        """Collision handling for game-field bounds.
           Alters an objects position to lie within the confines of the playfield

        Args:
            new_position (jnp.ndarray): new object position

        Returns:
            _type_: a valid position
        """

        permissible_lower_bounds: jnp.ndarray = jnp.array([0, 0], dtype=np.int32)
        permissible_upper_bound: jnp.ndarray = jnp.array([self.consts.WIDTH - self.consts.PLAYER_WIDTH , self.consts.HEIGHT - self.consts.PLAYER_HEIGHT ], dtype=np.int32)
        lower_bound_check: jnp.ndarray = jnp.vstack([new_position, permissible_lower_bounds])
        lower_checked = jnp.max(lower_bound_check, axis=0, keepdims=False)

        upper_bound_check: jnp.ndarray = jnp.vstack([lower_checked, permissible_upper_bound])
        checked = jnp.min(upper_bound_check, axis=0, keepdims=False)
        return checked


    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: AlienState, action: jnp.ndarray) -> PlayerState:
        """Handles full step for the player

        Args:
            state (AlienState): Current State of the Game
            action (jnp.ndarray): player action

        Returns:
            PlayerState: player information
        """
                
        velocity_horizontal = jax.lax.min(jnp.round(((117/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)
        velocity_vertical = jax.lax.min(jnp.round((((117*(2))/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117*(2)/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)
 
        old_orientation = state.player.orientation
        state_player_orientation = old_orientation
        
        # maps the action onto the relevant movement-actions
        moving_action = jax.lax.cond(jnp.greater(action,9),
                                      lambda ac: jnp.mod(ac,10)+2,
                                      lambda ac: ac,
                                      operand=action)
        
        # decides orientation based on the imput
        # TODO: THIS IS VERY BAD
        
        state_player_orientation = jax.lax.switch(moving_action, [
                lambda x, vel_v, vel_h: x.player.orientation,
                lambda x, vel_v, vel_h: x.player.orientation,
                lambda x, vel_v, vel_h: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(x, 0, -vel_v),
                                       lambda x: JAXAtariAction.UP, 
                                       lambda x: x,
                                       x.player.orientation),   # UP
                
                lambda x, vel_v, vel_h: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(x, vel_h, 0),
                                       lambda x: JAXAtariAction.RIGHT, 
                                       lambda x: x,
                                       x.player.orientation),   # RIGHT
                
                lambda x, vel_v, vel_h: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(x, -vel_h, 0),
                                       lambda x: JAXAtariAction.LEFT, 
                                       lambda x: x,
                                       x.player.orientation),   # LEFT
        
                lambda x, vel_v, vel_h: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(x, 0, vel_v),
                                       lambda x: JAXAtariAction.DOWN, 
                                       lambda x: x,
                                       x.player.orientation),   # DOWN
                
                lambda x, vel_v, vel_h: jax.lax.cond(jnp.logical_and(self.other_slightly_weirder_check_for_player_collision(x, vel_h, 0),jnp.not_equal(x.player.orientation,JAXAtariAction.RIGHT)),
                                       lambda x, y: JAXAtariAction.RIGHT, 
                                       lambda x, y: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(x, 0, -y),
                                                              lambda x: JAXAtariAction.UP,
                                                              lambda x: x,
                                                              x.player.orientation),
                                       x, vel_v),   # UPRIGHT 
                
                lambda x, vel_v, vel_h: jax.lax.cond(jnp.logical_and(self.other_slightly_weirder_check_for_player_collision(x, 0, -vel_v),jnp.not_equal(x.player.orientation,JAXAtariAction.UP)),
                                       lambda x, y: JAXAtariAction.UP, 
                                       lambda x, y: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(x, -y, 0),
                                                              lambda x: JAXAtariAction.LEFT,
                                                              lambda x: x,
                                                              x.player.orientation),
                                       x, vel_h),   # UPLEFT 
                
                lambda x, vel_v, vel_h: jax.lax.cond(jnp.logical_and(self.other_slightly_weirder_check_for_player_collision(x, 0, vel_v),jnp.not_equal(x.player.orientation,JAXAtariAction.DOWN)),
                                       lambda x, y: JAXAtariAction.DOWN, 
                                       lambda x, y: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(x, y, 0),
                                                              lambda x: JAXAtariAction.RIGHT,
                                                              lambda x: x,
                                                              x.player.orientation),
                                       x, vel_h),   #DOWNRIGHT
                
                lambda x, vel_v, vel_h: jax.lax.cond(jnp.logical_and(self.other_slightly_weirder_check_for_player_collision(x, -vel_h, 0),jnp.not_equal(x.player.orientation,JAXAtariAction.LEFT)),
                                       lambda x, y: JAXAtariAction.LEFT,
                                       lambda x, y: jax.lax.cond(self.other_slightly_weirder_check_for_player_collision(x, 0, y),
                                                              lambda x: JAXAtariAction.DOWN,
                                                              lambda x: x,
                                                              x.player.orientation),
                                       x, vel_v), #DOWNLEFT  
                              
            ], state, velocity_vertical, velocity_horizontal)
        #Determine last horizontal orientation, this is necessary for correctly displaying the player sprite
        
        last_horizontal_orientation = jax.lax.cond(
            jnp.logical_or(state_player_orientation == JAXAtariAction.LEFT, state_player_orientation == JAXAtariAction.RIGHT),
            lambda spo, lho: spo,
            lambda spo, lho: lho,
            state_player_orientation, state.player.last_horizontal_orientation
        )

        # Handle movement at this point:
        # Choose movement function according to index of proposed action.
        position = jnp.array([state.player.x, state.player.y])
        position = teleport_object(position, state.player.orientation, action)
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
                                   lambda np: np[0] - 6,
                                   lambda np: np[0] + 10,
                                   new_position)
        
        new_flame = state.player.flame._replace(
            x=jnp.array(new_flame_x).astype(jnp.int32),
            y=jnp.array(new_position[1] + 6).astype(jnp.int32),
            flame_counter=jnp.array(new_flame_counter).astype(jnp.int32),
            flame_flag=jnp.array(new_flame_flag).astype(jnp.int32)
        )

        return state.player._replace(
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
        """Handles step for all enemies."""
        
        main_key, *subkeys = jax.random.split(state.enemies.rng_key, self.consts.ENEMY_AMOUNT_BONUS_STAGE + 1)
        enemy_keys = jnp.array(subkeys)
        
        new_multiple_enemies = self.enemy_step(state.enemies.multiple_enemies, state)

        new_multiple_enemies = new_multiple_enemies._replace(
            key=enemy_keys,
            )

        new_enemies_state = state.enemies._replace(
            multiple_enemies=new_multiple_enemies,
            rng_key=main_key
        )
        
        return new_enemies_state


def enemy_step(enemy: SingleEnemyState, state: AlienState, cnsts: AlienConstants)-> SingleEnemyState:
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
                    jnp.less(random_value, enemy.mode_constants.mode_change_probability),  # Probability
                    lambda en: (0, en.mode_constants.chase_duration),  # Chase mode
                    lambda en: (1, en.mode_constants.scatter_duration),  # Scatter mode
                    enemy
                    )
    
            def check_wall_collison(new_pos):
                """Returns Boolean if step in a direction can be made"""
    
                tmp_pos = check_for_wall_collision(state.enemies.collision_map, state.level.collision_map, position, new_pos)
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
            allowed_directions_no_opp = allowed_directions.at[opposite_table[jnp.subtract(enemy.orientation, 2)]].set(0)
        
            allowed_directions_no_opp = jax.lax.cond(
                jnp.logical_and(jnp.equal(enemy.mode_type, 0), jnp.equal(enemy.mode_duration, enemy.mode_constants.chase_duration)),
                lambda a, b: a,
                lambda a, b: b,
                allowed_directions, allowed_directions_no_opp
            )
    
            new_mode_type, new_mode_duration = jax.lax.cond(
                jnp.equal(enemy.mode_duration, 0),
                lambda x: new_mode_fun(x),
                lambda x: (x.mode_type, jnp.subtract(x.mode_duration, 1)),
                enemy
            )
            
            new_mode_type = enemy.killable*2
        
            new_mode_type, new_mode_duration = jax.lax.cond(
                jnp.logical_and(
                    jnp.logical_and(
                        jnp.logical_and(
                            jnp.less(distance_to_point(position, state.player.x, state.player.y), 20), 
                            jnp.equal(state.player.flame.flame_flag, 1)), 
                        jnp.logical_or(
                            jnp.equal(enemy.orientation, JAXAtariAction.RIGHT), 
                            jnp.equal(enemy.orientation, JAXAtariAction.LEFT))), 
                    jnp.less(enemy.mode_type, 3)),
                lambda x: (3, cnsts.FLAME_FRIGHTENED_DURATION),  # Frightened mode
                lambda x: (new_mode_type, new_mode_duration), 
                enemy
            )
            
            new_direction = jax.lax.switch(new_mode_type,[
                lambda _state, _enemy, _dir, _dir_noop, _all_dir: chase_point(_state.player.x, _state.player.y, _dir_noop, _all_dir), # Chase mode
                lambda _state, _enemy, _dir, _dir_noop, _all_dir: chase_point(_enemy.mode_constants.scatter_point_x, _enemy.mode_constants.scatter_point_y, _dir_noop, _all_dir), # Scatter mode
                lambda _state, _enemy, _dir, _dir_noop, _all_dir: frightend(_state.player.x, _state.player.y, _dir_noop), # Frightened mode item  
                lambda _state, _enemy, _dir, _dir_noop, _all_dir: frightend(_state.player.x, _state.player.y, _dir) # Frightened mode flame                   
            ], state, enemy, allowed_directions, allowed_directions_no_opp, steps_in_all_directions)
   
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
                lambda _enemy: _enemy.orientation,
                lambda _enemy: _enemy.last_horizontal_orientation,
                enemy
            )
    
            position = teleport_object(position, enemy.orientation, new_direction)
    
            slowness = jnp.where(jnp.equal(enemy.mode_type,2),0.75,1)
            
            velocity_horizontal = jax.lax.min(jnp.round(((117/249)*slowness + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117/249)*slowness + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)
            velocity_vertical = jax.lax.min(jnp.round(((117*2*slowness/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117*2*slowness/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)

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
                    jnp.ndarray: bolean value for collision
                """
                has_collision = check_collision_between_two_sprites(sprite1=PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK, 
                                                                    pos1_x=new_position[0], 
                                                                    pos1_y=new_position[1], 
                                                                    sprite2=PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK, 
                                                                    pos2_x=state.player.x, pos2_y=state.player.y)
                has_collision = jnp.logical_and(
                    has_collision,
                    jnp.equal(enemy.enemy_spawn_frame,0)
                    )
                
                return has_collision
            
            new_enemy_existence = jax.lax.cond(jnp.logical_and(jnp.less_equal(enemy.enemy_spawn_frame, 9), jnp.equal(state.level.evil_item_frame_counter, state.level.evil_item_duration-1)),
                                                    lambda _: jnp.array(1),  # Enemy exists
                                                    lambda _: jax.lax.cond(jnp.equal(state.level.evil_item_frame_counter,1), 
                                                                           lambda _: jnp.array(0), 
                                                                           lambda _: enemy.existence, 
                                                                           None),#enemy.existence,
                                                    None)
            new_enemy_killable = jnp.logical_and(jnp.logical_and(jnp.logical_not(enemy.has_been_killed), jnp.greater(state.level.evil_item_frame_counter, 0)), new_enemy_existence).astype(jnp.int32)


            new_enemy_death_frame = jax.lax.cond(jnp.logical_and(jnp.logical_and(check_for_enemy_player_collision(state, new_position), new_enemy_killable),
                                                                 jnp.equal(enemy.enemy_death_frame,0)),
                                                 lambda _:  jnp.array(60),
                                                 lambda _: enemy.enemy_death_frame,
                                                 None)

            new_enemy_death_frame = jax.lax.cond(jnp.greater(new_enemy_death_frame, 0),
                                                 lambda _: jnp.subtract(new_enemy_death_frame, 1),
                                                 lambda _: enemy.enemy_death_frame,
                                                 None)
            
            spawn_offset_counter = jnp.sum(state.enemies.multiple_enemies.enemy_spawn_frame)

            new_enemy_spawn_frame, new_enemy_has_been_killed = jax.lax.cond(
                jnp.equal(new_enemy_death_frame,1), 
                lambda x,y: (jnp.full_like(enemy.enemy_spawn_frame, 11 + spawn_offset_counter + (spawn_offset_counter % 2)), jnp.array(1)), #(spawn_offset_counter | (spawn_offset_counter != 0))
                lambda x,y: jax.lax.cond(jnp.equal(state.level.evil_item_frame_counter, state.level.evil_item_duration - 1),
                        lambda z,c: (enemy.enemy_spawn_frame, jnp.array(0)),
                        lambda z,c: (enemy.enemy_spawn_frame, enemy.has_been_killed),
                None,None),
            None,None)
            
            new_direction, new_last_horizontal_orientation = jax.lax.cond(jnp.greater_equal(new_enemy_spawn_frame, 11),
                                                                          lambda _: (JAXAtariAction.RIGHT, JAXAtariAction.RIGHT),
                                                                          lambda _: (new_direction, new_last_horizontal_orientation),
                                                                          None)
            
            new_x, new_y = jax.lax.cond(
                jnp.greater_equal(new_enemy_spawn_frame, 11),
                lambda _: (cnsts.ENEMY_SPAWN_X, cnsts.ENEMY_SPAWN_Y),
                lambda _: (new_position[0], new_position[1]),
                None
            )
            new_position_at_death_x, new_position_at_death_y= jax.lax.cond(jnp.logical_or(jnp.equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 59),jnp.equal(jnp.abs(state.level.death_frame_counter),39)),
                                                                           lambda x,y : (enemy.x, enemy.y),
                                                                            lambda x,y : (x, y),
                                              enemy.position_at_death_x, enemy.position_at_death_y)

            new_enemy0 = enemy._replace(
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
            
            new_enemy1 = jax.lax.cond(jnp.logical_or(jnp.not_equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 0),jnp.not_equal(state.level.death_frame_counter,0)),
                                      lambda x: x._replace(
                                          x = new_position_at_death_x,
                                          y = new_position_at_death_y),
                                      lambda x: x,
                                      new_enemy0)
            return new_enemy1
    
        def spawn_enemy_step(enemy: SingleEnemyState, state: AlienState)-> SingleEnemyState:
            """Step for a single enemy that is currently in spawning animation"""
    
            def add_velocity(y, orientation, velocity, spawn_frame):
                return jnp.add(y, velocity), orientation, spawn_frame
    
            def substract_velocity(y, orientation, velocity, spawn_frame):
                return jnp.subtract(y, velocity), orientation, spawn_frame
    
            def change_to_down(y, spawn_frame):
                return jnp.add(y, 1), JAXAtariAction.LEFT, jnp.subtract(spawn_frame, 1)
    
            def change_to_up(y, spawn_frame):
                return jnp.subtract(y, 1), JAXAtariAction.RIGHT, jnp.subtract(spawn_frame, 1)
            
            velocity_vertical = jax.lax.min(jnp.round((((117*2)/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count)).astype(jnp.int32) - jnp.round(((117*2/249) + state.level.difficulty_stage.astype(jnp.int32)/10)*(state.level.frame_count - 1)).astype(jnp.int32),1)

            bound = jax.lax.cond(jnp.equal(jnp.mod(enemy.enemy_spawn_frame , 9), 1), lambda x: cnsts.ENEMY_START_Y, lambda x: cnsts.ENEMY_SPAWN_Y - 10, 0)

            new_y, new_orientation, new_spawn_frame = jax.lax.cond(jnp.equal(enemy.orientation, JAXAtariAction.RIGHT),
                                 lambda y, sp: jax.lax.cond(jnp.less_equal(y, bound),
                                                        lambda y, sp: change_to_down(y, sp),
                                                        lambda y, sp: substract_velocity(y, enemy.orientation, velocity_vertical, sp),
                                                        y, sp
                                                        ),
                                 lambda y, sp: jax.lax.cond(jnp.greater(y, cnsts.ENEMY_SPAWN_Y),
                                                        lambda y, sp: change_to_up(y, sp),
                                                        lambda y, sp: add_velocity(y, enemy.orientation, velocity_vertical, sp),
                                                        y, sp
                                                        ),
                                 enemy.y, enemy.enemy_spawn_frame
                                )
            new_enemy_has_been_killed = jax.lax.cond(jnp.equal(state.level.evil_item_frame_counter, state.level.evil_item_duration-1),
                                                      lambda _: jnp.array(0),  # Enemy has not been killed)
                                                      lambda _: enemy.has_been_killed,
                                                      None)
            new_enemy_existence = jax.lax.cond(jnp.logical_and(jnp.less_equal(new_spawn_frame, 9), jnp.equal(state.level.evil_item_frame_counter, state.level.evil_item_duration-1)),
                                                    lambda _: jnp.array(1),  # Enemy exists
                                                    lambda _: enemy.existence,
                                                    None)
            
            
            new_enemy_killable = jnp.logical_and(jnp.logical_and(jnp.logical_not(enemy.has_been_killed), jnp.greater(state.level.evil_item_frame_counter, 0)), new_enemy_existence).astype(jnp.int32)
            
            new_enemy_death_frame = jax.lax.cond(jnp.equal(enemy.enemy_death_frame, 1),
                                                 lambda _: jnp.array(0),
                                                 lambda _: enemy.enemy_death_frame,
                                                 None)
            
            new_position_at_death_x, new_position_at_death_y= jax.lax.cond(jnp.logical_or(jnp.equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 59),jnp.equal(jnp.abs(state.level.death_frame_counter),39)),
                                                                           lambda x,y : (enemy.x, enemy.y),
                                                                           lambda x,y : (x, y),
                                                                           enemy.position_at_death_x, enemy.position_at_death_y)
            
            y = jax.lax.cond(jnp.logical_or(jnp.not_equal(jnp.sum(state.enemies.multiple_enemies.enemy_death_frame), 0),jnp.not_equal(state.level.death_frame_counter,0)),
                                      lambda y:
                                           new_position_at_death_y,
                                      lambda y: y,
                               new_y)
            
            return enemy._replace(
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
    """step funtion for all enemies in the bonus stage

    Args:
        enemy (SingleEnemyState): Enemy
        state (AlienState): state of the game
        cnsts (AlienConstants): constats because we cant pass self through

    Returns:
        SingleEnemyState: augmented Enemy
    """
    
    new_x = jnp.clip((((enemy.x + (state.enemies.velocity * (2 * (enemy.last_horizontal_orientation % 2) - 1)))) % (cnsts.WIDTH - 35*((enemy.last_horizontal_orientation + 1) % 2))), (35*(enemy.last_horizontal_orientation % 2)), cnsts.WIDTH)
    
    position = jnp.array([enemy.x, enemy.y])
    
    shadow_position = jnp.array([new_x - 35*(2 * (enemy.last_horizontal_orientation % 2) - 1), enemy.y])
    
    player_positon = jnp.array([state.player.x, state.player.y])
    collides_1 = check_collision_between_two_sprites(sprite1=state.enemies.collision_map, pos1_x=position[0], pos1_y=position[1], 
                                            sprite2=state.player.collision_map, pos2_x=player_positon[0], pos2_y=player_positon[1])
    collides_2 = check_collision_between_two_sprites(sprite1=state.enemies.collision_map, pos1_x=shadow_position[0], pos1_y=shadow_position[1], 
                                            sprite2=state.player.collision_map, pos2_x=player_positon[0], pos2_y=player_positon[1])
    new_enemy_death_frame = jax.lax.cond(
        jnp.logical_or(
            collides_1, collides_2
        ),
        lambda en:  jnp.array(1),
        lambda en: en.enemy_death_frame,
        enemy
        )

    new_enemy = enemy._replace(
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
    def __init__(self, consts: AlienConstants = None):
        super().__init__()
        self.consts = consts or AlienConstants()

        (self.map_sprite, 
        self.player_sprite, 
        self.player_death_sprites, 
        self.teleport_sprites, 
        self.flame_sprite, 
        self.enemy_sprites, 
        self.enemy_tp_sprites,
        self.enemy_death_sprites, 
        self.evil_item_sprites, 
        self.items, 
        self.egg_sprite, 
        self.egg_sprite_half, 
        self.life_sprite, 
        self.digit_sprite
        ) = self.load_sprites()
        
    def load_sprites(self):
        """loads all sprites and returns them

        Returns:
            _type_: sprites
        """
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        #background sprites
        bg = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/bg/map_sprite.npy")), (1, 0, 2))
        bg_bs = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/bg/bonus_map_sprite.npy")), (1, 0, 2))
        
        #player sprites
        pl_f1 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_animation/player1.npy")), (1, 0, 2))
        pl_f2 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_animation/player2.npy")), (1, 0, 2))
        pl_f3 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_animation/player3.npy")), (1, 0, 2))
        
        #player death sprites
        pl_d1 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_death_animation/player_death_1_sprite.npy")), (1, 0, 2))
        pl_d2 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_death_animation/player_death_2_sprite.npy")), (1, 0, 2))
        pl_d3 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_death_animation/player_death_3_sprite.npy")), (1, 0, 2))
        pl_d4 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_death_animation/player_death_4_sprite.npy")), (1, 0, 2))
        
        #player teleport sprites
        pl_t1 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_teleport_animation/teleport1.npy")), (1, 0, 2))
        pl_t2 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_teleport_animation/teleport2.npy")), (1, 0, 2))
        pl_t3 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_teleport_animation/teleport3.npy")), (1, 0, 2))
        pl_t4 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/player_teleport_animation/teleport4.npy")), (1, 0, 2))
        
        #items
        ei_f1 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/items/evil_item_1.npy")), (1, 0, 2))
        ei_f2 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/items/evil_item_2.npy")), (1, 0, 2))
        pulsar = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/items/pulsar.npy")), (1, 0, 2))
        rocket = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/items/rocket.npy")), (1, 0, 2))
        saturn = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/items/saturn.npy")), (1, 0, 2))
        starship = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/items/starship.npy")), (1, 0, 2))
        pi = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/items/pi.npy")), (1, 0, 2))
        orb = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/items/orb.npy")), (1, 0, 2))
        
        #enemy sprite
        e_f1 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/enemy_animation/enemy_walk1.npy")), (1, 0, 2))
        e_f2 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/enemy_animation/enemy_walk2.npy")), (1, 0, 2))
        e_f3 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/enemy_animation/enemy_walk3.npy")), (1, 0, 2))
        
        #enemy death sprites
        e_d1 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/alien_death_animation/alien_death1.npy")), (1, 0, 2))
        e_d2 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/alien_death_animation/alien_death2.npy")), (1, 0, 2))
        e_d3 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/alien_death_animation/alien_death3.npy")), (1, 0, 2))
        e_d4 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/alien_death_animation/alien_death4.npy")), (1, 0, 2))

        #enemy teleport sprites
        e_t1 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/enemy_teleport_animation/1.npy")), (1, 0, 2))
        e_t2 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/enemy_teleport_animation/2.npy")), (1, 0, 2))
        e_t3 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/enemy_teleport_animation/3.npy")), (1, 0, 2))
        e_t4 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/enemy_teleport_animation/4.npy")), (1, 0, 2))
        
        digit_none = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/none.npy")), (1, 0, 2))
        digit_0 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/0.npy")), (1, 0, 2))
        digit_1 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/1.npy")), (1, 0, 2))
        digit_2 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/2.npy")), (1, 0, 2))
        digit_3 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/3.npy")), (1, 0, 2))
        digit_4 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/4.npy")), (1, 0, 2))
        digit_5 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/5.npy")), (1, 0, 2))
        digit_6 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/6.npy")), (1, 0, 2))
        digit_7 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/7.npy")), (1, 0, 2))
        digit_8 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/8.npy")), (1, 0, 2))
        digit_9 = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/digits/9.npy")), (1, 0, 2))
        
        
        #egg sprites
        egg = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/egg/egg.npy")), (1, 0, 2))
        egg_h = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/egg/half_egg.npy")), (1, 0, 2))
        
        bg_sprites, _ = jr.pad_to_match([bg,bg_bs])
        
        pl_sprites, _ = jr.pad_to_match([pl_f1, pl_f2, pl_f3])
        
        pl_d_sprites, _ = jr.pad_to_match([pl_d1, pl_d2, pl_d3, pl_d4])
        
        pl_t_sprites, _ = jr.pad_to_match([pl_t1, pl_t2, pl_t3, pl_t4])
        
        ei_sprites, _ = jr.pad_to_match([ei_f1, ei_f2])
        
        i_sprites, _ = jr.pad_to_match([pulsar, rocket, saturn, starship, orb, pi])
        
        e_sprites, _ = jr.pad_to_match([e_f1, e_f2, e_f3])
        
        e_t_sprites, _ = jr.pad_to_match([e_t1, e_t2, e_t3, e_t4])
        
        # Background sprite
        SPRITE_BG = jnp.concatenate(
            [
                jnp.repeat(bg_sprites[0][None], 1, axis=0),
                jnp.repeat(bg_sprites[1][None], 1, axis=0),
            ]        
        )
        
        # Player  sprites
        SPRITE_PL = jnp.concatenate(
            [
                jnp.repeat(pl_sprites[0][None], 1, axis=0),
                jnp.repeat(pl_sprites[1][None], 1, axis=0),
                jnp.repeat(pl_sprites[2][None], 1, axis=0),
                jnp.repeat(pl_sprites[1][None], 1, axis=0),
            ]
        )
        SPRITE_PL_DT = jnp.concatenate(
            [
                jnp.repeat(pl_d_sprites[3][None], 1, axis=0),
                jnp.repeat(pl_d_sprites[2][None], 1, axis=0),
                jnp.repeat(pl_d_sprites[1][None], 1, axis=0),
                jnp.repeat(pl_d_sprites[0][None], 1, axis=0),
            ]
        )
        SPRITE_PL_TLP = jnp.concatenate(
            [
                jnp.repeat(pl_t_sprites[0][None], 1, axis=0),
                jnp.repeat(pl_t_sprites[1][None], 1, axis=0),
                jnp.repeat(pl_t_sprites[2][None], 1, axis=0),
                jnp.repeat(pl_t_sprites[3][None], 1, axis=0),
                jnp.repeat(pl_t_sprites[3][None], 1, axis=0),
            ]
        )
        SPRITE_EVL_ITM = jnp.concatenate(
            [
                jnp.repeat(ei_sprites[0][None], 1, axis=0),
                jnp.repeat(ei_sprites[1][None], 1, axis=0),
            ]
        )
        SPRITE_ITM = jnp.concatenate(
            [
                jnp.repeat(i_sprites[0][None], 1, axis=0),
                jnp.repeat(i_sprites[1][None], 1, axis=0),
                jnp.repeat(i_sprites[2][None], 1, axis=0),
                jnp.repeat(i_sprites[3][None], 1, axis=0),
                jnp.repeat(i_sprites[4][None], 1, axis=0),
                jnp.repeat(i_sprites[5][None], 1, axis=0),
            ]
        )
        
        SPRITE_ENMY = jnp.concatenate(
            [
                jnp.repeat(e_sprites[0][None], 1, axis=0),
                jnp.repeat(e_sprites[1][None], 1, axis=0),
                jnp.repeat(e_sprites[2][None], 1, axis=0),
                jnp.repeat(e_sprites[1][None], 1, axis=0),
            ]
        ) 
        
        SPRITE_ENMY_TLP = jnp.concatenate(
            [
                jnp.repeat(e_t_sprites[0][None], 1, axis=0),
                jnp.repeat(e_t_sprites[1][None], 1, axis=0),
                jnp.repeat(e_t_sprites[2][None], 1, axis=0),
                jnp.repeat(e_t_sprites[3][None], 1, axis=0),
            ]
        )
        SPRITE_ENMY_DT = jnp.concatenate(
            [
                jnp.repeat(e_d1[None], 1, axis=0),
                jnp.repeat(e_d2[None], 1, axis=0),
                jnp.repeat(e_d3[None], 1, axis=0),
                jnp.repeat(e_d4[None], 1, axis=0),
            ]
        )
        #flame sprite
        FLAME_SPRITE = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/flame/flame_sprite.npy")), (1, 0, 2))
        
        #life sprite
        LIFES = jnp.transpose(jr.loadFrame(os.path.join(MODULE_DIR, "sprites/alien/life/life_sprite.npy")), (1, 0, 2))
        
        #digit sprites
        DIGITS = jnp.stack([digit_none, digit_0, digit_1, digit_2, digit_3, digit_4, digit_5, digit_6, digit_7, digit_8, digit_9])
    
        
        
        return (
            SPRITE_BG,
            SPRITE_PL,
            SPRITE_PL_DT,
            SPRITE_PL_TLP,
            FLAME_SPRITE,
            SPRITE_ENMY,
            SPRITE_ENMY_TLP,
            SPRITE_ENMY_DT,
            SPRITE_EVL_ITM,
            SPRITE_ITM,
            egg,
            egg_h,
            LIFES,
            DIGITS
        )
        
    @partial(jax.jit, static_argnums=(0))# This is ok in this context, as digits never changes throughout the game...
    def get_score_sprite(self, score: jnp.ndarray) -> jnp.ndarray:
        """Takes numerical representation of the current score and composes a score sprite from individual digit sprites

        Args:
            score (jnp.ndarray): current score

        Returns:
            jnp.ndarray: full score sprite matching the score
        """
        # Set dimensions of final sprite
        # Until 10k support
        final_sprite: jnp.ndarray = jnp.zeros((5*self.consts.DIGIT_WIDTH + 4*self.consts.DIGIT_OFFSET, self.consts.DIGIT_HEIGHT, 4), np.uint8)
        k_10_sprite_index: jnp.ndarray = jnp.mod(jnp.floor_divide(score, jnp.array([10000])), 10) + 1
        thousands_sprite_index: jnp.ndarray = jnp.mod(jnp.floor_divide(score, jnp.array([1000])), 10) + 1
        hundreds_sprite_index: jnp.ndarray = jnp.mod(jnp.floor_divide(score, jnp.array([100])), 10) + 1
        tens_sprite_index: jnp.ndarray = jnp.mod(jnp.floor_divide(score, jnp.array([10])), 10) + 1
        
        # Ones sprite is always displayed
        ones_sprite_index: jnp.ndarray = jnp.add(jnp.mod(score, jnp.array([10])), 1)
        # Now remove leading zeros:
        k_10_sprite_index = jnp.multiply(k_10_sprite_index, jnp.astype(jnp.greater_equal(score, 10000), jnp.int32))
        thousands_sprite_index = jnp.multiply(thousands_sprite_index, jnp.astype(jnp.greater_equal(score, 1000), jnp.int32))
        hundreds_sprite_index = jnp.multiply(hundreds_sprite_index, jnp.astype(jnp.greater_equal(score, 100), jnp.int32))
        tens_sprite_index = jnp.multiply(tens_sprite_index, jnp.astype(jnp.greater_equal(score, 10), jnp.int32))
        
        final_sprite = final_sprite.at[0:self.consts.DIGIT_WIDTH, ...].set(jnp.squeeze(self.digit_sprite[k_10_sprite_index, ...]))
        final_sprite = final_sprite.at[self.consts.DIGIT_WIDTH + self.consts.DIGIT_OFFSET:2*self.consts.DIGIT_WIDTH + self.consts.DIGIT_OFFSET,...].set(jnp.squeeze(self.digit_sprite[thousands_sprite_index, ...]))
        final_sprite = final_sprite.at[self.consts.DIGIT_WIDTH*2 + self.consts.DIGIT_OFFSET*2:self.consts.DIGIT_WIDTH*3 + self.consts.DIGIT_OFFSET*2,...].set(jnp.squeeze(self.digit_sprite[hundreds_sprite_index, ...]))
        final_sprite = final_sprite.at[self.consts.DIGIT_WIDTH*3 + self.consts.DIGIT_OFFSET*3:self.consts.DIGIT_WIDTH*4 + self.consts.DIGIT_OFFSET*3,...].set(jnp.squeeze(self.digit_sprite[tens_sprite_index, ...]))
        final_sprite = final_sprite.at[self.consts.DIGIT_WIDTH*4 + self.consts.DIGIT_OFFSET*4:self.consts.DIGIT_WIDTH*5 + self.consts.DIGIT_OFFSET*4,...].set(jnp.squeeze(self.digit_sprite[ones_sprite_index, ...]))
        
        return final_sprite
    

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: AlienState):
        """Jitted rendering function. receives the alien state, and returns a rendered frame

        Args:
            state (AlienState): state of the game

        Returns:
            jnp.ndarray: Returns only the RGB channels, no alpha
        """
        

        sprite_cycle_1 = jnp.array([3, 3, 2, 1, 0])  # Für Positionen 7-11 
        sprite_cycle_2 = jnp.array([0, 1, 2, 3, 3])
        sprite_cycle_3 = jnp.array([0,0,1,1,2,2,3,3])
        sprite_cycle_4 = jnp.array([3,3,2,2,1,1,0,0])
        
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
    
        # Alien rendering done with for_i loop, allows for dynamic # of aliens.
        def render_loop_alien(i, canvas):
            """render loop for all enemies

            Args:
                i (_type_): index
                canvas (_type_): canvas that is displayed

            Returns:
                _type_: augmented canvas
            """

            last_horizontal_orientations = state.enemies.multiple_enemies.last_horizontal_orientation
            killable = state.enemies.multiple_enemies.killable.at[i].get()
            enemy_death_frame = state.enemies.multiple_enemies.enemy_death_frame.at[i].get()
            x_positions =state.enemies.multiple_enemies.x
            y_positions = state.enemies.multiple_enemies.y
            x = x_positions[i]
            y = y_positions[i]
            last_horizontal_orientations = last_horizontal_orientations[i]

           
            color = jax.lax.cond(killable, 
                                 lambda _: self.consts.OTHER_BLUE,
                                 
                                 lambda _: self.consts.ENEMY_COLORS[i],
                                 operand=None)
            
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
            x_higher_player = jnp.add(state.player.x, self.consts.PLAYER_WIDTH)
            x_lower_player = jnp.add(state.player.x, 0)
            y_higher_player = jnp.add(state.player.y, self.consts.PLAYER_HEIGHT)
            y_lower_player = jnp.add(state.player.y, 0)

            # Check if player sprite crosses a certian point in enemy sprite
            has_collision_enemy = jnp.logical_and(x >= x_lower_player,
                                                   jnp.logical_and(x < x_higher_player,
                                                                   jnp.logical_and(jnp.add(y, self.consts.ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW) >= y_lower_player,
                                                                                   jnp.add(y, self.consts.ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH) < y_higher_player)))
            enemy_sprite0 = jax.lax.cond(
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

            enemy_sprite0 = colorize_sprite(enemy_sprite0, color)
            enemy_sprite = jax.lax.cond(jnp.greater(enemy_death_frame,0),
                                               lambda _:jax.lax.switch(
                                                   # Konvertiere die Bereiche in einzelne Fälle
                                                   jax.lax.cond(
                                                       jnp.logical_and(enemy_death_frame >= 47, enemy_death_frame <= 59),
                                                       lambda _: 0,  # Fall für alien_death1
                                                       lambda _: jax.lax.cond(
                                                           jnp.logical_and(enemy_death_frame >= 31, enemy_death_frame <= 46),
                                                           lambda _: 1,  # Fall für alien_death2
                                                           lambda _: jax.lax.cond(
                                                               jnp.logical_and(enemy_death_frame >= 15, enemy_death_frame <= 30),
                                                               lambda _: 2,  # Fall für alien_death3
                                                               lambda _: jax.lax.cond(
                                                                   jnp.logical_and(enemy_death_frame >= 2, enemy_death_frame <= 14),
                                                                   lambda _: 3,  # Fall für alien_death4
                                                                   lambda _: 4,  # Fall für player_zeroed bei death_frame == 1
                                                                   operand=None
                                                               ),
                                                               operand=None
                                                           ),
                                                           operand=None
                                                       ),
                                                       operand=None
                                                   ),
                                                   [
                                                       lambda _: jnp.flip(colorize_sprite(self.enemy_death_sprites[0].astype(jnp.uint8),self.consts.OTHER_BLUE),axis=0),  # alien_death1
                                                       lambda _: jnp.flip(colorize_sprite(self.enemy_death_sprites[1].astype(jnp.uint8),self.consts.OTHER_BLUE),axis=0),  # alien_death2
                                                       lambda _: jnp.flip(colorize_sprite(self.enemy_death_sprites[2].astype(jnp.uint8),self.consts.OTHER_BLUE),axis=0),# alien_death3
                                                       lambda _: jnp.flip(colorize_sprite(self.enemy_death_sprites[3].astype(jnp.uint8),self.consts.OTHER_BLUE),axis=0),# alien_death4
                                                       lambda _: jnp.zeros_like(enemy_sprite0,jnp.uint8)  # player_zeroed
                                                   ],
                                                   operand=None
                                               ),
                                               lambda _ : enemy_sprite0,
                                               operand=None)

            flipped_enemy_sprite = jax.lax.cond(
                    last_horizontal_orientations == JAXAtariAction.RIGHT,
                    lambda s: jnp.flip(s, axis=0),
                    lambda s: s,
                    enemy_sprite
                )
            
            # Handles blinking of the enemy sprite            
            blinking_enemy_sprite = jax.lax.cond( i == 2,
                                                 lambda _: jax.lax.cond(jnp.logical_and(jnp.mod(state.level.frame_count,2) ==1,state.enemies.multiple_enemies.blink[i]),
                                                                        lambda _: jnp.zeros(flipped_enemy_sprite.shape, jnp.uint8),
                                                                        lambda _: flipped_enemy_sprite,
                                                                        operand=None),
                                                 lambda _: jax.lax.cond(jnp.logical_and(jnp.mod(state.level.frame_count,2) ==0,state.enemies.multiple_enemies.blink[i]),
                                                                        lambda _: jnp.zeros(flipped_enemy_sprite.shape, jnp.uint8),
                                                                        lambda _: flipped_enemy_sprite,
                                                                        operand=None),
                                                 operand=None)

            blinking_enemy_sprite = blinking_enemy_sprite * state.enemies.multiple_enemies.active_enemy[i].astype(jnp.uint8)
            
            # Ensure collision_map is 3D (H, W, C) before rendering
            collision_map = PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK
            h, w = collision_map.shape[0], collision_map.shape[1]
            # Make RGB: 0 stays black, 1 (or nonzero) becomes white
            mask = (collision_map > 0).astype(jnp.uint8)
            rgb = jnp.stack([mask * 255, mask * 255, mask * 255], axis=-1)
            alpha = jnp.full((h, w, 1), 128, dtype=jnp.uint8)  # semi-transparent
            collision_map = jnp.concatenate([rgb, alpha], axis=-1)

            res = jr.render_at(canvas, y + self.consts.RENDER_OFFSET_Y, x + self.consts.RENDER_OFFSET_X, blinking_enemy_sprite)
            
            return jax.lax.cond(state.enemies.multiple_enemies.enemy_spawn_frame[i] > 9, lambda x: canvas,lambda x: res, 0)
        
        def render_loop_alien_bonus(i, canvas):
            """render loop for all aliens in the bonus stage

            Args:
                i (_type_): index
                canvas (_type_): canvas that is displayed

            Returns:
                _type_: augmented canvas
            """
            x_positions = state.enemies.multiple_enemies.x
            y_positions = state.enemies.multiple_enemies.y
            last_horizontal_orientations = state.enemies.multiple_enemies.last_horizontal_orientation
            x = x_positions[i]
            y = y_positions[i]
            last_horizontal_orientations = last_horizontal_orientations[i]
            
            color = self.consts.ENEMY_COLORS[jnp.mod(i, 3)]
            
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
            x_higher_player = jnp.add(state.player.x, self.consts.PLAYER_WIDTH)
            x_lower_player = jnp.add(state.player.x, 0)
            y_higher_player = jnp.add(state.player.y, self.consts.PLAYER_HEIGHT)
            y_lower_player = jnp.add(state.player.y, 0)

            # Check if player sprite crosses a certian point in enemy sprite
            has_collision_enemy = jnp.logical_and(x >= x_lower_player,
                                                   jnp.logical_and(x < x_higher_player,
                                                                   jnp.logical_and(jnp.add(y, self.consts.ENEMY_PLAYER_COLLISION_OFFSET_Y_LOW) >= y_lower_player,
                                                                                   jnp.add(y, self.consts.ENEMY_PLAYER_COLLISION_OFFSET_Y_HIGH) < y_higher_player)))
            enemy_sprite0 = jax.lax.cond(
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
            
            enemy_sprite0 = colorize_sprite(enemy_sprite0, color)
            
            flipped_enemy_sprite = jax.lax.cond(
                    last_horizontal_orientations == JAXAtariAction.RIGHT,
                    lambda s: jnp.flip(s, axis=0),
                    lambda s: s,
                    enemy_sprite0
                )
            
            collision_map = PLAYER_COLLISION_MAP_FOR_ENEMY_COLLISION_CHECK
            h, w = collision_map.shape[0], collision_map.shape[1]
            # Make RGB: 0 stays black, 1 (or nonzero) becomes white
            mask = (collision_map > 0).astype(jnp.uint8)
            rgb = jnp.stack([mask * 255, mask * 255, mask * 255], axis=-1)
            alpha = jnp.full((h, w, 1), 128, dtype=jnp.uint8)  # semi-transparent
            collision_map = jnp.concatenate([rgb, alpha], axis=-1)

            res = jr.render_at(canvas, y + self.consts.RENDER_OFFSET_Y, x + self.consts.RENDER_OFFSET_X, flipped_enemy_sprite)
            res = jr.render_at(res, y + self.consts.RENDER_OFFSET_Y, x + self.consts.RENDER_OFFSET_X - 35*(2 * (last_horizontal_orientations % 2) - 1), flipped_enemy_sprite)
            
            
            return res
            
        # Render loop for eggs, handles color & hiding of already collected eggs
        def render_loop_eggs(i, l_r, canvas):
            """render loop for the eggs

            Args:
                i (_type_): index
                l_r (_type_): left or right
                canvas (_type_): canvas that is displayed

            Returns:
                _type_: augmented canvas
            """
            x = state.eggs[l_r, i , 0]
            y = state.eggs[l_r, i , 1]

            egg_sprite = colorize_sprite(self.egg_sprite, self.consts.EGG_COLORS[state.eggs[l_r, i , 3],0])
            egg_sprite_half = colorize_sprite(self.egg_sprite_half, self.consts.EGG_COLORS[state.eggs[l_r, i , 3],1])
            render_egg = lambda canv : jr.render_at(jr.render_at(canv, y  + self.consts.RENDER_OFFSET_Y, x + self.consts.RENDER_OFFSET_X, egg_sprite),
                                                    y  + self.consts.RENDER_OFFSET_Y+1, x + self.consts.RENDER_OFFSET_X , egg_sprite_half)
            render_no_egg = lambda canv : jr.render_at(canv, y  + self.consts.RENDER_OFFSET_Y, x + self.consts.RENDER_OFFSET_X, jnp.zeros(egg_sprite.shape, jnp.uint8))
            
            rendered_c = jax.lax.cond(state.eggs[l_r, i, 2], render_egg, render_no_egg, canvas)
            return rendered_c
        
        def render_loop_items(i, canvas):
            """render loop for the items

            Args:
                i (_type_): index
                canvas (_type_): canvas that is displayed

            Returns:
                _type_: augmented canvas
            """
            x = state.items[i , 0]
            y = state.items[i , 1]
            item_sprite_identifier =  state.items[i , 3]
            sprite_index =  (state.level.frame_count // 8) % 4
            item_sprite = jax.lax.cond(
                item_sprite_identifier == 0,
                lambda x: jax.lax.cond(state.player.flame.flame_flag,
                                       lambda _: jnp.zeros(self.evil_item_sprites[x[1]].shape, jnp.uint8),
                                       lambda _: self.evil_item_sprites[x[1]],
                                       operand=None),
                lambda x: self.items[x[0] - 1],
                operand= (item_sprite_identifier, sprite_index)
            )

            colorized_item = jax.lax.cond(
                item_sprite_identifier == 0,
                lambda x: colorize_sprite(x,self.consts.ORANGE),
                lambda x: colorize_sprite(x,self.consts.YELLOW),
                item_sprite
            )
            
            # Handles blinking of items
            blinking_item_sprite = jax.lax.cond(item_sprite_identifier == 0,
                                                lambda _: jax.lax.cond(jnp.logical_and(jnp.mod(state.level.frame_count,2) == 0,state.level.blink_evil_item),
                                                                       lambda _: jnp.zeros(colorized_item.shape, jnp.uint8),
                                                                       lambda _: colorized_item,
                                                                       operand=None),
                                                lambda _: jax.lax.cond(jnp.logical_and(jnp.mod(state.level.frame_count,2) == 0,state.level.blink_current_active_item),
                                                                       lambda _: jnp.zeros(colorized_item.shape, jnp.uint8),
                                                                       lambda _: colorized_item,
                                                                       operand=None),
                                                operand=None)
            
            render_item = lambda canv : jr.render_at(canv, y + self.consts.RENDER_OFFSET_Y, x + self.consts.RENDER_OFFSET_X, blinking_item_sprite)
            render_no_item = lambda canv : jr.render_at(canv, y + self.consts.RENDER_OFFSET_Y, x + self.consts.RENDER_OFFSET_X, jnp.zeros(blinking_item_sprite.shape, jnp.uint8))

            rendered_c = jax.lax.cond(state.items[i, 2], render_item, render_no_item, canvas)
            return rendered_c

        # Render loop for lives, allwos for dynamic # of lives
        def render_loop_lifes(i, canvas):
            """render loop for the lifes

            Args:
                i (_type_): index
                canvas (_type_): canvas that is displayed

            Returns:
                _type_: augmented canvas
            """
            x = self.consts.LIFE_X + i * (self.consts.LIFE_WIDTH + self.consts.LIFE_OFFSET_X)
            y = self.consts.LIFE_Y
            colorized_life_sprite = colorize_sprite(self.life_sprite,self.consts.BASIC_BLUE)
            return jr.render_at(canvas, y , x + self.consts.RENDER_OFFSET_X , colorized_life_sprite)
        

        def player_death_animation(state: AlienState) -> jnp.ndarray:
            """returns

            Args:
                state (AlienState): the state of the game

            Returns:
                jnp.ndarray: the death player sprite
            """
            sprite_index = (jnp.abs(state.level.death_frame_counter) // 4) % 10
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


        player_sprite_frame0 = jax.lax.cond(jnp.not_equal(state.level.death_frame_counter, 0), lambda x: colorize_sprite(player_death_animation(x),self.consts.BASIC_BLUE), lambda x:jax.lax.cond(
                jnp.logical_or(
                    jnp.logical_and(jnp.logical_and(state.player.x >= 7, state.player.x <= 11), state.player.y == 80),
                    jnp.logical_and(jnp.logical_and(state.player.x >= 123, state.player.x <= 127), state.player.y == 80)
                ),
                lambda _: colorize_sprite(self.teleport_sprites[sprite_index],self.consts.BASIC_BLUE),
                lambda _: colorize_sprite(self.player_sprite[(state.level.frame_count // 4) % 4], jax.lax.cond(state.player.flame.flame_flag,
                                                                                                               lambda _: self.consts.ORANGE,
                                                                                                               lambda _: self.consts.BASIC_BLUE,
                                                                                                               operand=None)),
                operand=None
            ), state)

        enemy_death_frame = jnp.sum(state.enemies.multiple_enemies.enemy_death_frame) * (1 - state.level.bonus_flag).astype(jnp.uint8)
        player_sprite_frame = jax.lax.cond(
            jnp.not_equal(enemy_death_frame, 0),
                            lambda x: jnp.zeros(player_sprite_frame0.shape, jnp.uint8),
                            lambda x: player_sprite_frame0,
                            operand=None
        )
        # Handles horizontal flipping of the player sprite depending on the last horizontal direction (preserve sprite orientation during vertical movement)
        flipped_player_sprite = jax.lax.cond(
            jnp.logical_or(state.player.last_horizontal_orientation == JAXAtariAction.RIGHT, state.player.orientation == JAXAtariAction.RIGHT),
            lambda s: jnp.flip(s, axis=0),
            lambda s: s,
            player_sprite_frame
        )
    
        # Handles blinking of player sprite
        blinking_player_sprite = jax.lax.cond(jnp.logical_or(state.player.flame.flame_flag,state.player.blink),
                                           lambda _: jax.lax.cond(jnp.mod(state.level.frame_count,2) == 0,
                                                                  lambda _ :flipped_player_sprite,
                                                                  lambda _ : jnp.zeros(flipped_player_sprite.shape, jnp.uint8),
                                                                  operand=None),
                                           lambda _: flipped_player_sprite,
                                           operand=None)
        
        digit_sprite:jnp.ndarray = self.get_score_sprite(score=state.level.score)
        digit_sprite = colorize_sprite(digit_sprite,self.consts.BASIC_BLUE)

        def render_canvas_primary_stage(self, stage: AlienState):
            """renders the primary stage 

            Args:
                stage (AlienState): state of the game

            Returns:
                _type_: augmented canvas
            """
            canvas = jr.create_initial_frame(width=210, height=160)

            # render background
            map = jr.get_sprite_frame(self.map_sprite, 0)
            canvas = jr.render_at(canvas, self.consts.RENDER_OFFSET_Y, self.consts.RENDER_OFFSET_X, map)

            canvas = jax.lax.cond(jnp.mod(state.level.frame_count,2) ==1,
                                  lambda _: jax.lax.fori_loop(0, state.eggs.shape[1], lambda cnt, rnr, cnst=0 : render_loop_eggs(i=cnt, canvas=rnr, l_r=cnst), canvas),
                                  lambda _: jax.lax.fori_loop(0, state.eggs.shape[1], lambda cnt, rnr, cnst=1 : render_loop_eggs(i=cnt, canvas=rnr, l_r=cnst), canvas),
                                  operand=None)

            canvas = jax.lax.fori_loop(0, state.items.shape[0], render_loop_items, canvas)
            canvas = jax.lax.cond(state.level.frame_count < 100, lambda x: x, lambda x: jax.lax.fori_loop(0, self.consts.ENEMY_AMOUNT_BONUS_STAGE, render_loop_alien, x), canvas)
            canvas = jax.lax.fori_loop(0, state.level.lifes, render_loop_lifes, canvas)
        
            # rendering of player sprite
            
            collision_map = state.player.collision_map
            h, w = collision_map.shape[0], collision_map.shape[1]
            # Make RGB: 0 stays black, 1 (or nonzero) becomes white
            mask = (collision_map > 0).astype(jnp.uint8)
            rgb = jnp.stack([mask * 255, mask * 255, mask * 255], axis=-1)
            alpha = jnp.full((h, w, 1), 128, dtype=jnp.uint8)  # semi-transparent
            collision_map = jnp.concatenate([rgb, alpha], axis=-1)
            
            canvas = jr.render_at(canvas, state.player.y  + self.consts.RENDER_OFFSET_Y, state.player.x  + self.consts.RENDER_OFFSET_X, blinking_player_sprite)
        
            # Handles orientation of the flame sprite
            flipped_flame_sprite = jax.lax.cond(
                jnp.logical_or(state.player.last_horizontal_orientation == JAXAtariAction.RIGHT, state.player.orientation == JAXAtariAction.RIGHT),
                lambda s: jnp.flip(s, axis=0),
                lambda s: s,
                self.flame_sprite
            )
        
            colorized_flame_sprite = colorize_sprite(flipped_flame_sprite,self.consts.ORANGE) 

            # Handles blinking of flame sprite
            blinking_flame_sprite = jax.lax.cond(jnp.logical_and(state.player.flame.flame_flag,jnp.mod(state.level.frame_count,2) == 1),
                                                 lambda _: colorized_flame_sprite,
                                                 lambda _: jnp.zeros(colorized_flame_sprite.shape, jnp.uint8),
                                                 operand=None)


            # rendering of the flamethrower
            canvas = jr.render_at(canvas, state.player.flame.y + self.consts.RENDER_OFFSET_Y ,state.player.flame.x + self.consts.RENDER_OFFSET_X ,blinking_flame_sprite)

            canvas = jr.render_at(canvas, self.consts.SCORE_Y + self.consts.RENDER_OFFSET_Y, self.consts.SCORE_X + self.consts.RENDER_OFFSET_X, digit_sprite)

            canvas = jnp.transpose(canvas, (1, 0, 2))
            return canvas[..., 0:3]
        
        def render_canvas_bonus_stage(self, state: AlienState):
            """renders the bonus stage 

            Args:
                stage (AlienState): state of the game

            Returns:
                _type_: augmented canvas
            """
            canvas = jr.create_initial_frame(width=210, height=160)

            # render background
            map = jr.get_sprite_frame(self.map_sprite, 1)
            canvas = jr.render_at(canvas, 0, 0, map)
            
            canvas = jr.render_at(canvas, self.consts.SCORE_Y + self.consts.RENDER_OFFSET_Y, self.consts.SCORE_X + self.consts.RENDER_OFFSET_X, digit_sprite)

            canvas = jax.lax.fori_loop(0, state.level.lifes, render_loop_lifes, canvas)

            bonus_item_sprite = colorize_sprite(self.items[0], self.consts.GREEN) * (1 - jnp.clip(state.level.evil_item_frame_counter, 0, 1))

            canvas = jr.render_at(canvas, 10 + self.consts.RENDER_OFFSET_Y, self.consts.ENEMY_SPAWN_X + self.consts.RENDER_OFFSET_X, bonus_item_sprite)
            
            canvas = jax.lax.fori_loop(0, self.consts.ENEMY_AMOUNT_BONUS_STAGE, render_loop_alien_bonus, canvas)
            
            collision_map = state.player.collision_map
            h, w = collision_map.shape[0], collision_map.shape[1]
            # Make RGB: 0 stays black, 1 (or nonzero) becomes white
            mask = (collision_map > 0).astype(jnp.uint8)
            rgb = jnp.stack([mask * 255, mask * 255, mask * 255], axis=-1)
            alpha = jnp.full((h, w, 1), 128, dtype=jnp.uint8)  # semi-transparent
            collision_map = jnp.concatenate([rgb, alpha], axis=-1)
            
            canvas = jr.render_at(canvas, state.player.y  + self.consts.RENDER_OFFSET_Y, state.player.x  + self.consts.RENDER_OFFSET_X, blinking_player_sprite)
            
            canvas = jnp.transpose(canvas, (1, 0, 2))
            return canvas[..., 0:3]
            
        return jax.lax.cond(
            state.level.bonus_flag,
            lambda _: render_canvas_bonus_stage(self, state),
            lambda _: render_canvas_primary_stage(self, state),
            operand=None
        )


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
   
    fields = list(SingleEnemyState._fields)
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
    