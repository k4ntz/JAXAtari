import os
from functools import partial
from typing import List, NamedTuple, Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
import chex
from flax import struct
from jax import Array
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants

# Group: Kaan Yilmaz, Jonathan Frey
# Game: Berzerk
# Tested on Ubuntu Virtual Machine

def _create_static_procedural_sprites() -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # A black background
    procedural_bg = jnp.zeros((210, 160, 4), dtype=jnp.uint8)
    procedural_bg = procedural_bg.at[:, :, 3].set(255)

    # A 1x1 pixel for each color used in procedural recoloring
    # This ensures they are added to the palette.
    enemy_recolor_palette = jnp.array([
        [210, 210, 64, 255],    # Original enemy color
        [240, 170, 103, 255],   # Original bullet color
        [210, 210, 91, 255],    # yellow
        [186, 112, 69, 255],    # orange
        [214, 214, 214, 255],   # white
        [109, 210, 111, 255],   # green
        [239, 127, 128, 255],   # red
        [102, 158, 193, 255],   # blue
        [227, 205, 115, 255],   # yellow2
        [185, 96, 175, 255],    # pink
    ], dtype=jnp.uint8).reshape(-1, 1, 1, 4) # (N, 1, 1, 4)
    
    return {
        'background': procedural_bg,
        'recolor_palette': enemy_recolor_palette,
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Berzerk.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    static_procedural = _create_static_procedural_sprites()
    
    # Define sprite groups (for auto-padding)
    player_keys = [
        'player_idle', 'player_move_1', 'player_move_2', 'player_death',
        'player_shoot_up', 'player_shoot_right', 'player_shoot_down',
        'player_shoot_left', 'player_shoot_up_left', 'player_shoot_down_left'
    ]
    
    enemy_keys = [
        'enemy_idle_1', 'enemy_idle_2', 'enemy_idle_3', 'enemy_idle_4',
        'enemy_idle_5', 'enemy_idle_6', 'enemy_idle_7', 'enemy_idle_8',
        'enemy_move_horizontal_1', 'enemy_move_horizontal_2',
        'enemy_move_vertical_1', 'enemy_move_vertical_2', 'enemy_move_vertical_3',
        'enemy_death_1', 'enemy_death_2', 'enemy_death_3',
    ]
    
    otto_keys = ['evil_otto']

    wall_keys = [
        'mid_walls_1', 'mid_walls_2', 'mid_walls_3', 'mid_walls_4', 
        'level_outer_walls', 'door_vertical_left', 'door_vertical_right', 
        'door_horizontal_up', 'door_horizontal_down',
    ]

    config = (
        # Procedural assets
        {'name': 'background', 'type': 'background', 'data': static_procedural['background']},
        {'name': 'recolor_palette', 'type': 'procedural', 'data': static_procedural['recolor_palette']},

        # Groups (will be auto-padded)
        {'name': 'player_group', 'type': 'group', 'files': [f'{k}.npy' for k in player_keys]},
        {'name': 'enemy_group', 'type': 'group', 'files': [f'{k}.npy' for k in enemy_keys]},
        {'name': 'otto_group', 'type': 'group', 'files': [f'{k}.npy' for k in otto_keys]},
        {'name': 'wall_group', 'type': 'group', 'files': [f'{k}.npy' for k in wall_keys]},

        # Single sprites
        {'name': 'bullet_horizontal', 'type': 'single', 'file': 'bullet_horizontal.npy'},
        {'name': 'bullet_vertical', 'type': 'single', 'file': 'bullet_vertical.npy'},
        {'name': 'life', 'type': 'single', 'file': 'life.npy'},
        {'name': 'start_title', 'type': 'single', 'file': 'start_title.npy'},

        # Digits
        {'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
    )
    
    return config

class BerzerkConstants(AutoDerivedConstants):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(6, 20))
    PLAYER_SPEED: float = struct.field(pytree_node=False, default=0.4)

    EXTRA_LIFE_AT: int = struct.field(pytree_node=False, default=1000)

    ENEMY_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 16))
    MAX_NUM_ENEMIES: int = struct.field(pytree_node=False, default=7)
    MIN_NUM_ENEMIES: int = struct.field(pytree_node=False, default=5)
    MOVEMENT_PROB: float = struct.field(pytree_node=False, default=0.0025)  # probability for enemy to move
    ENEMY_SPEED: float = struct.field(pytree_node=False, default=0.1)
    ENEMY_SHOOT_PROB: float = struct.field(pytree_node=False, default=0.005)
    ENEMY_BULLET_SPEED: float = struct.field(pytree_node=False, default=0.47)

    BULLET_SIZE_HORIZONTAL: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 2))
    BULLET_SIZE_VERTICAL: Tuple[int, int] = struct.field(pytree_node=False, default=(1, 6))
    BULLET_SPEED: int = struct.field(pytree_node=False, default=2)
    MAX_BULLETS: int = struct.field(pytree_node=False, default=1)

    WALL_THICKNESS: int = struct.field(pytree_node=False, default=4)
    WALL_OFFSET: Tuple[int, int, int, int] = struct.field(pytree_node=False, default=(4, 4, 4, 30)) # left, top, right, bottom
    EXIT_WIDTH: int = struct.field(pytree_node=False, default=40)
    EXIT_HEIGHT: int = struct.field(pytree_node=False, default=64)

    DEATH_ANIMATION_FRAMES: int = struct.field(pytree_node=False, default=128)
    ENEMY_DEATH_ANIMATION_FRAMES: int = struct.field(pytree_node=False, default=8)
    
    TRANSITION_ANIMATION_FRAMES: int = struct.field(pytree_node=False, default=64)

    GAME_OVER_FRAMES: int = struct.field(pytree_node=False, default=32)

    UI_OFFSET: int = struct.field(pytree_node=False, default=30)  # pixels reserved for score at bottom
    
    # Derived constants (dynamic calculation based on static fields)
    SCORE_OFFSET_X: Optional[int] = struct.field(pytree_node=False, default=None)
    SCORE_OFFSET_Y: Optional[int] = struct.field(pytree_node=False, default=None)
    PLAYER_BOUNDS: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = struct.field(pytree_node=False, default=None)

    # Variations Evil Otto 
    ENABLE_EVIL_OTTO: bool = struct.field(pytree_node=False, default=False)    # Variation 1: enable immortal evil otto
    MORTAL_EVIL_OTTO: bool = struct.field(pytree_node=False, default=False)    # Variation 2: enable mortal evil otto (ENABLE_EVIL_OTTO has to be True)
    EVIL_OTTO_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 7))
    EVIL_OTTO_SPEED: float = struct.field(pytree_node=False, default=0.4)
    EVIL_OTTO_SPEED_SLOW: float = struct.field(pytree_node=False, default=0.2)  # Slower than player (0.4)
    EVIL_OTTO_SPEED_FAST: float = struct.field(pytree_node=False, default=0.5)  # "Amazing speed!" - faster than player
    EVIL_OTTO_DELAY: int = struct.field(pytree_node=False, default=422)
    EVIL_OTTO_RESPAWN_DELAY: int = struct.field(pytree_node=False, default=222)
    # Otto movement (bounce-phased) parameters
    OTTO_BOUNCE_CYCLE: int = struct.field(pytree_node=False, default=30)              # frames per bounce cycle
    OTTO_BOUNCE_HEIGHT: float = struct.field(pytree_node=False, default=1.5)            # pixels of peak vertical bounce 
    OTTO_VERTICAL_DRIFT_SCALE: float = struct.field(pytree_node=False, default=0.2) # drift factor towards player's Y
    OTTO_HORIZ_PHASE_START: float = struct.field(pytree_node=False, default=0.25)       # start of horizontal move phase in cycle
    OTTO_HORIZ_PHASE_END: float = struct.field(pytree_node=False, default=0.75)         # end of horizontal move phase in cycle

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: Tuple[dict, ...] = struct.field(pytree_node=False, default_factory=_get_default_asset_config)
    
    def compute_derived(self):
        """Compute derived constants based on static fields."""
        return {
            'SCORE_OFFSET_X': self.WIDTH - 58 - 6,  # window width - distance to the right - digit width
            'SCORE_OFFSET_Y': self.HEIGHT - 20 - 7,  # window height - distance to the bottom - digit height
            'PLAYER_BOUNDS': (
                (self.WALL_THICKNESS + self.WALL_OFFSET[0], self.WIDTH - self.WALL_THICKNESS - self.WALL_OFFSET[2]),
                (self.WALL_THICKNESS + self.WALL_OFFSET[1], self.HEIGHT - self.WALL_THICKNESS - self.WALL_OFFSET[3])
            ),
        }

@struct.dataclass
class PlayerState:
    pos: chex.Array                     # (2,)
    last_dir: chex.Array                # (2,)
    animation_counter: chex.Array       # (1,)
    is_firing: chex.Array               # (1,)
    bullet: chex.Array                 # (2,)
    bullet_dir: chex.Array             # (2,)
    bullet_active: chex.Array           # (1,)
    death_timer: chex.Array

@struct.dataclass
class EnemyState:
    pos: chex.Array                     # (NUM_ENEMIES, 2)
    move_axis: chex.Array               # (NUM_ENEMIES,)
    move_dir: chex.Array                # (NUM_ENEMIES,)
    alive: chex.Array                   # (NUM_ENEMIES,)
    bullets: chex.Array                 # (NUM_ENEMIES, 2)
    bullet_dirs: chex.Array             # (NUM_ENEMIES, 2)
    bullet_active: chex.Array           # (NUM_ENEMIES,)
    move_prob: chex.Array               # (1,)
    clear_bonus_given: chex.Array       # (1,)
    death_timer: chex.Array             # (NUM_ENEMIES,)
    death_pos: chex.Array               # (NUM_ENEMIES,)
    animation_counter: chex.Array       # (NUM_ENEMIES,)

@struct.dataclass
class OttoState:
    pos: chex.Array                     # (2,)
    active: chex.Array                  # (1,)
    timer: chex.Array                   # (1,)
    anim_counter: chex.Array            # (1,)

@struct.dataclass
class BerzerkState:
    player: PlayerState             
    enemy: EnemyState
    otto: OttoState
    rng: chex.PRNGKey                   # (1,)
    score: chex.Array                   # (1,)
    lives: chex.Array                   # (1,)
    room_counter: chex.Array            # (1,)
    extra_life_counter: chex.Array      # (1,)
    game_over_timer: chex.Array         # (1,)
    num_enemies: chex.Array             # (1,)
    entry_direction: chex.Array         # (1,)
    room_transition_timer: chex.Array   # (1,)


@struct.dataclass
class BerzerkObservation:
    # Player
    player_pos: jnp.ndarray        # (2,)
    player_dir: jnp.ndarray        # (2,)
    player_bullet: jnp.ndarray     # (1,2)
    player_bullet_dir: jnp.ndarray # (1,2)

    # Enemies
    enemy_pos: jnp.ndarray
    enemy_bullets: jnp.ndarray
    enemy_bullet_dirs: jnp.ndarray

    # Otto
    otto_pos: jnp.ndarray   

    # Game-level
    score: jnp.ndarray        
    lives: jnp.ndarray     


@struct.dataclass
class BerzerkInfo:
    enemies_killed: chex.Array      # (1,)
    level_cleared: chex.Array       # (1,)

@struct.dataclass
class WallGeometry:
    outer_walls: chex.Array
    door_blockers: chex.Array
    mid_walls: Tuple[chex.Array, ...]


class JaxBerzerk(JaxEnvironment[BerzerkState, BerzerkObservation, BerzerkInfo, BerzerkConstants]):
    # Minimal ALE action set (from scripts/action_space_helper.py)
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

    def __init__(self, consts: BerzerkConstants = None):
        super().__init__(consts)
        self.consts = consts or BerzerkConstants()
        self.obs_size = 111
        self.renderer = BerzerkRenderer(self.consts)
        # Initialize AABB wall geometry
        self.wall_geometry = self._define_wall_geometry()
        # Pre-pad mid wall AABBs to a uniform shape for JAX switch/indexing
        mid_list = list(self.wall_geometry.mid_walls)
        max_len = max(arr.shape[0] for arr in mid_list)
        def _pad_to(arr, target_len):
            cur_len = arr.shape[0]
            if cur_len == target_len:
                return arr
            pad_rows = jnp.zeros((target_len - cur_len, 4), dtype=arr.dtype)
            return jnp.concatenate([arr, pad_rows], axis=0)
        self.mid_walls_padded = jnp.stack([_pad_to(a, max_len) for a in mid_list], axis=0)  # (4, K, 4)


    @staticmethod   # has to be static to work for renderer
    def get_room_index(room_num):
        def get_current_index(room_num):
            prev = (room_num - 1) % 4
            offset = room_num + 1
            next_idx = (prev + offset) % 4
            return next_idx
        return jax.lax.cond(
            room_num == 0,
            lambda: jnp.array(0, dtype=jnp.int32),
            lambda: jnp.array(get_current_index(room_num), dtype=jnp.int32)
        )

    # AABB collision helpers
    @partial(jax.jit, static_argnums=(0, ))
    def check_object_hits_wall_list(self, object_pos, object_size, wall_list: chex.Array) -> chex.Array:
        def check_single_wall(wall_aabb):
            wall_pos = wall_aabb[0:2]
            wall_size = wall_aabb[2:4]
            return self.rects_overlap(object_pos, object_size, wall_pos, wall_size)
        return jnp.any(jax.vmap(check_single_wall)(wall_list))

    def _define_wall_geometry(self) -> WallGeometry:
        C = self.consts
        WT = C.WALL_THICKNESS
        WO = C.WALL_OFFSET
        W = C.WIDTH
        H = C.HEIGHT
        EW = C.EXIT_WIDTH
        EH = C.EXIT_HEIGHT
        left_bound = WO[0]
        top_bound = WO[1]
        right_bound = W - WO[2]
        bottom_bound = H - WO[3]
        inner_width = right_bound - left_bound
        inner_height = bottom_bound - top_bound
        top_exit_x_start = left_bound + (inner_width - EW) / 2
        top_exit_x_end = top_exit_x_start + EW
        bottom_exit_x_start = top_exit_x_start
        bottom_exit_x_end = top_exit_x_end
        left_exit_y_start = top_bound + (inner_height - EH) / 2
        left_exit_y_end = left_exit_y_start + EH
        right_exit_y_start = left_exit_y_start
        right_exit_y_end = left_exit_y_end
        outer = jnp.array([
            [left_bound, top_bound, top_exit_x_start - left_bound, WT],
            [top_exit_x_end, top_bound, right_bound - top_exit_x_end, WT],
            [left_bound, bottom_bound - WT, bottom_exit_x_start - left_bound, WT],
            [bottom_exit_x_end, bottom_bound - WT, right_bound - bottom_exit_x_end, WT],
            [left_bound, top_bound + WT, WT, left_exit_y_start - (top_bound + WT)],
            [left_bound, left_exit_y_end, WT, (bottom_bound - WT) - left_exit_y_end],
            [right_bound - WT, top_bound + WT, WT, right_exit_y_start - (top_bound + WT)],
            [right_bound - WT, right_exit_y_end, WT, (bottom_bound - WT) - right_exit_y_end],
        ], dtype=jnp.float32)
        blockers = jnp.array([
            [top_exit_x_start, top_bound, EW, WT],
            [bottom_exit_x_start, bottom_bound - WT, EW, WT],
            [left_bound, left_exit_y_start, WT, EH],
            [right_bound - WT, right_exit_y_start, WT, EH],
        ], dtype=jnp.float32)
        mid_walls_1 = jnp.array([[36, 76, 88, 4], [36, 28, 4, 124], [120, 28, 4, 124]], dtype=jnp.float32)
        mid_walls_2 = jnp.array([[36, 28, 4, 48], [36, 104, 4, 48], [120, 28, 4, 48], [120, 104, 4, 48]], dtype=jnp.float32)
        mid_walls_3 = jnp.array([[78, 28, 4, 124], [36, 76, 46, 4], [82, 76, 42, 4]], dtype=jnp.float32)
        mid_walls_4 = jnp.array([[36, 76, 42, 4], [82, 76, 42, 4], [78, 28, 4, 48], [78, 80, 4, 72]], dtype=jnp.float32)
        return WallGeometry(outer_walls=outer, door_blockers=blockers, mid_walls=(mid_walls_1, mid_walls_2, mid_walls_3, mid_walls_4))

    @partial(jax.jit, static_argnums=(0, ))
    def _get_current_walls(self, room_counter, entry_direction) -> chex.Array:
        room_idx = self.get_room_index(room_counter)
        # Select pre-padded mid walls by index to keep shapes consistent
        mid_walls_for_room = self.mid_walls_padded[room_idx]
        base = jnp.concatenate([self.wall_geometry.outer_walls, mid_walls_for_room], axis=0)
        block_top    = (entry_direction == 1)
        block_bottom = (entry_direction == 0)
        block_lr     = (entry_direction == 2) | (entry_direction == 3)
        zero = jnp.zeros((1, 4), dtype=base.dtype)
        top_block    = jax.lax.select(block_top,    self.wall_geometry.door_blockers[0:1], zero)
        bottom_block = jax.lax.select(block_bottom, self.wall_geometry.door_blockers[1:2], zero)
        left_block   = jax.lax.select(block_lr,     self.wall_geometry.door_blockers[2:3], zero)
        right_block  = jax.lax.select(block_lr,     self.wall_geometry.door_blockers[3:4], zero)
        current_walls = jnp.concatenate([base, top_block, bottom_block, left_block, right_block], axis=0)
        return current_walls


    @partial(jax.jit, static_argnums=(0, ))
    def rects_overlap(self, pos_a, size_a, pos_b, size_b):
        left_a, top_a = pos_a
        right_a = pos_a[0] + size_a[0]
        bottom_a = pos_a[1] + size_a[1]

        left_b, top_b = pos_b
        right_b = pos_b[0] + size_b[0]
        bottom_b = pos_b[1] + size_b[1]

        overlap_x = (left_a < right_b) & (right_a > left_b)
        overlap_y = (top_a < bottom_b) & (bottom_a > top_b)
        return overlap_x & overlap_y


    @partial(jax.jit, static_argnums=(0, ))
    def is_moving_action(self, action):
        moving_actions = jnp.array([
        Action.UP, 
        Action.DOWN, 
        Action.LEFT, 
        Action.RIGHT,
        Action.UPLEFT, 
        Action.UPRIGHT, 
        Action.DOWNLEFT, 
        Action.DOWNRIGHT,
    ])
        return jnp.any(action == moving_actions)

    
    @partial(jax.jit, static_argnums=(0, ))
    def player_step(
        self, state: BerzerkState, action: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        # implement all the possible movement directions for the player, the mapping is:
        # anything with left in it, add -1 to the x position
        # anything with right in it, add 1 to the x position
        # anything with up in it, add -1 to the y position
        # anything with down in it, add 1 to the y position
        up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                    action == Action.UPRIGHT,
                    action == Action.UPLEFT,
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                ]
            )
        )
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                ]
            )
        )
        
        dx = jnp.where(right, 1, jnp.where(left, -1, 0))
        dy = jnp.where(down, 1, jnp.where(up, -1, 0))

        # movement scaled
        player_x = state.player.pos[0] + dx * self.consts.PLAYER_SPEED
        player_y = state.player.pos[1] + dy * self.consts.PLAYER_SPEED

        player_direction = jnp.select(
            [
                action == Action.UPFIRE,
                action == Action.DOWNFIRE,
                action == Action.LEFTFIRE,
                action == Action.RIGHTFIRE,
                action == Action.UP,
                action == Action.DOWN,
                action == Action.LEFT,
                action == Action.RIGHT,
                action == Action.UPRIGHT,
                action == Action.UPLEFT,
                action == Action.DOWNRIGHT,
                action == Action.DOWNLEFT,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
            ],
            [
                jnp.array([0, -1]),   # UPFIRE
                jnp.array([0, 1]),    # DOWNFIRE
                jnp.array([-1, 0]),   # LEFTFIRE
                jnp.array([1, 0]),    # RIGHTFIRE
                jnp.array([0, -1]),   # UP
                jnp.array([0, 1]),    # DOWN
                jnp.array([-1, 0]),   # LEFT
                jnp.array([1, 0]),    # RIGHT
                jnp.array([1, -1]),   # UPRIGHT
                jnp.array([-1, -1]),  # UPLEFT
                jnp.array([1, 1]),    # DOWNRIGHT
                jnp.array([-1, 1]),   # DOWNLEFT
                jnp.array([1, -1]),   # UPRIGHTFIRE
                jnp.array([-1, -1]),  # UPLEFTFIRE
                jnp.array([1, 1]),    # DOWNRIGHTFIRE
                jnp.array([-1, 1]),   # DOWNLEFTFIRE
            ],
        default=state.player.last_dir
        )

        return player_x, player_y, player_direction
    

    @partial(jax.jit, static_argnums=(0, ))
    def shoot_bullet(self, state, player_pos, player_move_dir):
        # all possible directions
        dirs = jnp.array([
            [0, -1],   # up
            [1, -1],   # upright
            [1, 0],    # right
            [1, 1],    # downright
            [0, 1],    # down
            [-1, 1],   # downleft
            [-1, 0],   # left
            [-1, -1],  # upleft
        ], dtype=jnp.int32)

        # corresponding spawn positions (relative to player middle coordinates)
        offsets = jnp.array([
            [self.consts.PLAYER_SIZE[0] // 2, 0.0],                                     # up
            [self.consts.PLAYER_SIZE[0] // 2, 4.0],                                     # upright
            [3.0, self.consts.PLAYER_SIZE[1] // 2 - 4],                                 # right
            [self.consts.PLAYER_SIZE[0] // 2 + 1.0, self.consts.PLAYER_SIZE[1] - 10.0], # downright
            [self.consts.PLAYER_SIZE[0] // 2 + 2.0, self.consts.PLAYER_SIZE[1] - 10.0], # down
            [self.consts.PLAYER_SIZE[0] // 2 - 6.0, self.consts.PLAYER_SIZE[1] - 10.0], # downleft
            [-3.0, self.consts.PLAYER_SIZE[1] // 2 - 4],                                # left
            [self.consts.PLAYER_SIZE[0] -4.0 // 2 - 6.0, 4.0],                          # upleft
        ], dtype=jnp.float32)

        conds = jnp.all(dirs == player_move_dir[None, :], axis=1)   # check for active player direction

        default_offset = jnp.array([8.0, 8.0], dtype=jnp.float32)   # fallback (this should not happen) 

        offset = jnp.select(conds, offsets, default_offset)

        spawn_pos = player_pos + offset
        
        return jax.lax.cond(
            ~state.player.bullet_active[0],
            lambda _: (
                state.player.bullet.at[0].set(spawn_pos),
                state.player.bullet_dir.at[0].set(player_move_dir),
                state.player.bullet_active.at[0].set(True),
            ),
            lambda _: (state.player.bullet, state.player.bullet_dir, state.player.bullet_active),
            operand=None
        )


    @partial(jax.jit, static_argnums=(0, ))
    def object_hits_wall(self, object_pos, object_size, room_counter, entry_direction, num_points_per_side=3):
        # get current room id (0–3 → mid_walls_1 to _4)
        room_idx = JaxBerzerk.get_room_index(room_counter)

        # get respective wall mask (True = collision)
        # Use switch since masks have different shapes
        mid_mask = jax.lax.switch(
            room_idx,
            [
                lambda: self.renderer.room_collision_masks['mid_walls_1'],
                lambda: self.renderer.room_collision_masks['mid_walls_2'],
                lambda: self.renderer.room_collision_masks['mid_walls_3'],
                lambda: self.renderer.room_collision_masks['mid_walls_4'],
            ]
        )
        outer_mask = self.renderer.room_collision_masks['level_outer_walls']

        # get respective wall mask
        left_mask = self.renderer.room_collision_masks['door_vertical_left']
        right_mask = self.renderer.room_collision_masks['door_vertical_right']
        top_mask = self.renderer.room_collision_masks['door_horizontal_up']
        bottom_mask = self.renderer.room_collision_masks['door_horizontal_down']

        # calculate which doors should be opened
        block_left   = (entry_direction == 2) | (entry_direction == 3)
        block_right  = (entry_direction == 2) | (entry_direction == 3)
        block_top    = (entry_direction == 1)
        block_bottom = (entry_direction == 0)

        # get closed door masks
        collision_mask = mid_mask | outer_mask
        collision_mask = jax.lax.cond(block_left,   lambda: collision_mask | left_mask,   lambda: collision_mask)
        collision_mask = jax.lax.cond(block_right,  lambda: collision_mask | right_mask,  lambda: collision_mask)
        collision_mask = jax.lax.cond(block_top,    lambda: collision_mask | top_mask,    lambda: collision_mask)
        collision_mask = jax.lax.cond(block_bottom, lambda: collision_mask | bottom_mask, lambda: collision_mask)

        mask_height, mask_width = collision_mask.shape

        # check collision at all hit detection points
        def point_hits(px, py):
            i = jnp.floor(py).astype(jnp.int32)
            j = jnp.floor(px).astype(jnp.int32)
            in_bounds = (i >= 0) & (i < mask_height) & (j >= 0) & (j < mask_width)
            return jax.lax.select(in_bounds, collision_mask[i, j], False)

        x0, y0 = object_pos
        w, h = object_size
        top_edge = [(x0 + dx, y0) for dx in jnp.linspace(0, w, num_points_per_side)]
        right_edge = [(x0 + w, y0 + dy) for dy in jnp.linspace(0, h, num_points_per_side)]
        bottom_edge = [(x0 + dx, y0 + h) for dx in jnp.linspace(w, 0, num_points_per_side)]
        left_edge = [(x0, y0 + dy) for dy in jnp.linspace(h, 0, num_points_per_side)]

        all_edge_points = top_edge + right_edge + bottom_edge + left_edge
        return jnp.any(jnp.array([point_hits(x, y) for x, y in all_edge_points]))


    @partial(jax.jit, static_argnums=(0, ))
    def check_exit_crossing(self, player_pos: chex.Array) -> chex.Array:
        """Return True if player touches an exit region (centered on wall)."""
        x, y = player_pos[0], player_pos[1]

        # Top exit
        top = (self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 - self.consts.EXIT_WIDTH / 2,
            self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 + self.consts.EXIT_WIDTH / 2 - self.consts.PLAYER_SIZE[0])
        top_exit = (x > top[0]) & (x < top[1]) & (y < self.consts.PLAYER_BOUNDS[1][0] - self.consts.WALL_THICKNESS)
        
        # Bottom exit
        bottom_exit = (x > top[0]) & (x < top[1]) & (y > self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_SIZE[1] + self.consts.WALL_THICKNESS)

        # Left exit
        left = (self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 - self.consts.EXIT_HEIGHT / 2,
                self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 + self.consts.EXIT_HEIGHT / 2 - self.consts.PLAYER_SIZE[1])
        left_exit = (y > left[0]) & (y < left[1]) & (x < self.consts.PLAYER_BOUNDS[0][0] - self.consts.WALL_THICKNESS)

        # Right exit
        right_exit = (y > left[0]) & (y < left[1]) & (x > self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_SIZE[0] + self.consts.WALL_THICKNESS)

        return top_exit | bottom_exit | left_exit | right_exit


    @partial(jax.jit, static_argnums=(0, ))
    def get_exit_direction(self, player_pos: chex.Array) -> jnp.ndarray:
        """Returns direction index: 0=top, 1=bottom, 2=left, 3=right, -1=none"""
        x, y = player_pos[0], player_pos[1]
        top = (self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 - self.consts.EXIT_WIDTH / 2,
            self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 + self.consts.EXIT_WIDTH / 2 - self.consts.PLAYER_SIZE[0])
        left = (self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 - self.consts.EXIT_HEIGHT / 2,
                self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 + self.consts.EXIT_HEIGHT / 2 - self.consts.PLAYER_SIZE[1])
        
        return jax.lax.select(
            (x > top[0]) & (x < top[1]) & (y < self.consts.PLAYER_BOUNDS[1][0]), jnp.int32(0),
            jax.lax.select(
                (x > top[0]) & (x < top[1]) & (y > self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_SIZE[1]), jnp.int32(1),
                jax.lax.select(
                    (y > left[0]) & (y < left[1]) & (x < self.consts.PLAYER_BOUNDS[0][0]), jnp.int32(2),
                    jax.lax.select(
                        (y > left[0]) & (y < left[1]) & (x > self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_SIZE[0]), jnp.int32(3),
                        jnp.int32(-1)
                    )
                )
            )
        )


    @partial(jax.jit, static_argnums=(0, ))
    def update_enemy_positions(self, player_pos, enemy_pos, enemy_axis, enemy_dir, rng, move_prob, room_counter):
        """
        Update enemy positions with movement probability.
        Once started moving, continues until aligned with player.
        
        Args:
            enemy_move_axis: 0 for x-axis, 1 for y-axis movement
            enemy_move_dir: 1 for positive, -1 for negative direction
        """
        enemy_rngs = jax.random.split(rng, self.consts.MAX_NUM_ENEMIES)

        def update_one_enemy(_, inputs):
            rng, pos, axis, dir_, prob = inputs
            new_pos, new_axis, new_dir, _ = self.update_enemy_position(
                player_pos, pos, axis, dir_, rng, prob, enemy_pos, room_counter
            )
            return None, (new_pos, new_axis, new_dir)

        _, (positions, axes, dirs) = jax.lax.scan(
            update_one_enemy,
            None,
            (enemy_rngs, enemy_pos, enemy_axis, enemy_dir, move_prob)
        )

        return positions, axes, dirs


    @partial(jax.jit, static_argnums=(0, ))
    def update_enemy_position(self, player_pos: chex.Array, enemy_pos: chex.Array, 
                            enemy_move_axis: chex.Array, enemy_move_dir: chex.Array,
                            rng: chex.PRNGKey, move_prob: float, all_enemy_pos: chex.Array, room_counter: chex.Array):
        rng, move_rng = jax.random.split(rng)
        
        # check if already moving (axis != -1)
        is_moving = enemy_move_axis != -1
        
        # if not moving, decide whether to start moving
        start_moving = jax.random.bernoulli(move_rng, move_prob)
        should_move = is_moving | (~is_moving & start_moving)
        
        def start_new_movement(_):
            # choose random axis (0=x, 1=y) and direction (1 or -1)
            rng1, _ = jax.random.split(move_rng)
            axis = jax.random.bernoulli(rng1, 0.5).astype(jnp.int32)
            dir_ = jnp.where(player_pos[axis] > enemy_pos[axis], 1, -1)
            return axis, dir_
        
        # if not moving but should start, initialize movement
        new_axis, new_dir = jax.lax.cond(
            ~is_moving & should_move,
            start_new_movement,
            lambda _: (enemy_move_axis, enemy_move_dir),
            operand=None
        )
        
         # calculate movement vector
        move_vec = jnp.array([
            jnp.where(new_axis == 0, new_dir * self.get_enemy_speed(room_counter), 0),
            jnp.where(new_axis == 1, new_dir * self.get_enemy_speed(room_counter), 0),
        ])
        proposed_pos = enemy_pos + move_vec

        # check if enemy would walk in other enemy (5px puffer)
        def too_close(enemy_position):
            offset = jnp.array([
                jnp.where(new_axis == 0, new_dir * 5, 0),
                jnp.where(new_axis == 1, new_dir * 5, 0),
            ])
            future_pos = enemy_pos + move_vec + offset
            return (
                self.rects_overlap(future_pos, self.consts.ENEMY_SIZE, enemy_position, self.consts.ENEMY_SIZE)
                & ~jnp.all(enemy_position == enemy_pos)  # sich selbst ausschließen
            )
        overlap = jnp.any(jax.vmap(too_close)(all_enemy_pos))

        # if movement would lead to collision with other enemy stop
        final_pos = jax.lax.select(overlap, enemy_pos, proposed_pos)

        # stop if enemy axis aligned with player axis
        aligned_x = jnp.abs(player_pos[0] - final_pos[0]) < 5
        aligned_y = jnp.abs(player_pos[1] - final_pos[1]) < 5
        aligned = jax.lax.select(new_axis == 0, aligned_x, aligned_y)

        final_axis = jnp.where(overlap | aligned, -1, new_axis)
        final_dir  = jnp.where(overlap | aligned, 0, new_dir)
        
        return final_pos, final_axis, final_dir, rng
    

    @partial(jax.jit, static_argnums=(0, ))
    def get_enemy_bullet_speed(self, level: jnp.ndarray) -> jnp.ndarray:
        base_bullet_speed = BerzerkConstants.ENEMY_BULLET_SPEED
        bullet_speed_increment = 0.065      # best-guess-estimate of true value

        # cap bullet speed after 13 levels
        capped_level = jnp.minimum(level + 1, 13)

        # increment every 2 levels (2,3 = 0; 4,5 = 1; …; 12,13 = 5)
        step = (capped_level - 1) // 2
        step = jnp.maximum(step, 0)

        value = base_bullet_speed + step * bullet_speed_increment

        return value


    @partial(jax.jit, static_argnums=(0, ))
    def get_enemy_speed(self, level: jnp.ndarray) -> jnp.ndarray:
        base_enemy_speed = BerzerkConstants.ENEMY_SPEED
        enemy_speed_increment = 0.007       # best-guess-estimate of true value

        # increment speed every 2 levels and repeat after 16 levels
        step = ((level + 1) // 2) % 8

        return base_enemy_speed + step * enemy_speed_increment


    @partial(jax.jit, static_argnums=(0, ))
    def enemy_fire_logic(self, player_pos, enemy_pos, enemy_size, enemy_shoot_prob, alive, axis, rng):
        # only shoot if not moving
        is_moving = axis != -1

        aligned_x = jnp.abs(enemy_pos[0] - player_pos[0]) < 5
        aligned_y = jnp.abs(enemy_pos[1] - player_pos[1]) < 5
        aligned = aligned_x | aligned_y

        can_shoot = (~is_moving) & aligned & alive
        should_fire = jax.random.uniform(rng) < enemy_shoot_prob

        dx = jnp.where(aligned_x, 0.0, jnp.sign(player_pos[0] - enemy_pos[0]))
        dy = jnp.where(aligned_y, 0.0, jnp.sign(player_pos[1] - enemy_pos[1]))
        direction = jnp.array([dx, dy], dtype=jnp.float32)

        dirs = jnp.array([
            [0, -1],   # up
            [1, 0],    # right
            [0, 1],    # down
            [-1, 0],   # left
        ], dtype=jnp.float32)

        # bullet spawn offsets relative to enemy position
        offsets = jnp.array([
            [enemy_size[0], enemy_size[1] // 2 - 7],    # up
            [enemy_size[0], enemy_size[1] // 2],        # right
            [0.0, enemy_size[1] // 2 + 2],              # down
            [0.0, enemy_size[1] // 2],                  # left
        ], dtype=jnp.float32)

        conds = jnp.all(dirs == direction[None, :], axis=1)

        # fallback (should not be used)
        default_offset = jnp.array([enemy_size[0] // 2,
                                    enemy_size[1] // 2], dtype=jnp.float32)

        offset = jnp.select(conds, offsets, default_offset)

        spawn_pos = enemy_pos + offset

        return (
            spawn_pos,
            direction,
            should_fire & can_shoot
        )


    @partial(jax.jit, static_argnums=(0, ))
    def spawn_enemies(self, state, rng):
        # number of enemy: self.consts.MIN_NUM_ENEMIES - self.consts.MAX_NUM_ENEMIES
        rng, sub_num, sub_spawn = jax.random.split(rng, 3)
        num_enemies = jax.random.randint(sub_num, (), self.consts.MIN_NUM_ENEMIES, self.consts.MAX_NUM_ENEMIES+1)

        # initialize empty spawn vector
        placed_init = jnp.full((self.consts.MAX_NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)

        def sample_pos(r):
            return jax.random.uniform(
                r, shape=(2,),
                minval=jnp.array([self.consts.PLAYER_BOUNDS[0][0], self.consts.PLAYER_BOUNDS[1][0]]),
                maxval=jnp.array([self.consts.PLAYER_BOUNDS[0][1] - self.consts.ENEMY_SIZE[0],
                                self.consts.PLAYER_BOUNDS[1][1] - self.consts.ENEMY_SIZE[1]])
            )

        def cond_fn(carry2):
            pos, _, attempts, placed = carry2
            in_wall = self.object_hits_wall(pos, self.consts.ENEMY_SIZE,
                                            state.room_counter, state.entry_direction)
            on_player = self.rects_overlap(state.player.pos, self.consts.PLAYER_SIZE, pos, self.consts.ENEMY_SIZE)
            overlap_enemy = jnp.any(
                jax.vmap(lambda enemy_position: self.rects_overlap(
                    pos, self.consts.ENEMY_SIZE, enemy_position, self.consts.ENEMY_SIZE))(placed))
            invalid = in_wall | on_player | overlap_enemy
            return jnp.logical_and(invalid, attempts < 2)

        def body2(carry2):
            _, rng2, attempts, placed = carry2
            rng2, sub2 = jax.random.split(rng2)
            return sample_pos(sub2), rng2, attempts + 1, placed

        def body_fun(i, carry):
            placed, rng_inner = carry
            rng_inner, sub = jax.random.split(rng_inner)
            pos0 = sample_pos(sub)
            pos, rng_after, _, _ = jax.lax.while_loop(cond_fn, body2, (pos0, sub, jnp.int32(0), placed))
            placed = placed.at[i].set(pos)
            return (placed, rng_after)

        final_carry = jax.lax.fori_loop(0, num_enemies, body_fun, (placed_init, sub_spawn))
        placed_final, _ = final_carry
        enemy_alive = jnp.arange(self.consts.MAX_NUM_ENEMIES) < num_enemies
        return state.replace(
            enemy=state.enemy.replace(pos=placed_final, 
                                      alive=enemy_alive),
            num_enemies=num_enemies)


    @partial(jax.jit, static_argnums=(0, ))
    def move_otto(self, otto_pos, player_pos, otto_speed_slow, otto_speed_fast, otto_animation_counter, walls_to_check: chex.Array, robots_alive: bool):
        # --- Calculate Direction and Base Speed ---
        direction = player_pos - otto_pos
        dx = direction[0]
        dy = direction[1]
        norm = jnp.linalg.norm(direction) + 1e-6
        norm_direction = direction / norm
        current_otto_speed = jax.lax.select(robots_alive, otto_speed_slow, otto_speed_fast)

        # --- Bounce Animation Logic ---
        otto_animation_counter = otto_animation_counter + 1
        jump_cycle_length = self.consts.OTTO_BOUNCE_CYCLE
        jump_phase_float = (otto_animation_counter % jump_cycle_length) / jump_cycle_length
        vertical_bounce_factor = jnp.sin(jump_phase_float * 2 * jnp.pi)
        vertical_offset = vertical_bounce_factor * self.consts.OTTO_BOUNCE_HEIGHT

        # --- Phased Horizontal Movement ---
        move_horizontally = (jump_phase_float >= self.consts.OTTO_HORIZ_PHASE_START) & (jump_phase_float < self.consts.OTTO_HORIZ_PHASE_END)
        horizontal_move = jax.lax.select(
            move_horizontally,
            norm_direction[0] * current_otto_speed,
            0.0,
        )

        # --- Combine Movements ---
        vertical_drift = norm_direction[1] * current_otto_speed * self.consts.OTTO_VERTICAL_DRIFT_SCALE
        proposed_move_vec = jnp.array([horizontal_move, vertical_offset + vertical_drift])
        final_pos = otto_pos + proposed_move_vec

        return final_pos, otto_animation_counter

       
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state) -> BerzerkObservation:
        # Player as (2,)
        player_pos = jnp.array([state.player.pos[0], state.player.pos[1]], dtype=jnp.float32)
        player_dir = jnp.array([state.player.last_dir[0], state.player.last_dir[1]], dtype=jnp.float32)

        # Bullet as (1,2)
        player_bullet = jnp.array([state.player.bullet[0]], dtype=jnp.float32) if state.player.bullet.ndim == 2 else jnp.array([state.player.bullet], dtype=jnp.float32)
        player_bullet_dir = jnp.array([state.player.bullet_dir[0]], dtype=jnp.float32) if state.player.bullet_dir.ndim == 2 else jnp.array([state.player.bullet_dir], dtype=jnp.float32)

        # --- Enemies ---
        enemy_pos = state.enemy.pos.astype(jnp.float32)  # shape (MAX_NUM_ENEMIES, 2)
        enemy_bullets = state.enemy.bullets.astype(jnp.float32)  # shape (MAX_NUM_ENEMIES, 2)
        enemy_bullet_dirs = state.enemy.bullet_dirs.astype(jnp.float32)  # shape (MAX_NUM_ENEMIES, 2)

        # --- Otto ---
        otto_pos = state.otto.pos.astype(jnp.float32)

        # --- Global ---
        score = state.score.astype(jnp.float32)
        lives = state.lives.astype(jnp.float32)

        return BerzerkObservation(
            player_pos=player_pos,
            player_dir=player_dir,
            player_bullet=player_bullet,
            player_bullet_dir=player_bullet_dir,
            enemy_pos=enemy_pos,
            enemy_bullets=enemy_bullets,
            enemy_bullet_dirs=enemy_bullet_dirs,
            otto_pos=otto_pos,
            score=score,
            lives=lives,
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BerzerkState, state: BerzerkState) -> jnp.ndarray:
        return state.score - previous_state.score


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BerzerkState, previous_state: BerzerkState = None) -> BerzerkInfo:
        if previous_state is None:
            enemies_killed = jnp.array(0, dtype=jnp.int32)
        else:
            prev_alive = jnp.array(previous_state.enemy_alive, dtype=jnp.int32)
            curr_alive = jnp.array(state.enemy_alive, dtype=jnp.int32)
            enemies_killed = jnp.sum(prev_alive - curr_alive)

        level_cleared = jnp.array([state.room_counter], dtype=jnp.int32)

        return BerzerkInfo(
            enemies_killed=enemies_killed,
            level_cleared=level_cleared
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BerzerkState) -> bool:
        return state.lives < 0


    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: BerzerkObservation) -> chex.Array:
        return jnp.concatenate([
            obs.player_pos.flatten().astype(jnp.float32),
            obs.player_dir.flatten().astype(jnp.float32),
            obs.player_bullet.flatten().astype(jnp.float32),
            obs.player_bullet_dir.flatten().astype(jnp.float32),
            obs.enemy_pos.flatten().astype(jnp.float32),
            obs.enemy_bullets.flatten().astype(jnp.float32),
            obs.enemy_bullet_dirs.flatten().astype(jnp.float32),
            obs.otto_pos.flatten().astype(jnp.float32),
            obs.score.flatten().astype(jnp.float32),
            obs.lives.flatten().astype(jnp.float32),
        ])


    @partial(jax.jit, static_argnums=(0,))
    def info_to_flat_array(self, info: BerzerkInfo) -> chex.Array:
        return jnp.concatenate([
            info.enemies_killed.flatten(),
            info.level_cleared.flatten()
        ])


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BerzerkState) -> jnp.ndarray:
        return self.renderer.render(state)


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[BerzerkObservation, BerzerkState]:
        # --- Player init ---
        pos = jnp.array([
            self.consts.PLAYER_BOUNDS[0][0] + 2,
            self.consts.PLAYER_BOUNDS[1][1] // 2
        ], dtype=jnp.float32)
        last_dir = jnp.array([0.0, -1.0], dtype=jnp.float32)  # default = up
        bullets = jnp.zeros((1, 2), dtype=jnp.float32)
        bullet_dirs = jnp.zeros((1, 2), dtype=jnp.float32)
        bullet_active = jnp.zeros((1,), dtype=jnp.bool_)
        animation_counter = jnp.array(0, dtype=jnp.int32)
        death_timer = jnp.array(0, dtype=jnp.int32)
        player_is_firing = jnp.array(False, dtype=jnp.bool_)

        player_state = PlayerState(
            pos=pos,
            last_dir=last_dir,
            animation_counter=animation_counter,
            is_firing=player_is_firing,
            bullet=bullets,
            bullet_dir=bullet_dirs,
            bullet_active=bullet_active,
            death_timer=death_timer,
        )

        # --- Enemy init ---
        enemy_pos = jnp.full((self.consts.MAX_NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)
        enemy_move_axis = -jnp.ones((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_move_dir = jnp.zeros((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_alive = jnp.ones((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_bullets = jnp.zeros((self.consts.MAX_NUM_ENEMIES, 2), dtype=jnp.float32)
        enemy_bullet_dirs = jnp.zeros((self.consts.MAX_NUM_ENEMIES, 2), dtype=jnp.float32)
        enemy_bullet_active = jnp.zeros((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_move_prob = jnp.full((self.consts.MAX_NUM_ENEMIES,), self.consts.MOVEMENT_PROB, dtype=jnp.float32)
        enemy_clear_bonus_given = jnp.array(False)
        enemy_death_timer = jnp.zeros((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_death_pos = jnp.full((self.consts.MAX_NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)
        enemy_animation_counter = jnp.zeros((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)

        enemy_state = EnemyState(
            pos=enemy_pos,
            move_axis=enemy_move_axis,
            move_dir=enemy_move_dir,
            alive=enemy_alive,
            bullets=enemy_bullets,
            bullet_dirs=enemy_bullet_dirs,
            bullet_active=enemy_bullet_active,
            move_prob=enemy_move_prob,
            clear_bonus_given=enemy_clear_bonus_given,
            death_timer=enemy_death_timer,
            death_pos=enemy_death_pos,
            animation_counter=enemy_animation_counter,
        )

        # --- Otto init ---
        otto_pos = jnp.array([-100.0, -100.0], dtype=jnp.float32)
        otto_active = jnp.array(False)
        otto_timer = self.consts.EVIL_OTTO_DELAY
        otto_anim_counter = jnp.array(0, dtype=jnp.int32)
        otto_state = OttoState(
            pos=otto_pos,
            active=otto_active,
            timer=otto_timer,
            anim_counter=otto_anim_counter,
        )

        # --- Global game state ---
        lives = jnp.array(2, dtype=jnp.float32)
        score = jnp.array(0, dtype=jnp.float32)
        room_counter = jnp.array(0, dtype=jnp.int32)
        extra_life_counter = jnp.array(0, dtype=jnp.int32)
        game_over_timer = jnp.array(0, dtype=jnp.int32)
        num_enemies = jnp.array(self.consts.MAX_NUM_ENEMIES, dtype=jnp.int32)
        entry_direction = jnp.array(3, dtype=jnp.int32)
        room_transition_timer = jnp.array(0, dtype=jnp.int32)

        state = BerzerkState(
            player=player_state,
            enemy=enemy_state,
            otto=otto_state,
            rng=rng,
            score=score,
            lives=lives,
            room_counter=room_counter,
            extra_life_counter=extra_life_counter,
            game_over_timer=game_over_timer,
            num_enemies=num_enemies,
            entry_direction=entry_direction,
            room_transition_timer=room_transition_timer,
        )

        # Spawn enemies in valid positions
        state = self.spawn_enemies(state, jax.random.split(rng)[0])
        return self._get_observation(state), state

    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BerzerkState, action: chex.Array) -> Tuple[BerzerkObservation, BerzerkState, float, bool, BerzerkInfo]:
        # Translate compact agent action index to ALE console action
        action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        # Handle game over animation phase
        game_over_active = state.game_over_timer > 0
        game_over_timer = jnp.maximum(state.game_over_timer - 1, 0)

        def handle_game_over():
            new_state = state.replace(game_over_timer=game_over_timer)
            return (
                self._get_observation(new_state),
                new_state,
                0.0,
                game_over_timer == 0,
                self._get_info(new_state),
            )


        # Handle room transition animation phase
        room_transition_active = state.room_transition_timer > 0
        transition_timer = jnp.maximum(state.room_transition_timer - 1, 0)

        def handle_room_transition():
            new_state = state.replace(room_transition_timer=transition_timer)

            def finished_transition():
                player_spawn_pos = jax.lax.switch(
                    new_state.entry_direction,
                    [
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] // 2,
                            self.consts.PLAYER_BOUNDS[1][1] - 25
                            ], dtype=jnp.float32),  # top -> spawn at bottom
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] // 2, 
                            self.consts.PLAYER_BOUNDS[1][0] + 5], dtype=jnp.float32),  # bottom -> spawn at top
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] - 12, 
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                            ], dtype=jnp.float32),  # left -> spawn at right
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][0] + 2,
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                        ], dtype=jnp.float32),  # right -> spawn at left
                    ],
                    jnp.array([
                            self.consts.PLAYER_BOUNDS[0][0] + 2,
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                        ], dtype=jnp.float32)  # fallback
                )

                # load new level
                new_rng = jax.random.split(state.rng)[1]
                obs, base_state = self.reset(new_rng)
                base_state = base_state.replace(
                    player=base_state.player.replace(pos=player_spawn_pos),
                    room_counter=state.room_counter + 1,
                    lives=state.lives,
                    score=state.score,
                    entry_direction=state.entry_direction,
                    extra_life_counter=state.extra_life_counter
                )

                next_state = self.spawn_enemies(base_state, jax.random.split(new_rng)[1])

                return (
                    self._get_observation(next_state),
                    next_state,
                    0.0,
                    False,
                    self._get_info(next_state),
                )

            def in_transition():
                return (
                    self._get_observation(new_state),
                    new_state,
                    0.0,
                    False,
                    self._get_info(new_state),
                )

            return jax.lax.cond(
                transition_timer == 0,
                finished_transition,
                in_transition
            )


        # Handle normal gameplay
        def handle_normal():
            #######################################################
            # 1. Update Player
            #######################################################

            player_alive = state.player.death_timer == 0
            # forbid movement during death animation
            player_x, player_y, move_dir = jax.lax.cond(
                player_alive,
                lambda _: self.player_step(state, action),
                lambda _: (state.player.pos[0], state.player.pos[1], state.player.last_dir),
                operand=None
            )

            new_player_pos = jnp.array([player_x, player_y])

            moving = self.is_moving_action(action)

            animation_counter = jnp.where(
                player_alive & moving,
                state.player.animation_counter + 1,
                0
            )

            # handle player bullet
            is_shooting = jnp.any(jnp.array([
                action == Action.FIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
                action == Action.RIGHTFIRE,
                action == Action.LEFTFIRE,
                action == Action.UPFIRE,
            ]))
            is_shooting = player_alive & is_shooting

            player_is_firing = is_shooting.astype(jnp.bool_)

            player_bullet, player_bullet_dir, player_bullet_active = jax.lax.cond(
                is_shooting,
                lambda _: self.shoot_bullet(state, new_player_pos, move_dir),
                lambda _: (state.player.bullet, state.player.bullet_dir, state.player.bullet_active),
                operand=None
            )

            # choose bullet size (depending on direction)
            player_bullet_size = jax.vmap(
                lambda d: jax.lax.select(
                    d[0] == 0,
                    jnp.array(self.consts.BULLET_SIZE_VERTICAL, dtype=jnp.float32),
                    jnp.array(self.consts.BULLET_SIZE_HORIZONTAL, dtype=jnp.float32)
                )
            )(player_bullet_dir)
            
            player_bullet += player_bullet_dir * self.consts.BULLET_SPEED * player_bullet_active[:, None]
            # only 1 player bullet
            player_bullet_active = player_bullet_active & (~self.object_hits_wall(player_bullet[0], player_bullet_size[0], state.room_counter, state.entry_direction)) & (
                (player_bullet[:, 0] >= self.consts.PLAYER_BOUNDS[0][0]) &
                (player_bullet[:, 0] + player_bullet_size[:, 0] <= self.consts.PLAYER_BOUNDS[0][1]) &
                (player_bullet[:, 1] >= self.consts.PLAYER_BOUNDS[1][0]) &
                (player_bullet[:, 1] + player_bullet_size[:, 1] <= self.consts.PLAYER_BOUNDS[1][1])
            )


            #######################################################
            # 2. Update enemies
            #######################################################

            rng, enemy_rng = jax.random.split(state.rng)

            updated_enemy_pos, updated_enemy_axis, updated_enemy_dir = self.update_enemy_positions(
                new_player_pos, state.enemy.pos, state.enemy.move_axis, state.enemy.move_dir,
                enemy_rng, state.enemy.move_prob, state.room_counter
            )

            enemy_animation_counter = state.enemy.animation_counter + 1

            # only move living enemies
            updated_enemy_pos = jnp.where(state.enemy.alive[:, None], updated_enemy_pos, state.enemy.pos)
            updated_enemy_axis = jnp.where(state.enemy.alive, updated_enemy_axis, state.enemy.move_axis)
            updated_enemy_dir = jnp.where(state.enemy.alive, updated_enemy_dir, state.enemy.move_dir)

            # handle enemy bullets
            enemy_rngs = jax.random.split(rng, self.consts.MAX_NUM_ENEMIES)
            enemy_bullets_new, dirs_new, active_new = jax.vmap(self.enemy_fire_logic,
                                                               in_axes=(None, 0, None, None, 0, 0, 0))(
                new_player_pos,
                updated_enemy_pos,
                self.consts.ENEMY_SIZE,
                self.consts.ENEMY_SHOOT_PROB,
                state.enemy.alive,
                updated_enemy_axis,
                enemy_rngs
            )

            # only fire if no active bullet of this enemy already
            can_shoot = (state.room_counter > 0)
            can_shoot_mask = jnp.broadcast_to(can_shoot, active_new.shape)  # cast to length of NUM_ENEMIES

            enemy_bullets = jnp.where(
                ~state.enemy.bullet_active[:, None] & active_new[:, None] & can_shoot_mask[:, None],
                enemy_bullets_new,
                state.enemy.bullets
            )

            enemy_bullet_dirs = jnp.where(
                ~state.enemy.bullet_active[:, None] & active_new[:, None] & can_shoot_mask[:, None],
                dirs_new,
                state.enemy.bullet_dirs
            )

            enemy_bullet_active = state.enemy.bullet_active | (active_new & can_shoot_mask)

            enemy_bullet_sizes = jax.vmap(
                lambda d: jax.lax.select(
                    d[0] == 0,
                    jnp.array(self.consts.BULLET_SIZE_VERTICAL, dtype=jnp.float32),
                    jnp.array(self.consts.BULLET_SIZE_HORIZONTAL, dtype=jnp.float32)
                )
            )(enemy_bullet_dirs)

            enemy_bullets = enemy_bullets + enemy_bullet_dirs * self.get_enemy_bullet_speed(state.room_counter) * enemy_bullet_active[:, None]

            # deactivate bullets that are out of bounds
            enemy_bullet_active = enemy_bullet_active & (
                (enemy_bullets[:, 0] >= self.consts.PLAYER_BOUNDS[0][0]) &
                (enemy_bullets[:, 0] + enemy_bullet_sizes[:, 0] <= self.consts.PLAYER_BOUNDS[0][1]) &
                (enemy_bullets[:, 1] >= self.consts.PLAYER_BOUNDS[1][0]) &
                (enemy_bullets[:, 1] + enemy_bullet_sizes[:, 1] <= self.consts.PLAYER_BOUNDS[1][1])
            )


            #######################################################
            # 3. Collision Checks
            #######################################################

            player_hit_exit = self.check_exit_crossing(new_player_pos)  # player reached exit

            player_hit_wall = self.object_hits_wall((player_x, player_y), self.consts.PLAYER_SIZE, state.room_counter, state.entry_direction) & ~player_hit_exit

            player_hits_enemy = jax.vmap(
                lambda enemy_pos: self.rects_overlap(new_player_pos, self.consts.PLAYER_SIZE, enemy_pos, self.consts.ENEMY_SIZE)
            )(updated_enemy_pos)
            player_hit_by_enemy = jnp.any(player_hits_enemy)

            enemy_bullet_hits_player = jax.vmap(
                lambda b_pos, b_size, b_active: self.rects_overlap(b_pos, b_size, new_player_pos, self.consts.PLAYER_SIZE) & b_active
            )(enemy_bullets, enemy_bullet_sizes, enemy_bullet_active)
            player_hit_by_enemy_bullet = jnp.any(enemy_bullet_hits_player)

            enemy_hits_wall = jax.vmap(
                lambda enemy_pos: self.object_hits_wall(enemy_pos, self.consts.ENEMY_SIZE, state.room_counter, state.entry_direction)
            )(updated_enemy_pos)

            all_enemy_bullet_hits = jax.vmap(
                lambda enemy_pos: jax.vmap(
                    lambda bullet, size: self.rects_overlap(bullet, size, enemy_pos, self.consts.ENEMY_SIZE)
                )(player_bullet, player_bullet_size)
            )(updated_enemy_pos)
            enemy_hit_by_player_bullet = jnp.any(all_enemy_bullet_hits, axis=1)

            enemy_friendly_fire_hits = jax.vmap(
                lambda bullet_pos, bullet_size, shooter_pos, active: jax.vmap(
                    lambda target_pos: self.rects_overlap(
                        bullet_pos, bullet_size, target_pos, jnp.array(self.consts.ENEMY_SIZE, dtype=jnp.float32))
                          & active 
                          & ~jnp.all(target_pos == shooter_pos)
                )(updated_enemy_pos)
            )(enemy_bullets, enemy_bullet_sizes, updated_enemy_pos, enemy_bullet_active)

            enemy_touch_hits = jax.vmap(
                lambda pos_a, alive_a: jax.vmap(
                    lambda pos_b, alive_b: (
                        self.rects_overlap(pos_a, self.consts.ENEMY_SIZE, pos_b, self.consts.ENEMY_SIZE) &
                        ~jnp.all(pos_a == pos_b) &   # not the same enemy
                        alive_a & alive_b            # both alive
                    )
                )(updated_enemy_pos, state.enemy.alive)
            )(updated_enemy_pos, state.enemy.alive)
            enemy_hit_enemy = jnp.any(enemy_touch_hits, axis=1)

            # remove bullets on hit
            bullet_vs_bullet_hits = jax.vmap(
                lambda b_pos, b_size, b_active: jax.vmap(
                    lambda e_pos, e_size, e_active: 
                        self.rects_overlap(b_pos, b_size, e_pos, e_size) & b_active & e_active
                )(enemy_bullets, enemy_bullet_sizes, enemy_bullet_active)
            )(player_bullet, player_bullet_size, player_bullet_active)
            player_bullet_hit_enemy_bullet = jnp.any(bullet_vs_bullet_hits, axis=1)     # (player_bullets,)
            enemy_bullet_hit_by_player_bullet = jnp.any(bullet_vs_bullet_hits, axis=0)  # (enemy_bullets,)

            enemy_bullet_hit_enemy = jnp.any(enemy_friendly_fire_hits, axis=1)
            player_bullet_active = player_bullet_active & ~player_bullet_hit_enemy_bullet
            enemy_bullet_active = enemy_bullet_active & ~enemy_bullet_hit_enemy & ~enemy_bullet_hit_by_player_bullet

            enemy_bullet_hits_wall = jax.vmap(
                lambda pos, size: self.object_hits_wall(pos, size, state.room_counter, state.entry_direction)
            )(enemy_bullets, enemy_bullet_sizes)

            enemy_bullet_active = enemy_bullet_active & (~enemy_bullet_hits_wall)

            bullet_hit = jnp.any(all_enemy_bullet_hits, axis=0)
            player_bullet_active = player_bullet_active & ~bullet_hit


            #######################################################
            # 4. Handle Evil Otto (if active)
            #######################################################

            otto_pos = jnp.where(player_hit_exit, jnp.array([-100.0, -100.0], dtype=jnp.float32), state.otto.pos)
            otto_hits_player = self.rects_overlap(
                otto_pos, jnp.array(self.consts.EVIL_OTTO_SIZE, dtype=jnp.float32),
                new_player_pos, self.consts.PLAYER_SIZE
            )

            # spawn otto after timer has reached 0
            
            # Only count down the timer if Otto is enabled
            new_otto_timer = jax.lax.cond(
                self.consts.ENABLE_EVIL_OTTO,
                lambda: jnp.maximum(state.otto.timer - 1, 0), # Countdown if enabled
                lambda: state.otto.timer                          # Hold timer if disabled
            )
            
            # Check if he *should* activate (timer is 0, not already active, AND enabled)
            otto_should_activate = self.consts.ENABLE_EVIL_OTTO & jnp.logical_and(jnp.logical_not(new_otto_timer), jnp.logical_not(state.otto.active))

            # Define the spawn points for clarity
            spawn_top = lambda: jnp.array([
                self.consts.PLAYER_BOUNDS[0][1] // 2, 
                self.consts.PLAYER_BOUNDS[1][0] + 5], dtype=jnp.float32)
            
            spawn_bottom = lambda: jnp.array([
                self.consts.PLAYER_BOUNDS[0][1] // 2, 
                self.consts.PLAYER_BOUNDS[1][1] - 25], dtype=jnp.float32)
            spawn_left = lambda: jnp.array([
                self.consts.PLAYER_BOUNDS[0][0] + 2,
                self.consts.PLAYER_BOUNDS[1][1] // 2], dtype=jnp.float32)
            spawn_right = lambda: jnp.array([
                self.consts.PLAYER_BOUNDS[0][1] - 12, 
                self.consts.PLAYER_BOUNDS[1][1] // 2], dtype=jnp.float32)

            # Spawn Otto at the same position the player spawned into the room,
            # which is opposite of entry_direction (player entered from entry_direction)
            otto_spawn_pos = jax.lax.switch(
                state.entry_direction,
                [
                    spawn_bottom, # 0: Entered TOP -> player spawned BOTTOM
                    spawn_top,    # 1: Entered BOTTOM -> player spawned TOP
                    spawn_right,  # 2: Entered LEFT -> player spawned RIGHT
                    spawn_left,   # 3: Entered RIGHT -> player spawned LEFT
                ]
            )

            spawn_pos = jnp.where(otto_should_activate, otto_spawn_pos, state.otto.pos)

            # keep otto active after spawn (if he was already active, or just activated)
            otto_active = state.otto.active | otto_should_activate

            # move to player once active

            # Prepare walls for Otto collision checks
            walls_to_check = self._get_current_walls(state.room_counter, state.entry_direction)

            # Check if any robots are alive
            robots_alive = jnp.any(state.enemy.alive)

            # Call new bounce-phased movement; returns new pos and updated counter
            otto_pos, otto_anim_counter_next = jax.lax.cond(
                otto_active,
                lambda: self.move_otto(
                    spawn_pos,
                    new_player_pos,
                    self.consts.EVIL_OTTO_SPEED_SLOW,
                    self.consts.EVIL_OTTO_SPEED_FAST,
                    state.otto.anim_counter,
                    walls_to_check,
                    robots_alive,
                ),
                lambda: (spawn_pos, state.otto.anim_counter),
            )
            otto_anim_counter = jnp.where(otto_active, otto_anim_counter_next, 0)

            otto_hit_by_bullet = jnp.any(
                jax.vmap(lambda b_pos, b_size, b_active:
                    self.rects_overlap(b_pos, b_size, otto_pos, self.consts.EVIL_OTTO_SIZE) & b_active
                )(player_bullet, player_bullet_size, player_bullet_active)
            )

            # Rebound logic for mortal Otto
            otto_rebound_event = self.consts.MORTAL_EVIL_OTTO & otto_hit_by_bullet
            otto_respawn_delay = jnp.array(self.consts.EVIL_OTTO_RESPAWN_DELAY, dtype=jnp.int32)

            new_otto_timer = jax.lax.cond(
                otto_rebound_event,
                lambda: otto_respawn_delay,
                lambda: new_otto_timer
            )

            otto_active = jax.lax.cond(
                otto_rebound_event,
                lambda: jnp.array(False),
                lambda: otto_active
            )

            otto_pos = jax.lax.cond(
                otto_rebound_event,
                lambda: jnp.array([-100.0, -100.0], dtype=jnp.float32),
                lambda: otto_pos
            )

            # Destroy player's bullet if Otto enabled and was hit
            player_bullet_active = jnp.where(
                self.consts.ENABLE_EVIL_OTTO & otto_hit_by_bullet,
                player_bullet_active & False,
                player_bullet_active
            )


            #######################################################
            # 5. Handle Death, Score, Extra Lives
            #######################################################

            # player death
            hit_something = player_hit_by_enemy | player_hit_wall | player_hit_by_enemy_bullet | otto_hits_player
            death_timer = jnp.where(hit_something & (state.player.death_timer == 0), self.consts.DEATH_ANIMATION_FRAMES + 2, state.player.death_timer)
            death_timer = jnp.maximum(death_timer - 1, 0)

            # enemy death
            enemy_kill_events = (
                enemy_hit_by_player_bullet |
                enemy_hits_wall |
                enemy_hit_enemy |
                enemy_bullet_hit_enemy |
                player_hits_enemy
            )

            enemy_dies = state.enemy.alive & enemy_kill_events

            enemy_alive = state.enemy.alive & ~enemy_dies

            updated_enemy_axis = jnp.where(enemy_alive, updated_enemy_axis, 0)
            new_enemy_death_timer = jnp.where(enemy_dies, self.consts.ENEMY_DEATH_ANIMATION_FRAMES + 2, state.enemy.death_timer)
            new_enemy_death_pos = jnp.where(enemy_dies[:, None], updated_enemy_pos, state.enemy.death_pos)
            
            enemy_death_timer_next = jnp.maximum(new_enemy_death_timer - 1, 0)

            invisible = jnp.array([-100.0, -100.0])     # teleport dead enemies out of view
            updated_enemy_pos = jnp.where(enemy_alive[:, None], updated_enemy_pos, invisible)

            # calculate score: 50 points per dead enemy
            score_after = state.score + jnp.sum(enemy_dies) * 50

            # bonus score for killing all enemies in level
            give_bonus = (~jnp.any(enemy_alive)) & (~state.enemy.clear_bonus_given)
            bonus_score = jnp.where(give_bonus, state.num_enemies * 10, 0)

            score_after += bonus_score
            enemy_clear_bonus_given = state.enemy.clear_bonus_given | give_bonus

            # Handle live logic
            lives_lost_this_frame = ((death_timer == 0) & hit_something).astype(jnp.int32)
            lives_after_death = state.lives - lives_lost_this_frame
            # Extra Life Check using integer division milestones
            previous_milestone = jnp.floor_divide(state.score.astype(jnp.int32), self.consts.EXTRA_LIFE_AT)
            current_milestone = jnp.floor_divide(score_after.astype(jnp.int32), self.consts.EXTRA_LIFE_AT)
            earned_extra_life = (current_milestone > previous_milestone).astype(jnp.int32)
            lives_after = lives_after_death + earned_extra_life
            # Keep counter as total milestones reached
            extra_life_counter_after = current_milestone.astype(jnp.int32)
            # Reset score if game is over (lives hit -1)
            game_should_be_over = (lives_after < 0)
            # Ensure dtype consistency when resetting score
            score_after = jnp.where(game_should_be_over, jnp.asarray(0, score_after.dtype), score_after)

            # Trigger Room Transition oder Game Over automatisch
            transition_timer = jax.lax.cond(
                death_timer == 1,
                lambda: jax.lax.cond(
                    lives_after == -1,
                    lambda: jnp.array(self.consts.GAME_OVER_FRAMES, dtype=jnp.int32),
                    lambda: jnp.array(self.consts.TRANSITION_ANIMATION_FRAMES, dtype=jnp.int32)
                ),
                lambda: state.room_transition_timer
            )
            game_over_timer = jax.lax.cond(
                (death_timer == 1) & (lives_after == -1),
                lambda: self.consts.GAME_OVER_FRAMES,
                lambda: state.game_over_timer
            )



            #######################################################
            # 5. Update State
            #######################################################

            transition_timer = jnp.where(player_hit_exit, self.consts.TRANSITION_ANIMATION_FRAMES, transition_timer)
            entry_direction = jnp.where(player_hit_exit, self.get_exit_direction(new_player_pos), state.entry_direction)

            new_state = BerzerkState(
                player=PlayerState(
                    pos=new_player_pos,
                    last_dir=move_dir,
                    animation_counter=animation_counter,
                    is_firing=player_is_firing,
                    bullet=player_bullet,
                    bullet_dir=player_bullet_dir,
                    bullet_active=player_bullet_active,
                    death_timer=death_timer,
                ),
                enemy=EnemyState(
                    pos=updated_enemy_pos,
                    move_axis=updated_enemy_axis,
                    move_dir=updated_enemy_dir,
                    alive=enemy_alive,
                    bullets=enemy_bullets,
                    bullet_dirs=enemy_bullet_dirs,
                    bullet_active=enemy_bullet_active,
                    move_prob=state.enemy.move_prob,
                    clear_bonus_given=enemy_clear_bonus_given,
                    death_timer=enemy_death_timer_next,
                    death_pos=new_enemy_death_pos,
                    animation_counter=enemy_animation_counter,
                ),
                otto=OttoState(
                    pos=otto_pos,
                    active=otto_active,
                    timer=new_otto_timer,
                    anim_counter=otto_anim_counter,
                ),
                rng=rng,
                score=score_after,
                lives=lives_after,
                room_counter=state.room_counter,
                extra_life_counter=extra_life_counter_after,
                game_over_timer=game_over_timer,
                num_enemies=state.num_enemies,
                entry_direction=entry_direction,
                room_transition_timer=transition_timer,
            )


            #######################################################
            # 5. Observation + Info + Reward/Done
            #######################################################

            observation = self._get_observation(new_state)
            info = self._get_info(new_state)
            reward = 0.0
            done = jnp.equal(state.lives, -1) 

            return observation, new_state, reward, done, info
        

        # call appropriate step handler for current frame
        return jax.lax.cond(
            game_over_active,
            lambda _: handle_game_over(),
            lambda _: jax.lax.cond(
                room_transition_active,
                lambda _: handle_room_transition(),
                lambda _: handle_normal(),
                operand=None
            ),
            operand=None
        )


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))


    def observation_space(self) -> spaces.Dict:
        """Returns the simplified observation space for the agent."""
        return spaces.Dict({
            # Player
            "player_pos": spaces.Box(0, 255, (2,), jnp.float32),
            "player_dir": spaces.Box(-1, 1, (2,), jnp.float32),
            "player_bullet": spaces.Box(0, 255, (1,2), jnp.float32),
            "player_bullet_dir": spaces.Box(-1, 1, (1,2), jnp.float32),

            # Enemies
            "enemy_pos": spaces.Box(-255, 255, (self.consts.MAX_NUM_ENEMIES, 2), jnp.float32),
            "enemy_bullets": spaces.Box(-255, 255, (self.consts.MAX_NUM_ENEMIES, 2), jnp.float32),
            "enemy_bullet_dirs": spaces.Box(-1, 1, (self.consts.MAX_NUM_ENEMIES, 2), jnp.float32),

            # Otto
            "otto_pos": spaces.Box(-255, 255, (2,), jnp.float32),

            # Global
            "score": spaces.Box(0, 999999, (), jnp.float32),
            "lives": spaces.Box(0, 99, (), jnp.float32),
        })


    def image_space(self) -> spaces.Box:
        """
        Returns the pixel observation space of the environment.
        For now, we assume the game screen is 160x210 RGB.
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )


class BerzerkRenderer(JAXGameRenderer):
    # Type hints
    sprites: Dict[str, Any]
    pivots: Dict[str, Any]
    sprite_indices: Dict[str, int]
    group_offsets: Dict[str, chex.Array]

    def __init__(self, consts: BerzerkConstants = None):
        """
        Initializes the renderer by loading sprites using the
        new palette-based pipeline.
        """
        super().__init__()

        self.consts = consts or BerzerkConstants()

        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/berzerk"

        # 1. Configure the rendering utility
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 3. Make one call to load and process all assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

        # 4. Store key color IDs needed for rendering
        self.BLACK_ID = self.COLOR_TO_ID.get((0, 0, 0), 0)
        self.WALL_COLOR_TUPLE = (84, 92, 214) # Original wall color
        self.WALL_ID = self.COLOR_TO_ID.get(self.WALL_COLOR_TUPLE, 0)
        
        # Store colors needed for procedural recoloring
        self.ENEMY_ORIGINAL_COLOR_TUPLE = (210, 210, 64)
        self.ENEMY_BULLET_ORIGINAL_COLOR_TUPLE = (240, 170, 103)
        self.ENEMY_ORIGINAL_ID = self.COLOR_TO_ID.get(self.ENEMY_ORIGINAL_COLOR_TUPLE, 0)
        self.ENEMY_BULLET_ORIGINAL_ID = self.COLOR_TO_ID.get(self.ENEMY_BULLET_ORIGINAL_COLOR_TUPLE, 0)

        # Store the new color IDs in a JAX array for easy lookup
        color_cycle_tuples = [
            (210, 210, 91),    # yellow
            (186, 112, 69),    # orange
            (214, 214, 214),   # white
            (109, 210, 111),   # green
            (239, 127, 128),   # red
            (102, 158, 193),   # blue
            (227, 205, 115),   # yellow2
            (185, 96, 175),    # pink
        ]
        self.color_names_ids = jnp.array(
            [self.COLOR_TO_ID[c] for c in color_cycle_tuples]
        )

        # 5. Create helper mappings to bridge old logic with new SHAPE_MASKS
        self._create_helper_mappings()

        # 6. Generate collision masks (uses manual loading)
        self.room_collision_masks = self._generate_room_collision_masks()

        # Backward compatibility: map SHAPE_MASKS to old sprites dict
        self.sprites = self.SHAPE_MASKS
        self.pivots = {}  # Keep for compatibility but not used in new system

    def _create_helper_mappings(self):
        """
        Creates dictionaries to map old string keys to new group indices
        and to store group-level flip offsets.
        """
        # Map sprite name -> index in group
        self.sprite_indices = {
            # Player
            'player_idle': 0, 'player_move_1': 1, 'player_move_2': 2, 'player_death': 3,
            'player_shoot_up': 4, 'player_shoot_right': 5, 'player_shoot_down': 6,
            'player_shoot_left': 7, 'player_shoot_up_left': 8, 'player_shoot_down_left': 9,

            # Enemy
            'enemy_idle_1': 0, 'enemy_idle_2': 1, 'enemy_idle_3': 2, 'enemy_idle_4': 3,
            'enemy_idle_5': 4, 'enemy_idle_6': 5, 'enemy_idle_7': 6, 'enemy_idle_8': 7,
            'enemy_move_horizontal_1': 8, 'enemy_move_horizontal_2': 9,
            'enemy_move_vertical_1': 10, 'enemy_move_vertical_2': 11, 'enemy_move_vertical_3': 12,
            'enemy_death_1': 13, 'enemy_death_2': 14, 'enemy_death_3': 15,

            # Otto
            'evil_otto': 0, 'evil_otto_2': 1,

            # Walls
            'mid_walls_1': 0, 'mid_walls_2': 1, 'mid_walls_3': 2, 'mid_walls_4': 3,
            'level_outer_walls': 4, 'door_vertical_left': 5, 'door_vertical_right': 6,
            'door_horizontal_up': 7, 'door_horizontal_down': 8,
        }

        # Map group name -> flip offset
        self.group_offsets = {
            'player': self.FLIP_OFFSETS['player_group'],
            'enemy': self.FLIP_OFFSETS['enemy_group'],
            'otto': self.FLIP_OFFSETS['otto_group'],
            'wall': self.FLIP_OFFSETS['wall_group'],
            'bullet_horizontal': self.FLIP_OFFSETS['bullet_horizontal'],
            'bullet_vertical': self.FLIP_OFFSETS['bullet_vertical'],
            'life': self.FLIP_OFFSETS['life'],
            'start_title': self.FLIP_OFFSETS['start_title'],
            'digits': self.FLIP_OFFSETS['digits'],
        }

    def _generate_room_collision_masks(self) -> Dict[str, chex.Array]:
        """
        Manually loads RGBA sprites to generate boolean collision masks
        for game logic. This is separate from the rendering pipeline.
        """
        
        # Helper to load/convert frames manually, just for collision mask generation
        def _load_rgba_frame_for_collision(name: str) -> chex.Array:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = jnp.load(path) # Use simple jnp.load
            if isinstance(frame, jnp.ndarray) and frame.ndim == 2:
                frame = jnp.stack([frame]*3, axis=-1)  # grayscale → RGB
            
            if frame.shape[-1] == 3:
                # Pad RGB to RGBA, assuming alpha=255 for walls
                alpha = jnp.full(frame.shape[:2] + (1,), 255, dtype=jnp.uint8)
                frame = jnp.concatenate([frame, alpha], axis=-1)
            return frame.astype(jnp.uint8)

        wall_keys = [
            'mid_walls_1', 'mid_walls_2', 'mid_walls_3', 'mid_walls_4', 
            'level_outer_walls', 'door_vertical_left', 'door_horizontal_up', 
            'door_vertical_right', 'door_horizontal_down'
        ]

        # The original wall color in the .npy files
        wall_color_rgba = jnp.array([84, 92, 214, 255], dtype=jnp.uint8) 
        
        masks = {}
        target_height = self.consts.HEIGHT
        target_width = self.consts.WIDTH

        for name in wall_keys:
            sprite_rgba = _load_rgba_frame_for_collision(name)
            # Handle sprites that were already padded/expanded in the old logic
            sprite_frame = sprite_rgba[0] if sprite_rgba.ndim == 5 else sprite_rgba
            sprite_frame = sprite_frame[0] if sprite_frame.ndim == 4 and sprite_frame.shape[0] == 1 else sprite_frame
            
            # Create collision mask from sprite
            mask = jnp.all(sprite_frame == wall_color_rgba, axis=-1)  # shape (H, W)
            
            # Pad mask to game dimensions (top-left aligned)
            h, w = mask.shape
            pad_h = target_height - h
            pad_w = target_width - w
            
            # Create full-size mask with False (no collision) and place the actual mask at (0, 0)
            full_mask = jnp.zeros((target_height, target_width), dtype=jnp.bool_)
            full_mask = full_mask.at[:h, :w].set(mask)
            
            masks[name] = full_mask
        
        return masks

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BerzerkState) -> chex.Array:
        death_anim = state.player.death_timer > 0
        room_transition_anim = state.room_transition_timer > 0
        game_over_anim = state.game_over_timer > 0
        # --- 1. Create Base Raster & Dynamic Palette ---
        # Start with the background (all BLACK_ID)
        raster = self.jr.create_object_raster(self.BACKGROUND)
        # Create a dynamic palette for this frame. We'll recolor the enemy
        # and bullet entries based on the room.
        color_idx = self._get_enemy_color_index(state.room_counter)
        new_color_id = self.color_names_ids[color_idx]
        frame_palette = self.PALETTE
        
        # Conditionally update the palette:
        # If color_idx > 0, set the ENEMY_ORIGINAL_ID slot to the new color.
        frame_palette = jax.lax.cond(
            color_idx > 0,
            lambda p: p.at[self.ENEMY_ORIGINAL_ID].set(p[new_color_id]),
            lambda p: p,
            frame_palette
        )
        # Bullets match the enemy color
        frame_palette = jax.lax.cond(
            color_idx > 0,
            lambda p: p.at[self.ENEMY_BULLET_ORIGINAL_ID].set(p[new_color_id]),
            lambda p: p,
            frame_palette
        )

        # --- 2. Draw Static Walls ---
        room_idx = JaxBerzerk.get_room_index(state.room_counter)
        wall_group_offset = self.group_offsets['wall']

        # Select and draw the middle wall sprite for the room
        mid_wall_idx = jnp.array([
            self.sprite_indices['mid_walls_1'],
            self.sprite_indices['mid_walls_2'],
            self.sprite_indices['mid_walls_3'],
            self.sprite_indices['mid_walls_4'],
        ])[room_idx]
        mid_wall_mask = self.SHAPE_MASKS['wall_group'][mid_wall_idx]
        raster = self.jr.render_at(raster, 0, 0, mid_wall_mask, flip_offset=wall_group_offset)

        # Draw the outer walls
        outer_walls_mask = self.SHAPE_MASKS['wall_group'][self.sprite_indices['level_outer_walls']]
        raster = self.jr.render_at(raster, 0, 0, outer_walls_mask, flip_offset=wall_group_offset)
        
        # --- 3. Draw Entry/Exit Blocks ---
        def draw_entry_block(r):
            # Get masks
            vl_mask = self.SHAPE_MASKS['wall_group'][self.sprite_indices['door_vertical_left']]
            hu_mask = self.SHAPE_MASKS['wall_group'][self.sprite_indices['door_horizontal_up']]
            vr_mask = self.SHAPE_MASKS['wall_group'][self.sprite_indices['door_vertical_right']]
            hd_mask = self.SHAPE_MASKS['wall_group'][self.sprite_indices['door_horizontal_down']]
            
            # Define render functions
            block_top = lambda r_in: self.jr.render_at(r_in, 0, 0, hu_mask, flip_offset=wall_group_offset)
            block_bottom = lambda r_in: self.jr.render_at(r_in, 0, 0, hd_mask, flip_offset=wall_group_offset)
            block_left = lambda r_in: self.jr.render_at(r_in, 0, 0, vl_mask, flip_offset=wall_group_offset)
            block_right = lambda r_in: self.jr.render_at(r_in, 0, 0, vr_mask, flip_offset=wall_group_offset)
            
            # Apply conditional logic
            r = jax.lax.cond(
                (state.entry_direction == 2) | (state.entry_direction == 3),
                lambda r_in: block_left(block_right(r_in)), 
                lambda r_in: r_in, 
                r
            )
            r = jax.lax.cond(state.entry_direction == 0, block_bottom, lambda r_in: r_in, r)
            r = jax.lax.cond(state.entry_direction == 1, block_top, lambda r_in: r_in, r)
            return r

        raster = jax.lax.cond(~room_transition_anim, draw_entry_block, lambda r: r, raster)

        # --- 4. Draw Enemy Wall Lines (Vectorized) ---
        def draw_wall_lines(r):
            line_height = 1
            line_length = 6
            wall_x = 2
            
            is_alive = state.enemy.alive
            y_coords = jnp.clip(state.enemy.pos[:, 1].astype(jnp.int32) - 1, 0, self.consts.HEIGHT - line_height)
            x_coords = jnp.full_like(y_coords, wall_x)
            
            # Create (x, y) positions and (w, h) sizes
            positions = jnp.stack([x_coords, y_coords], axis=-1)
            sizes = jnp.full_like(positions, jnp.array([line_length, line_height]))
            
            # Hide inactive enemies by setting their position to -1
            positions = jnp.where(is_alive[:, None], positions, -1)
            
            return self.jr.draw_rects(r, positions, sizes, self.BLACK_ID)

        raster = jax.lax.cond(~room_transition_anim, draw_wall_lines, lambda r: r, raster)

        # --- 5. Draw Player Bullet ---
        def draw_player_bullet(r):
            bullet_pos = state.player.bullet[0]
            dx = state.player.bullet_dir[0][0]
            
            def draw_horizontal(r_in):
                mask = self.SHAPE_MASKS['bullet_horizontal']
                offset = self.group_offsets['bullet_horizontal']
                return self.jr.render_at_clipped(r_in, bullet_pos[0], bullet_pos[1], mask, flip_offset=offset)
            
            def draw_vertical(r_in):
                mask = self.SHAPE_MASKS['bullet_vertical']
                offset = self.group_offsets['bullet_vertical']
                return self.jr.render_at_clipped(r_in, bullet_pos[0], bullet_pos[1], mask, flip_offset=offset)
            
            # Bullets can go off-screen
            return jax.lax.cond(dx != 0, draw_horizontal, draw_vertical, r)

        raster = jax.lax.cond(
            state.player.bullet_active[0] & ~room_transition_anim, 
            draw_player_bullet, 
            lambda r: r, 
            raster
        )


        # --- 6. Draw Player ---
        def get_player_sprite_idx():
            def death_animation_idx():
                idx = (state.player.death_timer - 1) % 4
                return jnp.where(idx < 2, self.sprite_indices['player_idle'], self.sprite_indices['player_death'])

            def normal_or_shoot_idx():
                dir = state.player.last_dir
                shoot_dir_idx = jnp.select(
                    [
                        (dir[0] == 0) & (dir[1] == -1),     # up
                        (dir[0] == 1) & (dir[1] == -1),     # upright
                        (dir[0] == 1) & (dir[1] == 0),      # right
                        (dir[0] == 1) & (dir[1] == 1),      # downright
                        (dir[0] == 0) & (dir[1] == 1),      # down
                        (dir[0] == -1) & (dir[1] == 1),     # downleft
                        (dir[0] == -1) & (dir[1] == 0),     # left
                        (dir[0] == -1) & (dir[1] == -1),    # upleft
                    ],
                    jnp.arange(8), default=2
                )
                shoot_idx_map = jnp.array([
                    self.sprite_indices['player_shoot_up'],
                    self.sprite_indices['player_shoot_up'],
                    self.sprite_indices['player_shoot_right'],
                    self.sprite_indices['player_shoot_down'],
                    self.sprite_indices['player_shoot_down'],
                    self.sprite_indices['player_shoot_down_left'],
                    self.sprite_indices['player_shoot_left'],
                    self.sprite_indices['player_shoot_up_left'],
                ])
                shoot_sprite_idx = shoot_idx_map[shoot_dir_idx]
                
                move_idx_map = jnp.array([
                    self.sprite_indices['player_move_1'], self.sprite_indices['player_move_1'],
                    self.sprite_indices['player_move_2'], self.sprite_indices['player_move_2'],
                    self.sprite_indices['player_idle'], self.sprite_indices['player_idle'],
                ])
                move_sprite_idx = move_idx_map[(state.player.animation_counter - 1) % 6]
                idle_sprite_idx = self.sprite_indices['player_idle']
                return jax.lax.cond(
                    state.player.is_firing,
                    lambda: shoot_sprite_idx,
                    lambda: jax.lax.cond(
                        state.player.animation_counter > 0,
                        lambda: move_sprite_idx,
                        lambda: idle_sprite_idx
                    )
                )
            
            return jax.lax.cond(death_anim, death_animation_idx, normal_or_shoot_idx)

        def draw_player(r):
            player_idx = get_player_sprite_idx()
            player_mask = self.SHAPE_MASKS['player_group'][player_idx]
            flip_h = jnp.logical_and(state.player.last_dir[0] < 0, ~state.player.is_firing)
            
            return self.jr.render_at(
                r, state.player.pos[0], state.player.pos[1], 
                player_mask, 
                flip_horizontal=flip_h, 
                flip_offset=self.group_offsets['player']
            )

        raster = jax.lax.cond(~room_transition_anim, draw_player, lambda r: r, raster)


        # --- 7. Draw Enemies ---
        def get_enemy_sprite_idx(i):
            counter = state.enemy.animation_counter[i]
            axis = state.enemy.move_axis[i]
            death_timer = state.enemy.death_timer[i]
            
            death_idx_map = jnp.array([
                self.sprite_indices['enemy_death_3'], self.sprite_indices['enemy_death_3'],
                self.sprite_indices['enemy_death_3'], self.sprite_indices['enemy_death_3'],
                self.sprite_indices['enemy_death_2'], self.sprite_indices['enemy_death_2'],
                self.sprite_indices['enemy_death_1'], self.sprite_indices['enemy_death_1'],
            ])
            death_sprite_idx = death_idx_map[(death_timer - 1) % 8]
            
            idle_idx_map = jnp.array([self.sprite_indices[f'enemy_idle_{j}'] for j in range(1, 9) for _ in range(4)])
            idle_sprite_idx = idle_idx_map[(counter - 1) % 32]
            
            move_h_idx_map = jnp.array(
                [self.sprite_indices['enemy_move_horizontal_1']] * 7 +
                [self.sprite_indices['enemy_move_horizontal_2']] * 7
            )
            move_h_sprite_idx = move_h_idx_map[(counter - 1) % 14]
            
            move_v_idx_map = jnp.array(
                [self.sprite_indices['enemy_move_vertical_1']] * 6 +
                [self.sprite_indices['enemy_move_vertical_2']] * 6 +
                [self.sprite_indices['enemy_move_vertical_1']] * 6 +
                [self.sprite_indices['enemy_move_vertical_3']] * 6
            )
            move_v_sprite_idx = move_v_idx_map[(counter - 1) % 24]
            
            def normal_animation_idx():
                axis_idx = jnp.clip(axis + 1, 0, 2)
                # Select index from: [idle_idx, move_h_idx, move_v_idx]
                return jax.lax.switch(
                    axis_idx,
                    [
                        lambda: idle_sprite_idx,
                        lambda: move_h_sprite_idx,
                        lambda: move_v_sprite_idx,
                    ]
                )
            
            return jax.lax.cond(
                death_timer > 0,
                lambda: death_sprite_idx,
                normal_animation_idx,
            )

        def draw_enemy(i, r):
            is_dying = state.enemy.death_timer[i] > 0
            pos = jax.lax.cond(is_dying, lambda: state.enemy.death_pos[i], lambda: state.enemy.pos[i])
            
            enemy_idx = get_enemy_sprite_idx(i)
            enemy_mask = self.SHAPE_MASKS['enemy_group'][enemy_idx]
            flip_h = state.enemy.move_dir[i] < 0
            
            draw_fn = lambda r_in: self.jr.render_at(
                r_in, pos[0], pos[1], enemy_mask, 
                flip_horizontal=flip_h, 
                flip_offset=self.group_offsets['enemy']
            )
            return jax.lax.cond(state.enemy.alive[i] | is_dying, draw_fn, lambda r_in: r_in, r)

        raster = jax.lax.cond(
            ~room_transition_anim,
            lambda r: jax.lax.fori_loop(0, state.enemy.pos.shape[0], draw_enemy, r),
            lambda r: r,
            raster
        )

        # --- 8. Draw Enemy Bullets ---
        def draw_enemy_bullet(i, r):
            bullet_pos = state.enemy.bullets[i]
            dx = state.enemy.bullet_dirs[i][0]
            
            def draw_horizontal(r_in):
                mask = self.SHAPE_MASKS['bullet_horizontal']
                offset = self.group_offsets['bullet_horizontal']
                return self.jr.render_at_clipped(r_in, bullet_pos[0], bullet_pos[1], mask, flip_offset=offset)
            
            def draw_vertical(r_in):
                mask = self.SHAPE_MASKS['bullet_vertical']
                offset = self.group_offsets['bullet_vertical']
                return self.jr.render_at_clipped(r_in, bullet_pos[0], bullet_pos[1], mask, flip_offset=offset)
            
            def draw_bullet(r_in):
                return jax.lax.cond(dx != 0, draw_horizontal, draw_vertical, r_in)
            
            # Use _clipped as bullets can go off-screen
            return jax.lax.cond(
                state.enemy.bullet_active[i] & ~room_transition_anim, 
                draw_bullet, 
                lambda r_in: r_in, 
                r
            )
        
        raster = jax.lax.fori_loop(0, state.enemy.bullets.shape[0], draw_enemy_bullet, raster)

        # --- 9. Draw Otto (blink single sprite on cycle) ---
        def draw_otto(r):
            # Blink off when cycle is at peak (same cadence as before)
            is_blink_off = (state.otto.anim_counter // 15) % 5 == 0
            otto_mask = self.SHAPE_MASKS['otto_group'][self.sprite_indices['evil_otto']]
            return jax.lax.cond(
                ~is_blink_off,
                lambda r_in: self.jr.render_at(r_in, state.otto.pos[0], state.otto.pos[1], otto_mask, flip_offset=self.group_offsets['otto']),
                lambda r_in: r_in,
                r,
            )

        raster = jax.lax.cond(state.otto.active, draw_otto, lambda r: r, raster)

        # --- 10. Draw UI (Score, Title, Lives) ---
        digit_masks = self.SHAPE_MASKS['digits']
        
        def render_scores(r):
            max_score_digits = 5
            score_spacing = 8
            
            def draw_score(value, offset_x):
                score_digits = self.jr.int_to_digits(value.astype(jnp.int32), max_digits=max_score_digits)
                
                # Find first non-zero digit
                is_non_zero = score_digits != 0
                first_non_zero = jnp.argmax(is_non_zero)
                start_idx = jax.lax.select(jnp.any(is_non_zero), first_non_zero, max_score_digits - 1)
                num_to_render = max_score_digits - start_idx
                
                # Align right
                render_start_x = offset_x - score_spacing * (num_to_render - 1)
                
                return self.jr.render_label_selective(
                    r, render_start_x, self.consts.SCORE_OFFSET_Y,
                    score_digits, digit_masks,
                    start_idx, num_to_render,
                    spacing=score_spacing,
                    max_digits_to_render=max_score_digits
                )
            
            show_bonus = state.enemy.clear_bonus_given
            
            # Draw bonus if applicable
            r = jax.lax.cond(
                show_bonus,
                lambda r_in: draw_score(state.num_enemies * 10, self.consts.SCORE_OFFSET_X - 31),
                lambda r_in: r_in,
                r
            )
            # Draw score (if not 0 or if bonus is showing)
            r = jax.lax.cond(
                (state.score > 0) | show_bonus,
                lambda r_in: draw_score(state.score, self.consts.SCORE_OFFSET_X),
                lambda r_in: r_in,
                r
            )
            return r

        def render_title(r):
            title_mask = self.SHAPE_MASKS['start_title']
            x = (self.consts.WIDTH - title_mask.shape[1]) // 2 + 2
            y = self.consts.SCORE_OFFSET_Y
            return self.jr.render_at(r, x, y, title_mask, flip_offset=self.group_offsets['start_title'])
        
        # Render score or title, but not both
        raster = jax.lax.cond(
            ~room_transition_anim,
            lambda r: jax.lax.cond(
                state.score == 0,
                render_title,
                render_scores,
                r
            ),
            lambda r: r,
            raster
        )

        # Render lives (during transition or death)
        def render_lives(r):
            life_mask = self.SHAPE_MASKS['life']
            life_offset = self.group_offsets['life']
            life_spacing = 8
            start_x = self.consts.SCORE_OFFSET_X
            start_y = self.consts.SCORE_OFFSET_Y
            num_lives_to_draw = jax.lax.cond(
                death_anim,
                lambda: jnp.maximum(state.lives - 1, 0).astype(jnp.int32),
                lambda: state.lives.astype(jnp.int32)
            )
            def draw_life(i, r_in):
                x = start_x - i * life_spacing
                y = start_y
                return self.jr.render_at(r_in, x, y, life_mask, flip_offset=life_offset)
            
            # Assuming a reasonable max number of lives to show
            def body(i, r_in):
                return jax.lax.cond(
                    i < num_lives_to_draw,
                    lambda r0: draw_life(i, r0),
                    lambda r0: r0,
                    r_in,
                )
            return jax.lax.fori_loop(0, 5, body, r)

        raster = jax.lax.cond(room_transition_anim | death_anim, render_lives, lambda r: r, raster)

        # --- 11. Draw Transition/Game Over Overlays ---
        def apply_bar_overlay(r, progress: jnp.ndarray, mode_idx: int):
            # This function now writes BLACK_ID to the integer raster
            total_height, width = r.shape
            playfield_height = total_height - self.consts.WALL_OFFSET[3]
            covered_rows = jnp.floor(progress * playfield_height).astype(jnp.int32)
            rows = jnp.arange(total_height)
            
            top_down_mask = lambda: rows < covered_rows
            bottom_up_mask = lambda: rows >= (playfield_height - covered_rows)
            center_inward_mask = lambda: (rows < (covered_rows // 2)) | (rows >= (playfield_height - covered_rows // 2))
            mask = jax.lax.switch(
                mode_idx,
                [top_down_mask, bottom_up_mask, center_inward_mask, center_inward_mask]
            )
            
            return jnp.where(mask[:, None], self.BLACK_ID, r)

        # Apply transition overlay
        progress_transition = 1.0 - (state.room_transition_timer.astype(jnp.float32) / self.consts.TRANSITION_ANIMATION_FRAMES)
        raster = jax.lax.cond(
            room_transition_anim,
            lambda r: apply_bar_overlay(r, progress_transition, state.entry_direction),
            lambda r: r,
            raster
        )
        
        # Apply game over overlay (full black screen)
        raster = jax.lax.cond(
            game_over_anim,
            lambda _: jnp.full_like(raster, self.BLACK_ID),
            lambda _: raster,
            operand=None
        )

        # --- 12. Final Palette Lookup ---
        # We use the 'frame_palette' which contains our dynamic enemy/bullet colors
        return self.jr.render_from_palette(raster, frame_palette)

    # --- Helper function for recoloring logic ---
    @partial(jax.jit, static_argnums=(0,))
    def _get_enemy_color_index(self, room_counter: jnp.ndarray) -> jnp.ndarray:
        num_colors = self.color_names_ids.shape[0]
        return jax.lax.cond(
            room_counter == 0,
            lambda: jnp.array(0, dtype=jnp.int32),  # Default (yellow)
            lambda: ((room_counter - 1) // 2 + 1) % num_colors
        )