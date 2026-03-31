import os
from functools import partial
from typing import ClassVar, Tuple
from flax import struct
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

def _get_default_asset_config():
    return (
        # Background (could be procedural or a single tile)
        {"name": "background", "type": "background", "file": "background.npy"},

        # Player animation frames per direction
        # right(×3): p1-3  |  left(×3): p4-6  |  up(×5): p7-11  |  down(×5): p12-16
        {"name": "player_r0", "type": "single", "file": "p1.npy"},
        {"name": "player_r1", "type": "single", "file": "p2.npy"},
        {"name": "player_r2", "type": "single", "file": "p3.npy"},
        {"name": "player_l0", "type": "single", "file": "p4.npy"},
        {"name": "player_l1", "type": "single", "file": "p5.npy"},
        {"name": "player_l2", "type": "single", "file": "p6.npy"},
        {"name": "player_u0", "type": "single", "file": "p7.npy"},
        {"name": "player_u1", "type": "single", "file": "p8.npy"},
        {"name": "player_u2", "type": "single", "file": "p9.npy"},
        {"name": "player_u3", "type": "single", "file": "p10.npy"},
        {"name": "player_u4", "type": "single", "file": "p11.npy"},
        {"name": "player_d0", "type": "single", "file": "p12.npy"},
        {"name": "player_d1", "type": "single", "file": "p13.npy"},
        {"name": "player_d2", "type": "single", "file": "p14.npy"},
        {"name": "player_d3", "type": "single", "file": "p15.npy"},
        {"name": "player_d4", "type": "single", "file": "p16.npy"},

        # Zombie animation frames per direction                             # ENEMY_ZOMBIE = 1
        # right(×3): green_1-3  |  down(×8): custom cycle  |  up(×8): custom cycle  |  left(×3): green_12-14
        {"name": "green_r0", "type": "single", "file": "green_1.npy"},
        {"name": "green_r1", "type": "single", "file": "green_2.npy"},
        {"name": "green_r2", "type": "single", "file": "green_3.npy"},
        {"name": "green_d0", "type": "single", "file": "green_4.npy"},
        {"name": "green_d1", "type": "single", "file": "green_5.npy"},
        {"name": "green_d2", "type": "single", "file": "green_6.npy"},
        {"name": "green_d3", "type": "single", "file": "green_7.npy"},
        {"name": "green_d4", "type": "single", "file": "green_8.npy"},
        {"name": "green_u0", "type": "single", "file": "green_9.npy"},
        {"name": "green_u1", "type": "single", "file": "green_10.npy"},
        {"name": "green_u2", "type": "single", "file": "green_11.npy"},
        {"name": "green_l0", "type": "single", "file": "green_12.npy"},
        {"name": "green_l1", "type": "single", "file": "green_13.npy"},
        {"name": "green_l2", "type": "single", "file": "green_14.npy"},
        # Direct-name aliases for down/up 8-frame cycles
        {"name": "green_6",   "type": "single", "file": "green_6.npy"},
        {"name": "green_7",   "type": "single", "file": "green_7.npy"},
        {"name": "green_8",   "type": "single", "file": "green_8.npy"},
        {"name": "green_9",   "type": "single", "file": "green_9.npy"},
        {"name": "green_10",  "type": "single", "file": "green_10.npy"},
        {"name": "green_d_d", "type": "single", "file": "green_d_d.npy"},   # neutral down stance
        {"name": "green_dn_l","type": "single", "file": "green_dn_l.npy"},  # down left-lean step (screenshots/green_l1)
        {"name": "green_u_d", "type": "single", "file": "green_u_d.npy"},   # neutral up stance
        {"name": "green_u_r1","type": "single", "file": "green_u_r1.npy"},  # up right step 1
        {"name": "green_u_r2","type": "single", "file": "green_u_r2.npy"},  # up right step 2
        # Ghost (wraith) animation frames per direction   # ENEMY_WRAITH = 2
        # right(×2): ghost_1,ghost_2  |  left(×2): ghost_4,ghost_5
        # down(×3):  ghost_6-8        |  up(×3):   ghost_9-11
        {"name": "ghost_r0", "type": "single", "file": "ghost_1.npy"},
        {"name": "ghost_r1", "type": "single", "file": "ghost_2.npy"},
        {"name": "ghost_l0", "type": "single", "file": "ghost_4.npy"},
        {"name": "ghost_l1", "type": "single", "file": "ghost_5.npy"},
        {"name": "ghost_d0", "type": "single", "file": "ghost_6.npy"},
        {"name": "ghost_d1", "type": "single", "file": "ghost_7.npy"},
        {"name": "ghost_d2", "type": "single", "file": "ghost_8.npy"},
        {"name": "ghost_u0", "type": "single", "file": "ghost_9.npy"},
        {"name": "ghost_u1", "type": "single", "file": "ghost_10.npy"},
        {"name": "ghost_u2", "type": "single", "file": "ghost_11.npy"},
        # Skeleton animation frames per direction                          # ENEMY_SKELETON = 3
        # right(×3): skel_4-6  |  down(×3): skel_left/mid/right  |  left(×3): skel_1-3  |  up(×3): skel_up_left/mid/right
        {"name": "skel_r0", "type": "single", "file": "skel_4.npy"},
        {"name": "skel_r1", "type": "single", "file": "skel_5.npy"},
        {"name": "skel_r2", "type": "single", "file": "skel_6.npy"},
        {"name": "skel_d0", "type": "single", "file": "skel_left.npy"},
        {"name": "skel_d1", "type": "single", "file": "skel_mid.npy"},
        {"name": "skel_d2", "type": "single", "file": "skel_right.npy"},
        {"name": "skel_l0", "type": "single", "file": "skel_1.npy"},
        {"name": "skel_l1", "type": "single", "file": "skel_2.npy"},
        {"name": "skel_l2", "type": "single", "file": "skel_3.npy"},
        {"name": "skel_u0", "type": "single", "file": "skel_up_left.npy"},
        {"name": "skel_u1", "type": "single", "file": "skel_up_mid.npy"},
        {"name": "skel_u2", "type": "single", "file": "skel_up_right.npy"},
        # Direct-number aliases for down/up 4-frame cycles
        {"name": "skel_7",  "type": "single", "file": "skel_7.npy"},
        {"name": "skel_8",  "type": "single", "file": "skel_8.npy"},
        {"name": "skel_9",  "type": "single", "file": "skel_9.npy"},
        {"name": "skel_10", "type": "single", "file": "skel_10.npy"},
        {"name": "skel_11", "type": "single", "file": "skel_11.npy"},
        {"name": "skel_12", "type": "single", "file": "skel_12.npy"},
        # Wizard animation frames per direction                            # ENEMY_WIZARD = 4
        # right(×3): wizard_1-3  |  down(×5): wizard_4-8  |  up(×5): wizard_9-13  |  left(×3): wizard_14-16
        {"name": "wizard_r0", "type": "single", "file": "wizard_1.npy"},
        {"name": "wizard_r1", "type": "single", "file": "wizard_2.npy"},
        {"name": "wizard_r2", "type": "single", "file": "wizard_3.npy"},
        {"name": "wizard_d0", "type": "single", "file": "wizard_4.npy"},
        {"name": "wizard_d1", "type": "single", "file": "wizard_5.npy"},
        {"name": "wizard_d2", "type": "single", "file": "wizard_6.npy"},
        {"name": "wizard_d3", "type": "single", "file": "wizard_7.npy"},
        {"name": "wizard_d4", "type": "single", "file": "wizard_8.npy"},
        {"name": "wizard_u0", "type": "single", "file": "wizard_9.npy"},
        {"name": "wizard_u1", "type": "single", "file": "wizard_10.npy"},
        {"name": "wizard_u2", "type": "single", "file": "wizard_11.npy"},
        {"name": "wizard_u3", "type": "single", "file": "wizard_12.npy"},
        {"name": "wizard_u4", "type": "single", "file": "wizard_13.npy"},
        {"name": "wizard_l0", "type": "single", "file": "wizard_14.npy"},
        {"name": "wizard_l1", "type": "single", "file": "wizard_15.npy"},
        {"name": "wizard_l2", "type": "single", "file": "wizard_16.npy"},
        # Note: No sprite for ENEMY_GRIM_REAPER = 5 yet, will use colored box

        # Item sprites
        {"name": "pot", "type": "single", "file": "pot.npy"},             # Shield icon (was pot)
        {"name": "skull", "type": "single", "file": "skull.npy"},         # Poison/Trap (danger items)
        {"name": "stairs", "type": "single", "file": "stairs.npy"},       # Ladder up (exit door)
        {"name": "trapdoor", "type": "single", "file": "trapdoor.npy"},   # Ladder down (descent)

        # New props / tiles
        {"name": "apple", "type": "single", "file": "apple.npy"},
        {"name": "barrel", "type": "single", "file": "barrel.npy"},
        {"name": "candle", "type": "single", "file": "candle.npy"},
        {"name": "chain", "type": "single", "file": "chain.npy"},
        {"name": "door", "type": "single", "file": "door.npy"},
        {"name": "key", "type": "single", "file": "key.npy"},
        {"name": "bullet", "type": "single", "file": "bullet.npy"},
        {"name": "pistol", "type": "single", "file": "pistol.npy"},

        # Digits for UI - commented out, using hardcoded digit patterns instead
        # {"name": "digits", "type": "digits", "pattern": "digits/{}.npy"},
    )



# Game configuration and constants

GAME_H = 210  # Standard Atari height
GAME_W = 160  # Standard Atari width
UI_BAR_HEIGHT = 50  # Height of bottom UI bar (5% bigger than 48)
GAMEPLAY_H = GAME_H - UI_BAR_HEIGHT  # Viewport height for gameplay (160)
WORLD_W = GAME_W  # World width equals viewport width (no horizontal camera panning)
WORLD_H = 600  # Significantly taller world

# --- Nav grid for enemy pathfinding ---
CELL_SIZE = 8                      # size of one nav cell in pixels
GRID_W = WORLD_W // CELL_SIZE
GRID_H = WORLD_H // CELL_SIZE
BIG_DIST = 10_000                  # "infinity" for distance field

NUM_ENEMIES = 4  # Increased to allow more spawned enemies
NUM_SPAWNERS = 1  # Spawner entities

# Enemy types (5 = strongest, 1 = weakest)
ENEMY_GRIM_REAPER = 5  # Strongest
ENEMY_WIZARD = 4
ENEMY_SKELETON = 3
ENEMY_WRAITH = 2
ENEMY_ZOMBIE = 1  # Weakest

# Item configuration
NUM_CHECKPOINTS = 16
NUM_ITEMS = 24  # 5 fixed + 3 regular + 16 checkpoint slots
INITIAL_REGULAR_ITEM_COUNT = 5  # Fewer random items active at once to reduce clutter
# Item type codes
ITEM_HEART = 1          # +health, no points
ITEM_POISON = 2         # -4 health, no points
ITEM_TRAP = 3           # -6 health, no points
ITEM_STRONGBOX = 4      # +100 points
ITEM_AMBER_CHALICE = 5 # +500 points
ITEM_AMULET = 6         # +1000 points
ITEM_GOLD_CHALICE = 7   # +3000 points
ITEM_SHIELD = 8         # Damage reduction
ITEM_GUN = 9            # Faster shooting
ITEM_BOMB = 10          # Kill all enemies in area
ITEM_KEY = 11           # Key used to unlock gated extras
ITEM_LADDER_UP = 12     # Exit/door leading up (always accessible)
ITEM_LADDER_DOWN = 13   # Ladder leading down (always open)
ITEM_CAGE_DOOR = 14     # Door item that gates entry into the bonus cage

# New potion item types (mods)
ITEM_SPEED_POTION = 15  # Temporarily increases player movement speed (2x for 120 steps)
ITEM_HEAL_POTION = 16   # Fully restores player health to MAX_HEALTH on pickup
ITEM_POISON_POTION = 17 # Creates poison cloud that damages enemies in radius over time
ITEM_HAMMER = 18        # Kills all enemies within radius; limited uses per episode
ITEM_CHECKPOINT = 19    # Intra-level checkpoint; activates on touch, never collected

# Level configuration
MAX_LEVELS = 7          # Total number of levels (0..6)
LADDER_INTERACTION_TIME = 60  # Steps player must stand on ladder to change level
LADDER_WIDTH = 8        # Matches stairs.npy natural width
LADDER_HEIGHT = 28      # Matches stairs.npy natural height
LADDER_LEVEL_CHANGE_BLOCK_TICKS = 120  # Block immediate ladder re-trigger after a level change

# Bomb configuration
MAX_BOMBS = 15          # Maximum bombs player can carry
DOUBLE_TAP_WINDOW = 10  # Steps within which two fires count as double-tap
FIRE_RATE_LIMIT = 35    # Slower base fire rate (~1.2s at 30 FPS)
FIRE_RATE_LIMIT_WITH_GUN = 18  # Gun still speeds up shooting noticeably
BOMB_RADIUS = 80        # Kill radius for bomb in pixels
MAX_HAMMERS = 3         # Maximum hammers player can carry
HAMMER_RADIUS = 100     # Kill radius for hammer in pixels
ENEMY_HITS_PER_TIER = 2 # Bullet/poison hits needed before an enemy drops one tier

# Default base size (unused now, kept for reference)
ITEM_WIDTH = 6
ITEM_HEIGHT = 6

# Bullet configuration
# Player bullet configuration
MAX_BULLETS = 64             # Drastically higher simultaneous bullets
BULLET_WIDTH = 4
BULLET_HEIGHT = 4
BULLET_SPEED = 3             # Same as player speed
BULLET_SPEED_WITH_GUN = 5    # Faster when gun is active

# Wizard projectile configuration
ENEMY_MAX_BULLETS = 100
ENEMY_BULLET_WIDTH = 3
ENEMY_BULLET_HEIGHT = 3
ENEMY_BULLET_SPEED = 2
WIZARD_SHOOT_INTERVAL = 20  # base steps between shots for each wizard
WIZARD_SHOOT_OFFSET_MAX = 20  # random offset added to shooting interval (0-20)

# Object-centric observation caps to reduce dimensionality/noise for MLP agents.
OBS_MAX_PLAYER_BULLETS = 12
OBS_MAX_ENEMY_BULLETS = 12
OBS_MAX_WALLS = 24
OBS_SCORE_CLIP = 50_000
OBS_STEP_CLIP = 20_000

# Lives configuration
MAX_LIVES = 1  # Number of lives player starts with

# Death freeze configuration (ticks to freeze before respawn)
DEATH_FREEZE_TICKS = 60

# Chamber transition freeze configuration
CHAMBER_TRANSITION_STEPS = 3
CHAMBER_TRANSITION_TICKS_PER_STEP = 20
CHAMBER_TRANSITION_TICKS = CHAMBER_TRANSITION_STEPS * CHAMBER_TRANSITION_TICKS_PER_STEP
LEVEL_CHANGE_FREEZE_TICKS = 60

LEVEL_CHANGE_FONT_5X7 = {
    " ": (
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ),
    "A": (
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ),
    "C": (
        "01111",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "01111",
    ),
    "D": (
        "11110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11110",
    ),
    "E": (
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ),
    "G": (
        "01111",
        "10000",
        "10000",
        "10111",
        "10001",
        "10001",
        "01110",
    ),
    "H": (
        "10001",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ),
    "L": (
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "11111",
    ),
    "N": (
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ),
    "V": (
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01010",
        "00100",
    ),
}

# Spawner configuration
SPAWNER_WIDTH = 14
SPAWNER_HEIGHT = 28  # 2:1 aspect ratio (height = 2 * width, original width)
SPAWNER_HEALTH = 3  # Takes 3 hits to destroy
SPAWNER_SPAWN_INTERVAL = 320  # Spawn enemies significantly less often

ENEMY_COLLISION_MARGIN = 1
ENEMY_MOVE_EVERY = 4  # Enemies move every N game steps (used for movement throttle and animation timing)
ENEMY_PORTAL_KEEP_OUT_X = 18  # Keep enemies away from left/right portal entrances

# Directional sprite index lookup tables.
# Sprite order stored in PLAYER_DIRECTIONAL_MASKS / ENEMY_DIRECTIONAL_MASKS:
#   index 0 = right (_1), 1 = down (_2), 2 = left (_3), 3 = up (_4)
#
# player_direction: 0=right, 1=left, 2=up, 3=down  →  sprite index
PLAYER_DIR_TO_SPRITE = jnp.array([0, 2, 3, 1], dtype=jnp.int32)
# Number of animation frames per player direction (sprite index order: 0=right, 1=down, 2=left, 3=up)
PLAYER_NUM_FRAMES = jnp.array([3, 5, 3, 5], dtype=jnp.int32)
# Default (idle) frame index per player sprite direction (0=right, 1=down, 2=left, 3=up)
# right-idle=p1(idx 0), down-idle=p12(idx 0), left-idle=p6(idx 2), up-idle=p9(idx 2)
PLAYER_IDLE_FRAME = jnp.array([0, 0, 2, 2], dtype=jnp.int32)
#
# enemy_dir is an index into DIR8 (0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE)
# Diagonals snap to the dominant axis (horizontal wins).
DIR8_TO_SPRITE = jnp.array([0, 0, 3, 2, 2, 2, 1, 0], dtype=jnp.int32)

# How many game steps between animation frame advances for all entities (higher = slower)
ANIM_EVERY = 10

# Number of animation frames per ghost direction (sprite index order: 0=right, 1=down, 2=left, 3=up)
GHOST_NUM_FRAMES = jnp.array([2, 3, 2, 3], dtype=jnp.int32)
# Number of animation frames per wizard direction (0=right, 1=down, 2=left, 3=up)
WIZARD_NUM_FRAMES = jnp.array([8, 8, 7, 8], dtype=jnp.int32)
# Number of animation frames per skeleton direction (0=right, 1=down, 2=left, 3=up)
SKELETON_NUM_FRAMES = jnp.array([3, 4, 3, 4], dtype=jnp.int32)
# Number of animation frames per zombie direction (0=right, 1=down, 2=left, 3=up)
ZOMBIE_NUM_FRAMES = jnp.array([3, 8, 3, 8], dtype=jnp.int32)

CHASE_RADIUS = 80          # pixels
IDLE_SPEED = 1              # pixels per step (keep <= 1 for fewer collision issues)

# --- Enemy patrol/aggro state machine params ---
AGGRO_EVAL_EVERY = 12      # every N ticks, decide whether to aggro
AGGRO_RADIUS = 80
CHASE_TICKS = 90           # T_chase: how long to chase once triggered
CONFUSE_TICKS = 30         # back-off/confuse duration
CONFUSE_PROB_NUM = 1       # probability = CONFUSE_PROB_NUM / CONFUSE_PROB_DEN
CONFUSE_PROB_DEN = 256
STUCK_TICKS = 30           # if not making progress, trigger confuse (~0.5s at 60fps)



ORBIT_DIRS = jnp.array([
    [ 1,  0],  # E
    [ 1, -1],  # NE
    [ 0, -1],  # N
    [-1, -1],  # NW
    [-1,  0],  # W
    [-1,  1],  # SW
    [ 0,  1],  # S
    [ 1,  1],  # SE
], dtype=jnp.int32)



class DarkChambersConstants(struct.PyTreeNode):
    """Constants that define gameplay, visuals, and world layout."""
    # Dimensions
    WIDTH: int = struct.field(pytree_node=False, default=GAME_W)
    HEIGHT: int = struct.field(pytree_node=False, default=GAME_H)
    WORLD_WIDTH: int = struct.field(pytree_node=False, default=WORLD_W)
    WORLD_HEIGHT: int = struct.field(pytree_node=False, default=WORLD_H)
    
    # Color scheme
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(74, 74, 74))  # Dark gray floor from screenshot
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(200, 80, 60))
    # Enemy colors by type (from weakest to strongest)
    ZOMBIE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(100, 100, 100))  # Gray
    WRAITH_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(180, 180, 220))  # Light purple
    SKELETON_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(220, 220, 200))  # Bone white
    WIZARD_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(150, 80, 200))  # Purple
    GRIM_REAPER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(50, 50, 50))  # Dark gray/black
    WALL_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(213, 117, 114))  # Salmon/coral pink walls from screenshot
    HEART_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(220, 30, 30))
    POISON_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(50, 200, 50))  # Green poison
    TRAP_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(120, 70, 20))     # Brown trap
    TREASURE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(255, 220, 0)) # Yellow for all treasures
    SPAWNER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(180, 50, 180)) # Magenta/purple spawner
    SHIELD_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(80, 120, 255)) # Blue shield
    GUN_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(40, 40, 40)) # Black gun
    BOMB_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(240, 240, 240)) # White bomb
    KEY_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 255, 255)) # Cyan key (easy to distinguish from gold)
    LADDER_UP_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(140, 90, 40)) # Brownish exit door
    LADDER_DOWN_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(100, 60, 20)) # Darker brown ladder down
    UI_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))
    HUD_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(199, 108, 58))  # HUD health/score color
    BULLET_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(255, 200, 0))
    SPEED_POTION_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(255, 100, 0))   # Orange
    HEAL_POTION_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(255, 0, 255))    # Magenta
    POISON_POTION_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 255, 0))    # Green
    
    # Sizes
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=12)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=28)
    ENEMY_WIDTH: int = struct.field(pytree_node=False, default=10)
    ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=28)
    
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=1)
    WALL_THICKNESS: int = struct.field(pytree_node=False, default=8)
    
    PLAYER_START_X: int = struct.field(pytree_node=False, default=18)  # Mirrored from x=130 across map midline (WORLD_W=160, PLAYER_WIDTH=12)
    PLAYER_START_Y: int = struct.field(pytree_node=False, default=210)
    
    # Health mechanics (scaled to classic 31 strength units)
    MAX_HEALTH: int = struct.field(pytree_node=False, default=31)
    STARTING_HEALTH: int = struct.field(pytree_node=False, default=31)
    HEALTH_GAIN: int = struct.field(pytree_node=False, default=10)  # Heart potion gain
    POISON_DAMAGE: int = struct.field(pytree_node=False, default=4)   # Light damage
    TRAP_DAMAGE: int = struct.field(pytree_node=False, default=6)     # Heavier damage
    
    # Enemy damage by type (contact damage per step)
    ZOMBIE_DAMAGE: int = struct.field(pytree_node=False, default=1)  # Weakest
    WRAITH_DAMAGE: int = struct.field(pytree_node=False, default=1)
    SKELETON_DAMAGE: int = struct.field(pytree_node=False, default=3)
    WIZARD_DAMAGE: int = struct.field(pytree_node=False, default=4)
    GRIM_REAPER_DAMAGE: int = struct.field(pytree_node=False, default=5)  # Strongest
    
    # Enemy scoring (points awarded when killed)
    ZOMBIE_POINTS: int = struct.field(pytree_node=False, default=10)  # Weakest - explodes when killed
    WRAITH_POINTS: int = struct.field(pytree_node=False, default=20)
    SKELETON_POINTS: int = struct.field(pytree_node=False, default=30)
    WIZARD_POINTS: int = struct.field(pytree_node=False, default=50)
    GRIM_REAPER_POINTS: int = struct.field(pytree_node=False, default=100)  # Strongest
    
    # Potion effect tuning
    SPEED_POTION_DURATION: int = struct.field(pytree_node=False, default=120)  # 120 steps (~4 seconds at 30 FPS)
    SPEED_POTION_MULTIPLIER: int = struct.field(pytree_node=False, default=2)  # 2x movement speed
    POISON_DURATION: int = struct.field(pytree_node=False, default=360)        # 360 steps (~12 seconds at 30 FPS) - increased from 180 for longer effect
    POISON_RADIUS: int = struct.field(pytree_node=False, default=60)           # Damage radius in pixels - reduced from 80 for more targeted effect
    POISON_DAMAGE_INTERVAL: int = struct.field(pytree_node=False, default=30)  # Apply damage every 30 steps (once per second) - prevents instant kills

    # Potion item spawn toggles (disabled by default; enabled by mods via constants_overrides)
    ENABLE_SPEED_POTION_SPAWN: bool = struct.field(pytree_node=False, default=False)
    ENABLE_HEAL_POTION_SPAWN: bool = struct.field(pytree_node=False, default=False)
    ENABLE_POISON_POTION_SPAWN: bool = struct.field(pytree_node=False, default=False)

    # Base poison item spawn toggle (keeps poison logic intact, only disables spawning)
    ENABLE_DEFAULT_POISON_SPAWN: bool = struct.field(pytree_node=False, default=False)
    ENABLE_DEFAULT_TRAP_SPAWN: bool = struct.field(pytree_node=False, default=True)

    HAMMER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(148, 0, 211))  # Brown hammer
    ENABLE_HAMMER_SPAWN: bool = struct.field(pytree_node=False, default=False)

    # Advanced enemy features (disabled by default; enabled by mods)
    ENABLE_GRIM_REAPER_ENEMIES: bool = struct.field(pytree_node=False, default=False)
    ENABLE_WIZARD_BULLETS: bool = struct.field(pytree_node=False, default=False)

    # Checkpoint system (disabled by default; enabled by checkpoint mod)
    # Map index order: 0=middle chamber, 1=left chamber, 2=right chamber.
    ENABLE_CHECKPOINT_RESPAWN: bool = struct.field(pytree_node=False, default=False)
    CHECKPOINT_SPAWN_X_BY_MAP: Tuple[int, int, int] = struct.field(pytree_node=False, default=(50, 20, 130))
    CHECKPOINT_SPAWN_Y_BY_MAP: Tuple[int, int, int] = struct.field(pytree_node=False, default=(50, 70, 70))

    # Intra-level checkpoint objects (disabled by default; enabled by checkpoint mod)
    ENABLE_INTRA_LEVEL_CHECKPOINTS: bool = struct.field(pytree_node=False, default=False)
    CHECKPOINT_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(255, 220, 50))   # gold flag


@struct.dataclass
class DarkChambersState:
    """Immutable snapshot of the current game state."""
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array  # 0=right, 1=left, 2=up, 3=down
    player_moving: chex.Array     # 1 if a directional button is held this step, else 0

    enemy_positions: chex.Array  # shape: (NUM_ENEMIES, 2)
    enemy_types: chex.Array      # shape: (NUM_ENEMIES,) - 1=zombie, 2=wraith, 3=skeleton, 4=wizard, 5=grim_reaper
    enemy_active: chex.Array     # shape: (NUM_ENEMIES,) - 1=alive, 0=dead
    enemy_hitpoints: chex.Array  # shape: (NUM_ENEMIES,) - hits remaining before dropping one tier
    wizard_shoot_timers: chex.Array  # shape: (NUM_ENEMIES,) - countdown to next shot for wizards
    
    spawner_positions: chex.Array  # shape: (NUM_SPAWNERS, 2)
    spawner_health: chex.Array     # shape: (NUM_SPAWNERS,) - health remaining
    spawner_active: chex.Array     # shape: (NUM_SPAWNERS,) - 1=active, 0=destroyed
    spawner_timers: chex.Array     # shape: (NUM_SPAWNERS,) - countdown to next spawn
    
    bullet_positions: chex.Array  # (MAX_BULLETS, 4) - x, y, dx, dy
    bullet_active: chex.Array     # (MAX_BULLETS,) - 1=active, 0=inactive
    enemy_bullet_positions: chex.Array  # (ENEMY_MAX_BULLETS, 4) - x, y, dx, dy
    enemy_bullet_active: chex.Array     # (ENEMY_MAX_BULLETS,) - 1=active, 0=inactive
    
    health: chex.Array
    score: chex.Array
    
    item_positions: chex.Array  # (NUM_ITEMS, 2)
    item_types: chex.Array      # 1=heart, 2=poison
    item_active: chex.Array     # 1=active, 0=collected
    
    has_key: chex.Array         # 1=has key, 0=no key
    shield_active: chex.Array   # 1=shield active, 0=no shield
    gun_active: chex.Array      # 1=gun active, 0=no gun
    bomb_count: chex.Array      # number of bombs (0-15)
    hammer_count: chex.Array    # number of hammers (0-MAX_HAMMERS)
    last_fire_step: chex.Array  # step counter when fire was last pressed (for double-tap)
    last_shot_step: chex.Array  # step counter when a bullet was actually spawned
    fire_was_pressed: chex.Array  # 1 if fire was pressed last step (for single-shot detection)
    
    current_level: chex.Array   # current level index (0 to MAX_LEVELS-1)
    map_index: chex.Array       # current map variant (0=middle, 1=left, 2=right)
    ladder_timer: chex.Array    # time standing on ladder (0 to LADDER_INTERACTION_TIME)
    ladder_transition_cooldown: chex.Array  # blocks ladder transitions right after level travel
    lives: chex.Array           # remaining lives (0 to MAX_LIVES)
    
    step_counter: chex.Array
    death_counter: chex.Array    # >0 means freeze frames remaining before respawn
    chamber_transition_ticks: chex.Array  # >0 means side-chamber transition freeze with darkening
    chamber_transition_pending: chex.Array  # 1 while waiting to apply delayed side-chamber switch
    chamber_target_map_index: chex.Array  # target map index for delayed side-chamber switch
    chamber_target_x: chex.Array  # target player x after delayed side-chamber switch
    chamber_target_y: chex.Array  # target player y after delayed side-chamber switch
    level_transition_ticks: chex.Array  # >0 means level-change black screen freeze
    key: chex.PRNGKey

    damage_cooldown: chex.Array

    enemy_dir: chex.Array
    enemy_chase_timer: chex.Array
    enemy_confuse_timer: chex.Array
    enemy_patrol_box: chex.Array 
    enemy_stuck_counter: chex.Array

    enemy_idle_timer: chex.Array
    enemy_pause_timer: chex.Array
    
    # Potion effect state tracking
    speed_boost_timer: chex.Array      # Countdown for speed boost effect (0 = inactive)
    poison_cloud_x: chex.Array         # X position of active poison cloud
    poison_cloud_y: chex.Array         # Y position of active poison cloud
    poison_cloud_timer: chex.Array     # Countdown for poison effect (0 = inactive)

    enemy_contact_cooldown: chex.Array  # shape: (NUM_ENEMIES,) - frames until next contact damage tick
    enemy_moving: chex.Array            # shape: (NUM_ENEMIES,) - 1 if enemy moved this step, else 0

    checkpoint_idx: chex.Array    # index into item_positions of last activated intra-level checkpoint (-1 = none)
    checkpoint_map: chex.Array    # map_index of the room where checkpoint was activated
    checkpoint_level: chex.Array  # current_level of the room where checkpoint was activated



@struct.dataclass
class EntityPosition:
    """Axis-aligned rectangle: top-left position and size."""
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


@struct.dataclass
class DarkChambersObservation:
    """Compact observation used by agents and the UI."""
    player: EntityPosition
    enemies: jnp.ndarray  # (NUM_ENEMIES, 6): screen_x, screen_y, width, height, type, in_view
    items: jnp.ndarray    # (NUM_ITEMS, 6): screen_x, screen_y, width, height, type, in_view
    spawners: jnp.ndarray  # (NUM_SPAWNERS, 5): screen_x, screen_y, width, height, in_view
    player_bullets: jnp.ndarray  # (OBS_MAX_PLAYER_BULLETS, 5): screen_x, screen_y, width, height, in_view
    enemy_bullets: jnp.ndarray  # (OBS_MAX_ENEMY_BULLETS, 5): screen_x, screen_y, width, height, in_view
    portals: jnp.ndarray  # (6, 3): screen_x, screen_y, in_view
    walls: jnp.ndarray    # (OBS_MAX_WALLS, 5): screen_x, screen_y, width, height, in_view
    border_distances: jnp.ndarray  # (4,): left, right, top, bottom (player->camera viewport border)
    health: jnp.ndarray
    score: jnp.ndarray
    step: jnp.ndarray
    # Player equipment state (invisible to pixel agent only via HUD rendering)
    has_key: jnp.ndarray        # 1=has key, 0=no key
    shield_active: jnp.ndarray  # 1=shield active, 0=none
    gun_active: jnp.ndarray     # 1=gun active, 0=none
    bomb_count: jnp.ndarray     # 0..MAX_BOMBS
    hammer_count: jnp.ndarray   # 0..MAX_HAMMERS
    speed_boost_active: jnp.ndarray  # 1=speed boost active, 0=none
    lives: jnp.ndarray               # remaining lives (0 if not using CheckpointsMod)
    poison_cloud_x: jnp.ndarray      # x position of active poison cloud
    poison_cloud_y: jnp.ndarray      # y position of active poison cloud
    poison_cloud_active: jnp.ndarray # 1=poison cloud active, 0=none
    damage_cooldown: jnp.ndarray     # frames of invulnerability remaining


@struct.dataclass
class DarkChambersInfo:
    """Auxiliary info not intended for learning signals."""
    time: jnp.ndarray

class DarkChambersRenderer(JAXGameRenderer):
    """Software renderer for Dark Chambers."""
    
    def __init__(self, consts: DarkChambersConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or DarkChambersConstants()
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(GAME_H, GAME_W),  # (height, width)
                channels=3
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Load sprites using the asset system
        final_asset_config = list(_get_default_asset_config())
        
        # Create procedural sprites for colors not in the main sprites (enemies, items, UI, etc.)
        def create_color_sprite(color):
            """Make a 1×1 sprite for the given color (ensures it’s in the palette)."""
            rgba = (*color, 255)
            return jnp.array([[rgba]], dtype=jnp.uint8)
        
        # Add all game colors as procedural sprites to ensure they're in the palette
        color_sprites = {
            'background_color': create_color_sprite(self.consts.BACKGROUND_COLOR),
            'zombie_color': create_color_sprite(self.consts.ZOMBIE_COLOR),
            'wraith_color': create_color_sprite(self.consts.WRAITH_COLOR),
            'skeleton_color': create_color_sprite(self.consts.SKELETON_COLOR),
            'wizard_color': create_color_sprite(self.consts.WIZARD_COLOR),
            'grim_reaper_color': create_color_sprite(self.consts.GRIM_REAPER_COLOR),
            'wall_color': create_color_sprite(self.consts.WALL_COLOR),
            'heart_color': create_color_sprite(self.consts.HEART_COLOR),
            'poison_color': create_color_sprite(self.consts.POISON_COLOR),
            'trap_color': create_color_sprite(self.consts.TRAP_COLOR),
            'treasure_color': create_color_sprite(self.consts.TREASURE_COLOR),
            'spawner_color': create_color_sprite(self.consts.SPAWNER_COLOR),
            'shield_color': create_color_sprite(self.consts.SHIELD_COLOR),
            'gun_color': create_color_sprite(self.consts.GUN_COLOR),
            'bomb_color': create_color_sprite(self.consts.BOMB_COLOR),
            'key_color': create_color_sprite(self.consts.KEY_COLOR),
            'ladder_up_color': create_color_sprite(self.consts.LADDER_UP_COLOR),
            'ladder_down_color': create_color_sprite(self.consts.LADDER_DOWN_COLOR),
            'ui_color': create_color_sprite(self.consts.UI_COLOR),
            'hud_color': create_color_sprite(self.consts.HUD_COLOR),
            'bullet_color': create_color_sprite(self.consts.BULLET_COLOR),
            'speed_potion_color': create_color_sprite(self.consts.SPEED_POTION_COLOR),
            'heal_potion_color': create_color_sprite(self.consts.HEAL_POTION_COLOR),
            'poison_potion_color': create_color_sprite(self.consts.POISON_POTION_COLOR),
            'potion_cork_color': create_color_sprite((150, 75, 0)),
            'potion_glass_color': create_color_sprite((200, 200, 200)),
            'hammer_handle_color': create_color_sprite((180, 100, 30)),
            'hammer_color': create_color_sprite(self.consts.HAMMER_COLOR),
            'checkpoint_color': create_color_sprite(self.consts.CHECKPOINT_COLOR),
            'checkpoint_pole_color': create_color_sprite((100, 100, 100)),
            'checkpoint_star_color': create_color_sprite((255, 255, 255)),
        }
        
        # Append procedural color sprites to asset config
        for name, sprite_data in color_sprites.items():
            final_asset_config.append({'name': name, 'type': 'procedural', 'data': sprite_data})
        
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/darkchambers"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        # Override background with solid color based on BACKGROUND_COLOR constant
        bg_color_id = self.COLOR_TO_ID.get(self.consts.BACKGROUND_COLOR)
        if bg_color_id is not None:
            target_h = self.config.downscale[0] if self.config.downscale else GAME_H
            target_w = self.config.downscale[1] if self.config.downscale else GAME_W
            self.BACKGROUND = jnp.full((target_h, target_w), bg_color_id, dtype=jnp.uint8)
        
        # Helper to scale palette-index masks to a target size (centered pad/crop)
        # Use transparent ID for padding so sprite bounding boxes never erase walls/background.
        def _scale_mask(mask: jnp.ndarray, target_h: int, target_w: int, align: str = 'center') -> jnp.ndarray:
            if mask is None:
                return None
            if mask.ndim == 2:
                h, w = int(mask.shape[0]), int(mask.shape[1])
                # Use uniform scale factor to preserve aspect ratio (prevents distortion)
                scale = max(1, min(target_w // w, target_h // h))
                scaled = jnp.repeat(jnp.repeat(mask, scale, axis=0), scale, axis=1)
                sh, sw = int(scaled.shape[0]), int(scaled.shape[1])
                pad_h = max(0, target_h - sh)
                pad_w = max(0, target_w - sw)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                if align == 'right':
                    pad_left, pad_right = pad_w, 0
                elif align == 'left':
                    pad_left, pad_right = 0, pad_w
                else:  # 'center'
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                # Transparent padding prevents sprite rectangles from overwriting walls
                pad_value = self.jr.TRANSPARENT_ID
                scaled_padded = jnp.pad(scaled, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=pad_value)
                final_h = int(scaled_padded.shape[0])
                final_w = int(scaled_padded.shape[1])
                ct = max(0, (final_h - target_h) // 2)
                cl = max(0, (final_w - target_w) // 2)
                cb = ct + target_h
                cr = cl + target_w
                return scaled_padded[ct:cb, cl:cr]
            elif mask.ndim == 3:
                # Assume (N, H, W) stack
                N = int(mask.shape[0])
                def _scale_one(m):
                    return _scale_mask(m, target_h, target_w, align)
                return jax.vmap(_scale_one, in_axes=0, out_axes=0)(mask)
            else:
                return mask

        # Player directional animation: (4, 5, H, W) — max 5 frames, pad shorter directions
        # Sprite index order: 0=right, 1=down, 2=left, 3=up
        target_w = int(self.consts.PLAYER_WIDTH)
        target_h = int(self.consts.PLAYER_HEIGHT)
        _zero_player = jnp.zeros((target_h, target_w), dtype=jnp.int32)
        player_dir_frame_names = [
            ["player_r0", "player_r1", "player_r2", "player_r2", "player_r2"],  # right: 3 frames, pad to 5
            ["player_d0", "player_d1", "player_d2", "player_d3", "player_d4"],  # down:  5 frames
            ["player_l0", "player_l1", "player_l2", "player_l2", "player_l2"],  # left:  3 frames, pad to 5
            ["player_u0", "player_u1", "player_u2", "player_u3", "player_u4"],  # up:    5 frames
        ]
        player_dirs = []
        for dir_names in player_dir_frame_names:
            dir_frames = [
                _scale_mask(self.SHAPE_MASKS[n], target_h, target_w)
                if self.SHAPE_MASKS.get(n) is not None else _zero_player
                for n in dir_names
            ]
            player_dirs.append(jnp.stack(dir_frames))  # (5, H, W)
        self.PLAYER_ANIM_FRAMES = jnp.stack(player_dirs)  # (4, 5, H, W)
        # Legacy aliases
        self.PLAYER_DIRECTIONAL_MASKS = None
        self.PLAYER_SCALED_MASK = player_dirs[0][0]

        # Scale enemy sprites to match gameplay size (10×20) and stack 4 directional frames.
        # Sprite order per enemy: 0=right, 1=down, 2=left, 3=up
        target_enemy_w = int(self.consts.ENEMY_WIDTH)
        target_enemy_h = int(self.consts.ENEMY_HEIGHT)

        self.ENEMY_SCALED_MASKS = {}        # legacy single-sprite dict (frame 0)
        self.ENEMY_DIRECTIONAL_MASKS = {}   # (4, H, W) stacked dict; None for enemies with custom animation

        # All enemy types use custom multi-frame animation — no generic 4-dir path needed

        # Ghost (wraith) has custom multi-frame animation per direction.
        # Build GHOST_ANIM_FRAMES: shape (4, 3, H, W)
        # Direction order: 0=right, 1=down, 2=left, 3=up
        # Frame counts:    right=2, down=3, left=2, up=3  (padded to max 3 with last frame)
        _zero_frame = jnp.zeros((target_enemy_h, target_enemy_w), dtype=jnp.int32)
        ghost_dir_frame_names = [
            ["ghost_r1", "ghost_r0", "ghost_r0"],  # right: reversed (cx 3.31→3.71), pad 3rd
            ["ghost_d0", "ghost_d1", "ghost_d2"],  # down:  3 frames
            ["ghost_l0", "ghost_l1", "ghost_l1"],  # left:  2 frames, pad 3rd
            ["ghost_u0", "ghost_u1", "ghost_u2"],  # up:    3 frames
        ]
        _DIR_ALIGN = ['right', 'center', 'left', 'center']
        ghost_dirs = []
        for d_idx, dir_names in enumerate(ghost_dir_frame_names):
            dir_frames = []
            for name in dir_names:
                m = self.SHAPE_MASKS.get(name)
                dir_frames.append(_scale_mask(m, target_enemy_h, target_enemy_w, _DIR_ALIGN[d_idx]) if m is not None else _zero_frame)
            ghost_dirs.append(jnp.stack(dir_frames))  # (3, H, W)
        self.GHOST_ANIM_FRAMES = jnp.stack(ghost_dirs)  # (4, 3, H, W)

        # Ghost does not use the generic directional mask path
        self.ENEMY_DIRECTIONAL_MASKS["wraith"] = None
        self.ENEMY_SCALED_MASKS["wraith"] = None

        # Wizard directional animation: (4, 5, H, W) — max 5 frames, pad shorter directions
        # Direction order: 0=right, 1=down, 2=left, 3=up
        wizard_dir_frame_names = [
            ["wizard_r1", "wizard_r0", "wizard_r1", "wizard_r2", "wizard_r0", "wizard_r1", "wizard_r0", "wizard_r1"],  # right: 2,1,2,3,1,2,1,2
            ["wizard_d1", "wizard_d0", "wizard_d3", "wizard_d0", "wizard_d1", "wizard_d2", "wizard_d4", "wizard_d2"],  # down:  5,4,7,4,5,6,8,6
            ["wizard_l0", "wizard_l1", "wizard_l0", "wizard_l2", "wizard_l0", "wizard_l1", "wizard_l0", "wizard_l0"],  # left:  14,15,14,16,14,15,14 (7 frames, pad to 8)
            ["wizard_u0", "wizard_u1", "wizard_u2", "wizard_u1", "wizard_u0", "wizard_u3", "wizard_u4", "wizard_u3"],  # up:    9,10,11,10,9,12,13,12
        ]
        wizard_dirs = []
        for d_idx, dir_names in enumerate(wizard_dir_frame_names):
            dir_frames = []
            for name in dir_names:
                m = self.SHAPE_MASKS.get(name)
                dir_frames.append(_scale_mask(m, target_enemy_h, target_enemy_w, _DIR_ALIGN[d_idx]) if m is not None else _zero_frame)
            wizard_dirs.append(jnp.stack(dir_frames))  # (8, H, W)
        self.WIZARD_ANIM_FRAMES = jnp.stack(wizard_dirs)  # (4, 8, H, W)

        # Wizard does not use the generic directional mask path
        self.ENEMY_DIRECTIONAL_MASKS["wizard"] = None
        self.ENEMY_SCALED_MASKS["wizard"] = None

        # Skeleton directional animation: (4, 4, H, W) — right/left=3 frames padded to 4, down/up=4 frames
        # Direction order: 0=right, 1=down, 2=left, 3=up
        skel_dir_frame_names = [
            ["skel_r0", "skel_r1", "skel_r2", "skel_r2"],          # right: 3 frames, pad to 4
            ["skel_7",  "skel_8",  "skel_9",  "skel_8"],           # down:  4-frame cycle
            ["skel_l2", "skel_l0", "skel_l1", "skel_l1"],          # left:  3 frames, pad to 4
            ["skel_10", "skel_12", "skel_11", "skel_12"],           # up:    4-frame cycle
        ]
        skel_dirs = []
        for d_idx, dir_names in enumerate(skel_dir_frame_names):
            dir_frames = [
                _scale_mask(self.SHAPE_MASKS[n], target_enemy_h, target_enemy_w, _DIR_ALIGN[d_idx])
                if self.SHAPE_MASKS.get(n) is not None else _zero_frame
                for n in dir_names
            ]
            skel_dirs.append(jnp.stack(dir_frames))  # (4, H, W)
        self.SKELETON_ANIM_FRAMES = jnp.stack(skel_dirs)  # (4, 4, H, W)

        # Skeleton does not use the generic directional mask path
        self.ENEMY_DIRECTIONAL_MASKS["skeleton"] = None
        self.ENEMY_SCALED_MASKS["skeleton"] = None

        # Grim Reaper: skeleton frames, head removed, recolored blue
        _HEAD_ROWS = 11  # top 11 of 28 px = head
        _blue_id = self.COLOR_TO_ID[self.consts.SHIELD_COLOR]
        grim_dir_frame_names = [
            ["skel_r0", "skel_r1", "skel_r2", "skel_r2"],
            ["skel_7",  "skel_8",  "skel_9",  "skel_8"],
            ["skel_l2", "skel_l0", "skel_l1", "skel_l1"],
            ["skel_10", "skel_12", "skel_11", "skel_12"],
        ]
        grim_dirs = []
        for d_idx, dir_names in enumerate(grim_dir_frame_names):
            dir_frames = []
            for n in dir_names:
                if self.SHAPE_MASKS.get(n) is not None:
                    frame = _scale_mask(self.SHAPE_MASKS[n], target_enemy_h, target_enemy_w, _DIR_ALIGN[d_idx])
                    frame = frame.at[:_HEAD_ROWS, :].set(self.jr.TRANSPARENT_ID)  # remove head
                    frame = jnp.where(frame != self.jr.TRANSPARENT_ID, _blue_id, frame)  # recolor blue
                else:
                    frame = _zero_frame
                dir_frames.append(frame)
            grim_dirs.append(jnp.stack(dir_frames))  # (4, H, W)
        self.GRIM_REAPER_ANIM_FRAMES = jnp.stack(grim_dirs)  # (4, 4, H, W)

        # Zombie directional animation: (4, 8, H, W) — max 8 frames, pad shorter directions
        # Direction order: 0=right, 1=down, 2=left, 3=up
        zombie_dir_frame_names = [
            ["green_r2", "green_r1", "green_r0", "green_r0", "green_r0", "green_r0", "green_r0", "green_r0"],  # right: reversed (cx↑), pad to 8
            ["green_6",  "green_7",   "green_6",  "green_d_d", "green_dn_l", "green_8",  "green_dn_l", "green_d_d"],  # down:  8-frame cycle
            ["green_l2", "green_l0",  "green_l1", "green_l1",  "green_l1",   "green_l1", "green_l1",   "green_l1"],  # left:  cx↓ order, pad to 8
            ["green_10", "green_9",   "green_10", "green_u_d", "green_u_r1", "green_u_r2","green_u_r1", "green_u_d"], # up:    8-frame cycle
        ]
        zombie_dirs = []
        for d_idx, dir_names in enumerate(zombie_dir_frame_names):
            dir_frames = [
                _scale_mask(self.SHAPE_MASKS[n], target_enemy_h, target_enemy_w, _DIR_ALIGN[d_idx])
                if self.SHAPE_MASKS.get(n) is not None else _zero_frame
                for n in dir_names
            ]
            zombie_dirs.append(jnp.stack(dir_frames))  # (5, H, W)
        self.ZOMBIE_ANIM_FRAMES = jnp.stack(zombie_dirs)  # (4, 8, H, W)

        # Zombie does not use the generic directional mask path
        self.ENEMY_DIRECTIONAL_MASKS["zombie"] = None
        self.ENEMY_SCALED_MASKS["zombie"] = None


        # Scale item sprites to 12×12 boxes for consistent rendering
        target_item_w = 12
        target_item_h = 12
        self.ITEM_SCALED_MASKS = {}
        item_sprites = {
            "pot": "pot",          # ITEM_SHIELD (shield icon)
            "skull": "skull",      # ITEM_POISON
            "trapdoor": "trapdoor", # ITEM_TRAP
            "apple": "apple",      # ITEM_HEART
            "barrel": "barrel",    # ITEM_BOMB
            "candle": "candle",    # ITEM_AMBER_CHALICE
            "chain": "chain",      # ITEM_AMULET
            "key": "key",         # ITEM_KEY
            "door": "door",       # ITEM_CAGE_DOOR
            "pistol": "pistol",   # ITEM_GUN
        }
        for item_key, sprite_name in item_sprites.items():
            item_mask = self.SHAPE_MASKS.get(sprite_name)
            if item_mask is not None:
                self.ITEM_SCALED_MASKS[item_key] = _scale_mask(item_mask, target_item_h, target_item_w)
            else:
                self.ITEM_SCALED_MASKS[item_key] = None

        # Stairs sprite uses natural dimensions (LADDER_HEIGHT tall × LADDER_WIDTH wide)
        stairs_mask = self.SHAPE_MASKS.get("stairs")
        if stairs_mask is not None:
            self.ITEM_SCALED_MASKS["stairs"] = _scale_mask(stairs_mask, LADDER_HEIGHT, LADDER_WIDTH)
            self.ITEM_SCALED_MASKS["stairs_down"] = _scale_mask(stairs_mask, LADDER_HEIGHT, LADDER_WIDTH)
        else:
            self.ITEM_SCALED_MASKS["stairs"] = None
            self.ITEM_SCALED_MASKS["stairs_down"] = None

        # Pistol sprite pre-scaled for HUD indicator (ITEM_WIDTH × ITEM_HEIGHT)
        pistol_mask = self.SHAPE_MASKS.get("pistol")
        if pistol_mask is not None:
            self.GUN_HUD_SPRITE = _scale_mask(pistol_mask, ITEM_HEIGHT, ITEM_WIDTH)
        else:
            self.GUN_HUD_SPRITE = None

        # --- Spawner as skull: scale skull sprite to 14x28 for spawner (2:1 aspect, original width) ---
        skull_mask = self.SHAPE_MASKS.get("skull")
        if skull_mask is not None:
            self.SPAWNER_SKULL_MASK = _scale_mask(skull_mask, SPAWNER_HEIGHT, SPAWNER_WIDTH)
        else:
            self.SPAWNER_SKULL_MASK = None

        # Grim Reaper has no sprite, will use colored box
        self.HEART_MASK_6 = None
        self.POISON_MASK_6 = None
        self.TREASURE_MASKS = {}

        # Create color ID mapping for easy access
        self.WALL_ID = self.COLOR_TO_ID[self.consts.WALL_COLOR]
        self.HEART_ID = self.COLOR_TO_ID[self.consts.HEART_COLOR]
        self.POISON_ID = self.COLOR_TO_ID[self.consts.POISON_COLOR]
        self.TRAP_ID = self.COLOR_TO_ID[self.consts.TRAP_COLOR]
        self.TREASURE_ID = self.COLOR_TO_ID[self.consts.TREASURE_COLOR]
        self.SPAWNER_ID = self.COLOR_TO_ID[self.consts.SPAWNER_COLOR]
        self.SHIELD_ID = self.COLOR_TO_ID[self.consts.SHIELD_COLOR]
        self.GUN_ID = self.COLOR_TO_ID[self.consts.GUN_COLOR]
        self.BOMB_ID = self.COLOR_TO_ID[self.consts.BOMB_COLOR]
        self.KEY_ID = self.COLOR_TO_ID[self.consts.KEY_COLOR]
        self.LADDER_UP_ID = self.COLOR_TO_ID[self.consts.LADDER_UP_COLOR]
        self.LADDER_DOWN_ID = self.COLOR_TO_ID[self.consts.LADDER_DOWN_COLOR]
        self.UI_ID = self.COLOR_TO_ID[self.consts.UI_COLOR]
        self.HUD_ID = self.COLOR_TO_ID[self.consts.HUD_COLOR]
        self.BULLET_ID = self.COLOR_TO_ID[self.consts.BULLET_COLOR]
        self.BULLET_SPRITE = self.SHAPE_MASKS.get("bullet")
        self.ZOMBIE_ID = self.COLOR_TO_ID[self.consts.ZOMBIE_COLOR]
        self.WRAITH_ID = self.COLOR_TO_ID[self.consts.WRAITH_COLOR]
        self.SKELETON_ID = self.COLOR_TO_ID[self.consts.SKELETON_COLOR]
        self.WIZARD_ID = self.COLOR_TO_ID[self.consts.WIZARD_COLOR]
        self.GRIM_REAPER_ID = self.COLOR_TO_ID[self.consts.GRIM_REAPER_COLOR]
        self.HAMMER_ID = self.COLOR_TO_ID[self.consts.HAMMER_COLOR]
        self.CHECKPOINT_ID = self.COLOR_TO_ID[self.consts.CHECKPOINT_COLOR]
        self.CHECKPOINT_ACTIVATED_ID = self.SHIELD_ID  # bright blue when activated
        self.CHECKPOINT_POLE_ID = self.COLOR_TO_ID[(100, 100, 100)]
        self.CHECKPOINT_STAR_ID = self.COLOR_TO_ID[(255, 255, 255)]
        self.POTION_CORK_ID  = self.COLOR_TO_ID[(150, 75, 0)]
        self.POTION_GLASS_ID = self.COLOR_TO_ID[(200, 200, 200)]

        # Checkpoint flag sprites (10 wide × 16 tall): tapered pennant on a pole with base
        _P = int(self.CHECKPOINT_POLE_ID)
        _T = int(self.jr.TRANSPARENT_ID)

        def _make_flag(flag_id, star_id):
            _F, _S = int(flag_id), int(star_id)
            return jnp.array([
                [_T, _P, _F, _F, _F, _F, _F, _F, _T, _T],
                [_T, _P, _F, _F, _F, _F, _F, _F, _F, _T],
                [_T, _P, _F, _F, _S, _F, _F, _F, _T, _T],
                [_T, _P, _F, _S, _S, _S, _F, _T, _T, _T],
                [_T, _P, _F, _F, _S, _F, _T, _T, _T, _T],
                [_T, _P, _F, _F, _T, _T, _T, _T, _T, _T],
                [_T, _P, _T, _T, _T, _T, _T, _T, _T, _T],
                [_T, _P, _T, _T, _T, _T, _T, _T, _T, _T],
                [_T, _P, _T, _T, _T, _T, _T, _T, _T, _T],
                [_T, _P, _T, _T, _T, _T, _T, _T, _T, _T],
                [_T, _P, _T, _T, _T, _T, _T, _T, _T, _T],
                [_T, _P, _T, _T, _T, _T, _T, _T, _T, _T],
                [_T, _P, _T, _T, _T, _T, _T, _T, _T, _T],
                [_T, _P, _T, _T, _T, _T, _T, _T, _T, _T],
                [_P, _P, _P, _T, _T, _T, _T, _T, _T, _T],
                [_T, _T, _T, _T, _T, _T, _T, _T, _T, _T],
            ], dtype=jnp.int32)

        # Inactive: gold flag, no star (star pixels = flag color)
        self.CHECKPOINT_SPRITE = _make_flag(self.CHECKPOINT_ID, self.CHECKPOINT_ID)
        # Activated: blue flag, white star
        self.CHECKPOINT_SPRITE_ACTIVATED = _make_flag(self.CHECKPOINT_ACTIVATED_ID, self.CHECKPOINT_STAR_ID)

        # Potion bottle sprites (8×8): cork top, glass outline, colored liquid inside
        _T  = int(self.jr.TRANSPARENT_ID)
        _C  = int(self.POTION_CORK_ID)
        _G  = int(self.POTION_GLASS_ID)
        def _make_potion_sprite(liquid_color_id):
            _L = int(liquid_color_id)
            return jnp.array([
                [_T, _T, _C, _C, _C, _T, _T, _T],
                [_T, _T, _G, _G, _G, _T, _T, _T],
                [_T, _G, _L, _L, _L, _G, _T, _T],
                [_T, _G, _L, _L, _L, _G, _T, _T],
                [_T, _G, _L, _L, _L, _G, _T, _T],
                [_T, _G, _L, _L, _L, _G, _T, _T],
                [_T, _G, _G, _G, _G, _G, _T, _T],
                [_T, _T, _T, _T, _T, _T, _T, _T],
            ], dtype=jnp.int32)
        self.SPEED_POTION_SPRITE  = _make_potion_sprite(self.COLOR_TO_ID[self.consts.SPEED_POTION_COLOR])
        self.HEAL_POTION_SPRITE   = _make_potion_sprite(self.COLOR_TO_ID[self.consts.HEAL_POTION_COLOR])
        self.POISON_POTION_SPRITE = _make_potion_sprite(self.COLOR_TO_ID[self.consts.POISON_POTION_COLOR])

        # Hammer sprite (8×10): full-width T-shaped head + narrow centered handle
        _H  = int(self.HAMMER_ID)                    # head color (purple/violet)
        _HW = int(self.COLOR_TO_ID[(180, 100, 30)])  # handle wood (brown)
        self.HAMMER_SPRITE = jnp.array([
            [_T, _H, _H, _H, _H, _H, _H, _T],
            [_H, _H, _H, _H, _H, _H, _H, _H],
            [_H, _H, _H, _H, _H, _H, _H, _H],
            [_T, _H, _H, _H, _H, _H, _H, _T],
            [_T, _T, _T, _HW, _HW, _T, _T, _T],
            [_T, _T, _T, _HW, _HW, _T, _T, _T],
            [_T, _T, _T, _HW, _HW, _T, _T, _T],
            [_T, _T, _T, _HW, _HW, _T, _T, _T],
            [_T, _T, _T, _HW, _HW, _T, _T, _T],
            [_T, _T, _T, _T, _T, _T, _T, _T],
        ], dtype=jnp.int32)

        # Digit patterns (0-9) 3x5 bitmap (rows top->bottom, cols left->right)
        # 1 = pixel on, 0 = off
        self.DIGIT_PATTERNS = jnp.array([
            # 0
            [[1,1,1],
             [1,0,1],
             [1,0,1],
             [1,0,1],
             [1,1,1]],
            # 1
            [[0,1,0],
             [1,1,0],
             [0,1,0],
             [0,1,0],
             [1,1,1]],
            # 2
            [[1,1,1],
             [0,0,1],
             [1,1,1],
             [1,0,0],
             [1,1,1]],
            # 3
            [[1,1,1],
             [0,0,1],
             [1,1,1],
             [0,0,1],
             [1,1,1]],
            # 4
            [[1,0,1],
             [1,0,1],
             [1,1,1],
             [0,0,1],
             [0,0,1]],
            # 5
            [[1,1,1],
             [1,0,0],
             [1,1,1],
             [0,0,1],
             [1,1,1]],
            # 6
            [[1,1,1],
             [1,0,0],
             [1,1,1],
             [1,0,1],
             [1,1,1]],
            # 7
            [[1,1,1],
             [0,0,1],
             [0,1,0],
             [0,1,0],
             [0,1,0]],
            # 8
            [[1,1,1],
             [1,0,1],
             [1,1,1],
             [1,0,1],
             [1,1,1]],
            # 9
            [[1,1,1],
             [1,0,1],
             [1,1,1],
             [0,0,1],
             [1,1,1]],
        ], dtype=jnp.uint8)
        
        # Walls in world coordinates - format: [x, y, width, height]
        # Multiple levels with different layouts
        # Shape: (MAX_LEVELS, max_walls_per_level, 4)
        # Define 3 portal holes per side (top, middle, bottom)
        portal_hole_height = 40
        portal_gap = (self.consts.WORLD_HEIGHT - 3 * portal_hole_height) // 4
        portal_y_starts = [
            portal_gap,
            portal_gap * 2 + portal_hole_height,
            portal_gap * 3 + portal_hole_height * 2
        ]
        portal_y_ends = [y + portal_hole_height for y in portal_y_starts]
        
        # Middle map (map_index=0) - original layout, adjusted for new portals and height
        # Borders
        border_top = [0, 0, self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS]
        border_bottom = [0, self.consts.WORLD_HEIGHT - self.consts.WALL_THICKNESS, self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS]
        # Left wall - 3 gaps for portals
        left_walls = []
        prev_end = 0
        for y_start, y_end in zip(portal_y_starts, portal_y_ends):
            left_walls.append([0, prev_end, self.consts.WALL_THICKNESS, y_start - prev_end])
            prev_end = y_end
        left_walls.append([0, prev_end, self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT - prev_end])
        # Right wall - 3 gaps for portals
        right_walls = []
        prev_end = 0
        for y_start, y_end in zip(portal_y_starts, portal_y_ends):
            right_walls.append([self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, prev_end, self.consts.WALL_THICKNESS, y_start - prev_end])
            prev_end = y_end
        right_walls.append([self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, prev_end, self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT - prev_end])
        
        
        #Coordinate System
        #Origin (0, 0): Top-left corner of the world
        #X-axis: → (increases rightward)
        #Y-axis: ↓ (increases downward)
        #World size: 160 × 600 pixels

        #[x, y, width, height]
        #↓  ↓    ↓      ↓
        #│  │    │      └─ height in pixels (vertical)
        #│  │    └──────── width in pixels (horizontal)
        #│  └───────────── top-left Y coordinate (pixels from top)
        #└──────────────── top-left X coordinate (pixels from left)
        
        
        maze = [

            #chamber enclosing
            [48, 0, 6, 162],

            [0, 160, 54, 6],
            #ornament
            [8, 166, 8, 8],
            
            #chamber enclosing (mirror symmetric to left)
            [106, 0, 6, 162],
            
            [106, 160, 54, 6],
            #ornament (mirror)
            [144, 166, 8, 8],

            
            # ───────── MIDDLE DIVIDER (FLIPPED T) ─────────
            # Vertical wall down the middle
            [77, 0, 6, 300],
            # Single centered horizontal bar; leaves walkable gaps on both sides
            [40, 300, 80, 12],

            # ───────── BOTTOM LONG VERTICALS (SYMMETRIC) ─────────
            [48, 360, 6, 234],
            [106, 360, 6, 234]
        ]

        # Secondary maze set (v2) - starts as a copy and can be edited independently later.
        maze_2 = [

            #chamber enclosing
            [48, 0, 6, 162],

            [0, 160, 54, 6],
            [6, 166, 6, 6],
            
            #chamber enclosing (mirror symmetric to left)
            [106, 0, 6, 162],
            
            [106, 160, 54, 6],
            [148, 166, 6, 6],

            # long vertical maze pillars (v2)
            [30, 190, 6, 404],
            [66, 210, 6, 384],
            [88, 180, 6, 414],
            [124, 200, 6, 394]

        ]


        middle_level_0_walls = jnp.array(maze, dtype=jnp.int32)
        middle_level_1_walls = jnp.array(maze_2, dtype=jnp.int32)

        right_maze = [
            # top and bottom caps
            [0, 0, 160, 6],
            [0, 594, 160, 6],

            # upper split ledge (open center lane)
            [0, 208, 56, 6],
            [104, 208, 56, 6],

            # middle-right portal enclosure (above/below portal opening, no direct block)
            [110, 274, 50, 6],
            [110, 320, 50, 6],

            # bottom-right portal enclosure (above/below portal opening, no direct block)
            [110, 434, 50, 6],
            [110, 480, 50, 6],

            # lower-third full-width bar (placed above bottom portal opening)
            [0, 420, 160, 6],

            # lower split ledge
            [0, 520, 46, 6],
            [114, 520, 46, 6]
        ]

        # Secondary right maze set (v2) - copy of v1 for now.
        right_maze_2 = [
            #chamber enclosing (mirror symmetric to left)
            [106, 0, 6, 162],
            
            [106, 160, 54, 6],

            # long vertical maze pillars (v2)
            [30, 200, 6, 394],
            [58, 170, 6, 424],
            [82, 220, 6, 374],
            [110, 190, 6, 404]



        ]


        left_maze = [

            # top and bottom caps
            [0, 0, 160, 6],
            [0, 594, 160, 6],

            #chamber enclosing
            [48, 0, 6, 162],
            [0, 160, 54, 6],
            [6, 166, 12, 6],

            # upper split ledge (open center lane)
            [0, 208, 56, 6],
            [104, 208, 56, 6],





            # middle-left portal enclosure (above/below portal opening, no direct block)
            [0, 274, 50, 6],


            # seperator
            [0, 320, 300, 6],
            
            # vertical seperator
            [77, 0, 6, 350],


            # bottom-left portal enclosure (above/below portal opening, no direct block)
            [0, 434, 50, 6],
            [0, 480, 50, 6],

            # lower split ledge
            [0, 520, 46, 6],
            [114, 520, 46, 6]
        ]

        # Secondary left maze set (v2) - copy of v1 for now.
        left_maze_2 = [
            #chamber enclosing
            [48, 0, 6, 162],

            [0, 160, 54, 6],
            [6, 166, 6, 6],

            # long vertical maze pillars (v2)
            [34, 210, 6, 384],
            [62, 180, 6, 414],
            [92, 200, 6, 394],
            [118, 170, 6, 424]



        ]

        left_level_0_walls = jnp.array(left_maze, dtype=jnp.int32)
        left_level_1_walls = jnp.array(left_maze_2, dtype=jnp.int32)
        
        # Right map (map_index=2) - horizontal chambers, adjusted for portals/height
        right_level_0_walls = jnp.array(right_maze, dtype=jnp.int32)
        right_level_1_walls = jnp.array(right_maze_2, dtype=jnp.int32)




    # Middle map level 1 uses v2 walls
        
        #jnp.array([
        #    [50, 120, 100, 6],
        #    [180, 220, 100, 6],
        #    [100, 320, 120, 6],
        #    [60, 420, 100, 6],
        #    [81, 540, 6, 60],
        #], dtype=jnp.int32)
        
        # Left map level 1 uses v2 walls
        
        #jnp.array([
        #    [60, 90, 110, 6],
        #    [70, 170, 6, 90],
        #    [140, 250, 100, 6],
        #    [190, 320, 6, 80],
        #    [50, 440, 120, 6],
        #], dtype=jnp.int32)
        
        # Right map level 1 uses v2 walls
        
        #jnp.array([
        #    [80, 110, 120, 6],
        #    [160, 180, 6, 90],
        #    [40, 300, 130, 6],
        #    [60, 370, 6, 90],
        #    [170, 420, 90, 6],
        #], dtype=jnp.int32)

        # --- Hard-coded cage (slightly smaller than before) ---
        self.CAGE_INTERIOR_CELLS = 3
        self.CAGE_WALL_THICKNESS = 4
        self.CAGE_DOOR_GAP = 16  # wider opening for smooth in/out
        self.CAGE_DOOR_SIZE = 12  # independent of ladder sprite size
        cage_interior_size = self.CAGE_INTERIOR_CELLS * CELL_SIZE
        cage_outer_size = cage_interior_size + 2 * self.CAGE_WALL_THICKNESS
        self.CAGE_OUTER_SIZE = cage_outer_size
        self.CAGE_REWARD_TYPE = ITEM_AMULET
        reward_size = jnp.array([11, 11], dtype=jnp.int32)
        offscreen_pos = jnp.array([-1000, -1000], dtype=jnp.int32)

        # Boundary segments used for cage-fit validation (same portal gaps as the map border)
        boundary_walls_for_fit = jnp.array([
            [0, 0, self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS],
            [0, self.consts.WORLD_HEIGHT - self.consts.WALL_THICKNESS, self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS],
            [0, 0, self.consts.WALL_THICKNESS, portal_y_starts[0]],
            [0, portal_y_ends[0], self.consts.WALL_THICKNESS, portal_y_starts[1] - portal_y_ends[0]],
            [0, portal_y_ends[1], self.consts.WALL_THICKNESS, portal_y_starts[2] - portal_y_ends[1]],
            [0, portal_y_ends[2], self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT - portal_y_ends[2]],
            [self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, 0, self.consts.WALL_THICKNESS, portal_y_starts[0]],
            [self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, portal_y_ends[0], self.consts.WALL_THICKNESS, portal_y_starts[1] - portal_y_ends[0]],
            [self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, portal_y_ends[1], self.consts.WALL_THICKNESS, portal_y_starts[2] - portal_y_ends[1]],
            [self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, portal_y_ends[2], self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT - portal_y_ends[2]],
        ], dtype=jnp.int32)

        def rect_overlaps_any(rect, walls):
            rx, ry, rw, rh = rect[0], rect[1], rect[2], rect[3]
            wx = walls[:, 0]
            wy = walls[:, 1]
            ww = walls[:, 2]
            wh = walls[:, 3]
            overlap_x = (rx <= (wx + ww - 1)) & ((rx + rw - 1) >= wx)
            overlap_y = (ry <= (wy + wh - 1)) & ((ry + rh - 1) >= wy)
            return jnp.any(overlap_x & overlap_y)

        def build_cage(origin):
            cx, cy = origin
            gap_side = (cage_outer_size - self.CAGE_DOOR_GAP) // 2

            top = jnp.array([cx, cy, cage_outer_size, self.CAGE_WALL_THICKNESS], dtype=jnp.int32)
            bottom_left = jnp.array([cx, cy + cage_outer_size - self.CAGE_WALL_THICKNESS, gap_side, self.CAGE_WALL_THICKNESS], dtype=jnp.int32)
            bottom_right = jnp.array([
                cx + gap_side + self.CAGE_DOOR_GAP,
                cy + cage_outer_size - self.CAGE_WALL_THICKNESS,
                gap_side,
                self.CAGE_WALL_THICKNESS
            ], dtype=jnp.int32)
            left = jnp.array([cx, cy, self.CAGE_WALL_THICKNESS, cage_outer_size], dtype=jnp.int32)
            right = jnp.array([cx + cage_outer_size - self.CAGE_WALL_THICKNESS, cy, self.CAGE_WALL_THICKNESS, cage_outer_size], dtype=jnp.int32)
            cage_walls = jnp.stack([top, bottom_left, bottom_right, left, right], axis=0)

            # Door is centered at the bottom opening
            door_x = cx + gap_side + (self.CAGE_DOOR_GAP - self.CAGE_DOOR_SIZE) // 2
            door_y = cy + cage_outer_size - self.CAGE_DOOR_SIZE
            door_pos = jnp.array([door_x, door_y], dtype=jnp.int32)

            # Reward is centered inside the cage interior
            reward_x = cx + self.CAGE_WALL_THICKNESS + (cage_interior_size - reward_size[0]) // 2
            reward_y = cy + self.CAGE_WALL_THICKNESS + (cage_interior_size - reward_size[1]) // 2
            reward_pos = jnp.array([reward_x, reward_y], dtype=jnp.int32)

            # Entry position (where player teleports) is centered inside
            entry_x = cx + self.CAGE_WALL_THICKNESS + (cage_interior_size - self.consts.PLAYER_WIDTH) // 2
            entry_y = cy + self.CAGE_WALL_THICKNESS + (cage_interior_size - self.consts.PLAYER_HEIGHT) // 2
            entry_pos = jnp.array([entry_x, entry_y], dtype=jnp.int32)
            return cage_walls, door_pos, reward_pos, entry_pos

        def try_cage_at_position(level_walls, origin):
            """Try to place cage at specific position, return validity and components."""
            cage_walls, door_pos, reward_pos, entry_pos = build_cage(origin)
            overlaps_level = jnp.any(jax.vmap(lambda rect: rect_overlaps_any(rect, level_walls))(cage_walls))
            overlaps_boundary = jnp.any(jax.vmap(lambda rect: rect_overlaps_any(rect, boundary_walls_for_fit))(cage_walls))
            cage_valid = ~(overlaps_level | overlaps_boundary)
            return cage_valid, cage_walls, door_pos, reward_pos, entry_pos

        def generate_cage_candidates():
            """Generate 1000+ candidate positions covering the playable area.
            Prioritizes upper regions but covers entire space."""
            candidates = []
            top_side_candidates = []
            # Grid spacing for systematic coverage
            x_step = 8
            y_step = 8
            margin = 20  # Stay away from edges

            # Strongly prioritize top side-channel placement first.
            side_margin = 12
            left_x = side_margin
            right_x = self.consts.WORLD_WIDTH - cage_outer_size - side_margin
            for y in [24, 40, 56, 72, 88, 104, 120, 136]:
                top_side_candidates.append(jnp.array([left_x, y], dtype=jnp.int32))
                top_side_candidates.append(jnp.array([right_x, y], dtype=jnp.int32))
            
            # Cover entire playable area with grid
            for y in range(margin, self.consts.WORLD_HEIGHT - cage_outer_size - margin, y_step):
                for x in range(margin, self.consts.WORLD_WIDTH - cage_outer_size - margin, x_step):
                    candidates.append(jnp.array([x, y], dtype=jnp.int32))
            
            # Prioritize upper regions by adding them to front of list
            upper_candidates = []
            upper_y_limit = 250  # Upper half focus
            for y in range(margin, upper_y_limit, y_step):
                for x in range(margin, self.consts.WORLD_WIDTH - cage_outer_size - margin, x_step):
                    upper_candidates.append(jnp.array([x, y], dtype=jnp.int32))
            
            # Return top side candidates first, then upper region, then remaining map.
            return top_side_candidates + upper_candidates + [c for c in candidates if c[1] >= upper_y_limit]

        def add_cage(level_walls, candidate_origins):
            """Try multiple positions until one works. candidate_origins: list of [x,y] positions."""
            # Try each candidate position
            best_walls = level_walls
            best_door = offscreen_pos
            best_reward = offscreen_pos
            best_entry = offscreen_pos
            found_valid = False
            
            for origin in candidate_origins:
                cage_valid, cage_walls, door_pos, reward_pos, entry_pos = try_cage_at_position(level_walls, origin)
                if cage_valid:
                    best_walls = jnp.concatenate([level_walls, cage_walls], axis=0)
                    best_door = door_pos
                    best_reward = reward_pos
                    best_entry = entry_pos
                    found_valid = True
                    break
            
            return best_walls, best_door, best_reward, best_entry, jnp.array(1 if found_valid else 0, dtype=jnp.int32)

        # Generate 1000+ candidate positions (systematic grid covering playable area)
        cage_candidates = generate_cage_candidates()
        
        # Add cages to all 3 map variants - each tries 1000+ positions
        middle_level_0_walls, level0_door_m, level0_reward_m, level0_entry_m, level0_valid_m = add_cage(middle_level_0_walls, cage_candidates)
        middle_level_1_walls, level1_door_m, level1_reward_m, level1_entry_m, level1_valid_m = add_cage(middle_level_1_walls, cage_candidates)
        
        left_level_0_walls, level0_door_l, level0_reward_l, level0_entry_l, level0_valid_l = add_cage(left_level_0_walls, cage_candidates)
        left_level_1_walls, level1_door_l, level1_reward_l, level1_entry_l, level1_valid_l = add_cage(left_level_1_walls, cage_candidates)
        
        right_level_0_walls, level0_door_r, level0_reward_r, level0_entry_r, level0_valid_r = add_cage(right_level_0_walls, cage_candidates)
        right_level_1_walls, level1_door_r, level1_reward_r, level1_entry_r, level1_valid_r = add_cage(right_level_1_walls, cage_candidates)

        # Alternate wall sets by level parity: even=v1, odd=v2
        middle_level_2_walls = middle_level_0_walls
        middle_level_3_walls = middle_level_1_walls
        middle_level_4_walls = middle_level_0_walls
        middle_level_5_walls = middle_level_1_walls
        middle_level_6_walls = middle_level_0_walls
        
        left_level_2_walls = left_level_0_walls
        left_level_3_walls = left_level_1_walls
        left_level_4_walls = left_level_0_walls
        left_level_5_walls = left_level_1_walls
        left_level_6_walls = left_level_0_walls
        
        right_level_2_walls = right_level_0_walls
        right_level_3_walls = right_level_1_walls
        right_level_4_walls = right_level_0_walls
        right_level_5_walls = right_level_1_walls
        right_level_6_walls = right_level_0_walls

        # Cage positions for all levels (levels 2-6 identical to level 1)
        # MIDDLE MAP cage positions
        level0_door_pos_m = level0_door_m
        level0_reward_pos_m = level0_reward_m
        level0_entry_pos_m = level0_entry_m

        level1_door_pos_m = level1_door_m
        level1_reward_pos_m = level1_reward_m
        level1_entry_pos_m = level1_entry_m

        level2_door_pos_m = level0_door_m
        level2_reward_pos_m = level0_reward_m
        level2_entry_pos_m = level0_entry_m

        level3_door_pos_m = level1_door_m
        level3_reward_pos_m = level1_reward_m
        level3_entry_pos_m = level1_entry_m

        level4_door_pos_m = level0_door_m
        level4_reward_pos_m = level0_reward_m
        level4_entry_pos_m = level0_entry_m

        level5_door_pos_m = level1_door_m
        level5_reward_pos_m = level1_reward_m
        level5_entry_pos_m = level1_entry_m

        level6_door_pos_m = level0_door_m
        level6_reward_pos_m = level0_reward_m
        level6_entry_pos_m = level0_entry_m
        
        # LEFT MAP cage positions
        level0_door_pos_l = level0_door_l
        level0_reward_pos_l = level0_reward_l
        level0_entry_pos_l = level0_entry_l

        level1_door_pos_l = level1_door_l
        level1_reward_pos_l = level1_reward_l
        level1_entry_pos_l = level1_entry_l

        level2_door_pos_l = level0_door_l
        level2_reward_pos_l = level0_reward_l
        level2_entry_pos_l = level0_entry_l

        level3_door_pos_l = level1_door_l
        level3_reward_pos_l = level1_reward_l
        level3_entry_pos_l = level1_entry_l

        level4_door_pos_l = level0_door_l
        level4_reward_pos_l = level0_reward_l
        level4_entry_pos_l = level0_entry_l

        level5_door_pos_l = level1_door_l
        level5_reward_pos_l = level1_reward_l
        level5_entry_pos_l = level1_entry_l

        level6_door_pos_l = level0_door_l
        level6_reward_pos_l = level0_reward_l
        level6_entry_pos_l = level0_entry_l
        
        # RIGHT MAP cage positions
        level0_door_pos_r = level0_door_r
        level0_reward_pos_r = level0_reward_r
        level0_entry_pos_r = level0_entry_r

        level1_door_pos_r = level1_door_r
        level1_reward_pos_r = level1_reward_r
        level1_entry_pos_r = level1_entry_r

        level2_door_pos_r = level0_door_r
        level2_reward_pos_r = level0_reward_r
        level2_entry_pos_r = level0_entry_r

        level3_door_pos_r = level1_door_r
        level3_reward_pos_r = level1_reward_r
        level3_entry_pos_r = level1_entry_r

        level4_door_pos_r = level0_door_r
        level4_reward_pos_r = level0_reward_r
        level4_entry_pos_r = level0_entry_r

        level5_door_pos_r = level1_door_r
        level5_reward_pos_r = level1_reward_r
        level5_entry_pos_r = level1_entry_r

        level6_door_pos_r = level0_door_r
        level6_reward_pos_r = level0_reward_r
        level6_entry_pos_r = level0_entry_r

        # Stack into 3D arrays: (3 maps, 7 levels, 2 coords)
        self.CAGE_DOOR_POSITIONS = jnp.stack([
            jnp.stack([level0_door_pos_m, level1_door_pos_m, level2_door_pos_m, level3_door_pos_m, level4_door_pos_m, level5_door_pos_m, level6_door_pos_m], axis=0),
            jnp.stack([level0_door_pos_l, level1_door_pos_l, level2_door_pos_l, level3_door_pos_l, level4_door_pos_l, level5_door_pos_l, level6_door_pos_l], axis=0),
            jnp.stack([level0_door_pos_r, level1_door_pos_r, level2_door_pos_r, level3_door_pos_r, level4_door_pos_r, level5_door_pos_r, level6_door_pos_r], axis=0),
        ], axis=0)

        self.CAGE_REWARD_POSITIONS = jnp.stack([
            jnp.stack([level0_reward_pos_m, level1_reward_pos_m, level2_reward_pos_m, level3_reward_pos_m, level4_reward_pos_m, level5_reward_pos_m, level6_reward_pos_m], axis=0),
            jnp.stack([level0_reward_pos_l, level1_reward_pos_l, level2_reward_pos_l, level3_reward_pos_l, level4_reward_pos_l, level5_reward_pos_l, level6_reward_pos_l], axis=0),
            jnp.stack([level0_reward_pos_r, level1_reward_pos_r, level2_reward_pos_r, level3_reward_pos_r, level4_reward_pos_r, level5_reward_pos_r, level6_reward_pos_r], axis=0),
        ], axis=0)

        self.CAGE_ENTRY_POSITIONS = jnp.stack([
            jnp.stack([level0_entry_pos_m, level1_entry_pos_m, level2_entry_pos_m, level3_entry_pos_m, level4_entry_pos_m, level5_entry_pos_m, level6_entry_pos_m], axis=0),
            jnp.stack([level0_entry_pos_l, level1_entry_pos_l, level2_entry_pos_l, level3_entry_pos_l, level4_entry_pos_l, level5_entry_pos_l, level6_entry_pos_l], axis=0),
            jnp.stack([level0_entry_pos_r, level1_entry_pos_r, level2_entry_pos_r, level3_entry_pos_r, level4_entry_pos_r, level5_entry_pos_r, level6_entry_pos_r], axis=0),
        ], axis=0)

        self.CAGE_VALID = jnp.stack([
            jnp.stack([level0_valid_m, level1_valid_m, level0_valid_m, level1_valid_m, level0_valid_m, level1_valid_m, level0_valid_m], axis=0),
            jnp.stack([level0_valid_l, level1_valid_l, level0_valid_l, level1_valid_l, level0_valid_l, level1_valid_l, level0_valid_l], axis=0),
            jnp.stack([level0_valid_r, level1_valid_r, level0_valid_r, level1_valid_r, level0_valid_r, level1_valid_r, level0_valid_r], axis=0),
        ], axis=0)
        
        # Stack levels into 3D array: shape (3 maps, MAX_LEVELS, num_walls, 4)
        # Map index 0 = middle, 1 = left, 2 = right
        # NOTE: levels/maps can have different wall counts while editing maps.
        # Pad to a shared row count before stacking to avoid shape mismatch errors.
        middle_levels = [
            middle_level_0_walls, middle_level_1_walls, middle_level_2_walls,
            middle_level_3_walls, middle_level_4_walls, middle_level_5_walls, middle_level_6_walls,
        ]
        left_levels = [
            left_level_0_walls, left_level_1_walls, left_level_2_walls,
            left_level_3_walls, left_level_4_walls, left_level_5_walls, left_level_6_walls,
        ]
        right_levels = [
            right_level_0_walls, right_level_1_walls, right_level_2_walls,
            right_level_3_walls, right_level_4_walls, right_level_5_walls, right_level_6_walls,
        ]

        all_levels = middle_levels + left_levels + right_levels
        max_walls = max(int(w.shape[0]) for w in all_levels)

        def _pad_walls(walls: jnp.ndarray) -> jnp.ndarray:
            pad_rows = max_walls - int(walls.shape[0])
            if pad_rows <= 0:
                return walls
            pad = jnp.zeros((pad_rows, 4), dtype=jnp.int32)
            return jnp.concatenate([walls, pad], axis=0)

        middle_walls_stack = jnp.stack([_pad_walls(w) for w in middle_levels], axis=0)
        left_walls_stack = jnp.stack([_pad_walls(w) for w in left_levels], axis=0)
        right_walls_stack = jnp.stack([_pad_walls(w) for w in right_levels], axis=0)
        
        self.LEVEL_WALLS = jnp.stack([middle_walls_stack, left_walls_stack, right_walls_stack], axis=0)
        
        # Default to middle map level 0 for compatibility
        self.WALLS = middle_level_0_walls

        def build_boundary_walls():
            portal_hole_height = 40
            portal_gap = (self.consts.WORLD_HEIGHT - 3 * portal_hole_height) // 4
            portal_y_starts = jnp.array([
                portal_gap,
                portal_gap * 2 + portal_hole_height,
                portal_gap * 3 + portal_hole_height * 2
            ], dtype=jnp.int32)
            portal_y_ends = portal_y_starts + portal_hole_height

            bt = jnp.array([0, 0, self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS], dtype=jnp.int32)
            bb = jnp.array([0, self.consts.WORLD_HEIGHT - self.consts.WALL_THICKNESS, self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS], dtype=jnp.int32)
            lw1 = jnp.array([0, 0, self.consts.WALL_THICKNESS, portal_y_starts[0]], dtype=jnp.int32)
            lw2 = jnp.array([0, portal_y_ends[0], self.consts.WALL_THICKNESS, portal_y_starts[1] - portal_y_ends[0]], dtype=jnp.int32)
            lw3 = jnp.array([0, portal_y_ends[1], self.consts.WALL_THICKNESS, portal_y_starts[2] - portal_y_ends[1]], dtype=jnp.int32)
            lw4 = jnp.array([0, portal_y_ends[2], self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT - portal_y_ends[2]], dtype=jnp.int32)
            rw1 = jnp.array([self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, 0, self.consts.WALL_THICKNESS, portal_y_starts[0]], dtype=jnp.int32)
            rw2 = jnp.array([self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, portal_y_ends[0], self.consts.WALL_THICKNESS, portal_y_starts[1] - portal_y_ends[0]], dtype=jnp.int32)
            rw3 = jnp.array([self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, portal_y_ends[1], self.consts.WALL_THICKNESS, portal_y_starts[2] - portal_y_ends[1]], dtype=jnp.int32)
            rw4 = jnp.array([self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, portal_y_ends[2], self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT - portal_y_ends[2]], dtype=jnp.int32)
            return jnp.stack([bt, bb, lw1, lw2, lw3, lw4, rw1, rw2, rw3, rw4], axis=0)

        # Shared boundary walls (with portal gaps) for rendering and collision
        self.BOUNDARY_WALLS = build_boundary_walls()

        # --- Navigation grid: mark nav cells that are blocked by walls ---
        def make_occupancy(walls):
            # walls: (num_walls, 4) = [x, y, w, h]
            walls = jnp.concatenate([walls, self.BOUNDARY_WALLS], axis=0)
            xs = jnp.arange(GRID_W) * CELL_SIZE
            ys = jnp.arange(GRID_H) * CELL_SIZE
            cx, cy = jnp.meshgrid(xs, ys)   # shape (GRID_H, GRID_W)

            # Treat each cell as a CELL_SIZE x CELL_SIZE rect (top-left at cx,cy)
            cx = cx[..., None]
            cy = cy[..., None]

            wx = walls[:, 0]
            wy = walls[:, 1]
            ww = walls[:, 2]
            wh = walls[:, 3]

            overlap_x = (cx <= (wx + ww - 1)) & ((cx + CELL_SIZE - 1) >= wx)
            overlap_y = (cy <= (wy + wh - 1)) & ((cy + CELL_SIZE - 1) >= wy)
            blocked = jnp.any(overlap_x & overlap_y, axis=-1)  # (GRID_H, GRID_W) bool
            return blocked

        # Shape: (3 maps, MAX_LEVELS, GRID_H, GRID_W)
        self.LEVEL_WALL_GRID = jnp.stack([
            jnp.stack([make_occupancy(self.LEVEL_WALLS[map_idx, level_idx]) 
                      for level_idx in range(MAX_LEVELS)], axis=0)
            for map_idx in range(3)
        ], axis=0)
        
        # Per-item sizes (width, height) indexed by item type code
        # Index 0 unused placeholder for alignment
        self.ITEM_TYPE_SIZES = jnp.array([
            [0, 0],                 # 0 (unused)
            [6, 6],                 # 1 HEART
            [6, 6],                 # 2 POISON
            [6, 6],                 # 3 TRAP
            [7, 7],                 # 4 STRONGBOX (small)
            [9, 9],                 # 5 AMBER CHALICE (medium)
            [11, 11],               # 6 AMULET (large)
            [13, 13],               # 7 GOLD CHALICE (largest)
            [6, 6],                 # 8 SHIELD
            [6, 6],                 # 9 GUN
            [6, 6],                 # 10 BOMB
            [6, 6],                 # 11 KEY (small key)
            [LADDER_WIDTH, LADDER_HEIGHT],  # 12 LADDER_UP (larger)
            [LADDER_WIDTH, LADDER_HEIGHT],  # 13 LADDER_DOWN (larger)
            [12, 12],                       # 14 CAGE_DOOR (box-sized door)
            [8, 8],                 # 15 SPEED_POTION (medium)
            [8, 8],                 # 16 HEAL_POTION (medium)
            [8, 8],                 # 17 POISON_POTION (medium)
            [8, 8],                 # 18 HAMMER
            [6, 12],               # 19 CHECKPOINT (flag on pole)
        ], dtype=jnp.int32)

        # Color id mapping per item type (aligning with palette above)
        self.ITEM_TYPE_COLOR_IDS = jnp.array([
            0,   # unused
            8,   # HEART (red)
            9,   # POISON (green)
            13,  # TRAP (brown)
            12,  # STRONGBOX (yellow)
            12,  # AMBER CHALICE (yellow)
            12,  # AMULET (yellow)
            12,  # GOLD CHALICE (yellow)
            15,  # SHIELD (blue)
            16,  # GUN (black)
            17,  # BOMB (white)
            18,  # KEY (gold)
            19,  # LADDER_UP (brown exit door)
            20,  # LADDER_DOWN (dark brown ladder)
            19,  # CAGE_DOOR (reuse ladder-up tint)
            21,  # SPEED_POTION (orange)
            22,  # HEAL_POTION (magenta)
            23,  # POISON_POTION (green)
            24,  # HAMMER (brown)
        ], dtype=jnp.int32)
        # Python constants for item type color IDs (indexed by item_type - 1)
        self.ITEM_TYPE_COLOR_IDS_PY = [
            self.HEART_ID,         # 1: ITEM_HEART
            self.POISON_ID,        # 2: ITEM_POISON
            self.TRAP_ID,          # 3: ITEM_TRAP
            self.TREASURE_ID,      # 4: ITEM_STRONGBOX
            self.TREASURE_ID,      # 5: ITEM_AMBER_CHALICE
            self.TREASURE_ID,      # 6: ITEM_AMULET
            self.TREASURE_ID,      # 7: ITEM_GOLD_CHALICE
            self.SHIELD_ID,        # 8: ITEM_SHIELD
            self.GUN_ID,           # 9: ITEM_GUN
            self.BOMB_ID,          # 10: ITEM_BOMB
            self.KEY_ID,           # 11: ITEM_KEY
            self.LADDER_UP_ID,     # 12: ITEM_LADDER_UP
            self.LADDER_DOWN_ID,   # 13: ITEM_LADDER_DOWN
            self.LADDER_UP_ID,     # 14: ITEM_CAGE_DOOR
            self.COLOR_TO_ID[self.consts.SPEED_POTION_COLOR],   # 15: ITEM_SPEED_POTION
            self.COLOR_TO_ID[self.consts.HEAL_POTION_COLOR],    # 16: ITEM_HEAL_POTION
            self.COLOR_TO_ID[self.consts.POISON_POTION_COLOR],  # 17: ITEM_POISON_POTION
            self.COLOR_TO_ID[self.consts.HAMMER_COLOR],         # 18: ITEM_HAMMER
            self.CHECKPOINT_ID,                                  # 19: ITEM_CHECKPOINT
        ]
    
    def render(self, state: DarkChambersState) -> jnp.ndarray:
        """Return an RGB image for the current state (H×W×3, uint8)."""
        # Start with background sprite
        object_raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Camera follows player (viewport is reduced to GAMEPLAY_H due to UI bar)
        cam_x = jnp.clip(
            state.player_x - GAME_W // 2, 
            0, 
            self.consts.WORLD_WIDTH - GAME_W
        ).astype(jnp.int32)
        cam_y = jnp.clip(
            state.player_y - GAMEPLAY_H // 2, 
            0, 
            self.consts.WORLD_HEIGHT - GAMEPLAY_H
        ).astype(jnp.int32)
        
        # Draw walls for current level and map
        # Note: Wall textures can't be dynamically indexed in JAX due to tracing limitations
        # Using solid color rendering for walls
        current_level_walls = self.LEVEL_WALLS[state.map_index, state.current_level]
        wall_positions = (current_level_walls[:, 0:2] - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        wall_sizes = current_level_walls[:, 2:4]

        # Clip wall rects to visible gameplay viewport before drawing.
        # This avoids x/y == -1, which the shared renderer interprets as "hidden".
        wall_x0 = wall_positions[:, 0]
        wall_y0 = wall_positions[:, 1]
        wall_x1 = wall_x0 + wall_sizes[:, 0]
        wall_y1 = wall_y0 + wall_sizes[:, 1]
        wall_clip_x0 = jnp.clip(wall_x0, 0, GAME_W)
        wall_clip_y0 = jnp.clip(wall_y0, 0, GAMEPLAY_H)
        wall_clip_x1 = jnp.clip(wall_x1, 0, GAME_W)
        wall_clip_y1 = jnp.clip(wall_y1, 0, GAMEPLAY_H)
        wall_clip_w = jnp.maximum(0, wall_clip_x1 - wall_clip_x0)
        wall_clip_h = jnp.maximum(0, wall_clip_y1 - wall_clip_y0)
        wall_valid = (wall_clip_w > 0) & (wall_clip_h > 0)
        wall_positions = jnp.stack([
            jnp.where(wall_valid, wall_clip_x0, -100),
            jnp.where(wall_valid, wall_clip_y0, -100)
        ], axis=1).astype(jnp.int32)
        wall_sizes = jnp.stack([wall_clip_w, wall_clip_h], axis=1).astype(jnp.int32)

        object_raster = self.jr.draw_rects(
            object_raster, 
            positions=wall_positions, 
            sizes=wall_sizes, 
            color_id=self.WALL_ID
        )

        # Reinforce map borders every frame (with portal gaps) so edges never disappear
        boundary_walls = self.BOUNDARY_WALLS
        boundary_positions = (boundary_walls[:, 0:2] - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        boundary_sizes = boundary_walls[:, 2:4]

        # Same clipping workaround for boundary wall rects.
        boundary_x0 = boundary_positions[:, 0]
        boundary_y0 = boundary_positions[:, 1]
        boundary_x1 = boundary_x0 + boundary_sizes[:, 0]
        boundary_y1 = boundary_y0 + boundary_sizes[:, 1]
        boundary_clip_x0 = jnp.clip(boundary_x0, 0, GAME_W)
        boundary_clip_y0 = jnp.clip(boundary_y0, 0, GAMEPLAY_H)
        boundary_clip_x1 = jnp.clip(boundary_x1, 0, GAME_W)
        boundary_clip_y1 = jnp.clip(boundary_y1, 0, GAMEPLAY_H)
        boundary_clip_w = jnp.maximum(0, boundary_clip_x1 - boundary_clip_x0)
        boundary_clip_h = jnp.maximum(0, boundary_clip_y1 - boundary_clip_y0)
        boundary_valid = (boundary_clip_w > 0) & (boundary_clip_h > 0)
        boundary_positions = jnp.stack([
            jnp.where(boundary_valid, boundary_clip_x0, -100),
            jnp.where(boundary_valid, boundary_clip_y0, -100)
        ], axis=1).astype(jnp.int32)
        boundary_sizes = jnp.stack([boundary_clip_w, boundary_clip_h], axis=1).astype(jnp.int32)

        object_raster = self.jr.draw_rects(
            object_raster,
            positions=boundary_positions,
            sizes=boundary_sizes,
            color_id=self.WALL_ID
        )

        # Top UI removed - now using bottom bar like ALE reference

        # Items - mask inactive ones by moving off-screen
        items_world_pos = state.item_positions.astype(jnp.int32)
        items_screen_pos = (items_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)

        # Masking for inactive items
        off_screen = jnp.array([-100, -100], dtype=jnp.int32)
        masked_item_pos = jnp.where(
            state.item_active[:, None] == 1,
            items_screen_pos,
            off_screen
        )

        # Draw items by type using sprites (for heart, poison, trap, ladder_up) or rectangles (for others)
        def draw_item_type(raster, t, positions_world):
            mask = (state.item_types == t) & (state.item_active == 1)
            pos = jnp.where(mask[:, None], positions_world, off_screen)
            size_wh = self.ITEM_TYPE_SIZES[t]
            sizes = jnp.tile(size_wh[None, :], (NUM_ITEMS, 1))
            color_id_py = self.ITEM_TYPE_COLOR_IDS_PY[t-1]  # t in 1..12
            return self.jr.draw_rects(
                raster,
                positions=pos,
                sizes=sizes,
                color_id=color_id_py
            )

        # Render items with sprites for specific types
        def render_item_sprite(i, raster):
            item_type = state.item_types[i]
            is_active = state.item_active[i] == 1
            item_x = masked_item_pos[i, 0]
            item_y = masked_item_pos[i, 1]

            # ITEM_SHIELD = 8 -> pot sprite (shield icon)
            is_shield = (item_type == ITEM_SHIELD) & is_active
            pot_sprite = self.ITEM_SCALED_MASKS.get("pot")
            raster = jax.lax.cond(
                is_shield & (pot_sprite is not None),
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, pot_sprite),
                lambda r: r,
                raster
            )

            # ITEM_POISON = 2 -> skull sprite
            is_poison = (item_type == ITEM_POISON) & is_active
            skull_sprite = self.ITEM_SCALED_MASKS.get("skull")
            raster = jax.lax.cond(
                is_poison & (skull_sprite is not None),
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, skull_sprite),
                lambda r: r,
                raster
            )

            # ITEM_TRAP = 3 -> trapdoor sprite
            is_trap = (item_type == ITEM_TRAP) & is_active
            trapdoor_sprite = self.ITEM_SCALED_MASKS.get("trapdoor")
            # Draw a visible box under/for traps so they aren't invisible on dark backgrounds
            trap_size = self.ITEM_TYPE_SIZES[ITEM_TRAP]
            trap_sizes = jnp.array([trap_size], dtype=jnp.int32)
            trap_pos = jnp.array([[item_x, item_y]], dtype=jnp.int32)
            trap_color = self.ITEM_TYPE_COLOR_IDS_PY[ITEM_TRAP - 1]
            raster = jax.lax.cond(
                is_trap,
                lambda r: self.jr.draw_rects(r, positions=trap_pos, sizes=trap_sizes, color_id=trap_color),
                lambda r: r,
                raster
            )
            raster = jax.lax.cond(
                is_trap & (trapdoor_sprite is not None),
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, trapdoor_sprite),
                lambda r: r,
                raster
            )

            # ITEM_LADDER_UP = 12 -> stairs sprite
            is_ladder_up = (item_type == ITEM_LADDER_UP) & is_active
            stairs_sprite = self.ITEM_SCALED_MASKS.get("stairs")
            raster = jax.lax.cond(
                is_ladder_up & (stairs_sprite is not None),
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, stairs_sprite),
                lambda r: r,
                raster
            )

            # ITEM_CAGE_DOOR = 14 -> door sprite scaled to box
            is_cage_door = (item_type == ITEM_CAGE_DOOR) & is_active
            door_sprite = self.ITEM_SCALED_MASKS.get("door")
            raster = jax.lax.cond(
                is_cage_door & (door_sprite is not None),
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, door_sprite),
                lambda r: r,
                raster
            )

            # ITEM_LADDER_DOWN = 13 -> stairs sprite (same as ladder_up)
            is_ladder_down = (item_type == ITEM_LADDER_DOWN) & is_active
            stairs_down_sprite = self.ITEM_SCALED_MASKS.get("stairs_down")
            raster = jax.lax.cond(
                is_ladder_down & (stairs_down_sprite is not None),
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, stairs_down_sprite),
                lambda r: r,
                raster
            )

            # ITEM_HEART = 1 -> apple sprite
            apple_sprite = self.ITEM_SCALED_MASKS.get("apple")
            if apple_sprite is not None:
                is_heart = (item_type == ITEM_HEART) & is_active
                raster = jax.lax.cond(
                    is_heart,
                    lambda r: self.jr.render_at_clipped(r, item_x, item_y, apple_sprite),
                    lambda r: r,
                    raster
                )

            # ITEM_BOMB = 10 -> barrel sprite
            barrel_sprite = self.ITEM_SCALED_MASKS.get("barrel")
            if barrel_sprite is not None:
                is_bomb = (item_type == ITEM_BOMB) & is_active
                raster = jax.lax.cond(
                    is_bomb,
                    lambda r: self.jr.render_at_clipped(r, item_x, item_y, barrel_sprite),
                    lambda r: r,
                    raster
                )

            # ITEM_AMBER_CHALICE = 5 -> candle sprite (+500 points treasure)
            candle_sprite = self.ITEM_SCALED_MASKS.get("candle")
            if candle_sprite is not None:
                is_amber_chalice = (item_type == ITEM_AMBER_CHALICE) & is_active
                raster = jax.lax.cond(
                    is_amber_chalice,
                    lambda r: self.jr.render_at_clipped(r, item_x, item_y, candle_sprite),
                    lambda r: r,
                    raster
                )

            # ITEM_AMULET = 6 -> chain sprite (+1000 points treasure)
            chain_sprite = self.ITEM_SCALED_MASKS.get("chain")
            if chain_sprite is not None:
                is_amulet = (item_type == ITEM_AMULET) & is_active
                raster = jax.lax.cond(
                    is_amulet,
                    lambda r: self.jr.render_at_clipped(r, item_x, item_y, chain_sprite),
                    lambda r: r,
                    raster
                )

            # ITEM_KEY = 11 -> key sprite
            key_sprite = self.ITEM_SCALED_MASKS.get("key")
            if key_sprite is not None:
                is_key = (item_type == ITEM_KEY) & is_active
                raster = jax.lax.cond(
                    is_key,
                    lambda r: self.jr.render_at_clipped(r, item_x, item_y, key_sprite),
                    lambda r: r,
                    raster
                )

            # ITEM_GUN = 9 -> pistol sprite (faster shooting)
            is_gun = (item_type == ITEM_GUN) & is_active
            pistol_sprite = self.ITEM_SCALED_MASKS.get("pistol")
            if pistol_sprite is not None:
                raster = jax.lax.cond(
                    is_gun,
                    lambda r: self.jr.render_at_clipped(r, item_x, item_y, pistol_sprite),
                    lambda r: r,
                    raster
                )
            else:
                gun_size = self.ITEM_TYPE_SIZES[ITEM_GUN]
                gun_sizes = jnp.array([gun_size], dtype=jnp.int32)
                gun_pos = jnp.array([[item_x, item_y]], dtype=jnp.int32)
                gun_color = self.ITEM_TYPE_COLOR_IDS_PY[ITEM_GUN - 1]
                raster = jax.lax.cond(
                    is_gun,
                    lambda r: self.jr.draw_rects(r, positions=gun_pos, sizes=gun_sizes, color_id=gun_color),
                    lambda r: r,
                    raster
                )

            # ITEM_HAMMER = 18 -> hammer sprite
            is_hammer = (item_type == ITEM_HAMMER) & is_active
            raster = jax.lax.cond(
                is_hammer,
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, self.HAMMER_SPRITE),
                lambda r: r,
                raster
            )

            # ITEM_SPEED_POTION = 15, ITEM_HEAL_POTION = 16, ITEM_POISON_POTION = 17 -> bottle sprites
            is_speed_potion = (item_type == ITEM_SPEED_POTION) & is_active
            raster = jax.lax.cond(
                is_speed_potion,
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, self.SPEED_POTION_SPRITE),
                lambda r: r,
                raster
            )
            is_heal_potion = (item_type == ITEM_HEAL_POTION) & is_active
            raster = jax.lax.cond(
                is_heal_potion,
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, self.HEAL_POTION_SPRITE),
                lambda r: r,
                raster
            )
            is_poison_potion = (item_type == ITEM_POISON_POTION) & is_active
            raster = jax.lax.cond(
                is_poison_potion,
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, self.POISON_POTION_SPRITE),
                lambda r: r,
                raster
            )

            # ITEM_CHECKPOINT = 19 -> flag sprite (gold = inactive, blue = activated)
            is_chk = (item_type == ITEM_CHECKPOINT) & is_active
            is_activated = (state.checkpoint_idx == i) & (state.checkpoint_map == state.map_index) & (state.checkpoint_level == state.current_level)
            chk_sprite = jnp.where(is_activated, self.CHECKPOINT_SPRITE_ACTIVATED, self.CHECKPOINT_SPRITE)
            raster = jax.lax.cond(
                is_chk,
                lambda r: self.jr.render_at_clipped(r, item_x, item_y, chk_sprite),
                lambda r: r,
                raster
            )

            return raster

        # First render sprite-based items using fori_loop
        object_raster = jax.lax.fori_loop(0, NUM_ITEMS, render_item_sprite, object_raster)

        # Then render remaining items (treasures, powerups, etc.) as colored boxes
        # Skip types that now use sprites: HEART(1), POISON(2), TRAP(3), AMBER_CHALICE(5), AMULET(6), SHIELD(8), BOMB(10), KEY(11), LADDER_UP(12), LADDER_DOWN(13), CAGE_DOOR(14)
        for t in [4, 7]:  # STRONGBOX(4), HOURGLASS(7) — potions and hammer now use sprites
            object_raster = draw_item_type(object_raster, t, masked_item_pos)

        # Spawners: render as large skulls (24x24)
        spawner_world_pos = state.spawner_positions.astype(jnp.int32)
        spawner_screen_pos = (spawner_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        spawner_active_mask = state.spawner_active == 1
        for i in range(NUM_SPAWNERS):
            if self.SPAWNER_SKULL_MASK is not None:
                object_raster = jax.lax.cond(
                    spawner_active_mask[i],
                    lambda r: self.jr.render_at_clipped(r, spawner_screen_pos[i, 0], spawner_screen_pos[i, 1], self.SPAWNER_SKULL_MASK),
                    lambda r: r,
                    object_raster
                )
        # (If no skull sprite, fallback to colored box)
        if self.SPAWNER_SKULL_MASK is None:
            masked_spawner_pos = jnp.where(
                spawner_active_mask[:, None],
                spawner_screen_pos,
                off_screen
            )
            spawner_sizes = jnp.tile(
                jnp.array([SPAWNER_WIDTH, SPAWNER_HEIGHT], dtype=jnp.int32)[None, :],
                (NUM_SPAWNERS, 1)
            )
            object_raster = self.jr.draw_rects(
                object_raster,
                positions=masked_spawner_pos,
                sizes=spawner_sizes,
                color_id=self.SPAWNER_ID
            )

        # Enemies - use sprites for types 1-4, colored box for Grim Reaper (type 5)
        enemy_world_pos = state.enemy_positions.astype(jnp.int32)
        enemy_screen_pos = (enemy_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        num_enemies = enemy_screen_pos.shape[0]
        enemy_active_mask = (state.enemy_active == 1)
        
        # Use same approach as boxes: mask inactive enemies to off-screen position
        _off = jnp.array([-100, -100], dtype=jnp.int32)
        masked_enemy_pos = jnp.where(
            enemy_active_mask[:, None],
            enemy_screen_pos,
            _off
        )
        
        # Pre-fetch directional sprite stacks for each enemy type (static at trace time)
        zombie_anim_frames        = self.ZOMBIE_ANIM_FRAMES        # (4, 5, H, W)
        ghost_anim_frames         = self.GHOST_ANIM_FRAMES         # (4, 3, H, W)
        skeleton_anim_frames      = self.SKELETON_ANIM_FRAMES      # (4, 3, H, W)
        wizard_anim_frames        = self.WIZARD_ANIM_FRAMES        # (4, 5, H, W)
        grim_reaper_anim_frames   = self.GRIM_REAPER_ANIM_FRAMES   # (4, 5, H, W)

        # Render each enemy type using directional sprites (types 1-4)
        def render_one_enemy(i, raster):
            ex = masked_enemy_pos[i, 0]
            ey = masked_enemy_pos[i, 1]
            enemy_type = state.enemy_types[i]
            is_active = enemy_active_mask[i]

            # Map 8-direction enemy_dir to 4 sprite indices (0=right,1=down,2=left,3=up)
            dir_idx = DIR8_TO_SPRITE[state.enemy_dir[i].astype(jnp.int32)]

            anim_tick      = (state.step_counter // ANIM_EVERY).astype(jnp.int32)
            is_enemy_moving = state.enemy_moving[i]

            # Type 1: Zombie — multi-frame walking animation
            is_zombie = (enemy_type == 1) & is_active
            if zombie_anim_frames is not None:
                num_frames = ZOMBIE_NUM_FRAMES[dir_idx]   # 3 or 5
                anim_frame = jnp.where(is_enemy_moving, anim_tick % num_frames, jnp.array(0, dtype=jnp.int32))
                zombie_sprite = zombie_anim_frames[dir_idx, anim_frame]
                raster = jax.lax.cond(
                    is_zombie,
                    lambda r: self.jr.render_at_clipped(r, ex, ey, zombie_sprite),
                    lambda r: r,
                    raster
                )

            # Type 2: Wraith (ghost) — multi-frame walking animation
            is_wraith = (enemy_type == 2) & is_active
            if ghost_anim_frames is not None:
                num_frames = GHOST_NUM_FRAMES[dir_idx]   # 2 or 3 depending on direction
                anim_frame = jnp.where(is_enemy_moving, anim_tick % num_frames, jnp.array(0, dtype=jnp.int32))
                ghost_sprite = ghost_anim_frames[dir_idx, anim_frame]
                raster = jax.lax.cond(
                    is_wraith,
                    lambda r: self.jr.render_at_clipped(r, ex, ey, ghost_sprite),
                    lambda r: r,
                    raster
                )

            # Type 3: Skeleton — multi-frame walking animation
            is_skeleton = (enemy_type == 3) & is_active
            if skeleton_anim_frames is not None:
                num_frames = SKELETON_NUM_FRAMES[dir_idx]   # always 3
                anim_frame = jnp.where(is_enemy_moving, anim_tick % num_frames, jnp.array(0, dtype=jnp.int32))
                skel_sprite = skeleton_anim_frames[dir_idx, anim_frame]
                raster = jax.lax.cond(
                    is_skeleton,
                    lambda r: self.jr.render_at_clipped(r, ex, ey, skel_sprite),
                    lambda r: r,
                    raster
                )

            # Type 4: Wizard — multi-frame walking animation
            is_wizard = (enemy_type == 4) & is_active
            if wizard_anim_frames is not None:
                num_frames = WIZARD_NUM_FRAMES[dir_idx]   # 3 or 5
                anim_frame = jnp.where(is_enemy_moving, anim_tick % num_frames, jnp.array(0, dtype=jnp.int32))
                wizard_sprite = wizard_anim_frames[dir_idx, anim_frame]
                raster = jax.lax.cond(
                    is_wizard,
                    lambda r: self.jr.render_at_clipped(r, ex, ey, wizard_sprite),
                    lambda r: r,
                    raster
                )

            # Type 5: Grim Reaper — headless blue skeleton sprite
            is_grim_reaper = (enemy_type == 5) & is_active
            if grim_reaper_anim_frames is not None:
                num_frames = SKELETON_NUM_FRAMES[dir_idx]  # [3, 4, 3, 4]
                anim_frame = jnp.where(is_enemy_moving, anim_tick % num_frames, jnp.array(0, dtype=jnp.int32))
                grim_sprite = grim_reaper_anim_frames[dir_idx, anim_frame]
                raster = jax.lax.cond(
                    is_grim_reaper,
                    lambda r: self.jr.render_at_clipped(r, ex, ey, grim_sprite),
                    lambda r: r,
                    raster
                )

            return raster

        # Render all enemies with sprites
        object_raster = jax.lax.fori_loop(0, NUM_ENEMIES, render_one_enemy, object_raster)

        # Player (use directional sprite based on player_direction)
        player_screen_x = (state.player_x - cam_x).astype(jnp.int32)
        player_screen_y = (state.player_y - cam_y).astype(jnp.int32)

        if self.PLAYER_ANIM_FRAMES is not None:
            sprite_idx = PLAYER_DIR_TO_SPRITE[state.player_direction.astype(jnp.int32)]
            idle_frame = PLAYER_IDLE_FRAME[sprite_idx]
            anim_tick  = (state.step_counter // ANIM_EVERY).astype(jnp.int32)
            num_frames = PLAYER_NUM_FRAMES[sprite_idx]
            anim_frame = jnp.where(state.player_moving, anim_tick % num_frames, idle_frame)
            player_sprite = self.PLAYER_ANIM_FRAMES[sprite_idx, anim_frame]
            object_raster = jax.lax.cond(
                state.chamber_transition_ticks > 0,
                lambda r: r,
                lambda r: self.jr.render_at_clipped(r, player_screen_x, player_screen_y, player_sprite),
                object_raster,
            )

        # Bottom black UI bar (48 pixels tall - 3x original size)
        bar_y_start = GAMEPLAY_H  # Start at end of gameplay area (160)
        black_bar_pos = jnp.array([[0, bar_y_start]], dtype=jnp.int32)
        black_bar_size = jnp.array([[GAME_W, UI_BAR_HEIGHT]], dtype=jnp.int32)
        black_id = self.COLOR_TO_ID.get((0, 0, 0), self.COLOR_TO_ID.get(self.consts.BACKGROUND_COLOR))
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=black_bar_pos,
            sizes=black_bar_size,
            color_id=black_id
        )
        
        # Top row Y position for health bar and score (same height)
        top_row_y = bar_y_start + 8
        
        # Orange health bar on the LEFT (proportional to health)
        health_val = jnp.clip(state.health, 0, self.consts.MAX_HEALTH).astype(jnp.int32)
        max_health_bar_width = 70  # Wider for bigger bar
        health_bar_width = (health_val * max_health_bar_width) // self.consts.MAX_HEALTH
        health_bar_width = jnp.maximum(health_bar_width, 0)
        
        health_bar_height = 10  # Taller for bigger UI bar
        health_bar_y = top_row_y
        health_bar_x = 6  # Left side with padding
        
        health_bar_pos = jnp.array([[health_bar_x, health_bar_y]], dtype=jnp.int32)
        health_bar_size = jnp.array([[health_bar_width, health_bar_height]], dtype=jnp.int32)
        # Only draw if width > 0
        health_bar_pos = jnp.where(
            health_bar_width > 0,
            health_bar_pos,
            jnp.array([[-100, -100]], dtype=jnp.int32)
        )
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=health_bar_pos,
            sizes=health_bar_size,
            color_id=self.HUD_ID
        )
        
        # Score digits on the RIGHT (same row as health bar), suppress leading zeros
        score_val = jnp.clip(state.score, 0, 9999999).astype(jnp.int32)
        score_scale = 2
        digit_width = 3 * score_scale
        digit_height = 5 * score_scale
        spacing = 2

        millions = score_val // 1000000
        hundred_thousands = (score_val // 100000) % 10
        ten_thousands = (score_val // 10000) % 10
        thousands = (score_val // 1000) % 10
        hundreds = (score_val // 100) % 10
        tens = (score_val // 10) % 10
        ones = score_val % 10
        digits = jnp.array([millions, hundred_thousands, ten_thousands, thousands, hundreds, tens, ones], dtype=jnp.int32)

        active_mask = jnp.array([
            millions > 0,
            (millions > 0) | (hundred_thousands > 0),
            (millions > 0) | (hundred_thousands > 0) | (ten_thousands > 0),
            (millions > 0) | (hundred_thousands > 0) | (ten_thousands > 0) | (thousands > 0),
            (millions > 0) | (hundred_thousands > 0) | (ten_thousands > 0) | (thousands > 0) | (hundreds > 0),
            (millions > 0) | (hundred_thousands > 0) | (ten_thousands > 0) | (thousands > 0) | (hundreds > 0) | (tens > 0),
            True
        ])

        position_index = (jnp.cumsum(active_mask.astype(jnp.int32)) - 1)
        active_count = jnp.sum(active_mask.astype(jnp.int32))
        total_width = active_count * digit_width + (active_count - 1) * spacing
        score_start_x = GAME_W - total_width - 6
        base_x = jnp.where(active_mask, score_start_x + position_index * (digit_width + spacing), -100)

        patterns = self.DIGIT_PATTERNS[digits]
        patterns = jnp.repeat(jnp.repeat(patterns, score_scale, axis=1), score_scale, axis=2)
        xs = jnp.arange(digit_width)
        ys = jnp.arange(digit_height)
        grid_x = xs[None, None, :].repeat(7, axis=0).repeat(digit_height, axis=1)
        grid_y = ys[None, :, None].repeat(7, axis=0).repeat(digit_width, axis=2)

        px = base_x[:, None, None] + grid_x
        py = top_row_y + grid_y
        pixel_active = (patterns == 1) & (base_x[:, None, None] >= 0)
        px = jnp.where(pixel_active, px, -100)
        py = jnp.where(pixel_active, py, -100)

        score_positions = jnp.stack([px.reshape(-1), py.reshape(-1)], axis=1).astype(jnp.int32)
        score_sizes = jnp.ones((score_positions.shape[0], 2), dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=score_positions,
            sizes=score_sizes,
            color_id=self.HUD_ID
        )
        
        # Level indicator (bottom area of UI bar, centered)
        level_digit_width = 3
        level_digit_height = 5
        level_val = (state.current_level + 1).astype(jnp.int32)  # 1-based
        level_y = bar_y_start + 30  # Lower part of UI bar
        level_pattern = self.DIGIT_PATTERNS[level_val]
        level_x = GAME_W // 2 - level_digit_width // 2  # Centered
        
        xs = jnp.arange(level_digit_width)
        ys = jnp.arange(level_digit_height)
        px = level_x + xs[None, :].repeat(level_digit_height, axis=0)
        py = level_y + ys[:, None].repeat(level_digit_width, axis=1)
        
        pixel_active = (level_pattern == 1)
        px = jnp.where(pixel_active, px, -100)
        py = jnp.where(pixel_active, py, -100)
        
        level_positions = jnp.stack([px.reshape(-1), py.reshape(-1)], axis=1).astype(jnp.int32)
        level_sizes = jnp.ones((level_positions.shape[0], 2), dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=level_positions,
            sizes=level_sizes,
            color_id=self.UI_ID
        )
        
        # Removed step counter display from top-center UI
        
        # Bullets
        bullet_world_pos = state.bullet_positions[:, :2].astype(jnp.int32)
        bullet_screen_pos = (bullet_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        bullet_mask = (state.bullet_active == 1) & (bullet_screen_pos[:, 1] + BULLET_HEIGHT - 1 < GAMEPLAY_H)
        masked_bullet_pos = jnp.where(
            bullet_mask[:, None],
            bullet_screen_pos,
            off_screen
        )
        bullet_sprite = self.BULLET_SPRITE
        def render_one_bullet(i, raster):
            pos = masked_bullet_pos[i]
            return jax.lax.cond(
                bullet_mask[i],
                lambda r: self.jr.render_at_clipped(r, pos[0], pos[1], bullet_sprite),
                lambda r: r,
                raster,
            )
        object_raster = jax.lax.fori_loop(0, MAX_BULLETS, render_one_bullet, object_raster)
        # Enemy bullets
        ebullet_world_pos = state.enemy_bullet_positions[:, :2].astype(jnp.int32)
        ebullet_screen_pos = (ebullet_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        ebullet_mask = state.enemy_bullet_active == 1
        masked_ebullet_pos = jnp.where(ebullet_mask[:, None], ebullet_screen_pos, off_screen)
        ebullet_sizes = jnp.tile(
            jnp.array([ENEMY_BULLET_WIDTH, ENEMY_BULLET_HEIGHT], dtype=jnp.int32)[None, :],
            (ENEMY_MAX_BULLETS, 1)
        )
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=masked_ebullet_pos,
            sizes=ebullet_sizes,
            color_id=self.UI_ID
        )
        
        # Power-up indicators in UI bar (lower section, left side)
        indicator_y = bar_y_start + 30  # Same row as level indicator
        
        # Shield indicator
        shield_indicator_active = state.shield_active == 1
        shield_x = 10
        shield_pos = jnp.where(
            shield_indicator_active,
            jnp.array([[shield_x, indicator_y]], dtype=jnp.int32),
            jnp.array([[-100, -100]], dtype=jnp.int32)
        )
        shield_size = jnp.array([[ITEM_WIDTH, ITEM_HEIGHT]], dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=shield_pos,
            sizes=shield_size,
            color_id=self.SHIELD_ID
        )
        
        # Gun indicator next to shield
        gun_indicator_active = state.gun_active == 1
        gun_x = shield_x + ITEM_WIDTH + 3
        if self.GUN_HUD_SPRITE is not None:
            object_raster = jax.lax.cond(
                gun_indicator_active,
                lambda r: self.jr.render_at_clipped(r, gun_x, indicator_y, self.GUN_HUD_SPRITE),
                lambda r: r,
                object_raster
            )
        else:
            gun_pos = jnp.where(
                gun_indicator_active,
                jnp.array([[gun_x, indicator_y]], dtype=jnp.int32),
                jnp.array([[-100, -100]], dtype=jnp.int32)
            )
            gun_size = jnp.array([[ITEM_WIDTH, ITEM_HEIGHT]], dtype=jnp.int32)
            object_raster = self.jr.draw_rects(
                object_raster,
                positions=gun_pos,
                sizes=gun_size,
                color_id=self.GUN_ID
            )
        
        # Bomb indicator next to gun
        bomb_has_any = state.bomb_count > 0
        bomb_indicator_x = gun_x + ITEM_WIDTH + 3
        bomb_indicator_pos = jnp.where(
            bomb_has_any,
            jnp.array([[bomb_indicator_x, indicator_y]], dtype=jnp.int32),
            jnp.array([[-100, -100]], dtype=jnp.int32)
        )
        bomb_indicator_size = jnp.array([[ITEM_WIDTH, ITEM_HEIGHT]], dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=bomb_indicator_pos,
            sizes=bomb_indicator_size,
            color_id=17
        )
        
        # Bomb counter next to bomb icon
        bomb_count_x = bomb_indicator_x + ITEM_WIDTH + 2
        bomb_count_y = indicator_y
        
        # Bomb count digits (0-15) in bottom-right corner
        bomb_count_val = jnp.clip(state.bomb_count, 0, MAX_BOMBS).astype(jnp.int32)
        digit_width = 3
        digit_height = 5
        spacing = 1
        tens = bomb_count_val // 10
        ones = bomb_count_val % 10
        digits = jnp.array([tens, ones], dtype=jnp.int32)
        active_mask = jnp.array([tens > 0, True])
        bomb_digits_start_x = bomb_count_x
        bomb_digits_y = bomb_count_y + 1  # Slight offset for alignment
        position_index = (jnp.cumsum(active_mask.astype(jnp.int32)) - 1)
        base_x = jnp.where(active_mask, bomb_digits_start_x + position_index * (digit_width + spacing), -100)
        patterns = self.DIGIT_PATTERNS[digits]
        xs = jnp.arange(digit_width)
        ys = jnp.arange(digit_height)
        grid_x = xs[None, None, :].repeat(2, axis=0).repeat(digit_height, axis=1)
        grid_y = ys[None, :, None].repeat(2, axis=0).repeat(digit_width, axis=2)
        px = base_x[:, None, None] + grid_x
        py = bomb_digits_y + grid_y
        pixel_active = (patterns == 1) & (base_x[:, None, None] >= 0)
        px = jnp.where(pixel_active, px, -100)
        py = jnp.where(pixel_active, py, -100)
        flat_px = px.reshape(-1)
        flat_py = py.reshape(-1)
        bomb_digit_positions = jnp.stack([flat_px, flat_py], axis=1).astype(jnp.int32)
        bomb_digit_sizes = jnp.ones((bomb_digit_positions.shape[0], 2), dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=bomb_digit_positions,
            sizes=bomb_digit_sizes,
            color_id=self.BOMB_ID
        )

        # Hammer indicator in HUD (shown next to bomb count when player has hammers)
        hammer_has_any = state.hammer_count > 0
        hammer_indicator_x = bomb_count_x + 12  # Fixed offset after bomb count area
        object_raster = jax.lax.cond(
            hammer_has_any,
            lambda r: self.jr.render_at_clipped(r, jnp.array(hammer_indicator_x), jnp.array(indicator_y), self.HAMMER_SPRITE),
            lambda r: r,
            object_raster,
        )

        # Hammer count digit
        hammer_count_val_hud = jnp.clip(state.hammer_count, 0, MAX_HAMMERS).astype(jnp.int32)
        h_digit_x = hammer_indicator_x + ITEM_WIDTH + 5
        h_digit_y = indicator_y + 1
        h_pattern = self.DIGIT_PATTERNS[hammer_count_val_hud]
        h_xs = jnp.arange(3)
        h_ys = jnp.arange(5)
        h_px = h_digit_x + h_xs[None, :].repeat(5, axis=0)
        h_py = h_digit_y + h_ys[:, None].repeat(3, axis=1)
        h_active = (h_pattern == 1) & hammer_has_any
        h_px = jnp.where(h_active, h_px, -100)
        h_py = jnp.where(h_active, h_py, -100)
        hammer_digit_positions = jnp.stack([h_px.reshape(-1), h_py.reshape(-1)], axis=1).astype(jnp.int32)
        hammer_digit_sizes = jnp.ones((hammer_digit_positions.shape[0], 2), dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=hammer_digit_positions,
            sizes=hammer_digit_sizes,
            color_id=self.HAMMER_ID
        )

        # Convert to RGB
        img = self.jr.render_from_palette(object_raster, self.PALETTE)

        # Side-chamber transition darkening: 3-step fade while gameplay is frozen.
        elapsed = (CHAMBER_TRANSITION_TICKS - state.chamber_transition_ticks).astype(jnp.int32)
        phase = (elapsed // CHAMBER_TRANSITION_TICKS_PER_STEP) + 1
        phase = jnp.clip(phase, 0, CHAMBER_TRANSITION_STEPS)
        fade_factors = jnp.array([1.0, 0.9, 0.8, 0.7], dtype=jnp.float32)
        fade_factor = jnp.where(
            state.chamber_transition_ticks > 0,
            fade_factors[phase],
            jnp.array(1.0, dtype=jnp.float32),
        )
        img = jnp.clip(img.astype(jnp.float32) * fade_factor, 0, 255).astype(jnp.uint8)

        def render_level_changed_screen(_: None) -> jnp.ndarray:
            h, w, c = img.shape
            screen = jnp.zeros((h, w, c), dtype=jnp.uint8)
            message = "LEVEL CHANGED"
            scale = 2
            glyph_w = 5
            glyph_h = 7
            spacing = 1

            char_w = glyph_w * scale
            char_h = glyph_h * scale
            total_w = len(message) * char_w + (len(message) - 1) * spacing
            start_x = (w - total_w) // 2
            start_y = (h - char_h) // 2

            mask = jnp.zeros((h, w), dtype=bool)
            cursor_x = start_x
            for ch in message:
                glyph = LEVEL_CHANGE_FONT_5X7[ch]
                for gy, row in enumerate(glyph):
                    for gx, px in enumerate(row):
                        if px == "1":
                            y0 = start_y + gy * scale
                            x0 = cursor_x + gx * scale
                            mask = mask.at[y0:y0 + scale, x0:x0 + scale].set(True)
                cursor_x += char_w + spacing

            white = jnp.full((c,), 255, dtype=jnp.uint8)
            return jnp.where(mask[..., None], white, screen)

        raw_level_screen = jax.lax.cond(
            state.level_transition_ticks > 0,
            render_level_changed_screen,
            lambda x: x,
            operand=img,
        )
        return raw_level_screen


class DarkChambersEnv(JaxEnvironment[DarkChambersState, DarkChambersObservation, DarkChambersInfo, DarkChambersConstants]):
    """JAX environment for the game Dark Chambers."""

    _renderer_cache: ClassVar[dict] = {}

    def __new__(cls, consts: DarkChambersConstants = None):
        resolved = consts or DarkChambersConstants()
        try:
            import dataclasses as _dc
            _key = tuple(
                (f.name, getattr(resolved, f.name))
                for f in _dc.fields(resolved)
            )
        except Exception:
            return super().__new__(cls)
        import sys as _sys
        _registry = _sys.modules.setdefault('_jaxatari_darkchambers_env_registry', {})
        if _key in _registry:
            return _registry[_key]
        instance = super().__new__(cls)
        _registry[_key] = instance
        return instance

    def __init__(self, consts: DarkChambersConstants = None):
        # Guard: __init__ is called on the cached instance on every DarkChambersEnv()
        # call, so skip all initialisation after the first time.
        if getattr(self, '_dc_initialized', False):
            return
        self._dc_initialized = True

        super().__init__(consts=consts or DarkChambersConstants())
        w_eff = int(self.consts.ENEMY_WIDTH  - 2 * ENEMY_COLLISION_MARGIN)
        h_eff = int(self.consts.ENEMY_HEIGHT - 2 * ENEMY_COLLISION_MARGIN)

        r_x = max(0, (w_eff // 2) // CELL_SIZE)
        r_y = max(0, (h_eff // 2) // CELL_SIZE)

        self._wall_inflate_kh = 2 * r_y + 1
        self._wall_inflate_kw = 2 * r_x + 1

        # Reuse a single shared renderer for all instances with the same config.
        _cache_key = (type(self).__name__, self._wall_inflate_kh, self._wall_inflate_kw,
                      hash(self.consts))
        if _cache_key not in DarkChambersEnv._renderer_cache:
            DarkChambersEnv._renderer_cache[_cache_key] = DarkChambersRenderer(self.consts)
        self.renderer = DarkChambersEnv._renderer_cache[_cache_key]

    def __hash__(self):
        try:
            consts_hash = hash(self.consts)
        except Exception:
            consts_hash = id(self.consts)
        return hash((type(self).__name__, consts_hash,
                     self._wall_inflate_kh, self._wall_inflate_kw))

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        try:
            consts_eq = (self.consts == other.consts)
        except Exception:
            consts_eq = (self.consts is other.consts)
        return (consts_eq
                and self._wall_inflate_kh == other._wall_inflate_kh
                and self._wall_inflate_kw == other._wall_inflate_kw)

    def _pos_to_cell(self, x, y):
        """Convert pixel coordinates to grid-cell indices (cy, cx)."""
        cy = (y // CELL_SIZE).astype(jnp.int32)
        cx = (x // CELL_SIZE).astype(jnp.int32)
        cy = jnp.clip(cy, 0, GRID_H - 1)
        cx = jnp.clip(cx, 0, GRID_W - 1)
        return cy, cx

    def _distance_field(
        self,
        map_index: chex.Array,
        level: chex.Array,
        player_x: chex.Array,
        player_y: chex.Array,
        enemy_positions: chex.Array,
        enemy_active: chex.Array,
        enemy_exclude_idx: chex.Array,   # int32 scalar, -1 means “exclude nobody”
    ):

        """Compute 4-neighbor grid distances to the player.

        Alive enemies are treated as obstacles. Returns an int32 array
        of shape (GRID_H, GRID_W) with distances in cells.
        """
        # Base wall grid from the level and current map
        wall_grid = self.renderer.LEVEL_WALL_GRID[map_index, level]  # (H, W) bool
        H, W = wall_grid.shape

        # --- Inflate walls for enemy clearance (STATIC kernel size, JIT-safe) ---
        wall_i = wall_grid.astype(jnp.int32)

        inflated_i = jax.lax.reduce_window(
            wall_i,
            init_value=0,
            computation=jax.lax.max,
            window_dimensions=(self._wall_inflate_kh, self._wall_inflate_kw),
            window_strides=(1, 1),
            padding="SAME",
        )
        wall_grid = inflated_i.astype(bool)

        # --- Enemy occupancy grid (treat alive enemies as walls) ---
        # Use enemy centers to map to nav cells
        centers_x = enemy_positions[:, 0] + self.consts.ENEMY_WIDTH // 2
        centers_y = enemy_positions[:, 1] + self.consts.ENEMY_HEIGHT // 2

        enemy_cy, enemy_cx = self._pos_to_cell(centers_x, centers_y)  # arrays

        # clip indices for safe scatter
        enemy_cy = jnp.clip(enemy_cy, 0, H - 1)
        enemy_cx = jnp.clip(enemy_cx, 0, W - 1)

        # scatter as int32 sum then >0 (handles duplicates cleanly)
        enemy_grid_i = jnp.zeros((H, W), dtype=jnp.int32)
        idxs = jnp.arange(enemy_active.shape[0], dtype=jnp.int32)
        exclude = (idxs == enemy_exclude_idx) & (enemy_exclude_idx >= 0)
        add_vals = jnp.where(exclude, 0, enemy_active).astype(jnp.int32)

        enemy_grid_i = enemy_grid_i.at[enemy_cy, enemy_cx].add(add_vals)
        enemy_grid = enemy_grid_i > 0

        # Combined obstacle grid: walls OR enemies
        blocked = wall_grid | enemy_grid

        # --- Distance init ---
        dist = jnp.full((H, W), BIG_DIST, dtype=jnp.int32)

        py, px = self._pos_to_cell(player_x, player_y)
        py = jnp.clip(py, 0, H - 1)
        px = jnp.clip(px, 0, W - 1)

        # If player cell is blocked, still seed it (otherwise field becomes useless)
        dist = dist.at[py, px].set(0)
        frontier = jnp.zeros((H, W), dtype=bool).at[py, px].set(True)

        def body(t, carry):
            dist, frontier = carry

            # 4-neighbor expansion in parallel (up, down, left, right).
            shifted_frontier = jnp.stack(
                [
                    jnp.pad(frontier[1:, :], ((0, 1), (0, 0))),
                    jnp.pad(frontier[:-1, :], ((1, 0), (0, 0))),
                    jnp.pad(frontier[:, 1:], ((0, 0), (0, 1))),
                    jnp.pad(frontier[:, :-1], ((0, 0), (1, 0))),
                ],
                axis=0,
            )

            candidate_mask = (
                shifted_frontier
                & (~blocked)[None, :, :]
                & (dist[None, :, :] > (t + 1))
            )
            new_frontier = jnp.any(candidate_mask, axis=0)
            new_dist = jnp.where(new_frontier, t + 1, dist)

            return (new_dist, new_frontier)

        max_steps = GRID_W + GRID_H
        dist, _ = jax.lax.fori_loop(0, max_steps, body, (dist, frontier))
        return dist

    

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> Tuple[DarkChambersObservation, DarkChambersState]:
        """Reset the environment and return the initial observation and state."""
        
        # Use middle map (0), level 0 walls for initial spawn (plus boundaries)
        WALLS = jnp.concatenate([
            self.renderer.LEVEL_WALLS[0, 0],
            self.renderer.BOUNDARY_WALLS,
        ], axis=0)
        
        def check_wall_overlap(pos_x, pos_y, width, height):
            """Check if a rectangle overlaps with any wall."""
            wx = WALLS[:, 0]
            wy = WALLS[:, 1]
            ww = WALLS[:, 2]
            wh = WALLS[:, 3]
            overlap_x = (pos_x <= (wx + ww - 1)) & ((pos_x + width - 1) >= wx)
            overlap_y = (pos_y <= (wy + wh - 1)) & ((pos_y + height - 1) >= wy)
            return jnp.any(overlap_x & overlap_y)
        
        # Spawn enemies (retry until not on wall and not near player spawn)
        def spawn_enemy(carry, i):
            positions, key = carry
            
            # Define exclusion zone around player start (upper left corner)
            SPAWN_EXCLUSION_RADIUS = 60  # Keep enemies at least 60 pixels away from player start
            
            def try_spawn(retry_idx, retry_carry):
                pos, key_in, found_valid = retry_carry
                key_out, subkey = jax.random.split(key_in)
                x = jax.random.randint(subkey, (), 30, self.consts.WORLD_WIDTH - 30, dtype=jnp.int32)
                key_out, subkey = jax.random.split(key_out)
                y = jax.random.randint(subkey, (), 30, self.consts.WORLD_HEIGHT - 30, dtype=jnp.int32)
                
                on_wall = check_wall_overlap(x, y, self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT)
                
                # Check if too close to player spawn point
                dx = x - self.consts.PLAYER_START_X
                dy = y - self.consts.PLAYER_START_Y
                dist_sq = dx * dx + dy * dy
                too_close_to_spawn = dist_sq < (SPAWN_EXCLUSION_RADIUS * SPAWN_EXCLUSION_RADIUS)
                
                # Valid if not on wall AND not too close to spawn
                is_valid = (~on_wall) & (~too_close_to_spawn)
                # Update position if this is better (valid and haven't found valid yet)
                new_pos = jnp.where(is_valid & (~found_valid), jnp.array([x, y]), pos)
                new_found = found_valid | is_valid
                
                return (new_pos, key_out, new_found)
            
            # Try up to 20 times
            init_pos = jnp.array([30, 30])
            final_pos, key, _ = jax.lax.fori_loop(0, 20, try_spawn, (init_pos, key, False))
            
            new_positions = positions.at[i].set(final_pos)
            return (new_positions, key), None
        
        key, subkey = jax.random.split(key)
        enemy_positions_init = jnp.zeros((NUM_ENEMIES, 2), dtype=jnp.int32)
        (enemy_positions, key), _ = jax.lax.scan(spawn_enemy, (enemy_positions_init, subkey), jnp.arange(NUM_ENEMIES))
        
        # Spawn random number of enemies (3-6)
        key, subkey = jax.random.split(key)
        num_active_enemies = jax.random.randint(subkey, (), 3, 7, dtype=jnp.int32)
        
        # Spawn enemies with random types (favor stronger types; grim reaper is mod-gated)
        key, subkey = jax.random.split(key)
        enemy_type_high = jnp.where(
            jnp.array(self.consts.ENABLE_GRIM_REAPER_ENEMIES),
            ENEMY_GRIM_REAPER + 1,
            ENEMY_WIZARD + 1,
        )
        enemy_types = jax.random.randint(
            subkey, (NUM_ENEMIES,), ENEMY_WRAITH, enemy_type_high, dtype=jnp.int32
        )  # Random types from 2 (Wraith) to 4/5 (Wizard/Grim Reaper)
        # Only activate the first num_active_enemies
        enemy_active = jnp.where(jnp.arange(NUM_ENEMIES) < num_active_enemies, 1, 0)
        
        # Initialize wizard shooting timers with random offsets (base interval + 0-20 random offset)
        key, subkey = jax.random.split(key)
        wizard_shoot_timers = jax.random.randint(
            subkey, (NUM_ENEMIES,), 0, WIZARD_SHOOT_INTERVAL + WIZARD_SHOOT_OFFSET_MAX, dtype=jnp.int32
        )

        # --- Enemy movement state init ---
        key, subkey = jax.random.split(key)
        enemy_dir = jax.random.randint(subkey, (NUM_ENEMIES,), 0, 8, dtype=jnp.int32)  # 0..7 for ORBIT_DIRS
        enemy_chase_timer = jnp.zeros((NUM_ENEMIES,), dtype=jnp.int32)
        enemy_confuse_timer = jnp.zeros((NUM_ENEMIES,), dtype=jnp.int32)

        key, subkey = jax.random.split(key)
        enemy_idle_timer = jax.random.randint(subkey, (NUM_ENEMIES,), 10, 60, dtype=jnp.int32)  # how long to keep current dir

        key, subkey = jax.random.split(key)
        enemy_pause_timer = jax.random.randint(subkey, (NUM_ENEMIES,), 0, 10, dtype=jnp.int32)  # start mostly moving
        enemy_pause_timer = jnp.minimum(enemy_pause_timer, 1)  # basically 0 or 1 at start


        # --- Enemy patrol box + stuck counter init ---
        # patrol box = current viewport-sized "room" that contains the spawn point
        ex = enemy_positions[:, 0]
        ey = enemy_positions[:, 1]

        room_min_x = (ex // GAME_W) * GAME_W
        room_min_y = (ey // GAME_H) * GAME_H

        room_max_x = jnp.minimum(room_min_x + GAME_W - self.consts.ENEMY_WIDTH, self.consts.WORLD_WIDTH - self.consts.ENEMY_WIDTH)
        room_max_y = jnp.minimum(room_min_y + GAME_H - self.consts.ENEMY_HEIGHT, self.consts.WORLD_HEIGHT - self.consts.ENEMY_HEIGHT)

        enemy_patrol_box = jnp.stack([room_min_x, room_min_y, room_max_x, room_max_y], axis=1).astype(jnp.int32)  # (N,4)
        enemy_stuck_counter = jnp.zeros((NUM_ENEMIES,), dtype=jnp.int32)



        
        # Spawn spawners (retry until there's space for enemies to spawn inside)
        def spawn_spawner(carry, i):
            positions, key = carry
            
            def try_spawn_spawner(retry_idx, retry_carry):
                pos, key_in, found_valid = retry_carry
                key_out, subkey = jax.random.split(key_in)
                x = jax.random.randint(subkey, (), 50, self.consts.WORLD_WIDTH - 50, dtype=jnp.int32)
                key_out, subkey = jax.random.split(key_out)
                y = jax.random.randint(subkey, (), 50, self.consts.WORLD_HEIGHT - 50, dtype=jnp.int32)
                
                # Check if spawner itself doesn't overlap walls
                spawner_on_wall = check_wall_overlap(x, y, SPAWNER_WIDTH, SPAWNER_HEIGHT)
                
                # Also check if there's space for an enemy to spawn centered inside it
                spawn_offset_x = (SPAWNER_WIDTH - self.consts.ENEMY_WIDTH) // 2
                spawn_offset_y = (SPAWNER_HEIGHT - self.consts.ENEMY_HEIGHT) // 2
                enemy_x = x + spawn_offset_x
                enemy_y = y + spawn_offset_y
                enemy_has_space = check_wall_overlap(enemy_x, enemy_y, self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT)
                
                # Valid if spawner doesn't overlap AND there's space for enemy inside
                is_valid = (~spawner_on_wall) & (~enemy_has_space)
                
                # Update position if this is better (valid and haven't found valid yet)
                new_pos = jnp.where(is_valid & (~found_valid), jnp.array([x, y]), pos)
                new_found = found_valid | is_valid
                
                return (new_pos, key_out, new_found)
            
            # Try up to 20 times
            init_pos = jnp.array([100, 100])
            final_pos, key, _ = jax.lax.fori_loop(0, 20, try_spawn_spawner, (init_pos, key, False))
            
            new_positions = positions.at[i].set(final_pos)
            return (new_positions, key), None
        
        key, subkey = jax.random.split(key)
        spawner_positions_init = jnp.zeros((NUM_SPAWNERS, 2), dtype=jnp.int32)
        (spawner_positions, key), _ = jax.lax.scan(spawn_spawner, (spawner_positions_init, subkey), jnp.arange(NUM_SPAWNERS))
        
        spawner_health = jnp.full(NUM_SPAWNERS, SPAWNER_HEALTH, dtype=jnp.int32)
        spawner_active = jnp.ones(NUM_SPAWNERS, dtype=jnp.int32)  # ENABLED: all spawners active
        key, subkey = jax.random.split(key)
        spawner_timers = jax.random.randint(
            subkey, (NUM_SPAWNERS,), 0, SPAWNER_SPAWN_INTERVAL, dtype=jnp.int32
        )
        
        # Spawn items (retry until not on wall)
        def spawn_item(carry, i):
            positions, key = carry
            
            def try_spawn_item(retry_idx, retry_carry):
                pos, key_in, found_valid = retry_carry
                key_out, subkey = jax.random.split(key_in)
                x = jax.random.randint(subkey, (), 30, self.consts.WORLD_WIDTH - 30, dtype=jnp.int32)
                key_out, subkey = jax.random.split(key_out)
                y = jax.random.randint(subkey, (), 30, self.consts.WORLD_HEIGHT - 30, dtype=jnp.int32)
                
                # Use max item size to avoid large treasures spawning on walls
                on_wall = check_wall_overlap(x, y, 13, 13)
                # Update position if this is better (not on wall and haven't found valid yet)
                new_pos = jnp.where((~on_wall) & (~found_valid), jnp.array([x, y]), pos)
                new_found = found_valid | (~on_wall)
                
                return (new_pos, key_out, new_found)
            
            # Try up to 20 times
            init_pos = jnp.array([30, 30])
            final_pos, key, _ = jax.lax.fori_loop(0, 20, try_spawn_item, (init_pos, key, False))
            
            new_positions = positions.at[i].set(final_pos)
            return (new_positions, key), None
        
        key, subkey = jax.random.split(key)
        item_positions_init = jnp.zeros((NUM_ITEMS, 2), dtype=jnp.int32)
        # Spawn only regular items (indices 5+), leave first 5 for fixed key, ladders, and cage props
        (item_positions_temp, key), _ = jax.lax.scan(spawn_item, (item_positions_init, subkey), jnp.arange(5, NUM_ITEMS))
        
        # Place key at a random valid position away from player spawn; ladders fixed
        # Avoid walls and avoid upper-left spawn area (player starts around 24,24)
        KEY_EXCLUSION_RADIUS = 80

        def try_spawn_key(retry_idx, retry_carry):
            pos, key_in, found_valid = retry_carry
            key_out, sk = jax.random.split(key_in)
            x = jax.random.randint(sk, (), 16, self.consts.WORLD_WIDTH - 24, dtype=jnp.int32)
            key_out, sk = jax.random.split(key_out)
            y = jax.random.randint(sk, (), 16, self.consts.WORLD_HEIGHT - 24, dtype=jnp.int32)
            on_wall = check_wall_overlap(x, y, 13, 13)
            dx = x - self.consts.PLAYER_START_X
            dy = y - self.consts.PLAYER_START_Y
            dist_sq = dx * dx + dy * dy
            too_close = dist_sq < (KEY_EXCLUSION_RADIUS * KEY_EXCLUSION_RADIUS)
            is_valid = (~on_wall) & (~too_close)
            new_pos = jnp.where(is_valid & (~found_valid), jnp.array([x, y]), pos)
            new_found = found_valid | is_valid
            return (new_pos, key_out, new_found)

        init_key_pos = jnp.array([100, 100], dtype=jnp.int32)
        key_pos, key, _ = jax.lax.fori_loop(0, 20, try_spawn_key, (init_key_pos, key, False))
        # Main chamber level-0 ladder-up is in the upper-left corner.
        exit_pos = jnp.array([12, 105], dtype=jnp.int32)
        ladder_down_pos = jnp.array([40, 70], dtype=jnp.int32)
        main_map_corner_gun_pos = jnp.array([140, 20], dtype=jnp.int32)
        cage_door_pos = self.renderer.CAGE_DOOR_POSITIONS[0, 0]  # Middle map, level 0
        cage_reward_pos = self.renderer.CAGE_REWARD_POSITIONS[0, 0]  # Middle map, level 0
        
        # Combine: first 5 positions are key, exit, ladder_down, cage door, cage reward, rest are randomly spawned
        item_positions = item_positions_temp.at[0:5].set(jnp.array([key_pos, exit_pos, ladder_down_pos, cage_door_pos, cage_reward_pos]))
        
        # On level 0, hide the downward ladder by placing it off-screen (ladder_down is at index 2)
        item_positions = item_positions.at[2].set(jnp.array([-1000, -1000], dtype=jnp.int32))
        
        # Weighted distribution of item types (no extra keys here to enforce exactly one key per level)
        # Reserve first 5 slots for key, ladders, cage door, cage reward; rest are regular items without keys
        # Only spawn items that have sprites (exclude STRONGBOX, GOLD_CHALICE)
        key, subkey = jax.random.split(key)
        all_item_types = jnp.array([
            ITEM_HEART,          # health +10 -> apple sprite
            ITEM_POISON,         # -4 health -> skull sprite
            ITEM_TRAP,           # -6 health -> trapdoor sprite
            ITEM_AMBER_CHALICE,  # 500 pts -> candle sprite
            ITEM_AMULET,         # 1000 pts -> chain sprite
            ITEM_SHIELD,         # damage reduction -> pot sprite
            ITEM_GUN,            # faster shooting -> no sprite (colored box)
            ITEM_BOMB,           # kill all enemies -> barrel sprite
            ITEM_SPEED_POTION,   # 2x speed for 120 steps -> orange box
            ITEM_HEAL_POTION,    # restore to max health -> magenta box
            ITEM_POISON_POTION,  # create poison cloud -> green box
            ITEM_HAMMER,         # kills all in radius -> brown box
        ], dtype=jnp.int32)
        # Probabilities for regular items; potion spawns are mod-gated (default off)
        spawn_probs = jnp.array([
            0.18,  # heart (reduced from 0.20)
            0.08 if self.consts.ENABLE_DEFAULT_POISON_SPAWN else 0.0,  # poison
            0.15 if self.consts.ENABLE_DEFAULT_TRAP_SPAWN else 0.0,  # trap
            0.12,  # amber chalice (reduced from 0.14)
            0.08,  # amulet (reduced from 0.10)
            0.07,  # shield (reduced from 0.08)
            0.03,  # gun (further reduced)
            0.07,  # bomb (further reduced)
            0.08 if self.consts.ENABLE_SPEED_POTION_SPAWN else 0.0,   # speed potion
            0.08 if self.consts.ENABLE_HEAL_POTION_SPAWN else 0.0,     # heal potion
            0.07 if self.consts.ENABLE_POISON_POTION_SPAWN else 0.0,   # poison potion
            0.06 if self.consts.ENABLE_HAMMER_SPAWN else 0.0,          # hammer
        ], dtype=jnp.float32)
        spawn_probs = spawn_probs / jnp.sum(spawn_probs)
        # Spawn regular items (leave first 5 for key, ladders, cage contents)
        regular_items = jax.random.choice(subkey, all_item_types, shape=(NUM_ITEMS - 5 - NUM_CHECKPOINTS,), p=spawn_probs)
        # Add key, ladders, cage door and reward at beginning; checkpoints appended at end
        item_types = jnp.concatenate([
            jnp.array([ITEM_KEY, ITEM_LADDER_UP, ITEM_LADDER_DOWN, ITEM_CAGE_DOOR, self.renderer.CAGE_REWARD_TYPE], dtype=jnp.int32),
            regular_items,
            jnp.full((NUM_CHECKPOINTS,), ITEM_CHECKPOINT, dtype=jnp.int32),
        ])
        # Main map (spawn map): hardcoded upper-right gun.
        item_positions = item_positions.at[5].set(main_map_corner_gun_pos)
        item_types = item_types.at[5].set(jnp.array(ITEM_GUN, dtype=jnp.int32))
        disallowed_potions = (
            ((item_types == ITEM_SPEED_POTION) & (~jnp.array(self.consts.ENABLE_SPEED_POTION_SPAWN)))
            | ((item_types == ITEM_HEAL_POTION) & (~jnp.array(self.consts.ENABLE_HEAL_POTION_SPAWN)))
            | ((item_types == ITEM_POISON_POTION) & (~jnp.array(self.consts.ENABLE_POISON_POTION_SPAWN)))
            | ((item_types == ITEM_HAMMER) & (~jnp.array(self.consts.ENABLE_HAMMER_SPAWN)))
        )
        item_types = jnp.where(disallowed_potions, ITEM_HEART, item_types)
        regular_item_mask = (jnp.arange(NUM_ITEMS) < (5 + INITIAL_REGULAR_ITEM_COUNT)).astype(jnp.int32)
        item_active = jnp.where(jnp.arange(NUM_ITEMS) < 5, 1, regular_item_mask)
        item_active = item_active.at[5].set(jnp.array(1, dtype=jnp.int32))
        # Suppress all spawned items (6+) in the main chamber, keep only gun in the upper-right corner
        item_active = jnp.where(jnp.arange(NUM_ITEMS) > 5, 0, item_active)

        # Dynamically spawn intra-level checkpoints spread across vertical zones, avoiding walls
        def spawn_one_checkpoint_reset(carry, zone_idx):
            positions_arr, key_in = carry
            y_lo = zone_idx * self.consts.WORLD_HEIGHT // NUM_CHECKPOINTS
            y_hi = (zone_idx + 1) * self.consts.WORLD_HEIGHT // NUM_CHECKPOINTS

            def try_once_chk(_, inner):
                pos, key_loc, found = inner
                key_loc, sk = jax.random.split(key_loc)
                x = jax.random.randint(sk, (), 16, self.consts.WORLD_WIDTH - 16, dtype=jnp.int32)
                key_loc, sk = jax.random.split(key_loc)
                y_min = jnp.maximum(y_lo, 8)
                y_max = jnp.maximum(y_hi - 8, y_lo + 1)
                y = jax.random.randint(sk, (), y_min, y_max, dtype=jnp.int32)
                on_wall = check_wall_overlap(x, y, 6, 12)
                new_pos = jnp.where((~on_wall) & (~found), jnp.array([x, y]), pos)
                return (new_pos, key_loc, found | (~on_wall))

            fallback = jnp.array([75, (y_lo + y_hi) // 2], dtype=jnp.int32)
            final_pos, key_out, _ = jax.lax.fori_loop(0, 20, try_once_chk, (fallback, key_in, False))
            positions_arr = positions_arr.at[zone_idx].set(final_pos)
            return (positions_arr, key_out), None

        key, sk = jax.random.split(key)
        chk_init = jnp.zeros((NUM_CHECKPOINTS, 2), dtype=jnp.int32)
        (chk_positions_reset, key), _ = jax.lax.scan(
            spawn_one_checkpoint_reset, (chk_init, sk), jnp.arange(NUM_CHECKPOINTS)
        )
        item_positions = item_positions.at[NUM_ITEMS - NUM_CHECKPOINTS:].set(chk_positions_reset)
        item_active = item_active.at[NUM_ITEMS - NUM_CHECKPOINTS:].set(
            jnp.array(self.consts.ENABLE_INTRA_LEVEL_CHECKPOINTS, dtype=jnp.int32)
        )

        # Safety pass: suppress entities that still overlap walls after spawn attempts.
        def rect_overlaps_walls(pos: chex.Array, width: chex.Array, height: chex.Array) -> chex.Array:
            px = pos[0]
            py = pos[1]
            wx = WALLS[:, 0]
            wy = WALLS[:, 1]
            ww = WALLS[:, 2]
            wh = WALLS[:, 3]
            overlap_x = (px <= (wx + ww - 1)) & ((px + width - 1) >= wx)
            overlap_y = (py <= (wy + wh - 1)) & ((py + height - 1) >= wy)
            return jnp.any(overlap_x & overlap_y)

        enemy_wall_overlap = jax.vmap(
            lambda p: rect_overlaps_walls(p, self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT)
        )(enemy_positions)
        enemy_active = enemy_active & (~enemy_wall_overlap).astype(jnp.int32)
        enemy_hitpoints = jnp.where(enemy_active == 1, ENEMY_HITS_PER_TIER, 0).astype(jnp.int32)

        spawner_wall_overlap = jax.vmap(
            lambda p: rect_overlaps_walls(p, SPAWNER_WIDTH, SPAWNER_HEIGHT)
        )(spawner_positions)
        spawner_active = spawner_active & (~spawner_wall_overlap).astype(jnp.int32)

        item_sizes = self.renderer.ITEM_TYPE_SIZES[item_types]
        item_wall_overlap = jax.vmap(
            lambda p, sz: rect_overlaps_walls(p, sz[0], sz[1])
        )(item_positions, item_sizes)
        item_active = item_active & (~item_wall_overlap).astype(jnp.int32)

        # Cage door/reward should exist only if the cage placement is valid on this map/level.
        cage_valid_reset = self.renderer.CAGE_VALID[0, 0]
        item_active = item_active.at[3].set(cage_valid_reset)
        item_active = item_active.at[4].set(cage_valid_reset)

        # Ensure player spawn is never inside a wall.
        spawn_x0 = jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32)
        spawn_y0 = jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32)

        spawn_offsets = jnp.array([
            [0, 0],
            [8, 0], [-8, 0], [0, 8], [0, -8],
            [8, 8], [8, -8], [-8, 8], [-8, -8],
            [16, 0], [-16, 0], [0, 16], [0, -16],
            [16, 16], [16, -16], [-16, 16], [-16, -16],
            [24, 0], [-24, 0], [0, 24], [0, -24],
            [50 - self.consts.PLAYER_START_X, 50 - self.consts.PLAYER_START_Y],
            [40 - self.consts.PLAYER_START_X, 70 - self.consts.PLAYER_START_Y],
        ], dtype=jnp.int32)

        spawn_candidates = spawn_offsets + jnp.array([spawn_x0, spawn_y0], dtype=jnp.int32)
        spawn_candidates = spawn_candidates.at[:, 0].set(
            jnp.clip(spawn_candidates[:, 0], 0, self.consts.WORLD_WIDTH - self.consts.PLAYER_WIDTH)
        )
        spawn_candidates = spawn_candidates.at[:, 1].set(
            jnp.clip(spawn_candidates[:, 1], 0, self.consts.WORLD_HEIGHT - self.consts.PLAYER_HEIGHT)
        )

        spawn_valid = ~jax.vmap(
            lambda p: check_wall_overlap(
                p[0],
                p[1],
                self.consts.PLAYER_WIDTH,
                self.consts.PLAYER_HEIGHT,
            )
        )(spawn_candidates)
        has_valid_spawn = jnp.any(spawn_valid)
        first_valid_spawn_idx = jnp.argmax(spawn_valid)
        fallback_spawn = jnp.array([50, 50], dtype=jnp.int32)
        safe_spawn = jnp.where(
            has_valid_spawn,
            spawn_candidates[first_valid_spawn_idx],
            fallback_spawn,
        )
        spawn_x = safe_spawn[0]
        spawn_y = safe_spawn[1]

        # For the starting chamber (map_index=0, level=0), move spawners to the lower half
        # This moves the spawner from the right side down to the lower portion of the map
        adjusted_spawner_positions = spawner_positions.at[:, 1].set(
            jnp.maximum(spawner_positions[:, 1], 550)
        )

        state = DarkChambersState(
            player_x=spawn_x,
            player_y=spawn_y,
            player_direction=jnp.array(0, dtype=jnp.int32),
            player_moving=jnp.array(0, dtype=jnp.int32),
            enemy_positions=enemy_positions,
            enemy_types=enemy_types,
            enemy_active=enemy_active,
            enemy_hitpoints=enemy_hitpoints,
            wizard_shoot_timers=wizard_shoot_timers,
            spawner_positions=adjusted_spawner_positions,
            spawner_health=spawner_health,
            spawner_active=spawner_active,
            spawner_timers=spawner_timers,
            bullet_positions=jnp.zeros((MAX_BULLETS, 4), dtype=jnp.int32),
            bullet_active=jnp.zeros(MAX_BULLETS, dtype=jnp.int32),
            enemy_bullet_positions=jnp.zeros((ENEMY_MAX_BULLETS, 4), dtype=jnp.int32),
            enemy_bullet_active=jnp.zeros(ENEMY_MAX_BULLETS, dtype=jnp.int32),
            health=jnp.array(self.consts.STARTING_HEALTH, dtype=jnp.int32),
            damage_cooldown=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            item_positions=item_positions,
            item_types=item_types,
            item_active=item_active,
            has_key=jnp.array(0, dtype=jnp.int32),  # Start without key
            shield_active=jnp.array(0, dtype=jnp.int32),
            gun_active=jnp.array(0, dtype=jnp.int32),
            bomb_count=jnp.array(0, dtype=jnp.int32),
            hammer_count=jnp.array(0, dtype=jnp.int32),
            last_fire_step=jnp.array(-1000, dtype=jnp.int32),  # Initialize to far past
            last_shot_step=jnp.array(-1000, dtype=jnp.int32),  # Initialize to far past
            fire_was_pressed=jnp.array(0, dtype=jnp.int32),
            current_level=jnp.array(0, dtype=jnp.int32),  # Start at level 0
            map_index=jnp.array(0, dtype=jnp.int32),  # Start at middle map
            ladder_timer=jnp.array(0, dtype=jnp.int32),   # Not on ladder initially
            ladder_transition_cooldown=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(MAX_LIVES, dtype=jnp.int32),  # Start with MAX_LIVES
            step_counter=jnp.array(0, dtype=jnp.int32),
            death_counter=jnp.array(0, dtype=jnp.int32),
            chamber_transition_ticks=jnp.array(0, dtype=jnp.int32),
            chamber_transition_pending=jnp.array(0, dtype=jnp.int32),
            chamber_target_map_index=jnp.array(0, dtype=jnp.int32),
            chamber_target_x=spawn_x,
            chamber_target_y=spawn_y,
            level_transition_ticks=jnp.array(0, dtype=jnp.int32),
            key=key,
            enemy_dir=enemy_dir,
            enemy_chase_timer=enemy_chase_timer,
            enemy_confuse_timer=enemy_confuse_timer,
            enemy_patrol_box=enemy_patrol_box,
            enemy_stuck_counter=enemy_stuck_counter,
            enemy_idle_timer=enemy_idle_timer,
            enemy_pause_timer=enemy_pause_timer,
            speed_boost_timer=jnp.array(0, dtype=jnp.int32),
            poison_cloud_x=jnp.array(0, dtype=jnp.int32),
            poison_cloud_y=jnp.array(0, dtype=jnp.int32),
            poison_cloud_timer=jnp.array(0, dtype=jnp.int32),
            enemy_contact_cooldown=jnp.zeros(NUM_ENEMIES, dtype=jnp.int32),
            enemy_moving=jnp.ones(NUM_ENEMIES, dtype=jnp.int32),
            checkpoint_idx=jnp.array(-1, dtype=jnp.int32),
            checkpoint_map=jnp.array(0, dtype=jnp.int32),
            checkpoint_level=jnp.array(0, dtype=jnp.int32),
        )

        obs = self._get_observation(state)
        return obs, state
    

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DarkChambersState, action: int) -> Tuple[DarkChambersObservation, DarkChambersState, float, bool, DarkChambersInfo]:
        """Apply an action and return (obs, state, reward, done, info)."""
        is_pending_transition = state.chamber_transition_pending == 1
        a_raw = jnp.asarray(action, dtype=jnp.int32)
        a = jnp.where(is_pending_transition, jnp.array(Action.NOOP, dtype=jnp.int32), a_raw)

        # If we're in a death freeze, either decrement or respawn now
        def handle_freeze(_: None):
            # When counter > 1: just decrement and freeze everything else
            def dec_only(_: None):
                frozen_state = state.replace(
                    death_counter=state.death_counter - 1,
                    step_counter=state.step_counter + 1,
                )
                obs = self._get_observation(frozen_state)
                reward = self._get_reward(state, frozen_state)
                done = self._get_done(frozen_state)
                info = self._get_info(frozen_state)
                return obs, frozen_state, reward, done, info

            # When counter hits 1: perform respawn (keep game progress!)
            def respawn_now(_: None):
                # Respawn position depends on current level
                # Use safe positions that avoid walls in all map variants
                default_respawn_x = jnp.where(state.current_level == 0, jnp.array(50, dtype=jnp.int32), jnp.array(40, dtype=jnp.int32))
                default_respawn_y = jnp.where(state.current_level == 0, jnp.array(50, dtype=jnp.int32), jnp.array(70, dtype=jnp.int32))

                checkpoint_x_by_map = jnp.array(self.consts.CHECKPOINT_SPAWN_X_BY_MAP, dtype=jnp.int32)
                checkpoint_y_by_map = jnp.array(self.consts.CHECKPOINT_SPAWN_Y_BY_MAP, dtype=jnp.int32)
                checkpoint_respawn_x = checkpoint_x_by_map[state.map_index]
                checkpoint_respawn_y = checkpoint_y_by_map[state.map_index]

                has_intra_checkpoint = state.checkpoint_idx >= 0
                _safe_idx = jnp.maximum(state.checkpoint_idx, 0)
                _chk_respawn_pos = state.item_positions[_safe_idx]
                respawn_x = jnp.where(
                    has_intra_checkpoint,
                    _chk_respawn_pos[0],
                    jnp.where(jnp.array(self.consts.ENABLE_CHECKPOINT_RESPAWN), checkpoint_respawn_x, default_respawn_x),
                )
                respawn_y = jnp.where(
                    has_intra_checkpoint,
                    _chk_respawn_pos[1],
                    jnp.where(jnp.array(self.consts.ENABLE_CHECKPOINT_RESPAWN), checkpoint_respawn_y, default_respawn_y),
                )

                # Safety: if chosen respawn intersects walls, snap to nearest valid nearby point.
                respawn_walls = jnp.concatenate(
                    [
                        self.renderer.LEVEL_WALLS[state.map_index, state.current_level],
                        self.renderer.BOUNDARY_WALLS,
                    ],
                    axis=0,
                )

                def respawn_overlaps(px, py):
                    wx = respawn_walls[:, 0]
                    wy = respawn_walls[:, 1]
                    ww = respawn_walls[:, 2]
                    wh = respawn_walls[:, 3]
                    overlap_x = (px <= (wx + ww - 1)) & ((px + self.consts.PLAYER_WIDTH - 1) >= wx)
                    overlap_y = (py <= (wy + wh - 1)) & ((py + self.consts.PLAYER_HEIGHT - 1) >= wy)
                    return jnp.any(overlap_x & overlap_y)

                respawn_offsets = jnp.array([
                    [0, 0],
                    [8, 0], [-8, 0], [0, 8], [0, -8],
                    [8, 8], [8, -8], [-8, 8], [-8, -8],
                    [16, 0], [-16, 0], [0, 16], [0, -16],
                    [16, 16], [16, -16], [-16, 16], [-16, -16],
                    [24, 0], [-24, 0], [0, 24], [0, -24],
                    [32, 0], [-32, 0], [0, 32], [0, -32],
                ], dtype=jnp.int32)
                respawn_candidates = respawn_offsets + jnp.array([respawn_x, respawn_y], dtype=jnp.int32)
                respawn_candidates = jnp.concatenate(
                    [
                        respawn_candidates,
                        jnp.array([
                            [50, 50],
                            [40, 70],
                            [20, 70],
                            [260, 70],
                        ], dtype=jnp.int32),
                    ],
                    axis=0,
                )
                respawn_candidates = respawn_candidates.at[:, 0].set(
                    jnp.clip(respawn_candidates[:, 0], 0, self.consts.WORLD_WIDTH - self.consts.PLAYER_WIDTH)
                )
                respawn_candidates = respawn_candidates.at[:, 1].set(
                    jnp.clip(respawn_candidates[:, 1], 0, self.consts.WORLD_HEIGHT - self.consts.PLAYER_HEIGHT)
                )

                respawn_valid = ~jax.vmap(lambda p: respawn_overlaps(p[0], p[1]))(respawn_candidates)
                has_valid_respawn = jnp.any(respawn_valid)
                first_valid_respawn_idx = jnp.argmax(respawn_valid)
                safe_respawn = jnp.where(
                    has_valid_respawn,
                    respawn_candidates[first_valid_respawn_idx],
                    jnp.array([respawn_x, respawn_y], dtype=jnp.int32),
                )
                respawn_x = safe_respawn[0]
                respawn_y = safe_respawn[1]

                # KEEP EVERYTHING: score, collected items, level, powerups, etc.
                # Only restore health and clear bullets for clean respawn
                respawned_state = state.replace(
                    player_x=respawn_x,
                    player_y=respawn_y,
                    health=jnp.array(self.consts.STARTING_HEALTH, dtype=jnp.int32),
                    damage_cooldown=jnp.array(0, dtype=jnp.int32),
                    enemy_contact_cooldown=jnp.zeros(NUM_ENEMIES, dtype=jnp.int32),
                    bullet_positions=jnp.zeros((MAX_BULLETS, 4), dtype=jnp.int32),
                    bullet_active=jnp.zeros((MAX_BULLETS,), dtype=jnp.int32),
                    enemy_bullet_positions=jnp.zeros((ENEMY_MAX_BULLETS, 4), dtype=jnp.int32),
                    enemy_bullet_active=jnp.zeros((ENEMY_MAX_BULLETS,), dtype=jnp.int32),
                    death_counter=jnp.array(0, dtype=jnp.int32),
                    level_transition_ticks=jnp.array(0, dtype=jnp.int32),
                    ladder_transition_cooldown=jnp.array(0, dtype=jnp.int32),
                    step_counter=state.step_counter + 1,
                )
                obs = self._get_observation(respawned_state)
                reward = self._get_reward(state, respawned_state)
                done = self._get_done(respawned_state)
                info = self._get_info(respawned_state)
                return obs, respawned_state, reward, done, info

            return jax.lax.cond(state.death_counter > 1, dec_only, respawn_now, operand=None)

        # Side-chamber transition freeze: hold gameplay for a few frames.
        def handle_chamber_transition(_: None):
            frozen_state = state.replace(
                chamber_transition_ticks=jnp.maximum(state.chamber_transition_ticks - 1, 0),
                step_counter=state.step_counter + 1,
            )
            obs = self._get_observation(frozen_state)
            reward = self._get_reward(state, frozen_state)
            done = self._get_done(frozen_state)
            info = self._get_info(frozen_state)
            return obs, frozen_state, reward, done, info

        def handle_level_transition(_: None):
            frozen_state = state.replace(
                level_transition_ticks=jnp.maximum(state.level_transition_ticks - 1, 0),
                step_counter=state.step_counter + 1,
            )
            obs = self._get_observation(frozen_state)
            reward = self._get_reward(state, frozen_state)
            done = self._get_done(frozen_state)
            info = self._get_info(frozen_state)
            return obs, frozen_state, reward, done, info

        def handle_normal(_: None):
            # Normal logic continues below (existing code)
            # We will return at the end of this function.
            
            # Track player direction from action (prioritize horizontal for diagonals)
            new_direction = jnp.where((a == Action.RIGHT) | (a == Action.UPRIGHT) | (a == Action.DOWNRIGHT), 0,
                            jnp.where((a == Action.LEFT) | (a == Action.UPLEFT) | (a == Action.DOWNLEFT), 1,
                            jnp.where((a == Action.UP) | (a == Action.UPRIGHT) | (a == Action.UPLEFT), 2,
                            jnp.where((a == Action.DOWN) | (a == Action.DOWNRIGHT) | (a == Action.DOWNLEFT), 3, 
                                      state.player_direction))))
            
            # Calculate movement deltas - support diagonal movement
            # Apply speed boost if active (from speed potion mod)
            effective_speed = jnp.where(
                state.speed_boost_timer > 0,
                self.consts.PLAYER_SPEED * self.consts.SPEED_POTION_MULTIPLIER,
                self.consts.PLAYER_SPEED
            )
            dx = jnp.where((a == Action.LEFT) | (a == Action.UPLEFT) | (a == Action.DOWNLEFT), -effective_speed,
                   jnp.where((a == Action.RIGHT) | (a == Action.UPRIGHT) | (a == Action.DOWNRIGHT), effective_speed, 0))
            dy = jnp.where((a == Action.UP) | (a == Action.UPRIGHT) | (a == Action.UPLEFT), -effective_speed,
                   jnp.where((a == Action.DOWN) | (a == Action.DOWNRIGHT) | (a == Action.DOWNLEFT), effective_speed, 0))
            player_moving = jnp.where((dx != 0) | (dy != 0), 1, 0).astype(jnp.int32)
            
            dx_raw = dx
            dy_raw = dy

            PLAYER_MOVE_EVERY_X = 3  # slightly faster (was 4)
            dy = jnp.where((state.step_counter % 3) != 2, dy, 0)
            dx = jnp.where((state.step_counter % PLAYER_MOVE_EVERY_X) == 0, dx, 0)

            prop_x = state.player_x + dx
            prop_y = state.player_y + dy
            
            prop_x = jnp.clip(prop_x, 0, self.consts.WORLD_WIDTH - self.consts.PLAYER_WIDTH)
            prop_y = jnp.clip(prop_y, 0, self.consts.WORLD_HEIGHT - self.consts.PLAYER_HEIGHT)
            
            # Check if player is in ANY portal zone (top, middle, bottom)
            portal_hole_height = 40
            portal_gap = (self.consts.WORLD_HEIGHT - 3 * portal_hole_height) // 4
            portal_y_starts = jnp.array(
                [
                    portal_gap,
                    portal_gap * 2 + portal_hole_height,
                    portal_gap * 3 + portal_hole_height * 2,
                ],
                dtype=jnp.int32,
            )
            portal_y_ends = portal_y_starts + portal_hole_height
            # Player is in portal zone if in any of the three.
            in_portal_zone = jnp.any(
                (prop_y >= (portal_y_starts - self.consts.PLAYER_HEIGHT)) & (prop_y <= portal_y_ends)
            )
            
            # Cycle through maps: exit left → go left map, exit right → go right map
            # Middle(0) → Left(1) → Middle(0) → Right(2) → Middle(0)
            should_wrap_right = in_portal_zone & (prop_x <= 0)
            should_wrap_left = in_portal_zone & (prop_x >= self.consts.WORLD_WIDTH - self.consts.PLAYER_WIDTH)
            
            # Track if player crossed portal (for zombie spawning)
            crossed_portal = should_wrap_right | should_wrap_left
            start_chamber_transition = crossed_portal & (~is_pending_transition)
            
            # Update map_index based on which direction player exits.
            # Right-edge cycle: middle(0) -> right(2) -> left(1) -> middle(0)
            # Left-edge cycle:  middle(0) -> left(1)  -> right(2) -> middle(0)
            map_after_left_exit = jnp.where(
                state.map_index == 0,
                1,
                jnp.where(state.map_index == 1, 2, 0),
            )
            map_after_right_exit = jnp.where(
                state.map_index == 0,
                2,
                jnp.where(state.map_index == 2, 1, 0),
            )
            pending_target_map_index = jnp.where(
                should_wrap_right,
                map_after_left_exit,
                jnp.where(should_wrap_left, map_after_right_exit, state.map_index),
            )
            new_map_index = jnp.where(
                is_pending_transition,
                state.chamber_target_map_index,
                jnp.where(start_chamber_transition, state.map_index, pending_target_map_index),
            )
            
            # Teleport to opposite edge when crossing portal
            pending_target_x = jnp.where(should_wrap_right, self.consts.WORLD_WIDTH - self.consts.PLAYER_WIDTH - 2, prop_x)
            pending_target_x = jnp.where(should_wrap_left, 2, pending_target_x)
            pending_target_y = prop_y
            prop_x = jnp.where(
                is_pending_transition,
                state.chamber_target_x,
                jnp.where(start_chamber_transition, prop_x, pending_target_x),
            )
            prop_y = jnp.where(is_pending_transition, state.chamber_target_y, prop_y)
            
            # Wall collision check - include boundary walls (with portal gaps)
            WALLS = jnp.concatenate(
                [
                    self.renderer.LEVEL_WALLS[new_map_index, state.current_level],
                    self.renderer.BOUNDARY_WALLS,
                ],
                axis=0,
            )
            def check_enemy_collision(enemy_pos):
                # shrink collision box by a small margin to allow sliding along walls
                ex = enemy_pos[0] + ENEMY_COLLISION_MARGIN
                ey = enemy_pos[1] + ENEMY_COLLISION_MARGIN
                w = self.consts.ENEMY_WIDTH  - 2 * ENEMY_COLLISION_MARGIN
                h = self.consts.ENEMY_HEIGHT - 2 * ENEMY_COLLISION_MARGIN

                wx = WALLS[:, 0]
                wy = WALLS[:, 1]
                ww = WALLS[:, 2]
                wh = WALLS[:, 3]

                e_overlap_x = (ex <= (wx + ww - 1)) & ((ex + w - 1) >= wx)
                e_overlap_y = (ey <= (wy + wh - 1)) & ((ey + h - 1) >= wy)
                return jnp.any(e_overlap_x & e_overlap_y)

            def clip_and_portal_wrap_enemy(pos):
                clipped = jnp.clip(
                    pos,
                    jnp.array([0, 0], dtype=jnp.int32),
                    jnp.array(
                        [self.consts.WORLD_WIDTH - self.consts.ENEMY_WIDTH,
                        self.consts.WORLD_HEIGHT - self.consts.ENEMY_HEIGHT],
                        dtype=jnp.int32,
                    ),
                )

                portal_hole_height = 40
                portal_y_start = (self.consts.WORLD_HEIGHT - portal_hole_height) // 2
                portal_y_end = portal_y_start + portal_hole_height
                in_portal_zone = (clipped[1] >= portal_y_start - self.consts.ENEMY_HEIGHT) & (clipped[1] <= portal_y_end)

                should_wrap_right = in_portal_zone & (clipped[0] <= 0)
                should_wrap_left  = in_portal_zone & (clipped[0] >= self.consts.WORLD_WIDTH - self.consts.ENEMY_WIDTH)

                wrapped_x = jnp.where(
                    should_wrap_right,
                    self.consts.WORLD_WIDTH - self.consts.ENEMY_WIDTH - 2,
                    jnp.where(should_wrap_left, 2, clipped[0]),
                )
                return jnp.array([wrapped_x, clipped[1]], dtype=jnp.int32)

            def apply_enemy_portal_keepout(pos):
                """Clamp enemies away from portal entrances so they do not block traversal."""
                portal_hole_height = 40
                portal_gap = (self.consts.WORLD_HEIGHT - 3 * portal_hole_height) // 4
                portal_y_starts = jnp.array([
                    portal_gap,
                    portal_gap * 2 + portal_hole_height,
                    portal_gap * 3 + portal_hole_height * 2,
                ], dtype=jnp.int32)
                portal_y_ends = portal_y_starts + portal_hole_height

                in_portal_lane = jnp.any(
                    (pos[1] >= (portal_y_starts - self.consts.ENEMY_HEIGHT))
                    & (pos[1] <= portal_y_ends)
                )

                min_x = self.consts.WALL_THICKNESS + ENEMY_PORTAL_KEEP_OUT_X
                max_x = (
                    self.consts.WORLD_WIDTH
                    - self.consts.ENEMY_WIDTH
                    - self.consts.WALL_THICKNESS
                    - ENEMY_PORTAL_KEEP_OUT_X
                )
                safe_x = jnp.clip(pos[0], min_x, max_x)
                safe_x = jnp.where(in_portal_lane, safe_x, pos[0])
                return jnp.array([safe_x, pos[1]], dtype=jnp.int32)

            def enemy_in_keepout_zone(pos):
                """Return True if pos is inside the portal keepout band (in a portal lane and too close to left/right edge)."""
                portal_hole_height = 40
                portal_gap = (self.consts.WORLD_HEIGHT - 3 * portal_hole_height) // 4
                portal_y_starts = jnp.array([
                    portal_gap,
                    portal_gap * 2 + portal_hole_height,
                    portal_gap * 3 + portal_hole_height * 2,
                ], dtype=jnp.int32)
                portal_y_ends = portal_y_starts + portal_hole_height
                in_portal_lane = jnp.any(
                    (pos[1] >= (portal_y_starts - self.consts.ENEMY_HEIGHT))
                    & (pos[1] <= portal_y_ends)
                )
                min_x = self.consts.WALL_THICKNESS + ENEMY_PORTAL_KEEP_OUT_X
                max_x = (self.consts.WORLD_WIDTH - self.consts.ENEMY_WIDTH
                         - self.consts.WALL_THICKNESS - ENEMY_PORTAL_KEEP_OUT_X)
                outside_x = (pos[0] < min_x) | (pos[0] > max_x)
                return in_portal_lane & outside_x

            def move_enemy(enemy_pos, step_vec):
                """Apply step with wall sliding (full, else x-only, else y-only) + clipping + portal wrap."""
                cur = enemy_pos

                step_full = step_vec
                step_x = jnp.array([step_vec[0], 0], dtype=jnp.int32)
                step_y = jnp.array([0, step_vec[1]], dtype=jnp.int32)

                pos_full = clip_and_portal_wrap_enemy(cur + step_full)
                pos_x    = clip_and_portal_wrap_enemy(cur + step_x)
                pos_y    = clip_and_portal_wrap_enemy(cur + step_y)

                col_full = check_enemy_collision(pos_full) | enemy_in_keepout_zone(pos_full)
                col_x    = check_enemy_collision(pos_x)    | enemy_in_keepout_zone(pos_x)
                col_y    = check_enemy_collision(pos_y)    | enemy_in_keepout_zone(pos_y)

                use_full = ~col_full
                use_x    = col_full & ~col_x
                use_y    = col_full & col_x & ~col_y

                moved = jnp.where(
                    use_full, pos_full,
                    jnp.where(use_x, pos_x, jnp.where(use_y, pos_y, cur)),
                )
                return moved

            def collides(px, py):
                wx = WALLS[:, 0]
                wy = WALLS[:, 1]
                ww = WALLS[:, 2]
                wh = WALLS[:, 3]
                overlap_x = (px <= (wx + ww - 1)) & ((px + self.consts.PLAYER_WIDTH - 1) >= wx)
                overlap_y = (py <= (wy + wh - 1)) & ((py + self.consts.PLAYER_HEIGHT - 1) >= wy)
                return jnp.any(overlap_x & overlap_y)
            
            # Test each axis separately
            try_x = prop_x
            collide_x = collides(try_x, state.player_y)
            new_x = jnp.where(~collide_x, try_x, state.player_x)
            
            try_y = prop_y
            collide_y = collides(new_x, try_y)
            new_y = jnp.where(~collide_y, try_y, state.player_y)

            # Stop animation when the player is pressing against a wall in every pressed direction
            collide_x_wall = collides(state.player_x + dx_raw, state.player_y)
            collide_y_wall = collides(state.player_x, state.player_y + dy_raw)
            pressing_x = (dx_raw != 0)
            pressing_y = (dy_raw != 0)
            facing_wall_x = pressing_x & collide_x_wall
            facing_wall_y = pressing_y & collide_y_wall
            fully_facing_wall = jnp.where(
                pressing_x & pressing_y,
                facing_wall_x & facing_wall_y,   # diagonal: both axes must be blocked
                facing_wall_x | facing_wall_y,   # single axis: that axis blocked
            ) & (player_moving == 1)
            player_moving = jnp.where(fully_facing_wall, jnp.array(0, dtype=jnp.int32), player_moving)

            # Shooting - spawn bullet on FIRE action (or detonate bomb on double-tap)
            fire_pressed = (a == Action.FIRE) | (a == Action.UPFIRE) | (a == Action.DOWNFIRE) | (a == Action.LEFTFIRE) | (a == Action.RIGHTFIRE) | \
                           (a == Action.UPRIGHTFIRE) | (a == Action.UPLEFTFIRE) | (a == Action.DOWNRIGHTFIRE) | (a == Action.DOWNLEFTFIRE)
            
            # Only fire on initial press, not while held
            fire_just_pressed = fire_pressed & (state.fire_was_pressed == 0)

            # Double-tap detection: fire pressed within DOUBLE_TAP_WINDOW steps
            steps_since_last_fire = state.step_counter - state.last_fire_step
            is_double_tap = fire_just_pressed & (steps_since_last_fire <= DOUBLE_TAP_WINDOW) & (steps_since_last_fire > 0)
            has_bombs = state.bomb_count > 0
            has_hammers = state.hammer_count > 0
            should_detonate_bomb = is_double_tap & has_bombs
            should_use_hammer = is_double_tap & has_hammers & (~has_bombs)  # Use hammer if no bombs
            
            # Update last_fire_step when fire is first pressed
            new_last_fire_step = jnp.where(fire_just_pressed, state.step_counter, state.last_fire_step)
            new_fire_was_pressed = fire_pressed.astype(jnp.int32)
            
            # Find first inactive bullet slot
            first_inactive = jnp.argmax(state.bullet_active == 0)
            can_spawn = jnp.any(state.bullet_active == 0)
            steps_since_last_shot = state.step_counter - state.last_shot_step
            
            # Fire rate limiting: ensure minimum time between shots
            # Gun powerup lowers the delay further for faster follow-up shots
            fire_rate_threshold = jnp.where(state.gun_active == 1, FIRE_RATE_LIMIT_WITH_GUN, FIRE_RATE_LIMIT)
            can_fire_now = fire_just_pressed & (steps_since_last_shot >= fire_rate_threshold)
            can_spawn = can_spawn & can_fire_now
            
            # Determine bullet speed based on gun powerup
            current_bullet_speed = jnp.where(state.gun_active == 1, BULLET_SPEED_WITH_GUN, BULLET_SPEED)
            
            # Direction vectors based on fire action (not player direction)
            # Cardinal directions
            dir_x = jnp.where(a == Action.RIGHTFIRE, current_bullet_speed,
                    jnp.where(a == Action.LEFTFIRE, -current_bullet_speed,
                    jnp.where(a == Action.FIRE, jnp.where(new_direction == 0, current_bullet_speed,
                                                jnp.where(new_direction == 1, -current_bullet_speed, 0)),
                    0)))
            dir_y = jnp.where(a == Action.UPFIRE, -current_bullet_speed,
                    jnp.where(a == Action.DOWNFIRE, current_bullet_speed,
                    jnp.where(a == Action.FIRE, jnp.where(new_direction == 2, -current_bullet_speed,
                                                jnp.where(new_direction == 3, current_bullet_speed, 0)),
                    0)))
            
            # Diagonal directions
            dir_x = jnp.where(a == Action.UPRIGHTFIRE, current_bullet_speed,
                    jnp.where(a == Action.DOWNRIGHTFIRE, current_bullet_speed,
                    jnp.where(a == Action.UPLEFTFIRE, -current_bullet_speed,
                    jnp.where(a == Action.DOWNLEFTFIRE, -current_bullet_speed, dir_x))))
            dir_y = jnp.where(a == Action.UPRIGHTFIRE, -current_bullet_speed,
                    jnp.where(a == Action.UPLEFTFIRE, -current_bullet_speed,
                    jnp.where(a == Action.DOWNRIGHTFIRE, current_bullet_speed,
                    jnp.where(a == Action.DOWNLEFTFIRE, current_bullet_speed, dir_y))))
            
            # Spawn position offset from player center
            spawn_x = new_x + self.consts.PLAYER_WIDTH // 2
            spawn_y = new_y + self.consts.PLAYER_HEIGHT // 2
            
            # Create new bullet
            new_bullet = jnp.array([spawn_x, spawn_y, dir_x, dir_y], dtype=jnp.int32)
            
            # Update bullets array
            should_spawn = fire_pressed & can_spawn
            new_last_shot_step = jnp.where(should_spawn, state.step_counter, state.last_shot_step)
            new_bullet_positions = jnp.where(
                jnp.arange(MAX_BULLETS)[:, None] == first_inactive,
                jnp.where(should_spawn, new_bullet, state.bullet_positions),
                state.bullet_positions
            )
            new_bullet_active = jnp.where(
                (jnp.arange(MAX_BULLETS) == first_inactive) & should_spawn,
                1,
                state.bullet_active
            )
            
            # Move existing bullets
            moved_bullets = state.bullet_positions + jnp.concatenate([state.bullet_positions[:, 2:], jnp.zeros((MAX_BULLETS, 2), dtype=jnp.int32)], axis=1)
            
            # Check bounds and deactivate out-of-bounds bullets
            in_bounds = (moved_bullets[:, 0] >= 0) & (moved_bullets[:, 0] < self.consts.WORLD_WIDTH) & \
                        (moved_bullets[:, 1] >= 0) & (moved_bullets[:, 1] < self.consts.WORLD_HEIGHT)
            
            # Check wall collisions for bullets
            def check_bullet_wall_collision(bullet_pos):
                bx, by = bullet_pos[0], bullet_pos[1]
                b_overlap_x = (bx <= (WALLS[:,0] + WALLS[:,2] - 1)) & ((bx + BULLET_WIDTH - 1) >= WALLS[:,0])
                b_overlap_y = (by <= (WALLS[:,1] + WALLS[:,3] - 1)) & ((by + BULLET_HEIGHT - 1) >= WALLS[:,1])
                return jnp.any(b_overlap_x & b_overlap_y)
            
            bullet_wall_collisions = jax.vmap(check_bullet_wall_collision)(moved_bullets[:, :2])
            
            updated_bullet_positions = jnp.where(
                state.bullet_active[:, None] == 1,
                moved_bullets,
                state.bullet_positions
            )
            updated_bullet_keep_mask = (state.bullet_active == 1) & in_bounds & (~bullet_wall_collisions)
            updated_bullet_active = updated_bullet_keep_mask.astype(jnp.int32)
            
            # Merge spawned and moved bullets
            final_bullet_positions = jnp.where(
                should_spawn & (jnp.arange(MAX_BULLETS)[:, None] == first_inactive),
                new_bullet,
                updated_bullet_positions
            )
            final_bullet_active = jnp.where(
                should_spawn & (jnp.arange(MAX_BULLETS) == first_inactive),
                1,
                updated_bullet_active
            )
            # Wizard shooting: mod-gated
            new_wizard_timers = state.wizard_shoot_timers - 1
            wizard_shooting_enabled = jnp.array(self.consts.ENABLE_WIZARD_BULLETS)
            should_shoot = (
                wizard_shooting_enabled
                & (state.enemy_types == ENEMY_WIZARD)
                & (state.enemy_active == 1)
                & (new_wizard_timers <= 0)
            )

            # Reset timers for wizards that shoot (base interval + random offset)
            rng, subkey = jax.random.split(state.key)
            new_offsets = jax.random.randint(
                subkey, (NUM_ENEMIES,), 0, WIZARD_SHOOT_OFFSET_MAX + 1, dtype=jnp.int32
            )
            reset_timers = WIZARD_SHOOT_INTERVAL + new_offsets
            new_wizard_timers = jnp.where(should_shoot, reset_timers, new_wizard_timers)

            # Spawn bullets for all shooting wizards
            # Process each wizard that should shoot
            def spawn_wizard_bullet(carry, wizard_idx):
                bullet_positions, bullet_active, key_in = carry

                # Check if this wizard should shoot
                shoots = should_shoot[wizard_idx]

                # Find first inactive bullet slot
                first_inactive_idx = jnp.argmax(bullet_active == 0)
                can_spawn_bullet = jnp.any(bullet_active == 0) & shoots

                # Calculate bullet trajectory towards player
                wiz_pos = state.enemy_positions[wizard_idx]
                wiz_cx = wiz_pos[0] + self.consts.ENEMY_WIDTH // 2
                wiz_cy = wiz_pos[1] + self.consts.ENEMY_HEIGHT // 2
                p_cx = new_x + self.consts.PLAYER_WIDTH // 2
                p_cy = new_y + self.consts.PLAYER_HEIGHT // 2
                dir_x_e = jnp.sign(p_cx - wiz_cx) * ENEMY_BULLET_SPEED
                dir_y_e = jnp.sign(p_cy - wiz_cy) * ENEMY_BULLET_SPEED

                bullet_spawn = jnp.array([wiz_cx, wiz_cy, dir_x_e, dir_y_e], dtype=jnp.int32)

                # Update bullet arrays if this wizard can spawn
                new_positions = jnp.where(
                    (jnp.arange(ENEMY_MAX_BULLETS)[:, None] == first_inactive_idx) & can_spawn_bullet,
                    bullet_spawn,
                    bullet_positions
                )
                new_active = jnp.where(
                    (jnp.arange(ENEMY_MAX_BULLETS) == first_inactive_idx) & can_spawn_bullet,
                    1,
                    bullet_active
                )

                return (new_positions, new_active, key_in), None

            # Process all wizards sequentially to spawn bullets
            (new_enemy_bullet_positions, new_enemy_bullet_active, rng), _ = jax.lax.scan(
                spawn_wizard_bullet,
                (state.enemy_bullet_positions, state.enemy_bullet_active, rng),
                jnp.arange(NUM_ENEMIES)
            )

            # Move enemy bullets
            moved_enemy_bullets = new_enemy_bullet_positions + jnp.concatenate([
                new_enemy_bullet_positions[:, 2:], jnp.zeros((ENEMY_MAX_BULLETS, 2), dtype=jnp.int32)
            ], axis=1)
            e_in_bounds = (
                (moved_enemy_bullets[:, 0] >= 0)
                & (moved_enemy_bullets[:, 0] < self.consts.WORLD_WIDTH)
                & (moved_enemy_bullets[:, 1] >= 0)
                & (moved_enemy_bullets[:, 1] < self.consts.WORLD_HEIGHT)
            )
            def check_enemy_bullet_wall_collision(bpos):
                bx, by = bpos[0], bpos[1]
                b_overlap_x = (bx <= (WALLS[:,0] + WALLS[:,2] - 1)) & ((bx + ENEMY_BULLET_WIDTH - 1) >= WALLS[:,0])
                b_overlap_y = (by <= (WALLS[:,1] + WALLS[:,3] - 1)) & ((by + ENEMY_BULLET_HEIGHT - 1) >= WALLS[:,1])
                return jnp.any(b_overlap_x & b_overlap_y)
            enemy_bullet_wall_collisions = jax.vmap(check_enemy_bullet_wall_collision)(moved_enemy_bullets[:, :2])
            updated_enemy_bullet_positions = jnp.where(
                new_enemy_bullet_active[:, None] == 1,
                moved_enemy_bullets,
                new_enemy_bullet_positions
            )
            updated_enemy_bullet_keep_mask = (new_enemy_bullet_active == 1) & e_in_bounds & (~enemy_bullet_wall_collisions)
            updated_enemy_bullet_active = updated_enemy_bullet_keep_mask.astype(jnp.int32)
            
            # Final enemy bullet state
            final_enemy_bullet_positions = updated_enemy_bullet_positions
            final_enemy_bullet_active = updated_enemy_bullet_active
            
            # ---------------------------
            # Enemy movement: patrol -> aggro eval -> chase -> confuse/back-off
            # ---------------------------

            enemy_alive = state.enemy_active == 1

            # Player center
            player_center = jnp.array(
                [new_x + self.consts.PLAYER_WIDTH // 2,
                 new_y + self.consts.PLAYER_HEIGHT // 2],
                dtype=jnp.int32
            )

            # Enemy centers
            enemy_centers = state.enemy_positions + jnp.array(
                [self.consts.ENEMY_WIDTH // 2, self.consts.ENEMY_HEIGHT // 2],
                dtype=jnp.int32
            )

            # Distance check for aggro (squared)
            dxy = enemy_centers - player_center[None, :]
            dist2 = dxy[:, 0] * dxy[:, 0] + dxy[:, 1] * dxy[:, 1]
            in_aggro = dist2 <= (AGGRO_RADIUS * AGGRO_RADIUS)

            stuck_counter = state.enemy_stuck_counter

            # --- Parameters: tweak these to make enemies chase less/more ---
            NUDGE_RADIUS = 90          # px: only nudge if player is close (smaller => less chasing)
            NUDGE_EVERY = 25           # evaluate nudges every N ticks (bigger => less chasing)
            NUDGE_PROB_NUM = 1         # probability = NUM / DEN per evaluation
            NUDGE_PROB_DEN = 8         # 1/8 = 12.5% chance per eval tick
            NUDGE_MIN = 2              # nudge lasts 2..6 steps
            NUDGE_MAX = 6

            # Evaluate only every N ticks
            eval_tick = (state.step_counter % NUDGE_EVERY) == 0

            # Timers decrement
            chase_timer = jnp.maximum(state.enemy_chase_timer - 1, 0)     # now used as "nudge_timer"
            confuse_timer = jnp.maximum(state.enemy_confuse_timer - 1, 0)

            in_nudge_radius = dist2 <= (NUDGE_RADIUS * NUDGE_RADIUS)

            # Random trigger to start a nudge burst
            rng, sk = jax.random.split(rng)
            u = jax.random.randint(sk, (NUM_ENEMIES,), 0, NUDGE_PROB_DEN, dtype=jnp.int32)
            trigger = (u < NUDGE_PROB_NUM)

            # Start a burst only if not already nudging or confusing
            start_nudge = eval_tick & (chase_timer == 0) & (confuse_timer == 0) & in_nudge_radius & trigger & enemy_alive

            rng, sk = jax.random.split(rng)
            burst_len = jax.random.randint(sk, (NUM_ENEMIES,), NUDGE_MIN, NUDGE_MAX + 1, dtype=jnp.int32)
            chase_timer = jnp.where(start_nudge, burst_len, chase_timer)

            need_field = jnp.any((chase_timer > 0) & enemy_alive)

            dist_field = jax.lax.cond(
                need_field,
                lambda _: self._distance_field(
                    new_map_index,
                    state.current_level,
                    new_x,
                    new_y,
                    state.enemy_positions,
                    state.enemy_active,
                    jnp.array(-1, dtype=jnp.int32),
                ),
                # if no one nudges, fill with BIG_DIST so chase_step becomes no-op
                lambda _: jnp.full((GRID_H, GRID_W), BIG_DIST, dtype=jnp.int32),
                operand=None,
            )


            def chase_step(pos, alive):
                enemy_center = pos + jnp.array(
                    [self.consts.ENEMY_WIDTH // 2, self.consts.ENEMY_HEIGHT // 2],
                    dtype=jnp.int32,
                )
                cy, cx = self._pos_to_cell(enemy_center[0], enemy_center[1])

                neigh = jnp.array([
                    [cy - 1, cx    ],  # up
                    [cy + 1, cx    ],  # down
                    [cy,     cx - 1],  # left
                    [cy,     cx + 1],  # right
                    [cy - 1, cx - 1],  # up-left
                    [cy - 1, cx + 1],  # up-right
                    [cy + 1, cx - 1],  # down-left
                    [cy + 1, cx + 1],  # down-right
                ], dtype=jnp.int32)

                ny = jnp.clip(neigh[:, 0], 0, GRID_H - 1)
                nx = jnp.clip(neigh[:, 1], 0, GRID_W - 1)

                in_bounds = (
                    (neigh[:, 0] >= 0) & (neigh[:, 0] < GRID_H) &
                    (neigh[:, 1] >= 0) & (neigh[:, 1] < GRID_W)
                )

                dists = dist_field[ny, nx]

                target_centers = jnp.stack(
                    [nx * CELL_SIZE + CELL_SIZE // 2,
                     ny * CELL_SIZE + CELL_SIZE // 2],
                    axis=1,
                )

                step_vecs = jnp.sign(target_centers - enemy_center).astype(jnp.int32)

                def collides_with_wall(step_vec):
                    new_pos = pos + step_vec
                    ex = new_pos[0] + ENEMY_COLLISION_MARGIN
                    ey = new_pos[1] + ENEMY_COLLISION_MARGIN
                    w = self.consts.ENEMY_WIDTH  - 2 * ENEMY_COLLISION_MARGIN
                    h = self.consts.ENEMY_HEIGHT - 2 * ENEMY_COLLISION_MARGIN
                    wx = WALLS[:, 0]; wy = WALLS[:, 1]; ww = WALLS[:, 2]; wh = WALLS[:, 3]
                    e_overlap_x = (ex <= (wx + ww - 1)) & ((ex + w - 1) >= wx)
                    e_overlap_y = (ey <= (wy + wh - 1)) & ((ey + h - 1) >= wy)
                    return jnp.any(e_overlap_x & e_overlap_y)

                collisions = jax.vmap(collides_with_wall)(step_vecs)

                blocked_right = collides_with_wall(jnp.array([ 1,  0], dtype=jnp.int32))
                blocked_left  = collides_with_wall(jnp.array([-1,  0], dtype=jnp.int32))
                blocked_down  = collides_with_wall(jnp.array([ 0,  1], dtype=jnp.int32))
                blocked_up    = collides_with_wall(jnp.array([ 0, -1], dtype=jnp.int32))

                is_diag = (step_vecs[:, 0] != 0) & (step_vecs[:, 1] != 0)
                into_blocked_side = (
                    ((step_vecs[:, 0] > 0) & blocked_right) |
                    ((step_vecs[:, 0] < 0) & blocked_left)  |
                    ((step_vecs[:, 1] > 0) & blocked_down)  |
                    ((step_vecs[:, 1] < 0) & blocked_up)
                )
                diag_forbidden = is_diag & into_blocked_side

                valid = in_bounds & (~collisions) & (~diag_forbidden) & (alive == 1)
                dists_valid = jnp.where(valid, dists, BIG_DIST)

                best_idx = jnp.argmin(dists_valid)
                best_step = step_vecs[best_idx]
                any_valid = jnp.any(valid)
                best_step = jnp.where(any_valid, best_step, jnp.array([0, 0], dtype=jnp.int32))
                return best_step * alive.astype(jnp.int32)

            chase_steps = jax.vmap(chase_step)(state.enemy_positions, enemy_alive)

            # --- Confuse/back-off: step away from player (1px axis) ---
            away = jnp.sign(enemy_centers - player_center[None, :]).astype(jnp.int32)
            confuse_steps = away * enemy_alive[:, None].astype(jnp.int32)

            # --- Patrol: rectangle-like loop inside patrol box with wall/bounds turning ---
            # Directions: 0=E,1=S,2=W,3=N
            DIR4 = jnp.array([[1,0],[0,1],[-1,0],[0,-1]], dtype=jnp.int32)

            def patrol_one(pos, dir4, box, alive):
                step = DIR4[dir4] * IDLE_SPEED

                def would_collide_or_leave(step_vec):
                    new_pos = pos + step_vec

                    # patrol-box clamp check (no clipping; turning happens instead)
                    min_x, min_y, max_x, max_y = box
                    out_box = (new_pos[0] < min_x) | (new_pos[0] > max_x) | (new_pos[1] < min_y) | (new_pos[1] > max_y)

                    ex = new_pos[0] + ENEMY_COLLISION_MARGIN
                    ey = new_pos[1] + ENEMY_COLLISION_MARGIN
                    w = self.consts.ENEMY_WIDTH  - 2 * ENEMY_COLLISION_MARGIN
                    h = self.consts.ENEMY_HEIGHT - 2 * ENEMY_COLLISION_MARGIN

                    wx = WALLS[:, 0]; wy = WALLS[:, 1]; ww = WALLS[:, 2]; wh = WALLS[:, 3]
                    e_overlap_x = (ex <= (wx + ww - 1)) & ((ex + w - 1) >= wx)
                    e_overlap_y = (ey <= (wy + wh - 1)) & ((ey + h - 1) >= wy)
                    hit_wall = jnp.any(e_overlap_x & e_overlap_y)
                    return hit_wall | out_box

                blocked = would_collide_or_leave(step)

                # turn right when blocked
                new_dir = jnp.where(blocked, (dir4 + 1) & 3, dir4)

                # after turning, try that direction this tick
                step2 = DIR4[new_dir] * IDLE_SPEED
                blocked2 = would_collide_or_leave(step2)
                final_step = jnp.where(blocked2, jnp.array([0,0], dtype=jnp.int32), step2)

                final_step = final_step * alive.astype(jnp.int32)
                return final_step, new_dir

            # --- Dynamic patrol box: player's current viewport-sized room ---
            room_min_x = (new_x // GAME_W) * GAME_W
            room_min_y = (new_y // GAME_H) * GAME_H

            room_max_x = jnp.minimum(
                room_min_x + GAME_W - self.consts.ENEMY_WIDTH,
                self.consts.WORLD_WIDTH - self.consts.ENEMY_WIDTH,
            )
            room_max_y = jnp.minimum(
                room_min_y + GAME_H - self.consts.ENEMY_HEIGHT,
                self.consts.WORLD_HEIGHT - self.consts.ENEMY_HEIGHT,
            )

            player_patrol_box = jnp.array([room_min_x, room_min_y, room_max_x, room_max_y], dtype=jnp.int32)
            dynamic_patrol_box = jnp.tile(player_patrol_box[None, :], (NUM_ENEMIES, 1))
            
            box = player_patrol_box
            in_view = (
                (state.enemy_positions[:, 0] >= box[0]) &
                (state.enemy_positions[:, 0] <= box[2]) &
                (state.enemy_positions[:, 1] >= box[1]) &
                (state.enemy_positions[:, 1] <= box[3])
            ) & enemy_alive


            # 8-direction random walk (pixel steps)
            DIR8 = jnp.array([
                [ 1,  0], [ 1, -1], [ 0, -1], [-1, -1],
                [-1,  0], [-1,  1], [ 0,  1], [ 1,  1],
            ], dtype=jnp.int32)

            IDLE_SPEED = 1

            idle_timer = jnp.maximum(state.enemy_idle_timer - 1, 0)
            pause_timer = jnp.maximum(state.enemy_pause_timer - 1, 0)

            # Occasionally start a short pause (rare, short)
            PAUSE_EVERY = 30               # evaluate once per N ticks
            PAUSE_PROB_NUM = 1
            PAUSE_PROB_DEN = 12            # 1/12 chance per eval (~rare)
            PAUSE_MIN = 2
            PAUSE_MAX = 6

            pause_eval_tick = (state.step_counter % PAUSE_EVERY) == 0
            rng, sk = jax.random.split(rng)
            u_pause = jax.random.randint(sk, (NUM_ENEMIES,), 0, PAUSE_PROB_DEN, dtype=jnp.int32)
            start_pause = pause_eval_tick & (pause_timer == 0) & (u_pause < PAUSE_PROB_NUM) & in_view

            rng, sk = jax.random.split(rng)
            pause_len = jax.random.randint(sk, (NUM_ENEMIES,), PAUSE_MIN, PAUSE_MAX + 1, dtype=jnp.int32)
            pause_timer = jnp.where(start_pause, pause_len, pause_timer)

            can_patrol = enemy_alive  # move even off-screen

            pick_new_dir = (idle_timer == 0) & can_patrol & (pause_timer == 0)

            rng, sk = jax.random.split(rng)
            new_dir = jax.random.randint(sk, (NUM_ENEMIES,), 0, 8, dtype=jnp.int32)

            rng, sk = jax.random.split(rng)
            new_idle_len = jax.random.randint(sk, (NUM_ENEMIES,), 300, 700, dtype=jnp.int32)

            enemy_dir = jnp.where(pick_new_dir, new_dir, state.enemy_dir)
            idle_timer = jnp.where(pick_new_dir, new_idle_len, idle_timer)

            # Proposed step (0 if paused or not in view)
            step_vec = DIR8[enemy_dir] * IDLE_SPEED
            step_vec = jnp.where((pause_timer > 0)[:, None] | (~can_patrol)[:, None], 0, step_vec)

            # Apply movement
            cand_enemy_positions = jax.vmap(move_enemy)(state.enemy_positions, step_vec)

            # Detect if stuck (didn't move while trying to)
            moved = jnp.any(cand_enemy_positions != state.enemy_positions, axis=1)
            tried_to_move = jnp.any(step_vec != 0, axis=1)
            stuck_now = tried_to_move & (~moved)

            # If stuck, force direction change next tick
            idle_timer = jnp.where(stuck_now, 0, idle_timer)

            patrol_positions = cand_enemy_positions


            # --- Select mode (chase/confuse overrides wander) ---
            in_confuse = (confuse_timer > 0) & enemy_alive
            in_chase   = (confuse_timer == 0) & (chase_timer > 0) & enemy_alive

            desired_step = jnp.where(in_confuse[:, None], confuse_steps,
                        jnp.where(in_chase[:, None], chase_steps, (patrol_positions - state.enemy_positions)))


            # Save pre-collision, pre-throttle intent for wall-facing detection.
            # Use step_vec (not patrol_positions delta) because patrol_positions is already
            # after collision resolution — when wall-blocked the delta would be ~0.
            raw_intended_step = jnp.where(in_confuse[:, None], confuse_steps,
                                jnp.where(in_chase[:, None], chase_steps, step_vec))

            # --- Existing slow-down gate ---
            enemy_move_tick = (state.step_counter % ENEMY_MOVE_EVERY) == 0
            desired_step = jnp.where(enemy_move_tick, desired_step, jnp.zeros_like(desired_step))

            new_enemy_positions = jax.vmap(move_enemy)(state.enemy_positions, desired_step)


            # --- Derive animation direction from actual movement ---
            # This ensures up/down sprites play when chasing north/south, not just during patrol
            actual_delta = new_enemy_positions - state.enemy_positions
            dx = actual_delta[:, 0]
            dy = actual_delta[:, 1]
            # DIR8: 0=E(1,0) 1=NE(1,-1) 2=N(0,-1) 3=NW(-1,-1) 4=W(-1,0) 5=SW(-1,1) 6=S(0,1) 7=SE(1,1)
            actual_dir8 = jnp.where(
                (dx > 0) & (dy > 0), 7,
                jnp.where((dx > 0) & (dy < 0), 1,
                jnp.where((dx < 0) & (dy > 0), 5,
                jnp.where((dx < 0) & (dy < 0), 3,
                jnp.where(dx > 0, 0,
                jnp.where(dx < 0, 4,
                jnp.where(dy > 0, 6,
                jnp.where(dy < 0, 2,
                enemy_dir
                ))))))))
            is_moving = jnp.any(actual_delta != 0, axis=1)
            new_enemy_dir = jnp.where(is_moving, actual_dir8, enemy_dir)
            trying_to_move = jnp.any(raw_intended_step != 0, axis=1)
            wall_blocked_now = trying_to_move & ~is_moving
            new_moving_on_tick = jnp.where(wall_blocked_now,
                                           jnp.zeros(NUM_ENEMIES, dtype=jnp.int32),
                                           jnp.ones(NUM_ENEMIES, dtype=jnp.int32))
            # Latch: only update on move-ticks; carry previous value on off-ticks
            enemy_moving_arr = jnp.where(enemy_move_tick, new_moving_on_tick, state.enemy_moving)

            # --- Stuck detection => confuse (only while chasing) ---
            moved = jnp.any(new_enemy_positions != state.enemy_positions, axis=1)
            stuck_counter = jnp.where(in_chase & (~moved), state.enemy_stuck_counter + 1, 0)

            stuck_trigger = (stuck_counter >= STUCK_TICKS) & enemy_alive
            confuse_timer = jnp.where(stuck_trigger, CONFUSE_TICKS, confuse_timer)
            chase_timer = jnp.where(stuck_trigger, 0, chase_timer)
            stuck_counter = jnp.where(stuck_trigger, 0, stuck_counter)
            
            
            # Item pickup detection
            def check_item_collision(item_pos):
                overlap_x = (new_x <= (item_pos[0] + ITEM_WIDTH - 1)) & ((new_x + self.consts.PLAYER_WIDTH - 1) >= item_pos[0])
                overlap_y = (new_y <= (item_pos[1] + ITEM_HEIGHT - 1)) & ((new_y + self.consts.PLAYER_HEIGHT - 1) >= item_pos[1])
                return overlap_x & overlap_y
            
            item_collisions = jax.vmap(check_item_collision)(state.item_positions)
            item_collisions = item_collisions & (state.item_active == 1)

            def check_ladder_collision(item_pos):
                overlap_x = (new_x <= (item_pos[0] + LADDER_WIDTH - 1)) & ((new_x + self.consts.PLAYER_WIDTH - 1) >= item_pos[0])
                overlap_y = (new_y <= (item_pos[1] + LADDER_HEIGHT - 1)) & ((new_y + self.consts.PLAYER_HEIGHT - 1) >= item_pos[1])
                return overlap_x & overlap_y

            ladder_collisions = jax.vmap(check_ladder_collision)(state.item_positions)
            ladder_collisions = ladder_collisions & (state.item_active == 1)

            # Checkpoint activation: touch updates respawn position, item is never removed
            hit_checkpoint_mask = item_collisions & (state.item_types == ITEM_CHECKPOINT)
            hit_checkpoint = jnp.any(hit_checkpoint_mask)
            hit_checkpoint_idx = jnp.argmax(hit_checkpoint_mask)
            new_checkpoint_idx = jnp.where(hit_checkpoint, hit_checkpoint_idx, state.checkpoint_idx)
            new_checkpoint_map = jnp.where(hit_checkpoint, state.map_index, state.checkpoint_map)
            new_checkpoint_level = jnp.where(hit_checkpoint, state.current_level, state.checkpoint_level)

            # Apply item effects
            collected_hearts = jnp.sum(item_collisions & (state.item_types == ITEM_HEART))
            collected_poison = jnp.sum(item_collisions & (state.item_types == ITEM_POISON))
            collected_traps = jnp.sum(item_collisions & (state.item_types == ITEM_TRAP))
            collected_shields = jnp.any(item_collisions & (state.item_types == ITEM_SHIELD))
            collected_guns = jnp.any(item_collisions & (state.item_types == ITEM_GUN))
            collected_bombs = jnp.sum(item_collisions & (state.item_types == ITEM_BOMB))
            collected_hammers = jnp.sum(item_collisions & (state.item_types == ITEM_HAMMER))
            collected_keys = jnp.any(item_collisions & (state.item_types == ITEM_KEY))
            # Health change mapping: heart +HEALTH_GAIN, poison -POISON_DAMAGE, trap -TRAP_DAMAGE
            health_change = (
                collected_hearts * self.consts.HEALTH_GAIN
                - collected_poison * self.consts.POISON_DAMAGE
                - collected_traps * self.consts.TRAP_DAMAGE
            )
            new_health = jnp.clip(state.health + health_change, 0, self.consts.MAX_HEALTH)
            
            # Update shield status
            new_shield_active = jnp.where(collected_shields, 1, state.shield_active)
            
            # Update gun status
            new_gun_active = jnp.where(collected_guns, 1, state.gun_active)
            
            # Update key status (persists until level change)
            new_has_key = jnp.where(collected_keys, 1, state.has_key)

            # Cage door interaction (requires key)
            collided_cage_door = jnp.any(item_collisions & (state.item_types == ITEM_CAGE_DOOR))
            cage_entry_pos = self.renderer.CAGE_ENTRY_POSITIONS[new_map_index, state.current_level]
            can_enter_cage = collided_cage_door & (new_has_key == 1)
            blocked_cage = collided_cage_door & (new_has_key == 0)
            new_x = jnp.where(can_enter_cage, cage_entry_pos[0], jnp.where(blocked_cage, state.player_x, new_x))
            new_y = jnp.where(can_enter_cage, cage_entry_pos[1], jnp.where(blocked_cage, state.player_y, new_y))

            # If the dedicated cage reward (slot 4) is collected, force an exit just below the cage door.
            # This prevents getting trapped in custom cage layouts after taking the reward.
            collected_cage_reward = item_collisions[4] & (state.item_types[4] == self.renderer.CAGE_REWARD_TYPE)
            cage_door_pos = self.renderer.CAGE_DOOR_POSITIONS[new_map_index, state.current_level]
            cage_exit_x = jnp.clip(
                cage_door_pos[0],
                0,
                self.consts.WORLD_WIDTH - self.consts.PLAYER_WIDTH,
            )
            cage_exit_y = jnp.clip(
                cage_door_pos[1] + self.renderer.CAGE_DOOR_SIZE + 2,
                0,
                self.consts.WORLD_HEIGHT - self.consts.PLAYER_HEIGHT,
            )
            new_x = jnp.where(collected_cage_reward, cage_exit_x, new_x)
            new_y = jnp.where(collected_cage_reward, cage_exit_y, new_y)
            
            # Update bomb count (capped at MAX_BOMBS)
            new_bomb_count = jnp.clip(state.bomb_count + collected_bombs, 0, MAX_BOMBS)

            # Update hammer count (capped at MAX_HAMMERS)
            new_hammer_count = jnp.clip(state.hammer_count + collected_hammers, 0, MAX_HAMMERS)

            # Ladder interaction: check if player is standing on a ladder
            on_ladder_up = jnp.any(ladder_collisions & (state.item_types == ITEM_LADDER_UP))
            on_ladder_down = jnp.any(ladder_collisions & (state.item_types == ITEM_LADDER_DOWN))
            on_any_ladder = on_ladder_up | on_ladder_down
            ladder_transition_blocked = state.ladder_transition_cooldown > 0
            on_any_ladder_effective = on_any_ladder & (~ladder_transition_blocked)
            
            # Update ladder timer: increment if on ladder, reset if not
            new_ladder_timer = jnp.where(
                on_any_ladder_effective,
                state.ladder_timer + 1,
                jnp.array(0, dtype=jnp.int32)
            )
            
            # Level transition when timer reaches threshold
            should_change_level = new_ladder_timer >= LADDER_INTERACTION_TIME
            going_up = should_change_level & on_ladder_up
            going_down = should_change_level & on_ladder_down
            
            # Calculate new level (clamp to valid range)
            level_after_up = jnp.clip(state.current_level + 1, 0, MAX_LEVELS - 1)
            level_after_down = jnp.clip(state.current_level - 1, 0, MAX_LEVELS - 1)
            new_level = jnp.where(going_up, level_after_up,
                        jnp.where(going_down, level_after_down, state.current_level))
            
            # Reset ladder timer after transition
            new_ladder_timer = jnp.where(should_change_level, 0, new_ladder_timer)
            
            # Score mapping only for treasure items
            points_by_type = jnp.array([
                0,                    # 0 unused
                0,                    # 1 HEART
                0,                    # 2 POISON
                0,                    # 3 TRAP
                100,                  # 4 STRONGBOX
                500,                  # 5 AMBER CHALICE
                1000,                 # 6 AMULET
                3000,                 # 7 GOLD CHALICE
                0,                    # 8 SHIELD
                0,                    # 9 GUN
                0,                    # 10 BOMB
                0,                    # 11 KEY
                0,                    # 12 LADDER_UP
                0,                    # 13 LADDER_DOWN
                0,                    # 14 CAGE_DOOR (no points)
                0,                    # 15 SPEED_POTION
                0,                    # 16 HEAL_POTION
                0,                    # 17 POISON_POTION
                0,                    # 18 HAMMER
                0,                    # 19 CHECKPOINT
            ], dtype=jnp.int32)
            item_points = points_by_type[state.item_types]
            gained_points = jnp.sum(item_points * item_collisions.astype(jnp.int32))
            new_score = state.score + gained_points
            
            # Remove collected items; ladders, cage door (conditional), and checkpoints persist
            is_ladder = (state.item_types == ITEM_LADDER_UP) | (state.item_types == ITEM_LADDER_DOWN)
            is_cage_door = (state.item_types == ITEM_CAGE_DOOR)
            is_checkpoint = (state.item_types == ITEM_CHECKPOINT)
            should_remove_cage = is_cage_door & can_enter_cage
            should_remove = (item_collisions & (~is_ladder) & (~is_cage_door) & (~is_checkpoint)) | should_remove_cage
            new_item_active = jnp.where(should_remove, 0, state.item_active)
            
            # Initialize positions/types for potential updates from drops
            new_item_positions_after_drops = state.item_positions
            new_item_types_after_drops = state.item_types
            
            # Bullet-enemy collision detection
            def check_bullet_enemy_collision(bullet_pos, enemy_pos):
                """Check if bullet hits enemy."""
                b_overlap_x = (bullet_pos[0] <= (enemy_pos[0] + self.consts.ENEMY_WIDTH - 1)) & \
                             ((bullet_pos[0] + BULLET_WIDTH - 1) >= enemy_pos[0])
                b_overlap_y = (bullet_pos[1] <= (enemy_pos[1] + self.consts.ENEMY_HEIGHT - 1)) & \
                             ((bullet_pos[1] + BULLET_HEIGHT - 1) >= enemy_pos[1])
                return b_overlap_x & b_overlap_y
            
            # Vectorized collision for all bullet-enemy pairs
            def check_all_enemies_for_bullet(bullet_idx):
                bullet_pos = final_bullet_positions[bullet_idx]
                is_active = final_bullet_active[bullet_idx] == 1
                collisions = jax.vmap(lambda e_pos: check_bullet_enemy_collision(bullet_pos, e_pos))(new_enemy_positions)
                return collisions & is_active & (state.enemy_active == 1)
            
            # Check all bullets against all enemies
            all_collisions = jax.vmap(check_all_enemies_for_bullet)(jnp.arange(MAX_BULLETS))
            
            # Any bullet hit per enemy
            bullet_hits_enemies = jnp.any(all_collisions, axis=0) & (state.enemy_active == 1)
            
            # Bomb detonation: kill enemies within radius when bomb is used
            # Calculate distance from player to each enemy
            player_center_x = new_x + self.consts.PLAYER_WIDTH // 2
            player_center_y = new_y + self.consts.PLAYER_HEIGHT // 2
            enemy_center_x = new_enemy_positions[:, 0] + self.consts.ENEMY_WIDTH // 2
            enemy_center_y = new_enemy_positions[:, 1] + self.consts.ENEMY_HEIGHT // 2
            dx = enemy_center_x - player_center_x
            dy = enemy_center_y - player_center_y
            distance_sq = dx * dx + dy * dy
            within_radius = distance_sq <= (BOMB_RADIUS * BOMB_RADIUS)
            bomb_kills_enemies = should_detonate_bomb & within_radius & (state.enemy_active == 1)

            # Hammer detonation: instantly kill all enemies within HAMMER_RADIUS
            within_hammer_radius = distance_sq <= (HAMMER_RADIUS * HAMMER_RADIUS)
            hammer_kills_enemies = should_use_hammer & within_hammer_radius & (state.enemy_active == 1)

            # Poison cloud damage: apply gradual damage to enemies within radius
            # Only apply poison damage if cloud timer is active AND on damage interval (every 30 steps)
            # This prevents instant kills and makes it a gradual damage-over-time effect
            cloud_active = state.poison_cloud_timer > 0
            damage_tick = (state.step_counter % self.consts.POISON_DAMAGE_INTERVAL) == 0
            poison_center_x = state.poison_cloud_x
            poison_center_y = state.poison_cloud_y
            
            # Calculate distance from poison cloud to each enemy
            dx_poison = enemy_center_x - poison_center_x
            dy_poison = enemy_center_y - poison_center_y
            distance_sq_poison = dx_poison * dx_poison + dy_poison * dy_poison
            within_poison_radius = distance_sq_poison <= (self.consts.POISON_RADIUS * self.consts.POISON_RADIUS)
            # Poison hits enemies (decrements type by 1) every damage interval, NOT instant kill
            poison_hits_enemies = cloud_active & damage_tick & within_poison_radius & (state.enemy_active == 1)
            
            # Reduce bomb count when detonated
            bomb_count_after_detonation = jnp.where(should_detonate_bomb, new_bomb_count - 1, new_bomb_count)

            # Reduce hammer count when used
            hammer_count_after_use = jnp.where(should_use_hammer, new_hammer_count - 1, new_hammer_count)

            instant_kills = bomb_kills_enemies | hammer_kills_enemies
            non_instant_hits = (bullet_hits_enemies | poison_hits_enemies) & (~instant_kills) & (state.enemy_active == 1)
            enemy_hitpoints_after_hit = jnp.where(
                non_instant_hits,
                jnp.maximum(state.enemy_hitpoints - 1, 0),
                state.enemy_hitpoints,
            ).astype(jnp.int32)
            enemy_killed_by_hits = non_instant_hits & (enemy_hitpoints_after_hit <= 0) & (state.enemy_types == ENEMY_ZOMBIE)
            tier_dropped = non_instant_hits & (enemy_hitpoints_after_hit <= 0) & (state.enemy_types > ENEMY_ZOMBIE)

            # Enemy mutation system: every displayed enemy tier takes two hits.
            new_enemy_types = jnp.where(
                instant_kills | enemy_killed_by_hits,
                0,
                jnp.where(tier_dropped, state.enemy_types - 1, state.enemy_types),
            ).astype(jnp.int32)
            new_enemy_hitpoints = jnp.where(
                instant_kills | (new_enemy_types <= 0),
                0,
                jnp.where(tier_dropped, ENEMY_HITS_PER_TIER, enemy_hitpoints_after_hit),
            ).astype(jnp.int32)
            
            # Zombies do not drop tier; they die at hitpoint break, so score on actual zombie death.
            zombies_killed = enemy_killed_by_hits & (state.enemy_types == ENEMY_ZOMBIE)
            
            # Instant kills award points for all enemy types; bullet/poison only score when a tier actually breaks
            zombies_instant = instant_kills & (state.enemy_types == ENEMY_ZOMBIE)
            wraiths_instant = instant_kills & (state.enemy_types == ENEMY_WRAITH)
            skeletons_instant = instant_kills & (state.enemy_types == ENEMY_SKELETON)
            wizards_instant = instant_kills & (state.enemy_types == ENEMY_WIZARD)
            grim_reapers_instant = instant_kills & (state.enemy_types == ENEMY_GRIM_REAPER)
            wraiths_hit = tier_dropped & (state.enemy_types == ENEMY_WRAITH)
            skeletons_hit = tier_dropped & (state.enemy_types == ENEMY_SKELETON)
            wizards_hit = tier_dropped & (state.enemy_types == ENEMY_WIZARD)
            grim_reapers_hit = tier_dropped & (state.enemy_types == ENEMY_GRIM_REAPER)
            
            enemy_kill_score = (
                jnp.sum(zombies_killed | zombies_instant) * self.consts.ZOMBIE_POINTS +
                jnp.sum(wraiths_hit | wraiths_instant) * self.consts.WRAITH_POINTS +
                jnp.sum(skeletons_hit | skeletons_instant) * self.consts.SKELETON_POINTS +
                jnp.sum(wizards_hit | wizards_instant) * self.consts.WIZARD_POINTS +
                jnp.sum(grim_reapers_hit | grim_reapers_instant) * self.consts.GRIM_REAPER_POINTS
            )
            
            # Deactivate enemies that have been killed (type becomes 0)
            new_enemy_active = jnp.where(new_enemy_types <= 0, 0, state.enemy_active)
            
            # Drop items from killed zombies (20% chance)
            zombies_just_killed = (state.enemy_active == 1) & (new_enemy_active == 0) & (state.enemy_types == ENEMY_ZOMBIE)
            
            # Generate random values for each enemy to determine if they drop items
            rng, drop_rng = jax.random.split(rng)
            drop_chances = jax.random.uniform(drop_rng, shape=(NUM_ENEMIES,))
            should_drop = zombies_just_killed & (drop_chances < 0.05)  # 5% chance
            
            # Find inactive item slots to spawn drops
            def try_spawn_item_drop(idx, carry):
                rng, item_active, item_positions, item_types = carry
                
                # Check if this enemy should drop an item
                drop_item = should_drop[idx]
                
                # Find first inactive item slot
                inactive_slots = (item_active == 0)
                has_slot = jnp.any(inactive_slots)
                first_slot = jnp.argmax(inactive_slots)  # First True index
                
                # Generate random item type (only items with sprites)
                rng, item_type_rng = jax.random.split(rng)
                drop_types = jnp.array([
                    ITEM_HEART,
                    ITEM_POISON if self.consts.ENABLE_DEFAULT_POISON_SPAWN else ITEM_HEART,
                    ITEM_TRAP if self.consts.ENABLE_DEFAULT_TRAP_SPAWN else ITEM_HEART,
                    ITEM_AMBER_CHALICE, ITEM_AMULET,
                    ITEM_SHIELD, ITEM_GUN, ITEM_BOMB
                ], dtype=jnp.int32)
                random_item_type = jax.random.choice(item_type_rng, drop_types)
                
                # Spawn item at enemy position
                enemy_pos = state.enemy_positions[idx]
                
                # Only spawn if enemy drops, there's a slot, and enemy was at valid position
                should_spawn = drop_item & has_slot & (enemy_pos[0] > 0) & (enemy_pos[1] > 0)
                
                item_active = jax.lax.cond(
                    should_spawn,
                    lambda ia: ia.at[first_slot].set(1),
                    lambda ia: ia,
                    item_active
                )
                
                item_positions = jax.lax.cond(
                    should_spawn,
                    lambda ip: ip.at[first_slot].set(enemy_pos),
                    lambda ip: ip,
                    item_positions
                )
                
                item_types = jax.lax.cond(
                    should_spawn,
                    lambda it: it.at[first_slot].set(random_item_type),
                    lambda it: it,
                    item_types
                )
                
                return rng, item_active, item_positions, item_types
            
            # Process all enemies for item drops
            _, new_item_active_after_drops, new_item_positions_after_drops, new_item_types_after_drops = jax.lax.fori_loop(
                0, NUM_ENEMIES,
                try_spawn_item_drop,
                (rng, new_item_active, state.item_positions, state.item_types)
            )
            
            # Move dead enemies off-screen
            final_enemy_positions = jnp.where(
                new_enemy_active[:, None] == 1,
                new_enemy_positions,
                jnp.array([0, 0])
            )
            
            # Deactivate bullets that hit enemies
            bullet_hit_enemy = jnp.any(all_collisions, axis=1)
            
            # Bullet-spawner collision detection
            def check_bullet_spawner_collision(bullet_pos, spawner_pos):
                """Check if bullet hits spawner."""
                b_overlap_x = (bullet_pos[0] <= (spawner_pos[0] + SPAWNER_WIDTH - 1)) & \
                             ((bullet_pos[0] + BULLET_WIDTH - 1) >= spawner_pos[0])
                b_overlap_y = (bullet_pos[1] <= (spawner_pos[1] + SPAWNER_HEIGHT - 1)) & \
                             ((bullet_pos[1] + BULLET_HEIGHT - 1) >= spawner_pos[1])
                return b_overlap_x & b_overlap_y
            
            def check_all_spawners_for_bullet(bullet_idx):
                bullet_pos = final_bullet_positions[bullet_idx]
                is_active = final_bullet_active[bullet_idx] == 1
                collisions = jax.vmap(lambda s_pos: check_bullet_spawner_collision(bullet_pos, s_pos))(state.spawner_positions)
                return collisions & is_active & (state.spawner_active == 1)
            
            spawner_collisions = jax.vmap(check_all_spawners_for_bullet)(jnp.arange(MAX_BULLETS))
            spawner_hit = jnp.any(spawner_collisions, axis=0)
            
            # Reduce spawner health on hit
            new_spawner_health = jnp.where(spawner_hit, state.spawner_health - 1, state.spawner_health)
            new_spawner_active = jnp.where(new_spawner_health <= 0, 0, state.spawner_active)
            
            # Spawn item where spawner was destroyed
            spawner_destroyed = (state.spawner_active == 1) & (new_spawner_active == 0)
            
            # Find first inactive item slot for spawner drops
            def add_spawner_drop(carry, spawner_idx):
                item_pos, item_types_arr, item_active_arr, key = carry
                destroyed = spawner_destroyed[spawner_idx]
                
                # Find first inactive slot
                first_inactive = jnp.argmax(item_active_arr == 0)
                can_add = jnp.any(item_active_arr == 0) & destroyed
                
                # Random item type (heart or treasures with sprites only)
                key, subkey = jax.random.split(key)
                drop_types = jnp.array([ITEM_HEART, ITEM_AMBER_CHALICE, ITEM_AMULET], dtype=jnp.int32)
                drop_type = jax.random.choice(subkey, drop_types)
                
                # Use spawner position (already placed off walls during reset)
                spawner_pos = state.spawner_positions[spawner_idx]
                
                # Update arrays
                new_pos = jnp.where(
                    (jnp.arange(NUM_ITEMS)[:, None] == first_inactive) & can_add,
                    spawner_pos,
                    item_pos
                )
                new_types = jnp.where(
                    (jnp.arange(NUM_ITEMS) == first_inactive) & can_add,
                    drop_type,
                    item_types_arr
                )
                new_active = jnp.where(
                    (jnp.arange(NUM_ITEMS) == first_inactive) & can_add,
                    1,
                    item_active_arr
                )
                
                return (new_pos, new_types, new_active, key), None
            
            rng, subkey = jax.random.split(rng)
            (final_item_positions, final_item_types, final_item_active, rng), _ = jax.lax.scan(
                add_spawner_drop,
                (new_item_positions_after_drops, new_item_types_after_drops, new_item_active_after_drops, subkey),
                jnp.arange(NUM_SPAWNERS)
            )
            
            # Deactivate bullets that hit anything
            bullet_hit_spawner = jnp.any(spawner_collisions, axis=1)
            final_bullet_active2 = final_bullet_active & (~(bullet_hit_enemy | bullet_hit_spawner)).astype(jnp.int32)

            # Bullets must never persist across chamber/level transitions.
            level_changed_early = new_level != state.current_level
            map_changed_early = new_map_index != state.map_index
            world_changed_early = level_changed_early | map_changed_early

            transition_bullet_positions = jnp.where(
                world_changed_early,
                jnp.zeros((MAX_BULLETS, 4), dtype=jnp.int32),
                final_bullet_positions,
            )
            transition_bullet_active = jnp.where(
                world_changed_early,
                jnp.zeros((MAX_BULLETS,), dtype=jnp.int32),
                final_bullet_active2,
            )
            transition_enemy_bullet_positions = jnp.where(
                world_changed_early,
                jnp.zeros((ENEMY_MAX_BULLETS, 4), dtype=jnp.int32),
                final_enemy_bullet_positions,
            )
            transition_enemy_bullet_active = jnp.where(
                world_changed_early,
                jnp.zeros((ENEMY_MAX_BULLETS,), dtype=jnp.int32),
                final_enemy_bullet_active,
            )
            
            # Add enemy kill score to total
            final_score = new_score + enemy_kill_score
            
            # Enemy contact damage (varies by enemy type)
            alive_enemies_mask = new_enemy_active == 1
            enemy_x = final_enemy_positions[:, 0]
            enemy_y = final_enemy_positions[:, 1]
            overlap_x = (new_x <= (enemy_x + self.consts.ENEMY_WIDTH - 1)) & ((new_x + self.consts.PLAYER_WIDTH - 1) >= enemy_x)
            overlap_y = (new_y <= (enemy_y + self.consts.ENEMY_HEIGHT - 1)) & ((new_y + self.consts.PLAYER_HEIGHT - 1) >= enemy_y)
            enemy_contacts = overlap_x & overlap_y & alive_enemies_mask

            # Enemy takes damage when contacting player (same tier-drop system as bullets)
            # Each enemy has its own cooldown to prevent damage every single frame
            ENEMY_CONTACT_COOLDOWN_TICKS = 15  # ~0.5s at 30 FPS between contact hits
            contact_cooldown_expired = state.enemy_contact_cooldown == 0
            contact_hits_enemies = enemy_contacts & contact_cooldown_expired
            contact_enemy_hitpoints_after = jnp.where(
                contact_hits_enemies,
                jnp.maximum(new_enemy_hitpoints - 1, 0),
                new_enemy_hitpoints,
            ).astype(jnp.int32)
            contact_enemy_killed = contact_hits_enemies & (contact_enemy_hitpoints_after <= 0) & (new_enemy_types == ENEMY_ZOMBIE)
            contact_tier_dropped = contact_hits_enemies & (contact_enemy_hitpoints_after <= 0) & (new_enemy_types > ENEMY_ZOMBIE)
            new_enemy_types = jnp.where(
                contact_enemy_killed,
                0,
                jnp.where(contact_tier_dropped, new_enemy_types - 1, new_enemy_types),
            ).astype(jnp.int32)
            new_enemy_hitpoints = jnp.where(
                contact_tier_dropped,
                ENEMY_HITS_PER_TIER,
                jnp.where(contact_enemy_killed | (new_enemy_types <= 0), 0, contact_enemy_hitpoints_after),
            ).astype(jnp.int32)
            new_enemy_active = jnp.where(new_enemy_types <= 0, 0, new_enemy_active)
            # Move newly-dead enemies off-screen
            final_enemy_positions = jnp.where(
                new_enemy_active[:, None] == 1,
                final_enemy_positions,
                jnp.array([0, 0]),
            )
            # Update per-enemy contact cooldown
            new_enemy_contact_cooldown = jnp.where(
                contact_hits_enemies,
                ENEMY_CONTACT_COOLDOWN_TICKS,
                jnp.maximum(state.enemy_contact_cooldown - 1, 0),
            ).astype(jnp.int32)

            # Calculate damage based on enemy type
            damage_by_type = jnp.array([
                0,  # Dead enemy (type 0)
                self.consts.ZOMBIE_DAMAGE,
                self.consts.WRAITH_DAMAGE,
                self.consts.SKELETON_DAMAGE,
                self.consts.WIZARD_DAMAGE,
                self.consts.GRIM_REAPER_DAMAGE
            ], dtype=jnp.int32)
            
            cooldown0 = state.damage_cooldown
            cooldown1 = jnp.maximum(cooldown0 - 1, 0)
            can_take_damage = cooldown0 == 0

            contact_damage_each = jnp.where(
                enemy_contacts,
                damage_by_type[new_enemy_types],
                0
            )
            raw_damage = jnp.max(contact_damage_each)  # take only the strongest single hit this tick
            applied_damage = jnp.where(can_take_damage, raw_damage, 0)


            HIT_COOLDOWN_TICKS = 8

            shield_multiplier = jnp.where(new_shield_active == 1, 0.5, 1.0)
            reduced_damage = (applied_damage * shield_multiplier).astype(jnp.int32)

            
            # Enemy bullet damage to player only
            def enemy_bullet_hits_player(bpos):
                bx, by = bpos[0], bpos[1]
                overlap_x = (bx <= (new_x + self.consts.PLAYER_WIDTH - 1)) & ((bx + ENEMY_BULLET_WIDTH - 1) >= new_x)
                overlap_y = (by <= (new_y + self.consts.PLAYER_HEIGHT - 1)) & ((by + ENEMY_BULLET_HEIGHT - 1) >= new_y)
                return overlap_x & overlap_y
            enemy_bullet_hits = jax.vmap(enemy_bullet_hits_player)(transition_enemy_bullet_positions[:, :2]) & (transition_enemy_bullet_active == 1)
            enemy_bullet_damage = enemy_bullet_hits.astype(jnp.int32).any().astype(jnp.int32)
            applied_bullet_damage = jnp.where(can_take_damage, enemy_bullet_damage, 0)

            final_damage = jnp.maximum(applied_damage, applied_bullet_damage)

            final_health = jnp.clip(new_health - final_damage, 0, self.consts.MAX_HEALTH)

            cooldown2 = jnp.where(
                final_damage > 0,
                HIT_COOLDOWN_TICKS,
                cooldown1,
            )


            final_enemy_bullet_active3 = transition_enemy_bullet_active & (~enemy_bullet_hits).astype(jnp.int32)
            
            # Enemy spawning when crossing portals or changing levels
            
            # Spawner logic: spawn enemies only from active spawners
            new_spawner_timers = state.spawner_timers - 1
            should_spawn_enemy = (new_spawner_timers <= 0) & (new_spawner_active == 1)

            spawner_affected_types = new_enemy_types
            spawner_affected_active = new_enemy_active
            spawner_affected_hitpoints = new_enemy_hitpoints
            spawner_affected_timers = new_wizard_timers
            
            # spawn each enemy exactly in the middle of the spawner (or not at all)
            def try_spawn_from_spawner(carry, spawner_idx):
                enemy_pos, enemy_types_arr, enemy_active_arr, enemy_hp_arr, timers_arr, key = carry
                should_spawn = should_spawn_enemy[spawner_idx]
                
                # Find first inactive enemy slot
                first_inactive = jnp.argmax(enemy_active_arr == 0)
                can_spawn = jnp.any(enemy_active_arr == 0) & should_spawn
                
                # Random enemy type (2-4/5: wraith, skeleton, wizard, optional grim reaper)
                key, subkey = jax.random.split(key)
                enemy_type_high = jnp.where(
                    jnp.array(self.consts.ENABLE_GRIM_REAPER_ENEMIES),
                    ENEMY_GRIM_REAPER + 1,
                    ENEMY_WIZARD + 1,
                )
                spawn_type = jax.random.randint(subkey, (), ENEMY_WRAITH, enemy_type_high, dtype=jnp.int32)
                
                # Random wizard timer for new enemy (in case it's a wizard)
                key, subkey = jax.random.split(key)
                new_timer = jax.random.randint(subkey, (), 0, WIZARD_SHOOT_INTERVAL + WIZARD_SHOOT_OFFSET_MAX, dtype=jnp.int32)
                
                spawner_pos = state.spawner_positions[spawner_idx]
                
                # Use destination world walls so a portal/map change in this same step
                # cannot place a newly spawned enemy into walls of the new map.
                WALLS_SPAWNER = jnp.concatenate(
                    [
                        self.renderer.LEVEL_WALLS[new_map_index, new_level],
                        self.renderer.BOUNDARY_WALLS,
                    ],
                    axis=0,
                )
                
                # Spawn enemy directly centered in the spawner - NO FALLBACK
                spawn_offset_x = (SPAWNER_WIDTH - self.consts.ENEMY_WIDTH) // 2
                spawn_offset_y = (SPAWNER_HEIGHT - self.consts.ENEMY_HEIGHT) // 2
                spawn_pos = spawner_pos + jnp.array([spawn_offset_x, spawn_offset_y])
                
                # Check if centered position overlaps with walls - if yes, DON'T spawn
                wx = WALLS_SPAWNER[:, 0]
                wy = WALLS_SPAWNER[:, 1]
                ww = WALLS_SPAWNER[:, 2]
                wh = WALLS_SPAWNER[:, 3]
                overlap_x = (spawn_pos[0] <= (wx + ww - 1)) & ((spawn_pos[0] + self.consts.ENEMY_WIDTH - 1) >= wx)
                overlap_y = (spawn_pos[1] <= (wy + wh - 1)) & ((spawn_pos[1] + self.consts.ENEMY_HEIGHT - 1) >= wy)
                pos_on_wall = jnp.any(overlap_x & overlap_y)
                
                # Only spawn if position is valid (not on wall)
                can_spawn_at_pos = can_spawn & (~pos_on_wall)
                
                # Update arrays
                new_pos = jnp.where(
                    (jnp.arange(NUM_ENEMIES)[:, None] == first_inactive) & can_spawn_at_pos,
                    spawn_pos,
                    enemy_pos
                )
                new_types = jnp.where(
                    (jnp.arange(NUM_ENEMIES) == first_inactive) & can_spawn_at_pos,
                    spawn_type,
                    enemy_types_arr
                )
                new_active = jnp.where(
                    (jnp.arange(NUM_ENEMIES) == first_inactive) & can_spawn_at_pos,
                    1,
                    enemy_active_arr
                )
                new_hitpoints = jnp.where(
                    (jnp.arange(NUM_ENEMIES) == first_inactive) & can_spawn_at_pos,
                    ENEMY_HITS_PER_TIER,
                    enemy_hp_arr
                )
                new_timers = jnp.where(
                    (jnp.arange(NUM_ENEMIES) == first_inactive) & can_spawn_at_pos,
                    new_timer,
                    timers_arr
                )
                
                return (new_pos, new_types, new_active, new_hitpoints, new_timers, key), None

            (enemy_positions_after_spawner, enemy_types_after_spawner, enemy_active_after_spawner, enemy_hitpoints_after_spawner, wizard_timers_after_spawner, rng), _ = jax.lax.scan(
                try_spawn_from_spawner,
                (new_enemy_positions, spawner_affected_types, spawner_affected_active, spawner_affected_hitpoints, spawner_affected_timers, rng),
                jnp.arange(NUM_SPAWNERS)
            )
            
            # Reset spawner timers when they spawn
            final_spawner_timers = jnp.where(should_spawn_enemy, SPAWNER_SPAWN_INTERVAL, new_spawner_timers)
            
            # Level transition: respawn items and reset player position when level OR map changes
            level_changed = new_level != state.current_level
            map_changed = new_map_index != state.map_index
            world_changed = level_changed | map_changed
            
            # Reset player position on level/map change
            # Keep level-transition respawn near top-middle to avoid upper-left placement.
            spawn_x_up = jnp.array((self.consts.WORLD_WIDTH - self.consts.PLAYER_WIDTH) // 2, dtype=jnp.int32)
            spawn_y_up = jnp.array(70, dtype=jnp.int32)
            spawn_x_down = jnp.array((self.consts.WORLD_WIDTH - self.consts.PLAYER_WIDTH) // 2, dtype=jnp.int32)
            spawn_y_down = jnp.array(70, dtype=jnp.int32)

            WALLS_SPAWN = jnp.concatenate(
                [
                    self.renderer.LEVEL_WALLS[new_map_index, new_level],
                    self.renderer.BOUNDARY_WALLS,
                ],
                axis=0,
            )

            def spawn_overlaps_or_too_close_to_wall(pos: chex.Array) -> chex.Array:
                # Require a tiny clearance around the player so they don't spawn glued to a wall.
                clearance = jnp.array(2, dtype=jnp.int32)
                px = pos[0] - clearance
                py = pos[1] - clearance
                pw = self.consts.PLAYER_WIDTH + 2 * clearance
                ph = self.consts.PLAYER_HEIGHT + 2 * clearance
                wx = WALLS_SPAWN[:, 0]
                wy = WALLS_SPAWN[:, 1]
                ww = WALLS_SPAWN[:, 2]
                wh = WALLS_SPAWN[:, 3]
                overlap_x = (px <= (wx + ww - 1)) & ((px + pw - 1) >= wx)
                overlap_y = (py <= (wy + wh - 1)) & ((py + ph - 1) >= wy)
                return jnp.any(overlap_x & overlap_y)

            spawn_offsets = jnp.array([
                [0, 0],
                [8, 0], [-8, 0], [0, 8], [0, -8],
                [16, 0], [-16, 0], [0, 16], [0, -16],
                [24, 0], [-24, 0], [0, 24], [0, -24],
                [8, 8], [8, -8], [-8, 8], [-8, -8],
                [16, 16], [16, -16], [-16, 16], [-16, -16],
                [32, 0], [-32, 0], [0, 32], [0, -32],
            ], dtype=jnp.int32)

            def find_safe_spawn(base_x: chex.Array, base_y: chex.Array) -> chex.Array:
                base = jnp.array([base_x, base_y], dtype=jnp.int32)

                def try_candidate(i, carry):
                    best_pos, found = carry
                    cand = base + spawn_offsets[i]
                    cand = jnp.array([
                        jnp.clip(cand[0], 1, self.consts.WORLD_WIDTH - self.consts.PLAYER_WIDTH - 1),
                        jnp.clip(cand[1], 1, self.consts.WORLD_HEIGHT - self.consts.PLAYER_HEIGHT - 1),
                    ], dtype=jnp.int32)
                    blocked = spawn_overlaps_or_too_close_to_wall(cand)
                    can_use = (~blocked) & (~found)
                    best_pos = jnp.where(can_use, cand, best_pos)
                    found = found | (~blocked)
                    return (best_pos, found)

                initial_blocked = spawn_overlaps_or_too_close_to_wall(base)
                init_found = ~initial_blocked
                init_best = base
                best_pos, _ = jax.lax.fori_loop(0, spawn_offsets.shape[0], try_candidate, (init_best, init_found))
                return best_pos

            safe_spawn_up = find_safe_spawn(spawn_x_up, spawn_y_up)
            safe_spawn_down = find_safe_spawn(spawn_x_down, spawn_y_down)
            
            # Determine spawn position based on which ladder was used or map portal
            # For map portal transitions (left/right exit), use portal position
            spawn_x = jnp.where(going_up, safe_spawn_up[0],
                      jnp.where(going_down, safe_spawn_down[0], new_x))  # Use new_x for portal transitions
            spawn_y = jnp.where(going_up, safe_spawn_up[1],
                      jnp.where(going_down, safe_spawn_down[1], new_y))  # Use new_y for portal transitions
            
            transition_x = jnp.where(world_changed, spawn_x, new_x)
            transition_y = jnp.where(world_changed, spawn_y, new_y)
            
            # Respawn all items on level change
            def respawn_items_for_level(key):
                # New level and map walls for collision checks
                WALLS_NEW = self.renderer.LEVEL_WALLS[new_map_index, new_level]
                
                def check_wall_overlap_item(x, y):
                    wx = WALLS_NEW[:, 0]
                    wy = WALLS_NEW[:, 1]
                    ww = WALLS_NEW[:, 2]
                    wh = WALLS_NEW[:, 3]
                    # Use max item size to be safe
                    overlap_x = (x <= (wx + ww - 1)) & ((x + 13 - 1) >= wx)
                    overlap_y = (y <= (wy + wh - 1)) & ((y + 13 - 1) >= wy)
                    return jnp.any(overlap_x & overlap_y)
                
                # Key: random valid position away from upper-left spawn area; ladders fixed
                KEY_EXCLUSION_RADIUS = 80
                def try_spawn_key(_, inner):
                    pos, key_loc, found = inner
                    key_loc, sk = jax.random.split(key_loc)
                    x = jax.random.randint(sk, (), 16, self.consts.WORLD_WIDTH - 24, dtype=jnp.int32)
                    key_loc, sk = jax.random.split(key_loc)
                    y = jax.random.randint(sk, (), 16, self.consts.WORLD_HEIGHT - 24, dtype=jnp.int32)
                    on_wall = check_wall_overlap_item(x, y)
                    dx = x - self.consts.PLAYER_START_X
                    dy = y - self.consts.PLAYER_START_Y
                    dist_sq = dx * dx + dy * dy
                    too_close = dist_sq < (KEY_EXCLUSION_RADIUS * KEY_EXCLUSION_RADIUS)
                    is_valid = (~on_wall) & (~too_close)
                    new_pos = jnp.where(is_valid & (~found), jnp.array([x, y]), pos)
                    new_found = found | is_valid
                    return (new_pos, key_loc, new_found)

                key, key_sk = jax.random.split(key)
                init_kpos = jnp.array([100, 100], dtype=jnp.int32)
                key_pos, key_sk, _ = jax.lax.fori_loop(0, 20, try_spawn_key, (init_kpos, key_sk, False))
                
                # Hardcoded exit ladder for visibility consistency across custom mazes
                exit_pos = jnp.array([74, 548], dtype=jnp.int32)
                
                # Hide ladder_down on level 0, otherwise spawn at safe position
                def try_spawn_ladder_down(_, inner):
                    pos, key_loc, found = inner
                    key_loc, sk = jax.random.split(key_loc)
                    x = jax.random.randint(sk, (), 20, 100, dtype=jnp.int32)
                    key_loc, sk = jax.random.split(key_loc)
                    y = jax.random.randint(sk, (), 40, 120, dtype=jnp.int32)
                    on_wall = check_wall_overlap_item(x, y)
                    new_pos = jnp.where((~on_wall) & (~found), jnp.array([x, y]), pos)
                    new_found = found | (~on_wall)
                    return (new_pos, key_loc, new_found)
                
                key, key_sk = jax.random.split(key)
                init_ladder = jnp.array([40, 70], dtype=jnp.int32)
                ladder_down_temp, key_sk, _ = jax.lax.fori_loop(0, 20, try_spawn_ladder_down, (init_ladder, key_sk, False))
                ladder_down_pos = jnp.where(new_level == 0, jnp.array([-1000, -1000], dtype=jnp.int32), ladder_down_temp)
                main_map_corner_gun_pos = jnp.array([140, 20], dtype=jnp.int32)
                cage_door_pos = self.renderer.CAGE_DOOR_POSITIONS[new_map_index, new_level]
                cage_reward_pos = self.renderer.CAGE_REWARD_POSITIONS[new_map_index, new_level]
                
                # Generate random positions for regular items with retries to avoid walls
                NUM_REGULAR = NUM_ITEMS - 5 - NUM_CHECKPOINTS
                def spawn_regular_item(carry, idx):
                    positions_arr, key_in = carry

                    def try_once(_, inner):
                        pos, key_loc, found = inner
                        key_loc, sk = jax.random.split(key_loc)
                        x = jax.random.randint(sk, (), 16, self.consts.WORLD_WIDTH - 24, dtype=jnp.int32)
                        key_loc, sk = jax.random.split(key_loc)
                        y = jax.random.randint(sk, (), 16, self.consts.WORLD_HEIGHT - 24, dtype=jnp.int32)
                        on_wall = check_wall_overlap_item(x, y)
                        new_pos = jnp.where((~on_wall) & (~found), jnp.array([x, y]), pos)
                        new_found = found | (~on_wall)
                        return (new_pos, key_loc, new_found)

                    init = (jnp.array([30, 30], dtype=jnp.int32), key_in, False)
                    final_pos, key_out, _ = jax.lax.fori_loop(0, 20, try_once, init)
                    positions_arr = positions_arr.at[idx].set(final_pos)
                    return (positions_arr, key_out), None

                key, sk = jax.random.split(key)
                init_positions = jnp.zeros((NUM_REGULAR, 2), dtype=jnp.int32)
                (regular_positions, key), _ = jax.lax.scan(spawn_regular_item, (init_positions, sk), jnp.arange(NUM_REGULAR))

                # Dynamically spawn checkpoints spread across vertical zones, avoiding walls
                def spawn_one_checkpoint_level(carry, zone_idx):
                    positions_arr, key_in = carry
                    y_lo = zone_idx * self.consts.WORLD_HEIGHT // NUM_CHECKPOINTS
                    y_hi = (zone_idx + 1) * self.consts.WORLD_HEIGHT // NUM_CHECKPOINTS

                    def try_once_chk(_, inner):
                        pos, key_loc, found = inner
                        key_loc, sk = jax.random.split(key_loc)
                        x = jax.random.randint(sk, (), 16, self.consts.WORLD_WIDTH - 16, dtype=jnp.int32)
                        key_loc, sk = jax.random.split(key_loc)
                        y_min = jnp.maximum(y_lo, 8)
                        y_max = jnp.maximum(y_hi - 8, y_lo + 1)
                        y = jax.random.randint(sk, (), y_min, y_max, dtype=jnp.int32)
                        on_wall = check_wall_overlap_item(x, y)
                        new_pos = jnp.where((~on_wall) & (~found), jnp.array([x, y]), pos)
                        return (new_pos, key_loc, found | (~on_wall))

                    fallback = jnp.array([75, (y_lo + y_hi) // 2], dtype=jnp.int32)
                    final_pos, key_out, _ = jax.lax.fori_loop(0, 20, try_once_chk, (fallback, key_in, False))
                    positions_arr = positions_arr.at[zone_idx].set(final_pos)
                    return (positions_arr, key_out), None

                key, sk = jax.random.split(key)
                chk_init = jnp.zeros((NUM_CHECKPOINTS, 2), dtype=jnp.int32)
                (chk_positions, key), _ = jax.lax.scan(
                    spawn_one_checkpoint_level, (chk_init, sk), jnp.arange(NUM_CHECKPOINTS)
                )

                # Combine: first 5 are key, ladders, cage door and reward, then regular items, then checkpoints
                new_positions = jnp.concatenate([
                    key_pos[None, :],
                    exit_pos[None, :],
                    ladder_down_pos[None, :],
                    cage_door_pos[None, :],
                    cage_reward_pos[None, :],
                    regular_positions,
                    chk_positions,
                ], axis=0)

                # Generate new item types - only items with sprites (exclude STRONGBOX, GOLD_CHALICE)
                key, subkey = jax.random.split(key)
                all_item_types = jnp.array([
                    ITEM_HEART, ITEM_POISON, ITEM_TRAP,
                    ITEM_AMBER_CHALICE, ITEM_AMULET,
                    ITEM_SHIELD, ITEM_GUN, ITEM_BOMB,
                    ITEM_SPEED_POTION, ITEM_HEAL_POTION, ITEM_POISON_POTION,
                    ITEM_HAMMER,
                ], dtype=jnp.int32)
                spawn_probs = jnp.array([
                    0.18,
                    0.08 if self.consts.ENABLE_DEFAULT_POISON_SPAWN else 0.0,
                    0.15 if self.consts.ENABLE_DEFAULT_TRAP_SPAWN else 0.0,
                    0.12,
                    0.08,
                    0.07,
                    0.03,
                    0.07,
                    0.08 if self.consts.ENABLE_SPEED_POTION_SPAWN else 0.0,
                    0.08 if self.consts.ENABLE_HEAL_POTION_SPAWN else 0.0,
                    0.07 if self.consts.ENABLE_POISON_POTION_SPAWN else 0.0,
                    0.06 if self.consts.ENABLE_HAMMER_SPAWN else 0.0,
                ], dtype=jnp.float32)
                spawn_probs = spawn_probs / jnp.sum(spawn_probs)
                regular_items = jax.random.choice(subkey, all_item_types, shape=(NUM_REGULAR,), p=spawn_probs)
                new_types = jnp.concatenate([
                    jnp.array([ITEM_KEY, ITEM_LADDER_UP, ITEM_LADDER_DOWN, ITEM_CAGE_DOOR, self.renderer.CAGE_REWARD_TYPE]),
                    regular_items,
                    jnp.full((NUM_CHECKPOINTS,), ITEM_CHECKPOINT, dtype=jnp.int32),
                ])
                regular_item_mask = (jnp.arange(NUM_ITEMS) < (5 + INITIAL_REGULAR_ITEM_COUNT)).astype(jnp.int32)
                new_active = jnp.where(jnp.arange(NUM_ITEMS) < 5, 1, regular_item_mask)
                new_active = new_active.at[NUM_ITEMS - NUM_CHECKPOINTS:].set(
                    jnp.array(self.consts.ENABLE_INTRA_LEVEL_CHECKPOINTS, dtype=jnp.int32)
                )

                is_main_spawn_map = (new_map_index == 0) & (new_level == 0)
                new_positions = jnp.where(
                    is_main_spawn_map,
                    new_positions.at[5].set(main_map_corner_gun_pos),
                    new_positions,
                )
                new_types = jnp.where(
                    is_main_spawn_map,
                    new_types.at[5].set(jnp.array(ITEM_GUN, dtype=jnp.int32)),
                    new_types,
                )
                new_active = jnp.where(
                    is_main_spawn_map,
                    new_active.at[5].set(jnp.array(1, dtype=jnp.int32)),
                    new_active,
                )

                cage_valid_here = self.renderer.CAGE_VALID[new_map_index, new_level]
                new_active = new_active.at[3].set(cage_valid_here)
                new_active = new_active.at[4].set(cage_valid_here)
                
                return new_positions, new_types, new_active, key
            
            # Apply respawn if level OR map changed
            respawned_positions, respawned_types, respawned_active, rng = respawn_items_for_level(rng)
            transition_item_positions = jnp.where(world_changed, respawned_positions, final_item_positions)
            transition_item_types = jnp.where(world_changed, respawned_types, final_item_types)
            transition_item_active = jnp.where(world_changed, respawned_active, final_item_active)

            # Keep ladder-up hardcoded and always active; main chamber level-0 uses upper-left placement.
            ladder_up_default_pos = jnp.array([74, 548], dtype=jnp.int32)
            ladder_up_main_spawn_pos = jnp.array([12, 105], dtype=jnp.int32)
            ladder_up_target_pos = jnp.where(
                (new_map_index == 0) & (new_level == 0),
                ladder_up_main_spawn_pos,
                ladder_up_default_pos,
            )
            transition_item_positions = transition_item_positions.at[1].set(ladder_up_target_pos)
            transition_item_types = transition_item_types.at[1].set(jnp.array(ITEM_LADDER_UP, dtype=jnp.int32))
            transition_item_active = transition_item_active.at[1].set(jnp.array(1, dtype=jnp.int32))

            # Intra-level checkpoints persist across room changes (positions and collected state preserved)
            _chk_start = NUM_ITEMS - NUM_CHECKPOINTS
            transition_item_positions = transition_item_positions.at[_chk_start:].set(
                final_item_positions[_chk_start:]
            )
            transition_item_types = transition_item_types.at[_chk_start:].set(
                final_item_types[_chk_start:]
            )
            transition_item_active = transition_item_active.at[_chk_start:].set(
                final_item_active[_chk_start:]
            )

            disallowed_potions = (
                ((transition_item_types == ITEM_SPEED_POTION) & (~jnp.array(self.consts.ENABLE_SPEED_POTION_SPAWN)))
                | ((transition_item_types == ITEM_HEAL_POTION) & (~jnp.array(self.consts.ENABLE_HEAL_POTION_SPAWN)))
                | ((transition_item_types == ITEM_POISON_POTION) & (~jnp.array(self.consts.ENABLE_POISON_POTION_SPAWN)))
                | ((transition_item_types == ITEM_HAMMER) & (~jnp.array(self.consts.ENABLE_HAMMER_SPAWN)))
            )
            transition_item_types = jnp.where(disallowed_potions, ITEM_HEART, transition_item_types)
            
            # On level change, ensure enemies are not overlapping new level walls
            WALLS_NEWLVL = self.renderer.LEVEL_WALLS[new_map_index, new_level]
            def enemy_on_wall(enemy_pos):
                ex = enemy_pos[0]
                ey = enemy_pos[1]
                wx = WALLS_NEWLVL[:, 0]
                wy = WALLS_NEWLVL[:, 1]
                ww = WALLS_NEWLVL[:, 2]
                wh = WALLS_NEWLVL[:, 3]
                overlap_x = (ex <= (wx + ww - 1)) & ((ex + self.consts.ENEMY_WIDTH - 1) >= wx)
                overlap_y = (ey <= (wy + wh - 1)) & ((ey + self.consts.ENEMY_HEIGHT - 1) >= wy)
                return jnp.any(overlap_x & overlap_y)
            
            def relocate_enemy(carry, i):
                positions_arr, key_in = carry
                cur = enemy_positions_after_spawner[i]
                is_active = enemy_active_after_spawner[i] == 1
            
                def try_once(_, inner):
                    pos, key_loc, found = inner
                    key_loc, kx, ky = jax.random.split(key_loc, 3)
                    dx = jax.random.randint(kx, (), -12, 13, dtype=jnp.int32)
                    dy = jax.random.randint(ky, (), -12, 13, dtype=jnp.int32)
                    cand = cur + jnp.array([dx, dy])
                    cand = jnp.clip(cand, jnp.array([16,16]), jnp.array([self.consts.WORLD_WIDTH - 24, self.consts.WORLD_HEIGHT - 24]))
                    on_wall = enemy_on_wall(cand)
                    new_pos = jnp.where((~on_wall) & (~found), cand, pos)
                    new_found = found | (~on_wall)
                    return (new_pos, key_loc, new_found)
            
                init = (cur, key_in, False)
                final_pos, key_out, _ = jax.lax.fori_loop(0, 20, try_once, init)
                # If enemy inactive or world didn't change, keep original
                final_pos = jnp.where(world_changed & is_active, final_pos, cur)
                positions_arr = positions_arr.at[i].set(final_pos)
                return (positions_arr, key_out), None
            
            rng, sk = jax.random.split(rng)
            init_epos = enemy_positions_after_spawner
            (relocated_enemy_positions, rng), _ = jax.lax.scan(relocate_enemy, (init_epos, sk), jnp.arange(NUM_ENEMIES))
            
            # On level change going UP, spawn additional zombies
            def spawn_level_enemies(key_in, positions_in, types_in, active_in, hitpoints_in, timers_in):
                """Spawn 3-5 zombies when entering a new higher level."""
                rng_local, subkey = jax.random.split(key_in)
                num_to_spawn = 4  # Spawn 4 zombies per new level
                
                def spawn_one_zombie(carry, spawn_idx):
                    pos_arr, typ_arr, act_arr, hp_arr, tim_arr, key_loc = carry
                    # Find first inactive enemy slot
                    first_inactive_idx = jnp.argmax(act_arr == 0)
                    slot_available = jnp.any(act_arr == 0)
                    
                    # Random timer for new zombie (zombies don't shoot but keep consistent state)
                    key_loc, sk = jax.random.split(key_loc)
                    new_timer = jax.random.randint(sk, (), 0, WIZARD_SHOOT_INTERVAL + WIZARD_SHOOT_OFFSET_MAX, dtype=jnp.int32)
                    
                    # Random position avoiding walls
                    def try_pos(_, inner):
                        p, k, found = inner
                        k, kx = jax.random.split(k)
                        x = jax.random.randint(kx, (), 40, self.consts.WORLD_WIDTH - 40, dtype=jnp.int32)
                        k, ky = jax.random.split(k)
                        y = jax.random.randint(ky, (), 40, self.consts.WORLD_HEIGHT - 40, dtype=jnp.int32)
                        on_wall = enemy_on_wall(jnp.array([x, y]))
                        new_p = jnp.where((~on_wall) & (~found), jnp.array([x, y]), p)
                        new_found = found | (~on_wall)
                        return (new_p, k, new_found)
                    
                    init_p = jnp.array([100, 100], dtype=jnp.int32)
                    final_p, key_loc, _ = jax.lax.fori_loop(0, 15, try_pos, (init_p, key_loc, False))
                    
                    # Update arrays if slot available
                    pos_arr = jnp.where(
                        (jnp.arange(NUM_ENEMIES)[:, None] == first_inactive_idx) & slot_available,
                        final_p,
                        pos_arr
                    )
                    typ_arr = jnp.where(
                        (jnp.arange(NUM_ENEMIES) == first_inactive_idx) & slot_available,
                        ENEMY_ZOMBIE,
                        typ_arr
                    )
                    act_arr = jnp.where(
                        (jnp.arange(NUM_ENEMIES) == first_inactive_idx) & slot_available,
                        1,
                        act_arr
                    )
                    hp_arr = jnp.where(
                        (jnp.arange(NUM_ENEMIES) == first_inactive_idx) & slot_available,
                        ENEMY_HITS_PER_TIER,
                        hp_arr
                    )
                    tim_arr = jnp.where(
                        (jnp.arange(NUM_ENEMIES) == first_inactive_idx) & slot_available,
                        new_timer,
                        tim_arr
                    )
                    return (pos_arr, typ_arr, act_arr, hp_arr, tim_arr, key_loc), None
                
                (new_pos, new_typ, new_act, new_hp, new_tim, rng_local), _ = jax.lax.scan(
                    spawn_one_zombie,
                    (positions_in, types_in, active_in, hitpoints_in, timers_in, subkey),
                    jnp.arange(num_to_spawn)
                )
                return new_pos, new_typ, new_act, new_hp, new_tim, rng_local
            
            # Apply enemy spawning only if going up to a new level
            spawned_pos, spawned_typ, spawned_act, spawned_hp, spawned_tim, rng = jax.lax.cond(
                level_changed & going_up,
                lambda _: spawn_level_enemies(rng, relocated_enemy_positions, enemy_types_after_spawner, enemy_active_after_spawner, enemy_hitpoints_after_spawner, wizard_timers_after_spawner),
                lambda _: (relocated_enemy_positions, enemy_types_after_spawner, enemy_active_after_spawner, enemy_hitpoints_after_spawner, wizard_timers_after_spawner, rng),
                operand=None
            )
            
            final_enemy_positions_after_level = jax.vmap(apply_enemy_portal_keepout)(spawned_pos)
            final_enemy_types_after_level = spawned_typ
            final_enemy_active_after_level = spawned_act
            final_enemy_hitpoints_after_level = spawned_hp
            final_wizard_timers_after_level = spawned_tim
            
            # On level change, respawn spawners avoiding new level walls
            def respawn_spawners_for_level(key):
                # Use renderer-level walls structure with map index
                WALLS_NEW = jnp.concatenate(
                    [
                        self.renderer.LEVEL_WALLS[new_map_index, new_level],
                        self.renderer.BOUNDARY_WALLS,
                    ],
                    axis=0,
                )
            
                def check_wall_overlap_sp(x, y):
                    """Check if spawner position has space for enemy to spawn inside it."""
                    # Calculate where enemy would be centered in this spawner
                    spawn_offset_x = (SPAWNER_WIDTH - self.consts.ENEMY_WIDTH) // 2
                    spawn_offset_y = (SPAWNER_HEIGHT - self.consts.ENEMY_HEIGHT) // 2
                    enemy_x = x + spawn_offset_x
                    enemy_y = y + spawn_offset_y
                    
                    wx = WALLS_NEW[:, 0]
                    wy = WALLS_NEW[:, 1]
                    ww = WALLS_NEW[:, 2]
                    wh = WALLS_NEW[:, 3]
                    
                    # Check if enemy space overlaps with walls
                    overlap_x = (enemy_x <= (wx + ww - 1)) & ((enemy_x + self.consts.ENEMY_WIDTH - 1) >= wx)
                    overlap_y = (enemy_y <= (wy + wh - 1)) & ((enemy_y + self.consts.ENEMY_HEIGHT - 1) >= wy)
                    return jnp.any(overlap_x & overlap_y)
            
                def spawn_one(carry, i):
                    pos_arr, key_in = carry
            
                    def try_once(_, inner):
                        pos, key_loc, found = inner
                        key_loc, sk = jax.random.split(key_loc)
                        x = jax.random.randint(sk, (), 50, self.consts.WORLD_WIDTH - 50, dtype=jnp.int32)
                        key_loc, sk = jax.random.split(key_loc)
                        y = jax.random.randint(sk, (), 50, self.consts.WORLD_HEIGHT - 50, dtype=jnp.int32)
                        on_wall = check_wall_overlap_sp(x, y)
                        new_pos = jnp.where((~on_wall) & (~found), jnp.array([x, y]), pos)
                        new_found = found | (~on_wall)
                        return (new_pos, key_loc, new_found)
            
                    init = (jnp.array([100, 100], dtype=jnp.int32), key_in, False)
                    final_pos, key_out, _ = jax.lax.fori_loop(0, 20, try_once, init)
                    pos_arr = pos_arr.at[i].set(final_pos)
                    return (pos_arr, key_out), None
            
                key, sk = jax.random.split(rng)
                init_pos = jnp.zeros((NUM_SPAWNERS, 2), dtype=jnp.int32)
                (new_sp_positions, key), _ = jax.lax.scan(spawn_one, (init_pos, sk), jnp.arange(NUM_SPAWNERS))
                new_sp_health = jnp.full(NUM_SPAWNERS, SPAWNER_HEALTH, dtype=jnp.int32)
                new_sp_active = jnp.ones(NUM_SPAWNERS, dtype=jnp.int32)  # ENABLED - spawners continue to generate enemies on new levels
                key, sk = jax.random.split(key)
                new_sp_timers = jax.random.randint(sk, (NUM_SPAWNERS,), 0, SPAWNER_SPAWN_INTERVAL, dtype=jnp.int32)
                return new_sp_positions, new_sp_health, new_sp_active, new_sp_timers, key
            
            (transition_sp_positions, transition_sp_health, transition_sp_active, transition_sp_timers, rng) = respawn_spawners_for_level(rng)
            
            # For the starting chamber (map_index=0, level=0), move spawners to the lower half
            is_starting_chamber = (new_map_index == 0) & (new_level == 0)
            adjusted_transition_sp_positions = jnp.where(
                is_starting_chamber,
                transition_sp_positions.at[:, 1].set(
                    jnp.maximum(transition_sp_positions[:, 1], 550)
                ),
                transition_sp_positions
            )
            
            # When level changes, use freshly respawned spawners; otherwise use the updated values
            use_sp_positions = jnp.where(level_changed[..., None], adjusted_transition_sp_positions, state.spawner_positions)
            use_sp_health = jnp.where(level_changed, transition_sp_health, new_spawner_health)
            use_sp_active = jnp.where(level_changed, transition_sp_active, new_spawner_active)
            use_sp_timers = jnp.where(level_changed, transition_sp_timers, final_spawner_timers)
            
            # Safety pass: suppress entities that overlap walls after all spawn/transition logic.
            WALLS_FINAL = jnp.concatenate(
                [
                    self.renderer.LEVEL_WALLS[new_map_index, new_level],
                    self.renderer.BOUNDARY_WALLS,
                ],
                axis=0,
            )

            def rect_overlaps_walls_step(pos: chex.Array, width: chex.Array, height: chex.Array) -> chex.Array:
                px = pos[0]
                py = pos[1]
                wx = WALLS_FINAL[:, 0]
                wy = WALLS_FINAL[:, 1]
                ww = WALLS_FINAL[:, 2]
                wh = WALLS_FINAL[:, 3]
                overlap_x = (px <= (wx + ww - 1)) & ((px + width - 1) >= wx)
                overlap_y = (py <= (wy + wh - 1)) & ((py + height - 1) >= wy)
                return jnp.any(overlap_x & overlap_y)

            # Only apply wall overlap safety check during level transitions, not every frame
            def apply_wall_safety():
                enemy_wall_overlap = jax.vmap(
                    lambda p: rect_overlaps_walls_step(p, self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT)
                )(final_enemy_positions_after_level)
                safe_enemies = final_enemy_active_after_level & (~enemy_wall_overlap).astype(jnp.int32)

                spawner_wall_overlap = jax.vmap(
                    lambda p: rect_overlaps_walls_step(p, SPAWNER_WIDTH, SPAWNER_HEIGHT)
                )(use_sp_positions)
                safe_spawners = use_sp_active & (~spawner_wall_overlap).astype(jnp.int32)

                item_sizes_check = self.renderer.ITEM_TYPE_SIZES[transition_item_types]
                item_wall_overlap = jax.vmap(
                    lambda p, sz: rect_overlaps_walls_step(p, sz[0], sz[1])
                )(transition_item_positions, item_sizes_check)
                safe_items = transition_item_active & (~item_wall_overlap).astype(jnp.int32)
                
                return safe_enemies, safe_spawners, safe_items
            
            def no_wall_safety():
                return final_enemy_active_after_level, use_sp_active, transition_item_active
            
            # Only suppress wall-overlapping entities when the world changes (level or map)
            safe_enemy_active, safe_spawner_active, safe_item_active = jax.lax.cond(
                world_changed,
                lambda _: apply_wall_safety(),
                lambda _: no_wall_safety(),
                operand=None
            )

            # Keep only valid final enemy positions; clear inactive slots to avoid stale coordinate reuse
            final_enemy_positions_state = jnp.where(
                safe_enemy_active[:, None] == 1,
                final_enemy_positions_after_level,
                jnp.array([0, 0], dtype=jnp.int32),
            )
            final_enemy_hitpoints_state = jnp.where(
                safe_enemy_active == 1,
                final_enemy_hitpoints_after_level,
                0,
            ).astype(jnp.int32)

            # Death handling: start freeze instead of instant respawn
            player_died = final_health <= 0
            new_lives = jnp.where(player_died, state.lives - 1, state.lives)
            
            # Keep position; set health to 0 if dead; score unchanged here
            final_x = transition_x
            final_y = transition_y
            final_health_after = jnp.where(player_died, jnp.array(0, dtype=jnp.int32), final_health)
            death_counter_after = jnp.where(player_died & (new_lives > 0), jnp.array(DEATH_FREEZE_TICKS, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
            chamber_transition_ticks_after = jnp.where(
                start_chamber_transition,
                jnp.array(CHAMBER_TRANSITION_TICKS, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32),
            )
            chamber_transition_pending_after = jnp.where(
                player_died,
                jnp.array(0, dtype=jnp.int32),
                jnp.where(
                    start_chamber_transition,
                    jnp.array(1, dtype=jnp.int32),
                    jnp.where(
                        is_pending_transition,
                        jnp.array(0, dtype=jnp.int32),
                        state.chamber_transition_pending,
                    ),
                ),
            )
            chamber_target_map_index_after = jnp.where(
                start_chamber_transition,
                pending_target_map_index,
                state.chamber_target_map_index,
            )
            chamber_target_x_after = jnp.where(
                start_chamber_transition,
                pending_target_x,
                state.chamber_target_x,
            )
            chamber_target_y_after = jnp.where(
                start_chamber_transition,
                pending_target_y,
                state.chamber_target_y,
            )
            chamber_transition_ticks_after = jnp.where(
                player_died,
                jnp.array(0, dtype=jnp.int32),
                chamber_transition_ticks_after,
            )
            ladder_transition_cooldown_after = jnp.maximum(state.ladder_transition_cooldown - 1, 0)
            ladder_transition_cooldown_after = jnp.where(
                level_changed,
                jnp.array(LADDER_LEVEL_CHANGE_BLOCK_TICKS, dtype=jnp.int32),
                ladder_transition_cooldown_after,
            )
            ladder_transition_cooldown_after = jnp.where(
                player_died,
                jnp.array(0, dtype=jnp.int32),
                ladder_transition_cooldown_after,
            )
            level_transition_ticks_after = jnp.where(
                level_changed,
                jnp.array(LEVEL_CHANGE_FREEZE_TICKS, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32),
            )
            level_transition_ticks_after = jnp.where(
                player_died,
                jnp.array(0, dtype=jnp.int32),
                level_transition_ticks_after,
            )
            
            new_state = DarkChambersState(
                player_x=final_x,
                player_y=final_y,
                player_direction=new_direction,
                player_moving=player_moving,
                enemy_types=final_enemy_types_after_level,
                enemy_active=safe_enemy_active,
                enemy_hitpoints=final_enemy_hitpoints_state,
                wizard_shoot_timers=final_wizard_timers_after_level,
                spawner_positions=use_sp_positions,
                spawner_health=use_sp_health,
                spawner_active=safe_spawner_active,
                spawner_timers=use_sp_timers,
                bullet_positions=transition_bullet_positions,
                bullet_active=transition_bullet_active,
                enemy_bullet_positions=transition_enemy_bullet_positions,
                enemy_bullet_active=final_enemy_bullet_active3,
                health=final_health_after,
                damage_cooldown=cooldown2,
                score=final_score,
                item_positions=transition_item_positions,
                item_types=transition_item_types,
                item_active=safe_item_active,
                has_key=jnp.where(level_changed, 0, new_has_key),
                shield_active=new_shield_active,
                gun_active=new_gun_active,
                bomb_count=bomb_count_after_detonation,
                hammer_count=hammer_count_after_use,
                last_fire_step=new_last_fire_step,
                last_shot_step=new_last_shot_step,
                fire_was_pressed=new_fire_was_pressed,
                current_level=new_level,
                map_index=new_map_index,
                ladder_timer=new_ladder_timer,
                ladder_transition_cooldown=ladder_transition_cooldown_after,
                lives=new_lives,
                step_counter=state.step_counter + 1,
                death_counter=death_counter_after,
                chamber_transition_ticks=chamber_transition_ticks_after,
                chamber_transition_pending=chamber_transition_pending_after,
                chamber_target_map_index=chamber_target_map_index_after,
                chamber_target_x=chamber_target_x_after,
                chamber_target_y=chamber_target_y_after,
                level_transition_ticks=level_transition_ticks_after,
                key=rng,
                enemy_positions=final_enemy_positions_state,
                enemy_dir=new_enemy_dir,
                enemy_chase_timer=chase_timer,
                enemy_confuse_timer=confuse_timer,
                enemy_stuck_counter=stuck_counter,
                enemy_patrol_box=state.enemy_patrol_box,
                enemy_idle_timer=idle_timer,
                enemy_pause_timer=pause_timer,
                speed_boost_timer=state.speed_boost_timer,
                poison_cloud_x=state.poison_cloud_x,
                poison_cloud_y=state.poison_cloud_y,
                poison_cloud_timer=state.poison_cloud_timer,
                enemy_contact_cooldown=new_enemy_contact_cooldown,
                enemy_moving=enemy_moving_arr,
                checkpoint_idx=jnp.where(level_changed, jnp.array(-1, dtype=jnp.int32), new_checkpoint_idx),
                checkpoint_map=jnp.where(level_changed, jnp.array(0, dtype=jnp.int32), new_checkpoint_map),
                checkpoint_level=jnp.where(level_changed, jnp.array(0, dtype=jnp.int32), new_checkpoint_level),
            )
            
            obs = self._get_observation(new_state)
            reward = self._get_reward(state, new_state)
            done = self._get_done(new_state)
            info = self._get_info(new_state)
            return obs, new_state, reward, done, info


        # Top-level choose death freeze, chamber transition freeze, or normal update.
        return jax.lax.cond(
            state.death_counter > 0,
            handle_freeze,
            lambda _: jax.lax.cond(
                state.chamber_transition_ticks > 0,
                handle_chamber_transition,
                lambda __: jax.lax.cond(
                    state.level_transition_ticks > 0,
                    handle_level_transition,
                    handle_normal,
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )
    
    def render(self, state: DarkChambersState) -> jnp.ndarray:
        return self.renderer.render(state)
    
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(19)
    
    def observation_space(self) -> spaces.Dict:
        import numpy as np
        _W = max(GAME_W, GAMEPLAY_H)
        # Per-column bounds for entity arrays so that type and in_view features
        # are properly scaled to [0,1] instead of being squashed to ~0.006–0.037.
        # Columns: [x, y, w, h, type, in_view]
        enemy_low  = np.tile(np.array([[-1, -1, 0, 0, 0, 0]], dtype=np.int32), (NUM_ENEMIES, 1))
        enemy_high = np.tile(np.array([[_W, _W, _W, _W, ENEMY_GRIM_REAPER, 1]], dtype=np.int32), (NUM_ENEMIES, 1))
        # item type max includes checkpoint entries.
        item_low  = np.tile(np.array([[-1, -1, 0, 0, 0, 0]], dtype=np.int32), (NUM_ITEMS, 1))
        item_high = np.tile(np.array([[_W, _W, _W, _W, ITEM_CHECKPOINT, 1]], dtype=np.int32), (NUM_ITEMS, 1))
        # Columns without type: [x, y, w, h, in_view]
        spawner_low  = np.tile(np.array([[-1, -1, 0, 0, 0]], dtype=np.int32), (NUM_SPAWNERS, 1))
        spawner_high = np.tile(np.array([[_W, _W, _W, _W, 1]], dtype=np.int32), (NUM_SPAWNERS, 1))
        pbullet_low  = np.tile(np.array([[-1, -1, 0, 0, 0]], dtype=np.int32), (OBS_MAX_PLAYER_BULLETS, 1))
        pbullet_high = np.tile(np.array([[_W, _W, _W, _W, 1]], dtype=np.int32), (OBS_MAX_PLAYER_BULLETS, 1))
        ebullet_low  = np.tile(np.array([[-1, -1, 0, 0, 0]], dtype=np.int32), (OBS_MAX_ENEMY_BULLETS, 1))
        ebullet_high = np.tile(np.array([[_W, _W, _W, _W, 1]], dtype=np.int32), (OBS_MAX_ENEMY_BULLETS, 1))
        # Portals: [x, y, in_view]
        portal_low  = np.tile(np.array([[-1, -1, 0]], dtype=np.int32), (6, 1))
        portal_high = np.tile(np.array([[_W, _W, 1]], dtype=np.int32), (6, 1))
        wall_low  = np.tile(np.array([[-1, -1, 0, 0, 0]], dtype=np.int32), (OBS_MAX_WALLS, 1))
        wall_high = np.tile(np.array([[_W, _W, _W, _W, 1]], dtype=np.int32), (OBS_MAX_WALLS, 1))
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=GAME_W - 1, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=GAMEPLAY_H - 1, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=GAME_W, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=GAMEPLAY_H, shape=(), dtype=jnp.int32),
            }),
            "enemies": spaces.Box(low=enemy_low, high=enemy_high, dtype=jnp.int32),
            "items": spaces.Box(low=item_low, high=item_high, dtype=jnp.int32),
            "spawners": spaces.Box(low=spawner_low, high=spawner_high, dtype=jnp.int32),
            "player_bullets": spaces.Box(low=pbullet_low, high=pbullet_high, dtype=jnp.int32),
            "enemy_bullets": spaces.Box(low=ebullet_low, high=ebullet_high, dtype=jnp.int32),
            "portals": spaces.Box(low=portal_low, high=portal_high, dtype=jnp.int32),
            "walls": spaces.Box(low=wall_low, high=wall_high, dtype=jnp.int32),
            "border_distances": spaces.Box(
                low=0,
                high=max(GAME_W, GAMEPLAY_H),
                shape=(4,),
                dtype=jnp.int32,
            ),
            "health": spaces.Box(low=0, high=self.consts.MAX_HEALTH, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=OBS_SCORE_CLIP, shape=(), dtype=jnp.int32),
            "step": spaces.Box(low=0, high=OBS_STEP_CLIP, shape=(), dtype=jnp.int32),
            "has_key": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "shield_active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "gun_active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "bomb_count": spaces.Box(low=0, high=MAX_BOMBS, shape=(), dtype=jnp.int32),
            "hammer_count": spaces.Box(low=0, high=MAX_HAMMERS, shape=(), dtype=jnp.int32),
            "speed_boost_active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=5, shape=(), dtype=jnp.int32),
            "poison_cloud_x": spaces.Box(low=0, high=self.consts.WORLD_WIDTH, shape=(), dtype=jnp.int32),
            "poison_cloud_y": spaces.Box(low=0, high=self.consts.WORLD_HEIGHT, shape=(), dtype=jnp.int32),
            "poison_cloud_active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "damage_cooldown": spaces.Box(low=0, high=120, shape=(), dtype=jnp.int32),
        })
    
    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0, 
            high=255, 
            shape=(GAME_H, GAME_W, 3), 
            dtype=jnp.uint8
        )
    
    def _get_observation(self, state: DarkChambersState) -> DarkChambersObservation:
        cam_x = jnp.clip(
            state.player_x - GAME_W // 2,
            0,
            self.consts.WORLD_WIDTH - GAME_W,
        ).astype(jnp.int32)
        cam_y = jnp.clip(
            state.player_y - GAMEPLAY_H // 2,
            0,
            self.consts.WORLD_HEIGHT - GAMEPLAY_H,
        ).astype(jnp.int32)

        player_screen_x = (state.player_x - cam_x).astype(jnp.int32)
        player_screen_y = (state.player_y - cam_y).astype(jnp.int32)

        def _select_nearest_visible(
            x_raw: jnp.ndarray,
            y_raw: jnp.ndarray,
            widths: jnp.ndarray,
            heights: jnp.ndarray,
            in_view_mask: jnp.ndarray,
            k: int,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            cx = x_raw + widths // 2
            cy = y_raw + heights // 2
            dx = cx - player_screen_x
            dy = cy - player_screen_y
            dist2 = dx * dx + dy * dy
            big = jnp.array(1_000_000_000, dtype=jnp.int32)
            masked_dist2 = jnp.where(in_view_mask == 1, dist2, big)
            sorted_idx = jnp.argsort(masked_dist2)
            sel_idx = sorted_idx[:k]

            sel_x_raw = x_raw[sel_idx]
            sel_y_raw = y_raw[sel_idx]
            sel_w = widths[sel_idx]
            sel_h = heights[sel_idx]
            sel_in_view = in_view_mask[sel_idx].astype(jnp.int32)
            sel_x = jnp.where(sel_in_view == 1, jnp.clip(sel_x_raw, 0, GAME_W - 1), -1).astype(jnp.int32)
            sel_y = jnp.where(sel_in_view == 1, jnp.clip(sel_y_raw, 0, GAMEPLAY_H - 1), -1).astype(jnp.int32)
            sel_w = jnp.where(sel_in_view == 1, sel_w, 0).astype(jnp.int32)
            sel_h = jnp.where(sel_in_view == 1, sel_h, 0).astype(jnp.int32)
            return sel_x, sel_y, sel_w, sel_h, sel_in_view
        player = EntityPosition(
            x=jnp.asarray(player_screen_x, dtype=jnp.int32),
            y=jnp.asarray(player_screen_y, dtype=jnp.int32),
            width=jnp.asarray(self.consts.PLAYER_WIDTH, dtype=jnp.int32),
            height=jnp.asarray(self.consts.PLAYER_HEIGHT, dtype=jnp.int32),
        )

        enemy_widths = jnp.full(NUM_ENEMIES, self.consts.ENEMY_WIDTH, dtype=jnp.int32)
        enemy_heights = jnp.full(NUM_ENEMIES, self.consts.ENEMY_HEIGHT, dtype=jnp.int32)
        enemy_screen_positions = (state.enemy_positions - jnp.array([cam_x, cam_y], dtype=jnp.int32)).astype(jnp.int32)
        enemy_x_raw = enemy_screen_positions[:, 0]
        enemy_y_raw = enemy_screen_positions[:, 1]
        enemy_is_active = state.enemy_active.astype(jnp.int32)
        enemy_in_view = (
            (enemy_is_active == 1)
            & (enemy_x_raw + enemy_widths > 0)
            & (enemy_x_raw < GAME_W)
            & (enemy_y_raw + enemy_heights > 0)
            & (enemy_y_raw < GAMEPLAY_H)
        ).astype(jnp.int32)
        enemy_x = jnp.where(enemy_in_view, jnp.clip(enemy_x_raw, 0, GAME_W - 1), -1).astype(jnp.int32)
        enemy_y = jnp.where(enemy_in_view, jnp.clip(enemy_y_raw, 0, GAMEPLAY_H - 1), -1).astype(jnp.int32)
        enemy_widths  = jnp.where(enemy_in_view, enemy_widths, 0)
        enemy_heights = jnp.where(enemy_in_view, enemy_heights, 0)

        enemies_array = jnp.stack([
            enemy_x,
            enemy_y,
            enemy_widths,
            enemy_heights,
            jnp.where(enemy_in_view == 1, state.enemy_types.astype(jnp.int32), 0).astype(jnp.int32),
            enemy_in_view,
        ], axis=1)

        item_screen_positions = (state.item_positions - jnp.array([cam_x, cam_y], dtype=jnp.int32)).astype(jnp.int32)
        item_sizes = self.renderer.ITEM_TYPE_SIZES[state.item_types].astype(jnp.int32)
        item_x_raw = item_screen_positions[:, 0]
        item_y_raw = item_screen_positions[:, 1]
        item_w = item_sizes[:, 0]
        item_h = item_sizes[:, 1]
        item_active = state.item_active.astype(jnp.int32)
        item_in_view = (
            (item_active == 1)
            & (item_x_raw + item_w > 0)
            & (item_x_raw < GAME_W)
            & (item_y_raw + item_h > 0)
            & (item_y_raw < GAMEPLAY_H)
        ).astype(jnp.int32)
        item_x = jnp.where(item_in_view == 1, jnp.clip(item_x_raw, 0, GAME_W - 1), -1).astype(jnp.int32)
        item_y = jnp.where(item_in_view == 1, jnp.clip(item_y_raw, 0, GAMEPLAY_H - 1), -1).astype(jnp.int32)
        item_type_visible = jnp.where(item_in_view == 1, state.item_types.astype(jnp.int32), 0).astype(jnp.int32)
        item_w = jnp.where(item_in_view == 1, item_w, 0)
        item_h = jnp.where(item_in_view == 1, item_h, 0)
        items_array = jnp.stack([
            item_x,
            item_y,
            item_w,
            item_h,
            item_type_visible,
            item_in_view,
        ], axis=1)

        spawner_screen_positions = (state.spawner_positions - jnp.array([cam_x, cam_y], dtype=jnp.int32)).astype(jnp.int32)
        spawner_x_raw = spawner_screen_positions[:, 0]
        spawner_y_raw = spawner_screen_positions[:, 1]
        spawner_w = jnp.full(NUM_SPAWNERS, SPAWNER_WIDTH, dtype=jnp.int32)
        spawner_h = jnp.full(NUM_SPAWNERS, SPAWNER_HEIGHT, dtype=jnp.int32)
        spawner_in_view = (
            (state.spawner_active == 1)
            & (spawner_x_raw + spawner_w > 0)
            & (spawner_x_raw < GAME_W)
            & (spawner_y_raw + spawner_h > 0)
            & (spawner_y_raw < GAMEPLAY_H)
        ).astype(jnp.int32)
        spawner_x = jnp.where(spawner_in_view == 1, jnp.clip(spawner_x_raw, 0, GAME_W - 1), -1).astype(jnp.int32)
        spawner_y = jnp.where(spawner_in_view == 1, jnp.clip(spawner_y_raw, 0, GAMEPLAY_H - 1), -1).astype(jnp.int32)
        spawners_array = jnp.stack([
            spawner_x,
            spawner_y,
            spawner_w,
            spawner_h,
            spawner_in_view,
        ], axis=1)

        bullet_screen_positions = (state.bullet_positions[:, :2] - jnp.array([cam_x, cam_y], dtype=jnp.int32)).astype(jnp.int32)
        bullet_x_raw = bullet_screen_positions[:, 0]
        bullet_y_raw = bullet_screen_positions[:, 1]
        bullet_w = jnp.full(MAX_BULLETS, BULLET_WIDTH, dtype=jnp.int32)
        bullet_h = jnp.full(MAX_BULLETS, BULLET_HEIGHT, dtype=jnp.int32)
        bullet_in_view = (
            (state.bullet_active == 1)
            & (bullet_x_raw + bullet_w > 0)
            & (bullet_x_raw < GAME_W)
            & (bullet_y_raw + bullet_h > 0)
            & (bullet_y_raw < GAMEPLAY_H)
        ).astype(jnp.int32)
        bullet_x, bullet_y, bullet_w, bullet_h, bullet_in_view = _select_nearest_visible(
            bullet_x_raw,
            bullet_y_raw,
            bullet_w,
            bullet_h,
            bullet_in_view,
            OBS_MAX_PLAYER_BULLETS,
        )
        player_bullets_array = jnp.stack([
            bullet_x,
            bullet_y,
            bullet_w,
            bullet_h,
            bullet_in_view,
        ], axis=1)

        enemy_bullet_screen_positions = (state.enemy_bullet_positions[:, :2] - jnp.array([cam_x, cam_y], dtype=jnp.int32)).astype(jnp.int32)
        enemy_bullet_x_raw = enemy_bullet_screen_positions[:, 0]
        enemy_bullet_y_raw = enemy_bullet_screen_positions[:, 1]
        enemy_bullet_w = jnp.full(ENEMY_MAX_BULLETS, ENEMY_BULLET_WIDTH, dtype=jnp.int32)
        enemy_bullet_h = jnp.full(ENEMY_MAX_BULLETS, ENEMY_BULLET_HEIGHT, dtype=jnp.int32)
        enemy_bullet_in_view = (
            (state.enemy_bullet_active == 1)
            & (enemy_bullet_x_raw + enemy_bullet_w > 0)
            & (enemy_bullet_x_raw < GAME_W)
            & (enemy_bullet_y_raw + enemy_bullet_h > 0)
            & (enemy_bullet_y_raw < GAMEPLAY_H)
        ).astype(jnp.int32)
        enemy_bullet_x, enemy_bullet_y, enemy_bullet_w, enemy_bullet_h, enemy_bullet_in_view = _select_nearest_visible(
            enemy_bullet_x_raw,
            enemy_bullet_y_raw,
            enemy_bullet_w,
            enemy_bullet_h,
            enemy_bullet_in_view,
            OBS_MAX_ENEMY_BULLETS,
        )
        enemy_bullets_array = jnp.stack([
            enemy_bullet_x,
            enemy_bullet_y,
            enemy_bullet_w,
            enemy_bullet_h,
            enemy_bullet_in_view,
        ], axis=1)

        portal_hole_height = 40
        portal_gap = (self.consts.WORLD_HEIGHT - 3 * portal_hole_height) // 4
        portal_y_centers = jnp.array([
            portal_gap + portal_hole_height // 2,
            portal_gap * 2 + portal_hole_height + portal_hole_height // 2,
            portal_gap * 3 + 2 * portal_hole_height + portal_hole_height // 2,
        ], dtype=jnp.int32)
        portal_x = jnp.concatenate([
            jnp.zeros((3,), dtype=jnp.int32),
            jnp.full((3,), self.consts.WORLD_WIDTH - 1, dtype=jnp.int32),
        ])
        portal_y = jnp.concatenate([portal_y_centers, portal_y_centers])
        portal_world = jnp.stack([portal_x, portal_y], axis=1)
        portal_screen = (portal_world - jnp.array([cam_x, cam_y], dtype=jnp.int32)).astype(jnp.int32)
        portal_in_view = (
            (portal_screen[:, 0] >= 0)
            & (portal_screen[:, 0] < GAME_W)
            & (portal_screen[:, 1] >= 0)
            & (portal_screen[:, 1] < GAMEPLAY_H)
        ).astype(jnp.int32)
        portal_x = jnp.where(portal_in_view == 1, jnp.clip(portal_screen[:, 0], 0, GAME_W - 1), -1).astype(jnp.int32)
        portal_y = jnp.where(portal_in_view == 1, jnp.clip(portal_screen[:, 1], 0, GAMEPLAY_H - 1), -1).astype(jnp.int32)
        portals_array = jnp.stack([
            portal_x,
            portal_y,
            portal_in_view,
        ], axis=1)

        current_level_walls = self.renderer.LEVEL_WALLS[state.map_index, state.current_level]
        all_walls = jnp.concatenate([current_level_walls, self.renderer.BOUNDARY_WALLS], axis=0).astype(jnp.int32)
        wall_screen_pos = (all_walls[:, 0:2] - jnp.array([cam_x, cam_y], dtype=jnp.int32)).astype(jnp.int32)
        wall_sizes = all_walls[:, 2:4].astype(jnp.int32)

        wall_x0 = wall_screen_pos[:, 0]
        wall_y0 = wall_screen_pos[:, 1]
        wall_x1 = wall_x0 + wall_sizes[:, 0]
        wall_y1 = wall_y0 + wall_sizes[:, 1]

        wall_clip_x0 = jnp.clip(wall_x0, 0, GAME_W)
        wall_clip_y0 = jnp.clip(wall_y0, 0, GAMEPLAY_H)
        wall_clip_x1 = jnp.clip(wall_x1, 0, GAME_W)
        wall_clip_y1 = jnp.clip(wall_y1, 0, GAMEPLAY_H)
        wall_clip_w = jnp.maximum(0, wall_clip_x1 - wall_clip_x0).astype(jnp.int32)
        wall_clip_h = jnp.maximum(0, wall_clip_y1 - wall_clip_y0).astype(jnp.int32)
        wall_in_view = ((wall_clip_w > 0) & (wall_clip_h > 0)).astype(jnp.int32)

        wall_x, wall_y, wall_clip_w, wall_clip_h, wall_in_view = _select_nearest_visible(
            wall_clip_x0.astype(jnp.int32),
            wall_clip_y0.astype(jnp.int32),
            wall_clip_w,
            wall_clip_h,
            wall_in_view,
            OBS_MAX_WALLS,
        )
        walls_array = jnp.stack([
            wall_x,
            wall_y,
            wall_clip_w,
            wall_clip_h,
            wall_in_view,
        ], axis=1)

        border_distances = jnp.stack([
            jnp.maximum(0, player_screen_x),
            jnp.maximum(0, GAME_W - (player_screen_x + self.consts.PLAYER_WIDTH)),
            jnp.maximum(0, player_screen_y),
            jnp.maximum(0, GAMEPLAY_H - (player_screen_y + self.consts.PLAYER_HEIGHT)),
        ], axis=0).astype(jnp.int32)

        
        return DarkChambersObservation(
            player=player,
            enemies=enemies_array,
            items=items_array,
            spawners=spawners_array,
            player_bullets=player_bullets_array,
            enemy_bullets=enemy_bullets_array,
            portals=portals_array,
            walls=walls_array,
            border_distances=border_distances,
            health=state.health.astype(jnp.int32),
            score=jnp.clip(state.score, 0, OBS_SCORE_CLIP).astype(jnp.int32),
            step=jnp.clip(state.step_counter, 0, OBS_STEP_CLIP).astype(jnp.int32),
            has_key=state.has_key.astype(jnp.int32),
            shield_active=state.shield_active.astype(jnp.int32),
            gun_active=state.gun_active.astype(jnp.int32),
            bomb_count=jnp.clip(state.bomb_count, 0, MAX_BOMBS).astype(jnp.int32),
            hammer_count=jnp.clip(state.hammer_count, 0, MAX_HAMMERS).astype(jnp.int32),
            speed_boost_active=(state.speed_boost_timer > 0).astype(jnp.int32),
            lives=state.lives.astype(jnp.int32),
            poison_cloud_x=state.poison_cloud_x.astype(jnp.int32),
            poison_cloud_y=state.poison_cloud_y.astype(jnp.int32),
            poison_cloud_active=(state.poison_cloud_timer > 0).astype(jnp.int32),
            damage_cooldown=state.damage_cooldown.astype(jnp.int32),
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: DarkChambersState, all_rewards: jnp.array = None) -> DarkChambersInfo:
        return DarkChambersInfo(time=state.step_counter)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: DarkChambersState, state: DarkChambersState) -> float:
        return state.score - previous_state.score
    
    # Episode ends when the player has no lives left and health reaches zero.
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: DarkChambersState) -> bool:
        return (state.lives <= 0) & (state.health <= 0)
    

    def obs_to_flat_array(self, obs: DarkChambersObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.height.flatten(),
            obs.player.width.flatten(),
            obs.enemies.flatten(),
            obs.items.flatten(),
            obs.spawners.flatten(),
            obs.player_bullets.flatten(),
            obs.enemy_bullets.flatten(),
            obs.portals.flatten(),
            obs.walls.flatten(),
            obs.border_distances.flatten(),
            obs.health.flatten(),
            obs.score.flatten(),
            obs.step.flatten(),
            obs.has_key.flatten(),
            obs.shield_active.flatten(),
            obs.gun_active.flatten(),
            obs.bomb_count.flatten(),
            obs.hammer_count.flatten(),
            obs.speed_boost_active.flatten(),
            obs.lives.flatten(),
            obs.poison_cloud_x.flatten(),
            obs.poison_cloud_y.flatten(),
            obs.poison_cloud_active.flatten(),
            obs.damage_cooldown.flatten(),
        ])