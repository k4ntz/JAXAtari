import os
from functools import partial
from typing import NamedTuple, Tuple
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

        # Player sprite
        {"name": "player", "type": "single", "file": "player.npy"},

        # Digits for UI - commented out, using hardcoded digit patterns instead
        # {"name": "digits", "type": "digits", "pattern": "digits/{}.npy"},
    )



# Game configuration and constants

GAME_H = 210
GAME_W = 160
WORLD_W = GAME_W * 2  # World is 2x viewport size
WORLD_H = GAME_H * 2

# --- Nav grid for enemy pathfinding ---
CELL_SIZE = 8                      # size of one nav cell in pixels
GRID_W = WORLD_W // CELL_SIZE
GRID_H = WORLD_H // CELL_SIZE
BIG_DIST = 10_000                  # "infinity" for distance field

NUM_ENEMIES = 20  # Increased to allow more spawned enemies
NUM_SPAWNERS = 3  # Spawner entities

# Enemy types (5 = strongest, 1 = weakest)
ENEMY_GRIM_REAPER = 5  # Strongest
ENEMY_WIZARD = 4
ENEMY_SKELETON = 3
ENEMY_WRAITH = 2
ENEMY_ZOMBIE = 1  # Weakest

# Item configuration
NUM_ITEMS = 20  # Increased variety
# Item type codes
ITEM_HEART = 1          # +health, no points
ITEM_POISON = 2         # -4 health, no points
ITEM_TRAP = 3           # -6 health, no points
ITEM_STRONGBOX = 4      # +100 points
ITEM_SILVER_CHALICE = 5 # +500 points
ITEM_AMULET = 6         # +1000 points
ITEM_GOLD_CHALICE = 7   # +3000 points
ITEM_SHIELD = 8         # Damage reduction
ITEM_GUN = 9            # Faster shooting
ITEM_BOMB = 10          # Kill all enemies in area
ITEM_KEY = 11           # Key to unlock exit (gated door)
ITEM_LADDER_UP = 12     # Exit/door leading up (requires key)
ITEM_LADDER_DOWN = 13   # Ladder leading down (always open)

# Level configuration
MAX_LEVELS = 7          # Total number of levels (0..6)
LADDER_INTERACTION_TIME = 60  # Steps player must stand on ladder to change level
LADDER_WIDTH = 12       # Larger than regular items
LADDER_HEIGHT = 12

# Bomb configuration
MAX_BOMBS = 15          # Maximum bombs player can carry
DOUBLE_TAP_WINDOW = 10  # Steps within which two fires count as double-tap
FIRE_RATE_LIMIT = 1     # Minimum steps between shots (fire rate limiter)
BOMB_RADIUS = 80        # Kill radius for bomb in pixels

# Default base size (unused now, kept for reference)
ITEM_WIDTH = 6
ITEM_HEIGHT = 6

# Bullet configuration
# Player bullet configuration
MAX_BULLETS = 64             # Drastically higher simultaneous bullets
BULLET_WIDTH = 4
BULLET_HEIGHT = 4
BULLET_SPEED = 3             # Slightly faster than player speed (2)
BULLET_SPEED_WITH_GUN = 6    # Much faster when gun is active

# Wizard projectile configuration
ENEMY_MAX_BULLETS = 8
ENEMY_BULLET_WIDTH = 3
ENEMY_BULLET_HEIGHT = 3
ENEMY_BULLET_SPEED = 2
WIZARD_SHOOT_INTERVAL = 40  # steps between shots (gate) - increased from 20 to slow down shooting

# Lives configuration
MAX_LIVES = 5  # Number of lives player starts with

# Death freeze configuration (ticks to freeze before respawn)
DEATH_FREEZE_TICKS = 60

# Spawner configuration
SPAWNER_WIDTH = 14
SPAWNER_HEIGHT = 14
SPAWNER_HEALTH = 3  # Takes 3 hits to destroy
SPAWNER_SPAWN_INTERVAL = 150  # Spawn enemy every 150 steps

ENEMY_COLLISION_MARGIN = 1  # or 2 pixels if you want more slide room

CHASE_RADIUS = 120          # pixels
IDLE_SPEED = 1              # pixels per step (keep <= 1 for fewer collision issues)

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



class DarkChambersConstants(NamedTuple):
    """Game constants and configuration."""
    # Dimensions
    WIDTH: int = GAME_W
    HEIGHT: int = GAME_H
    WORLD_WIDTH: int = WORLD_W
    WORLD_HEIGHT: int = WORLD_H
    
    # Color scheme
    BACKGROUND_COLOR: Tuple[int, int, int] = (8, 10, 20)
    PLAYER_COLOR: Tuple[int, int, int] = (200, 80, 60)
    # Enemy colors by type (from weakest to strongest)
    ZOMBIE_COLOR: Tuple[int, int, int] = (100, 100, 100)  # Gray
    WRAITH_COLOR: Tuple[int, int, int] = (180, 180, 220)  # Light purple
    SKELETON_COLOR: Tuple[int, int, int] = (220, 220, 200)  # Bone white
    WIZARD_COLOR: Tuple[int, int, int] = (150, 80, 200)  # Purple
    GRIM_REAPER_COLOR: Tuple[int, int, int] = (50, 50, 50)  # Dark gray/black
    WALL_COLOR: Tuple[int, int, int] = (150, 120, 70)
    HEART_COLOR: Tuple[int, int, int] = (220, 30, 30)
    POISON_COLOR: Tuple[int, int, int] = (50, 200, 50)  # Green poison
    TRAP_COLOR: Tuple[int, int, int] = (120, 70, 20)     # Brown trap
    TREASURE_COLOR: Tuple[int, int, int] = (255, 220, 0) # Yellow for all treasures
    SPAWNER_COLOR: Tuple[int, int, int] = (180, 50, 180) # Magenta/purple spawner
    SHIELD_COLOR: Tuple[int, int, int] = (80, 120, 255) # Blue shield
    GUN_COLOR: Tuple[int, int, int] = (40, 40, 40) # Black gun
    BOMB_COLOR: Tuple[int, int, int] = (240, 240, 240) # White bomb
    KEY_COLOR: Tuple[int, int, int] = (0, 255, 255) # Cyan key (easy to distinguish from gold)
    LADDER_UP_COLOR: Tuple[int, int, int] = (140, 90, 40) # Brownish exit door
    LADDER_DOWN_COLOR: Tuple[int, int, int] = (100, 60, 20) # Darker brown ladder down
    UI_COLOR: Tuple[int, int, int] = (236, 236, 236)
    BULLET_COLOR: Tuple[int, int, int] = (255, 200, 0)
    
    # Sizes
    PLAYER_WIDTH: int = 12
    PLAYER_HEIGHT: int = 12
    ENEMY_WIDTH: int = 10
    ENEMY_HEIGHT: int = 10
    
    PLAYER_SPEED: int = 2
    WALL_THICKNESS: int = 8
    
    PLAYER_START_X: int = 24
    PLAYER_START_Y: int = 24
    
    # Health mechanics (scaled to classic 31 strength units)
    MAX_HEALTH: int = 31
    STARTING_HEALTH: int = 31
    HEALTH_GAIN: int = 10  # Heart potion gain
    POISON_DAMAGE: int = 4   # Light damage
    TRAP_DAMAGE: int = 6     # Heavier damage
    
    # Enemy damage by type (contact damage per step)
    ZOMBIE_DAMAGE: int = 1  # Weakest
    WRAITH_DAMAGE: int = 1
    SKELETON_DAMAGE: int = 3
    WIZARD_DAMAGE: int = 4
    GRIM_REAPER_DAMAGE: int = 5  # Strongest
    
    # Enemy scoring (points awarded when killed)
    ZOMBIE_POINTS: int = 10  # Weakest - explodes when killed
    WRAITH_POINTS: int = 20
    SKELETON_POINTS: int = 30
    WIZARD_POINTS: int = 50
    GRIM_REAPER_POINTS: int = 100  # Strongest


class DarkChambersState(NamedTuple):
    """Game state."""
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array  # 0=right, 1=left, 2=up, 3=down
    
    enemy_positions: chex.Array  # shape: (NUM_ENEMIES, 2)
    enemy_types: chex.Array      # shape: (NUM_ENEMIES,) - 1=zombie, 2=wraith, 3=skeleton, 4=wizard, 5=grim_reaper
    enemy_active: chex.Array     # shape: (NUM_ENEMIES,) - 1=alive, 0=dead
    
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
    last_fire_step: chex.Array  # step counter when fire was last pressed (for double-tap)
    
    current_level: chex.Array   # current level index (0 to MAX_LEVELS-1)
    ladder_timer: chex.Array    # time standing on ladder (0 to LADDER_INTERACTION_TIME)
    lives: chex.Array           # remaining lives (0 to MAX_LIVES)
    
    step_counter: chex.Array
    death_counter: chex.Array    # >0 means freeze frames remaining before respawn
    key: chex.PRNGKey


class EntityPosition(NamedTuple):
    """Entity position and dimensions."""
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class DarkChambersObservation(NamedTuple):
    """Observation space."""
    player: EntityPosition
    enemies: jnp.ndarray  # (NUM_ENEMIES, 5): x, y, width, height, active
    health: jnp.ndarray
    score: jnp.ndarray
    step: jnp.ndarray


class DarkChambersInfo(NamedTuple):
    """Extra info."""
    time: jnp.ndarray

class DarkChambersRenderer(JAXGameRenderer):
    """Handles rendering."""
    
    def __init__(self, consts: DarkChambersConstants = None):
        super().__init__(consts)
        self.consts = consts or DarkChambersConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(GAME_H, GAME_W),
            channels=3
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Load sprites using the asset system
        final_asset_config = list(_get_default_asset_config())
        
        # Create procedural sprites for colors not in the main sprites (enemies, items, UI, etc.)
        def create_color_sprite(color):
            """Create a 1x1 pixel sprite for a color to ensure it's in the palette."""
            rgba = (*color, 255)
            return jnp.array([[rgba]], dtype=jnp.uint8)
        
        # Add all game colors as procedural sprites to ensure they're in the palette
        color_sprites = {
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
            'bullet_color': create_color_sprite(self.consts.BULLET_COLOR),
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
        
        # Helper to scale palette-index masks to a target size (centered pad/crop)
        def _scale_mask(mask: jnp.ndarray, target_h: int, target_w: int) -> jnp.ndarray:
            if mask is None:
                return None
            if mask.ndim == 2:
                h, w = int(mask.shape[0]), int(mask.shape[1])
                sx = max(1, target_w // w)
                sy = max(1, target_h // h)
                scaled = jnp.repeat(jnp.repeat(mask, sy, axis=0), sx, axis=1)
                sh, sw = int(scaled.shape[0]), int(scaled.shape[1])
                pad_h = max(0, target_h - sh)
                pad_w = max(0, target_w - sw)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                scaled_padded = jnp.pad(scaled, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
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
                    return _scale_mask(m, target_h, target_w)
                return jax.vmap(_scale_one, in_axes=0, out_axes=0)(mask)
            else:
                return mask

        # Ensure player sprite visually matches gameplay size by upscaling if needed
        player_mask = self.SHAPE_MASKS.get("player")
        if player_mask is not None:
            target_w = int(self.consts.PLAYER_WIDTH)
            target_h = int(self.consts.PLAYER_HEIGHT)
            self.PLAYER_SCALED_MASK = _scale_mask(player_mask, target_h, target_w)
        else:
            # Fallback: use original player mask without scaling
            self.PLAYER_SCALED_MASK = self.SHAPE_MASKS.get("player")

        # No enemy/item sprite scaling; enemies/items stay as colored boxes
        self.ENEMY_SCALED_MASKS = None
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
        self.BULLET_ID = self.COLOR_TO_ID[self.consts.BULLET_COLOR]
        self.ZOMBIE_ID = self.COLOR_TO_ID[self.consts.ZOMBIE_COLOR]
        self.WRAITH_ID = self.COLOR_TO_ID[self.consts.WRAITH_COLOR]
        self.SKELETON_ID = self.COLOR_TO_ID[self.consts.SKELETON_COLOR]
        self.WIZARD_ID = self.COLOR_TO_ID[self.consts.WIZARD_COLOR]
        self.GRIM_REAPER_ID = self.COLOR_TO_ID[self.consts.GRIM_REAPER_COLOR]

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
        level_0_walls = jnp.array([
            # Border
            [0, 0, self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS],
            [0, self.consts.WORLD_HEIGHT - self.consts.WALL_THICKNESS, 
             self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS],
            [0, 0, self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT],
            [self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, 0, 
             self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT],
            # Labyrinth structure - Level 0
            [60, 80, 120, 8],
            [200, 150, 80, 8],
            [80, 200, 100, 8],
            [40, 280, 140, 8],
            [220, 50, 8, 140],
            [120, 180, 8, 100],
            [280, 120, 8, 160],
            [160, 300, 8, 80],
        ], dtype=jnp.int32)
        
        level_1_walls = jnp.array([
            # Border
            [0, 0, self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS],
            [0, self.consts.WORLD_HEIGHT - self.consts.WALL_THICKNESS, 
             self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS],
            [0, 0, self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT],
            [self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, 0, 
             self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT],
            # Different labyrinth structure - Level 1
            [50, 100, 100, 8],
            [180, 180, 100, 8],
            [100, 250, 120, 8],
            [60, 320, 100, 8],
            [250, 80, 8, 120],
            [140, 150, 8, 140],
            [260, 200, 8, 140],
            [80, 340, 8, 60],
        ], dtype=jnp.int32)

        # Create additional levels by offsetting and varying walls
        def offset_walls(base, dx, dy):
            # Preserve border walls (first 4 entries) and offset only interior labyrinth walls
            borders = base[:4]
            interior = base[4:]
            xy = interior[:, 0:2] + jnp.array([dx, dy])
            wh = interior[:, 2:4]
            interior_off = jnp.concatenate([xy, wh], axis=1)
            return jnp.concatenate([borders, interior_off], axis=0)

        level_2_walls = offset_walls(level_1_walls, 10, -10)
        level_3_walls = offset_walls(level_0_walls, -15, 20)
        level_4_walls = offset_walls(level_1_walls, 25, 10)
        level_5_walls = offset_walls(level_0_walls, -30, -20)
        level_6_walls = offset_walls(level_1_walls, 5, 25)
        
        # Stack levels: shape (MAX_LEVELS, num_walls, 4)
        self.LEVEL_WALLS = jnp.stack([level_0_walls, level_1_walls, level_2_walls, level_3_walls, level_4_walls, level_5_walls, level_6_walls], axis=0)
        
        # Default to level 0 for compatibility
        self.WALLS = level_0_walls

        # --- Navigation grid: mark nav cells that are blocked by walls ---
        def make_occupancy(walls):
            # walls: (num_walls, 4) = [x, y, w, h]
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

        # Shape: (MAX_LEVELS, GRID_H, GRID_W)
        self.LEVEL_WALL_GRID = jnp.stack(
            [make_occupancy(self.LEVEL_WALLS[i]) for i in range(MAX_LEVELS)],
            axis=0
        )
        
        # Per-item sizes (width, height) indexed by item type code
        # Index 0 unused placeholder for alignment
        self.ITEM_TYPE_SIZES = jnp.array([
            [0, 0],                 # 0 (unused)
            [6, 6],                 # 1 HEART
            [6, 6],                 # 2 POISON
            [6, 6],                 # 3 TRAP
            [7, 7],                 # 4 STRONGBOX (small)
            [9, 9],                 # 5 SILVER CHALICE (medium)
            [11, 11],               # 6 AMULET (large)
            [13, 13],               # 7 GOLD CHALICE (largest)
            [6, 6],                 # 8 SHIELD
            [6, 6],                 # 9 GUN
            [6, 6],                 # 10 BOMB
            [6, 6],                 # 11 KEY (small key)
            [LADDER_WIDTH, LADDER_HEIGHT],  # 12 LADDER_UP (larger)
            [LADDER_WIDTH, LADDER_HEIGHT],  # 13 LADDER_DOWN (larger)
        ], dtype=jnp.int32)

        # Color id mapping per item type (aligning with palette above)
        self.ITEM_TYPE_COLOR_IDS = jnp.array([
            0,   # unused
            8,   # HEART (red)
            9,   # POISON (green)
            13,  # TRAP (brown)
            12,  # STRONGBOX (yellow)
            12,  # SILVER CHALICE (yellow)
            12,  # AMULET (yellow)
            12,  # GOLD CHALICE (yellow)
            15,  # SHIELD (blue)
            16,  # GUN (black)
            17,  # BOMB (white)
            18,  # KEY (gold)
            19,  # LADDER_UP (brown exit door)
            20,  # LADDER_DOWN (dark brown ladder)
        ], dtype=jnp.int32)
        # Python constants for item type color IDs (indexed by item_type - 1)
        self.ITEM_TYPE_COLOR_IDS_PY = [
            self.HEART_ID,         # 1: ITEM_HEART
            self.POISON_ID,        # 2: ITEM_POISON
            self.TRAP_ID,          # 3: ITEM_TRAP
            self.TREASURE_ID,      # 4: ITEM_STRONGBOX
            self.TREASURE_ID,      # 5: ITEM_SILVER_CHALICE
            self.TREASURE_ID,      # 6: ITEM_AMULET
            self.TREASURE_ID,      # 7: ITEM_GOLD_CHALICE
            self.SHIELD_ID,        # 8: ITEM_SHIELD
            self.GUN_ID,           # 9: ITEM_GUN
            self.BOMB_ID,          # 10: ITEM_BOMB
            self.KEY_ID,           # 11: ITEM_KEY
            self.LADDER_UP_ID,     # 12: ITEM_LADDER_UP
            self.LADDER_DOWN_ID,   # 13: ITEM_LADDER_DOWN
        ]
    
    def render(self, state: DarkChambersState) -> jnp.ndarray:
        """Render current game state."""
        # Start with background sprite
        object_raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Camera follows player
        cam_x = jnp.clip(
            state.player_x - GAME_W // 2, 
            0, 
            self.consts.WORLD_WIDTH - GAME_W
        ).astype(jnp.int32)
        cam_y = jnp.clip(
            state.player_y - GAME_H // 2, 
            0, 
            self.consts.WORLD_HEIGHT - GAME_H
        ).astype(jnp.int32)
        
        # Draw walls for current level
        current_level_walls = self.LEVEL_WALLS[state.current_level]
        wall_positions = (current_level_walls[:, 0:2] - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        wall_sizes = current_level_walls[:, 2:4]
        object_raster = self.jr.draw_rects(
            object_raster, 
            positions=wall_positions, 
            sizes=wall_sizes, 
            color_id=self.WALL_ID
        )

        # Level indicator: "LEVEL X" under health bar (top-left)
        # Render the word LEVEL (fixed 5 letters) and a single digit X = current_level+1
        # Using 3x5 digit patterns; for letters, approximate with simple blocks
        level_text_x = 4
        level_text_y = 20  # below health bar

        # Render "LEVEL" as five 3x5 glyphs encoded similarly to digits
        # Simple L,E,V,E,L patterns (manually defined)
        L = jnp.array([[1,0,0],
                       [1,0,0],
                       [1,0,0],
                       [1,0,0],
                       [1,1,1]], dtype=jnp.uint8)
        E = jnp.array([[1,1,1],
                       [1,0,0],
                       [1,1,1],
                       [1,0,0],
                       [1,1,1]], dtype=jnp.uint8)
        V = jnp.array([[1,0,1],
                       [1,0,1],
                       [1,0,1],
                       [0,1,0],
                       [0,1,0]], dtype=jnp.uint8)
        letters = jnp.stack([L, E, V, E, L], axis=0)  # shape (5,5,3)
        spacing = 1
        letter_width = 3
        letter_height = 5
        # Compute positions for letters
        letter_offsets = jnp.arange(5) * (letter_width + spacing)
        xs = jnp.arange(letter_width)
        ys = jnp.arange(letter_height)
        grid_x = xs[None, None, :].repeat(5, axis=0).repeat(letter_height, axis=1)
        grid_y = ys[None, :, None].repeat(5, axis=0).repeat(letter_width, axis=2)
        base_x = level_text_x + letter_offsets[:, None, None]
        px_letters = base_x + grid_x
        py_letters = level_text_y + grid_y
        active_letters = (letters == 1)
        # Mask out inactive pixels by moving them off-screen (-100)
        px_letters_masked = jnp.where(active_letters, px_letters, -100)
        py_letters_masked = jnp.where(active_letters, py_letters, -100)
        flat_px_letters = px_letters_masked.reshape(-1)
        flat_py_letters = py_letters_masked.reshape(-1)
        letter_positions = jnp.stack([flat_px_letters, flat_py_letters], axis=1).astype(jnp.int32)
        letter_sizes = jnp.ones((letter_positions.shape[0], 2), dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=letter_positions,
            sizes=letter_sizes,
            color_id=self.UI_ID
        )

        # Render level number (1-based)
        level_num = (state.current_level + 1).astype(jnp.int32)
        tens = level_num // 10
        ones = level_num % 10
        # Only render ones (levels 1..7)
        digit = ones
        pattern = self.DIGIT_PATTERNS[digit]
        digit_x = level_text_x + 5 * (letter_width + spacing) + 2  # after "LEVEL"
        digit_y = level_text_y
        xs_d = jnp.arange(3)
        ys_d = jnp.arange(5)
        grid_x_d = xs_d[None, :].repeat(5, axis=0)
        grid_y_d = ys_d[:, None].repeat(3, axis=1)
        px_d = digit_x + grid_x_d
        py_d = digit_y + grid_y_d
        active_d = (pattern == 1)
        # Avoid boolean advanced indexing in JIT: mask and flatten
        px_d_masked = jnp.where(active_d, px_d, -100)
        py_d_masked = jnp.where(active_d, py_d, -100)
        digit_positions = jnp.stack([px_d_masked.reshape(-1), py_d_masked.reshape(-1)], axis=1).astype(jnp.int32)
        digit_sizes = jnp.ones((digit_positions.shape[0], 2), dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=digit_positions,
            sizes=digit_sizes,
            color_id=self.UI_ID
        )
        
        # Player (use scaled sprite if available)
        player_screen_x = (state.player_x - cam_x).astype(jnp.int32)
        player_screen_y = (state.player_y - cam_y).astype(jnp.int32)
        object_raster = self.jr.render_at(object_raster, player_screen_x, player_screen_y, self.PLAYER_SCALED_MASK)
        
        # Enemies (colored rectangles)
        enemy_world_pos = state.enemy_positions.astype(jnp.int32)
        enemy_screen_pos = (enemy_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        num_enemies = enemy_screen_pos.shape[0]
        enemy_sizes = jnp.tile(
            jnp.array([self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT], dtype=jnp.int32)[None, :],
            (num_enemies, 1)
        )
        _off = jnp.array([-100, -100], dtype=jnp.int32)
        enemy_active_mask = (state.enemy_active == 1)
        masked_enemy_pos = jnp.where(
            enemy_active_mask[:, None],
            enemy_screen_pos,
            _off
        )
        # Draw each enemy type separately with its color
        for enemy_type, color_id in [(1, self.ZOMBIE_ID), (2, self.WRAITH_ID), (3, self.SKELETON_ID), (4, self.WIZARD_ID), (5, self.GRIM_REAPER_ID)]:
            type_mask = (state.enemy_types == enemy_type) & enemy_active_mask
            type_positions = jnp.where(
                type_mask[:, None],
                masked_enemy_pos,
                _off
            )
            object_raster = self.jr.draw_rects(
                object_raster,
                positions=type_positions,
                sizes=enemy_sizes,
                color_id=color_id
            )
        
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
        
        # Draw items by type using size and color mappings (rectangles)
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
        for t in range(1, 13):  # 1..12 item types (including ladders)
            object_raster = draw_item_type(object_raster, t, masked_item_pos)
        
        # Spawners
        spawner_world_pos = state.spawner_positions.astype(jnp.int32)
        spawner_screen_pos = (spawner_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        spawner_active_mask = state.spawner_active == 1
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
        
        # Health bar (5 boxes proportional to max health=31) top-left
        health_val = jnp.clip(state.health, 0, self.consts.MAX_HEALTH).astype(jnp.int32)
        # Number of filled segments = floor(health * 5 / MAX_HEALTH)
        segments = (health_val * 5) // self.consts.MAX_HEALTH  # 0..5
        bar_indices = jnp.arange(5)
        bar_spacing = ITEM_WIDTH + 2
        bar_x = 4 + bar_indices * bar_spacing
        bar_y = jnp.full(5, 4, dtype=jnp.int32)
        bar_active = bar_indices < segments
        bar_pos_x = jnp.where(bar_active, bar_x, -100)
        bar_positions = jnp.stack([bar_pos_x, bar_y], axis=1).astype(jnp.int32)
        bar_sizes = jnp.tile(
            jnp.array([ITEM_WIDTH, ITEM_HEIGHT], dtype=jnp.int32)[None, :],
            (5, 1)
        )
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=bar_positions,
            sizes=bar_sizes,
            color_id=self.HEART_ID
        )
        # Health digits below bar (supports up to 1000)
        digit_width = 3
        digit_height = 5
        spacing = 1
        thousands = health_val // 1000
        hundreds = (health_val // 100) % 10
        tens = (health_val // 10) % 10
        ones = health_val % 10
        digits = jnp.array([thousands, hundreds, tens, ones], dtype=jnp.int32)
        active_mask = jnp.array([
            thousands > 0,
            (thousands > 0) | (hundreds > 0),
            (thousands > 0) | (hundreds > 0) | (tens > 0),
            True
        ])
        active_count = jnp.sum(active_mask.astype(jnp.int32))
        start_x_digits = 4  # left aligned
        start_y_digits = 4 + ITEM_HEIGHT + 2  # below bar
        position_index = (jnp.cumsum(active_mask.astype(jnp.int32)) - 1)
        base_x = jnp.where(active_mask, start_x_digits + position_index * (digit_width + spacing), -100)
        patterns = self.DIGIT_PATTERNS[digits]
        xs = jnp.arange(digit_width)
        ys = jnp.arange(digit_height)
        grid_x = xs[None, None, :].repeat(4, axis=0).repeat(digit_height, axis=1)
        grid_y = ys[None, :, None].repeat(4, axis=0).repeat(digit_width, axis=2)
        px = base_x[:, None, None] + grid_x
        py = start_y_digits + grid_y
        pixel_active = (patterns == 1) & (base_x[:, None, None] >= 0)
        px = jnp.where(pixel_active, px, -100)
        py = jnp.where(pixel_active, py, -100)
        flat_px = px.reshape(-1)
        flat_py = py.reshape(-1)
        digit_positions = jnp.stack([flat_px, flat_py], axis=1).astype(jnp.int32)
        pixel_sizes = jnp.ones((digit_positions.shape[0], 2), dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=digit_positions,
            sizes=pixel_sizes,
            color_id=self.HEART_ID
        )
        
        # Lives count below health digits
        lives_val = jnp.clip(state.lives, 0, 99).astype(jnp.int32)
        digit_width = 3
        digit_height = 5
        spacing = 1
        lives_tens = lives_val // 10
        lives_ones = lives_val % 10
        lives_digits = jnp.array([lives_tens, lives_ones], dtype=jnp.int32)
        lives_active_mask = jnp.array([lives_tens > 0, True])
        lives_position_index = (jnp.cumsum(lives_active_mask.astype(jnp.int32)) - 1)
        lives_base_x = jnp.where(lives_active_mask, 4 + lives_position_index * (digit_width + spacing), -100)
        lives_patterns = self.DIGIT_PATTERNS[lives_digits]
        lives_xs = jnp.arange(digit_width)
        lives_ys = jnp.arange(digit_height)
        lives_grid_x = lives_xs[None, None, :].repeat(2, axis=0).repeat(digit_height, axis=1)
        lives_grid_y = lives_ys[None, :, None].repeat(2, axis=0).repeat(digit_width, axis=2)
        lives_px = lives_base_x[:, None, None] + lives_grid_x
        lives_py = (4 + ITEM_HEIGHT + 2 + digit_height + 2 + 8) + lives_grid_y  # below health digits
        lives_pixel_active = (lives_patterns == 1) & (lives_base_x[:, None, None] >= 0)
        lives_px = jnp.where(lives_pixel_active, lives_px, -100)
        lives_py = jnp.where(lives_pixel_active, lives_py, -100)
        lives_flat_px = lives_px.reshape(-1)
        lives_flat_py = lives_py.reshape(-1)
        lives_digit_positions = jnp.stack([lives_flat_px, lives_flat_py], axis=1).astype(jnp.int32)
        lives_digit_sizes = jnp.ones((lives_digit_positions.shape[0], 2), dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=lives_digit_positions,
            sizes=lives_digit_sizes,
            color_id=self.HEART_ID
        )
        
        # Numeric score display (7 digits: 0-9999999, no bar)
        score_val = jnp.clip(state.score, 0, 9999999).astype(jnp.int32)
        digit_width = 3
        digit_height = 5
        spacing = 1
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
            True  # ones always shown
        ])
        position_index = (jnp.cumsum(active_mask.astype(jnp.int32)) - 1)
        score_start_x = self.config.game_dimensions[1] - 7 * (digit_width + spacing) - 4  # right-aligned
        base_x = jnp.where(active_mask, score_start_x + position_index * (digit_width + spacing), -100)
        patterns = self.DIGIT_PATTERNS[digits]
        xs = jnp.arange(digit_width)
        ys = jnp.arange(digit_height)
        grid_x = xs[None, None, :].repeat(7, axis=0).repeat(digit_height, axis=1)
        grid_y = ys[None, :, None].repeat(7, axis=0).repeat(digit_width, axis=2)
        px = base_x[:, None, None] + grid_x
        py = 4 + grid_y  # top-right, no bar above
        pixel_active = (patterns == 1) & (base_x[:, None, None] >= 0)
        px = jnp.where(pixel_active, px, -100)
        py = jnp.where(pixel_active, py, -100)
        flat_px = px.reshape(-1)
        flat_py = py.reshape(-1)
        score_digit_positions = jnp.stack([flat_px, flat_py], axis=1).astype(jnp.int32)
        score_digit_sizes = jnp.ones((score_digit_positions.shape[0], 2), dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=score_digit_positions,
            sizes=score_digit_sizes,
            color_id=self.UI_ID
        )
        
        # Removed step counter display from top-center UI
        
        # Bullets
        bullet_world_pos = state.bullet_positions[:, :2].astype(jnp.int32)
        bullet_screen_pos = (bullet_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        bullet_mask = state.bullet_active == 1
        masked_bullet_pos = jnp.where(
            bullet_mask[:, None],
            bullet_screen_pos,
            off_screen
        )
        bullet_sizes = jnp.tile(
            jnp.array([BULLET_WIDTH, BULLET_HEIGHT], dtype=jnp.int32)[None, :],
            (MAX_BULLETS, 1)
        )
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=masked_bullet_pos,
            sizes=bullet_sizes,
            color_id=self.BULLET_ID
        )
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
        
        # Shield indicator in bottom-left corner
        shield_indicator_active = state.shield_active == 1
        shield_x = 4
        shield_y = self.config.game_dimensions[0] - ITEM_HEIGHT - 4
        shield_pos = jnp.where(
            shield_indicator_active,
            jnp.array([[shield_x, shield_y]], dtype=jnp.int32),
            jnp.array([[-100, -100]], dtype=jnp.int32)
        )
        shield_size = jnp.array([[ITEM_WIDTH, ITEM_HEIGHT]], dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=shield_pos,
            sizes=shield_size,
            color_id=self.SHIELD_ID
        )
        
        # Gun indicator next to shield (bottom-left corner)
        gun_indicator_active = state.gun_active == 1
        gun_x = 4 + ITEM_WIDTH + 2  # Next to shield
        gun_y = self.config.game_dimensions[0] - ITEM_HEIGHT - 4
        gun_pos = jnp.where(
            gun_indicator_active,
            jnp.array([[gun_x, gun_y]], dtype=jnp.int32),
            jnp.array([[-100, -100]], dtype=jnp.int32)
        )
        gun_size = jnp.array([[ITEM_WIDTH, ITEM_HEIGHT]], dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=gun_pos,
            sizes=gun_size,
            color_id=self.GUN_ID
        )
        
        # Bomb indicator next to gun (bottom-left corner)
        bomb_has_any = state.bomb_count > 0
        bomb_indicator_x = 4 + (ITEM_WIDTH + 2) * 2  # Next to gun
        bomb_indicator_y = self.config.game_dimensions[0] - ITEM_HEIGHT - 4
        bomb_indicator_pos = jnp.where(
            bomb_has_any,
            jnp.array([[bomb_indicator_x, bomb_indicator_y]], dtype=jnp.int32),
            jnp.array([[-100, -100]], dtype=jnp.int32)
        )
        bomb_indicator_size = jnp.array([[ITEM_WIDTH, ITEM_HEIGHT]], dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=bomb_indicator_pos,
            sizes=bomb_indicator_size,
            color_id=17
        )
        
        # Bomb counter in bottom-right corner
        bomb_count_x = self.config.game_dimensions[1] - 12 - 4  # Space for 2-digit number
        bomb_count_y = self.config.game_dimensions[0] - ITEM_HEIGHT - 4
        
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
        
        # Convert to RGB
        img = self.jr.render_from_palette(object_raster, self.PALETTE)
        return img


class DarkChambersEnv(JaxEnvironment[DarkChambersState, DarkChambersObservation, DarkChambersInfo, DarkChambersConstants]):
    """Main environment."""
    
    def __init__(self, consts: DarkChambersConstants = None):
        super().__init__(consts=consts or DarkChambersConstants())
        self.renderer = DarkChambersRenderer(self.consts)

    def _pos_to_cell(self, x, y):
        """Convert pixel coords (x,y) to nav-grid indices (cy,cx)."""
        cy = (y // CELL_SIZE).astype(jnp.int32)
        cx = (x // CELL_SIZE).astype(jnp.int32)
        cy = jnp.clip(cy, 0, GRID_H - 1)
        cx = jnp.clip(cx, 0, GRID_W - 1)
        return cy, cx

    def _distance_field(
        self,
        level: chex.Array,
        player_x: chex.Array,
        player_y: chex.Array,
        enemy_positions: chex.Array,
        enemy_active: chex.Array,
    ):
        """
        BFS / Dijkstra distance from each nav cell to the player.
        Returns (GRID_H, GRID_W) int32 array of distances.

        Now treats alive enemies as additional obstacles on the nav grid.
        """
        # Base wall grid from the level
        wall_grid = self.renderer.LEVEL_WALL_GRID[level]  # (GRID_H, GRID_W) bool
        H, W = wall_grid.shape

        # --- Enemy occupancy grid (treat alive enemies as walls) ---
        # Use enemy centers to map to nav cells
        centers_x = enemy_positions[:, 0] + self.consts.ENEMY_WIDTH // 2
        centers_y = enemy_positions[:, 1] + self.consts.ENEMY_HEIGHT // 2

        enemy_cy, enemy_cx = self._pos_to_cell(centers_x, centers_y)  # can handle arrays

        enemy_grid = jnp.zeros((H, W), dtype=bool)
        enemy_grid = enemy_grid.at[enemy_cy, enemy_cx].set(enemy_active.astype(bool))

        # Combined obstacle grid: walls OR enemies
        blocked = wall_grid | enemy_grid

        # Distance init
        dist = jnp.full((H, W), BIG_DIST, dtype=jnp.int32)

        # Player seed cell
        py, px = self._pos_to_cell(player_x, player_y)
        dist = dist.at[py, px].set(0)

        frontier = jnp.zeros((H, W), dtype=bool).at[py, px].set(True)

        def body(t, carry):
            dist, frontier = carry

            def shift_frontier(f, dy, dx):
                if dy == -1:
                    f_n = jnp.pad(f[1:, :], ((0, 1), (0, 0)))
                elif dy == 1:
                    f_n = jnp.pad(f[:-1, :], ((1, 0), (0, 0)))
                elif dx == -1:
                    f_n = jnp.pad(f[:, 1:], ((0, 0), (0, 1)))
                else:  # dx == 1
                    f_n = jnp.pad(f[:, :-1], ((0, 0), (1, 0)))
                return f_n

            # 4-neighbourhood
            dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

            new_frontier = jnp.zeros_like(frontier)
            new_dist = dist

            for dy, dx in dirs:
                f_n = shift_frontier(frontier, dy, dx)
                # Only expand into non-blocked cells
                cand = f_n & ~blocked & (dist > t + 1)
                new_frontier = new_frontier | cand
                new_dist = jnp.where(cand, t + 1, new_dist)

            return (new_dist, new_frontier)

        max_steps = GRID_W + GRID_H  # enough to cross the map
        dist, _ = jax.lax.fori_loop(0, max_steps, body, (dist, frontier))
        return dist

    

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> Tuple[DarkChambersObservation, DarkChambersState]:
        """Reset game."""
        
        # Use level 0 walls for initial spawn
        WALLS = self.renderer.LEVEL_WALLS[0]
        
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
        
        # Spawn enemies with random types (favor stronger types)
        key, subkey = jax.random.split(key)
        enemy_types = jax.random.randint(
            subkey, (NUM_ENEMIES,), ENEMY_WRAITH, ENEMY_GRIM_REAPER + 1, dtype=jnp.int32
        )  # Random types from 2 (Wraith) to 5 (Grim Reaper)
        enemy_active = jnp.ones(NUM_ENEMIES, dtype=jnp.int32)
        
        # Spawn spawners (retry until not on wall)
        def spawn_spawner(carry, i):
            positions, key = carry
            
            def try_spawn_spawner(retry_idx, retry_carry):
                pos, key_in, found_valid = retry_carry
                key_out, subkey = jax.random.split(key_in)
                x = jax.random.randint(subkey, (), 50, self.consts.WORLD_WIDTH - 50, dtype=jnp.int32)
                key_out, subkey = jax.random.split(key_out)
                y = jax.random.randint(subkey, (), 50, self.consts.WORLD_HEIGHT - 50, dtype=jnp.int32)
                
                on_wall = check_wall_overlap(x, y, SPAWNER_WIDTH, SPAWNER_HEIGHT)
                # Update position if this is better (not on wall and haven't found valid yet)
                new_pos = jnp.where((~on_wall) & (~found_valid), jnp.array([x, y]), pos)
                new_found = found_valid | (~on_wall)
                
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
        spawner_active = jnp.ones(NUM_SPAWNERS, dtype=jnp.int32)
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
        # Spawn only regular items (indices 3+), leave first 3 for fixed key, exit, ladder_down
        (item_positions_temp, key), _ = jax.lax.scan(spawn_item, (item_positions_init, subkey), jnp.arange(3, NUM_ITEMS))
        
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
        exit_pos = jnp.array([300, 350], dtype=jnp.int32)
        ladder_down_pos = jnp.array([40, 70], dtype=jnp.int32)
        
        # Combine: first 3 positions are key, exit, ladder_down, rest are randomly spawned
        item_positions = item_positions_temp.at[0:3].set(jnp.array([key_pos, exit_pos, ladder_down_pos]))
        
        # Weighted distribution of item types (no extra keys here to enforce exactly one key per level)
        # Reserve first 3 slots for key, exit, ladder_down; rest are regular items without keys
        key, subkey = jax.random.split(key)
        all_item_types = jnp.array([
            ITEM_HEART,          # health +10
            ITEM_POISON,         # -4 health
            ITEM_TRAP,           # -6 health (increase frequency)
            ITEM_STRONGBOX,      # 100 pts
            ITEM_SILVER_CHALICE, # 500 pts
            ITEM_AMULET,         # 1000 pts
            ITEM_GOLD_CHALICE,   # 3000 pts
            ITEM_SHIELD,         # damage reduction
            ITEM_GUN,            # faster shooting
            ITEM_BOMB,           # kill all enemies
        ], dtype=jnp.int32)
        # Probabilities for regular items; normalized to sum to 1.0
        spawn_probs = jnp.array([
            0.12,  # heart
            0.06,  # poison
            0.14,  # trap
            0.08,  # strongbox
            0.08,  # silver chalice (has actually color, just a little bit smaller)
            0.06,  # amulet
            0.06,  # gold chalice
            0.04,  # shield (rarer – single-pickup buff)
            0.035, # gun (rarer – single-pickup buff)
            0.08,  # bomb (grenade) – more common
        ], dtype=jnp.float32)
        spawn_probs = spawn_probs / jnp.sum(spawn_probs)
        # Spawn regular items (leave first 3 for key, exit, ladder_down)
        regular_items = jax.random.choice(subkey, all_item_types, shape=(NUM_ITEMS - 3,), p=spawn_probs)
        # Add key, exit, ladder_down at beginning
        item_types = jnp.concatenate([
            jnp.array([ITEM_KEY, ITEM_LADDER_UP, ITEM_LADDER_DOWN], dtype=jnp.int32),
            regular_items
        ])
        item_active = jnp.ones(NUM_ITEMS, dtype=jnp.int32)
        
        state = DarkChambersState(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            player_direction=jnp.array(0, dtype=jnp.int32),
            enemy_positions=enemy_positions,
            enemy_types=enemy_types,
            enemy_active=enemy_active,
            spawner_positions=spawner_positions,
            spawner_health=spawner_health,
            spawner_active=spawner_active,
            spawner_timers=spawner_timers,
            bullet_positions=jnp.zeros((MAX_BULLETS, 4), dtype=jnp.int32),
            bullet_active=jnp.zeros(MAX_BULLETS, dtype=jnp.int32),
            enemy_bullet_positions=jnp.zeros((ENEMY_MAX_BULLETS, 4), dtype=jnp.int32),
            enemy_bullet_active=jnp.zeros(ENEMY_MAX_BULLETS, dtype=jnp.int32),
            health=jnp.array(self.consts.STARTING_HEALTH, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            item_positions=item_positions,
            item_types=item_types,
            item_active=item_active,
            has_key=jnp.array(0, dtype=jnp.int32),  # Start without key
            shield_active=jnp.array(0, dtype=jnp.int32),
            gun_active=jnp.array(0, dtype=jnp.int32),
            bomb_count=jnp.array(0, dtype=jnp.int32),
            last_fire_step=jnp.array(-1000, dtype=jnp.int32),  # Initialize to far past
            current_level=jnp.array(0, dtype=jnp.int32),  # Start at level 0
            ladder_timer=jnp.array(0, dtype=jnp.int32),   # Not on ladder initially
            lives=jnp.array(MAX_LIVES, dtype=jnp.int32),  # Start with MAX_LIVES
            step_counter=jnp.array(0, dtype=jnp.int32),
            death_counter=jnp.array(0, dtype=jnp.int32),
            key=key,
        )
        
        obs = self._get_observation(state)
        return obs, state
    

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DarkChambersState, action: int) -> Tuple[DarkChambersObservation, DarkChambersState, float, bool, DarkChambersInfo]:
        """Step forward."""
        a = jnp.asarray(action)

        # If we're in a death freeze, either decrement or respawn now
        def handle_freeze(_: None):
            # When counter > 1: just decrement and freeze everything else
            def dec_only(_: None):
                frozen_state = state._replace(
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
                respawn_x = jnp.where(state.current_level == 0, self.consts.PLAYER_START_X, jnp.array(40, dtype=jnp.int32))
                respawn_y = jnp.where(state.current_level == 0, self.consts.PLAYER_START_Y, jnp.array(70, dtype=jnp.int32))

                # KEEP EVERYTHING: score, collected items, level, powerups, etc.
                # Only restore health and clear bullets for clean respawn
                respawned_state = state._replace(
                    player_x=respawn_x,
                    player_y=respawn_y,
                    health=jnp.array(self.consts.STARTING_HEALTH, dtype=jnp.int32),
                    bullet_positions=jnp.zeros((MAX_BULLETS, 4), dtype=jnp.int32),
                    bullet_active=jnp.zeros((MAX_BULLETS,), dtype=jnp.int32),
                    enemy_bullet_positions=jnp.zeros((ENEMY_MAX_BULLETS, 4), dtype=jnp.int32),
                    enemy_bullet_active=jnp.zeros((ENEMY_MAX_BULLETS,), dtype=jnp.int32),
                    death_counter=jnp.array(0, dtype=jnp.int32),
                    step_counter=state.step_counter + 1,
                )
                obs = self._get_observation(respawned_state)
                reward = self._get_reward(state, respawned_state)
                done = self._get_done(respawned_state)
                info = self._get_info(respawned_state)
                return obs, respawned_state, reward, done, info

            return jax.lax.cond(state.death_counter > 1, dec_only, respawn_now, operand=None)

        def handle_normal(_: None):
            # Normal logic continues below (existing code)
            # We will return at the end of this function.
            
            # Track player direction from action
            new_direction = jnp.where(a == Action.RIGHT, 0,
                            jnp.where(a == Action.LEFT, 1,
                            jnp.where(a == Action.UP, 2,
                            jnp.where(a == Action.DOWN, 3, state.player_direction))))
            
            dx = jnp.where(a == Action.LEFT, -self.consts.PLAYER_SPEED, 
                   jnp.where(a == Action.RIGHT, self.consts.PLAYER_SPEED, 0))
            dy = jnp.where(a == Action.UP, -self.consts.PLAYER_SPEED, 
                   jnp.where(a == Action.DOWN, self.consts.PLAYER_SPEED, 0))
            
            prop_x = state.player_x + dx
            prop_y = state.player_y + dy
            
            prop_x = jnp.clip(prop_x, 0, self.consts.WORLD_WIDTH - 1)
            prop_y = jnp.clip(prop_y, 0, self.consts.WORLD_HEIGHT - 1)
            
            # Wall collision check - use walls for current level
            WALLS = self.renderer.LEVEL_WALLS[state.current_level]
            
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
            
            # Shooting - spawn bullet on FIRE action (or detonate bomb on double-tap)
            fire_pressed = (a == Action.FIRE) | (a == Action.UPFIRE) | (a == Action.DOWNFIRE) | (a == Action.LEFTFIRE) | (a == Action.RIGHTFIRE) | \
                           (a == Action.UPRIGHTFIRE) | (a == Action.UPLEFTFIRE) | (a == Action.DOWNRIGHTFIRE) | (a == Action.DOWNLEFTFIRE)
            
            # Double-tap detection: fire pressed within DOUBLE_TAP_WINDOW steps
            steps_since_last_fire = state.step_counter - state.last_fire_step
            is_double_tap = fire_pressed & (steps_since_last_fire <= DOUBLE_TAP_WINDOW) & (steps_since_last_fire > 0)
            has_bombs = state.bomb_count > 0
            should_detonate_bomb = is_double_tap & has_bombs
            
            # Update last_fire_step when fire is pressed
            new_last_fire_step = jnp.where(fire_pressed, state.step_counter, state.last_fire_step)
            
            # Find first inactive bullet slot
            first_inactive = jnp.argmax(state.bullet_active == 0)
            can_spawn = jnp.any(state.bullet_active == 0)
            
            # Fire rate limiting: ensure minimum time between shots
            # Gun powerup reduces fire rate limit to 1 (even faster shooting)
            fire_rate_threshold = jnp.where(state.gun_active == 1, 1, FIRE_RATE_LIMIT)
            can_fire_now = steps_since_last_fire >= fire_rate_threshold
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
            updated_bullet_active = state.bullet_active & in_bounds.astype(jnp.int32) & (~bullet_wall_collisions).astype(jnp.int32)
            
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

            # Wizard shooting: sporadically fire towards player
            can_enemy_spawn_gate = (state.step_counter % WIZARD_SHOOT_INTERVAL) == 0
            wizard_mask = (state.enemy_active == 1) & (state.enemy_types == ENEMY_WIZARD)
            has_wizard = jnp.any(wizard_mask)
            first_wizard_idx = jnp.argmax(wizard_mask.astype(jnp.int32))
            enemy_can_spawn = can_enemy_spawn_gate & has_wizard
            # Use current enemy positions for shooting; movement updates happen later
            wiz_pos = state.enemy_positions[first_wizard_idx]
            wiz_cx = wiz_pos[0] + self.consts.ENEMY_WIDTH // 2
            wiz_cy = wiz_pos[1] + self.consts.ENEMY_HEIGHT // 2
            p_cx = new_x + self.consts.PLAYER_WIDTH // 2
            p_cy = new_y + self.consts.PLAYER_HEIGHT // 2
            dir_x_e = jnp.sign(p_cx - wiz_cx) * ENEMY_BULLET_SPEED
            dir_y_e = jnp.sign(p_cy - wiz_cy) * ENEMY_BULLET_SPEED
            enemy_bullet_spawn = jnp.array([wiz_cx, wiz_cy, dir_x_e, dir_y_e], dtype=jnp.int32)
            e_first_inactive = jnp.argmax(state.enemy_bullet_active == 0)
            e_can_spawn = jnp.any(state.enemy_bullet_active == 0) & enemy_can_spawn
            new_enemy_bullet_positions = jnp.where(
                (jnp.arange(ENEMY_MAX_BULLETS)[:, None] == e_first_inactive) & e_can_spawn,
                enemy_bullet_spawn,
                state.enemy_bullet_positions
            )
            new_enemy_bullet_active = jnp.where(
                (jnp.arange(ENEMY_MAX_BULLETS) == e_first_inactive) & e_can_spawn,
                1,
                state.enemy_bullet_active
            )
            # Move enemy bullets
            moved_enemy_bullets = state.enemy_bullet_positions + jnp.concatenate([
                state.enemy_bullet_positions[:, 2:], jnp.zeros((ENEMY_MAX_BULLETS, 2), dtype=jnp.int32)
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
                state.enemy_bullet_active[:, None] == 1,
                moved_enemy_bullets,
                state.enemy_bullet_positions
            )
            updated_enemy_bullet_active = state.enemy_bullet_active & e_in_bounds.astype(jnp.int32) & (~enemy_bullet_wall_collisions).astype(jnp.int32)
            final_enemy_bullet_positions = jnp.where(
                (jnp.arange(ENEMY_MAX_BULLETS)[:, None] == e_first_inactive) & e_can_spawn,
                enemy_bullet_spawn,
                updated_enemy_bullet_positions
            )
            final_enemy_bullet_active = jnp.where(
                (jnp.arange(ENEMY_MAX_BULLETS) == e_first_inactive) & e_can_spawn,
                1,
                updated_enemy_bullet_active
            )
            
            # Enemy random walk
            rng, subkey = jax.random.split(state.key)
            enemy_deltas = jax.random.randint(subkey, (NUM_ENEMIES, 2), -1, 2, dtype=jnp.int32)
            enemy_alive = state.enemy_active == 1
            
            player_center = jnp.array(
                [new_x + self.consts.PLAYER_WIDTH // 2,
                new_y + self.consts.PLAYER_HEIGHT // 2],
                dtype=jnp.int32
            )
            
            enemy_centers = state.enemy_positions + jnp.array(
                [self.consts.ENEMY_WIDTH // 2, self.consts.ENEMY_HEIGHT // 2],
                dtype=jnp.int32
            )
            
            dx = enemy_centers[:, 0] - player_center[0]
            dy = enemy_centers[:, 1] - player_center[1]
            dist2 = dx * dx + dy * dy
            
            chase_mask = (dist2 <= (CHASE_RADIUS * CHASE_RADIUS)) & enemy_alive
            
            
            # distance field to player over nav grid
            # enemies are treated as dynamic obstacles
            dist_field = self._distance_field(
                state.current_level,
                new_x,
                new_y,
                state.enemy_positions,
                state.enemy_active,
            )
            
            
            def enemy_step(pos, alive):
                # pos: [x, y] top-left of enemy
                # compute nav-cell from enemy center
                enemy_center = pos + jnp.array(
                    [self.consts.ENEMY_WIDTH // 2, self.consts.ENEMY_HEIGHT // 2],
                    dtype=jnp.int32,
                )
                cy, cx = self._pos_to_cell(enemy_center[0], enemy_center[1])
            
                cy = jnp.clip(cy, 0, GRID_H - 1)
                cx = jnp.clip(cx, 0, GRID_W - 1)
            
                # 4-neighbourhood (no "stay in place" cell)
                # 8-neighbourhood around (cy, cx)
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
            
                # distances from distance field
                dists = dist_field[ny, nx]
            
                # target centers of neighbour cells
                target_centers = jnp.stack(
                    [
                        nx * CELL_SIZE + CELL_SIZE // 2,
                        ny * CELL_SIZE + CELL_SIZE // 2,
                    ],
                    axis=1,
                )  # shape (4, 2)
            
                # step vectors: 1 px per axis towards target center
                step_vecs = jnp.sign(target_centers - enemy_center).astype(jnp.int32)
            
                # --- collision check for each candidate step ---
                def collides_with_wall(step_vec):
                    new_pos = pos + step_vec
            
                    ex = new_pos[0] + ENEMY_COLLISION_MARGIN
                    ey = new_pos[1] + ENEMY_COLLISION_MARGIN
                    w = self.consts.ENEMY_WIDTH  - 2 * ENEMY_COLLISION_MARGIN
                    h = self.consts.ENEMY_HEIGHT - 2 * ENEMY_COLLISION_MARGIN
            
                    wx = WALLS[:, 0]
                    wy = WALLS[:, 1]
                    ww = WALLS[:, 2]
                    wh = WALLS[:, 3]
            
                    e_overlap_x = (ex <= (wx + ww - 1)) & ((ex + w - 1) >= wx)
                    e_overlap_y = (ey <= (wy + wh - 1)) & ((ey + h - 1) >= wy)
                    return jnp.any(e_overlap_x & e_overlap_y)
            
                collisions = jax.vmap(collides_with_wall)(step_vecs)  # (4,)
            
                # valid = in grid, no wall collision, and enemy is alive
                valid = in_bounds & (~collisions) & (alive == 1)
            
                # neighbours that are invalid get BIG_DIST so they won’t be chosen
                dists_valid = jnp.where(valid, dists, BIG_DIST)
            
                best_idx = jnp.argmin(dists_valid)
                best_step = step_vecs[best_idx]
            
                # if *all* neighbours invalid, don't move
                any_valid = jnp.any(valid)
                best_step = jnp.where(
                    any_valid,
                    best_step,
                    jnp.array([0, 0], dtype=jnp.int32),
                )
            
                # final step is zero if enemy is dead
                return best_step * alive.astype(jnp.int32)
            
            
            chosen_step = jax.vmap(enemy_step)(state.enemy_positions, enemy_alive)
            
            # --- idle "circular-ish" movement (8-dir orbit) ---
            idxs = jnp.arange(NUM_ENEMIES, dtype=jnp.int32)
            phase = (state.step_counter // 12 + idxs) % 8          # 12 = frames per segment
            idle_step = ORBIT_DIRS[phase] * IDLE_SPEED             # (NUM_ENEMIES, 2)
            idle_step = idle_step * enemy_alive[:, None].astype(jnp.int32)  # dead enemies don't move
            
            # --- choose chase vs idle ---
            final_step = jnp.where(chase_mask[:, None], chosen_step, idle_step)
            
            # Helper: check collision for a single enemy position (keep your existing function)
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
            
            # Re-use your axis-separate movement logic
            def move_enemy(enemy_pos, step_vec):
                cur = enemy_pos
            
                step_full = step_vec
                step_x = jnp.array([step_vec[0], 0], dtype=jnp.int32)
                step_y = jnp.array([0, step_vec[1]], dtype=jnp.int32)
            
                def clip_pos(pos):
                    return jnp.clip(
                        pos,
                        jnp.array([0, 0], dtype=jnp.int32),
                        jnp.array([self.consts.WORLD_WIDTH - 1, self.consts.WORLD_HEIGHT - 1], dtype=jnp.int32),
                    )
            
                pos_full = clip_pos(cur + step_full)
                pos_x    = clip_pos(cur + step_x)
                pos_y    = clip_pos(cur + step_y)
            
                col_full = check_enemy_collision(pos_full)
                col_x    = check_enemy_collision(pos_x)
                col_y    = check_enemy_collision(pos_y)
            
                use_full = ~col_full
                use_x    = col_full & ~col_x
                use_y    = col_full & col_x & ~col_y
            
                pos_after = jnp.where(
                    use_full, pos_full,
                    jnp.where(
                        use_x, pos_x,
                        jnp.where(use_y, pos_y, cur),
                    ),
                )
                return pos_after
            
            # move enemies using your existing collision-aware move_enemy()
            new_enemy_positions = jax.vmap(move_enemy)(state.enemy_positions, final_step)
            
            def resolve_enemy_overlaps(cand_positions, prev_positions, active):
                # cand_positions, prev_positions: (NUM_ENEMIES, 2)
                # active: (NUM_ENEMIES,) 1/0
                w = self.consts.ENEMY_WIDTH
                h = self.consts.ENEMY_HEIGHT
            
                ex = cand_positions[:, 0]
                ey = cand_positions[:, 1]
            
                ex_i = ex[:, None]
                ex_j = ex[None, :]
                ey_i = ey[:, None]
                ey_j = ey[None, :]
            
                overlap_x = (ex_i <= ex_j + w - 1) & (ex_i + w - 1 >= ex_j)
                overlap_y = (ey_i <= ey_j + h - 1) & (ey_i + h - 1 >= ey_j)
                overlap = overlap_x & overlap_y
            
                # only consider active enemies
                a_i = active[:, None].astype(bool)
                a_j = active[None, :].astype(bool)
                overlap = overlap & a_i & a_j
            
                # enemy with smaller index "wins" the tile, later ones lose
                idx = jnp.arange(NUM_ENEMIES)
                earlier = idx[:, None] < idx[None, :]  # True for (i,j) with i<j
            
                loser_matrix = overlap & earlier  # (i,j) True => j loses to i
                loser_flags = jnp.any(loser_matrix, axis=0)  # for each j
            
                # losers revert to previous position
                final_positions = jnp.where(
                    loser_flags[:, None],
                    prev_positions,
                    cand_positions,
                )
                return final_positions
            
            new_enemy_positions = resolve_enemy_overlaps(
                new_enemy_positions,
                state.enemy_positions,
                state.enemy_active,
            )
            
            
            # Item pickup detection
            def check_item_collision(item_pos):
                overlap_x = (new_x <= (item_pos[0] + ITEM_WIDTH - 1)) & ((new_x + self.consts.PLAYER_WIDTH - 1) >= item_pos[0])
                overlap_y = (new_y <= (item_pos[1] + ITEM_HEIGHT - 1)) & ((new_y + self.consts.PLAYER_HEIGHT - 1) >= item_pos[1])
                return overlap_x & overlap_y
            
            item_collisions = jax.vmap(check_item_collision)(state.item_positions)
            item_collisions = item_collisions & (state.item_active == 1)
            
            # Apply item effects
            collected_hearts = jnp.sum(item_collisions & (state.item_types == ITEM_HEART))
            collected_poison = jnp.sum(item_collisions & (state.item_types == ITEM_POISON))
            collected_traps = jnp.sum(item_collisions & (state.item_types == ITEM_TRAP))
            collected_shields = jnp.any(item_collisions & (state.item_types == ITEM_SHIELD))
            collected_guns = jnp.any(item_collisions & (state.item_types == ITEM_GUN))
            collected_bombs = jnp.sum(item_collisions & (state.item_types == ITEM_BOMB))
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
            
            # Update bomb count (capped at MAX_BOMBS)
            new_bomb_count = jnp.clip(state.bomb_count + collected_bombs, 0, MAX_BOMBS)
            
            # Ladder interaction: check if player is standing on a ladder
            on_ladder_up = jnp.any(item_collisions & (state.item_types == ITEM_LADDER_UP))
            on_ladder_down = jnp.any(item_collisions & (state.item_types == ITEM_LADDER_DOWN))
            on_any_ladder = on_ladder_up | on_ladder_down
            
            # Gate UP ladder on key possession
            on_ladder_up_gated = on_ladder_up & (new_has_key == 1)
            
            # Update ladder timer: increment if on ladder, reset if not
            new_ladder_timer = jnp.where(
                on_any_ladder,
                state.ladder_timer + 1,
                jnp.array(0, dtype=jnp.int32)
            )
            
            # Level transition when timer reaches threshold
            # Only allow UP transition if player has key
            should_change_level = new_ladder_timer >= LADDER_INTERACTION_TIME
            going_up = should_change_level & on_ladder_up_gated
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
                500,                  # 5 SILVER CHALICE
                1000,                 # 6 AMULET
                3000,                 # 7 GOLD CHALICE
                0,                    # 8 SHIELD
                0,                    # 9 GUN
                0,                    # 10 BOMB
                0,                    # 11 LADDER_UP
                0,                    # 12 LADDER_DOWN
            ], dtype=jnp.int32)
            item_points = points_by_type[state.item_types]
            gained_points = jnp.sum(item_points * item_collisions.astype(jnp.int32))
            new_score = state.score + gained_points
            
            # Remove collected items (but NOT ladders - they stay persistent, and keys are consumable)
            is_ladder = (state.item_types == ITEM_LADDER_UP) | (state.item_types == ITEM_LADDER_DOWN)
            is_key = (state.item_types == ITEM_KEY)
            should_remove = item_collisions & (~is_ladder) | (is_key & item_collisions)  # Remove keys when collected
            new_item_active = jnp.where(should_remove, 0, state.item_active)
            
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
            enemy_hit = jnp.any(all_collisions, axis=0)
            
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
            
            enemy_hit = enemy_hit | bomb_kills_enemies
            
            # Reduce bomb count when detonated
            bomb_count_after_detonation = jnp.where(should_detonate_bomb, new_bomb_count - 1, new_bomb_count)
            
            # Enemy mutation system: when hit, enemy mutates to weaker form
            new_enemy_types = jnp.where(bomb_kills_enemies, 0, 
                              jnp.where(enemy_hit & (~bomb_kills_enemies), state.enemy_types - 1, state.enemy_types))
            
            # Award points when enemy is killed
            zombies_killed = (enemy_hit & (~bomb_kills_enemies)) & (state.enemy_types == ENEMY_ZOMBIE)
            
            # Bomb kills award points for all enemy types and bullet progression hits
            zombies_bombed = bomb_kills_enemies & (state.enemy_types == ENEMY_ZOMBIE)
            wraiths_bombed = bomb_kills_enemies & (state.enemy_types == ENEMY_WRAITH)
            skeletons_bombed = bomb_kills_enemies & (state.enemy_types == ENEMY_SKELETON)
            wizards_bombed = bomb_kills_enemies & (state.enemy_types == ENEMY_WIZARD)
            grim_reapers_bombed = bomb_kills_enemies & (state.enemy_types == ENEMY_GRIM_REAPER)
            wraiths_hit = (enemy_hit & (~bomb_kills_enemies)) & (state.enemy_types == ENEMY_WRAITH)
            skeletons_hit = (enemy_hit & (~bomb_kills_enemies)) & (state.enemy_types == ENEMY_SKELETON)
            wizards_hit = (enemy_hit & (~bomb_kills_enemies)) & (state.enemy_types == ENEMY_WIZARD)
            grim_reapers_hit = (enemy_hit & (~bomb_kills_enemies)) & (state.enemy_types == ENEMY_GRIM_REAPER)
            
            enemy_kill_score = (
                jnp.sum(zombies_killed | zombies_bombed) * self.consts.ZOMBIE_POINTS +
                jnp.sum(wraiths_hit | wraiths_bombed) * self.consts.WRAITH_POINTS +
                jnp.sum(skeletons_hit | skeletons_bombed) * self.consts.SKELETON_POINTS +
                jnp.sum(wizards_hit | wizards_bombed) * self.consts.WIZARD_POINTS +
                jnp.sum(grim_reapers_hit | grim_reapers_bombed) * self.consts.GRIM_REAPER_POINTS
            )
            
            # Deactivate enemies that have been killed (type becomes 0)
            new_enemy_active = jnp.where(new_enemy_types <= 0, 0, state.enemy_active)
            
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
                
                # Random item type (heart or treasures)
                key, subkey = jax.random.split(key)
                drop_types = jnp.array([ITEM_HEART, ITEM_STRONGBOX, ITEM_SILVER_CHALICE, ITEM_AMULET], dtype=jnp.int32)
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
            
            rng, subkey = jax.random.split(state.key)
            (final_item_positions, final_item_types, final_item_active, rng), _ = jax.lax.scan(
                add_spawner_drop,
                (state.item_positions, state.item_types, new_item_active, subkey),
                jnp.arange(NUM_SPAWNERS)
            )
            
            # Deactivate bullets that hit anything
            bullet_hit_spawner = jnp.any(spawner_collisions, axis=1)
            final_bullet_active2 = final_bullet_active & (~(bullet_hit_enemy | bullet_hit_spawner)).astype(jnp.int32)
            
            # Add enemy kill score to total
            final_score = new_score + enemy_kill_score
            
            # Enemy contact damage (varies by enemy type)
            alive_enemies_mask = new_enemy_active == 1
            enemy_x = final_enemy_positions[:, 0]
            enemy_y = final_enemy_positions[:, 1]
            overlap_x = (new_x <= (enemy_x + self.consts.ENEMY_WIDTH - 1)) & ((new_x + self.consts.PLAYER_WIDTH - 1) >= enemy_x)
            overlap_y = (new_y <= (enemy_y + self.consts.ENEMY_HEIGHT - 1)) & ((new_y + self.consts.PLAYER_HEIGHT - 1) >= enemy_y)
            enemy_contacts = overlap_x & overlap_y & alive_enemies_mask
            
            # Calculate damage based on enemy type
            damage_by_type = jnp.array([
                0,  # Dead enemy (type 0)
                self.consts.ZOMBIE_DAMAGE,
                self.consts.WRAITH_DAMAGE,
                self.consts.SKELETON_DAMAGE,
                self.consts.WIZARD_DAMAGE,
                self.consts.GRIM_REAPER_DAMAGE
            ], dtype=jnp.int32)
            
            # Sum damage from all contacted enemies
            contact_damage = jnp.sum(
                jnp.where(enemy_contacts, damage_by_type[new_enemy_types], 0)
            )
            # Apply shield damage reduction (50% if shield active)
            shield_multiplier = jnp.where(new_shield_active == 1, 0.5, 1.0)
            reduced_damage = (contact_damage * shield_multiplier).astype(jnp.int32)
            final_health = jnp.clip(new_health - reduced_damage, 0, self.consts.MAX_HEALTH)
            
            # Enemy bullet damage to player only
            def enemy_bullet_hits_player(bpos):
                bx, by = bpos[0], bpos[1]
                overlap_x = (bx <= (new_x + self.consts.PLAYER_WIDTH - 1)) & ((bx + ENEMY_BULLET_WIDTH - 1) >= new_x)
                overlap_y = (by <= (new_y + self.consts.PLAYER_HEIGHT - 1)) & ((by + ENEMY_BULLET_HEIGHT - 1) >= new_y)
                return overlap_x & overlap_y
            enemy_bullet_hits = jax.vmap(enemy_bullet_hits_player)(final_enemy_bullet_positions[:, :2]) & (final_enemy_bullet_active == 1)
            enemy_bullet_damage = jnp.sum(enemy_bullet_hits.astype(jnp.int32))
            final_health = jnp.clip(final_health - enemy_bullet_damage, 0, self.consts.MAX_HEALTH)
            final_enemy_bullet_active3 = final_enemy_bullet_active & (~enemy_bullet_hits).astype(jnp.int32)
            
            # Spawner logic: spawn enemies near active spawners
            new_spawner_timers = state.spawner_timers - 1
            should_spawn_enemy = (new_spawner_timers <= 0) & (new_spawner_active == 1)
            
            # spawn each enemy exactly in the middle of the spawner
            def try_spawn_from_spawner(carry, spawner_idx):
                enemy_pos, enemy_types_arr, enemy_active_arr, key = carry
                should_spawn = should_spawn_enemy[spawner_idx]
                
                # Find first inactive enemy slot
                first_inactive = jnp.argmax(enemy_active_arr == 0)
                can_spawn = jnp.any(enemy_active_arr == 0) & should_spawn
                
                # Random enemy type
                key, subkey = jax.random.split(key)
                spawn_type = jax.random.randint(subkey, (), ENEMY_WRAITH, ENEMY_GRIM_REAPER + 1, dtype=jnp.int32)
                
                spawner_pos = state.spawner_positions[spawner_idx]
                
                # Spawn enemy directly inside the spawner (centered)
                spawn_offset_x = (SPAWNER_WIDTH - self.consts.ENEMY_WIDTH) // 2
                spawn_offset_y = (SPAWNER_HEIGHT - self.consts.ENEMY_HEIGHT) // 2
                spawn_pos = spawner_pos + jnp.array([spawn_offset_x, spawn_offset_y])
                
                # Update arrays
                new_pos = jnp.where(
                    (jnp.arange(NUM_ENEMIES)[:, None] == first_inactive) & can_spawn,
                    spawn_pos,
                    enemy_pos
                )
                new_types = jnp.where(
                    (jnp.arange(NUM_ENEMIES) == first_inactive) & can_spawn,
                    spawn_type,
                    enemy_types_arr
                )
                new_active = jnp.where(
                    (jnp.arange(NUM_ENEMIES) == first_inactive) & can_spawn,
                    1,
                    enemy_active_arr
                )
                
                return (new_pos, new_types, new_active, key), None
            
            (spawned_enemy_positions, spawned_enemy_types, spawned_enemy_active, rng), _ = jax.lax.scan(
                try_spawn_from_spawner,
                (final_enemy_positions, new_enemy_types, new_enemy_active, rng),
                jnp.arange(NUM_SPAWNERS)
            )
            
            # Reset spawner timers when they spawn
            final_spawner_timers = jnp.where(should_spawn_enemy, SPAWNER_SPAWN_INTERVAL, new_spawner_timers)
            
            # Level transition: respawn items and reset player position when level changes
            level_changed = new_level != state.current_level
            
            # Reset player position on level change
            spawn_x_up = jnp.array(40, dtype=jnp.int32)  # At down ladder after going up
            spawn_y_up = jnp.array(70, dtype=jnp.int32)
            spawn_x_down = jnp.array(40, dtype=jnp.int32)  # At up ladder after going down
            spawn_y_down = jnp.array(40, dtype=jnp.int32)
            
            # Determine spawn position based on which ladder was used
            spawn_x = jnp.where(going_up, spawn_x_up,
                      jnp.where(going_down, spawn_x_down, self.consts.PLAYER_START_X))
            spawn_y = jnp.where(going_up, spawn_y_up,
                      jnp.where(going_down, spawn_y_down, self.consts.PLAYER_START_Y))
            
            transition_x = jnp.where(level_changed, spawn_x, new_x)
            transition_y = jnp.where(level_changed, spawn_y, new_y)
            
            # Respawn all items on level change
            def respawn_items_for_level(key):
                # New level walls for collision checks
                WALLS_NEW = self.renderer.LEVEL_WALLS[new_level]
                
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
                exit_pos = jnp.array([300, 350], dtype=jnp.int32)
                ladder_down_pos = jnp.array([40, 70], dtype=jnp.int32)
                
                # Generate random positions for regular items with retries to avoid walls
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
                init_positions = jnp.zeros((NUM_ITEMS - 3, 2), dtype=jnp.int32)  # -3 for key, exit, ladder_down
                (regular_positions, key), _ = jax.lax.scan(spawn_regular_item, (init_positions, sk), jnp.arange(NUM_ITEMS - 3))
                
                # Combine: first 3 are key, exit, ladder_down, rest are regular items
                new_positions = jnp.concatenate([
                    key_pos[None, :],
                    exit_pos[None, :],
                    ladder_down_pos[None, :],
                    regular_positions
                ], axis=0)
                
                # Generate new item types
                key, subkey = jax.random.split(key)
                all_item_types = jnp.array([
                    ITEM_HEART, ITEM_POISON, ITEM_TRAP, ITEM_STRONGBOX,
                    ITEM_SILVER_CHALICE, ITEM_AMULET, ITEM_GOLD_CHALICE,
                    ITEM_SHIELD, ITEM_GUN, ITEM_BOMB
                ], dtype=jnp.int32)
                spawn_probs = jnp.array([0.18, 0.08, 0.16, 0.11, 0.10, 0.08, 0.08, 0.06, 0.05, 0.10], dtype=jnp.float32)
                regular_items = jax.random.choice(subkey, all_item_types, shape=(NUM_ITEMS - 3,), p=spawn_probs)
                new_types = jnp.concatenate([jnp.array([ITEM_KEY, ITEM_LADDER_UP, ITEM_LADDER_DOWN]), regular_items])
                new_active = jnp.ones(NUM_ITEMS, dtype=jnp.int32)
                
                return new_positions, new_types, new_active, key
            
            # Apply respawn only if level changed
            respawned_positions, respawned_types, respawned_active, rng = respawn_items_for_level(rng)
            transition_item_positions = jnp.where(level_changed, respawned_positions, final_item_positions)
            transition_item_types = jnp.where(level_changed, respawned_types, final_item_types)
            transition_item_active = jnp.where(level_changed, respawned_active, final_item_active)
            
            # On level change, ensure enemies are not overlapping new level walls
            WALLS_NEWLVL = self.renderer.LEVEL_WALLS[new_level]
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
                cur = spawned_enemy_positions[i]
                is_active = spawned_enemy_active[i] == 1
            
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
                # If enemy inactive or level didn't change, keep original
                final_pos = jnp.where(level_changed & is_active, final_pos, cur)
                positions_arr = positions_arr.at[i].set(final_pos)
                return (positions_arr, key_out), None
            
            rng, sk = jax.random.split(rng)
            init_epos = spawned_enemy_positions
            (relocated_enemy_positions, rng), _ = jax.lax.scan(relocate_enemy, (init_epos, sk), jnp.arange(NUM_ENEMIES))
            
            # On level change going UP, spawn additional zombies
            def spawn_level_enemies(key_in, positions_in, types_in, active_in):
                """Spawn 3-5 zombies when entering a new higher level."""
                rng_local, subkey = jax.random.split(key_in)
                num_to_spawn = 4  # Spawn 4 zombies per new level
                
                def spawn_one_zombie(carry, spawn_idx):
                    pos_arr, typ_arr, act_arr, key_loc = carry
                    # Find first inactive enemy slot
                    first_inactive_idx = jnp.argmax(act_arr == 0)
                    slot_available = jnp.any(act_arr == 0)
                    
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
                    return (pos_arr, typ_arr, act_arr, key_loc), None
                
                (new_pos, new_typ, new_act, rng_local), _ = jax.lax.scan(
                    spawn_one_zombie,
                    (positions_in, types_in, active_in, subkey),
                    jnp.arange(num_to_spawn)
                )
                return new_pos, new_typ, new_act, rng_local
            
            # Apply enemy spawning only if going up to a new level
            spawned_pos, spawned_typ, spawned_act, rng = jax.lax.cond(
                level_changed & going_up,
                lambda _: spawn_level_enemies(rng, relocated_enemy_positions, spawned_enemy_types, spawned_enemy_active),
                lambda _: (relocated_enemy_positions, spawned_enemy_types, spawned_enemy_active, rng),
                operand=None
            )
            
            final_enemy_positions_after_level = spawned_pos
            final_enemy_types_after_level = spawned_typ
            final_enemy_active_after_level = spawned_act
            
            # On level change, respawn spawners avoiding new level walls
            def respawn_spawners_for_level(key):
                # Use renderer-level walls structure
                WALLS_NEW = self.renderer.LEVEL_WALLS[new_level]
            
                def check_wall_overlap_sp(x, y):
                    wx = WALLS_NEW[:, 0]
                    wy = WALLS_NEW[:, 1]
                    ww = WALLS_NEW[:, 2]
                    wh = WALLS_NEW[:, 3]
                    overlap_x = (x <= (wx + ww - 1)) & ((x + SPAWNER_WIDTH - 1) >= wx)
                    overlap_y = (y <= (wy + wh - 1)) & ((y + SPAWNER_HEIGHT - 1) >= wy)
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
                new_sp_active = jnp.ones(NUM_SPAWNERS, dtype=jnp.int32)
                key, sk = jax.random.split(key)
                new_sp_timers = jax.random.randint(sk, (NUM_SPAWNERS,), 0, SPAWNER_SPAWN_INTERVAL, dtype=jnp.int32)
                return new_sp_positions, new_sp_health, new_sp_active, new_sp_timers, key
            
            (transition_sp_positions, transition_sp_health, transition_sp_active, transition_sp_timers, rng) = respawn_spawners_for_level(rng)
            # When level changes, use freshly respawned spawners; otherwise use the updated values
            use_sp_positions = jnp.where(level_changed[..., None], transition_sp_positions, state.spawner_positions)
            use_sp_health = jnp.where(level_changed, transition_sp_health, new_spawner_health)
            use_sp_active = jnp.where(level_changed, transition_sp_active, new_spawner_active)
            use_sp_timers = jnp.where(level_changed, transition_sp_timers, final_spawner_timers)
            
            # Death handling: start freeze instead of instant respawn
            player_died = final_health <= 0
            new_lives = jnp.where(player_died, state.lives - 1, state.lives)
            
            # Keep position; set health to 0 if dead; score unchanged here
            final_x = transition_x
            final_y = transition_y
            final_health_after = jnp.where(player_died, jnp.array(0, dtype=jnp.int32), final_health)
            death_counter_after = jnp.where(player_died & (new_lives > 0), jnp.array(DEATH_FREEZE_TICKS, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
            
            new_state = DarkChambersState(
                player_x=final_x,
                player_y=final_y,
                player_direction=new_direction,
                enemy_positions=final_enemy_positions_after_level,
                enemy_types=final_enemy_types_after_level,
                enemy_active=final_enemy_active_after_level,
                spawner_positions=use_sp_positions,
                spawner_health=use_sp_health,
                spawner_active=use_sp_active,
                spawner_timers=use_sp_timers,
                bullet_positions=final_bullet_positions,
                bullet_active=final_bullet_active2,
                enemy_bullet_positions=final_enemy_bullet_positions,
                enemy_bullet_active=final_enemy_bullet_active3,
                health=final_health_after,
                score=final_score,
                item_positions=transition_item_positions,
                item_types=transition_item_types,
                item_active=transition_item_active,
                has_key=jnp.where(level_changed, 0, new_has_key),
                shield_active=new_shield_active,
                gun_active=new_gun_active,
                bomb_count=bomb_count_after_detonation,
                last_fire_step=new_last_fire_step,
                current_level=new_level,
                ladder_timer=new_ladder_timer,
                lives=new_lives,
                step_counter=state.step_counter + 1,
                death_counter=death_counter_after,
                key=rng,
            )
            
            # During an active freeze (from previous step), override updates and freeze or respawn
            def do_freeze(_: None):
                def dec_only(_: None):
                    return state._replace(
                        death_counter=state.death_counter - 1,
                        step_counter=state.step_counter + 1,
                    )
                def respawn_now(_: None):
                    respawn_x2 = jnp.where(state.current_level == 0, self.consts.PLAYER_START_X, jnp.array(40, dtype=jnp.int32))
                    respawn_y2 = jnp.where(state.current_level == 0, self.consts.PLAYER_START_Y, jnp.array(70, dtype=jnp.int32))
                    # KEEP EVERYTHING: score, items, level, powerups - just restore health and clear bullets
                    return state._replace(
                        player_x=respawn_x2,
                        player_y=respawn_y2,
                        health=jnp.array(self.consts.STARTING_HEALTH, dtype=jnp.int32),
                        bullet_positions=jnp.zeros((MAX_BULLETS, 4), dtype=jnp.int32),
                        bullet_active=jnp.zeros((MAX_BULLETS,), dtype=jnp.int32),
                        enemy_bullet_positions=jnp.zeros((ENEMY_MAX_BULLETS, 4), dtype=jnp.int32),
                        enemy_bullet_active=jnp.zeros((ENEMY_MAX_BULLETS,), dtype=jnp.int32),
                        death_counter=jnp.array(0, dtype=jnp.int32),
                        step_counter=state.step_counter + 1,
                    )
                return jax.lax.cond(state.death_counter > 1, dec_only, respawn_now, operand=None)
            
            return_state = jax.lax.cond(state.death_counter > 0, do_freeze, lambda _: new_state, operand=None)
            
            obs = self._get_observation(return_state)
            reward = self._get_reward(state, return_state)
            done = self._get_done(return_state)
            info = self._get_info(return_state)
            
            return obs, return_state, reward, done, info

        # Top-level choose freeze vs normal
        return jax.lax.cond(state.death_counter > 0, handle_freeze, handle_normal, operand=None)
    
    def render(self, state: DarkChambersState) -> jnp.ndarray:
        return self.renderer.render(state)
    
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)
    
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WORLD_WIDTH - 1, shape=(), dtype=jnp.float64),
                "y": spaces.Box(low=0, high=self.consts.WORLD_HEIGHT - 1, shape=(), dtype=jnp.float64),
                "width": spaces.Box(low=0, high=self.consts.WORLD_WIDTH, shape=(), dtype=jnp.float64),
                "height": spaces.Box(low=0, high=self.consts.WORLD_HEIGHT, shape=(), dtype=jnp.float64),
            }),
            "enemies": spaces.Box(
                low=0,
                high=float(max(self.consts.WORLD_WIDTH, self.consts.WORLD_HEIGHT)),
                shape=(NUM_ENEMIES, 5),
                dtype=jnp.float64,
            ),
            "health": spaces.Box(low=0, high=float(self.consts.MAX_HEALTH), shape=(), dtype=jnp.float64),
            "score": spaces.Box(low=0, high=1e9, shape=(), dtype=jnp.float64),
            "step": spaces.Box(low=0, high=1e9, shape=(), dtype=jnp.float64),
        })
    
    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0, 
            high=255, 
            shape=(GAME_H, GAME_W, 3), 
            dtype=jnp.uint8
        )
    
    def _get_observation(self, state: DarkChambersState) -> DarkChambersObservation:
        player = EntityPosition(
        x=jnp.asarray(state.player_x, dtype=jnp.float32),
        y=jnp.asarray(state.player_y, dtype=jnp.float32),
        width=jnp.asarray(self.consts.PLAYER_WIDTH, dtype=jnp.float32),
        height=jnp.asarray(self.consts.PLAYER_HEIGHT, dtype=jnp.float32),
    )

        enemy_widths  = jnp.full(NUM_ENEMIES, self.consts.ENEMY_WIDTH,  dtype=jnp.float32)
        enemy_heights = jnp.full(NUM_ENEMIES, self.consts.ENEMY_HEIGHT, dtype=jnp.float32)

        enemies_array = jnp.stack([
            state.enemy_positions[:, 0].astype(jnp.float32),
            state.enemy_positions[:, 1].astype(jnp.float32),
            enemy_widths,
            enemy_heights,
            state.enemy_active.astype(jnp.float32),
        ], axis=1)

        
        return DarkChambersObservation(
            player=player,
            enemies=enemies_array,
            health=state.health,
            score=state.score,
            step=state.step_counter
        )
    
    def _get_info(self, state: DarkChambersState, all_rewards: jnp.array = None) -> DarkChambersInfo:
        return DarkChambersInfo(time=state.step_counter)
    

    """
    TODO reward for RE Agent later
    """
    def _get_reward(self, previous_state: DarkChambersState, state: DarkChambersState) -> float:
        return 0.1
    


    """
    Game is over when player has no lives left and health is 0
    """
    def _get_done(self, state: DarkChambersState) -> bool:
        return (state.lives <= 0) & (state.health <= 0)
    

    def obs_to_flat_array(self, obs: DarkChambersObservation) -> jnp.ndarray:
        player_data = jnp.array([obs.player.x, obs.player.y, obs.player.width, obs.player.height], dtype=jnp.float64)
        enemies_flat = obs.enemies.flatten()
        health_data = jnp.array([obs.health], dtype=jnp.float64)
        score_data = jnp.array([obs.score], dtype=jnp.float64)
        step_data = jnp.array([obs.step], dtype=jnp.float64)
        
        return jnp.concatenate([player_data, enemies_flat, health_data, score_data, step_data])