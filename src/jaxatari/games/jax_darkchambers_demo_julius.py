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


# Game configuration and constants

GAME_H = 210
GAME_W = 160
WORLD_W = GAME_W * 2  # World is 2x viewport size
WORLD_H = GAME_H * 2

NUM_ENEMIES = 5

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

# Default base size (unused now, kept for reference)
ITEM_WIDTH = 6
ITEM_HEIGHT = 6

# Bullet configuration
MAX_BULLETS = 5
BULLET_WIDTH = 4
BULLET_HEIGHT = 4
BULLET_SPEED = 4


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
    
    bullet_positions: chex.Array  # (MAX_BULLETS, 4) - x, y, dx, dy
    bullet_active: chex.Array     # (MAX_BULLETS,) - 1=active, 0=inactive
    
    health: chex.Array
    score: chex.Array
    
    item_positions: chex.Array  # (NUM_ITEMS, 2)
    item_types: chex.Array      # 1=heart, 2=poison
    item_active: chex.Array     # 1=active, 0=collected
    
    step_counter: chex.Array
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
        config = render_utils.RendererConfig(
            game_dimensions=(GAME_H, GAME_W),
            channels=3
        )
        super().__init__(consts=consts, config=config)
        self.consts = consts or DarkChambersConstants()
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Color palette
        self.PALETTE = jnp.array([
            self.consts.BACKGROUND_COLOR,  # 0
            self.consts.PLAYER_COLOR,       # 1
            self.consts.ZOMBIE_COLOR,       # 2
            self.consts.WRAITH_COLOR,       # 3
            self.consts.SKELETON_COLOR,     # 4
            self.consts.WIZARD_COLOR,       # 5
            self.consts.GRIM_REAPER_COLOR,  # 6
            self.consts.WALL_COLOR,         # 7
            self.consts.HEART_COLOR,        # 8
            self.consts.POISON_COLOR,       # 9
            self.consts.UI_COLOR,           # 10
            self.consts.BULLET_COLOR,       # 11
            self.consts.TREASURE_COLOR,     # 12 (treasures)
            self.consts.TRAP_COLOR,         # 13 (trap)
        ], dtype=jnp.uint8)

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
        self.WALLS = jnp.array([
            # Border
            [0, 0, self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS],
            [0, self.consts.WORLD_HEIGHT - self.consts.WALL_THICKNESS, 
             self.consts.WORLD_WIDTH, self.consts.WALL_THICKNESS],
            [0, 0, self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT],
            [self.consts.WORLD_WIDTH - self.consts.WALL_THICKNESS, 0, 
             self.consts.WALL_THICKNESS, self.consts.WORLD_HEIGHT],
            # Labyrinth structure
            [60, 80, 120, 8],
            [200, 150, 80, 8],
            [80, 200, 100, 8],
            [40, 280, 140, 8],
            [220, 50, 8, 140],
            [120, 180, 8, 100],
            [280, 120, 8, 160],
            [160, 300, 8, 80],
        ], dtype=jnp.int32)
        
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
        ], dtype=jnp.int32)
        # Python constants (avoid tracing int() on JAX 0-D arrays inside jit)
        self.ITEM_TYPE_COLOR_IDS_PY = [8, 9, 13, 12, 12, 12, 12]
    
    def render(self, state: DarkChambersState) -> jnp.ndarray:
        """Render current game state."""
        object_raster = jnp.full(
            (self.config.game_dimensions[0], self.config.game_dimensions[1]), 
            0,
            dtype=jnp.uint8
        )
        
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
        
        # Draw walls
        wall_positions = (self.WALLS[:, 0:2] - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        wall_sizes = self.WALLS[:, 2:4]
        object_raster = self.jr.draw_rects(
            object_raster, 
            positions=wall_positions, 
            sizes=wall_sizes, 
            color_id=7
        )
        
        # Player
        player_screen_x = (state.player_x - cam_x).astype(jnp.int32)
        player_screen_y = (state.player_y - cam_y).astype(jnp.int32)
        player_pos = jnp.array([[player_screen_x, player_screen_y]], dtype=jnp.int32)
        player_size = jnp.array([[self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT]], dtype=jnp.int32)
        object_raster = self.jr.draw_rects(
            object_raster, 
            positions=player_pos, 
            sizes=player_size, 
            color_id=1
        )
        
        # Enemies (render by type with different colors, mask out dead ones)
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
        for enemy_type in range(1, 6):  # 1=zombie to 5=grim_reaper
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
                color_id=enemy_type + 1  # +1 because palette starts at 0, zombie=2, wraith=3, etc.
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
        
        # Draw items by type using size and color mappings
        def draw_item_type(raster, t, positions_world):
            mask = (state.item_types == t) & (state.item_active == 1)
            pos = jnp.where(mask[:, None], positions_world, off_screen)
            size_wh = self.ITEM_TYPE_SIZES[t]
            sizes = jnp.tile(size_wh[None, :], (NUM_ITEMS, 1))
            color_id_py = self.ITEM_TYPE_COLOR_IDS_PY[t-1]  # t in 1..7
            return self.jr.draw_rects(
                raster,
                positions=pos,
                sizes=sizes,
                color_id=color_id_py
            )
        for t in range(1, 8):  # 1..7 item types
            object_raster = draw_item_type(object_raster, t, masked_item_pos)
        
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
            color_id=8
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
            color_id=8
        )
        
        # Score display moved to top-right (bar) and numeric score below
        max_score_units = 10  # Reduced to fit on right side
        score_units = jnp.clip(state.score // 10, 0, max_score_units).astype(jnp.int32)
        score_spacing = ITEM_WIDTH + 2
        score_indices = jnp.arange(max_score_units)
        total_score_width = max_score_units * score_spacing
        score_start_x = self.config.game_dimensions[1] - total_score_width - 4
        score_x = score_start_x + score_indices * score_spacing
        score_y = jnp.full(max_score_units, 4, dtype=jnp.int32)
        score_visible = score_indices < score_units
        score_pos_x = jnp.where(score_visible, score_x, -100)
        score_positions = jnp.stack([score_pos_x, score_y], axis=1).astype(jnp.int32)
        score_sizes = jnp.tile(
            jnp.array([ITEM_WIDTH, ITEM_HEIGHT], dtype=jnp.int32)[None, :],
            (max_score_units, 1)
        )
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=score_positions,
            sizes=score_sizes,
            color_id=10
        )
        # Numeric score digits (0-9999) below score bar
        score_val = jnp.clip(state.score, 0, 9999).astype(jnp.int32)
        digit_width = 3
        digit_height = 5
        spacing = 1
        thousands = score_val // 1000
        hundreds = (score_val // 100) % 10
        tens = (score_val // 10) % 10
        ones = score_val % 10
        digits = jnp.array([thousands, hundreds, tens, ones], dtype=jnp.int32)
        active_mask = jnp.array([
            thousands > 0,
            (thousands > 0) | (hundreds > 0),
            (thousands > 0) | (hundreds > 0) | (tens > 0),
            True
        ])
        position_index = (jnp.cumsum(active_mask.astype(jnp.int32)) - 1)
        base_x = jnp.where(active_mask, score_start_x + position_index * (digit_width + spacing), -100)
        patterns = self.DIGIT_PATTERNS[digits]
        xs = jnp.arange(digit_width)
        ys = jnp.arange(digit_height)
        grid_x = xs[None, None, :].repeat(4, axis=0).repeat(digit_height, axis=1)
        grid_y = ys[None, :, None].repeat(4, axis=0).repeat(digit_width, axis=2)
        px = base_x[:, None, None] + grid_x
        py = (4 + ITEM_HEIGHT + 2) + grid_y  # below score bar
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
            color_id=10
        )
        
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
            color_id=11
        )
        
        # Convert to RGB
        img = self.jr.render_from_palette(object_raster, self.PALETTE)
        return img


class DarkChambersEnv(JaxEnvironment[DarkChambersState, DarkChambersObservation, DarkChambersInfo, DarkChambersConstants]):
    """Main environment."""
    
    def __init__(self, consts: DarkChambersConstants = None):
        super().__init__(consts=consts or DarkChambersConstants())
        self.renderer = DarkChambersRenderer(self.consts)
    
    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[DarkChambersObservation, DarkChambersState]:
        """Reset game."""
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Spawn enemies
        key, subkey = jax.random.split(key)
        enemy_x_positions = jax.random.randint(
            subkey, (NUM_ENEMIES,), 30, self.consts.WORLD_WIDTH - 30, dtype=jnp.int32
        )
        key, subkey = jax.random.split(key)
        enemy_y_positions = jax.random.randint(
            subkey, (NUM_ENEMIES,), 30, self.consts.WORLD_HEIGHT - 30, dtype=jnp.int32
        )
        enemy_positions = jnp.stack([enemy_x_positions, enemy_y_positions], axis=1)
        
        # Spawn enemies with random types (favor stronger types)
        key, subkey = jax.random.split(key)
        enemy_types = jax.random.randint(
            subkey, (NUM_ENEMIES,), ENEMY_WRAITH, ENEMY_GRIM_REAPER + 1, dtype=jnp.int32
        )  # Random types from 2 (Wraith) to 5 (Grim Reaper)
        enemy_active = jnp.ones(NUM_ENEMIES, dtype=jnp.int32)
        
        # Spawn items
        key, subkey = jax.random.split(key)
        item_x_positions = jax.random.randint(
            subkey, (NUM_ITEMS,), 30, self.consts.WORLD_WIDTH - 30, dtype=jnp.int32
        )
        key, subkey = jax.random.split(key)
        item_y_positions = jax.random.randint(
            subkey, (NUM_ITEMS,), 30, self.consts.WORLD_HEIGHT - 30, dtype=jnp.int32
        )
        item_positions = jnp.stack([item_x_positions, item_y_positions], axis=1)
        
        # Weighted distribution of item types (more traps)
        key, subkey = jax.random.split(key)
        all_item_types = jnp.array([
            ITEM_HEART,          # health +10
            ITEM_POISON,         # -4 health
            ITEM_TRAP,           # -6 health (increase frequency)
            ITEM_STRONGBOX,      # 100 pts
            ITEM_SILVER_CHALICE, # 500 pts
            ITEM_AMULET,         # 1000 pts
            ITEM_GOLD_CHALICE,   # 3000 pts
        ], dtype=jnp.int32)
        # Probabilities sum to 1.0; traps boosted
        spawn_probs = jnp.array([
            0.22,  # heart
            0.12,  # poison
            0.20,  # trap (increased)
            0.14,  # strongbox
            0.12,  # silver chalice
            0.10,  # amulet
            0.10,  # gold chalice
        ], dtype=jnp.float32)
        item_types = jax.random.choice(subkey, all_item_types, shape=(NUM_ITEMS,), p=spawn_probs)
        item_active = jnp.ones(NUM_ITEMS, dtype=jnp.int32)
        
        state = DarkChambersState(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            player_direction=jnp.array(0, dtype=jnp.int32),
            enemy_positions=enemy_positions,
            enemy_types=enemy_types,
            enemy_active=enemy_active,
            bullet_positions=jnp.zeros((MAX_BULLETS, 4), dtype=jnp.int32),
            bullet_active=jnp.zeros(MAX_BULLETS, dtype=jnp.int32),
            health=jnp.array(self.consts.STARTING_HEALTH, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            item_positions=item_positions,
            item_types=item_types,
            item_active=item_active,
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=key,
        )
        
        obs = self._get_observation(state)
        return obs, state
    
    def step(self, state: DarkChambersState, action: int) -> Tuple[DarkChambersObservation, DarkChambersState, float, bool, DarkChambersInfo]:
        """Step forward."""
        a = jnp.asarray(action)
        
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
        
        # Wall collision check
        WALLS = self.renderer.WALLS
        
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
        
        # Shooting - spawn bullet on FIRE action
        fire_pressed = (a == Action.FIRE) | (a == Action.UPFIRE) | (a == Action.DOWNFIRE) | (a == Action.LEFTFIRE) | (a == Action.RIGHTFIRE) | \
                       (a == Action.UPRIGHTFIRE) | (a == Action.UPLEFTFIRE) | (a == Action.DOWNRIGHTFIRE) | (a == Action.DOWNLEFTFIRE)
        
        # Find first inactive bullet slot
        first_inactive = jnp.argmax(state.bullet_active == 0)
        can_spawn = jnp.any(state.bullet_active == 0)
        
        # Direction vectors based on fire action (not player direction)
        # Cardinal directions
        dir_x = jnp.where(a == Action.RIGHTFIRE, BULLET_SPEED,
                jnp.where(a == Action.LEFTFIRE, -BULLET_SPEED,
                jnp.where(a == Action.FIRE, jnp.where(new_direction == 0, BULLET_SPEED,
                                            jnp.where(new_direction == 1, -BULLET_SPEED, 0)),
                0)))
        dir_y = jnp.where(a == Action.UPFIRE, -BULLET_SPEED,
                jnp.where(a == Action.DOWNFIRE, BULLET_SPEED,
                jnp.where(a == Action.FIRE, jnp.where(new_direction == 2, -BULLET_SPEED,
                                            jnp.where(new_direction == 3, BULLET_SPEED, 0)),
                0)))
        
        # Diagonal directions
        dir_x = jnp.where(a == Action.UPRIGHTFIRE, BULLET_SPEED,
                jnp.where(a == Action.DOWNRIGHTFIRE, BULLET_SPEED,
                jnp.where(a == Action.UPLEFTFIRE, -BULLET_SPEED,
                jnp.where(a == Action.DOWNLEFTFIRE, -BULLET_SPEED, dir_x))))
        dir_y = jnp.where(a == Action.UPRIGHTFIRE, -BULLET_SPEED,
                jnp.where(a == Action.UPLEFTFIRE, -BULLET_SPEED,
                jnp.where(a == Action.DOWNRIGHTFIRE, BULLET_SPEED,
                jnp.where(a == Action.DOWNLEFTFIRE, BULLET_SPEED, dir_y))))
        
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
        
        updated_bullet_positions = jnp.where(
            state.bullet_active[:, None] == 1,
            moved_bullets,
            state.bullet_positions
        )
        updated_bullet_active = state.bullet_active & in_bounds.astype(jnp.int32)
        
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
        
        # Enemy random walk (dead enemies stay put and never move again)
        rng, subkey = jax.random.split(state.key)
        enemy_deltas = jax.random.randint(subkey, (NUM_ENEMIES, 2), -1, 2, dtype=jnp.int32)
        enemy_alive = state.enemy_active == 1
        
        prop_enemy_positions = state.enemy_positions + enemy_deltas
        # Keep dead enemies exactly where they are (typically [0,0])
        prop_enemy_positions = jnp.where(
            enemy_alive[:, None],
            prop_enemy_positions,
            state.enemy_positions
        )
        prop_enemy_positions = jnp.clip(
            prop_enemy_positions,
            jnp.array([0, 0]),
            jnp.array([self.consts.WORLD_WIDTH - 1, self.consts.WORLD_HEIGHT - 1])
        )
        
        def check_enemy_collision(enemy_pos):
            ex, ey = enemy_pos[0], enemy_pos[1]
            e_overlap_x = (ex <= (WALLS[:,0] + WALLS[:,2] - 1)) & ((ex + self.consts.ENEMY_WIDTH - 1) >= WALLS[:,0])
            e_overlap_y = (ey <= (WALLS[:,1] + WALLS[:,3] - 1)) & ((ey + self.consts.ENEMY_HEIGHT - 1) >= WALLS[:,1])
            return jnp.any(e_overlap_x & e_overlap_y)
        
        enemy_collisions = jax.vmap(check_enemy_collision)(prop_enemy_positions)
        # Dead enemies cannot collide with walls or move
        enemy_collisions = enemy_collisions & enemy_alive
        new_enemy_positions = jnp.where(
            enemy_collisions[:, None],
            state.enemy_positions,
            prop_enemy_positions
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
        # Health change mapping: heart +HEALTH_GAIN, poison -POISON_DAMAGE, trap -TRAP_DAMAGE
        health_change = (
            collected_hearts * self.consts.HEALTH_GAIN
            - collected_poison * self.consts.POISON_DAMAGE
            - collected_traps * self.consts.TRAP_DAMAGE
        )
        new_health = jnp.clip(state.health + health_change, 0, self.consts.MAX_HEALTH)
        
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
        ], dtype=jnp.int32)
        item_points = points_by_type[state.item_types]
        gained_points = jnp.sum(item_points * item_collisions.astype(jnp.int32))
        new_score = state.score + gained_points
        
        # Remove collected items
        new_item_active = jnp.where(item_collisions, 0, state.item_active)
        
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
        
        # Enemy mutation system: when hit, enemy mutates to weaker form
        # Grim Reaper (5) -> Wizard (4) -> Skeleton (3) -> Wraith (2) -> Zombie (1) -> Dead (0)
        new_enemy_types = jnp.where(enemy_hit, state.enemy_types - 1, state.enemy_types)
        
        # Award points when enemy is killed (mutates from zombie to dead)
        zombies_killed = enemy_hit & (state.enemy_types == ENEMY_ZOMBIE)
        wraiths_hit = enemy_hit & (state.enemy_types == ENEMY_WRAITH)
        skeletons_hit = enemy_hit & (state.enemy_types == ENEMY_SKELETON)
        wizards_hit = enemy_hit & (state.enemy_types == ENEMY_WIZARD)
        grim_reapers_hit = enemy_hit & (state.enemy_types == ENEMY_GRIM_REAPER)
        
        enemy_kill_score = (
            jnp.sum(zombies_killed) * self.consts.ZOMBIE_POINTS +
            jnp.sum(wraiths_hit) * self.consts.WRAITH_POINTS +
            jnp.sum(skeletons_hit) * self.consts.SKELETON_POINTS +
            jnp.sum(wizards_hit) * self.consts.WIZARD_POINTS +
            jnp.sum(grim_reapers_hit) * self.consts.GRIM_REAPER_POINTS
        )
        
        # Deactivate enemies that have been killed (type becomes 0)
        new_enemy_active = jnp.where(new_enemy_types <= 0, 0, state.enemy_active)
        
        # Move dead enemies off-screen
        final_enemy_positions = jnp.where(
            new_enemy_active[:, None] == 1,
            new_enemy_positions,
            jnp.array([0, 0])
        )
        
        # Deactivate bullets that hit
        bullet_hit_any = jnp.any(all_collisions, axis=1)
        final_bullet_active = final_bullet_active & (~bullet_hit_any).astype(jnp.int32)
        
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
        final_health = jnp.clip(new_health - contact_damage, 0, self.consts.MAX_HEALTH)
        
        new_state = DarkChambersState(
            player_x=new_x,
            player_y=new_y,
            player_direction=new_direction,
            enemy_positions=final_enemy_positions,
            enemy_types=new_enemy_types,
            enemy_active=new_enemy_active,
            bullet_positions=final_bullet_positions,
            bullet_active=final_bullet_active,
            health=final_health,
            score=final_score,
            item_positions=state.item_positions,
            item_types=state.item_types,
            item_active=new_item_active,
            step_counter=state.step_counter + 1,
            key=rng,
        )
        
        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = False
        info = self._get_info(new_state)
        
        return obs, new_state, reward, done, info
    
    def render(self, state: DarkChambersState) -> jnp.ndarray:
        return self.renderer.render(state)
    
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)
    
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player_x": spaces.Box(low=0, high=self.consts.WORLD_WIDTH - 1, shape=(), dtype=jnp.int32),
            "player_y": spaces.Box(low=0, high=self.consts.WORLD_HEIGHT - 1, shape=(), dtype=jnp.int32),
            "enemies": spaces.Box(
                low=0, 
                high=max(self.consts.WORLD_WIDTH, self.consts.WORLD_HEIGHT), 
                shape=(NUM_ENEMIES, 5),
                dtype=jnp.float32
            ),
            "health": spaces.Box(low=0, high=1000, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=10**9, shape=(), dtype=jnp.int32),
            "step": spaces.Box(low=0, high=10**9, shape=(), dtype=jnp.int32),
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
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_WIDTH),
            height=jnp.array(self.consts.PLAYER_HEIGHT)
        )
        
        # Pack enemy data
        enemy_widths = jnp.full(NUM_ENEMIES, self.consts.ENEMY_WIDTH, dtype=jnp.float32)
        enemy_heights = jnp.full(NUM_ENEMIES, self.consts.ENEMY_HEIGHT, dtype=jnp.float32)
        enemy_active = state.enemy_active.astype(jnp.float32)
        
        enemies_array = jnp.stack([
            state.enemy_positions[:, 0].astype(jnp.float32),
            state.enemy_positions[:, 1].astype(jnp.float32),
            enemy_widths,
            enemy_heights,
            enemy_active
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
    
    def _get_reward(self, previous_state: DarkChambersState, state: DarkChambersState) -> float:
        return 0.1
    
    def obs_to_flat_array(self, obs: DarkChambersObservation) -> jnp.ndarray:
        player_data = jnp.array([obs.player.x, obs.player.y], dtype=jnp.float32)
        enemies_flat = obs.enemies.flatten()
        health_data = jnp.array([obs.health], dtype=jnp.float32)
        score_data = jnp.array([obs.score], dtype=jnp.float32)
        step_data = jnp.array([obs.step], dtype=jnp.float32)
        
        return jnp.concatenate([player_data, enemies_flat, health_data, score_data, step_data])
