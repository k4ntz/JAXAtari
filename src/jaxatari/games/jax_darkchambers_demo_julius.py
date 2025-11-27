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

# Item configuration
NUM_ITEMS = 8
ITEM_HEART = 1
ITEM_POISON = 2
ITEM_WIDTH = 6
ITEM_HEIGHT = 6


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
    ENEMY_COLOR: Tuple[int, int, int] = (80, 200, 120)
    WALL_COLOR: Tuple[int, int, int] = (150, 120, 70)
    HEART_COLOR: Tuple[int, int, int] = (220, 30, 30)
    POISON_COLOR: Tuple[int, int, int] = (50, 200, 50)
    UI_COLOR: Tuple[int, int, int] = (236, 236, 236)
    
    # Sizes
    PLAYER_WIDTH: int = 12
    PLAYER_HEIGHT: int = 12
    ENEMY_WIDTH: int = 10
    ENEMY_HEIGHT: int = 10
    
    PLAYER_SPEED: int = 2
    WALL_THICKNESS: int = 8
    
    PLAYER_START_X: int = 24
    PLAYER_START_Y: int = 24
    
    # Health mechanics
    MAX_HEALTH: int = 10
    STARTING_HEALTH: int = 5
    HEALTH_GAIN: int = 1
    POISON_DAMAGE: int = 1


class DarkChambersState(NamedTuple):
    """Game state."""
    player_x: chex.Array
    player_y: chex.Array
    
    enemy_positions: chex.Array  # shape: (NUM_ENEMIES, 2)
    
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
            self.consts.ENEMY_COLOR,        # 2
            self.consts.WALL_COLOR,         # 3
            self.consts.HEART_COLOR,        # 4
            self.consts.POISON_COLOR,       # 5
            self.consts.UI_COLOR,           # 6
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
            color_id=3
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
        
        # Enemies
        enemy_world_pos = state.enemy_positions.astype(jnp.int32)
        enemy_screen_pos = (enemy_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        num_enemies = enemy_screen_pos.shape[0]
        enemy_sizes = jnp.tile(
            jnp.array([self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT], dtype=jnp.int32)[None, :],
            (num_enemies, 1)
        )
        object_raster = self.jr.draw_rects(
            object_raster, 
            positions=enemy_screen_pos, 
            sizes=enemy_sizes, 
            color_id=2
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
        
        # Hearts
        heart_mask = (state.item_types == ITEM_HEART) & (state.item_active == 1)
        heart_pos = jnp.where(
            heart_mask[:, None],
            masked_item_pos,
            off_screen
        )
        heart_sizes = jnp.tile(
            jnp.array([ITEM_WIDTH, ITEM_HEIGHT], dtype=jnp.int32)[None, :],
            (NUM_ITEMS, 1)
        )
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=heart_pos,
            sizes=heart_sizes,
            color_id=4
        )
        
        # Poison
        poison_mask = (state.item_types == ITEM_POISON) & (state.item_active == 1)
        poison_pos = jnp.where(
            poison_mask[:, None],
            masked_item_pos,
            off_screen
        )
        poison_sizes = jnp.tile(
            jnp.array([ITEM_WIDTH, ITEM_HEIGHT], dtype=jnp.int32)[None, :],
            (NUM_ITEMS, 1)
        )
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=poison_pos,
            sizes=poison_sizes,
            color_id=5
        )
        
        # Health bar at top-right
        heart_spacing = ITEM_WIDTH + 2
        health_count = jnp.clip(state.health, 0, self.consts.MAX_HEALTH).astype(jnp.int32)
        max_hearts = self.consts.MAX_HEALTH
        start_x = self.config.game_dimensions[1] - (max_hearts * heart_spacing) - 4
        health_indices = jnp.arange(max_hearts)
        health_x = health_indices * heart_spacing + start_x
        health_y = jnp.full(max_hearts, 4, dtype=jnp.int32)
        health_visible = health_indices < health_count
        health_pos_x = jnp.where(health_visible, health_x, -100)
        health_positions = jnp.stack([health_pos_x, health_y], axis=1).astype(jnp.int32)
        health_sizes = jnp.tile(
            jnp.array([ITEM_WIDTH, ITEM_HEIGHT], dtype=jnp.int32)[None, :],
            (max_hearts, 1)
        )
        object_raster = self.jr.draw_rects(
            object_raster,
            positions=health_positions,
            sizes=health_sizes,
            color_id=4
        )
        
        # Score display at top-left (static value for now)
        max_score_units = 20
        score_units = jnp.clip(state.score // 10, 0, max_score_units).astype(jnp.int32)
        score_spacing = ITEM_WIDTH + 2
        score_indices = jnp.arange(max_score_units)
        score_x = score_indices * score_spacing + 4
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
            color_id=6
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
        
        # 50/50 split between hearts and poison
        key, subkey = jax.random.split(key)
        item_types = jax.random.choice(
            subkey, jnp.array([ITEM_HEART, ITEM_POISON]), shape=(NUM_ITEMS,)
        )
        item_active = jnp.ones(NUM_ITEMS, dtype=jnp.int32)
        
        state = DarkChambersState(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            enemy_positions=enemy_positions,
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
        
        # Enemy random walk
        rng, subkey = jax.random.split(state.key)
        enemy_deltas = jax.random.randint(subkey, (NUM_ENEMIES, 2), -1, 2, dtype=jnp.int32)
        
        prop_enemy_positions = state.enemy_positions + enemy_deltas
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
        health_change = (collected_hearts * self.consts.HEALTH_GAIN) - (collected_poison * self.consts.POISON_DAMAGE)
        new_health = jnp.clip(state.health + health_change, 0, self.consts.MAX_HEALTH)
        
        new_score = state.score + jnp.sum(item_collisions) * 10
        
        # Remove collected items
        new_item_active = jnp.where(item_collisions, 0, state.item_active)
        
        new_state = DarkChambersState(
            player_x=new_x,
            player_y=new_y,
            enemy_positions=new_enemy_positions,
            health=new_health,
            score=new_score,
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
            "health": spaces.Box(low=0, high=self.consts.MAX_HEALTH, shape=(), dtype=jnp.int32),
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
        enemy_active = jnp.ones(NUM_ENEMIES, dtype=jnp.float32)
        
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
