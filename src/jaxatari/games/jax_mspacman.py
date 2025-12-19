from functools import partial
from typing import NamedTuple, Tuple
from enum import IntEnum
import os

import chex
import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.games.mspacman_mazes import MsPacmanMaze


# -------- Enums --------
class GhostMode(IntEnum):
    RANDOM = 0
    CHASE = 1
    SCATTER = 2
    FRIGHTENED = 3
    BLINKING = 4
    RETURNING = 5
    ENJAILED = 6


class FruitType(IntEnum):
    CHERRY = 0
    STRAWBERRY = 1
    ORANGE = 2
    PRETZEL = 3
    APPLE = 4
    PEAR = 5
    BANANA = 6
    NONE = 7


# -------- Constants --------
# General
RESET_TIMER = 40  # Frames to freeze after death
PELLETS_TO_COLLECT = 154  # Total pellets in maze (including power pellets counted separately)
COLLISION_THRESHOLD = 8  # Pixels for collision detection
INITIAL_LIVES = 3

# Ghost timing (in steps)
CHASE_DURATION = 640       # ~20 seconds
SCATTER_DURATION = 224     # ~7 seconds
FRIGHTENED_DURATION = 496  # Power pellet effect duration
BLINKING_DURATION = 80     # Warning before frightened ends
ENJAILED_DURATION = 120    # Time in jail before return
RETURN_DURATION = 16       # Time to return from jail

# Power pellet positions (pixel coordinates - corners of maze)
POWER_PELLET_POSITIONS = jnp.array([[8, 20], [148, 20], [8, 152], [148, 152]], dtype=jnp.int32)

# Scatter targets (corners for each ghost type)
SCATTER_TARGETS = jnp.array([
    [156, 0],    # Blinky - top right
    [0, 0],      # Pinky - top left
    [156, 172],  # Inky - bottom right
    [0, 172],    # Sue - bottom left
], dtype=jnp.int32)

# Ghost spawn/jail position
JAIL_POSITION = jnp.array([80, 78], dtype=jnp.int32)

# Fruit
FRUIT_SPAWN_THRESHOLDS = jnp.array([50, 100], dtype=jnp.int32)
FRUIT_WANDER_DURATION = 160  # Steps before fruit exits
FRUIT_REWARDS = jnp.array([100, 200, 500, 700, 1000, 2000, 5000], dtype=jnp.int32)

# Scoring
PELLET_POINTS = 10
POWER_PELLET_POINTS = 50
GHOST_BASE_POINTS = 200  # Doubles for each ghost eaten
LEVEL_COMPLETE_BONUS = 500

# Direction mappings for ghost movement
DIRECTION_VECTORS = jnp.array([
    [0, 0],   # NOOP
    [0, 0],   # FIRE
    [0, -1],  # UP
    [1, 0],   # RIGHT
    [-1, 0],  # LEFT
    [0, 1],   # DOWN
], dtype=jnp.int32)


class EntityPosition(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array


class MsPacmanObservation(NamedTuple):
    pacman: EntityPosition
    ghosts: chex.Array
    pellets: chex.Array
    power_timer: chex.Array
    lives: chex.Array


class MsPacmanInfo(NamedTuple):
    time: chex.Array
    pellets_remaining: chex.Array
    lives: chex.Array


class MsPacmanConstants(NamedTuple):
    # Screen dimensions (pixel-based)
    screen_width: int = 160
    screen_height: int = 210
    
    # Maze configuration
    maze_id: int = 0  # Which maze to use (0-3)
    tile_scale: int = 4  # Pixels per tile
    maze_width: int = 160  # Maze pixel width (40 tiles * 4)
    maze_offset_x: int = 0  # No offset (maze fills screen)
    
    # Game rules
    num_ghosts: int = 4
    pellet_reward: int = 10
    power_pellet_reward: int = 50
    ghost_reward: int = 200
    collision_penalty: int = 0
    frightened_duration: int = 180
    initial_lives: int = 3
    max_steps: int = 5000
    ghost_spawn_delay: int = 60
    
    # Movement speeds (pixels per frame when moving)
    player_speed: int = 2
    ghost_speed: int = 1
    player_move_period: int = 1  # Move every N frames
    ghost_move_period: int = 2   # Ghosts move every N frames
    
    # Sprite dimensions
    pacman_size: int = 10
    ghost_size: int = 10
    
    # Colors (from MsPacmanMaze)
    background_color: Tuple[int, int, int] = (0, 28, 136)  # PATH_COLOR
    wall_color: Tuple[int, int, int] = (228, 111, 111)     # WALL_COLOR
    pellet_color: Tuple[int, int, int] = (228, 111, 111)   # Same as wall
    power_pellet_color: Tuple[int, int, int] = (255, 255, 255)
    pacman_color: Tuple[int, int, int] = (255, 255, 0)
    ghost_colors: Tuple[Tuple[int, int, int], ...] = (
        (255, 0, 0),      # Blinky (red)
        (255, 184, 222),  # Pinky (pink)
        (0, 255, 255),    # Inky (cyan)
        (255, 128, 0),    # Sue (orange)
    )
    
    # Spawn positions (pixel coordinates)
    pacman_spawn_x: int = 80   # Center of maze
    pacman_spawn_y: int = 164  # Lower area
    ghost_spawn_x: int = 80    # Center (ghost house)
    ghost_spawn_y: int = 78    # Ghost house area


class MsPacmanState(NamedTuple):
    pacman_x: chex.Array
    pacman_y: chex.Array
    direction: chex.Array
    facing_direction: chex.Array  # Last non-zero direction for sprite rotation
    ghost_positions: chex.Array
    ghost_modes: chex.Array       # Int[4] - Mode for each ghost (GhostMode enum)
    ghost_timers: chex.Array      # Int[4] - Timer for mode transitions
    ghost_directions: chex.Array  # Int[4, 2] - Current direction vector per ghost
    pellets: chex.Array
    power_pellets: chex.Array     # Bool[4] - Power pellet availability
    power_timer: chex.Array       # Frames remaining in frightened mode
    eaten_ghosts: chex.Array      # Ghosts eaten since last power pellet
    level: chex.Array             # Current level (1-indexed)
    collected_pellets: chex.Array # Pellets collected this level
    freeze_timer: chex.Array      # Death animation freeze timer
    fruit_type: chex.Array        # Current fruit type (FruitType.NONE if inactive)
    fruit_position: chex.Array    # Int[2] - Fruit position
    fruit_timer: chex.Array       # Steps until fruit exits
    score: chex.Array
    time: chex.Array
    lives: chex.Array
    pellets_remaining: chex.Array
    game_over: chex.Array


class JaxMsPacman(JaxEnvironment[MsPacmanState, MsPacmanObservation, MsPacmanInfo, MsPacmanConstants]):
    def __init__(self, consts: MsPacmanConstants = None, reward_funcs: list[callable] = None):
        consts = consts or MsPacmanConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        # Load maze data from MsPacmanMaze
        self.maze_id = consts.maze_id
        self.maze_grid = jnp.array(MsPacmanMaze.MAZES[self.maze_id], dtype=jnp.int32)
        self.tile_scale = consts.tile_scale
        
        # Precompute degree of freedom for movement
        self.dof_grid = MsPacmanMaze.precompute_dof(MsPacmanMaze.MAZES[self.maze_id])
        
        # Precompute background image
        self.background_image = MsPacmanMaze.load_background(self.maze_id)
        
        # Pellet grid (18x14 from BASE_PELLETS)
        self.initial_pellets = jnp.array(MsPacmanMaze.BASE_PELLETS, dtype=jnp.int32)
        self.pellet_grid_height, self.pellet_grid_width = self.initial_pellets.shape
        self.initial_pellet_count = jnp.sum(self.initial_pellets > 0)
        
        # Power pellet positions (corners in the pellet grid)
        # Power pellets are at corners: (0,0), (0,17), (17,0), (17,13) in original grid
        # These correspond to screen positions we need to calculate
        
        # Spawn positions (pixel coordinates)
        self.pacman_spawn = jnp.array([consts.pacman_spawn_x, consts.pacman_spawn_y], dtype=jnp.int32)
        
        # Ghost spawn positions - staggered in ghost house
        ghost_spawn_list = []
        for i in range(consts.num_ghosts):
            gx = consts.ghost_spawn_x + (i - consts.num_ghosts // 2) * 12
            gy = consts.ghost_spawn_y
            ghost_spawn_list.append([gx, gy])
        self.ghost_spawn_positions = jnp.array(ghost_spawn_list, dtype=jnp.int32)

        # Action deltas (pixel movement directions)
        self._action_deltas = jnp.array(
            [
                [0, 0],   # NOOP
                [0, 0],   # FIRE treated as NOOP
                [0, -1],  # UP
                [1, 0],   # RIGHT
                [-1, 0],  # LEFT
                [0, 1],   # DOWN
                [1, -1],  # UPRIGHT
                [-1, -1], # UPLEFT
                [1, 1],   # DOWNRIGHT
                [-1, 1],  # DOWNLEFT
                [0, -1],  # UPFIRE
                [1, 0],   # RIGHTFIRE
                [-1, 0],  # LEFTFIRE
                [0, 1],   # DOWNFIRE
                [1, -1],  # UPRIGHTFIRE
                [-1, -1], # UPLEFTFIRE
                [1, 1],   # DOWNRIGHTFIRE
                [-1, 1],  # DOWNLEFTFIRE
            ],
            dtype=jnp.int32,
        )
        
        # Tunnel y-positions for this maze
        self.tunnel_heights = MsPacmanMaze.TUNNEL_HEIGHTS[self.maze_id]

        self.state = self.reset()[1]
        self.renderer = MsPacmanRenderer(
            self.consts,
            self.maze_grid,
            self.initial_pellets,
            self.background_image,
        )

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[MsPacmanObservation, MsPacmanState]:
        # Initialize ghost modes - start in SCATTER mode
        initial_ghost_modes = jnp.array([
            GhostMode.SCATTER, GhostMode.SCATTER, 
            GhostMode.SCATTER, GhostMode.SCATTER
        ], dtype=jnp.int32)
        
        # Initialize ghost timers with SCATTER duration
        initial_ghost_timers = jnp.full((self.consts.num_ghosts,), SCATTER_DURATION, dtype=jnp.int32)
        
        # Initialize ghost directions (all facing left initially)
        initial_ghost_directions = jnp.tile(jnp.array([[-1, 0]], dtype=jnp.int32), (self.consts.num_ghosts, 1))
        
        state = MsPacmanState(
            pacman_x=self.pacman_spawn[0],
            pacman_y=self.pacman_spawn[1],
            direction=jnp.array([0, 0], dtype=jnp.int32),
            facing_direction=jnp.array([-1, 0], dtype=jnp.int32),  # Start facing left
            ghost_positions=self.ghost_spawn_positions,
            ghost_modes=initial_ghost_modes,
            ghost_timers=initial_ghost_timers,
            ghost_directions=initial_ghost_directions,
            pellets=self.initial_pellets,
            power_pellets=jnp.ones(4, dtype=jnp.bool_),  # All 4 power pellets available
            power_timer=jnp.array(0, dtype=jnp.int32),
            eaten_ghosts=jnp.array(0, dtype=jnp.int32),
            level=jnp.array(1, dtype=jnp.int32),
            collected_pellets=jnp.array(0, dtype=jnp.int32),
            freeze_timer=jnp.array(0, dtype=jnp.int32),
            fruit_type=jnp.array(FruitType.NONE, dtype=jnp.int32),
            fruit_position=jnp.array([80, 100], dtype=jnp.int32),  # Default position
            fruit_timer=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.initial_lives, dtype=jnp.int32),
            pellets_remaining=self.initial_pellet_count,
            game_over=jnp.array(False, dtype=jnp.bool_),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: MsPacmanState, action: int) -> Tuple[MsPacmanObservation, MsPacmanState, float, bool, MsPacmanInfo]:
        # Handle freeze timer (death animation)
        is_frozen = state.freeze_timer > 0
        freeze_timer = jnp.maximum(state.freeze_timer - 1, 0)
        
        # Check if we should reset entities after death freeze
        should_reset_entities = jnp.logical_and(state.freeze_timer == 1, state.lives > 0)
        
        action = jnp.asarray(action, dtype=jnp.int32)
        action_idx = jnp.clip(action, 0, self._action_deltas.shape[0] - 1)
        delta = self._action_deltas[action_idx]

        # Determine movement direction
        has_new_direction = jnp.any(delta != 0)
        direction = jnp.where(has_new_direction, delta, state.direction)
        can_move = (state.time % jnp.maximum(self.consts.player_move_period, 1)) == 0
        
        # Move by player_speed pixels in the direction
        speed = self.consts.player_speed
        tentative_x = state.pacman_x + direction[0] * speed
        tentative_y = state.pacman_y + direction[1] * speed
        
        # Constrain to maze area (offset to offset + maze_width)
        offset = self.consts.maze_offset_x
        maze_w = self.consts.maze_width
        
        # Tunnel wrapping within maze bounds
        maze_x = tentative_x - offset  # Convert to maze-local coordinates
        maze_x = jnp.mod(maze_x, maze_w)  # Wrap within maze
        tentative_x = maze_x + offset  # Convert back to screen coordinates
        
        # Vertical bounds
        tentative_y = jnp.clip(tentative_y, 4, self.consts.screen_height - 10)
        
        # Check for wall collision at pixel position
        sprite_half = 3  # Collision radius
        
        corner_offsets = jnp.array([
            [-sprite_half, -sprite_half],
            [sprite_half, -sprite_half],
            [-sprite_half, sprite_half],
            [sprite_half, sprite_half],
        ])
        
        def check_corner(off):
            cx = tentative_x + off[0] - offset
            cy = tentative_y + off[1]
            cx = jnp.mod(cx, maze_w)
            tile_x = cx // self.tile_scale
            tile_y = (cy - 1) // self.tile_scale
            tile_x = jnp.clip(tile_x, 0, self.maze_grid.shape[1] - 1)
            tile_y = jnp.clip(tile_y, 0, self.maze_grid.shape[0] - 1)
            return self.maze_grid[tile_y, tile_x] == 1
        
        hit_wall = jnp.any(jax.vmap(check_corner)(corner_offsets))
        
        zero_dir = jnp.array([0, 0], dtype=jnp.int32)
        pacman_x = jnp.where(hit_wall, state.pacman_x, tentative_x)
        pacman_y = jnp.where(hit_wall, state.pacman_y, tentative_y)
        direction = jnp.where(hit_wall, zero_dir, direction)
        
        # Apply movement only when allowed and not frozen
        pacman_x = jnp.where(jnp.logical_and(can_move, ~is_frozen), pacman_x, state.pacman_x)
        pacman_y = jnp.where(jnp.logical_and(can_move, ~is_frozen), pacman_y, state.pacman_y)

        # ---- Regular Pellet collection ----
        maze_local_x = pacman_x - offset
        pellet_x = jnp.clip((maze_local_x - 4) // 8, 0, 17)
        pellet_y = jnp.clip((pacman_y - 4) // 12, 0, 13)
        
        pellet_value = state.pellets[pellet_x, pellet_y]
        ate_pellet = jnp.logical_and(pellet_value > 0, ~is_frozen)
        
        pellets = state.pellets.at[pellet_x, pellet_y].set(jnp.where(ate_pellet, 0, pellet_value))

        # ---- Power Pellet collection ----
        # Check distance to each power pellet position
        power_pellet_dists = jnp.abs(POWER_PELLET_POSITIONS - jnp.array([pacman_x, pacman_y])).sum(axis=1)
        power_pellet_close = power_pellet_dists < COLLISION_THRESHOLD
        ate_power_pellet_mask = jnp.logical_and(power_pellet_close, state.power_pellets)
        ate_any_power_pellet = jnp.logical_and(jnp.any(ate_power_pellet_mask), ~is_frozen)
        
        # Update power pellets (remove eaten one)
        power_pellets = jnp.where(
            jnp.logical_and(ate_any_power_pellet, ate_power_pellet_mask),
            False,
            state.power_pellets
        )

        # Calculate pellet rewards
        pellet_reward = jnp.where(ate_pellet, PELLET_POINTS, 0)
        power_reward = jnp.where(ate_any_power_pellet, POWER_PELLET_POINTS, 0)
        
        collected_pellets = state.collected_pellets + ate_pellet.astype(jnp.int32)
        pellets_remaining = jnp.maximum(
            state.pellets_remaining - ate_pellet.astype(jnp.int32),
            0,
        )

        # ---- Ghost mode updates ----
        # Decay power timer
        decayed_timer = jnp.maximum(state.power_timer - 1, 0)
        power_timer = jnp.where(ate_any_power_pellet, FRIGHTENED_DURATION, decayed_timer)
        frightened = power_timer > 0
        blinking = jnp.logical_and(power_timer > 0, power_timer <= BLINKING_DURATION)
        
        # Reset eaten ghost counter on new power pellet
        eaten_ghosts = jnp.where(ate_any_power_pellet, 0, state.eaten_ghosts)
        
        # Update ghost modes based on power pellet
        ghost_modes = jnp.where(
            jnp.logical_and(ate_any_power_pellet, 
                          jnp.logical_and(state.ghost_modes != GhostMode.ENJAILED, 
                                         state.ghost_modes != GhostMode.RETURNING)),
            GhostMode.FRIGHTENED,
            state.ghost_modes
        )
        
        # Transition FRIGHTENED -> BLINKING when timer is low
        ghost_modes = jnp.where(
            jnp.logical_and(ghost_modes == GhostMode.FRIGHTENED, blinking),
            GhostMode.BLINKING,
            ghost_modes
        )
        
        # Transition FRIGHTENED/BLINKING -> CHASE when power timer ends
        ghost_modes = jnp.where(
            jnp.logical_and(
                jnp.logical_or(ghost_modes == GhostMode.FRIGHTENED, ghost_modes == GhostMode.BLINKING),
                power_timer == 0
            ),
            GhostMode.CHASE,
            ghost_modes
        )

        # ---- Move ghosts ----
        ghost_positions, ghost_directions = self._move_ghosts_with_modes(
            state.ghost_positions, 
            state.ghost_directions,
            ghost_modes,
            pacman_x, pacman_y, 
            state.facing_direction,
            state.time,
            is_frozen
        )

        # ---- Ghost collision detection ----
        ghost_dist_x = jnp.abs(ghost_positions[:, 0] - pacman_x)
        ghost_dist_y = jnp.abs(ghost_positions[:, 1] - pacman_y)
        ghost_overlap = jnp.logical_and(ghost_dist_x < COLLISION_THRESHOLD, ghost_dist_y < COLLISION_THRESHOLD)
        ghost_overlap = jnp.logical_and(ghost_overlap, ~is_frozen)
        
        # Check for ghost eating (when frightened/blinking)
        ghost_is_vulnerable = jnp.logical_or(ghost_modes == GhostMode.FRIGHTENED, ghost_modes == GhostMode.BLINKING)
        ghosts_eaten_mask = jnp.logical_and(ghost_overlap, ghost_is_vulnerable)
        
        # Calculate ghost eating reward (doubles for each ghost: 200, 400, 800, 1600)
        # We need to process sequentially for correct doubling
        def calc_ghost_reward(carry, is_eaten):
            total_reward, count = carry
            reward = jnp.where(is_eaten, GHOST_BASE_POINTS * (2 ** count), 0)
            new_count = count + is_eaten.astype(jnp.int32)
            return (total_reward + reward, new_count), reward
        
        (ghost_total_reward, final_eaten_count), _ = jax.lax.scan(
            calc_ghost_reward, 
            (0, eaten_ghosts), 
            ghosts_eaten_mask
        )
        eaten_ghosts = final_eaten_count

        # Reset eaten ghosts to jail
        ghost_positions = jnp.where(
            jnp.broadcast_to(ghosts_eaten_mask[:, None], ghost_positions.shape),
            jnp.broadcast_to(JAIL_POSITION[None, :], ghost_positions.shape),
            ghost_positions,
        )
        
        # Set eaten ghosts to ENJAILED mode
        ghost_modes = jnp.where(ghosts_eaten_mask, GhostMode.ENJAILED, ghost_modes)
        ghost_timers = jnp.where(ghosts_eaten_mask, ENJAILED_DURATION, state.ghost_timers)

        # Check for deadly collision (not frightened/blinking)
        ghost_is_dangerous = jnp.logical_and(
            ~ghost_is_vulnerable,
            jnp.logical_and(ghost_modes != GhostMode.ENJAILED, ghost_modes != GhostMode.RETURNING)
        )
        deadly_collision = jnp.any(jnp.logical_and(ghost_overlap, ghost_is_dangerous))

        # Handle deadly collision
        lives = state.lives - deadly_collision.astype(jnp.int32)
        new_freeze_timer = jnp.where(deadly_collision, RESET_TIMER, freeze_timer)

        # Reset positions on death (applied after freeze timer expires via should_reset_entities)
        pacman_x = jax.lax.select(should_reset_entities, self.pacman_spawn[0], pacman_x)
        pacman_y = jax.lax.select(should_reset_entities, self.pacman_spawn[1], pacman_y)
        ghost_positions = jax.lax.select(should_reset_entities, self.ghost_spawn_positions, ghost_positions)
        ghost_modes = jax.lax.select(
            should_reset_entities,
            jnp.full((self.consts.num_ghosts,), GhostMode.SCATTER, dtype=jnp.int32),
            ghost_modes
        )
        ghost_timers = jax.lax.select(
            should_reset_entities,
            jnp.full((self.consts.num_ghosts,), SCATTER_DURATION, dtype=jnp.int32),
            ghost_timers
        )
        power_timer = jax.lax.select(should_reset_entities, jnp.array(0, dtype=jnp.int32), power_timer)
        direction = jax.lax.select(should_reset_entities, zero_dir, direction)
        eaten_ghosts = jax.lax.select(should_reset_entities, jnp.array(0, dtype=jnp.int32), eaten_ghosts)

        # ---- Calculate total score ----
        score = state.score + pellet_reward + power_reward + ghost_total_reward
        
        # ---- Level completion check ----
        level_complete = pellets_remaining <= 0
        score = score + jnp.where(level_complete, LEVEL_COMPLETE_BONUS, 0)
        level = state.level + level_complete.astype(jnp.int32)
        
        # Reset pellets and power pellets on level complete (simplified - full reset would reload maze)
        pellets = jnp.where(level_complete, self.initial_pellets, pellets)
        power_pellets = jnp.where(level_complete, jnp.ones(4, dtype=jnp.bool_), power_pellets)
        pellets_remaining = jnp.where(level_complete, self.initial_pellet_count, pellets_remaining)
        collected_pellets = jnp.where(level_complete, jnp.array(0, dtype=jnp.int32), collected_pellets)
        pacman_x = jax.lax.select(level_complete, self.pacman_spawn[0], pacman_x)
        pacman_y = jax.lax.select(level_complete, self.pacman_spawn[1], pacman_y)
        ghost_positions = jax.lax.select(level_complete, self.ghost_spawn_positions, ghost_positions)

        time = state.time + 1

        game_over = jnp.logical_or(
            state.game_over,
            jnp.logical_or(
                lives <= 0,
                time >= self.consts.max_steps,
            ),
        )

        # Update facing_direction only when actually moving
        is_moving = jnp.any(direction != 0)
        facing_direction = jnp.where(
            jnp.logical_or(deadly_collision, should_reset_entities),
            jnp.array([-1, 0], dtype=jnp.int32),
            jnp.where(is_moving, direction, state.facing_direction)
        )

        new_state = MsPacmanState(
            pacman_x=pacman_x,
            pacman_y=pacman_y,
            direction=direction,
            facing_direction=facing_direction,
            ghost_positions=ghost_positions,
            ghost_modes=ghost_modes,
            ghost_timers=ghost_timers,
            ghost_directions=ghost_directions,
            pellets=pellets,
            power_pellets=power_pellets,
            power_timer=power_timer,
            eaten_ghosts=eaten_ghosts,
            level=level,
            collected_pellets=collected_pellets,
            freeze_timer=new_freeze_timer,
            fruit_type=state.fruit_type,  # Fruit handling to be added
            fruit_position=state.fruit_position,
            fruit_timer=state.fruit_timer,
            score=score,
            time=time,
            lives=lives,
            pellets_remaining=pellets_remaining,
            game_over=game_over,
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _move_ghosts(
        self,
        ghost_positions: chex.Array,
        pacman_x: chex.Array,
        pacman_y: chex.Array,
        frightened: chex.Array,
        time: chex.Array,
    ) -> chex.Array:
        idx = jnp.arange(self.consts.num_ghosts, dtype=jnp.int32)
        active = time >= (idx * self.consts.ghost_spawn_delay)
        can_move = (time % jnp.maximum(self.consts.ghost_move_period, 1)) == 0
        
        gx = ghost_positions[:, 0]
        gy = ghost_positions[:, 1]
        dx = pacman_x - gx
        dy = pacman_y - gy

        dx_sign = jnp.sign(dx)
        dy_sign = jnp.sign(dy)

        # Reverse direction when frightened
        dx_sign = jnp.where(frightened, -dx_sign, dx_sign)
        dy_sign = jnp.where(frightened, -dy_sign, dy_sign)

        prefer_x = jnp.abs(dx) >= jnp.abs(dy)
        
        # Ghost speed in pixels
        speed = self.consts.ghost_speed
        offset = self.consts.maze_offset_x
        maze_w = self.consts.maze_width
        
        # Primary movement direction
        primary_dx = jnp.where(prefer_x, dx_sign * speed, 0)
        primary_dy = jnp.where(prefer_x, 0, dy_sign * speed)
        
        # Calculate tentative positions with maze wrapping
        cand1_local_x = jnp.mod(gx - offset + primary_dx, maze_w)
        cand1_x = cand1_local_x + offset
        cand1_y = jnp.clip(gy + primary_dy, 4, self.consts.screen_height - 10)
        
        # Check for wall collision (pixel to tile conversion)
        tile_x1 = jnp.clip(cand1_local_x // self.tile_scale, 0, self.maze_grid.shape[1] - 1)
        tile_y1 = jnp.clip((cand1_y - 1) // self.tile_scale, 0, self.maze_grid.shape[0] - 1)
        blocked1 = self.maze_grid[tile_y1, tile_x1] == 1

        # Secondary movement direction
        secondary_dx = jnp.where(prefer_x, 0, dx_sign * speed)
        secondary_dy = jnp.where(prefer_x, dy_sign * speed, 0)
        
        cand2_local_x = jnp.mod(gx - offset + secondary_dx, maze_w)
        cand2_x = cand2_local_x + offset
        cand2_y = jnp.clip(gy + secondary_dy, 4, self.consts.screen_height - 10)
        
        tile_x2 = jnp.clip(cand2_local_x // self.tile_scale, 0, self.maze_grid.shape[1] - 1)
        tile_y2 = jnp.clip((cand2_y - 1) // self.tile_scale, 0, self.maze_grid.shape[0] - 1)
        blocked2 = self.maze_grid[tile_y2, tile_x2] == 1

        use_second = jnp.logical_and(blocked1, jnp.logical_not(blocked2))
        stay = jnp.logical_and(blocked1, blocked2)

        new_x = jnp.where(stay, gx, jnp.where(use_second, cand2_x, cand1_x))
        new_y = jnp.where(stay, gy, jnp.where(use_second, cand2_y, cand1_y))
        new_x = jnp.where(can_move, new_x, gx)
        new_y = jnp.where(can_move, new_y, gy)

        # Respect spawn delays: inactive ghosts stay at spawn
        spawn_x = self.ghost_spawn_positions[:, 0]
        spawn_y = self.ghost_spawn_positions[:, 1]
        final_x = jnp.where(active, new_x, spawn_x)
        final_y = jnp.where(active, new_y, spawn_y)

        return jnp.stack([final_x, final_y], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def _move_ghosts_with_modes(
        self,
        ghost_positions: chex.Array,
        ghost_directions: chex.Array,
        ghost_modes: chex.Array,
        pacman_x: chex.Array,
        pacman_y: chex.Array,
        pacman_dir: chex.Array,
        time: chex.Array,
        is_frozen: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Move ghosts based on their current mode."""
        idx = jnp.arange(self.consts.num_ghosts, dtype=jnp.int32)
        active = time >= (idx * self.consts.ghost_spawn_delay)
        can_move = jnp.logical_and(
            (time % jnp.maximum(self.consts.ghost_move_period, 1)) == 0,
            ~is_frozen
        )
        
        gx = ghost_positions[:, 0]
        gy = ghost_positions[:, 1]
        
        # Get target for each ghost based on mode
        # CHASE mode: target player based on ghost type
        # SCATTER mode: target corners
        # FRIGHTENED/BLINKING: move randomly
        
        # Calculate chase targets for each ghost type
        pacman_pos = jnp.array([pacman_x, pacman_y])
        blinky_pos = ghost_positions[0]  # Blinky is always index 0
        
        # Blinky targets pacman directly
        chase_target_0 = pacman_pos
        
        # Pinky targets 4 tiles ahead of pacman
        pinky_offset = pacman_dir * 16  # 4 tiles * 4 pixels
        chase_target_1 = pacman_pos + pinky_offset
        
        # Inky targets vector from blinky doubled
        two_ahead = pacman_pos + pacman_dir * 8
        inky_vec = two_ahead - blinky_pos
        chase_target_2 = blinky_pos + 2 * inky_vec
        
        # Sue targets pacman if far, else corner
        sue_dist = jnp.sqrt(jnp.sum((ghost_positions[3] - pacman_pos) ** 2))
        chase_target_3 = jnp.where(sue_dist > 32, pacman_pos, SCATTER_TARGETS[3])
        
        chase_targets = jnp.stack([chase_target_0, chase_target_1, chase_target_2, chase_target_3])
        
        # Choose target based on mode
        is_chase = ghost_modes == GhostMode.CHASE
        is_scatter = ghost_modes == GhostMode.SCATTER
        is_frightened = jnp.logical_or(ghost_modes == GhostMode.FRIGHTENED, ghost_modes == GhostMode.BLINKING)
        
        targets = jnp.where(
            is_chase[:, None],
            chase_targets,
            jnp.where(
                is_scatter[:, None],
                SCATTER_TARGETS,
                pacman_pos  # Default for frightened (will use random)
            )
        )
        
        # Calculate movement direction
        dx = targets[:, 0] - gx
        dy = targets[:, 1] - gy
        
        # For frightened mode, reverse direction
        dx = jnp.where(is_frightened, -dx, dx)
        dy = jnp.where(is_frightened, -dy, dy)
        
        dx_sign = jnp.sign(dx)
        dy_sign = jnp.sign(dy)
        
        prefer_x = jnp.abs(dx) >= jnp.abs(dy)
        
        speed = self.consts.ghost_speed
        offset = self.consts.maze_offset_x
        maze_w = self.consts.maze_width
        
        # Primary movement direction
        primary_dx = jnp.where(prefer_x, dx_sign * speed, 0).astype(jnp.int32)
        primary_dy = jnp.where(prefer_x, 0, dy_sign * speed).astype(jnp.int32)
        
        # Calculate tentative positions with maze wrapping
        cand1_local_x = jnp.mod(gx - offset + primary_dx, maze_w)
        cand1_x = cand1_local_x + offset
        cand1_y = jnp.clip(gy + primary_dy, 4, self.consts.screen_height - 10)
        
        # Check for wall collision
        tile_x1 = jnp.clip(cand1_local_x // self.tile_scale, 0, self.maze_grid.shape[1] - 1)
        tile_y1 = jnp.clip((cand1_y - 1) // self.tile_scale, 0, self.maze_grid.shape[0] - 1)
        blocked1 = self.maze_grid[tile_y1, tile_x1] == 1

        # Secondary movement direction
        secondary_dx = jnp.where(prefer_x, 0, dx_sign * speed).astype(jnp.int32)
        secondary_dy = jnp.where(prefer_x, dy_sign * speed, 0).astype(jnp.int32)
        
        cand2_local_x = jnp.mod(gx - offset + secondary_dx, maze_w)
        cand2_x = cand2_local_x + offset
        cand2_y = jnp.clip(gy + secondary_dy, 4, self.consts.screen_height - 10)
        
        tile_x2 = jnp.clip(cand2_local_x // self.tile_scale, 0, self.maze_grid.shape[1] - 1)
        tile_y2 = jnp.clip((cand2_y - 1) // self.tile_scale, 0, self.maze_grid.shape[0] - 1)
        blocked2 = self.maze_grid[tile_y2, tile_x2] == 1

        use_second = jnp.logical_and(blocked1, jnp.logical_not(blocked2))
        stay = jnp.logical_and(blocked1, blocked2)

        new_x = jnp.where(stay, gx, jnp.where(use_second, cand2_x, cand1_x))
        new_y = jnp.where(stay, gy, jnp.where(use_second, cand2_y, cand1_y))
        new_x = jnp.where(can_move, new_x, gx)
        new_y = jnp.where(can_move, new_y, gy)

        # Compute new directions
        new_dir_x = jnp.where(use_second, secondary_dx, primary_dx)
        new_dir_y = jnp.where(use_second, secondary_dy, primary_dy)
        new_dir_x = jnp.where(stay, ghost_directions[:, 0], new_dir_x)
        new_dir_y = jnp.where(stay, ghost_directions[:, 1], new_dir_y)
        new_directions = jnp.stack([new_dir_x, new_dir_y], axis=-1)

        # Enjailed ghosts stay at jail position
        is_enjailed = ghost_modes == GhostMode.ENJAILED
        new_x = jnp.where(is_enjailed, JAIL_POSITION[0], new_x)
        new_y = jnp.where(is_enjailed, JAIL_POSITION[1], new_y)

        # Respect spawn delays: inactive ghosts stay at spawn
        spawn_x = self.ghost_spawn_positions[:, 0]
        spawn_y = self.ghost_spawn_positions[:, 1]
        final_x = jnp.where(active, new_x, spawn_x)
        final_y = jnp.where(active, new_y, spawn_y)

        return jnp.stack([final_x, final_y], axis=-1), new_directions

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: MsPacmanState) -> MsPacmanObservation:
        pacman = EntityPosition(
            x=state.pacman_x,
            y=state.pacman_y,
            width=jnp.array(1, dtype=jnp.int32),
            height=jnp.array(1, dtype=jnp.int32),
        )

        ghost_dims = jnp.ones((self.consts.num_ghosts, 2), dtype=jnp.int32)
        ghosts = jnp.concatenate([state.ghost_positions, ghost_dims], axis=1)

        return MsPacmanObservation(
            pacman=pacman,
            ghosts=ghosts,
            pellets=state.pellets,
            power_timer=state.power_timer,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: MsPacmanState) -> MsPacmanInfo:
        return MsPacmanInfo(
            time=state.time,
            pellets_remaining=state.pellets_remaining,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: MsPacmanState, state: MsPacmanState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: MsPacmanState) -> bool:
        return state.game_over

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)

    def observation_space(self) -> spaces.Dict:
        # Pellet grid dimensions from MsPacmanMaze.BASE_PELLETS
        pellet_grid_h, pellet_grid_w = 18, 14
        return spaces.Dict({
            "pacman": spaces.Dict({
                # Pixel-based coordinates
                "x": spaces.Box(low=0, high=self.consts.screen_width - 1, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.screen_height - 1, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.pacman_size, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.pacman_size, shape=(), dtype=jnp.int32),
            }),
            "ghosts": spaces.Box(
                low=0,
                high=max(self.consts.screen_width, self.consts.screen_height),
                shape=(self.consts.num_ghosts, 4),
                dtype=jnp.int32,
            ),
            "pellets": spaces.Box(
                low=0,
                high=2,
                shape=(pellet_grid_h, pellet_grid_w),
                dtype=jnp.int32,
            ),
            "power_timer": spaces.Box(low=0, high=self.consts.frightened_duration, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.initial_lives, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.screen_height, self.consts.screen_width, 3),
            dtype=jnp.uint8,
        )

    def render(self, state: MsPacmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: MsPacmanObservation) -> jnp.ndarray:
        pacman_flat = jnp.array([obs.pacman.x, obs.pacman.y, obs.pacman.width, obs.pacman.height], dtype=jnp.int32)
        ghosts_flat = obs.ghosts.reshape(-1)
        pellets_flat = obs.pellets.reshape(-1)
        extras = jnp.array([obs.power_timer, obs.lives], dtype=jnp.int32)
        return jnp.concatenate([pacman_flat, ghosts_flat, pellets_flat, extras]).astype(jnp.int32)


class MsPacmanRenderer(JAXGameRenderer):
    """Renderer for the Ms. Pac-Man game using authentic sprites."""
    
    def __init__(self, consts: MsPacmanConstants = None, maze_grid: jnp.ndarray = None, 
                 pellet_template: jnp.ndarray = None, background_image: jnp.ndarray = None):
        super().__init__()
        self.consts = consts or MsPacmanConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.screen_height, self.consts.screen_width),
            channels=3,
        )
        
        # Store maze grid for collision reference
        self.maze_grid = maze_grid
        self.offset_x = self.consts.maze_offset_x
        
        # Create base canvas with background color
        self._base_canvas = jnp.ones(
            (self.consts.screen_height, self.consts.screen_width, 3), 
            dtype=jnp.uint8
        ) * jnp.array(self.consts.background_color, dtype=jnp.uint8)
        
        # Place pre-rendered maze background (160px wide) centered in 200px canvas
        if background_image is not None:
            # Background from MsPacmanMaze.load_background() is (W, H, 3) = (160, 210, 3)
            if background_image.shape[0] == self.consts.maze_width:
                bg = jnp.swapaxes(background_image, 0, 1)  # (210, 160, 3)
            else:
                bg = background_image
            # Place centered in canvas
            self._base_canvas = self._base_canvas.at[:, self.offset_x:self.offset_x + self.consts.maze_width, :].set(bg)

        self.background_color = jnp.asarray(self.consts.background_color, dtype=jnp.uint8)
        self.wall_color = jnp.asarray(self.consts.wall_color, dtype=jnp.uint8)
        self.pellet_color = jnp.asarray(self.consts.pellet_color, dtype=jnp.uint8)
        self.power_pellet_color = jnp.asarray(self.consts.power_pellet_color, dtype=jnp.uint8)
        self.pacman_color = jnp.asarray(self.consts.pacman_color, dtype=jnp.uint8)
        self.ghost_colors = jnp.asarray(self.consts.ghost_colors, dtype=jnp.uint8)
        
        # Load Ms. Pac-Man specific sprites (authentic extracted sprites)
        # Load Pac-Man animation frames (pacman_0 through pacman_3)
        self.pac_sprites = self._load_pacman_sprites()
        self.pac_sprite = self.pac_sprites[0] if self.pac_sprites else None
        
        # Load ghost sprites: Blinky (red), Pinky (pink), Inky (cyan), Sue (orange)
        self.ghost_sprites = self._load_ghost_sprites()
        self.ghost_frightened_sprite = self._load_sprite("ghost_blue")
        self.ghost_frightened_white_sprite = self._load_sprite("ghost_white")
        
        # Pre-stack sprites into JAX arrays for efficient rendering
        self._pac_sprite_array = self._build_pacman_sprite_array()
        self._ghost_sprite_array = self._build_ghost_sprite_array()
        self._frightened_sprite_array = self._build_frightened_sprite_array()
        self._blinking_sprite_array = self._build_blinking_sprite_array()

    def _load_sprite(self, name: str) -> jnp.ndarray | None:
        """Load a sprite from the mspacman sprites folder."""
        try:
            path = os.path.join(os.path.dirname(__file__), "sprites", "mspacman", f"{name}.npy")
            sprite_rgba = np.load(path)
            return jnp.asarray(sprite_rgba, dtype=jnp.uint8)
        except Exception:
            return None

    def _load_pacman_sprites(self) -> list:
        """Load all Pac-Man animation frames."""
        sprites = []
        for i in range(4):
            sprite = self._load_sprite(f"pacman_{i}")
            if sprite is not None:
                sprites.append(sprite)
        return sprites

    def _build_pacman_sprite_array(self) -> jnp.ndarray | None:
        """Stack Pac-Man animation frames with all rotations.
        
        Returns array of shape (4, 4, H, W, 4) where:
        - First 4: animation frames
        - Second 4: rotation directions (0=left, 1=down, 2=right, 3=up)
        
        Note: The original sprite faces LEFT.
        """
        if not self.pac_sprites:
            return None
        
        # Build rotated versions for each animation frame
        # Direction indices: 0=left (original), 1=up, 2=right, 3=down
        # Note: rot90 with k rotates counter-clockwise
        all_rotations = []
        for sprite in self.pac_sprites:
            rotations = [
                sprite,                         # 0: Left 
                jnp.rot90(sprite, k=3),         # 1: Up 
                jnp.rot90(sprite, k=2),         # 2: Right 
                jnp.rot90(sprite, k=1),         # 3: Down 
            ]
            all_rotations.append(jnp.stack(rotations, axis=0))
        
        return jnp.stack(all_rotations, axis=0)

    def _load_ghost_sprites(self) -> list:
        """Load all ghost sprites in order: Blinky, Pinky, Inky, Sue."""
        ghost_names = ["ghost_blinky", "ghost_pinky", "ghost_inky", "ghost_sue"]
        sprites = []
        for name in ghost_names:
            sprite = self._load_sprite(name)
            sprites.append(sprite)
        return sprites

    def _build_ghost_sprite_array(self) -> jnp.ndarray | None:
        """Stack ghost sprites into a single array for vectorized rendering."""
        if not self.ghost_sprites:
            return None
        # Only use as many sprites as we have ghosts in the game
        # Cycle through available sprites if we have more ghosts than sprites
        sprites_to_stack = []
        for i in range(self.consts.num_ghosts):
            sprite_idx = i % len(self.ghost_sprites)
            sprite = self.ghost_sprites[sprite_idx]
            if sprite is None:
                return None  # Can't build array if any sprite is missing
            sprites_to_stack.append(sprite)
        return jnp.stack(sprites_to_stack, axis=0)

    def _build_frightened_sprite_array(self) -> jnp.ndarray | None:
        """Build array of frightened sprites for all ghosts."""
        if self.ghost_frightened_sprite is None:
            return None
        # Repeat the frightened sprite for each ghost
        return jnp.stack([self.ghost_frightened_sprite] * self.consts.num_ghosts, axis=0)

    def _build_blinking_sprite_array(self) -> jnp.ndarray | None:
        """Build array of blinking (white) sprites for all ghosts."""
        if self.ghost_frightened_white_sprite is None:
            return None
        # Repeat the white/blinking sprite for each ghost
        return jnp.stack([self.ghost_frightened_white_sprite] * self.consts.num_ghosts, axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: MsPacmanState) -> jnp.ndarray:
        # Start with the pre-rendered background (includes maze and pellets)
        canvas = self._base_canvas.copy()
        
        # Draw Ms. Pac-Man using animated sprite with direction-based rotation
        pac_x = state.pacman_x
        pac_y = state.pacman_y
        
        if self._pac_sprite_array is not None:
            # Select animation frame based on game time (ping-pong)
            frame_cycle = jnp.array([0, 1, 2, 3, 2, 1], dtype=jnp.int32)
            cycle_idx = (state.time // 3) % 6
            frame_idx = frame_cycle[cycle_idx]
            
            # Determine rotation based on facing_direction
            dx, dy = state.facing_direction[0], state.facing_direction[1]
            
            # Map to rotation indices: 0=left, 1=up, 2=right, 3=down
            rot_idx = jnp.where(
                dx > 0, 2,  # Right
                jnp.where(
                    dx < 0, 0,  # Left
                    jnp.where(
                        dy > 0, 3,  # Down
                        jnp.where(dy < 0, 1, 0)  # Up, default to left
                    )
                )
            )
            
            # Get sprite with correct animation frame and rotation
            pac_sprite = self._pac_sprite_array[frame_idx, rot_idx]
            sprite_h, sprite_w = pac_sprite.shape[:2]
            
            # Center sprite on position
            sprite_x = pac_x - sprite_w // 2
            sprite_y = pac_y - sprite_h // 2
            
            # Clamp to canvas bounds
            sprite_x = jnp.clip(sprite_x, 0, self.consts.screen_width - sprite_w)
            sprite_y = jnp.clip(sprite_y, 0, self.consts.screen_height - sprite_h)
            
            # Alpha blend the sprite
            h, w = pac_sprite.shape[:2]
            region = jax.lax.dynamic_slice(canvas, (sprite_y, sprite_x, 0), (h, w, 3))
            alpha = pac_sprite[:, :, 3:4].astype(jnp.uint16)
            fg_rgb = pac_sprite[:, :, :3].astype(jnp.uint16)
            bg_rgb = region.astype(jnp.uint16)
            blended = ((fg_rgb * alpha) + (bg_rgb * (255 - alpha))) // 255
            canvas = jax.lax.dynamic_update_slice(canvas, blended.astype(jnp.uint8), (sprite_y, sprite_x, 0))
        else:
            # Fallback: draw Pac-Man as colored block
            pac_block = jnp.ones((self.consts.pacman_size, self.consts.pacman_size, 3), dtype=jnp.uint8) * self.pacman_color
            block_x = jnp.clip(pac_x - 5, 0, self.consts.screen_width - self.consts.pacman_size)
            block_y = jnp.clip(pac_y - 5, 0, self.consts.screen_height - self.consts.pacman_size)
            canvas = jax.lax.dynamic_update_slice(canvas, pac_block, (block_y, block_x, 0))

        # Draw ghosts
        ghost_positions = state.ghost_positions
        ghost_modes = state.ghost_modes
        frightened = state.power_timer > 0
        # Blinking alternates between blue and white based on time
        blink_white = (state.time // 8) % 2 == 1  # Toggle every 8 frames

        if self._ghost_sprite_array is not None:
            ghost_sprite_h, ghost_sprite_w = self.ghost_sprites[0].shape[:2]
            
            # Build sprite selection for each ghost based on their mode
            # For each ghost: use normal if not frightened/blinking, blue if frightened, 
            # alternate blue/white if blinking
            
            def select_ghost_sprite(idx):
                mode = ghost_modes[idx]
                is_frightened = jnp.logical_or(mode == GhostMode.FRIGHTENED, mode == GhostMode.BLINKING)
                is_blinking = mode == GhostMode.BLINKING
                is_enjailed = mode == GhostMode.ENJAILED
                
                # Use white sprite when blinking and blink_white is True
                use_white = jnp.logical_and(is_blinking, blink_white)
                
                normal_sprite = self._ghost_sprite_array[idx]
                frightened_sprite = self._frightened_sprite_array[idx]
                blinking_sprite = self._blinking_sprite_array[idx] if self._blinking_sprite_array is not None else frightened_sprite
                
                sprite = jnp.where(
                    is_frightened,
                    jnp.where(use_white, blinking_sprite, frightened_sprite),
                    normal_sprite
                )
                return sprite
            
            # Pre-select sprites for all ghosts
            sprites_to_use = jax.vmap(select_ghost_sprite)(jnp.arange(self.consts.num_ghosts, dtype=jnp.int32))

            def draw_ghost_sprite(img, inputs):
                idx, pos = inputs
                gx, gy = pos[0], pos[1]
                sprite = sprites_to_use[idx]
                
                # Center sprite on position
                sprite_x = gx - ghost_sprite_w // 2
                sprite_y = gy - ghost_sprite_h // 2
                
                # Clamp to canvas bounds
                sprite_x = jnp.clip(sprite_x, 0, self.consts.screen_width - ghost_sprite_w)
                sprite_y = jnp.clip(sprite_y, 0, self.consts.screen_height - ghost_sprite_h)
                
                # Alpha blend
                h, w = ghost_sprite_h, ghost_sprite_w
                region = jax.lax.dynamic_slice(img, (sprite_y, sprite_x, 0), (h, w, 3))
                alpha = sprite[:, :, 3:4].astype(jnp.uint16)
                fg_rgb = sprite[:, :, :3].astype(jnp.uint16)
                bg_rgb = region.astype(jnp.uint16)
                blended = ((fg_rgb * alpha) + (bg_rgb * (255 - alpha))) // 255
                img = jax.lax.dynamic_update_slice(img, blended.astype(jnp.uint8), (sprite_y, sprite_x, 0))
                return img, None

            canvas, _ = jax.lax.scan(
                draw_ghost_sprite,
                canvas,
                (jnp.arange(self.consts.num_ghosts, dtype=jnp.int32), ghost_positions),
            )
        else:
            # Fallback: draw ghosts as colored blocks
            def draw_ghost_block(img, inputs):
                idx, pos = inputs
                gx, gy = pos[0], pos[1]
                color_idx = idx % self.ghost_colors.shape[0]
                color = jax.lax.select(frightened, jnp.array([0, 0, 255], dtype=jnp.uint8), self.ghost_colors[color_idx])
                block = jnp.ones((self.consts.ghost_size, self.consts.ghost_size, 3), dtype=jnp.uint8) * color
                block_x = jnp.clip(gx - 5, 0, self.consts.screen_width - self.consts.ghost_size)
                block_y = jnp.clip(gy - 5, 0, self.consts.screen_height - self.consts.ghost_size)
                img = jax.lax.dynamic_update_slice(img, block, (block_y, block_x, 0))
                return img, None

            canvas, _ = jax.lax.scan(
                draw_ghost_block,
                canvas,
                (jnp.arange(self.consts.num_ghosts, dtype=jnp.int32), ghost_positions),
            )

        return canvas
