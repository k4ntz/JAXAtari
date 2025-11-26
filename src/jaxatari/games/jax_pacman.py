import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Pacman.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'group', 'files': [
            'player/player_right_1.npy', 'player/player_right_2.npy',
            'player/player_left_1.npy', 'player/player_left_2.npy',
            'player/player_up_1.npy', 'player/player_up_2.npy',
            'player/player_down_1.npy', 'player/player_down_2.npy',
        ]},
        {'name': 'ghost_blinky', 'type': 'group', 'files': [
            'ghost_blinky/ghost_right.npy', 'ghost_blinky/ghost_left.npy',
            'ghost_blinky/ghost_up.npy', 'ghost_blinky/ghost_down.npy',
        ]},
        {'name': 'ghost_pinky', 'type': 'group', 'files': [
            'ghost_pinky/ghost_right.npy', 'ghost_pinky/ghost_left.npy',
            'ghost_pinky/ghost_up.npy', 'ghost_pinky/ghost_down.npy',
        ]},
        {'name': 'ghost_inky', 'type': 'group', 'files': [
            'ghost_inky/ghost_right.npy', 'ghost_inky/ghost_left.npy',
            'ghost_inky/ghost_up.npy', 'ghost_inky/ghost_down.npy',
        ]},
        {'name': 'ghost_clyde', 'type': 'group', 'files': [
            'ghost_clyde/ghost_right.npy', 'ghost_clyde/ghost_left.npy',
            'ghost_clyde/ghost_up.npy', 'ghost_clyde/ghost_down.npy',
        ]},
        {'name': 'ghost_frightened', 'type': 'group', 'files': [
            'ghost_frightened/ghost_frightened_1.npy', 'ghost_frightened/ghost_frightened_2.npy',
        ]},
        {'name': 'ghost_eyes', 'type': 'group', 'files': [
            'ghost_eyes/eyes_right.npy', 'ghost_eyes/eyes_left.npy',
            'ghost_eyes/eyes_up.npy', 'ghost_eyes/eyes_down.npy',
        ]},
        {'name': 'pellet_dot', 'type': 'single', 'file': 'pellet_dot.npy'},
        {'name': 'pellet_power', 'type': 'single', 'file': 'pellet_power.npy'},
        {'name': 'wall', 'type': 'single', 'file': 'wall/0.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/digit_{}.npy'},
    )


class PacmanConstants(NamedTuple):
    # Screen dimensions (Atari 2600 Pacman uses 224x288, but we'll use standard 210x160)
    WIDTH: int = 160
    HEIGHT: int = 210
    
    # Tile size for maze (8x8 pixels per tile)
    TILE_SIZE: int = 8
    
    # Maze dimensions in tiles
    MAZE_WIDTH: int = 28  # 28 tiles wide
    MAZE_HEIGHT: int = 31  # 31 tiles tall
    
    # Player constants
    PLAYER_SIZE: Tuple[int, int] = (8, 8)
    PLAYER_SPEED: int = 1  # pixels per step
    PLAYER_START_X: int = 76  # Center of maze
    PLAYER_START_Y: int = 188  # Bottom area
    
    # Ghost constants
    GHOST_SIZE: Tuple[int, int] = (8, 8)
    GHOST_SPEED_NORMAL: int = 1
    GHOST_SPEED_FRIGHTENED: int = 1
    GHOST_SPEED_EATEN: int = 2
    
    # Ghost starting positions (in ghost house area)
    GHOST_START_X: int = 76
    GHOST_START_Y: int = 100
    
    # Ghost colors (RGB)
    GHOST_BLINKY_COLOR: Tuple[int, int, int] = (255, 0, 0)  # Red
    GHOST_PINKY_COLOR: Tuple[int, int, int] = (255, 192, 203)  # Pink
    GHOST_INKY_COLOR: Tuple[int, int, int] = (0, 255, 255)  # Cyan
    GHOST_CLYDE_COLOR: Tuple[int, int, int] = (255, 165, 0)  # Orange
    GHOST_FRIGHTENED_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Blue
    
    # Pellet constants
    PELLET_DOT_SIZE: Tuple[int, int] = (2, 2)
    PELLET_POWER_SIZE: Tuple[int, int] = (6, 6)
    PELLET_DOT_SCORE: int = 10
    PELLET_POWER_SCORE: int = 50
    GHOST_SCORE_BASE: int = 200  # Base score for first ghost, doubles for each
    
    # Game timing constants
    FRIGHTENED_DURATION: int = 200  # frames
    SCATTER_DURATION: int = 7000  # frames
    CHASE_DURATION: int = 20000  # frames
    
    # Colors
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)  # Black
    WALL_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Blue
    PELLET_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Yellow
    SCORE_COLOR: Tuple[int, int, int] = (255, 255, 255)  # White
    
    # Maze layout - simplified representation
    # 0 = empty, 1 = wall, 2 = dot, 3 = power pellet, 4 = ghost house
    # This will be a 2D array representing the maze
    MAZE_LAYOUT: chex.Array = None  # Will be initialized
    
    # Asset config
    ASSET_CONFIG: tuple = _get_default_asset_config()


# Ghost states: 0=normal, 1=frightened, 2=eaten
class GhostState(NamedTuple):
    x: chex.Array
    y: chex.Array
    direction: chex.Array  # 0=right, 1=left, 2=up, 3=down
    state: chex.Array  # 0=normal, 1=frightened, 2=eaten
    target_x: chex.Array
    target_y: chex.Array


class PacmanState(NamedTuple):
    # Player state
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array  # 0=right, 1=left, 2=up, 3=down
    player_next_direction: chex.Array  # Queued direction for cornering
    player_animation_frame: chex.Array  # 0 or 1 for mouth open/close
    
    # Ghost states (4 ghosts)
    ghosts: chex.Array  # Shape: (4, 6) - [x, y, direction, state, target_x, target_y] for each ghost
    
    # Pellet states - simplified: track number of dots remaining
    dots_remaining: chex.Array
    power_pellets_active: chex.Array  # Bitmask for 4 power pellets
    
    # Game state
    score: chex.Array
    lives: chex.Array
    level: chex.Array
    pellets_collected: chex.Array  # 31x28 mask: 0=not collected, 1=collected
    
    # Timers
    frightened_timer: chex.Array
    scatter_chase_timer: chex.Array
    is_scatter_mode: chex.Array  # True for scatter, False for chase
    
    # Step counter and RNG
    step_counter: chex.Array
    key: chex.PRNGKey


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray


class PacmanObservation(NamedTuple):
    player: EntityPosition
    ghosts: jnp.ndarray  # Shape: (4, 5) - [x, y, width, height, state] for each ghost
    dots_remaining: jnp.ndarray
    power_pellets_active: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    level: jnp.ndarray
    frightened_timer: jnp.ndarray


class PacmanInfo(NamedTuple):
    step_counter: jnp.ndarray
    level: jnp.ndarray


class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo, PacmanConstants]):
    def __init__(self, consts: PacmanConstants = None):
        consts = consts or PacmanConstants()
        super().__init__(consts)
        self.renderer = PacmanRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]
        
        # Initialize maze layout if not provided
        if consts.MAZE_LAYOUT is None:
            self.consts = consts._replace(MAZE_LAYOUT=self._create_default_maze())
        else:
            self.consts = consts

    def _create_default_maze(self) -> jnp.ndarray:
        """Create a classic Pacman maze layout.
        
        Maze values:
        0 = empty path
        1 = wall
        2 = dot pellet
        3 = power pellet
        4 = ghost house area
        """
        # Create maze as numpy array for easier initialization, then convert to jax
        maze = np.zeros((self.consts.MAZE_HEIGHT, self.consts.MAZE_WIDTH), dtype=np.int32)

        maze_layout = [
            "1111111111111111111111111111",
            "1222222222222112222222222221",
            "1211112111112112111112111121",
            "1311112111112112111112111131",
            "1211112111112112111112111121",
            "1222222222222222222222222221",
            "1211112112111111112112111121",
            "1211112112111111112112111121",
            "1222222112222112222112222221",
            "1111112111110110111112111111",
            "1111112111110110111112111111",
            "1111112110000000001112111111",
            "1111112110111441110112111111",
            "1111112110144441110112111111",
            "0000002000144441000002000000",
            "1111112110144441110112111111",
            "1111112110111111110112111111",
            "1111112110000000001112111111",
            "1111112110111111110112111111",
            "1222222222222112222222222221",
            "1211112111112112111112111121",
            "1211112111112112111112111121",
            "1322112222222222222222112231",
            "1112112112111111112112112111",
            "1112112112111111112112112111",
            "1222222112222112222112222221",
            "1211111111112112111111111121",
            "1211111111112112111111111121",
            "1222222222222222222222222221",
            "1111111111111111111111111111",
        ]

        for row_idx, row_str in enumerate(maze_layout):
            for col_idx, char in enumerate(row_str):
                maze[row_idx, col_idx] = int(char)
        
        return jnp.array(maze, dtype=jnp.int32)

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[PacmanObservation, PacmanState]:
        state_key, _ = jax.random.split(key)
        
        # Initialize player at valid path with dots (row 25, col 12)
        # Don't add offset - keep within bounds
        player_x = jnp.array(12 * self.consts.TILE_SIZE, dtype=jnp.int32)  # Column 12 = 96
        player_y = jnp.array(25 * self.consts.TILE_SIZE, dtype=jnp.int32)  # Row 25 = 200
        player_direction = jnp.array(0, dtype=jnp.int32)  # Start facing right
        player_next_direction = jnp.array(-1, dtype=jnp.int32)
        player_animation_frame = jnp.array(0, dtype=jnp.int32)
        
        # Initialize ghosts (4 ghosts at starting positions)
        ghosts = jnp.zeros((4, 6), dtype=jnp.int32)
        for i in range(4):
            ghosts = ghosts.at[i, 0].set(self.consts.GHOST_START_X + i * 8)  # x
            ghosts = ghosts.at[i, 1].set(self.consts.GHOST_START_Y)  # y
            ghosts = ghosts.at[i, 2].set(0)  # direction (right)
            ghosts = ghosts.at[i, 3].set(0)  # state (normal)
            ghosts = ghosts.at[i, 4].set(self.consts.GHOST_START_X)  # target_x
            ghosts = ghosts.at[i, 5].set(self.consts.GHOST_START_Y)  # target_y
        
        # Initial game state
        score = jnp.array(0, dtype=jnp.int32)
        lives = jnp.array(3, dtype=jnp.int32)
        level = jnp.array(1, dtype=jnp.int32)
        dots_remaining = jnp.array(240, dtype=jnp.int32)  # Approximate
        frightened_timer = jnp.array(0, dtype=jnp.int32)
        scatter_chase_timer = jnp.array(0, dtype=jnp.int32)
        is_scatter_mode = jnp.array(True)  # Start in scatter mode
        
        # Initialize pellet collection mask (0 = not collected)
        pellets_collected = jnp.zeros((self.consts.MAZE_HEIGHT, self.consts.MAZE_WIDTH), dtype=jnp.int32)
        
        # Create state
        state = PacmanState(
            player_x=player_x,
            player_y=player_y,
            player_direction=player_direction,
            player_next_direction=player_next_direction,
            player_animation_frame=player_animation_frame,
            ghosts=ghosts,
            dots_remaining=dots_remaining,
            power_pellets_active=jnp.array(15, dtype=jnp.int32), # Re-added as it's part of PacmanState
            score=score,
            lives=lives,
            level=level,
            pellets_collected=pellets_collected,
            frightened_timer=frightened_timer,
            scatter_chase_timer=scatter_chase_timer,
            is_scatter_mode=is_scatter_mode,
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=state_key,
        )
        
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array) -> Tuple[PacmanObservation, PacmanState, float, bool, PacmanInfo]:
        new_state_key, step_key = jax.random.split(state.key)
        previous_state = state
        
        # Update state key
        state = state._replace(key=step_key)
        
        # Update player movement
        state = self._player_step(state, action)
        
        # Split key for ghost updates
        key, ghost_key = jax.random.split(state.key)
        state = state._replace(key=key)
        
        # 2. Ghost Step
        state = self._ghost_step(state, ghost_key)
        
        # Check collisions
        state = self._check_collisions(state)
        
        # Update timers
        state = self._update_timers(state)
        
        # Update animation frames
        state = state._replace(
            player_animation_frame=(state.player_animation_frame + 1) % 2,
            step_counter=state.step_counter + 1,
            key=new_state_key,
        )
        
        done = self._get_done(state)
        reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)
        
        return observation, state, reward, done, info

    def _player_step(self, state: PacmanState, action: chex.Array) -> PacmanState:
        """Handle player movement based on action."""
        # Check if this is a NOOP action
        is_noop = action == Action.NOOP
        
        # Map Action enum to our direction: 0=right, 1=left, 2=up, 3=down
        # Action.NOOP=0, Action.UP=2, Action.DOWN=5, Action.LEFT=4, Action.RIGHT=3
        new_direction = jnp.where(
            action == Action.RIGHT, 0,
            jnp.where(action == Action.LEFT, 1,
            jnp.where(action == Action.UP, 2,
            jnp.where(action == Action.DOWN, 3, state.player_direction)))
        )
        
        # Check if we can move in the new direction
        can_move_direction = self._can_move_in_direction(state.player_x, state.player_y, new_direction)
        
        # Only move if NOT noop AND can move in direction
        should_move = jnp.logical_and(jnp.logical_not(is_noop), can_move_direction)
        
        # Use the new direction
        current_dir = new_direction
        
        # Calculate movement deltas
        # 0=right(+x), 1=left(-x), 2=up(-y), 3=down(+y)
        dx = jnp.where(current_dir == 0, self.consts.PLAYER_SPEED, 
             jnp.where(current_dir == 1, -self.consts.PLAYER_SPEED, 0))
        dy = jnp.where(current_dir == 2, -self.consts.PLAYER_SPEED, 
             jnp.where(current_dir == 3, self.consts.PLAYER_SPEED, 0))
        
        # Apply movement only if should_move is True
        new_x = jnp.where(should_move, state.player_x + dx, state.player_x)
        new_y = jnp.where(should_move, state.player_y + dy, state.player_y)
        
        # Wrap around screen edges (tunnel effect)
        new_x = jnp.where(new_x < 0, self.consts.WIDTH - 1, new_x)
        new_x = jnp.where(new_x >= self.consts.WIDTH, 0, new_x)
        
        # Clamp to screen bounds
        new_x = jnp.clip(new_x, 0, self.consts.WIDTH - self.consts.PLAYER_SIZE[0])
        new_y = jnp.clip(new_y, 0, self.consts.HEIGHT - self.consts.PLAYER_SIZE[1])
        
        return state._replace(
            player_x=new_x.astype(jnp.int32),
            player_y=new_y.astype(jnp.int32),
            player_direction=current_dir,
            player_next_direction=jnp.array(-1, dtype=jnp.int32),
        )

    def _can_move_in_direction(self, x: chex.Array, y: chex.Array, direction: chex.Array) -> chex.Array:
        """Check if player can move in given direction based on maze walls."""
        # Handle invalid direction
        invalid = direction < 0
        
        # Calculate next position
        dx = jnp.where(direction == 0, self.consts.PLAYER_SPEED,
             jnp.where(direction == 1, -self.consts.PLAYER_SPEED, 0))
        dy = jnp.where(direction == 2, -self.consts.PLAYER_SPEED,
             jnp.where(direction == 3, self.consts.PLAYER_SPEED, 0))
        
        next_x = x + dx
        next_y = y + dy
        
        # Convert to tile coordinates (center of player sprite)
        tile_x = next_x // self.consts.TILE_SIZE
        tile_y = next_y // self.consts.TILE_SIZE
        
        # Clamp to valid maze bounds
        tile_x = jnp.clip(tile_x, 0, self.consts.MAZE_WIDTH - 1)
        tile_y = jnp.clip(tile_y, 0, self.consts.MAZE_HEIGHT - 1)
        
        # Check if tile is not a wall
        tile_value = self.consts.MAZE_LAYOUT[tile_y, tile_x]
        not_wall = tile_value != 1
        
        # Also check screen bounds
        in_bounds = jnp.logical_and(
            jnp.logical_and(next_x >= 0, next_x < self.consts.WIDTH),
            jnp.logical_and(next_y >= 0, next_y < self.consts.HEIGHT)
        )
        
        return jnp.logical_and(jnp.logical_and(not_wall, in_bounds), jnp.logical_not(invalid))

    def _ghost_step(self, state: PacmanState, keys: chex.PRNGKey) -> PacmanState:
        """Update ghost positions and states."""
        
        def update_ghost(ghost_data, ghost_key):
            gx, gy, gdir, gstate, gtx, gty = ghost_data
            ghost_idx = jnp.where(
                (state.ghosts[:, 0] == gx) & (state.ghosts[:, 1] == gy),
                jnp.arange(4),
                -1
            )[0] # This is a bit hacky to get index, but we can pass index via vmap if needed
            # Better approach: vmap over indices
            
            return ghost_data # Placeholder, we'll implement the logic inside the vmap wrapper below
            
        # We need to process ghosts with their indices to assign scatter targets
        ghost_indices = jnp.arange(4)
        
        def process_single_ghost(idx, ghost_data, key):
            gx, gy, gdir, gstate, _, _ = ghost_data
            
            # 1. Select target
            # Scatter targets (corners)
            scatter_targets_x = jnp.array([
                self.consts.WIDTH - self.consts.GHOST_SIZE[0], # Blinky -> Top Right
                0,                                             # Pinky -> Top Left
                self.consts.WIDTH - self.consts.GHOST_SIZE[0], # Inky -> Bottom Right
                0                                              # Clyde -> Bottom Left
            ])
            scatter_targets_y = jnp.array([
                0,                                             # Blinky -> Top Right
                0,                                             # Pinky -> Top Left
                self.consts.HEIGHT - self.consts.GHOST_SIZE[1],# Inky -> Bottom Right
                self.consts.HEIGHT - self.consts.GHOST_SIZE[1] # Clyde -> Bottom Left
            ])
            
            target_x = jnp.where(
                gstate == 2, self.consts.GHOST_START_X, # Eaten -> Home
                jnp.where(
                    gstate == 1, gx, # Frightened -> Random (handled by random dir choice)
                    jnp.where(
                        state.is_scatter_mode,
                        scatter_targets_x[idx],
                        state.player_x # Chase -> Player
                    )
                )
            )
            
            target_y = jnp.where(
                gstate == 2, self.consts.GHOST_START_Y, # Eaten -> Home
                jnp.where(
                    gstate == 1, gy, # Frightened -> Random
                    jnp.where(
                        state.is_scatter_mode,
                        scatter_targets_y[idx],
                        state.player_y # Chase -> Player
                    )
                )
            )
            
            # 2. Check valid moves
            
            # Check if move is valid
            def is_valid_move(d):
                dx = jnp.where(d == 0, 1, jnp.where(d == 1, -1, 0)) * self.consts.GHOST_SPEED_NORMAL
                dy = jnp.where(d == 2, -1, jnp.where(d == 3, 1, 0)) * self.consts.GHOST_SPEED_NORMAL
                
                # Collision check
                check_x = gx + dx + self.consts.GHOST_SIZE[0] // 2
                check_y = gy + dy + self.consts.GHOST_SIZE[1] // 2
                
                # Convert to tile coordinates
                tx = check_x // self.consts.TILE_SIZE
                ty = check_y // self.consts.TILE_SIZE
                
                # Boundary check
                in_bounds = (tx >= 0) & (tx < self.consts.MAZE_WIDTH) & \
                            (ty >= 0) & (ty < self.consts.MAZE_HEIGHT)
                
                # Wall check (1=wall)
                is_wall = jnp.where(
                    in_bounds,
                    self.consts.MAZE_LAYOUT[ty, tx] == 1,
                    True # Out of bounds is wall
                )
                
                # Prevent immediate reverse
                # 0(R) <-> 1(L)
                # 2(U) <-> 3(D)
                is_reverse = jnp.where(
                    gdir == 0, d == 1,
                    jnp.where(gdir == 1, d == 0,
                    jnp.where(gdir == 2, d == 3,
                    jnp.where(gdir == 3, d == 2, False)))
                )
                
                # Allow reverse if frightened or eaten, otherwise forbid
                allow_reverse = (gstate != 0) 
                
                return jnp.logical_and(jnp.logical_not(is_wall), jnp.logical_or(jnp.logical_not(is_reverse), allow_reverse))

            # Check all 4 directions
            valid_dirs = jax.vmap(is_valid_move)(jnp.arange(4))
            
            # 3. Pick best direction
            # Calc distances
            def get_dist(d):
                dx = jnp.where(d == 0, 1, jnp.where(d == 1, -1, 0)) * self.consts.GHOST_SPEED_NORMAL
                dy = jnp.where(d == 2, -1, jnp.where(d == 3, 1, 0)) * self.consts.GHOST_SPEED_NORMAL
                nx, ny = gx + dx, gy + dy
                return (nx - target_x)**2 + (ny - target_y)**2
            
            dists = jax.vmap(get_dist)(jnp.arange(4))
            
            # Mask invalid directions with infinity
            masked_dists = jnp.where(valid_dirs, dists, jnp.inf)
            
            # Pick direction with min distance
            # Tie-break: Up > Left > Down > Right
            # We can achieve this by adding small offsets to distances based on priority
            # Lower distance is better.
            # Priority: 2 > 1 > 3 > 0
            # Add: 0.0 for 2, 0.1 for 1, 0.2 for 3, 0.3 for 0
            priority_offsets = jnp.array([0.3, 0.1, 0.0, 0.2])
            masked_dists = masked_dists + priority_offsets
            
            best_dir = jnp.argmin(masked_dists)
            
            # Random move if frightened
            random_dir_idx = jax.random.randint(key, (), 0, 4)
            # We want a random VALID direction.
            # Simple way: add large random noise to dists if frightened
            is_frightened = (gstate == 1)
            noise = jax.random.uniform(key, (4,)) * 10000.0
            frightened_dists = jnp.where(valid_dirs, noise, jnp.inf)
            
            final_dir = jnp.where(
                is_frightened,
                jnp.argmin(frightened_dists),
                best_dir
            )
            
            # Fallback if stuck
            any_valid = jnp.any(valid_dirs)
            final_dir = jnp.where(any_valid, final_dir, (gdir + 1) % 4) # fallback
            
            # 4. Update position
            speed = jnp.where(
                gstate == 2, self.consts.GHOST_SPEED_EATEN,
                jnp.where(gstate == 1, self.consts.GHOST_SPEED_FRIGHTENED, self.consts.GHOST_SPEED_NORMAL)
            )
            
            move_dx = jnp.where(final_dir == 0, 1, jnp.where(final_dir == 1, -1, 0))
            move_dy = jnp.where(final_dir == 2, -1, jnp.where(final_dir == 3, 1, 0))
            
            new_x = gx + move_dx * speed
            new_y = gy + move_dy * speed
            
            # Wrap around
            new_x = jnp.where(new_x < 0, self.consts.WIDTH - 1, new_x)
            new_x = jnp.where(new_x >= self.consts.WIDTH, 0, new_x)
            
            return jnp.array([new_x, new_y, final_dir, gstate, target_x, target_y], dtype=jnp.int32)

        # Process all ghosts
        keys = jax.random.split(keys, 4)
        new_ghosts = jax.vmap(process_single_ghost)(ghost_indices, state.ghosts, keys)
        
        # Update state key (consume one more)
        new_key, _ = jax.random.split(keys[0]) # Just mixing keys to get a new one
        
        return state._replace(ghosts=new_ghosts, key=new_key)

    def _check_collisions(self, state: PacmanState) -> PacmanState:
        """Check player-ghost and player-pellet collisions."""
        # Check ghost collisions
        player_left = state.player_x
        player_right = state.player_x + self.consts.PLAYER_SIZE[0]
        player_top = state.player_y
        player_bottom = state.player_y + self.consts.PLAYER_SIZE[1]
        
        def check_ghost_collision(ghost_data):
            gx, gy, _, gstate, _, _ = ghost_data
            ghost_left = gx
            ghost_right = gx + self.consts.GHOST_SIZE[0]
            ghost_top = gy
            ghost_bottom = gy + self.consts.GHOST_SIZE[1]
            
            # AABB collision
            collision = jnp.logical_and(
                jnp.logical_and(player_left < ghost_right, player_right > ghost_left),
                jnp.logical_and(player_top < ghost_bottom, player_bottom > ghost_top)
            )
            
            return collision
        
        collisions = jax.vmap(check_ghost_collision)(state.ghosts)
        
        # Process collisions for all ghosts
        def process_collision(ghost_idx, ghost_data, collided):
            gx, gy, gdir, gstate, gtx, gty = ghost_data
            
            # If frightened and collided, ghost is eaten
            new_state = jnp.where(
                jnp.logical_and(collided, gstate == 1),
                2,  # eaten
                gstate
            )
            
            # If normal and collided, player loses life
            life_lost = jnp.logical_and(collided, gstate == 0)
            
            # Calculate score for eating ghost
            ghost_score = jnp.where(
                jnp.logical_and(collided, gstate == 1),
                self.consts.GHOST_SCORE_BASE * (2 ** ghost_idx),  # Doubles for each ghost
                0
            )
            
            return new_state, life_lost, ghost_score
        
        results = jax.vmap(process_collision)(jnp.arange(4), state.ghosts, collisions)
        new_ghost_states = results[0]
        any_life_lost = jnp.any(results[1])
        ghost_scores = jnp.sum(results[2])
        
        # Update ghost states
        new_ghosts = state.ghosts.at[:, 3].set(new_ghost_states)
        


        # Update score
        new_score = state.score + ghost_scores
        
        # Update lives
        final_lives = jnp.where(any_life_lost, state.lives - 1, state.lives)
        final_lives = jnp.maximum(final_lives, 0)
        
        # Reset player position if life lost
        new_player_x = jnp.where(any_life_lost, self.consts.PLAYER_START_X, state.player_x)
        new_player_y = jnp.where(any_life_lost, self.consts.PLAYER_START_Y, state.player_y)
        
        # Check pellet collisions based on player's tile position
        # Convert player position to tile coordinates (simple division)
        player_tile_x = state.player_x // self.consts.TILE_SIZE
        player_tile_y = state.player_y // self.consts.TILE_SIZE
        
        # Clamp to maze bounds
        player_tile_x = jnp.clip(player_tile_x, 0, self.consts.MAZE_WIDTH - 1)
        player_tile_y = jnp.clip(player_tile_y, 0, self.consts.MAZE_HEIGHT - 1)
        
        # Get tile value at player position
        tile_val = self.consts.MAZE_LAYOUT[player_tile_y, player_tile_x]
        
        # Check if pellet was already collected
        already_collected = state.pellets_collected[player_tile_y, player_tile_x] == 1
        
        # Check if dot pellet (2) or power pellet (3) is at this position AND not yet collected
        dot_collected = jnp.logical_and(tile_val == 2, jnp.logical_not(already_collected))
        power_pellet_collected = jnp.logical_and(tile_val == 3, jnp.logical_not(already_collected))
        
        # Add scores for collected pellets
        pellet_score = jnp.where(dot_collected, self.consts.PELLET_DOT_SCORE, 0)
        pellet_score = jnp.where(power_pellet_collected, self.consts.PELLET_POWER_SCORE, pellet_score)
        new_score = new_score + pellet_score
        
        # Update dots remaining
        new_dots_remaining = jnp.where(dot_collected, state.dots_remaining - 1, state.dots_remaining)
        
        # Mark pellet as collected
        pellet_collected = jnp.logical_or(dot_collected, power_pellet_collected)
        new_pellets_collected = jnp.where(
            pellet_collected,
            state.pellets_collected.at[player_tile_y, player_tile_x].set(1),
            state.pellets_collected
        )
        
        # Activate frightened mode if power pellet eaten
        power_pellet_eaten = power_pellet_collected
        new_frightened_timer = jnp.where(
            power_pellet_eaten,
            self.consts.FRIGHTENED_DURATION,
            state.frightened_timer
        )
        
        # Set all ghosts to frightened if power pellet eaten
        new_ghost_states_final = jnp.where(
            power_pellet_eaten,
            jnp.where(new_ghost_states == 2, 2, 1),  # Keep eaten ghosts as eaten, others to frightened
            new_ghost_states
        )
        new_ghosts_final = new_ghosts.at[:, 3].set(new_ghost_states_final)
        
        return state._replace(
            ghosts=new_ghosts_final,
            score=new_score,
            lives=final_lives,
            frightened_timer=new_frightened_timer,
            dots_remaining=new_dots_remaining,
            pellets_collected=new_pellets_collected,
        )

    def _update_timers(self, state: PacmanState) -> PacmanState:
        """Update frightened and scatter/chase timers."""
        # Update frightened timer
        new_frightened = jnp.maximum(state.frightened_timer - 1, 0)
        
        # If frightened timer expired, return frightened ghosts to normal (but not eaten ghosts)
        def update_ghost_state(ghost_data):
            gx, gy, gdir, gstate, gtx, gty = ghost_data
            # If frightened and timer expired, return to normal
            new_state = jnp.where(
                jnp.logical_and(gstate == 1, new_frightened == 0),
                0,  # normal
                gstate
            )
            return jnp.array([gx, gy, gdir, new_state, gtx, gty], dtype=jnp.int32)
        
        new_ghosts = jax.vmap(update_ghost_state)(state.ghosts)
        
        # Update scatter/chase timer
        new_scatter_chase = state.scatter_chase_timer + 1
        cycle_length = self.consts.SCATTER_DURATION + self.consts.CHASE_DURATION
        
        # Toggle mode when timer exceeds current phase
        new_is_scatter = jnp.where(
            new_scatter_chase % cycle_length < self.consts.SCATTER_DURATION,
            True,
            False
        )
        
        return state._replace(
            ghosts=new_ghosts,
            frightened_timer=new_frightened,
            scatter_chase_timer=new_scatter_chase,
            is_scatter_mode=new_is_scatter,
        )

    def _get_observation(self, state: PacmanState) -> PacmanObservation:
        """Convert state to observation."""
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0], dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_SIZE[1], dtype=jnp.int32),
            active=jnp.array(1, dtype=jnp.int32),
        )
        
        # Ghosts observation: [x, y, width, height, state] for each
        ghosts_obs = jnp.zeros((4, 5), dtype=jnp.int32)
        for i in range(4):
            ghosts_obs = ghosts_obs.at[i, 0].set(state.ghosts[i, 0])  # x
            ghosts_obs = ghosts_obs.at[i, 1].set(state.ghosts[i, 1])  # y
            ghosts_obs = ghosts_obs.at[i, 2].set(self.consts.GHOST_SIZE[0])  # width
            ghosts_obs = ghosts_obs.at[i, 3].set(self.consts.GHOST_SIZE[1])  # height
            ghosts_obs = ghosts_obs.at[i, 4].set(state.ghosts[i, 3])  # state
        
        return PacmanObservation(
            player=player,
            ghosts=ghosts_obs,
            dots_remaining=state.dots_remaining,
            power_pellets_active=state.power_pellets_active,
            score=state.score,
            lives=state.lives,
            level=state.level,
            frightened_timer=state.frightened_timer,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: PacmanObservation) -> jnp.ndarray:
        """Convert observation to flat array."""
        return jnp.concatenate([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.width.flatten(),
            obs.player.height.flatten(),
            obs.player.active.flatten(),
            obs.ghosts.flatten(),
            obs.dots_remaining.flatten(),
            obs.power_pellets_active.flatten(),
            obs.score.flatten(),
            obs.lives.flatten(),
            obs.level.flatten(),
            obs.frightened_timer.flatten(),
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(5)

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "ghosts": spaces.Box(low=0, high=210, shape=(4, 5), dtype=jnp.int32),
            "dots_remaining": spaces.Box(low=0, high=240, shape=(), dtype=jnp.int32),
            "power_pellets_active": spaces.Box(low=0, high=15, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=1, high=255, shape=(), dtype=jnp.int32),
            "frightened_timer": spaces.Box(low=0, high=200, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )
    
    def render(self, state: PacmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PacmanState) -> PacmanInfo:
        return PacmanInfo(
            step_counter=state.step_counter,
            level=state.level,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PacmanState, state: PacmanState) -> float:
        """Reward is score difference."""
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PacmanState) -> bool:
        """Game over when no lives remaining or all dots collected."""
        return jnp.logical_or(
            state.lives <= 0,
            state.dots_remaining <= 0
        )


class PacmanRenderer(JAXGameRenderer):
    def __init__(self, consts: PacmanConstants = None):
        super().__init__(consts)
        self.consts = consts or PacmanConstants()
        # Initialize maze if not already done
        if self.consts.MAZE_LAYOUT is None:
            self.consts = self.consts._replace(MAZE_LAYOUT=self._create_default_maze())
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Load assets
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/pacman"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(self.consts.ASSET_CONFIG, sprite_path)

        # Pre-render static maze background (walls and ghost house)
        self.maze_background = self._create_maze_background()

    def _create_default_maze(self) -> jnp.ndarray:
        """Create a classic Pacman maze layout (same as in JaxPacman)."""
        maze = np.zeros((31, 28), dtype=np.int32)
        
        maze_layout = [
            "1111111111111111111111111111",
            "1222222222222112222222222221",
            "1211112111112112111112111121",
            "1311112111112112111112111131",
            "1211112111112112111112111121",
            "1222222222222222222222222221",
            "1211112112111111112112111121",
            "1211112112111111112112111121",
            "1222222112222112222112222221",
            "1111112111110110111112111111",
            "1111112111110110111112111111",
            "1111112110000000001112111111",
            "1111112110111441110112111111",
            "1111112110144441110112111111",
            "0000002000144441000002000000",
            "1111112110144441110112111111",
            "1111112110111111110112111111",
            "1111112110000000001112111111",
            "1111112110111111110112111111",
            "1222222222222112222222222221",
            "1211112111112112111112111121",
            "1211112111112112111112111121",
            "1322112222222222222222112231",
            "1112112112111111112112112111",
            "1112112112111111112112112111",
            "1222222112222112222112222221",
            "1211111111112112111111111121",
            "1211111111112112111111111121",
            "1222222222222222222222222221",
            "1111111111111111111111111111",
        ]
        
        for row_idx, row_str in enumerate(maze_layout):
            for col_idx, char in enumerate(row_str):
                maze[row_idx, col_idx] = int(char)
        
        return jnp.array(maze, dtype=jnp.int32)
    
    def _create_maze_background(self) -> jnp.ndarray:
        """Create static background with maze walls."""
        import numpy as np
        bg = np.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=np.uint8)
        
        for row in range(self.consts.MAZE_HEIGHT):
            for col in range(self.consts.MAZE_WIDTH):
                tile_val = int(self.consts.MAZE_LAYOUT[row, col])
                if tile_val == 1:  # Wall
                    x, y = col * self.consts.TILE_SIZE, row * self.consts.TILE_SIZE
                    for dy in range(self.consts.TILE_SIZE):
                        for dx in range(self.consts.TILE_SIZE):
                            px, py = x + dx, y + dy
                            if px < self.consts.WIDTH and py < self.consts.HEIGHT:
                                bg[py, px] = [33, 33, 222]  # Blue walls
        
        return jnp.array(bg, dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # Init raster
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Render maze elements
        wall_mask = self.SHAPE_MASKS["wall"]
        pellet_dot_mask = self.SHAPE_MASKS["pellet_dot"]
        pellet_power_mask = self.SHAPE_MASKS["pellet_power"]
        
        # Draw tiles
        def render_tile(i, raster_state):
            row = i // self.consts.MAZE_WIDTH
            col = i % self.consts.MAZE_WIDTH
            tile_val = self.consts.MAZE_LAYOUT[row, col]
            x = col * self.consts.TILE_SIZE
            y = row * self.consts.TILE_SIZE
            
            # Draw wall
            raster_state = jax.lax.cond(
                tile_val == 1,
                lambda r: self.jr.render_at(r, x, y, wall_mask),
                lambda r: r,
                raster_state
            )
            
            # Draw dot
            is_dot = jnp.logical_and(tile_val == 2, state.pellets_collected[row, col] == 0)
            raster_state = jax.lax.cond(
                is_dot,
                lambda r: self.jr.render_at(r, x, y, pellet_dot_mask),
                lambda r: r,
                raster_state
            )
            
            # Draw power pellet  
            is_power = jnp.logical_and(tile_val == 3, state.pellets_collected[row, col] == 0)
            raster_state = jax.lax.cond(
                is_power,
                lambda r: self.jr.render_at(r, x, y, pellet_power_mask),
                lambda r: r,
                raster_state
            )
            
            return raster_state
        
        # Draw all tiles
        num_tiles = self.consts.MAZE_HEIGHT * self.consts.MAZE_WIDTH
        raster = jax.lax.fori_loop(0, num_tiles, render_tile, raster)
        
        # Draw player
        player_dir_idx = state.player_direction
        player_frame = state.player_animation_frame
        # Player sprite indices: 0-1=right, 2-3=left, 4-5=up, 6-7=down
        player_sprite_idx = player_dir_idx * 2 + player_frame
        player_mask = self.SHAPE_MASKS["player"][player_sprite_idx]
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)
        
        
        # Draw ghosts
        frightened_frame = (state.step_counter // 10) % 2
        
        # Ghost 0 - Blinky
        g0 = state.ghosts[0]
        g0_mask = jax.lax.switch(
            g0[3].astype(jnp.int32),
            [
                lambda: self.SHAPE_MASKS["ghost_blinky"][g0[2].astype(jnp.int32)],
                lambda: self.SHAPE_MASKS["ghost_frightened"][frightened_frame],
                lambda: self.SHAPE_MASKS["ghost_eyes"][g0[2].astype(jnp.int32)]
            ]
        )
        raster = self.jr.render_at(raster, g0[0], g0[1], g0_mask)
        
        # Ghost 1 - Pinky
        g1 = state.ghosts[1]
        g1_mask = jax.lax.switch(
            g1[3].astype(jnp.int32),
            [
                lambda: self.SHAPE_MASKS["ghost_pinky"][g1[2].astype(jnp.int32)],
                lambda: self.SHAPE_MASKS["ghost_frightened"][frightened_frame],
                lambda: self.SHAPE_MASKS["ghost_eyes"][g1[2].astype(jnp.int32)]
            ]
        )
        raster = self.jr.render_at(raster, g1[0], g1[1], g1_mask)
        
        # Ghost 2 - Inky
        g2 = state.ghosts[2]
        g2_mask = jax.lax.switch(
            g2[3].astype(jnp.int32),
            [
                lambda: self.SHAPE_MASKS["ghost_inky"][g2[2].astype(jnp.int32)],
                lambda: self.SHAPE_MASKS["ghost_frightened"][frightened_frame],
                lambda: self.SHAPE_MASKS["ghost_eyes"][g2[2].astype(jnp.int32)]
            ]
        )
        raster = self.jr.render_at(raster, g2[0], g2[1], g2_mask)
        
        # Ghost 3 - Clyde
        g3 = state.ghosts[3]
        g3_mask = jax.lax.switch(
            g3[3].astype(jnp.int32),
            [
                lambda: self.SHAPE_MASKS["ghost_clyde"][g3[2].astype(jnp.int32)],
                lambda: self.SHAPE_MASKS["ghost_frightened"][frightened_frame],
                lambda: self.SHAPE_MASKS["ghost_eyes"][g3[2].astype(jnp.int32)]
            ]
        )
        raster = self.jr.render_at(raster, g3[0], g3[1], g3_mask)
        

        
        # Render score
        score_digits = self.jr.int_to_digits(state.score, max_digits=6)
        score_digit_masks = self.SHAPE_MASKS["digits"]
        raster = self.jr.render_label_selective(
            raster, 10, 5, score_digits, score_digit_masks, 0, 6, spacing=8
        )
        
        return self.jr.render_from_palette(raster, self.PALETTE)

