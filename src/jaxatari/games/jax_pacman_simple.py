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
from jaxatari.games.pacmanMaps.nodes import NodeGroup, Node


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Pacman (simplified - player and walls only).
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
        {'name': 'wall', 'type': 'single', 'file': 'wall/0.npy'},
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
    PLAYER_SPEED: int = 1  # pixels per step (smooth movement speed) - can be reduced for slower movement
    PLAYER_START_X: int = 76  # Center of maze
    PLAYER_START_Y: int = 188  # Bottom area
    
    # Colors
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)  # Black
    WALL_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Blue
    
    # Asset config
    ASSET_CONFIG: tuple = _get_default_asset_config()


class PacmanState(NamedTuple):
    # Player state
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array  # Action enum value (Action.UP, Action.DOWN, etc.)
    player_next_direction: chex.Array  # Queued direction for cornering
    player_animation_frame: chex.Array  # 0 or 1 for mouth open/close
    current_node_index: chex.Array  # Index in nodeList for current node
    target_node_index: chex.Array  # Index in nodeList for target node (node being moved towards)
    
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


class PacmanInfo(NamedTuple):
    step_counter: jnp.ndarray


class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo, PacmanConstants]):
    def __init__(self, consts: PacmanConstants = None):
        consts = consts or PacmanConstants()
        super().__init__(consts)
        
        # Load NodeList from maze file (or use default if file doesn't exist)
        maze_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pacmanMaps", "maze1.txt"
        )
        # Check if file exists, otherwise use None for default map
        if not os.path.exists(maze_file_path):
            maze_file_path = None
        self.node_group = NodeGroup.from_maze_file(maze_file_path, tile_size=self.consts.TILE_SIZE)
        
        # Load maze data for rendering (convert to numeric: 0=path/node, 1=wall)
        if maze_file_path is not None:
            maze_data = np.loadtxt(maze_file_path, dtype='<U1')
            maze_numeric = np.zeros_like(maze_data, dtype=np.int32)
            maze_numeric[maze_data == 'X'] = 1  # X = wall
            self.maze_data_np = maze_numeric  # Keep as numpy for Python-level rendering
            self.maze_data = jnp.array(maze_numeric, dtype=jnp.int32)  # JAX array version (if needed)
            self.maze_height, self.maze_width = maze_data.shape
            
            # Pre-compute wall positions as JAX array for JIT rendering
            wall_positions = []
            for row in range(self.maze_height):
                for col in range(self.maze_width):
                    if maze_numeric[row, col] == 1:  # Wall
                        x = col * self.consts.TILE_SIZE
                        y = row * self.consts.TILE_SIZE
                        wall_positions.append([x, y])
            
            if wall_positions:
                self.wall_positions = jnp.array(wall_positions, dtype=jnp.int32)
            else:
                self.wall_positions = jnp.zeros((0, 2), dtype=jnp.int32)
        else:
            # Default map - no walls
            self.maze_data_np = None
            self.maze_data = None
            self.maze_height = 0
            self.maze_width = 0
            self.wall_positions = jnp.zeros((0, 2), dtype=jnp.int32)
        
        # Pre-compute node positions for JIT-compatible movement
        num_nodes = len(self.node_group.nodeList)
        self.node_positions_x = jnp.array([node.position.x for node in self.node_group.nodeList], dtype=jnp.int32)
        self.node_positions_y = jnp.array([node.position.y for node in self.node_group.nodeList], dtype=jnp.int32)
        
        # Each node.neighbor_indices is a JAX array: neighbor_indices[action] = node_index (-1 if no neighbor)
        # Stack all node neighbor_indices into a 2D array for easy indexing: [node_idx][action] -> neighbor_idx
        neighbor_arrays = [node.neighbor_indices for node in self.node_group.nodeList]
        self.neighbor_lookup = jnp.stack(neighbor_arrays)  # Shape: (num_nodes, 18)
        
        self.consts = consts
        
        self.renderer = PacmanRenderer(self.consts, wall_positions=self.wall_positions)
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[PacmanObservation, PacmanState]:
        state_key, _ = jax.random.split(key)
        
        # Initialize player at first node
        start_node = self.node_group.nodeList[0]
        player_x = jnp.array(start_node.position.x, dtype=jnp.int32)
        player_y = jnp.array(start_node.position.y, dtype=jnp.int32)
        player_direction = jnp.array(Action.NOOP, dtype=jnp.int32)
        player_next_direction = jnp.array(Action.NOOP, dtype=jnp.int32)
        player_animation_frame = jnp.array(0, dtype=jnp.int32)
        current_node_index = jnp.array(0, dtype=jnp.int32)  # Start at first node
        target_node_index = jnp.array(0, dtype=jnp.int32)  # Initially target is same as current
        
        # Create state
        state = PacmanState(
            player_x=player_x,
            player_y=player_y,
            player_direction=player_direction,
            player_next_direction=player_next_direction,
            player_animation_frame=player_animation_frame,
            current_node_index=current_node_index,
            target_node_index=target_node_index,
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=state_key,
        )
        
        initial_obs = self._get_observation(state)
        return initial_obs, state

    def step(self, state: PacmanState, action: chex.Array) -> Tuple[PacmanObservation, PacmanState, float, bool, PacmanInfo]:
        new_state_key, step_key = jax.random.split(state.key)
        previous_state = state
        
        # Update state key
        state = state._replace(key=step_key)
        
        # Update player movement (node-to-node jumping)
        state = self._player_step(state, action)
        
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
        """
        Handle player movement - smooth movement between nodes.
        Translated from pacman.py: update() logic with smooth movement.
        Uses neighbors already computed in nodes.py (via NodeGroup.from_maze_file).
        Key behavior: When reaching a node, stop (set direction to NOOP) and wait for new action.
        """
        # Check if this is a NOOP action
        is_noop = action == Action.NOOP
        
        # Get current and target node positions
        current_node_idx = state.current_node_index
        target_node_idx = state.target_node_index
        
        current_node_x = self.node_positions_x[current_node_idx]
        current_node_y = self.node_positions_y[current_node_idx]
        target_node_x = self.node_positions_x[target_node_idx]
        target_node_y = self.node_positions_y[target_node_idx]
        
        # Check if we're currently at a node (position matches current node position)
        at_current_node = jnp.logical_and(
            state.player_x == current_node_x,
            state.player_y == current_node_y
        )
        
        # Check if we've reached/passed the target node (overshot check)
        # Calculate direction vector from current node to target node
        dx_to_target = target_node_x - current_node_x
        dy_to_target = target_node_y - current_node_y
        vec_to_target_sq = dx_to_target * dx_to_target + dy_to_target * dy_to_target
        
        # Calculate distance from current node to current position
        vec_to_self_sq = (state.player_x - current_node_x) * (state.player_x - current_node_x) + (state.player_y - current_node_y) * (state.player_y - current_node_y)
        reached_target = vec_to_self_sq >= vec_to_target_sq
        
        # If we've reached the target node and we're moving, stop at it
        is_moving = state.player_direction != Action.NOOP
        should_stop_at_target = jnp.logical_and(is_moving, reached_target)
        
        # Update current node if we reached target
        new_current_idx = jnp.where(should_stop_at_target, target_node_idx, current_node_idx)
        
        # If we stopped at target, snap position to target node
        snapped_x = jnp.where(should_stop_at_target, target_node_x, state.player_x)
        snapped_y = jnp.where(should_stop_at_target, target_node_y, state.player_y)
        
        # After stopping, we're at the new current node
        at_node_after_stop = jnp.logical_or(at_current_node, should_stop_at_target)
        
        # Get new target based on action (equivalent to getNewTarget() in pacman.py)
        # Only check action if we're at a node (either was already at node, or just reached target)
        new_target_from_action = self.neighbor_lookup[new_current_idx, action]
        has_new_target = new_target_from_action >= 0
        valid_action = jnp.logical_and(jnp.logical_not(is_noop), has_new_target)
        
        # Only start moving if we're at a node AND have a valid action
        # This ensures we stop at each node and wait for a new action
        should_start_moving = jnp.logical_and(at_node_after_stop, valid_action)
        
        # Update target: if we should start moving, use action target; 
        # if we stopped, target becomes same as current (we're at the node);
        # otherwise keep current target (still moving towards it)
        new_target_idx = jnp.where(should_start_moving, new_target_from_action,
                          jnp.where(should_stop_at_target, new_current_idx, target_node_idx))
        
        # Update direction: 
        # - if we should start moving (at node + valid action), use action
        # - if we stopped at target, use NOOP (stop moving)
        # - otherwise keep current direction (continue moving)
        new_direction = jnp.where(should_start_moving, action,
                         jnp.where(should_stop_at_target, jnp.array(Action.NOOP, dtype=jnp.int32),
                                  state.player_direction))
        
        # Update target node position for movement calculation
        final_target_x = self.node_positions_x[new_target_idx]
        final_target_y = self.node_positions_y[new_target_idx]
        
        # Calculate movement direction based on new direction
        # Action values: Action.UP=2, Action.DOWN=5, Action.LEFT=4, Action.RIGHT=3
        # Movement deltas: RIGHT=+x, LEFT=-x, DOWN=+y, UP=-y
        move_dx = jnp.where(new_direction == Action.RIGHT, self.consts.PLAYER_SPEED,
                   jnp.where(new_direction == Action.LEFT, -self.consts.PLAYER_SPEED, 0))
        move_dy = jnp.where(new_direction == Action.DOWN, self.consts.PLAYER_SPEED,
                   jnp.where(new_direction == Action.UP, -self.consts.PLAYER_SPEED, 0))
        
        # Move incrementally towards target (only if moving, otherwise stay at snapped position)
        new_x = jnp.where(new_direction != Action.NOOP, snapped_x + move_dx, snapped_x)
        new_y = jnp.where(new_direction != Action.NOOP, snapped_y + move_dy, snapped_y)
        
        return state._replace(
            player_x=new_x.astype(jnp.int32),
            player_y=new_y.astype(jnp.int32),
            player_direction=new_direction,
            player_next_direction=jnp.array(Action.NOOP, dtype=jnp.int32),
            current_node_index=new_current_idx,
            target_node_index=new_target_idx,
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
        
        return PacmanObservation(player=player)

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: PacmanObservation) -> jnp.ndarray:
        """Convert observation to flat array."""
        return jnp.concatenate([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.width.flatten(),
            obs.player.height.flatten(),
            obs.player.active.flatten(),
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
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PacmanState, state: PacmanState) -> float:
        """Simple reward - just return 0 for now."""
        return jnp.array(0.0, dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PacmanState) -> bool:
        """Game never ends in simplified version."""
        return jnp.array(False)


class PacmanRenderer(JAXGameRenderer):
    def __init__(self, consts: PacmanConstants = None, wall_positions=None):
        super().__init__(consts)
        self.consts = consts or PacmanConstants()
        # Wall positions for JIT-compatible rendering
        self.wall_positions = wall_positions if wall_positions is not None else jnp.zeros((0, 2), dtype=jnp.int32)
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
        
        # Pre-render static maze background with walls
        self.maze_background = self._create_maze_background()


    def _create_maze_background(self) -> jnp.ndarray:
        """Create static background with maze walls from wall positions."""
        import numpy as np
        # Start with the base background raster (palette-indexed, 2D)
        bg_raster = np.array(self.BACKGROUND, dtype=np.uint8).copy()
        
        # Render walls at each wall position using the wall mask
        wall_positions_np = np.array(self.wall_positions)
        wall_mask = np.array(self.SHAPE_MASKS["wall"], dtype=np.int32)
        wall_h, wall_w = wall_mask.shape[:2]
        
        # Get wall color ID from the mask (the mask contains color IDs, not RGB)
        wall_color_id = None
        for dy in range(wall_h):
            for dx in range(wall_w):
                if wall_mask[dy, dx] != 0:
                    wall_color_id = int(wall_mask[dy, dx])
                    break
            if wall_color_id is not None:
                break
        
        if wall_color_id is None:
            # Default to a blue color ID if we can't find one
            # Find an unused color ID or use a default
            wall_color_id = 1  # Assuming 1 is a wall color
        
        # Render each wall tile
        for i in range(wall_positions_np.shape[0]):
            x = int(wall_positions_np[i, 0])
            y = int(wall_positions_np[i, 1])
            # Render wall tile using the mask
            for dy in range(wall_h):
                for dx in range(wall_w):
                    px, py = x + dx, y + dy
                    if px < self.consts.WIDTH and py < self.consts.HEIGHT:
                        mask_val = int(wall_mask[dy, dx])
                        if mask_val != 0:  # Non-transparent pixel
                            bg_raster[py, px] = mask_val
        
        return jnp.array(bg_raster, dtype=jnp.uint8)
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # Start with pre-rendered maze background (includes walls)
        raster = self.maze_background
        
        # Draw player on top
        # Map Action enum to sprite direction index: Action.RIGHT=3->0, Action.LEFT=4->1, Action.UP=2->2, Action.DOWN=5->3
        player_dir_idx = jnp.where(
            state.player_direction == Action.RIGHT, 0,
            jnp.where(state.player_direction == Action.LEFT, 1,
            jnp.where(state.player_direction == Action.UP, 2,
            jnp.where(state.player_direction == Action.DOWN, 3, 0)))
        )
        player_frame = state.player_animation_frame
        # Player sprite indices: 0-1=right, 2-3=left, 4-5=up, 6-7=down
        player_sprite_idx = player_dir_idx * 2 + player_frame
        player_mask = self.SHAPE_MASKS["player"][player_sprite_idx]
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)
        
        return self.jr.render_from_palette(raster, self.PALETTE)

