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
    Default asset configuration for Pacman.
    Uses a tuple of dictionaries for immutability and JAX compatibility.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'group', 'files': [
            'player/player_right_1.npy', 'player/player_right_2.npy',
            'player/player_left_1.npy', 'player/player_left_2.npy',
            'player/player_up_1.npy', 'player/player_up_2.npy',
            'player/player_down_1.npy', 'player/player_down_2.npy',
        ]},
        {'name': 'player_death', 'type': 'group', 'files': [
            'player/death_0.npy', 'player/death_1.npy', 'player/death_2.npy', 'player/death_3.npy',
            'player/death_4.npy', 'player/death_5.npy', 'player/death_6.npy', 'player/death_7.npy',
            'player/death_8.npy', 'player/death_9.npy', 'player/death_10.npy', 'player/death_11.npy',
        ]},
        {'name': 'ghost_blinky', 'type': 'group', 'files': [
            'ghost_blinky/ghost_right_1.npy', 'ghost_blinky/ghost_right_2.npy',
            'ghost_blinky/ghost_left_1.npy', 'ghost_blinky/ghost_left_2.npy',
            'ghost_blinky/ghost_up_1.npy', 'ghost_blinky/ghost_up_2.npy',
            'ghost_blinky/ghost_down_1.npy', 'ghost_blinky/ghost_down_2.npy',
        ]},
        {'name': 'ghost_pinky', 'type': 'group', 'files': [
            'ghost_pinky/ghost_right_1.npy', 'ghost_pinky/ghost_right_2.npy',
            'ghost_pinky/ghost_left_1.npy', 'ghost_pinky/ghost_left_2.npy',
            'ghost_pinky/ghost_up_1.npy', 'ghost_pinky/ghost_up_2.npy',
            'ghost_pinky/ghost_down_1.npy', 'ghost_pinky/ghost_down_2.npy',
        ]},
        {'name': 'ghost_inky', 'type': 'group', 'files': [
            'ghost_inky/ghost_right_1.npy', 'ghost_inky/ghost_right_2.npy',
            'ghost_inky/ghost_left_1.npy', 'ghost_inky/ghost_left_2.npy',
            'ghost_inky/ghost_up_1.npy', 'ghost_inky/ghost_up_2.npy',
            'ghost_inky/ghost_down_1.npy', 'ghost_inky/ghost_down_2.npy',
        ]},
        {'name': 'ghost_clyde', 'type': 'group', 'files': [
            'ghost_clyde/ghost_right_1.npy', 'ghost_clyde/ghost_right_2.npy',
            'ghost_clyde/ghost_left_1.npy', 'ghost_clyde/ghost_left_2.npy',
            'ghost_clyde/ghost_up_1.npy', 'ghost_clyde/ghost_up_2.npy',
            'ghost_clyde/ghost_down_1.npy', 'ghost_clyde/ghost_down_2.npy',
        ]},
        {'name': 'ghost_frightened', 'type': 'group', 'files': [
            'ghost_frightened/ghost_frightened_1.npy', 'ghost_frightened/ghost_frightened_2.npy',
            'ghost_frightened/ghost_frightened_white_1.npy', 'ghost_frightened/ghost_frightened_white_2.npy',
        ]},
        {'name': 'ghost_eyes', 'type': 'group', 'files': [
            'ghost_eyes/eyes_right.npy', 'ghost_eyes/eyes_left.npy',
            'ghost_eyes/eyes_up.npy', 'ghost_eyes/eyes_down.npy',
        ]},
        {'name': 'pellet_dot', 'type': 'single', 'file': 'pellet_dot.npy'},
        {'name': 'pellet_power', 'type': 'group', 'files': [
            'pellet_power/pellet_power_on.npy',
            'pellet_power/pellet_power_off.npy'
        ]},
        {'name': 'wall', 'type': 'group', 'files': [
            'wall/wall_0.npy',   
            'wall/wall_1.npy',   
            'wall/wall_2.npy',   
            'wall/wall_3.npy',   
            'wall/wall_4.npy',   
            'wall/wall_5.npy',   
            'wall/wall_6.npy',   
            'wall/wall_7.npy',   
            'wall/wall_8.npy',   
            'wall/wall_9.npy',   
            'wall/wall_10.npy',  
            'wall/wall_11.npy',  
            'wall/wall_12.npy',  
            'wall/wall_13.npy',  
            'wall/wall_14.npy',  
            'wall/wall_15.npy',  
        ]},
        {'name': 'ghost_door', 'type': 'single', 'file': 'ghost_door.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/digit_{}.npy'},
    )


def _get_sprite_lookup() -> chex.Array:
    """
    Creates lookup table for mapping Actions to Sprites.
    """
    # Define sprite offsets
    SPRITE_RIGHT = 0
    SPRITE_LEFT = 2
    SPRITE_UP = 4
    SPRITE_DOWN = 6

    # Create lookup table for mapping Actions to Sprites
    lookup = np.zeros(18, dtype=np.int32)
    
    # Default to Right facing
    lookup[:] = SPRITE_RIGHT
    
    # Map Actions to Sprites
    lookup[Action.UP] = SPRITE_UP
    lookup[Action.DOWN] = SPRITE_DOWN
    lookup[Action.LEFT] = SPRITE_LEFT
    lookup[Action.RIGHT] = SPRITE_RIGHT
    
    # Ghost direction mapping (Action -> Sprite Offset)
    ghost_lookup = np.zeros(18, dtype=np.int32)
    ghost_lookup[:] = 0 # Default to Right
    ghost_lookup[Action.RIGHT] = 0
    ghost_lookup[Action.LEFT] = 2
    ghost_lookup[Action.UP] = 4
    ghost_lookup[Action.DOWN] = 6
    return jnp.array(lookup, dtype=jnp.int32), jnp.array(ghost_lookup, dtype=jnp.int32)


class PacmanConstants(NamedTuple):
    # Screen dimensions (Atari 2600 Pacman uses 224x288)
    WIDTH: int = 224
    HEIGHT: int = 288  # 36 tiles * 8 pixels
    
    # Tile size for maze (8x8 pixels per tile)
    TILE_SIZE: int = 8
    ANIMATION_SPEED: int = 5 # Frames per animation step
    
    # Maze dimensions in tiles (224/8 = 28, 288/8 = 36)
    MAZE_WIDTH: int = 28  # 28 tiles wide
    MAZE_HEIGHT: int = 36  # 36 tiles tall
    
    # Player constants
    PLAYER_SIZE: Tuple[int, int] = (8, 8)
    PLAYER_SPEED: int = 1  # pixels per step
    PLAYER_START_X: int = 112  # Center of maze (224/2 = 112)
    PLAYER_START_Y: int = 280  # Bottom area (near bottom of 288 height)
    
    # Ghost constants
    GHOST_SIZE: Tuple[int, int] = (8, 8)
    GHOST_SPEED_NORMAL: int = 1
    GHOST_SPEED_FRIGHTENED: int = 1
    GHOST_SPEED_EATEN: int = 2
    
    # Ghost starting positions (in ghost house area)
    GHOST_START_X: int = 112  # Center of maze (224/2 = 112)
    GHOST_START_Y: int = 144  # Adjusted for new map size (approximately center vertically)
    
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
    LEVEL_TRANSITION_DURATION: int = 60 # frames (1 second at 60fps)
    FREEZE_DURATION: int = 15 # frames (0.25 second)
    DEATH_DURATION: int = 60 # frames (1 second)
    
    # Colors
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)  # Black
    WALL_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Blue
    PELLET_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Yellow
    SCORE_COLOR: Tuple[int, int, int] = (255, 255, 255)  # White
    
    # Maze layout grid where each cell represents a tile type.
    # 0: Empty path
    # 1: Wall
    # 2: Dot
    # 3: Power Pellet
    # 4: Ghost House
    MAZE_LAYOUT: chex.Array = None
    
    # Asset config
    ASSET_CONFIG: tuple = _get_default_asset_config()
    SPRITE_LOOKUP: Tuple[chex.Array, chex.Array] = _get_sprite_lookup()


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
    player_current_node_index: chex.Array  # Current node index for node-based movement
    player_target_node_index: chex.Array  # Target node index for node-based movement
    
    # Ghost states (4 ghosts)
    ghosts: chex.Array  # Shape: (4, 8) - [x, y, direction, state, target_x, target_y, current_node, target_node]
    
    # Pellet states - simplified: track number of dots remaining
    dots_remaining: chex.Array
    power_pellets_active: chex.Array  # Bitmask for 4 power pellets
    
    # Game state
    score: chex.Array
    lives: chex.Array
    level: chex.Array
    pellets_collected: chex.Array  # 25x20 mask: 0=not collected, 1=collected
    
    # Timers
    frightened_timer: chex.Array
    ghosts_eaten_count: chex.Array  # Tracks ghosts eaten during current power pellet
    scatter_chase_timer: chex.Array
    is_scatter_mode: chex.Array  # True for scatter, False for chase
    
    # Level transition
    level_transition_timer: chex.Array
    
    # Freeze and Death
    freeze_timer: chex.Array
    death_timer: chex.Array
    player_state: chex.Array # 0=Alive, 1=Dying
    
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
    level_transition_timer: jnp.ndarray
    player_state: jnp.ndarray


class PacmanInfo(NamedTuple):
    step_counter: jnp.ndarray
    level: jnp.ndarray


class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo, PacmanConstants]):
    def __init__(self, consts: PacmanConstants = None):
        consts = consts or PacmanConstants()
        super().__init__(consts)
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]
        
        # Determine maze file path once (single source of truth)
        from jaxatari.games.pacmanMaps.nodes import NodeGroup
        maze_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pacmanMaps", "maze1.txt"
        )
        
        maze_file_pellet_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pacmanMaps", "maze1_pellet.txt"
        )
        
        # Check if maze file exists, raise error if not found
        if not os.path.exists(maze_file_path):
            raise FileNotFoundError(f"Maze file not found: {maze_file_path}")
        
        # Initialize maze layout if not provided (load from file)
        if consts.MAZE_LAYOUT is None:
            self.consts = consts._replace(MAZE_LAYOUT=self._load_maze_from_file(maze_file_pellet_path))
        else:
            self.consts = consts
        
        # Create renderer after maze layout is loaded (so it can create maze_background correctly)
        self.renderer = PacmanRenderer(self.consts)
        
        # Load NodeGroup using the same maze file path
        self.node_group = NodeGroup.from_maze_file(maze_file_path, tile_size=self.consts.TILE_SIZE)
        
        # Pre-compute node positions for JIT-compatible movement
        self.node_positions_x = jnp.array([node.position.x for node in self.node_group.nodeList], dtype=jnp.int32)
        self.node_positions_y = jnp.array([node.position.y for node in self.node_group.nodeList], dtype=jnp.int32)
        
        # Pre-compute neighbor lookup: neighbor_lookup[node_idx][action] -> next_node_idx
        neighbor_arrays = [node.neighbor_indices for node in self.node_group.nodeList]
        self.neighbor_lookup = jnp.stack(neighbor_arrays)  # Shape: (num_nodes, 18)
        
        # Find ghost house node (look for tile value 4)
        # Default to center if not found
        center_x = (self.consts.MAZE_WIDTH * self.consts.TILE_SIZE) // 2
        center_y = (self.consts.MAZE_HEIGHT * self.consts.TILE_SIZE) // 2
        self.ghost_house_node_idx = self._find_nearest_node_idx(center_x, center_y)
        
        # Try to find a node that is actually inside the ghost house (tile 4)
        for i, node in enumerate(self.node_group.nodeList):
            tx = int(node.position.x) // self.consts.TILE_SIZE
            ty = int(node.position.y) // self.consts.TILE_SIZE
            if tx < self.consts.MAZE_WIDTH and ty < self.consts.MAZE_HEIGHT:
                if self.consts.MAZE_LAYOUT[ty, tx] == 4:
                    self.ghost_house_node_idx = jnp.array(i, dtype=jnp.int32)
                    break

        # Pre-compute which edges correspond to crossing the ghost door (or entering ghost house)
        num_nodes, num_actions = self.neighbor_lookup.shape
        door_edge_mask = np.zeros((num_nodes, num_actions), dtype=np.bool_)
        ghost_entry_mask = np.zeros((num_nodes, num_actions), dtype=np.bool_)

        # Helper to get tile type at a node
        def tile_type_for_node(node):
            tx = int(node.position.x) // self.consts.TILE_SIZE
            ty = int(node.position.y) // self.consts.TILE_SIZE
            if 0 <= tx < self.consts.MAZE_WIDTH and 0 <= ty < self.consts.MAZE_HEIGHT:
                return int(self.consts.MAZE_LAYOUT[ty, tx])
            return 1  # treat out-of-bounds as wall

        node_tile_types = [tile_type_for_node(node) for node in self.node_group.nodeList]

        for n in range(num_nodes):
            tile_n = node_tile_types[n]
            for a in range(num_actions):
                nb = int(self.neighbor_lookup[n, a])
                if nb < 0:
                    continue
                tile_nb = node_tile_types[nb]

                # Option 1: treat any edge that touches a door node as a "door edge"
                if tile_n == 5 or tile_nb == 5:
                    door_edge_mask[n, a] = True
                
                # Option 2: Ghost entry mask (Block entry to Door/House from Outside)
                is_house_complex_n = (tile_n == 4 or tile_n == 5)
                is_house_complex_nb = (tile_nb == 4 or tile_nb == 5)
                
                if is_house_complex_nb and not is_house_complex_n:
                     # Attempting to enter House/Door from Outside -> BLOCK for non-eaten ghosts
                     ghost_entry_mask[n, a] = True

        # Store mask as JAX array; used only for Pacman
        self.player_door_edge_mask = jnp.array(door_edge_mask, dtype=jnp.bool_)
        # Store ghost one-way mask
        self.ghost_entry_mask = jnp.array(ghost_entry_mask, dtype=jnp.bool_)

    def _load_maze_from_file(self, maze_file_path: str) -> jnp.ndarray:
        """
        Loads maze layout from the provided maze file path.
        
        Parses text file into numerical grid:
        0: Empty path
        1: Wall (X)
        2: Dot (.) or Regular Node (o)
        3: Power Pellet (+)
        4: Ghost House (H)
        
        Args:
            maze_file_path: Path to maze file (must exist)
        """
        maze = np.zeros((self.consts.MAZE_HEIGHT, self.consts.MAZE_WIDTH), dtype=np.int32)
        with open(maze_file_path, 'r') as f:
            lines = f.readlines()
            for row, line in enumerate(lines):
                if row >= self.consts.MAZE_HEIGHT:
                    break
                # Remove spaces and newline
                line = line.strip().replace(' ', '')
                for col, char in enumerate(line):
                    if col >= self.consts.MAZE_WIDTH:
                        break
                    if char == 'X':
                        maze[row, col] = 1
                    elif char == '.':
                        maze[row, col] = 2
                    elif char == 'o':
                        maze[row, col] = 2
                    elif char == '+':
                        maze[row, col] = 3
                    elif char == 'H':
                        maze[row, col] = 4
                    elif char == 'D':
                        maze[row, col] = 5
                    else:
                        maze[row, col] = 0
        return jnp.array(maze, dtype=jnp.int32)

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[PacmanObservation, PacmanState]:
        state_key, _ = jax.random.split(key)
        
        # Initialize player at nearest node
        player_start_node_idx = self._find_nearest_node_idx(
            self.consts.PLAYER_START_X,
            self.consts.PLAYER_START_Y
        )
        player_x = jnp.array(self.node_positions_x[player_start_node_idx], dtype=jnp.int32)
        player_y = jnp.array(self.node_positions_y[player_start_node_idx], dtype=jnp.int32)
        player_direction = jnp.array(0, dtype=jnp.int32)  # Start facing right
        player_next_direction = jnp.array(-1, dtype=jnp.int32)
        player_animation_frame = jnp.array(0, dtype=jnp.int32)
        player_current_node_index = jnp.array(player_start_node_idx, dtype=jnp.int32)
        player_target_node_index = jnp.array(player_start_node_idx, dtype=jnp.int32)
        
        # Initialize ghosts (4 ghosts at starting positions)
        ghosts = jnp.zeros((4, 8), dtype=jnp.int32)
        for i in range(4):
            ghost_x = self.consts.GHOST_START_X + i * 8
            ghost_y = self.consts.GHOST_START_Y
            # Find nearest node for this ghost
            ghost_node_idx = self._find_nearest_node_idx(ghost_x, ghost_y)
            
            ghosts = ghosts.at[i, 0].set(self.node_positions_x[ghost_node_idx])  # x at node
            ghosts = ghosts.at[i, 1].set(self.node_positions_y[ghost_node_idx])  # y at node
            ghosts = ghosts.at[i, 2].set(0)  # direction (right)
            ghosts = ghosts.at[i, 3].set(0)  # state (normal)
            ghosts = ghosts.at[i, 4].set(self.node_positions_x[ghost_node_idx])  # target_x
            ghosts = ghosts.at[i, 5].set(self.node_positions_y[ghost_node_idx])  # target_y
            ghosts = ghosts.at[i, 6].set(ghost_node_idx)  # current_node
            ghosts = ghosts.at[i, 7].set(ghost_node_idx)  # target_node
        
        # Initial game state
        score = jnp.array(0, dtype=jnp.int32)
        lives = jnp.array(1, dtype=jnp.int32)
        level = jnp.array(1, dtype=jnp.int32)
        pellets_collected = jnp.zeros((self.consts.MAZE_HEIGHT, self.consts.MAZE_WIDTH), dtype=jnp.int32)
        
        dots_remaining = jnp.array(240, dtype=jnp.int32)  # Approximate
        frightened_timer = jnp.array(0, dtype=jnp.int32)
        scatter_chase_timer = jnp.array(0, dtype=jnp.int32)
        is_scatter_mode = jnp.array(True)  # Start in scatter mode
        
        freeze_timer = jnp.array(0, dtype=jnp.int32)
        death_timer = jnp.array(0, dtype=jnp.int32)
        player_state = jnp.array(0, dtype=jnp.int32) # Alive
        
        # Create state
        state = PacmanState(
            player_x=player_x,
            player_y=player_y,
            player_direction=player_direction,
            player_next_direction=player_next_direction,
            player_animation_frame=player_animation_frame,
            player_current_node_index=player_current_node_index,
            player_target_node_index=player_target_node_index,
            ghosts=ghosts,
            dots_remaining=dots_remaining,
            power_pellets_active=jnp.array(15, dtype=jnp.int32),
            score=score,
            lives=lives,
            level=level,
            pellets_collected=pellets_collected,
            frightened_timer=frightened_timer,
            ghosts_eaten_count=jnp.array(0, dtype=jnp.int32),
            scatter_chase_timer=scatter_chase_timer,
            is_scatter_mode=is_scatter_mode,
            level_transition_timer=jnp.array(0, dtype=jnp.int32),
            freeze_timer=freeze_timer,
            death_timer=death_timer,
            player_state=player_state,
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=state_key,
        )
        
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array) -> Tuple[PacmanObservation, PacmanState, float, bool, PacmanInfo]:
        new_state_key, step_key = jax.random.split(state.key)
        previous_state = state
        
        # Check if we are in level transition
        is_transitioning = state.level_transition_timer > 0
        
        # Check if we are frozen (ghost eaten)
        is_frozen = state.freeze_timer > 0
        
        # Check if player is dying
        is_dying = state.player_state == 1
        
        def transition_step(state):
            # Only update transition timer
            new_timer = state.level_transition_timer - 1
            state = state._replace(level_transition_timer=new_timer)
            
            # If timer hits 0, reset for new level
            state = jax.lax.cond(
                new_timer == 0,
                self._reset_for_new_level,
                lambda x: x,
                state
            )
            
            # Update key
            state = state._replace(key=new_state_key)
            
            # Return current state (paused)
            return self._get_observation(state), state, jnp.array(0.0, dtype=jnp.float32), self._get_done(state), self._get_info(state)

        def freeze_step(state):
            # Only update freeze timer
            new_timer = state.freeze_timer - 1
            state = state._replace(freeze_timer=new_timer)
            state = state._replace(key=new_state_key)
            return self._get_observation(state), state, jnp.array(0.0, dtype=jnp.float32), self._get_done(state), self._get_info(state)

        def death_step(state):
            # Update death timer
            new_timer = state.death_timer - 1
            state = state._replace(death_timer=new_timer)
            
            # If timer hits 0, reset after death (lives - 1, reset positions)
            state = jax.lax.cond(
                new_timer == 0,
                self._reset_after_death,
                lambda x: x,
                state
            )
            
            state = state._replace(key=new_state_key)
            return self._get_observation(state), state, jnp.array(0.0, dtype=jnp.float32), self._get_done(state), self._get_info(state)

        def normal_step(state):
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
                player_animation_frame=(state.step_counter // self.consts.ANIMATION_SPEED) % 2,
                step_counter=state.step_counter + 1,
                key=new_state_key,
            )
            
            done = self._get_done(state)
            reward = self._get_reward(previous_state, state)
            info = self._get_info(state)
            observation = self._get_observation(state)
            
            return observation, state, reward, done, info
            
        # Dispatch based on state priority: Transition > Death > Freeze > Normal
        
        return jax.lax.cond(
            is_transitioning,
            transition_step,
            lambda s: jax.lax.cond(
                is_dying,
                death_step,
                lambda s2: jax.lax.cond(
                    is_frozen,
                    freeze_step,
                    normal_step,
                    s2
                ),
                s
            ),
            state
        )

    def _reset_after_death(self, state: PacmanState) -> PacmanState:
        """Resets positions and decrements lives after death animation."""
        new_lives = state.lives - 1
        new_lives = jnp.maximum(new_lives, 0)
        
        # Reset positions (same logic as reset_for_new_level but keep pellets)
        player_start_node_idx = self._find_nearest_node_idx(
            self.consts.PLAYER_START_X,
            self.consts.PLAYER_START_Y
        )
        player_x = jnp.array(self.node_positions_x[player_start_node_idx], dtype=jnp.int32)
        player_y = jnp.array(self.node_positions_y[player_start_node_idx], dtype=jnp.int32)
        
        ghosts = jnp.zeros((4, 8), dtype=jnp.int32)
        for i in range(4):
            ghost_x = self.consts.GHOST_START_X + i * 8
            ghost_y = self.consts.GHOST_START_Y
            ghost_node_idx = self._find_nearest_node_idx(ghost_x, ghost_y)
            ghosts = ghosts.at[i, 0].set(self.node_positions_x[ghost_node_idx])
            ghosts = ghosts.at[i, 1].set(self.node_positions_y[ghost_node_idx])
            ghosts = ghosts.at[i, 2].set(0)
            ghosts = ghosts.at[i, 3].set(0)
            ghosts = ghosts.at[i, 4].set(self.node_positions_x[ghost_node_idx])
            ghosts = ghosts.at[i, 5].set(self.node_positions_y[ghost_node_idx])
            ghosts = ghosts.at[i, 6].set(ghost_node_idx)
            ghosts = ghosts.at[i, 7].set(ghost_node_idx)
            
        return state._replace(
            lives=new_lives,
            player_x=player_x,
            player_y=player_y,
            player_direction=jnp.array(0, dtype=jnp.int32),
            player_next_direction=jnp.array(-1, dtype=jnp.int32),
            player_current_node_index=jnp.array(player_start_node_idx, dtype=jnp.int32),
            player_target_node_index=jnp.array(player_start_node_idx, dtype=jnp.int32),
            ghosts=ghosts,
            frightened_timer=jnp.array(0, dtype=jnp.int32),
            scatter_chase_timer=jnp.array(0, dtype=jnp.int32),
            is_scatter_mode=jnp.array(True),
            player_state=jnp.array(0, dtype=jnp.int32), # Alive again
            death_timer=jnp.array(0, dtype=jnp.int32)
        )

    def _reset_for_new_level(self, state: PacmanState) -> PacmanState:
        """Resets entities and pellets for a new level."""
        # Increment level
        new_level = state.level + 1
        
        # Reset player
        player_start_node_idx = self._find_nearest_node_idx(
            self.consts.PLAYER_START_X,
            self.consts.PLAYER_START_Y
        )
        player_x = jnp.array(self.node_positions_x[player_start_node_idx], dtype=jnp.int32)
        player_y = jnp.array(self.node_positions_y[player_start_node_idx], dtype=jnp.int32)
        
        # Reset ghosts
        ghosts = jnp.zeros((4, 8), dtype=jnp.int32)
        for i in range(4):
            ghost_x = self.consts.GHOST_START_X + i * 8
            ghost_y = self.consts.GHOST_START_Y
            ghost_node_idx = self._find_nearest_node_idx(ghost_x, ghost_y)
            ghosts = ghosts.at[i, 0].set(self.node_positions_x[ghost_node_idx])
            ghosts = ghosts.at[i, 1].set(self.node_positions_y[ghost_node_idx])
            ghosts = ghosts.at[i, 2].set(0)
            ghosts = ghosts.at[i, 3].set(0)
            ghosts = ghosts.at[i, 4].set(self.node_positions_x[ghost_node_idx])
            ghosts = ghosts.at[i, 5].set(self.node_positions_y[ghost_node_idx])
            ghosts = ghosts.at[i, 6].set(ghost_node_idx)
            ghosts = ghosts.at[i, 7].set(ghost_node_idx)

        # Reset pellets
        pellets_collected = jnp.zeros((self.consts.MAZE_HEIGHT, self.consts.MAZE_WIDTH), dtype=jnp.int32)
        dots_remaining = jnp.array(240, dtype=jnp.int32) # Reset count

        return state._replace(
            level=new_level,
            player_x=player_x,
            player_y=player_y,
            player_direction=jnp.array(0, dtype=jnp.int32),
            player_next_direction=jnp.array(-1, dtype=jnp.int32),
            player_current_node_index=jnp.array(player_start_node_idx, dtype=jnp.int32),
            player_target_node_index=jnp.array(player_start_node_idx, dtype=jnp.int32),
            ghosts=ghosts,
            pellets_collected=pellets_collected,
            dots_remaining=dots_remaining,
            frightened_timer=jnp.array(0, dtype=jnp.int32),
            level_transition_timer=jnp.array(0, dtype=jnp.int32),
            scatter_chase_timer=jnp.array(0, dtype=jnp.int32),
            is_scatter_mode=jnp.array(True),
            freeze_timer=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            player_state=jnp.array(0, dtype=jnp.int32)
        )

    def _player_step(self, state: PacmanState, action: chex.Array) -> PacmanState:
        """
        Handle player movement - full Pacman movement implementation.
        Key behaviors:
        - Only stops at node if can't continue in current direction
        - Can reverse direction at any time (when not overshooting)
        - Input direction takes precedence when overshooting a node
        """
        # Check if this is a NOOP action
        is_noop = action == Action.NOOP
        
        # Get current and target node positions
        current_node_idx = state.player_current_node_index
        target_node_idx = state.player_target_node_index
        
        current_node_x = self.node_positions_x[current_node_idx]
        current_node_y = self.node_positions_y[current_node_idx]
        target_node_x = self.node_positions_x[target_node_idx]
        target_node_y = self.node_positions_y[target_node_idx]
        
        # Check if we've overshot the target node (equivalent to overshotTarget() in pacman.py)
        # Calculate direction vector from current node to target node
        dx_to_target = target_node_x - current_node_x
        dy_to_target = target_node_y - current_node_y
        vec_to_target_sq = dx_to_target * dx_to_target + dy_to_target * dy_to_target
        
        # Calculate distance from current node to current position
        vec_to_self_sq = (state.player_x - current_node_x) * (state.player_x - current_node_x) + (state.player_y - current_node_y) * (state.player_y - current_node_y)
        overshot = vec_to_self_sq >= vec_to_target_sq
        
        # If overshot target (reached/passed the target node)
        # Update current node to target, then get new target
        new_current_idx = jnp.where(overshot, target_node_idx, current_node_idx)
        
        # Get new target from input direction (equivalent to getNewTarget(direction) in pacman.py)
        # Raw graph neighbors
        new_target_from_action = self.neighbor_lookup[new_current_idx, action]
        new_target_from_current_dir = self.neighbor_lookup[new_current_idx, state.player_direction]

        # Block edges that go through the door for Pacman only
        # door_edge_mask[new_current_idx, action] == True -> treat as no neighbor (-1)
        blocked_action = self.player_door_edge_mask[new_current_idx, action]
        blocked_curr  = self.player_door_edge_mask[new_current_idx, state.player_direction]

        new_target_from_action = jnp.where(blocked_action, -1, new_target_from_action)
        new_target_from_current_dir = jnp.where(blocked_curr, -1, new_target_from_current_dir)

        has_target_from_action = new_target_from_action >= 0
        valid_action = jnp.logical_and(jnp.logical_not(is_noop), has_target_from_action)
        
        # Get new target from current direction (equivalent to getNewTarget(self.direction))
        has_target_from_current_dir = new_target_from_current_dir >= 0
        valid_current_dir = jnp.logical_and(state.player_direction != Action.NOOP, has_target_from_current_dir)
        
        # When overshot: input direction takes precedence, then current direction, else stop
        # When not overshot: keep current target
        new_target_idx = jnp.where(
            overshot,
            jnp.where(valid_action, new_target_from_action,
                     jnp.where(valid_current_dir, new_target_from_current_dir, new_current_idx)),
            target_node_idx
        )
        
        # Update direction when overshot:
        # - If input direction gives valid target, use input direction
        # - Else if current direction gives valid target, use current direction
        # - Else stop (NOOP)
        new_direction = jnp.where(
            overshot,
            jnp.where(valid_action, action,
                     jnp.where(valid_current_dir, state.player_direction, jnp.array(Action.NOOP, dtype=jnp.int32))),
            state.player_direction
        )
        
        # If overshot, snap position to target node (equivalent to setPosition() in pacman.py)
        snapped_x = jnp.where(overshot, target_node_x, state.player_x)
        snapped_y = jnp.where(overshot, target_node_y, state.player_y)
        
        # If NOT overshot, check if we can reverse direction (oppositeDirection check)
        # Action values: UP=2, DOWN=5, LEFT=4, RIGHT=3
        # Opposite pairs: UP(2) <-> DOWN(5), LEFT(4) <-> RIGHT(3)
        is_opposite = jnp.where(
            action == Action.UP, state.player_direction == Action.DOWN,
            jnp.where(
                action == Action.DOWN, state.player_direction == Action.UP,
                jnp.where(
                    action == Action.LEFT, state.player_direction == Action.RIGHT,
                    jnp.where(
                        action == Action.RIGHT, state.player_direction == Action.LEFT,
                        False
                    )
                )
            )
        )
        can_reverse = jnp.logical_and(jnp.logical_not(overshot), 
                                     jnp.logical_and(jnp.logical_not(is_noop), is_opposite))
        
        # If reversing direction, swap node and target, and change direction
        # reverseDirection() in pseudo code: direction *= -1, swap(node, target)
        final_current_idx = jnp.where(can_reverse, target_node_idx, new_current_idx)
        final_target_idx = jnp.where(can_reverse, current_node_idx, new_target_idx)
        final_direction = jnp.where(can_reverse, action, new_direction)
        
        # Update target node position for movement calculation
        final_target_x = self.node_positions_x[final_target_idx]
        final_target_y = self.node_positions_y[final_target_idx]
        
        # Calculate movement direction based on final direction
        # Action values: Action.UP=2, Action.DOWN=5, Action.LEFT=4, Action.RIGHT=3
        # Movement deltas: RIGHT=+x, LEFT=-x, DOWN=+y, UP=-y
        move_dx = jnp.where(final_direction == Action.RIGHT, self.consts.PLAYER_SPEED,
                   jnp.where(final_direction == Action.LEFT, -self.consts.PLAYER_SPEED, 0))
        move_dy = jnp.where(final_direction == Action.DOWN, self.consts.PLAYER_SPEED,
                   jnp.where(final_direction == Action.UP, -self.consts.PLAYER_SPEED, 0))
        
        # Move incrementally towards target (equivalent to position += directions[direction]*speed*dt)
        new_x = jnp.where(final_direction != Action.NOOP, snapped_x + move_dx, snapped_x)
        new_y = jnp.where(final_direction != Action.NOOP, snapped_y + move_dy, snapped_y)
        
        return state._replace(
            player_x=new_x.astype(jnp.int32),
            player_y=new_y.astype(jnp.int32),
            player_direction=final_direction,
            player_next_direction=jnp.array(Action.NOOP, dtype=jnp.int32),
            player_current_node_index=final_current_idx,
            player_target_node_index=final_target_idx,
        )
    
    def _find_nearest_node_idx(self, x: int, y: int) -> int:
        """Find nearest node index to given position (non-JIT for init)."""
        min_dist = float('inf')
        nearest_idx = 0
        for idx in range(len(self.node_group.nodeList)):
            node_x = int(self.node_group.nodeList[idx].position.x)
            node_y = int(self.node_group.nodeList[idx].position.y)
            dist = (x - node_x)**2 + (y - node_y)**2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        return nearest_idx

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
        """
        Updates ghost positions and states.
        
        Ghosts use node-based movement and select targets based on personality
        (Blinky, Pinky, Inky, Clyde) or state (Frightened, Eaten).
        """
        
        def process_single_ghost(idx, ghost_data, key):
            gx, gy, gdir, gstate, gtx, gty, gcurrent, gtarget = ghost_data
            
            # Get current and target node positions
            current_node_x = self.node_positions_x[gcurrent]
            current_node_y = self.node_positions_y[gcurrent]
            target_node_x = self.node_positions_x[gtarget]
            target_node_y = self.node_positions_y[gtarget]
            
            # Check if reached target node
            at_current = jnp.logical_and(gx == current_node_x, gy == current_node_y)
            
            # Calculate speed early to check for overshoot
            speed = jnp.where(gstate == 2, self.consts.GHOST_SPEED_EATEN, self.consts.GHOST_SPEED_NORMAL)
            
            dx_to_target = target_node_x - current_node_x
            dy_to_target = target_node_y - current_node_y
            vec_to_target_sq = dx_to_target * dx_to_target + dy_to_target * dy_to_target
            vec_to_self_sq = (gx - current_node_x) * (gx - current_node_x) + (gy - current_node_y) * (gy - current_node_y)
            
            # Check if already passed target
            already_passed = vec_to_self_sq >= vec_to_target_sq
            
            # Check if will reach target in this step (prevent overshoot)
            dist_remaining_sq = (gx - target_node_x) * (gx - target_node_x) + (gy - target_node_y) * (gy - target_node_y)
            will_reach = dist_remaining_sq <= speed * speed
            
            is_moving = gdir != Action.NOOP
            should_stop = jnp.logical_and(is_moving, jnp.logical_or(already_passed, will_reach))
            
            # Update current node if stopped
            new_current = jnp.where(should_stop, gtarget, gcurrent)
            snapped_x = jnp.where(should_stop, target_node_x, gx)
            snapped_y = jnp.where(should_stop, target_node_y, gy)
            
            at_node = jnp.logical_or(at_current, should_stop)
            
            # Choose next target when at node
            # Simple AI: pick random valid neighbor
            valid_neighbors = self.neighbor_lookup[new_current, :]

            # Apply One-Way Door restriction for Ghosts
            # If ghost is NOT eaten (state != 2), it cannot traverse "entry" edges into the house
            is_restricted = gstate != 2
            entry_mask = self.ghost_entry_mask[new_current, :]
            
            # If restricted and edge is an entry edge, treat as invalid (-1)
            # mask: True if edge should be blocked
            should_block = jnp.logical_and(is_restricted, entry_mask)
            
            valid_neighbors = jnp.where(should_block, -1, valid_neighbors)
            
            # Filter valid directions (not -1)
            valid_mask = valid_neighbors >= 0
            
            # Count valid neighbors
            num_valid = jnp.sum(valid_mask)
            
            # Pick random valid direction
            rand_idx = jax.random.randint(key, (), 0, 18)
            # Find the rand_idx-th valid neighbor (with wraparound)
            rand_idx = rand_idx % jnp.maximum(num_valid, 1)
            
            # Get that neighbor
            cumsum = jnp.cumsum(valid_mask.astype(jnp.int32))
            # FIX: Ensure we only select indices that are VALID neighbors
            selected_neighbor_mask = jnp.logical_and(valid_mask, cumsum == (rand_idx + 1))
            
            new_target_from_random = jnp.where(
                jnp.any(selected_neighbor_mask),
                jnp.sum(jnp.where(selected_neighbor_mask, valid_neighbors, 0)),
                new_current
            )
            
            # Update target when at node
            new_target = jnp.where(at_node, new_target_from_random, gtarget)
            
            # Eaten ghosts (state 2) ignore other logic and return to ghost house.
            is_eaten = gstate == 2
            
            # Ghost house is around center of maze
            # Use pre-computed ghost house node index
            ghost_house_node = self.ghost_house_node_idx
            
            # Check if eaten ghost reached the house
            at_house = jnp.logical_and(
                is_eaten,
                jnp.logical_and(
                    jnp.abs(gx - self.node_positions_x[ghost_house_node]) < 8,
                    jnp.abs(gy - self.node_positions_y[ghost_house_node]) < 8
                )
            )
            
            # Respawn: change state back to normal
            respawned_state = jnp.where(at_house, 0, gstate)
            
            # For eaten ghosts, override target to be ghost house
            # Pick neighbor that gets closer to ghost house
            def get_distance_to_house(node_idx):
                nx = self.node_positions_x[node_idx]
                ny = self.node_positions_y[node_idx]
                hx = self.node_positions_x[ghost_house_node]
                hy = self.node_positions_y[ghost_house_node]
                return (nx - hx) * (nx - hx) + (ny - hy) * (ny - hy)
            
            # When eaten and at node, pick neighbor closest to house
            def pick_best_neighbor_to_house():
                valid_neighbors = self.neighbor_lookup[new_current, :]
                valid_mask = valid_neighbors >= 0
                
                # Calculate distance for each valid neighbor
                def calc_dist(i):
                    neighbor = valid_neighbors[i]
                    is_valid = valid_mask[i]
                    dist = jnp.where(
                        is_valid,
                        get_distance_to_house(neighbor),
                        jnp.inf
                    )
                    return dist
                
                distances = jax.vmap(calc_dist)(jnp.arange(18))
                best_idx = jnp.argmin(distances)
                return valid_neighbors[best_idx]
            
            # Unique personalities for each ghost (when not eaten).
            # Blinky (Red): Direct chaser.
            # Pinky (Pink): Ambusher (targets ahead of player).
            # Inky (Cyan): Patroller (mix of Blinky and player position).
            # Clyde (Orange): Shy (chases when far, retreats when close).
            
            # Calculate target node for each ghost personality
            def get_target_for_personality():
                # Use player's current node from state
                player_node = state.player_current_node_index
                
                # Blinky: Direct chase - target player's node
                blinky_target = player_node
                
                # Pinky: Target ahead of player
                player_dir = state.player_direction
                pinky_neighbors = self.neighbor_lookup[player_node, :]
                pinky_target = jnp.where(
                    pinky_neighbors[player_dir] >= 0,
                    pinky_neighbors[player_dir],
                    player_node
                )
                
                # Inky: Patrol - cycle through corners
                inky_corners = jnp.array([0, 4, 45, 48], dtype=jnp.int32)
                inky_idx = (state.step_counter // 200) % 4
                inky_target = inky_corners[inky_idx]
                
                # Clyde: Shy - chase when far, retreat when close
                dx = gx - state.player_x
                dy = gy - state.player_y
                dist_sq = dx * dx + dy * dy
                clyde_target = jnp.where(dist_sq > 2500, player_node, 0)
                
                # Select based on ghost index
                targets = jnp.array([blinky_target, pinky_target, inky_target, clyde_target], dtype=jnp.int32)
                return targets[idx]
            
            # Get personality-based target (only for normal ghosts)
            personality_target_node = get_target_for_personality()
            
            # Pick neighbor that gets closest to personality target
            def pick_best_neighbor_to_target(target_node):
                valid_neighbors = self.neighbor_lookup[new_current, :]
                valid_mask = valid_neighbors >= 0
                
                def calc_dist_to_target(i):
                    neighbor = valid_neighbors[i]
                    is_valid = valid_mask[i]
                    # Distance to target node
                    nx = self.node_positions_x[neighbor]
                    ny = self.node_positions_y[neighbor]
                    tx = self.node_positions_x[target_node]
                    ty = self.node_positions_y[target_node]
                    dist = (nx - tx) * (nx - tx) + (ny - ty) * (ny - ty)
                    return jnp.where(is_valid, dist, jnp.inf)
                
                distances = jax.vmap(calc_dist_to_target)(jnp.arange(18))
                best_idx = jnp.argmin(distances)
                return valid_neighbors[best_idx]
            
            # Choose target based on state
            target_for_eaten = pick_best_neighbor_to_house()
            target_for_normal = pick_best_neighbor_to_target(personality_target_node)
            target_for_frightened = new_target_from_random  # Random when frightened
            
            # Update target based on ghost state
            new_target = jnp.where(
                at_node,
                jnp.where(
                    is_eaten,
                    target_for_eaten,  # Eaten: go to house
                    jnp.where(
                        gstate == 1,
                        target_for_frightened,  # Frightened: random
                        target_for_normal  # Normal: personality-based
                    )
                ),
                gtarget  # Not at node: keep current target
            )
            
            # Update direction based on movement
            target_x_pos = self.node_positions_x[new_target]
            target_y_pos = self.node_positions_y[new_target]
            
            # Determine direction
            dx = target_x_pos - snapped_x
            dy = target_y_pos - snapped_y
            
            new_direction = jnp.where(
                dx > 0, Action.RIGHT,
                jnp.where(dx < 0, Action.LEFT,
                jnp.where(dy > 0, Action.DOWN,
                jnp.where(dy < 0, Action.UP, Action.NOOP)))
            )
            
            # Move toward target with appropriate speed
            speed = jnp.where(is_eaten, self.consts.GHOST_SPEED_EATEN, self.consts.GHOST_SPEED_NORMAL)
            move_dx = jnp.where(new_direction == Action.RIGHT, speed,
                       jnp.where(new_direction == Action.LEFT, -speed, 0))
            move_dy = jnp.where(new_direction == Action.DOWN, speed,
                       jnp.where(new_direction == Action.UP, -speed, 0))
            
            new_x = jnp.where(new_direction != Action.NOOP, snapped_x + move_dx, snapped_x)
            new_y = jnp.where(new_direction != Action.NOOP, snapped_y + move_dy, snapped_y)
            
            return jnp.array([new_x, new_y, new_direction, respawned_state, target_x_pos, target_y_pos, new_current, new_target], dtype=jnp.int32)
        
        # Process all ghosts
        ghost_indices = jnp.arange(4)
        keys = jax.random.split(keys, 4)
        new_ghosts = jax.vmap(process_single_ghost)(ghost_indices, state.ghosts, keys)
        
        # Update state key
        new_key, _ = jax.random.split(keys[0])
        
        return state._replace(ghosts=new_ghosts, key=new_key)

    def _check_collisions(self, state: PacmanState) -> PacmanState:
        """Check player-ghost and player-pellet collisions."""
        # Check ghost collisions
        player_left = state.player_x
        player_right = state.player_x + self.consts.PLAYER_SIZE[0]
        player_top = state.player_y
        player_bottom = state.player_y + self.consts.PLAYER_SIZE[1]
        
        def check_ghost_collision(ghost_data):
            gx, gy, _, gstate, _, _, gcurrent, gtarget = ghost_data
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
            gx, gy, gdir, gstate, gtx, gty, gcurrent, gtarget = ghost_data
            
            # If frightened and collided, ghost is eaten
            new_state = jnp.where(
                jnp.logical_and(collided, gstate == 1),
                2,  # eaten
                gstate
            )
            
            # If normal and collided, player loses life
            life_lost = jnp.logical_and(collided, gstate == 0)
            
            # Calculate score for eating ghost using progressive scoring
            # First ghost: 200, second: 400, third: 800, fourth: 1600
            ghost_eaten = jnp.logical_and(collided, gstate == 1)
            ghost_score = jnp.where(
                ghost_eaten,
                self.consts.GHOST_SCORE_BASE * (2 ** state.ghosts_eaten_count),
                0
            )
            
            return new_state, life_lost, ghost_score, ghost_eaten
        
        results = jax.vmap(process_collision)(jnp.arange(4), state.ghosts, collisions)
        new_ghost_states = results[0]
        any_life_lost = jnp.any(results[1])
        ghost_scores = jnp.sum(results[2])
        any_ghost_eaten = jnp.any(results[3])
        
        # Update ghosts eaten counter
        new_ghosts_eaten_count = jnp.where(
            any_ghost_eaten,
            state.ghosts_eaten_count + 1,
            state.ghosts_eaten_count
        )
        
        # Update ghost states
        new_ghosts = state.ghosts.at[:, 3].set(new_ghost_states)
        


        # Update score
        new_score = state.score + ghost_scores
        
        # Handle Death: If life lost, set player state to dying and start death timer
        new_player_state = jnp.where(any_life_lost, 1, state.player_state)
        new_death_timer = jnp.where(any_life_lost, self.consts.DEATH_DURATION, state.death_timer)
        
        # Handle Freeze: If ghost eaten, start freeze timer
        new_freeze_timer = jnp.where(any_ghost_eaten, self.consts.FREEZE_DURATION, state.freeze_timer)
        
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
        
        # Reset ghosts_eaten_count when new power pellet is eaten
        final_ghosts_eaten_count = jnp.where(
            power_pellet_eaten,
            0,  # Reset counter on new power pellet
            new_ghosts_eaten_count
        )
        
        # Check level completion
        level_complete = jnp.logical_and(new_dots_remaining <= 0, state.level_transition_timer == 0)
        
        new_transition_timer = jnp.where(
            level_complete,
            self.consts.LEVEL_TRANSITION_DURATION,
            state.level_transition_timer
        )

        return state._replace(
            ghosts=new_ghosts_final,
            score=new_score,
            lives=state.lives, # Lives updated after death animation
            frightened_timer=new_frightened_timer,
            ghosts_eaten_count=final_ghosts_eaten_count,
            dots_remaining=new_dots_remaining,
            pellets_collected=new_pellets_collected,
            level_transition_timer=new_transition_timer,
            player_state=new_player_state,
            death_timer=new_death_timer,
            freeze_timer=new_freeze_timer
        )

    def _update_timers(self, state: PacmanState) -> PacmanState:
        """Update frightened and scatter/chase timers."""
        # Update frightened timer
        new_frightened = jnp.maximum(state.frightened_timer - 1, 0)
        
        # If frightened timer expired, return frightened ghosts to normal (but not eaten ghosts)
        def update_ghost_state(ghost_data):
            gx, gy, gdir, gstate, gtx, gty, gcurrent, gtarget = ghost_data
            # If frightened and timer expired, return to normal
            new_state = jnp.where(
                jnp.logical_and(gstate == 1, new_frightened == 0),
                0,  # normal
                gstate
            )
            return jnp.array([gx, gy, gdir, new_state, gtx, gty, gcurrent, gtarget], dtype=jnp.int32)
        
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
        
        # Reset ghosts_eaten_count when frightened mode ends
        final_ghosts_eaten_count = jnp.where(
            jnp.logical_and(state.frightened_timer > 0, new_frightened == 0),
            0,  # Reset counter when mode ends
            state.ghosts_eaten_count
        )
        
        # Update level transition timer
        new_transition = jnp.maximum(state.level_transition_timer - 1, 0)

        return state._replace(
            ghosts=new_ghosts,
            frightened_timer=new_frightened,
            ghosts_eaten_count=final_ghosts_eaten_count,
            scatter_chase_timer=new_scatter_chase,
            is_scatter_mode=new_is_scatter,
            level_transition_timer=new_transition
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
            level_transition_timer=state.level_transition_timer,
            player_state=state.player_state,
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
            obs.level_transition_timer.flatten(),
            obs.player_state.flatten(),
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(5)

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=224, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=288, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=224, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=288, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "ghosts": spaces.Box(low=0, high=288, shape=(4, 5), dtype=jnp.int32),
            "dots_remaining": spaces.Box(low=0, high=240, shape=(), dtype=jnp.int32),
            "power_pellets_active": spaces.Box(low=0, high=15, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=1, high=255, shape=(), dtype=jnp.int32),
            "frightened_timer": spaces.Box(low=0, high=200, shape=(), dtype=jnp.int32),
            "level_transition_timer": spaces.Box(low=0, high=200, shape=(), dtype=jnp.int32),
            "player_state": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(288, 224, 3),  # (height, width, channels)
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
        """Game over when no lives remaining."""
        # Note: Completing a level is NOT done, it just transitions.
        return state.lives <= 0


class PacmanRenderer(JAXGameRenderer):
    def __init__(self, consts: PacmanConstants = None):
        super().__init__(consts)
        self.consts = consts or PacmanConstants()
        # Maze layout will be set by JaxPacman after initialization
        self.config = render_utils.RendererConfig(
            game_dimensions=(288, 224),  # (height, width) format
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
        
        # Resize BACKGROUND to match new game dimensions if needed
        bg_h, bg_w = self.BACKGROUND.shape[:2]
        target_h, target_w = self.config.game_dimensions
        if bg_h != target_h or bg_w != target_w:
            from scipy.ndimage import zoom
            zoom_h = target_h / bg_h
            zoom_w = target_w / bg_w
            self.BACKGROUND = jnp.array(zoom(self.BACKGROUND, (zoom_h, zoom_w), order=0).astype(np.uint8))
        

        
        # Find Wall and Background IDs dynamically from the loaded palette
        # Use COLOR_TO_ID to get correct indices

        self.bg_id = jnp.array(self.COLOR_TO_ID[tuple(self.consts.BACKGROUND_COLOR)], dtype=jnp.uint8)
        self.white_id = jnp.array(self.COLOR_TO_ID[(255, 255, 255)], dtype=jnp.uint8)

        # Pre-render static maze background (walls and ghost house)
        self.maze_background = self._create_maze_background()

        # NEW: precompute per-tile wall connectivity mask indices (0–15)
        self.wall_mask_indices = self._compute_wall_masks()
    
    def _compute_wall_masks(self) -> jnp.ndarray:
        """Precompute 4-way connectivity mask for wall-like tiles.

        Bit layout (0–15):
          bit 0: neighbor wall above (up)
          bit 1: neighbor wall to the right
          bit 2: neighbor wall below (down)
          bit 3: neighbor wall to the left
        """
        import numpy as np

        layout_np = np.array(self.consts.MAZE_LAYOUT)  # (H, W)
        H, W = layout_np.shape

        def is_wall_like(val: int) -> bool:
            # Treat these tile types as solid walls for connectivity purposes.
            # 1 = normal wall
            return val == 1

        mask_indices = np.zeros_like(layout_np, dtype=np.int32)

        for row in range(H):
            for col in range(W):
                if not is_wall_like(int(layout_np[row, col])):
                    continue  # leave 0 for non-wall tiles

                up    = is_wall_like(int(layout_np[row - 1, col])) if row > 0 else False
                right = is_wall_like(int(layout_np[row, col + 1])) if col < W - 1 else False
                down  = is_wall_like(int(layout_np[row + 1, col])) if row < H - 1 else False
                left  = is_wall_like(int(layout_np[row, col - 1])) if col > 0 else False

                mask = (
                    (1 if up else 0)   |  # bit 0
                    (2 if right else 0)|  # bit 1
                    (4 if down else 0) |  # bit 2
                    (8 if left else 0)    # bit 3
                )
                mask_indices[row, col] = mask

        return jnp.array(mask_indices, dtype=jnp.int32)
    
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
        wall_masks = self.SHAPE_MASKS["wall"]
        door_mask = self.SHAPE_MASKS["ghost_door"]
        pellet_dot_mask = self.SHAPE_MASKS["pellet_dot"]
        # pellet_power_mask is now accessed dynamically via self.SHAPE_MASKS["pellet_power"][frame]
        digit_masks = self.SHAPE_MASKS["digits"]
        
        # Draw tiles
        def render_tile(i, raster_state):
            row = i // self.consts.MAZE_WIDTH
            col = i % self.consts.MAZE_WIDTH
            tile_val = self.consts.MAZE_LAYOUT[row, col]
            x = col * self.consts.TILE_SIZE
            y = row * self.consts.TILE_SIZE
            
            # Draw wall / ghost-house walls with connectivity-based sprite
            def draw_wall(r):
                mask_idx = self.wall_mask_indices[row, col]  # 0–15
                wall_mask = wall_masks[mask_idx]
                return self.jr.render_at(r, x, y, wall_mask)

            # Tiles that should be rendered using wall connectivity
            is_wall_tile = (tile_val == 1)  # Only normal walls

            raster_state = jax.lax.cond(
                is_wall_tile,
                draw_wall,
                lambda r: r,
                raster_state
            )

            # Draw door (pink)
            raster_state = jax.lax.cond(
                tile_val == 5,   # door tile
                lambda r: self.jr.render_at(r, x, y, door_mask),
                lambda r: r,
                raster_state
            )

            # Draw dot (centered)
            # Dot is pre-centered in 8x8 sprite
            is_dot = jnp.logical_and(tile_val == 2, state.pellets_collected[row, col] == 0)
            raster_state = jax.lax.cond(
                is_dot,
                lambda r: self.jr.render_at(r, x, y, pellet_dot_mask),
                lambda r: r,
                raster_state
            )
            
            # Draw power pellet
            is_power = jnp.logical_and(tile_val == 3, state.pellets_collected[row, col] == 0)
            
            # Blink logic
            power_frame = (state.step_counter // 10) % 2
            power_mask = self.SHAPE_MASKS["pellet_power"][power_frame]
            
            raster_state = jax.lax.cond(
                is_power,
                lambda r: self.jr.render_at(r, x, y, power_mask),
                lambda r: r,
                raster_state
            )
            
            return raster_state
        
        # Draw all tiles
        num_tiles = self.consts.MAZE_HEIGHT * self.consts.MAZE_WIDTH
        raster = jax.lax.fori_loop(0, num_tiles, render_tile, raster)

        # Check if dying
        is_dying = state.player_state == 1
        
        def render_alive(r):
            player_dir_idx = state.player_direction
            player_frame = state.player_animation_frame
            # Use pre-computed lookup table
            base_sprite_idx = self.consts.SPRITE_LOOKUP[0][player_dir_idx]
            player_sprite_idx = base_sprite_idx + player_frame
            player_mask = self.SHAPE_MASKS["player"][player_sprite_idx]
            return self.jr.render_at(r, state.player_x, state.player_y, player_mask)
            
        def render_dying(r):
            # Calculate death frame (0 to 11)
            progress = (self.consts.DEATH_DURATION - state.death_timer) / self.consts.DEATH_DURATION
            frame = (progress * 12).astype(jnp.int32)
            frame = jnp.clip(frame, 0, 11)
            
            death_mask = self.SHAPE_MASKS["player_death"][frame]
            
            # Rotate mask based on direction (assuming default sprites are RIGHT facing)
            dir_idx = state.player_direction
            
            death_mask = jax.lax.switch(
                dir_idx,
                [
                    lambda: death_mask, # 0: NOOP (Default Right)
                    lambda: death_mask, # 1: FIRE (Default Right)
                    lambda: jnp.rot90(death_mask, k=1), # 2: UP (90 CCW)
                    lambda: death_mask, # 3: RIGHT (No change)
                    lambda: jnp.fliplr(death_mask), # 4: LEFT (Flip H)
                    lambda: jnp.rot90(death_mask, k=3), # 5: DOWN (90 CW)
                ]
            )
            
            return self.jr.render_at(r, state.player_x, state.player_y, death_mask)
            
        raster = jax.lax.cond(is_dying, render_dying, render_alive, raster)
        
        # Helper to draw ghost
        def draw_ghost(g_idx, r):
            g = state.ghosts[g_idx]
            # Select mask based on state: 0=normal, 1=frightened, 2=eaten
            
            # Animation frame for normal ghosts
            anim_frame = (state.step_counter // 10) % 2
            
            # Get ghost direction index
            g_dir = g[2].astype(jnp.int32)
            
            # Lookup sprite offset for direction
            g_base_idx = self.consts.SPRITE_LOOKUP[1][g_dir]
            g_sprite_idx = g_base_idx + anim_frame
            
            # Frightened flashing (Blue/White) near end
            is_flashing = jnp.logical_and(state.frightened_timer < 60, (state.step_counter // 8) % 2 == 0)
            frightened_idx = jnp.where(is_flashing, 2 + anim_frame, anim_frame)

            g_mask = jax.lax.switch(
                g[3].astype(jnp.int32),
                [
                    # 0: Normal
                    lambda: [
                        lambda: self.SHAPE_MASKS["ghost_blinky"][g_sprite_idx],
                        lambda: self.SHAPE_MASKS["ghost_pinky"][g_sprite_idx],
                        lambda: self.SHAPE_MASKS["ghost_inky"][g_sprite_idx],
                        lambda: self.SHAPE_MASKS["ghost_clyde"][g_sprite_idx],
                    ][g_idx](),
                    # 1: Frightened
                    lambda: self.SHAPE_MASKS["ghost_frightened"][frightened_idx],
                    # 2: Eaten (Eyes)
                    lambda: jax.lax.switch(
                        g_dir,
                        [
                            lambda: self.SHAPE_MASKS["ghost_eyes"][0], # NOOP -> Right
                            lambda: self.SHAPE_MASKS["ghost_eyes"][0], # FIRE -> Right
                            lambda: self.SHAPE_MASKS["ghost_eyes"][2], # UP
                            lambda: self.SHAPE_MASKS["ghost_eyes"][0], # RIGHT
                            lambda: self.SHAPE_MASKS["ghost_eyes"][1], # LEFT
                            lambda: self.SHAPE_MASKS["ghost_eyes"][3], # DOWN
                        ]
                    )
                ]
            )
            return self.jr.render_at(r, g[0], g[1], g_mask)

        # Draw all 4 ghosts
        for i in range(4):
            raster = draw_ghost(i, raster)

        # Draw Score
        # Draw up to 6 digits, right aligned at top left padding
        score = state.score
        score_x = 10
        score_y = 2
        
        def draw_digit(i, r_val):
            # Extract digit: (score // 10^i) % 10
            divisor = jnp.power(10, i)
            digit = (score // divisor) % 10
            
            # Only draw if score >= divisor (except for 0)
            should_draw = jnp.logical_or(score >= divisor, i == 0)
            
            return jax.lax.cond(
                should_draw,
                lambda r: self.jr.render_at(r, score_x + (5 - i) * 8, score_y, digit_masks[digit]),
                lambda r: r,
                r_val
            )

        raster = jax.lax.fori_loop(0, 6, draw_digit, raster)
        
        # Draw Level Indicator (Top Right)
        level = state.level
        level_x = 130
        level_y = 2
        
        def draw_level_digit(i, r_val):
            divisor = jnp.power(10, i)
            digit = (level // divisor) % 10
            should_draw = jnp.logical_or(level >= divisor, i == 0)
            return jax.lax.cond(
                should_draw,
                lambda r: self.jr.render_at(r, level_x + (2 - i) * 8, level_y, digit_masks[digit]),
                lambda r: r,
                r_val
            )
        raster = jax.lax.fori_loop(0, 3, draw_level_digit, raster)

        # Level Transition Flash
        is_transitioning = state.level_transition_timer > 0
        flash_on = (state.level_transition_timer // 4) % 2 == 0
        
        # Black screen
        is_black_screen = jnp.logical_and(is_transitioning, state.level_transition_timer < 10)
        
        # Determine final raster
        final_raster = jax.lax.cond(
            is_black_screen,
            lambda: jnp.full_like(raster, self.bg_id),
            lambda: jax.lax.cond(
                jnp.logical_and(is_transitioning, flash_on),
                lambda: jnp.full_like(raster, self.white_id),
                lambda: raster
            )
        )
        
        return self.jr.render_from_palette(final_raster, self.PALETTE)
