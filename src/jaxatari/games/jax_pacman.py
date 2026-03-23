import os
import sys
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
            'player/player_right_0.npy', 'player/player_right_1.npy', 'player/player_right_2.npy',
            'player/player_left_0.npy', 'player/player_left_1.npy', 'player/player_left_2.npy',
        ]},
        {'name': 'player_death', 'type': 'group', 'files': [
            'player/death_0.npy', 'player/death_1.npy', 'player/death_2.npy', 'player/death_3.npy',
            'player/death_4.npy', 'player/death_5.npy', 'player/death_6.npy', 'player/death_7.npy',
            'player/death_8.npy', 'player/death_9.npy', 'player/death_10.npy', 'player/death_11.npy',
        ]},
        {'name': 'ghost_normal', 'type': 'group', 'files': [
            'ghost_normal/ghost_1.npy', 'ghost_normal/ghost_2.npy',
        ]},
        {'name': 'ghost_frightened', 'type': 'group', 'files': [
            'ghost_frightened/ghost_frightened_1.npy', 'ghost_frightened/ghost_frightened_2.npy',
        ]},
        {'name': 'ghost_eyes', 'type': 'single', 'file': 'ghost_eyes/eyes.npy'},
        {'name': 'ghost_eyes_pink', 'type': 'single', 'file': 'ghost_eyes/eyes_pink.npy'},
        {'name': 'pellet_dot', 'type': 'single', 'file': 'pellet_dot.npy'},
        {'name': 'pellet_power', 'type': 'group', 'files': [
            'pellet_power/pellet_power_on.npy',
            'pellet_power/pellet_power_off.npy'
        ]},
        {'name': 'pellet_power_frightened', 'type': 'group', 'files': [
            'pellet_power/pellet_power_frightened_on.npy',
            'pellet_power/pellet_power_frightened_off.npy'
        ]},
        {'name': 'vitamin', 'type': 'single', 'file': 'vitamin.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/digit_{}.npy'},
    )


def _get_sprite_lookup() -> chex.Array:
    """
    Creates lookup table for mapping Actions to Sprites.
    """
    # Define sprite offsets (3 frames per direction for player)
    SPRITE_RIGHT = 0
    SPRITE_LEFT = 3

    # Create lookup table for mapping Actions to Sprites
    lookup = np.zeros(18, dtype=np.int32)
    
    # Default to Right facing
    lookup[:] = SPRITE_RIGHT
    
    # Map Actions to Sprites
    # Pacman does not change sprite when going up or down, only left or right
    lookup[Action.UP] = SPRITE_RIGHT
    lookup[Action.DOWN] = SPRITE_RIGHT
    lookup[Action.LEFT] = SPRITE_LEFT
    lookup[Action.RIGHT] = SPRITE_RIGHT
    
    return jnp.array(lookup, dtype=jnp.int32)


class PacmanConstants(NamedTuple):
    # Screen dimensions (Atari 2600 Pacman native resolution)
    WIDTH: int = 160
    HEIGHT: int = 250  # 200 for Maze + 50 for UI
    
    TILE_SIZE: int = 8 
    ANIMATION_SPEED: int = 5 
    
    # Maze dimensions (Matches maze_atari.txt: 21x17)
    MAZE_WIDTH: int = 21
    MAZE_HEIGHT: int = 17
    
    # Player constants
    PLAYER_SIZE: Tuple[int, int] = (8, 8) 
    PLAYER_SPEED: int = 1 
    PLAYER_SPEED_VERTICAL: int = 2
    PLAYER_START_X: int = 80   # Col 10 * 8
    PLAYER_START_Y: int = 152  # Row 16 * 8 (Where 'S' is) + 32 offset
    
    # Ghost constants
    GHOST_SIZE: Tuple[int, int] = (8, 8)
    GHOST_SPEED_NORMAL: int = 1
    GHOST_SPEED_FRIGHTENED: int = 1
    GHOST_SPEED_EATEN: int = 1
    
    # Ghost starting positions (Ghost House Center)
    GHOST_START_X: int = 80   # Col 10 * 8 (Center of 'HHHH')
    GHOST_START_Y: int = 88   # Row 7 * 8 + 32 offset
    
    # Ghost colors
    GHOST_COLOR: Tuple[int, int, int] = (252, 144, 200)
    GHOST_FRIGHTENED_COLOR: Tuple[int, int, int] = (144, 144, 252)
    
    # Pellet constants
    PELLET_DOT_SIZE: Tuple[int, int] = (2, 2)
    PELLET_POWER_SIZE: Tuple[int, int] = (4, 4)
    PELLET_DOT_SCORE: int = 1
    PELLET_POWER_SCORE: int = 5
    GHOST_SCORE_BASE: int = 20
    
    # Game timing
    FRIGHTENED_DURATION: int = 200
    SCATTER_DURATION: int = 7000
    CHASE_DURATION: int = 20000
    LEVEL_TRANSITION_DURATION: int = 60
    
    # Vitamin constants
    VITAMIN_SCORE: int = 100
    VITAMIN_TRIGGER_PELLETS: int = 170  # Appears after this many pellets eaten
    VITAMIN_DURATION: int = 600  # ~10 seconds at 60fps
    FREEZE_DURATION: int = 15
    DEATH_DURATION: int = 60
    
    # Colors (User Update: Wall Yellow, BG Blue for Atari Look)
    # Using Atari Palette approximations
    BACKGROUND_COLOR: Tuple[int, int, int] = (50, 50, 176) # Blue
    WALL_COLOR: Tuple[int, int, int] = (223, 192, 111)   # Yellow/Orange/Gold
    PELLET_COLOR: Tuple[int, int, int] = (223, 192, 111)
    SCORE_COLOR: Tuple[int, int, int] = (255, 255, 0)
    
    # Maze layout grid where each cell represents a tile type.
    # 0: Empty path
    # 1: Wall
    # 2: Dot
    # 3: Power Pellet
    # 4: Ghost House
    MAZE_LAYOUT: chex.Array = None
    
    # Asset config
    ASSET_CONFIG: tuple = _get_default_asset_config()
    SPRITE_LOOKUP: chex.Array = _get_sprite_lookup()


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
    player_last_horizontal_dir: chex.Array # Visual: Keep track of last horizontal face
    player_animation_frame: chex.Array  # 0 or 1 for mouth open/close
    player_current_node_index: chex.Array  # Current node index for node-based movement
    player_target_node_index: chex.Array  # Target node index for node-based movement

    # Optional co-op support (kept inactive for default gameplay)
    multiplayer_active: chex.Array  # 0=disabled, 1=enabled by coop mod
    player2_x: chex.Array
    player2_y: chex.Array
    player2_direction: chex.Array
    player2_last_horizontal_dir: chex.Array
    player2_current_node_index: chex.Array
    player2_target_node_index: chex.Array
    
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
    
    # Vitamin bonus item
    vitamin_active: chex.Array  # 0=inactive, 1=active
    vitamin_timer: chex.Array   # Countdown timer for vitamin visibility
    vitamin_collected: chex.Array  # 1=already collected this level
    vitamin_x: chex.Array  # Pixel x position
    vitamin_y: chex.Array  # Pixel y position
    total_pellets_eaten: chex.Array  # Tracks total pellets eaten for vitamin trigger


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
    def __init__(
        self,
        consts: PacmanConstants = None,
        maze_file_path: str = None,
        maze_file_pellet_path: str = None,
    ):
        consts = consts or PacmanConstants()
        super().__init__(consts)
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]
        # Default vitamin tile; overwritten if '*' exists in maze map.
        self.vitamin_tile_row = 9
        self.vitamin_tile_col = 10
        
        # Determine paths relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        pacman_maps_dir = os.path.join(current_dir, "pacmanMaps")
        maze_file_path = os.path.join(pacman_maps_dir, "maze_atari.txt")
        # Use separate pellet map if it exists, otherwise fallback to the base layout
        maze_file_pellet_path = os.path.join(pacman_maps_dir, "maze_atari_pellet.txt")
        if not os.path.exists(maze_file_pellet_path):
            maze_file_pellet_path = maze_file_path
        import pacmanMaps.nodes as nodes_mod
        NodeGroup = nodes_mod.NodeGroup
        
        # Initialize maze layout if not provided (load from file)
        if consts.MAZE_LAYOUT is None:
            loaded_maze = self._load_maze_from_file(maze_file_pellet_path)
            if loaded_maze is None:
                raise ValueError("Failed to load maze layout.")
            self.consts = consts._replace(MAZE_LAYOUT=loaded_maze)
        else:
            self.consts = consts
        
        # Create renderer after maze layout is loaded (so it can create maze_background correctly)
        # Check to ensure layout is valid for renderer
        if self.consts.MAZE_LAYOUT is None:
             raise ValueError("MAZE_LAYOUT is None before initializing renderer!")
             
        self.renderer = PacmanRenderer(self.consts)
        
        # Load NodeGroup using the same maze file path
        # NodeGroup.from_maze_file handles scaling by tile_size
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
        center_y = ((self.consts.MAZE_HEIGHT * self.consts.TILE_SIZE) // 2) + 20
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
        ghost_wall_mask = np.zeros((num_nodes, num_actions), dtype=np.bool_)

        # Helper to get tile type at a node
        def tile_type_for_node(node):
            # x is unmodified in nodes (gen_map doesn't bake x_offset of -4)
            tx = int(node.position.x) // self.consts.TILE_SIZE
            # y has a +32 offset baked into the generated node coordinates (y=32 is top row 0)
            ty = (int(node.position.y) - 32) // self.consts.TILE_SIZE
            
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

                # Option 1: treat any edge that touches a house node as a "door edge" for Pac-Man
                if tile_n == 4 or tile_nb == 4:
                    door_edge_mask[n, a] = True
                
                # Option 2: Ghost entry mask (Block entry to House from non-right side)
                if tile_nb == 4 and tile_n != 4:
                     # Only allow entering H via LEFT action (approaching from the right)
                     if a != Action.LEFT:
                          ghost_entry_mask[n, a] = True
                     
                # Option 3: Ghost House Structural Wall constraints
                # From inside H, ghosts can only leave by moving RIGHT
                if tile_n == 4 and a != Action.RIGHT:
                     ghost_wall_mask[n, a] = True
                # From outside, approaching H from non-right side is blocked
                if tile_nb == 4 and a != Action.LEFT:
                     ghost_wall_mask[n, a] = True

        # Store mask as JAX array; used only for Pacman
        self.player_door_edge_mask = jnp.array(door_edge_mask, dtype=jnp.bool_)
        # Store ghost one-way mask
        self.ghost_entry_mask = jnp.array(ghost_entry_mask, dtype=jnp.bool_)
        # Store ghost structural boundaries
        self.ghost_wall_mask = jnp.array(ghost_wall_mask, dtype=jnp.bool_)

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
                    elif char == '*':
                        maze[row, col] = 2  # Treat as regular pellet
                        self.vitamin_tile_row = row
                        self.vitamin_tile_col = col
                    else:
                        maze[row, col] = 0
        # Default vitamin position if not found in map
        if not hasattr(self, 'vitamin_tile_row'):
            self.vitamin_tile_row = 9
            self.vitamin_tile_col = 10
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

        # Secondary player spawn (used only when multiplayer_active=1)
        player2_start_node_idx = self._find_nearest_node_idx(
            self.consts.PLAYER_START_X + (self.consts.TILE_SIZE * 2),
            self.consts.PLAYER_START_Y
        )
        player2_x = jnp.array(self.node_positions_x[player2_start_node_idx], dtype=jnp.int32)
        player2_y = jnp.array(self.node_positions_y[player2_start_node_idx], dtype=jnp.int32)
        
        # Initialize ghosts (4 ghosts at starting positions)
        ghosts = jnp.zeros((4, 8), dtype=jnp.int32)
        for i in range(4):
            ghost_x = self.consts.GHOST_START_X
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
        lives = jnp.array(3, dtype=jnp.int32)
        level = jnp.array(1, dtype=jnp.int32)
        pellets_collected = jnp.zeros((self.consts.MAZE_HEIGHT, self.consts.MAZE_WIDTH), dtype=jnp.int32)
        
        dots_remaining = jnp.array(195, dtype=jnp.int32)  # 191 regular + 4 power pellets
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
            player_last_horizontal_dir=jnp.array(0, dtype=jnp.int32),
            player_animation_frame=player_animation_frame,
            player_current_node_index=player_current_node_index,
            player_target_node_index=player_target_node_index,
            multiplayer_active=jnp.array(0, dtype=jnp.int32),
            player2_x=player2_x,
            player2_y=player2_y,
            player2_direction=jnp.array(0, dtype=jnp.int32),
            player2_last_horizontal_dir=jnp.array(0, dtype=jnp.int32),
            player2_current_node_index=jnp.array(player2_start_node_idx, dtype=jnp.int32),
            player2_target_node_index=jnp.array(player2_start_node_idx, dtype=jnp.int32),
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
            vitamin_active=jnp.array(0, dtype=jnp.int32),
            vitamin_timer=jnp.array(0, dtype=jnp.int32),
            vitamin_collected=jnp.array(0, dtype=jnp.int32),
            vitamin_x=jnp.array(self.vitamin_tile_col * self.consts.TILE_SIZE - 4, dtype=jnp.int32),
            vitamin_y=jnp.array(self.vitamin_tile_row * self.consts.TILE_SIZE + 32, dtype=jnp.int32),
            total_pellets_eaten=jnp.array(0, dtype=jnp.int32),
        )
        
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array) -> Tuple[PacmanObservation, PacmanState, float, bool, PacmanInfo]:
        new_state_key, step_key = jax.random.split(state.key)
        previous_state = state

        # Backward compatible input handling:
        # - scalar action: player2 uses mirrored fallback
        # - vector action [a1, a2]: explicit independent controls
        action_vec = jnp.atleast_1d(action).astype(jnp.int32)
        player1_action = action_vec[0]
        if action_vec.shape[0] >= 2:
            player2_action = action_vec[1]
        else:
            player2_action = self._map_player2_action(player1_action)
        
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
            state = self._player_step(state, player1_action)

            # Optional co-op helper player movement (kept disabled by default).
            state = jax.lax.cond(
                state.multiplayer_active == 1,
                lambda s: self._player2_step(s, player2_action),
                lambda s: s,
                state
            )
            
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

        player2_start_node_idx = self._find_nearest_node_idx(
            self.consts.PLAYER_START_X + (self.consts.TILE_SIZE * 2),
            self.consts.PLAYER_START_Y
        )
        player2_x = jnp.array(self.node_positions_x[player2_start_node_idx], dtype=jnp.int32)
        player2_y = jnp.array(self.node_positions_y[player2_start_node_idx], dtype=jnp.int32)
        
        ghosts = jnp.zeros((4, 8), dtype=jnp.int32)
        for i in range(4):
            ghost_x = self.consts.GHOST_START_X
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
            player_last_horizontal_dir=jnp.array(0, dtype=jnp.int32),
            player_current_node_index=jnp.array(player_start_node_idx, dtype=jnp.int32),
            player_target_node_index=jnp.array(player_start_node_idx, dtype=jnp.int32),
            player2_x=player2_x,
            player2_y=player2_y,
            player2_direction=jnp.array(0, dtype=jnp.int32),
            player2_last_horizontal_dir=jnp.array(0, dtype=jnp.int32),
            player2_current_node_index=jnp.array(player2_start_node_idx, dtype=jnp.int32),
            player2_target_node_index=jnp.array(player2_start_node_idx, dtype=jnp.int32),
            ghosts=ghosts,
            frightened_timer=jnp.array(0, dtype=jnp.int32),
            ghosts_eaten_count=jnp.array(0, dtype=jnp.int32),
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

        player2_start_node_idx = self._find_nearest_node_idx(
            self.consts.PLAYER_START_X + (self.consts.TILE_SIZE * 2),
            self.consts.PLAYER_START_Y
        )
        player2_x = jnp.array(self.node_positions_x[player2_start_node_idx], dtype=jnp.int32)
        player2_y = jnp.array(self.node_positions_y[player2_start_node_idx], dtype=jnp.int32)
        
        # Reset ghosts
        ghosts = jnp.zeros((4, 8), dtype=jnp.int32)
        for i in range(4):
            ghost_x = self.consts.GHOST_START_X
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
        dots_remaining = jnp.array(195, dtype=jnp.int32) # 191 regular + 4 power pellets

        return state._replace(
            level=new_level,
            lives=jnp.array(3, dtype=jnp.int32),
            player_x=player_x,
            player_y=player_y,
            player_direction=jnp.array(0, dtype=jnp.int32),
            player_next_direction=jnp.array(-1, dtype=jnp.int32),
            player_last_horizontal_dir=jnp.array(0, dtype=jnp.int32),
            player_current_node_index=jnp.array(player_start_node_idx, dtype=jnp.int32),
            player_target_node_index=jnp.array(player_start_node_idx, dtype=jnp.int32),
            player2_x=player2_x,
            player2_y=player2_y,
            player2_direction=jnp.array(0, dtype=jnp.int32),
            player2_last_horizontal_dir=jnp.array(0, dtype=jnp.int32),
            player2_current_node_index=jnp.array(player2_start_node_idx, dtype=jnp.int32),
            player2_target_node_index=jnp.array(player2_start_node_idx, dtype=jnp.int32),
            ghosts=ghosts,
            pellets_collected=pellets_collected,
            dots_remaining=dots_remaining,
            frightened_timer=jnp.array(0, dtype=jnp.int32),
            level_transition_timer=jnp.array(0, dtype=jnp.int32),
            scatter_chase_timer=jnp.array(0, dtype=jnp.int32),
            is_scatter_mode=jnp.array(True),
            freeze_timer=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            player_state=jnp.array(0, dtype=jnp.int32),
            vitamin_active=jnp.array(0, dtype=jnp.int32),
            vitamin_timer=jnp.array(0, dtype=jnp.int32),
            vitamin_collected=jnp.array(0, dtype=jnp.int32),
            total_pellets_eaten=jnp.array(0, dtype=jnp.int32),
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
        
        # Portal Detection Logic
        # 1. Get movement vector from current direction
        move_dx = jnp.where(state.player_direction == Action.RIGHT, 1, 
                       jnp.where(state.player_direction == Action.LEFT, -1, 0))
        move_dy = jnp.where(state.player_direction == Action.DOWN, 1,
                       jnp.where(state.player_direction == Action.UP, -1, 0))
        
        # 2. Check if we are moving AWAY from the target (Dot Product < 0)
        #    This happens when target is wrapped (e.g. Left<->Right or Top<->Bottom)
        dot_prod = move_dx * dx_to_target + move_dy * dy_to_target
        
        # 3. Check if nodes are far apart (Portal)
        #    is_portal = (dist > 2*TILE_SIZE) AND (moving away from target)
        nodes_dist_sq = vec_to_target_sq
        is_portal = jnp.logical_and(
            nodes_dist_sq > (self.consts.TILE_SIZE * 2)**2,
            dot_prod < 0
        )
        
        # 4. Define overshot condition
        #    Normal: reached target (dist_to_self >= dist_between_nodes)
        #    Portal: travel a short distance from current node before wrap (spatial delay)
        normal_overshot = vec_to_self_sq >= vec_to_target_sq
        
        # Portal overshot: use direction-dependent threshold to fix asymmetric delay.
        # Works for both left-right and up-down portals.
        # - Moving LEFT or UP (negative): we often go off-screen, so 2-pixel threshold
        #   gives 2 frames of "tunnel" that are not visible → feels quick.
        # - Moving RIGHT or DOWN (positive): we may stay on-screen, so the same
        #   threshold would show Pacman for 2 extra frames → feels like more delay.
        # Use threshold 0 when moving in positive direction (RIGHT/DOWN) so we wrap
        # after 1 pixel; use 1 for negative direction (LEFT/UP). Both directions
        # then feel like a quick wrap (horizontal and vertical).
        moving_positive = jnp.logical_or(
            state.player_direction == Action.RIGHT,
            state.player_direction == Action.DOWN
        )
        portal_threshold_sq = jnp.where(moving_positive, 0, 1)
        portal_overshot = vec_to_self_sq > portal_threshold_sq
        
        overshot = jnp.where(is_portal, portal_overshot, normal_overshot)
        
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
        # Alternate between 1 and 2 pixels vertically for ~1.5x speed
        vert_speed = jnp.where(state.step_counter % 2 == 0, 2, 1)
        move_dy = jnp.where(final_direction == Action.DOWN, vert_speed,
                   jnp.where(final_direction == Action.UP, -vert_speed, 0))
        
        # Move incrementally towards target (equivalent to position += directions[direction]*speed*dt)
        new_x = jnp.where(final_direction != Action.NOOP, snapped_x + move_dx, snapped_x)
        new_y = jnp.where(final_direction != Action.NOOP, snapped_y + move_dy, snapped_y)
        
        # Update last horizontal direction
        # If moving LEFT or RIGHT, update it. Else keep state.
        is_horizontal = jnp.logical_or(final_direction == Action.RIGHT, final_direction == Action.LEFT)
        new_last_h = jnp.where(is_horizontal, final_direction, state.player_last_horizontal_dir)
        
        return state._replace(
            player_x=new_x.astype(jnp.int32),
            player_y=new_y.astype(jnp.int32),
            player_direction=final_direction,
            player_next_direction=jnp.array(Action.NOOP, dtype=jnp.int32),
            player_last_horizontal_dir=new_last_h,
            player_current_node_index=final_current_idx,
            player_target_node_index=final_target_idx,
        )

    def _map_player2_action(self, action: chex.Array) -> chex.Array:
        """
        Maps the primary action to the helper player's action.
        LEFT/RIGHT are mirrored so both players split routes more often.
        """
        return jnp.where(
            action == Action.LEFT,
            jnp.array(Action.RIGHT, dtype=jnp.int32),
            jnp.where(
                action == Action.RIGHT,
                jnp.array(Action.LEFT, dtype=jnp.int32),
                action.astype(jnp.int32)
            )
        )

    def _player2_step(self, state: PacmanState, action: chex.Array) -> PacmanState:
        """
        Reuses the main player movement routine for player2 by temporarily
        projecting player2 fields into player1 slots.
        """
        projected_state = state._replace(
            player_x=state.player2_x,
            player_y=state.player2_y,
            player_direction=state.player2_direction,
            player_last_horizontal_dir=state.player2_last_horizontal_dir,
            player_current_node_index=state.player2_current_node_index,
            player_target_node_index=state.player2_target_node_index,
        )

        moved_state = self._player_step(projected_state, action)

        return state._replace(
            player2_x=moved_state.player_x,
            player2_y=moved_state.player_y,
            player2_direction=moved_state.player_direction,
            player2_last_horizontal_dir=moved_state.player_last_horizontal_dir,
            player2_current_node_index=moved_state.player_current_node_index,
            player2_target_node_index=moved_state.player_target_node_index,
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
        tile_y = (next_y - 32) // self.consts.TILE_SIZE
        
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
        
        Ghosts use node-based movement or state (Frightened, Eaten).
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
            valid_neighbors = self.neighbor_lookup[new_current, :]

            # Apply ghost structural wall boundaries (sides and floor of ghost house H)
            g_wall_mask = self.ghost_wall_mask[new_current, :]
            valid_neighbors = jnp.where(g_wall_mask, -1, valid_neighbors)

            # Apply One-Way Door restriction for Ghosts
            is_restricted = gstate != 2
            entry_mask = self.ghost_entry_mask[new_current, :]
            should_block = jnp.logical_and(is_restricted, entry_mask)
            valid_neighbors = jnp.where(should_block, -1, valid_neighbors)
            
            # Prevent reversing direction (no U-turns)
            def get_opposite_action(a):
                return jax.lax.switch(
                    a.astype(jnp.int32),
                    [
                        lambda: 0, lambda: 1, lambda: 5, lambda: 4, lambda: 3, lambda: 2
                    ] + [lambda: 0]*12
                )
            
            opposite_action = get_opposite_action(gdir)
            num_valid_before = jnp.sum(valid_neighbors >= 0)
            
            # Only block reverse if there are other valid options (no infinite lock in a true dead end)
            should_block_reverse = jnp.logical_and(num_valid_before > 1, gdir != Action.NOOP)
            
            # valid_neighbors has shape (18,), where index i corresponds to Action i
            valid_neighbors = jnp.where(
                 jnp.logical_and(should_block_reverse, jnp.arange(18) == opposite_action),
                 -1,
                 valid_neighbors
            )
            
            # Random neighbor fallback logic (for frightened ghosts)
            valid_mask = valid_neighbors >= 0
            num_valid = jnp.sum(valid_mask)
            rand_idx = jax.random.randint(key, (), 0, 18) % jnp.maximum(num_valid, 1)
            cumsum = jnp.cumsum(valid_mask.astype(jnp.int32))
            selected_neighbor_mask = jnp.logical_and(valid_mask, cumsum == (rand_idx + 1))
            new_target_from_random = jnp.where(
                jnp.any(selected_neighbor_mask),
                jnp.sum(jnp.where(selected_neighbor_mask, valid_neighbors, 0)),
                new_current
            )
            
            # Helper to pick valid neighbor closest to coordinate
            def pick_closest_neighbor_to_coord(t_x, t_y):
                def calc_dist(i):
                    nb = valid_neighbors[i]
                    nx = jnp.where(nb >= 0, self.node_positions_x[nb], 999999)
                    ny = jnp.where(nb >= 0, self.node_positions_y[nb], 999999)
                    dist = (nx - t_x) * (nx - t_x) + (ny - t_y) * (ny - t_y)
                    return jnp.where(nb >= 0, dist, 99999999)
                dists = jax.vmap(calc_dist)(jnp.arange(18))
                best_idx = jnp.argmin(dists)
                return jnp.where(valid_neighbors[best_idx] >= 0, valid_neighbors[best_idx], new_target_from_random)
            
            # Eaten state handling
            is_eaten = gstate == 2
            ghost_house_node = self.ghost_house_node_idx
            house_x = self.node_positions_x[ghost_house_node]
            house_y = self.node_positions_y[ghost_house_node]
            near_house = jnp.logical_and(
                jnp.abs(gx - house_x) < 8,
                jnp.abs(gy - house_y) < 8
            )
            at_house = jnp.logical_and(is_eaten, near_house)
            # Only respawn if frightened timer has expired
            fright_expired = state.frightened_timer <= 0
            can_respawn = jnp.logical_and(at_house, fright_expired)
            respawned_state = jnp.where(can_respawn, 0, gstate)
            # Waiting at house: eaten, at house, but fright still active
            waiting_at_house = jnp.logical_and(at_house, jnp.logical_not(fright_expired))
            
            # Target computation
            target_for_eaten = pick_closest_neighbor_to_coord(house_x, house_y)
            
            # Apply dynamic travel-vector flanking to ensure aggressive distinct intersection branches
            dir_dx = jnp.array([0, 0, 0, 1, -1, 0], dtype=jnp.int32)
            dir_dy = jnp.array([0, 0, -1, 0, 0, 1], dtype=jnp.int32)
            p_dir = state.player_direction.astype(jnp.int32)
            
            px_val = dir_dx[p_dir] * 32
            py_val = dir_dy[p_dir] * 32
            ortho_x = dir_dy[p_dir] * 32
            ortho_y = dir_dx[p_dir] * 32
            
            offset_x = jax.lax.switch(idx, [
                lambda: 0,         # Ghost 0: Exact Player position
                lambda: px_val,    # Ghost 1: 4 Tiles Ahead
                lambda: -px_val,   # Ghost 2: 4 Tiles Behind
                lambda: ortho_x    # Ghost 3: 4 Tiles Orthogonal
            ])
            offset_y = jax.lax.switch(idx, [
                lambda: 0,
                lambda: py_val,
                lambda: -py_val,
                lambda: ortho_y
            ])
            
            target_for_normal = pick_closest_neighbor_to_coord(state.player_x + offset_x, state.player_y + offset_y)
            target_for_frightened = new_target_from_random
            
            new_target_from_logic = jnp.where(
                is_eaten, target_for_eaten,
                jnp.where(gstate == 1, target_for_frightened, target_for_normal)
            )
            
            # Update target when at node
            new_target = jnp.where(at_node, new_target_from_logic, gtarget)
            
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
            
            # Freeze ghost at house if waiting for fright to expire
            new_x = jnp.where(waiting_at_house, house_x, new_x)
            new_y = jnp.where(waiting_at_house, house_y, new_y)
            new_direction = jnp.where(waiting_at_house, Action.NOOP, new_direction)
            new_current = jnp.where(waiting_at_house, ghost_house_node, new_current)
            new_target = jnp.where(waiting_at_house, ghost_house_node, new_target)
            target_x_pos = jnp.where(waiting_at_house, house_x, target_x_pos)
            target_y_pos = jnp.where(waiting_at_house, house_y, target_y_pos)
            
            return jnp.array([new_x, new_y, new_direction, respawned_state, target_x_pos, target_y_pos, new_current, new_target], dtype=jnp.int32)
        
        # Process all ghosts
        ghost_indices = jnp.arange(4)
        keys = jax.random.split(keys, 4)
        new_ghosts = jax.vmap(process_single_ghost)(ghost_indices, state.ghosts, keys)
        
        # Update state key
        new_key, _ = jax.random.split(keys[0])
        
        return state._replace(ghosts=new_ghosts, key=new_key)

    def _check_collisions(self, state: PacmanState) -> PacmanState:
        """Check ghost, pellet, and vitamin collisions for one or two players."""
        multiplayer_enabled = state.multiplayer_active == 1

        def ghost_collisions_for_player(player_x: chex.Array, player_y: chex.Array) -> chex.Array:
            player_left = player_x
            player_right = player_x + self.consts.PLAYER_SIZE[0]
            player_top = player_y
            player_bottom = player_y + self.consts.PLAYER_SIZE[1]

            def check_ghost_collision(ghost_data):
                gx, gy, _, _, _, _, _, _ = ghost_data
                ghost_left = gx
                ghost_right = gx + self.consts.GHOST_SIZE[0]
                ghost_top = gy
                ghost_bottom = gy + self.consts.GHOST_SIZE[1]
                return jnp.logical_and(
                    jnp.logical_and(player_left < ghost_right, player_right > ghost_left),
                    jnp.logical_and(player_top < ghost_bottom, player_bottom > ghost_top),
                )

            return jax.vmap(check_ghost_collision)(state.ghosts)

        player1_collisions = ghost_collisions_for_player(state.player_x, state.player_y)

        player2_collisions = jax.lax.cond(
            multiplayer_enabled,
            lambda _: ghost_collisions_for_player(state.player2_x, state.player2_y),
            lambda _: jnp.zeros_like(player1_collisions),
            operand=None,
        )

        collisions = jnp.logical_or(player1_collisions, player2_collisions)

        def process_collision(ghost_data, collided):
            _, _, _, gstate, _, _, _, _ = ghost_data

            new_state = jnp.where(
                jnp.logical_and(collided, gstate == 1),
                2,
                gstate,
            )

            life_lost = jnp.logical_and(collided, gstate == 0)

            ghost_eaten = jnp.logical_and(collided, gstate == 1)
            ghost_score = jnp.where(
                ghost_eaten,
                self.consts.GHOST_SCORE_BASE * (2 ** state.ghosts_eaten_count),
                0,
            )

            return new_state, life_lost, ghost_score, ghost_eaten

        results = jax.vmap(process_collision)(state.ghosts, collisions)
        new_ghost_states = results[0]
        any_life_lost = jnp.any(results[1])
        ghost_scores = jnp.sum(results[2])
        any_ghost_eaten = jnp.any(results[3])

        new_ghosts_eaten_count = jnp.where(
            any_ghost_eaten,
            state.ghosts_eaten_count + 1,
            state.ghosts_eaten_count,
        )

        new_ghosts = state.ghosts.at[:, 3].set(new_ghost_states)

        new_score = state.score + ghost_scores

        new_player_state = jnp.where(any_life_lost, 1, state.player_state)
        new_death_timer = jnp.where(any_life_lost, self.consts.DEATH_DURATION, state.death_timer)
        new_freeze_timer = jnp.where(any_ghost_eaten, self.consts.FREEZE_DURATION, state.freeze_timer)

        def tile_coords(player_x: chex.Array, player_y: chex.Array) -> Tuple[chex.Array, chex.Array]:
            tile_x = player_x // self.consts.TILE_SIZE
            tile_y = (player_y - 32) // self.consts.TILE_SIZE
            tile_x = jnp.clip(tile_x, 0, self.consts.MAZE_WIDTH - 1)
            tile_y = jnp.clip(tile_y, 0, self.consts.MAZE_HEIGHT - 1)
            return tile_x, tile_y

        def collect_tile(
            pellets_collected: chex.Array,
            tile_x: chex.Array,
            tile_y: chex.Array,
        ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
            tile_val = self.consts.MAZE_LAYOUT[tile_y, tile_x]
            already_collected = pellets_collected[tile_y, tile_x] == 1
            dot_collected = jnp.logical_and(tile_val == 2, jnp.logical_not(already_collected))
            power_pellet_collected = jnp.logical_and(tile_val == 3, jnp.logical_not(already_collected))
            pellet_collected = jnp.logical_or(dot_collected, power_pellet_collected)

            pellet_score = jnp.where(dot_collected, self.consts.PELLET_DOT_SCORE, 0)
            pellet_score = jnp.where(power_pellet_collected, self.consts.PELLET_POWER_SCORE, pellet_score)

            new_pellets_collected = jax.lax.cond(
                pellet_collected,
                lambda p: p.at[tile_y, tile_x].set(1),
                lambda p: p,
                pellets_collected,
            )

            return new_pellets_collected, pellet_collected, power_pellet_collected, pellet_score

        player1_tile_x, player1_tile_y = tile_coords(state.player_x, state.player_y)
        player2_tile_x, player2_tile_y = tile_coords(state.player2_x, state.player2_y)

        pellets_after_player1, p1_pellet_collected, p1_power_collected, p1_pellet_score = collect_tile(
            state.pellets_collected,
            player1_tile_x,
            player1_tile_y,
        )

        pellets_after_player2, p2_pellet_collected, p2_power_collected, p2_pellet_score = jax.lax.cond(
            multiplayer_enabled,
            lambda pellets: collect_tile(pellets, player2_tile_x, player2_tile_y),
            lambda pellets: (
                pellets,
                jnp.array(False),
                jnp.array(False),
                jnp.array(0, dtype=jnp.int32),
            ),
            pellets_after_player1,
        )

        pellets_collected_count = (
            p1_pellet_collected.astype(jnp.int32) + p2_pellet_collected.astype(jnp.int32)
        )
        power_pellet_eaten = jnp.logical_or(p1_power_collected, p2_power_collected)
        pellet_score_total = p1_pellet_score + p2_pellet_score

        new_score = new_score + pellet_score_total
        new_dots_remaining = state.dots_remaining - pellets_collected_count

        new_frightened_timer = jnp.where(
            power_pellet_eaten,
            self.consts.FRIGHTENED_DURATION,
            state.frightened_timer,
        )

        new_ghost_states_final = jnp.where(
            power_pellet_eaten,
            jnp.where(new_ghost_states == 2, 2, 1),
            new_ghost_states,
        )
        new_ghosts_final = new_ghosts.at[:, 3].set(new_ghost_states_final)

        final_ghosts_eaten_count = jnp.where(
            power_pellet_eaten,
            0,
            new_ghosts_eaten_count,
        )

        level_complete = jnp.logical_and(new_dots_remaining <= 0, state.level_transition_timer == 0)

        new_transition_timer = jnp.where(
            level_complete,
            self.consts.LEVEL_TRANSITION_DURATION,
            state.level_transition_timer,
        )

        new_total_pellets_eaten = state.total_pellets_eaten + pellets_collected_count

        should_spawn_vitamin = jnp.logical_and(
            new_total_pellets_eaten >= self.consts.VITAMIN_TRIGGER_PELLETS,
            jnp.logical_and(state.vitamin_collected == 0, state.vitamin_active == 0),
        )
        new_vitamin_active = jnp.where(should_spawn_vitamin, 1, state.vitamin_active)
        new_vitamin_timer = jnp.where(should_spawn_vitamin, self.consts.VITAMIN_DURATION, state.vitamin_timer)

        vitamin_dx_p1 = jnp.abs(state.player_x - state.vitamin_x)
        vitamin_dy_p1 = jnp.abs(state.player_y - state.vitamin_y)
        touching_vitamin_p1 = jnp.logical_and(vitamin_dx_p1 < 6, vitamin_dy_p1 < 6)

        vitamin_dx_p2 = jnp.abs(state.player2_x - state.vitamin_x)
        vitamin_dy_p2 = jnp.abs(state.player2_y - state.vitamin_y)
        touching_vitamin_p2 = jnp.logical_and(vitamin_dx_p2 < 6, vitamin_dy_p2 < 6)

        touching_vitamin = jnp.logical_or(
            touching_vitamin_p1,
            jnp.logical_and(multiplayer_enabled, touching_vitamin_p2),
        )

        vitamin_eaten = jnp.logical_and(new_vitamin_active == 1, touching_vitamin)

        new_score = jnp.where(vitamin_eaten, new_score + self.consts.VITAMIN_SCORE, new_score)
        new_vitamin_active = jnp.where(vitamin_eaten, 0, new_vitamin_active)
        new_vitamin_collected = jnp.where(vitamin_eaten, 1, state.vitamin_collected)

        return state._replace(
            ghosts=new_ghosts_final,
            score=new_score,
            lives=state.lives,
            frightened_timer=new_frightened_timer,
            ghosts_eaten_count=final_ghosts_eaten_count,
            dots_remaining=new_dots_remaining,
            pellets_collected=pellets_after_player2,
            level_transition_timer=new_transition_timer,
            player_state=new_player_state,
            death_timer=new_death_timer,
            freeze_timer=new_freeze_timer,
            total_pellets_eaten=new_total_pellets_eaten,
            vitamin_active=new_vitamin_active,
            vitamin_timer=new_vitamin_timer,
            vitamin_collected=new_vitamin_collected,
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
        
        # Update vitamin timer
        new_vitamin_timer = jnp.maximum(state.vitamin_timer - 1, 0)
        # Deactivate vitamin when timer expires
        new_vitamin_active = jnp.where(
            jnp.logical_and(state.vitamin_active == 1, new_vitamin_timer == 0),
            0,
            state.vitamin_active
        )

        return state._replace(
            ghosts=new_ghosts,
            frightened_timer=new_frightened,
            ghosts_eaten_count=final_ghosts_eaten_count,
            scatter_chase_timer=new_scatter_chase,
            is_scatter_mode=new_is_scatter,
            level_transition_timer=new_transition,
            vitamin_timer=new_vitamin_timer,
            vitamin_active=new_vitamin_active,
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
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "ghosts": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(4, 5), dtype=jnp.int32),
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
            shape=(250, 160, 3),  # (height, width, channels)
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
        # Fix: Updated dimensions to 250x160 (H, W)
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Pre-render static maze background (walls / HUD / etc)
        # We do this BEFORE loading assets so we can inject it
        if self.consts.MAZE_LAYOUT is not None:
            print("Pre-rendering maze background...")
            self.maze_background = self._create_maze_background()
        else:
            self.maze_background = None

        # Load assets
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/pacman"
        
        # Inject generated background into the asset config
        # We need to iterate and replace the 'background' entry
        final_asset_config = []
        for asset in self.consts.ASSET_CONFIG:
            if asset['type'] == 'background':
                new_asset = asset.copy()
                if self.maze_background is not None:
                    # Use our generated data
                    new_asset['data'] = self.maze_background
                    # Ensure 'file' is removed if present to avoid confusion/errors in utils
                    if 'file' in new_asset: 
                        del new_asset['file']
                final_asset_config.append(new_asset)
            else:
                final_asset_config.append(asset)

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        # Resize BACKGROUND to match new game dimensions if needed (should match now)
        bg_h, bg_w = self.BACKGROUND.shape[:2]
        target_h, target_w = self.config.game_dimensions
        if bg_h != target_h or bg_w != target_w:
            from scipy.ndimage import zoom
            zoom_h = target_h / bg_h
            zoom_w = target_w / bg_w
            self.BACKGROUND = jnp.array(zoom(self.BACKGROUND, (zoom_h, zoom_w), order=0).astype(np.uint8))
        
        # Find Wall and Background IDs dynamically from the loaded palette
        # Use COLOR_TO_ID to get correct indices
        
        # Note: BACKGROUND_COLOR might not be in the palette if _create_maze_background
        # used different RGB values. But typically it should be consistent.
        # Fallback to 0 if not found (though it should be found).
        bg_rgb = tuple(self.consts.BACKGROUND_COLOR)
        self.bg_id = jnp.array(self.COLOR_TO_ID.get(bg_rgb, 0), dtype=jnp.uint8)
        self.white_id = jnp.array(self.COLOR_TO_ID.get((255, 255, 255), 0), dtype=jnp.uint8)

    
    def _create_maze_background(self) -> jnp.ndarray:
        """Create static background with procedureal wall shapes (Atari Style)."""
        import numpy as np
        
        # Dimensions
        H, W = self.consts.HEIGHT, self.consts.WIDTH
        TILE = self.consts.TILE_SIZE # 8
        rows, cols = self.consts.MAZE_HEIGHT, self.consts.MAZE_WIDTH # 25, 20
        
        # Colors
        bg_color = np.array(self.consts.BACKGROUND_COLOR, dtype=np.uint8)
        wall_color = np.array(self.consts.WALL_COLOR, dtype=np.uint8) # Atari Blue
        hud_color = np.array([72, 176, 110], dtype=np.uint8) # Green
        
        # Initialize canvas with black background
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Draw Maze
        # User wants the maze roughly vertically centered. 
        # Screen is 250 high. Maze is 17 tiles * 8 = 136 pixels high.
        # Let's shift the maze down by y_offset = 32 pixels.
        y_offset = 32
        
        # Draw blue background only for the maze area
        maze_pixel_height = rows * TILE
        if y_offset + maze_pixel_height <= H:
            canvas[y_offset:y_offset+maze_pixel_height, :, :] = bg_color
        else:
            canvas[y_offset:, :, :] = bg_color

        # Helper to check bounds and get wall type
        layout = self.consts.MAZE_LAYOUT
        def is_wall(r, c):
            if 0 <= r < rows and 0 <= c < cols:
                val = int(layout[r, c])
                return val == 1 # Only generic walls connect
            return False

        # Draw Maze
        # User wants the maze roughly vertically centered. 
        # Screen is 250 high. Maze is 17 tiles * 8 = 136 pixels high.
        # Shift the maze down by y_offset = 32 pixels.
        # Shift the maze left by x_offset = -4 pixels to perfectly center 21-col map in 160px width.
        y_offset = 32
        x_offset = -4
        for r in range(rows):
            for c in range(cols):
                tile_val = int(layout[r, c])
                x, y = c * TILE + x_offset, r * TILE + y_offset
                
                x_start = max(0, x)
                x_end = min(W, x + TILE)
                
                if tile_val == 1 and x_start < x_end: # Wall
                    # Draw a solid block
                    canvas[y:y+TILE, x_start:x_end] = wall_color
                    
                elif tile_val == 4 and x_start < x_end: # Ghost House Box
                    # Draw a perfectly symmetrical enclosing box framing the 8x8 cell
                    # Outer bounds: 10px width, 14px height. Inner housing: 8x8.
                    # Left Wall (1px outer)
                    if x_start - 1 >= 0:
                         canvas[max(0, y-3):min(H, y+11), x_start-1:x_start] = wall_color
                    # Right Wall (1px outer)
                    if x_end + 1 <= W:
                         canvas[max(0, y-3):min(H, y+11), x_end:x_end+1] = wall_color
                    # Top Wall (3px outer)
                    if y - 3 >= 0:
                         canvas[max(0, y-3):y, max(0, x_start-1):min(W, x_end+1)] = wall_color
                    # Bottom Wall (3px outer)
                    if y + 11 <= H:
                         canvas[y+8:min(H, y+11), max(0, x_start-1):min(W, x_end+1)] = wall_color
                    
        # Draw HUD (Green Bar)
        hud_start = 172
        hud_end = 181
        if H >= hud_end:
            canvas[hud_start:hud_end, :, :] = hud_color
            
        # Add Alpha Channel
        alpha = np.full((H, W, 1), 255, dtype=np.uint8)
        canvas_rgba = np.concatenate([canvas, alpha], axis=-1)
        
        return jnp.array(canvas_rgba)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # 1. Render Maze & Sprites (Raster)
        # Init raster with background color (Includes Maze + Green HUD from _create_maze_background)
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Masks
        pellet_dot_mask = self.SHAPE_MASKS["pellet_dot"]
        pellet_power_masks = self.SHAPE_MASKS["pellet_power"]
        
        # Draw tiles (Pellets only in Raster - Walls are now in Background)
        def render_tile(i, raster_state):
            row = i // self.consts.MAZE_WIDTH
            col = i % self.consts.MAZE_WIDTH
            tile_val = self.consts.MAZE_LAYOUT[row, col]
            x = col * self.consts.TILE_SIZE - 4 # Apply -4 offset
            y = row * self.consts.TILE_SIZE + 32 # Apply 32 pixel offset
            
            # Pellets
            is_collected = state.pellets_collected[row, col] > 0
            
            def render_pellet(r):
                 return self.jr.render_at(r, x, y, pellet_dot_mask)
                 
            def render_power(r):
                anim_frame = (state.step_counter // 10) % 2
                is_frightened = state.frightened_timer > 0
                pp_mask = jax.lax.cond(
                    is_frightened,
                    lambda: self.SHAPE_MASKS["pellet_power_frightened"][anim_frame],
                    lambda: pellet_power_masks[anim_frame]
                )
                return self.jr.render_at(r, x, y, pp_mask)
            
            raster_state = jax.lax.cond(
                 jnp.logical_and(tile_val == 2, jnp.logical_not(is_collected)),
                 render_pellet,
                 lambda r: jax.lax.cond(
                     jnp.logical_and(tile_val == 3, jnp.logical_not(is_collected)),
                     render_power,
                     lambda rr: rr,
                     r
                 ),
                 raster_state
            )
            return raster_state

        raster = jax.lax.fori_loop(0, self.consts.MAZE_HEIGHT * self.consts.MAZE_WIDTH, render_tile, raster)
        
        # Render Entities (Raster)
        def render_alive(r):
            player_dir_idx = state.player_direction
            is_vertical = jnp.logical_or(player_dir_idx == Action.UP, player_dir_idx == Action.DOWN)
            sprite_dir_action = jnp.where(is_vertical, state.player_last_horizontal_dir, player_dir_idx)
            anim_step = (state.step_counter // self.consts.ANIMATION_SPEED) % 4
            player_frame = 2 - jnp.abs(anim_step - 2)
            base_sprite_idx = self.consts.SPRITE_LOOKUP[sprite_dir_action]
            player_sprite_idx = base_sprite_idx + player_frame
            player_mask = self.SHAPE_MASKS["player"][player_sprite_idx]
            return self.jr.render_at(r, state.player_x - 4, state.player_y, player_mask)

        def render_dying(r):
            progress = (self.consts.DEATH_DURATION - state.death_timer) / self.consts.DEATH_DURATION
            frame = (progress * 12).astype(jnp.int32)
            frame = jnp.clip(frame, 0, 11)
            death_mask = self.SHAPE_MASKS["player_death"][frame]
            return self.jr.render_at(r, state.player_x - 4, state.player_y, death_mask)

        raster = jax.lax.cond(state.player_state == 1, render_dying, render_alive, raster)

        def render_player2(r):
            player_dir_idx = state.player2_direction
            is_vertical = jnp.logical_or(player_dir_idx == Action.UP, player_dir_idx == Action.DOWN)
            sprite_dir_action = jnp.where(is_vertical, state.player2_last_horizontal_dir, player_dir_idx)
            anim_step = (state.step_counter // self.consts.ANIMATION_SPEED) % 4
            player_frame = 2 - jnp.abs(anim_step - 2)
            base_sprite_idx = self.consts.SPRITE_LOOKUP[sprite_dir_action]
            player_sprite_idx = base_sprite_idx + player_frame
            player_mask = self.SHAPE_MASKS["player"][player_sprite_idx]
            return self.jr.render_at(r, state.player2_x - 4, state.player2_y, player_mask)

        raster = jax.lax.cond(
            jnp.logical_and(state.multiplayer_active == 1, state.player_state == 0),
            render_player2,
            lambda r: r,
            raster,
        )

        def render_ghosts(r):
            def render_single_ghost(i, rr):
                g_x, g_y = state.ghosts[i, 0] - 4, state.ghosts[i, 1]
                g_dir = state.ghosts[i, 2]
                g_state = state.ghosts[i, 3] # 0=Normal, 1=Frightened, 2=Eaten
                anim_frame = (state.step_counter // 10) % 2
                
                # Desynchronized flashing: each ghost has a different phase offset
                flash_phase = (state.step_counter + i) % 2
                is_visible = flash_phase == 0  # Visible every other frame
                
                # Ghost Masks logic
                mask_normal = self.SHAPE_MASKS["ghost_normal"][anim_frame]
                mask_fright = self.SHAPE_MASKS["ghost_frightened"][anim_frame]
                mask_dead_eyes = self.SHAPE_MASKS["ghost_eyes"]
                mask_dead_eyes_pink = self.SHAPE_MASKS["ghost_eyes_pink"]
                # Pink eyes when eaten and fright expired (travelling home post-fright)
                fright_active = state.frightened_timer > 0
                mask_dead = jax.lax.cond(
                    fright_active,
                    lambda: mask_dead_eyes,
                    lambda: mask_dead_eyes_pink
                )
                final_mask = jax.lax.switch(
                    g_state.astype(jnp.int32),
                     [lambda: mask_normal, lambda: mask_fright, lambda: mask_dead]
                )
                # Apply flashing - only render if visible this frame
                return jax.lax.cond(is_visible, lambda r: self.jr.render_at(r, g_x, g_y, final_mask), lambda r: r, rr)
            return jax.lax.fori_loop(0, 4, render_single_ghost, r)
            
        raster = render_ghosts(raster)
        
        # Render vitamin bonus item
        def render_vitamin(r):
            vitamin_mask = self.SHAPE_MASKS["vitamin"]
            return self.jr.render_at(r, state.vitamin_x, state.vitamin_y, vitamin_mask)
        raster = jax.lax.cond(state.vitamin_active == 1, render_vitamin, lambda r: r, raster)
        
        # Convert to RGB (Background with Walls/HUD is already baked in)
        output = self.jr.render_from_palette(object_raster=raster, base_palette=self.PALETTE)

        # Draw Score RGB Overlay (Beige)
        # (This remains as overlay since we don't have this color in the palette typically, 
        # unless we add it to the background dummy pixels)
        
        # 4. SCORE (RGB Overlay)
        score = state.score
        score_x = 60
        score_y = 173
        digit_masks = self.SHAPE_MASKS["digits"]
        
        def draw_score_digit_rgb(i, out_img):
            divisor = jnp.power(10, i)
            digit = (score // divisor) % 10
            should_draw = jnp.logical_or(score >= divisor, i == 0)
            
            # Digit mask is 8x8 (0 or 1). We need a color.
            # Colored black as per standard Atari HUD style.
            text_color = jnp.array([0, 0, 0], dtype=jnp.uint8) # Black
            
            # Get the mask for this digit
            mask = digit_masks[digit] # (8, 8) or similar.
            # Extract only the exact drawn digit using the white_id
            mask_bool = (mask == self.white_id)[..., None] # (8, 8, 1)
            
            dx = score_x + (5 - i) * 8
            dy = score_y
            
            # Extract slice
            region = jax.lax.dynamic_slice(out_img, (dy, dx, 0), (8, 8, 3))
            
            # Draw text color where mask is true
            region = jnp.where(mask_bool, text_color, region)
            
            # Put back
            return jax.lax.cond(
                should_draw,
                lambda: jax.lax.dynamic_update_slice(out_img, region, (dy, dx, 0)),
                lambda: out_img
            )

        output = jax.lax.fori_loop(0, 6, draw_score_digit_rgb, output)
        
        # 4.5. LIVES INDICATOR
        life_color = jnp.array([72, 176, 110], dtype=jnp.uint8)
        
        def draw_life_rect(i, out_img):
            # User wants them to disappear left to right
            # If 3 lives: draw i=0, 1, 2
            # If 2 lives: draw i=1, 2 (i=0 drops off)
            # If 1 life: draw i=2 (i=0, 1 drop off)
            should_draw = (3 - state.lives) <= i
            
            life_x = 12 + i * 8
            life_y = 182         # Below green bar
            
            # Green vertical rectangle: 8 tall, 4 wide
            region = jnp.full((6, 4, 3), life_color, dtype=jnp.uint8)
            
            # Bounds check just in case
            return jax.lax.cond(
                should_draw,
                lambda: jax.lax.dynamic_update_slice(out_img, region, (life_y, life_x, 0)),
                lambda: out_img
            )

        output = jax.lax.fori_loop(0, 3, draw_life_rect, output)
        
        # 5. STRICT CROP (160x250)
        output = output[:250, :160, :]
        
        return output
