from typing import NamedTuple, Tuple
from functools import partial
from collections import deque
import os

import chex
import jax
import jax.numpy as jnp
import jax.image as jim
import numpy as np
from flax import struct

from jaxatari.environment import JaxEnvironment
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


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


@struct.dataclass
class MsPacmanConstants:
    screen_width: int = 160
    screen_height: int = 210
    cell_size: int = 4
    grid_width: int = 40
    grid_height: int = 42
    num_ghosts: int = 4
    # ALE-accurate scoring
    pellet_reward: int = 10
    power_pellet_reward: int = 50
    ghost_chain_rewards: Tuple[int, ...] = (200, 400, 800, 1600)
    # ALE-accurate timing
    frightened_duration: int = 360       # ~6 seconds at 60fps
    frightened_flash_start: int = 120    # start flashing ~2s before end
    initial_lives: int = 3
    ghost_spawn_delays: Tuple[int, ...] = (0, 60, 120, 180)  # per-ghost staggered release
    ghost_move_period: int = 6           # ghosts move every N frames
    player_move_period: int = 4          # pacman moves every N frames
    # Scatter/Chase mode timing (in frames)
    scatter_duration: int = 420          # ~7 seconds
    chase_duration: int = 1200           # ~20 seconds
    # Ghost pen position (center of maze, grid coords)
    pen_x: int = 19
    pen_y: int = 16
    pen_door_y: int = 14                 # y-coord of pen exit
    # Start-of-level delay
    start_delay: int = 60               # frames before movement begins
    # Death animation
    death_freeze_duration: int = 60     # frames of death animation
    # Fruit
    fruit_spawn_pellet_counts: Tuple[int, int] = (70, 170)
    fruit_move_period: int = 4          # fruit advances one tile every N frames
    fruit_score: int = 100              # cherry = 100 for level 1
    fruit_path_max: int = 128           # max length of fruit path (padded)
    # Power pellet blink
    power_pellet_blink_period: int = 15  # toggle every N frames
    # Colors
    background_color: Tuple[int, int, int] = (0, 0, 0)
    wall_color: Tuple[int, int, int] = (228, 111, 111)
    blocked_color: Tuple[int, int, int] = (0, 28, 136)
    pellet_color: Tuple[int, int, int] = (228, 111, 111)
    power_pellet_color: Tuple[int, int, int] = (228, 111, 111)
    pacman_color: Tuple[int, int, int] = (255, 255, 0)
    ghost_colors: Tuple[Tuple[int, int, int], ...] = (
        (200, 72, 72),     # Blinky (red)
        (252, 188, 252),    # Pinky (pink)
        (0, 255, 255),      # Inky (cyan)
        (180, 122, 48),     # Sue (orange)
    )
    # Scatter targets (corner grid positions for each ghost)
    scatter_targets: Tuple[Tuple[int, int], ...] = (
        (37, 0),    # Blinky -> top-right
        (2, 0),     # Pinky -> top-left
        (37, 41),   # Inky -> bottom-right
        (2, 41),    # Sue -> bottom-left
    )
    # Level selection and layouts
    level: int = 0
    maze_layouts: Tuple[Tuple[str, ...], ...] = ()
    # Kept for backward compat; when set, level 0 uses this
    maze_layout: Tuple[str, ...] = ()

# Level layouts
maze_layout_level1 = (
        "1000000000100000000000000000010000000001",
        "1023232320102323232332323232010232323201",
        "1030000030103000000000000003010300000301",
        "1030111030103011111111111103010301110301",
        "10S0111020102011111111111102010201110S01",
        "1030111030103011111111111103010301110301",
        "1030000030003000000000000003000300000301",
        "1023232323232323232332323232323232323201",
        "1000300030000030000000000300000300030001",
        "1110301030111030111111110301110301030111",
        "1110201020111020111111110201110201020111",
        "1110301030111030111111110301110301030111",
        "0000301030000030000000000300000301030000",
        "3323201023232323232332323232323201023233",
        "0000301000000030000000000300000001030000",
        "1110301111111030111111110301111111030111",
        "1110201111111020100G00010201111111020111",
        "1110301111111030100000010301111111030111",
        "0010300000000030100000010300000000030100",
        "0010232323232320100000010232323232320100",
        "0010300000000030100000010300000000030100",
        "1110301111111030100000010301111111030111",
        "1110201111111020100000010201111111020111",
        "1110301111111030111111110301111111030111",
        "0000301000000030000000000300000001030000",
        "3323201023232323232P32323232323201023233",
        "0000301030003000300000030003000301030000",
        "1110301030103010301111030103010301030111",
        "1110201020102010201111020102010201020111",
        "1110301030103010301111030103010301030111",
        "1000300030103010300000030103010300030001",
        "1023232320102010232332320102010232323201",
        "1030000030103010300000030103010300000301",
        "1030111030103000301111030003010301110301",
        "1020111020102323201111023232010201110201",
        "1030111030100000301111030000010301110301",
        "1030111030111110301111030111110301110301",
        "10S0111020111110201111020111110201110S01",
        "1030111030111110301111030111110301110301",
        "1030000030000000300000030000000300000301",
        "1023232323232323232332323232323232323201",
        "1000000000000000000000000000000000000001",
    )

maze_layout_level2 = (
        "1000000000000000000000000000000000000001",
        "1023232323232323232332323232323232323201",
        "1030000000003000000000000003000000000301",
        "1030111111103011111001111103011111110301",
        "10S0111111102011111001111102011111110S01",
        "1030111111103011111001111103011111110301",
        "1030001000003010000000000103000001000301",
        "1023201023232010232332320102323201023201",
        "1000301030003010300000030103000301030001",
        "1110301030103010301111030103010301030111",
        "1110201020102010201111020102010201020111",
        "1110301030103010301111030103010301030111",
        "1110301030103000300000030003010301030111",
        "1110201020102323232332323232010201020111",
        "1110301030100030000000000300010301030111",
        "0000300030111030111111110301110300030000",
        "3323232320111020100G00010201110232323233",
        "0030000030111030100000010301110300000300",
        "1030111030000030100000010300000301110301",
        "1020111023232320100000010232323201110201",
        "1030111030000030100000010300000301110301",
        "1030111030111030100000010301110301110301",
        "1020111020111020100000010201110201110201",
        "1030111030111030111111110301110301110301",
        "1030111030100030000000000300010301110301",
        "1020111020102323232P32323232010201110201",
        "1030111030103000000000000003010301110301",
        "1030000030003011111111111103000300000301",
        "1023232323232011111111111102323232323201",
        "1000003000003011111111111103000003000001",
        "1111103011103010000000000103011103011111",
        "1111102011102010232332320102011102011111",
        "1111103011103010300000030103011103011111",
        "1000003000103000301001030003010003000001",
        "1023232320102323201001023232010232323201",
        "1030000030103000001001000003010300000301",
        "1030111030103011111001111103010301110301",
        "10S0111020102011111001111102010201110S01",
        "1030111030103011111001111103010301110301",
        "0030000030103000000000000003010300000300",
        "3323232320102323232332323232010232323233",
        "0000000000100000000000000000010000000000",
    )

maze_layout_level3 = (
        "1000000000000010000000000100000000000001",
        "1023232323232010232332320102323232323201",
        "1030000030003010300000030103000300000301",
        "1030111030103000301111030003010301110301",
        "10S0111020102323201111023232010201110S01",
        "1030111030103000301111030003010301110301",
        "1030000030103010300000030103010300000301",
        "1023232320102010232332320102010232323201",
        "1000003000103010000000000103010003000001",
        "1111103011103011111001111103011103011111",
        "1111102011102011111001111102011102011111",
        "1111103011103011111001111103011103011111",
        "1000003000003000000000000003000003000001",
        "1023232323232323232332323232323232323201",
        "1030000030000030000000000300000300000301",
        "1030111030111030111111110301110301110301",
        "1020111020111020100G00010201110201110201",
        "1030111030111030100000010301110301110301",
        "1030100030001030100000010301000300010301",
        "1020102323201020100000010201023232010201",
        "1030103000301030100000010301030003010301",
        "1030103010301030100000010301030103010301",
        "1020102010201020100000010201020102010201",
        "1030103010301030111111110301030103010301",
        "0030003010300030000000000300030103000300",
        "3323232010232323232P32323232320102323233",
        "0000003010003000300000030003000103000000",
        "1111103011103010301111030103011103011111",
        "1111102011102010201111020102011102011111",
        "1111103011103010301111030103011103011111",
        "1000003000003010300000030103000003000001",
        "1023232323232010232332320102323232323201",
        "1030003000003010300000030103000003000301",
        "1030103011103000301111030003011103010301",
        "1020102011102323201111023232011102010201",
        "1030103011103000301111030003011103010301",
        "1030103011103010301111030103011103010301",
        "10S0102011102010201111020102011102010S01",
        "1030103011103010301111030103011103010301",
        "1030003000003010300000030103000003000301",
        "1023232323232010232332320102323232323201",
        "1000000000000010000000000100000000000001",
    )

maze_layout_level4 = (
        "1000000000000000000000000000000000000001",
        "1023232323232323232332323232323232323201",
        "1030003000003000000000000003000003000301",
        "1030103011103011111111111103011103010301",
        "10S0102011102011111111111102011102010S01",
        "1030103011103011111111111103011103010301",
        "1030103000003010000000000103000003010301",
        "1020102323232010232332320102323232010201",
        "1030100030003010300000030103000300010301",
        "1030111030103010301111030103010301110301",
        "1020111020102010201111020102010201110201",
        "1030111030103010301111030103010301110301",
        "1030000030103000300000030003010300000301",
        "1023232320102323232332323232010232323201",
        "1000300030100030000000000300010300030001",
        "1110301030111030111111110301110301030111",
        "1110201020111020100G00010201110201020111",
        "1110301030111030100000010301110301030111",
        "0000301030000030100000010300000301030000",
        "3323201023232320100000010232323201023233",
        "0000001030000030100000010300000301000000",
        "1111111030111030100000010301110301111111",
        "1111111020111020100000010201110201111111",
        "1111111030111030111111110301110301111111",
        "0000001030100030000000000300010301000000",
        "3323201020102323232P32323232010201023233",
        "0000301030103000300000030003010301030000",
        "1110301030103010301111030103010301030111",
        "1110201020102010201111020102010201020111",
        "1110301030103010301111030103010301030111",
        "1000300030003010300000030103000300030001",
        "1023232323232010232332320102323232323201",
        "1030003000003010000000000103000003000301",
        "1030103011103011111001111103011103010301",
        "1020102011102011111001111102011102010201",
        "1030103011103011111001111103011103010301",
        "1030103011103010000000000103011103010301",
        "10S0102011102010232332320102011102010S01",
        "1030103011103010300000030103011103010301",
        "1030003000003000301111030003000003000301",
        "1023232323232323201111023232323232323201",
        "1000000000000000001111000000000000000001",
    )

# Create default constants instance
DEFAULT_MSPACMAN_CONSTANTS = MsPacmanConstants(
    level=0,
    maze_layout=maze_layout_level1,
    maze_layouts=(
        maze_layout_level1,
        maze_layout_level2,
        maze_layout_level3,
        maze_layout_level4,
    ),
)
# Define the game board's Y-offset and the UI's Y-offset
GAME_BOARD_OFFSET_Y = 0
UI_OFFSET_Y = 176 # Based on 44 rows * 4 cell_size = 176 pixels

class MsPacmanState(NamedTuple):
    pacman_x: chex.Array
    pacman_y: chex.Array
    direction: chex.Array            # [dx, dy] current movement direction
    buffered_direction: chex.Array   # [dx, dy] queued input direction
    ghost_positions: chex.Array      # (num_ghosts, 2) grid positions
    ghost_directions: chex.Array     # (num_ghosts, 2) last movement direction per ghost
    ghost_modes: chex.Array          # (num_ghosts,) 0=normal, 1=frightened, 2=eaten, 3=in_pen
    ghost_global_mode: chex.Array    # 0=scatter, 1=chase
    mode_timer: chex.Array           # frames until next scatter/chase toggle
    ghosts_eaten_count: chex.Array   # chain counter per power pellet (0-4)
    pellets: chex.Array
    power_timer: chex.Array
    score: chex.Array
    time: chex.Array
    lives: chex.Array
    pellets_remaining: chex.Array
    pellets_eaten: chex.Array        # total pellets eaten (for fruit trigger)
    game_over: chex.Array
    game_phase: chex.Array           # 0=start_screen, 1=playing, 2=game_over
    start_timer: chex.Array          # countdown before movement allowed
    death_timer: chex.Array          # countdown during death animation
    level_idx: chex.Array            # current level index
    fruit_active: chex.Array         # 1 if fruit is currently visible
    fruit_path_idx: chex.Array       # current index along fruit path
    fruit_pos_x: chex.Array         # current fruit grid x
    fruit_pos_y: chex.Array         # current fruit grid y
    fruit_spawned_count: chex.Array  # how many fruits have spawned this level (0, 1, or 2)
    key: chex.Array                  # PRNG key


def _parse_layout(layout: Tuple[str, ...], expected_ghosts: int):
    wall_rows = []
    pellet_rows = []
    pacman_spawn = None
    ghost_positions: list[Tuple[int, int]] = []

    for y, row in enumerate(layout):
        wall_row = []
        pellet_row = []
        for x, char in enumerate(row):
            if char in ("#", "1"):
                wall_row.append(1)
                pellet_row.append(0)
            elif char == "0":
                # Visually corridor but blocked for player movement (pellet=-1 signals visual)
                wall_row.append(1)
                pellet_row.append(-1)
            elif char in ("2", "3", "S"):
                # Walkable path (2=pellet path, 3=no-pellet path, S=power pellet)
                wall_row.append(0)
                pellet_row.append(0)
            elif char == ".":
                wall_row.append(0)
                pellet_row.append(1)
            elif char == "o":
                wall_row.append(0)
                pellet_row.append(2)
            elif char == "P":
                pacman_spawn = (x, y)
                wall_row.append(0)
                pellet_row.append(0)
            elif char == "G":
                ghost_positions.append((x, y))
                wall_row.append(0)
                pellet_row.append(0)
            else:
                wall_row.append(0)
                pellet_row.append(0)
        wall_rows.append(wall_row)
        pellet_rows.append(pellet_row)

    if pacman_spawn is None:
        pacman_spawn = (1, 1)

    if not ghost_positions:
        ghost_positions.append(pacman_spawn)

    while len(ghost_positions) < expected_ghosts:
        ghost_positions.append(ghost_positions[-1])

    ghost_positions = ghost_positions[:expected_ghosts]
    pellet_count = sum(1 for row in pellet_rows for cell in row if cell > 0)

    return (
        jnp.array(wall_rows, dtype=jnp.int32),
        jnp.array(pellet_rows, dtype=jnp.int32),
        jnp.array(pacman_spawn, dtype=jnp.int32),
        jnp.array(ghost_positions, dtype=jnp.int32),
        jnp.array(pellet_count, dtype=jnp.int32),
    )


class JaxMsPacman(JaxEnvironment[MsPacmanState, MsPacmanObservation, MsPacmanInfo, MsPacmanConstants]):
    def __init__(self, consts: MsPacmanConstants = None, reward_funcs: list[callable] = None):
        consts = consts or DEFAULT_MSPACMAN_CONSTANTS
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        # Select layout by level (fallback to legacy maze_layout for level 0)
        layouts = self.consts.maze_layouts or (self.consts.maze_layout,)
        level_idx = min(max(int(self.consts.level), 0), len(layouts) - 1)
        self.selected_layout = layouts[level_idx]

        # Precompute per-level assets
        level_wall_grids = []
        level_pellet_templates = []
        level_initial_pellets = []
        level_pac_spawns = []
        level_ghost_spawns = []
        level_pellet_counts = []
        level_ghost_wall_grids = []
        level_fruit_paths = []
        level_fruit_path_lens = []

        # Use enumerate to track the level index (0, 1, 2, 3...)
        for level_idx_iter, layout in enumerate(layouts):
            wall_grid, pellets_raw, pac_spawn, ghost_spawns, pellet_count_raw = _parse_layout(layout, self.consts.num_ghosts)
            # Pellet template keeps -1 markers
            pellet_template = pellets_raw
            # Initial pellets (0/1/2)
            grid_h, grid_w = wall_grid.shape
            pellets_grid = jnp.zeros((grid_h, grid_w), dtype=jnp.int32)
            for y, row in enumerate(layout):
                for x, char in enumerate(row):
                    if char == '2':
                        pellets_grid = pellets_grid.at[y, x].set(1)
                    elif char == 'S':
                        pellets_grid = pellets_grid.at[y, x].set(2)
            # Ghost-walkable grid: same as wall_grid but pen area is passable
            ghost_wall_grid = wall_grid.at[16:21, 17:22].set(0)
            ghost_wall_grid = ghost_wall_grid.at[13:16, 19].set(0)
            
            # Pass the level index into our function!
            f_path, f_len = self._compute_fruit_path_for_grid(wall_grid, level_idx_iter)

            level_wall_grids.append(wall_grid)
            level_pellet_templates.append(pellet_template)
            level_pac_spawns.append(pac_spawn)
            level_ghost_spawns.append(ghost_spawns)
            level_initial_pellets.append(pellets_grid)
            level_pellet_counts.append(jnp.sum(pellets_grid > 0))
            level_ghost_wall_grids.append(ghost_wall_grid)
            level_fruit_paths.append(f_path)
            level_fruit_path_lens.append(f_len)

        self.level_wall_grids = jnp.stack(level_wall_grids)
        self.level_pellet_templates = jnp.stack(level_pellet_templates)
        self.level_pac_spawns = jnp.stack(level_pac_spawns)
        self.level_ghost_spawns = jnp.stack(level_ghost_spawns)
        self.level_initial_pellets = jnp.stack(level_initial_pellets)
        self.level_pellet_counts = jnp.stack(level_pellet_counts)
        self.level_ghost_wall_grids = jnp.stack(level_ghost_wall_grids)
        self.level_fruit_paths = jnp.stack(level_fruit_paths)
        self.level_fruit_path_lens = jnp.stack(level_fruit_path_lens)

        self.num_levels = self.level_wall_grids.shape[0]

        # Select starting level assets
        self.wall_grid = self.level_wall_grids[level_idx]
        self.initial_pellets_raw = self.level_pellet_templates[level_idx]
        self.initial_pellets = self.level_initial_pellets[level_idx]
        self.pacman_spawn = self.level_pac_spawns[level_idx]
        self.ghost_spawn_positions = self.level_ghost_spawns[level_idx]
        self.initial_pellet_count_raw = self.level_pellet_counts[level_idx]
        self.fruit_path = self.level_fruit_paths[level_idx]
        self.fruit_path_len = self.level_fruit_path_lens[level_idx]

        # Preserve visual template (contains -1 markers for blocked-but-path-colored tiles)
        self.pellet_template = self.initial_pellets_raw

        # Override ghost spawn positions: Blinky outside pen, others inside pen
        pen_x = self.consts.pen_x
        pen_y = self.consts.pen_y
        pen_door_y = self.consts.pen_door_y
        self.ghost_spawn_positions = jnp.array([
            [pen_x, pen_door_y],       # Blinky: just at pen door
            [pen_x - 1, pen_y],        # Pinky: left in pen
            [pen_x, pen_y],            # Inky: center of pen
            [pen_x + 1, pen_y],        # Sue: right in pen
        ], dtype=jnp.int32)

        # Generate pellets using maze layout '2' and 'S' markers
        grid_h, grid_w = self.wall_grid.shape
        pellets = jnp.zeros((grid_h, grid_w), dtype=jnp.int32)
        for y, row in enumerate(self.selected_layout):
            for x, char in enumerate(row):
                if char == '2':
                    pellets = pellets.at[y, x].set(1)
                elif char == 'S':
                    pellets = pellets.at[y, x].set(2)

        # Already per-level
        self.initial_pellets = self.initial_pellets
        self.initial_pellet_count = int(self.level_pellet_counts[level_idx])

        # Ghost-walkable grid: same as wall_grid but pen area is passable (selected level)
        self.ghost_wall_grid = self.level_ghost_wall_grids[level_idx]

        # Scatter targets as jnp array for ghost AI
        self.scatter_targets = jnp.array(self.consts.scatter_targets, dtype=jnp.int32)
        self.ghost_spawn_delays = jnp.array(self.consts.ghost_spawn_delays, dtype=jnp.int32)
        self.ghost_chain_rewards = jnp.array(self.consts.ghost_chain_rewards, dtype=jnp.int32)

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

        # Renderer uses stacked per-level grids and will select per state.level_idx
        self.renderer = MsPacmanRenderer(
            self.consts,
            self.level_wall_grids,
            self.level_pellet_templates,
        )

    def _compute_fruit_path_for_grid(self, wall_grid, level_idx):
        """Compute fruit path using BFS. Uses full path for Level 1, and just start/end for the rest."""
        wall_np = np.array(wall_grid)
        h, w = wall_np.shape

        def _bfs(start, goal):
            q = deque([(start, [start])])
            vis = {start}
            while q:
                (cx, cy), path = q.popleft()
                if (cx, cy) == goal:
                    return path
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and wall_np[ny, nx] == 0 and (nx, ny) not in vis:
                        vis.add((nx, ny))
                        q.append(((nx, ny), path + [(nx, ny)]))
            return None

        # Level 0 keeps the classic U-shape. 
        # Levels 1, 2, and 3 only have a Start and End coordinate! 
        # The BFS will automatically calculate the shortest valid path between them!
        level_waypoints = {
            0: [(38, 13), (30, 13), (30, 25), (4, 25), (4, 13), (1, 13)],  
            1: [(38, 16), (1, 16)],  # Level 2 (Just start and end)
            2: [(38, 25), (1, 25)],  # Level 3 (Just start and end)
            3: [(38, 19), (1, 19)],  # Level 4 (Just start and end)
        }

        # Fetch the waypoints for this level, fallback to Level 1 if not found
        waypoints = level_waypoints.get(level_idx, level_waypoints[0])
        
        segments = []
        for i in range(len(waypoints) - 1):
            seg = _bfs(waypoints[i], waypoints[i + 1])
            if seg is None:
                # Safe Fallback: Just sit at the start point if the pathfinder hits a wall
                fallback = [waypoints[0]]
                path_arr = np.array(fallback, dtype=np.int32)
                padded = np.zeros((self.consts.fruit_path_max, 2), dtype=np.int32)
                padded[0] = path_arr[0]
                return jnp.array(padded), 1
            segments.append(seg if i == 0 else seg[1:]) 

        full_path = []
        for seg in segments:
            full_path.extend(seg)

        # Pad the path array up to 128 steps to satisfy JAX static shapes
        path_arr = np.array(full_path, dtype=np.int32)
        plen = min(len(full_path), self.consts.fruit_path_max)
        padded = np.zeros((self.consts.fruit_path_max, 2), dtype=np.int32)
        padded[:plen] = path_arr[:plen]
        return jnp.array(padded), plen
    
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[MsPacmanObservation, MsPacmanState]:
        # Ghost initial modes: 0=in_pen for ghosts 1-3, 0=normal for ghost 0 (Blinky)
        ghost_modes = jnp.array([0, 3, 3, 3], dtype=jnp.int32)
        ghost_directions = jnp.zeros((self.consts.num_ghosts, 2), dtype=jnp.int32)

        # Starting level index
        level_idx = jnp.array(min(max(int(self.consts.level), 0), self.num_levels - 1), dtype=jnp.int32)

        state = MsPacmanState(
            pacman_x=self.pacman_spawn[0],
            pacman_y=self.pacman_spawn[1],
            direction=jnp.array([-1, 0], dtype=jnp.int32),
            buffered_direction=jnp.array([0, 0], dtype=jnp.int32),
            ghost_positions=self.ghost_spawn_positions,
            ghost_directions=ghost_directions,
            ghost_modes=ghost_modes,
            ghost_global_mode=jnp.array(0, dtype=jnp.int32),  # start in scatter
            mode_timer=jnp.array(self.consts.scatter_duration, dtype=jnp.int32),
            ghosts_eaten_count=jnp.array(0, dtype=jnp.int32),
            pellets=self.initial_pellets,
            power_timer=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.initial_lives, dtype=jnp.int32),
            pellets_remaining=jnp.array(self.initial_pellet_count, dtype=jnp.int32),
            pellets_eaten=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            game_phase=jnp.array(1, dtype=jnp.int32),  # Start playing directly (ALE behavior)
            start_timer=jnp.array(self.consts.start_delay, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            level_idx=jnp.array(level_idx, dtype=jnp.int32),
            fruit_active=jnp.array(0, dtype=jnp.int32),
            fruit_path_idx=jnp.array(0, dtype=jnp.int32),
            fruit_pos_x=jnp.array(0, dtype=jnp.int32),
            fruit_pos_y=jnp.array(0, dtype=jnp.int32),
            fruit_spawned_count=jnp.array(0, dtype=jnp.int32),
            key=key,
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: MsPacmanState, action: int) -> Tuple[MsPacmanObservation, MsPacmanState, float, bool, MsPacmanInfo]:
        action = jnp.asarray(action, dtype=jnp.int32)
        new_key, step_key = jax.random.split(state.key)
        state = state._replace(key=step_key)

        # If game is over, just return
        def game_over_fn():
            return self._get_observation(state), state._replace(key=new_key), jnp.float32(0.0), True, self._get_info(state)

        def playing_fn():
            return self._step_game(state, action, new_key)

        return jax.lax.cond(
            state.game_over,
            game_over_fn,
            playing_fn,
        )

    def _step_game(self, state: MsPacmanState, action: int, new_key: chex.Array) -> Tuple[MsPacmanObservation, MsPacmanState, float, bool, MsPacmanInfo]:
        # --- Start delay: no movement allowed ---
        in_start_delay = state.start_timer > 0
        start_timer = jnp.maximum(state.start_timer - 1, 0)

        # --- Death animation: no movement, just count down ---
        in_death = state.death_timer > 0
        death_timer = jnp.maximum(state.death_timer - 1, 0)

        frozen = jnp.logical_or(in_start_delay, in_death)

        # --- Parse action into direction delta ---
        action_idx = jnp.clip(action, 0, self._action_deltas.shape[0] - 1)
        delta = self._action_deltas[action_idx]
        has_new_direction = jnp.any(delta != 0)

        # --- Input buffering: store desired direction ---
        buffered_direction = jnp.where(has_new_direction, delta, state.buffered_direction)

        # --- Player movement ---
        can_move_player = jnp.logical_and(
            (state.time % jnp.maximum(self.consts.player_move_period, 1)) == 0,
            ~frozen,
        )

        # Try buffered direction first
        buf_x = jnp.mod(state.pacman_x + buffered_direction[0], self.consts.grid_width)
        buf_y = jnp.clip(state.pacman_y + buffered_direction[1], 0, self.consts.grid_height - 1)
        # Use level-specific wall grid
        wall_grid = self.level_wall_grids[state.level_idx]
        ghost_wall_grid = self.level_ghost_wall_grids[state.level_idx]
        fruit_path = self.level_fruit_paths[state.level_idx]
        fruit_path_len = self.level_fruit_path_lens[state.level_idx]
        buf_blocked = wall_grid[buf_y, buf_x] == 1

        # If buffered direction works, use it; otherwise try current direction
        use_buffered = jnp.logical_and(has_new_direction | jnp.any(buffered_direction != 0), ~buf_blocked)
        direction = jnp.where(use_buffered, buffered_direction, state.direction)

        # Clear buffer if we used it
        buffered_direction = jnp.where(use_buffered, jnp.array([0, 0], dtype=jnp.int32), buffered_direction)

        tentative_x = jnp.mod(state.pacman_x + direction[0], self.consts.grid_width)
        tentative_y = jnp.clip(state.pacman_y + direction[1], 0, self.consts.grid_height - 1)

        hit_wall = wall_grid[tentative_y, tentative_x] == 1
        pacman_x = jnp.where(hit_wall, state.pacman_x, tentative_x)
        pacman_y = jnp.where(hit_wall, state.pacman_y, tentative_y)

        # Only apply movement if can_move
        pacman_x = jnp.where(can_move_player, pacman_x, state.pacman_x)
        pacman_y = jnp.where(can_move_player, pacman_y, state.pacman_y)

        # --- Pellet collection ---
        pellet_value = state.pellets[pacman_y, pacman_x]
        actually_moved = jnp.logical_or(pacman_x != state.pacman_x, pacman_y != state.pacman_y)
        ate_pellet = jnp.logical_and(pellet_value > 0, actually_moved)
        ate_power = jnp.logical_and(pellet_value == 2, actually_moved)

        pellets = state.pellets.at[pacman_y, pacman_x].set(
            jnp.where(ate_pellet, 0, pellet_value)
        )

        pellet_reward = jnp.where(
            ate_power,
            self.consts.power_pellet_reward,
            jnp.where(ate_pellet, self.consts.pellet_reward, 0),
        )
        score = state.score + pellet_reward
        pellets_remaining = jnp.maximum(
            state.pellets_remaining - ate_pellet.astype(jnp.int32), 0,
        )
        pellets_eaten = state.pellets_eaten + ate_pellet.astype(jnp.int32)

        # --- Power pellet / frightened mode ---
        decayed_timer = jnp.maximum(state.power_timer - 1, 0)
        power_timer = jnp.where(
            ate_power,
            jnp.array(self.consts.frightened_duration, dtype=jnp.int32),
            decayed_timer,
        )
        frightened = power_timer > 0

        # Reset ghost eaten chain when new power pellet eaten
        ghosts_eaten_count = jnp.where(
            ate_power, jnp.array(0, dtype=jnp.int32), state.ghosts_eaten_count
        )

        # Update ghost modes: set normal ghosts to frightened when power pellet eaten
        ghost_modes = state.ghost_modes
        ghost_modes = jnp.where(
            jnp.logical_and(ate_power, ghost_modes == 0),  # normal -> frightened
            jnp.ones_like(ghost_modes),
            ghost_modes,
        )
        # End frightened when timer runs out
        ghost_modes = jnp.where(
            jnp.logical_and(~frightened, ghost_modes == 1),  # frightened -> normal
            jnp.zeros_like(ghost_modes),
            ghost_modes,
        )

        # --- Scatter/Chase mode timer ---
        mode_timer = state.mode_timer - 1
        toggle_mode = mode_timer <= 0
        ghost_global_mode = jnp.where(
            toggle_mode,
            1 - state.ghost_global_mode,  # toggle between 0 and 1
            state.ghost_global_mode,
        )
        new_duration = jnp.where(
            ghost_global_mode == 0,
            self.consts.scatter_duration,
            self.consts.chase_duration,
        )
        mode_timer = jnp.where(toggle_mode, new_duration, mode_timer)

        # --- Ghost movement ---
        can_move_ghosts = jnp.logical_and(
            (state.time % jnp.maximum(self.consts.ghost_move_period, 1)) == 0,
            ~frozen,
        )
        ghost_positions, ghost_directions, ghost_modes = self._move_ghosts(
            state.ghost_positions, state.ghost_directions, ghost_modes,
            pacman_x, pacman_y, direction,
            ghost_global_mode, can_move_ghosts, state.time, state.key, ghost_wall_grid,
        )

        # --- Ghost-Pacman collision ---
        # Direct overlap: both on same tile after movement
        ghost_overlap = jnp.logical_and(
            ghost_positions[:, 0] == pacman_x,
            ghost_positions[:, 1] == pacman_y,
        )
        # Swap-through: pac moved to ghost's old tile AND ghost moved to pac's old tile
        swapped = jnp.logical_and(
            jnp.logical_and(
                state.ghost_positions[:, 0] == pacman_x,
                state.ghost_positions[:, 1] == pacman_y,
            ),
            jnp.logical_and(
                ghost_positions[:, 0] == state.pacman_x,
                ghost_positions[:, 1] == state.pacman_y,
            ),
        )
        ghost_overlap = jnp.logical_or(ghost_overlap, swapped)

        # Eat frightened ghosts
        ghosts_eaten = jnp.logical_and(ghost_overlap, ghost_modes == 1)
        num_eaten_this_step = jnp.sum(ghosts_eaten.astype(jnp.int32))

        # Chain scoring: 200, 400, 800, 1600 for consecutive ghosts
        def compute_ghost_bonus(carry, eaten):
            total_score, chain_idx = carry
            bonus = jnp.where(
                eaten,
                self.ghost_chain_rewards[jnp.minimum(chain_idx, 3)],
                jnp.array(0, dtype=jnp.int32),
            )
            new_chain = jnp.where(eaten, chain_idx + 1, chain_idx)
            return (total_score + bonus, new_chain), None

        (ghost_bonus, ghosts_eaten_count), _ = jax.lax.scan(
            compute_ghost_bonus,
            (jnp.array(0, dtype=jnp.int32), ghosts_eaten_count),
            ghosts_eaten,
        )
        score = score + ghost_bonus

        # Send eaten ghosts back to pen (mode=2)
        ghost_modes = jnp.where(ghosts_eaten, jnp.full_like(ghost_modes, 2), ghost_modes)

        # Check for lethal collision (non-frightened, non-eaten ghost)
        lethal_ghost = jnp.logical_and(ghost_overlap, ghost_modes == 0)
        hit_without_power = jnp.logical_and(jnp.any(lethal_ghost), ~in_death)

        # --- Death handling ---
        lives = state.lives - hit_without_power.astype(jnp.int32)
        death_timer = jnp.where(
            hit_without_power,
            jnp.array(self.consts.death_freeze_duration, dtype=jnp.int32),
            death_timer,
        )
        # After death animation ends, reset positions
        death_just_ended = jnp.logical_and(state.death_timer == 1, ~in_start_delay)
        pacman_x = jnp.where(death_just_ended, self.pacman_spawn[0], pacman_x)
        pacman_y = jnp.where(death_just_ended, self.pacman_spawn[1], pacman_y)
        ghost_positions = jnp.where(
            death_just_ended,
            self.ghost_spawn_positions,
            ghost_positions,
        )
        ghost_modes = jnp.where(
            death_just_ended,
            jnp.array([0, 3, 3, 3], dtype=jnp.int32),
            ghost_modes,
        )
        power_timer = jnp.where(death_just_ended, jnp.array(0, dtype=jnp.int32), power_timer)
        direction = jnp.where(
            death_just_ended,
            jnp.array([-1, 0], dtype=jnp.int32),
            direction,
        )
        start_timer = jnp.where(
            death_just_ended,
            jnp.array(self.consts.start_delay, dtype=jnp.int32),
            start_timer,
        )

        # On hit, freeze immediately
        pacman_x = jnp.where(hit_without_power, state.pacman_x, pacman_x)
        pacman_y = jnp.where(hit_without_power, state.pacman_y, pacman_y)

        # --- Fruit spawning & movement ---
        fruit_active = state.fruit_active
        fruit_path_idx = state.fruit_path_idx
        fruit_pos_x = state.fruit_pos_x
        fruit_pos_y = state.fruit_pos_y
        fruit_spawned_count = state.fruit_spawned_count

        # Check if we should spawn a fruit
        should_spawn_first = jnp.logical_and(
            pellets_eaten >= self.consts.fruit_spawn_pellet_counts[0],
            fruit_spawned_count == 0,
        )
        should_spawn_second = jnp.logical_and(
            pellets_eaten >= self.consts.fruit_spawn_pellet_counts[1],
            fruit_spawned_count == 1,
        )
        should_spawn = jnp.logical_or(should_spawn_first, should_spawn_second)
        fruit_active = jnp.where(should_spawn, jnp.array(1, dtype=jnp.int32), fruit_active)
        fruit_path_idx = jnp.where(should_spawn, jnp.array(0, dtype=jnp.int32), fruit_path_idx)
        # Set initial position from path start
        spawn_pos = fruit_path[0]
        fruit_pos_x = jnp.where(should_spawn, spawn_pos[0], fruit_pos_x)
        fruit_pos_y = jnp.where(should_spawn, spawn_pos[1], fruit_pos_y)
        fruit_spawned_count = fruit_spawned_count + should_spawn.astype(jnp.int32)

        # Move fruit along path
        can_move_fruit = jnp.logical_and(
            fruit_active == 1,
            (state.time % jnp.maximum(self.consts.fruit_move_period, 1)) == 0,
        )
        new_path_idx = jnp.minimum(fruit_path_idx + 1, self.consts.fruit_path_max - 1)
        fruit_path_idx = jnp.where(can_move_fruit, new_path_idx, fruit_path_idx)
        # Look up position from path
        path_pos = fruit_path[fruit_path_idx]
        fruit_pos_x = jnp.where(fruit_active == 1, path_pos[0], fruit_pos_x)
        fruit_pos_y = jnp.where(fruit_active == 1, path_pos[1], fruit_pos_y)

        # Fruit reached end of path → disappears
        fruit_active = jnp.where(
            fruit_path_idx >= fruit_path_len - 1,
            jnp.array(0, dtype=jnp.int32),
            fruit_active,
        )

        # --- Level progression when pellets cleared ---
        level_complete = pellets_remaining == 0
        next_level_idx = jnp.minimum(state.level_idx + 1, self.num_levels - 1)
        level_idx = jnp.where(level_complete, next_level_idx, state.level_idx)

        # Select assets for (potentially) new level
        next_pellets = self.level_initial_pellets[level_idx]
        next_pellet_count = self.level_pellet_counts[level_idx]
        next_pac_spawn = self.level_pac_spawns[level_idx]
        next_ghost_spawn = self.level_ghost_spawns[level_idx]
        next_fruit_path = self.level_fruit_paths[level_idx]
        next_fruit_path_len = self.level_fruit_path_lens[level_idx]
        next_ghost_wall_grid = self.level_ghost_wall_grids[level_idx]

        # Reset per-level state on level completion
        pacman_x = jnp.where(level_complete, next_pac_spawn[0], pacman_x)
        pacman_y = jnp.where(level_complete, next_pac_spawn[1], pacman_y)
        direction = jnp.where(level_complete, jnp.array([-1, 0], dtype=jnp.int32), direction)
        buffered_direction = jnp.where(level_complete, jnp.array([0, 0], dtype=jnp.int32), buffered_direction)
        ghost_positions = jnp.where(level_complete, next_ghost_spawn, ghost_positions)
        ghost_directions = jnp.where(level_complete, jnp.zeros_like(ghost_directions), ghost_directions)
        ghost_modes = jnp.where(level_complete, jnp.array([0,3,3,3], dtype=jnp.int32), ghost_modes)
        ghost_global_mode = jnp.where(level_complete, jnp.array(0, dtype=jnp.int32), ghost_global_mode)
        mode_timer = jnp.where(level_complete, jnp.array(self.consts.scatter_duration, dtype=jnp.int32), mode_timer)
        ghosts_eaten_count = jnp.where(level_complete, jnp.array(0, dtype=jnp.int32), ghosts_eaten_count)
        pellets = jnp.where(level_complete, next_pellets, pellets)
        pellets_remaining = jnp.where(level_complete, next_pellet_count.astype(jnp.int32), pellets_remaining)
        pellets_eaten = jnp.where(level_complete, jnp.array(0, dtype=jnp.int32), pellets_eaten)
        power_timer = jnp.where(level_complete, jnp.array(0, dtype=jnp.int32), power_timer)
        start_timer = jnp.where(level_complete, jnp.array(self.consts.start_delay, dtype=jnp.int32), start_timer)
        death_timer = jnp.where(level_complete, jnp.array(0, dtype=jnp.int32), death_timer)
        fruit_active = jnp.where(level_complete, jnp.array(0, dtype=jnp.int32), fruit_active)
        fruit_path_idx = jnp.where(level_complete, jnp.array(0, dtype=jnp.int32), fruit_path_idx)
        fruit_pos_x = jnp.where(level_complete, next_fruit_path[0][0], fruit_pos_x)
        fruit_pos_y = jnp.where(level_complete, next_fruit_path[0][1], fruit_pos_y)
        fruit_spawned_count = jnp.where(level_complete, jnp.array(0, dtype=jnp.int32), fruit_spawned_count)
        fruit_path = jnp.where(level_complete, next_fruit_path, fruit_path)
        fruit_path_len = jnp.where(level_complete, next_fruit_path_len, fruit_path_len)
        ghost_wall_grid = jnp.where(level_complete, next_ghost_wall_grid, ghost_wall_grid)

        # Check if pacman ate the fruit
        ate_fruit = jnp.logical_and(
            fruit_active == 1,
            jnp.logical_and(pacman_x == fruit_pos_x, pacman_y == fruit_pos_y),
        )
        score = score + jnp.where(ate_fruit, self.consts.fruit_score, 0)
        fruit_active = jnp.where(ate_fruit, jnp.array(0, dtype=jnp.int32), fruit_active)

        # --- Time and game over ---
        time = state.time + 1
        game_over = jnp.logical_or(state.game_over, lives <= 0)
        game_phase = jnp.where(game_over, jnp.array(2, dtype=jnp.int32), state.game_phase)

        new_state = MsPacmanState(
            pacman_x=pacman_x,
            pacman_y=pacman_y,
            direction=direction,
            buffered_direction=buffered_direction,
            ghost_positions=ghost_positions,
            ghost_directions=ghost_directions,
            ghost_modes=ghost_modes,
            ghost_global_mode=ghost_global_mode,
            mode_timer=mode_timer,
            ghosts_eaten_count=ghosts_eaten_count,
            pellets=pellets,
            power_timer=power_timer,
            score=score,
            time=time,
            lives=lives,
            pellets_remaining=pellets_remaining,
            pellets_eaten=pellets_eaten,
            game_over=game_over,
            game_phase=game_phase,
            start_timer=start_timer,
            death_timer=death_timer,
            level_idx=level_idx,
            fruit_active=fruit_active,
            fruit_path_idx=fruit_path_idx,
            fruit_pos_x=fruit_pos_x,
            fruit_pos_y=fruit_pos_y,
            fruit_spawned_count=fruit_spawned_count,
            key=new_key,
        )

        done = self._get_done(new_state)
        env_reward = jnp.float32(self._get_reward(state, new_state))
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)
        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_ghost_target(
        self,
        ghost_idx: int,
        ghost_pos: chex.Array,
        ghost_mode: chex.Array,
        pacman_x: chex.Array,
        pacman_y: chex.Array,
        pacman_dir: chex.Array,
        blinky_pos: chex.Array,
        ghost_global_mode: chex.Array,
    ) -> chex.Array:
        """Compute target tile for a single ghost based on its personality."""
        pac_pos = jnp.array([pacman_x, pacman_y], dtype=jnp.int32)

        # Scatter targets
        scatter_target = self.scatter_targets[ghost_idx]

        # Chase targets per ghost personality
        # Blinky (0): target pac-man directly
        blinky_target = pac_pos

        # Pinky (1): target 4 tiles ahead of pac-man
        pinky_target = pac_pos + pacman_dir * 4
        pinky_target = jnp.array([
            jnp.mod(pinky_target[0], self.consts.grid_width),
            jnp.clip(pinky_target[1], 0, self.consts.grid_height - 1),
        ])

        # Inky (2): vector from blinky through point 2 tiles ahead of pacman, doubled
        ahead_2 = pac_pos + pacman_dir * 2
        inky_target = 2 * ahead_2 - blinky_pos
        inky_target = jnp.array([
            jnp.mod(inky_target[0], self.consts.grid_width),
            jnp.clip(inky_target[1], 0, self.consts.grid_height - 1),
        ])

        # Sue (3): chase when far (>8 tiles), scatter when close
        sue_dist = jnp.abs(pac_pos[0] - ghost_pos[0]) + jnp.abs(pac_pos[1] - ghost_pos[1])
        sue_target = jnp.where(sue_dist > 8, pac_pos, scatter_target)

        # Select based on ghost index
        chase_target = jnp.where(
            ghost_idx == 0, blinky_target,
            jnp.where(ghost_idx == 1, pinky_target,
            jnp.where(ghost_idx == 2, inky_target,
            sue_target))
        )

        # Eaten ghosts target the pen
        pen_target = jnp.array([self.consts.pen_x, self.consts.pen_door_y], dtype=jnp.int32)

        # Select target based on mode
        target = jnp.where(
            ghost_mode == 2,  # eaten -> go to pen
            pen_target,
            jnp.where(
                ghost_mode == 1,  # frightened -> handled separately (random)
                scatter_target,   # placeholder, won't be used
                jnp.where(
                    ghost_global_mode == 0,  # scatter
                    scatter_target,
                    chase_target,  # chase
                ),
            ),
        )
        return target

    @partial(jax.jit, static_argnums=(0,))
    def _move_ghosts(
        self,
        ghost_positions: chex.Array,
        ghost_directions: chex.Array,
        ghost_modes: chex.Array,
        pacman_x: chex.Array,
        pacman_y: chex.Array,
        pacman_dir: chex.Array,
        ghost_global_mode: chex.Array,
        can_move: chex.Array,
        time: chex.Array,
        key: chex.Array,
        ghost_wall_grid: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Move all ghosts using per-ghost targeting and tile-based pathfinding."""
        blinky_pos = ghost_positions[0]

        # 4 cardinal directions
        dir_table = jnp.array([[1, 0], [-1, 0], [0, -1], [0, 1]], dtype=jnp.int32)

        def move_single_ghost(carry, ghost_idx):
            positions, directions, modes = carry
            gx = positions[ghost_idx, 0]
            gy = positions[ghost_idx, 1]
            mode = modes[ghost_idx]
            prev_dir = directions[ghost_idx]

            # Get target for this ghost
            target = self._get_ghost_target(
                ghost_idx, positions[ghost_idx], mode,
                pacman_x, pacman_y, pacman_dir, blinky_pos, ghost_global_mode,
            )

            # Check if ghost is in pen (mode 3)
            in_pen = mode == 3
            should_leave_pen = time >= self.ghost_spawn_delays[ghost_idx]

            # Pen exit: first center on exit column (pen_x), then move up
            dx_to_exit = jnp.sign(self.consts.pen_x - gx)
            at_exit_col = gx == self.consts.pen_x
            pen_exit_dir = jnp.where(
                at_exit_col,
                jnp.array([0, -1], dtype=jnp.int32),  # move up
                jnp.array([dx_to_exit, 0], dtype=jnp.int32),  # move toward exit column
            )

            # Compute reverse of previous direction (ghosts can't reverse)
            reverse_dir = -prev_dir

            # For each candidate direction, compute Manhattan distance to target
            def eval_direction(d_idx):
                d = dir_table[d_idx]
                nx = jnp.mod(gx + d[0], self.consts.grid_width)
                ny = jnp.clip(gy + d[1], 0, self.consts.grid_height - 1)
                blocked = ghost_wall_grid[ny, nx] == 1
                is_reverse = jnp.logical_and(d[0] == reverse_dir[0], d[1] == reverse_dir[1])
                # Manhattan distance to target
                dist = jnp.abs(nx - target[0]) + jnp.abs(ny - target[1])
                # Penalize blocked and reverse directions heavily
                penalty = jnp.where(blocked, jnp.array(9999, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
                penalty = penalty + jnp.where(is_reverse, jnp.array(9998, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
                return dist + penalty

            costs = jax.vmap(eval_direction)(jnp.arange(4))

            # Frightened mode: pick random valid direction
            ghost_key = jax.random.fold_in(key, ghost_idx)
            rand_perm = jax.random.permutation(ghost_key, 4)
            # For frightened ghosts, use random permutation of costs to randomize choice
            frightened_costs = costs[rand_perm]
            frightened_best_idx = rand_perm[jnp.argmin(frightened_costs)]

            # Normal mode: pick direction with lowest cost (ties broken by priority: up, left, down, right)
            normal_best_idx = jnp.argmin(costs)

            best_idx = jnp.where(mode == 1, frightened_best_idx, normal_best_idx)
            best_dir = dir_table[best_idx]

            # In-pen behavior
            best_dir = jnp.where(
                jnp.logical_and(in_pen, should_leave_pen),
                pen_exit_dir,
                jnp.where(in_pen, jnp.array([0, 0], dtype=jnp.int32), best_dir),
            )

            new_x = jnp.mod(gx + best_dir[0], self.consts.grid_width)
            new_y = jnp.clip(gy + best_dir[1], 0, self.consts.grid_height - 1)

            # Check if new position is blocked
            new_blocked = self.ghost_wall_grid[new_y, new_x] == 1
            new_x = jnp.where(new_blocked, gx, new_x)
            new_y = jnp.where(new_blocked, gy, new_y)
            best_dir = jnp.where(new_blocked, jnp.array([0, 0], dtype=jnp.int32), best_dir)

            # Only move if can_move flag is set
            final_x = jnp.where(can_move, new_x, gx)
            final_y = jnp.where(can_move, new_y, gy)
            final_dir = jnp.where(can_move, best_dir, prev_dir)

            # Update mode: in_pen ghost that reached pen door becomes normal
            reached_door = jnp.logical_and(
                in_pen,
                jnp.logical_and(final_y <= self.consts.pen_door_y, should_leave_pen),
            )
            new_mode = jnp.where(reached_door, jnp.array(0, dtype=jnp.int32), mode)

            # Eaten ghost that reached pen becomes normal (respawned)
            is_eaten = mode == 2
            reached_pen = jnp.logical_and(
                is_eaten,
                jnp.logical_and(
                    jnp.abs(final_x - self.consts.pen_x) <= 1,
                    jnp.abs(final_y - self.consts.pen_y) <= 1,
                ),
            )
            new_mode = jnp.where(reached_pen, jnp.array(0, dtype=jnp.int32), new_mode)

            # Update arrays
            new_positions = positions.at[ghost_idx].set(jnp.array([final_x, final_y]))
            new_directions = directions.at[ghost_idx].set(final_dir)
            new_modes = modes.at[ghost_idx].set(new_mode)

            return (new_positions, new_directions, new_modes), None

        (ghost_positions, ghost_directions, ghost_modes), _ = jax.lax.scan(
            move_single_ghost,
            (ghost_positions, ghost_directions, ghost_modes),
            jnp.arange(self.consts.num_ghosts, dtype=jnp.int32),
        )

        return ghost_positions, ghost_directions, ghost_modes

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
        return spaces.Dict({
            "pacman": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.grid_width - 1, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.grid_height - 1, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "ghosts": spaces.Box(
                low=0,
                high=max(self.consts.grid_width, self.consts.grid_height),
                shape=(self.consts.num_ghosts, 4),
                dtype=jnp.int32,
            ),
            "pellets": spaces.Box(
                low=0,
                high=2,
                shape=(self.consts.grid_height, self.consts.grid_width),
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
    def __init__(self, consts: MsPacmanConstants = None, wall_grid: jnp.ndarray = None, pellet_template: jnp.ndarray = None, config=None):
        self.consts = consts or DEFAULT_MSPACMAN_CONSTANTS
        # Honor provided config (for native downscaling/grayscale); otherwise use default
        self.config = config or render_utils.RendererConfig(
            game_dimensions=(self.consts.screen_height, self.consts.screen_width),
            channels=3,
        )
        super().__init__(consts, config=self.config)
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Support per-level arrays or single grids; renderer keeps stacked grids and per-level backgrounds
        if wall_grid is None:
            wall_grid = jnp.zeros((self.consts.grid_height, self.consts.grid_width), dtype=jnp.int32)
        if wall_grid.ndim == 2:
            wall_grid = wall_grid[None, ...]
        if pellet_template is None:
            pellet_template = jnp.zeros_like(wall_grid)
        if pellet_template.ndim == 2:
            pellet_template = pellet_template[None, ...]

        self.wall_grids = jnp.asarray(wall_grid, dtype=jnp.int32)
        self.pellet_templates = jnp.asarray(pellet_template, dtype=jnp.int32)
        # For convenience keep level 0 references
        self.wall_grid = self.wall_grids[0]
        self.pellet_template = self.pellet_templates[0]

        # Grid layout
        grid_h, grid_w = self.wall_grid.shape
        self.cell = self.consts.cell_size
        grid_px_w = grid_w * self.cell
        grid_px_h = grid_h * self.cell
        self.offset_x = max((self.consts.screen_width - grid_px_w) // 2, 0)
        self.offset_y = 2#GAME_BOARD_OFFSET_Y

        # Build per-level RGBA backgrounds
        backgrounds = []
        for idx in range(self.wall_grids.shape[0]):
            bg_rgba = self._build_background(self.wall_grids[idx], self.pellet_templates[idx])
            backgrounds.append(np.array(bg_rgba))
        self.backgrounds_rgba = jnp.asarray(np.stack(backgrounds, axis=0))
        # Use level 0 for asset setup; render will pick per-level
        background_rgba = self.backgrounds_rgba[0]

        # Asset configuration for palette-based rendering
        sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites", "mspacman")

        # Create procedural eyes sprite for eaten ghosts (10x9 RGBA)
        eyes_sprite = np.zeros((10, 9, 4), dtype=np.uint8)
        # White eyeballs (2x2 each)
        eyes_sprite[3:5, 1:3] = [255, 255, 255, 255]  # left eye
        eyes_sprite[3:5, 5:7] = [255, 255, 255, 255]  # right eye
        # Blue pupils (1x1 each)
        eyes_sprite[4, 2] = [0, 28, 136, 255]          # left pupil
        eyes_sprite[4, 6] = [0, 28, 136, 255]          # right pupil

        asset_config = [
            {'name': 'bg', 'type': 'background', 'data': background_rgba},
            {'name': 'pacman', 'type': 'group', 'files': [f'pacman_{i}.npy' for i in range(3)]},
            {'name': 'pacman_lives', 'type': 'group', 'files': ['MsPacman_lives.npy']},

            {'name': 'ghost', 'type': 'group', 'files': [
                'ghost_blinky.npy', 'ghost_pinky.npy', 'ghost_inky.npy', 'ghost_sue.npy',
                'ghost_blue.npy', 'ghost_white.npy',
            ]},
            {'name': 'ghost_eyes', 'type': 'procedural', 'data': jnp.array(eyes_sprite)},
            {'name': 'score', 'type': 'digits', 'pattern': 'score_{}.npy'},
            {'name': 'fruit', 'type': 'group', 'files': [
                'fruit_cherry.npy', 
                'fruit_banana.npy', 
                'fruit_pear.npy', 
                'fruit_apple.npy'
            ]},        
            ]

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        # Convert RGBA backgrounds to palette ID masks for rendering
        bg_ids = []
        for idx in range(self.backgrounds_rgba.shape[0]):
            bg_rgba = np.array(self.backgrounds_rgba[idx])  # (H, W, 4)
            h, w = bg_rgba.shape[:2]
            ids = np.zeros((h, w), dtype=np.uint8)
            for color, cid in self.COLOR_TO_ID.items():
                if len(color) == 3:
                    rgba = np.array([color[0], color[1], color[2], 255], dtype=np.uint8)
                else:
                    rgba = np.array(color, dtype=np.uint8)
                mask = np.all(bg_rgba == rgba, axis=-1)
                ids[mask] = cid
            bg_ids.append(ids)
        self.backgrounds = jnp.asarray(np.stack(bg_ids, axis=0))

        # Pre-compute directional pacman masks: (4 frames, 4 dirs, H, W)
        # dir 0=left (base), 1=right (flip_h), 2=up (rot90 CW), 3=down (rot90 CCW)
        pac_masks = self.SHAPE_MASKS['pacman']  # (4, H, W)
        pac_flip_offset = self.FLIP_OFFSETS['pacman']

        # Pad all frames to a shared square size to allow rotations/stacking
        max_h = int(max([pac_masks[i].shape[0] for i in range(pac_masks.shape[0])]))
        max_w = int(max([pac_masks[i].shape[1] for i in range(pac_masks.shape[0])]))
        size = max(max_h, max_w)

        def pad_to_size(mask):
            h, w = mask.shape
            pad_h = size - h
            pad_w = size - w
            return jnp.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

        directional = []
        for frame_idx in range(4):
            m = pad_to_size(pac_masks[frame_idx])
            left = m                                    # base faces left
            right = jnp.flip(m, axis=1)                 # horizontal flip
            up = jnp.rot90(m, k=3)                      # 90° CW (mouth points up)
            down = jnp.rot90(m, k=1)                    # 90° CCW (mouth points down)
            directional.append(jnp.stack([left, right, up, down]))
        self.pacman_dir_masks = jnp.stack(directional)  # (4, 4, H, W)

        # Precompute pellet color IDs for render_grid_inverse
        pellet_rgb = self.consts.pellet_color
        power_rgb = self.consts.power_pellet_color
        bg_rgb = self.consts.background_color
        # Color map: index 0 = no pellet (transparent), 1 = regular pellet, 2 = power pellet
        pellet_color_id = self.COLOR_TO_ID.get(pellet_rgb, 0)
        power_color_id = self.COLOR_TO_ID.get(power_rgb, pellet_color_id)
        self.pellet_color_map = jnp.array([0, pellet_color_id, power_color_id], dtype=jnp.uint8)

    def _build_background(self, wall_grid: jnp.ndarray, pellet_template: jnp.ndarray):
        """Build background RGBA image from wall grid and pellet template."""
        cell = self.cell
        grid_h, grid_w = wall_grid.shape

        # Target output size respects renderer config (native downscaling)
        target_h, target_w = (
            self.config.downscale if self.config.downscale is not None
            else (self.consts.screen_height, self.consts.screen_width)
        )

        # Build per-tile colors
        bg_color = np.array(self.consts.background_color, dtype=np.uint8)
        wall_color = np.array(self.consts.wall_color, dtype=np.uint8)
        corridor_color = np.array(self.consts.blocked_color, dtype=np.uint8)

        # Create tile-level image
        tile_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        wall_np = np.array(wall_grid)
        pellet_np = np.array(pellet_template)

        for y in range(grid_h):
            for x in range(grid_w):
                if wall_np[y, x] == 1:
                    if pellet_np[y, x] == -1:
                        tile_img[y, x] = corridor_color  # soft wall = blue
                    else:
                        tile_img[y, x] = wall_color
                else:
                    tile_img[y, x] = corridor_color  # corridors

        # Scale up to pixel resolution
        pixel_img = np.repeat(np.repeat(tile_img, cell, axis=0), cell, axis=1)

        # Place on full screen canvas
        canvas = np.zeros((self.consts.screen_height, self.consts.screen_width, 3), dtype=np.uint8)
        # Fill with background color
        canvas[:] = bg_color
        h = min(pixel_img.shape[0], canvas.shape[0] - self.offset_y)
        w = min(pixel_img.shape[1], canvas.shape[1] - self.offset_x)
        canvas[self.offset_y:self.offset_y + h, self.offset_x:self.offset_x + w] = pixel_img[:h, :w]
        
        # Stripes at the bottom of the maze (rows 170 and 171)
        canvas[1, :] = wall_color
        canvas[170:172, :] = wall_color

        # Convert to RGBA
        alpha = np.full((self.consts.screen_height, self.consts.screen_width, 1), 255, dtype=np.uint8)
        canvas_rgba = np.concatenate([canvas, alpha], axis=2)

        # If renderer is downscaling, resize background to target dimensions
        if self.config.downscale is not None:
            canvas_rgba = np.array(
                jim.resize(jnp.asarray(canvas_rgba), (target_h, target_w, 4), method="nearest")
            )
        return jnp.asarray(canvas_rgba)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: MsPacmanState) -> jnp.ndarray:
        # Select correct background for current level
        bg = self.backgrounds[state.level_idx]
        # Ensure palette ID background is 2D (H, W)
        bg = jnp.squeeze(bg)
        if bg.ndim == 3:
            bg = bg[..., 0]
        raster = self.jr.create_object_raster(bg)
        cell = self.cell

        # --- Draw pellets using render_grid_inverse ---
        # Power pellet blinking
        blink_on = (state.time // self.consts.power_pellet_blink_period) % 2 == 0
        # Create pellet grid: 0=empty, 1=regular, 2=power (but blink power pellets)
        pellet_grid = state.pellets
        # When blink is off, hide power pellets (set 2 -> 0)
        pellet_grid = jnp.where(
            jnp.logical_and(pellet_grid == 2, ~blink_on),
            jnp.zeros_like(pellet_grid),
            pellet_grid,
        )

        # Regular pellets (value==1): small 2x2 dots centered in 4x4 cells
        regular_grid = jnp.where(pellet_grid == 1, jnp.ones_like(pellet_grid), jnp.zeros_like(pellet_grid))
        raster = self.jr.render_grid_inverse(
            raster,
            regular_grid,
            grid_origin=(self.offset_x + 0, self.offset_y + 1),
            cell_size=(4, 2),
            color_map=self.pellet_color_map,
            cell_padding=(0, 2),
        )
        # Power pellets (value==2): full 4x4 dots
        power_grid = jnp.where(pellet_grid == 2, jnp.ones_like(pellet_grid), jnp.zeros_like(pellet_grid))
        raster = self.jr.render_grid_inverse(
            raster,
            power_grid,
            grid_origin=(self.offset_x, self.offset_y),
            cell_size=(cell, cell),
            color_map=self.pellet_color_map,
        )

        # --- Draw Pac-Man ---
        anim_frame = (state.time // 4) % 4
        dir_x, dir_y = state.direction[0], state.direction[1]
        # Direction index: 0=left, 1=right, 2=up, 3=down
        normal_dir_idx = jnp.where(dir_x < 0, 0,
                         jnp.where(dir_x > 0, 1,
                         jnp.where(dir_y < 0, 2, 3)))

        # Handle death animation: spin counter-clockwise
        is_dead = state.death_timer > 0
        death_elapsed = self.consts.death_freeze_duration - state.death_timer
        # Spin cycle directions: 2=up, 0=left, 3=down, 1=right
        spin_dirs = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
        spin_idx = (death_elapsed // 4) % 4
        death_dir_idx = spin_dirs[spin_idx]

        dir_idx = jnp.where(is_dead, death_dir_idx, normal_dir_idx)

        pacman_mask = self.pacman_dir_masks[anim_frame, dir_idx]
        pac_px = self.offset_x + state.pacman_x * cell - 3  # center sprite
        pac_py = self.offset_y + state.pacman_y * cell - 3
        raster = self.jr.render_at_clipped(
            raster, pac_px, pac_py, pacman_mask,
        )

        # --- Draw ghosts ---
        eyes_mask = self.SHAPE_MASKS['ghost_eyes']

        def draw_ghost(raster_carry, ghost_idx):
            mode = state.ghost_modes[ghost_idx]
            pos = state.ghost_positions[ghost_idx]

            # Select sprite based on mode
            normal_idx = ghost_idx  # 0=blinky, 1=pinky, 2=inky, 3=sue
            blue_idx = jnp.array(4, dtype=jnp.int32)
            white_idx = jnp.array(5, dtype=jnp.int32)

            # Frightened: alternate blue/white based on timer for flashing
            is_flashing = state.power_timer < self.consts.frightened_flash_start
            flash_white = jnp.logical_and(is_flashing, (state.time // 8) % 2 == 0)
            frightened_idx = jnp.where(flash_white, white_idx, blue_idx)

            # Mode: 0=normal, 1=frightened, 2=eaten, 3=in_pen
            sprite_idx = jnp.where(
                mode == 1, frightened_idx,
                jnp.where(mode == 3, normal_idx, normal_idx),
            )

            ghost_mask = self.SHAPE_MASKS['ghost'][sprite_idx]
            gx = self.offset_x + pos[0] * cell - 2  # center sprite
            gy = self.offset_y + pos[1] * cell - 3

            is_eaten = mode == 2
            # Eaten ghosts: draw eyes sprite; others: draw ghost sprite
            chosen_mask = jnp.where(is_eaten, eyes_mask, ghost_mask)

            new_raster = self.jr.render_at_clipped(
                raster_carry, gx, gy, chosen_mask,
                flip_offset=self.FLIP_OFFSETS['ghost'],
            )
            return new_raster, None

        raster, _ = jax.lax.scan(
            draw_ghost, raster,
            jnp.arange(self.consts.num_ghosts, dtype=jnp.int32),
        )

        # --- Draw fruit ---
        def draw_fruit(r):
            # Find how many fruits we loaded to prevent out-of-bounds crashing
            num_fruits = self.SHAPE_MASKS['fruit'].shape[0]
            
            # Use the level_idx, but cap it at the max available fruits
            fruit_sprite_idx = jnp.minimum(state.level_idx, num_fruits - 1)
            
            # Select the dynamic fruit mask!
            fruit_mask = self.SHAPE_MASKS['fruit'][fruit_sprite_idx]
            
            fx = self.offset_x + state.fruit_pos_x * cell - 4
            fy = self.offset_y + state.fruit_pos_y * cell - 4
            return self.jr.render_at_clipped(r, fx, fy, fruit_mask)

        raster = jax.lax.cond(
            state.fruit_active == 1,
            draw_fruit,
            lambda r: r,
            raster,
        )

        # --- Draw score ---
        score_digits = self.jr.int_to_digits(state.score, max_digits=6)
        num_digits = jnp.where(state.score == 0, 1,
                               jnp.floor(jnp.log10(jnp.maximum(state.score, 1))).astype(jnp.int32) + 1)
        start_index = 6 - num_digits
        digit_masks = self.SHAPE_MASKS['score']

        # Score position: right-aligned near top-right
        score_x = 87
        score_y = self.consts.screen_height - 23

        raster = self.jr.render_label_selective(
            raster, score_x, score_y,
            score_digits, digit_masks,
            start_index, num_digits,
            spacing=8, max_digits_to_render=6,
        )

        # --- Draw lives (lives - 1 remaining icons) ---
        lives_to_show = jnp.maximum(state.lives - 1, 0)
        life_mask = self.SHAPE_MASKS['pacman_lives'][0]  # Use first pacman frame
        raster = self.jr.render_indicator(
            raster, 9, self.consts.screen_height - 37,
            lives_to_show, life_mask,
            spacing=16, max_value=3,
        )
        
        # --- Draw level fruit on scoreboard ---
        num_fruits = self.SHAPE_MASKS['fruit'].shape[0]
        fruit_sprite_idx = jnp.minimum(state.level_idx, num_fruits - 1)
        scoreboard_fruit_mask = self.SHAPE_MASKS['fruit'][fruit_sprite_idx]
        
        fruit_ui_x = 128 
        fruit_ui_y = self.consts.screen_height - 37
        
        raster = self.jr.render_at_clipped(
            raster, fruit_ui_x, fruit_ui_y, scoreboard_fruit_mask
        )

        return self.jr.render_from_palette(raster, self.PALETTE)

